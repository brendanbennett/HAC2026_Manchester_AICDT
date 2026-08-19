#!/usr/bin/env python3
"""Train the LPD as a conditional flow.

Stage 1 builds the corpus. Each training body is fitted by the implicit field -- h plus the 32
tokens -- by regressing the field onto the body's signed distance. That fit is what supplies
x1, the target token code. Fitting
needs no mesh extraction, only field evaluations at sampled points, so it is cheap.

Stage 2 trains the flow. For a draw x0 ~ N(0, I) and a time t, the state is
x_t = (1-t) x0 + t x1, the operator is applied to x_t, the dual network reduces the residual
against the data, and the primal network predicts the velocity, whose target is x1 - x0 at
every t.

The loss is a sum over the six step times; sampling one k per draw estimates it without bias
at a sixth of the cost, which matters because each operator evaluation needs a mesh.

--steps is a cap, not a schedule. A few corpus bodies are held out of training and scored
every --val-every steps at fixed draws; training stops once that score has gone --patience
evaluations without improving, and the saved checkpoint is the best-scoring one rather than
whatever the last step happened to leave behind. --val-bodies 0 turns all of that off.

Public bodies are never in the corpus. They are the test set, held out from the held-out
set too, and appear only at reconstruction time.

Training checkpoints itself every --ckpt-every steps to --ckpt-file (default <--out>.ckpt)
and resumes from it by default, so a run longer than one batch slot can be spread over
several jobs: --steps is the total cap, and each job trains from wherever the last one
stopped up to that cap. The checkpoint carries the optimiser and early-stopping state, not
just the weights. Only a completed run writes --out, which is what the pipeline reads --
to reconstruct from an unfinished run, point reconstruct_lpd.py --ckpt at the .ckpt itself.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.conventions import S_LAB, cameras, psi_grid, to_body          # noqa: E402
from hac26.field import DESIGN_N, ImplicitBody, extract_mesh    # noqa: E402
from hac26.solvers.lpd_flow import N_MODES, N_STEPS, LPDFlow           # noqa: E402
from hac26.forward.mesh.radiosity import facet_geometry                               # noqa: E402
from hac26.forward.learned_surrogate import (Surrogate, camera_features,                 # noqa: E402
                             sun_features)                               # noqa: E402


_DECODER = {}


def _corpus_meta(n, psi, op_res) -> dict:
    return {
        "schema": 3,
        "bodies": int(n),
        "phases": int(len(psi)),
        "n_geoms": int(len(cameras())),
        "operator_res": int(op_res),
        "design_n": int(DESIGN_N),
    }


def _decode_meta(z) -> dict | None:
    if "meta" not in z.files:
        return None
    try:
        return json.loads(str(z["meta"]))
    except Exception:                                  # noqa: BLE001  corrupt metadata
        return None


def _load_valid_corpus_cache(cache: str, expected: dict):
    try:
        z = np.load(cache, allow_pickle=False)
    except Exception as exc:                           # noqa: BLE001  corrupt: rebuild
        print(f"  ignoring unreadable corpus cache {cache}: {exc}", flush=True)
        return None
    meta = _decode_meta(z)
    if meta is None:
        print(f"  ignoring legacy corpus cache {cache}: no metadata", flush=True)
        return None
    bad = [k for k, v in expected.items() if meta.get(k) != v]
    if bad:
        print(f"  ignoring stale corpus cache {cache}: metadata mismatch {bad}",
              flush=True)
        return None
    if not {"codes", "curves", "support"}.issubset(z.files):
        print(f"  ignoring stale corpus cache {cache}: missing arrays", flush=True)
        return None
    codes, curves, support = z["codes"], z["curves"], z["support"]
    if (len(codes) == 0 or support.shape != (len(codes), DESIGN_N)
            or curves.shape != (len(codes), len(cameras()), 2, expected["phases"])):
        print(f"  ignoring stale corpus cache {cache}: bad array shapes", flush=True)
        return None
    print(f"  loaded corpus of {len(codes)} bodies from {cache}", flush=True)
    return torch.tensor(codes), torch.tensor(curves), torch.tensor(support)


def _load_valid_corpus_part(path: Path, expected: dict, body_index: int):
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as exc:                           # noqa: BLE001  corrupt: redo
        print(f"  ignoring unreadable corpus part {path}: {exc}", flush=True)
        return None
    meta = _decode_meta(z)
    if meta is None or any(meta.get(k) != v for k, v in expected.items()):
        print(f"  ignoring stale corpus part {path}", flush=True)
        return None
    if int(z["body_index"]) != int(body_index):
        print(f"  ignoring corpus part {path}: body index mismatch", flush=True)
        return None
    code, curve, support = z["code"], z["curve"], z["support"]
    if support.shape != (DESIGN_N,) or curve.shape != (len(cameras()), 2, expected["phases"]):
        print(f"  ignoring corpus part {path}: bad array shapes", flush=True)
        return None
    return code, curve, support


def _save_corpus_part(path: Path, body_index: int, code, curve, support, expected: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, body_index=int(body_index), code=code, curve=curve, support=support,
             meta=json.dumps(expected, sort_keys=True))


def load_decoder(body, path="runs/token_decoder.pt"):
    """Install the shared token decoder.

    A fresh ImplicitBody carries randomly initialised cross-attention and MLP weights, so
    without this the code is decoded by a decoder it was never fitted against and the token
    field contributes nothing. It would also make A(x) non-deterministic."""
    if path not in _DECODER:
        _DECODER[path] = torch.load(path, map_location="cpu")
    body.tokens.load_state_dict(_DECODER[path], strict=False)
    return body


def code_of(body: ImplicitBody) -> torch.Tensor:
    return torch.cat([body.tokens.p.reshape(-1), body.tokens.z.reshape(-1)])


def set_code(body: ImplicitBody, code: torch.Tensor) -> None:
    n = body.tokens.p.numel()
    with torch.no_grad():
        body.tokens.p.copy_(code[:n].reshape_as(body.tokens.p))
        body.tokens.z.copy_(code[n:].reshape_as(body.tokens.z))


def fit_body(verts, faces, radius=1.0, steps=250, n_pts=6000, device="cpu", seed=0):
    """Fit the implicit field to a mesh by SDF regression. Returns (body, code, loss)."""
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=False)
    rng = np.random.default_rng(seed)
    ext = float(np.abs(verts).max()) * 1.3
    pts = rng.uniform(-ext, ext, (n_pts, 3))
    surf, _ = trimesh.sample.sample_surface(m, n_pts // 2)
    pts = np.vstack([pts, surf + rng.normal(0, 0.03, surf.shape)])
    sd = -m.nearest.signed_distance(pts)          # trimesh: positive inside
    P = torch.tensor(pts, dtype=torch.float32, device=device)
    S = torch.tensor(sd, dtype=torch.float32, device=device)
    body = ImplicitBody(radius=radius).to(device)
    h0 = np.maximum(verts @ body.core.n.cpu().numpy().T, 1e-3).max(axis=0).astype(np.float32)
    body.core.set_support(torch.tensor(h0))
    opt = torch.optim.Adam(body.parameters(), lr=0.02)
    for _ in range(steps):
        loss = ((body(P) - S) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return body, code_of(body).detach(), float(loss)


def curves_from_code(code, radius, surro, psi, res=32, device=None, geoms=None,
                     chunk=4, support=None, decoder_path="runs/token_decoder.pt"):
    """A(x): decode a token code to a body, extract, tokenise, and run the surrogate.

    Covers all 28 geometries. The dual attends across them, which is what recovers the m = 0
    content that a single lightcurve cannot constrain.

    Two economies, neither changing the arithmetic: the mesh is extracted once per code rather
    than per camera, and the source is fixed in the lab frame, so light visibility and the
    gathered bounce are traced once and shared across all 28. What remains per camera is one
    batched ray cast. Chunking the surrogate over geometries bounds the (G, T, P, W)
    activations.
    """
    # The field evaluation dominates this call, not the mesh extraction.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    body = ImplicitBody(radius=radius).to(device)
    # h is supplied by the caller and comes from the convex stage; the flow generates only
    # the token correction.
    n_norm = body.core.n.shape[0]
    if support is None:
        h = torch.full((n_norm,), 0.8 * radius, dtype=torch.float32, device=device)
    else:
        h = torch.as_tensor(support, dtype=torch.float32, device=device)
        if h.numel() != n_norm:
            raise ValueError(f"support has {h.numel()} entries, but this field uses "
                             f"{n_norm} normals; rerun fit_shapes.py after changing "
                             "DESIGN_N")
    body.core.set_support(h)
    load_decoder(body, path=decoder_path)
    set_code(body, code.to(device))
    ext = radius * 1.6
    v, f = extract_mesh(lambda y: body(y), ext, res=res, device=device)
    if len(f) < 8:
        return None
    # Decimate to the token count the surrogate was trained at; ray tracing scales as
    # tokens x phases.
    return curves_from_mesh(v, f, surro, psi, geoms=geoms, chunk=chunk)


def curves_from_mesh(v, f, surro, psi, geoms=None, chunk=4):
    """The second half of A(x): a mesh in, the 28 reduced curve pairs out.

    Separate from the code path so the planar snap can be evaluated on a candidate mesh that
    did not come from a code.
    """
    import trimesh
    from hac26.calibrate import decimate
    v, f = decimate(v, f, 600)
    c, n, a = facet_geometry(v, f)
    mesh = trimesh.Trimesh(v, f, process=False)
    cams = list(cameras()) if geoms is None else [list(cameras())[i] for i in geoms]
    sun_d = np.stack([to_body(S_LAB, np.array([p]))[0] for p in psi])
    sun = sun_features(mesh, c, n, sun_d)
    fe = np.stack([camera_features(
        mesh, c, n,
        np.stack([to_body(np.asarray(cm.v), np.array([p]))[0] for p in psi]),
        sun, areas=a) for cm in cams])
    dv = next(surro.parameters()).device
    ar = torch.tensor(np.tile((a / a.sum())[None], (len(cams), 1)),
                      dtype=torch.float32, device=dv)
    return _run_chunked(surro, fe, ar, dv, chunk)   # (G, 2, P)


def _run_chunked(surro, fe, ar, dv, chunk):
    """Evaluate the surrogate over geometry chunks, halving on OOM and falling back to CPU.

    The activations are (G, T, P, W) = (28, 600, 96, 96) at full width, which is 2 GB per
    four geometries. Whether that fits depends on what else holds the card -- a training run
    in another process, for one -- so the chunk cannot be a fixed constant chosen once. An
    unattended reconstruction that dies on a transient allocation failure is worse than a
    slow one, and the CPU path gives identical numbers.
    """
    while chunk >= 1:
        try:
            out = []
            with torch.no_grad():
                for i in range(0, len(fe), chunk):
                    x = torch.tensor(fe[i:i + chunk], dtype=torch.float32, device=dv)
                    out.append(surro(x, ar[i:i + chunk]).cpu())
            return torch.cat(out)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            chunk //= 2
    surro_cpu = surro.to("cpu")
    with torch.no_grad():
        return surro_cpu(torch.tensor(fe, dtype=torch.float32), ar.cpu())


def corpus(n, psi, surro, device, seed=0, cache=None, op_res: int = 32,
           codes_file="runs/corpus_codes.npz", decoder_path="runs/token_decoder.pt"):
    """Curves for the corpus, from codes fitted by scripts/fit_shapes.py.

    The codes and the shared decoder come from the AUTODECODER fit, not from per-body fits.
    Fitting each body separately gives each its own q/k/v/mlp, and a code is then meaningless
    to any other decoder -- which is what made the token channel inert. Run fit_shapes.py
    first; this only applies the operator.
    """
    if not Path(codes_file).exists():
        raise SystemExit(f"{codes_file} missing -- run scripts/fit_shapes.py first")
    if not Path(decoder_path).exists():
        raise SystemExit(f"{decoder_path} missing -- run scripts/fit_shapes.py first")
    zz = np.load(codes_file)
    if "codes" not in zz.files or "support" not in zz.files:
        raise SystemExit(f"{codes_file} must contain 'codes' and 'support' arrays")
    if len(zz["codes"]) < n:
        raise SystemExit(f"{codes_file} contains {len(zz['codes'])} bodies, "
                         f"but --bodies requested {n}")
    if zz["support"].ndim != 2 or zz["support"].shape[1] != DESIGN_N:
        raise SystemExit(f"{codes_file} support has shape {zz['support'].shape}, "
                         f"but DESIGN_N={DESIGN_N}; rerun fit_shapes.py")
    all_codes, all_sup = zz["codes"][:n], zz["support"][:n]
    expected_meta = _corpus_meta(n, psi, op_res)
    if cache and Path(cache).exists():
        cached = _load_valid_corpus_cache(cache, expected_meta)
        if cached is not None:
            return cached

    part_dir = Path(f"{cache}.parts") if cache else None
    codes, curves, sup = [], [], []
    for i in range(len(all_codes)):
        t0 = time.time()
        if part_dir is not None:
            part = part_dir / f"body_{i:05d}.npz"
            cached_part = _load_valid_corpus_part(part, expected_meta, i)
            if cached_part is not None:
                code_i, cur_i, sup_i = cached_part
                codes.append(code_i); curves.append(cur_i); sup.append(sup_i)
                print(f"  body {i}: resumed from {part}", flush=True)
                continue
        code = torch.tensor(all_codes[i]); h = torch.tensor(all_sup[i])
        cur = curves_from_code(code, 1.0, surro, psi, res=op_res, support=h,
                               decoder_path=decoder_path)
        if cur is None:
            print(f"  body {i}: degenerate, skipped", flush=True); continue
        cur_np = cur.numpy()
        codes.append(all_codes[i]); curves.append(cur_np); sup.append(all_sup[i])
        if part_dir is not None:
            _save_corpus_part(part_dir / f"body_{i:05d}.npz", i, all_codes[i], cur_np,
                              all_sup[i], expected_meta)
        print(f"  body {i}: h {float(h.min()):.3f}-{float(h.max()):.3f}, "
              f"{time.time()-t0:.1f}s", flush=True)
    if not codes:
        raise SystemExit("every corpus body decoded to a degenerate mesh")
    codes = np.stack(codes); curves = np.stack(curves); sup = np.stack(sup)
    if cache:
        np.savez(cache, codes=codes, curves=curves, support=sup,
                 meta=json.dumps(expected_meta, sort_keys=True))
    return torch.tensor(codes), torch.tensor(curves), torch.tensor(sup)


def flow_loss(net, surro, psi, codes, curves, sup, idx, x0, k, M, tag, mask,
              decoder_path, op_res=32, train_geoms=None):
    """The flow-matching loss for one batch, given the draws (idx, x0, k).

    Split out of the training loop so validation scores the same objective through the same
    code path -- operator included. The only difference on the validation side is that the
    draws are fixed instead of resampled, which is what makes two evaluations comparable:
    a fresh x0 and k per evaluation would move the loss by more than the training does.
    """
    B, C = len(idx), tag.shape[1]
    x1 = codes[idx]
    t = k.float() / N_STEPS
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    # THE RESIDUAL, which means actually applying the operator to the current state.
    # The operator is applied to the current state x_t. Feeding only the data's Fourier
    # content would leave the network without any signal about where it currently is.
    g_dat = torch.fft.rfft(curves[idx], dim=-1)[..., 1:M + 1]      # (B, G, 2, M)
    geoms_t = None
    geoms = None
    step_mask = mask.expand(B, C)
    if train_geoms is not None:
        n_geoms = max(1, min(C, int(train_geoms)))
        if n_geoms < C:
            geoms_t = torch.randperm(C)[:n_geoms].sort().values
            geoms = geoms_t.tolist()
            step_mask = torch.zeros(B, C)
            step_mask[:, geoms_t] = 1.0

    preds = []
    for b in range(B):
        cur = curves_from_code(xt[b].detach(), 1.0, surro, psi,
                               res=op_res, geoms=geoms, support=sup[idx[b]],
                               decoder_path=decoder_path)
        pred = torch.zeros_like(curves[0])
        if cur is not None:
            if geoms_t is None:
                pred = cur
            else:
                pred[geoms_t] = cur
        preds.append(pred)
    g_cur = torch.fft.rfft(torch.stack(preds), dim=-1)[..., 1:M + 1]
    r = g_dat - g_cur                                   # (B, G, 2, M) complex
    # the convex operator explains the low orders best, so its complement is what the
    # profiling channel must carry: strip the first few m and keep the rest
    r_perp = r.clone()
    r_perp[..., :4] = 0
    feats = torch.zeros(B, C, N_MODES, 6)
    perp = torch.zeros(B, C, N_MODES, 6)
    for ch in range(2):                                 # EVERY geometry, not slot 0
        feats[:, :, :M, 2 * ch] = r[:, :, ch].real
        feats[:, :, :M, 2 * ch + 1] = r[:, :, ch].imag
        perp[:, :, :M, 2 * ch] = r_perp[:, :, ch].real
        perp[:, :, :M, 2 * ch + 1] = r_perp[:, :, ch].imag
    feats[:, :, :M, 4] = g_dat[:, :, 0].real
    feats[:, :, :M, 5] = g_dat[:, :, 1].real
    u = net.velocity(xt, feats, perp, tag.expand(B, C, 4), step_mask, t)
    return ((u - (x1 - x0)) ** 2).mean()


def validate(net, surro, psi, codes, curves, sup, val_idx, val_x0, val_k, M, tag, mask,
             decoder_path, chunk, op_res=32):
    """Mean flow loss over the held-out bodies, at fixed draws. Chunked to bound memory."""
    was_training = net.training
    net.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val_idx), chunk):
            sl = slice(i, i + chunk)
            b = len(val_idx[sl])
            tot += b * float(flow_loss(net, surro, psi, codes, curves, sup, val_idx[sl],
                                       val_x0[sl], val_k[sl], M, tag, mask, decoder_path,
                                       op_res=op_res))
            n += b
    net.train(was_training)
    return tot / max(n, 1)


def _hms(sec: float) -> str:
    """Seconds as h:mm:ss, for lines a human reads while a job is running."""
    sec = int(max(sec, 0.0))
    return f"{sec // 3600}:{(sec // 60) % 60:02d}:{sec % 60:02d}"


def _now() -> str:
    return time.strftime("%H:%M:%S")


def save_checkpoint(path, net, opt, step, best, best_state, best_step, stale, elapsed,
                    meta):
    """Write a resumable training checkpoint: weights, optimiser, schedule and RNG.

    Everything the loop needs to carry on is in here, not just the weights -- the Adam
    moments, the best-so-far state and the early-stopping counters included, since a resume
    that dropped them would restart the optimiser cold and re-earn a patience it had already
    spent. The RNG state goes too, so the batches drawn after a resume are the ones an
    uninterrupted run would have drawn.

    Written beside the target and renamed: a job killed mid-write cannot leave a truncated
    checkpoint for the next one to load.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step,
                "best": best, "best_state": best_state, "best_step": best_step,
                "stale": stale, "elapsed": elapsed, "rng": torch.get_rng_state(),
                **meta}, tmp)
    tmp.replace(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=40)
    ap.add_argument("--steps", type=int, default=1500)
    # The dual uses m = 1..40, and an rFFT of n phases yields floor(n/2)+1
    # coefficients, so fewer than 81 phases cannot supply 40 modes. At 16 phases
    # only 8 exist and the dual silently ran at a fifth of its specified width.
    ap.add_argument("--phases", type=int, default=96)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--operator-res", type=int, default=32,
                    help="FlexiCubes resolution used inside curves_from_code")
    ap.add_argument("--train-geoms", type=int, default=28,
                    help="number of camera geometries sampled per flow step")
    ap.add_argument("--out", default="runs/lpd_flow.pt")
    ap.add_argument("--codes-file", default="runs/corpus_codes.npz",
                    help="output of scripts/fit_shapes.py --out")
    ap.add_argument("--decoder-file", default="runs/token_decoder.pt",
                    help="output of scripts/fit_shapes.py --decoder")
    # --steps is the cap; training stops earlier when the held-out loss stops improving.
    ap.add_argument("--val-bodies", type=int, default=8,
                    help="bodies held out of training to score early stopping on; 0 trains "
                         "the full --steps and keeps the final weights")
    ap.add_argument("--val-every", type=int, default=200,
                    help="steps between held-out evaluations")
    ap.add_argument("--patience", type=int, default=5,
                    help="consecutive evaluations without improvement before stopping; "
                         "0 evaluates and checkpoints but never stops early")
    ap.add_argument("--min-delta", type=float, default=1e-4,
                    help="held-out loss must drop by at least this much to count as an "
                         "improvement")
    # --steps is a total across jobs, not a per-job budget: a run resumed from a checkpoint
    # trains up to the same cap, so a 300-step slot chips away at it a slot at a time.
    ap.add_argument("--ckpt-every", type=int, default=100,
                    help="steps between resumable checkpoints; 0 disables them (the run "
                         "then has to finish in one job to leave anything behind)")
    ap.add_argument("--ckpt-file", default=None,
                    help="where the resumable checkpoint goes; defaults to <--out>.ckpt. "
                         "Keep it on persistent storage -- a worker's /tmp does not "
                         "survive the job")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="pick training back up from --ckpt-file when it exists "
                         "(--no-resume starts from step 0 and overwrites it)")
    ap.add_argument("--log-every", type=int, default=10,
                    help="steps between training-loss lines")
    ap.add_argument("--cache-tag", default="shared",
                    help="distinguishes the /tmp curve cache between runs that use the "
                         "same --phases but different --codes-file/--bodies; the cache key "
                         "otherwise ignores both, so a small test run and a production run "
                         "at the same --phases would silently read each other's curves")
    a = ap.parse_args()
    dev = "cpu"                     # extraction runs on CPU; the nets are small
    psi = psi_grid(a.phases)

    gdev = "cuda" if torch.cuda.is_available() else "cpu"
    surro = Surrogate(width=96, modes=8, blocks=3)
    sd = Path("runs/surrogate.pt")
    if sd.exists():
        surro.load_state_dict(torch.load(sd, map_location="cpu"))
        print("  loaded the trained surrogate", flush=True)
    surro = surro.to(gdev).eval()
    print(f"  surrogate on {gdev}; operator covers all {len(cameras())} geometries",
          flush=True)

    print(f"[{_now()}] [stage 1] corpus", flush=True)
    codes, curves, sup = corpus(
        a.bodies, psi, surro, dev, codes_file=a.codes_file, decoder_path=a.decoder_file,
        cache=f"/tmp/lpd_corpus_{a.phases}_g{len(cameras())}_"
              f"res{a.operator_res}_n{DESIGN_N}_{a.cache_tag}.npz",
        op_res=a.operator_res)
    print(f"  corpus: codes {tuple(codes.shape)}, curves {tuple(curves.shape)}", flush=True)

    print(f"[{_now()}] [stage 2] flow", flush=True)
    net = LPDFlow()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    C = 28
    M = min(N_MODES, a.phases // 2)
    if M < N_MODES:
        print(f'  WARNING: only {M} modes available at {a.phases} phases; '
              f'{N_MODES} are required', flush=True)
    tag = torch.zeros(1, C, 4)
    for i, cam in enumerate(cameras()):
        tag[0, i] = torch.tensor([np.cos(np.radians(cam.azimuth_deg)),
                                  np.sin(np.radians(cam.azimuth_deg)),
                                  np.sin(np.radians(cam.elevation_deg)), 1.0])
    mask = torch.ones(1, C)
    train_geoms = max(1, min(C, int(a.train_geoms)))
    print(f"  operator extraction res {a.operator_res}; training samples "
          f"{train_geoms}/{C} geometries per step", flush=True)

    # The held-out bodies are drawn from a fixed permutation, so the split is the same on a
    # resumed or repeated run over the same corpus, and a body never scores the network that
    # trained on it.
    perm = torch.randperm(len(codes), generator=torch.Generator().manual_seed(0))
    n_val = max(0, min(a.val_bodies, len(codes) - 1))
    if n_val < a.val_bodies:
        print(f"  WARNING: corpus has {len(codes)} bodies; holding out {n_val} for "
              f"validation instead of {a.val_bodies}", flush=True)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if n_val:
        # Fixed noise and fixed step times: the same draws at every evaluation, so a change
        # in the score is a change in the network. The times cycle through the schedule
        # rather than being sampled, which is the low-variance form of the training estimator.
        val_x0 = torch.randn(n_val, codes.shape[1], dtype=codes.dtype,
                             generator=torch.Generator().manual_seed(1234))
        val_k = torch.arange(n_val) % N_STEPS
        print(f"  {len(train_idx)} training bodies, {n_val} held out; validating every "
              f"{a.val_every} steps, patience {a.patience}", flush=True)
    else:
        print("  no held-out bodies: training the full --steps, keeping the final weights",
              flush=True)

    best, best_state, best_step, stale = float("inf"), None, -1, 0
    stopped_at = a.steps
    start_step, elapsed_before = 0, 0.0
    ckpt_path = a.ckpt_file or f"{a.out}.ckpt"
    meta = {
        "dim": int(codes.shape[1]),
        "n_val": n_val,
        "phases": int(a.phases),
        "operator_res": int(a.operator_res),
        "train_geoms": train_geoms,
    }

    # A job that dies at step 250 of 300 should not cost the 250 steps it already paid for.
    if a.resume and Path(ckpt_path).exists():
        stale_ckpt = any(Path(f).exists() and Path(f).stat().st_mtime
                         > Path(ckpt_path).stat().st_mtime
                         for f in (a.codes_file, a.decoder_file))
        st = None if stale_ckpt else torch.load(ckpt_path, map_location="cpu",
                                                weights_only=False)
        if stale_ckpt:
            # Weights trained against codes that have since been refitted are worse than no
            # weights: they would be reported as progress towards a corpus they never saw.
            print(f"  WARNING: {ckpt_path} predates {a.codes_file} -- ignoring it and "
                  f"training from step 0", flush=True)
        else:
            bad = [f"{k}: checkpoint={st.get(k)!r}, current={v!r}"
                   for k, v in meta.items() if st.get(k) != v]
            if bad:
                raise SystemExit(
                    f"{ckpt_path} was written for different flow settings "
                    f"({'; '.join(bad)}). Delete it or pass --no-resume.")
            net.load_state_dict(st["net"])
            opt.load_state_dict(st["opt"])
            torch.set_rng_state(st["rng"])
            start_step = st["step"] + 1
            best, best_step, stale = st["best"], st["best_step"], st["stale"]
            best_state = st["best_state"]
            elapsed_before = st.get("elapsed", 0.0)
            print(f"  [{_now()}] resumed {ckpt_path} at step {start_step} of {a.steps} "
                  f"({_hms(elapsed_before)} trained so far; best val "
                  f"{best:.5f} from step {best_step}, {stale}/{a.patience} without "
                  f"improvement)", flush=True)
    if start_step >= a.steps:
        print(f"  the checkpoint is already at the --steps cap ({a.steps}); nothing left "
              f"to train -- raise --steps to continue", flush=True)

    t_run = t_step = time.time()
    for s in range(start_step, a.steps):
        idx = train_idx[torch.randint(0, len(train_idx), (a.batch,))]
        x0 = torch.randn(a.batch, codes.shape[1], dtype=codes.dtype)
        k = torch.randint(0, N_STEPS, (a.batch,))          # unbiased single-k estimator
        loss = flow_loss(net, surro, psi, codes, curves, sup, idx, x0, k, M, tag, mask,
                         a.decoder_file, op_res=a.operator_res,
                         train_geoms=train_geoms)
        opt.zero_grad(); loss.backward(); opt.step()
        now = time.time()
        step_s = now - t_step
        elapsed = elapsed_before + (now - t_run)
        if (a.log_every and s % a.log_every == 0) or s == a.steps - 1:
            # Wall clock, seconds per step and a projection to the cap: on a batch worker
            # with a wall-clock limit, what matters is whether the remaining steps fit in
            # the slot, and that is not something a bare loss line can answer.
            rate = (now - t_run) / (s - start_step + 1)
            print(f"  [{_now()}] step {s:>5}  flow loss {float(loss.detach()):.5f}  "
                  f"{step_s:.1f}s/step  elapsed {_hms(elapsed)}  "
                  f"eta {_hms(rate * (a.steps - s - 1))}", flush=True)

        stop = False
        if n_val and ((s + 1) % a.val_every == 0 or s == a.steps - 1):
            t_val = time.time()
            vl = validate(net, surro, psi, codes, curves, sup, val_idx, val_x0, val_k, M,
                          tag, mask, a.decoder_file, a.batch, op_res=a.operator_res)
            if vl < best - a.min_delta:
                best, best_step, stale = vl, s, 0
                best_state = {k_: v.detach().clone() for k_, v in net.state_dict().items()}
                print(f"  [{_now()}] step {s:>5}  val {vl:.5f}  (best, "
                      f"{time.time()-t_val:.0f}s)", flush=True)
            else:
                stale += 1
                print(f"  [{_now()}] step {s:>5}  val {vl:.5f}  (no improvement on "
                      f"{best:.5f} from step {best_step}, {stale}/{a.patience}, "
                      f"{time.time()-t_val:.0f}s)", flush=True)
                if a.patience and stale >= a.patience:
                    stopped_at = s + 1
                    stop = True

        # After the evaluation, so the checkpoint carries the best state it just found.
        if a.ckpt_every and (stop or (s + 1) % a.ckpt_every == 0 or s == a.steps - 1):
            save_checkpoint(ckpt_path, net, opt, s, best, best_state, best_step, stale,
                            elapsed_before + (time.time() - t_run), meta)
            print(f"  [{_now()}] step {s:>5}  checkpointed to {ckpt_path}", flush=True)
        if stop:
            print(f"  early stop at step {s}: {stale} evaluations without improvement",
                  flush=True)
            break
        t_step = time.time()

    if best_state is not None:
        # The last weights are not the best ones once the score has been climbing back --
        # that is the whole point of watching it, so restore the best before saving.
        net.load_state_dict(best_state)
        print(f"  restored step {best_step} (val {best:.5f}) after {stopped_at} steps",
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), a.out)
    print(f"[{_now()}] wrote {a.out} after {_hms(elapsed_before + (time.time() - t_run))} "
          f"of training", flush=True)


if __name__ == "__main__":
    main()
