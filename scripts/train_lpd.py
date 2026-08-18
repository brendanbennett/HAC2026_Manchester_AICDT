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

Public bodies are never in the corpus. They are the validation set and appear only at
reconstruction time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.conventions import S_LAB, cameras, psi_grid, to_body          # noqa: E402
from hac26.field import ImplicitBody, extract_mesh    # noqa: E402
from hac26.solvers.lpd_flow import N_MODES, N_STEPS, LPDFlow           # noqa: E402
from hac26.forward.mesh.radiosity import facet_geometry                               # noqa: E402
from hac26.forward.learned_surrogate import (Surrogate, camera_features,                 # noqa: E402
                             sun_features)                               # noqa: E402


_DECODER = {}


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
    h0 = np.array([max(1e-3, float((verts @ n).max()))
                   for n in body.core.n.cpu().numpy()], dtype=np.float32)
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
    body.core.set_support(torch.full((n_norm,), 0.8 * radius) if support is None
                          else torch.as_tensor(support, dtype=torch.float32).to(device))
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


def corpus(n, psi, surro, device, seed=0, cache=None,
           codes_file="runs/corpus_codes.npz", decoder_path="runs/token_decoder.pt"):
    """Curves for the corpus, from codes fitted by scripts/fit_shapes.py.

    The codes and the shared decoder come from the AUTODECODER fit, not from per-body fits.
    Fitting each body separately gives each its own q/k/v/mlp, and a code is then meaningless
    to any other decoder -- which is what made the token channel inert. Run fit_shapes.py
    first; this only applies the operator.
    """
    if cache and Path(cache).exists():
        z = np.load(cache)
        print(f"  loaded corpus of {len(z['codes'])} bodies", flush=True)
        return (torch.tensor(z["codes"]), torch.tensor(z["curves"]),
                torch.tensor(z["support"]))
    if not Path(codes_file).exists():
        raise SystemExit(f"{codes_file} missing -- run scripts/fit_shapes.py first")
    zz = np.load(codes_file)
    all_codes, all_sup = zz["codes"][:n], zz["support"][:n]
    codes, curves, sup = [], [], []
    for i in range(len(all_codes)):
        t0 = time.time()
        code = torch.tensor(all_codes[i]); h = torch.tensor(all_sup[i])
        cur = curves_from_code(code, 1.0, surro, psi, support=h, decoder_path=decoder_path)
        if cur is None:
            print(f"  body {i}: degenerate, skipped", flush=True); continue
        codes.append(all_codes[i]); curves.append(cur.numpy()); sup.append(all_sup[i])
        print(f"  body {i}: h {float(h.min()):.3f}-{float(h.max()):.3f}, "
              f"{time.time()-t0:.1f}s", flush=True)
    codes = np.stack(codes); curves = np.stack(curves); sup = np.stack(sup)
    if cache:
        np.savez(cache, codes=codes, curves=curves, support=sup)
    return torch.tensor(codes), torch.tensor(curves), torch.tensor(sup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=40)
    ap.add_argument("--steps", type=int, default=1500)
    # The dual uses m = 1..40, and an rFFT of n phases yields floor(n/2)+1
    # coefficients, so fewer than 81 phases cannot supply 40 modes. At 16 phases
    # only 8 exist and the dual silently ran at a fifth of its specified width.
    ap.add_argument("--phases", type=int, default=96)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default="runs/lpd_flow.pt")
    ap.add_argument("--codes-file", default="runs/corpus_codes.npz",
                    help="output of scripts/fit_shapes.py --out")
    ap.add_argument("--decoder-file", default="runs/token_decoder.pt",
                    help="output of scripts/fit_shapes.py --decoder")
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

    print("[stage 1] corpus", flush=True)
    codes, curves, sup = corpus(
        a.bodies, psi, surro, dev, codes_file=a.codes_file, decoder_path=a.decoder_file,
        cache=f"/tmp/lpd_corpus_{a.phases}_g{len(cameras())}_{a.cache_tag}.npz")
    print(f"  corpus: codes {tuple(codes.shape)}, curves {tuple(curves.shape)}", flush=True)

    print("[stage 2] flow", flush=True)
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
    for s in range(a.steps):
        idx = torch.randint(0, len(codes), (a.batch,))
        x1 = codes[idx]
        x0 = torch.randn_like(x1)
        k = torch.randint(0, N_STEPS, (a.batch,))          # unbiased single-k estimator
        t = k.float() / N_STEPS
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        # THE RESIDUAL, which means actually applying the operator to the current state.
        # The operator is applied to the current state x_t. Feeding only the data's Fourier
        # content would leave the network without any signal about where it currently is.
        g_dat = torch.fft.rfft(curves[idx], dim=-1)[..., 1:M + 1]      # (B, G, 2, M)
        preds = []
        for b in range(a.batch):
            cur = curves_from_code(xt[b].detach(), 1.0, surro, psi,
                                   support=sup[idx[b]], decoder_path=a.decoder_file)
            preds.append(torch.zeros_like(curves[0]) if cur is None else cur)
        g_cur = torch.fft.rfft(torch.stack(preds), dim=-1)[..., 1:M + 1]
        r = g_dat - g_cur                                   # (B, G, 2, M) complex
        # the convex operator explains the low orders best, so its complement is what the
        # profiling channel must carry: strip the first few m and keep the rest
        r_perp = r.clone()
        r_perp[..., :4] = 0
        feats = torch.zeros(a.batch, C, N_MODES, 6)
        perp = torch.zeros(a.batch, C, N_MODES, 6)
        for ch in range(2):                                 # EVERY geometry, not slot 0
            feats[:, :, :M, 2 * ch] = r[:, :, ch].real
            feats[:, :, :M, 2 * ch + 1] = r[:, :, ch].imag
            perp[:, :, :M, 2 * ch] = r_perp[:, :, ch].real
            perp[:, :, :M, 2 * ch + 1] = r_perp[:, :, ch].imag
        feats[:, :, :M, 4] = g_dat[:, :, 0].real
        feats[:, :, :M, 5] = g_dat[:, :, 1].real
        u = net.velocity(xt, feats, perp, tag.expand(a.batch, C, 4),
                         mask.expand(a.batch, C), t)
        loss = ((u - (x1 - x0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0 or s == a.steps - 1:
            print(f"  step {s:>5}  flow loss {float(loss):.5f}", flush=True)
    Path("model").mkdir(exist_ok=True)
    torch.save(net.state_dict(), a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
