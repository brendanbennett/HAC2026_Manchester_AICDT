#!/usr/bin/env python3
"""Train the LPD as a conditional flow.

Stage 1 builds the corpus. Each training body is fitted by the implicit field -- the support h
plus the lattice amplitudes g -- by regressing the field onto the body's signed distance. That
fit supplies x1. Fitting needs no mesh extraction, only field evaluations at sampled points,
so it is cheap.

Stage 2 trains the flow. For a draw x0 ~ N(0, I) and a time t, the state is
x_t = (1-t) x0 + t x1, the operator is applied to x_t, the dual network reduces the residual
against the data, and the primal network predicts the velocity, whose target is x1 - x0 at
every t.

t is continuous and stratified across the batch, which is an O(1/B^2) estimator of the same
integral instead of O(1/B) -- one operator call per draw either way, so it is free.

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
from hac26.field import (CODE_DIM, DESIGN_N, LATTICE_EXTENT, LATTICE_SHAPE,   # noqa: E402
                         N_DIR, N_SITES, GaussianLattice, ImplicitBody, design_sha,
                         dir_design, extract_mesh, sh_expand, spherical_design)
from hac26.solvers.lpd_flow import N_MODES, N_STEPS, LPDFlow           # noqa: E402
from hac26.forward.mesh.radiosity import facet_geometry                               # noqa: E402
from hac26.forward.learned_surrogate import (Surrogate, camera_features,                 # noqa: E402
                             sun_features)                               # noqa: E402


def corpus_cache_path(phases, n_geoms, op_res, tag) -> str:
    """The corpus cache filename. ONE definition, imported by ablate_flow.py, because it used
    to be spelled out in four places -- two Python files and two shell scripts, one of them
    with the geometry count hardcoded -- and none of them encoded the code length. A stale
    608-dimensional cache therefore passed every check and died inside the primal."""
    return (f"/tmp/lpd_corpus_{phases}_g{n_geoms}_res{op_res}_n{DESIGN_N}_"
            f"c{CODE_DIM}_{tag}.npz")


def _corpus_meta(n, psi, op_res) -> dict:
    return {
        "schema": 4,
        "bodies": int(n),
        "phases": int(len(psi)),
        "n_geoms": int(len(cameras())),
        "operator_res": int(op_res),
        "design_n": int(DESIGN_N),
        "code_dim": int(CODE_DIM),
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


DH_EPS_STD = 0.02      # std of the training-time perturbation of h, in units of R. The
                       # measured dh a fitted body needs is 0.0199 R, so this is the scale the
                       # flow has to learn to undo -- and because eps is redrawn every draw it
                       # doubles as augmentation at no operator cost.


def _dh_perturbation(n: int, generator=None) -> torch.Tensor:
    """Band-limited dh perturbations, one row per draw, drawn on the N_DIR directions.

    Band-limited by construction: an out-of-band dh kills facets, and a dead facet has an
    exactly zero row in the area Jacobian, i.e. no gradient at all rather than a bad one.
    Measured facet death from white dh at 2% R: 3% at N=64, 7% at 128, 27% at 256, 53% at 512.
    """
    global _DH_SELF
    if _DH_SELF is None:
        d = dir_design(N_DIR)
        _DH_SELF = torch.from_numpy(sh_expand(d, d))
    raw = torch.randn(n, N_DIR, generator=generator)
    band = raw @ _DH_SELF.T
    band = band / band.std(dim=1, keepdim=True).clamp_min(1e-8) * DH_EPS_STD
    return band


_DH_SELF = None


def perturb_support(h: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """softplus(inv_softplus(h) + expand(eps)) -- positivity automatic, no clamp needed.

    Exactly the operation ImplicitBody.support() performs with dh, so what the flow learns to
    emit at reconstruction is the same object it was supervised on here.
    """
    global _EXPAND_CACHE
    if _EXPAND_CACHE is None:
        _EXPAND_CACHE = torch.from_numpy(sh_expand(dir_design(N_DIR), spherical_design(DESIGN_N)))
    e = _EXPAND_CACHE.to(h.device)
    hh = h.clamp_min(1e-6)
    raw = hh + torch.log(-torch.expm1(-hh))            # stable inverse softplus
    return torch.nn.functional.softplus(raw + eps.to(h.device) @ e.T)


_EXPAND_CACHE = None


_A_DIR = None


def support_residual_channel(r_phase: torch.Tensor, n_phases: int,
                             c_lambert: float = 0.1) -> torch.Tensor:
    """A^T r, on the dh directions: which support directions the residual implicates.

    `build_A` takes ARBITRARY normals, so the operator is built directly on dir_design(N_DIR)
    rather than on the default 1152-cell lat-lon NormalGrid. That matters: a design pins the
    six coordinate axes and has cells at neither the equator nor the poles, and on flat-faced
    bodies -- which every ground truth is -- it is exact where the grid is 13.2% off.

    WHAT THIS IS AND IS NOT. It is the transpose of the convex photometric operator. It does
    NOT include the normalisation Jacobian DN, and it does NOT include J = d(areas)/dh, so it
    is not the exact gradient of the misfit with respect to h. Both would need the current
    body's polytope and areas rebuilt per sample; J alone is a 4096x4096 object per body. The
    direction information -- which is the whole job of a conditioning channel -- survives
    without them, and the branch whitens and re-scales its own input anyway.

    Read it as a hint, not a gradient. A_conv is provably blind to concavity: a notched cube's
    unshadowed curves are reproduced exactly, as a fatter box, and 90% of its row-space power
    sits at l <= 3. It can say which face is wrong; it can never say what shape the dent is.
    That is exactly why this is a conditioning input and is never added to the velocity.
    """
    global _A_DIR
    if _A_DIR is None or _A_DIR[1] != n_phases:
        # hac26.geometry.Camera, not hac26.conventions.Camera: build_A calls cam.omega(),
        # which only the geometry one has. The two carry the same 28 azimuth/elevation pairs.
        from hac26.forward.convex_egi import build_A
        from hac26.geometry import build_cameras
        cams = list(build_cameras()) + list(build_cameras())
        types = ["intensity"] * len(build_cameras()) + ["binary"] * len(build_cameras())
        A = build_A(dir_design(N_DIR), cams, n_phases, types, c_lambert=c_lambert)
        _A_DIR = (torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32)), n_phases)
    A = _A_DIR[0].to(r_phase.device)                       # (56, P, N_DIR)
    stacked = torch.cat([r_phase[:, :, 0], r_phase[:, :, 1]], 1)      # (B, 56, P)
    a = torch.einsum("cpn,bcp->bn", A, stacked)
    # whitened per sample: the raw magnitude is order 1e2 and varies with the residual scale,
    # which would otherwise dominate the branch's first layer
    return a / a.std(dim=1, keepdim=True).clamp_min(1e-8)


def cond_channels(support: torch.Tensor, device=None):
    """The per-body conditioning the two branches see, built once per body.

    Sphere branch (B, N_DIR, 5): the base support resampled onto the dh directions, the three
    components of the direction itself, and a slot for the aligned adjoint channel.

    Volume branch (B, 5, nx, ny, nz): the convex core's signed distance AT THE LATTICE SITES,
    the inside indicator, and normalised x, y, z. `core_sdf` is what replaces culling -- the
    index set has to stay fixed for the code to mean anything, so a site deep inside or far
    outside cannot be removed, but it can be identified.
    """
    global _DIR_CACHE
    dev = device or support.device
    sup = support if support.dim() == 2 else support[None]
    B = sup.shape[0]
    if _DIR_CACHE is None:
        d = dir_design(N_DIR)
        nrm = spherical_design(DESIGN_N)
        _DIR_CACHE = (torch.from_numpy(d.astype(np.float32)),
                      torch.from_numpy(sh_expand(nrm, d)),
                      GaussianLattice().p)
    dirs, to_dir, sites = (t.to(dev) for t in _DIR_CACHE)
    h_dir = sup.to(dev) @ to_dir.T                                    # (B, N_DIR)
    sph = torch.cat([h_dir[..., None],
                     dirs[None].expand(B, -1, -1),
                     torch.zeros(B, N_DIR, 1, device=dev)], -1)       # (B, N_DIR, 5)

    core = ImplicitBody(radius=1.0).core.to(dev)
    sdf = []
    for b in range(B):
        sdf.append(core(sites, h=sup[b].to(dev)))
    sdf = torch.stack(sdf)                                            # (B, N_SITES)
    xyz = (sites / LATTICE_EXTENT).T[None].expand(B, -1, -1)          # (B, 3, N_SITES)
    vol = torch.cat([sdf[:, None], (sdf < 0).float()[:, None], xyz], 1)
    return sph, vol.reshape(B, 5, *LATTICE_SHAPE)


_DIR_CACHE = None


def code_of(body: ImplicitBody) -> torch.Tensor:
    """RAW code: [dh on N_DIR directions, g on N_SITES lattice amplitudes]."""
    return torch.cat([body.dh.detach().reshape(-1), body.delta.g.detach().reshape(-1)])


def set_code(body: ImplicitBody, code: torch.Tensor) -> None:
    """Install a RAW code. The caller decodes from the flow's whitened space first."""
    if code.numel() != CODE_DIM:
        raise ValueError(f"code has {code.numel()} entries, expected CODE_DIM={CODE_DIM} "
                         f"= N_DIR {N_DIR} + N_SITES {N_SITES}")
    with torch.no_grad():
        body.dh.copy_(code[:N_DIR].reshape_as(body.dh))
        body.delta.g.copy_(code[N_DIR:].reshape_as(body.delta.g))


def curves_from_code(code, radius, surro, psi, res=32, device=None, geoms=None,
                     chunk=4, support=None):
    """A(x): decode a RAW code to a body, extract, tokenise, and run the surrogate.

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
    # h is the BASE support -- the convex stage's answer at reconstruction, the fitted corpus
    # support during training. The code's dh block corrects it, inside the softplus.
    n_norm = body.core.n.shape[0]
    if support is None:
        h = torch.full((n_norm,), 0.8 * radius, dtype=torch.float32, device=device)
    else:
        h = torch.as_tensor(support, dtype=torch.float32, device=device)
        if h.numel() != n_norm:
            raise ValueError(f"support has {h.numel()} entries, but this field uses "
                             f"{n_norm} normals; rerun fit_shapes.py after changing "
                             "DESIGN_N")
    body.set_support(h)
    set_code(body, code.to(device))
    ext = radius * 1.6
    v, f = extract_mesh(lambda y: body(y), ext, res=res, device=device)
    if len(f) < 8:
        return None
    # Decimate to the facet count the surrogate was trained at; ray tracing scales as
    # facets x phases.
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
           codes_file="runs/corpus_codes.npz"):
    """Curves for the corpus, from codes fitted by scripts/fit_shapes.py.

    The codes come from scripts/fit_shapes.py. There is no decoder to load: with a fixed
    lattice, site k means the same place for every body, so a code is portable by construction
    and the operator only has to apply it. Run fit_shapes.py first.
    """
    if not Path(codes_file).exists():
        raise SystemExit(f"{codes_file} missing -- run scripts/fit_shapes.py first")
    zz = np.load(codes_file)
    if "codes" not in zz.files or "support" not in zz.files:
        raise SystemExit(f"{codes_file} must contain 'codes' and 'support' arrays")
    if len(zz["codes"]) < n:
        raise SystemExit(f"{codes_file} contains {len(zz['codes'])} bodies, "
                         f"but --bodies requested {n}")
    if zz["codes"].ndim != 2 or zz["codes"].shape[1] != CODE_DIM:
        raise SystemExit(f"{codes_file} codes have shape {zz['codes'].shape}, but "
                         f"CODE_DIM={CODE_DIM}; rerun scripts/fit_shapes.py")
    if zz["support"].ndim != 2 or zz["support"].shape[1] != DESIGN_N:
        raise SystemExit(f"{codes_file} support has shape {zz['support'].shape}, "
                         f"but DESIGN_N={DESIGN_N}; rerun fit_shapes.py")
    # h is indexed BY NORMAL, so matching lengths is not enough: a design generated
    # independently on another machine has the same n and different points, and pairing the
    # two silently reindexes every body. Older corpora carry no digest; do not reject those.
    codes_meta = _decode_meta(zz) or {}
    for key, live, what in (("design_sha", lambda: design_sha(spherical_design()),
                             f"hac26/design{DESIGN_N}.npy"),
                            ("dir_sha", lambda: design_sha(dir_design(N_DIR)),
                             f"hac26/design{N_DIR}.npy, the dh directions")):
        if key in codes_meta:
            here = live()
            if codes_meta[key] != here:
                raise SystemExit(
                    f"{codes_file} was fitted against {key} {codes_meta[key]}, but {what} "
                    f"is {here}. Support and dh are indexed BY DIRECTION, so these cannot be "
                    f"mixed. Use the design the corpus was built with, or rerun "
                    f"scripts/fit_shapes.py.")
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
        cur = curves_from_code(code, 1.0, surro, psi, res=op_res, support=h)
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


def flow_loss(net, surro, psi, codes, curves, sup, idx, x0, t, M, tag, mask,
              op_res=32, train_geoms=None, eps=None, return_diag=False,
              ablate=False):
    """The flow-matching loss for one batch, given the draws (idx, x0, t, eps).

    Split out of the training loop so validation scores the same objective through the same
    code path -- operator included. The only difference on the validation side is that the
    draws are fixed instead of resampled, which is what makes two evaluations comparable:
    a fresh x0 and t per evaluation would move the loss by more than the training does.

    THE dh SUPERVISION. A corpus body's h is exact, so its dh block is zero and there is
    nothing there to learn from. Instead h is PERTURBED and the perturbation becomes its own
    target: the operator runs at softplus(inv_softplus(h) + eps), so the residual it returns
    genuinely reflects an h error, and the flow's target for that block is -eps. No new
    pipeline stage, no extra operator call, and eps doubles as free augmentation -- which is
    what makes 600 bodies enough to teach a 128-dimensional support correction.

    Everything here is in the CODEC's whitened space. x0 ~ N(0, I) only means something once
    the code has been transformed to match it: raw dh has std 0.0199 and raw g has std 0.0483,
    about fifty times narrower, and g has kurtosis 14.5 whose tail IS the deep carves.

    `ablate` additionally scores the same draws with every curve channel switched off, and
    returns (loss, ablated_loss, n_degenerate). One operator call serves both, so the two
    arms differ by the switch and by nothing else. Off by default; the training path does
    not enter the branch.
    """
    if ablate and return_diag:
        raise ValueError("flow_loss: ablate and return_diag return different tuples; "
                         "ask for one or the other")
    B, C = len(idx), tag.shape[1]
    dev = codes.device
    if eps is None:
        eps = _dh_perturbation(B).to(dev)
    x1_raw = codes[idx].clone()
    x1_raw[:, :N_DIR] = -eps                    # the correction that undoes the perturbation
    x1 = net.codec.encode(x1_raw)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1

    g_dat = torch.fft.rfft(curves[idx], dim=-1)[..., 1:M + 1]      # (B, G, 2, M)
    geoms_t = None
    geoms = None
    step_mask = mask.expand(B, C)
    if train_geoms is not None:
        n_geoms = max(1, min(C, int(train_geoms)))
        if n_geoms < C:
            geoms_t = torch.randperm(C)[:n_geoms].sort().values
            geoms = geoms_t.tolist()
            step_mask = torch.zeros(B, C, device=dev)
            step_mask[:, geoms_t] = 1.0

    # h the operator actually runs at: the corpus support, perturbed.
    h_pert = perturb_support(sup[idx].to(dev), eps)
    sph, vol = cond_channels(h_pert, device=dev)
    # THE OPERATOR IS APPLIED AT x1_hat = x_t + (1-t) v0(x_t), NOT AT x_t.
    #
    # The velocity is constant along a straight path, so that extrapolation is the model's own
    # estimate of the endpoint. It matters because x_t at small t is mostly x0, and x0 is a
    # standard Gaussian in the codec's space: decoded, its tail draws are amplitudes several
    # times anything the corpus contains. A residual measured at a body like that says almost
    # nothing about the body being reconstructed. v0 is a first pass with the residual
    # features ZEROED -- it costs one network forward and NO operator call, which is the
    # expensive part -- so the correction is free in the only currency that matters here.
    with torch.no_grad():
        v0 = net.velocity(xt, torch.zeros(B, C, N_MODES, 6, device=dev),
                          tag.expand(B, C, 4), step_mask, t, sph, vol)
        x1_hat = xt + (1 - t[:, None]) * v0
    xt_raw = net.codec.decode(x1_hat)
    preds, n_bad = [], 0
    for b in range(B):
        cur = curves_from_code(xt_raw[b], 1.0, surro, psi,
                               res=op_res, geoms=geoms, support=h_pert[b])
        pred = torch.zeros_like(curves[0])
        if cur is None:
            n_bad += 1                  # decoded to under 8 faces; A(x) is unavailable
        else:
            if geoms_t is None:
                pred = cur
            else:
                pred[geoms_t] = cur
        preds.append(pred)
    pred_stack = torch.stack(preds)
    g_cur = torch.fft.rfft(pred_stack, dim=-1)[..., 1:M + 1]
    r = g_dat - g_cur                                   # (B, G, 2, M) complex
    # the same residual in PHASE space, which is the space the convex operator lives in
    sph = sph.clone()
    sph[..., 4] = support_residual_channel(curves[idx].to(dev) - pred_stack, curves.shape[-1])
    feats = torch.zeros(B, C, N_MODES, 6, device=dev)
    for ch in range(2):                                 # EVERY geometry, not slot 0
        feats[:, :, :M, 2 * ch] = r[:, :, ch].real
        feats[:, :, :M, 2 * ch + 1] = r[:, :, ch].imag
    feats[:, :, :M, 4] = g_dat[:, :, 0].real
    feats[:, :, :M, 5] = g_dat[:, :, 1].real

    def _score(v):
        # Per-block weights. Unweighted, g's 1728 dimensions take about 89% of the
        # gradient and the 128-dimensional support block -- the one the operator can
        # actually see -- gets the rest. Weighting by 1/N_block makes the two blocks
        # contribute equally per block.
        err = (v - (x1 - x0)) ** 2
        return 0.5 * (err[:, :N_DIR].mean() + err[:, N_DIR:].mean())

    u = net.velocity(xt, feats, tag.expand(B, C, 4), step_mask, t, sph, vol)
    loss = _score(u)
    if ablate:
        # THE ABLATION ARM (scripts/ablate_flow.py), scored off the SAME operator call so
        # the two arms cannot differ by anything except the switch. Both paths the curves
        # take, not just the dual's: sph[..., 4] is the adjoint channel, and it feeds the
        # sphere branch, which is the branch that emits the dh block.
        sph0 = sph.clone(); sph0[..., 4] = 0.0
        u0 = net.velocity(xt, torch.zeros_like(feats), tag.expand(B, C, 4), step_mask,
                          t, sph0, vol)
        return loss, _score(u0), n_bad
    if not return_diag:
        return loss
    with torch.no_grad():
        g_hat = net.codec.decode(xt + (1 - t[:, None]) * u)[:, N_DIR:]
        diag = (float(g_hat.abs().mean()),
                float(g_hat.std(0).mean()) if len(g_hat) > 1 else float("nan"))
    return loss, diag


def validate(net, surro, psi, codes, curves, sup, val_idx, val_x0, val_t, val_eps, M, tag,
             mask, chunk, op_res=32):
    """Mean flow loss over the held-out bodies at fixed draws, plus a collapse diagnostic.

    Chunked to bound memory. Returns (loss, (mean |g| of the endpoint estimate, its spread
    across draws)) -- both computed from the velocity this call already produced, so they are
    free, and both are what a collapse to the conditional mean would move first.
    """
    was_training = net.training
    net.eval()
    tot, n, dg, ds = 0.0, 0, 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(val_idx), chunk):
            sl = slice(i, i + chunk)
            b = len(val_idx[sl])
            l, d = flow_loss(net, surro, psi, codes, curves, sup, val_idx[sl],
                             val_x0[sl], val_t[sl], M, tag, mask,
                             op_res=op_res, eps=val_eps[sl], return_diag=True)
            tot += b * float(l); dg += b * d[0]
            ds += b * (0.0 if d[1] != d[1] else d[1])
            n += b
    net.train(was_training)
    m = max(n, 1)
    return tot / m, (dg / m, ds / m)


def _hms(sec: float) -> str:
    """Seconds as h:mm:ss, for lines a human reads while a job is running."""
    sec = int(max(sec, 0.0))
    return f"{sec // 3600}:{(sec // 60) % 60:02d}:{sec % 60:02d}"


def _now() -> str:
    return time.strftime("%H:%M:%S")


class EMA:
    """Exponential moving average of the weights, with a warm-up correction.

    The correction matters at the start: an average initialised at the init weights is biased
    towards them for roughly 1/(1-decay) steps, which at 0.999 is a thousand -- longer than a
    budgeted run. Dividing by (1 - decay^n) removes that exactly, the same correction Adam
    applies to its own moments.

    The buffer therefore starts at ZERO, not at the init weights. The two conventions are
    mutually exclusive and mixing them is not a small error: correcting a buffer that already
    started at w0 divides by 1 - decay^n, which is 0.002 at n = 2, and the weights blow up by
    500x into NaN on the first validation.
    """

    def __init__(self, net, decay: float = 0.999):
        self.decay = float(decay)
        self.n = 0
        self.shadow = {k: torch.zeros_like(v, dtype=torch.float32)
                       for k, v in net.state_dict().items() if v.dtype.is_floating_point}

    def update(self, net):
        self.n += 1
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                         alpha=1.0 - self.decay)

    def state(self, net):
        """Bias-corrected weights, in the layout state_dict() wants."""
        if self.n == 0:
            return {k: v.detach().clone() for k, v in net.state_dict().items()}
        c = 1.0 - self.decay ** self.n
        out = {}
        for k, v in net.state_dict().items():
            out[k] = (self.shadow[k] / c).to(v.dtype) if k in self.shadow else v.detach().clone()
        return out

    def load(self, d, n):
        self.shadow = {k: v.detach().clone().float() for k, v in d.items()}
        self.n = int(n)


class _Swapped:
    """Run a block with the EMA weights installed, then put the raw ones back."""

    def __init__(self, net, ema):
        self.net, self.ema = net, ema

    def __enter__(self):
        self.saved = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(self.ema.state(self.net))

    def __exit__(self, *exc):
        self.net.load_state_dict(self.saved)
        return False


def save_checkpoint(path, net, opt, step, best, best_state, best_step, stale, elapsed,
                    meta, ema=None):
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
    # The EMA shadow goes in too. Without it a resume silently restarts the average, so the
    # weights that get shipped are an average over the wrong window.
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step,
                "best": best, "best_state": best_state, "best_step": best_step,
                "stale": stale, "elapsed": elapsed, "rng": torch.get_rng_state(),
                "ema": None if ema is None else ema.shadow,
                "ema_n": 0 if ema is None else ema.n,
                **meta}, tmp)
    tmp.replace(p)


def _enable_tf32():
    """TF32 on the matmul path. The operator's field evaluation is (points x normals) and
    (points x sites) matmuls at about three decimal places of useful precision; TF32 keeps ten
    bits of mantissa, which is more than the surrogate's own accuracy, and is several times
    faster on any Ampere-or-later GPU. No effect on CPU or on older cards."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


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
    ap.add_argument("--ema", type=float, default=0.999,
                    help="EMA decay on the weights. Validation scores the averaged weights "
                         "and the saved checkpoint is those weights, so the model that is "
                         "selected is the model that was measured. 0 disables it.")
    ap.add_argument("--time-budget", type=float, default=20.0,
                    help="hours. After the first few steps the finish time is projected and "
                         "compared against this, and the largest --steps that would fit is "
                         "printed. It warns rather than exits: the run checkpoints and "
                         "resumes, so an overrun costs a restart, not the work.")
    ap.add_argument("--cache-tag", default="shared",
                    help="distinguishes the /tmp curve cache between runs that use the "
                         "same --phases but different --codes-file/--bodies; the cache key "
                         "otherwise ignores both, so a small test run and a production run "
                         "at the same --phases would silently read each other's curves")
    a = ap.parse_args()
    _enable_tf32()
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
        a.bodies, psi, surro, dev, codes_file=a.codes_file,
        cache=corpus_cache_path(a.phases, len(cameras()), a.operator_res, a.cache_tag),
        op_res=a.operator_res)
    print(f"  corpus: codes {tuple(codes.shape)}, curves {tuple(curves.shape)}", flush=True)

    print(f"[{_now()}] [stage 2] flow", flush=True)
    net = LPDFlow()
    # The codec is fitted from the corpus and lives IN the network, so it rides the state_dict
    # into every checkpoint and back out at reconstruction. Statistics kept anywhere else
    # would silently desync between training and inference.
    net.codec.fit(codes, eps_std=DH_EPS_STD)
    with torch.no_grad():
        z = net.codec.encode(codes)
    print(f"  codec: g scale {float(net.codec.g_s):.5f}, dh sd {float(net.codec.sd[0]):.5f}, "
          f"g sd {float(net.codec.sd[1]):.5f}; corpus in whitened space reaches "
          f"|z| = {float(z[:, N_DIR:].abs().max()):.2f}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    # EMA of the weights. There was none anywhere in the repo. The flow's velocity is a noisy
    # regression target -- one operator call per draw, one t per draw -- so the last iterate
    # is a worse estimate of the trained field than an average of the recent ones. Validation
    # scores the AVERAGED weights, not the raw ones: scoring one model and shipping another is
    # how early stopping ends up selecting a checkpoint nobody evaluated.
    # The window is tied to the RUN, not fixed. decay = 0.999 averages over 1/(1-d) = 1000
    # steps; on a 2500-step budget that is 40% of training, so the shipped weights lag deep
    # into the regime where the velocity is still close to E[x1 - x0], which transports every
    # draw towards the corpus MEAN code -- and a mean over bodies is smoother, hence more
    # convex, than any of them. Ten percent of the run keeps the averaging useful and the lag
    # proportionate.
    ema_decay = min(a.ema, 1.0 - 1.0 / max(a.steps / 10.0, 10.0)) if a.ema else 0.0
    ema = EMA(net, decay=ema_decay)
    if a.ema:
        print(f"  EMA decay {ema_decay:.5f} (window ~{1/(1-ema_decay):.0f} steps of "
              f"{a.steps})", flush=True)
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
    g_ref = float(codes[:, N_DIR:].abs().mean())
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
        gen = torch.Generator().manual_seed(1234)
        val_x0 = torch.randn(n_val, codes.shape[1], dtype=codes.dtype, generator=gen)
        # Times stratified over [0,1), not sampled and not snapped to the six step times: the
        # flow is now trained at continuous t, and validation has to score the same objective.
        val_t = (torch.arange(n_val, dtype=codes.dtype) + 0.5) / max(n_val, 1)
        val_eps = _dh_perturbation(n_val, generator=gen)
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
        "n_modes": int(N_MODES),        # `modes` is a buffer and IS in the state_dict, so a
        "n_steps": int(N_STEPS),        # mismatch would otherwise surface as a bare
        "loss": "per_block_mse_v2",     # RuntimeError from load_state_dict rather than here
        "n_val": n_val,
        "phases": int(a.phases),
        "operator_res": int(a.operator_res),
        "train_geoms": train_geoms,
    }

    # A job that dies at step 250 of 300 should not cost the 250 steps it already paid for.
    if a.resume and Path(ckpt_path).exists():
        stale_ckpt = any(Path(f).exists() and Path(f).stat().st_mtime
                         > Path(ckpt_path).stat().st_mtime
                         for f in (a.codes_file,))
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
            if st.get("ema") is not None:
                ema.load(st["ema"], st.get("ema_n", 0))
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
        # Continuous t, STRATIFIED across the batch. Six discrete step times meant the network
        # only ever saw six points on its own trajectory. Stratifying gives an O(1/B^2)
        # estimator instead of O(1/B), which is what makes a larger batch superlinearly
        # better; the operator is called once per draw either way, so this costs nothing.
        t = ((torch.arange(a.batch, dtype=codes.dtype) + torch.rand(a.batch)) / a.batch)
        t = t[torch.randperm(a.batch)]
        loss = flow_loss(net, surro, psi, codes, curves, sup, idx, x0, t, M, tag, mask,
                         op_res=a.operator_res, train_geoms=train_geoms)
        opt.zero_grad(); loss.backward(); opt.step(); ema.update(net)
        now = time.time()
        step_s = now - t_step
        elapsed = elapsed_before + (now - t_run)
        if a.time_budget and s - start_step == 2:
            # Projected from three measured steps, not from a guess. Checked here rather than
            # at the end because the point is to fail while there is still time to act.
            per = (now - t_run) / 3.0
            proj = elapsed + per * (a.steps - s - 1)
            n_fit = int((a.time_budget * 3600.0 - elapsed) / max(per, 1e-9)) + s + 1
            print(f"  [budget] {per:.1f}s/step -> {_hms(proj)} projected for {a.steps} steps; "
                  f"{_hms(a.time_budget * 3600)} allowed. Largest --steps that fits: {n_fit}",
                  flush=True)
            if proj > a.time_budget * 3600.0:
                # A WARNING, not an exit. An earlier version refused to start, on the premise
                # that being killed on a deadline loses everything. It does not: this script
                # checkpoints every --ckpt-every steps and resumes from the optimiser state,
                # the RNG state and the early-stopping counters, and the corpus cache is
                # preserved to runs/ by a trap on EXIT/INT/TERM. A run that overruns is
                # therefore resumed, not lost -- and refusing to start it would have thrown
                # away the progress it would have made.
                print(f"  [budget] WARNING: {_hms(proj)} exceeds the budget by "
                      f"{_hms(proj - a.time_budget * 3600)}. This run will be cut short and "
                      f"resumed from {ckpt_path}; pass --steps {n_fit} if you would rather it "
                      f"finish inside one window.", flush=True)
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
            with _Swapped(net, ema):
                vl, diag = validate(net, surro, psi, codes, curves, sup, val_idx, val_x0,
                                    val_t, val_eps, M, tag, mask, a.batch,
                                    op_res=a.operator_res)
            if diag is not None:
                # A convexity-collapse tripwire, free: |g| of the model's own endpoint
                # estimate against the corpus. A flow that has regressed to the conditional
                # mean produces a code smaller and flatter than any real body, and a small
                # spread ACROSS draws means it is producing one body regardless of x0.
                print(f"  [{_now()}] step {s:>5}  |g|hat {diag[0]:.5f} vs corpus "
                      f"{g_ref:.5f} ({100*diag[0]/max(g_ref,1e-12):.0f}%), "
                      f"across-draw spread {diag[1]:.5f}", flush=True)
            if vl < best - a.min_delta:
                best, best_step, stale = vl, s, 0
                best_state = ema.state(net)      # ship the weights that were scored
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
                            elapsed_before + (time.time() - t_run), meta, ema=ema)
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
    elif a.ema:
        # Either there were no held-out bodies, or no evaluation ever improved. Either way
        # nothing selected a step, so ship the average rather than whatever the last gradient
        # happened to leave behind.
        net.load_state_dict(ema.state(net))
        print(f"  no best checkpoint was selected; keeping the EMA weights over {ema.n} "
              f"steps", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), a.out)
    print(f"[{_now()}] wrote {a.out} after {_hms(elapsed_before + (time.time() - t_run))} "
          f"of training", flush=True)


if __name__ == "__main__":
    main()
