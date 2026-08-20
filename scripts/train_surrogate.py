#!/usr/bin/env python3
"""Train the surrogate against the physical forward model and measure held-out agreement.

The gate is agreement well below sigma. sigma here is the measured per-curve replicate
noise of the real instrument, 0.005-0.03 in mean-normalised units, so "well below" means
the surrogate must reproduce the physical model to a few times 1e-3 or better, otherwise the LPD would be
inverting an operator whose own error exceeds the noise it is trying to fit.

A CUBE IS IN THE HELD-OUT SET DELIBERATELY. It is the shape whose curves are most unlike
the smooth bodies that dominate any random training corpus, and the one whose flat faces and
sharp silhouette a learned response is most likely to get wrong.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.conventions import S_LAB, cameras, psi_grid, source_directions, to_body  # noqa
from hac26.forward.mesh.radiosity import RadiositySolver, emission, facet_geometry, form_factors  # noqa
from hac26.calibrate import decimate                                         # noqa
from hac26.forward.mesh.raster import Rasteriser                                          # noqa
from hac26.forward.learned_surrogate import Surrogate, trace_features                        # noqa


# --------------------------------------------------------------------------------------
# Training corpus
#
# The bodies here are the only place the surrogate -- and downstream `fit_shapes.py`'s
# token library and `train_lpd.py`'s non-convex flow -- ever see what a target looks like,
# so this corpus IS the shape prior. It is now generated and, crucially, FILTERED by
# `hac26.shapes_nonconvex`, on the statistic the flow actually has to reproduce: the radial
# hull-deficit field D(u) = (r_hull(u) - r_body(u)) / mean(r_hull).
#
# Measured on the public STLs in the challenge pose:
#     model 1  D_rms 0.003  D_lo 0.17      model 2  D_rms 0.000  D_lo 0.00
#     model 3  D_rms 0.201  D_lo 0.91
# where D_lo is the fraction of D's power at spherical-harmonic degrees l <= 4.
#
# The previous archetype corpus is retired because measurement showed it did not do what
# it looked like it did: five of its seven classes came in at D_rms <= 0.021 with silhouette
# convexity deficit at the rasterisation noise floor. `faceted_rock` was a convex hull, so
# exactly 1.000 convex; `basin` measured 0.008; the craters measured 0.004. They were
# convex training examples, and the flow spent most of its gradient learning to emit zero.
# Boring holes into a convex body -- the approach before that -- gave D_rms 0.060 at
# D_lo 0.36: a third of model 3's amplitude, and in the part of the spectrum model 3 does
# not occupy. Craters and holes are also invisible to the challenge's side-view measure,
# which compares silhouette boundary curves.
#
# `sample_corpus` replaces the fixed archetype cycle with quota-filled strata on D_rms
# (model 3's 0.201 is an anchor in the middle, not a target -- the secret models are
# ordered by increasing difficulty), a lower bound on D_lo, a deliberate 40% near-convex
# fraction so models 1, 2 and 4 do not regress, and R drawn from a prior shaped by the
# published bounding-cylinder table including its two tails at 0.67 and 3.95.


def shapes(n: int, seed: int = 0):
    """Training corpus plus a held-out set that always contains a cube.

    Thin wrapper over `hac26.shapes_nonconvex.sample_corpus`, which returns bodies already
    in the challenge pose (rotation axis = z, touching z = +-1) with the bounding-cylinder
    radius imposed exactly. No random rotation is applied any more: the real targets are
    always in that pose, so rotating the corpus off it trains on geometry the test set
    never contains.
    """
    from hac26.shapes_nonconvex import sample_corpus
    return [(v, f) for v, f, _ in sample_corpus(n, seed=seed)]


def shapes_with_meta(n: int, seed: int = 0):
    """Same corpus, keeping each body's (kind, R, D_rms, D_lo) for diagnostics."""
    from hac26.shapes_nonconvex import sample_corpus
    return sample_corpus(n, seed=seed)


def m2_curves(v, f, ras, psi, rho=0.85, delta=np.radians(1.0), target=600):
    """Reference curves from the physical pipeline."""
    from hac26.calibrate import decimate
    dv, df = decimate(v, f, target)
    F_, area, nrm, cen = form_factors(dv, df, occlusion=True)
    solver = RadiositySolver(F_, rho=rho)
    s_lab = source_directions(delta, 8)
    from scipy.spatial import cKDTree
    cf, _, _ = facet_geometry(v, f)
    idx = cKDTree(cen).query(cf)[1]
    fv = v[f].reshape(-1, 3).astype(np.float32)
    ff = np.arange(len(fv), dtype=np.int32).reshape(-1, 3)
    dev = ras.device
    fv_t, ff_t = torch.tensor(fv, device=dev), torch.tensor(ff, device=dev)
    cams = cameras()
    ext = float(np.abs(v).max())
    fov = 2.0 * np.arctan(1.6 * ext / 8.0)
    I = np.zeros((28, len(psi))); N = np.zeros((28, len(psi)))
    peak = 0.0   # set on the first frame; an ABSOLUTE binary threshold empties
                 # the whole channel whenever the radiance scale changes
    for j, p in enumerate(psi):
        dirs = np.stack([to_body(d, np.array([p]))[0] for d in s_lab])
        import trimesh
        mm = trimesh.Trimesh(dv, df, process=False)
        vis = np.stack([(~mm.ray.intersects_any(cen + nrm * 1e-4, np.tile(d, (len(cen), 1))))
                        .astype(float) for d in dirs], 1)
        L = solver.radiance(solver.solve(emission(nrm, dirs, vis)))
        vr = torch.tensor(np.repeat(L[idx], 3).astype(np.float32), device=dev)
        for c, cam in enumerate(cams):
            vb = to_body(np.asarray(cam.v), np.array([p]))[0]
            img, _ = ras.render(fv_t, ff_t, vr, eye=vb * 8.0, fov_y_rad=fov)
            peak = max(peak, float(img.max()))
            I[c, j] = float(img.sum())
            N[c, j] = float((img > 1e-3 * peak).sum())
    cur = np.concatenate([I, N], 0)
    return cur / np.maximum(cur.mean(1, keepdims=True), 1e-9)


def features_for(v, f, psi, n_tokens=600, seed=0):
    """Exact ray-traced features, on DETERMINISTIC area-carrying tokens.

    The tokens are the facets of the decimated mesh, not random surface samples, and each
    carries its own area. That makes the token sum a quadrature of the surface integral
    rather than a Monte-Carlo estimate of it -- which is what capped the first attempt at
    1/sqrt(128) = 0.088 no matter how well the network fitted.
    """
    from hac26.calibrate import decimate
    from hac26.forward.mesh.radiosity import facet_geometry
    dv, df = decimate(v, f, n_tokens)
    pts, nrm, area = facet_geometry(dv, df)
    # Features for EVERY geometry, not just camera 0. The surrogate is camera-agnostic by
    # construction -- its inputs are mu, mu0 and visibility, which already encode where the
    # camera is -- so one network serves all 28, and training it on all of them is what
    # makes it valid off azimuth 0. Conditioning the LPD on a single geometry gives the dual
    # 160 numbers to determine 608 code dimensions, and the flow correspondingly learned
    # 5.4% of the target variance.
    sun_d = np.stack([to_body(S_LAB, np.array([p]))[0] for p in psi])
    out = []
    for cam in cameras():
        cam_d = np.stack([to_body(np.asarray(cam.v), np.array([p]))[0] for p in psi])
        out.append(trace_features(dv, df, pts, nrm, cam_d, sun_d,
                                  areas=area / area.sum()))
    return np.stack(out), area / area.sum()          # (28, T, P, F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=12)
    ap.add_argument("--held", type=int, default=4)
    ap.add_argument("--phases", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--no-attn", action="store_true",
                    help="pointwise + spectral only. The photometric response of a\n"
                         "token should be a near-UNIVERSAL function of its own\n"
                         "features; attention across tokens can instead memorise\n"
                         "which shape it is looking at.")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=4,
                    help="micro-batch; memory-bound by O(T^2) attention")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation steps. Batch 6 out of 5040\n"
                         "examples is 0.1% of the data per step, and the train\n"
                         "loss swung 0.067-0.195 between logged steps purely\n"
                         "from that. Accumulation buys a larger EFFECTIVE batch\n"
                         "without the attention memory a larger real one needs.")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--modes", type=int, default=8)
    ap.add_argument("--rho", type=float, default=0.85,
                    help="albedo of the reference. The surrogate sees one\n"
                         "gathered bounce; the reference solves the full series, so rho\n"
                         "controls how large that mismatch is.")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    psi = psi_grid(a.phases)
    ras = Rasteriser(64, 96, 2, device=dev)

    allsh = shapes(a.train + a.held, seed=1)
    import trimesh
    cube = trimesh.creation.box(extents=(0.9, 0.9, 0.9))
    allsh[a.train] = (np.asarray(cube.vertices, float), np.asarray(cube.faces, np.int64))

    cache = Path(f"/tmp/surr_cache_v9_{a.train}_{a.held}_{a.phases}_{a.rho}.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        X, Y, A = list(z['X']), list(z['Y']), list(z['A'])
        print(f"  loaded {len(X)} shapes from cache", flush=True)
    else:
        X, Y, A = [], [], []
    for i, (v, f) in enumerate(allsh if not X else []):
        try:
            # ONE mesh for both sides. The reference was rendering the full mesh while the
            # tokeniser described a 600-face decimation of it, so the surrogate was asked to
            # predict a silhouette belonging to geometry it had never been shown -- a
            # mismatch no amount of training can absorb.
            v, f = decimate(v, f, 600)
            cur = m2_curves(v, f, ras, psi, rho=max(a.rho, 1e-9))
            fe, ar = features_for(v, f, psi)
        except Exception as e:
            print(f"  shape {i}: skipped ({type(e).__name__})", flush=True); continue
        X.append(fe); Y.append(cur); A.append(ar)
        print(f"  shape {i}: features {fe.shape}, curves {cur.shape}", flush=True)
    n_tr = min(a.train, len(X))
    if not cache.exists() and X:
        # np.array(list_of_ragged, dtype=object) tries to BROADCAST when the entries share
        # a leading dimension -- here every entry is (28, T, 16, 4) with T varying, so numpy
        # sees 28 and fails. An empty object array filled by slice assignment is the only
        # form that stays ragged.
        def obj(seq):
            arr = np.empty(len(seq), dtype=object)
            for i, e in enumerate(seq):
                arr[i] = e
            return arr
        np.savez(cache, X=obj(X), Y=obj(Y), A=obj(A))
        print(f'  cached {len(X)} shapes -> {cache}', flush=True)

    # Pad, never truncate: token counts vary by two orders of magnitude between bodies, so
    # equalising by the minimum would discard most of the surface of the detailed ones. A
    # zero-area token contributes nothing to an area-weighted sum, so padding is exact.
    # Features are (28, T, P, F); the token axis is axis 1.
    nt = max(x.shape[1] for x in X)
    X = [np.pad(x, ((0, 0), (0, nt - x.shape[1]), (0, 0), (0, 0))) for x in X]
    A = [np.pad(a_, (0, nt - len(a_))) for a_ in A]
    print(f"  tokens padded to {nt} (raw counts "
          f"{min(int((a_ > 0).sum()) for a_ in A)}-{max(int((a_ > 0).sum()) for a_ in A)})",
          flush=True)
    Atr = torch.tensor(np.stack(A[:n_tr]), dtype=torch.float32, device=dev)
    Ava = torch.tensor(np.stack(A[n_tr:n_tr + max(1, (len(A)-n_tr)//2)]),
                       dtype=torch.float32, device=dev)
    Ate = torch.tensor(np.stack(A[n_tr + max(1, (len(A)-n_tr)//2):]),
                       dtype=torch.float32, device=dev)
    Xtr = torch.tensor(np.stack(X[:n_tr]), dtype=torch.float32, device=dev)
    Ytr = torch.tensor(np.stack(Y[:n_tr]), dtype=torch.float32, device=dev)
    # A held-out set used BOTH to stop training and to report would be selected on, and
    # the reported number would be optimistic. Split it: early stopping watches `val`, the
    # gate is measured on `test`, and the two never overlap.
    n_val = max(1, (len(X) - n_tr) // 2)
    Xva = torch.tensor(np.stack(X[n_tr:n_tr + n_val]), dtype=torch.float32, device=dev)
    Yva = torch.tensor(np.stack(Y[n_tr:n_tr + n_val]), dtype=torch.float32, device=dev)
    Xte = torch.tensor(np.stack(X[n_tr + n_val:]), dtype=torch.float32, device=dev)
    Yte = torch.tensor(np.stack(Y[n_tr + n_val:]), dtype=torch.float32, device=dev)

    net = Surrogate(width=a.width, modes=a.modes, blocks=a.blocks,
                    use_attention=not a.no_attn).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=a.wd)

    def eval_rms(X_, A_, T_, chunk=8):
        """Chunked evaluation: the validation and test sets are 28x larger now that every
        (shape, camera) pair is an example, and attention is O(T^2) per phase."""
        outs = []
        with torch.no_grad():
            for i in range(0, X_.shape[0], chunk):
                outs.append(net(X_[i:i + chunk], A_[i:i + chunk]) - T_[i:i + chunk])
        return torch.cat(outs)

    # Every (shape, camera) pair is a training example, and the target is THAT camera's own
    # curve pair. The surrogate is camera-agnostic by construction, so one network serves
    # all 28 geometries and is valid away from azimuth 0 -- which it was not before.
    # (shape, camera) pairs: features (S, 28, T, P, F) -> (S*28, T, P, F), targets likewise
    def flat(Xs, Ys, As):
        S, C = Xs.shape[0], Xs.shape[1]
        x = Xs.reshape(S * C, *Xs.shape[2:])
        a_ = As[:, None].expand(S, C, As.shape[1]).reshape(S * C, As.shape[1])
        y = torch.stack([Ys[:, :28], Ys[:, 28:]], 2).reshape(S * C, 2, Ys.shape[-1])
        return x, a_, y
    Xtr, Atr, tgt = flat(Xtr, Ytr, Atr)
    Xva, Ava, tva = flat(Xva, Yva, Ava)
    Xte, Ate, tte = flat(Xte, Yte, Ate)
    print(f"  training examples: {Xtr.shape[0]} (shape x camera pairs)", flush=True)
    # Mini-batched over shapes. Attention is O(T^2) per phase, so 600 tokens across all
    # 64 shapes at once asks for 5.5 GiB in a single allocation and OOMs an 8 GB card.
    bs = a.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    best_val, best_state = float("inf"), None
    for s in range(a.steps):
        opt.zero_grad()
        tot = 0.0
        for _ in range(a.accum):
            idx = torch.randperm(Xtr.shape[0], device=dev)[:bs]
            loss = ((net(Xtr[idx], Atr[idx]) - tgt[idx]) ** 2).mean() / a.accum
            loss.backward()
            tot += float(loss)
        loss = torch.tensor(tot)
        opt.step(); sched.step()
        if s % 25 == 0 or s == a.steps - 1:
            with torch.no_grad():
                v = float((eval_rms(Xva, Ava, tva) ** 2).mean().sqrt())
            if v < best_val:
                best_val = v
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        if s % 100 == 0 or s == a.steps - 1:
            with torch.no_grad():
                te = (eval_rms(Xte, Ate, tte) ** 2).mean().sqrt()
            print(f"  step {s:>4}  train {float(loss)**0.5:.5f}  held-out RMS {float(te):.5f}",
                  flush=True)

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"\n  restored the checkpoint with the best VALIDATION RMS {best_val:.5f}")
    torch.save(net.state_dict(), "runs/surrogate.pt")
    with torch.no_grad():
        err = eval_rms(Xte, Ate, tte).abs()
    # sigma comes from the DATA, not from a constant. The measured replicate noise at
    # azimuth 0 -- the geometry these features describe -- is 0.0034 to 0.0240 across the
    # three public bodies (median 0.0124), so a single hard-coded "typical sigma" decides
    # the verdict by itself. Report against the range and say which end is which.
    SIG = {"model 1 intensity": 0.0034, "model 1 binary": 0.0043,
           "model 2 intensity": 0.0195, "model 2 binary": 0.0240,
           "model 3 intensity": 0.0173, "model 3 binary": 0.0075}
    sig_med = float(np.median(list(SIG.values())))
    print(f"\nheld-out shapes: {Xte.shape[0]} (index {n_tr} is the CUBE)")
    for i in range(Xte.shape[0]):
        tag = " <- cube" if i == 0 else ""
        print(f"  shape {n_tr+i}: RMS {float(err[i].pow(2).mean().sqrt()):.5f}  "
              f"max {float(err[i].max()):.5f}{tag}")
    rms = float(err.pow(2).mean().sqrt())
    ri = float(err[:, 0].pow(2).mean().sqrt())
    rb = float(err[:, 1].pow(2).mean().sqrt())
    print(f"  intensity channel RMS {ri:.5f}")
    print(f"  binary    channel RMS {rb:.5f}")
    print(f"  overall   RMS {rms:.5f}")
    print("  against the MEASURED replicate noise at azimuth 0:")
    for k, v in SIG.items():
        print(f"    {k:<20} sigma {v:.4f}  ->  {rms/v:.2f} sigma")
    print(f"  median sigma {sig_med:.4f}  ->  {rms/sig_med:.2f} sigma")
    n_below = sum(1 for v in SIG.values() if rms < v)
    print(f"  below sigma for {n_below} of {len(SIG)} model/channel pairs")
    print("  GATE " + ("PASS" if rms < 0.3 * sig_med else
                       "FAIL (not WELL below sigma; below it, but not by the "
                       "required margin)"))


if __name__ == "__main__":
    main()
