#!/usr/bin/env python3
"""Fit the shape library: per-body support h and per-body correction amplitudes g.

There is no decoder. The previous field decoded a code through cross-attention weights that
had to be fitted jointly across the library, so a code only meant something together with the
decoder it came from -- hence an autodecoder, a shared `runs/token_decoder.pt`, and a
`--decoder-file` threaded through every downstream script. With a fixed lattice the code IS
the amplitude vector: site k always means the same place, so codes are portable by
construction and every body can be fitted independently.

What is fitted per body:
    delta.g      N_SITES signed amplitudes on the fixed lattice

What is FROZEN: h, at the analytic support of the body's own convex hull. It is not fitted --
see BatchedFit for why solving for h and g jointly is both ambiguous and inconsistent with how
h is obtained at reconstruction time.

What is NOT fitted: dh. A corpus body's h is exact, so its dh is zero by definition. The flow
learns dh against a fresh perturbation of the fitted h -- see flow_loss in train_lpd.py -- so
the corpus stores a zero block and the perturbation supplies both the signal and free
augmentation.

Public bodies are never in the library.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hac26.field import (CODE_DIM, DESIGN_N, LATTICE_ALPHA, LATTICE_EXTENT,   # noqa: E402
                         LATTICE_SHAPE, N_DIR, N_SITES, ImplicitBody, design_sha,
                         dir_design)


# The SDF supervision has to cover the correction lattice, not just the body: a site the
# sampler never sees is an amplitude the fit cannot determine, and it is then free at sampling
# time. The lattice reaches LATTICE_EXTENT and its kernels reach 3 sigma beyond that.
SAMPLE_EXTENT = LATTICE_EXTENT + 3.0 * LATTICE_ALPHA * (2.0 * LATTICE_EXTENT / LATTICE_SHAPE[0])


def sample_arrays(verts, faces, n_pts=6000, seed=0):
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=False)
    rng = np.random.default_rng(seed)
    ext = max(float(np.abs(verts).max()) * 1.3, SAMPLE_EXTENT)
    pts = rng.uniform(-ext, ext, (n_pts, 3))
    surf, _ = trimesh.sample.sample_surface(m, n_pts // 2)
    pts = np.vstack([pts, surf + rng.normal(0, 0.03, surf.shape)])
    sd = -m.nearest.signed_distance(pts)       # trimesh: positive inside
    return pts.astype(np.float32), sd.astype(np.float32)


def samples(verts, faces, n_pts=6000, seed=0):
    pts, sd = sample_arrays(verts, faces, n_pts=n_pts, seed=seed)
    return torch.tensor(pts), torch.tensor(sd)


def _prepare_shape(args):
    i, verts, faces, normals, n_pts = args
    pts, sd = sample_arrays(verts, faces, n_pts=n_pts, seed=i)
    h0 = np.maximum((verts @ normals.T).max(axis=0), 1e-3).astype(np.float32)
    return i, pts, sd, h0


class BatchedFit:
    """Every body's support and amplitudes as two stacked tensors, fitted in one batched pass.

    The predecessor ran a Python loop over the bodies in each minibatch, so a step was B
    separate (n_pts x 3) @ (3 x DESIGN_N) matmuls. On a GPU those are launch-bound, not
    compute-bound: the same arithmetic as one (B*n_pts x 3) @ (3 x DESIGN_N) matmul, at a
    fraction of the utilisation. It could not have been written this way before -- the old
    field shared a decoder across the library, which coupled every body to every other -- and
    deleting that decoder is what makes the whole fit embarrassingly parallel.

    The sample points are stacked once onto the device and stay there: 600 bodies x 6000
    points is 43 MB, so nothing is gained by streaming them.

    THE ENCODER IS SEQUENTIAL, NOT JOINT. h is pinned to the analytic support of the body's
    own convex hull and never moves; only g is fitted. Two independent reasons, and the second
    is the one that would have quietly ruined a run:

    1. h and g overlap. Any body can be written as a larger core carved more deeply or a
       smaller core carved less, and the two blocks agree exactly in the low spherical-harmonic
       degrees -- principal cosines 1.00000 at l=0, 0.996 for the three l=1 translations,
       0.987-0.993 at l=2. Solved jointly, the SAME body at the SAME accuracy admits a whole
       family of (h, g) pairs, and a flow trained on that family faithfully learns the
       ambiguity as spurious multimodality.

    2. Worse, and specific to this pipeline: at reconstruction h does not come from the fit at
       all. It comes from support_from_convex(), which is the convex stage's estimate of the
       body's HULL. If the corpus's h were free to drift away from the hull, every corpus g
       would have been fitted against a core that means something different from the core it
       is decoded against. Measured on the smoke library, joint fitting drifts h by 0.35 in
       units where h itself is order 1 -- a third of the support, silently.

    The cost is convergence rate, not accuracy: joint reached 0.000231 by step 250 and frozen
    was still at 0.000446, but frozen passes it by step 800 (0.000089) with only g to fit.
    |g| is larger, as it must be -- the core is now the full hull, so every concavity has to
    be carved rather than partly absorbed by shrinking the core.
    """

    def __init__(self, n, data, h0s, dev):
        self.dev = dev
        self.P = torch.stack([d[0] for d in data]).to(dev)          # (n, n_pts, 3)
        self.S = torch.stack([d[1] for d in data]).to(dev)          # (n, n_pts)
        ref = ImplicitBody(radius=1.0).to(dev)
        self.n_normals = ref.core.n                                 # (DESIGN_N, 3)
        self.lat = ref.delta
        h = torch.tensor(np.stack(h0s), dtype=torch.float32, device=dev).clamp_min(1e-6)
        # FROZEN, not a parameter. h is pinned to the analytic support of the body's own
        # convex hull, h(u) = max_v <u, v>, and only g is fitted. See the class docstring.
        self.raw_h = h + torch.log(-torch.expm1(-h))                       # inverse softplus
        self.g = torch.nn.Parameter(torch.zeros(n, N_SITES, device=dev))

    def _delta(self, P, g):
        """Batched lattice evaluation, same expanded-square matmul as GaussianLattice."""
        lat = self.lat
        d2 = (P ** 2 * lat.inv2).sum(-1, keepdim=True) + lat.pb \
            - 2.0 * ((P * lat.inv2) @ lat.p.T)                      # (B, n_pts, N_SITES)
        return torch.einsum("bps,bs->bp", torch.exp(-0.5 * d2.clamp_min(0.0)), g)

    def loss(self, idx):
        P, S = self.P[idx], self.S[idx]
        h = torch.nn.functional.softplus(self.raw_h[idx])           # (B, DESIGN_N)
        core = (P @ self.n_normals.T - h[:, None, :]).amax(-1)      # (B, n_pts)
        return ((core + self._delta(P, self.g[idx]) - S) ** 2).mean(-1).mean()

    def bodies(self):
        """Materialise per-body ImplicitBody objects for the diagnostics and the corpus."""
        out = []
        for i in range(len(self.raw_h)):
            b = ImplicitBody(radius=1.0).to(self.dev)
            with torch.no_grad():
                b.core.raw_h.copy_(self.raw_h[i])
                b.delta.g.copy_(self.g[i])
            out.append(b)
        return out


def report_corpus(bodies, data, codes, trace) -> bool:
    """Report on the finished corpus. Returns True if it looks usable.

    Called AFTER the corpus is written, never before: a fit that took hours must be flagged,
    not thrown away. The caller turns a False into a non-zero exit.

    There is no longer a pre-training liveness check. Its predecessor existed because the old
    cross-attention field had exactly zero gradient at its zero init -- identical tokens make
    the softmax uniform, so Delta was identically zero AND unrecoverable. A lattice of fixed
    sites is LINEAR in g, so dDelta/dg_k = exp(-||(y-p_k)/sigma||^2/2) is strictly positive
    everywhere and g = 0 is a perfectly good starting point. Keeping that check would have
    aborted every run.
    """
    with torch.no_grad():
        dmax = max(float(b.delta(data[i][0].to(b.delta.g.device)).abs().max())
                   for i, b in enumerate(bodies[:min(8, len(bodies))]))
    gvar = float(codes[:, N_DIR:].var(0).mean())
    gmax = float(np.abs(codes[:, N_DIR:]).max())
    # medians, not the first and last step: each step's loss is one random minibatch of
    # `--batch` bodies out of `--bodies`, and per-body SDF loss spans two orders of magnitude,
    # so consecutive converged steps differ by up to 34x. Comparing single steps false-fails.
    k = max(1, min(25, len(trace) // 4))
    l0 = float(np.median(trace[:k])) if trace else float("nan")
    l1 = float(np.median(trace[-k:])) if trace else float("nan")
    bad = []
    if dmax < 1e-4:
        bad.append(f"the correction is dead: max|Delta| = {dmax:.3e}. Every body in this "
                   f"corpus IS its convex core.")
    if gvar < 1e-12:
        bad.append(f"codes do not vary across bodies: amplitude variance {gvar:.3e}. "
                   f"Every body was assigned the same code.")
    loss_note = (f"sdf loss (median of {k}) {l0:.6f} -> {l1:.6f}" if trace
                 else "no training steps were run")
    print(f"  [check] max|Delta| {dmax:.4f}, |g|max {gmax:.4f}, g variance {gvar:.5f}, "
          f"{loss_note}", flush=True)
    if trace and not (l1 < 0.8 * l0):
        print(f"  WARNING: the sdf loss barely moved ({l0:.6f} -> {l1:.6f}). The corpus was "
              f"still written; raise --steps if this was not deliberate.", flush=True)
    for b_ in bad:
        print(f"  ERROR: {b_}", flush=True)
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=40)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4, help="bodies per step")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel workers for independent SDF/support preprocessing")
    ap.add_argument("--points", type=int, default=6000,
                    help="SDF sample points per body")
    ap.add_argument("--out", default="runs/corpus_codes.npz")
    ap.add_argument("--device", default=None,
                    help="cuda when available, else cpu. The fit is "
                         "FIT_STEPS x FIT_BATCH x FIT_POINTS x (DESIGN_N + N_SITES) "
                         "element-ops -- about 1.7e12 at the remote defaults -- so this is "
                         "hours on a CPU and minutes on a GPU.")
    ap.add_argument("--shapes-dir", default=None,
                    help="directory written by scripts/build_shape_library.py; if given, "
                         "the corpus is drawn from it instead of train_surrogate.shapes()")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle seed when reading --shapes-dir (ignored otherwise: "
                         "train_surrogate.shapes() is deterministic in draw order already)")
    a = ap.parse_args()

    if a.shapes_dir:
        from hac26.library_io import load_library_dir
        print(f"[1] loading {a.bodies} bodies from {a.shapes_dir}", flush=True)
        shape_list = load_library_dir(a.shapes_dir, n=a.bodies, seed=a.seed)
        if len(shape_list) < a.bodies:
            print(f"  WARNING: only {len(shape_list)} bodies available in {a.shapes_dir}, "
                  f"requested {a.bodies}", flush=True)
    else:
        from train_surrogate import shapes
        print(f"[1] sampling SDF for {a.bodies} bodies", flush=True)
        shape_list = shapes(a.bodies, seed=0)

    data, h0s = [None] * len(shape_list), [None] * len(shape_list)
    ref = ImplicitBody(radius=1.0)
    nrm = ref.core.n.detach().cpu().numpy()
    jobs = [(i, np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64), nrm,
             a.points)
            for i, (v, f) in enumerate(shape_list)]
    workers = max(1, int(a.workers))
    pool = None
    if workers == 1:
        iterator = map(_prepare_shape, jobs)
    else:
        pool = Pool(workers)
        iterator = pool.imap_unordered(_prepare_shape, jobs, chunksize=2)
    failed = False
    try:
        for n_done, (i, pts, sd, h0) in enumerate(iterator, start=1):
            data[i] = (torch.tensor(pts), torch.tensor(sd))
            h0s[i] = h0
            if n_done % 10 == 0 or n_done == len(jobs):
                print(f"    preprocessed {n_done}/{len(jobs)} bodies", flush=True)
    except Exception:
        failed = True
        if pool is not None:
            pool.terminate()
        raise
    finally:
        if pool is not None:
            if not failed:
                pool.close()
            pool.join()

    # No shared decoder. With fixed lattice sites the code IS g, so a code means the same
    # thing to any reader, there is nothing to fit jointly, and -- the point of this section --
    # the bodies are now COMPLETELY INDEPENDENT of one another. That is what makes the fit
    # vectorisable: one batched matmul over B bodies instead of B separate small ones.
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda":
        # The SDF fit is a regression to about three decimal places; TF32 costs it nothing and
        # is several times faster on the (points x normals) matmul that dominates.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    fitter = BatchedFit(len(data), data, h0s, dev)
    print(f"[2] fit on {dev}: {len(data)} bodies x ({DESIGN_N} support + {N_SITES} "
          f"amplitudes), batched {a.batch} at a time", flush=True)

    # One parameter block: g. h is frozen at the hull -- see BatchedFit. lr 0.005 because g is
    # the SDF correction itself now rather than 0.15 R times it; deleting CORE_SCALE moved the
    # effective step size by that factor.
    opt = torch.optim.Adam([{"params": [fitter.g], "lr": 0.005}])
    t0 = time.time()
    trace = []
    for s in range(a.steps):
        idx = torch.from_numpy(
            np.random.default_rng(s).integers(0, len(data), a.batch)).to(dev)
        loss = fitter.loss(idx)
        opt.zero_grad(); loss.backward(); opt.step()
        trace.append(float(loss.detach()))
        if s % 250 == 0 or s == a.steps - 1:
            print(f"    step {s:>5}  sdf loss {trace[-1]:.5f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    bodies = fitter.bodies()

    # The dh block is stored explicitly as zeros rather than omitted, so codes.shape[1] is
    # CODE_DIM everywhere and one shape gate covers the whole pipeline.
    codes = np.stack([torch.cat([b.dh.detach(), b.delta.g.detach()]).cpu().numpy()
                      for b in bodies])
    sup = np.stack([b.core.h.detach().cpu().numpy() for b in bodies])   # the property

    # the directories the run actually writes to, not a fixed "model" that no path uses:
    # np.savez raised on a fresh clone BEFORE the diagnostics below ever printed
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": 3,
        "bodies": int(len(bodies)),
        "design_n": int(DESIGN_N),
        "design_sha": design_sha(nrm),
        # The dh directions are a SECOND design and are just as load-bearing: sh_expand, the
        # sphere-convolution operators and the A^T r channel are all indexed by them, so two
        # machines that generated dir_design(N_DIR) independently would disagree about what
        # every dh coefficient means, silently.
        "dir_sha": design_sha(dir_design(N_DIR)),
        "code_dim": int(CODE_DIM),
        "n_dir": int(N_DIR),
        "n_sites": int(N_SITES),
        "lattice_shape": list(LATTICE_SHAPE),
        "lattice_extent": float(LATTICE_EXTENT),
        "lattice_alpha": float(LATTICE_ALPHA),
        "points": int(a.points),
        "steps": int(a.steps),
        "batch": int(a.batch),
        "seed": int(a.seed),
        "shapes_dir": None if a.shapes_dir is None else str(Path(a.shapes_dir)),
    }
    np.savez(a.out, codes=codes, support=sup, meta=json.dumps(meta, sort_keys=True))
    print(f"  codes {codes.shape}, amplitude variance {codes[:, N_DIR:].var(0).mean():.5f}")
    print(f"  wrote {a.out}")
    if not report_corpus(bodies, data, codes, trace):
        raise SystemExit("fit_shapes: the corpus above is degenerate. It was written so the "
                         "fit is not lost, but do not train on it.")


if __name__ == "__main__":
    main()
