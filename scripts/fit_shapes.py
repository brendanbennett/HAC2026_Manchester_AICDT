#!/usr/bin/env python3
"""Fit the implicit field to every body of the shape library, giving the training corpus.

Per body, the support h is set to the body's own convex hull and frozen, and the lattice
amplitudes g are fitted by regressing the field onto the body's signed distance at sampled
points. See BatchedFit for why h is not fitted jointly with g. The dh block is stored as
zeros: a corpus body's h is exact here, and scripts/build_corpus.py later sets dh to the
correction from the convex stage's start to this h.

With a fixed lattice the code is the amplitude vector itself, so every body is fitted
independently and codes mean the same thing to every reader.

After the fit, a sample of bodies is decoded and its Dice overlap with the mesh it was fitted
to is reported by library family. This is the check that the representation can express the
shapes at all: a family with low fitted Dice (necks, sharp craters) is a family the flow
cannot reconstruct however well it is trained. The per-body values go into the output file.

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

from hac26.shapes import canonicalize_r, rescale_touch_z   # noqa: E402
from hac26.field import (CODE_DIM, DESIGN_N, EXTRACT_EXTENT, LATTICE_ALPHA,   # noqa: E402
                         LATTICE_EXTENT, LATTICE_SHAPE, N_DIR, N_SITES, ImplicitBody,
                         design_sha, dir_design, extract_mesh)
from hac26.recon import dice, mesh_occupancy                # noqa: E402

DICE_RES = 64          # voxel grid of the fitted-Dice check
DICE_EXTENT = 1.35     # half-width of that grid; covers a posed body with its published radius


# The sampled points have to cover the lattice and its kernels, not just the body: a site the
# sampler never sees is an amplitude the fit cannot determine.
SAMPLE_EXTENT = LATTICE_EXTENT + 3.0 * LATTICE_ALPHA * (2.0 * LATTICE_EXTENT / LATTICE_SHAPE[0])


def sample_arrays(verts, faces, n_pts=6000, seed=0):
    """Sample points for the fit and the body's signed distance at them (positive outside):
    n_pts uniform in a box covering the lattice, plus half as many jittered surface points."""
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=False)
    rng = np.random.default_rng(seed)
    ext = max(float(np.abs(verts).max()) * 1.3, SAMPLE_EXTENT)
    pts = rng.uniform(-ext, ext, (n_pts, 3))
    surf, _ = trimesh.sample.sample_surface(m, n_pts // 2)
    pts = np.vstack([pts, surf + rng.normal(0, 0.03, surf.shape)])
    sd = -m.nearest.signed_distance(pts)       # trimesh: positive inside
    return pts.astype(np.float32), sd.astype(np.float32)


def _prepare_shape(args):
    """One body's sample points, signed distances and hull support, for a worker pool."""
    i, verts, faces, normals, n_pts = args
    pts, sd = sample_arrays(verts, faces, n_pts=n_pts, seed=i)
    h0 = np.maximum((verts @ normals.T).max(axis=0), 1e-3).astype(np.float32)
    return i, pts, sd, h0


class BatchedFit:
    """Every body's support and amplitudes as two stacked tensors, fitted in one batched pass.
    The bodies are independent, so a step is one matrix product over the whole minibatch.

    h is pinned to the support of the body's own convex hull and never moves; only g is
    fitted. Two reasons:

    1. h and g overlap. The same body can be written as a larger core carved more deeply or a
       smaller core carved less. Fitted jointly, one body admits a whole family of (h, g)
       pairs, and a flow trained on that family learns the ambiguity as if it were real.

    2. At reconstruction h does not come from a fit. It comes from the convex stage's estimate
       of the hull. If the corpus's h drifted away from the hull, every corpus g would have
       been fitted against a core that means something different from the one it is decoded
       against.

    With the core at the full hull, every concavity has to be carved by g rather than partly
    absorbed by a smaller core, so |g| is larger than a joint fit would give.
    """

    def __init__(self, n, data, h0s, dev):
        self.dev = dev
        self.P = torch.stack([d[0] for d in data]).to(dev)          # (n, n_pts, 3)
        self.S = torch.stack([d[1] for d in data]).to(dev)          # (n, n_pts)
        ref = ImplicitBody().to(dev)
        self.n_normals = ref.core.n                                 # (DESIGN_N, 3)
        self.lat = ref.delta
        h = torch.tensor(np.stack(h0s), dtype=torch.float32, device=dev).clamp_min(1e-6)
        # frozen, not a parameter: h is the hull support h(u) = max over vertices of <u, v>
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
            b = ImplicitBody().to(self.dev)
            with torch.no_grad():
                b.core.raw_h.copy_(self.raw_h[i])
                b.delta.g.copy_(self.g[i])
            out.append(b)
        return out


def fitted_dice(bodies, shape_list, families, n_sample: int, seed: int = 0) -> np.ndarray:
    """Dice of the decoded fitted body against the mesh it was fitted to, for a random sample
    of n_sample bodies (NaN for the rest), printed by family."""
    out = np.full(len(bodies), np.nan)
    idx = np.random.default_rng(seed).permutation(len(bodies))[:n_sample]
    for i in idx:
        b = bodies[i]
        v, f = extract_mesh(lambda y: b(y), EXTRACT_EXTENT, res=DICE_RES,
                            device=str(b.delta.g.device))
        if len(f) < 8:
            out[i] = 0.0
            continue
        sv, sf = shape_list[i]
        out[i] = dice(mesh_occupancy(v, f, DICE_RES, DICE_EXTENT),
                      mesh_occupancy(np.asarray(sv), np.asarray(sf), DICE_RES, DICE_EXTENT))
    print(f"  [dice] fitted body against its mesh, {len(idx)} bodies sampled:")
    for fam in sorted(set(families[i] for i in idx)):
        d = np.array([out[i] for i in idx if families[i] == fam])
        print(f"    {fam:<16} n {len(d):>3}  mean {d.mean():.3f}  min {d.min():.3f}", flush=True)
    return out


def report_corpus(bodies, data, codes, trace) -> bool:
    """Report on the finished corpus. Returns True if it looks usable.

    Called after the corpus is written, never before: a fit that took hours must be flagged,
    not thrown away. The caller turns a False into a non-zero exit.
    """
    with torch.no_grad():
        dmax = max(float(b.delta(data[i][0].to(b.delta.g.device)).abs().max())
                   for i, b in enumerate(bodies[:min(8, len(bodies))]))
    gvar = float(codes[:, N_DIR:].var(0).mean())
    gmax = float(np.abs(codes[:, N_DIR:]).max())
    # medians over a stretch of steps, not single steps: each step is one small random
    # minibatch and the per-body loss varies a lot
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
                    help="uniform SDF sample points per body; half as many jittered "
                         "surface points are added on top")
    ap.add_argument("--out", default="runs/corpus_codes.npz")
    ap.add_argument("--device", default=None,
                    help="cuda when available, else cpu; the fit is hours on a CPU and "
                         "minutes on a GPU")
    ap.add_argument("--shapes-dir", required=True,
                    help="directory written by scripts/build_shape_library.py")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle seed when reading --shapes-dir")
    ap.add_argument("--dice-bodies", type=int, default=64,
                    help="bodies sampled for the fitted-Dice check by family; 0 skips it")
    a = ap.parse_args()

    from hac26.library_io import load_library_dir
    print(f"[1] loading {a.bodies} bodies from {a.shapes_dir}", flush=True)
    loaded = load_library_dir(a.shapes_dir, n=a.bodies, seed=a.seed, with_entries=True)
    shape_list = [(v, f) for v, f, _ in loaded]
    families = [str(e.get("base", "unknown")) for _, _, e in loaded]
    # the width over half-height each body was mounted with; a library written before that
    # was recorded carries none, and build_corpus.py then draws a radius instead
    radii = np.array([float(e.get("radius", np.nan)) for _, _, e in loaded])
    if len(shape_list) < a.bodies:
        print(f"  WARNING: only {len(shape_list)} bodies available in {a.shapes_dir}, "
              f"requested {a.bodies}", flush=True)

    # Every body into the canonical frame, whichever source it came from: the lattice is
    # fixed in that frame, so site k only means the same place across bodies if the bodies
    # share it. An already-posed body is untouched.
    n_posed = 0
    posed = []
    for v, f in shape_list:
        v = np.asarray(v, dtype=np.float64)
        # faces passed so the pose centres on the solid centroid, as the library does
        c = canonicalize_r(rescale_touch_z(v, np.asarray(f, dtype=np.int64)))
        if float(np.abs(c - v).max()) > 1e-9:
            n_posed += 1
        posed.append((c, f))
    shape_list = posed
    if n_posed:
        print(f"  posed {n_posed}/{len(shape_list)} bodies into the canonical frame "
              f"(z span 2, xy r_max 1)", flush=True)

    data, h0s = [None] * len(shape_list), [None] * len(shape_list)
    ref = ImplicitBody()
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

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    fitter = BatchedFit(len(data), data, h0s, dev)
    print(f"[2] fit on {dev}: {len(data)} bodies x ({DESIGN_N} support + {N_SITES} "
          f"amplitudes), batched {a.batch} at a time", flush=True)

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

    # the dh block is stored as zeros rather than omitted, so codes.shape[1] is CODE_DIM
    # everywhere
    codes = np.stack([torch.cat([b.dh.detach(), b.delta.g.detach()]).cpu().numpy()
                      for b in bodies])
    sup = np.stack([b.core.h.detach().cpu().numpy() for b in bodies])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": 3,
        "bodies": int(len(bodies)),
        "design_n": int(DESIGN_N),
        "design_sha": design_sha(nrm),
        # the dh directions are a second design, and dh is indexed by them
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
        "shapes_dir": str(Path(a.shapes_dir)),
    }
    fit_d = (fitted_dice(bodies, shape_list, families, a.dice_bodies, seed=a.seed)
             if a.dice_bodies > 0 else np.full(len(bodies), np.nan))
    np.savez(a.out, codes=codes, support=sup, fit_dice=fit_d, family=np.array(families),
             radius=radii, meta=json.dumps(meta, sort_keys=True))
    print(f"  codes {codes.shape}, amplitude variance {codes[:, N_DIR:].var(0).mean():.5f}")
    print(f"  wrote {a.out}")
    if not report_corpus(bodies, data, codes, trace):
        raise SystemExit("fit_shapes: the corpus above is degenerate. It was written so the "
                         "fit is not lost, but do not train on it.")


if __name__ == "__main__":
    main()
