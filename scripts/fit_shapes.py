#!/usr/bin/env python3
"""Fit the implicit field to every body of the shape library, giving the training corpus.

Per body, the support h is set to the body's own convex hull and frozen, and the lattice
amplitudes g are fitted by regressing the field onto the body's signed distance at sampled
points. See LatticeFit for why h is not fitted jointly with g. The dh block is stored as
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

# Only the zero level set of the field is the body, so the fit weights a sample point by how
# close it is to the surface. The band is about one lattice kernel width; the floor keeps far
# points contributing, which pins the field's sign where nothing else constrains it.
SURFACE_BAND = 0.15
WEIGHT_FLOOR = 0.05
RIDGE = 1e-3           # relative to the mean diagonal of the normal matrix. Neighbouring
                       # kernels overlap, so many amplitude vectors describe nearly the same
                       # body and the sample points alone do not choose between them: fitted
                       # twice from different points, one body gets two different codes, and
                       # the difference is a property of the sampling rather than of the body.
                       # Both the ridge and more sample points settle that choice, but the
                       # ridge also costs a little of the shape while points cost only time,
                       # so the ridge is set low and the points carry the work.
SOLVE_CHUNK = 20000    # sample points per block of the solve. Both the core support function
                       # over the design normals and the normal matrix are sums or maxima over
                       # the points, so evaluating them a block at a time bounds the memory of
                       # the solve and leaves the point count limited only by the cost of the
                       # signed distances.


# The sampled points have to cover the lattice and its kernels, not just the body: a site the
# sampler never sees is an amplitude the fit cannot determine.
SAMPLE_EXTENT = LATTICE_EXTENT + 3.0 * LATTICE_ALPHA * (2.0 * LATTICE_EXTENT / LATTICE_SHAPE[0])
SD_CHUNK = 20000       # query points per call to the mesh's signed distance; see sample_arrays
GPU_PAIRS = 4_000_000  # point-triangle pairs per block of the GPU signed distance. Each block
                       # is scored against every face at once, so this and nothing else sets
                       # the memory the query needs, about a gigabyte here. Measured flat in
                       # speed from this size to four times larger, so it is set at the low
                       # end and leaves the card free for whatever else is on it.
SDF_TOL = 1e-4         # how far the GPU signed distance may sit from trimesh's; see
                       # check_sdf_backend. Measured at most 9e-6 over the library.
POINTS_PER_SITE = 12   # fewest sample points per amplitude the fit will accept. Measured
                       # on the library: at five per amplitude a carved body's fit
                       # overshoots and decodes to a body unlike itself, at seventeen it
                       # reproduces one at convexity 0.34 to a Dice of 0.97. A smooth body
                       # needs far fewer, since most of its amplitudes are near zero, so
                       # this floor is set by the bodies that matter. See main.
DICE_FLOOR = 0.75      # median fitted Dice below which the corpus is refused; see
                       # report_corpus
FIT_CONVEXITY_BINS = (0.55, 0.7, 0.85, 0.95)


def _trimesh_signed_distance(m, pts):
    """Signed distances with this file's convention: positive outside."""
    return -np.concatenate([m.nearest.signed_distance(pts[i:i + SD_CHUNK])
                            for i in range(0, len(pts), SD_CHUNK)])


def _prepare_mesh(verts, faces):
    """The body as a cleaned trimesh whose faces face outward."""
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=True)
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    m.fix_normals()
    if m.volume < 0.0:
        m.invert()
    return m


def _safe_div(num, den):
    """num / den clamped to [0, 1], a zero denominator giving zero."""
    return (num / torch.where(den.abs() < 1e-30, torch.ones_like(den), den)).clamp(0.0, 1.0)


def _point_triangle_sq(p, a, b, c):
    """Squared distance from each point to each triangle: p is (n, 1, 3) and a, b, c are
    (1, f, 3), so the result is (n, f).

    The closest point of a triangle is either inside it, on one of its three edges or at one
    of its three vertices, and which of the seven it is follows from the signs of the six dot
    products below (Ericson, Real-Time Collision Detection, 5.1.5). The cases are masks over
    the whole block rather than branches, applied in the reverse of the order the sequential
    test checks them: where two of them overlap on a boundary the last one written wins, so
    reversing the order leaves the case the sequential test would have returned.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4          # the barycentric coordinates of the projection into the
    vb = d5 * d2 - d1 * d6          # plane of the triangle, before they are normalised
    vc = d1 * d4 - d3 * d2
    v = _safe_div(vb, va + vb + vc)
    w = _safe_div(vc, va + vb + vc)
    q = a + v[..., None] * ab + w[..., None] * ac                       # inside the face
    for mask, val in (
            ((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
             b + _safe_div(d4 - d3, (d4 - d3) + (d5 - d6))[..., None] * (c - b)),
            ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + _safe_div(d2, d2 - d6)[..., None] * ac),
            ((d6 >= 0) & (d5 <= d6), c.expand_as(q)),
            ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + _safe_div(d1, d1 - d3)[..., None] * ab),
            ((d3 >= 0) & (d4 <= d3), b.expand_as(q)),
            ((d1 <= 0) & (d2 <= 0), a.expand_as(q))):
        q = torch.where(mask[..., None], val, q)
    d = p - q
    return (d * d).sum(-1)


def _winding_number(p, a, b, c):
    """The mesh's generalized winding number at each point, from (n, 1, 3) against (1, f, 3).

    The solid angle each triangle subtends at the point (Van Oosterom and Strackee, 1983),
    summed over the faces and divided by 4 pi: one inside a closed surface and zero outside
    it. It is read off the face winding, so it reports an inward-facing mesh as the
    complement of itself, exactly as a sign taken from the face normals would.
    """
    pa, pb, pc = a - p, b - p, c - p
    la, lb, lc = pa.norm(dim=-1), pb.norm(dim=-1), pc.norm(dim=-1)
    num = (pa * torch.linalg.cross(pb, pc, dim=-1)).sum(-1)
    den = (la * lb * lc + (pa * pb).sum(-1) * lc + (pb * pc).sum(-1) * la
           + (pc * pa).sum(-1) * lb)
    return torch.atan2(num, den).sum(-1) / (2.0 * np.pi)


def _gpu_signed_distance(m, pts, device):
    """Signed distances with this file's convention, positive outside, computed on a GPU.

    The unsigned distance is the least point-triangle distance over every face and the sign
    is the generalized winding number, both of them a minimum or a sum over the faces with
    no acceleration structure at all. That is the shape a GPU wants. trimesh's rtree prunes
    far more work per point, but it does the pruning one call at a time in Python, and over
    the library this path is several times quicker for it even on a small card.

    The sign comes from the mesh's winding exactly as trimesh's does, so an inward-facing
    mesh is negated here too and sample_arrays' check for that covers this path as well.
    """
    tri = torch.as_tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)[
        torch.as_tensor(np.asarray(m.faces), dtype=torch.int64, device=device)]   # (f, 3, 3)
    a, b, c = tri[:, 0][None], tri[:, 1][None], tri[:, 2][None]
    q = torch.as_tensor(np.asarray(pts), dtype=torch.float32, device=device)
    # every point of a block is scored against every face, so the block is sized to hold the
    # pair count the card has room for however many faces this particular body has
    chunk = max(1, GPU_PAIRS // max(len(tri), 1))
    out = torch.empty(len(q), dtype=torch.float32, device=device)
    for i in range(0, len(q), chunk):
        p = q[i:i + chunk, None, :]
        d = _point_triangle_sq(p, a, b, c).amin(-1).clamp_min(0.0).sqrt()
        out[i:i + chunk] = torch.where(_winding_number(p, a, b, c) > 0.5, -d, d)
    return out.double().cpu().numpy()


def _signed_distance(m, pts, device=None):
    """The body's signed distance at pts, positive outside, on `device` when that is a GPU
    and through trimesh otherwise."""
    if device is not None and torch.device(device).type != "cpu":
        return _gpu_signed_distance(m, pts, device)
    return _trimesh_signed_distance(m, pts)


def check_sdf_backend(verts, faces, device, n_pts=1000, seed=0):
    """The GPU signed distance against trimesh's on one body, before the run commits to it.

    This path is arithmetic the file does itself rather than a library call it can take on
    trust, and a wrong answer from it would not announce itself: every body would be fitted
    to a shape that is not the one on disk, the residuals would fall as usual, and only the
    fitted Dice at the very end would show it. A thousand points against the path it
    replaces costs a fraction of a second, once, and is the whole of the evidence that the
    two agree.

    Returns the largest disagreement. Points within a hair of the surface are left out of
    the sign comparison, being the one place the two may legitimately differ.
    """
    m = _prepare_mesh(verts, faces)
    rng = np.random.default_rng(seed)
    ext = max(float(np.abs(np.asarray(m.vertices, dtype=np.float64)).max()) * 1.3,
              SAMPLE_EXTENT)
    pts = rng.uniform(-ext, ext, (n_pts, 3))
    ref = _trimesh_signed_distance(m, pts)
    got = _gpu_signed_distance(m, pts, device)
    err = float(np.abs(got - ref).max())
    off = np.abs(ref) > 1e-5
    flips = int((np.sign(got[off]) != np.sign(ref[off])).sum())
    if err > SDF_TOL or flips:
        raise SystemExit(
            f"the GPU signed distance disagrees with trimesh's on the first body: largest "
            f"difference {err:.2e} against a tolerance of {SDF_TOL}, and {flips} of "
            f"{int(off.sum())} points put on the wrong side of the surface. Rerun with "
            f"--sdf cpu, the path this one is checked against.")
    return err


def _contains_signed_distance(m, pts):
    """Robust fallback: unsigned nearest-surface distance, sign from ray parity."""
    out = []
    for i in range(0, len(pts), SD_CHUNK):
        block = pts[i:i + SD_CHUNK]
        _, distance, _ = m.nearest.on_surface(block)
        signed = np.asarray(distance, dtype=np.float64)
        inside = np.asarray(m.contains(block), dtype=bool)
        signed[inside] *= -1.0
        out.append(signed)
    return np.concatenate(out)


def _bad_signed_distances(sd, far):
    if not np.isfinite(sd).all():
        return True
    return bool(far.any() and sd[far].min() <= 0.0)


def sample_arrays(verts, faces, n_pts=6000, seed=0, device=None):
    """Sample points for the fit and the body's signed distance at them (positive outside):
    n_pts uniform in a box covering the lattice, plus half as many jittered surface points.

    With `device` a GPU the distances go through _gpu_signed_distance, which is the whole of
    this stage's cost and several times quicker there. Otherwise they are queried through
    trimesh a block of points at a time: that query allocates per point against the whole
    mesh, so asking for all of them at once needs memory in proportion to the point count,
    which is the one thing the point count must not cost."""
    import trimesh
    m = _prepare_mesh(verts, faces)
    rng = np.random.default_rng(seed)
    verts = np.asarray(m.vertices, dtype=np.float64)
    ext = max(float(np.abs(verts).max()) * 1.3, SAMPLE_EXTENT)
    pts = rng.uniform(-ext, ext, (n_pts, 3))
    surf, _ = trimesh.sample.sample_surface(m, n_pts // 2, seed=seed)
    pts = np.vstack([pts, surf + rng.normal(0, 0.03, surf.shape)])
    sd = _signed_distance(m, pts, device)
    # The sign comes from the mesh's winding, so a mesh that is inside out returns the whole
    # field negated and the body is fitted as its own complement, with no sign of it in the
    # residual. A point beyond the body's own bounding sphere is outside whatever the mesh
    # says, so it is the cheapest thing that can tell the two apart.
    far = np.linalg.norm(pts, axis=1) > float(np.linalg.norm(verts, axis=1).max()) + 1e-6
    if _bad_signed_distances(sd, far):
        m.invert()
        sd = _signed_distance(m, pts, device)
    if _bad_signed_distances(sd, far):
        sd = _contains_signed_distance(m, pts)
        # A point beyond every vertex radius is outside by construction, even when a
        # degeneracy makes the ray-parity fallback undecidable on that exact ray.
        sd[far] = np.maximum(sd[far], 1e-6)
    if _bad_signed_distances(sd, far):
        raise ValueError("the signed distance calls points outside the body's bounding sphere "
                         "inside it: the mesh is oriented inward. Check mesh_volume.")
    return pts.astype(np.float32), sd.astype(np.float32)


def _prepare_shape(args):
    """One body's sample points, signed distances and hull support, for a worker pool.

    `device` is a GPU only when the caller has already forced the pool to one worker and is
    therefore running this in the main process: a forked worker cannot use the parent's CUDA
    context, and initialising its own would put one context per worker on the card."""
    i, verts, faces, normals, n_pts, device = args
    try:
        pts, sd = sample_arrays(verts, faces, n_pts=n_pts, seed=i, device=device)
    except Exception as exc:
        raise RuntimeError(f"failed to prepare library body {i}") from exc
    h0 = np.maximum((verts @ normals.T).max(axis=0), 1e-3).astype(np.float32)
    return i, pts, sd, h0


class LatticeFit:
    """The amplitudes of every body, solved exactly rather than descended to.

    h is pinned to the support of the body's own convex hull and never moves; only g is
    fitted. Two reasons:

    1. h and g overlap. The same body can be written as a larger core carved more deeply or a
       smaller core carved less. Fitted jointly, one body admits a whole family of (h, g)
       pairs, and a flow trained on that family learns the ambiguity as if it were real.

    2. At reconstruction h does not come from a fit. It comes from the convex stage's estimate
       of the hull. If the corpus's h drifted away from the hull, every corpus g would have
       been fitted against a core that means something different from the one it is decoded
       against.

    With h fixed, the field core(y) + sum_k g_k phi_k(y) is linear in g, so the amplitudes
    that minimise the weighted residual against the signed distance solve

        (Phi^T W Phi + lambda I) g = Phi^T W (S - core),

    one small symmetric system per body. Solving it reaches whatever amplitudes a body needs,
    however large. A gradient fit does not: each coordinate moves by about the learning rate
    per step, so the depth a body can be carved to is capped by the steps it is given, and a
    deeply carved body comes out shallow with no sign that anything went wrong.

    The target S - core is the carve itself. It vanishes wherever the body agrees with its
    hull and is largest in the concavities, which is what g has to supply.
    """

    def __init__(self, dev):
        ref = ImplicitBody().to(dev)
        self.dev = dev
        self.normals = ref.core.n                                   # (DESIGN_N, 3)
        self.lat = ref.delta

    def _phi(self, pts):
        """The lattice kernels at the sample points, (n_pts, N_SITES): the same expanded
        square GaussianLattice evaluates."""
        lat = self.lat
        d2 = (pts ** 2 * lat.inv2).sum(-1, keepdim=True) + lat.pb \
            - 2.0 * ((pts * lat.inv2) @ lat.p.T)
        return torch.exp(-0.5 * d2.clamp_min(0.0))

    def solve(self, pts, sd, h):
        """One body's amplitudes from its sample points, their signed distances (positive
        outside) and its hull support. Returns (g, rms before, rms after), the residuals
        being the weighted root mean square of the field against the signed distance with
        no amplitudes and with the fitted ones.

        Both the core and the normal matrix are evaluated a block of points at a time, so
        the memory the solve needs is set by the block size rather than by how many points
        the body is fitted from, and the point count costs only time."""
        pts, sd = pts.to(self.dev), sd.to(self.dev)
        h = torch.as_tensor(h, dtype=torch.float32, device=self.dev)
        core = torch.cat([(pts[i:i + SOLVE_CHUNK] @ self.normals.T - h[None, :]).amax(-1)
                          for i in range(0, len(pts), SOLVE_CHUNK)])       # (n_pts,)
        target = sd - core
        w = torch.exp(-0.5 * (sd / SURFACE_BAND) ** 2) + WEIGHT_FLOOR
        A = torch.zeros(N_SITES, N_SITES, dtype=torch.float64, device=self.dev)
        b = torch.zeros(N_SITES, dtype=torch.float64, device=self.dev)
        for i in range(0, len(pts), SOLVE_CHUNK):
            sl = slice(i, i + SOLVE_CHUNK)
            phi = self._phi(pts[sl])                                # (chunk, N_SITES)
            wphi = w[sl, None] * phi
            A += (phi.T @ wphi).double()
            b += (wphi.T @ target[sl]).double()
            del phi, wphi
        A.diagonal().add_(RIDGE * A.diagonal().mean().clamp_min(1e-12))
        try:
            g = torch.cholesky_solve(b[:, None], torch.linalg.cholesky(A))[:, 0]
        except RuntimeError:            # not positive definite: fall back to a general solve
            g = torch.linalg.solve(A, b)
        g = g.float()
        left = torch.cat([target[i:i + SOLVE_CHUNK] - self._phi(pts[i:i + SOLVE_CHUNK]) @ g
                          for i in range(0, len(pts), SOLVE_CHUNK)])
        wn = w / w.sum()
        return g, float((wn * target ** 2).sum().sqrt()), float((wn * left ** 2).sum().sqrt())


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


def _bin_name(lo: float, hi: float) -> str:
    if not np.isfinite(lo):
        return f"<{hi:g}"
    if not np.isfinite(hi):
        return f">={lo:g}"
    return f"{lo:g}-{hi:g}"


def report_corpus(bodies, data, codes, before, after, fit_dice=None,
                  convexity: np.ndarray | None = None) -> bool:
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
    r0 = float(np.median(before)) if before else float("nan")
    r1 = float(np.median(after)) if after else float("nan")
    bad = []
    if dmax < 1e-4:
        bad.append(f"the correction is dead: max|Delta| = {dmax:.3e}. Every body in this "
                   f"corpus IS its convex core.")
    if gvar < 1e-12:
        bad.append(f"codes do not vary across bodies: amplitude variance {gvar:.3e}. "
                   f"Every body was assigned the same code.")
    loss_note = f"weighted residual (median over bodies) {r0:.4f} -> {r1:.4f}"
    print(f"  [check] max|Delta| {dmax:.4f}, |g|max {gmax:.4f}, g variance {gvar:.5f}, "
          f"{loss_note}", flush=True)
    if before and not (r1 < 0.5 * r0):
        print(f"  WARNING: the amplitudes barely reduced the residual ({r0:.4f} -> {r1:.4f}). "
              f"The corpus was still written, but these bodies are close to their own hulls.",
              flush=True)
    # The decisive check, because it compares the decoded body with the body itself rather
    # than with the sample points it was fitted from. An under-determined solve reproduces
    # its points and not its body: the residual falls, the amplitudes vary, nothing above
    # fires, and the decoded bodies are wrong. Only Dice sees that.
    if fit_dice is not None:
        d = np.asarray(fit_dice, dtype=float)
        d = d[np.isfinite(d)]
        if len(d) and float(np.median(d)) < DICE_FLOOR:
            bad.append(f"the fitted bodies do not reproduce the bodies they were fitted to: "
                       f"median Dice {float(np.median(d)):.3f} over {len(d)} sampled bodies, "
                       f"against a floor of {DICE_FLOOR}. The usual cause is too few sample "
                       f"points for how deeply carved the library is; raise --points.")
        if convexity is not None:
            conv = np.asarray(convexity, dtype=float)
            raw_dice = np.asarray(fit_dice, dtype=float)
            edges = (-np.inf,) + tuple(FIT_CONVEXITY_BINS) + (np.inf,)
            print("  [dice] fitted Dice by source convexity bin:", flush=True)
            for lo, hi in zip(edges[:-1], edges[1:]):
                in_bin = (conv >= lo) & (conv < hi)
                sampled = in_bin & np.isfinite(raw_dice)
                if not in_bin.any():
                    continue
                if sampled.any():
                    med = float(np.median(raw_dice[sampled]))
                    mn = float(np.min(raw_dice[sampled]))
                    print(f"    {_bin_name(lo, hi):<9} bodies {int(in_bin.sum()):>5}  "
                          f"sampled {int(sampled.sum()):>3}  median {med:.3f}  min {mn:.3f}",
                          flush=True)
                    if hi <= 0.85 and int(sampled.sum()) >= 3 and med < DICE_FLOOR:
                        bad.append(f"nonconvex fitted bodies in convexity bin "
                                   f"{_bin_name(lo, hi)} have median Dice {med:.3f}, "
                                   f"below {DICE_FLOOR}; raise --points or simplify the "
                                   f"library before training the flow.")
                else:
                    print(f"    {_bin_name(lo, hi):<9} bodies {int(in_bin.sum()):>5}  "
                          f"sampled   0  WARNING no fitted-Dice coverage", flush=True)
    for b_ in bad:
        print(f"  ERROR: {b_}", flush=True)
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=40)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel workers for independent SDF/support preprocessing")
    ap.add_argument("--points", type=int, default=60000,
                    help="uniform SDF sample points per body; half as many jittered "
                         "surface points are added on top. The amplitudes are solved from "
                         "these, and how far the code is a property of the body rather than "
                         "of the sampling improves as their square root, so this is worth "
                         "as much as the signed distances can be afforded")
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
    ap.add_argument("--sdf", choices=("auto", "gpu", "cpu"), default="auto",
                    help="where the sample points' signed distances are computed, which is "
                         "the whole cost of the preprocessing. auto takes the GPU when "
                         "there is one; it is several times quicker than the trimesh path "
                         "and, being in this process, leaves --workers unused. cpu keeps "
                         "the trimesh path over --workers processes, which can be the "
                         "quicker of the two on a machine with many cores and a weak GPU.")
    a = ap.parse_args()
    # The amplitudes are the solution of a system with N_SITES unknowns, and the sample points
    # are its equations. Below a few equations per unknown only the ridge decides the answer,
    # and the fit returns large amplitudes that reproduce the sample points and not the body:
    # the corpus is then quietly wrong, and the flow trains on it for as long as the run takes.
    # A carved body needs the margin more than a smooth one, because more of its amplitudes
    # are doing work. This is a precondition, so it is checked before any body is loaded.
    n_samples = a.points + a.points // 2
    if n_samples < POINTS_PER_SITE * N_SITES:
        raise SystemExit(
            f"--points {a.points} gives {n_samples} sample points for {N_SITES} amplitudes, "
            f"under the {POINTS_PER_SITE} per amplitude the solve needs to be determined by "
            f"the body rather than by the ridge. Use --points "
            f"{int(np.ceil(POINTS_PER_SITE * N_SITES / 1.5))} or more.")

    from hac26.library_io import load_library_dir
    print(f"[1] loading {a.bodies} bodies from {a.shapes_dir}", flush=True)
    loaded = load_library_dir(a.shapes_dir, n=a.bodies, seed=a.seed, with_entries=True)
    shape_list = [(v, f) for v, f, _ in loaded]
    families = [str(e.get("base", "unknown")) for _, _, e in loaded]
    convexity = np.array([float(e.get("convexity", np.nan)) for _, _, e in loaded])
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

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if a.sdf == "gpu" and dev.type != "cuda":
        raise SystemExit("--sdf gpu was asked for but the fit is on the CPU; drop the flag "
                         "or pass --device cuda.")
    sdf_dev = dev if (a.sdf == "gpu" or (a.sdf == "auto" and dev.type == "cuda")) else None

    data, h0s = [None] * len(shape_list), [None] * len(shape_list)
    ref = ImplicitBody()
    nrm = ref.core.n.detach().cpu().numpy()
    jobs = [(i, np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64), nrm,
             a.points, sdf_dev)
            for i, (v, f) in enumerate(shape_list)]
    workers = max(1, int(a.workers))
    if sdf_dev is not None:
        # the signed distances are the whole cost here and they are now on the card, so the
        # pool would only fork processes to wait on it
        v0, f0 = shape_list[0]
        err = check_sdf_backend(np.asarray(v0, dtype=np.float64),
                                np.asarray(f0, dtype=np.int64), sdf_dev)
        print(f"  signed distances on {sdf_dev}, checked against trimesh on the first body "
              f"to {err:.1e}" + (f"; --workers {workers} not used" if workers > 1 else ""),
              flush=True)
        workers = 1
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

    fitter = LatticeFit(dev)
    print(f"[2] fit on {dev}: {len(data)} bodies x {N_SITES} amplitudes against "
          f"{DESIGN_N} core normals, solved one body at a time", flush=True)

    t0 = time.time()
    bodies, before, after = [], [], []
    for i, (pts, sd) in enumerate(data):
        g, r0, r1 = fitter.solve(pts, sd, h0s[i])
        b = ImplicitBody().to(dev)
        with torch.no_grad():
            b.core.set_support(torch.as_tensor(h0s[i]))
            b.delta.g.copy_(g)
        bodies.append(b)
        before.append(r0); after.append(r1)
        if (i + 1) % 25 == 0 or i + 1 == len(data):
            print(f"    fitted {i + 1}/{len(data)} bodies, residual median "
                  f"{np.median(before):.4f} -> {np.median(after):.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

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
        "surface_band": float(SURFACE_BAND),
        "weight_floor": float(WEIGHT_FLOOR),
        "ridge": float(RIDGE),
        "seed": int(a.seed),
        "shapes_dir": str(Path(a.shapes_dir)),
    }
    fit_d = (fitted_dice(bodies, shape_list, families, a.dice_bodies, seed=a.seed)
             if a.dice_bodies > 0 else np.full(len(bodies), np.nan))
    np.savez(a.out, codes=codes, support=sup, fit_dice=fit_d, family=np.array(families),
             convexity=convexity.astype(np.float32), radius=radii,
             meta=json.dumps(meta, sort_keys=True))
    print(f"  codes {codes.shape}, amplitude variance {codes[:, N_DIR:].var(0).mean():.5f}")
    print(f"  wrote {a.out}")
    if not report_corpus(bodies, data, codes, before, after, fit_dice=fit_d,
                         convexity=convexity):
        raise SystemExit("fit_shapes: the corpus above is degenerate. It was written so the "
                         "fit is not lost, but do not train on it.")


if __name__ == "__main__":
    main()
