"""Adapter between the team dataset and the trainer. Self-contained contract:

Expected layout (flexible; see load_pairs):
    <root>/<name>.stl|.obj          ground-truth mesh (challenge pose preferred)
    <root>/<name>_curves.npz        with 'curves' (56, m) [28 intensity + 28 binary,
                                    per-curve mean-normalized], optional 'mask' (56,)
If a mesh has no curves file, curves are SIMULATED here with the exact convex
operator (conventions sigma=-1, delta=+1 baked in) so training can start anyway.

Yields training triples (d, mask, p) like hac26.train.SyntheticCurves, plus the
optional Track-1 residual channel r = d - T(p_hull) (concavity signal).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from hac26.data_io import _resample
from forward_models.convex_egi import normalize_np
from hac26.geometry import make_grid
from hac26.radial import fibonacci_sphere, mesh_radial
from hac26.shapes import (canonicalize_r, hull_mesh, mesh_support, mesh_to_egi,
                          rescale_touch_z)
from hac26.stl_io import load_stl
from hac26.noise import apply_noise


def _load_obj(path: str) -> tuple:
    v, f = [], []
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            v.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            f.append([int(w.split("/")[0]) - 1 for w in t[1:4]])
    return np.asarray(v), np.asarray(f)


def curves_from_npz(path: str, m: int, eps: float = 1e-3) -> tuple:
    """(d (56,m) float32, mask (56,) float32) from one stored curves npz.

    Accepts the legacy single-array layout ('curves' (56,m)) and the team
    make_dataset schema ('intensity'/'binary' (frames,28) + 'azimuth'/'elevation'
    (28,)), reordering the latter into challenge-camera order. Stored curves are
    periodically resampled onto m, then per-curve mean-normalized.
    """
    z = np.load(path)
    if "curves" in z:                          # legacy single-array layout
        raw = z["curves"]
    else:                                      # team schema: (frames,28) x 2 + geometry
        from hac26.geometry import build_cameras
        cols = list(zip(np.round(z["azimuth"], 3), np.round(z["elevation"], 3)))
        order, used = [], set()
        for cam in build_cameras():
            j = next(k for k, (a, e) in enumerate(cols) if k not in used
                     and a == round(cam.azimuth_deg, 3)
                     and abs(e - cam.elevation_deg) < 0.51)
            order.append(j)
            used.add(j)
        raw = np.concatenate([z["intensity"].T[order], z["binary"].T[order]])
    d = normalize_np(_resample(np.asarray(raw, dtype=float), m), eps=eps).astype(np.float32)
    mask = np.asarray(z.get("mask", np.ones(len(d)))).astype(np.float32)
    return d, mask


def load_pairs(root: str, grid, A: np.ndarray, eps: float = 1e-3,
               canonical_r: bool = False, rays: np.ndarray | None = None) -> list:
    """[(d (56,m), mask (56,), p (N,), h (N,)), ...] for every mesh under root.

    h is the support function of the posed hull, the target for the
    support-function head (see hac26.shapes.mesh_support).

    m is taken from the operator (A is (56, m, N)); stored curves with a
    different frame count are periodically resampled onto it, so a dataset can
    be trained with any preset (no-op when they already agree, e.g. the team
    dataset's 360 frames against the `gpu` preset)."""
    m = A.shape[1]
    pairs = []
    for mp in sorted(glob.glob(str(Path(root) / "**" / "*.stl"), recursive=True)
                     + glob.glob(str(Path(root) / "**" / "*.obj"), recursive=True)):
        verts, faces = (_load_obj(mp) if mp.endswith(".obj") else load_stl(mp))
        verts = rescale_touch_z(verts)
        hv, hf = hull_mesh(verts)
        g_true = mesh_to_egi(hv, hf, grid)      # curves (when simulated) use the TRUE body
        if canonical_r:
            hv, hf = hull_mesh(canonicalize_r(hv))
        g = mesh_to_egi(hv, hf, grid)
        p = (g / max(g.sum(), 1e-12)).astype(np.float32)
        h = mesh_support(hv, grid.normals).astype(np.float32)
        r_true = float(np.sqrt((verts[:, :2] ** 2).sum(1)).max())
        cands = [Path(mp).with_suffix("").as_posix() + "_curves.npz",
                 Path(mp).with_suffix(".npz").as_posix()]
        cp = next((c for c in cands if Path(c).exists()), None)
        if cp:
            d, mask = curves_from_npz(cp, m, eps=eps)
        else:  # simulate with the exact convex operator
            raw = np.einsum("cmn,n->cm", A, g_true)
            d = normalize_np(raw, eps=eps).astype(np.float32)
            mask = np.ones(len(d), dtype=np.float32)
        rho = (mesh_radial(hv, hf, rays).astype(np.float32) if rays is not None
               else np.zeros(1, dtype=np.float32))
        pairs.append((d, mask, p, h, r_true, rho))
    return pairs


class FigurineCurves(IterableDataset):
    """Streams dataset pairs with the same augmentations as SyntheticCurves
    (noise, +-shift, curve dropout) applied on top of the stored curves."""

    def __init__(self, pairs: list, pr, mix_synthetic: float = 0.25, grid=None,
                 A: np.ndarray | None = None):
        self.pairs, self.pr = pairs, pr
        self.mix, self.grid, self.A = mix_synthetic, grid, A
        self.rays = (fibonacci_sphere(pr.n_rays)
                     if getattr(pr, "dice_weight", 0.0) else None)

    def __iter__(self):
        from hac26.shapes import sample_damit_shape, sample_training_shape
        wi = get_worker_info()
        rng = np.random.default_rng(self.pr.seed + (wi.id + 1) * 9973 if wi else self.pr.seed)
        pool = None
        if getattr(self.pr, "shape_source", "synthetic") == "damit":
            from hac26.damit import load_damit_pool
            pool = load_damit_pool(self.pr.damit_dir, max_models=self.pr.damit_max_models)
            if not pool:
                raise RuntimeError(f"no DAMIT shape.txt found under {self.pr.damit_dir}")
        while True:
            if self.mix > 0 and rng.random() < self.mix:  # breadth reserve
                s = (sample_damit_shape(rng, self.grid, pool) if pool is not None
                     else sample_training_shape(rng, self.grid,
                                                p_flat=getattr(self.pr, "p_flat", 0.0)))
                raw = np.einsum("cmn,n->cm", self.A, s["g"])
                d0 = normalize_np(raw, eps=self.pr.eps_norm).astype(np.float32)
                mask, p = np.ones(len(d0), np.float32), s["p"].astype(np.float32)
                tv, tf = s["verts"], s["faces"]
                if getattr(self.pr, "canonical_r", False):
                    tv, tf = hull_mesh(canonicalize_r(tv))
                    gc = mesh_to_egi(tv, tf, self.grid)
                    p = (gc / max(gc.sum(), 1e-12)).astype(np.float32)
                h = mesh_support(tv, self.grid.normals).astype(np.float32)
                r_true = float(np.sqrt((s["verts"][:, :2] ** 2).sum(1)).max())
                rho = (mesh_radial(tv, tf, self.rays).astype(np.float32)
                       if self.rays is not None else np.zeros(1, dtype=np.float32))
            else:
                d0, mask, p, h, r_true, rho = self.pairs[rng.integers(len(self.pairs))]
                d0, mask = d0.copy(), mask.copy()
            C, m = d0.shape
            # heteroscedastic, from the co-located replicate pairs (hac26.noise).
            # Curves here are already mean-normalised, hence relative=False.
            d0 = apply_noise(d0, rng, self.pr.noise_lo, self.pr.noise_hi,
                             profile=getattr(self.pr, "noise_profile", None),
                             relative=False).astype(np.float32)
            for c in range(C):
                sh = int(rng.integers(-self.pr.shift_max, self.pr.shift_max + 1))
                if sh:
                    d0[c] = np.roll(d0[c], sh)
            drop = rng.random(C) < self.pr.drop_p
            mask = mask * (~drop)
            d0 = d0 * mask[:, None]
            r_in = r_true * float(np.exp(rng.normal(0.0, getattr(self.pr, "r_jitter", 0.05))))
            yield (torch.from_numpy(d0), torch.from_numpy(mask.astype(np.float32)),
                   torch.from_numpy(p), torch.from_numpy(h),
                   torch.tensor(np.log(max(r_in, 1e-6)), dtype=torch.float32),
                   torch.from_numpy(rho))
