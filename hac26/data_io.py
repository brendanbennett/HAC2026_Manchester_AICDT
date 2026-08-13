"""Challenge data IO: the 29-column lightcurve format, file discovery, and the
convention-fitting routine to run on the public models once data is present.

Column layout (challenge page, 1-indexed): column 1 = time stamp / frame index;
then 7 groups of 4 columns for azimuths 0, 45, 90, 135, 225, 270, 315 deg;
within each group: (horizontal, horizontal, top, virtual bottom) — same order as
geometry.build_cameras().

File names observed in the page's News section:
    Asteroid01_lightcurve_intensity.txt
    Asteroid01_lightcurve_binary.txt
    Asteroid01_lightcurve_intensity_blender.txt
    Asteroid01_lightcurve_binary_blender.txt

UNVERIFIED-UNTIL-DATA (kept configurable, fitted by fit_conventions):
    sigma (rotation sense), delta (azimuth handedness), semantics of the two
    horizontal columns, c_lambert of the LS+L scattering law.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .forward import normalize_np
from .geometry import build_cameras
from .shapes import hull_mesh, mesh_curves_convex, mesh_to_egi

N_CAMS = 28


def read_curves29(path: str) -> dict:
    """Read one lightcurve file -> {'time': (m,), 'curves': (28, m)}.
    Real challenge files are comma-separated; delimiter is auto-detected."""
    with open(path) as fh:
        first = fh.readline()
    delim = "," if "," in first else None
    mat = np.loadtxt(path, delimiter=delim)
    if mat.ndim == 1:
        mat = mat[None, :]
    if mat.shape[1] != 29:
        raise ValueError(f"{path}: expected 29 columns, got {mat.shape[1]}")
    return {"time": mat[:, 0].copy(), "curves": mat[:, 1:].T.copy()}


def write_curves29(path: str, time: np.ndarray, curves: np.ndarray) -> None:
    assert curves.shape[0] == N_CAMS
    np.savetxt(path, np.column_stack([time, curves.T]))


def _resample(curves: np.ndarray, m: int) -> np.ndarray:
    """Periodic linear resampling of each curve onto m uniform frames."""
    m0 = curves.shape[-1]
    if m0 == m:
        return curves
    x0 = np.arange(m0 + 1) / m0
    x1 = np.arange(m) / m
    ext = np.concatenate([curves, curves[..., :1]], axis=-1)
    return np.stack([np.interp(x1, x0, c) for c in ext], axis=0)[..., :m]


def load_model_curves(data_dir: str, model_idx: int, m: int = 360,
                      use_blender: bool = False, renormalize: bool = True) -> dict:
    """Assemble the 56-curve stack [28 intensity, 28 binary] + availability mask.

    Missing files yield zero curves with mask 0 (the LPD input convention).
    Data files are already mean-normalized per the page; renormalize is an
    idempotent safeguard.
    """
    import glob as _glob

    suffix = "_blender" if use_blender else ""
    stack = np.zeros((2 * N_CAMS, m))
    mask = np.zeros(2 * N_CAMS, dtype=np.float32)
    found = {}
    for j, ctype in enumerate(("intensity", "binary")):
        # names in the released archive: Asteroid01..Asteroid09, but Asteroid010;
        # files live in nested per-model subfolders -> recursive search
        names = {f"Asteroid{model_idx:02d}_lightcurve_{ctype}{suffix}.txt",
                 f"Asteroid0{model_idx}_lightcurve_{ctype}{suffix}.txt"}
        hits: list = []
        for nm in names:
            hits += _glob.glob(str(Path(data_dir) / "**" / nm), recursive=True)
            hits += _glob.glob(str(Path(data_dir) / nm))
        p = Path(sorted(hits)[0]) if hits else None
        if p is not None and p.exists():
            cur = _resample(read_curves29(str(p))["curves"], m)
            if renormalize:
                cur = normalize_np(cur)
            sl = slice(j * N_CAMS, (j + 1) * N_CAMS)
            stack[sl] = cur
            mask[sl] = 1.0
            found[ctype] = str(p)
    return {"curves": stack, "mask": mask, "files": found}


def fit_conventions(verts: np.ndarray, faces: np.ndarray, curves56: np.ndarray,
                    mask: np.ndarray, m: int,
                    c_grid=(0.0, 0.05, 0.1, 0.2, 0.4, 0.8)) -> dict:
    """Estimate (sigma, delta, c_lambert) on a public model with known mesh.

    Defined estimator: minimize the summed squared misfit between the measured
    normalized curves and the normalized brute-force convex-hull curves over the
    finite candidate set {+-1} x {+-1} x c_grid. (The true models may be nonconvex;
    the hull curves are the convex-model surrogate — sufficient to identify signs.)
    """
    cams = build_cameras()
    types = ["intensity"] * N_CAMS + ["binary"] * N_CAMS
    hv, hf = hull_mesh(verts)
    best = None
    for sigma in (1.0, -1.0):
        for delta in (1.0, -1.0):
            for c in c_grid:
                sim = mesh_curves_convex(hv, hf, cams + cams, m, types,
                                         c_lambert=c, sigma=sigma, delta=delta)
                r = (normalize_np(sim) - curves56) * mask[:, None]
                err = float((r ** 2).sum())
                if best is None or err < best["err"]:
                    best = {"sigma": sigma, "delta": delta, "c_lambert": c, "err": err}
    return best
