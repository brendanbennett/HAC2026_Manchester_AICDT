"""Reading the challenge curve files.

A curve file has one row per frame and N_CAMS + 1 columns: the frame time, then one curve
per camera in the released column order (per azimuth: two horizontal cameras, top, virtual
bottom), the same order as hac26.geometry.build_cameras(). Files are named
Asteroid<NN>_lightcurve_<intensity|binary>[_blender].txt.

fit_conventions estimates the rotation sense, azimuth handedness and Lambert weight of the
convex operator on a public model whose shape is known.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hac26.forward.convex_egi import normalize_np
from .conventions import PUBLIC_MODELS
from .geometry import build_cameras
from .shapes import hull_mesh, mesh_curves_convex

N_CAMS = 28


def public_stl(data_dir: str, model: int) -> str:
    """Path of a public model's released shape inside the dataset directory."""
    if model not in PUBLIC_MODELS:
        raise ValueError(f"model {model} has no released shape; public models are "
                         f"{PUBLIC_MODELS}")
    return str(Path(data_dir) / f"AsteroidModel0{model}_shape_public" / f"asteroid{model}.stl")


def read_curves29(path: str) -> dict:
    """Read one curve file into {'time': (m,), 'curves': (N_CAMS, m)}. The delimiter, comma
    or whitespace, is detected from the first line."""
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
    """Write curves in the layout read_curves29 reads, whitespace-separated."""
    assert curves.shape[0] == N_CAMS
    np.savetxt(path, np.column_stack([time, curves.T]))


def resample_curves(curves: np.ndarray, m: int) -> np.ndarray:
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
    """Assemble one model's [intensity, binary] curve stack, resampled to m frames.

    Returns {'curves': (2 * N_CAMS, m), 'mask': (2 * N_CAMS,), 'files': {type: path},
    'native': {type: (N_CAMS, m0)}}. A missing file leaves its block at zero with mask 0.
    `renormalize` divides each curve by its mean, which leaves an already mean-normalised
    file unchanged.

    'native' holds the curves at the frame rate of the files, before the resampling. The
    noise has to be estimated there: `hac26.noise.sigma_from_highfreq` reads successive
    differences, and resampling ~841 frames down to the operator's phase grid turns those
    into the curvature of the signal. `native_sigma` does that for a whole model.
    """
    import glob as _glob

    suffix = "_blender" if use_blender else ""
    stack = np.zeros((2 * N_CAMS, m))
    mask = np.zeros(2 * N_CAMS, dtype=np.float32)
    found = {}
    native = {}
    for j, ctype in enumerate(("intensity", "binary")):
        # The released archive spells the model number both zero-padded (Asteroid10) and
        # zero-prefixed (Asteroid010), and files sit in nested per-model folders, so both
        # spellings are searched recursively.
        names = {f"Asteroid{model_idx:02d}_lightcurve_{ctype}{suffix}.txt",
                 f"Asteroid0{model_idx}_lightcurve_{ctype}{suffix}.txt"}
        hits: list = []
        for nm in names:
            hits += _glob.glob(str(Path(data_dir) / "**" / nm), recursive=True)
            hits += _glob.glob(str(Path(data_dir) / nm))
        p = Path(sorted(hits)[0]) if hits else None
        if p is not None and p.exists():
            raw = read_curves29(str(p))["curves"]
            cur = resample_curves(raw, m)
            if renormalize:
                raw = normalize_np(raw)
                cur = normalize_np(cur)
            sl = slice(j * N_CAMS, (j + 1) * N_CAMS)
            stack[sl] = cur
            mask[sl] = 1.0
            found[ctype] = str(p)
            native[ctype] = raw
    return {"curves": stack, "mask": mask, "files": found, "native": native}


def native_sigma(d: dict) -> np.ndarray:
    """(2 * N_CAMS,) noise sigma per curve of a `load_model_curves` result, estimated at the
    files' own frame rate. Blocks whose file is missing get the median of the present ones;
    a result with no files at all raises."""
    from .noise import sigma_from_highfreq

    out = np.full(2 * N_CAMS, np.nan)
    for j, ctype in enumerate(("intensity", "binary")):
        if ctype in d["native"]:
            out[j * N_CAMS:(j + 1) * N_CAMS] = sigma_from_highfreq(d["native"][ctype])
    if np.isnan(out).all():
        raise ValueError("no curve file was found; cannot estimate the noise")
    return np.where(np.isnan(out), np.nanmedian(out), out)


def fit_conventions(verts: np.ndarray, faces: np.ndarray, curves56: np.ndarray,
                    mask: np.ndarray, m: int,
                    c_grid=(0.0, 0.05, 0.1, 0.2, 0.4, 0.8)) -> dict:
    """Estimate (sigma, delta, c_lambert) for a public model with a known mesh.

    Minimises the summed squared misfit between the measured normalised curves and the
    normalised convex-operator curves of the mesh's convex hull, over the finite candidate
    set {+-1} x {+-1} x c_grid. The true body may be non-convex; the hull is enough to
    identify the signs. Returns the best candidate with its misfit under 'err'.
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
