"""Measurement noise of the lightcurves.

At each azimuth two cameras sit at the same place and see the same body at the same instant,
so their difference is measurement noise with no geometry in it:

    sigma_c^2 = mean( (x_a - x_b)^2 ) / 2

This only holds because the pair is co-located: two cameras at slightly different azimuths
would differ by geometry, and calling that noise would inflate sigma.

The noise level differs a lot between azimuths. NOISE_PROFILE is its per-azimuth shape,
measured on the public models with sigma_from_replicates and normalised to mean 1; training
uses it to distribute synthetic noise across the curves, at an overall level drawn per body
from [NOISE_LO, NOISE_HI], a range that covers the levels measured on the public models. It
is a snapshot of the data and can be recomputed from the public curves with
sigma_from_replicates.
"""
from __future__ import annotations

import numpy as np

__all__ = ["AZIMUTHS_DEG", "NOISE_PROFILE", "NOISE_LO", "NOISE_HI", "sigma_from_replicates",
           "apply_noise"]

AZIMUTHS_DEG = (0.0, 45.0, 90.0, 135.0, 225.0, 270.0, 315.0)
NOISE_LO, NOISE_HI = 0.005, 0.03      # range of the mean noise level of a mean-normalised curve

# Per-azimuth median of sigma over the public models, normalised to mean 1. Curve layout:
# the intensity curves then the binary curves, four cameras per azimuth in the order
# (horizontal a, horizontal b, top, bottom).
_AZ_INTENSITY = (0.831, 0.234, 0.425, 2.449, 2.579, 0.263, 0.220)
_AZ_BINARY = (0.314, 1.153, 0.597, 1.843, 2.081, 0.703, 0.309)
NOISE_PROFILE = np.array([v for v in _AZ_INTENSITY for _ in range(4)]
                         + [v for v in _AZ_BINARY for _ in range(4)], dtype=np.float32)


def _pairs(n_curves: int):
    """Indices of the two co-located horizontal cameras at each azimuth, per channel."""
    for offset in range(0, n_curves, 28):
        for i in range(len(AZIMUTHS_DEG)):
            a, b = offset + 4 * i, offset + 4 * i + 1
            if b < n_curves:
                yield i, a, b


def sigma_from_replicates(curves: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per-curve sigma from the co-located horizontal pairs.

    curves: (C, m) mean-normalised curves. mask: (C,) non-zero where a curve is present.
    Returns (C,). All four cameras at an azimuth inherit that azimuth's sigma, since only
    the horizontal pair is replicated; curves with no usable pair get the median.
    """
    C = curves.shape[0]
    mask = np.ones(C) if mask is None else np.asarray(mask)
    out = np.full(C, np.nan)
    for _, a, b in _pairs(C):
        if mask[a] > 0 and mask[b] > 0:
            s = float(np.sqrt(((curves[a] - curves[b]) ** 2).mean() / 2.0))
            out[(a // 4) * 4: (a // 4) * 4 + 4] = s
    if np.isnan(out).all():
        raise ValueError("no usable replicate pair; cannot estimate sigma")
    out[np.isnan(out)] = np.nanmedian(out)
    return np.maximum(out, 1e-6)


def apply_noise(curves: np.ndarray, rng: np.random.Generator,
                scale_lo: float = NOISE_LO, scale_hi: float = NOISE_HI,
                profile: np.ndarray | None = None,
                relative: bool = True) -> np.ndarray:
    """Add Gaussian noise to generated curves. One overall scale is drawn per body from
    [scale_lo, scale_hi]; `profile` (default NOISE_PROFILE) distributes it across the curves
    without changing its mean level. relative=True scales the noise by each curve's own mean,
    which is right before the per-curve mean normalisation; pass False for curves that are
    already normalised."""
    curves = np.asarray(curves, dtype=np.float64)
    C = curves.shape[0]
    p = NOISE_PROFILE if profile is None else np.asarray(profile)
    if len(p) < C:
        raise ValueError(f"profile has {len(p)} entries, need at least {C}")
    p = p[:C, None]
    sig = rng.uniform(scale_lo, scale_hi)
    level = curves.mean(axis=1, keepdims=True) if relative else 1.0
    return curves + sig * p * level * rng.standard_normal(curves.shape)
