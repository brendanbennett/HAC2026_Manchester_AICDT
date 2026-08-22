"""Measurement noise for the HAC 2026 lightcurves.

Depends on numpy only, so it can be dropped into any pipeline.

At each azimuth two cameras sit at the same place and see the same body at the same instant,
so their difference is measurement noise with no geometry in it:

    sigma_c^2 = mean( (x_a - x_b)^2 ) / 2

`replicate_offset_r2` checks the pair really is co-located rather than assuming it; a pair at
slightly different azimuths would differ by geometry, and calling that noise would inflate
sigma.

The noise is strongly heteroscedastic across the array and concentrates at the high-phase
azimuths, so treating all 56 curves as equally reliable is wrong. NOISE_PROFILE carries the
per-azimuth shape, normalised to mean 1 so applying it changes the distribution of noise
across the array without changing its overall level.
"""
from __future__ import annotations

import numpy as np

__all__ = ["AZIMUTHS_DEG", "NOISE_PROFILE", "sigma_from_replicates",
           "replicate_offset_r2", "apply_noise"]

AZIMUTHS_DEG = (0.0, 45.0, 90.0, 135.0, 225.0, 270.0, 315.0)

# Per-azimuth median over the three public models, normalised to mean 1 so that applying it
# changes the DISTRIBUTION of noise across the array without changing its overall scale.
# Curve layout: 28 intensity curves then 28 binary, four cameras per azimuth in the order
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


def replicate_offset_r2(curves: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Guard on `sigma_from_replicates`: is the pair difference noise, or geometry?

    Rotating the body by dpsi is equivalent to moving a camera in azimuth, so to first
    order a genuine offset gives  x_a - x_b = dpsi * d/dpsi(mean curve).  Regressing the
    difference on that derivative returns R^2 = the FRACTION of the difference explained by
    a rigid offset. Near 0 means the pair is co-located and the difference is honest noise;
    near 1 means the cameras are not where they are documented to be and sigma is inflated.

    On the public models this comes out near zero on the geometric (binary) channel and at
    the noisy azimuths, so the spread across curves is noise rather than geometry.
    """
    C, m = curves.shape
    mask = np.ones(C) if mask is None else np.asarray(mask)
    out = np.zeros(C)
    for _, a, b in _pairs(C):
        if not (mask[a] > 0 and mask[b] > 0):
            continue
        mean = 0.5 * (curves[a] + curves[b])
        der = np.gradient(mean) * m / (2.0 * np.pi)
        diff = curves[a] - curves[b]
        den = float(der @ der)
        k = float(der @ diff) / den if den > 1e-12 else 0.0
        res = diff - k * der
        r2 = 1.0 - float(res @ res) / max(float(diff @ diff), 1e-12)
        out[(a // 4) * 4: (a // 4) * 4 + 4] = r2
    return out


def apply_noise(curves: np.ndarray, rng: np.random.Generator,
                scale_lo: float = 0.005, scale_hi: float = 0.03,
                profile: np.ndarray | None = None,
                relative: bool = True) -> np.ndarray:
    """Add heteroscedastic noise to generated curves.

    One overall scale is drawn per body from [scale_lo, scale_hi]; `profile` then
    redistributes it across the array. With the default profile the near-backlit curves
    receive more noise than the well-lit ones, while the mean level
    is unchanged -- so switching a pipeline from homoscedastic to this changes only WHERE
    the noise goes, and any existing noise-scale tuning stays valid.

    relative=True scales the noise by each curve's own mean, appropriate before the
    organisers' per-curve mean normalisation; pass False if the curves are already
    normalised.
    """
    curves = np.asarray(curves, dtype=np.float64)
    C = curves.shape[0]
    p = NOISE_PROFILE if profile is None else np.asarray(profile)
    if len(p) < C:
        raise ValueError(f"profile has {len(p)} entries, need at least {C}")
    p = p[:C, None]
    sig = rng.uniform(scale_lo, scale_hi)
    level = curves.mean(axis=1, keepdims=True) if relative else 1.0
    return curves + sig * p * level * rng.standard_normal(curves.shape)
