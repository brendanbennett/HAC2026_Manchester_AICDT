"""Measurement noise for the HAC 2026 lightcurves.

Depends on numpy only, and on nothing else in this package, so it can be dropped into any
generation or training pipeline.

WHY THIS EXISTS. Synthetic curves are generated noiselessly. A model trained on them learns
that all 56 curves are equally reliable. They are not: the measured noise spans 226x across
the camera array and is concentrated at two of the seven azimuths, so a noiseless prior
systematically over-trusts the worst geometries at inference time.

HOW THE NOISE IS MEASURED, with no model. At each azimuth two cameras sit at the same
elevation and view the same body at the same instant, so their difference is pure
measurement noise:

    sigma_c^2 = mean( (x_a - x_b)^2 ) / 2

That identity is the whole method. It assumes only that the pair really is co-located,
which `replicate_offset_r2` below checks rather than assumes -- a pair at slightly
different azimuths would differ by GEOMETRY, and calling that noise would inflate sigma.

WHAT WAS MEASURED (three public models, real curves, mean-normalised units):

    model  type        0      45      90     135     225     270     315
      1  intensity  0.0035  0.0024  0.0052  0.0513  0.0540  0.0055  0.0023
      1  binary     0.0041  0.0039  0.0072  0.0370  0.0491  0.0083  0.0046
      2  intensity  0.0205  0.0215  0.0602  0.4120  0.1299  0.0444  0.0180
      2  binary     0.0262  0.0272  0.0223  0.1510  0.1563  0.0789  0.0183
      3  intensity  0.0174  0.0049  0.0089  0.0322  0.0328  0.0018  0.0046
      3  binary     0.0074  0.0293  0.0141  0.0435  0.0483  0.0166  0.0073

min 0.0018, max 0.4120. The spread is not scattered: it concentrates at 135 and 225
degrees, where the body is near-backlit, the lit area collapses, and the organisers'
per-curve mean normalisation then amplifies what little signal remains.
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

    Measured on the public models: R^2 <= 0.002 on the geometric (binary) channel, and
    ~0.001 at the noisy azimuths 135/225 -- so the 226x spread is real noise, not geometry.
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
    receive 2.51x the noise and the well-lit ones 0.53x, a 4.7x ratio, while the mean level
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
