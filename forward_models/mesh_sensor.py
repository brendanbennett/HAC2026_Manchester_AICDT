"""The sensor chain, applied in the order the hardware applies it.

    radiance L
      -> natural off-axis falloff cos^4(theta_off) x fitted radial vignetting
      -> convolution with the measured PSF
      -> OETF (a monotone spline, not a power law)
      -> clip at saturation
      -> quantise to 8 bits, straight-through in the backward pass
      -> box-downsample from 4x supersampling

What is absent: any 1/d^2 factor. Radiance is conserved along a ray, so the
image irradiance produced by an extended surface does not depend on how far away it is. The
perspective effect is entirely in how many PIXELS a surface element covers, which the
rasteriser already handles. Putting a 1/d^2 on pixel VALUES would double-count it; the
roughly 2x difference between near and far limb is a projected-area effect, not a
brightness one. The only radiometric falloff here is off-axis cos^4 and fitted vignetting.

WHY THE OETF IS A SPLINE. A power law has one parameter and forces the same curvature
everywhere. Real camera transfer curves have a toe and a shoulder, and it is the shoulder
that decides which pixels survive the Otsu threshold -- exactly the pixels that carry the
grazing-incidence geometry. Monotonicity is enforced by construction (softplus increments)
rather than hoped for, because a non-monotone OETF would make the value threshold
multi-valued and the coarea derivative meaningless.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SensorModel", "gaussian_psf", "box_downsample", "quantise_ste"]


def gaussian_psf(sigma_px: float, radius: int | None = None,
                 device=None, dtype=torch.float32) -> torch.Tensor:
    """Separable Gaussian PSF kernel. Stands in for the measured PSF until it is fitted."""
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * sigma_px)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / max(sigma_px, 1e-6)) ** 2)
    return k / k.sum()


def _separable_conv(img: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """img (B,H,W) convolved with the separable kernel k, edges replicated."""
    b, h, w = img.shape
    r = (len(k) - 1) // 2
    x = img[:, None]
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x[:, 0]


def box_downsample(img: torch.Tensor, factor: int) -> torch.Tensor:
    """Box average over factor x factor blocks. Supersample, then average, then threshold."""
    if factor == 1:
        return img
    b, h, w = img.shape
    if h % factor or w % factor:
        raise ValueError(f"image {h}x{w} is not divisible by the supersample factor {factor}")
    return img.reshape(b, h // factor, factor, w // factor, factor).mean(dim=(2, 4))


class _QuantiseSTE(torch.autograd.Function):
    """8-bit quantisation, identity on the backward pass."""

    @staticmethod
    def forward(ctx, x, levels: int):
        return torch.round(x * (levels - 1)) / (levels - 1)

    @staticmethod
    def backward(ctx, g):
        return g, None


def quantise_ste(x: torch.Tensor, levels: int = 256) -> torch.Tensor:
    return _QuantiseSTE.apply(x, levels)


class SensorModel(nn.Module):
    """The sensor chain. Every fitted quantity is a parameter here, pinned by the calibration.

    `vignette` is the radial polynomial in normalised image radius r in [0, 1]:
        V(r) = 1 + a1 r^2 + a2 r^4 + a3 r^6
    even powers only, because a lens is radially symmetric and an odd term would put a cusp
    on the optical axis.
    """

    def __init__(self, n_oetf_knots: int = 8, psf_sigma_px: float = 1.0,
                 saturation: float = 1.0, quantise: bool = True, levels: int = 256):
        super().__init__()
        self.raw_vignette = nn.Parameter(torch.zeros(3))
        self.raw_psf = nn.Parameter(torch.tensor(float(np.log(np.expm1(psf_sigma_px)))))
        # OETF as cumulative positive increments -> monotone by construction
        self.raw_oetf = nn.Parameter(torch.zeros(n_oetf_knots))
        self.raw_sat = nn.Parameter(torch.tensor(float(np.log(np.expm1(saturation)))))
        self.quantise, self.levels = quantise, levels

    # -------------------------------------------------------------- fitted quantities
    @property
    def psf_sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_psf)

    @property
    def saturation(self) -> torch.Tensor:
        return F.softplus(self.raw_sat)

    def oetf_knots(self) -> torch.Tensor:
        """Monotone knot values on [0, 1], starting at 0 and ending at 1."""
        inc = F.softplus(self.raw_oetf) + 1e-4
        c = torch.cumsum(inc, 0)
        return torch.cat([torch.zeros(1, device=c.device, dtype=c.dtype), c / c[-1]])

    def oetf(self, x: torch.Tensor) -> torch.Tensor:
        """Piecewise-linear monotone spline through the knots, evaluated on x in [0, 1]."""
        k = self.oetf_knots()
        n = len(k) - 1
        u = x.clamp(0.0, 1.0) * n
        i = u.floor().clamp(max=n - 1)
        t = u - i
        i = i.long()
        return torch.lerp(k[i], k[i + 1], t)

    def vignette(self, r: torch.Tensor) -> torch.Tensor:
        a = self.raw_vignette
        r2 = r ** 2
        return 1.0 + a[0] * r2 + a[1] * r2 ** 2 + a[2] * r2 ** 3

    # -------------------------------------------------------------- the chain
    def forward(self, radiance: torch.Tensor, cos_off: torch.Tensor,
                radius: torch.Tensor, supersample: int = 4) -> torch.Tensor:
        """radiance, cos_off and radius are all (B, H, W) at the SUPERSAMPLED resolution.

        cos_off is the cosine of the off-axis angle of each pixel's ray; radius is the
        normalised distance from the optical axis, in [0, 1] at the frame corner.
        """
        x = radiance * cos_off.clamp_min(0.0) ** 4 * self.vignette(radius)
        x = _separable_conv(x, gaussian_psf(float(self.psf_sigma), device=x.device,
                                            dtype=x.dtype))
        x = self.oetf(x / self.saturation.clamp_min(1e-6))
        x = x.clamp(0.0, 1.0)                       # saturation
        if self.quantise:
            x = quantise_ste(x, self.levels)
        return box_downsample(x, supersample)
