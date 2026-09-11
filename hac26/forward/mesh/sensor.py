"""The sensor chain, from a radiance image to the pixel values the curves are computed from.

    radiance L at the supersampled resolution
      -> off-axis falloff cos^4(theta_off) times a fitted radial vignetting polynomial
      -> Gaussian PSF with a fitted width
      -> divide by the fitted saturation level
      -> OETF: a monotone piecewise-linear spline on [0, 1] with fitted knots
      -> clamp to [0, 1]
      -> quantise to `levels` grey levels, identity in the backward pass
      -> box-average down to the sensor resolution

There is no 1/d^2 factor. Radiance is conserved along a ray, so the pixel value from a surface
does not depend on how far away it is; distance changes only how many pixels the surface
covers, which the rasteriser handles.

The OETF is a spline rather than a power law because real transfer curves have a toe and a
shoulder, and the shoulder decides which faint pixels survive the binary threshold. The knots
are cumulative positive increments, so the curve is monotone by construction.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SensorModel", "gaussian_psf", "box_downsample", "quantise_ste"]


def gaussian_psf(sigma_px, radius: int | None = None,
                 device=None, dtype=torch.float32) -> torch.Tensor:
    """One-dimensional Gaussian kernel, differentiable in sigma when a tensor is passed. The
    support `radius` is an integer chosen from the current sigma and is not differentiated."""
    sig = torch.as_tensor(sigma_px, device=device, dtype=dtype)
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * float(sig.detach()))))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sig.clamp_min(1e-6)) ** 2)
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
    """Box average over factor x factor blocks."""
    if factor == 1:
        return img
    b, h, w = img.shape
    if h % factor or w % factor:
        raise ValueError(f"image {h}x{w} is not divisible by the supersample factor {factor}")
    return img.reshape(b, h // factor, factor, w // factor, factor).mean(dim=(2, 4))


class _QuantiseSTE(torch.autograd.Function):
    """Quantisation to `levels` grey levels, identity on the backward pass."""

    @staticmethod
    def forward(ctx, x, levels: int):
        return torch.round(x * (levels - 1)) / (levels - 1)

    @staticmethod
    def backward(ctx, g):
        return g, None


def quantise_ste(x: torch.Tensor, levels: int = 256) -> torch.Tensor:
    return _QuantiseSTE.apply(x, levels)


class SensorModel(nn.Module):
    """The sensor chain. Every fitted quantity is a parameter here, set by the calibration.

    `vignette` is the radial polynomial in the normalised image radius r in [0, 1]:
        V(r) = 1 + a1 r^2 + a2 r^4 + a3 r^6
    even powers only, because a lens is radially symmetric.
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
        """Piecewise-linear monotone spline through the knots, evaluated on x in [0, 1].

        Each pixel's two knot values are picked by one torch.where per segment, not by
        indexing the knots with the segment index. The values are the same, but the backward
        of the index accumulates millions of pixels onto nine knots, which the indexing kernel
        does nearly serially: 0.94 s of a 0.99 s image backward for a block of 56 images,
        against a masked sum per segment here."""
        k = self.oetf_knots()
        n = len(k) - 1
        u = x.clamp(0.0, 1.0) * n
        i = u.floor().clamp(max=n - 1)
        t = u - i
        lo, hi = k[0].expand_as(u), k[1].expand_as(u)
        for m in range(1, n):
            seg = i == m
            lo = torch.where(seg, k[m], lo)
            hi = torch.where(seg, k[m + 1], hi)
        return torch.lerp(lo, hi, t)

    def vignette(self, r: torch.Tensor) -> torch.Tensor:
        a = self.raw_vignette
        r2 = r ** 2
        return 1.0 + a[0] * r2 + a[1] * r2 ** 2 + a[2] * r2 ** 3

    # -------------------------------------------------------------- the chain
    def forward(self, radiance: torch.Tensor, cos_off: torch.Tensor,
                radius: torch.Tensor, supersample: int = 4) -> torch.Tensor:
        """radiance is (B, H, W) at the SUPERSAMPLED resolution; cos_off and radius are (1, H, W)
        or (B, H, W) and broadcast against it.

        cos_off is the cosine of the off-axis angle of each pixel's ray; radius is the
        normalised distance from the optical axis, in [0, 1] at the frame corner.
        """
        gain = cos_off.clamp_min(0.0) ** 4 * self.vignette(radius)    # the same for every image
        x = radiance * gain
        x = _separable_conv(x, gaussian_psf(self.psf_sigma, device=x.device,
                                            dtype=x.dtype))
        x = self.oetf(x / self.saturation.clamp_min(1e-6))
        x = x.clamp(0.0, 1.0)                       # saturation
        if self.quantise:
            x = quantise_ste(x, self.levels)
        return box_downsample(x, supersample)
