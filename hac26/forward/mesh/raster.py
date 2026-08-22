"""Perspective rasterisation and the reduction to the two curves.

    I_c(psi) = sum_p val_p * 1[val_p > tau_I]          summed pixel value
    N_c(psi) = sum_p       1[val_p > tau_B,c]          pixel count
    then each curve is divided by its own mean over psi

tau_B,c is Otsu, computed on the FULL frame at a single reference phase -- not on a crop
around the body. A crop changes the class weights that Otsu balances, which moves the
threshold, and it moves it in the direction that matters: toward or away from the dim
grazing-incidence pixels that carry the terminator geometry. It is recomputed once per
outer iteration and stop-gradiented within the iteration, so the threshold is a constant of
the current linearisation rather than something the optimiser can chase.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["perspective", "look_at", "Rasteriser", "otsu_threshold", "reduce_curves",
           "normalise_curves"]


def perspective(fov_y_rad: float, aspect: float, near: float = 0.1,
                far: float = 100.0, device=None) -> torch.Tensor:
    """Standard OpenGL-style projection matrix, which is what nvdiffrast expects."""
    f = 1.0 / np.tan(fov_y_rad / 2.0)
    m = torch.zeros(4, 4, device=device, dtype=torch.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0),
            device=None) -> torch.Tensor:
    eye = np.asarray(eye, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    u = np.asarray(up, dtype=np.float64)
    f = t - eye
    f = f / np.linalg.norm(f)
    if abs(float(f @ (u / np.linalg.norm(u)))) > 0.999:      # camera on the up axis
        u = np.array([0.0, 1.0, 0.0])
    s = np.cross(f, u); s /= np.linalg.norm(s)
    v = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, v, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return torch.tensor(m, device=device, dtype=torch.float32)


class Rasteriser:
    """Perspective first-hit rasterisation with nvdiffrast, plus the per-pixel geometry
    the sensor chain needs (off-axis cosine and normalised radius)."""

    def __init__(self, height: int = 1080, width: int = 1920, supersample: int = 4,
                 device: str = "cuda"):
        from ..shared._nvdr import load
        self.dr = load()
        self.h, self.w, self.ss = height, width, supersample
        self.device = device
        self.ctx = self.dr.RasterizeCudaContext(device=device)
        self._px_cache: dict = {}

    @property
    def resolution(self) -> list:
        return [self.h * self.ss, self.w * self.ss]

    def pixel_geometry(self, fov_y_rad: float):
        """cos(off-axis angle) and normalised radius for every supersampled pixel."""
        key = (fov_y_rad, self.h, self.w, self.ss)
        if key in self._px_cache:
            return self._px_cache[key]
        H, W = self.resolution
        aspect = self.w / self.h
        ty = np.tan(fov_y_rad / 2.0)
        yy = torch.linspace(ty, -ty, H, device=self.device)
        xx = torch.linspace(-ty * aspect, ty * aspect, W, device=self.device)
        gx, gy = torch.meshgrid(xx, yy, indexing="xy")
        cos_off = 1.0 / torch.sqrt(1.0 + gx ** 2 + gy ** 2)
        r = torch.sqrt(gx ** 2 + gy ** 2)
        r = r / r.max()
        out = (cos_off[None], r[None])
        self._px_cache[key] = out
        return out

    def render(self, verts: torch.Tensor, faces: torch.Tensor,
               vert_radiance: torch.Tensor, eye: np.ndarray,
               fov_y_rad: float, antialias: bool = True):
        """Returns (radiance image (1,H,W), coverage mask (1,H,W)) at supersampled size."""
        return self.render_batch(verts, faces, vert_radiance, eye[None], fov_y_rad, antialias)

    def render_batch(self, verts: torch.Tensor, faces: torch.Tensor,
                     vert_radiance: torch.Tensor, eyes: np.ndarray,
                     fov_y_rad: float, antialias: bool = True):
        """Same, for many eyes at once. Returns (radiance (B,H,W), coverage (B,H,W)).

        The radiance is shared across the batch, which is the case that matters: it depends
        on the source direction, so every camera at one rotation phase sees the same one.
        Rasterising them together is one kernel launch instead of 28.
        """
        dev = self.device
        proj = perspective(fov_y_rad, self.w / self.h, device=dev)
        mvp = torch.stack([proj @ look_at(np.asarray(e), device=dev) for e in eyes])
        v_h = torch.cat([verts, torch.ones(len(verts), 1, device=dev, dtype=verts.dtype)], 1)
        clip = torch.einsum("vj,bij->bvi", v_h, mvp).contiguous()
        tri = faces.to(torch.int32).contiguous()
        rast, _ = self.dr.rasterize(self.ctx, clip, tri, resolution=self.resolution)
        attr = vert_radiance.reshape(1, -1, 1)
        img, _ = self.dr.interpolate(attr, rast, tri)
        if antialias:
            img = self.dr.antialias(img, rast, clip, tri)
        return img[..., 0], (rast[..., 3] > 0).to(img.dtype)


def otsu_threshold(frame: torch.Tensor, bins: int = 256) -> float:
    """Otsu's threshold on the FULL frame. Returns a plain float: it is stop-gradiented.

    Standard between-class variance maximisation, computed on the histogram of the whole
    image rather than a crop -- see the module docstring for why the crop matters.
    """
    x = frame.detach().reshape(-1).clamp(0.0, 1.0)
    hist = torch.histc(x, bins=bins, min=0.0, max=1.0)
    p = hist / hist.sum().clamp_min(1.0)
    centres = (torch.arange(bins, device=x.device, dtype=x.dtype) + 0.5) / bins
    w0 = torch.cumsum(p, 0)
    w1 = 1.0 - w0
    m0 = torch.cumsum(p * centres, 0) / w0.clamp_min(1e-12)
    mt = (p * centres).sum()
    m1 = (mt - torch.cumsum(p * centres, 0)) / w1.clamp_min(1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centres[int(torch.argmax(between))])


def reduce_curves(value_image: torch.Tensor, tau_i: float, tau_b: float):
    """(I, N) for one frame: summed value above tau_I, and pixel count above tau_B."""
    v = value_image
    i = (v * (v > tau_i)).sum(dim=(-2, -1))
    n = (v > tau_b).to(v.dtype).sum(dim=(-2, -1))
    return i, n


def normalise_curves(curves: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Divide each curve by its own mean over phase, as the organisers do."""
    return curves / curves.mean(dim=-1, keepdim=True).clamp_min(eps)
