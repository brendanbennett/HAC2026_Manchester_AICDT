"""Shaded orthographic renders of a mesh, via first-hit depth on a voxel grid.

Rasterising 800k triangles in numpy is slow; voxelising them is not (eval_gate
already does it by ray-stabbing z columns). So render from the occupancy grid:
first occupied voxel along the view axis gives a depth map, the gradient of the
depth map gives normals, and Lambert plus a cheap ambient term gives an image.
Chunky at the silhouette, but it treats a 6k-face reconstruction and an 800k-face
ground truth identically, which is the point of a comparison figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_gate import occupancy  # noqa: E402
from hac26.shapes import rescale_touch_z  # noqa: E402


def _rot(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = np.radians(az_deg), np.radians(el_deg)
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    return np.array([[ca, -sa, 0.0], [sa * ce, ca * ce, -se], [sa * se, ca * se, ce]])


def pose(v: np.ndarray, f: np.ndarray, centroid: bool = True) -> np.ndarray:
    """Challenge pose: z touching +-1, xy on the volume centroid (not the vertex mean)."""
    import trimesh

    v = rescale_touch_z(np.asarray(v, float))
    if centroid:
        m = trimesh.Trimesh(v, f, process=False)
        try:
            c = np.asarray(m.centroid, float) if m.is_volume else v.mean(0)
        except Exception:
            c = v.mean(0)
        v = v.copy()
        v[:, 0] -= c[0]
        v[:, 1] -= c[1]
    return v


def render(v, f, n=176, az=35.0, el=22.0, light=(-0.42, -0.66, 0.62)):
    """-> (image HxW in [0,1], alpha mask). Camera looks along -y after rotation."""
    v = np.asarray(v, float) @ _rot(az, el).T
    extent = float(np.abs(v).max()) * 1.08
    occ = occupancy(v, np.asarray(f, np.int64), n, extent)

    # first occupied cell along the view axis (y), from the camera side
    seen = occ.any(axis=1)
    depth = np.where(seen, occ.argmax(axis=1).astype(float), np.nan)

    d = depth.copy()
    d[~seen] = np.nan
    filled = np.nan_to_num(d, nan=float(n))
    filled = ndimage.gaussian_filter(filled, 0.9)          # take the staircase off
    # surface is p(x,z) = (x, D, z), so the outward (camera-facing) normal is
    # (dD/dx, -1, dD/dz): axis 0 of the depth map is x, axis 1 is z.
    gx, gz = np.gradient(filled)
    scale = 2.2 * n / 128.0
    nrm = np.stack([gx / scale, -np.ones_like(gx), gz / scale], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)

    L = np.array(light, float)
    L /= np.linalg.norm(L)
    lam = np.clip(nrm @ L, 0, 1)

    # cheap ambient occlusion: how much closer a cell is than its neighbourhood
    near = ndimage.uniform_filter(filled, 11)
    ao = np.clip(1.0 - (filled - near) / (0.08 * n), 0.35, 1.0)

    img = 0.20 + 0.78 * lam * ao
    img = np.clip(img, 0, 1)
    return np.flipud(img.T), np.flipud(seen.T)


def draw(ax, v, f, title=None, sub=None, color=(0.82, 0.79, 0.74), **kw):
    img, mask = render(v, f, **kw)
    rgb = np.zeros(img.shape + (4,))
    rgb[..., :3] = img[..., None] * np.array(color)
    rgb[..., 3] = mask.astype(float)
    ax.imshow(rgb, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, pad=3)
    if sub:
        ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=7, color="0.35", family="monospace")
