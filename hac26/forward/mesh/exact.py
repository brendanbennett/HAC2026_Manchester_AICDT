"""The exact forward model: a triangle mesh in, the two measured curves of every geometry out,
with the derivative of any scalar function of the curves with respect to the vertices.

The chain, in the body frame, per rotation phase psi:

  1. Direct light. The mesh is rasterised from each of the K sample directions of the source
     disc, with the vertices duplicated per face. The antialiased coverage of face i in that
     view is the projected area of the lit part of face i, so the irradiance per unit area is
     e_i = (1/K) sum_k coverage_ik / A_i. Cast shadows, self-shadowing and penumbra come out
     of the rasteriser, and the antialiasing gives the exact derivative of that area with
     respect to the vertices, including the motion of a shadow edge (LitCoverage).
  2. Interreflection. The faces are grouped into patches (a decimated copy of the mesh, each
     face assigned to the nearest patch), the irradiance is averaged over each patch by area,
     and B = rho (I - rho F)^-1 E is solved with F the form factors between the patches, held
     constant. The radiance leaving a face is that of its patch, L = B / pi. Direct light
     therefore has the resolution of the mesh, interreflected light that of the patches, and
     the form-factor matrix stays small whatever the mesh.
  3. Cameras. The mesh is rasterised from every camera with L as a per-face attribute,
     antialiased, and passed through the sensor chain and the per-curve pedestal.
  4. Reduction. The intensity curve sums the pixel values above tau_i; the binary curve counts
     the pixels above tau_b, with tau_b Otsu's threshold of that camera's first frame, as the
     organisers compute it. Both derivatives come from the coarea formula.

Nothing is smoothed by hand: the only widths in the chain are the pixel of the sun view, the
pixel of the camera, and the sensor's own PSF. Four things are held constant in the
derivative: the form factors and the face-to-patch map (so the change of interreflection due
to the motion of an occluder is left out), Otsu's threshold, and the field of view, which is
set from the body's extent.

`ExactForward.raw_curves` gives the unnormalised curves. `ExactForward.vjp` gives the
vector-Jacobian product of the unnormalised curves with a cotangent, with respect to the
vertices and to any parameters that require grad, running in phase chunks so the memory stays
bounded. The per-curve mean normalisation and its adjoint are `normalise` and
`normalise_vjp`; they are kept outside the operator because the cotangent of a normalised
curve depends on every phase at once, which is why `vjp` also accepts the cotangent as a
function of the full raw curves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from hac26.conventions import S_LAB, cameras
from .instrument import Instrument
from .radiosity import RadiosityError, RadiositySolver, form_factors
from .raster import Rasteriser, flat_faces, look_at, orthographic, otsu_threshold, perspective
from ..shared.coarea import threshold_count, threshold_sum

__all__ = ["RenderConfig", "ExactForward", "LitCoverage", "normalise", "normalise_vjp",
           "rotate_z", "decimate", "source_dirs"]


@dataclass(frozen=True)
class RenderConfig:
    """The discretisation of the chain. Every value here is a resolution, not a model
    parameter; the model parameters live in Instrument."""
    height: int = 108          # sensor image, before supersampling
    width: int = 192
    supersample: int = 2
    sun_res: int = 512         # side of the square sun view, in pixels
    fov_scale: float = 1.6     # the camera's half-height covers this many body extents
    n_source: int = 8          # sample directions on the source disc
    phase_chunk: int = 8       # phases per rendering batch
    radiosity_faces: int = 600 # patches the interreflection is solved on
    form_factor_samples: int = 4
    bad_rows: str = "raise"    # what RadiositySolver does with a broken form-factor row


class Prepared(NamedTuple):
    """Everything about one mesh that does not depend on the phase."""
    solver: RadiositySolver
    patch: torch.Tensor        # (F,) patch index of every face
    fv: torch.Tensor           # (3F, 3) per-face vertices, differentiable
    ff: torch.Tensor           # (F, 3) faces indexing fv
    area: torch.Tensor         # (F,) face areas, differentiable
    extent: float              # radius of the body, with a margin


def rotate_z(w: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """R_z(angle) w for directions w (..., 3) and angles (P,), broadcast to (P, ..., 3)."""
    c, s = torch.cos(angle), torch.sin(angle)
    shape = (len(angle),) + (1,) * (w.dim() - 1)
    c, s = c.reshape(shape), s.reshape(shape)
    x, y, z = w[..., 0], w[..., 1], w[..., 2]
    return torch.stack([c * x - s * y, s * x + c * y, z.expand_as(c * x)], -1)


def decimate(verts: np.ndarray, faces: np.ndarray, target: int):
    """Quadric decimation of a mesh to about `target` faces. Meshes at or below the target
    are returned unchanged."""
    import trimesh
    if len(faces) <= target:
        return verts, faces
    d = trimesh.Trimesh(verts, faces, process=False).simplify_quadric_decimation(
        face_count=target)
    return np.asarray(d.vertices, dtype=np.float64), np.asarray(d.faces, dtype=np.int64)


def source_dirs(delta: torch.Tensor, k: int) -> torch.Tensor:
    """K unit directions on a ring of angular radius delta / sqrt(2) about the lab source
    direction, differentiable in delta. One ring gives the ring the same mean squared offset
    as a uniform disc of radius delta, so the penumbra has the right width."""
    s = torch.as_tensor(S_LAB, dtype=torch.float32, device=delta.device)
    r = delta / np.sqrt(2.0)
    e1 = torch.tensor([0.0, 1.0, 0.0], device=delta.device)
    e2 = torch.tensor([0.0, 0.0, 1.0], device=delta.device)
    ang = 2.0 * np.pi * torch.arange(k, device=delta.device, dtype=torch.float32) / k
    d = (s[None] * torch.cos(r) + torch.sin(r) * (torch.cos(ang)[:, None] * e1[None]
                                                 + torch.sin(ang)[:, None] * e2[None]))
    return d / d.norm(dim=1, keepdim=True)


class LitCoverage(torch.autograd.Function):
    """Per-face lit projected area in the view along each of several directions.

    forward(verts (3F, 3) per-face vertices, tri (F, 3), dirs (B, 3), half_width, depth,
    ras) -> (B, F): the antialiased coverage of face f in view b times the pixel area. The
    antialiasing is linear in the per-face attribute, so the coverage of every face is read
    off in one pass as the gradient of the total image sum with respect to a per-face
    attribute of ones.

    backward: the same rendering with the cotangent as the per-face attribute; the position
    gradient of the antialiasing is then exactly sum_f cot_f d coverage_f / d vertices,
    including the motion of shadow edges, and the direction gradient follows through the
    projection.
    """

    @staticmethod
    def _image_sum(verts, tri, dirs, half_width, depth, ras, attr_face):
        """sum over pixels of the antialiased image of `attr_face` (B, F), per view (B,)."""
        m = torch.stack([orthographic(d, half_width, depth, device=verts.device) for d in dirs])
        hom = torch.cat([verts, torch.ones(len(verts), 1, device=verts.device)], 1)
        clip = torch.einsum("vj,bij->bvi", hom, m)                             # (B, 3F, 4)
        attr = attr_face.repeat_interleave(3, dim=1)[..., None]                 # (B, 3F, 1)
        img, _ = ras.render(clip, tri, attr)
        return img[..., 0].sum((1, 2)) * (2.0 * half_width / ras.resolution[0]) ** 2

    @staticmethod
    def forward(ctx, verts, tri, dirs, half_width, depth, ras):
        F_ = tri.shape[0]
        with torch.enable_grad():
            ones = torch.ones(dirs.shape[0], F_, device=verts.device, requires_grad=True)
            s = LitCoverage._image_sum(verts.detach(), tri, dirs.detach(), half_width, depth,
                                       ras, ones)
            cov = torch.autograd.grad(s.sum(), ones)[0]
        ctx.save_for_backward(verts, dirs)
        ctx.extra = (tri, half_width, depth, ras)
        return cov.detach()

    @staticmethod
    def backward(ctx, grad_cov):
        verts, dirs = ctx.saved_tensors
        tri, half_width, depth, ras = ctx.extra
        with torch.enable_grad():
            v = verts.detach().requires_grad_(True)
            d = dirs.detach().requires_grad_(True)
            s = LitCoverage._image_sum(v, tri, d, half_width, depth, ras, grad_cov.detach())
            if not s.requires_grad:              # a backend without position gradients
                return None, None, None, None, None, None
            gv, gd = torch.autograd.grad(s.sum(), [v, d], allow_unused=True)
        return gv, None, gd, None, None, None


def normalise(raw: torch.Tensor) -> torch.Tensor:
    """Each curve divided by its own mean over the phases (last axis)."""
    return raw / raw.mean(-1, keepdim=True).clamp_min(1e-12)


def normalise_vjp(raw: torch.Tensor, cot: torch.Tensor) -> torch.Tensor:
    """The cotangent on the unnormalised curves that corresponds to `cot` on the normalised
    ones: with y = x / m and m the mean of x over P phases,
    d/dx_p sum_q cot_q y_q = cot_p / m - (sum_q cot_q x_q) / (P m^2)."""
    m = raw.mean(-1, keepdim=True).clamp_min(1e-12)
    return cot / m - (cot * raw).sum(-1, keepdim=True) / (raw.shape[-1] * m ** 2)


class ExactForward:
    """The exact forward model for one phase grid and one set of cameras.

    `psi` is the rotation angle of every frame (P,), `instrument` holds the scene and sensor
    parameters, `config` the discretisation. `psi0` shifts every frame by a per-body start
    phase and may be a tensor, in which case the curves are differentiable in it. `geoms`
    selects a subset of the cameras by index; None means all of them.
    """

    def __init__(self, instrument: Instrument, psi, config: RenderConfig = RenderConfig(),
                 device: str = "cuda", backend: str | None = None):
        self.inst = instrument.to(device)
        self.cfg = config
        self.device = device
        self.psi = torch.as_tensor(np.asarray(psi), dtype=torch.float32, device=device)
        self.cams = cameras()
        self.cam_v = torch.tensor(np.stack([np.asarray(c.v) for c in self.cams]),
                                  dtype=torch.float32, device=device)
        self.ras_cam = Rasteriser(config.height, config.width, config.supersample, device,
                                  backend)
        self.ras_sun = Rasteriser(config.sun_res, config.sun_res, 1, device, backend)

    # ------------------------------------------------------------------ per-body setup
    def _prepare(self, verts: torch.Tensor, faces: torch.Tensor) -> Prepared:
        """The form factors of the patches and their factorisation (constants of the mesh),
        the face-to-patch map, and the per-face vertices and areas, which stay
        differentiable. Raises RadiosityError when the patches cannot be built or their form
        factors are unusable."""
        v_np = verts.detach().cpu().double().numpy()
        f_np = faces.detach().cpu().numpy()
        try:
            pv, pf = decimate(v_np, f_np, self.cfg.radiosity_faces)
        except Exception as exc:                        # noqa: BLE001  decimation failed
            raise RadiosityError(f"could not build the radiosity patches: {exc}") from exc
        if len(pf) < 4:
            raise RadiosityError("the mesh decimates to fewer than four patches")
        F, _, _, pc = form_factors(pv, pf, n_samples=self.cfg.form_factor_samples,
                                   device=self.device)
        solver = RadiositySolver(F.to(torch.float32), self.inst.rho, bad_rows=self.cfg.bad_rows)
        from scipy.spatial import cKDTree
        patch = torch.as_tensor(cKDTree(pc).query(v_np[f_np].mean(1))[1], device=self.device)
        fv, ff = flat_faces(verts, faces)
        tv = verts[faces.long()]
        area = 0.5 * torch.linalg.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]).norm(dim=1)
        extent = float(verts.detach().norm(dim=1).max()) * 1.05
        return Prepared(solver, patch, fv, ff, area.clamp_min(1e-12), extent)

    def _radiance(self, prep: Prepared, cov: torch.Tensor, k: int) -> torch.Tensor:
        """Per-face radiance (P, F) from the lit coverage (P K, F) of K source samples."""
        P = cov.shape[0] // k
        e = cov.reshape(P, k, -1).mean(1) / prep.area[None]                        # (P, F)
        n_patch = len(prep.solver.F)
        idx = prep.patch[None].expand(P, -1)
        w = prep.area[None].expand(P, -1)
        E = torch.zeros(P, n_patch, device=cov.device).scatter_add(1, idx, e * w)
        A = torch.zeros(P, n_patch, device=cov.device).scatter_add(1, idx, w)
        B = prep.solver.solve((E / A.clamp_min(1e-12)).T).T                       # (P, patches)
        return prep.solver.radiance(B).gather(1, idx)                               # (P, F)

    def _images(self, prep: Prepared, L: torch.Tensor, angle: torch.Tensor, geoms):
        """Sensor images (P, G, h, w) of the radiance L (P, F) at the body angles (P,)."""
        inst, cfg = self.inst, self.cfg
        P, G = len(angle), len(geoms)
        eyes = rotate_z(self.cam_v[geoms], angle) * inst.eye_distance                # (P, G, 3)
        fov = 2.0 * torch.atan(cfg.fov_scale * prep.extent / inst.eye_distance)
        # clip planes just around the body, whatever the camera distance: a far camera is
        # how a nearly orthographic view is expressed, and it must not clip the body
        near = (inst.eye_distance - 2.0 * prep.extent).clamp_min(0.05 * inst.eye_distance)
        far = inst.eye_distance + 2.0 * prep.extent
        proj = perspective(fov, cfg.width / cfg.height, near, far, device=self.device)
        mvp = proj @ look_at(eyes.reshape(-1, 3), device=self.device)                # (P G, 4, 4)
        hom = torch.cat([prep.fv, torch.ones(len(prep.fv), 1, device=self.device)], 1)
        clip = torch.einsum("vj,bij->bvi", hom, mvp)                                 # (P G, 3F, 4)
        attr = L[:, None, :].expand(P, G, -1).reshape(P * G, -1)
        attr = attr.repeat_interleave(3, dim=1)[..., None]                           # (P G, 3F, 1)
        img, _ = self.ras_cam.render(clip, prep.ff, attr)
        cos_off, radius = self.ras_cam.pixel_geometry(float(fov.detach()))
        n = img.shape[0]
        val = inst.sensor(img[..., 0], cos_off.expand(n, -1, -1), radius.expand(n, -1, -1),
                          supersample=cfg.supersample)
        return val.reshape(P, G, *val.shape[-2:])

    def _images_chunk(self, prep: Prepared, phases: torch.Tensor, psi0, geoms) -> torch.Tensor:
        """Sensor images (P_chunk, G, h, w) of a block of phases, before the pedestals."""
        inst, cfg = self.inst, self.cfg
        angle = -(phases + psi0)                       # lab -> body frame, see conventions.to_body
        sd = rotate_z(source_dirs(inst.delta, cfg.n_source), angle).reshape(-1, 3)   # (P K, 3)
        cov = LitCoverage.apply(prep.fv, prep.ff, sd, prep.extent, prep.extent, self.ras_sun)
        return self._images(prep, self._radiance(prep, cov, cfg.n_source), angle, geoms)

    def _count(self, val: torch.Tensor, geoms, tau_b: torch.Tensor) -> torch.Tensor:
        """The count curve (G, P_chunk) of the images `val`: pixels above tau_b (G,) after the
        binary pedestal."""
        gi = torch.as_tensor(geoms, device=self.device)
        ped_b = self.inst.pedestal[len(self.cams) + gi].reshape(1, len(geoms), 1, 1)
        return threshold_count(val + ped_b, tau_b.reshape(1, -1)).T

    def _reduce(self, val: torch.Tensor, geoms, tau_b: torch.Tensor) -> torch.Tensor:
        """Raw (I, N) curves (G, 2, P_chunk) of the images `val`: the summed value above tau_i
        after the intensity pedestal, and the count above tau_b."""
        gi = torch.as_tensor(geoms, device=self.device)
        ped_i = self.inst.pedestal[gi].reshape(1, len(geoms), 1, 1)
        I = threshold_sum(val + ped_i, self.inst.tau_i).T                            # (G, P)
        return torch.stack([I, self._count(val, geoms, tau_b)], 1)                   # (G, 2, P)

    def _chunk(self, prep: Prepared, phases: torch.Tensor, psi0, geoms, tau_b):
        """Raw (I, N) curves (G, 2, P_chunk) for a block of phases."""
        return self._reduce(self._images_chunk(prep, phases, psi0, geoms), geoms, tau_b)

    def _binary_thresholds(self, prep: Prepared, psi0, geoms, frame: int = 0) -> torch.Tensor:
        """Otsu's threshold of each camera's image at `frame` (the first frame, as the
        organisers take it), as a constant (G,)."""
        with torch.no_grad():
            val = self._images_chunk(prep, self.psi[frame:frame + 1], psi0, geoms)
            gi = torch.as_tensor(geoms, device=self.device)
            val = val[0] + self.inst.pedestal[len(self.cams) + gi].reshape(-1, 1, 1)
            return otsu_threshold(val)

    # ------------------------------------------------------------------ public interface
    def _geoms(self, geoms):
        return list(range(len(self.cams))) if geoms is None else [int(g) for g in geoms]

    def _phase_chunks(self):
        step = self.cfg.phase_chunk
        return [(i, self.psi[i:i + step]) for i in range(0, len(self.psi), step)]

    def raw_curves(self, verts: torch.Tensor, faces: torch.Tensor, geoms=None,
                   psi0=0.0) -> torch.Tensor:
        """Unnormalised curves (G, 2, P): intensity then count, for the requested geometries.
        No gradient. Raises RadiosityError when the mesh cannot be used."""
        geoms = self._geoms(geoms)
        with torch.no_grad():
            prep = self._prepare(verts.detach(), faces)
            tau_b = self._binary_thresholds(prep, psi0, geoms)
            out = [self._chunk(prep, phases, psi0, geoms, tau_b) for _, phases in
                   self._phase_chunks()]
        return torch.cat(out, -1)

    def raw_curves_turned(self, verts: torch.Tensor, faces: torch.Tensor, geoms=None,
                          psi0=0.0):
        """raw_curves, and beside it the count curves (3, G, P) the body would have if its
        first frame were frame P/4, P/2 or 3P/4: the counts above Otsu's threshold of that
        frame. A body turned by a quarter turn about its spin axis has the intensity curves of
        the body shifted by P/4 phases, exactly; its count curves are shifted the same way
        but take their threshold from what is then the first frame, which is what these
        supply (train_lpd.quarter_turns). One render pass serves all four. P must be
        divisible by four. No gradient."""
        geoms = self._geoms(geoms)
        P = len(self.psi)
        if P % 4:
            raise ValueError(f"turned counts need a phase count divisible by four, not {P}")
        with torch.no_grad():
            prep = self._prepare(verts.detach(), faces)
            taus = [self._binary_thresholds(prep, psi0, geoms, frame=j * P // 4)
                    for j in range(4)]
            out, extra = [], []
            for _, phases in self._phase_chunks():
                val = self._images_chunk(prep, phases, psi0, geoms)
                out.append(self._reduce(val, geoms, taus[0]))
                extra.append(torch.stack([self._count(val, geoms, tau) for tau in taus[1:]]))
        return torch.cat(out, -1), torch.cat(extra, -1)

    def vjp(self, verts: torch.Tensor, faces: torch.Tensor, cot, geoms=None, psi0=0.0,
            params: list | None = None):
        """The unnormalised curves and the vector-Jacobian product with `cot` (G, 2, P), with
        respect to `verts` (when it requires grad) and to the tensors in `params`. `cot` may
        also be a function of the full unnormalised curves, for cotangents that depend on
        every phase at once, such as the adjoint of the mean normalisation; the curves are
        then computed once without gradient first. Returns (curves, grad_verts,
        [grad_param, ...]); a gradient is None for an input that does not require grad. Runs
        in phase chunks, so the memory is that of one chunk. Raises RadiosityError when the
        mesh cannot be used."""
        geoms = self._geoms(geoms)
        params = list(params or [])
        wrt = ([verts] if verts.requires_grad else []) + params
        with torch.enable_grad():          # the caller may be inside no_grad
            prep = self._prepare(verts, faces)
            tau_b = self._binary_thresholds(prep, psi0, geoms)
            if callable(cot):
                with torch.no_grad():
                    raw = torch.cat([self._chunk(prep, phases, psi0, geoms, tau_b)
                                     for _, phases in self._phase_chunks()], -1)
                cot = cot(raw)
            grads = [torch.zeros_like(t) for t in wrt]
            out = []
            for i, phases in self._phase_chunks():
                raw = self._chunk(prep, phases, psi0, geoms, tau_b)
                if wrt:
                    # retain_graph: the part of the graph before the chunk (the mesh, the
                    # form-factor solve) is shared by every chunk
                    g = torch.autograd.grad(raw, wrt, grad_outputs=cot[..., i:i + raw.shape[-1]],
                                            allow_unused=True, retain_graph=True)
                    for acc, gi in zip(grads, g):
                        if gi is not None:
                            acc += gi
                out.append(raw.detach())
        curves = torch.cat(out, -1)
        grad_v = grads[0] if verts.requires_grad else None
        grad_p = grads[1:] if verts.requires_grad else grads
        return curves, grad_v, grad_p
