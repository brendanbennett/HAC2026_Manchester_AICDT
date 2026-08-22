"""The fast operator the LPD needs, at thousands of evaluations.

The split that makes this work: the GEOMETRY is ray-traced exactly and never learned, and
only the tone-mapped response is. Per surface token and per phase:

    camera visibility            is the token seen from this camera
    light visibility             penumbra-weighted over the K source samples
    perspective solid angle      how much image area the token subtends
    one gathered bounce          FREE: it is the blocker hit of the shadow rays, which
                                 have already been cast for light visibility
    the token's own area         quadrature weight, and an input too: a facet's share of
                                 the binary curve is its projected area

The first four carry all the non-convexity there is -- occlusion, cast shadow and
interreflection are exactly the phenomena a convex model cannot express -- so learning them
would be learning something computable directly. What is learned is only the map from these
features to the reduced curve value, which is where the shared sensor chain lives. The
per-curve terms of the calibration are applied nowhere; see scripts/train_surrogate.py.

EQUIVARIANCE, exactly AND NOT APPROXIMATELY. Rotating the body by one frame permutes the
phase axis cyclically. The operator must commute with that, so the network is built only
from operations that do:

  * pointwise MLPs across the feature axis (act identically at every phase);
  * circular spectral convolution along psi (multiplication in the rFFT domain, which is
    diagonal in the shift and therefore exactly equivariant, not merely trained to be);
  * attention over TOKENS with weights shared across phase.

No positional encoding of absolute phase appears anywhere, because that is precisely what
would break the symmetry. This matters beyond elegance: the physics is block-diagonal in
the rotation order m, so an operator that mixes m is modelling a coupling that does not
exist and will fit noise with it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["TokenFeatures", "trace_features", "sun_features", "camera_features",
           "PhaseEquivariantNet", "Surrogate"]

N_FEAT = 5          # camera visibility, light visibility, solid angle, gathered bounce,
                    # and the token's own area. Area is the quadrature weight and also an
                    # input, since a facet's contribution to the binary curve is its
                    # projected area.


def trace_features(verts, faces, token_pts, token_nrm, cam_dirs, sun_dirs,
                   eye_distance: float = 8.0, eps: float = 1e-4, areas=None):
    """Exact geometric features per (token, phase). Nothing here is learned.

    cam_dirs and sun_dirs are (n_phase, 3) body-frame directions. The gathered bounce is
    read off the same shadow rays used for light visibility -- when a shadow ray is blocked,
    the facet it hits is the one lighting the token indirectly, so the bounce costs nothing
    beyond a lookup.
    """
    import trimesh
    mesh = trimesh.Trimesh(verts, faces, process=False)
    n_t, n_p = len(token_pts), len(cam_dirs)
    out = np.zeros((n_t, n_p, N_FEAT), dtype=np.float32)
    origins = token_pts + token_nrm * eps
    face_norm = mesh.face_normals
    for j in range(n_p):
        v, s = cam_dirs[j], sun_dirs[j]
        # Per-token direction to the eye. The rasteriser is perspective: at distance 8 with
        # a unit body the view direction varies by about 7 degrees across the surface, so a
        # single global camera axis would be an orthographic approximation.
        eye = v * eye_distance
        d_eye = eye[None, :] - token_pts
        d_eye = d_eye / np.linalg.norm(d_eye, axis=1, keepdims=True)
        mu = (token_nrm * d_eye).sum(1)
        mu0 = token_nrm @ s
        seen = (mu > 0) & (~mesh.ray.intersects_any(origins, d_eye))
        # light visibility and the gathered bounce come from the SAME cast
        loc, idx_ray, idx_tri = mesh.ray.intersects_location(
            origins, np.tile(s, (n_t, 1)), multiple_hits=False)
        lit = (mu0 > 0).astype(np.float32)
        bounce = np.zeros(n_t, dtype=np.float32)
        if len(idx_ray):
            lit[idx_ray] = 0.0
            # the blocker's own orientation to the light sets how much it can re-radiate
            bounce[idx_ray] = np.clip(face_norm[idx_tri] @ s, 0, None)
        out[:, j, 0] = np.clip(mu, 0, None) * seen
        out[:, j, 1] = lit * np.clip(mu0, 0, None)
        out[:, j, 2] = (np.clip(mu, 0, None) * seen
                        / np.maximum(((eye[None, :] - token_pts) ** 2).sum(1), 1e-9))
        out[:, j, 3] = bounce
        if areas is not None:
            out[:, j, 4] = areas
    return out


def sun_features(mesh, token_pts, token_nrm, sun_dirs, eps: float = 1e-4):
    """Light visibility and the gathered bounce -- the two features that do not depend on
    the camera, so they are computed once and shared by all 28 geometries.

    The source is fixed in the lab frame, so at a given phase every camera sees the same
    illumination. Re-tracing the shadow rays per camera repeats identical work once per
    camera. Checked against trace_features: the two agree to float summation order.
    """
    n_t, n_p = len(token_pts), len(sun_dirs)
    o = np.repeat(token_pts + token_nrm * eps, n_p, axis=0)
    d = np.tile(sun_dirs, (n_t, 1))
    mu0 = (token_nrm @ sun_dirs.T).reshape(-1)
    lit = (mu0 > 0).astype(np.float32)
    bounce = np.zeros(n_t * n_p, dtype=np.float32)
    _, idx_ray, idx_tri = mesh.ray.intersects_location(o, d, multiple_hits=False)
    if len(idx_ray):
        lit[idx_ray] = 0.0
        bounce[idx_ray] = np.clip((mesh.face_normals[idx_tri] * d[idx_ray]).sum(1), 0, None)
    return ((lit * np.clip(mu0, 0, None)).reshape(n_t, n_p).astype(np.float32),
            bounce.reshape(n_t, n_p))


def camera_features(mesh, token_pts, token_nrm, cam_dirs, sun, areas=None,
                    eye_distance: float = 8.0, eps: float = 1e-4):
    """The per-camera half, with every phase cast in ONE call rather than n_phase calls."""
    n_t, n_p = len(token_pts), len(cam_dirs)
    eye = cam_dirs * eye_distance
    d = eye[None, :, :] - token_pts[:, None, :]
    r2 = (d ** 2).sum(-1)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    mu = (token_nrm[:, None, :] * d).sum(-1)
    o = np.repeat(token_pts + token_nrm * eps, n_p, axis=0)
    hit = mesh.ray.intersects_any(o, d.reshape(-1, 3)).reshape(n_t, n_p)
    vis = np.clip(mu, 0, None) * ((mu > 0) & (~hit))
    out = np.zeros((n_t, n_p, N_FEAT), dtype=np.float32)
    out[..., 0] = vis
    out[..., 1] = sun[0]
    out[..., 2] = vis / np.maximum(r2, 1e-9)
    out[..., 3] = sun[1]
    if areas is not None:
        out[..., 4] = areas[:, None]
    return out


class TokenFeatures(nn.Module):
    """Pointwise lift of the N_FEAT per-token features, identical at every phase."""

    def __init__(self, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU())

    def forward(self, x):                       # (B, T, P, F) -> (B, T, P, width)
        return self.net(x)


class CircularSpectralConv(nn.Module):
    """Circular convolution along the phase axis, as multiplication in the rFFT domain.

    Exactly equivariant to cyclic phase shift by construction: a shift multiplies each
    Fourier coefficient by a phase, and a diagonal operator in that basis commutes with it.
    Truncating to `modes` is a low-pass, which commutes with shifts as well.
    """

    def __init__(self, width: int, modes: int = 20):
        super().__init__()
        self.modes = modes
        self.w = nn.Parameter(torch.randn(width, modes, 2) * 0.02)

    def forward(self, x):                       # (B, T, P, W)
        p = x.shape[-2]
        f = torch.fft.rfft(x, dim=-2)
        m = min(self.modes, f.shape[-2])
        wt = torch.view_as_complex(self.w[:, :m].contiguous())
        # Built out of place: writing the bands into a cloned tensor would be two in-place
        # operations on a graph leaf, which autograd rejects.
        low = f[..., :m, :] * wt.T.unsqueeze(0).unsqueeze(0)
        high = torch.zeros_like(f[..., m:, :])
        return torch.fft.irfft(torch.cat([low, high], dim=-2), n=p, dim=-2)


class PhaseSharedAttention(nn.Module):
    """Attention over TOKENS with weights shared across phase; never across phase."""

    def __init__(self, width: int, heads: int = 4):
        super().__init__()
        self.att = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):                       # (B, T, P, W)
        b, t, p, w = x.shape
        y = x.permute(0, 2, 1, 3).reshape(b * p, t, w)
        # need_weights=False, and it is not cosmetic. With it True (the torch default)
        # F.multi_head_attention_forward takes the bmm+softmax path and materialises the
        # (bsz*heads, L, S) attention matrix PLUS a (bsz, L, S) head-mean, both of which are
        # discarded here. That is quadratic in the token count, per layer, and it is most of
        # what a chunked run spends. False routes to scaled_dot_product_attention, i.e.
        # flash / memory-efficient attention with O(L) memory on CUDA. Same output either
        # way.
        y, _ = self.att(y, y, y, need_weights=False)
        y = y.reshape(b, p, t, w).permute(0, 2, 1, 3)
        return self.norm(x + y)


class PhaseEquivariantNet(nn.Module):
    """Exactly equivariant to cyclic shift of the phase axis. No absolute-phase encoding."""

    def __init__(self, width: int = 64, modes: int = 20, blocks: int = 3,
                 use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.lift = TokenFeatures(width)
        self.spec = nn.ModuleList([CircularSpectralConv(width, modes) for _ in range(blocks)])
        self.attn = nn.ModuleList([PhaseSharedAttention(width) for _ in range(blocks)])
        self.mix = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.SiLU())
                                  for _ in range(blocks)])
        self.head = nn.Linear(width, 2)          # intensity and binary contribution

    def forward(self, feats, areas=None):       # (B, T, P, F), (B, T) -> (B, 2, P)
        """Tokens are summed weighted by area, which makes this a quadrature.

An unweighted sum over sampled tokens is a Monte-Carlo estimate of the surface
        integral, with error 1/sqrt(T) independent of the fit. Weighting by facet area makes
        it the same quadrature the radiosity solve uses, so the error is set by the mesh.
        """
        x = self.lift(feats)
        for s, a, m in zip(self.spec, self.attn, self.mix):
            x = x + m(s(x))
            if self.use_attention:
                x = a(x)
        # Softplus because both reductions are sums of non-negative contributions: a token
        # cannot remove intensity from the frame, nor un-count a pixel. A linear head lets
        # contributions cancel, and the curve is then divided by its own mean, which is
        # unstable where that mean approaches zero.
        h = torch.nn.functional.softplus(self.head(x))       # (B, T, P, 2)
        if areas is not None:
            h = h * areas[:, :, None, None]
        return h.sum(dim=1).permute(0, 2, 1)


class Surrogate(nn.Module):
    """Exact geometry in, curves out; trained against the physical forward model."""

    def __init__(self, **kw):
        super().__init__()
        self.net = PhaseEquivariantNet(**kw)

    def forward(self, feats, areas=None):
        c = self.net(feats, areas)
        return c / c.mean(dim=-1, keepdim=True).clamp_min(1e-9)


# --------------------------------------------------------------- checkpoint with its config

SURROGATE_DEFAULTS = {"width": 96, "modes": 8, "blocks": 3}


def save_surrogate(net, path, meta: dict) -> None:
    """Weights AND the configuration they were built and trained under.

    A bare state_dict records neither the architecture nor the phase grid, so a rerun of
    train_surrogate.py with different flags produced a file the pipeline either refused to
    load (wrong width/blocks) or loaded and silently misused (wrong phase count). Both have
    happened. Consumers read this through load_surrogate and check it.
    """
    import torch
    torch.save({"net": net.state_dict(), "meta": dict(meta)}, path)


def load_surrogate(path, phases: int | None = None, device="cpu"):
    """(net, meta). Accepts the legacy bare state_dict, assuming SURROGATE_DEFAULTS for it.

    `phases` is the grid the CALLER will evaluate on. A surrogate fitted on a coarser grid is
    not the same operator: rFFT bin k is rotation order k at any P, so keeping modes 0..8 is
    the same band either way -- but sampling at P frames ALIASES every true order above P/2
    down into that band, so a net trained at P = 16 learned to reproduce aliased curves and
    is then asked for unaliased ones.
    """
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "net" in blob and "meta" in blob:
        meta, sd = dict(blob["meta"]), blob["net"]
    else:
        meta, sd = dict(SURROGATE_DEFAULTS), blob
        meta["legacy"] = True
        print(f"  WARNING: {path} carries no metadata; assuming "
              f"{SURROGATE_DEFAULTS} and an unknown phase grid", flush=True)
    net = Surrogate(width=int(meta.get("width", 96)), modes=int(meta.get("modes", 8)),
                    blocks=int(meta.get("blocks", 3)),
                    use_attention=bool(meta.get("attention", True)))
    net.load_state_dict(sd)
    got = meta.get("phases")
    if phases is not None and got is not None and int(got) != int(phases):
        print(f"  WARNING: {path} was trained at {got} phases and is being evaluated at "
              f"{phases}. Sampling at {got} aliases every rotation order above {int(got)//2} "
              f"into the band the network kept, so it was fitted to a different operator. "
              f"Retrain with --phases {phases}.", flush=True)
    return net.to(device).eval(), meta

