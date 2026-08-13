"""Volumetric forward model: a mollified SDF, rendered by transmittance line integrals.

THE FIELD. phi(x) is a signed distance function, negative inside, |grad phi| = 1, so the
unit outward normal is n(x) = grad phi(x). Every formula below assumes a true SDF; use
`normalised_field` or an eikonal penalty if the field is only approximately one.

THE MOLLIFIER. Psi_w(d) = (1/w) f(d/w) with f(t) = 1 / (4 cosh^2(t/2)), the logistic
density, which satisfies

    INT f = 1,      INT t f = 0,      INT t^2 f = pi^2 / 3,      f(t) ~ e^{-|t|}

Phi_s(x) = 1 / (1 + e^{-x/w}) is its CDF, so Phi_s' = Psi_w.

DENSITY. sigma = kappa Psi_w(phi) is NOT used. Along a ray of direction omega crossing the
surface once, dphi/ds = omega . n, so

    INT sigma ds = kappa INT Psi_w(phi) dphi / |omega . n| = kappa / |omega . n|

independent of w: w sets only the sharpness of the shell, while the opacity is set by kappa.
At finite kappa the body is then semi-transparent in a view-angle-dependent way -- opaque at
grazing incidence, e^{-kappa} at normal incidence -- a systematic error resembling limb
darkening that does not vanish as w -> 0. A hard surface would need w -> 0 and kappa -> oo
jointly.

The occlusion-exact construction is used instead. With

    rho(t) = max( -d/dt ln Phi_s(phi(r(t))), 0 )

the transmittance on a segment where phi decreases along the ray is

    T(t) = exp(-INT_0^t rho du) = Phi_s(phi(r(t))) / Phi_s(phi(r(0)))  ->  Phi_s(phi(r(t)))

starting far outside where Phi_s ~ 1. That is exactly 1 outside and 0 inside at every
incidence angle, with no kappa and no angular bias. The rendering weight

    W(t) = T(t) rho(t) = -d/dt Phi_s(phi(r(t))) = Psi_w(phi) |dphi/dt|

peaks exactly at phi = 0, so the surface location is unbiased, and integrates to 1 across a
crossing because INT Psi_w(phi) |dphi/dt| dt = INT Psi_w(phi) |dphi| = 1.

THE TWO LINE INTEGRALS.

    L_pixel     = INT_0^oo W_v(t) c(r(t)) dt,   W_v = -d/dt Phi_s(phi(r(t))), clamped >= 0
    T_s(x, w_k) = Phi_s( min_{u > 0} phi(x + u w_k) )

The shadow form is the closest-approach transmittance, exact for this density on a ray with a
single approach and recession: deep inside gives phi_min << 0 and T_s -> 0, a miss gives
phi_min >> 0 and T_s -> 1, grazing gives T_s ~ 1/2. min phi is found by sphere tracing, which
is exact for an SDF.

BIAS, TO SECOND ORDER IN w. Because |grad phi| = 1 the coarea formula gives
INT g dV = INT dd INT_{phi=d} g dA. On the offset surface {phi = d} the area element relative
to the base surface is the Steiner factor J(d) = (1 + k1 d)(1 + k2 d) = 1 + 2Hd + Kd^2.
Expanding g(x + d n) and multiplying,

    INT_{phi=d} g dA = INT_{A0} [ g + d(dn_g + 2Hg) + d^2(dnn_g/2 + 2H dn_g + Kg) ] dA0 + O(d^3)

Psi_w is even, so INT d Psi_w dd = 0 and INT d^2 Psi_w dd = m2 w^2 with m2 = pi^2/3, giving

    INT Psi_w(phi) g dV = INT dA [ g(1 + K m2 w^2) + 2H m2 w^2 dn_g + (m2 w^2 / 2) dnn_g ]
                          + O(w^4) + O(e^{-reach/w})

The Steiner expansion holds only for |d| < reach, the distance to the medial axis, and the
logistic has exponential rather than compact tails, so the mass outside that band is about
2 e^{-reach/w} and is not covered. Two consequences:

  * at an edge or crease the reach is zero, the expansion fails, and the local error is O(w)
    rather than O(w^2). Every ground truth here is a 3-D print, so the whole edge set is in
    this regime;
  * across a thin neck of half-thickness t the two tails overlap and add spurious density
    about e^{-2t/w}, filling the neck -- a bias toward convexity produced by the mollifier
    itself, on exactly the feature the model exists to recover.

Both biases point toward convexity, so this model must not be the final one. `w_bounds`
returns the two hard limits and `anneal_schedule` the path down to them.

GRADIENTS. The soft model is evaluated and the same soft model is differentiated, so at each
w the gradient is exact for the objective actually being minimised and the w-schedule is an
ordinary continuation method along a well-defined path of objectives. Phi_s, the ray
integrals and the shading are all smooth in phi, so ordinary autodiff applies with no
discontinuity handling. Evaluating a hard model and differentiating a soft surrogate is a
straight-through estimator with no convergence guarantee, and is not done here.

Quadrature nodes along the ray are treated as fixed: the integral is over t with the nodes
supplied by the marcher, and the dependence on the shape enters through phi evaluated at
those nodes. This is why smoothness of the integrand in phi is sufficient.

The one quantity that remains non-smooth is the observable, because both thresholds are hard.
Those are handled by the coarea formula on the rendered image -- contour segments at level
tau weighted by 1/|grad u| -- and are not softened: the geometric softness and the threshold
softness are different bandwidths, and only the first is physically justified.
"""
from __future__ import annotations

import numpy as np
import torch

from hac26.coarea import threshold_count, threshold_sum

__all__ = ["M2_LOGISTIC", "SHELL", "N_NODES", "logistic_cdf", "logistic_pdf",
           "normalised_field", "eikonal_penalty", "field_normal", "closest_approach",
           "shell_entry",
           "shadow_transmittance", "march_view", "direct_radiance", "gathered_bounce",
           "render_frame", "reduce_frame", "shadow_deficit", "w_bounds",
           "anneal_schedule", "curves"]

M2_LOGISTIC = float(np.pi ** 2 / 3.0)      # variance of the unit-scale logistic density
SHELL = 10.0        # gate half-width in units of w; 1 - 2/(1 + e^SHELL) = 1 - 9e-5 of the
                    # weight is inside it, against 1.35% missing at a half-width of 5w
N_NODES = 64        # in-shell quadrature nodes per ray, fixed so the count carries no
                    # geometry dependence and the gradient does not jump by a node


# ----------------------------------------------------------------------------- mollifier

def logistic_cdf(x: torch.Tensor, w: float) -> torch.Tensor:
    """Phi_s(x) = sigmoid(x / w)."""
    return torch.sigmoid(x / w)


def logistic_pdf(x: torch.Tensor, w: float) -> torch.Tensor:
    """Psi_w(x) = Phi_s'(x) = f(x/w)/w with f(t) = 1/(4 cosh^2(t/2))."""
    s = torch.sigmoid(x / w)
    return s * (1.0 - s) / w


# ----------------------------------------------------------------------------- the field

def field_normal(field, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """n = grad phi / |grad phi|, by autodiff of the field with respect to position."""
    xr = x.detach().requires_grad_(True)
    phi = field(xr)
    g, = torch.autograd.grad(phi.sum(), xr, create_graph=create_graph)
    return g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def eikonal_penalty(field, points: torch.Tensor) -> torch.Tensor:
    """mean (|grad phi| - 1)^2 at sampled points.

    A CONSTRAINT ON THE FIELD, not a prior on the shape. Every formula in this module assumes
    |grad phi| = 1: sphere tracing may not overstep, the closest-approach shadow form reads
    phi as a distance, and the Steiner expansion of the mollified integral is stated in terms
    of the offset distance. A field that is only approximately a distance function violates
    all three.

    It is therefore not interchangeable with a curvature or total-variation penalty, which
    say something about what shapes are like. Those are supplied by the shape prior Gamma in
    solvers.map_gauss_newton, estimated from a library, and nowhere else.

    The alternative to this penalty is `normalised_field`, which divides by |grad phi| at
    evaluation time instead of driving it to 1 during optimisation.
    """
    x = points.detach().requires_grad_(True)
    v = field(x)
    g, = torch.autograd.grad(v.sum(), x, create_graph=True)
    return ((g.norm(dim=-1) - 1.0) ** 2).mean()


def normalised_field(field):
    """phi / |grad phi|, so the formulas that assume a true SDF hold near the zero set.

    The alternative is an eikonal penalty during optimisation; this is the evaluation-time
    version and costs one extra gradient per call.
    """
    def phi(x):
        xr = x.detach().requires_grad_(True)
        v = field(xr)
        g, = torch.autograd.grad(v.sum(), xr, create_graph=True)
        return field(x) / g.norm(dim=-1).clamp_min(1e-6)
    return phi


# ------------------------------------------------------------------------------ tracing

def closest_approach(o: torch.Tensor, d: torch.Tensor, field, iters: int = 96,
                     tmax: float = 8.0, t0: float = 1e-3):
    """min_{u > 0} phi(o + u d), and the point attaining it.

    Marching by |phi| is the SDF's own guarantee: no step can pass through the surface, and
    once inside the same bound holds for the distance back out.
    """
    t = torch.full(o.shape[:-1], t0, device=o.device, dtype=o.dtype)
    best = field(o + t[..., None] * d)
    best_t = t.clone()
    for _ in range(iters):
        x = o + t[..., None] * d
        v = field(x)
        closer = v < best
        best = torch.where(closer, v, best)
        best_t = torch.where(closer, t, best_t)
        t = torch.minimum(t + v.abs().clamp_min(1e-4),
                          torch.full_like(t, tmax))
    return best, o + best_t[..., None] * d


def shadow_transmittance(x: torch.Tensor, dirs: torch.Tensor, field, w: float,
                         normal: torch.Tensor | None = None, shell: float = SHELL,
                         cos_floor: float = 0.05) -> torch.Tensor:
    """T_s(x, omega_k) = Phi_s(min_u phi(x + u omega_k)), one scalar per (point, direction).

    x is (P, 3) and dirs is (K, 3); the result is (P, K).

    THE MINIMUM STARTS BEYOND THE POINT'S OWN SHELL. Taken literally from u = 0, the minimum
    over a shading point that lies on the surface is phi(x) = 0, so T_s = Phi_s(0) = 1/2 for
    every lit point whether or not anything occludes it -- a uniform halving of the direct
    term that no amount of geometry can produce. The shadow ray must measure occlusion by
    OTHER geometry, so the trace starts where the point's own mollified shell ends,

        u0 = shell * w / max(n . omega, cos_floor)

    at which phi has risen to about shell*w on a locally flat surface. An unoccluded point
    then returns Phi_s(shell) = 1 - 2/(1 + e^shell), which is 9e-5 short of 1 at shell = 10:
    the same telescoping residual as the view gate, and the reason the zero-phase deficit is
    that size rather than identically zero at finite w.
    """
    P, K = x.shape[0], dirs.shape[0]
    n = field_normal(field, x, create_graph=False) if normal is None else normal
    mu = (n[:, None, :] * dirs[None, :, :]).sum(-1).abs().clamp_min(cos_floor)   # (P, K)
    u0 = (shell * w / mu)[..., None]                                            # (P, K, 1)
    o = (x[:, None, :] + u0 * dirs[None, :, :]).reshape(-1, 3)
    d = dirs[None, :, :].expand(P, K, 3).reshape(-1, 3)
    phi_min, _ = closest_approach(o, d, field, t0=0.0)
    return logistic_cdf(phi_min.reshape(P, K), w)


# ------------------------------------------------------------------- view line integral

def shell_entry(o: torch.Tensor, d: torch.Tensor, field, w: float, shell: float = SHELL,
                iters: int = 128, tmax: float = 8.0) -> torch.Tensor:
    """First t at which the ray reaches the shell, phi <= shell * w, by sphere tracing.

    Stepping by phi - shell*w approaches the shell boundary without entering it, which is the
    SDF's own guarantee applied to the offset surface {phi = shell*w}.
    """
    t = torch.zeros(o.shape[:-1], device=o.device, dtype=o.dtype)
    for _ in range(iters):
        v = field(o + t[..., None] * d) - shell * w
        t = torch.where((v > 0) & (t < tmax), t + v.clamp_min(1e-4), t)
    return t


def march_view(o: torch.Tensor, d: torch.Tensor, field, w: float, shade_fn,
               shell: float = SHELL, n_nodes: int = N_NODES, tmax: float = 8.0,
               cos_floor: float = 0.05):
    """L = INT W(t) c(r(t)) dt along each ray, with W = -d/dt Phi_s(phi).

    GATE WIDTH. The weight across a crossing telescopes exactly,
    sum_i W_i = Phi_s(phi_start) - Phi_s(phi_end), so gating at |phi| < D w delivers
    Phi_s(Dw) - Phi_s(-Dw) = 1 - 2/(1 + e^D) rather than 1. At D = 5 that is a 1.35%
    deficit, above the 0.4-0.9% median noise floor; at D = 10 it is 9e-5. The deficit is a
    uniform multiplicative factor and so largely cancels under per-curve mean normalisation,
    but not entirely, because tau_I is a fixed level -- and a grazing ray that never crosses
    picks up opacity 1 - Phi_s(phi_min), so a narrow gate puts a discontinuity at the
    silhouette, which is where the binary channel lives.

    COMPOSITING IN LOG SPACE. The transmittance ratio Phi_s(phi_cur)/Phi_s(phi_prev)
    underflows deep inside the body, so the increment is accumulated as

        log T <- log T + logsigmoid(phi_cur / w) - logsigmoid(phi_prev / w)

    clamped at zero increment, which is what makes a receding segment contribute nothing and
    so splits the ray at sign changes of dphi/dt without explicit segmentation. The per-step
    weight is then T (1 - e^dlog), evaluated with expm1.

    FIXED NODE COUNT. Exactly n_nodes quadrature nodes are placed uniformly in t across the
    shell crossing, so the node count does not depend on the geometry and the gradient does
    not jump when a marcher would have taken one step more or fewer. The window length scales
    as 2 shell w / |d . n| at the entry point, since a ray at incidence |d . n| takes that
    much longer in t to cross the same range of phi.
    """
    t_in = shell_entry(o, d, field, w, shell, tmax=tmax).detach()
    n_entry = field_normal(field, o + t_in[..., None] * d, create_graph=False)
    cosang = (n_entry * d).sum(-1).abs().clamp_min(cos_floor)
    span = (2.0 * shell * w) / cosang
    dt = (span / n_nodes).detach()

    log_T = torch.zeros(o.shape[:-1], device=o.device, dtype=o.dtype)
    L = torch.zeros_like(log_T)
    phi_prev = field(o + t_in[..., None] * d)
    for i in range(n_nodes):
        t = (t_in + (i + 1) * dt).detach()
        x = (o + t[..., None] * d).detach()
        phi_cur = field(x)
        dlog = (torch.nn.functional.logsigmoid(phi_cur / w)
                - torch.nn.functional.logsigmoid(phi_prev / w)).clamp(max=0.0)
        T = torch.exp(log_T)
        wgt = T * (-torch.expm1(dlog))
        m = wgt > 1e-9
        if bool(m.any()):
            c = torch.zeros_like(L)
            c[m] = shade_fn(x[m])
            L = L + wgt * c
        log_T = log_T + dlog
        phi_prev = phi_cur
    return L, torch.exp(log_T)


# ------------------------------------------------------------------------------ shading

def direct_radiance(x: torch.Tensor, field, w: float, sun_dirs: torch.Tensor,
                    rho_alb: float = 0.85, e0: float = 1.0,
                    shadows: bool = True) -> torch.Tensor:
    """c(x) = (rho/pi)(E0/K) sum_k (n . omega_k)+ T_s(x, omega_k).

    The K source directions span the source's finite angular disc, so the penumbra is
    physical and stays separate from w: the mollifier is not used to stand in for it.
    """
    n = field_normal(field, x)
    mu0 = (n[:, None, :] * sun_dirs[None, :, :]).sum(-1).clamp_min(0.0)
    ts = (shadow_transmittance(x, sun_dirs, field, w) if shadows
          else torch.ones_like(mu0))
    return (rho_alb / np.pi) * (e0 / sun_dirs.shape[0]) * (mu0 * ts).sum(-1)


def _cosine_directions(n: torch.Tensor, m: int, generator=None) -> torch.Tensor:
    """m cosine-sampled directions about each normal. Returns (P, m, 3)."""
    P = n.shape[0]
    u1 = torch.rand(P, m, device=n.device, dtype=n.dtype, generator=generator)
    u2 = torch.rand(P, m, device=n.device, dtype=n.dtype, generator=generator)
    r = u1.sqrt()
    theta = 2.0 * np.pi * u2
    local = torch.stack([r * torch.cos(theta), r * torch.sin(theta),
                         (1.0 - u1).clamp_min(0.0).sqrt()], dim=-1)
    a = torch.where(n[..., 2:3].abs() < 0.9,
                    torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype).expand_as(n),
                    torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype).expand_as(n))
    t1 = torch.cross(a, n, dim=-1); t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    t2 = torch.cross(n, t1, dim=-1)
    return (local[..., 0:1] * t1[:, None, :] + local[..., 1:2] * t2[:, None, :]
            + local[..., 2:3] * n[:, None, :])


def gathered_bounce(x: torch.Tensor, field, w: float, sun_dirs: torch.Tensor,
                    m_dirs: int = 32, rho_alb: float = 0.85, e0: float = 1.0,
                    shadows: bool = True):
    """One gathered bounce, reusing the closest-approach trace.

        c1(x) = (rho/pi) sum_m (1 - T_s(x, omega_m)) c0(y_m) (n . omega_m)+ (2 pi / M)

    y_m is the closest-approach point along omega_m. At albedo 0.85 this term is first order.
    """
    n = field_normal(field, x)
    dirs = _cosine_directions(n, m_dirs)                      # (P, M, 3)
    P, M = dirs.shape[0], dirs.shape[1]
    o = (x[:, None, :] + 1e-3 * dirs).reshape(-1, 3)
    dd = dirs.reshape(-1, 3)
    phi_min, y = closest_approach(o, dd, field)
    ts = logistic_cdf(phi_min, w).reshape(P, M)
    c0 = direct_radiance(y, field, w, sun_dirs, rho_alb, e0, shadows).reshape(P, M)
    mu = (n[:, None, :] * dirs).sum(-1).clamp_min(0.0)
    return (rho_alb / np.pi) * ((1.0 - ts) * c0 * mu).sum(-1) * (2.0 * np.pi / M)


# -------------------------------------------------------------------------------- frame

def render_frame(field, view: torch.Tensor, sun_lab_dirs: torch.Tensor, w: float,
                 sensor=None, res: int = 128, extent: float = 1.6, rho_alb: float = 0.85,
                 e0: float = 1.0, bounce: bool = True, m_dirs: int = 32,
                 supersample: int = 4, eye_distance: float = 8.0,
                 shadows: bool = True):
    """One frame's radiance image, through the sensor chain if one is given.

    Returns the image at the reduced resolution, ready for the two thresholded reductions.
    """
    from .common import camera_basis
    ex, ey = camera_basis(view)
    n_px = res * supersample
    a = torch.linspace(-extent, extent, n_px, device=view.device, dtype=view.dtype)
    gx, gy = torch.meshgrid(a, a, indexing="ij")
    o = (gx.reshape(-1, 1) * ex + gy.reshape(-1, 1) * ey) + 2.0 * extent * view
    d = (-view)[None, :].expand(o.shape[0], 3)

    def shade_fn(pts):
        c = direct_radiance(pts, field, w, sun_lab_dirs, rho_alb, e0, shadows)
        if bounce:
            c = c + gathered_bounce(pts, field, w, sun_lab_dirs, m_dirs,
                                    rho_alb, e0, shadows)
        return c

    L, _ = march_view(o, d, field, w, shade_fn)
    img = L.reshape(1, n_px, n_px)
    if sensor is None:
        return img[0]
    # off-axis cosine and normalised radius of each pixel ray, for cos^4 and vignetting
    rr = (gx ** 2 + gy ** 2).sqrt()
    cos_off = (eye_distance / (eye_distance ** 2 + rr ** 2).sqrt())[None]
    radius = (rr / rr.max().clamp_min(1e-9))[None]
    return sensor(img, cos_off, radius, supersample=supersample)[0]


def reduce_frame(img: torch.Tensor, tau_i: float, tau_b: float):
    """The two observables, with exact derivatives through both thresholds by coarea.

    The thresholds are NOT softened to match the geometric softness: they are different
    bandwidths and only the geometric one is physically justified.
    """
    return threshold_sum(img, tau_i), threshold_count(img, tau_b)


def shadow_deficit(field, view: torch.Tensor, sun_dirs: torch.Tensor, w: float,
                   **kw) -> torch.Tensor:
    """D_shadow = sum over pixels of [ L(shadows=False) - L(shadows=True) ].

    Three properties must hold, and each failure is a distinct fault:

      * D >= 0 everywhere. Shadow removes light and never adds it.
      * D = 0 exactly at azimuth 0, elevation 0. There v = s, so a point is shadowed only
        when it is blocked along the view direction, i.e. when it is not visible at all.
      * D varies with psi for a non-convex body. A psi-flat term is annihilated by the
        per-curve mean normalisation and carries no information.

    Identically zero means the shadow ray is not wired in. Non-zero but flat in psi means the
    source is not rotating in the body frame; it must be recomputed per phase as
    s_body(psi) = R_z(-psi - psi0) s_lab.
    """
    lit = render_frame(field, view, sun_dirs, w, shadows=False, **kw)
    shd = render_frame(field, view, sun_dirs, w, shadows=True, **kw)
    return (lit - shd).sum()


# --------------------------------------------------------------------------- w schedule

def w_bounds(t_min: float, radius: float, eps: float = 1e-3, psf_px: float = 1.5,
             body_px: float = 800.0) -> dict:
    """The two hard bounds on w, and their minimum.

    Neck: a neck of half-thickness t_min fills by about e^{-2 t_min / w}, so holding that
    below eps requires w <= 2 t_min / ln(1/eps).

    Silhouette: a grazing ray that does not cross accumulates opacity 1 - Phi_s(phi_min), so
    the outline is blurred over a scale w in phi. Keeping that below the optical PSF
    footprint expressed in object units stops the mollification from biasing the thresholded
    pixel count. At 1.5 px across a body spanning 800 px this is about 0.004 R.
    """
    neck = 2.0 * t_min / np.log(1.0 / eps)
    silhouette = 2.0 * (psf_px / body_px) * radius
    return {"neck": neck, "silhouette": silhouette, "w_max": min(neck, silhouette)}


def anneal_schedule(radius: float, w_max: float, levels: int = 6, hold: int = 400):
    """Geometric path from w = 0.05 R down to w_max, held `hold` steps at each level.

    Held rather than swept so the continuation path is followed rather than jumped.
    """
    w0 = 0.05 * radius
    ws = np.geomspace(w0, max(w_max, 1e-6), levels)
    return [(float(w), int(hold)) for w in ws]


# -------------------------------------------------------------------------------- curves

def curves(field, cam_dir, sun_lab, psi, w: float, sensor=None, psi0: float = 0.0,
           delta_rad: float = 0.0, n_source: int = 8, tau_i: float = 0.0,
           tau_b: float | None = None, **kw):
    """Both curves over a rotation, in the body frame, normalised by their mean over psi.

    The field is fixed and the camera and source directions rotate by R_z(-psi - psi0). The
    binary threshold defaults to the full-frame Otsu level of frame 0, as in the rasterised
    pipeline; the intensity threshold is the fixed low one.
    """
    from hac26.conventions import source_directions, to_body
    from .mesh_raster import otsu_threshold

    src = source_directions(delta_rad, n_source)          # (K, 3) in the lab frame
    cam = to_body(np.asarray(cam_dir, dtype=float), np.asarray(psi, dtype=float), psi0)
    frames = []
    for j in range(len(psi)):
        sun_b = np.stack([to_body(s, np.array([psi[j]]), psi0)[0] for s in src])
        img = render_frame(field, torch.tensor(cam[j], dtype=torch.float32),
                           torch.tensor(sun_b, dtype=torch.float32), w, sensor=sensor, **kw)
        frames.append(img)
    if tau_b is None:
        tau_b = otsu_threshold(frames[0])
    I, N = [], []
    for img in frames:
        i_, n_ = reduce_frame(img, tau_i, tau_b)
        I.append(i_); N.append(n_)
    I = torch.stack(I); N = torch.stack(N)
    return I / I.mean().clamp_min(1e-12), N / N.mean().clamp_min(1e-12)
