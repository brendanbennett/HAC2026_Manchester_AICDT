"""Fitting a body to its curves by damped Gauss-Newton, without differentiating the renderer.

The unknown is the pair (c, g): nine coefficients that reshape the convex core and one
amplitude per lattice site that carves it. Both move in the same step, because the correction
from a convex inversion's answer to the body is a conjunction of the two and neither half
lowers the misfit on its own. A method that takes one coordinate direction at a time, or one
block at a time, walks away from the answer.

No derivative of the renderer is used. The chain's derivative with respect to the vertices is
missing its boundary term on the pure-torch rasteriser and has never been compared against a
finite difference on the other, and the misfit is in any case a rough function of the code at
the step sizes a line search takes. A secant Jacobian costs one render per active coordinate
and removes both risks, and the response of a curve to a carve is superlinear in its depth, so
a secant over the depth an answer is likely to have is the right linearisation where a tangent
at zero depth is far too short.

The carve is fitted coarse to fine. A stage's coordinates are combinations of the site
amplitudes: first the cells of a coarse sub-lattice, then a finer one, then random smooth
fields, so that the number of renders per iteration stays bounded while the last stage still
reaches every amplitude. Every coordinate is scaled so that one unit of it is one body unit of
carve depth at the sites, which is what makes a single finite-difference step size right for
all of them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, sparse

__all__ = ["Stage", "DEFAULT_STAGES", "CarveFit", "block_basis", "subspace_basis",
           "waist_amplitudes", "conjunction_start", "SHRINK_RANGE", "STEP_G", "STEP_C",
           "AREA_WEIGHT", "AREA_WINDOW", "VOLUME_TRUST", "DEPTH_TRUST", "TARGET_SIGMA"]

# The secant steps, in body units of the canonical pose. Each is the size the answer's own
# correction is likely to have, not a small number: the response of a curve to either half of
# the correction begins rather than scales, so a probe much shorter than the answer measures
# the wrong regime. STEP_G is a carve depth and STEP_C a displacement of the hull.
STEP_G = 0.20
STEP_C = 0.08

# Weight of the surface area in the objective, in inverse area of the canonical pose. Its
# admissible window is bounded on both sides and was measured on a span of bodies around the
# one released non-convex model: below the bottom of it a corrugation one grid cell wide still
# lowers the objective, and above the top the body stops being the minimiser, beaten by itself
# displaced slightly inward. The value sits near the top because what it risks there is
# bounded and small while what it must refuse below is not.
AREA_WEIGHT = 0.9
AREA_WINDOW = (0.60, 0.99)

# Largest change of volume an accepted step may make, as a fraction. The cheapest area in this
# representation is a hull shrink, and a penalty on area alone walks down it until the body is
# gone; refusing a step that moves the volume by more than this removes that and costs nothing,
# since the extraction has already given the volume. A correction larger than this is reached
# in several steps, which is why the line search carries lengths short enough to land inside
# it: a trust region that refuses every trial is a fit that cannot move at all.
VOLUME_TRUST = 0.08

# Deepest carve one accepted step may add or remove, in body units. Nothing else bounds the
# amplitudes: what the objective charges is the body's surface and its misfit, and both are
# properties of the level set, so amplitudes may grow without limit in the directions that do
# not move it. A step is a carve, and a carve deeper than the body is not one.
DEPTH_TRUST = 0.5

# Smallest curvature the damping will believe, as a fraction of the mean over the stage's
# coordinates. See _normal_equations: it is what bounds a step into a direction the curves do
# not see, and without it the area term sends the amplitudes there without limit.
DAMP_FLOOR = 1e-2

# The misfit, in model errors, at which the fit stops. It is not one: one asks the body to
# explain the curves to the noise, and the representation cannot, its own floor at the
# released non-convex body being about this. A target below the floor makes the stopping rule
# dead and lets the fit spend its budget buying misfit with surface.
TARGET_SIGMA = 2.6

SHRINK_RANGE = (0.02, 0.14)   # inward displacements of the hull a start draws from, in body
                              # units of the canonical pose, where a body spans [-1, 1]. The
                              # range covers a body that is barely non-convex through a
                              # contact binary; which of them a body is, is what the restarts
                              # decide.


@dataclass(frozen=True)
class Stage:
    """One refinement of the carve. `side` is the sub-lattice whose cells are the
    coordinates, or 0 for a random subspace of `n_dirs` smooth fields."""
    side: int
    n_dirs: int
    iters: int

    @property
    def name(self) -> str:
        return f"blocks {self.side}^3" if self.side else f"subspace {self.n_dirs}"


DEFAULT_STAGES = (Stage(side=6, n_dirs=0, iters=6),
                  Stage(side=12, n_dirs=0, iters=6),
                  Stage(side=0, n_dirs=192, iters=10))


def block_basis(shape, side: int) -> sparse.csr_matrix:
    """(n_sites, side^3): one column per cell of a sub-lattice, driving the sites in that
    cell together. `side` must divide every axis of the lattice."""
    shape = tuple(int(s) for s in shape)
    if any(s % side for s in shape):
        raise ValueError(f"sub-lattice side {side} does not divide the lattice {shape}")
    idx = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
    idx = idx.reshape(-1, 3)
    cell = idx // np.array([s // side for s in shape])
    col = cell[:, 0] * side * side + cell[:, 1] * side + cell[:, 2]
    row = np.arange(len(idx))
    return sparse.csr_matrix((np.ones(len(idx)), (row, col)),
                             shape=(len(idx), side ** 3))


def subspace_basis(shape, n_dirs: int, rng, correlation: float = 2.0) -> np.ndarray:
    """(n_sites, n_dirs): smooth random fields on the lattice, orthonormalised.

    Smooth rather than white because neighbouring kernels overlap, so a white direction on
    the amplitudes is almost entirely in the null space of the field and a render along it
    sees nothing. The correlation length is in site spacings.
    """
    shape = tuple(int(s) for s in shape)
    q = np.stack([ndimage.gaussian_filter(rng.standard_normal(shape), correlation,
                                          mode="nearest").reshape(-1)
                  for _ in range(n_dirs)], 1)
    return np.linalg.qr(q)[0]


def waist_amplitudes(sites: np.ndarray, kernel, rng, equatorial_bias: float = 0.9):
    """Amplitudes of one Gaussian slab across the body, with its recipe.

    A waist across the spin axis is the feature a convex inversion cannot see, and a contact
    binary is one, so a fit that has to cross from the convex answer to a carved body is
    started from such a feature as well as from the convex answer itself.

    The amplitudes are the requested field at the sites, rescaled so that the field they
    actually make is as deep as was asked for. Dividing by the kernels' overlap alone is not
    enough: a slab a fraction of a site spacing wide is covered by far fewer kernels along
    its normal than a slowly varying field is, and comes out shallower than requested by a
    factor of two or more. One sparse product measures the depth instead of estimating it.
    """
    if rng.random() < equatorial_bias:
        a = rng.uniform(0, 2 * np.pi)
        n = np.array([np.cos(a), np.sin(a), rng.normal(0, 0.25)])
    else:
        n = rng.normal(size=3)
    n = n / np.linalg.norm(n)
    d = rng.uniform(-0.45, 0.45)
    w = rng.uniform(0.10, 0.50)
    depth = rng.uniform(0.10, 0.60)
    target = np.exp(-(((sites @ n) - d) / w) ** 2)
    made = float(np.abs(kernel @ target).max())
    g = target * (depth / made) if made > 1e-9 else target * 0.0
    return g, {"axis": n.tolist(), "offset": float(d), "width": float(w),
               "depth": float(depth)}


def conjunction_start(sites: np.ndarray, kernel, rng, n_radial: int = 9) -> tuple:
    """A start with both halves of the correction present: (c, g, recipe).

    A convex inversion of a non-convex body does not return that body's hull. It returns a
    larger one, because enlarging the hull is how a convex body imitates the shadowing of a
    concavity, so the correction from that answer to the body shrinks the hull and carves it
    at the same time. A start that leaves the hull where the convex inversion put it asks the
    fit to find both from a linearisation taken where neither is active, and there a hull
    shrink alone raises the misfit; the fit then spends the step on the carve, which is the
    convex inversion's own mistake made once more one level down. Drawing the shrink with the
    waist puts the first linearisation somewhere both are already doing something.

    The shrink is uniform, the degree-zero coefficient alone, because that is the part of the
    excess that does not depend on which way a body is turned; the rest is left to the fit.
    """
    g, rec = waist_amplitudes(sites, kernel, rng)
    c = np.zeros(int(n_radial))
    c[0] = rng.uniform(*SHRINK_RANGE)
    return c, g, {**rec, "shrink": float(c[0])}


class CarveFit:
    """Damped Gauss-Newton on (c, g) against one body's curves.

    `render(c, g)` returns the kept curves as a flat array in the same order as `data`, or
    None for a body the forward model cannot render. `scale` is the per-entry model error the
    residual is divided by, so the reported misfit is in standard deviations of it.

    Both secant steps are taken over the size the answer's own correction is likely to have
    rather than at zero, because the response of the curves to either half of the correction
    begins rather than scales: a dent shadows itself only once it is deep enough to, and a
    hull shrink lowers the misfit only once there is a carve for it to uncover. A probe much
    shorter than the answer therefore returns a column of the Jacobian whose sign is that of
    the wrong regime.
    """

    def __init__(self, render, data: np.ndarray, scale: np.ndarray, kernel,
                 lattice_shape, n_radial: int = 9,
                 area_weight: float = AREA_WEIGHT, volume_trust: float = VOLUME_TRUST,
                 depth_trust: float = DEPTH_TRUST,
                 step_g: float = STEP_G, step_c: float = STEP_C,
                 damping=(1e-1, 1e-2, 1e-3, 1.0, 10.0),
                 lengths=(1.0, 0.5, 0.25, 0.1, 0.04),
                 subspace_tries: int = 3, seed: int = 0, site_mask=None):
        if area_weight and not AREA_WINDOW[0] <= area_weight <= AREA_WINDOW[1]:
            raise ValueError(f"area weight {area_weight} is outside the measured window "
                             f"{AREA_WINDOW}; below it a corrugation one cell wide is still "
                             f"profitable and above it the body stops being the minimum")
        self.render = render
        self.data = np.asarray(data, dtype=np.float64).ravel()
        self.iscale = 1.0 / np.asarray(scale, dtype=np.float64).ravel()
        self.n_obs = len(self.data)
        self.kernel = kernel
        self.shape = tuple(int(s) for s in lattice_shape)
        self.n_sites = int(np.prod(self.shape))
        self.site_mask = None if site_mask is None else np.asarray(site_mask, bool).ravel()
        if self.site_mask is not None and len(self.site_mask) != self.n_sites:
            raise ValueError(f"the site mask has {len(self.site_mask)} entries for "
                             f"{self.n_sites} sites")
        self.n_radial = int(n_radial)
        self.area_weight = float(area_weight)
        self.volume_trust = float(volume_trust)
        self.depth_trust = float(depth_trust)
        self.step_g, self.step_c = float(step_g), float(step_c)
        self.damping, self.lengths = tuple(damping), tuple(lengths)
        self.subspace_tries = int(subspace_tries)
        self.rng = np.random.default_rng(seed)
        self.renders = 0

    # ------------------------------------------------------------------ the objective
    def residual(self, y) -> np.ndarray | None:
        """The whitened residual, scaled so that its Euclidean norm is the misfit in standard
        deviations."""
        if y is None:
            return None
        r = (np.asarray(y, dtype=np.float64).ravel() - self.data) * self.iscale
        return r / np.sqrt(self.n_obs)

    def _render(self, c, g):
        """(residual, area, volume) of the body at (c, g), or (None, nan, nan).

        `render` returns the kept curves together with the surface area and the volume of the
        canonically posed body. Both come from the mesh the extraction has already built, so
        they cost a per cent of the render, and every Jacobian column therefore carries the
        secant derivative of the area for nothing."""
        self.renders += 1
        out = self.render(c, g)
        if out is None:
            return None, float("nan"), float("nan")
        y, area, vol = out
        return self.residual(y), float(area), float(vol)

    def objective(self, r, area: float) -> float:
        """log chi^2 plus the area of the body, which is the total variation of its indicator.

        Area rather than a ridge on the amplitudes because the two charge different things. A
        ridge charges by how large a coefficient is, so it prefers a shallow answer to a deep
        one and prefers the fit's own answer to the body. Area charges by how much surface a
        shape has, so a smooth dent of any depth passes nearly free while a corrugation of the
        same amplitude does not, and a corrugation is what a fit buys misfit with when the
        misfit is mostly reporting how finely the surface is resolved.

        The logarithm rather than a plain weight because the two terms have to stay in balance
        as the fit descends. The gradient of chi^2 falls with chi while the gradient of an area
        term does not, so no fixed weight both charges a corrugation at the start and leaves
        the body a minimum at the end: the window for one is empty by a factor of five, and the
        scale-free form opens it. Stationarity then compares a relative change of misfit with
        an absolute change of area.
        """
        rr = float(r @ r)
        return float(np.log(max(rr, 1e-300)) + self.area_weight * float(area))

    def _basis(self, stage: Stage):
        """The stage's coordinates as columns over the site amplitudes, scaled so that one
        unit of a coordinate is one body unit of carve depth at the sites.

        Where a site mask is given, the coordinates are confined to it and any column left
        with nothing to drive is dropped. A secant Jacobian spends one render per column, so a
        column over sites that cannot move the surface is a render thrown away; field
        .searchable_sites says which those are and why the band is one-sided.
        """
        if stage.side:
            b = block_basis(self.shape, stage.side)
        else:
            b = sparse.csc_matrix(subspace_basis(self.shape, stage.n_dirs, self.rng))
        if self.site_mask is not None:
            b = sparse.csc_matrix(sparse.diags(self.site_mask.astype(float)) @ b)
            b.eliminate_zeros()
            alive = np.diff(b.indptr) > 0
            if alive.any():
                b = b[:, np.flatnonzero(alive)]
        peak = np.zeros(b.shape[1])
        for i in range(0, b.shape[1], 256):                    # a block at a time, dense
            sl = slice(i, min(i + 256, b.shape[1]))
            peak[sl] = np.abs(self.kernel @ b[:, sl].toarray()).max(axis=0)
        b = sparse.csc_matrix(b.multiply(1.0 / np.maximum(peak, 1e-12)[None, :]))
        return b, (b.T @ b).toarray()

    # ------------------------------------------------------------------ one iteration
    def _jacobian(self, c, g, basis, r0, area0, vol0):
        """(J, a, v): the secant derivatives of the residual, the area and the volume, in the
        stage's coordinates. The last two cost no render of their own, since the body whose
        residual a column measures is the body whose area and volume it measures."""
        m = basis.shape[1]
        J = np.zeros((len(r0), self.n_radial + m))
        a = np.zeros(self.n_radial + m)
        v = np.zeros(self.n_radial + m)
        for j in range(self.n_radial):
            cc = c.copy()
            cc[j] += self.step_c
            r, ar, vr = self._render(cc, g)
            if r is not None:
                J[:, j] = (r - r0) / self.step_c
                a[j] = (ar - area0) / self.step_c
                v[j] = (vr - vol0) / self.step_c
        for j in range(m):
            col = np.asarray(basis[:, j].todense()).ravel()
            r, ar, vr = self._render(c, g + self.step_g * col)
            if r is not None:
                J[:, self.n_radial + j] = (r - r0) / self.step_g
                a[self.n_radial + j] = (ar - area0) / self.step_g
                v[self.n_radial + j] = (vr - vol0) / self.step_g
        return J, a, v

    def _trust_length(self, step, basis, vprime, vol) -> float:
        """The longest step length that keeps both the change of volume and the depth of the
        carve it adds inside their trust regions, capped at one.

        A region has to shorten the step rather than merely refuse it. The cheapest area in
        this representation is a hull shrink, so the linear term of a penalised step points
        down it and can be enormous: unconstrained, the first step of a coarse stage collapses
        the body altogether, and a line search over a fixed ladder of lengths then finds every
        one of them outside the region and takes no step at all.

        The two regions bound different things and both are needed. Volume alone does not
        bound the amplitudes, because a step that carves deeply in one place and fills deeply
        in another leaves the volume where it was; the objective does not bound them either,
        since it charges the surface and the misfit and both are properties of the level set.
        The depth of the carve a step makes is what says whether it is a step between bodies.
        """
        rate = abs(float(vprime @ step))
        lv = 1.0 if rate <= 1e-12 else self.volume_trust * max(abs(vol), 1e-9) / rate
        depth = float(np.abs(self.kernel @ (basis @ step[self.n_radial:])).max())
        ld = 1.0 if depth <= 1e-12 else self.depth_trust / depth
        return float(min(1.0, lv, ld))

    def _normal_equations(self, J, r0, a):
        """(A, b, diagonal) of the penalised Gauss-Newton system, before the damping.

        Differentiating log(r.r) + mu A gives the ordinary Gauss-Newton system with one extra
        linear term. The matrix is unchanged, so the penalty costs nothing in conditioning:

            (J^T J) d = -J^T r - (mu/2)(r^T r) a.
        """
        A = J.T @ J
        b = -(J.T @ r0) - 0.5 * self.area_weight * float(r0 @ r0) * a
        d = np.diag(A).copy()
        pos = d[d > 0]
        mean = float(pos.mean()) if pos.size else 1.0
        # A direction the curves cannot see has no curvature, and the damping, being relative
        # to the curvature, does not bound the step there. That is harmless while the right
        # side is J^T r, which lies in the range of J; it is not harmless once the right side
        # carries the area's gradient, which does not, and the step in such a direction then
        # grows without bound as the damping is loosened. Flooring the diagonal gives those
        # directions the damping of an average one, so a step into them is bounded by the same
        # trust as a step into a direction the curves do determine.
        d = np.maximum(d, DAMP_FLOOR * mean)
        return A, b, d

    @staticmethod
    def _solve(A, b, d, mu):
        try:
            return np.linalg.solve(A + mu * np.diag(d), b)
        except np.linalg.LinAlgError:
            return np.zeros(len(b))

    def run(self, c0, g0, stages=DEFAULT_STAGES, target: float = TARGET_SIGMA, log=None,
            area_weight: float | None = None):
        """Fit from (c0, g0). Returns (c, g, history); the history has one row per accepted
        or refused iteration.

        `area_weight` overrides the penalty for this run alone, and zero turns it off, which
        is what the polish uses: once the shape is where the penalty puts it, minimising the
        misfit alone with the trust region still on recovers the misfit without giving the
        shape back.
        """
        was, self.area_weight = self.area_weight, (self.area_weight if area_weight is None
                                                   else float(area_weight))
        try:
            return self._run(c0, g0, stages, target, log)
        finally:
            self.area_weight = was

    def _run(self, c0, g0, stages, target, log):
        c, g = np.asarray(c0, float).copy(), np.asarray(g0, float).copy()
        history = []
        r, area, vol = self._render(c, g)
        if r is None:
            return c, g, [{"stage": "start", "chi": float("inf"), "note": "no curves"}]
        chi = float(np.linalg.norm(r))
        for stage in stages:
            basis, _ = self._basis(stage)
            tries = 0
            it = 0
            while it < stage.iters:
                if chi <= target:
                    break
                J, a, vprime = self._jacobian(c, g, basis, r, area, vol)
                A, rhs, diag = self._normal_equations(J, r, a)
                obj = self.objective(r, area)
                # The best of the trials rather than the first that descends. A damping and a
                # step length that happen to be tried early are not the best step available,
                # and the trials cost fifteen renders against the Jacobian's own hundreds.
                best = None
                for mu in self.damping:
                    step = self._solve(A, rhs, diag, mu)
                    cap = self._trust_length(step, basis, vprime, vol)
                    for length in (cap * np.asarray(self.lengths)):
                        cc = c + length * step[:self.n_radial]
                        gg = g + length * (basis @ step[self.n_radial:])
                        rr, aa, vv = self._render(cc, gg)
                        if rr is None:
                            continue
                        if abs(vv - vol) > self.volume_trust * max(abs(vol), 1e-9):
                            continue          # outside the trust region on volume
                        o = self.objective(rr, aa)
                        if o < obj and (best is None or o < best[0]):
                            best = (o, cc, gg, rr, aa, vv)
                moved = best is not None
                if moved:
                    _, c, g, r, area, vol = best
                    chi = float(np.linalg.norm(r))
                row = {"stage": stage.name, "iteration": it, "chi": chi, "area": area,
                       "volume": vol, "objective": self.objective(r, area),
                       "area_weight": self.area_weight, "accepted": moved,
                       "renders": self.renders}
                history.append(row)
                if log is not None:
                    log(row)
                if not moved:
                    # a subspace that finds no step is redrawn; a fixed block basis that
                    # finds none has converged for this stage
                    tries += 1
                    if stage.side or tries >= self.subspace_tries:
                        break
                    basis, _ = self._basis(stage)
                    continue
                it += 1
            if chi <= target:
                break
        return c, g, history
