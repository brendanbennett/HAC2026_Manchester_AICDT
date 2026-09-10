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
           "waist_amplitudes"]


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


class CarveFit:
    """Damped Gauss-Newton on (c, g) against one body's curves.

    `render(c, g)` returns the kept curves as a flat array in the same order as `data`, or
    None for a body the forward model cannot render. `scale` is the per-entry model error the
    residual is divided by, so the reported misfit is in standard deviations of it.
    """

    def __init__(self, render, data: np.ndarray, scale: np.ndarray, kernel,
                 lattice_shape, n_radial: int = 9, ridge_frac: float = 1e-2,
                 step_g: float = 0.20, step_c: float = 0.03,
                 damping=(1e-1, 1e-2, 1e-3, 1.0, 10.0), lengths=(1.0, 0.5, 0.25),
                 subspace_tries: int = 3, seed: int = 0):
        self.render = render
        self.data = np.asarray(data, dtype=np.float64).ravel()
        self.iscale = 1.0 / np.asarray(scale, dtype=np.float64).ravel()
        self.n_obs = len(self.data)
        self.kernel = kernel
        self.shape = tuple(int(s) for s in lattice_shape)
        self.n_sites = int(np.prod(self.shape))
        self.n_radial = int(n_radial)
        self.ridge_frac = float(ridge_frac)
        self.step_g, self.step_c = float(step_g), float(step_c)
        self.damping, self.lengths = tuple(damping), tuple(lengths)
        self.subspace_tries = int(subspace_tries)
        self.rng = np.random.default_rng(seed)
        self.lam_g = None
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
        self.renders += 1
        return self.render(c, g)

    def objective(self, r, g, c) -> float:
        """chi^2 in standard deviations, plus the ridge on the amplitudes. The ridge is in
        the objective and therefore in both sides of the step, not only in the damping."""
        lam = 0.0 if self.lam_g is None else self.lam_g
        return float(r @ r + lam * (g @ g))

    def _basis(self, stage: Stage):
        """The stage's coordinates as columns over the site amplitudes, scaled so that one
        unit of a coordinate is one body unit of carve depth at the sites."""
        if stage.side:
            b = block_basis(self.shape, stage.side)
        else:
            b = sparse.csc_matrix(subspace_basis(self.shape, stage.n_dirs, self.rng))
        peak = np.zeros(b.shape[1])
        for i in range(0, b.shape[1], 256):                    # a block at a time, dense
            sl = slice(i, min(i + 256, b.shape[1]))
            peak[sl] = np.abs(self.kernel @ b[:, sl].toarray()).max(axis=0)
        b = sparse.csc_matrix(b.multiply(1.0 / np.maximum(peak, 1e-12)[None, :]))
        return b, (b.T @ b).toarray()

    # ------------------------------------------------------------------ one iteration
    def _jacobian(self, c, g, basis, r0):
        """Columns (d residual / d coordinate) by a secant over the step of that block."""
        m = basis.shape[1]
        J = np.zeros((len(r0), self.n_radial + m))
        for j in range(self.n_radial):
            cc = c.copy()
            cc[j] += self.step_c
            r = self.residual(self._render(cc, g))
            if r is not None:
                J[:, j] = (r - r0) / self.step_c
        for j in range(m):
            col = np.asarray(basis[:, j].todense()).ravel()
            r = self.residual(self._render(c, g + self.step_g * col))
            if r is not None:
                J[:, self.n_radial + j] = (r - r0) / self.step_g
        return J

    def _normal_equations(self, J, r0, g, basis, btb):
        """(A, b, diagonal) of the penalised Gauss-Newton system, before the damping."""
        A = J.T @ J
        b = -(J.T @ r0)
        if self.lam_g:
            # the ridge is on g, and g moves by the basis applied to the carve coordinates
            A[self.n_radial:, self.n_radial:] += self.lam_g * btb
            b[self.n_radial:] -= self.lam_g * (basis.T @ g)
        d = np.diag(A).copy()
        d[d <= 0] = np.mean(d[d > 0]) if np.any(d > 0) else 1.0
        return A, b, d

    @staticmethod
    def _solve(A, b, d, mu):
        try:
            return np.linalg.solve(A + mu * np.diag(d), b)
        except np.linalg.LinAlgError:
            return np.zeros(len(b))

    def run(self, c0, g0, stages=DEFAULT_STAGES, target: float = 1.0, log=None):
        """Fit from (c0, g0). Returns (c, g, history); the history has one row per accepted
        or refused iteration."""
        c, g = np.asarray(c0, float).copy(), np.asarray(g0, float).copy()
        history = []
        r = self.residual(self._render(c, g))
        if r is None:
            return c, g, [{"stage": "start", "chi": float("inf"), "note": "no curves"}]
        chi = float(np.linalg.norm(r))
        for si, stage in enumerate(stages):
            basis, btb = self._basis(stage)
            tries = 0
            it = 0
            while it < stage.iters:
                if chi <= target:
                    break
                J = self._jacobian(c, g, basis, r)
                if self.lam_g is None:
                    carve = np.diag(J.T @ J)[self.n_radial:]
                    scale = float(np.mean(np.maximum(carve, 0.0)))
                    unit = float(np.mean(np.diag(btb)))
                    self.lam_g = self.ridge_frac * scale / max(unit, 1e-12)
                A, rhs, diag = self._normal_equations(J, r, g, basis, btb)
                obj = self.objective(r, g, c)
                moved = False
                for mu in self.damping:
                    step = self._solve(A, rhs, diag, mu)
                    for length in self.lengths:
                        cc = c + length * step[:self.n_radial]
                        gg = g + length * (basis @ step[self.n_radial:])
                        rr = self.residual(self._render(cc, gg))
                        if rr is None:
                            continue
                        if self.objective(rr, gg, cc) < obj:
                            c, g, r = cc, gg, rr
                            chi = float(np.linalg.norm(r))
                            moved = True
                            break
                    if moved:
                        break
                row = {"stage": stage.name, "iteration": it, "chi": chi,
                       "accepted": moved, "renders": self.renders}
                history.append(row)
                if log is not None:
                    log(row)
                if not moved:
                    # a subspace that finds no step is redrawn; a fixed block basis that
                    # finds none has converged for this stage
                    tries += 1
                    if stage.side or tries >= self.subspace_tries:
                        break
                    basis, btb = self._basis(stage)
                    continue
                it += 1
            if chi <= target:
                break
        return c, g, history
