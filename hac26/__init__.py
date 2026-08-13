"""hac26 — Learned Primal-Dual lightcurve inversion for the Helsinki Asteroid
Challenge 2026. See docs/HAC2026_LPD_forward_model.md for the exact mathematics."""

from .geometry import (AZIMUTHS_DEG, OMEGA0, TOP_ALPHA_DEG, Camera, NormalGrid,
                       build_cameras, make_grid, project_closure, psi_grid)
from .forward import build_A, stack_A, forward_np, deriv_adjoint_np, normalize_np
from .shapes import (hull_mesh, mesh_curves_convex, mesh_to_egi, rescale_touch_z,
                     sample_training_shape)
from .minkowski import solve_minkowski
from .stl_io import load_stl, save_stl
from .recon import dice, save_submission_stl, voxel_grid, voxelize_convex

__version__ = "0.1.0"
