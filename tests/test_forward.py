"""Shadow-deficit test.

    D = (radiosity prediction with all visibility set to 1) - (raycast prediction)

must be >= 0 everywhere, and exactly zero at azimuth 0 elevation 0. Both are asserted. D is
never clamped: a clamp would turn a broken transport model into a plausible-looking one,
which is the failure mode this test exists to catch.

WHY D >= 0 IS A THEOREM AND NOT A HOPE. Setting every visibility to 1 can only raise the
emission, so e_novis >= e_vis componentwise. The transport operator inherits that ordering:
(I - rho F)^-1 = I + rho F + rho^2 F^2 + ... has non-negative entries, since F >= 0 and the
spectral radius is below 1. So B_novis >= B_vis, and therefore every pixel, every summed
intensity and every pixel count is >= as well.

WHY IT IS EXACTLY ZERO AT ZERO PHASE, AND UNDER WHAT CONDITION. At azimuth 0 elevation 0
the camera looks along the light, so a surface point is shadowed exactly when it is blocked
along the view direction, i.e. exactly when it is invisible. Removing shadows therefore
changes only pixels that were never seen. That argument is about DIRECT light. With
interreflection on, a shadowed facet still receives and re-radiates, so it perturbs the
visible facets through F, and the equality becomes approximate rather than exact. The test
asserts exactness at rho = 0 and the inequality with rho > 0, which is what the physics
actually supports.
"""
import numpy as np
import pytest
import torch

from hac26.conventions import S_LAB, cameras, source_directions
from forward_models.mesh_radiosity import RadiositySolver, emission, facet_geometry, form_factors

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
H, W, SS = 128, 128, 2          # small on purpose; the gate is about signs, not resolution


def contact_binary(n_phi: int = 14, n_th: int = 9, sep: float = 0.75, r: float = 0.55):
    """Two overlapping spheres: genuinely non-convex, so it casts real shadows."""
    import trimesh
    a = trimesh.creation.icosphere(subdivisions=1, radius=r)
    b = trimesh.creation.icosphere(subdivisions=1, radius=r)
    a.apply_translation([-sep / 2, 0, 0])
    b.apply_translation([+sep / 2, 0, 0])
    m = trimesh.util.concatenate([a, b])
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def flat_shade(verts, faces, facet_value):
    """Duplicate vertices per face so each facet carries its own constant radiance.

    Radiosity is a per-FACET quantity. Interpolating it over shared vertices would smear it
    across edges, which on a faceted body is exactly the error the binary channel is most
    sensitive to.
    """
    v = verts[faces].reshape(-1, 3)
    f = np.arange(len(v), dtype=np.int64).reshape(-1, 3)
    val = np.repeat(np.asarray(facet_value, dtype=np.float64), 3)
    return v, f, val


def _curves(rho, occlude_light, cam, dev="cuda"):
    """Reduced (I, N) for one camera, with light visibility on or off."""
    from forward_models.mesh_raster import Rasteriser, reduce_curves
    verts, faces = contact_binary()
    F, area, nrm, cen = form_factors(verts, faces, occlusion=(rho > 0))
    solver = RadiositySolver(F, rho=max(rho, 1e-12))
    s_dirs = source_directions(0.0)                       # point source: one direction
    if occlude_light:
        import trimesh
        m = trimesh.Trimesh(verts, faces, process=False)
        vis = np.ones((len(cen), len(s_dirs)))
        for k, d in enumerate(s_dirs):
            o = cen + nrm * 1e-4
            hit = m.ray.intersects_any(o, np.tile(d, (len(o), 1)))
            vis[:, k] = (~hit).astype(float)
    else:
        vis = None
    e = emission(nrm, s_dirs, vis)
    B = solver.solve(e) if rho > 0 else e / np.pi
    L = solver.radiance(B) if rho > 0 else B
    v2, f2, val = flat_shade(verts, faces, L)
    ras = Rasteriser(H, W, SS, device=dev)
    img, _ = ras.render(torch.tensor(v2, dtype=torch.float32, device=dev),
                        torch.tensor(f2, dtype=torch.int32, device=dev),
                        torch.tensor(val, dtype=torch.float32, device=dev),
                        eye=np.asarray(cam.v) * 8.0, fov_y_rad=np.radians(20.0))
    return reduce_curves(img, tau_i=0.0, tau_b=1e-6)


@cuda
def test_shadow_deficit_is_non_negative():
    """D >= 0 at an oblique geometry, where shadows genuinely exist."""
    cam = [c for c in cameras() if c.azimuth_deg == 90.0 and c.kind == "hor_a"][0]
    i_novis, n_novis = _curves(rho=0.0, occlude_light=False, cam=cam)
    i_vis, n_vis = _curves(rho=0.0, occlude_light=True, cam=cam)
    d_i = float(i_novis - i_vis)
    d_n = float(n_novis - n_vis)
    assert d_i >= -1e-4, f"intensity deficit is NEGATIVE ({d_i:.6e}); not clamped, by design"
    assert d_n >= -1e-4, f"count deficit is NEGATIVE ({d_n:.6e}); not clamped, by design"
    assert d_i > 0, "a non-convex body at 90 deg phase must cast some shadow"


@cuda
def test_shadow_deficit_vanishes_at_zero_phase():
    """At azimuth 0 elevation 0 with direct light only, shadowing changes nothing seen."""
    cam = [c for c in cameras() if c.azimuth_deg == 0.0 and c.kind == "hor_a"][0]
    assert cam.phase_angle_deg == pytest.approx(0.0, abs=1e-9)
    i_novis, _ = _curves(rho=0.0, occlude_light=False, cam=cam)
    i_vis, _ = _curves(rho=0.0, occlude_light=True, cam=cam)
    rel = abs(float(i_novis - i_vis)) / max(float(i_novis), 1e-12)
    assert rel < 1e-6, f"zero-phase deficit should vanish, got relative {rel:.3e}"


@cuda
def test_transport_preserves_the_ordering():
    """The inequality is inherited from (I - rho F)^-1 having non-negative entries."""
    verts, faces = contact_binary()
    F, area, nrm, cen = form_factors(verts, faces, occlusion=True)
    solver = RadiositySolver(F, rho=0.85)
    s = source_directions(0.0)
    e_full = emission(nrm, s, None)
    e_shad = emission(nrm, s, np.zeros((len(nrm), len(s))))
    assert (e_full >= e_shad - 1e-12).all()
    assert (solver.solve(e_full) >= solver.solve(e_shad) - 1e-9).all()
