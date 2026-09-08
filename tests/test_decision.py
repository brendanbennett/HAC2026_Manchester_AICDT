"""The choice of the answer: candidates scored against the draws, and the consensus bodies."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hac26.recon import mesh_occupancy                     # noqa: E402
from hac26.solvers.output import metric_medoid            # noqa: E402
from reconstruct_lpd import consensus_bodies              # noqa: E402


def _ball(radius, n=32, extent=1.3):
    g = (np.arange(n) + 0.5) / n * 2 * extent - extent
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    return x ** 2 + y ** 2 + z ** 2 < radius ** 2


def test_candidates_are_scored_against_the_draws_only():
    """With three balls as draws and a fourth candidate equal to the middle one, the middle
    ball and its copy tie for the best mean Dice to the draws, and the copy is never scored
    against itself as a draw would be."""
    draws = [_ball(0.6), _ball(0.8), _ball(1.0)]
    k = metric_medoid(draws + [_ball(0.8)], n_ref=3)
    assert k in (1, 3)


def test_consensus_levels_keep_or_fill_a_dent():
    """Two draws with a dent and one without: at level 0.5 the dent survives (only a third
    of the draws fill it), at level 0.3 it is filled in; the rest of the ball is inside at
    both levels."""
    extent = 1.3
    n = 32
    g = (np.arange(n) + 0.5) / n * 2 * extent - extent
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    ball = x ** 2 + y ** 2 + z ** 2 < 1.0
    dent = (x - 1.0) ** 2 + y ** 2 + z ** 2 < 0.25
    draws = [ball & ~dent, ball & ~dent, ball]
    bodies = consensus_bodies(draws, extent, 1.0, levels=(0.5, 0.3))
    assert [lv for lv, _, _ in bodies] == [0.5, 0.3]
    occ = [mesh_occupancy(v, f, n, extent) for _, v, f in bodies]
    dent_cells = dent & ball
    core = (x ** 2 + y ** 2 + z ** 2 < 0.5) & ~dent
    assert occ[0][dent_cells].mean() < 0.2 and occ[0][core].mean() > 0.95
    assert occ[1][dent_cells].mean() > 0.8 and occ[1][core].mean() > 0.95


def test_the_derived_level_is_the_fixed_point_of_half_the_score():
    """The level a consensus body should sit at is not a free choice: adding a voxel the draws
    occupy with probability p changes the expected voxel score by (2p - D), so the level that
    is right for the body it produces is the fixed point of t -> D(t)/2. Checked by taking the
    level the derivation returns and confirming that the body at that level does score about
    twice it against the draws."""
    from reconstruct_lpd import consensus_bodies, dice_optimal_level
    from hac26.recon import dice as dice_of
    n, extent = 48, 1.3
    g = (np.arange(n) + 0.5) / n * 2 * extent - extent
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    ball = x ** 2 + y ** 2 + z ** 2 < 1.0
    rng = np.random.default_rng(0)
    centres = rng.normal(0.0, 0.4, (8, 2)) + np.array([0.85, 0.0])
    draws = [ball & ~(((x - c[0]) ** 2 + (y - c[1]) ** 2 + z ** 2) < 0.28) for c in centres]
    lv = dice_optimal_level(draws, extent, 1.0)
    assert 0.2 <= lv <= 0.7
    made = consensus_bodies(draws, extent, 1.0, levels=(lv,))
    assert made, "the derived level has to produce a closed surface"
    occ = mesh_occupancy(made[0][1], made[0][2], n, extent)
    d = float(np.mean([dice_of(occ, o) for o in draws]))
    assert abs(lv - 0.5 * d) < 0.02, (lv, d)


def test_the_consensus_grid_is_fine_enough_not_to_swamp_the_choice():
    """A draw is a mesh voxelised once; a consensus body is built out of the voxels and then
    meshed and voxelised again. That round trip costs it overlap, and if it costs more than
    the differences between candidates the rule is choosing on the grid rather than on the
    bodies. Checked against the same ensemble at the resolution actually used."""
    from reconstruct_lpd import OCC_RES, consensus_bodies, dice_optimal_level
    from hac26.recon import dice as dice_of
    n, extent = OCC_RES, 1.3
    g = (np.arange(n) + 0.5) / n * 2 * extent - extent
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    rng = np.random.default_rng(0)
    ball = x ** 2 + y ** 2 + z ** 2 < 1.0
    draws = [ball & ~(((x - c[0]) ** 2 + (y - c[1]) ** 2 + z ** 2) < 0.28)
             for c in rng.normal(0.0, 0.25, (8, 2)) + np.array([0.85, 0.0])]
    prob = np.mean([d.astype(np.float32) for d in draws], axis=0)
    lv = dice_optimal_level(draws, extent, 1.0)
    made = consensus_bodies(draws, extent, 1.0, levels=(lv,))
    assert made
    _, v, f = made[0]
    round_trip = dice_of(mesh_occupancy(v, f, n, extent), prob >= lv)
    assert round_trip > 0.95, round_trip
