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
