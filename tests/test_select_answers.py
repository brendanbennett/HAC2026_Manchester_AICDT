"""The rule that lets a refined body replace a convex answer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_answers import decide  # noqa: E402


def test_a_refinement_must_beat_the_convex_answer_on_held_out_geometries_by_the_ratio():
    meta = {"held_out": [3, 9], "chi_held_convex": 2.0, "chi_held_export": 1.2}
    assert decide(meta, 0.7)[0]
    assert decide(meta, 0.6)[0]
    assert not decide(meta, 0.5)[0]


def test_a_fit_without_held_out_geometries_or_without_a_written_misfit_is_refused():
    assert not decide({"held_out": [], "chi_held_convex": 2.0, "chi_held_export": 0.5}, 0.9)[0]
    assert not decide({"held_out": [3], "chi_held_convex": 2.0,
                       "chi_held_export": float("inf")}, 0.9)[0]
    assert not decide({"held_out": [3]}, 0.9)[0]
