"""The rule that lets a refined body replace a convex answer."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_answers import calibrated_ratio, decide  # noqa: E402


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


def test_the_ratio_comes_from_a_public_refinement_that_improved_the_overlap():
    """The gate's number is measured, not chosen: it is the misfit ratio a refinement of a
    body with released truth reached, and it is only a number at all if that refinement moved
    the body toward its truth."""
    good = {"model": 3, "held_out": [1, 2], "convex_dice": 0.71, "final_dice": 0.98,
            "chi_held_convex": 2.0, "chi_held_export": 0.8}
    assert calibrated_ratio(good) == pytest.approx(0.4)

    # a refinement that fit the curves better while moving away from the truth is the failure
    # the gate exists to catch, so it licenses nothing
    for bad in ({**good, "final_dice": 0.70},                   # the overlap fell
                {**good, "final_dice": good["convex_dice"]},    # and did not move
                {**good, "held_out": []},                       # nothing was held out
                {**good, "convex_dice": float("nan"),           # the truth is not released
                 "final_dice": float("nan")},
                {**good, "chi_held_export": float("inf")}):     # nothing was written
        with pytest.raises(SystemExit):
            calibrated_ratio(bad)
