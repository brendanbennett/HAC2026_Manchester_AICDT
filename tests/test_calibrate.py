"""The calibration's convergence report: it has to notice when the fit stopped because the
step budget ran out rather than because the data was satisfied."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "calibrate", Path(__file__).resolve().parents[1] / "scripts" / "calibrate.py")
calibrate = importlib.util.module_from_spec(_spec)
sys.modules["calibrate"] = calibrate
_spec.loader.exec_module(calibrate)


def _report(start, now, budget=4.5):
    s = {k: torch.tensor(v) for k, v in start.items()}
    n = {k: torch.tensor(v) for k, v in now.items()}
    return calibrate.movement_report(s, n, {k: budget for k in start})


def test_movement_is_measured_in_raw_space_against_steps_times_lr():
    rep = _report({"raw_eye": 8.0}, {"raw_eye": 9.5}, budget=4.5)
    assert rep["raw_eye"]["moved"] == pytest.approx(1.5)
    assert rep["raw_eye"]["fraction"] == pytest.approx(1.5 / 4.5)


def test_the_shipped_calibration_is_flagged_as_budget_limited():
    """The values in models/instrument_calibration.json came from 150 steps at lr 0.03, a
    travel budget of 4.5. rho moved 3.7 of that, so it was still moving when the run ended;
    eye_distance could not have passed softplus(8 + 4.5) = 12.5 whatever the data said."""
    rep = _report({"raw_rho": -1.386, "raw_eye": 8.0, "raw_tau_i": -3.892},
                  {"raw_rho": 2.330, "raw_eye": 8.510, "raw_tau_i": -4.345})
    assert calibrate.print_movement(rep) == ["raw_rho"]
    assert rep["raw_rho"]["fraction"] > 0.8


def test_a_settled_fit_is_not_flagged():
    rep = _report({"raw_rho": -1.386, "raw_eye": 8.0}, {"raw_rho": -1.2, "raw_eye": 8.3})
    assert calibrate.print_movement(rep) == []


def test_a_vector_parameter_reports_its_largest_component():
    rep = _report({"raw_oetf": [0.0, 0.0, 0.0]}, {"raw_oetf": [0.1, -3.0, 0.4]})
    assert rep["raw_oetf"]["moved"] == pytest.approx(3.0)
