"""Load the two artefacts that are meant to be reused outside this repository.

    lpd_convex.pt                trained convex LPD; produced every reconstruction in
                                 results/convex. Carries its own `preset`, so the
                                 architecture is rebuilt from the file with no other input.

    instrument_calibration.pt    fitted to the real lab curves of models 1-3: vignetting,
                                 PSF width, OETF knots, clip knee, per-curve pedestal and
                                 per-curve model error. Independent of any reconstruction
                                 method.

                                 Also in the file but NOT fitted: the two per-curve
                                 thresholds (a hard comparison passes no gradient), and rho
                                 and delta_deg, which are whatever was passed on the command
                                 line. There is no psi0 in the file. See scripts/calibrate.py.

Both are torch pickles, so torch.load executes their contents; load only copies you trust.
"""
from __future__ import annotations

from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def load_lpd(device: str = "cpu"):
    """Return (net, preset, grid) for the trained convex LPD."""
    import sys
    sys.path.insert(0, str(HERE.parent))
    from hac26.train import load_net
    return load_net(str(HERE / "lpd_convex.pt"), device=device)


def load_calibration() -> dict:
    """Fitted instrument parameters. Keys: sensor, raw_tau_i, raw_tau_b, pedestal,
    raw_eta, rho, delta_deg."""
    return torch.load(HERE / "instrument_calibration.pt", map_location="cpu",
                      weights_only=False)
