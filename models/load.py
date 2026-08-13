"""Load the two artefacts that are meant to be reused outside this repository.

    lpd_convex.pt                trained convex LPD; produced every reconstruction in
                                 results/convex. Carries its own `preset`, so the
                                 architecture is rebuilt from the file with no other input.
                                 Summed Dice 1.9459 over the three public models
                                 (0.8682 / 0.4647 / 0.6130, 128^3 voxel grid, both meshes
                                 posed by rescale_touch_z).

    instrument_calibration.pt    parameters fitted to the real lab curves of models 1-3:
                                 albedo, source angular radius, PSF width, OETF knots,
                                 per-curve intensity and binary thresholds, per-curve
                                 pedestal, per-body initial phase, per-curve model error.
                                 Independent of any reconstruction method.

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
