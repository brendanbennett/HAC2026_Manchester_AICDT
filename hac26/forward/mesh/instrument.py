"""Everything about the measurement that is not the shape: the scene and sensor parameters
the calibration fits on the public models and the exact forward model then uses.

    rho           albedo of the surface, in the radiosity solve
    delta         angular radius of the source disc, in radians
    eye_distance  distance of every camera from the body centre, in canonical units
    tau_i         the fixed threshold below which pixels do not count toward the intensity
                  curve (the binary threshold is Otsu's, computed from the first frame and
                  not a parameter)
    pedestal      a per-curve offset added to every pixel value before thresholding
    eta           a per-curve model-error scale: the part of the residual at the true shape
                  that the noise does not explain. It weights the residual in the
                  calibration and at reconstruction; it does not enter the rendering
    sensor        the SensorModel: vignetting, PSF, OETF, saturation

Every quantity with a range is stored through a squashing function so it stays in range:
sigmoid for rho and tau_i, softplus for delta, eye_distance and eta.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sensor import SensorModel

__all__ = ["Instrument", "N_CURVES"]

N_CURVES = 56          # the released curves: every camera geometry, intensity then binary


def _inv_softplus(x: float) -> float:
    return float(np.log(np.expm1(x)))


def _inv_sigmoid(x: float) -> float:
    return float(np.log(x / (1.0 - x)))


class Instrument(nn.Module):
    """The fitted scene and sensor parameters. Construct with the starting values; load a
    calibration with `load`."""

    def __init__(self, rho: float = 0.85, delta_deg: float = 1.0, eye_distance: float = 8.0,
                 tau_i: float = 0.02, eta: float = 0.02, n_curves: int = N_CURVES,
                 quantise: bool = True):
        super().__init__()
        self.sensor = SensorModel(quantise=quantise)
        self.raw_rho = nn.Parameter(torch.tensor(_inv_sigmoid(rho)))
        self.raw_delta = nn.Parameter(torch.tensor(_inv_softplus(np.radians(delta_deg))))
        self.raw_eye = nn.Parameter(torch.tensor(_inv_softplus(eye_distance)))
        self.raw_tau_i = nn.Parameter(torch.tensor(_inv_sigmoid(tau_i)))
        self.pedestal = nn.Parameter(torch.zeros(n_curves))
        self.raw_eta = nn.Parameter(torch.full((n_curves,), _inv_softplus(eta)))

    @property
    def rho(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_rho)

    @property
    def delta(self) -> torch.Tensor:
        return F.softplus(self.raw_delta)

    @property
    def eye_distance(self) -> torch.Tensor:
        return F.softplus(self.raw_eye)

    @property
    def tau_i(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_tau_i)

    @property
    def eta(self) -> torch.Tensor:
        return F.softplus(self.raw_eta)

    def scene_parameters(self) -> list:
        """The parameters that change the rendered geometry or transport, as opposed to the
        sensor chain: rho, delta and the eye distance."""
        return [self.raw_rho, self.raw_delta, self.raw_eye]

    def summary(self) -> str:
        def f(x):
            return float(x.detach())
        return (f"rho {f(self.rho):.3f}, source radius {np.degrees(f(self.delta)):.2f} deg, "
                f"eye distance {f(self.eye_distance):.2f}, tau_i {f(self.tau_i):.4f}, "
                f"psf sigma {f(self.sensor.psf_sigma):.2f} px, "
                f"eta median {f(self.eta.median()):.4f}")

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, device="cpu") -> "Instrument":
        inst = cls()
        state = torch.load(path, map_location="cpu", weights_only=True)
        try:
            inst.load_state_dict(state)
        except (RuntimeError, TypeError) as exc:
            raise RuntimeError(f"{path} is not a saved Instrument; rerun "
                               f"scripts/calibrate.py to write one") from exc
        return inst.to(device)
