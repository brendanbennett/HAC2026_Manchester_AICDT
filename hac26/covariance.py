"""Measurement covariance: photon noise plus fitted model error, in the psi-Fourier basis.

Every data residual is whitened by 1/s^2 from here.

DFT convention, used by every caller through `psi_dft`: with psi_k = 2 pi k / N,

    rhat_m = (1/N) sum_k r_k e^{-i m psi_k}

so white noise of variance sigma^2 per sample gives Var(rhat_m) = sigma^2 / N.

Photon noise: two horizontal cameras share each azimuth, so their difference carries no
geometry and sigma_c follows from the replicate identity in hac26.noise. It appears on a
mean-normalised curve as sigma_c = sigma_photon / mean_c, which is why whitening is per curve.

Model error: the residual of the forward model against a body whose truth is known is not
noise. Its Fourier content is fitted as

    eta^2_{c,m} = gamma_c tau0^2 (1 + m/m0)^(-2p)

by maximum likelihood. gamma_c and tau0^2 are multiplicatively degenerate, so gamma is
normalised to unit geometric mean. A smooth parametric form is used because the empirical
per-(c, m) variance would rest on as many samples as there are bodies.

    s^2_{c,m} = sigma^2_{c,m} + eta^2_{c,m}

A direction determined only by model error therefore carries a large s^2 and is demoted
wherever this covariance is used.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["psi_dft", "photon_modes", "model_error", "fit_model_error", "whitening",
           "whitened_misfit", "save_covariance", "load_covariance"]


def psi_dft(curves: torch.Tensor, n_modes: int) -> torch.Tensor:
    """rhat_m for m = 1..n_modes, with the 1/N convention above. Input (..., N)."""
    n = curves.shape[-1]
    return torch.fft.rfft(curves, dim=-1)[..., 1:n_modes + 1] / n


def photon_modes(sigma_c: np.ndarray | torch.Tensor, n_phase: int,
                 n_modes: int) -> torch.Tensor:
    """sigma^2_{c,m} = sigma^2_c / N, constant in m because the noise is white in psi."""
    s = torch.as_tensor(sigma_c, dtype=torch.float32)
    return (s ** 2 / n_phase)[:, None].expand(-1, n_modes).contiguous()


def model_error(gamma_c: torch.Tensor, tau0: torch.Tensor, m0: torch.Tensor,
                p: torch.Tensor, n_modes: int) -> torch.Tensor:
    """eta^2_{c,m} = gamma_c tau0^2 (1 + m/m0)^(-2p), shape (C, n_modes)."""
    m = torch.arange(1, n_modes + 1, dtype=torch.float32, device=gamma_c.device)
    shape = (1.0 + m[None, :] / m0.clamp_min(1e-3)) ** (-2.0 * p.clamp_min(0.0))
    return gamma_c[:, None].clamp_min(1e-12) * (tau0 ** 2) * shape


def fit_model_error(rhat: torch.Tensor, sigma2_m: torch.Tensor,
                    mask: torch.Tensor | None = None, steps: int = 3000,
                    lr: float = 0.05, verbose: bool = True) -> dict:
    """Maximum likelihood fit of eta^2 to residual Fourier coefficients.

    rhat is (B, C, M) complex -- B bodies, C curves, M modes -- and sigma2_m is (C, M).
    Each coefficient is treated as complex Gaussian with variance s^2 = sigma^2 + eta^2, so

        -log L = sum [ log s^2 + |rhat|^2 / s^2 ]

    Returns the fitted parameters and the whitened covariance.
    """
    B, C, M = rhat.shape
    power = (rhat.real ** 2 + rhat.imag ** 2)                     # (B, C, M)
    if mask is None:
        mask = torch.ones(B, C, dtype=power.dtype)
    w = mask[:, :, None]

    log_gamma = torch.zeros(C, requires_grad=True)
    log_tau0 = torch.tensor(float(np.log(np.sqrt(power.mean().clamp_min(1e-12)))),
                            requires_grad=True)
    log_m0 = torch.tensor(1.0, requires_grad=True)
    log_p = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([log_gamma, log_tau0, log_m0, log_p], lr=lr)

    for i in range(steps):
        gamma = torch.exp(log_gamma - log_gamma.mean())            # unit geometric mean
        eta2 = model_error(gamma, torch.exp(log_tau0), torch.exp(log_m0),
                           torch.exp(log_p), M)
        s2 = (sigma2_m + eta2)[None].clamp_min(1e-18)
        nll = ((torch.log(s2) + power / s2) * w).sum() / w.sum().clamp_min(1.0)
        opt.zero_grad(); nll.backward(); opt.step()
        if verbose and (i % 500 == 0 or i == steps - 1):
            print(f"    step {i:>5}  -logL {float(nll):.5f}  tau0 {float(torch.exp(log_tau0)):.5f}"
                  f"  m0 {float(torch.exp(log_m0)):.3f}  p {float(torch.exp(log_p)):.3f}",
                  flush=True)

    gamma = torch.exp(log_gamma - log_gamma.mean()).detach()
    tau0 = torch.exp(log_tau0).detach(); m0 = torch.exp(log_m0).detach()
    p = torch.exp(log_p).detach()
    eta2 = model_error(gamma, tau0, m0, p, M).detach()
    return {"gamma_c": gamma, "tau0": tau0, "m0": m0, "p": p,
            "eta2": eta2, "sigma2": sigma2_m, "s2": sigma2_m + eta2,
            "n_modes": M, "nll": float(nll)}


def whitening(sigma2_m: torch.Tensor, eta2: torch.Tensor) -> torch.Tensor:
    """s^2 = sigma^2 + eta^2. The only weight any objective may use."""
    return sigma2_m + eta2


def whitened_misfit(pred: torch.Tensor, real: torch.Tensor, s2: torch.Tensor,
                    mask: torch.Tensor | None = None) -> torch.Tensor:
    """chi^2 per coefficient: mean over (c, m) of |pred_hat - real_hat|^2 / s^2.

    Both curve sets are (C, N) and already mean-normalised. This is the only weighting any
    objective should apply to a data residual; a weight of 1 in curve space would be a
    different, unstated prior.
    """
    n_modes = s2.shape[-1]
    r = psi_dft(pred, n_modes) - psi_dft(real, n_modes)
    chi = (r.real ** 2 + r.imag ** 2) / s2.clamp_min(1e-18)
    if mask is not None:
        chi = chi * mask[:, None]
        return chi.sum() / mask.sum().clamp_min(1.0) / n_modes
    return chi.mean()


def save_covariance(fit: dict, path: str) -> None:
    torch.save({k: (v if torch.is_tensor(v) else torch.tensor(v)) for k, v in fit.items()
                if k != "nll"} | {"nll": float(fit["nll"])}, path)


def load_covariance(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
