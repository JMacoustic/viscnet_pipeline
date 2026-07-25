"""Gaussian-mixture (UQ) head helpers.

The GMM model's flat [B, 3K] output is interpreted as [mu_K | log_sigma_K | pi_logits_K].
The mixture is collapsed to a single predictive (mean, std) by the law of total
variance. Everything here is in STANDARDIZED log10(cP) space.
"""

from __future__ import annotations

import numpy as np
import torch


def split_gmm_params(outputs: torch.Tensor, k: int):
    """[B, 3K] -> (mu [B,K], sigma [B,K], pi [B,K])."""
    mu = outputs[:, :k]
    sigma = torch.exp(outputs[:, k : 2 * k].clamp(-6.0, 3.0))
    pi = torch.softmax(outputs[:, 2 * k : 3 * k].float(), dim=-1)
    return mu, sigma, pi


def gmm_moments(mu: np.ndarray, sigma: np.ndarray, pi: np.ndarray):
    """Mixture mean and std (law of total variance): [N,K] -> ([N], [N])."""
    mean = np.sum(pi * mu, axis=-1)
    var = np.sum(pi * (sigma ** 2 + mu ** 2), axis=-1) - mean ** 2
    return mean, np.sqrt(np.maximum(var, 0.0))
