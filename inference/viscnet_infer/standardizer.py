"""Target standardizer: model output is standardized log10(cP).

Inverse:  log10(cP) = z * std + mean   ->   cP = 10 ** (z * std + mean)
mean/std are fit on log10(cP) over the training split and stored in a small JSON.
"""

from __future__ import annotations

import json

import numpy as np


class Standardizer:
    def __init__(self, mean: float, std: float, space: str = "standardized_log10"):
        self.mean = float(mean)
        self.std = float(std)
        self.space = space

    @classmethod
    def from_json(cls, path: str) -> "Standardizer":
        j = json.load(open(path))
        return cls(j["mean"][0], j["std"][0], j.get("space", "standardized_log10"))

    def inverse_log10(self, z) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.std + self.mean

    def inverse_cP(self, z) -> np.ndarray:
        return 10.0 ** self.inverse_log10(z)
