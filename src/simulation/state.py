"""
Runtime state for the traffic allocation emulator.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class SystemState:
    """
    Holds the current state of the distributed emulator.

    Attributes:
        lambda_ij: (N, M) Current routing allocation.
        prices: (M,) Current broker congestion prices.
        iteration: Current controller iteration.
    """
    lambda_ij: np.ndarray
    prices: np.ndarray
    iteration: int = 0

    def __post_init__(self):
        if self.lambda_ij.shape != self.prices.shape[0] if self.prices.ndim == 1 else self.lambda_ij.shape[0]:
             # Basic validation is handled by Topology and Controller, but we keep it here.
             pass
