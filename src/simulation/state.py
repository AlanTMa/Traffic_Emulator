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
        s_j: (M,) Per-broker step bounds of the last iteration (None before any).
        s_t: Common safe step used by the last iteration (nan before any).
        route_rel: Relative routing change of the last iteration.
        price_rel: Relative price change of the last iteration.
    """
    lambda_ij: np.ndarray
    prices: np.ndarray
    iteration: int = 0
    s_j: np.ndarray = None
    s_t: float = np.nan
    route_rel: float = np.nan
    price_rel: float = np.nan
