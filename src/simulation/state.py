"""Algorithm 1 state."""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SystemState:
    """lambda_ij, prices, and the last iteration's s_j, s_t, route_rel, price_rel
    and br_failures (the sources that were held)."""
    lambda_ij: np.ndarray
    prices: np.ndarray
    iteration: int = 0
    s_j: np.ndarray = None
    s_t: float = np.nan
    route_rel: float = np.nan
    price_rel: float = np.nan
    br_failures: list = field(default_factory=list)
