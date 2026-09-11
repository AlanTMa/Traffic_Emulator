"""M/M/1 marginal cost."""
import numpy as np

def mm1_marginal_cost(x: float, mu: float, eps: float = 1e-8) -> float:
    """C(x) = mu / (mu - x)^2."""
    if x >= mu:
        return mu / (eps**2)
    return mu / ((mu - x)**2)

def mm1_marginal_cost_vectorized(x: np.ndarray, mu: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """C(x) = mu / (mu - x)^2, elementwise."""
    return mu / np.maximum(mu - x, eps)**2
