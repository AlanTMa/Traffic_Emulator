"""
Marginal cost functions for the M/M/1 queueing model.
"""
import numpy as np

def mm1_marginal_cost(x: float, mu: float, eps: float = 1e-8) -> float:
    """
    Calculate the M/M/1 marginal cost: C(x) = mu / (mu - x)^2

    Args:
        x: Arrival rate (lambda)
        mu: Service capacity
        eps: Numerical protection

    Returns:
        The marginal cost.
    """
    if x >= mu:
        return mu / (eps**2)
    return mu / ((mu - x)**2)

def mm1_marginal_cost_vectorized(x: np.ndarray, mu: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Vectorized M/M/1 marginal cost for numpy arrays.
    """
    return mu / np.maximum(mu - x, eps)**2
