"""
Delay functions for the M/M/1 queueing model.
"""
import numpy as np

def mm1_delay(x: float, mu: float, eps: float = 1e-8) -> float:
    """
    Calculate the M/M/1 delay: D(x) = 1 / (mu - x)

    Args:
        x: Arrival rate (lambda)
        mu: Service capacity
        eps: Numerical protection to avoid division by zero

    Returns:
        The average delay.
    """
    if x >= mu:
        # In the stability domain x < mu, but for numerical stability
        # we cap the denominator.
        return 1.0 / eps
    return 1.0 / (mu - x)

def mm1_delay_vectorized(x: np.ndarray, mu: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Vectorized M/M/1 delay for numpy arrays.
    """
    # Use np.maximum to ensure we don't divide by zero or negative values
    return 1.0 / np.maximum(mu - x, eps)
