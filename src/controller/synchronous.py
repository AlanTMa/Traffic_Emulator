"""
Synchronous implementation of the WiOpt distributed routing algorithm.
"""
import numpy as np
from src.model.marginal_costs import mm1_marginal_cost_vectorized
from src.controller.best_response import best_response_mm1
from src.simulation.state import SystemState

def compute_broker_loads(lambda_ij: np.ndarray) -> np.ndarray:
    """Compute aggregate load at each broker: Lambda_j = sum_i lambda_ij"""
    return lambda_ij.sum(axis=0)

def update_prices(current_prices: np.ndarray, loads: np.ndarray, mu_brokers: np.ndarray, gamma: float, eps: float = 1e-12) -> np.ndarray:
    """
    Update broker prices using price damping.
    p_new = (1-gamma)*p + gamma * (mu / (mu - Lambda)^2)
    """
    # Current marginal prices (p_hat)
    p_hat = mm1_marginal_cost_vectorized(loads, mu_brokers, eps)
    return (1.0 - gamma) * current_prices + gamma * p_hat

def compute_safe_step(lambda_ij: np.ndarray, lambda_br: np.ndarray, loads: np.ndarray, mu_brokers: np.ndarray, eps: float = 1e-12) -> float:
    """
    Calculate the common safe step s_t to ensure broker feasibility.
    s_t = min(1, min_j (mu_j - Lambda_j) / Delta_Lambda_j) for Delta_Lambda_j > 0.
    """
    direction = lambda_br - lambda_ij
    load_direction = direction.sum(axis=0)

    safe_fraction = 1.0
    for j in range(len(mu_brokers)):
        if load_direction[j] > 0:
            # Capacity headroom / proposed increase
            fraction = (mu_brokers[j] - eps - loads[j]) / load_direction[j]
            safe_fraction = min(safe_fraction, fraction)

    return max(0.0, min(1.0, safe_fraction))

def iteration_step(state: SystemState, topology, eta: float, gamma: float, eps: float = 1e-12):
    """
    Perform one iteration of the distributed synchronous algorithm.
    """
    l_prev = state.lambda_ij.copy()
    p_prev = state.prices.copy()

    # 1. Calculate current loads
    loads = compute_broker_loads(state.lambda_ij)

    # 2. Update prices
    new_prices = update_prices(state.prices, loads, topology.mu_brokers, gamma, eps)

    # 3. Each source solves best response
    lambda_br = np.zeros_like(state.lambda_ij)
    for i in range(topology.n_sources):
        lambda_br[i, :] = best_response_mm1(
            topology.mu_links[i, :],
            new_prices,
            topology.lambdas_total[i]
        )

    # 4. Calculate safe step
    s_t = compute_safe_step(state.lambda_ij, lambda_br, loads, topology.mu_brokers, eps)

    # 5. Update routing
    step = eta * s_t
    new_lambda_ij = (1.0 - step) * state.lambda_ij + step * lambda_br

    # Calculate residuals for convergence
    route_rel = np.linalg.norm(new_lambda_ij - l_prev) / max(np.linalg.norm(l_prev), eps)
    price_rel = np.linalg.norm(new_prices - p_prev) / max(np.linalg.norm(p_prev), eps)

    # Update state
    state.lambda_ij = new_lambda_ij
    state.prices = new_prices
    state.iteration += 1

    return state, s_t, max(route_rel, price_rel)

