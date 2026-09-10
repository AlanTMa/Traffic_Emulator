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

def compute_safe_step(lambda_ij: np.ndarray, lambda_br: np.ndarray, loads: np.ndarray, mu_brokers: np.ndarray,
                      eta: float, delta_s: float = 1e-8) -> float:
    """
    Common safe step s_t of Algorithm 1, with capacity margin delta_s.

    With Delta_j = sum_i (lambda_br_ij - lambda_ij), for every broker with Delta_j > 0:
        s_j = min(1, (mu_j - delta_s - Lambda_j) / (eta * Delta_j))
    and s_j = 1 otherwise; s_t = min_j s_j. The routing update then moves by
    eta * s_t, so Lambda_j + eta * s_t * Delta_j <= mu_j - delta_s for every j.
    """
    load_direction = (lambda_br - lambda_ij).sum(axis=0)

    s_t = 1.0
    for j in range(len(mu_brokers)):
        if load_direction[j] > 0:
            s_j = (mu_brokers[j] - delta_s - loads[j]) / (eta * load_direction[j])
            s_t = min(s_t, s_j)

    return max(0.0, s_t)

def iteration_step(state: SystemState, topology, eta: float, gamma: float, eps: float = 1e-12,
                   delta_s: float = 1e-8):
    """
    Perform one iteration of Algorithm 1.

    delta_s is the capacity margin kept below every broker's capacity by the
    safe step (paper: delta_s; the reference notebook uses 1e-8). eps only
    guards divisions against zero and is not a modeling parameter.
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
    s_t = compute_safe_step(state.lambda_ij, lambda_br, loads, topology.mu_brokers, eta, delta_s)

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

