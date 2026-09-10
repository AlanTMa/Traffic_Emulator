"""
Runtime diagnostics for Proposition 1 (paper Sec. IV-H).

Proposition 1: if a routing/price pair (lambda*, p*) is feasible and satisfies
  (17) price consistency   p*_j = C_j(Lambda*_j), and
  (18) local best response C_ij(lambda*_ij) + p*_j = alpha*_i on used routes,
                           >= alpha*_i on unused routes,
then lambda* is globally system-optimal for (4). The paper does not claim
global convergence of Algorithm 1 on the open M/M/1 domain; these
diagnostics certify a *given state*, they say nothing about reaching one.

Definitions follow the reference notebook's verification_residuals() with
two deliberate differences, documented where they occur: the fixed point is
evaluated under the published prices p (condition 18), and inactive-route
complementarity compares against the minimum used-route marginal cost.
"""
import numpy as np
from src.controller.best_response import best_response_mm1

# Reference notebook's predetermined tolerances (residual_tolerances)
DEFAULT_TOLERANCES = {
    "r_conservation": 1e-8,
    "r_nonnegative": 1e-10,
    "r_access_capacity": 1e-10,
    "r_service_capacity": 1e-10,
    "r_price": 1e-8,
    "r_fixed_point": 1e-6,
    "r_kkt_complementarity": 1e-8,
    "r_active_stationarity": 1e-6,
    "r_inactive_complementarity": 1e-8,
}

CERTIFIED_MESSAGE = "Current state satisfies feasible fixed-point optimality conditions"

def marginal_costs(lambda_ij: np.ndarray, mu_links: np.ndarray, mu_brokers: np.ndarray, eps: float = 1e-12):
    """
    M/M/1 delays and marginal costs (paper eqs. 1, 2, 12, 13).

    Returns dict with D_ij, D_j, C_ij = mu_ij/(mu_ij-lambda_ij)^2,
    C_j = mu_j/(mu_j-Lambda_j)^2 and M_ij = C_ij + C_j.
    """
    loads = lambda_ij.sum(axis=0)
    link_gap = np.maximum(mu_links - lambda_ij, eps)
    broker_gap = np.maximum(mu_brokers - loads, eps)
    C_ij = mu_links / link_gap ** 2
    C_j = mu_brokers / broker_gap ** 2
    return {
        "D_ij": 1.0 / link_gap,
        "D_j": 1.0 / broker_gap,
        "C_ij": C_ij,
        "C_j": C_j,
        "M_ij": C_ij + C_j[None, :],
    }

def compute_diagnostics(lambda_ij: np.ndarray, prices: np.ndarray, topology, active_tol: float = 1e-7,
                        tolerances: dict = None, eps: float = 1e-12) -> dict:
    """
    Certificate residuals for a routing lambda_ij and published prices p.

    Scalars (r_*) are worst cases over sources/brokers; per-source and
    per-broker arrays are included for telemetry. 'certified' is True only if
    every residual is within its tolerance; 'failed' lists those that are not.
    """
    tolerances = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    lambda_ij = np.asarray(lambda_ij, dtype=float)
    prices = np.asarray(prices, dtype=float)
    lambdas, mu_links, mu_brokers = topology.lambdas_total, topology.mu_links, topology.mu_brokers
    n_sources, n_brokers = lambda_ij.shape
    loads = lambda_ij.sum(axis=0)
    mc = marginal_costs(lambda_ij, mu_links, mu_brokers, eps)
    M = mc["M_ij"]
    active = lambda_ij > active_tol

    # Local best-response structure, per source
    alpha = np.full(n_sources, np.nan)          # mean used-route marginal cost
    spread = np.full(n_sources, np.inf)         # max - min over used routes
    stationarity = np.full(n_sources, np.inf)   # max |M_ij - alpha_i| over used routes
    inactive_gap = np.zeros(n_sources)          # how far an unused route undercuts the used minimum
    for i in range(n_sources):
        if not np.any(active[i]):
            continue
        used = M[i, active[i]]
        alpha[i] = used.mean()
        spread[i] = used.max() - used.min()
        stationarity[i] = np.max(np.abs(used - alpha[i]))
        if np.any(~active[i]):
            # Condition (18): unused routes must cost at least as much as the
            # cheapest used route (notebook compares against the mean alpha_i)
            inactive_gap[i] = max(0.0, used.min() - M[i, ~active[i]].min())

    # Threshold-free KKT complementarity: beta_ij = M_ij - min_j M_ij >= 0 is
    # a valid dual slack; lambda_ij * beta_ij must vanish.
    beta = M - M.min(axis=1, keepdims=True)

    # Fixed point under the published prices (condition 18); the notebook
    # uses the model prices C_j(Lambda), which coincide at a price-consistent state.
    br = np.vstack([best_response_mm1(mu_links[i], prices, lambdas[i]) for i in range(n_sources)])
    fixed_point_per_source = np.linalg.norm(lambda_ij - br, axis=1)

    access_margin = float(np.max(lambda_ij - mu_links))
    service_margin = float(np.max(loads - mu_brokers))
    result = {
        # Primal feasibility
        "r_conservation": float(np.max(np.abs(lambda_ij.sum(axis=1) - lambdas))),
        "r_nonnegative": float(np.max(np.maximum(-lambda_ij, 0.0))),
        "access_margin": access_margin,     # max(lambda_ij - mu_ij); negative = headroom
        "service_margin": service_margin,   # max(Lambda_j - mu_j)
        "r_access_capacity": max(access_margin, 0.0),
        "r_service_capacity": max(service_margin, 0.0),
        # Condition (17)
        "r_price": float(np.max(np.abs(prices - mc["C_j"]))),
        # Condition (18) and KKT
        "r_fixed_point": float(np.linalg.norm(fixed_point_per_source)),
        "r_active_stationarity": float(np.max(stationarity)),
        "r_inactive_complementarity": float(np.max(inactive_gap)),
        "r_kkt_complementarity": float(np.max(np.abs(lambda_ij * beta))),
        # Per-source / per-broker detail
        "alpha_i": alpha,
        "active_spread_i": spread,
        "fixed_point_i": fixed_point_per_source,
        "best_response_ij": br,
        "active_ij": active,
        "broker_loads": loads,
        "broker_utilization": loads / mu_brokers,
        **mc,
    }
    failed = [k for k, tol in tolerances.items() if not (abs(result[k]) <= tol)]
    result["failed"] = failed
    result["certified"] = not failed
    result["status"] = CERTIFIED_MESSAGE if not failed else "Not certified: " + ", ".join(failed)
    return result
