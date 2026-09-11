"""
Proposition 1 residuals (Sec. IV-H): a feasible (lambda, p) with p_j = C_j(Lambda_j)
(17) and C_ij + p_j = alpha_i on used routes, >= alpha_i on unused ones (18) is
optimal. This certifies a given state; it says nothing about reaching one.

Two differences from the notebook's verification_residuals: the fixed point
uses the published prices p, and unused routes are compared with the cheapest
used route rather than the mean.
"""
import numpy as np
from src.controller.best_response import best_response_available

# the notebook's residual_tolerances
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
    "r_unavailable_routes": 0.0,     # no flow on missing links
}

CERTIFIED_MESSAGE = "Current state satisfies feasible fixed-point optimality conditions"

def marginal_costs(lambda_ij: np.ndarray, mu_links: np.ndarray, mu_brokers: np.ndarray, eps: float = 1e-12):
    """D_ij, D_j, C_ij, C_j and M_ij = C_ij + C_j (eqs. 1, 2, 12, 13); NaN on missing links."""
    loads = lambda_ij.sum(axis=0)
    link_gap = np.maximum(mu_links - lambda_ij, eps)
    broker_gap = np.maximum(mu_brokers - loads, eps)
    D_ij = 1.0 / link_gap
    C_ij = mu_links / link_gap ** 2
    C_j = mu_brokers / broker_gap ** 2
    M_ij = C_ij + C_j[None, :]
    unavailable = ~(mu_links > 0)
    if unavailable.any():
        D_ij, C_ij, M_ij = (np.where(unavailable, np.nan, a) for a in (D_ij, C_ij, M_ij))
    return {"D_ij": D_ij, "D_j": 1.0 / broker_gap, "C_ij": C_ij, "C_j": C_j, "M_ij": M_ij}

def compute_diagnostics(lambda_ij: np.ndarray, prices: np.ndarray, topology, active_tol: float = 1e-7,
                        tolerances: dict = None, eps: float = 1e-12) -> dict:
    """Residuals of (lambda_ij, p); the r_* scalars are worst cases, 'certified' means
    every one is within tolerance, 'failed' lists those that are not."""
    tolerances = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    lambda_ij = np.asarray(lambda_ij, dtype=float)
    prices = np.asarray(prices, dtype=float)
    lambdas, mu_links, mu_brokers = topology.lambdas_total, topology.mu_links, topology.mu_brokers
    n_sources, n_brokers = lambda_ij.shape
    loads = lambda_ij.sum(axis=0)
    mc = marginal_costs(lambda_ij, mu_links, mu_brokers, eps)
    M = mc["M_ij"]
    available = mu_links > 0
    active = (lambda_ij > active_tol) & available

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
        unused = available[i] & ~active[i]
        if np.any(unused):
            # (18): an unused route must cost at least the cheapest used one (the notebook uses the mean)
            inactive_gap[i] = max(0.0, used.min() - M[i, unused].min())

    # lambda_ij (M_ij - min_j M_ij) must vanish
    if available.all():
        kkt_complementarity = float(np.max(np.abs(lambda_ij * (M - M.min(axis=1, keepdims=True)))))
    else:
        row_min = np.array([M[i, available[i]].min() if available[i].any() else 0.0 for i in range(n_sources)])
        kkt_complementarity = float(np.max(np.where(available, np.abs(lambda_ij * (M - row_min[:, None])), 0.0)))

    # fixed point under the published prices (the notebook uses C_j(Lambda))
    br = np.vstack([best_response_available(mu_links[i], prices, lambdas[i]) for i in range(n_sources)])
    fixed_point_per_source = np.linalg.norm(lambda_ij - br, axis=1)

    access_margin = float(np.max((lambda_ij - mu_links)[available]))
    service_margin = float(np.max(loads - mu_brokers))
    result = {
        # feasibility
        "r_conservation": float(np.max(np.abs(lambda_ij.sum(axis=1) - lambdas))),
        "r_nonnegative": float(np.max(np.maximum(-lambda_ij, 0.0))),
        "access_margin": access_margin,     # max(lambda_ij - mu_ij); negative = headroom
        "service_margin": service_margin,   # max(Lambda_j - mu_j)
        "r_access_capacity": max(access_margin, 0.0),
        "r_service_capacity": max(service_margin, 0.0),
        # (17)
        "r_price": float(np.max(np.abs(prices - mc["C_j"]))),
        # (18), KKT
        "r_fixed_point": float(np.linalg.norm(fixed_point_per_source)),
        "r_active_stationarity": float(np.max(stationarity)),
        "r_inactive_complementarity": float(np.max(inactive_gap)),
        "r_kkt_complementarity": kkt_complementarity,
        "r_unavailable_routes": float(np.max(np.abs(lambda_ij[~available]), initial=0.0)),
        # per source / per broker
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
