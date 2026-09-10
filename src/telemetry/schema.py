"""
Telemetry record schema: one JSON object per controller iteration/window.

Matrices are nested lists indexed [source][broker]; vectors are per source
(_i) or per broker (_j). Model quantities (D, C, objective, residuals) are
evaluated on the controller's routing lambda_ij with the M/M/1 formulas; in
windowed_stochastic mode that routing is the *planned* split, and measured
quantities are recorded alongside under their own keys.

Units: rates are normalized work units/s (see src/model/topology.py); the
latency_* fields are end-to-end sojourn times of one work unit in seconds.
"""
import time
import numpy as np
from src.controller.central import system_objective
from src.controller.diagnostics import compute_diagnostics

SCHEMA_VERSION = 1

def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value

def controller_snapshot(topology, lambda_ij: np.ndarray, prices: np.ndarray, *, iteration: int, sim_time: float,
                        controller_mode: str, s_j: np.ndarray = None, s_t: float = np.nan,
                        route_rel: float = np.nan, price_rel: float = np.nan, eps: float = 1e-12,
                        **extra) -> dict:
    """
    Build a telemetry record for a controller state (lambda_ij, published prices).

    extra: mode-specific fields (e.g. measured rates, queue lengths), stored as is.
    """
    lambda_ij = np.asarray(lambda_ij, dtype=float)
    lambdas = topology.lambdas_total
    diag = compute_diagnostics(lambda_ij, prices, topology, eps=eps)
    D_ij, D_j = diag["D_ij"], diag["D_j"]
    # Model mean end-to-end delay per source: sum_j lambda_ij (D_ij + D_j) / lambda_i
    e2e_i = (lambda_ij * (D_ij + D_j[None, :])).sum(axis=1) / np.maximum(lambdas, eps)

    record = {
        "schema": SCHEMA_VERSION,
        "controller_mode": controller_mode,
        "iteration": iteration,
        "sim_time": sim_time,
        "wall_time": time.time(),
        # Routing and loads
        "lambda_ij": lambda_ij,
        "fraction_ij": lambda_ij / np.maximum(lambdas, eps)[:, None],
        "load_j": diag["broker_loads"],
        "util_j": diag["broker_utilization"],
        "access_util_ij": lambda_ij / topology.mu_links,
        # Prices and multipliers
        "price_j": np.asarray(prices, dtype=float),
        "alpha_i": diag["alpha_i"],
        # Delays and marginal costs
        "D_ij": D_ij, "D_j": D_j, "e2e_i": e2e_i,
        "C_ij": diag["C_ij"], "C_j": diag["C_j"], "M_ij": diag["M_ij"],
        "active_ij": diag["active_ij"],
        "objective": system_objective(lambda_ij, topology.mu_links, topology.mu_brokers),
        # Algorithm 1 step (NaN/None where the controller has no safe step)
        "s_j": s_j, "s_t": s_t,
        # Residuals
        "route_rel": route_rel, "price_rel": price_rel,
        "r_conservation": diag["r_conservation"],
        "r_capacity": max(diag["r_access_capacity"], diag["r_service_capacity"]),
        "access_margin": diag["access_margin"], "service_margin": diag["service_margin"],
        "r_price": diag["r_price"],
        "r_fixed_point": diag["r_fixed_point"],
        "r_kkt_stationarity": diag["r_active_stationarity"],
        "active_spread_i": diag["active_spread_i"],
        "r_kkt_complementarity": diag["r_kkt_complementarity"],
        "r_inactive_complementarity": diag["r_inactive_complementarity"],
        "certified": diag["certified"],
        "certificate_status": diag["status"],
        **extra,
    }
    return {k: _jsonable(v) for k, v in record.items()}
