"""
One telemetry record per iteration/window. Matrices are [source][broker]
lists; _i and _j are per source and per broker. Model quantities are
evaluated on the planned routing; measured quantities have their own keys.
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
    """Record for a controller state; `extra` holds the mode-specific fields."""
    lambda_ij = np.asarray(lambda_ij, dtype=float)
    lambdas = topology.lambdas_total
    diag = compute_diagnostics(lambda_ij, prices, topology, eps=eps)
    D_ij, D_j = diag["D_ij"], diag["D_j"]
    # per-source mean delay
    e2e_i = (lambda_ij * (D_ij + D_j[None, :])).sum(axis=1) / np.maximum(lambdas, eps)

    record = {
        "schema": SCHEMA_VERSION,
        "controller_mode": controller_mode,
        "iteration": iteration,
        "sim_time": sim_time,
        "wall_time": time.time(),
        # routing
        "lambda_ij": lambda_ij,
        "fraction_ij": lambda_ij / np.maximum(lambdas, eps)[:, None],
        "load_j": diag["broker_loads"],
        "util_j": diag["broker_utilization"],
        "access_util_ij": np.divide(lambda_ij, topology.mu_links, out=np.full(lambda_ij.shape, np.nan),
                                    where=topology.mu_links > 0),     # NaN where no link
        # prices
        "price_j": np.asarray(prices, dtype=float),
        "alpha_i": diag["alpha_i"],
        # delays and marginal costs
        "D_ij": D_ij, "D_j": D_j, "e2e_i": e2e_i,
        "C_ij": diag["C_ij"], "C_j": diag["C_j"], "M_ij": diag["M_ij"],
        "active_ij": diag["active_ij"],
        "objective": system_objective(lambda_ij, topology.mu_links, topology.mu_brokers),
        # safe step (NaN without one)
        "s_j": s_j, "s_t": s_t,
        # residuals
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
