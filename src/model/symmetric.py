"""
Closed-form optimum of the symmetric N x M instance (docs/symmetric_case.md):
the equal split lambda_ij = lam/M, when lam/M < mu_access and N lam/M < mu_server.
"""
import numpy as np
from src.model.topology import Topology

def symmetric_topology(n_sources: int, n_brokers: int, lam: float, mu_access: float, mu_server: float) -> Topology:
    return Topology(np.full(n_sources, float(lam)), np.full((n_sources, n_brokers), float(mu_access)),
                    np.full(n_brokers, float(mu_server)),
                    [f"P{i}" for i in range(n_sources)], [f"SN{j + 1}" for j in range(n_brokers)])

def symmetric_solution(n_sources: int, n_brokers: int, lam: float, mu_access: float, mu_server: float) -> dict:
    """Equal-split optimum; ValueError when infeasible."""
    x = lam / n_brokers                       # per-route flow lambda_ij
    load = n_sources * lam / n_brokers        # broker load Lambda_j
    if not x < mu_access:
        raise ValueError(f"infeasible: lam/M = {x} >= mu_access = {mu_access}")
    if not load < mu_server:
        raise ValueError(f"infeasible: N*lam/M = {load} >= mu_server = {mu_server}")
    c_access = mu_access / (mu_access - x) ** 2          # C_ij = D + x D'
    price = mu_server / (mu_server - load) ** 2          # p_j = C_j(Lambda_j)
    d_access, d_server = 1.0 / (mu_access - x), 1.0 / (mu_server - load)
    return {
        "lambda_ij": np.full((n_sources, n_brokers), x),
        "broker_load": load,
        "rho_server": load / mu_server,                  # N*lam / (M*mu_server)
        "rho_access": x / mu_access,
        "access_marginal_cost": c_access,
        "price": price,
        "alpha": c_access + price,                       # identical on every route
        "mean_delay": d_access + d_server,               # per work unit, every route
        "objective": n_sources * lam * (d_access + d_server),
    }
