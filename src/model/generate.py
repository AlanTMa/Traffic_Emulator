"""
Random topology generation, using the notebook's capacity distributions.
"""
import numpy as np
import yaml
from src.controller.feasibility import transportation_feasibility
from src.model.config import topology_from_config

# Notebook (Phase 2) baseline capacity ranges (MB/s there; normalized work units/s here)
LINK_CAP_RANGE = (40.0, 60.0)
BROKER_CAP_RANGE = (100.0, 200.0)

def generate_topology_config(n_sources: int, n_brokers: int, load: float, seed: int = 42) -> dict:
    """
    Build a topology section (same shape as the YAML configs) for an
    n_sources x n_brokers instance.

    Source rates are heterogeneous (lognormal weights, like the notebook's
    mix of heavy and light producers) and scaled so the total offered rate is
    `load` times the total broker capacity.

    Raises:
        ValueError: if the access links cannot carry that load.
    """
    rng = np.random.default_rng(seed)
    mu_links = rng.uniform(*LINK_CAP_RANGE, size=(n_sources, n_brokers))
    mu_brokers = rng.uniform(*BROKER_CAP_RANGE, size=n_brokers)
    weights = rng.lognormal(mean=0.0, sigma=1.0, size=n_sources)
    lambdas = weights / weights.sum() * load * mu_brokers.sum()

    sources = [f"P{i}" for i in range(n_sources)]
    brokers = [f"SN{j + 1}" for j in range(n_brokers)]
    # Full float precision: rounding for readability could turn a feasible
    # instance into an infeasible saved one.
    topology = {
        "sources": [{"id": s, "rate": float(r)} for s, r in zip(sources, lambdas)],
        "brokers": [{"id": b, "capacity": float(c)} for b, c in zip(brokers, mu_brokers)],
        "access_capacities": {
            f"{s}->{b}": float(mu_links[i, j])
            for i, s in enumerate(sources) for j, b in enumerate(brokers)
        },
    }

    # Check feasibility of exactly what will be saved and read back
    saved = yaml.safe_load(yaml.safe_dump({"topology": topology}))
    topo = topology_from_config(saved)
    try:
        transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    except ValueError:
        raise ValueError(
            f"A {n_sources}x{n_brokers} topology at {load:.0%} load is infeasible: some "
            f"source sends more than its access links can carry. Lower the load or add brokers."
        ) from None
    return topology
