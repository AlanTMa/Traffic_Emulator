import numpy as np
import pytest
from scripts.sweep_5x3 import base_topology, event_experiment, max_multiplier, static_experiment
from src.controller.feasibility import transportation_feasibility

def test_max_multiplier_is_set_by_access_links():
    r_max = max_multiplier()
    assert r_max == pytest.approx(4.8695, abs=1e-3)
    topo = base_topology(r_max)
    # P2 saturates its access links while aggregate broker load stays ~63%
    assert topo.lambdas_total[2] / topo.mu_links[2].sum() == pytest.approx(1.0, abs=1e-6)
    assert topo.lambdas_total.sum() / topo.mu_brokers.sum() == pytest.approx(0.634, abs=1e-3)

def test_static_experiment_high_load():
    topo = base_topology(0.95 * max_multiplier())
    L, s = static_experiment(topo)
    assert s["certified"]
    assert s["iterations"] > s["iterations_to_tol"]
    assert abs(s["relative_objective_gap"]) < 1e-7
    assert s["min_broker_headroom"] > 0 and s["min_link_headroom"] > 0
    assert L.sum(axis=1) == pytest.approx(topo.lambdas_total, abs=1e-8)

def test_event_experiment_smoke():
    topo = base_topology(0.5 * max_multiplier())
    x0 = transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    ev = event_experiment(topo, x0, frozen=False, duration=60.0, warmup_time=20.0, seed=0)
    assert ev["completed"] > 1000
    assert ev["latency_p50"] <= ev["latency_p95"] <= ev["latency_p99"]
    assert len(ev["util_measured_j"]) == 3
