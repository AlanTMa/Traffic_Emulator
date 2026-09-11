"""Sparse access topologies (some source->broker links do not exist)."""
import numpy as np
import pytest

from src.controller.feasibility import random_feasible_routing, transportation_feasibility

def test_notebook_sparse_lp():
    # Reference notebook sparse_feasibility_regression(): both sources can only
    # reach SN1 (capacity 1) with total demand 2. Aggregate capacity (2) would
    # suggest feasibility; the masked LP must reject it.
    with pytest.raises(ValueError):
        transportation_feasibility(np.array([1.0, 1.0]), np.ones((2, 2)), np.ones(2),
                                   route_mask=np.array([[1, 0], [1, 0]], dtype=bool), margin=1e-9)
    feasible = transportation_feasibility(np.array([1.0, 1.0]), 2.0 * np.ones((2, 2)), np.array([1.2, 1.2]),
                                          route_mask=np.ones((2, 2), dtype=bool), margin=1e-9)
    assert np.max(np.abs(feasible.sum(axis=1) - 1.0)) <= 1e-10
    assert np.max(np.maximum(feasible.sum(axis=0) - 1.2, 0.0)) <= 1e-10

def test_masked_routes_carry_no_flow():
    mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    mu_links = np.where(mask, 5.0, 0.0)          # default mask = mu_links > 0
    L = transportation_feasibility(np.array([3.0, 3.0]), mu_links, np.array([4.0, 4.0, 4.0]))
    assert np.all(L[~mask] == 0.0)
    assert L.sum(axis=1) == pytest.approx([3.0, 3.0])
    R = random_feasible_routing(np.array([3.0, 3.0]), mu_links, np.array([4.0, 4.0, 4.0]), seed=1)
    assert np.all(R[~mask] == 0.0) and R.sum(axis=1) == pytest.approx([3.0, 3.0])

# --- Sparse topologies end to end ---

import json
import yaml

from src.cli import run_simulation
from src.controller.central import solve_central, system_objective
from src.controller.diagnostics import compute_diagnostics, marginal_costs
from src.controller.synchronous import run_algorithm1
from src.model.config import load_config, topology_from_config
from src.model.topology import Topology
from src.runtime.experiments import event_experiment

SPARSE_CONFIG = {
    "topology": {
        # P0 cannot reach SN3, P1 cannot reach SN1; P2 reaches every broker
        "sources": [{"id": "P0", "rate": 6.0}, {"id": "P1", "rate": 5.0}, {"id": "P2", "rate": 8.0}],
        "brokers": [{"id": "SN1", "capacity": 12.0}, {"id": "SN2", "capacity": 9.0}, {"id": "SN3", "capacity": 10.0}],
        "access_capacities": {"P0->SN1": 7.0, "P0->SN2": 6.0, "P1->SN2": 5.0, "P1->SN3": 8.0,
                              "P2->SN1": 6.0, "P2->SN2": 4.0, "P2->SN3": 7.0},
        "unavailable_links": ["P0->SN3", "P1->SN1"],
    }
}
MASK = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=bool)

def sparse_topology():
    return topology_from_config(SPARSE_CONFIG)

def test_unavailable_links_mask():
    t = sparse_topology()
    assert t.route_mask.tolist() == MASK.tolist()
    assert t.mu_links[0, 2] == 0.0 and t.mu_links[1, 0] == 0.0

@pytest.mark.parametrize("change, match", [
    (lambda t: t["unavailable_links"].append("P0->SN1"), "specified more than once"),     # also has a capacity
    (lambda t: t["unavailable_links"].append("P0->SN3"), "specified more than once"),     # listed twice
    (lambda t: t["unavailable_links"].append("P9->SN1"), "unknown source"),
    (lambda t: t["unavailable_links"].append("P0-SN1"), "malformed"),
    (lambda t: t.update(unavailable_links="P0->SN3"), "must be a list"),
    (lambda t: t["unavailable_links"].remove("P1->SN1"), r"missing access capacities for \['P1->SN1'\]"),
    (lambda t: t.update(unavailable_links=["P0->SN3", "P1->SN1", "P1->SN2", "P1->SN3"],
                        access_capacities={k: v for k, v in t["access_capacities"].items() if not k.startswith("P1")}),
     "no available access link"),
])
def test_invalid_sparse_configs(change, match):
    import copy
    cfg = copy.deepcopy(SPARSE_CONFIG)
    change(cfg["topology"])
    with pytest.raises(ValueError, match=match):
        topology_from_config(cfg)

def test_zero_capacity_needs_mask():
    with pytest.raises(ValueError, match="access capacities must be > 0"):
        Topology([1.0], [[1.0, 0.0]], [5.0, 5.0], ["A"], ["S1", "S2"])
    t = Topology([1.0], [[1.0, 3.0]], [5.0, 5.0], ["A"], ["S1", "S2"], route_mask=[[True, False]])
    assert t.mu_links.tolist() == [[1.0, 0.0]]     # the masked capacity is dropped

def test_sparse_marginal_costs_and_objective():
    t = sparse_topology()
    L = np.where(MASK, 1.0, 0.0)
    mc = marginal_costs(L, t.mu_links, t.mu_brokers)
    assert np.all(np.isnan(mc["M_ij"][~MASK])) and np.all(np.isfinite(mc["M_ij"][MASK]))
    assert np.isfinite(system_objective(L, t.mu_links, t.mu_brokers))
    assert system_objective(np.where(MASK, 1.0, 0.5), t.mu_links, t.mu_brokers) == np.inf   # flow on a missing link

def test_sparse_algorithm1_vs_central():
    t = sparse_topology()
    L_c, info = solve_central(t.lambdas_total, t.mu_links, t.mu_brokers)
    assert np.all(L_c[~MASK] == 0.0)
    state = run_algorithm1(t, tol=1e-12, max_iter=5000, require_certificate=True)
    assert np.all(state.lambda_ij[~MASK] == 0.0)                  # exactly zero, every iterate
    d = compute_diagnostics(state.lambda_ij, state.prices, t)
    assert d["certified"], d["failed"]
    assert d["r_unavailable_routes"] == 0.0
    assert state.lambda_ij == pytest.approx(L_c, abs=1e-6)
    gap = (system_objective(state.lambda_ij, t.mu_links, t.mu_brokers) - info["objective"]) / info["objective"]
    assert abs(gap) <= 1e-7
    # A routing that uses a missing link is not certified
    bad = state.lambda_ij.copy()
    bad[0, 2], bad[0, 0] = 0.1, bad[0, 0] - 0.1
    assert "r_unavailable_routes" in compute_diagnostics(bad, state.prices, t)["failed"]

@pytest.mark.parametrize("mode", ["capacity_safe_event_driven", "windowed_stochastic"])
def test_no_flow_on_missing_links(mode):
    t = sparse_topology()
    ev = event_experiment(t, mode, duration=100.0, warmup_time=20.0, seed=0)
    assert ev["completed"] > 1000
    assert np.all(np.isnan(np.array(ev["access_util_planned_ij"], dtype=float)[~MASK]))   # no link: NaN
    assert np.all(np.isnan(ev["access_util_actual_ij"][~MASK]))
    assert np.all(ev["queue_access_mean_ij"][~MASK] == 0)
    assert np.all(np.array(ev["fraction_ij"])[~MASK] == 0.0)

def test_sparse_config_runs_from_cli(tmp_path):
    cfg = dict(SPARSE_CONFIG, simulation={"controller_mode": "capacity_safe_event_driven", "duration": 20, "seed": 1})
    path = tmp_path / "sparse.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run_simulation(str(path), real_time=False, output_dir=str(tmp_path / "out"))
    rows = [json.loads(l) for l in (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    for r in rows:
        lam = np.array(r["lambda_ij"])
        assert np.all(lam[~MASK] == 0.0)
        assert np.all(np.array(r["queue_access_ij"])[~MASK] == 0)
    assert load_config(path)["topology"]["unavailable_links"] == ["P0->SN3", "P1->SN1"]
