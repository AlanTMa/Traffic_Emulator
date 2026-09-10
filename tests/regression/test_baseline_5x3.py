"""
5x3 regression gate against the ANRG reference notebook.

The expected values in data/baseline_5x3.json were produced by running the
notebook's own functions on its exact seed-42 instance (see "provenance" in
the file). The canonical objective is F* = 2.0157473649138 (notebook-printed).
The value 2.015766 that earlier versions of this test asserted comes from
running the same algorithm on the instance rounded to 3 decimals (e.g.
lambda_P2 = 30.059 instead of 30.05859375); it is not a paper value.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.controller.central import solve_central, system_objective
from src.controller.diagnostics import CERTIFIED_MESSAGE, DEFAULT_TOLERANCES, compute_diagnostics, marginal_costs
from src.controller.synchronous import run_algorithm1
from src.model.topology import Topology

BASELINE = json.loads((Path(__file__).parent / "data" / "baseline_5x3.json").read_text())
PARAMS = BASELINE["parameters"]
EXPECTED = BASELINE["distributed"]

@pytest.fixture(scope="module")
def topology():
    return Topology(np.array(PARAMS["lambdas_total"]), np.array(PARAMS["mu_links"]),
                    np.array(PARAMS["mu_brokers"]), [f"P{i}" for i in range(5)], ["SN1", "SN2", "SN3"])

@pytest.fixture(scope="module")
def distributed(topology):
    return run_algorithm1(topology, eta=PARAMS["eta"], gamma=PARAMS["gamma"], tol=PARAMS["tol"],
                          max_iter=PARAMS["max_iter"], delta_s=PARAMS["delta_s"])

@pytest.fixture(scope="module")
def central(topology):
    return solve_central(topology.lambdas_total, topology.mu_links, topology.mu_brokers, margin=PARAMS["delta_s"])

def test_iterations_match_notebook(distributed):
    assert distributed.iteration == EXPECTED["iterations"]

def test_routing_matrix(distributed):
    assert distributed.lambda_ij == pytest.approx(np.array(EXPECTED["lambda_ij"]), abs=1e-9)

def test_source_conservation(distributed, topology):
    assert np.max(np.abs(distributed.lambda_ij.sum(axis=1) - topology.lambdas_total)) <= 1e-8

def test_broker_loads_and_utilization(distributed, topology):
    loads = distributed.lambda_ij.sum(axis=0)
    assert loads == pytest.approx(np.array(EXPECTED["broker_loads"]), abs=1e-9)
    assert loads / topology.mu_brokers == pytest.approx(np.array(EXPECTED["broker_utilization"]), abs=1e-11)
    assert np.all(loads <= topology.mu_brokers - PARAMS["delta_s"])

def test_advertised_prices(distributed):
    assert distributed.prices == pytest.approx(np.array(EXPECTED["prices"]), rel=1e-9)

def test_active_route_set(distributed):
    active = (distributed.lambda_ij > 1e-7).astype(int)
    assert active.tolist() == EXPECTED["active_routes"]
    # P0, P1, P4 use only SN3; P2, P3 use all three brokers
    assert active.tolist() == [[0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1]]

def test_objective(distributed, topology):
    obj = system_objective(distributed.lambda_ij, topology.mu_links, topology.mu_brokers)
    assert obj == pytest.approx(EXPECTED["objective"], abs=1e-12)
    assert obj == pytest.approx(2.0157473649138, abs=1e-12)

def test_final_route_and_price_residuals(distributed):
    assert distributed.route_rel < PARAMS["tol"]
    assert distributed.price_rel < PARAMS["tol"]

def test_centralized_matches_notebook(central):
    L_c, info = central
    assert info["kkt_polish_residual"] is not None
    assert info["objective"] == pytest.approx(BASELINE["central"]["objective"], abs=1e-12)
    assert L_c == pytest.approx(np.array(BASELINE["central"]["lambda_ij"]), abs=1e-9)

def test_distributed_matches_centralized(distributed, central, topology):
    _, info = central
    F_d = system_objective(distributed.lambda_ij, topology.mu_links, topology.mu_brokers)
    gap = (F_d - info["objective"]) / abs(info["objective"])
    assert abs(gap) <= BASELINE["residual_tolerances"]["relative_objective_gap"]
    assert distributed.lambda_ij == pytest.approx(central[0], abs=1e-7)

# --- Proposition 1 diagnostics on the baseline ---

CERT_KEYS = ["r_conservation", "r_access_capacity", "r_service_capacity", "r_price", "r_fixed_point",
             "r_kkt_complementarity", "r_active_stationarity", "r_inactive_complementarity"]

def test_distributed_endpoint_is_certified(distributed, topology):
    d = compute_diagnostics(distributed.lambda_ij, distributed.prices, topology)
    assert d["certified"], d["failed"]
    assert d["status"] == CERTIFIED_MESSAGE

def test_distributed_residuals_match_notebook_table(distributed, topology):
    d = compute_diagnostics(distributed.lambda_ij, distributed.prices, topology)
    notebook = BASELINE["residuals"]["distributed"]
    for key in CERT_KEYS:
        assert d[key] <= DEFAULT_TOLERANCES[key], key
        # Same state, same definitions: residuals agree with the notebook's
        # audit to within a factor 2 (or both are at the rounding floor).
        assert d[key] <= max(2 * notebook[key], 1e-15), key

def test_centralized_solution_is_certified(central, topology):
    L_c, _ = central
    model_prices = marginal_costs(L_c, topology.mu_links, topology.mu_brokers)["C_j"]
    d = compute_diagnostics(L_c, model_prices, topology)
    assert d["certified"], d["failed"]

def test_kkt_multipliers(distributed, topology):
    d = compute_diagnostics(distributed.lambda_ij, distributed.prices, topology)
    # Wardrop/KKT equalization: sources splitting across all brokers see equal
    # marginal cost on every used route
    assert d["active_spread_i"][2] < 1e-10 and d["active_spread_i"][3] < 1e-10
    assert d["alpha_i"][2] == pytest.approx(0.0417122, abs=1e-7)
    assert d["alpha_i"][3] == pytest.approx(0.0417734, abs=1e-7)

def test_certificate_rejects_non_optimal_states(topology):
    # The LP initializer is price-consistent but not a best-response fixed point
    initial = run_algorithm1(topology, max_iter=0, delta_s=PARAMS["delta_s"])
    d0 = compute_diagnostics(initial.lambda_ij, initial.prices, topology)
    assert d0["r_price"] < 1e-12
    assert not d0["certified"] and "r_fixed_point" in d0["failed"]
    # Mid-trajectory the damped prices lag the loads as well
    early = run_algorithm1(topology, max_iter=5, delta_s=PARAMS["delta_s"])
    d5 = compute_diagnostics(early.lambda_ij, early.prices, topology)
    assert not d5["certified"] and "r_price" in d5["failed"]

def test_step5_stopping_rule_certifies_at_high_load():
    # At 95% of the largest feasible multiplier, stopping on route/price change
    # alone (notebook rule) leaves KKT complementarity above its absolute
    # tolerance; the paper's Step 5 rule keeps iterating until certified.
    high = Topology(4.626 * np.array(PARAMS["lambdas_total"]), np.array(PARAMS["mu_links"]),
                    np.array(PARAMS["mu_brokers"]), [f"P{i}" for i in range(5)], ["SN1", "SN2", "SN3"])
    notebook_rule = run_algorithm1(high, tol=1e-10)
    step5 = run_algorithm1(high, tol=1e-10, require_certificate=True)
    assert not compute_diagnostics(notebook_rule.lambda_ij, notebook_rule.prices, high)["certified"]
    assert compute_diagnostics(step5.lambda_ij, step5.prices, high)["certified"]
    assert step5.iteration > notebook_rule.iteration
