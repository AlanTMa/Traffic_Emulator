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
