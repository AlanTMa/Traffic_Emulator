import numpy as np
import pytest
from src.controller.diagnostics import CERTIFIED_MESSAGE, compute_diagnostics, marginal_costs
from src.model.topology import Topology

def _symmetric():
    # 2 sources x 2 identical brokers: the optimum splits every source evenly
    topo = Topology(np.array([4.0, 4.0]), np.full((2, 2), 10.0), np.array([20.0, 20.0]), ["A", "B"], ["S1", "S2"])
    optimum = np.full((2, 2), 2.0)
    prices = marginal_costs(optimum, topo.mu_links, topo.mu_brokers)["C_j"]
    return topo, optimum, prices

def test_marginal_cost_formulas():
    mc = marginal_costs(np.array([[1.0]]), np.array([[3.0]]), np.array([5.0]))
    assert mc["D_ij"][0, 0] == pytest.approx(1 / 2)
    assert mc["C_ij"][0, 0] == pytest.approx(3 / 4)
    assert mc["D_j"][0] == pytest.approx(1 / 4)
    assert mc["C_j"][0] == pytest.approx(5 / 16)
    assert mc["M_ij"][0, 0] == pytest.approx(3 / 4 + 5 / 16)

def test_optimum_is_certified():
    topo, optimum, prices = _symmetric()
    d = compute_diagnostics(optimum, prices, topo)
    assert d["certified"], d["failed"]
    assert d["status"] == CERTIFIED_MESSAGE
    assert d["r_fixed_point"] < 1e-9
    assert d["active_spread_i"] == pytest.approx([0.0, 0.0], abs=1e-12)

def test_stale_prices_fail_price_consistency():
    topo, optimum, prices = _symmetric()
    d = compute_diagnostics(optimum, prices * 1.1, topo)
    assert not d["certified"]
    assert "r_price" in d["failed"]

def test_unbalanced_routing_fails_kkt_and_fixed_point():
    topo, _, _ = _symmetric()
    lam = np.array([[3.0, 1.0], [3.0, 1.0]])
    prices = marginal_costs(lam, topo.mu_links, topo.mu_brokers)["C_j"]  # price-consistent
    d = compute_diagnostics(lam, prices, topo)
    assert d["r_price"] < 1e-12
    assert not d["certified"]
    assert {"r_fixed_point", "r_active_stationarity", "r_kkt_complementarity"} <= set(d["failed"])
    assert np.all(d["active_spread_i"] > 0.1)

def test_cheaper_unused_route_fails_inactive_complementarity():
    topo, _, _ = _symmetric()
    lam = np.array([[4.0, 0.0], [4.0, 0.0]])      # everyone on the busier broker
    prices = marginal_costs(lam, topo.mu_links, topo.mu_brokers)["C_j"]
    d = compute_diagnostics(lam, prices, topo)
    assert d["r_inactive_complementarity"] > 0.1
    assert "r_inactive_complementarity" in d["failed"]

def test_conservation_violation_is_reported():
    topo, _, _ = _symmetric()
    d = compute_diagnostics(np.array([[3.0, 2.0], [2.0, 2.0]]), np.zeros(2), topo)   # source A sends 5, not 4
    assert d["r_conservation"] == pytest.approx(1.0)
    assert "r_conservation" in d["failed"]

def test_access_capacity_violation_is_reported():
    topo, _, _ = _symmetric()
    d = compute_diagnostics(np.array([[10.5, 0.0], [2.0, 2.0]]), np.zeros(2), topo)  # link A->S1 has mu=10
    assert d["access_margin"] == pytest.approx(0.5)
    assert "r_access_capacity" in d["failed"]
    assert d["service_margin"] < 0   # brokers still have headroom
