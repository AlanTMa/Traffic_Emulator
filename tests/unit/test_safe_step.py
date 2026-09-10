import numpy as np
import pytest
from src.controller.best_response import best_response_mm1
from src.controller.synchronous import compute_safe_step, iteration_step
from src.model.marginal_costs import mm1_marginal_cost_vectorized
from src.model.topology import Topology
from src.simulation.state import SystemState

def test_non_binding_example_from_paper_formula():
    # eta=0.25, headroom/Delta=0.5: s_j = min(1, 0.5/0.25) = 1, so the full
    # eta step is taken (the old headroom/Delta form gave s=0.5).
    lam = np.array([[0.0]])
    br = np.array([[1.0]])      # Delta = 1
    loads = np.array([0.0])
    mu = np.array([0.5])        # headroom 0.5 with eps=0
    assert compute_safe_step(lam, br, loads, mu, eta=0.25, delta_s=0.0) == pytest.approx(1.0)

def test_binding_single_broker():
    # headroom/Delta = 0.1 < eta: s = 0.1/0.25 = 0.4, effective step 0.1,
    # which lands exactly on mu - eps.
    lam = np.array([[0.0]])
    br = np.array([[1.0]])
    loads = np.array([0.0])
    mu = np.array([0.1 + 1e-8])
    s = compute_safe_step(lam, br, loads, mu, eta=0.25, delta_s=1e-8)
    assert s == pytest.approx(0.4)
    assert loads[0] + 0.25 * s * 1.0 == pytest.approx(mu[0] - 1e-8)

def test_min_over_brokers_ignores_decreasing_loads():
    lam = np.array([[1.0, 1.0, 2.0]])
    br = np.array([[2.0, 1.5, 0.5]])            # Delta = [+1, +0.5, -1.5]
    loads = lam.sum(axis=0)
    mu = np.array([1.2, 1.1, 2.0 + 1e-9])       # headroom [0.2, 0.1, ~0]
    eta = 0.5
    # s_0 = 0.2/(0.5*1) = 0.4, s_1 = 0.1/(0.5*0.5) = 0.4, broker 2 shrinks: s_2 = 1
    s_0 = min(1.0, (mu[0] - loads[0]) / (eta * 1.0))
    s_1 = min(1.0, (mu[1] - loads[1]) / (eta * 0.5))
    assert compute_safe_step(lam, br, loads, mu, eta=eta, delta_s=0.0) == pytest.approx(min(s_0, s_1))

def _binding_instance():
    # Source 0 wants to move all 10 units to broker 0 (its price is stale and
    # low); source 1 already fills broker 0 to within 0.5 of capacity and can
    # barely reach broker 1, so it stays put.
    lambdas = np.array([10.0, 10.0])
    mu_links = np.array([[100.0, 100.0], [100.0, 0.01]])
    mu_brokers = np.array([10.5, 1000.0])
    topo = Topology(lambdas, mu_links, mu_brokers, ["A", "B"], ["SN1", "SN2"])
    lam0 = np.array([[0.0, 10.0], [10.0, 0.0]])
    prices0 = np.array([0.0, 1e3])
    return topo, lam0, prices0

def test_iteration_step_binding_matches_algorithm_1():
    topo, lam0, prices0 = _binding_instance()
    eta, gamma, eps, delta_s = 0.25, 0.01, 1e-12, 1e-6
    state = SystemState(lambda_ij=lam0.copy(), prices=prices0.copy())
    state, s_t, _ = iteration_step(state, topo, eta, gamma, eps=eps, delta_s=delta_s)

    # Recompute Algorithm 1 independently
    loads = lam0.sum(axis=0)
    p_new = (1 - gamma) * prices0 + gamma * mm1_marginal_cost_vectorized(loads, topo.mu_brokers, eps)
    lam_br = np.vstack([best_response_mm1(topo.mu_links[i], p_new, topo.lambdas_total[i]) for i in range(2)])
    delta = (lam_br - lam0).sum(axis=0)
    s_j = [min(1.0, (topo.mu_brokers[j] - delta_s - loads[j]) / (eta * delta[j])) if delta[j] > 0 else 1.0
           for j in range(2)]
    s_expected = min(s_j)

    assert s_expected < 1.0, "test instance must make the safe step bind"
    assert s_t == pytest.approx(s_expected, rel=1e-12)
    expected = (1 - eta * s_t) * lam0 + eta * s_t * lam_br
    assert state.lambda_ij == pytest.approx(expected, rel=1e-12, abs=1e-12)

    new_loads = state.lambda_ij.sum(axis=0)
    # Broker capacity: the binding broker lands on mu - delta_s (not eps), never beyond
    assert np.all(new_loads <= topo.mu_brokers - delta_s + 1e-9)
    assert new_loads[0] == pytest.approx(topo.mu_brokers[0] - delta_s, abs=1e-9)
    # Source conservation
    assert state.lambda_ij.sum(axis=1) == pytest.approx(topo.lambdas_total, abs=1e-9)
    assert np.all(state.lambda_ij >= 0)

def test_binding_step_stays_feasible_over_iterations():
    topo, lam0, prices0 = _binding_instance()
    state = SystemState(lambda_ij=lam0.copy(), prices=prices0.copy())
    for _ in range(200):
        state, s_t, _ = iteration_step(state, topo, 0.25, 0.01, delta_s=1e-8)
        loads = state.lambda_ij.sum(axis=0)
        assert np.all(loads < topo.mu_brokers)
        assert np.all(state.lambda_ij < topo.mu_links)
        assert state.lambda_ij.sum(axis=1) == pytest.approx(topo.lambdas_total, abs=1e-9)

def test_best_response_failure_raises_by_default_and_is_recorded_when_held(monkeypatch):
    import src.controller.synchronous as sync
    topo, lam0, prices0 = _binding_instance()
    real = sync.best_response_mm1
    def flaky(mu_row, p, lam_i, *args, **kwargs):
        if mu_row[1] == 100.0:          # source A's row
            raise RuntimeError("solver did not converge")
        return real(mu_row, p, lam_i, *args, **kwargs)
    monkeypatch.setattr(sync, "best_response_mm1", flaky)

    with pytest.raises(RuntimeError):
        iteration_step(SystemState(lam0.copy(), prices0.copy()), topo, 0.25, 0.01)

    state, _, _ = iteration_step(SystemState(lam0.copy(), prices0.copy()), topo, 0.25, 0.01,
                                 on_best_response_failure="hold")
    assert state.br_failures == [0]
    assert state.lambda_ij[0] == pytest.approx(lam0[0])      # held source keeps its split
    assert state.lambda_ij.sum(axis=1) == pytest.approx(topo.lambdas_total)
