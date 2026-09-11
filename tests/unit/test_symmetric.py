"""
Symmetric N x M instances against the closed form (docs/symmetric_case.md).
Algorithm 1 runs with eta=0.01, gamma=0.02; the paper's 5x3 values cycle, which is
asserted as a counterexample.
"""
import numpy as np
import pytest

from src.controller.best_response import best_response_mm1
from src.controller.central import solve_central, system_objective
from src.controller.diagnostics import compute_diagnostics
from src.controller.feasibility import transportation_feasibility
from src.controller.synchronous import algorithm1_initial_state, iteration_step, run_algorithm1
from src.model.symmetric import symmetric_solution, symmetric_topology

# (N, M, lam, mu_access, mu_server): light to heavy broker load (rho_s up to 0.9)
CASES = [
    (1, 1, 2.0, 5.0, 4.0),
    (2, 3, 6.0, 5.0, 10.0),
    (5, 3, 9.0, 10.0, 20.0),     # rho_s = 0.75
    (4, 4, 3.0, 2.0, 6.0),
    (10, 2, 1.8, 3.0, 10.0),     # rho_s = 0.9
    (3, 7, 14.0, 4.0, 12.0),     # rho_a = 0.5
]
IDS = [f"{n}x{m}" for n, m, *_ in CASES]
CONSERVATIVE = dict(eta=0.01, gamma=0.02)   # notebook load-sweep step sizes

def unbalanced_start(n, m, lam, mu_a, mu_s):
    """As much of every source's rate on broker 1 as its capacity (and the links) allow."""
    first = min(0.99 * mu_a, 0.99 * mu_s / n, lam)
    return np.tile(np.r_[first, np.full(m - 1, (lam - first) / (m - 1))], (n, 1))

@pytest.mark.parametrize("n, m, lam, mu_a, mu_s", CASES, ids=IDS)
def test_closed_form_values(n, m, lam, mu_a, mu_s):
    s = symmetric_solution(n, m, lam, mu_a, mu_s)
    x, load = lam / m, n * lam / m
    assert s["rho_server"] == pytest.approx(n * lam / (m * mu_s))
    assert s["price"] == pytest.approx(mu_s / (mu_s - load) ** 2)
    assert s["access_marginal_cost"] == pytest.approx(mu_a / (mu_a - x) ** 2)
    assert s["alpha"] == pytest.approx(s["access_marginal_cost"] + s["price"])
    topo = symmetric_topology(n, m, lam, mu_a, mu_s)
    assert s["objective"] == pytest.approx(system_objective(s["lambda_ij"], topo.mu_links, topo.mu_brokers))

@pytest.mark.parametrize("n, m, lam, mu_a, mu_s", CASES, ids=IDS)
def test_centralized_solver_finds_equal_split(n, m, lam, mu_a, mu_s):
    topo = symmetric_topology(n, m, lam, mu_a, mu_s)
    s = symmetric_solution(n, m, lam, mu_a, mu_s)
    L, info = solve_central(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    assert L == pytest.approx(s["lambda_ij"], abs=1e-6)
    assert info["objective"] == pytest.approx(s["objective"], rel=1e-10)

@pytest.mark.parametrize("n, m, lam, mu_a, mu_s", CASES, ids=IDS)
def test_equal_split_fixed_point(n, m, lam, mu_a, mu_s):
    s = symmetric_solution(n, m, lam, mu_a, mu_s)
    br = best_response_mm1(np.full(m, mu_a), np.full(m, s["price"]), lam)
    assert br == pytest.approx(np.full(m, lam / m), abs=1e-10)
    topo = symmetric_topology(n, m, lam, mu_a, mu_s)
    d = compute_diagnostics(s["lambda_ij"], np.full(m, s["price"]), topo)
    assert d["certified"], d["failed"]
    assert d["M_ij"] == pytest.approx(np.full((n, m), s["alpha"]), rel=1e-12)   # identical on every route

@pytest.mark.parametrize("n, m, lam, mu_a, mu_s", CASES, ids=IDS)
def test_algorithm1_reaches_equal_split(n, m, lam, mu_a, mu_s):
    topo = symmetric_topology(n, m, lam, mu_a, mu_s)
    s = symmetric_solution(n, m, lam, mu_a, mu_s)
    state = run_algorithm1(topo, **CONSERVATIVE, tol=1e-10, max_iter=10000, require_certificate=True)
    assert state.lambda_ij == pytest.approx(s["lambda_ij"], abs=1e-7)
    assert state.prices == pytest.approx(np.full(m, s["price"]), rel=1e-7)
    d = compute_diagnostics(state.lambda_ij, state.prices, topo)
    assert d["certified"], d["failed"]
    assert d["broker_utilization"] == pytest.approx(np.full(m, s["rho_server"]), rel=1e-7)

@pytest.mark.parametrize("n, m, lam, mu_a, mu_s", [CASES[3], CASES[2]], ids=["4x4", "5x3"])
def test_algorithm1_from_unbalanced_start(n, m, lam, mu_a, mu_s):
    topo = symmetric_topology(n, m, lam, mu_a, mu_s)
    state = run_algorithm1(topo, **CONSERVATIVE, tol=1e-10, max_iter=10000,
                           initial_lambda=unbalanced_start(n, m, lam, mu_a, mu_s), require_certificate=True)
    assert state.lambda_ij == pytest.approx(symmetric_solution(n, m, lam, mu_a, mu_s)["lambda_ij"], abs=1e-7)

def test_paper_steps_cycle_on_5x3():
    """eta=0.25, gamma=0.5 on the symmetric 5x3: a feasible period-2 cycle, never certified."""
    case = CASES[2]
    topo = symmetric_topology(*case)
    state = algorithm1_initial_state(topo)
    history = []
    for _ in range(200):
        state, s_t, _ = iteration_step(state, topo, 0.25, 0.5)
        history.append(state.lambda_ij.copy())
        assert s_t == 1.0
        assert np.all(state.lambda_ij.sum(axis=0) < topo.mu_brokers - 1e-8)
        assert state.lambda_ij.sum(axis=1) == pytest.approx(topo.lambdas_total, abs=1e-9)
    tail = history[-20:]
    period_1 = max(np.abs(tail[k] - tail[k - 1]).max() for k in range(1, 20))
    period_2 = max(np.abs(tail[k] - tail[k - 2]).max() for k in range(2, 20))
    assert period_1 > 0.5 and period_2 < 1e-3
    loads = sorted([tuple(np.round(L.sum(axis=0), 1)) for L in tail[-2:]])
    assert loads == [(12.4, 16.0, 16.6), (16.6, 15.6, 12.9)]
    assert not compute_diagnostics(state.lambda_ij, state.prices, topo)["certified"]

@pytest.mark.parametrize("n, m, lam, mu_a, mu_s, which", [
    (2, 2, 10.0, 5.0, 100.0, "mu_access"),     # lam/M = 5 = mu_access
    (4, 2, 3.0, 10.0, 6.0, "mu_server"),       # N lam / M = 6 = mu_server
    (4, 2, 3.5, 10.0, 6.0, "mu_server"),       # beyond
])
def test_infeasible_rejected(n, m, lam, mu_a, mu_s, which):
    with pytest.raises(ValueError, match=which):
        symmetric_solution(n, m, lam, mu_a, mu_s)
    topo = symmetric_topology(n, m, lam, mu_a, mu_s)
    with pytest.raises(ValueError):
        transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
