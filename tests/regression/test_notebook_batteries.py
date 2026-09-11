"""
The notebook's verification batteries (cells 4, 6, 8) with its tolerances: best
response against an SLSQP optimizer, stressed best responses, finite-difference
derivatives, the load sweep and multistart (slow). The last two use the notebook's
safe-step bound: with the paper's, multistart seed 0 at load 0.85 does not converge.
"""
import numpy as np
import pytest
from scipy.optimize import minimize

from src.controller.best_response import best_response_mm1
from src.controller.central import solve_central, system_objective
from src.controller.diagnostics import compute_diagnostics, marginal_costs
from src.controller.feasibility import random_feasible_routing
from src.controller.synchronous import run_algorithm1
from src.model.topology import Topology
from src.runtime.experiments import canonical_5x3

def optimizer_best_response(mu, p, demand, eps=1e-9):
    """Independent SLSQP solution of the source problem (notebook's validation optimizer)."""
    objective = lambda x: float(np.sum(x / (mu - x) + p * x))
    result = minimize(objective, demand * mu / mu.sum(), jac=lambda x: mu / (mu - x) ** 2 + p, method="SLSQP",
                      bounds=[(0.0, float(v - eps)) for v in mu],
                      constraints={"type": "eq", "fun": lambda x: x.sum() - demand, "jac": lambda x: np.ones_like(x)},
                      options={"ftol": 1e-13, "maxiter": 3000})
    assert result.success, result.message
    return result.x, objective

# --- Source best response ---

@pytest.mark.parametrize("demand, active", [(0.10, 1), (3.00, 2), (6.00, 3)])
def test_validate_source_best_response(demand, active):
    # Notebook validate_source_best_response: one, partial and all routes active
    mu, p = np.array([3.0, 4.0, 5.0]), np.array([1.0, 0.20, 0.0])
    closed = best_response_mm1(mu, p, demand)
    independent, _ = optimizer_best_response(mu, p, demand)
    assert int(np.sum(closed > 1e-7)) == active
    assert abs(closed.sum() - demand) <= 1e-12 * max(1.0, demand)   # the solver's conservation tolerance
    assert np.max(np.abs(closed - independent)) <= 5e-6

def test_randomized_best_response_regression():
    rng = np.random.default_rng(20260824)
    active_counts = set()
    worst = dict(conservation=0.0, stationarity=0.0, inactive=0.0, allocation=0.0, objective_gap=0.0)
    for _ in range(250):
        n = int(rng.integers(2, 7))
        mu = rng.uniform(0.5, 10.0, n)
        p = rng.uniform(0.0, 2.0, n)
        demand = float(rng.uniform(1e-4, 0.90 * mu.sum()))
        closed = best_response_mm1(mu, p, demand)
        independent, objective = optimizer_best_response(mu, p, demand)
        marginal = mu / (mu - closed) ** 2 + p
        active = closed > 1e-8
        alpha = float(np.mean(marginal[active]))
        active_counts.add(int(active.sum()))
        worst["conservation"] = max(worst["conservation"], abs(closed.sum() - demand))
        worst["stationarity"] = max(worst["stationarity"], float(np.max(np.abs(marginal[active] - alpha))))
        if np.any(~active):
            worst["inactive"] = max(worst["inactive"], float(np.max(np.maximum(alpha - marginal[~active], 0.0))))
        worst["allocation"] = max(worst["allocation"], float(np.max(np.abs(closed - independent))))
        worst["objective_gap"] = max(worst["objective_gap"], abs(objective(closed) - objective(independent))
                                     / max(1.0, abs(objective(independent))))
    assert active_counts == set(range(1, 7))
    # Notebook tolerances
    assert worst["conservation"] <= 1e-9
    assert worst["stationarity"] <= 1e-10
    assert worst["inactive"] <= 1e-10
    assert worst["allocation"] <= 5e-6
    assert worst["objective_gap"] <= 1e-9

def test_stressed_best_response_regression():
    rng = np.random.default_rng(20260825)
    cases = [
        (np.array([1.0, 1.0]), np.array([100.0, 100.0]), 1.8),
        (np.array([0.7, 1.3, 5.0]), np.array([1e6, 1e6 + 0.1, 1e6 + 200.0]), 0.95 * 7.0),
        (np.array([0.01, 1.0, 100.0]), np.array([1e8, 1e8 + 10.0, 1e8 + 1e4]), 0.9999 * 101.01),
        (np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1e8, 1e8, 1e8]), 3.8),
    ]
    fractions = np.array([1e-8, 1e-4, 0.1, 0.5, 0.9, 0.99, 0.9999])
    for _ in range(400):
        n = int(rng.integers(2, 9))
        mu = 10.0 ** rng.uniform(-2.0, 2.0, n)
        price_scale = 10.0 ** rng.uniform(-2.0, 8.0)
        p = price_scale + rng.uniform(0.0, max(1.0, 0.2 * price_scale), n)
        cases.append((mu, p, float(rng.choice(fractions) * mu.sum())))

    for mu, p, demand in cases:
        x = best_response_mm1(mu, p, demand)
        marginal = mu / (mu - x) ** 2 + p
        active = x > max(1e-14, 1e-10 * demand)
        if not np.any(active):
            active[np.argmax(x)] = True
        alpha = float(np.median(marginal[active]))
        scale = max(1.0, abs(alpha))
        # Notebook tolerances
        assert abs(float(x.sum()) - demand) / max(1.0, abs(demand)) <= 1.1e-12
        assert np.max(np.abs(marginal[active] - alpha)) / scale <= 5e-10
        if np.any(~active):
            assert np.max(np.maximum(alpha - marginal[~active], 0.0)) / scale <= 5e-10
        assert np.min(x) >= 0.0
        assert np.min(mu - x) > 0.0

# --- Derivatives ---

def test_finite_difference_derivatives():
    rng = np.random.default_rng(20260824)
    worst = 0.0
    for _ in range(1000):
        mu = float(rng.uniform(0.5, 20.0))
        x = float(rng.uniform(0.0, 0.85 * mu))
        price = float(rng.uniform(0.0, 3.0))
        h = 1e-6 * max(1.0, mu)
        f = lambda z: z / (mu - z) + price * z
        worst = max(worst, abs((f(x + h) - f(x - h)) / (2 * h) - (mu / (mu - x) ** 2 + price)))
    assert worst <= 1e-7

    topo = canonical_5x3()
    L = np.tile(topo.lambdas_total[:, None] / topo.n_brokers, (1, topo.n_brokers))
    analytic = marginal_costs(L, topo.mu_links, topo.mu_brokers)["M_ij"]   # dF/dlambda_ij = C_ij + C_j
    h = 1e-6
    numeric = np.zeros_like(L)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            plus, minus = L.copy(), L.copy()
            plus[i, j] += h
            minus[i, j] -= h
            numeric[i, j] = (system_objective(plus, topo.mu_links, topo.mu_brokers)
                             - system_objective(minus, topo.mu_links, topo.mu_brokers)) / (2 * h)
    assert np.max(np.abs(analytic - numeric)) <= 1e-7

# --- Load sweep and multistart (Algorithm 1 at eta=0.01, gamma=0.02) ---

TARGETS = (0.20, 0.55, 0.85)
CONSERVATIVE = dict(eta=0.01, gamma=0.02, tol=1e-10, max_iter=15000, delta_s=1e-8, safe_step_variant="notebook")

def load_case(target):
    """The 5x3 instance with broker capacities scaled so sum(lambda)/sum(mu) = target."""
    base = canonical_5x3()
    mu_case = base.mu_brokers / base.mu_brokers.sum() * (base.lambdas_total.sum() / target)
    return Topology(base.lambdas_total, base.mu_links, mu_case, base.sources, base.brokers)

def assert_certificate(topo, lam, prices, central_objective):
    d = compute_diagnostics(lam, prices, topo)
    assert d["certified"], d["failed"]
    gap = (system_objective(lam, topo.mu_links, topo.mu_brokers) - central_objective) / abs(central_objective)
    assert abs(gap) <= 1e-7

@pytest.mark.slow
@pytest.mark.parametrize("target", TARGETS)
def test_load_sweep(target):
    topo = load_case(target)
    L_c, info = solve_central(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    model_prices = marginal_costs(L_c, topo.mu_links, topo.mu_brokers)["C_j"]
    assert compute_diagnostics(L_c, model_prices, topo)["certified"]
    state = run_algorithm1(topo, **CONSERVATIVE)
    assert_certificate(topo, state.lambda_ij, state.prices, info["objective"])
    assert np.max(state.lambda_ij.sum(axis=0) / topo.mu_brokers) == pytest.approx(
        np.max(L_c.sum(axis=0) / topo.mu_brokers), abs=1e-6)

@pytest.mark.slow
@pytest.mark.parametrize("target", TARGETS)
def test_multistart_regression(target):
    topo = load_case(target)
    _, info = solve_central(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    for seed in range(5):
        start = random_feasible_routing(topo.lambdas_total, topo.mu_links, topo.mu_brokers, seed=seed)
        state = run_algorithm1(topo, **CONSERVATIVE, initial_lambda=start)
        assert_certificate(topo, state.lambda_ij, state.prices, info["objective"])

@pytest.mark.slow
def test_paper_safe_step_multistart_0_85():
    """With the paper's bound, multistart seed 0 at load 0.85 stops short of the optimum."""
    topo = load_case(0.85)
    _, info = solve_central(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    start = random_feasible_routing(topo.lambdas_total, topo.mu_links, topo.mu_brokers, seed=0)
    paper = run_algorithm1(topo, **{**CONSERVATIVE, "max_iter": 3000, "safe_step_variant": "paper"},
                           initial_lambda=start)
    gap = (system_objective(paper.lambda_ij, topo.mu_links, topo.mu_brokers) - info["objective"]) / info["objective"]
    assert gap > 1e-3
    assert not compute_diagnostics(paper.lambda_ij, paper.prices, topo)["certified"]
