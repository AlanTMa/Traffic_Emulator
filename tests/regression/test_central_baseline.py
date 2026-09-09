import numpy as np
import pytest
from src.model.topology import Topology
from src.simulation.state import SystemState
from src.controller.synchronous import iteration_step
from src.controller.feasibility import transportation_feasibility
from src.model.marginal_costs import mm1_marginal_cost_vectorized
from src.controller.best_response import best_response_mm1
from scipy.optimize import minimize, Bounds, LinearConstraint, least_squares

def objective_flow_weighted_at(l_flat, lambdas_total, mu_links, mu_brokers):
    l_mat = np.asarray(l_flat, dtype=float).reshape(mu_links.shape)
    loads = l_mat.sum(axis=0)
    return float(np.sum(l_mat / (mu_links - l_mat)) + np.sum(loads / (mu_brokers - loads)))

def grad_flow_weighted_at(l_flat, lambdas_total, mu_links, mu_brokers):
    l_mat = np.asarray(l_flat, dtype=float).reshape(mu_links.shape)
    loads = l_mat.sum(axis=0)
    return (mu_links / (mu_links - l_mat)**2 + mu_brokers[None, :] / (mu_brokers - loads)[None, :]**2).ravel()

def solve_central_reference(lambdas_total, mu_links, mu_brokers, eps=1e-8):
    m, n = mu_links.shape
    l0 = transportation_feasibility(lambdas_total, mu_links, mu_brokers, margin=eps)

    a_eq = np.zeros((m, m * n))
    for i in range(m):
        a_eq[i, i * n : (i + 1) * n] = 1.0
    a_col = np.zeros((n, m * n))
    for j in range(n):
        a_col[j, j::n] = 1.0

    result = minimize(
        objective_flow_weighted_at,
        l0.ravel(),
        args=(lambdas_total, mu_links, mu_brokers),
        jac=grad_flow_weighted_at,
        method="SLSQP",
        bounds=[(0.0, float(v - eps)) for v in mu_links.ravel()],
        constraints=[
            {"type": "eq", "fun": lambda x: a_eq @ x - lambdas_total, "jac": lambda x: a_eq},
            {"type": "ineq", "fun": lambda x: mu_brokers - eps - a_col @ x, "jac": lambda x: -a_col},
        ],
        options={"maxiter": 5000, "ftol": 1e-12, "disp": False},
    )
    return result

def test_compare_central_objective():
    sources = [f"P{i}" for i in range(5)]
    brokers = [f"SN{i+1}" for i in range(3)]
    lambdas_total = np.array([0.156, 0.098, 30.059, 30.010, 0.107])
    mu_links = np.array([
        [47.491, 54.640, 43.120],
        [41.162, 52.022, 40.412],
        [56.649, 43.636, 46.085],
        [48.639, 52.237, 45.843],
        [49.121, 43.993, 51.848]
    ])
    mu_brokers = np.array([160.754, 106.505, 196.563])

    res = solve_central_reference(lambdas_total, mu_links, mu_brokers)
    obj = objective_flow_weighted_at(res.x, lambdas_total, mu_links, mu_brokers)
    print(f"\nCentral Objective: {obj:.6f}")
    # Compare with notebook F* = 2.015747
    assert obj == pytest.approx(2.015747, abs=1e-4)
