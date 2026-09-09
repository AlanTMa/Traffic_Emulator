import numpy as np
import pytest
from src.model.topology import Topology
from src.simulation.state import SystemState
from src.controller.synchronous import iteration_step
from src.controller.feasibility import transportation_feasibility
from src.model.marginal_costs import mm1_marginal_cost_vectorized

def test_reproduce_5x3_baseline():
    # 5 Producers, 3 Brokers
    sources = [f"P{i}" for i in range(5)]
    brokers = [f"SN{i+1}" for i in range(3)]

    # Exact Notebook Values
    lambdas_total = np.array([0.156, 0.098, 30.059, 30.010, 0.107])

    # MU_links (P x B)
    mu_links = np.array([
        [47.491, 54.640, 43.120],
        [41.162, 52.022, 40.412],
        [56.649, 43.636, 46.085],
        [48.639, 52.237, 45.843],
        [49.121, 43.993, 51.848]
    ])

    # MU_brokers
    mu_brokers = np.array([160.754, 106.505, 196.563])

    topo = Topology(lambdas_total, mu_links, mu_brokers, sources, brokers)

    # Initialization: Feasible L0
    L0 = transportation_feasibility(lambdas_total, mu_links, mu_brokers)

    # Initial prices from L0
    loads = L0.sum(axis=0)
    p0 = mu_brokers / (mu_brokers - loads)**2

    state = SystemState(lambda_ij=L0, prices=p0)

    # Algorithm Parameters
    eta = 0.25
    gamma = 0.50
    tol = 1e-10
    max_iter = 4000

    # Run the synchronous loop until convergence
    for i in range(max_iter):
        state, s_t, rel_change = iteration_step(state, topo, eta, gamma)

        if rel_change < tol:
            break


    # Verification against notebook baseline
    # F_dist approx 2.015747
    # Using the a-priori objective formula: sum (L / (mu-L)) + sum (Lambda / (mu-Lambda))
    l_mat = state.lambda_ij
    loads_final = l_mat.sum(axis=0)

    obj = np.sum(l_mat / (mu_links - l_mat)) + np.sum(loads_final / (mu_brokers - loads_final))

    print(f"\nFinal Objective: {obj:.6f}")
    assert obj == pytest.approx(2.015766, abs=1e-5)

    # Check Broker Utilizations
    utils = loads_final / mu_brokers
    print(f"Broker Utils: {utils}")
    # SN1: 0.1581, SN2: 0.1386, SN3: 0.1030
    assert utils[0] == pytest.approx(0.1581, abs=1e-3)
    assert utils[1] == pytest.approx(0.1386, abs=1e-3)
    assert utils[2] == pytest.approx(0.1030, abs=1e-3)
