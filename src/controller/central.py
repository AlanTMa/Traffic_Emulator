"""
Centralized reference solver for the joint routing problem (paper eq. 4).

Port of solve_central_reference() from the ANRG reference notebook
(WiOpt26JNSC_Extended.ipynb, cell 4): SLSQP identifies the active route set,
then the open-domain KKT equations on that active set are polished with
least squares.
"""
import numpy as np
from scipy.optimize import least_squares, minimize
from src.controller.feasibility import transportation_feasibility
from src.model.marginal_costs import mm1_marginal_cost_vectorized

def system_objective(lambda_ij: np.ndarray, mu_links: np.ndarray, mu_brokers: np.ndarray) -> float:
    """
    F = sum_ij lambda_ij / (mu_ij - lambda_ij) + sum_j Lambda_j / (mu_j - Lambda_j)
    (flow-weighted M/M/1 delay, paper eq. 4); +inf outside the open domain.
    """
    lambda_ij = np.asarray(lambda_ij, dtype=float)
    loads = lambda_ij.sum(axis=0)
    if np.any(lambda_ij < 0) or np.any(lambda_ij >= mu_links) or np.any(loads >= mu_brokers):
        return np.inf
    return float(np.sum(lambda_ij / (mu_links - lambda_ij)) + np.sum(loads / (mu_brokers - loads)))

def _objective_gradient(lambda_ij, mu_links, mu_brokers):
    loads = lambda_ij.sum(axis=0)
    return mu_links / (mu_links - lambda_ij) ** 2 + (mu_brokers / (mu_brokers - loads) ** 2)[None, :]

def solve_central(lambdas_total, mu_links, mu_brokers, margin: float = 1e-8, active_tol: float = 1e-7):
    """
    Solve min F subject to source conservation, lambda_ij in [0, mu_ij - margin]
    and Lambda_j <= mu_j - margin.

    Returns:
        (lambda_ij, info) where info has 'objective', 'slsqp_message' and
        'kkt_polish_residual' (None if the polish was not accepted).
    """
    lambdas_total = np.asarray(lambdas_total, dtype=float)
    mu_links = np.asarray(mu_links, dtype=float)
    mu_brokers = np.asarray(mu_brokers, dtype=float)
    m, n = mu_links.shape
    l0 = transportation_feasibility(lambdas_total, mu_links, mu_brokers, margin=margin)

    a_eq = np.zeros((m, m * n))
    for i in range(m):
        a_eq[i, i * n:(i + 1) * n] = 1.0
    a_col = np.zeros((n, m * n))
    for j in range(n):
        a_col[j, j::n] = 1.0

    result = minimize(
        lambda x: system_objective(x.reshape(m, n), mu_links, mu_brokers),
        l0.ravel(),
        jac=lambda x: _objective_gradient(x.reshape(m, n), mu_links, mu_brokers).ravel(),
        method="SLSQP",
        bounds=[(0.0, float(v - margin)) for v in mu_links.ravel()],
        constraints=[
            {"type": "eq", "fun": lambda x: a_eq @ x - lambdas_total, "jac": lambda x: a_eq},
            {"type": "ineq", "fun": lambda x: mu_brokers - margin - a_col @ x, "jac": lambda x: -a_col},
        ],
        options={"maxiter": 5000, "ftol": 1e-12, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"central solver failed: {result.message}")

    # SLSQP reliably identifies the active set but may stop with a small KKT
    # error; polish that active set by solving the open-domain KKT equations.
    l_slsqp = result.x.reshape(m, n)
    active = l_slsqp > active_tol
    edges = list(zip(*np.where(active)))
    marginal0 = mu_links / (mu_links - l_slsqp) ** 2 + mm1_marginal_cost_vectorized(l_slsqp.sum(axis=0), mu_brokers)[None, :]
    alpha0 = np.array([np.mean(marginal0[i, active[i]]) for i in range(m)])
    z0 = np.r_[np.array([l_slsqp[i, j] for i, j in edges]), alpha0]

    def kkt_equations(z):
        candidate = np.zeros((m, n))
        for value, (i, j) in zip(z[:len(edges)], edges):
            candidate[i, j] = value
        alpha = z[len(edges):]
        prices = mu_brokers / (mu_brokers - candidate.sum(axis=0)) ** 2
        stationarity = [mu_links[i, j] / (mu_links[i, j] - candidate[i, j]) ** 2 + prices[j] - alpha[i]
                        for i, j in edges]
        return np.r_[stationarity, candidate.sum(axis=1) - lambdas_total]

    lower = np.r_[np.zeros(len(edges)), np.full(m, -np.inf)]
    upper = np.r_[np.array([mu_links[i, j] - margin for i, j in edges]), np.full(m, np.inf)]
    polish = least_squares(kkt_equations, z0, bounds=(lower, upper),
                           xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=5000)
    l_polished = np.zeros((m, n))
    for value, (i, j) in zip(polish.x[:len(edges)], edges):
        l_polished[i, j] = value

    info = {"slsqp_message": result.message, "kkt_polish_residual": None}
    if (polish.success
            and np.max(np.abs(l_polished.sum(axis=1) - lambdas_total)) < 1e-8
            and np.all(l_polished.sum(axis=0) < mu_brokers)):
        solution = l_polished
        info["kkt_polish_residual"] = float(np.max(np.abs(kkt_equations(polish.x))))
    else:
        solution = l_slsqp
    info["objective"] = system_objective(solution, mu_links, mu_brokers)
    return solution, info
