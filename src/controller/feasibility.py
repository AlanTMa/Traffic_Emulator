"""
Feasibility and initialization tools for routing.
"""
import numpy as np
from scipy.optimize import linprog

def transportation_feasibility(lambdas_total, mu_links, mu_brokers, margin=1e-8):
    """
    Return an LP-certified feasible routing matrix.
    """
    m, n = mu_links.shape

    # Equality constraints: sum_j lambda_ij = lambda_i
    a_eq = np.zeros((m, m * n))
    for i in range(m):
        a_eq[i, i * n : (i + 1) * n] = 1.0

    # Inequality constraints:
    # 1. sum_i lambda_ij <= mu_j
    # 2. lambda_ij <= mu_ij
    ub_rows, ub_rhs = [], []
    for j in range(n):
        row = np.zeros(m * n + 1)
        row[j : m * n : n] = 1.0
        row[-1] = 1.0 # Slack variable for max-headroom
        ub_rows.append(row)
        ub_rhs.append(mu_brokers[j])

    for i in range(m):
        for j in range(n):
            row = np.zeros(m * n + 1)
            row[i * n + j] = 1.0
            row[-1] = 1.0
            ub_rows.append(row)
            ub_rhs.append(mu_links[i, j])

    bounds = [(0.0, None)] * (m * n) + [(margin, None)]
    objective = np.zeros(m * n + 1)
    objective[-1] = -1.0 # Maximize minimum headroom

    result = linprog(
        objective,
        A_ub=np.asarray(ub_rows),
        b_ub=np.asarray(ub_rhs),
        A_eq=np.column_stack([a_eq, np.zeros(m)]),
        b_eq=lambdas_total,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise ValueError(f"Transportation LP infeasible: {result.message}")

    routing = np.maximum(result.x[:-1].reshape(m, n), 0.0)
    # HiGHS accepts constraint violations up to its feasibility tolerance
    # (~1e-7), which exceeds the default margin: at an exactly critical
    # instance it returns zero headroom (lambda_ij = mu_ij). Check the routing
    # itself and reject anything short of the requested margin.
    link_headroom = float(np.min(mu_links - routing))
    broker_headroom = float(np.min(mu_brokers - routing.sum(axis=0)))
    if min(link_headroom, broker_headroom) < 0.5 * margin:
        raise ValueError(
            f"Transportation LP infeasible: no routing keeps headroom >= {margin:g} "
            f"(best found: links {link_headroom:.3g}, brokers {broker_headroom:.3g})")
    return routing
