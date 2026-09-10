"""
Feasibility and initialization tools for routing.
"""
import numpy as np
from scipy.optimize import linprog

def transportation_feasibility(lambdas_total, mu_links, mu_brokers, margin=1e-8, route_mask=None):
    """
    Return an LP-certified feasible routing matrix: the routing that
    maximizes the minimum headroom over all access links and brokers.

    route_mask: (N, M) bool, True where the access link exists (reference
    notebook's route_mask). Defaults to mu_links > 0, which is all True for a
    fully connected topology; flow on unavailable routes is fixed at 0, so the
    certificate stays valid for sparse bipartite graphs where aggregate
    capacity checks are not enough.
    """
    lambdas_total = np.asarray(lambdas_total, dtype=float)
    mu_links = np.asarray(mu_links, dtype=float)
    mu_brokers = np.asarray(mu_brokers, dtype=float)
    m, n = mu_links.shape
    route_mask = (mu_links > 0) if route_mask is None else np.asarray(route_mask, dtype=bool)
    if route_mask.shape != (m, n):
        raise ValueError("route_mask must have the same shape as mu_links")

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
            if route_mask[i, j]:
                row = np.zeros(m * n + 1)
                row[i * n + j] = 1.0
                row[-1] = 1.0
                ub_rows.append(row)
                ub_rhs.append(mu_links[i, j])

    bounds = [(0.0, None) if route_mask[i, j] else (0.0, 0.0)
              for i in range(m) for j in range(n)] + [(margin, None)]
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

    routing = np.where(route_mask, np.maximum(result.x[:-1].reshape(m, n), 0.0), 0.0)
    # HiGHS accepts constraint violations up to its feasibility tolerance
    # (~1e-7), which exceeds the default margin: at an exactly critical
    # instance it returns zero headroom (lambda_ij = mu_ij). Check the routing
    # itself and reject anything short of the requested margin.
    link_headroom = float(np.min((mu_links - routing)[route_mask])) if route_mask.any() else np.inf
    broker_headroom = float(np.min(mu_brokers - routing.sum(axis=0)))
    if min(link_headroom, broker_headroom) < 0.5 * margin:
        raise ValueError(
            f"Transportation LP infeasible: no routing keeps headroom >= {margin:g} "
            f"(best found: links {link_headroom:.3g}, brokers {broker_headroom:.3g})")
    return routing

def random_feasible_routing(lambdas_total, mu_links, mu_brokers, seed: int, margin: float = 1e-8,
                            interior_weight: float = 0.30, route_mask=None) -> np.ndarray:
    """
    Reproducible, diverse feasible routing (reference notebook's
    random_feasible_initialization): a random vertex of the transportation
    polytope (capacities reduced by margin) blended with the max-headroom
    routing, which keeps the result strictly inside the open domain.
    """
    lambdas_total = np.asarray(lambdas_total, dtype=float)
    mu_links = np.asarray(mu_links, dtype=float)
    mu_brokers = np.asarray(mu_brokers, dtype=float)
    m, n = mu_links.shape
    route_mask = (mu_links > 0) if route_mask is None else np.asarray(route_mask, dtype=bool)
    rng = np.random.default_rng(seed)
    a_eq = np.zeros((m, m * n))
    a_ub = np.zeros((n, m * n))
    for i in range(m):
        a_eq[i, i * n:(i + 1) * n] = 1.0
    for j in range(n):
        a_ub[j, j::n] = 1.0
    vertex = linprog(rng.normal(size=m * n), A_ub=a_ub, b_ub=mu_brokers - margin, A_eq=a_eq, b_eq=lambdas_total,
                     bounds=[(0.0, float(v - margin)) if ok else (0.0, 0.0)
                             for v, ok in zip(mu_links.ravel(), route_mask.ravel())], method="highs")
    if not vertex.success:
        raise ValueError(f"random feasibility LP failed: {vertex.message}")
    safe = transportation_feasibility(lambdas_total, mu_links, mu_brokers, margin=margin, route_mask=route_mask)
    vertex_routing = np.where(route_mask, np.maximum(vertex.x.reshape(m, n), 0.0), 0.0)
    return (1.0 - interior_weight) * vertex_routing + interior_weight * safe
