"""Initial routings: the max-headroom transportation LP and the notebook's random feasible routing."""
import numpy as np
from scipy.optimize import linprog

def transportation_feasibility(lambdas_total, mu_links, mu_brokers, margin=1e-8, route_mask=None):
    """
    The routing that maximizes the minimum headroom over links and brokers (LP).
    route_mask: True where a link exists (default mu_links > 0); missing links
    carry no flow.
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

    # sum_i lambda_ij + h <= mu_j and lambda_ij + h <= mu_ij, h = headroom
    ub_rows, ub_rhs = [], []
    for j in range(n):
        row = np.zeros(m * n + 1)
        row[j : m * n : n] = 1.0
        row[-1] = 1.0
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
    objective[-1] = -1.0    # maximize h

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
    # HiGHS tolerates ~1e-7 violations, more than the margin: check the routing itself
    link_headroom = float(np.min((mu_links - routing)[route_mask])) if route_mask.any() else np.inf
    broker_headroom = float(np.min(mu_brokers - routing.sum(axis=0)))
    if min(link_headroom, broker_headroom) < 0.5 * margin:
        raise ValueError(
            f"Transportation LP infeasible: no routing keeps headroom >= {margin:g} "
            f"(best found: links {link_headroom:.3g}, brokers {broker_headroom:.3g})")
    return routing

def random_feasible_routing(lambdas_total, mu_links, mu_brokers, seed: int, margin: float = 1e-8,
                            interior_weight: float = 0.30, route_mask=None) -> np.ndarray:
    """The notebook's random_feasible_initialization: a random vertex of the transportation
    polytope blended with the max-headroom routing, which keeps it strictly inside."""
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
