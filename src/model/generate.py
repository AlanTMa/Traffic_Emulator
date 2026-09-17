"""
N x M instances with the paper's structure. Sources tile its four message
classes and five subscription patterns (two of every five carry the 2 MB
class); servers tile its three capacities, scaled so the aggregate server
utilization sum(lambda)/sum(mu_j) is rho_server; links are drawn from its
range and, given beta, scaled until the stage balance C_ij/C_j at the optimum
(flow-weighted over used routes) is beta. The paper's instance sits at
rho_server 0.13 and beta 4.0.
"""
import numpy as np
import yaml
from scipy.optimize import least_squares

from src.controller.best_response import best_response_batch
from src.controller.central import solve_central
from src.controller.diagnostics import compute_diagnostics, marginal_costs
from src.controller.feasibility import transportation_feasibility
from src.controller.synchronous import algorithm1_initial_state
from src.model.config import topology_from_config
from src.model.marginal_costs import mm1_marginal_cost_vectorized

# notebook cells 0-2: msg/s and MB per message
CLASSES = {"t1": (50, 2 / 1024), "t2": (30, 2 / 1024), "t3": (15, 2.0), "t4": (10, 1 / 1024)}
SUBSCRIPTIONS = (("t1", "t2"), ("t1",), ("t2", "t3"), ("t3", "t4"), ("t1", "t4"))     # P0..P4
PAPER_SERVERS = (160.75448519014384, 106.50515929852796, 196.56320330745592)         # SN1..SN3
LINK_CAP_RANGE = (40.0, 60.0)
SLSQP_MAX_ROUTES = 100

def source_rates(n_sources: int) -> np.ndarray:
    """MB/s of the first n patterns, tiled."""
    per_pattern = [sum(CLASSES[t][0] * CLASSES[t][1] for t in pattern) for pattern in SUBSCRIPTIONS]
    return np.array([per_pattern[i % len(SUBSCRIPTIONS)] for i in range(n_sources)])

def optimal_routing(topo, start: np.ndarray = None) -> np.ndarray:
    """
    SLSQP on small instances. On large ones the optimum is found as the fixed
    point of the M prices, p = C_j(Lambda_j(BR(p))) (Proposition 1), by
    Levenberg-Marquardt on log p with the batched best response; loads are
    capped just below capacity inside C_j so the map stays finite where a best
    response overloads a server. Heavily loaded instances are reached by
    continuation from lower server load. `start` is a routing whose loads give
    the starting prices. RuntimeError if no certified optimum is found.
    """
    if topo.mu_links.size <= SLSQP_MAX_ROUTES:
        return np.asarray(solve_central(topo.lambdas_total, topo.mu_links, topo.mu_brokers)[0])
    mu_l, lam = topo.mu_links, topo.lambdas_total

    def solve(mu_s, prices):
        cap = (1.0 - 1e-6) * mu_s
        def residual(q):
            loads = best_response_batch(mu_l, np.exp(q), lam).sum(axis=0)
            return q - np.log(mm1_marginal_cost_vectorized(np.minimum(loads, cap), mu_s))
        fit = least_squares(residual, np.log(prices), method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15,
                            max_nfev=400 * (len(prices) + 1))
        prices = np.exp(fit.x)
        routing = best_response_batch(mu_l, prices, lam)
        loads = routing.sum(axis=0)
        certified = bool(np.all(loads < mu_s)) and compute_diagnostics(
            routing, mm1_marginal_cost_vectorized(loads, mu_s), topo)["certified"]
        return routing, prices, certified

    if start is None:
        start = algorithm1_initial_state(topo).lambda_ij
    prices = mm1_marginal_cost_vectorized(start.sum(axis=0), topo.mu_brokers)
    routing, prices, certified = solve(topo.mu_brokers, prices)
    if certified:
        return routing
    for relief in (4.0, 2.0, 1.5, 1.2, 1.0):        # continuation: more server capacity first
        routing, prices, certified = solve(relief * topo.mu_brokers, prices)
        if not certified:
            break
    if certified:
        return routing
    raise RuntimeError(f"no certified optimum found for the {topo.n_sources}x{topo.n_brokers} instance")

def measured_beta(lambda_ij: np.ndarray, topo) -> float:
    """C_ij / C_j, flow-weighted over used routes."""
    mc = marginal_costs(lambda_ij, topo.mu_links, topo.mu_brokers)
    used = (lambda_ij > 1e-9) & topo.route_mask
    weight = lambda_ij[used] / lambda_ij[used].sum()
    return float((weight * (mc["C_ij"] / mc["C_j"][None, :])[used]).sum())

def generate_instance(n_sources: int, n_brokers: int, rho_server: float = 0.3, beta: float = None,
                      seed: int = 42, beta_tol: float = 0.01) -> tuple:
    """
    (topology section, stats). stats has rho_server, and with beta the measured
    value and the rounds the adjust loop took. ValueError if no routing fits.
    """
    if not 0 < rho_server < 1:
        raise ValueError("rho_server must be in (0, 1)")
    if beta is not None and beta <= 0:
        raise ValueError("beta must be > 0")
    rng = np.random.default_rng(seed)
    lambdas = source_rates(n_sources)
    pattern = np.array([PAPER_SERVERS[j % 3] for j in range(n_brokers)])
    mu_brokers = pattern * lambdas.sum() / (rho_server * pattern.sum())
    links = rng.uniform(*LINK_CAP_RANGE, size=(n_sources, n_brokers))
    sources = [f"P{i}" for i in range(n_sources)]
    brokers = [f"SN{j + 1}" for j in range(n_brokers)]

    def section(scale):
        # full precision: rounding could make the saved instance infeasible
        return {
            "sources": [{"id": s, "rate": float(r)} for s, r in zip(sources, lambdas)],
            "brokers": [{"id": b, "capacity": float(c)} for b, c in zip(brokers, mu_brokers)],
            "access_capacities": {f"{s}->{b}": float(scale * links[i, j])
                                  for i, s in enumerate(sources) for j, b in enumerate(brokers)},
        }

    def check(topology):
        # what will actually be read back
        topo = topology_from_config(yaml.safe_load(yaml.safe_dump({"topology": topology})))
        try:
            transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
        except ValueError:
            raise ValueError(f"{n_sources}x{n_brokers} at rho_server {rho_server:.2f}"
                             f"{f', beta {beta:g}' if beta is not None else ''} is infeasible: a source "
                             "exceeds its access links") from None
        return topo

    stats = {"rho_server": float(rho_server), "beta": None}
    if beta is None:
        topology = section(1.0)
        check(topology)
        return topology, stats
    # matched utilization gives beta = mu_j / mu_ij; then a secant on the measured value, in log-log,
    # since beta responds more than proportionally to the link scale once links are utilized.
    # Steps are capped at 4x per round; a scale that leaves a source without enough link
    # capacity backs off toward the last feasible one.
    scale = float(mu_brokers.mean() / (beta * np.mean(LINK_CAP_RANGE)))
    history, routing = [], None
    for rounds in range(1, 13):
        topology = section(scale)
        try:
            topo = check(topology)
        except ValueError:
            if not history:
                raise
            scale = float(np.exp(0.5 * (np.log(scale) + history[-1][0])))
            continue
        routing = optimal_routing(topo, start=routing)
        got = measured_beta(routing, topo)
        stats.update(beta=got, beta_rounds=rounds)
        if abs(got / beta - 1) <= beta_tol:
            break
        history.append((np.log(scale), np.log(got)))
        slope = -1.0
        if len(history) > 1 and history[-1][0] != history[-2][0]:
            slope = float(np.clip((history[-1][1] - history[-2][1]) / (history[-1][0] - history[-2][0]), -3.0, -0.5))
        step = float(np.clip((np.log(beta) - history[-1][1]) / slope, -np.log(4.0), np.log(4.0)))
        scale = float(np.exp(history[-1][0] + step))
    return topology, stats

def generate_topology_config(n_sources: int, n_brokers: int, rho_server: float = 0.3, beta: float = None,
                             seed: int = 42, beta_tol: float = 0.01) -> dict:
    return generate_instance(n_sources, n_brokers, rho_server, beta, seed, beta_tol)[0]
