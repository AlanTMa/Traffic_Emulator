"""
Shared experiment runners for comparing controller/simulation modes.

- static_experiment: static_algorithm1 to tol and the Proposition 1
  certificate (paper Step 5), with trajectory statistics.
- event_experiment: an event-driven run under one of
    "frozen"                     routing fixed at x0 (queueing validation)
    "windowed_stochastic"        notebook windowed controller
    "capacity_safe_event_driven" Algorithm 1 steps on planned rates
  with measured statistics taken after `warmup_time`.

Measured event statistics are finite-run estimates; they are not expected
to equal the analytical M/M/1 means exactly.
"""
from pathlib import Path

import numpy as np

from src.controller.central import solve_central, system_objective
from src.controller.diagnostics import compute_diagnostics
from src.controller.feasibility import transportation_feasibility
from src.controller.synchronous import algorithm1_initial_state, iteration_step
from src.model.config import load_config, topology_from_config
from src.model.topology import Topology
from src.simulation.engine import SimulationEngine
from src.simulation.events import Event, EventType
from src.simulation.handler import SimulationHandler
from src.simulation.queues import SimulationState
from src.telemetry.metrics import TelemetryBuffer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_5X3 = PROJECT_ROOT / "config" / "paper_5x3.yaml"
EVENT_MODES = ("frozen", "windowed_stochastic", "capacity_safe_event_driven")

def scaled_topology(topology: Topology, r: float) -> Topology:
    """Same topology with every source rate multiplied by r."""
    return Topology(r * topology.lambdas_total, topology.mu_links.copy(), topology.mu_brokers.copy(),
                    list(topology.sources), list(topology.brokers), route_mask=topology.route_mask.copy())

def canonical_5x3(r: float = 1.0) -> Topology:
    """The notebook's exact 5x3 instance, source rates scaled by r."""
    return scaled_topology(topology_from_config(load_config(CANONICAL_5X3)), r)

def max_multiplier(topology: Topology, margin: float = 1e-8) -> float:
    """Largest r for which r * lambda has a feasible routing (bisection on the transportation LP)."""
    def feasible(r):
        try:
            transportation_feasibility(r * topology.lambdas_total, topology.mu_links, topology.mu_brokers,
                                       margin=margin)
            return True
        except ValueError:
            return False
    lo, hi = 1.0, 2.0
    if not feasible(lo):
        raise ValueError("the unscaled topology is already infeasible")
    while feasible(hi):
        lo, hi = hi, 2 * hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if feasible(mid) else (lo, mid)
    return lo

def static_experiment(topo: Topology, eta=0.25, gamma=0.5, tol=1e-10, max_iter=20000, delta_s=1e-8, eps=1e-12):
    """static_algorithm1 until tol AND the certificate pass, with trajectory statistics."""
    state = algorithm1_initial_state(topo, delta_s, eps)
    objective = [system_objective(state.lambda_ij, topo.mu_links, topo.mu_brokers)]
    s_hist, broker_headroom, link_headroom, route_rel = [], [], [], []
    iterations_to_tol = None
    for _ in range(max_iter):
        state, s_t, residual = iteration_step(state, topo, eta, gamma, eps=eps, delta_s=delta_s)
        objective.append(system_objective(state.lambda_ij, topo.mu_links, topo.mu_brokers))
        s_hist.append(s_t)
        route_rel.append(state.route_rel)
        broker_headroom.append(np.min(topo.mu_brokers - state.lambda_ij.sum(axis=0)))
        link_headroom.append(np.min((topo.mu_links - state.lambda_ij)[topo.route_mask]))
        if residual < tol:
            # Notebook stopping point; continue to the paper's Step 5 (certificate)
            iterations_to_tol = iterations_to_tol or state.iteration
            if compute_diagnostics(state.lambda_ij, state.prices, topo, eps=eps)["certified"]:
                break
    L = state.lambda_ij
    diag = compute_diagnostics(L, state.prices, topo, eps=eps)
    _, info = solve_central(topo.lambdas_total, topo.mu_links, topo.mu_brokers, margin=delta_s)
    objective = np.array(objective)
    return L, {
        "iterations_to_tol": iterations_to_tol,          # notebook stopping rule
        "iterations": state.iteration,                   # + certificate (paper Step 5)
        "objective": objective[-1],
        "central_objective": info["objective"],
        "relative_objective_gap": (objective[-1] - info["objective"]) / info["objective"],
        "model_mean_sojourn": objective[-1] / topo.lambdas_total.sum(),
        "fraction_ij": L / topo.lambdas_total[:, None],
        "util_j": L.sum(axis=0) / topo.mu_brokers,
        "access_util_ij": np.where(topo.route_mask, L / np.where(topo.route_mask, topo.mu_links, 1.0), np.nan),
        "max_access_util": float(np.max(L[topo.route_mask] / topo.mu_links[topo.route_mask])),
        "price_j": state.prices,
        "certified": diag["certified"],
        "failed": diag["failed"],
        "r_price": diag["r_price"], "r_fixed_point": diag["r_fixed_point"],
        "r_kkt_stationarity": diag["r_active_stationarity"],
        "r_kkt_complementarity": diag["r_kkt_complementarity"],
        "r_inactive_complementarity": diag["r_inactive_complementarity"],
        "safe_step_binding_iterations": int(np.sum(np.array(s_hist) < 1.0)),
        "min_s_t": float(np.min(s_hist)),
        "min_broker_headroom": float(np.min(broker_headroom)),
        "min_link_headroom": float(np.min(link_headroom)),
        "objective_increases": int(np.sum(np.diff(objective) > 1e-15)),
        "total_route_variation": float(np.sum(route_rel)),
    }

def _growth(series: np.ndarray) -> float:
    q = max(len(series) // 4, 1)
    first, last = float(np.mean(series[:q])), float(np.mean(series[-q:]))
    return last / first if first > 0 else np.nan

def event_experiment(topo: Topology, mode: str, *, x0: np.ndarray = None, duration: float, warmup_time: float,
                     seed: int, window: float = 5.0, eta: float = None, gamma: float = 0.5, beta: float = 0.3,
                     warmup_windows: int = 4, delta_s: float = 1e-8, eps: float = 1e-12):
    """
    One event-driven run; statistics over what happens after `warmup_time`.

    mode "frozen" needs x0 (e.g. a static optimum); the controller modes
    default to their own initial routing (LP; margin delta_s for Algorithm 1)
    and step size (eta 0.35 windowed, 0.25 Algorithm 1).
    """
    if mode not in EVENT_MODES:
        raise ValueError(f"mode must be one of {EVENT_MODES}")
    algorithm1 = mode == "capacity_safe_event_driven"
    if x0 is None:
        if mode == "frozen":
            raise ValueError("mode 'frozen' needs x0")
        x0 = transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers,
                                        margin=delta_s if algorithm1 else 1e-8)
    if eta is None:
        eta = 0.25 if algorithm1 else 0.35

    engine = SimulationEngine()
    state = SimulationState(topo.n_sources, topo.n_brokers, topo.mu_links, topo.mu_brokers, keep_requests=False)
    telemetry = TelemetryBuffer(max_history=int(duration / window) + 10)
    log = []
    if algorithm1:
        handler = SimulationHandler(engine, topo, state, np.array(x0, dtype=float), telemetry=telemetry, eta=eta,
                                    gamma=gamma, beta=beta, window=window, eps=eps, delta_s=delta_s,
                                    controller_mode=mode, rng=np.random.default_rng(seed), latency_log=log)
    else:
        handler = SimulationHandler(engine, topo, state, np.array(x0, dtype=float), telemetry=telemetry, eta=eta,
                                    gamma=gamma, beta=beta, window=window,
                                    warmup_windows=10**9 if mode == "frozen" else warmup_windows,
                                    rng=np.random.default_rng(seed), latency_log=log)
    for i in range(topo.n_sources):
        engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=i))
    engine.run(duration=duration, handler=handler.handle_event)

    all_rows = list(telemetry.history)
    rows = [r for r in all_rows if r["sim_time"] > warmup_time]
    t, src, lat = (np.array(c) for c in zip(*log))
    post = t > warmup_time
    lat, src = lat[post], src[post].astype(int)
    q_broker = np.array([r["queue_broker_j"] for r in rows])
    q_access = np.array([r["queue_access_ij"] for r in rows])
    frac = np.array([r["fraction_ij"] for r in rows])
    planned_loads = np.array([r["load_j"] for r in all_rows])
    planned_links = np.array([r["lambda_ij"] for r in all_rows])
    s_t = np.array([r["s_t"] for r in all_rows], dtype=float)
    last = all_rows[-1]
    model_mean = system_objective(handler.x_ij, topo.mu_links, topo.mu_brokers) / topo.lambdas_total.sum()
    return {
        "mode": mode,
        "windows": len(all_rows),
        "completed": int(lat.size),
        # Measured (post warm-up)
        "latency_mean": float(lat.mean()),
        "latency_p50": float(np.percentile(lat, 50)),
        "latency_p95": float(np.percentile(lat, 95)),
        "latency_p99": float(np.percentile(lat, 99)),
        "latency_mean_i": [float(lat[src == i].mean()) if np.any(src == i) else np.nan for i in range(topo.n_sources)],
        "model_mean_sojourn": model_mean,               # F / sum(lambda) of the final planned routing
        "latency_vs_model": float(lat.mean() / model_mean - 1.0),
        "util_measured_j": np.mean([r["util_measured_j"] for r in rows], axis=0),          # EWMA, as notebook
        # Raw arrival rates averaged over the post-warm-up windows: actual utilizations
        "util_actual_j": np.mean([r["broker_rate_measured_j"] for r in rows], axis=0) / topo.mu_brokers,
        "access_util_actual_ij": np.mean([r["access_util_measured_ij"] for r in rows], axis=0),
        "queue_broker_mean_j": q_broker.mean(axis=0),
        "queue_broker_max": int(q_broker.max()),
        "queue_access_mean_ij": q_access.mean(axis=0),
        "queue_access_max": int(q_access.max()),
        # Total queue occupancy in the last quarter of the measured period vs
        # the first quarter (>> 1 suggests growth rather than a stationary queue)
        "queue_growth": _growth(q_broker.sum(axis=1) + q_access.sum(axis=(1, 2))),
        # Planned (controller state)
        "fraction_ij": np.array(last["fraction_ij"]),
        "util_planned_j": np.array(last["util_j"]),
        "access_util_planned_ij": np.array(last["access_util_ij"]),
        "max_access_util_planned": float(np.nanmax(np.array(last["access_util_ij"], dtype=float))),
        "price_j": np.array(last["price_j"]),
        "objective": last["objective"],
        "r_price": last["r_price"], "r_fixed_point": last["r_fixed_point"],
        "r_kkt_stationarity": last["r_kkt_stationarity"],
        "r_kkt_complementarity": last["r_kkt_complementarity"],
        "certified": last["certified"],
        "min_planned_broker_headroom": float(np.min(topo.mu_brokers - planned_loads)),
        "min_planned_link_headroom": float(np.min((topo.mu_links - planned_links)[:, topo.route_mask])),
        "planned_broker_overload_windows": int(np.sum(np.any(planned_loads >= topo.mu_brokers, axis=1))),
        "safe_step_binding_windows": int(np.sum(s_t < 1.0)) if algorithm1 else None,
        "min_s_t": float(np.nanmin(s_t)) if algorithm1 else None,
        "br_failures_total": last.get("br_failures_total", 0),
        # Route oscillation after warm-up
        "mean_route_rel": float(np.nanmean([r["route_rel"] for r in rows])),
        "fraction_std": float(frac.std(axis=0).mean()),
    }
