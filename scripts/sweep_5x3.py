"""
High-load sweep on the canonical 5x3 instance: lambda_i(r) = r * lambda_i.

The multipliers are fractions of r_max, the largest r for which a feasible
routing exists (transportation LP). In this topology r_max ~= 4.87 is set by
the access links of P2/P3, while aggregate broker load is only ~63%; the
sweep therefore reports the utilizations actually reached rather than
forcing a target.

For every r:
  A. static_algorithm1 run until tol AND the Proposition 1 certificate pass
     (paper Step 5; the notebook stops at tol only): optimum, certificate, safe-step
     activity (iterations with s_t < 1), capacity headroom along the trajectory,
     objective increases (oscillation), gap to the centralized solver.
  B. Event simulation of the optimum routing A with the controller frozen:
     queueing validation (measured work-unit sojourn times vs the M/M/1 model).
  C. Event simulation under the windowed_stochastic controller from the LP
     initializer: measured utilization, queues, latency percentiles, residuals,
     route oscillation.

Usage:
    python -m scripts.sweep_5x3 [--fractions 0.2,0.5,0.7,0.8,0.9,0.95]
                                [--duration 1000] [--warmup-time 200] [--seed 0]
                                [--output-dir runs/sweep_5x3] [--skip-events]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.controller.central import solve_central, system_objective
from src.controller.diagnostics import compute_diagnostics
from src.controller.feasibility import transportation_feasibility
from src.controller.synchronous import iteration_step
from src.model.marginal_costs import mm1_marginal_cost_vectorized
from src.model.topology import Topology
from src.runtime.metadata import git_revision
from src.simulation.engine import SimulationEngine
from src.simulation.events import Event, EventType
from src.simulation.handler import SimulationHandler
from src.simulation.queues import SimulationState
from src.simulation.state import SystemState
from src.telemetry.metrics import TelemetryBuffer

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((ROOT / "tests" / "regression" / "data" / "baseline_5x3.json").read_text())["parameters"]

def base_topology(r: float = 1.0) -> Topology:
    return Topology(r * np.array(BASELINE["lambdas_total"]), np.array(BASELINE["mu_links"]),
                    np.array(BASELINE["mu_brokers"]), [f"P{i}" for i in range(5)], ["SN1", "SN2", "SN3"])

def max_multiplier(margin: float = 1e-8) -> float:
    """Largest r with a feasible routing (bisection on the transportation LP)."""
    topo = base_topology()
    def feasible(r):
        try:
            transportation_feasibility(r * topo.lambdas_total, topo.mu_links, topo.mu_brokers, margin=margin)
            return True
        except ValueError:
            return False
    lo, hi = 1.0, 2.0
    while feasible(hi):
        lo, hi = hi, 2 * hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if feasible(mid) else (lo, mid)
    return lo

def static_experiment(topo: Topology, eta=0.25, gamma=0.5, tol=1e-10, max_iter=20000, delta_s=1e-8, eps=1e-12):
    """A: Algorithm 1 to convergence with trajectory statistics."""
    lam0 = transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers, margin=delta_s)
    state = SystemState(lambda_ij=lam0.copy(),
                        prices=mm1_marginal_cost_vectorized(lam0.sum(axis=0), topo.mu_brokers, eps))
    objective = [system_objective(lam0, topo.mu_links, topo.mu_brokers)]
    s_hist, broker_headroom, link_headroom, route_rel = [], [], [], []
    iterations_to_tol = None
    for _ in range(max_iter):
        state, s_t, residual = iteration_step(state, topo, eta, gamma, eps=eps, delta_s=delta_s)
        objective.append(system_objective(state.lambda_ij, topo.mu_links, topo.mu_brokers))
        s_hist.append(s_t)
        route_rel.append(state.route_rel)
        broker_headroom.append(np.min(topo.mu_brokers - state.lambda_ij.sum(axis=0)))
        link_headroom.append(np.min(topo.mu_links - state.lambda_ij))
        if residual < tol:
            # Notebook stopping point; continue to the paper's Step 5 (certificate)
            iterations_to_tol = iterations_to_tol or state.iteration
            diag = compute_diagnostics(state.lambda_ij, state.prices, topo, eps=eps)
            if diag["certified"]:
                break
    L = state.lambda_ij
    diag = compute_diagnostics(L, state.prices, topo, eps=eps)
    L_c, info = solve_central(topo.lambdas_total, topo.mu_links, topo.mu_brokers, margin=delta_s)
    objective = np.array(objective)
    return L, {
        "iterations_to_tol": iterations_to_tol,          # notebook stopping rule
        "iterations": state.iteration,                   # + certificate (paper Step 5)
        "objective": objective[-1],
        "central_objective": info["objective"],
        "relative_objective_gap": (objective[-1] - info["objective"]) / info["objective"],
        "model_mean_sojourn": objective[-1] / topo.lambdas_total.sum(),
        "util_j": L.sum(axis=0) / topo.mu_brokers,
        "max_access_util": float(np.max(L / topo.mu_links)),
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

def event_experiment(topo: Topology, x0: np.ndarray, *, frozen: bool, duration: float, warmup_time: float,
                     seed: int, window=5.0, eta=0.35, gamma=0.5, beta=0.3, warmup_windows=4):
    """B (frozen=True) or C (windowed controller): event simulation statistics after warmup_time."""
    engine = SimulationEngine()
    state = SimulationState(topo.n_sources, topo.n_brokers, topo.mu_links, topo.mu_brokers, keep_requests=False)
    telemetry = TelemetryBuffer(max_history=int(duration / window) + 10)
    log = []
    handler = SimulationHandler(engine, topo, state, x0.copy(), telemetry=telemetry, eta=eta, gamma=gamma, beta=beta,
                                window=window, warmup_windows=10**9 if frozen else warmup_windows,
                                rng=np.random.default_rng(seed), latency_log=log)
    for i in range(topo.n_sources):
        engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=i))
    engine.run(duration=duration, handler=handler.handle_event)

    rows = [r for r in telemetry.history if r["sim_time"] > warmup_time]
    t, src, lat = (np.array(c) for c in zip(*log))
    post = t > warmup_time
    lat, src = lat[post], src[post].astype(int)
    q_broker = np.array([r["queue_broker_j"] for r in rows])
    q_access = np.array([r["queue_access_ij"] for r in rows])
    frac = np.array([r["fraction_ij"] for r in rows])
    last = telemetry.history[-1]
    return {
        "completed": int(lat.size),
        "latency_mean": float(lat.mean()),
        "latency_p50": float(np.percentile(lat, 50)),
        "latency_p95": float(np.percentile(lat, 95)),
        "latency_p99": float(np.percentile(lat, 99)),
        "latency_mean_i": [float(lat[src == i].mean()) if np.any(src == i) else np.nan for i in range(topo.n_sources)],
        "model_mean_sojourn": system_objective(handler.x_ij, topo.mu_links, topo.mu_brokers) / topo.lambdas_total.sum(),
        "util_measured_j": np.mean([r["util_measured_j"] for r in rows], axis=0),
        "util_planned_j": np.array(last["util_j"]),
        "max_access_util_planned": float(np.max(handler.x_ij / topo.mu_links)),
        "queue_broker_mean_j": q_broker.mean(axis=0),
        "queue_broker_max": int(q_broker.max()),
        "queue_access_max": int(q_access.max()),
        "objective": last["objective"],
        "r_price": last["r_price"], "r_fixed_point": last["r_fixed_point"],
        "r_kkt_stationarity": last["r_kkt_stationarity"],
        "certified": last["certified"],
        "mean_route_rel": float(np.nanmean([r["route_rel"] for r in rows])),
        "fraction_std": float(frac.std(axis=0).mean()),   # route oscillation around the mean split
        "planned_broker_overload_windows": int(sum(np.any(np.array(r["util_j"]) >= 1.0) for r in rows)),
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--fractions", default="0.2,0.5,0.7,0.8,0.9,0.95", help="multipliers as fractions of r_max")
    parser.add_argument("--duration", type=float, default=1000.0, help="simulated seconds per event run")
    parser.add_argument("--warmup-time", type=float, default=200.0, help="simulated seconds discarded from statistics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/sweep_5x3")
    parser.add_argument("--skip-events", action="store_true", help="only run the static Algorithm 1 part")
    args = parser.parse_args()

    r_max = max_multiplier()
    fractions = [float(f) for f in args.fractions.split(",")]
    print(f"r_max = {r_max:.6f} (aggregate broker load {r_max * base_topology().lambdas_total.sum() / base_topology().mu_brokers.sum():.1%})")

    results = []
    for k, frac in enumerate(fractions):
        r = frac * r_max
        topo = base_topology(r)
        L_opt, static = static_experiment(topo)
        row = {"fraction_of_r_max": frac, "r": r, "offered_load_over_broker_capacity": r * BASELINE_TOTAL / BROKER_TOTAL,
               "static": static}
        print(f"\n[{frac:.0%} of r_max, r={r:.3f}] static: {static['iterations_to_tol']}/{static['iterations']} it "
              f"(tol/certified), F={static['objective']:.6f}, "
              f"util_j={np.round(static['util_j'], 3)}, max access util={static['max_access_util']:.3f}, "
              f"certified={static['certified']}, s_t<1 in {static['safe_step_binding_iterations']} it")
        if not args.skip_events:
            lam0 = transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
            for name, x0, frozen in (("optimum_frozen", L_opt, True), ("windowed", lam0, False)):
                ev = event_experiment(topo, x0, frozen=frozen, duration=args.duration,
                                      warmup_time=args.warmup_time, seed=args.seed * 1000 + k)
                row[name] = ev
                print(f"  {name:>15}: mean {ev['latency_mean']:.4f}s (model {ev['model_mean_sojourn']:.4f}s), "
                      f"p95 {ev['latency_p95']:.4f}, p99 {ev['latency_p99']:.4f}, "
                      f"measured util_j={np.round(ev['util_measured_j'], 3)}, max broker queue {ev['queue_broker_max']}, "
                      f"max access queue {ev['queue_access_max']}, fraction std {ev['fraction_std']:.4f}")
        results.append(row)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"r_max": r_max, "fractions": fractions, "duration": args.duration, "warmup_time": args.warmup_time,
            "seed": args.seed, "git": git_revision(),
            "note": "lambda_i(r) = r * lambda_i on the notebook's exact 5x3 instance; units: normalized work units/s"}
    (out / "sweep.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2, default=_json_default))
    flat = pd.json_normalize(json.loads(json.dumps(results, default=_json_default)))
    flat.to_csv(out / "summary.csv", index=False)
    print(f"\nWrote {out / 'sweep.json'} and {out / 'summary.csv'}")

def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(type(value))

BASELINE_TOTAL = float(np.sum(BASELINE["lambdas_total"]))
BROKER_TOTAL = float(np.sum(BASELINE["mu_brokers"]))

if __name__ == "__main__":
    main()
