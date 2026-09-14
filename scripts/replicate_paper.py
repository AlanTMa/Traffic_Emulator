"""
The numbers Sec. V of arXiv:2602.03246 reports on its 5x3 instance, next to
what this emulator gives: the centralized benchmark, Algorithm 1's endpoint
and trajectory, the Table I residuals, and the windowed stochastic
experiment. --run adds a distributed run's telemetry (src.cli up).

    python -m scripts.replicate_paper [--seeds 10] [--run runs/distributed]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.controller.central import solve_central, system_objective
from src.controller.diagnostics import compute_diagnostics, marginal_costs
from src.controller.feasibility import transportation_feasibility
from src.controller.synchronous import algorithm1_initial_state, iteration_step, run_algorithm1
from src.model.config import load_config, topology_from_config
from src.runtime.experiments import CANONICAL_5X3
from src.simulation.engine import SimulationEngine
from src.simulation.events import Event, EventType
from src.simulation.handler import SimulationHandler
from src.simulation.queues import SimulationState
from src.telemetry.metrics import TelemetryBuffer

# Sec. V-B, V-C, V-D, V-E and Table I
PAPER = {
    "F_star": 2.0157, "slsqp_iterations": 29,
    "util": (0.158136, 0.138642, 0.102983),
    "iterations": 70, "F_dist": 2.0157, "objective_gap": 1.76e-13, "util_gap": 6.10e-13,
    "F_first": 2.0345, "F_10": 2.0158,
    "residuals": {  # (central, distributed)
        "r_conservation": (0.0, 1.31e-11), "r_capacity": (0.0, 0.0), "r_price": (None, 1.86e-14),
        "r_fixed_point": (1.23e-11, 5.86e-9), "r_kkt_complementarity": (6.94e-17, 1.41e-10),
        "r_active_stationarity": (6.94e-18, 5.18e-12)},
    # Sec. V-F: one 300 s run, 5 s windows, 4 warm-up windows
    "util_stoch": (0.1571, 0.1403, 0.0984), "util_stoch_rel": (0.65, 1.18, 4.41),
    "split_gap_pp": {"P2": 0.34, "P3": 0.27, "P4": 0.0}, "light_on_sn2": 3.3,
}

def rounded(a, digits):
    return tuple(float(x) for x in np.round(a, digits))

def deterministic(topo):
    F = lambda L: system_objective(L, topo.mu_links, topo.mu_brokers)
    Lc, info = solve_central(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    st = run_algorithm1(topo, eta=0.25, gamma=0.5, tol=1e-10, max_iter=4000, delta_s=1e-8, eps=1e-12)
    Ld = st.lambda_ij
    s = algorithm1_initial_state(topo, 1e-8, 1e-12)
    F_t, util_t = [], [s.lambda_ij.sum(axis=0) / topo.mu_brokers]
    for _ in range(70):
        s, _, _ = iteration_step(s, topo, 0.25, 0.5, 1e-12, 1e-8)
        F_t.append(F(s.lambda_ij))
        util_t.append(s.lambda_ij.sum(axis=0) / topo.mu_brokers)
    uc, ud = Lc.sum(axis=0) / topo.mu_brokers, Ld.sum(axis=0) / topo.mu_brokers
    gap = abs(F(Ld) - info["objective"])
    rows = [
        ("F* (V-B)", PAPER["F_star"], round(info["objective"], 4), round(info["objective"], 4) == PAPER["F_star"]),
        ("utilizations, central (V-B)", PAPER["util"], rounded(uc, 6), rounded(uc, 6) == PAPER["util"]),
        ("Algorithm 1 iterations (V-C)", PAPER["iterations"], st.iteration, st.iteration == PAPER["iterations"]),
        ("F_dist (V-C)", PAPER["F_dist"], round(F(Ld), 4), round(F(Ld), 4) == PAPER["F_dist"]),
        ("|F_dist - F*| (V-C)", PAPER["objective_gap"], float(f"{gap:.3g}"), gap < 1e-12),
        ("F after iteration 1 (V-D, first plotted point)", PAPER["F_first"], round(F_t[0], 4), round(F_t[0], 4) == PAPER["F_first"]),
        ("F after iteration 10 (V-D)", PAPER["F_10"], round(F_t[9], 4), round(F_t[9], 4) == PAPER["F_10"]),
        ("utilization trend (V-D)", "SN1, SN2 down, SN3 up", f"{np.round(util_t[0], 4)} -> {np.round(util_t[70], 4)}",
         bool(util_t[70][0] < util_t[0][0] and util_t[70][1] < util_t[0][1] and util_t[70][2] > util_t[0][2])),
        ("utilizations, distributed (V-E)", PAPER["util"], rounded(ud, 6), rounded(ud, 6) == PAPER["util"]),
        ("max utilization difference (V-E)", PAPER["util_gap"], float(f"{np.max(np.abs(ud - uc)):.3g}"), np.max(np.abs(ud - uc)) < 1e-11),
    ]
    mc = marginal_costs(Ld, topo.mu_links, topo.mu_brokers)
    delay = (Ld * (mc["D_ij"] + mc["D_j"][None, :])).sum(axis=1) / topo.lambdas_total
    order = [topo.sources[k] for k in np.argsort(delay)]
    rows.append(("per-source delay order (V-D)", "P4 smallest, P3 largest", f"{order[0]} smallest, {order[-1]} largest",
                 order[0] == "P4" and order[-1] == "P3"))
    pattern = {topo.sources[i]: [topo.brokers[j] for j in np.where(Ld[i] > 1e-7)[0]] for i in range(topo.n_sources)}
    expected = {"P0": ["SN3"], "P1": ["SN3"], "P2": ["SN1", "SN2", "SN3"], "P3": ["SN1", "SN2", "SN3"], "P4": ["SN3"]}
    rows.append(("active routes (V-E)", "P0,P1,P4: SN3; P2,P3: all", "; ".join(f"{k}: {'+'.join(v)}" for k, v in pattern.items()),
                 pattern == expected))
    dc = compute_diagnostics(Lc, marginal_costs(Lc, topo.mu_links, topo.mu_brokers)["C_j"], topo)
    dd = compute_diagnostics(Ld, st.prices, topo)
    for key, (pc, pd) in PAPER["residuals"].items():
        for label, diag, p in (("central", dc, pc), ("distributed", dd, pd)):
            if p is None:
                continue
            got = max(diag["r_access_capacity"], diag["r_service_capacity"]) if key == "r_capacity" else diag[key]
            rows.append((f"Table I {key}, {label}", p, float(f"{got:.3g}"), got <= max(2 * p, 1e-15)))
    rows.append(("SLSQP iterations (V-B)", PAPER["slsqp_iterations"], info.get("slsqp_iterations"), None))
    return rows

def windowed_run(topo, seed, duration=300.0, window=5.0, warmup=4):
    """The notebook's windowed experiment (cell 14) as src.cli run runs it; rows after warm-up."""
    engine = SimulationEngine()
    state = SimulationState(topo.n_sources, topo.n_brokers, topo.mu_links, topo.mu_brokers, keep_requests=False)
    telemetry = TelemetryBuffer(max_history=int(duration / window) + 10)
    x0 = transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    handler = SimulationHandler(engine, topo, state, x0, telemetry=telemetry, eta=0.35, gamma=0.5, beta=0.3,
                                window=window, warmup_windows=warmup, rng=np.random.default_rng(seed))
    for i in range(topo.n_sources):
        engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=i))
    engine.run(duration=duration, handler=handler.handle_event)
    return [r for r in telemetry.history if r["iteration"] > warmup]

def windowed(topo, seeds):
    u_det = np.array(PAPER["util"])
    means, fracs, qmax = [], [], 0
    for seed in seeds:
        post = windowed_run(topo, seed)
        means.append(np.mean([r["util_measured_j"] for r in post], axis=0))
        fracs.append(np.mean([r["fraction_ij"] for r in post], axis=0))
        qmax = max(qmax, max(max(r["queue_broker_j"]) for r in post))
    mean, frac = np.mean(means, axis=0), np.mean(fracs, axis=0)
    rel = 100 * (mean / u_det - 1)
    close = bool(np.all(np.abs(rel) <= 6.0))
    rows = [(f"post-warm-up mean utilization, {len(seeds)} seeds (V-F)", PAPER["util_stoch"], rounded(mean, 4), close),
            ("  relative to the deterministic values, %", PAPER["util_stoch_rel"], rounded(rel, 2), close)]
    Ld = run_algorithm1(topo, eta=0.25, gamma=0.5, tol=1e-10, max_iter=4000, delta_s=1e-8, eps=1e-12).lambda_ij
    x_det = Ld / topo.lambdas_total[:, None]
    for name, limit in PAPER["split_gap_pp"].items():
        i = topo.sources.index(name)
        gap = 100 * np.max(np.abs(frac[i] - x_det[i]))
        rows.append((f"window-averaged split vs deterministic, {name}, pp (Fig. 8)", limit, round(gap, 2), gap <= 1.0))
    j = topo.brokers.index("SN2")
    light = 100 * np.mean([frac[topo.sources.index(n)][j] for n in ("P0", "P1")])
    rows.append(("P0, P1 average share on SN2, % (Fig. 8)", PAPER["light_on_sn2"], round(light, 2), 2.0 <= light <= 5.0))
    rows.append(("largest broker queue (V-F: zero in the paper's count model)", 0, int(qmax), None))
    return rows

def distributed_run(topo, run_dir):
    rows_t = [json.loads(line) for line in open(Path(run_dir) / "metrics.jsonl", encoding="utf-8")]
    if len(rows_t) < 70:
        return [(f"distributed run {run_dir}", "70 rounds", f"{len(rows_t)} rounds", False)]
    r70 = rows_t[69]
    certified = next((r["iteration"] for r in rows_t if r["certified"]), None)
    post = [r for r in rows_t if r["iteration"] > 4]
    mean = np.mean([r["util_measured_j"] for r in post], axis=0)
    rel = 100 * (mean / np.array(PAPER["util"]) - 1)
    close = bool(np.all(np.abs(rel) <= 8.0))
    return [
        ("distributed run: F at round 70", PAPER["F_dist"], round(r70["objective"], 4), round(r70["objective"], 4) == PAPER["F_dist"]),
        ("distributed run: utilizations at round 70", PAPER["util"], rounded(r70["util_j"], 6),
         rounded(r70["util_j"], 6) == PAPER["util"]),
        ("distributed run: first certified round", "<= 70", certified, certified is not None and certified <= 70),
        (f"distributed run: measured utilization, rounds 5-{rows_t[-1]['iteration']}", PAPER["util_stoch"], rounded(mean, 4), close),
        ("  relative to the deterministic values, %", PAPER["util_stoch_rel"], rounded(rel, 2), close),
    ]

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=10, help="windowed experiment runs (10)")
    parser.add_argument("--run", help="telemetry directory of a distributed run")
    args = parser.parse_args()
    topo = topology_from_config(load_config(CANONICAL_5X3))
    rows = deterministic(topo) + windowed(topo, range(args.seeds))
    if args.run:
        rows += distributed_run(topo, args.run)
    width = max(len(r[0]) for r in rows)
    failed = 0
    for name, paper, value, ok in rows:
        failed += ok is False
        print(f"{name:<{width}}  paper {str(paper):<30} emulator {str(value):<42} {'info' if ok is None else 'ok' if ok else 'FAIL'}")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
