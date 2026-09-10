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
  C. capacity_safe_event_driven: event-driven queues with Algorithm 1 steps
     on planned rates: planned utilization and capacity feasibility, safe-step
     activity, queue occupancy and growth, sojourn distribution.
  D. windowed_stochastic from the LP initializer: EWMA-measured utilization,
     planned routing, queues, latency distribution, planned-overload windows.

Utilization is reported per broker and per access path, planned (lambda_ij /
mu_ij, Lambda_j / mu_j) and actual (measured arrival rates averaged after the
warm-up) - never as total offered traffic over total capacity alone.

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

from src.controller.feasibility import transportation_feasibility
from src.runtime.experiments import canonical_5x3, event_experiment, max_multiplier as _max_multiplier
from src.runtime.experiments import static_experiment
from src.runtime.metadata import git_revision

def base_topology(r: float = 1.0):
    """The notebook's exact 5x3 instance with source rates scaled by r."""
    return canonical_5x3(r)

def max_multiplier(margin: float = 1e-8) -> float:
    """Largest feasible multiplier of the canonical 5x3 instance."""
    return _max_multiplier(canonical_5x3(), margin)

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
            for name, mode, x0 in (("optimum_frozen", "frozen", L_opt),
                                   ("capacity_safe", "capacity_safe_event_driven", None),
                                   ("windowed", "windowed_stochastic", None)):
                ev = event_experiment(topo, mode, x0=x0, duration=args.duration,
                                      warmup_time=args.warmup_time, seed=args.seed * 1000 + k)
                row[name] = ev
                steps = (f", s_t<1 in {ev['safe_step_binding_windows']} windows (min {ev['min_s_t']:.3f})"
                         if ev["min_s_t"] is not None else f", planned overload windows {ev['planned_broker_overload_windows']}")
                print(f"  {name:>15}: mean {ev['latency_mean']:.4f}s (model {ev['model_mean_sojourn']:.4f}s, "
                      f"{ev['latency_vs_model']:+.1%}), p95 {ev['latency_p95']:.4f}, p99 {ev['latency_p99']:.4f}\n"
                      f"{'':>19}util planned {np.round(ev['util_planned_j'], 3)} actual {np.round(ev['util_actual_j'], 3)}, "
                      f"max access util planned {ev['max_access_util_planned']:.3f} "
                      f"actual {np.max(ev['access_util_actual_ij']):.3f}\n"
                      f"{'':>19}min planned headroom broker {ev['min_planned_broker_headroom']:.3g} "
                      f"link {ev['min_planned_link_headroom']:.3g}{steps}\n"
                      f"{'':>19}queues max broker {ev['queue_broker_max']} access {ev['queue_access_max']}, "
                      f"growth {ev['queue_growth']:.2f}, fraction std {ev['fraction_std']:.4f}, "
                      # frozen routing has no controller, so no meaningful certificate
                      f"certified {ev['certified'] if mode != 'frozen' else 'n/a'}")
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

BASELINE_TOTAL = float(np.sum(canonical_5x3().lambdas_total))
BROKER_TOTAL = float(np.sum(canonical_5x3().mu_brokers))

if __name__ == "__main__":
    main()
