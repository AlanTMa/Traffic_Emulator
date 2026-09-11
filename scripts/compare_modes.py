"""
The three controller modes side by side on one topology. Both event modes use
the same seed; their statistics are taken after --warmup-time.

    python -m scripts.compare_modes [--config config/paper_5x3.yaml] [--multiplier 1.0]
                                    [--duration 1000] [--warmup-time 200] [--seed 0]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.config import load_config, topology_from_config
from src.runtime.experiments import CANONICAL_5X3, event_experiment, scaled_topology, static_experiment
from src.runtime.metadata import git_revision

MODES = ("static_algorithm1", "capacity_safe_event_driven", "windowed_stochastic")

def compare(topology, *, duration: float, warmup_time: float, seed: int, window: float = 5.0) -> dict:
    """{mode: stats}; the static stats have no measured fields."""
    _, static = static_experiment(topology)
    results = {"static_algorithm1": static}
    for mode in MODES[1:]:
        results[mode] = event_experiment(topology, mode, duration=duration, warmup_time=warmup_time,
                                         seed=seed, window=window)
    return results

def comparison_table(results: dict, topology) -> pd.DataFrame:
    """Quantities by mode."""
    rows = {}
    get = lambda r, k: r.get(k, np.nan) if r.get(k) is not None else np.nan
    for mode in MODES:
        r = results[mode]
        planned_util = r["util_j"] if mode == "static_algorithm1" else r["util_planned_j"]
        max_access = r["max_access_util"] if mode == "static_algorithm1" else r["max_access_util_planned"]
        col = {
            "objective (planned)": r["objective"],
            **{f"planned util {b}": u for b, u in zip(topology.brokers, planned_util)},
            "planned max access util": max_access,
            **{f"price {b}": p for b, p in zip(topology.brokers, r["price_j"])},
            "r_price": r["r_price"], "r_fixed_point": r["r_fixed_point"],
            "r_kkt_stationarity": r["r_kkt_stationarity"], "r_kkt_complementarity": r["r_kkt_complementarity"],
            "certified": r["certified"],
            "model mean sojourn F/sum(lambda)": r["model_mean_sojourn"],
        }
        if mode != "static_algorithm1":
            col.update({
                **{f"measured util {b}": u for b, u in zip(topology.brokers, r["util_measured_j"])},
                **{f"mean queue {b}": q for b, q in zip(topology.brokers, r["queue_broker_mean_j"])},
                "max broker queue": r["queue_broker_max"], "max access queue": r["queue_access_max"],
                "measured mean sojourn": r["latency_mean"], "p50": r["latency_p50"],
                "p95": r["latency_p95"], "p99": r["latency_p99"],
                "measured vs model mean": r["latency_vs_model"],
                "safe-step binding windows": get(r, "safe_step_binding_windows"),
                "route oscillation (fraction std)": r["fraction_std"],
            })
        rows[mode] = col
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Compare controller/simulation modes on one topology")
    parser.add_argument("--config", default=str(CANONICAL_5X3), help="config whose topology is used")
    parser.add_argument("--multiplier", type=float, default=1.0, help="scale all source rates by this factor")
    parser.add_argument("--duration", type=float, default=1000.0, help="simulated seconds per event run")
    parser.add_argument("--warmup-time", type=float, default=200.0, help="simulated seconds excluded from statistics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--output-dir", default="runs/compare_modes")
    args = parser.parse_args()

    topology = scaled_topology(topology_from_config(load_config(args.config)), args.multiplier)
    results = compare(topology, duration=args.duration, warmup_time=args.warmup_time, seed=args.seed,
                      window=args.window)
    table = comparison_table(results, topology)

    with pd.option_context("display.float_format", "{:.6g}".format, "display.width", 160,
                           "display.max_rows", 200):
        print(table.to_string())
    print("\nPlanned routing fractions lambda_ij / lambda_i:")
    for mode in MODES:
        frac = pd.DataFrame(results[mode]["fraction_ij"], index=topology.sources, columns=topology.brokers)
        print(f"\n{mode}\n{frac.round(4).to_string()}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison.csv")
    meta = {"config": args.config, "multiplier": args.multiplier, "duration": args.duration,
            "warmup_time": args.warmup_time, "seed": args.seed, "window": args.window, "git": git_revision()}
    (out / "comparison.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2,
                                                    default=_json_default))
    print(f"\nWrote {out / 'comparison.csv'} and {out / 'comparison.json'}")

def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(type(value))

if __name__ == "__main__":
    main()
