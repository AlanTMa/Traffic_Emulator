"""
Replay Algorithm 1 from a run's resolved config and compare every recorded
round (lambda_ij, prices, s_t); also check conservation, headroom and that
work was served. Same platform: bit-identical (--tol 0). Linux containers
checked from Windows or macOS differ in the last bits (libm), hence the
default tolerance.

    python -m scripts.verify_distributed_run runs/distributed [--tol 1e-12]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.controller.synchronous import algorithm1_initial_state, iteration_step
from src.model.config import load_config, topology_from_config

def verify(run_dir: Path, tol: float = 1e-12) -> list:
    """Problems found (empty when the run matches the reference)."""
    rows = [json.loads(line) for line in open(run_dir / "metrics.jsonl", encoding="utf-8")]
    meta = json.load(open(run_dir / "run.json", encoding="utf-8"))
    config = load_config(run_dir / "resolved_config.yaml")
    topo = topology_from_config(config)
    p = config["algorithm"]
    problems = []
    if meta.get("execution_backend") != "distributed":
        problems.append(f"execution_backend is {meta.get('execution_backend')!r}")
    for role, count in (("source", topo.n_sources), ("broker", topo.n_brokers)):
        if len(meta.get("processes", {}).get(role, {})) != count:
            problems.append(f"run.json lists {len(meta.get('processes', {}).get(role, {}))} {role} processes, "
                            f"topology has {count}")
    if not rows:
        return problems + ["no telemetry rounds"]

    state = algorithm1_initial_state(topo, p["delta_s"], p["eps"])
    worst = 0.0
    for k, row in enumerate(rows, start=1):
        if row["iteration"] != k:
            problems.append(f"record {k} has iteration {row['iteration']}")
            break
        if row.get("br_failures"):              # the reference raises; the controller held those sources
            problems.append(f"round {k}: best-response failures {row['br_failures']}")
            break
        state, _, _ = iteration_step(state, topo, p["eta"], p["gamma"], p["eps"], p["delta_s"],
                                     safe_step_variant=p["safe_step_variant"])
        diff = max(np.max(np.abs(np.array(row["lambda_ij"]) - state.lambda_ij)),
                   np.max(np.abs(np.array(row["price_j"]) - state.prices)), abs(row["s_t"] - state.s_t))
        worst = max(worst, diff)
    if worst > tol:
        problems.append(f"max |distributed - reference| = {worst:.3g} exceeds {tol:g}")

    last = rows[-1]
    lam = np.array(last["lambda_ij"])
    conservation = np.max(np.abs(lam.sum(axis=1) - topo.lambdas_total))
    headroom = np.min(topo.mu_brokers - p["delta_s"] - lam.sum(axis=0))
    if conservation > 1e-8:
        problems.append(f"source conservation off by {conservation:.3g}")
    if headroom < -1e-9:
        problems.append(f"planned broker load exceeds mu_j - delta_s by {-headroom:.3g}")
    if not last.get("completed_total"):
        problems.append("no work units completed")

    print(f"{run_dir}: {len(rows)} rounds, {topo.n_sources} sources x {topo.n_brokers} brokers, "
          f"max |distributed - reference| {worst:.3g}")
    print(f"  objective {last['objective']:.13f}, conservation {conservation:.2g}, "
          f"min broker headroom {headroom:.4f}, completed {last['completed_total']} units")
    print(f"  {last['certificate_status']}")
    return problems

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--tol", type=float, default=1e-12, help="0 requires bit-identical rounds")
    args = parser.parse_args()
    problems = verify(args.run_dir, args.tol)
    for problem in problems:
        print(f"  FAIL: {problem}")
    sys.exit(1 if problems else 0)

if __name__ == "__main__":
    main()
