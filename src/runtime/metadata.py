"""
Run metadata written next to the telemetry (run.json) for reproducibility.
"""
import json
import os
import platform
import secrets
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def resolve_seed(config: dict) -> int:
    """simulation.seed from the config, or a fresh random seed (which is then recorded)."""
    seed = config.get("simulation", {}).get("seed")
    return int(seed) if seed is not None else secrets.randbits(32)

def git_revision() -> dict:
    """
    Commit SHA of the working tree and whether it has uncommitted changes.
    Inside a container (no .git) the launcher passes them as TE_GIT_SHA / TE_GIT_DIRTY.
    """
    def git(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    try:
        sha = git("rev-parse", "HEAD") or None
        if sha:
            return {"sha": sha, "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError):
        pass
    dirty = os.environ.get("TE_GIT_DIRTY")
    return {"sha": os.environ.get("TE_GIT_SHA") or None, "dirty": None if not dirty else dirty == "1"}

def write_run_metadata(output_dir: Path, config: dict, topology, *, seed: int, controller_mode: str,
                       real_time: bool, parameters: dict, config_path: str = None, extra: dict = None) -> dict:
    """Write output_dir/run.json and return its contents."""
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_path": str(config_path) if config_path else None,
        "controller_mode": controller_mode,
        "real_time": real_time,
        "seed": seed,
        "git": git_revision(),
        "n_sources": topology.n_sources,
        "n_brokers": topology.n_brokers,
        "sources": list(topology.sources),
        "brokers": list(topology.brokers),
        "lambdas_total": topology.lambdas_total.tolist(),
        "mu_links": topology.mu_links.tolist(),
        "mu_brokers": topology.mu_brokers.tolist(),
        "units": "normalized work units/s (5x3 instance: 1 unit = 1 MB)",
        "parameters": parameters,   # eta, gamma, beta, delta_s, eps, window, warmup as used
        "config": config,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "execution_backend": "in_process",
        **(extra or {}),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta
