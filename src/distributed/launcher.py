"""
One-command launcher of the distributed emulator.

    python -m src.cli up [--config config/paper_5x3.yaml | --sources N --brokers M]
                         [--backend docker|local] [--window 5] [--output-dir runs/distributed]

Resolves the run (topology, Algorithm 1 parameters, seed) into
<output-dir>/resolved_config.yaml, then starts one controller, M broker
processes, N source processes and the dashboard:

- backend "docker": `docker compose --profile distributed up --build
  --scale source=N --scale broker=M` (docker-compose.yml); every process is
  its own container.
- backend "local": the same processes as child OS processes on this
  machine, connected over localhost TCP (no Docker required).

Ctrl+C stops everything.
"""
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

from src.distributed.protocol import CONTROL_PORT
from src.distributed.settings import distributed_settings, resolved_config
from src.model.config import load_config, topology_from_config
from src.model.generate import generate_topology_config
from src.runtime.metadata import git_revision

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def prepare_run(config_path=None, sources=None, brokers=None, load=0.3, seed=42, output_dir="runs/distributed",
                window=None, overrides=None) -> dict:
    """Resolve and write the run's config; returns paths and N, M."""
    if config_path:
        config = load_config(config_path)
        topo = topology_from_config(config)
        for flag, given, actual in (("--sources", sources, topo.n_sources), ("--brokers", brokers, topo.n_brokers)):
            if given is not None and given != actual:
                raise ValueError(f"{flag} {given} does not match {config_path}, which has {actual}")
    else:
        if sources is None or brokers is None:
            raise ValueError("give --config, or --sources and --brokers for a generated topology")
        config = {"simulation": {"controller_mode": "capacity_safe_event_driven", "seed": int(seed)},
                  "topology": generate_topology_config(sources, brokers, load, int(seed))}
    if window is not None:
        config.setdefault("simulation", {})["window"] = float(window)
    settings = distributed_settings(config, overrides)
    resolved = resolved_config(config, settings)
    topo = topology_from_config(resolved)                  # validates the resolved topology again
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "resolved_config.yaml"
    path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return {"config_path": path, "output_dir": out, "n": topo.n_sources, "m": topo.n_brokers,
            "settings": settings, "source": config_path or f"generated {sources}x{brokers}"}

def describe(run: dict):
    p = run["settings"]["params"]
    print(f"Distributed emulator: {run['n']} sources x {run['m']} brokers from {run['source']}", flush=True)
    print(f"  Algorithm 1: eta={p['eta']} gamma={p['gamma']} delta_s={p['delta_s']} "
          f"({p['safe_step_variant']} safe step), round every {p['window']} s, seed {run['settings']['seed']}")
    for note in run["settings"]["notes"]:
        print(f"  note: {note}")
    print(f"  resolved config: {run['config_path']}; telemetry: {run['output_dir'] / 'metrics.jsonl'}")

# ---------------------------------------------------------------- local processes

def start_local(run: dict, port: int = CONTROL_PORT, dashboard: bool = True, dashboard_port: int = 8501,
                duration: float = None, quiet: bool = False) -> list:
    """Start controller, brokers, sources (and dashboard) as child processes. Returns them, controller first."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "ADVERTISE_HOST": "127.0.0.1"}
    out = subprocess.DEVNULL if quiet else None
    py = sys.executable
    controller = [py, "-m", "src.distributed.controller", "--config", str(run["config_path"]),
                  "--output-dir", str(run["output_dir"]), "--host", "127.0.0.1", "--port", str(port)]
    if duration is not None:
        controller += ["--duration", str(duration)]
    procs = [subprocess.Popen(controller, cwd=PROJECT_ROOT, env=env, stdout=out, stderr=out)]
    address = f"127.0.0.1:{port}"
    for j in range(run["m"]):
        procs.append(subprocess.Popen([py, "-m", "src.distributed.broker", "--controller", address,
                                       "--instance", f"broker-{j}"], cwd=PROJECT_ROOT, env=env, stdout=out, stderr=out))
    for i in range(run["n"]):
        procs.append(subprocess.Popen([py, "-m", "src.distributed.source", "--controller", address,
                                       "--instance", f"source-{i}"], cwd=PROJECT_ROOT, env=env, stdout=out, stderr=out))
    if dashboard:
        dash_env = {**env, "TRAFFIC_EMULATOR_OUTPUT_DIR": str(Path(run["output_dir"]).resolve()),
                    "TRAFFIC_EMULATOR_OBSERVER": "1"}
        procs.append(subprocess.Popen([py, "-m", "streamlit", "run", "src/dashboard/app.py", "--server.headless",
                                       "true", "--server.port", str(dashboard_port)], cwd=PROJECT_ROOT, env=dash_env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        print(f"  dashboard: http://localhost:{dashboard_port}")
    return procs

def request_shutdown(port: int = CONTROL_PORT, host: str = "127.0.0.1"):
    """Ask a running controller to stop the whole emulation (works the same on every OS)."""
    try:
        with socket.create_connection((host, port), timeout=2) as s:
            s.sendall(b'{"type": "shutdown"}\n')
            s.recv(1024)
    except OSError:
        pass

def stop_local(procs: list, port: int = CONTROL_PORT, grace: float = 10.0):
    """Ask the controller to stop (it tells every worker), then make sure everything exits."""
    if procs[0].poll() is None:
        request_shutdown(port)
    deadline = time.time() + grace
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.terminate()

def run_local(run: dict, port: int = CONTROL_PORT, dashboard: bool = True, dashboard_port: int = 8501,
              duration: float = None) -> int:
    procs = start_local(run, port, dashboard, dashboard_port, duration)
    print(f"  {len(procs)} processes started; Ctrl+C stops everything")
    try:
        code = procs[0].wait()                       # the controller runs until stopped
    except KeyboardInterrupt:
        # Ctrl+C also reaches the children in this console; make sure they all stop
        code = 130
    stop_local(procs, port)
    return code

# ---------------------------------------------------------------- Docker Compose

def compose_command(run: dict, dashboard: bool = True, duration: float = None) -> list:
    services = [] if dashboard else ["controller", "broker", "source"]
    finite = ["--exit-code-from", "controller"] if duration is not None else []   # end with the controller
    return ["docker", "compose", "--profile", "distributed", "up", "--build", *finite,
            "--scale", f"source={run['n']}", "--scale", f"broker={run['m']}", *services]

def compose_env(run: dict, duration: float = None, dashboard_port: int = 8501) -> dict:
    out = Path(run["output_dir"]).resolve()
    try:
        rel = out.relative_to(PROJECT_ROOT / "runs")
    except ValueError:
        raise ValueError("with --backend docker the output directory must be under runs/ (mounted in the containers)")
    rel_posix = rel.as_posix()
    env = {**os.environ, "TE_CONFIG": f"runs/{rel_posix}/resolved_config.yaml", "TE_OUTPUT_DIR": f"runs/{rel_posix}"}
    env["TE_DURATION"] = "" if duration is None else str(duration)
    env["TE_DASHBOARD_PORT"] = str(dashboard_port)
    git = git_revision()                     # the image has no .git; run.json records the host's revision
    env["TE_GIT_SHA"] = git["sha"] or ""
    env["TE_GIT_DIRTY"] = "" if git["dirty"] is None else str(int(git["dirty"]))
    return env

def run_docker(run: dict, dashboard: bool = True, duration: float = None, dashboard_port: int = 8501) -> int:
    if shutil.which("docker") is None:
        print("docker was not found on PATH. Install Docker, or run the same processes locally with --backend local.")
        return 2
    env = compose_env(run, duration, dashboard_port)
    cmd = compose_command(run, dashboard, duration)
    print("  " + " ".join(cmd))
    if dashboard:
        print(f"  dashboard: http://localhost:{dashboard_port}")
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        # Compose got the same Ctrl+C and stops the containers (SIGTERM: each process
        # shuts down cleanly); wait for that before removing them
        try:
            code = proc.wait(timeout=60)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            proc.terminate()
            code = 130
    subprocess.call(["docker", "compose", "--profile", "distributed", "down"], cwd=PROJECT_ROOT, env=env)
    return code
