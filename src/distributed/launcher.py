"""
The `up` command: resolve the run into <output-dir>/resolved_config.yaml,
then start the controller, M brokers, N sources and the dashboard as Docker
Compose services (docker-compose.yml) or as child processes (--backend
local). Ctrl+C stops everything.
"""
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
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
    topo = topology_from_config(resolved)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "resolved_config.yaml"
    path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return {"config_path": path, "output_dir": out, "n": topo.n_sources, "m": topo.n_brokers,
            "settings": settings, "source": config_path or f"generated {sources}x{brokers}"}

def describe(run: dict):
    p = run["settings"]["params"]
    print(f"{run['n']} sources x {run['m']} brokers from {run['source']}", flush=True)
    print(f"  Algorithm 1: eta={p['eta']} gamma={p['gamma']} delta_s={p['delta_s']} "
          f"({p['safe_step_variant']} safe step), round every {p['window']} s, seed {run['settings']['seed']}")
    for note in run["settings"]["notes"]:
        print(f"  note: {note}")
    print(f"  resolved config: {run['config_path']}; telemetry: {run['output_dir'] / 'metrics.jsonl'}")

def lan_address():
    """This machine's address on its local network (the interface of the default route), or None."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))                 # UDP: picks the route, sends nothing
            address = s.getsockname()[0]
    except OSError:
        return None
    return None if address.startswith("127.") else address

def print_dashboard_urls(port: int):
    lines = ["", f"dashboard: http://localhost:{port}"]
    address = lan_address()
    if address:
        lines.append(f"           http://{address}:{port} from this network")
    print("\n".join(lines + [""]), flush=True)

def port_in_use(port: int) -> bool:
    """Whether something on this machine already accepts connections on the port."""
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

def dashboard_ready(port: int, timeout: float) -> bool:
    """Wait until the dashboard on this machine answers its health check."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as reply:
                if reply.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False

def first_round_written(metrics: Path, since: float, timeout: float) -> bool:
    """Wait until this run's controller has written its first telemetry record."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            info = metrics.stat()
            if info.st_size > 0 and info.st_mtime >= since:
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False

def announce_dashboard(port: int, hint: str, timeout: float, metrics: Path = None):
    """Print the dashboard URLs once it answers and the first round is written; from a
    thread, so they land below the startup output instead of scrolling away."""
    since = time.time()
    def wait():
        if not dashboard_ready(port, timeout):
            print(f"no dashboard on port {port} after {timeout:.0f}s; {hint}", flush=True)
            return
        if metrics is not None:
            first_round_written(metrics, since, timeout=120)
        print_dashboard_urls(port)
    threading.Thread(target=wait, daemon=True).start()

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
        announce_dashboard(dashboard_port, "port in use? try --dashboard-port",
                           timeout=60, metrics=Path(run["output_dir"]) / "metrics.jsonl")
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
    for used, flag in ((port, "--port"), (dashboard_port, "--dashboard-port")):
        if (used != dashboard_port or dashboard) and port_in_use(used):
            print(f"port {used} is already in use; stop what holds it or pass {flag}")
            return 2
    procs = start_local(run, port, dashboard, dashboard_port, duration)
    print(f"  {len(procs)} processes; Ctrl+C stops everything")
    try:
        code = procs[0].wait()
    except KeyboardInterrupt:
        code = 130
    stop_local(procs, port)
    return code

# ---------------------------------------------------------------- Docker Compose

def compose_command(run: dict, dashboard: bool = True, duration: float = None) -> list:
    services = [] if dashboard else ["controller", "broker", "source"]
    finite = ["--exit-code-from", "controller"] if duration is not None else []   # end with the controller
    # the dashboard container's own URLs are unusable from the host
    quiet = ["--no-attach", "dashboard"] if dashboard else []
    return ["docker", "compose", "--profile", "distributed", "up", "--build", *finite, *quiet,
            "--scale", f"source={run['n']}", "--scale", f"broker={run['m']}", *services]

def docker_executable():
    """docker on PATH, or in Docker Desktop's install folders (a terminal opened
    before the install still has the old PATH)."""
    found = shutil.which("docker")
    if found:
        return found
    exe = "docker.exe" if os.name == "nt" else "docker"
    folders = [Path(os.environ.get("LOCALAPPDATA", "~")) / "Programs" / "DockerDesktop" / "resources" / "bin",
               Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Docker" / "Docker" / "resources" / "bin",
               Path("/Applications/Docker.app/Contents/Resources/bin"), Path("/usr/local/bin")]
    for folder in folders:
        if (folder / exe).is_file():
            return str(folder / exe)
    return None

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
    git = git_revision()                     # the image has no .git
    env["TE_GIT_SHA"] = git["sha"] or ""
    env["TE_GIT_DIRTY"] = "" if git["dirty"] is None else str(int(git["dirty"]))
    return env

def run_docker(run: dict, dashboard: bool = True, duration: float = None, dashboard_port: int = 8501) -> int:
    docker = docker_executable()
    if docker is None:
        print("docker not found; install Docker Desktop or use --backend local")
        return 2
    env = compose_env(run, duration, dashboard_port)
    env["PATH"] = str(Path(docker).parent) + os.pathsep + env.get("PATH", "")   # compose plugin, credential helper
    try:
        running = subprocess.run([docker, "info"], env=env, capture_output=True, timeout=30).returncode == 0
    except subprocess.TimeoutExpired:
        running = False
    if not running:
        print("docker is not running")
        return 2
    # a second compose up would take over the running containers
    active = subprocess.run([docker, "compose", "--profile", "distributed", "--profile", "reference", "ps", "-q"],
                            cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    if active.stdout.strip():
        print("an emulator is already running (docker compose --profile distributed down)")
        return 2
    if dashboard and port_in_use(dashboard_port):
        print(f"port {dashboard_port} is already in use; pass --dashboard-port")
        return 2
    cmd = compose_command(run, dashboard, duration)
    print("  " + " ".join(cmd))
    if dashboard:
        announce_dashboard(dashboard_port, "docker compose --profile distributed logs dashboard",
                           timeout=900,                           # the first build takes a few minutes
                           metrics=Path(run["output_dir"]) / "metrics.jsonl")
    proc = subprocess.Popen([docker, *cmd[1:]], cwd=PROJECT_ROOT, env=env)
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        # compose got the same Ctrl+C and is stopping the containers; wait for it
        try:
            code = proc.wait(timeout=60)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            proc.terminate()
            code = 130
    subprocess.call([docker, "compose", "--profile", "distributed", "down"], cwd=PROJECT_ROOT, env=env)
    return code
