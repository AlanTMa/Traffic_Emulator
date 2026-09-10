"""
Command-line interface for the traffic allocation emulator.
"""
import argparse
import itertools
import numpy as np
import signal
import sys
import time
import yaml
from pathlib import Path
from src.model.config import CONTROLLER_MODES, controller_mode, load_config, topology_from_config
from src.model.generate import generate_topology_config
from src.model.dynamics import CapacityVariation
from src.simulation.engine import SimulationEngine
from src.simulation.events import Event, EventType
from src.simulation.queues import SimulationState
from src.simulation.handler import SimulationHandler
from src.controller.feasibility import transportation_feasibility
from src.telemetry.metrics import TelemetryBuffer
from src.telemetry.schema import controller_snapshot
from src.controller.synchronous import algorithm1_initial_state, iteration_step
from src.runtime.metadata import resolve_seed, write_run_metadata

# Global flag for graceful shutdown
RUNNING = True

def signal_handler(sig, frame):
    global RUNNING
    print("\n[SIGINT] Shutdown signal received. Stopping simulation...")
    RUNNING = False

def run_simulation(config_path: str, real_time: bool = True, output_dir: str = "runs/latest"):
    global RUNNING
    signal.signal(signal.SIGINT, signal_handler)

    # 1. Load configuration
    config = load_config(config_path)
    topo = topology_from_config(config)
    # float(): PyYAML reads values like 1e-8 (no decimal point) as strings
    alg_cfg = config.get('algorithm', {})
    sim_cfg = config.get('simulation', {})
    window = float(sim_cfg.get('window', 5.0))
    # No duration: run until stopped (Ctrl+C, or Stop in the dashboard)
    duration = float(sim_cfg.get('duration', 'inf'))

    # Optional time-varying capacities (dynamics.capacity_variation); topo is
    # then the topology in effect at t = 0
    mode = controller_mode(config)
    seed = resolve_seed(config)
    base_topo = topo
    capacity_model = CapacityVariation.from_config(config, topo, seed)
    if capacity_model is not None:
        topo = capacity_model.topology_at(0.0)
        print(f"Time-varying capacities: link {capacity_model.link}, broker {capacity_model.broker}")

    # 2. Initialization
    x_ij = transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    output_dir = Path(output_dir)
    telemetry = TelemetryBuffer(path=output_dir / "metrics.jsonl")
    print(f"Writing telemetry to {output_dir / 'metrics.jsonl'}")

    if mode in ('static_algorithm1', 'capacity_safe_event_driven'):
        params = {"eta": float(alg_cfg.get('eta', 0.25)), "gamma": float(alg_cfg.get('gamma', 0.5)),
                  "delta_s": float(alg_cfg.get('delta_s', 1e-8)),   # Algorithm 1 capacity margin
                  "eps": float(alg_cfg.get('eps', 1e-12)),          # numerical guard only
                  "window": window}
        if mode == 'capacity_safe_event_driven':
            params["beta"] = float(alg_cfg.get('beta', 0.3))       # measured-rate telemetry only
    else:
        params = {"eta": float(alg_cfg.get('eta', 0.35)), "gamma": float(alg_cfg.get('gamma', 0.5)),
                  "beta": float(alg_cfg.get('beta', 0.3)), "delta_s": None,   # no safe step in this mode
                  "window": window, "warmup": int(sim_cfg.get('warmup', 4))}
    write_run_metadata(output_dir, config, base_topo, seed=seed, controller_mode=mode, real_time=real_time,
                       parameters=params, config_path=config_path)
    print(f"Topology: {topo.n_sources} sources, {topo.n_brokers} brokers; controller: {mode}; seed: {seed}")
    if np.isinf(duration):
        print(f"Running until stopped, controller every {window}s. Press Ctrl+C to stop.", flush=True)
    else:
        print(f"Duration: {duration}s, controller every {window}s (~{round(duration / window)} iterations). Press Ctrl+C to stop.", flush=True)

    if mode == 'static_algorithm1':
        run_static(topo, params, duration, telemetry, real_time, capacity_model)
        telemetry.close()
        return

    # Create the event-driven simulation components
    engine = SimulationEngine()
    state = SimulationState(len(topo.sources), len(topo.brokers), topo.mu_links, topo.mu_brokers,
                            keep_requests=False)
    if mode == 'capacity_safe_event_driven':
        # Algorithm 1 on planned rates; initial routing from the LP with margin delta_s
        x_ij = transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers,
                                          margin=params["delta_s"])
        handler = SimulationHandler(
            engine, topo, state, x_ij, telemetry=telemetry,
            eta=params["eta"], gamma=params["gamma"], beta=params["beta"], window=window,
            eps=params["eps"], delta_s=params["delta_s"], controller_mode=mode,
            rng=np.random.default_rng(seed), capacity_model=capacity_model,
        )
    else:
        handler = SimulationHandler(
            engine, topo, state, x_ij, telemetry=telemetry,
            eta=params["eta"], gamma=params["gamma"], beta=params["beta"],
            window=window, warmup_windows=params["warmup"],
            rng=np.random.default_rng(seed), capacity_model=capacity_model,
        )

    # Initial events: first arrivals for all sources
    for i in range(len(topo.sources)):
        engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=i))

    print(f"Starting event-driven simulation from config: {config_path}")

    # 4. Execution
    def wrapper(event):
        if not RUNNING:
            engine.stop()
            return
        handler.handle_event(event)

    try:
        # real_time paces simulated time to the wall clock so the live
        # dashboard updates as the run progresses.
        engine.run(duration=duration, handler=wrapper, real_time=real_time)
    except KeyboardInterrupt:
        pass
    telemetry.close()

    # 5. Final Results
    print("\n=== Simulation Complete ===")
    if state.completed:
        print(f"Completed {state.completed} work units.")
        print(f"Mean end-to-end work-unit sojourn time: {state.latency_sum / state.completed:.4f}s")
    else:
        print("No work units completed during the simulation.")

def run_static(topo, params, duration, telemetry, real_time, capacity_model=None):
    """
    static_algorithm1: paper Algorithm 1 on the analytic model (no events or queues),
    one iteration per window. Reproduces the notebook's static convergence plots.
    """
    eta, gamma, eps, delta_s, window = (params[k] for k in ("eta", "gamma", "eps", "delta_s", "window"))

    # Algorithm 1 initialization: LP routing with Lambda_j <= mu_j - delta_s, model prices
    state = algorithm1_initial_state(topo, delta_s, eps)
    wall_start = time.perf_counter()
    n_iter = None if np.isinf(duration) else round(duration / window)
    for k in itertools.count(1):
        if n_iter is not None and k > n_iter:
            break
        if not RUNNING:
            break
        if real_time:
            time.sleep(max(0.0, wall_start + k * window - time.perf_counter()))

        if capacity_model is not None:
            topo = capacity_model.topology_at(k * window)   # capacities for this iteration
        state, _, residual = iteration_step(state, topo, eta, gamma, eps=eps, delta_s=delta_s,
                                            on_best_response_failure="hold" if capacity_model else "raise")
        extra = ({"mu_links_t": topo.mu_links, "mu_brokers_t": topo.mu_brokers,
                  "br_failures": list(state.br_failures)} if capacity_model is not None else {})
        record = controller_snapshot(
            topo, state.lambda_ij, state.prices, iteration=k, sim_time=k * window,
            controller_mode="static_algorithm1", s_j=state.s_j, s_t=state.s_t,
            route_rel=state.route_rel, price_rel=state.price_rel, eps=eps, **extra)
        telemetry.record(record)

    print("\n=== Static Solve Complete ===")
    print(f"Iterations: {state.iteration}, objective: {record['objective']:.10f}, final residual: {residual:.2e}")
    print(record["certificate_status"])

# Per-mode defaults for generated runs (same as the shipped configs)
GENERATED_DEFAULTS = {
    "static_algorithm1": {"eta": 0.25, "gamma": 0.5, "delta_s": 1e-8, "eps": 1e-12},
    "capacity_safe_event_driven": {"eta": 0.25, "gamma": 0.5, "delta_s": 1e-8, "eps": 1e-12, "beta": 0.3},
    "windowed_stochastic": {"eta": 0.35, "gamma": 0.5, "beta": 0.3},
}

def generated_config(args) -> dict:
    """
    Config for a generated N x M topology (src/model/generate.py). The same
    seed drives topology generation and the event simulation's RNG, so the
    written config reproduces the run.
    """
    algorithm = dict(GENERATED_DEFAULTS[args.controller])
    for key in ("eta", "gamma", "beta", "delta_s"):
        value = getattr(args, key)
        if value is not None:
            if key not in algorithm:
                raise ValueError(f"--{key.replace('_', '-')} does not apply to controller {args.controller}")
            algorithm[key] = value
    simulation = {"controller_mode": args.controller, "window": args.window, "seed": args.seed}
    if args.duration is not None:
        simulation["duration"] = args.duration
    if args.controller == "windowed_stochastic":
        simulation["warmup"] = args.warmup
    return {
        "simulation": simulation,
        "algorithm": algorithm,
        "topology": generate_topology_config(args.sources, args.brokers, args.load, args.seed),
    }

def main():
    parser = argparse.ArgumentParser(description="Traffic Allocation Emulator")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="Run a simulation",
        description="Run a config (--config), or generate an N x M topology (--sources/--brokers/...). "
                    "Generated runs write their full config to <output-dir>/generated_config.yaml.")
    source = run_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=str, help="Path to YAML config file")
    source.add_argument("--sources", type=int, help="Generate a topology with this many sources")
    run_parser.add_argument("--brokers", type=int, help="Brokers in the generated topology")
    run_parser.add_argument("--load", type=float, default=0.3,
                            help="Generated: total offered rate / total broker capacity (default 0.3)")
    run_parser.add_argument("--seed", type=int, default=42,
                            help="Generated: seed for the topology and the simulation (default 42)")
    run_parser.add_argument("--controller", choices=CONTROLLER_MODES, default="windowed_stochastic",
                            help="Generated: controller mode (default windowed_stochastic)")
    run_parser.add_argument("--window", type=float, default=5.0, help="Generated: seconds per window (default 5)")
    run_parser.add_argument("--duration", type=float, help="Generated: simulated seconds (default: until stopped)")
    run_parser.add_argument("--warmup", type=int, default=4, help="Generated windowed_stochastic: warm-up windows")
    for name in ("eta", "gamma", "beta", "delta-s"):
        run_parser.add_argument(f"--{name}", type=float, help=f"Generated: override {name} (mode defaults otherwise)")
    run_parser.add_argument("--no-realtime", action="store_true",
                            help="Run as fast as possible instead of pacing to the wall clock")
    run_parser.add_argument("--output-dir", default="runs/latest",
                            help="Directory for metrics.jsonl (default: runs/latest)")

    up_parser = subparsers.add_parser(
        "up", help="Start the distributed emulator (controller, sources, brokers, dashboard)",
        description="One command for the distributed emulator: one controller process, one process per source "
                    "and per broker, and the dashboard. Algorithm 1 (paper) drives the routing of live traffic "
                    "over explicit access and broker queues. Runs until Ctrl+C.")
    up_parser.add_argument("--config", help="config whose topology is used (default config/paper_5x3.yaml "
                                            "unless --sources/--brokers generate one)")
    up_parser.add_argument("--sources", type=int, help="number of source processes (must match --config, "
                                                       "or generates an N x M topology)")
    up_parser.add_argument("--brokers", type=int, help="number of broker processes")
    up_parser.add_argument("--load", type=float, default=0.3, help="generated topology: offered/broker capacity")
    up_parser.add_argument("--seed", type=int, default=42, help="generated topology and root RNG seed")
    up_parser.add_argument("--backend", choices=["docker", "local"], default="docker",
                           help="docker: Docker Compose, one container per process (default); "
                                "local: the same processes on this machine")
    up_parser.add_argument("--window", type=float, help="seconds between controller rounds (config default 5)")
    for name in ("eta", "gamma", "delta-s"):
        up_parser.add_argument(f"--{name}", type=float, help=f"override Algorithm 1 {name}")
    up_parser.add_argument("--output-dir", default="runs/distributed", help="telemetry and run metadata")
    up_parser.add_argument("--no-dashboard", action="store_true")
    up_parser.add_argument("--port", type=int, default=7000, help="local backend: controller port")
    up_parser.add_argument("--dashboard-port", type=int, default=8501, help="dashboard port on this machine")
    up_parser.add_argument("--duration", type=float, help="stop after this many seconds (default: run until Ctrl+C)")

    args = parser.parse_args()

    if args.command == "up":
        from src.distributed import launcher
        config = args.config or (None if args.sources is not None or args.brokers is not None
                                 else "config/paper_5x3.yaml")
        try:
            run = launcher.prepare_run(config, args.sources, args.brokers, args.load, args.seed, args.output_dir,
                                       args.window, {"eta": args.eta, "gamma": args.gamma, "delta_s": args.delta_s})
        except ValueError as e:
            parser.error(str(e))
        launcher.describe(run)
        if args.backend == "local":
            sys.exit(launcher.run_local(run, args.port, not args.no_dashboard, args.dashboard_port, args.duration))
        sys.exit(launcher.run_docker(run, not args.no_dashboard, args.duration, args.dashboard_port))

    if args.command == "run":
        config_path = args.config
        if args.sources is not None:
            if args.brokers is None:
                parser.error("--sources needs --brokers")
            try:
                config = generated_config(args)
            except ValueError as e:
                parser.error(str(e))
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            config_path = out / "generated_config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(f"Generated {args.sources}x{args.brokers} topology at load {args.load:.0%} (seed {args.seed}); "
                  f"config written to {config_path}")
        run_simulation(str(config_path), real_time=not args.no_realtime, output_dir=args.output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
