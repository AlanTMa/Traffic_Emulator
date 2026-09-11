"""Command line: `run` (one process) and `up` (the distributed emulator)."""
import argparse
import copy
import itertools
import numpy as np
import signal
import sys
import time
import yaml
from pathlib import Path
from src.model.config import (CONTROLLER_MODES, MODE_DEFAULTS, controller_mode, load_config, topology_from_config,
                              with_controller_mode)
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

RUNNING = True

def signal_handler(sig, frame):
    global RUNNING
    print("\nstopping")
    RUNNING = False

def run_simulation(config_path: str, real_time: bool = True, output_dir: str = "runs/latest"):
    global RUNNING
    signal.signal(signal.SIGINT, signal_handler)

    config = load_config(config_path)
    topo = topology_from_config(config)
    alg_cfg = config.get('algorithm', {})       # float() below: PyYAML reads 1e-8 as a string
    sim_cfg = config.get('simulation', {})
    window = float(sim_cfg.get('window', 5.0))
    duration = float(sim_cfg.get('duration', 'inf'))

    mode = controller_mode(config)
    seed = resolve_seed(config)
    base_topo = topo
    capacity_model = CapacityVariation.from_config(config, topo, seed)
    if capacity_model is not None:
        topo = capacity_model.topology_at(0.0)
        print(f"Time-varying capacities: link {capacity_model.link}, broker {capacity_model.broker}")

    x_ij = transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    output_dir = Path(output_dir)
    telemetry = TelemetryBuffer(path=output_dir / "metrics.jsonl")
    print(f"Writing telemetry to {output_dir / 'metrics.jsonl'}")

    if mode in ('static_algorithm1', 'capacity_safe_event_driven'):
        params = {"eta": float(alg_cfg.get('eta', 0.25)), "gamma": float(alg_cfg.get('gamma', 0.5)),
                  "delta_s": float(alg_cfg.get('delta_s', 1e-8)), "eps": float(alg_cfg.get('eps', 1e-12)),
                  "window": window}
        if mode == 'capacity_safe_event_driven':
            params["beta"] = float(alg_cfg.get('beta', 0.3))
    else:
        params = {"eta": float(alg_cfg.get('eta', 0.35)), "gamma": float(alg_cfg.get('gamma', 0.5)),
                  "beta": float(alg_cfg.get('beta', 0.3)), "delta_s": None,   # no safe step
                  "window": window, "warmup": int(sim_cfg.get('warmup', 4))}
    write_run_metadata(output_dir, config, base_topo, seed=seed, controller_mode=mode, real_time=real_time,
                       parameters=params, config_path=config_path)
    print(f"Topology: {topo.n_sources} sources, {topo.n_brokers} brokers; controller: {mode}; seed: {seed}")
    if np.isinf(duration):
        print(f"running until Ctrl+C, one window every {window}s", flush=True)
    else:
        print(f"{duration}s, one window every {window}s ({round(duration / window)} iterations)", flush=True)

    if mode == 'static_algorithm1':
        run_static(topo, params, duration, telemetry, real_time, capacity_model)
        telemetry.close()
        return

    engine = SimulationEngine()
    state = SimulationState(len(topo.sources), len(topo.brokers), topo.mu_links, topo.mu_brokers,
                            keep_requests=False)
    if mode == 'capacity_safe_event_driven':
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

    for i in range(len(topo.sources)):
        engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=i))

    def wrapper(event):
        if not RUNNING:
            engine.stop()
            return
        handler.handle_event(event)

    try:
        engine.run(duration=duration, handler=wrapper, real_time=real_time)
    except KeyboardInterrupt:
        pass
    telemetry.close()

    if state.completed:
        print(f"\n{state.completed} work units, mean sojourn {state.latency_sum / state.completed:.4f}s")
    else:
        print("\nno work units completed")

def run_static(topo, params, duration, telemetry, real_time, capacity_model=None):
    """Algorithm 1 on the analytic model, one iteration per window."""
    eta, gamma, eps, delta_s, window = (params[k] for k in ("eta", "gamma", "eps", "delta_s", "window"))

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
            topo = capacity_model.topology_at(k * window)
        state, _, residual = iteration_step(state, topo, eta, gamma, eps=eps, delta_s=delta_s,
                                            on_best_response_failure="hold" if capacity_model else "raise")
        extra = ({"mu_links_t": topo.mu_links, "mu_brokers_t": topo.mu_brokers,
                  "br_failures": list(state.br_failures)} if capacity_model is not None else {})
        record = controller_snapshot(
            topo, state.lambda_ij, state.prices, iteration=k, sim_time=k * window,
            controller_mode="static_algorithm1", s_j=state.s_j, s_t=state.s_t,
            route_rel=state.route_rel, price_rel=state.price_rel, eps=eps, **extra)
        telemetry.record(record)

    print(f"\n{state.iteration} iterations, F = {record['objective']:.10f}, residual {residual:.2e}")
    print(record["certificate_status"])

def configure(config: dict, args) -> dict:
    """Apply the run flags (--controller, --window, --duration, --warmup, --seed, --eta ...) to a copy."""
    config = with_controller_mode(config, args.controller) if args.controller else copy.deepcopy(config)
    mode = controller_mode(config)
    algorithm = config.get("algorithm") or {}
    for key in ("eta", "gamma", "beta", "delta_s"):
        value = getattr(args, key)
        if value is not None:
            if key not in MODE_DEFAULTS[mode]:
                raise ValueError(f"--{key.replace('_', '-')} does not apply to controller {mode}")
            algorithm[key] = value
    config["algorithm"] = algorithm
    simulation = config.get("simulation") or {}
    for key in ("window", "duration", "warmup", "seed"):
        value = getattr(args, key)
        if value is not None:
            simulation[key] = value
    config["simulation"] = simulation
    return config

def generated_config(args) -> dict:
    """Config for a generated N x M topology; the seed drives both the topology and the run."""
    mode = args.controller or "windowed_stochastic"
    seed = 42 if args.seed is None else args.seed
    simulation = {"controller_mode": mode, "window": 5.0, "seed": seed}
    if mode == "windowed_stochastic":
        simulation["warmup"] = 4
    config = {"simulation": simulation, "algorithm": dict(MODE_DEFAULTS[mode]),
              "topology": generate_topology_config(args.sources, args.brokers, args.load, seed)}
    return configure(config, args)

def main():
    parser = argparse.ArgumentParser(description="Traffic emulator")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="run in one process",
        description="Run a config, or generate an N x M topology. The config actually run is written "
                    "to the output directory.")
    source = run_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=str, help="YAML config")
    source.add_argument("--sources", type=int, help="generate a topology with this many sources")
    run_parser.add_argument("--brokers", type=int, help="brokers in the generated topology")
    run_parser.add_argument("--load", type=float, default=0.3, help="generated: offered rate / broker capacity (0.3)")
    run_parser.add_argument("--seed", type=int, help="run seed, and the topology's when generated (42)")
    run_parser.add_argument("--controller", choices=CONTROLLER_MODES, help="controller mode (default: the config's)")
    run_parser.add_argument("--window", type=float, help="seconds per window (5)")
    run_parser.add_argument("--duration", type=float, help="simulated seconds (until stopped)")
    run_parser.add_argument("--warmup", type=int, help="warm-up windows, windowed_stochastic (4)")
    for name in ("eta", "gamma", "beta", "delta-s"):
        run_parser.add_argument(f"--{name}", type=float, help=f"override {name}")
    run_parser.add_argument("--no-realtime", action="store_true", help="don't pace to the wall clock")
    run_parser.add_argument("--output-dir", default="runs/latest", help="(runs/latest)")

    up_parser = subparsers.add_parser(
        "up", help="start the distributed emulator",
        description="One controller, one process per source and per broker, and the dashboard, as "
                    "containers or local processes. Runs until Ctrl+C.")
    up_parser.add_argument("--config", help="topology config (config/paper_5x3.yaml, unless --sources/--brokers "
                                            "generate one)")
    up_parser.add_argument("--sources", type=int, help="source processes (must match --config)")
    up_parser.add_argument("--brokers", type=int, help="broker processes")
    up_parser.add_argument("--load", type=float, default=0.3, help="generated: offered rate / broker capacity (0.3)")
    up_parser.add_argument("--seed", type=int, default=42, help="root seed (42)")
    up_parser.add_argument("--backend", choices=["docker", "local"], default="docker",
                           help="docker (default) or local processes")
    up_parser.add_argument("--window", type=float, help="seconds between rounds (5)")
    for name in ("eta", "gamma", "delta-s"):
        up_parser.add_argument(f"--{name}", type=float, help=f"override {name}")
    up_parser.add_argument("--output-dir", default="runs/distributed", help="(runs/distributed)")
    up_parser.add_argument("--no-dashboard", action="store_true")
    up_parser.add_argument("--port", type=int, default=7000, help="controller port, local backend (7000)")
    up_parser.add_argument("--dashboard-port", type=int, default=8501, help="(8501)")
    up_parser.add_argument("--duration", type=float, help="stop after this many seconds (until Ctrl+C)")

    args = parser.parse_args()

    if args.command == "up":
        from src.distributed import launcher
        sys.stdout.reconfigure(line_buffering=True)          # stay in order with docker's output
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
        config_path, written = args.config, None
        try:
            if args.sources is not None:
                if args.brokers is None:
                    parser.error("--sources needs --brokers")
                config, written = generated_config(args), "generated_config.yaml"
            else:
                base = load_config(args.config)
                config = configure(base, args)
                if config != base:
                    written = "resolved_config.yaml"
                    if controller_mode(config) != controller_mode(base):
                        print(f"controller {controller_mode(config)}, config is for {controller_mode(base)}; "
                              f"parameters {config['algorithm']}")
        except ValueError as e:
            parser.error(str(e))
        if written:
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            config_path = out / written
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            if args.sources is not None:
                print(f"generated {args.sources}x{args.brokers} at load {args.load:.0%}, seed {config['simulation']['seed']}")
            print(f"config: {config_path}")
        run_simulation(str(config_path), real_time=not args.no_realtime, output_dir=args.output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
