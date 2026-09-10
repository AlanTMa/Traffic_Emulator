"""
Command-line interface for the traffic allocation emulator.
"""
import argparse
import itertools
import numpy as np
import signal
import sys
import time
from pathlib import Path
from src.model.config import controller_mode, load_config, topology_from_config
from src.simulation.engine import SimulationEngine
from src.simulation.events import Event, EventType
from src.simulation.queues import SimulationState
from src.simulation.handler import SimulationHandler
from src.controller.feasibility import transportation_feasibility
from src.telemetry.metrics import TelemetryBuffer
from src.telemetry.schema import controller_snapshot
from src.controller.synchronous import iteration_step
from src.simulation.state import SystemState
from src.model.marginal_costs import mm1_marginal_cost_vectorized

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

    # 2. Initialization
    x_ij = transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    output_dir = Path(output_dir)
    telemetry = TelemetryBuffer(path=output_dir / "metrics.jsonl")
    print(f"Writing telemetry to {output_dir / 'metrics.jsonl'}")

    mode = controller_mode(config)
    print(f"Topology: {topo.n_sources} sources, {topo.n_brokers} brokers; controller: {mode}")
    if np.isinf(duration):
        print(f"Running until stopped, controller every {window}s. Press Ctrl+C to stop.", flush=True)
    else:
        print(f"Duration: {duration}s, controller every {window}s (~{round(duration / window)} iterations). Press Ctrl+C to stop.", flush=True)

    if mode == 'static_algorithm1':
        run_static(topo, x_ij, alg_cfg, window, duration, telemetry, real_time)
        telemetry.close()
        return

    # Create the event-driven simulation components
    engine = SimulationEngine()
    state = SimulationState(len(topo.sources), len(topo.brokers), topo.mu_links, topo.mu_brokers,
                            keep_requests=False)
    handler = SimulationHandler(
        engine, topo, state, x_ij, telemetry=telemetry,
        eta=float(alg_cfg.get('eta', 0.35)),
        gamma=float(alg_cfg.get('gamma', 0.5)),
        beta=float(alg_cfg.get('beta', 0.3)),
        window=window,
        warmup_windows=int(sim_cfg.get('warmup', 4)),
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

def run_static(topo, x_ij, alg_cfg, window, duration, telemetry, real_time):
    """
    static_algorithm1: paper Algorithm 1 on the analytic model (no events or queues),
    one iteration per window. Reproduces the notebook's static convergence plots.
    """
    eta = float(alg_cfg.get('eta', 0.25))
    gamma = float(alg_cfg.get('gamma', 0.5))
    eps = float(alg_cfg.get('eps', 1e-12))          # numerical guard only
    delta_s = float(alg_cfg.get('delta_s', 1e-8))   # Algorithm 1 capacity margin

    # Algorithm 1 requires an initial routing with Lambda_j <= mu_j - delta_s
    x_ij = transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers, margin=delta_s)
    loads = x_ij.sum(axis=0)
    state = SystemState(lambda_ij=x_ij, prices=mm1_marginal_cost_vectorized(loads, topo.mu_brokers, eps))
    wall_start = time.perf_counter()
    n_iter = None if np.isinf(duration) else round(duration / window)
    for k in itertools.count(1):
        if n_iter is not None and k > n_iter:
            break
        if not RUNNING:
            break
        if real_time:
            time.sleep(max(0.0, wall_start + k * window - time.perf_counter()))

        state, _, residual = iteration_step(state, topo, eta, gamma, eps=eps, delta_s=delta_s)
        record = controller_snapshot(
            topo, state.lambda_ij, state.prices, iteration=k, sim_time=k * window,
            controller_mode="static_algorithm1", s_j=state.s_j, s_t=state.s_t,
            route_rel=state.route_rel, price_rel=state.price_rel, eps=eps)
        telemetry.record(record)

    print("\n=== Static Solve Complete ===")
    print(f"Iterations: {state.iteration}, objective: {record['objective']:.10f}, final residual: {residual:.2e}")
    print(record["certificate_status"])

def main():
    parser = argparse.ArgumentParser(description="Traffic Allocation Emulator")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a simulation")
    run_parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    run_parser.add_argument("--no-realtime", action="store_true",
                            help="Run as fast as possible instead of pacing to the wall clock")
    run_parser.add_argument("--output-dir", default="runs/latest",
                            help="Directory for metrics.jsonl (default: runs/latest)")

    args = parser.parse_args()

    if args.command == "run":
        run_simulation(args.config, real_time=not args.no_realtime, output_dir=args.output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
