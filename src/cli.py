"""
Command-line interface for the traffic allocation emulator.
"""
import argparse
import numpy as np
import signal
import sys
import time
from src.model.config import load_config, topology_from_config
from src.simulation.engine import SimulationEngine
from src.simulation.events import Event, EventType
from src.simulation.queues import SimulationState
from src.simulation.handler import SimulationHandler
from src.controller.feasibility import transportation_feasibility
from src.telemetry.metrics import TelemetryBuffer
from src.controller.synchronous import iteration_step
from src.simulation.state import SystemState
from src.model.marginal_costs import mm1_marginal_cost_vectorized

# Global flag for graceful shutdown
RUNNING = True

def signal_handler(sig, frame):
    global RUNNING
    print("\n[SIGINT] Shutdown signal received. Stopping simulation...")
    RUNNING = False

def run_simulation(config_path: str, real_time: bool = True):
    global RUNNING
    signal.signal(signal.SIGINT, signal_handler)

    # 1. Load configuration
    config = load_config(config_path)
    topo = topology_from_config(config)
    # float(): PyYAML reads values like 1e-8 (no decimal point) as strings
    alg_cfg = config.get('algorithm', {})
    sim_cfg = config.get('simulation', {})
    window = float(sim_cfg.get('window', 5.0))
    # Default to a very large duration for "infinite" runs
    duration = float(sim_cfg.get('duration', 1e12))

    # 2. Initialization
    x_ij = transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    telemetry = TelemetryBuffer()

    print(f"Topology: {topo.n_sources} sources, {topo.n_brokers} brokers")
    print(f"Duration: {duration}s, controller every {window}s (~{round(duration / window)} iterations). Press Ctrl+C to stop.")

    if sim_cfg.get('mode') == 'static':
        run_static(topo, x_ij, alg_cfg, window, duration, telemetry, real_time)
        return

    # Create the event-driven simulation components
    engine = SimulationEngine()
    state = SimulationState(len(topo.sources), len(topo.brokers), topo.mu_links, topo.mu_brokers)
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

    # 5. Final Results
    print("\n=== Simulation Complete ===")
    latencies = []
    for req in state.requests.values():
        if "broker_complete" in req:
            latencies.append(req["broker_complete"] - req["arrival"])

    if latencies:
        print(f"Processed {len(latencies)} requests.")
        print(f"Mean End-to-End Latency: {np.mean(latencies):.4f}s")
    else:
        print("No requests completed during the simulation.")

def run_static(topo, x_ij, alg_cfg, window, duration, telemetry, real_time):
    """
    Run the deterministic synchronous solver (no packets), one iteration per
    window. Reproduces the notebook's static distributed convergence plots.
    """
    eta = float(alg_cfg.get('eta', 0.25))
    gamma = float(alg_cfg.get('gamma', 0.5))
    eps = float(alg_cfg.get('eps', 1e-12))

    loads = x_ij.sum(axis=0)
    state = SystemState(lambda_ij=x_ij, prices=mm1_marginal_cost_vectorized(loads, topo.mu_brokers, eps))
    wall_start = time.perf_counter()
    for k in range(1, round(duration / window) + 1):
        if not RUNNING:
            break
        if real_time:
            time.sleep(max(0.0, wall_start + k * window - time.perf_counter()))

        state, _, residual = iteration_step(state, topo, eta, gamma, eps)
        L = state.lambda_ij
        L_j = L.sum(axis=0)
        obj = np.sum(L / (topo.mu_links - L)) + np.sum(L_j / (topo.mu_brokers - L_j))
        util = np.max(L_j / topo.mu_brokers)
        telemetry.record(timestamp=k * window, iteration=k, state_data={
            "objective": obj, "max_util": util, "max_util_planned": util, "rel_change": residual,
        })
        telemetry.save_to_csv("simulation_metrics.csv")

    print("\n=== Static Solve Complete ===")
    print(f"Iterations: {state.iteration}, objective: {obj:.6f}, final residual: {residual:.2e}")

def main():
    parser = argparse.ArgumentParser(description="Traffic Allocation Emulator")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a simulation")
    run_parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    run_parser.add_argument("--no-realtime", action="store_true",
                            help="Run as fast as possible instead of pacing to the wall clock")

    args = parser.parse_args()

    if args.command == "run":
        run_simulation(args.config, real_time=not args.no_realtime)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
