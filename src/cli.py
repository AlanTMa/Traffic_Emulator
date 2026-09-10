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

# Global flag for graceful shutdown
RUNNING = True

def signal_handler(sig, frame):
    global RUNNING
    print("\n[SIGINT] Shutdown signal received. Stopping simulation...")
    RUNNING = False

def run_simulation(config_path: str):
    global RUNNING
    signal.signal(signal.SIGINT, signal_handler)

    # 1. Load configuration
    config = load_config(config_path)
    topo = topology_from_config(config)

    # 2. Initialization
    x_ij = transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers)

    # Create the event-driven simulation components
    engine = SimulationEngine()
    state = SimulationState(len(topo.sources), len(topo.brokers), topo.mu_links, topo.mu_brokers)

    # Telemetry
    telemetry = TelemetryBuffer()
    # float(): PyYAML reads values like 1e-8 (no decimal point) as strings
    alg_cfg = config.get('algorithm', {})
    handler = SimulationHandler(
        engine, topo, state, x_ij, telemetry=telemetry,
        eta=float(alg_cfg.get('eta', 0.25)),
        gamma=float(alg_cfg.get('gamma', 0.5)),
        eps=float(alg_cfg.get('eps', 1e-12)),
    )

    # Initial events: first arrivals for all sources
    for i in range(len(topo.sources)):
        engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=i))

    # 3. Simulation Parameters
    sim_cfg = config.get('simulation', {})
    # Default to a very large duration for "infinite" runs
    duration = float(sim_cfg.get('duration', 1e12))

    print(f"Starting event-driven simulation from config: {config_path}")
    print(f"Topology: {topo.n_sources} sources, {topo.n_brokers} brokers")
    print(f"Duration: {duration}s. Press Ctrl+C to stop.")

    # 4. Execution
    def wrapper(event):
        if not RUNNING:
            return
        handler.handle_event(event)

    try:
        # Set real_time=True so the simulation matches the wall clock
        # This allows the live dashboard to update in real-time.
        engine.run(duration=duration, handler=wrapper, real_time=True)
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

def main():
    parser = argparse.ArgumentParser(description="Traffic Allocation Emulator")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a simulation")
    run_parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    args = parser.parse_args()

    if args.command == "run":
        run_simulation(args.config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
