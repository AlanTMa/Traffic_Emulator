"""
Command-line interface for the traffic allocation emulator.
"""
import argparse
import numpy as np
import signal
import sys
import time
from src.model.config import load_config, topology_from_config
from src.simulation.state import SystemState
from src.controller.synchronous import iteration_step
from src.controller.feasibility import transportation_feasibility
from src.telemetry.metrics import TelemetryBuffer

# Global flag for graceful shutdown
RUNNING = True

def signal_handler(sig, frame):
    global RUNNING
    print("\n[SIGINT] Shutdown signal received. Flushing metrics and exiting...")
    RUNNING = False

def run_simulation(config_path: str):
    global RUNNING
    signal.signal(signal.SIGINT, signal_handler)

    # 1. Load configuration
    config = load_config(config_path)
    topo = topology_from_config(config)

    # 2. Initialization
    L0 = transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    loads = L0.sum(axis=0)
    p0 = topo.mu_brokers / (topo.mu_brokers - loads)**2
    state = SystemState(lambda_ij=L0, prices=p0)

    # 3. Algorithm Parameters
    algo = config['algorithm']
    eta = float(algo['eta'])
    gamma = float(algo['gamma'])
    eps = float(algo['eps'])
    tol = float(algo['convergence_tol'])

    # Telemetry
    telemetry = TelemetryBuffer()

    print(f"Starting continuous simulation from config: {config_path}")
    print(f"Topology: {topo.n_sources} sources, {topo.n_brokers} brokers")
    print("Press Ctrl+C to stop.")

    # 4. Continuous Loop
    iteration = 0
    start_time = time.time()

    while RUNNING:
        # Controller Tick
        state, s_t, rel_change = iteration_step(state, topo, eta, gamma, eps)
        iteration += 1

        # Calculate objective for telemetry
        l_mat = state.lambda_ij
        loads_final = l_mat.sum(axis=0)
        obj = np.sum(l_mat / (topo.mu_links - l_mat)) + np.sum(loads_final / (topo.mu_brokers - loads_final))

        # Record telemetry
        telemetry.record(
            timestamp=time.time() - start_time,
            iteration=iteration,
            state_data={
                "objective": obj,
                "rel_change": rel_change,
                "max_util": np.max(loads_final / topo.mu_brokers),
                "s_t": s_t
            }
        )

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Obj={obj:.6f}, rel_change={rel_change:.3e}")

        if iteration >= 100:
            print("Test limit reached (100 iterations). Shutting down.")
            break

        # In a real emulator, we would sleep here to match the window duration
        # time.sleep(config['simulation'].get('window', 1))

    # 5. Shutdown and Save
    print("\n=== Final Results ===")
    print(f"Final Objective: {obj:.6f}")
    telemetry.save_to_csv("simulation_metrics.csv")
    print("Metrics saved to simulation_metrics.csv")

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
