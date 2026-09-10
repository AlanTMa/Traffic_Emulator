import numpy as np
import pandas as pd
from typing import List, Dict
from src.simulation.engine import SimulationEngine
from src.simulation.events import Event, EventType
from src.simulation.queues import SimulationState
from src.simulation.handler import SimulationHandler
from src.model.topology import Topology

def theoretical_delay(lam: float, mu_link: float, mu_broker: float) -> float:
    """Theoretical mean end-to-end delay for 1 source, 1 broker M/M/1 system."""
    return 1.0 / (mu_link - lam) + 1.0 / (mu_broker - lam)

def run_simulation(lam: float, mu_link: float, mu_broker: float, trials: int = 3) -> float:
    """Runs the simulation for several trials and returns the average mean latency."""
    d_theoretical = theoretical_delay(lam, mu_link, mu_broker)

    # Convergence strategy
    t_warmup = 10 * d_theoretical
    t_sim = max(2000.0, 100 * d_theoretical)
    t_total = t_warmup + t_sim

    trial_latencies = []

    for trial in range(trials):
        # Setup
        sources = ["P0"]
        brokers = ["SN1"]
        lambdas_total = np.array([lam])
        mu_links = np.array([[mu_link]])
        mu_brokers = np.array([mu_broker])

        topo = Topology(lambdas_total, mu_links, mu_brokers, sources, brokers)
        engine = SimulationEngine()
        state = SimulationState(1, 1, mu_links, mu_brokers)
        # Routing flows, not fractions: sum_j x_ij must equal lambda_i
        x_ij = np.array([[lam]])
        assert np.allclose(x_ij.sum(axis=1), lambdas_total)
        handler = SimulationHandler(engine, topo, state, x_ij)

        # Initial arrival
        engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=0))

        # Run
        engine.run(duration=t_total, handler=handler.handle_event)

        # Collect latencies for requests that completed AFTER warm-up
        latencies = []
        for req in state.requests.values():
            if "broker_complete" in req and req["broker_complete"] > t_warmup:
                latencies.append(req["broker_complete"] - req["arrival"])

        if latencies:
            trial_latencies.append(np.mean(latencies))
        else:
            # Fallback if no packets completed after warm-up (shouldn't happen for these parameters)
            trial_latencies.append(np.nan)

    return np.nanmean(trial_latencies)

def main():
    # Parameters
    mu_link = 1.0
    mu_broker = 1.0
    lambdas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]

    results = []

    print(f"Starting load sweep for mu_link={mu_link}, mu_broker={mu_broker}...")
    print("-" * 60)

    for lam in lambdas:
        theoretical = theoretical_delay(lam, mu_link, mu_broker)
        simulated = run_simulation(lam, mu_link, mu_broker)

        error_pct = abs(simulated - theoretical) / theoretical * 100

        results.append({
            "Lambda": lam,
            "Theoretical": theoretical,
            "Simulated": simulated,
            "Error (%)": error_pct
        })
        print(f"Lambda {lam:.2f} | Theoretical: {theoretical:.4f} | Simulated: {simulated:.4f} | Error: {error_pct:.2f}%")

    # Format as table
    df = pd.DataFrame(results)
    print("\nFinal Load Sweep Results:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    main()
