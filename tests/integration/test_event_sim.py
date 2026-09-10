import numpy as np
import pytest
from src.simulation.engine import SimulationEngine
from src.simulation.events import Event, EventType
from src.simulation.queues import SimulationState
from src.simulation.handler import SimulationHandler
from src.model.topology import Topology

def test_event_driven_mm1_latency():
    # Simple 1 source, 1 broker system
    sources = ["P0"]
    brokers = ["SN1"]
    # Moderate load: at rho=0.9 the queue mixes too slowly for a short run to
    # converge (~50% of unseeded runs fell outside 20%).
    lambdas_total = np.array([0.5]) # 0.5 pkts/s
    mu_links = np.array([[1.0]])    # 1.0 pkt/s
    mu_brokers = np.array([1.0])   # 1.0 pkt/s

    topo = Topology(lambdas_total, mu_links, mu_brokers, sources, brokers)

    # state and handler
    engine = SimulationEngine()
    state = SimulationState(1, 1, mu_links, mu_brokers)
    x_ij = np.array([[1.0]])
    handler = SimulationHandler(engine, topo, state, x_ij, rng=np.random.default_rng(0))

    # Initial arrival
    engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=0))

    # Run for a while to gather samples
    engine.run(duration=20000.0, handler=handler.handle_event)

    # Calculate end-to-end latency: broker_complete - arrival
    latencies = []
    for req in state.requests.values():
        if "broker_complete" in req:
            latencies.append(req["broker_complete"] - req["arrival"])

    mean_latency = np.mean(latencies)

    # Theoretical M/M/1 delay: 1/(mu_link - lam) + 1/(mu_broker - lam)
    # = 1/(1-0.5) + 1/(1-0.5) = 2 + 2 = 4.0
    expected = 4.0

    print(f"\nSimulated Mean Latency: {mean_latency:.4f}")
    print(f"Expected Mean Latency: {expected:.4f}")

    # Across 30 seeds this setup stayed within 5%; 10% leaves headroom.
    assert mean_latency == pytest.approx(expected, rel=0.1)
