import numpy as np
import pytest
from src.simulation.engine import SimulationEngine
from src.simulation.events import Event, EventType
from src.simulation.queues import SimulationState
from src.simulation.handler import SimulationHandler
from src.model.topology import Topology

def test_async_controller_convergence():
    # 2 sources, 2 brokers
    sources = ["P0", "P1"]
    brokers = ["SN1", "SN2"]
    lambdas_total = np.array([0.4, 0.4]) # Total 0.8
    mu_links = np.array([[1.0, 1.0], [1.0, 1.0]])
    mu_brokers = np.array([1.0, 1.0])

    topo = Topology(lambdas_total, mu_links, mu_brokers, sources, brokers)

    # Start with unbalanced routing: all flow to SN1 (rows sum to lambda_i)
    x_ij = np.array([[0.4, 0.0], [0.4, 0.0]])

    engine = SimulationEngine()
    state = SimulationState(2, 2, mu_links, mu_brokers)
    handler = SimulationHandler(engine, topo, state, x_ij)

    # Initial arrival
    engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=0))
    engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=1))

    # Run for a while to see if routing balances
    engine.run(duration=100.0, handler=handler.handle_event)

    # Check if routing has shifted away from [1.0, 0.0]
    print(f"\nFinal Routing Matrix:\n{handler.x_ij}")

    # Symmetric optimum: each source splits its 0.4 evenly across both brokers
    assert handler.x_ij == pytest.approx(np.full((2, 2), 0.2), abs=1e-3)

if __name__ == "__main__":
    pytest.main([__file__])
