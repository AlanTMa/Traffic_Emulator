import numpy as np
import pytest
from src.simulation.engine import SimulationEngine
from src.simulation.events import Event, EventType
from src.simulation.queues import SimulationState
from src.simulation.handler import SimulationHandler
from src.model.topology import Topology

def test_measured_price_controller_balances():
    # 2 sources, 2 identical brokers. Rates are high enough (~200 packets per
    # broker per window) for the measured-rate prices to be informative.
    sources = ["P0", "P1"]
    brokers = ["SN1", "SN2"]
    lambdas_total = np.array([40.0, 40.0])
    mu_links = np.full((2, 2), 100.0)
    mu_brokers = np.array([100.0, 100.0])

    topo = Topology(lambdas_total, mu_links, mu_brokers, sources, brokers)

    # Start with unbalanced routing: all flow to SN1 (rows sum to lambda_i)
    x_ij = np.array([[40.0, 0.0], [40.0, 0.0]])

    engine = SimulationEngine()
    state = SimulationState(2, 2, mu_links, mu_brokers)
    handler = SimulationHandler(engine, topo, state, x_ij, rng=np.random.default_rng(0))

    for i in range(2):
        engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=i))

    # 60 windows of 5s, 4 of them warm-up (notebook defaults)
    engine.run(duration=300.0, handler=handler.handle_event)

    print(f"\nFinal Routing Matrix:\n{handler.x_ij}")

    # Symmetric optimum is an even split. Prices come from measured traffic,
    # so allow for noise: across seeds 0-9 the worst deviation was ~1.0.
    assert handler.x_ij == pytest.approx(np.full((2, 2), 20.0), abs=2.0)

def test_splits_frozen_during_warmup():
    topo = Topology(np.array([40.0]), np.array([[100.0, 100.0]]), np.array([100.0, 100.0]), ["P0"], ["SN1", "SN2"])
    engine = SimulationEngine()
    handler = SimulationHandler(engine, topo, SimulationState(1, 2, topo.mu_links, topo.mu_brokers),
                                np.array([[40.0, 0.0]]), window=5.0, warmup_windows=4,
                                rng=np.random.default_rng(0))
    engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=0))

    # Through the 4th window (t=20) the split must not move; the 5th adapts it
    engine.run(duration=20.0, handler=handler.handle_event)
    assert handler.x_ij.tolist() == [[40.0, 0.0]]
    engine.run(duration=25.0, handler=handler.handle_event)
    assert handler.x_ij[0, 1] > 0.0

if __name__ == "__main__":
    pytest.main([__file__])
