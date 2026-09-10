import numpy as np
import pytest
from src.model.topology import Topology
from src.simulation.engine import SimulationEngine
from src.simulation.events import Event, EventType
from src.simulation.handler import SimulationHandler
from src.simulation.queues import SimulationState
from src.telemetry.metrics import TelemetryBuffer

def test_windowed_records_queue_and_latency_metrics():
    np.random.seed(1)
    topo = Topology(np.array([30.0, 10.0]), np.full((2, 2), 60.0), np.array([80.0, 60.0]), ["A", "B"], ["S1", "S2"])
    engine = SimulationEngine()
    state = SimulationState(2, 2, topo.mu_links, topo.mu_brokers, keep_requests=False)
    telemetry = TelemetryBuffer()
    handler = SimulationHandler(engine, topo, state, np.array([[15.0, 15.0], [5.0, 5.0]]), telemetry=telemetry)
    for i in range(2):
        engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=i))
    engine.run(duration=100.0, handler=handler.handle_event)

    rows = list(telemetry.history)
    assert len(rows) == 20
    for r in rows:
        assert np.array(r["queue_access_ij"]).shape == (2, 2)
        assert len(r["queue_broker_j"]) == 2 and min(r["queue_broker_j"]) >= 0
        assert r["latency_p50"] <= r["latency_p95"] <= r["latency_p99"]
        assert len(r["latency_mean_i"]) == 2
    # Window counts add up to the run total
    assert sum(r["completed_window"] for r in rows) == rows[-1]["completed_total"] == state.completed
    # ~40 arrivals/s over 100 s
    assert 3500 < state.completed < 4500
    # Mean latency is consistent with the M/M/1 model at this light load
    model = np.mean([r["e2e_i"][0] for r in rows[-5:]])
    assert rows[-1]["latency_mean_total"] == pytest.approx(model, rel=0.15)
