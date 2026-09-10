"""
controller_mode: capacity_safe_event_driven

Event-driven queues whose planned routing is advanced by exact Algorithm 1
steps (synchronous.iteration_step) at every window boundary.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.cli import run_simulation
from src.controller.feasibility import transportation_feasibility
from src.controller.synchronous import iteration_step, run_algorithm1
from src.model.config import load_config, topology_from_config
from src.model.topology import Topology
from src.simulation.engine import SimulationEngine
from src.simulation.events import Event, EventType
from src.simulation.handler import SimulationHandler
from src.simulation.queues import SimulationState
from src.simulation.state import SystemState
from src.telemetry.metrics import TelemetryBuffer

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "capacity_safe_5x3.yaml"
DELTA_S = 1e-8

def make_handler(topo, x0=None, *, seed=0, delta_s=DELTA_S, eta=0.25, gamma=0.5, telemetry=True, **kwargs):
    if x0 is None:
        x0 = transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers, margin=delta_s)
    engine = SimulationEngine()
    state = SimulationState(topo.n_sources, topo.n_brokers, topo.mu_links, topo.mu_brokers, keep_requests=False)
    handler = SimulationHandler(engine, topo, state, np.array(x0, dtype=float),
                                telemetry=TelemetryBuffer() if telemetry else None, eta=eta, gamma=gamma,
                                eps=1e-12, delta_s=delta_s, controller_mode="capacity_safe_event_driven",
                                rng=np.random.default_rng(seed), **kwargs)
    for i in range(topo.n_sources):
        engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=i))
    return engine, state, handler

@pytest.fixture(scope="module")
def topo_5x3():
    return topology_from_config(load_config(CONFIG))

@pytest.fixture(scope="module")
def run_5x3(topo_5x3):
    engine, state, handler = make_handler(topo_5x3)
    engine.run(duration=100.0, handler=handler.handle_event)   # 20 windows
    return engine, state, handler

def binding_instance():
    # Same 2x2 instance as tests/unit/test_safe_step.py: a stale low price on
    # SN1 pulls source A onto SN1, which source B already fills to within 0.5.
    topo = Topology(np.array([10.0, 10.0]), np.array([[100.0, 100.0], [100.0, 0.01]]),
                    np.array([10.5, 1000.0]), ["A", "B"], ["SN1", "SN2"])
    return topo, np.array([[0.0, 10.0], [10.0, 0.0]]), np.array([0.0, 1e3])

# 1-3. Planned conservation, broker feasibility (margin delta_s), access feasibility

def test_planned_state_is_feasible_at_every_update(run_5x3, topo_5x3):
    _, _, handler = run_5x3
    rows = list(handler.telemetry.history)
    assert len(rows) == 20
    for r in rows:
        lam = np.array(r["lambda_ij"])
        assert np.max(np.abs(lam.sum(axis=1) - topo_5x3.lambdas_total)) <= 1e-8
        assert np.all(np.array(r["load_j"]) <= topo_5x3.mu_brokers - DELTA_S + 1e-12)
        assert np.all(lam < topo_5x3.mu_links) and np.all(lam >= 0)
        assert r["controller_mode"] == "capacity_safe_event_driven"
        assert r["s_t"] == 1.0 and len(r["s_j"]) == 3 and r["br_failures"] == []

# 4. One event-mode update equals one static Algorithm 1 step from the same state

def test_single_update_equals_iteration_step(topo_5x3):
    engine, _, handler = make_handler(topo_5x3, telemetry=False)
    before = SystemState(lambda_ij=handler.x_ij.copy(), prices=handler.prices.copy())
    engine.run(duration=5.0, handler=handler.handle_event)   # exactly one controller tick
    assert handler.window_idx == 1
    expected, s_t, _ = iteration_step(before, topo_5x3, 0.25, 0.5, eps=1e-12, delta_s=DELTA_S)
    assert np.array_equal(handler.x_ij, expected.lambda_ij)
    assert np.array_equal(handler.prices, expected.prices)
    assert handler.s_t == s_t and np.array_equal(handler.s_j, expected.s_j)

def test_update_sequence_equals_static_algorithm1(run_5x3, topo_5x3):
    _, _, handler = run_5x3
    static = run_algorithm1(topo_5x3, eta=0.25, gamma=0.5, tol=0.0, max_iter=20, eps=1e-12, delta_s=DELTA_S)
    assert np.array_equal(handler.x_ij, static.lambda_ij)
    assert np.array_equal(handler.prices, static.prices)
    # The stochastic arrivals never touch the planned controller state
    _, _, other_seed = make_handler(topo_5x3, seed=99, telemetry=False)
    other_seed.engine.run(duration=100.0, handler=other_seed.handle_event)
    assert np.array_equal(other_seed.x_ij, handler.x_ij)

# 5. The safe step binds and holds the margin

def test_safe_step_binds_in_event_mode():
    topo, lam0, stale_prices = binding_instance()
    delta_s = 1e-6
    engine, _, handler = make_handler(topo, lam0, delta_s=delta_s, gamma=0.01, telemetry=False)
    handler.controller_state.prices = stale_prices.copy()        # stale published prices
    before = SystemState(lambda_ij=lam0.copy(), prices=stale_prices.copy())
    engine.run(duration=5.0, handler=handler.handle_event)
    expected, s_t, _ = iteration_step(before, topo, 0.25, 0.01, eps=1e-12, delta_s=delta_s)
    # s_t = (10.5 - delta_s - 10) / (eta * Delta_SN1) = (0.5 - 1e-6) / (0.25 * 10)
    assert handler.s_t == pytest.approx((0.5 - delta_s) / 2.5, rel=1e-9) and handler.s_t == s_t
    assert np.array_equal(handler.x_ij, expected.lambda_ij)
    assert handler.x_ij.sum(axis=0)[0] == pytest.approx(10.5 - delta_s, abs=1e-9)

def test_initial_routing_must_respect_the_margin():
    topo, _, _ = binding_instance()
    too_full = np.array([[0.5, 9.5], [10.0, 0.0]])   # SN1 planned 10.5 > 10.5 - delta_s
    with pytest.raises(ValueError, match="mu_j - delta_s"):
        make_handler(topo, too_full)

# 6. Queues keep working across controller updates

def test_work_units_are_conserved_across_updates(run_5x3):
    _, state, handler = run_5x3
    in_system = lambda q: len(q.queue) + int(q.is_busy)
    queued = sum(in_system(q) for row in state.access_queues.values() for q in row.values())
    queued += sum(in_system(q) for q in state.broker_queues.values())
    assert handler.request_id_counter == state.completed + queued
    rows = list(handler.telemetry.history)
    assert sum(r["completed_window"] for r in rows) == state.completed
    assert all(r["completed_window"] > 0 for r in rows)          # service continues every window
    assert all(r["latency_p50"] <= r["latency_p95"] <= r["latency_p99"] for r in rows)
    # Mean sojourn time is consistent with the planned M/M/1 model at this load
    assert rows[-1]["latency_mean_total"] == pytest.approx(np.mean([r["e2e_i"][2] for r in rows]), rel=0.2)

# 7. Arrivals are routed with the updated planned fractions

def test_routing_probabilities_follow_updated_plan(topo_5x3):
    engine, _, handler = make_handler(topo_5x3, telemetry=False)
    draws = []

    class RecordingRng:
        """Generator proxy recording the broker-choice probabilities of each arrival."""
        def __init__(self, rng):
            self._rng = rng
        def exponential(self, scale):
            return self._rng.exponential(scale)
        def choice(self, n, p):
            draws.append((engine.now, np.array(p)))
            return self._rng.choice(n, p=p)
    handler.rng = RecordingRng(handler.rng)

    fractions_0 = handler.x_ij / handler.x_ij.sum(axis=1, keepdims=True)
    engine.run(duration=5.0, handler=handler.handle_event)
    # Routing probabilities are lambda_ij / sum_j lambda_ij (= lambda_i up to the
    # ~1e-11 conservation residual) of the updated plan
    fractions_1 = handler.x_ij / handler.x_ij.sum(axis=1, keepdims=True)
    assert not np.allclose(fractions_0, fractions_1, atol=1e-3)   # the update moved the plan
    n_before = len(draws)
    engine.run(duration=10.0, handler=handler.handle_event)
    after = [p for t, p in draws[n_before:] if 5.0 < t <= 10.0]
    assert len(after) > 100
    for p in after:
        assert any(np.allclose(p, row, rtol=0, atol=1e-15) for row in fractions_1)

# 8-9. Seeded reproducibility, mode in telemetry and run.json

def _cli_rows(tmp_path, name, seed):
    cfg = load_config(CONFIG)
    cfg["simulation"].update(duration=30, seed=seed)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run_simulation(str(path), real_time=False, output_dir=str(tmp_path / name))
    rows = [{k: v for k, v in json.loads(line).items() if k != "wall_time"}
            for line in (tmp_path / name / "metrics.jsonl").read_text().splitlines()]
    return rows, json.loads((tmp_path / name / "run.json").read_text())

def test_seeded_runs_reproduce_and_mode_is_recorded(tmp_path):
    rows_a, meta = _cli_rows(tmp_path, "a", 5)
    rows_b, _ = _cli_rows(tmp_path, "b", 5)
    rows_c, _ = _cli_rows(tmp_path, "c", 6)
    assert rows_a == rows_b and len(rows_a) == 6
    assert rows_a != rows_c
    # Different seeds change the measured traffic but not the planned controller
    assert [r["lambda_ij"] for r in rows_a] == [r["lambda_ij"] for r in rows_c]
    assert {r["controller_mode"] for r in rows_a} == {"capacity_safe_event_driven"}
    assert meta["controller_mode"] == "capacity_safe_event_driven"
    assert meta["parameters"]["delta_s"] == 1e-8 and meta["parameters"]["eta"] == 0.25
