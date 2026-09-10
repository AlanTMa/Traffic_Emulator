"""Time-varying capacities (src/model/dynamics.py), ported from the notebook's vary_* functions."""
import json

import numpy as np
import pytest
import yaml

from src.cli import run_simulation
from src.model.config import load_config, with_controller_mode
from src.model.dynamics import NOTEBOOK_COMBOS, CapacityVariation, vary_broker_capacity, vary_link_capacity
from src.model.topology import Topology
from src.runtime.experiments import canonical_5x3
from src.simulation.engine import SimulationEngine
from src.simulation.events import Event, EventType
from src.simulation.handler import SimulationHandler
from src.simulation.queues import SimulationState

def test_notebook_formulas_and_floors():
    rng = np.random.default_rng(0)
    base = np.array([10.0, 20.0])
    # No noise: exact sinusoid / cosinusoid
    t, T = 15.0, 60.0
    assert vary_link_capacity(base, t, 0.15, T, 0.0, 0.0, rng) == pytest.approx(base * (1 + 0.15 * np.sin(2 * np.pi * t / T)))
    assert vary_broker_capacity(base, t, 0.25, T, 0.0, 0.0, rng) == pytest.approx(base * (1 + 0.25 * np.cos(2 * np.pi * t / T)))
    # Floors: 0.2 of base for links, 0.3 for brokers
    trough_sin, trough_cos = 45.0, 30.0          # sin = -1, cos = -1 for T = 60
    assert vary_link_capacity(base, trough_sin, 0.9, T, 0.0, 0.0, rng) == pytest.approx(0.2 * base)
    assert vary_broker_capacity(base, trough_cos, 0.9, T, 0.0, 0.0, rng) == pytest.approx(0.3 * base)

def test_config_combos_defaults_and_validation():
    topo = canonical_5x3()
    m3 = CapacityVariation.from_config({"dynamics": {"capacity_variation": {"combo": 3}}}, topo, seed=1)
    la, lp, ln, ba, bp, bn = NOTEBOOK_COMBOS[3]
    assert m3.link == {"amp": la, "period": lp, "noise": ln} and m3.broker == {"amp": ba, "period": bp, "noise": bn}
    default = CapacityVariation.from_config({"dynamics": {"capacity_variation": {"link": {"amp": 0.5}}}}, topo, seed=1)
    assert default.link == {"amp": 0.5, "period": 60.0, "noise": 0.02}
    assert default.broker == {"amp": 0.25, "period": 120.0, "noise": 0.03}
    assert CapacityVariation.from_config({}, topo, seed=1) is None
    with pytest.raises(ValueError, match="combo"):
        CapacityVariation.from_config({"dynamics": {"capacity_variation": {"combo": 12}}}, topo, seed=1)
    with pytest.raises(ValueError, match="period > 0"):
        CapacityVariation.from_config({"dynamics": {"capacity_variation": {"broker": {"period": 0}}}}, topo, seed=1)

def test_seeded_and_mask_preserving():
    topo = Topology([2.0, 1.0], [[5.0, 5.0], [5.0, 5.0]], [8.0, 8.0], ["A", "B"], ["S1", "S2"],
                    route_mask=[[True, True], [True, False]])
    cfg = {"dynamics": {"capacity_variation": {"combo": 5}}}
    a = [CapacityVariation.from_config(cfg, topo, seed=3) for _ in range(2)]
    b = CapacityVariation.from_config(cfg, topo, seed=4)
    ta = [a[0].topology_at(t).mu_brokers for t in (0, 5, 10)]
    assert all(np.array_equal(x, y) for x, y in zip(ta, [a[1].topology_at(t).mu_brokers for t in (0, 5, 10)]))
    assert not np.array_equal(ta[1], b.topology_at(0).mu_brokers)
    assert a[0].topology_at(20).mu_links[1, 1] == 0.0          # the missing link stays missing

class StepDown:
    """Test capacity model: broker capacity halves at t >= 50 (deterministic)."""
    def __init__(self, topo):
        self.topo = topo
    def topology_at(self, t):
        return self.topo.with_capacities(self.topo.mu_links, self.topo.mu_brokers * (0.5 if t >= 50 else 1.0))

def test_event_service_uses_current_capacity():
    topo = Topology([3.0], [[1000.0]], [10.0], ["A"], ["S"])
    engine = SimulationEngine()
    log = []
    handler = SimulationHandler(engine, topo, SimulationState(1, 1, topo.mu_links, topo.mu_brokers),
                                np.array([[3.0]]), rng=np.random.default_rng(0), latency_log=log,
                                capacity_model=StepDown(topo))
    engine.schedule(Event(timestamp=0.0, event_type=EventType.SOURCE_ARRIVAL, source_id=0))
    engine.run(duration=4000.0, handler=handler.handle_event)
    t, _, lat = (np.array(c) for c in zip(*log))
    before, after = lat[(t > 5) & (t < 50)].mean(), lat[t > 100].mean()
    # M/M/1 broker (access delay ~1e-3): 1/(10-3) = 0.143 before, 1/(5-3) = 0.5 after
    assert after == pytest.approx(0.5, rel=0.1)
    assert before < 0.25

@pytest.mark.parametrize("mode", ["capacity_safe_event_driven", "windowed_stochastic", "static_algorithm1"])
def test_cli_runs_with_time_varying_capacities(tmp_path, mode):
    cfg = with_controller_mode(load_config("config/paper_5x3.yaml"), mode)
    cfg["simulation"].update(duration=60, seed=2)
    cfg["dynamics"] = {"capacity_variation": {"combo": 2}}
    path = tmp_path / "dyn.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run_simulation(str(path), real_time=False, output_dir=str(tmp_path / "out"))
    rows = [json.loads(l) for l in (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()]
    mu_b = np.array([r["mu_brokers_t"] for r in rows])
    assert len(rows) == 12 and np.all(mu_b > 0)
    assert np.ptp(mu_b[:, 0]) > 1.0                       # capacities actually vary
    for r in rows:
        # model quantities are evaluated against the capacities in effect
        assert np.array(r["util_j"]) == pytest.approx(np.array(r["load_j"]) / np.array(r["mu_brokers_t"]))
        if mode != "windowed_stochastic":
            # Algorithm 1 modes keep the planned loads under the current capacities
            # here (mild combo, light load): capacity safety w.r.t. mu(t)
            assert np.all(np.array(r["load_j"]) < np.array(r["mu_brokers_t"]))
    meta = json.loads((tmp_path / "out" / "run.json").read_text())
    assert meta["config"]["dynamics"]["capacity_variation"]["combo"] == 2
