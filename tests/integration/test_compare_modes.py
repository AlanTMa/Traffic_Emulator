import numpy as np
import pytest
from scripts.compare_modes import MODES, compare, comparison_table
from src.runtime.experiments import canonical_5x3

def test_compare_modes_smoke():
    topo = canonical_5x3()
    results = compare(topo, duration=120.0, warmup_time=30.0, seed=0)
    assert set(results) == set(MODES)
    table = comparison_table(results, topo)
    assert list(table.columns) == list(MODES)
    # Static optimum is certified; capacity-safe planned state follows the same
    # Algorithm 1 iterates (24 windows here), so it is close to the optimum
    assert results["static_algorithm1"]["certified"]
    assert results["capacity_safe_event_driven"]["objective"] == pytest.approx(
        results["static_algorithm1"]["objective"], rel=1e-5)
    assert results["capacity_safe_event_driven"]["min_planned_broker_headroom"] > 0
    for mode in MODES[1:]:
        r = results[mode]
        assert r["completed"] > 1000
        assert r["latency_p50"] <= r["latency_p95"] <= r["latency_p99"]
        assert np.isnan(table.loc["p95", "static_algorithm1"])
