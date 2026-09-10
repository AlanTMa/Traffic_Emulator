import json
import numpy as np
import pytest
from src.model.topology import Topology
from src.telemetry.metrics import TelemetryBuffer
from src.telemetry.schema import controller_snapshot

TOPO = Topology(np.array([4.0, 2.0]), np.full((2, 3), 10.0), np.array([20.0, 15.0, 10.0]), ["A", "B"], ["S1", "S2", "S3"])
LAM = np.array([[2.0, 1.0, 1.0], [0.0, 1.0, 1.0]])

def test_snapshot_shapes_and_values():
    r = controller_snapshot(TOPO, LAM, np.array([0.1, 0.1, 0.2]), iteration=3, sim_time=15.0,
                            controller_mode="static_algorithm1", s_j=np.array([1.0, 0.5, 1.0]), s_t=0.5,
                            route_rel=1e-3, price_rel=2e-3)
    assert np.array(r["lambda_ij"]).shape == (2, 3)
    for key in ("fraction_ij", "access_util_ij", "D_ij", "C_ij", "M_ij", "active_ij"):
        assert np.array(r[key]).shape == (2, 3), key
    for key in ("load_j", "util_j", "price_j", "D_j", "C_j", "s_j"):
        assert len(r[key]) == 3, key
    for key in ("alpha_i", "e2e_i", "active_spread_i"):
        assert len(r[key]) == 2, key
    assert np.array(r["fraction_ij"]).sum(axis=1) == pytest.approx([1.0, 1.0])
    assert r["load_j"] == pytest.approx([2.0, 2.0, 2.0])
    assert r["util_j"] == pytest.approx([0.1, 2 / 15, 0.2])
    assert r["s_t"] == 0.5 and r["iteration"] == 3 and r["sim_time"] == 15.0
    # e2e_A = (2*(1/8+1/18) + 1*(1/9+1/13) + 1*(1/9+1/8)) / 4
    expected = (2 * (1/8 + 1/18) + (1/9 + 1/13) + (1/9 + 1/8)) / 4
    assert r["e2e_i"][0] == pytest.approx(expected)
    assert isinstance(r["certified"], bool)

def test_records_are_appended_as_json_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    buf = TelemetryBuffer(path=path)
    for k in range(3):
        buf.record(controller_snapshot(TOPO, LAM, np.zeros(3), iteration=k, sim_time=k, controller_mode="windowed_stochastic"))
    buf.close()  # writes are flushed periodically and on close
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [0, 1, 2]
    assert rows[0]["s_j"] is None
