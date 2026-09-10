import json
import yaml
import pytest
from src.cli import run_simulation

CONFIG = {
    "simulation": {"controller_mode": "windowed_stochastic", "duration": 40, "window": 5, "warmup": 2},
    "algorithm": {"eta": 0.35, "gamma": 0.5, "beta": 0.3},
    "topology": {
        "sources": [{"id": "A", "rate": 30.0}, {"id": "B", "rate": 10.0}],
        "brokers": [{"id": "S1", "capacity": 80.0}, {"id": "S2", "capacity": 60.0}],
        "access_capacities": {"A->S1": 60.0, "A->S2": 60.0, "B->S1": 60.0, "B->S2": 60.0},
    },
}
VOLATILE = {"wall_time"}

def _run(tmp_path, name, seed):
    cfg = json.loads(json.dumps(CONFIG))
    if seed is not None:
        cfg["simulation"]["seed"] = seed
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / name
    run_simulation(str(path), real_time=False, output_dir=str(out))
    rows = [{k: v for k, v in json.loads(line).items() if k not in VOLATILE}
            for line in (out / "metrics.jsonl").read_text().splitlines()]
    return rows, json.loads((out / "run.json").read_text())

def test_same_seed_reproduces_run(tmp_path):
    rows_a, meta_a = _run(tmp_path, "a", seed=123)
    rows_b, _ = _run(tmp_path, "b", seed=123)
    assert len(rows_a) == 8
    assert rows_a == rows_b
    assert meta_a["seed"] == 123

def test_different_seed_differs(tmp_path):
    rows_a, _ = _run(tmp_path, "a", seed=123)
    rows_c, _ = _run(tmp_path, "c", seed=124)
    assert rows_a != rows_c

def test_unseeded_run_records_its_seed(tmp_path):
    rows, meta = _run(tmp_path, "u", seed=None)
    assert isinstance(meta["seed"], int)
    rows_again, _ = _run(tmp_path, "u2", seed=meta["seed"])
    assert rows == rows_again

def test_run_metadata_contents(tmp_path):
    _, meta = _run(tmp_path, "m", seed=7)
    assert meta["controller_mode"] == "windowed_stochastic"
    assert (meta["n_sources"], meta["n_brokers"]) == (2, 2)
    assert meta["lambdas_total"] == [30.0, 10.0]
    assert meta["mu_brokers"] == [80.0, 60.0]
    assert meta["parameters"] == {"eta": 0.35, "gamma": 0.5, "beta": 0.3, "delta_s": None, "window": 5.0, "warmup": 2}
    assert set(meta["git"]) == {"sha", "dirty"}
    assert meta["config"]["simulation"]["seed"] == 7
