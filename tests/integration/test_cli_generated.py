import json
import sys

import numpy as np
import pytest
import yaml

from src import cli
from src.controller.synchronous import run_algorithm1
from src.model.config import load_config, topology_from_config, with_controller_mode

def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["src.cli", "run", *argv])
    cli.main()

def rows(path):
    return [{k: v for k, v in json.loads(line).items() if k != "wall_time"}
            for line in (path / "metrics.jsonl").read_text().splitlines()]

@pytest.mark.parametrize("controller", ["static_algorithm1", "capacity_safe_event_driven", "windowed_stochastic"])
def test_generated_run_reproducible(monkeypatch, tmp_path, controller):
    out = tmp_path / "gen"
    run_cli(monkeypatch, "--sources", "6", "--brokers", "3", "--load", "0.4", "--seed", "7",
            "--controller", controller, "--duration", "20", "--window", "2", "--no-realtime", "--output-dir", str(out))
    cfg = yaml.safe_load((out / "generated_config.yaml").read_text())
    assert cfg["simulation"]["controller_mode"] == controller and cfg["simulation"]["seed"] == 7
    assert len(cfg["topology"]["sources"]) == 6 and len(cfg["topology"]["brokers"]) == 3
    meta = json.loads((out / "run.json").read_text())
    assert meta["controller_mode"] == controller and (meta["n_sources"], meta["n_brokers"]) == (6, 3)
    assert meta["config"] == cfg

    again = tmp_path / "again"
    run_cli(monkeypatch, "--config", str(out / "generated_config.yaml"), "--no-realtime", "--output-dir", str(again))
    assert rows(out) == rows(again) and len(rows(out)) == 10

def test_config_mode_switch(monkeypatch, tmp_path):
    # The canonical 5x3 (written for windowed_stochastic, eta = split inertia 0.35) under static Algorithm 1
    out = tmp_path / "static"
    run_cli(monkeypatch, "--config", "config/paper_5x3.yaml", "--controller", "static_algorithm1",
            "--window", "1", "--duration", "70", "--no-realtime", "--output-dir", str(out))
    cfg = yaml.safe_load((out / "resolved_config.yaml").read_text())
    assert cfg["simulation"]["controller_mode"] == "static_algorithm1"
    assert cfg["algorithm"] == {"eta": 0.25, "gamma": 0.5, "delta_s": 1e-8, "eps": 1e-12}
    topo = topology_from_config(load_config("config/paper_5x3.yaml"))
    ref = run_algorithm1(topo, eta=0.25, gamma=0.5, tol=0.0, max_iter=70, delta_s=1e-8, eps=1e-12)
    last = rows(out)[-1]
    assert len(rows(out)) == 70 and np.array_equal(np.array(last["lambda_ij"]), ref.lambda_ij)
    assert last["objective"] == pytest.approx(2.0157473649139757, rel=0, abs=1e-12)   # notebook F_dist, 70 iterations

    # Switching between the two Algorithm 1 modes keeps the parameters
    switched = with_controller_mode({**cfg, "algorithm": {**cfg["algorithm"], "eta": 0.1}}, "capacity_safe_event_driven")
    assert switched["algorithm"]["eta"] == 0.1 and switched["algorithm"]["beta"] == 0.3

def test_mode_parameter_overrides(monkeypatch, tmp_path):
    out = tmp_path / "o"
    run_cli(monkeypatch, "--sources", "2", "--brokers", "2", "--controller", "capacity_safe_event_driven",
            "--eta", "0.1", "--delta-s", "1e-6", "--duration", "10", "--no-realtime", "--output-dir", str(out))
    alg = yaml.safe_load((out / "generated_config.yaml").read_text())["algorithm"]
    assert alg["eta"] == 0.1 and alg["delta_s"] == 1e-6 and alg["gamma"] == 0.5

@pytest.mark.parametrize("argv, message", [
    (["--sources", "3"], "--sources needs --brokers"),
    (["--sources", "2", "--brokers", "1", "--load", "0.9"], "infeasible"),
    (["--sources", "2", "--brokers", "2", "--controller", "static_algorithm1", "--beta", "0.2"], "does not apply"),
    (["--config", "x.yaml", "--sources", "2"], "not allowed with"),
    ([], "one of the arguments --config --sources is required"),
])
def test_invalid_cli_arguments(monkeypatch, tmp_path, capsys, argv, message):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, *argv, "--output-dir", str(tmp_path / "x"))
    captured = capsys.readouterr()
    assert message in captured.err + captured.out
