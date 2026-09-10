import json
import sys

import pytest
import yaml

from src import cli

def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["src.cli", "run", *argv])
    cli.main()

def rows(path):
    return [{k: v for k, v in json.loads(line).items() if k != "wall_time"}
            for line in (path / "metrics.jsonl").read_text().splitlines()]

@pytest.mark.parametrize("controller", ["static_algorithm1", "capacity_safe_event_driven", "windowed_stochastic"])
def test_generated_run_is_reproducible_from_its_config(monkeypatch, tmp_path, controller):
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
