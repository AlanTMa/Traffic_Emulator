"""
Distributed backend: controller, source and broker processes.

Mathematical correctness is tested separately from process scheduling:
- the controller's round logic, fed by the real actor-side functions over the
  real wire encoding, must reproduce synchronous.iteration_step bit-for-bit;
- live multi-process runs must reproduce the reference Algorithm 1 iterates
  exactly (planned rates do not depend on traffic timing) and satisfy the
  feasibility invariants, while traffic, queues and latency keep flowing.
"""
import asyncio
import json
import socket
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.controller.diagnostics import compute_diagnostics
from src.controller.synchronous import (algorithm1_initial_state, broker_price, iteration_step, run_algorithm1,
                                        source_best_response)
from src.distributed import launcher
from src.distributed.controller import ControllerService
from src.distributed.protocol import encode
from src.distributed.settings import distributed_settings
from src.model.config import load_config, topology_from_config
from src.simulation.state import SystemState
from src.telemetry.metrics import TelemetryBuffer

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "config" / "paper_5x3.yaml"

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

# ---------------------------------------------------------------- one round, in process

class FakeActor:
    """Stands in for a worker process: same actor-side function, same wire encoding."""
    def __init__(self, handler):
        self.handler = handler
    async def request(self, message, timeout):
        wire = json.loads(encode(message))                # exactly what crosses the socket
        return json.loads(encode({"req": 1, **self.handler(wire)}))
    async def send(self, message):
        json.loads(encode(message))

def controller_with_fake_actors(config: dict, state: SystemState = None) -> ControllerService:
    ctl = ControllerService(config, output_dir="unused")
    if state is not None:
        ctl.state = state
    topo, p = ctl.topology, ctl.p
    metrics_b = {"arrivals": 0, "completed": 0, "in_system": 0, "completed_total": 0, "latency_sum_total": 0.0,
                 "samples": []}
    ctl.channels["broker"] = [FakeActor(lambda m, j=j: {"price": broker_price(m["price"], m["load"],
                                                        float(topo.mu_brokers[j]), p["gamma"], p["eps"]),
                                                        "metrics": metrics_b})
                              for j in range(topo.n_brokers)]
    ctl.channels["source"] = [FakeActor(lambda m, i=i: {"lambda_br": source_best_response(
                                  topo.mu_links[i], np.array(m["prices"]), float(topo.lambdas_total[i])),
                                  "metrics": {"routed": [0] * topo.n_brokers, "in_system": [0] * topo.n_brokers,
                                              "generated_total": 0}})
                              for i in range(topo.n_sources)]
    ctl.telemetry = TelemetryBuffer()
    ctl.t0 = time.time()
    return ctl

def reference_config(path=PAPER):
    cfg = load_config(path)
    cfg["simulation"]["controller_mode"] = "capacity_safe_event_driven"
    cfg["algorithm"] = {"eta": 0.25, "gamma": 0.5, "delta_s": 1e-8, "eps": 1e-12}
    return cfg

def test_distributed_round_matches_reference_iteration_step():
    cfg = reference_config()
    ctl = controller_with_fake_actors(cfg)
    topo = ctl.topology
    reference = algorithm1_initial_state(topo, 1e-8, 1e-12)
    for k in range(1, 6):
        before = SystemState(ctl.state.lambda_ij.copy(), ctl.state.prices.copy())
        asyncio.run(ctl._round(k, time.time() - 1.0))
        reference, s_t, _ = iteration_step(reference, topo, 0.25, 0.5, eps=1e-12, delta_s=1e-8)
        assert np.array_equal(ctl.state.prices, reference.prices)            # broker prices
        assert np.array_equal(ctl.state.s_j, reference.s_j)                  # per-broker bounds
        assert ctl.state.s_t == reference.s_t == s_t                        # common step
        assert np.array_equal(ctl.state.lambda_ij, reference.lambda_ij)      # routing update
        # Best responses and Delta_ij: recover them from the step lambda_new = lambda + eta s_t Delta
        br = np.vstack([source_best_response(topo.mu_links[i], reference.prices, topo.lambdas_total[i])
                        for i in range(topo.n_sources)])
        assert np.allclose(ctl.state.lambda_ij - before.lambda_ij, 0.25 * s_t * (br - before.lambda_ij),
                           rtol=0, atol=1e-12)
    # Telemetry carries the same diagnostics as the reference
    record = ctl.telemetry.history[-1]
    ref_diag = compute_diagnostics(reference.lambda_ij, reference.prices, topo)
    assert record["r_fixed_point"] == ref_diag["r_fixed_point"]
    assert record["certificate_status"] == ref_diag["status"]
    assert record["execution_backend"] == "distributed"

def test_distributed_round_matches_reference_when_safe_step_binds():
    # 2x2 instance where a stale low price makes the safe step bind (s_t = 0.2)
    cfg = {"simulation": {"controller_mode": "capacity_safe_event_driven"},
           "algorithm": {"eta": 0.25, "gamma": 0.01, "delta_s": 1e-6, "eps": 1e-12},
           "topology": {"sources": [{"id": "A", "rate": 10.0}, {"id": "B", "rate": 10.0}],
                        "brokers": [{"id": "S1", "capacity": 10.5}, {"id": "S2", "capacity": 1000.0}],
                        "access_capacities": {"A->S1": 100.0, "A->S2": 100.0, "B->S1": 100.0, "B->S2": 0.01}}}
    start = SystemState(np.array([[0.0, 10.0], [10.0, 0.0]]), np.array([0.0, 1e3]))
    ctl = controller_with_fake_actors(cfg, SystemState(start.lambda_ij.copy(), start.prices.copy()))
    asyncio.run(ctl._round(1, time.time() - 1.0))
    ref, s_t, _ = iteration_step(start, ctl.topology, 0.25, 0.01, eps=1e-12, delta_s=1e-6)
    assert s_t < 1.0 and ctl.state.s_t == s_t
    assert np.array_equal(ctl.state.lambda_ij, ref.lambda_ij) and np.array_equal(ctl.state.prices, ref.prices)
    assert ctl.state.lambda_ij.sum(axis=0)[0] == pytest.approx(10.5 - 1e-6, abs=1e-9)

def test_registration_is_idempotent_and_bounded():
    ctl = ControllerService(reference_config(), output_dir="unused")
    assert [ctl._assign("source", f"s{i}") for i in range(5)] == [0, 1, 2, 3, 4]
    assert ctl._assign("source", "s2") == 2                 # a restarted process keeps its id
    assert ctl._assign("source", "extra") is None           # topology has 5 sources
    assert [ctl._assign("broker", f"b{j}") for j in range(3)] == [0, 1, 2]
    assert ctl._assign("broker", "b9") is None

def test_settings_resolution():
    s = distributed_settings(load_config(PAPER))             # windowed config: paper Algorithm 1 defaults
    assert s["params"]["eta"] == 0.25 and s["params"]["gamma"] == 0.5 and s["params"]["delta_s"] == 1e-8
    assert s["notes"] and s["controller_mode"] == "capacity_safe_event_driven"
    cs = distributed_settings(load_config(ROOT / "config" / "capacity_safe_5x3.yaml"))
    assert cs["params"]["eta"] == 0.25 and not cs["notes"]
    assert distributed_settings(load_config(PAPER), {"eta": 0.1})["params"]["eta"] == 0.1
    with pytest.raises(ValueError, match="static capacities"):
        distributed_settings({**load_config(PAPER), "dynamics": {"capacity_variation": {"combo": 1}}})

# ---------------------------------------------------------------- live processes

def run_live(tmp_path, *, config=None, sources=None, brokers=None, window=0.5, duration=8.0):
    run = launcher.prepare_run(config, sources, brokers, output_dir=tmp_path / "out", window=window, seed=11)
    port = free_port()
    procs = launcher.start_local(run, port=port, dashboard=False, duration=duration, quiet=True)
    try:
        assert procs[0].wait(timeout=duration + 60) == 0
        codes = [p.wait(timeout=20) for p in procs[1:]]
    finally:
        launcher.stop_local(procs, port, grace=5)
    rows = [json.loads(l) for l in (run["output_dir"] / "metrics.jsonl").read_text().splitlines()]
    meta = json.loads((run["output_dir"] / "run.json").read_text())
    return run, rows, meta, codes

def test_live_5x3_matches_reference_and_keeps_traffic_flowing(tmp_path):
    run, rows, meta, codes = run_live(tmp_path, config=str(PAPER))
    assert codes == [0] * 8                                  # every worker exited cleanly on stop
    topo = topology_from_config(yaml.safe_load(run["config_path"].read_text()))
    assert topo.lambdas_total.tolist() == topology_from_config(load_config(PAPER)).lambdas_total.tolist()
    assert len(rows) >= 10
    for r in rows:
        # Planned Algorithm 1 state across processes == the reference, bit for bit
        ref = run_algorithm1(topo, eta=0.25, gamma=0.5, tol=0.0, max_iter=r["iteration"], delta_s=1e-8, eps=1e-12)
        lam = np.array(r["lambda_ij"])
        assert np.array_equal(lam, ref.lambda_ij) and np.array_equal(np.array(r["price_j"]), ref.prices)
        assert np.max(np.abs(lam.sum(axis=1) - topo.lambdas_total)) <= 1e-8          # conservation
        assert np.all(lam >= 0) and np.all(lam < topo.mu_links)                      # access feasibility
        assert np.all(np.array(r["load_j"]) <= topo.mu_brokers - 1e-8 + 1e-12)       # Lambda_j <= mu_j - delta_s
        assert r["controller_mode"] == "capacity_safe_event_driven" and r["execution_backend"] == "distributed"
        assert np.array(r["queue_access_ij"]).shape == (5, 3) and len(r["queue_broker_j"]) == 3
    last = rows[-1]
    assert last["completed_total"] > 200                     # ~60 work units/s were generated and served
    assert last["latency_p50"] <= last["latency_p95"] <= last["latency_p99"]
    assert set(last["latency_parts"]) == {"access_wait", "access_service", "transfer", "broker_wait", "broker_service"}
    # Queueing part of the sojourn is consistent with the planned M/M/1 model: mean over every
    # completed unit (one 0.5 s window holds only ~30 units, too few to test on its own)
    n_done = np.array([r["completed_window"] if np.isfinite(r["queueing_sojourn_mean"]) else 0 for r in rows], float)
    measured = np.sum(n_done * np.nan_to_num([r["queueing_sojourn_mean"] for r in rows])) / n_done.sum()
    model = np.sum(n_done * np.array([r["objective"] for r in rows])) / n_done.sum() / topo.lambdas_total.sum()
    assert measured == pytest.approx(model, rel=0.25)
    assert meta["execution_backend"] == "distributed"
    assert sorted(meta["processes"]["source"].values()) == ["P0", "P1", "P2", "P3", "P4"]
    assert sorted(meta["processes"]["broker"].values()) == ["SN1", "SN2", "SN3"]
    assert meta["parameters"]["eta"] == 0.25 and meta["parameters"]["delta_s"] == 1e-8

def test_live_generated_topology_is_not_hardcoded_to_5x3(tmp_path):
    run, rows, meta, codes = run_live(tmp_path, sources=4, brokers=2, duration=4.0)
    assert (run["n"], run["m"]) == (4, 2) and codes == [0] * 6
    assert len(rows) >= 4 and np.array(rows[-1]["lambda_ij"]).shape == (4, 2)
    assert rows[-1]["completed_total"] > 0
    assert len(meta["processes"]["source"]) == 4 and len(meta["processes"]["broker"]) == 2

# ---------------------------------------------------------------- launcher and compose

def test_prepare_run_validation(tmp_path):
    with pytest.raises(ValueError, match="--sources 3 does not match"):
        launcher.prepare_run(str(PAPER), sources=3, output_dir=tmp_path)
    run = launcher.prepare_run(str(PAPER), sources=5, brokers=3, output_dir=tmp_path, window=2.0)
    resolved = yaml.safe_load(run["config_path"].read_text())
    assert resolved["simulation"]["execution_backend"] == "distributed"
    assert resolved["simulation"]["window"] == 2.0 and resolved["algorithm"]["eta"] == 0.25
    assert launcher.compose_command(run)[-4:] == ["--scale", "source=5", "--scale", "broker=3"]
    assert "--exit-code-from" in launcher.compose_command(run, duration=30)
    with pytest.raises(ValueError, match="under runs/"):         # only runs/ is mounted in the containers
        launcher.compose_env(run)
    env = launcher.compose_env({**run, "output_dir": ROOT / "runs" / "ci"}, 30)
    assert env["TE_CONFIG"] == "runs/ci/resolved_config.yaml" and env["TE_DURATION"] == "30"

def test_compose_file_defines_replicable_services():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    services = compose["services"]
    for name in ("controller", "source", "broker", "dashboard"):
        assert "distributed" in services[name]["profiles"]
    for name in ("source", "broker"):                         # replicated with --scale: no fixed names/ports
        assert "container_name" not in services[name] and "ports" not in services[name]
        assert services[name]["depends_on"]["controller"]["condition"] == "service_healthy"
        assert f"src.distributed.{name}" in services[name]["command"]
    assert "healthcheck" in services["controller"]
    assert services["dashboard"]["environment"]["TRAFFIC_EMULATOR_OBSERVER"] == "1"
