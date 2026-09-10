"""
Controller / coordinator process (distributed backend).

Responsibilities:
- load the run config and topology; decide Algorithm 1 parameters (settings.py)
- register broker and source processes and assign them logical ids
  (idempotent per process instance; N and M come from the topology)
- run synchronous Algorithm 1 rounds every `window` seconds, forever:
    1. planned loads Lambda_j = sum_i lambda_ij (controller state)
    2. each BROKER returns its damped price  (synchronous.broker_price)
    3. each SOURCE returns its best response (synchronous.source_best_response)
    4. the controller applies the common safe step (synchronous.apply_common_step)
       and sends every source its new routing row
- Proposition 1 diagnostics and telemetry for every round (telemetry.schema),
  aggregating the measurements that brokers and sources report
- stop on Ctrl+C / SIGTERM (or after --duration), telling every process

Reaching the certificate does not stop the run: traffic keeps flowing under
the converged routing and rounds continue.

    python -m src.distributed.controller --config config/paper_5x3.yaml
"""
import argparse
import asyncio
import os
import time
from pathlib import Path

import numpy as np

from src.controller.synchronous import algorithm1_initial_state, apply_common_step, compute_broker_loads
from src.distributed.protocol import (CONTROL_PORT, LINE_LIMIT, PROTOCOL_VERSION, Channel, actor_seed)
from src.distributed.runtime import install_stop_handlers, log
from src.distributed.settings import distributed_settings
from src.model.config import load_config, topology_from_config
from src.runtime.metadata import write_run_metadata
from src.telemetry.metrics import TelemetryBuffer
from src.telemetry.schema import controller_snapshot

class ControllerService:
    def __init__(self, config: dict, output_dir: str, host: str = "0.0.0.0", port: int = CONTROL_PORT,
                 duration: float = None, config_path: str = None, round_timeout: float = None):
        self.config, self.config_path = config, config_path
        self.topology = topology_from_config(config)
        self.settings = distributed_settings(config)
        self.p = self.settings["params"]
        self.seed = self.settings["seed"]
        self.output_dir = Path(output_dir)
        self.host, self.port, self.duration = host, port, duration
        self.round_timeout = round_timeout or max(5.0, 2 * self.p["window"])
        self.n, self.m = self.topology.n_sources, self.topology.n_brokers

        # Algorithm 1 state on planned rates (same initialization as the reference backend)
        self.state = algorithm1_initial_state(self.topology, self.p["delta_s"], self.p["eps"])

        self.slots = {"source": {}, "broker": {}}              # instance -> logical index
        self.channels = {"source": [None] * self.n, "broker": [None] * self.m}
        self.broker_addresses = [None] * self.m
        self.ready_sources = set()
        self.brokers_registered = asyncio.Event()
        self.sources_ready = asyncio.Event()
        self.stop = asyncio.Event()

        self.lambda_hat = np.zeros(self.m)
        self.br_failures_total = 0
        self.skipped_rounds = 0
        self.t0 = None

    # ---------------------------------------------------------------- registration
    def _assign(self, role: str, instance: str):
        slots = self.slots[role]
        if instance not in slots:
            limit = self.n if role == "source" else self.m
            if len(slots) >= limit:
                return None
            slots[instance] = len(slots)
        return slots[instance]

    async def _on_actor(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        channel = Channel(reader, writer)
        hello = await channel.recv()
        if hello is None:                                    # health-check probe: connect and close
            return await channel.close()
        if hello.get("type") == "shutdown":                  # from the launcher / an operator
            log("shutdown requested")
            self.stop.set()
            await channel.send({"type": "ok"})
            return await channel.close()
        if not hello or hello.get("type") != "register" or hello.get("protocol") != PROTOCOL_VERSION:
            await channel.send({"type": "refused", "reason": f"expected register (protocol {PROTOCOL_VERSION})"})
            return await channel.close()
        role, instance = hello.get("role"), str(hello.get("instance"))
        index = self._assign(role, instance) if role in self.slots else None
        if index is None:
            await channel.send({"type": "refused", "reason": f"no free {role} slot (topology has "
                                                             f"{self.n} sources, {self.m} brokers)"})
            return await channel.close()

        topo = self.topology
        if role == "broker":
            self.broker_addresses[index] = hello["data_address"]
            await channel.send({"type": "assignment", "index": index, "id": topo.brokers[index],
                                "mu": float(topo.mu_brokers[index]), "gamma": self.p["gamma"], "eps": self.p["eps"],
                                "price": float(self.state.prices[index]),
                                "seed": actor_seed(self.seed, "broker", index)})
            channel.start_reader()
            self.channels["broker"][index] = channel
            log(f"broker {topo.brokers[index]} registered ({instance}, data {hello['data_address']})")
            if all(a is not None for a in self.broker_addresses):
                self.brokers_registered.set()
        else:
            await self.brokers_registered.wait()          # sources need every broker's address
            brokers = [{"index": j, "id": topo.brokers[j], "host": self.broker_addresses[j][0],
                        "port": self.broker_addresses[j][1]} for j in range(self.m)]
            await channel.send({"type": "assignment", "index": index, "id": topo.sources[index],
                                "rate": float(topo.lambdas_total[index]), "mu_row": topo.mu_links[index],
                                "lambda_row": self.state.lambda_ij[index], "brokers": brokers,
                                "seed": actor_seed(self.seed, "source", index)})
            channel.start_reader()
            self.channels["source"][index] = channel
            ready = await channel.inbox.get()
            if ready.get("type") == "ready":
                self.ready_sources.add(index)
                log(f"source {topo.sources[index]} ready ({instance})")
                if self.t0 is not None:        # a restarted source rejoins a running emulation
                    await channel.send({"type": "start", "t0": time.time()})
                if len(self.ready_sources) == self.n:
                    self.sources_ready.set()
        await channel.closed.wait()
        if self.channels[role][index] is channel:
            self.channels[role][index] = None
            if role == "source":
                self.ready_sources.discard(index)
            if not self.stop.is_set():
                log(f"{role} {index} disconnected")

    # ---------------------------------------------------------------- rounds
    async def _gather(self, role: str, messages: list):
        async def one(index, message):
            channel = self.channels[role][index]
            if channel is None:
                raise ConnectionError(f"{role} {index} not connected")
            return await channel.request(message, self.round_timeout)
        return await asyncio.gather(*(one(k, msg) for k, msg in enumerate(messages)), return_exceptions=True)

    async def _round(self, k: int, last_time: float) -> float:
        topo, state, p = self.topology, self.state, self.p
        loads = compute_broker_loads(state.lambda_ij)

        # Step 1 at the brokers: damped price at the planned load
        replies = await self._gather("broker", [{"type": "price", "price": float(state.prices[j]),
                                                 "load": float(loads[j])} for j in range(self.m)])
        failed = [j for j, r in enumerate(replies) if isinstance(r, Exception)]
        if failed:
            self.skipped_rounds += 1
            log(f"round {k} skipped: no price from brokers {failed}")
            return last_time
        new_prices = np.array([r["price"] for r in replies], dtype=float)
        broker_metrics = [r["metrics"] for r in replies]

        # Step 2 at the sources: best responses to the new prices
        replies = await self._gather("source", [{"type": "best_response", "prices": new_prices}] * self.n)
        lambda_br = state.lambda_ij.copy()
        br_failures, source_metrics = [], []
        for i, r in enumerate(replies):
            if isinstance(r, Exception) or "failed" in r:
                br_failures.append(i)          # hold this source's split (Delta_ij = 0)
                source_metrics.append(None)
            else:
                lambda_br[i] = np.array(r["lambda_br"], dtype=float)
                source_metrics.append(r["metrics"])
        self.br_failures_total += len(br_failures)

        # Steps 3-4 at the controller: common safe step, routing update
        self.state, _, _ = apply_common_step(state, topo, lambda_br, new_prices, p["eta"], p["eps"], p["delta_s"],
                                             p["safe_step_variant"], br_failures)
        for i, channel in enumerate(self.channels["source"]):
            if channel is not None:
                await channel.send({"type": "routing", "round": k, "lambda_row": self.state.lambda_ij[i]})

        now = time.time()
        self._record(k, now, now - last_time, broker_metrics, source_metrics, br_failures)
        return now

    def _record(self, k, now, dt, broker_metrics, source_metrics, br_failures):
        topo, p = self.topology, self.p
        n, m = self.n, self.m
        arrivals = np.array([b["arrivals"] for b in broker_metrics], dtype=float)
        broker_rate = arrivals / dt
        self.lambda_hat = (1.0 - p["beta"]) * self.lambda_hat + p["beta"] * broker_rate
        routed = np.array([s["routed"] if s else [np.nan] * m for s in source_metrics], dtype=float)
        access_rate = routed / dt
        queue_access = np.array([s["in_system"] if s else [np.nan] * m for s in source_metrics], dtype=float)
        samples = [row for b in broker_metrics for row in b["samples"]]
        totals = np.array([s[1] for s in samples], dtype=float)
        if totals.size:
            p50, p95, p99 = np.percentile(totals, [50, 95, 99])
            mean = totals.mean()
            src = np.array([s[0] for s in samples], dtype=int)
            parts = np.array([s[2:] for s in samples], dtype=float).mean(axis=0)
        else:
            p50 = p95 = p99 = mean = np.nan
            src, parts = np.array([], dtype=int), np.full(5, np.nan)
        completed_total = int(sum(b["completed_total"] for b in broker_metrics))
        latency_sum_total = float(sum(b["latency_sum_total"] for b in broker_metrics))
        record = controller_snapshot(
            topo, self.state.lambda_ij, self.state.prices, iteration=k, sim_time=now - self.t0,
            controller_mode="capacity_safe_event_driven", s_j=self.state.s_j, s_t=self.state.s_t,
            route_rel=self.state.route_rel, price_rel=self.state.price_rel, eps=p["eps"],
            execution_backend="distributed",
            round_duration=dt,
            # Measured by the processes (the controller never uses these for control)
            lambda_hat_j=self.lambda_hat, util_measured_j=self.lambda_hat / topo.mu_brokers,
            broker_rate_measured_j=broker_rate,
            access_rate_measured_ij=access_rate,
            access_util_measured_ij=np.divide(access_rate, topo.mu_links, out=np.full((n, m), np.nan),
                                              where=topo.mu_links > 0),
            queue_access_ij=queue_access,
            queue_broker_j=[b["in_system"] for b in broker_metrics],
            completed_window=int(totals.size) if totals.size else int(sum(b["completed"] for b in broker_metrics)),
            completed_total=completed_total,
            latency_mean=mean, latency_p50=p50, latency_p95=p95, latency_p99=p99,
            latency_mean_i=[float(totals[src == i].mean()) if np.any(src == i) else np.nan for i in range(n)],
            latency_mean_total=latency_sum_total / completed_total if completed_total else np.nan,
            # End-to-end sojourn = queueing (the modeled M/M/1 stages) + transfer
            # (process-to-process hand-off: TCP plus timer granularity, not modeled)
            latency_parts={"access_wait": parts[0], "access_service": parts[1], "transfer": parts[2],
                           "broker_wait": parts[3], "broker_service": parts[4]},
            queueing_sojourn_mean=float(parts[0] + parts[1] + parts[3] + parts[4]),
            br_failures=br_failures, br_failures_total=self.br_failures_total,
            skipped_rounds=self.skipped_rounds,
        )
        self.telemetry.record(record)

    def _write_metadata(self):
        processes = {role: {inst: (self.topology.sources if role == "source" else self.topology.brokers)[idx]
                            for inst, idx in slots.items()} for role, slots in self.slots.items()}
        write_run_metadata(self.output_dir, self.config, self.topology, seed=self.seed,
                           controller_mode=self.settings["controller_mode"], real_time=True,
                           parameters=self.p, config_path=self.config_path,
                           extra={"execution_backend": "distributed", "notes": self.settings["notes"],
                                  "processes": processes,
                                  "actor_seeds": "numpy default_rng([root_seed, role(1=source, 2=broker), index])"})

    # ---------------------------------------------------------------- lifecycle
    async def run(self):
        self.telemetry = TelemetryBuffer(path=self.output_dir / "metrics.jsonl")
        self._write_metadata()
        for note in self.settings["notes"]:
            log(f"note: {note}")
        server = await asyncio.start_server(self._on_actor, self.host, self.port, limit=LINE_LIMIT)
        log(f"controller on {self.host}:{self.port}: waiting for {self.m} brokers and {self.n} sources; "
            f"Algorithm 1 eta={self.p['eta']} gamma={self.p['gamma']} delta_s={self.p['delta_s']}, "
            f"round every {self.p['window']}s")
        try:
            ready = asyncio.create_task(self.sources_ready.wait())
            stopping = asyncio.create_task(self.stop.wait())
            await asyncio.wait({ready, stopping}, return_when=asyncio.FIRST_COMPLETED)
            if not self.stop.is_set():
                self._write_metadata()                       # now with the process -> id mapping
                self.t0 = time.time() + 0.2
                for channel in self.channels["source"]:
                    await channel.send({"type": "start", "t0": self.t0})
                log(f"all {self.n + self.m} processes up; traffic started (Ctrl+C to stop)")
                await self._rounds()
        finally:
            await self._shutdown(server)

    async def _rounds(self):
        k, last, next_at = 0, self.t0, self.t0 + self.p["window"]
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=max(0.0, next_at - time.time()))
                break
            except asyncio.TimeoutError:
                pass
            k += 1
            last = await self._round(k, last)
            if k % 10 == 0 or k == 1:
                rec = self.telemetry.history[-1] if self.telemetry.history else {}
                log(f"round {k}: objective {rec.get('objective', float('nan')):.6f}, "
                    f"completed {rec.get('completed_total', 0)}, {rec.get('certificate_status', '')}")
            next_at += self.p["window"]
            if self.duration is not None and time.time() - self.t0 >= self.duration:
                break

    async def _shutdown(self, server):
        self.stop.set()
        for role in ("source", "broker"):
            for channel in self.channels[role]:
                if channel is not None:
                    try:
                        await channel.send({"type": "stop"})
                    except (ConnectionError, OSError):
                        pass
        await asyncio.sleep(0.2)
        server.close()
        self.telemetry.close()
        log(f"controller stopped; telemetry in {self.output_dir / 'metrics.jsonl'}")

def main():
    parser = argparse.ArgumentParser(description="Distributed emulator controller")
    parser.add_argument("--config", default=os.environ.get("TE_CONFIG", "config/paper_5x3.yaml"))
    parser.add_argument("--output-dir", default=os.environ.get("TE_OUTPUT_DIR", "runs/distributed"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CONTROL_PORT", CONTROL_PORT)))
    parser.add_argument("--duration", type=float, help="stop after this many seconds (default: run until stopped)")
    args = parser.parse_args()
    service = ControllerService(load_config(args.config), args.output_dir, args.host, args.port,
                                duration=args.duration, config_path=args.config)

    async def run():
        install_stop_handlers(service.stop)
        await service.run()
    asyncio.run(run())

if __name__ == "__main__":
    main()
