"""
Source process: one source's Poisson arrivals, one access queue per link
served at Exp(mu_ij), routing by the current plan, and its best response
(synchronous.source_best_response). The TCP hop to the broker is transport,
not a model of mu_ij.

    python -m src.distributed.source --controller HOST:PORT
"""
import argparse
import asyncio
import itertools
import os
import socket
import time
from collections import deque

import numpy as np

from src.controller.synchronous import source_best_response
from src.distributed.protocol import PROTOCOL_VERSION, connect_with_retry, encode, parse_address
from src.distributed.runtime import install_stop_handlers, log

class SourceWorker:
    def __init__(self, controller: str, instance: str):
        self.controller = parse_address(controller)
        self.instance = instance
        self.stop = asyncio.Event()
        self.unit_ids = itertools.count()
        self.generated_total = 0

    # ---------------------------------------------------------------- traffic
    def _route(self) -> int:
        """Pick a broker with probabilities lambda_ij / lambda_i."""
        probs = self.x_row / self.x_row.sum() if self.x_row.sum() > 0 else self.mask / self.mask.sum()
        return int(self.rng_route.choice(len(probs), p=probs))

    async def _generate(self, t0: float):
        next_t = t0
        while not self.stop.is_set():
            next_t += self.rng_arrivals.exponential(1.0 / self.rate)
            delay = next_t - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            j = self._route()
            self.routed[j] += 1
            self.generated_total += 1
            self.queues[j].append({"w": f"{self.index}-{next(self.unit_ids)}", "s": self.index, "b": j, "g": next_t})
            self.wakeups[j].set()

    async def _serve_access(self, j: int):
        """Access link (i, j): FIFO, Exp(mu_ij) service on the link's virtual clock."""
        queue, wake, writer, rng = self.queues[j], self.wakeups[j], self.writers[j], self.rng_service[j]
        free_at = 0.0
        while not self.stop.is_set():
            if not queue:
                wake.clear()
                await wake.wait()
                continue
            unit = queue.popleft()
            self.busy[j] = True
            start = max(unit["g"], free_at)
            free_at = start + rng.exponential(1.0 / self.mu_row[j])
            delay = free_at - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self.busy[j] = False
            unit["as"], unit["ac"] = start, free_at
            writer.write(encode(unit))
            await writer.drain()

    def metrics(self) -> dict:
        report = {
            "routed": self.routed.tolist(),                            # this round
            "in_system": [len(q) + int(b) for q, b in zip(self.queues, self.busy)],
            "generated_total": self.generated_total,
        }
        self.routed[:] = 0
        return report

    # ---------------------------------------------------------------- control plane
    async def run(self):
        control = await connect_with_retry(*self.controller)
        await control.send({"type": "register", "role": "source", "instance": self.instance,
                            "protocol": PROTOCOL_VERSION})
        a = await control.recv()                 # sent once every broker has registered
        if a is None or a.get("type") != "assignment":
            raise RuntimeError(f"registration refused: {a}")
        self.index, self.id, self.rate = a["index"], a["id"], a["rate"]
        self.mu_row = np.array(a["mu_row"], dtype=float)
        self.mask = self.mu_row > 0
        self.x_row = np.array(a["lambda_row"], dtype=float)
        m = len(self.mu_row)
        # separate streams, so draws don't depend on task interleaving
        streams = [np.random.default_rng(s) for s in np.random.SeedSequence(a["seed"]).spawn(2 + m)]
        self.rng_arrivals, self.rng_route, self.rng_service = streams[0], streams[1], streams[2:]
        self.queues = [deque() for _ in range(m)]
        self.wakeups = [asyncio.Event() for _ in range(m)]
        self.busy = [False] * m
        self.routed = np.zeros(m, dtype=int)

        # one data connection per reachable broker
        self.writers = [None] * m
        for broker in a["brokers"]:
            if self.mask[broker["index"]]:
                channel = await connect_with_retry(broker["host"], broker["port"])
                self.writers[broker["index"]] = channel.writer
        await control.send({"type": "ready"})
        log(f"source {self.id} (lambda={self.rate}) connected to {int(self.mask.sum())} brokers")

        tasks = []
        stop_wait = asyncio.create_task(self.stop.wait())
        try:
            while True:
                recv = asyncio.create_task(control.recv())
                done, _ = await asyncio.wait({recv, stop_wait}, return_when=asyncio.FIRST_COMPLETED)
                if stop_wait in done:
                    recv.cancel()
                    break
                message = recv.result()
                kind = None if message is None else message.get("type")
                if kind in (None, "stop"):
                    break
                if kind == "start":
                    tasks = [asyncio.create_task(self._generate(message["t0"]))]
                    tasks += [asyncio.create_task(self._serve_access(j)) for j in range(m) if self.mask[j]]
                elif kind == "best_response":
                    reply = {"type": "best_response", "req": message["req"], "metrics": self.metrics()}
                    try:
                        reply["lambda_br"] = source_best_response(self.mu_row, np.array(message["prices"]), self.rate)
                    except (RuntimeError, ValueError) as e:
                        reply["failed"] = str(e)          # the controller holds the split
                    await control.send(reply)
                elif kind == "routing":
                    self.x_row = np.array(message["lambda_row"], dtype=float)
        finally:
            for task in tasks:
                task.cancel()
            await control.close()
            log(f"source {getattr(self, 'id', '?')} stopped")

def main():
    parser = argparse.ArgumentParser(description="Distributed emulator source worker")
    parser.add_argument("--controller", default=os.environ.get("CONTROLLER", "localhost:7000"))
    parser.add_argument("--instance", default=os.environ.get("INSTANCE", socket.gethostname() + f"-{os.getpid()}"))
    args = parser.parse_args()
    worker = SourceWorker(args.controller, args.instance)

    async def run():
        install_stop_handlers(worker.stop)
        await worker.run()
    asyncio.run(run())

if __name__ == "__main__":
    main()
