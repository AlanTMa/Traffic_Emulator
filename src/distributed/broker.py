"""
Broker worker process (distributed backend).

Owns one broker j: its service rate mu_j, its queue of incoming work units,
their exponential service, completion/latency accounting, and its published
congestion price, updated each controller round with the shared Algorithm 1
formula (synchronous.broker_price).

Service is paced on a virtual clock tied to the wall clock: a unit starts
service at max(its arrival, the server's free time) and completes Exp(mu_j)
later; the worker sleeps until that completion time. Late wake-ups therefore
never slow the server down, so the queue is an M/M/1 queue in wall time.
mu_j is this model parameter - not the container's CPU speed.

Run directly (normally started by the launcher or Docker Compose):
    python -m src.distributed.broker --controller HOST:PORT [--port 0]
"""
import argparse
import asyncio
import os
import socket
import time
from collections import deque

import numpy as np

from src.controller.synchronous import broker_price
from src.distributed.protocol import (LINE_LIMIT, Channel, advertised_host, connect_with_retry, encode,
                                      parse_address, PROTOCOL_VERSION)
from src.distributed.runtime import install_stop_handlers, log

MAX_SAMPLES = 5000   # latency samples reported per round (uniformly thinned beyond this)

class BrokerWorker:
    def __init__(self, controller: str, instance: str, port: int = 0, bind: str = "0.0.0.0"):
        self.controller = parse_address(controller)
        self.instance = instance
        self.port, self.bind = port, bind
        self.queue = deque()
        self.queue_event = asyncio.Event()
        self.busy = False
        self.stop = asyncio.Event()
        self._reset_window()
        self.completed_total, self.latency_sum_total = 0, 0.0

    def _reset_window(self):
        self.arrivals = 0
        # (source, total, access_wait, access_service, transfer, broker_wait, broker_service)
        self.samples = []
        self.completed = 0

    # ---------------------------------------------------------------- data plane
    async def _on_source(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        channel = Channel(reader, writer)
        while True:
            unit = await channel.recv()
            if unit is None:
                break
            unit["ba"] = time.time()          # broker arrival
            self.arrivals += 1
            self.queue.append(unit)
            self.queue_event.set()

    async def _serve(self):
        free_at = 0.0
        while not self.stop.is_set():
            if not self.queue:
                self.queue_event.clear()
                await self.queue_event.wait()
                continue
            unit = self.queue.popleft()
            self.busy = True
            start = max(unit["ba"], free_at)
            free_at = start + self.rng.exponential(1.0 / self.mu)
            delay = free_at - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self.busy = False
            unit["bs"], unit["bc"] = start, free_at
            self._complete(unit)

    def _complete(self, u):
        total = u["bc"] - u["g"]
        self.completed += 1
        self.completed_total += 1
        self.latency_sum_total += total
        self.samples.append((u["s"], total, u["as"] - u["g"], u["ac"] - u["as"], u["ba"] - u["ac"],
                             u["bs"] - u["ba"], u["bc"] - u["bs"]))

    # ---------------------------------------------------------------- control plane
    def metrics(self) -> dict:
        """Measurements since the previous round, then reset the window."""
        samples = self.samples
        if len(samples) > MAX_SAMPLES:       # bounded message size
            keep = np.sort(self.sample_rng.choice(len(samples), MAX_SAMPLES, replace=False))
            samples = [samples[k] for k in keep]
        report = {
            "arrivals": self.arrivals, "completed": self.completed,
            "in_system": len(self.queue) + int(self.busy),
            "completed_total": self.completed_total, "latency_sum_total": self.latency_sum_total,
            "samples": samples,
        }
        self._reset_window()
        return report

    async def run(self):
        server = await asyncio.start_server(self._on_source, self.bind, self.port, limit=LINE_LIMIT)
        port = server.sockets[0].getsockname()[1]
        control = await connect_with_retry(*self.controller)
        await control.send({"type": "register", "role": "broker", "instance": self.instance,
                            "protocol": PROTOCOL_VERSION, "data_address": [advertised_host(), port]})
        assignment = await control.recv()
        if assignment is None or assignment.get("type") != "assignment":
            raise RuntimeError(f"registration refused: {assignment}")
        self.index, self.id = assignment["index"], assignment["id"]
        self.mu, self.gamma, self.eps = assignment["mu"], assignment["gamma"], assignment["eps"]
        self.price = assignment["price"]
        self.rng = np.random.default_rng(assignment["seed"])
        self.sample_rng = np.random.default_rng(assignment["seed"] + [7])
        log(f"broker {self.id} (mu={self.mu}) serving on port {port}")

        serve = asyncio.create_task(self._serve())
        stop_wait = asyncio.create_task(self.stop.wait())
        try:
            while True:
                recv = asyncio.create_task(control.recv())
                done, _ = await asyncio.wait({recv, stop_wait}, return_when=asyncio.FIRST_COMPLETED)
                if stop_wait in done:
                    recv.cancel()
                    break
                message = recv.result()
                if message is None or message.get("type") == "stop":
                    break
                if message.get("type") == "price":
                    # Algorithm 1 Step 1 at this broker: damped price at the planned load. The
                    # controller sends the current published price with the request, so a
                    # skipped round cannot leave broker and controller disagreeing.
                    self.price = broker_price(message["price"], message["load"], self.mu, self.gamma, self.eps)
                    await control.send({"type": "price", "req": message["req"], "price": self.price,
                                        "metrics": self.metrics()})
        finally:
            serve.cancel()
            server.close()
            await control.close()
            log(f"broker {getattr(self, 'id', '?')} stopped")

def main():
    parser = argparse.ArgumentParser(description="Distributed emulator broker worker")
    parser.add_argument("--controller", default=os.environ.get("CONTROLLER", "localhost:7000"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DATA_PORT", "0")),
                        help="data-plane port for incoming work (0: any free port)")
    parser.add_argument("--instance", default=os.environ.get("INSTANCE", socket.gethostname() + f"-{os.getpid()}"))
    args = parser.parse_args()
    worker = BrokerWorker(args.controller, args.instance, args.port)

    async def run():
        install_stop_handlers(worker.stop)
        await worker.run()
    asyncio.run(run())

if __name__ == "__main__":
    main()
