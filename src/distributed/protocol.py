"""
Wire protocol of the distributed emulator: newline-delimited JSON over TCP.

Two kinds of connection:

- Control plane, actor <-> controller (one persistent connection per
  source/broker process). The actor sends `register`; the controller answers
  with the actor's assignment (logical id and parameters). Afterwards the
  controller sends requests (`price`, `best_response`) carrying a `req` id
  that the actor echoes in its reply, and notifications (`start`, `routing`,
  `stop`) that need no reply.
- Data plane, source -> broker (one persistent connection per source/broker
  pair): a stream of `work` messages, one per work unit, no replies.

Floats are serialized with repr, so numbers survive a round trip exactly;
the distributed controller therefore runs Algorithm 1 bit-for-bit like the
in-process reference.
"""
import asyncio
import itertools
import json
import os
import socket
from typing import Any, Dict, Optional

import numpy as np

PROTOCOL_VERSION = 1
LINE_LIMIT = 32 * 2 ** 20            # max bytes per message (latency sample batches)
CONTROL_PORT = 7000                  # controller default
DATA_PORT = 7100                     # broker data plane default (in containers)
ROLE_CODES = {"source": 1, "broker": 2}

def encode(message: Dict[str, Any]) -> bytes:
    return (json.dumps(message, default=_jsonable) + "\n").encode("utf-8")

def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")

def actor_seed(root_seed: int, role: str, index: int) -> list:
    """Deterministic per-process RNG seed derived from the run's root seed."""
    return [int(root_seed), ROLE_CODES[role], int(index)]

def parse_address(address: str, default_port: int = CONTROL_PORT) -> tuple:
    host, _, port = address.rpartition(":")
    return (host or address, int(port)) if port.isdigit() else (address, default_port)

def advertised_host() -> str:
    """Address other processes can reach this one at: ADVERTISE_HOST, else this host's IP."""
    if os.environ.get("ADVERTISE_HOST"):
        return os.environ["ADVERTISE_HOST"]
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"

class Channel:
    """
    One JSON-lines connection. `request()` sends a message with a fresh
    `req` id and waits for the reply with the same id; replies are routed by
    a background reader started with `start_reader()`. Messages without a
    matching pending request go to `inbox`.
    """
    _ids = itertools.count(1)

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader, self.writer = reader, writer
        self.pending: Dict[int, asyncio.Future] = {}
        self.inbox: asyncio.Queue = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task] = None
        self.closed = asyncio.Event()

    async def send(self, message: Dict[str, Any]):
        self.writer.write(encode(message))
        await self.writer.drain()

    async def recv(self) -> Optional[Dict[str, Any]]:
        """Next message, or None when the peer closed the connection."""
        try:
            line = await self.reader.readline()
        except (ConnectionError, asyncio.IncompleteReadError):
            return None
        if not line:
            return None
        return json.loads(line)

    def start_reader(self):
        self._reader_task = asyncio.create_task(self._read_loop())
        return self._reader_task

    async def _read_loop(self):
        try:
            while True:
                message = await self.recv()
                if message is None:
                    break
                future = self.pending.pop(message.get("req"), None) if "req" in message else None
                if future is not None and not future.done():
                    future.set_result(message)
                else:
                    await self.inbox.put(message)
        finally:
            self.closed.set()
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(ConnectionError("connection closed"))
            self.pending.clear()

    async def request(self, message: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        req = next(self._ids)
        future = asyncio.get_running_loop().create_future()
        self.pending[req] = future
        try:
            await self.send({**message, "req": req})
            return await asyncio.wait_for(future, timeout)
        finally:
            self.pending.pop(req, None)

    async def close(self):
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except (ConnectionError, OSError):
            pass

async def connect_with_retry(host: str, port: int, *, attempts: int = 120, delay: float = 0.5) -> Channel:
    """Open a Channel, retrying while the peer is not up yet (containers start in any order)."""
    last_error = None
    for _ in range(attempts):
        try:
            reader, writer = await asyncio.open_connection(host, port, limit=LINE_LIMIT)
            return Channel(reader, writer)
        except OSError as e:
            last_error = e
            await asyncio.sleep(delay)
    raise ConnectionError(f"could not reach {host}:{port}: {last_error}")
