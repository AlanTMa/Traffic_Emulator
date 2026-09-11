"""Queues of the event-driven simulation."""
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict
import numpy as np

@dataclass
class Queue:
    """Waiting work-unit ids and a busy flag."""
    capacity: float  # mu
    queue: Deque[int] = field(default_factory=deque)
    is_busy: bool = False

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def push(self, request_id: int):
        self.queue.append(request_id)

    def pop(self) -> int:
        return self.queue.popleft()

@dataclass
class SimulationState:
    """Access queues [source][broker], broker queues, and per-request timestamps."""
    access_queues: Dict[int, Dict[int, Queue]]
    broker_queues: Dict[int, Queue]
    requests: Dict[int, Dict[str, float]]

    def __init__(self, n_sources: int, n_brokers: int, mu_links: np.ndarray, mu_brokers: np.ndarray,
                 keep_requests: bool = True):
        # keep_requests=False drops a request's record on completion (open-ended runs)
        self.keep_requests = keep_requests
        self.completed = 0
        self.latency_sum = 0.0
        self.access_queues = {
            i: {j: Queue(capacity=mu_links[i, j]) for j in range(n_brokers)}
            for i in range(n_sources)
        }
        self.broker_queues = {
            j: Queue(capacity=mu_brokers[j]) for j in range(n_brokers)
        }
        self.requests = {}
