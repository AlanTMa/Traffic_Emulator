"""
Queue state management for the event-driven simulation.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict
import numpy as np

@dataclass
class Queue:
    """
    Represents an M/M/1 queue.
    """
    capacity: float  # mu
    # Waiting work-unit ids, FIFO. deque: O(1) popleft even when a queue grows
    # long under high utilization (list.pop(0) is O(n)).
    queue: Deque[int] = field(default_factory=deque)
    is_busy: bool = False # True if the server is currently processing a request


    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def push(self, request_id: int):
        self.queue.append(request_id)

    def pop(self) -> int:
        return self.queue.popleft()

@dataclass
class SimulationState:
    """
    Full state of the event-driven simulation.
    """
    # Access queues: source_id -> {broker_id: Queue}
    access_queues: Dict[int, Dict[int, Queue]]
    # Broker queues: broker_id -> Queue
    broker_queues: Dict[int, Queue]
    # Request tracking: request_id -> {timestamps}
    requests: Dict[int, Dict[str, float]]

    def __init__(self, n_sources: int, n_brokers: int, mu_links: np.ndarray, mu_brokers: np.ndarray,
                 keep_requests: bool = True):
        # keep_requests=False drops each request's record once it completes,
        # so open-ended runs don't grow memory without bound; completed/
        # latency_sum still track the mean end-to-end latency.
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
