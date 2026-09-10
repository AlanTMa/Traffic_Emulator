"""
Discrete-event simulation engine.
"""
import heapq
import time
from typing import List, Callable
from src.simulation.events import Event, EventType

class SimulationEngine:
    """
    Priority-queue based event engine for the traffic emulator.
    """
    def __init__(self):
        self.now = 0.0
        self.event_queue: List[Event] = []
        self.event_count = 0

    def schedule(self, event: Event):
        """Schedule a new event into the priority queue."""
        heapq.heappush(self.event_queue, event)
        self.event_count += 1

    def run(self, duration: float, handler: Callable[[Event], None], real_time: bool = False):
        """
        Run the simulation until the duration is reached or the queue is empty.

        Args:
            duration: Total simulation time to run.
            handler: A callback function that processes each event.
            real_time: If True, pace the simulation to match the wall clock.
        """
        # Wall-clock time corresponding to simulated time zero. Sleeping until
        # start + timestamp (rather than for each inter-event gap) keeps handler
        # time and sleep overshoot from accumulating as drift.
        wall_start = time.perf_counter() - self.now
        while self.event_queue and self.event_queue[0].timestamp <= duration:
            event = heapq.heappop(self.event_queue)

            if real_time:
                delay = wall_start + event.timestamp - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

            self.now = event.timestamp
            handler(event)

    def stop(self):
        """Clear the queue to stop the simulation."""
        self.event_queue = []
