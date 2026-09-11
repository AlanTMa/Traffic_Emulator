"""Discrete-event engine."""
import heapq
import time
from typing import List, Callable
from src.simulation.events import Event, EventType

class SimulationEngine:
    """A heap of events; run() paces to the wall clock when asked."""
    def __init__(self):
        self.now = 0.0
        self.event_queue: List[Event] = []
        self.event_count = 0

    def schedule(self, event: Event):
        """Add an event."""
        heapq.heappush(self.event_queue, event)
        self.event_count += 1

    def run(self, duration: float, handler: Callable[[Event], None], real_time: bool = False):
        """Run until `duration` or an empty queue; real_time sleeps until each event's wall time."""
        # sleep to absolute times so handler time and overshoot don't accumulate
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
        """Clear the queue."""
        self.event_queue = []
