"""
Discrete-event simulation engine.
"""
import heapq
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

    def run(self, duration: float, handler: Callable[[Event], None]):
        """
        Run the simulation until the duration is reached or the queue is empty.

        Args:
            duration: Total simulation time to run.
            handler: A callback function that processes each event.
        """
        while self.event_queue and self.now < duration:
            event = heapq.heappop(self.event_queue)
            self.now = event.timestamp
            handler(event)

    def stop(self):
        """Clear the queue to stop the simulation."""
        self.event_queue = []
