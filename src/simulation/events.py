"""
Event definitions for the discrete-event simulation.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

class EventType(Enum):
    SOURCE_ARRIVAL = auto()
    ACCESS_SERVICE_START = auto()
    ACCESS_SERVICE_COMPLETE = auto()
    BROKER_ARRIVAL = auto()
    BROKER_SERVICE_START = auto()
    BROKER_SERVICE_COMPLETE = auto()
    # End of a controller window (price + routing update)
    CONTROLLER_TICK = auto()

@dataclass(order=True)
class Event:
    """
    A simulation event.
    Ordered by timestamp for the priority queue.
    """
    timestamp: float
    event_type: EventType = field(compare=False)
    source_id: Optional[int] = field(default=None, compare=False)
    broker_id: Optional[int] = field(default=None, compare=False)
    request_id: Optional[int] = field(default=None, compare=False)
    payload: Any = field(default=None, compare=False)
