"""Events of the discrete-event simulation."""
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
    CONTROLLER_TICK = auto()      # end of a controller window

@dataclass(order=True)
class Event:
    """Ordered by timestamp."""
    timestamp: float
    event_type: EventType = field(compare=False)
    source_id: Optional[int] = field(default=None, compare=False)
    broker_id: Optional[int] = field(default=None, compare=False)
    request_id: Optional[int] = field(default=None, compare=False)
    payload: Any = field(default=None, compare=False)
