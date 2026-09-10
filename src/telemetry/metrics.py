"""
Telemetry buffer for recording system metrics over time.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

@dataclass
class TelemetryBuffer:
    """
    Stores telemetry records (see src/telemetry/schema.py). If `path` is set,
    every record is also appended to that JSON Lines file as it arrives.
    """
    path: Optional[Path] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.path is not None:
            self.path = Path(self.path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")  # a new run starts a new file

    def record(self, record: Dict[str, Any]):
        """Add one record (a JSON-serializable dict)."""
        self.history.append(record)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to a pandas DataFrame for analysis."""
        return pd.DataFrame(self.history)
