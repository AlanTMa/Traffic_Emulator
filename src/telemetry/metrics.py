"""
Telemetry buffer for recording system metrics over time.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import pandas as pd

@dataclass
class TelemetryBuffer:
    """
    Stores time-series data for all system components.
    """
    history: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, timestamp: float, iteration: int, state_data: Dict[str, Any]):
        """
        Add a new record to the history.
        state_data should contain values like 'objective', 'utilizations', etc.
        """
        record = {
            "timestamp": timestamp,
            "iteration": iteration,
            **state_data
        }
        self.history.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to a pandas DataFrame for analysis."""
        return pd.DataFrame(self.history)

    def save_to_csv(self, path: str):
        """Save the recorded data to a CSV file."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
