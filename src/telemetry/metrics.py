"""
Telemetry storage: append-only JSON Lines on disk, bounded history in memory.
"""
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

class TelemetryBuffer:
    """Keeps the last max_history records in memory and, given a path, appends every
    record to a JSONL file, flushed at most every flush_interval seconds."""
    def __init__(self, path: Optional[Path] = None, max_history: int = 2000, flush_interval: float = 1.0):
        self.history = deque(maxlen=max_history)
        self.path = Path(path) if path is not None else None
        self.flush_interval = flush_interval
        self.records_written = 0
        self._file = None
        self._last_flush = 0.0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "w", encoding="utf-8")  # a new run starts a new file

    def record(self, record: Dict[str, Any]):
        """Add one record (a JSON-serializable dict)."""
        self.history.append(record)
        if self._file is not None:
            self._file.write(json.dumps(record) + "\n")
            self.records_written += 1
            now = time.monotonic()
            if now - self._last_flush >= self.flush_interval:
                self._file.flush()
                self._last_flush = now

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

    def to_dataframe(self) -> pd.DataFrame:
        """The in-memory (most recent) records as a DataFrame."""
        return pd.DataFrame(list(self.history))

class JsonlTail:
    """
    Incremental reader for a growing JSON Lines file: each read() parses only
    the bytes appended since the previous call and keeps the last `max_rows`
    records. A replaced or truncated file (a new run) resets the reader.
    """
    def __init__(self, path: Path, max_rows: int = 3000):
        self.path = Path(path)
        self.rows = deque(maxlen=max_rows)
        self.total_rows = 0
        self._offset = 0
        self._first_line = None   # identifies the run

    def _reset(self):
        self.rows.clear()
        self.total_rows = 0
        self._offset = 0
        self._first_line = None

    def read(self) -> List[Dict[str, Any]]:
        for _ in range(2):  # a second pass after detecting a replaced file
            try:
                return self._read_new()
            except ValueError:
                self._reset()           # replaced under us
        return list(self.rows)

    def _read_new(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            self._reset()
            return []
        with open(self.path, "rb") as f:
            first_line = f.readline()
            size = f.seek(0, 2)
            # a shorter file or a different first line is a new run
            if size < self._offset or (self._first_line is not None and first_line != self._first_line):
                self._reset()
            if self._first_line is None and first_line.endswith(b"\n"):
                self._first_line = first_line
            f.seek(self._offset)
            chunk = f.read()
        # complete lines only
        end = chunk.rfind(b"\n") + 1
        parsed = [json.loads(line) for line in chunk[:end].splitlines() if line.strip()]
        self.rows.extend(parsed)
        self.total_rows += len(parsed)
        self._offset += end
        return list(self.rows)
