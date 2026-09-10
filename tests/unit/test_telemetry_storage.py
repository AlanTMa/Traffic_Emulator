import json
from src.telemetry.metrics import JsonlTail, TelemetryBuffer

def test_history_is_bounded_but_file_is_complete(tmp_path):
    path = tmp_path / "metrics.jsonl"
    buf = TelemetryBuffer(path=path, max_history=10)
    for k in range(100):
        buf.record({"iteration": k})
    buf.close()
    assert [r["iteration"] for r in buf.history] == list(range(90, 100))
    assert [json.loads(l)["iteration"] for l in path.read_text().splitlines()] == list(range(100))

def test_tail_reads_only_new_complete_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    tail = JsonlTail(path, max_rows=5)
    assert tail.read() == []
    with open(path, "w") as f:
        f.write('{"i": 0}\n{"i": 1}\n{"i": ')          # last line still being written
    assert [r["i"] for r in tail.read()] == [0, 1]
    with open(path, "a") as f:
        f.write('2}\n{"i": 3}\n')
    assert [r["i"] for r in tail.read()] == [0, 1, 2, 3]
    with open(path, "a") as f:
        f.write("".join(f'{{"i": {k}}}\n' for k in range(4, 10)))
    rows = tail.read()
    assert [r["i"] for r in rows] == [5, 6, 7, 8, 9]    # bounded to max_rows
    assert tail.total_rows == 10

def test_tail_resets_on_new_run(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"run": "a", "i": 0}\n{"run": "a", "i": 1}\n')
    tail = JsonlTail(path)
    assert len(tail.read()) == 2
    # A new run rewrites the file with a different first record
    path.write_text('{"run": "b", "i": 0}\n{"run": "b", "i": 1}\n{"run": "b", "i": 2}\n')
    rows = tail.read()
    assert [r["run"] for r in rows] == ["b", "b", "b"] and tail.total_rows == 3
