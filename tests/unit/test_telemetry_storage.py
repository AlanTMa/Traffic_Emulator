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

def test_tail_detects_new_run_that_outgrows_old_offset(tmp_path):
    # Records of every run start with the same long prefix; only the first
    # record's wall_time tells runs apart. If the new run has already grown
    # past the old offset, a prefix check would resume mid-record.
    path = tmp_path / "metrics.jsonl"
    prefix = '{"schema": 1, "controller_mode": "windowed_stochastic", "iteration": %d, "wall_time": %s, "x": "%s"}\n'
    path.write_text("".join(prefix % (k, "100.0", "a" * 50) for k in range(3)))
    tail = JsonlTail(path)
    assert len(tail.read()) == 3
    path.write_text("".join(prefix % (k, "200.0", "b" * 80) for k in range(6)))
    rows = tail.read()
    assert [r["wall_time"] for r in rows] == [200.0] * 6
    assert tail.total_rows == 6

def test_tail_first_seen_mid_write_then_new_run(tmp_path):
    # The reader first sees run A's first record half-written (only the
    # prefix shared by all runs), then A's complete records, then run B,
    # which has already outgrown A's offset. B must be read from its start.
    path = tmp_path / "metrics.jsonl"
    record = '{"schema": 1, "controller_mode": "windowed_stochastic", "iteration": %d, "wall_time": %s}\n'
    run_a = "".join(record % (k, "100.0") for k in range(3))
    path.write_text(run_a[:40])
    tail = JsonlTail(path)
    assert tail.read() == []
    path.write_text(run_a)
    assert len(tail.read()) == 3
    path.write_text("".join(record % (k, "200.0") for k in range(8)))
    rows = tail.read()
    assert [r["wall_time"] for r in rows] == [200.0] * 8
    assert [r["iteration"] for r in rows] == list(range(8))
