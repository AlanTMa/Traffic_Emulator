from collections import deque
from src.simulation.queues import Queue

def test_queue_is_fifo_deque():
    q = Queue(capacity=1.0)
    assert isinstance(q.queue, deque) and q.is_empty()
    for rid in range(5):
        q.push(rid)
    assert [q.pop() for _ in range(5)] == [0, 1, 2, 3, 4]
    assert q.is_empty()

def test_queues_do_not_share_storage():
    a, b = Queue(capacity=1.0), Queue(capacity=1.0)
    a.push(1)
    assert b.is_empty()
