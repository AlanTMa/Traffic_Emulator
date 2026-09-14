"""The numbers Sec. V of the paper reports (scripts/replicate_paper.py)."""
import pytest

from scripts.replicate_paper import PAPER, deterministic, windowed
from src.model.config import load_config, topology_from_config
from src.runtime.experiments import CANONICAL_5X3

@pytest.fixture(scope="module")
def topo():
    return topology_from_config(load_config(CANONICAL_5X3))

def failures(rows):
    return [(name, paper, value) for name, paper, value, ok in rows if ok is False]

def test_deterministic_results(topo):
    rows = deterministic(topo)
    assert failures(rows) == []
    value = {name: v for name, _, v, _ in rows}
    assert value["F* (V-B)"] == 2.0157 and value["F_dist (V-C)"] == 2.0157
    assert value["Algorithm 1 iterations (V-C)"] == 70
    assert value["utilizations, distributed (V-E)"] == PAPER["util"]

def test_windowed_experiment(topo):
    # three 300 s runs; single runs land within 1.2 / 4.6 / 6 % of the deterministic utilizations
    assert failures(windowed(topo, range(3))) == []
