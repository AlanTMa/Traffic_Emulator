import copy
import math

import numpy as np
import pytest

from src.model.config import load_config, topology_from_config
from src.model.topology import Topology

VALID = {
    "topology": {
        "sources": [{"id": "A", "rate": 3.0}, {"id": "B", "rate": 1.0}],
        "brokers": [{"id": "S1", "capacity": 10.0}, {"id": "S2", "capacity": 8.0}],
        "access_capacities": {"A->S1": 5.0, "A->S2": 6.0, "B->S1": 4.0, "B->S2": 7.0},
    }
}

def with_change(fn):
    cfg = copy.deepcopy(VALID)
    fn(cfg["topology"])
    return cfg

def test_valid_config_parses():
    t = topology_from_config(VALID)
    assert t.mu_links.tolist() == [[5.0, 6.0], [4.0, 7.0]]
    assert t.sources == ["A", "B"] and t.brokers == ["S1", "S2"]

def test_whitespace_around_link_ids_is_accepted():
    t = topology_from_config(with_change(lambda t: t.update(access_capacities={
        "A -> S1": 5.0, "A->S2": 6.0, "B->S1": 4.0, "B->S2": 7.0})))
    assert t.mu_links[0, 0] == 5.0

# --- topology_from_config: structure ---

@pytest.mark.parametrize("change, match", [
    (lambda t: t["access_capacities"].pop("B->S2"), r"missing access capacities for \['B->S2'\]"),
    (lambda t: t["access_capacities"].update({"C->S1": 1.0}), "unknown source 'C'"),
    (lambda t: t["access_capacities"].update({"A->S9": 1.0}), "unknown broker 'S9'"),
    (lambda t: t["access_capacities"].update({"A->S1 ": 2.0}), "specified more than once"),
    (lambda t: t["access_capacities"].update({"A-S1": 1.0}), "malformed access link key"),
    (lambda t: t["access_capacities"].update({"A->S1->S2": 1.0}), "malformed access link key"),
    (lambda t: t["access_capacities"].update({"->S1": 1.0}), "malformed access link key"),
    (lambda t: t["sources"].append({"id": "A", "rate": 1.0}), r"duplicate ids in topology.sources: \['A'\]"),
    (lambda t: t["brokers"].append({"id": "S2", "capacity": 1.0}), r"duplicate ids in topology.brokers"),
    (lambda t: t.pop("access_capacities"), "topology.access_capacities is missing"),
    (lambda t: t.update(sources=[]), "must be a non-empty list"),
    (lambda t: t["sources"][0].pop("rate"), "needs 'id' and 'rate'"),
    (lambda t: t["brokers"][1].update(capacity="fast"), "must be a number"),
    (lambda t: t["access_capacities"].update({"A->S1": None}), "must be a number"),
    (lambda t: t.update(access_capacities=[["A", "S1", 5.0]]), "must be a mapping"),
])
def test_malformed_configs_fail_loudly(change, match):
    with pytest.raises(ValueError, match=match):
        topology_from_config(with_change(change))

def test_missing_topology_section():
    with pytest.raises(ValueError, match="'topology' mapping"):
        topology_from_config({"simulation": {}})

# --- topology_from_config -> Topology: values ---

@pytest.mark.parametrize("change, match", [
    (lambda t: t["sources"][1].update(rate=-1.0), r"source rates must be >= 0 \(negative for \['B'\]\)"),
    (lambda t: t["brokers"][0].update(capacity=0.0), r"broker capacities must be > 0 \(not for \['S1'\]\)"),
    (lambda t: t["brokers"][1].update(capacity=-3.0), "broker capacities must be > 0"),
    (lambda t: t["access_capacities"].update({"A->S2": 0.0}), r"access capacities must be > 0 \(not for \['A->S2'\]\)"),
    (lambda t: t["access_capacities"].update({"B->S1": -1.0}), "access capacities must be > 0"),
    (lambda t: t["sources"][0].update(rate=math.nan), "lambdas_total contains NaN or infinite"),
    (lambda t: t["brokers"][0].update(capacity=math.inf), "mu_brokers contains NaN or infinite"),
    (lambda t: t["access_capacities"].update({"A->S1": "nan"}), "mu_links contains NaN or infinite"),
])
def test_invalid_values_fail_loudly(change, match):
    with pytest.raises(ValueError, match=match):
        topology_from_config(with_change(change))

def test_zero_rate_source_is_allowed():
    t = topology_from_config(with_change(lambda t: t["sources"][1].update(rate=0.0)))
    assert t.lambdas_total[1] == 0.0

# --- Topology directly: dimensions ---

@pytest.mark.parametrize("kwargs, match", [
    (dict(lambdas_total=[1.0, 2.0, 3.0]), r"lambdas_total shape \(3,\) must be \(2,\)"),
    (dict(mu_links=[[5.0, 6.0]]), r"mu_links shape \(1, 2\) must be \(2, 2\)"),
    (dict(mu_links=[5.0, 6.0, 4.0, 7.0]), r"mu_links shape \(4,\) must be \(2, 2\)"),
    (dict(mu_brokers=[[10.0, 8.0]]), r"mu_brokers shape \(1, 2\) must be \(2,\)"),
    (dict(sources=[], lambdas_total=[], mu_links=np.zeros((0, 2))), "at least one source"),
    (dict(sources=["A", "A"]), "duplicate source ids"),
    (dict(mu_brokers=["ten", 8.0]), "must be numeric"),
])
def test_topology_rejects_malformed_arrays(kwargs, match):
    base = dict(lambdas_total=[3.0, 1.0], mu_links=[[5.0, 6.0], [4.0, 7.0]], mu_brokers=[10.0, 8.0],
                sources=["A", "B"], brokers=["S1", "S2"])
    with pytest.raises(ValueError, match=match):
        Topology(**{**base, **kwargs})

# --- YAML loading ---

def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "dup.yaml"
    path.write_text("topology:\n  access_capacities:\n    A->S1: 5.0\n    A->S1: 9.0\n")
    with pytest.raises(ValueError, match="duplicate key 'A->S1'"):
        load_config(path)

def test_shipped_configs_are_valid():
    from pathlib import Path
    for path in sorted(Path(__file__).resolve().parents[2].joinpath("config").glob("*.yaml")):
        topology_from_config(load_config(path))

# --- Feasibility LP at the boundary ---

def test_feasibility_rejects_exactly_critical_instances():
    from src.controller.feasibility import transportation_feasibility
    # One source sending exactly its two links' capacity: zero headroom
    with pytest.raises(ValueError, match="headroom"):
        transportation_feasibility(np.array([10.0]), np.array([[5.0, 5.0]]), np.array([100.0, 100.0]))
    # Brokers exactly full
    with pytest.raises(ValueError, match="headroom"):
        transportation_feasibility(np.array([6.0, 6.0]), np.full((2, 2), 10.0), np.array([6.0, 6.0]))
    # Just inside the boundary: feasible, with the requested headroom
    L = transportation_feasibility(np.array([9.99]), np.array([[5.0, 5.0]]), np.array([100.0, 100.0]))
    assert np.min(5.0 - L) >= 1e-8 and L.sum() == pytest.approx(9.99)
