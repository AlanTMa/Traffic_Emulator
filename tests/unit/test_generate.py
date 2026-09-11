import numpy as np
import pytest
import yaml
from src.controller.feasibility import transportation_feasibility
from src.model.config import topology_from_config
from src.model.generate import generate_topology_config

@pytest.mark.parametrize("n, m, load", [(5, 3, 0.3), (20, 10, 0.9), (3, 7, 0.5)])
def test_generated_round_trip(n, m, load):
    topo_cfg = generate_topology_config(n, m, load, seed=7)
    saved = yaml.safe_load(yaml.safe_dump({"topology": topo_cfg}))
    assert saved["topology"] == topo_cfg          # no precision lost when saved
    topo = topology_from_config(saved)
    transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    assert topo.lambdas_total.sum() / topo.mu_brokers.sum() == pytest.approx(load, rel=1e-12)

def test_infeasible_request_is_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        generate_topology_config(2, 1, 0.9, seed=42)
