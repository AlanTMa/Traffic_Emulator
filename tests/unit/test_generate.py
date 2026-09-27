import numpy as np
import pytest
import yaml
from src.controller.feasibility import transportation_feasibility
from src.model.config import load_config, topology_from_config
from src.model.generate import (generate_instance, generate_topology_config, measured_beta, optimal_routing,
                                source_rates)
from src.runtime.experiments import CANONICAL_5X3

def test_paper_structure_tiled():
    paper = topology_from_config(load_config(CANONICAL_5X3))
    assert source_rates(5).tolist() == paper.lambdas_total.tolist()
    rates = source_rates(12)
    assert rates[5:10].tolist() == rates[:5].tolist() and rates[10:].tolist() == rates[:2].tolist()
    big = rates > 1.0
    assert big.sum() == 4 and rates[big].sum() / rates.sum() > 0.99      # two of every five carry 99.5%
    topo = topology_from_config({"topology": generate_topology_config(5, 3, 0.130290, seed=1)})
    assert topo.mu_brokers == pytest.approx(paper.mu_brokers, rel=1e-4)  # the paper's servers at its load

@pytest.mark.parametrize("n, m, rho", [(5, 3, 0.3), (20, 10, 0.9), (3, 7, 0.5)])
def test_generated_round_trip(n, m, rho):
    topo_cfg = generate_topology_config(n, m, rho, seed=7)
    saved = yaml.safe_load(yaml.safe_dump({"topology": topo_cfg}))
    assert saved["topology"] == topo_cfg          # no precision lost when saved
    topo = topology_from_config(saved)
    transportation_feasibility(topo.lambdas_total, topo.mu_links, topo.mu_brokers)
    assert topo.lambdas_total.sum() / topo.mu_brokers.sum() == pytest.approx(rho, rel=1e-12)
    assert np.all(topo.mu_links >= 40.0) and np.all(topo.mu_links <= 60.0)   # no beta: the paper's range

@pytest.mark.parametrize("beta", [4.0, 1.0, 0.5])
def test_beta_is_hit(beta):
    topology, stats = generate_instance(10, 5, 0.3, beta=beta, seed=42)
    assert stats["beta"] == pytest.approx(beta, rel=0.01) and stats["beta_rounds"] <= 8
    topo = topology_from_config({"topology": topology})
    assert measured_beta(optimal_routing(topo), topo) == pytest.approx(beta, rel=0.01)
    assert topo.lambdas_total.sum() / topo.mu_brokers.sum() == pytest.approx(0.3, rel=1e-12)

@pytest.mark.parametrize("rho, beta", [(0.5, 0.5), (0.7, 2.0)])
def test_fixed_point_failure_falls_back_to_slsqp(rho, beta):
    # 160 routes; on both, the fixed-point solver used above 100 routes found no certified point
    topology, stats = generate_instance(20, 8, rho, beta=beta, seed=42)
    assert stats["beta"] == pytest.approx(beta, rel=0.01)

def test_unreached_beta_is_an_error():
    with pytest.raises(RuntimeError, match="not reached"):
        generate_instance(5, 3, 0.13, beta=4.0, seed=42, beta_tol=-1.0)     # a negative tolerance is never met

def test_infeasible_request_is_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        generate_topology_config(5, 3, 0.13, beta=20.0, seed=42)     # links too tight for the 30 MB/s sources
    with pytest.raises(ValueError, match="rho_server"):
        generate_topology_config(5, 3, 1.0)
