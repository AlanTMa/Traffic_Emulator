"""
Configuration loader for the emulator.
"""
import yaml
from pathlib import Path
from src.model.topology import Topology
import numpy as np

def load_config(config_path: str) -> dict:
    """Load experiment configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def topology_from_config(config: dict) -> Topology:
    """
    Construct a Topology object from a configuration dictionary.
    """
    topo_cfg = config['topology']

    # 1. Sources and their offered rates
    sources = []
    lambdas_total = []
    for src in topo_cfg['sources']:
        sources.append(src['id'])
        lambdas_total.append(float(src['rate'])) # Ensure float

    # 2. Brokers and their capacities
    brokers = []
    mu_brokers = []
    for brk in topo_cfg['brokers']:
        brokers.append(brk['id'])
        mu_brokers.append(float(brk['capacity'])) # Ensure float


    # 3. Access capacities (mu_ij)
    # Expects a map: "SrcID->BrkID": value
    # Or a matrix if provided.
    access_cfg = topo_cfg['access_capacities']
    n_src = len(sources)
    n_brk = len(brokers)
    mu_links = np.zeros((n_src, n_brk))

    # Create mapping for quick lookup
    src_map = {name: i for i, name in enumerate(sources)}
    brk_map = {name: i for i, name in enumerate(brokers)}

    for link, cap in access_cfg.items():
        src_id, brk_id = link.split('->')
        if src_id in src_map and brk_id in brk_map:
            mu_links[src_map[src_id], brk_map[brk_id]] = float(cap) # Ensure float

    return Topology(
        lambdas_total=np.array(lambdas_total),
        mu_links=mu_links,
        mu_brokers=np.array(mu_brokers),
        sources=sources,
        brokers=brokers
    )
