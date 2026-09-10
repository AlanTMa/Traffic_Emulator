"""
Configuration loader for the emulator.
"""
import yaml
from pathlib import Path
from src.model.topology import Topology
import numpy as np

# Controller modes (simulation.controller_mode):
#   static_algorithm1   - paper Algorithm 1 on the analytic model: no packets or
#                         queues; every iterate keeps Lambda_j <= mu_j - delta_s
#                         through the common safe step.
#   windowed_stochastic - event-driven queues driven by the reference notebook's
#                         windowed stochastic scheme (EWMA of measured arrival
#                         rates, damped prices, split inertia). No safe step:
#                         no per-iteration capacity guarantee.
CONTROLLER_MODES = ("static_algorithm1", "windowed_stochastic")
_LEGACY_MODES = {"static": "static_algorithm1", "synchronous": "windowed_stochastic",
                 "asynchronous": "windowed_stochastic"}

def load_config(config_path: str) -> dict:
    """Load experiment configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def controller_mode(config: dict) -> str:
    """
    Resolve simulation.controller_mode. The older simulation.mode values
    ('static', 'synchronous', 'asynchronous') are mapped for existing configs;
    none of them selected an asynchronous controller.
    """
    sim_cfg = config.get('simulation', {})
    mode = sim_cfg.get('controller_mode') or _LEGACY_MODES.get(sim_cfg.get('mode'), 'windowed_stochastic')
    if mode not in CONTROLLER_MODES:
        raise ValueError(f"unknown controller_mode {mode!r}; expected one of {CONTROLLER_MODES}")
    return mode

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
