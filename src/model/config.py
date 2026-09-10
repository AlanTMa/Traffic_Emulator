"""
Configuration loader for the emulator.
"""
import yaml
from pathlib import Path
from src.model.topology import Topology
import numpy as np

# Controller modes (simulation.controller_mode):
#   static_algorithm1   - paper Algorithm 1 on the analytic model: no events or
#                         queues; every iterate keeps Lambda_j <= mu_j - delta_s
#                         through the common safe step.
#   windowed_stochastic - event-driven queues driven by the reference notebook's
#                         windowed stochastic scheme (EWMA of measured arrival
#                         rates, damped prices, split inertia). No safe step:
#                         no per-iteration capacity guarantee.
#   capacity_safe_event_driven - the same event-driven queues, with the planned
#                         routing advanced by one exact Algorithm 1 step per
#                         window (same function as static_algorithm1): planned
#                         Lambda_j <= mu_j - delta_s at every update. Measured
#                         quantities are recorded, never used for control.
CONTROLLER_MODES = ("static_algorithm1", "windowed_stochastic", "capacity_safe_event_driven")
_LEGACY_MODES = {"static": "static_algorithm1", "synchronous": "windowed_stochastic",
                 "asynchronous": "windowed_stochastic"}

class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys (PyYAML keeps the last one silently)."""

def _construct_unique_mapping(loader, node, deep=False):
    keys = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in keys:
            raise ValueError(f"duplicate key {key!r} in config (line {key_node.start_mark.line + 1})")
        keys.add(key)
    return loader.construct_mapping(node, deep=deep)

_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)

def load_config(config_path: str) -> dict:
    """Load experiment configuration from a YAML file (duplicate keys are an error)."""
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=_UniqueKeyLoader)

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
    Construct a Topology from a configuration dictionary.

    Expects topology.sources [{id, rate}], topology.brokers [{id, capacity}]
    and topology.access_capacities {"<source>-><broker>": capacity} with
    every source->broker pair exactly once. Missing, unknown, duplicate or
    malformed entries raise ValueError; no capacity is ever left at zero by
    omission. Values are then validated by Topology.
    """
    if not isinstance(config, dict) or not isinstance(config.get('topology'), dict):
        raise ValueError("config needs a 'topology' mapping")
    topo_cfg = config['topology']
    for section in ('sources', 'brokers', 'access_capacities'):
        if section not in topo_cfg:
            raise ValueError(f"topology.{section} is missing")

    def entries(section, value_key):
        items = topo_cfg[section]
        if not isinstance(items, list) or not items:
            raise ValueError(f"topology.{section} must be a non-empty list")
        ids, values = [], []
        for k, item in enumerate(items):
            if not isinstance(item, dict) or 'id' not in item or value_key not in item:
                raise ValueError(f"topology.{section}[{k}] needs 'id' and '{value_key}'")
            ids.append(str(item['id']))
            values.append(_number(item[value_key], f"topology.{section}[{k}].{value_key}"))
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate ids in topology.{section}: {dupes}")
        return ids, values

    sources, lambdas_total = entries('sources', 'rate')
    brokers, mu_brokers = entries('brokers', 'capacity')

    access_cfg = topo_cfg['access_capacities']
    if not isinstance(access_cfg, dict):
        raise ValueError('topology.access_capacities must be a mapping "<source>-><broker>": capacity')
    src_map = {name: i for i, name in enumerate(sources)}
    brk_map = {name: j for j, name in enumerate(brokers)}
    mu_links = np.zeros((len(sources), len(brokers)))
    specified = np.zeros(mu_links.shape, dtype=bool)   # not a value sentinel: NaN must stay an error
    for link, cap in access_cfg.items():
        parts = str(link).split('->')
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(f'malformed access link key {link!r}; expected "<source>-><broker>"')
        src_id, brk_id = parts[0].strip(), parts[1].strip()
        if src_id not in src_map:
            raise ValueError(f"access link {link!r}: unknown source {src_id!r}")
        if brk_id not in brk_map:
            raise ValueError(f"access link {link!r}: unknown broker {brk_id!r}")
        i, j = src_map[src_id], brk_map[brk_id]
        if specified[i, j]:
            raise ValueError(f"access link {src_id}->{brk_id} is specified more than once")
        mu_links[i, j] = _number(cap, f"access capacity {link!r}")
        specified[i, j] = True
    missing = [f"{sources[i]}->{brokers[j]}" for i, j in zip(*np.where(~specified))]
    if missing:
        raise ValueError(f"missing access capacities for {missing}")

    return Topology(
        lambdas_total=np.array(lambdas_total),
        mu_links=mu_links,
        mu_brokers=np.array(mu_brokers),
        sources=sources,
        brokers=brokers
    )

def _number(value, what: str) -> float:
    """float(value), with a clear error; PyYAML reads e.g. 1e-8 (no decimal point) as a string."""
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {value!r}") from None
