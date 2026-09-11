"""Config loading and validation."""
import copy
import yaml
from pathlib import Path
from src.model.topology import Topology
import numpy as np

# static_algorithm1: Algorithm 1 on the analytic model, no queues
# windowed_stochastic: the notebook's scheme over event-driven queues, no safe step
# capacity_safe_event_driven: Algorithm 1 steps on planned rates over event-driven queues
CONTROLLER_MODES = ("static_algorithm1", "windowed_stochastic", "capacity_safe_event_driven")
_LEGACY_MODES = {"static": "static_algorithm1", "synchronous": "windowed_stochastic",
                 "asynchronous": "windowed_stochastic"}
ALGORITHM1_MODES = ("static_algorithm1", "capacity_safe_event_driven")
# defaults: paper Sec. V-A for Algorithm 1, notebook cell 14 for the windowed scheme
MODE_DEFAULTS = {
    "static_algorithm1": {"eta": 0.25, "gamma": 0.5, "delta_s": 1e-8, "eps": 1e-12},
    "capacity_safe_event_driven": {"eta": 0.25, "gamma": 0.5, "delta_s": 1e-8, "eps": 1e-12, "beta": 0.3},
    "windowed_stochastic": {"eta": 0.35, "gamma": 0.5, "beta": 0.3},
}

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
    """simulation.controller_mode; the old simulation.mode names are mapped."""
    sim_cfg = config.get('simulation', {})
    mode = sim_cfg.get('controller_mode') or _LEGACY_MODES.get(sim_cfg.get('mode'), 'windowed_stochastic')
    if mode not in CONTROLLER_MODES:
        raise ValueError(f"unknown controller_mode {mode!r}; expected one of {CONTROLLER_MODES}")
    return mode

def with_controller_mode(config: dict, mode: str) -> dict:
    """
    Copy of the config under `mode`. eta is the split inertia in the windowed
    scheme and the step size otherwise, so switching families swaps in that
    mode's defaults; between the two Algorithm 1 modes the parameters are kept.
    """
    if mode not in CONTROLLER_MODES:
        raise ValueError(f"unknown controller_mode {mode!r}; expected one of {CONTROLLER_MODES}")
    current = controller_mode(config)
    config = copy.deepcopy(config)
    simulation = config.get("simulation") or {}
    simulation.pop("mode", None)                                   # legacy key
    simulation["controller_mode"] = mode
    config["simulation"] = simulation
    if mode != current:
        keep = current in ALGORITHM1_MODES and mode in ALGORITHM1_MODES
        previous = config.get("algorithm") or {}
        config["algorithm"] = {k: previous.get(k, v) if keep else v for k, v in MODE_DEFAULTS[mode].items()}
    return config

def topology_from_config(config: dict) -> Topology:
    """
    topology.sources [{id, rate}], brokers [{id, capacity}], access_capacities
    {"S->B": mu} and optional unavailable_links ["S->B"]; every pair must appear
    in exactly one of the two.
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
    specified = np.zeros(mu_links.shape, dtype=bool)   # NaN must stay an error
    route_mask = np.ones(mu_links.shape, dtype=bool)

    def link_index(link):
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
        specified[i, j] = True
        return i, j

    for link, cap in access_cfg.items():
        i, j = link_index(link)
        mu_links[i, j] = _number(cap, f"access capacity {link!r}")
    unavailable = topo_cfg.get('unavailable_links') or []
    if not isinstance(unavailable, list):
        raise ValueError('topology.unavailable_links must be a list of "<source>-><broker>" links')
    for link in unavailable:
        i, j = link_index(link)       # also rejects a link both listed here and given a capacity
        route_mask[i, j] = False
    missing = [f"{sources[i]}->{brokers[j]}" for i, j in zip(*np.where(~specified))]
    if missing:
        raise ValueError(f"missing access capacities for {missing} "
                         "(list links that do not exist under topology.unavailable_links)")

    return Topology(
        lambdas_total=np.array(lambdas_total),
        mu_links=mu_links,
        mu_brokers=np.array(mu_brokers),
        sources=sources,
        brokers=brokers,
        route_mask=route_mask,
    )

def _number(value, what: str) -> float:
    """float(value), with a clear error; PyYAML reads e.g. 1e-8 (no decimal point) as a string."""
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {value!r}") from None
