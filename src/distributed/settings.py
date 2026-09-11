"""
Parameters of a distributed run. A config written for the windowed scheme
(paper_5x3.yaml: its eta is the split inertia) contributes topology, window
and seed only; Algorithm 1 then uses the paper's eta 0.25, gamma 0.5,
delta_s 1e-8, and the substitution is recorded in `notes`.
"""
from src.controller.synchronous import SAFE_STEP_VARIANTS
from src.model.config import controller_mode
from src.runtime.metadata import resolve_seed

ALGORITHM1_MODES = ("static_algorithm1", "capacity_safe_event_driven")
ALGORITHM1_DEFAULTS = {"eta": 0.25, "gamma": 0.5, "delta_s": 1e-8, "eps": 1e-12}

def distributed_settings(config: dict, overrides: dict = None) -> dict:
    """Resolved controller parameters, seed and notes for a distributed run."""
    if (config.get("dynamics") or {}).get("capacity_variation"):
        raise ValueError("the distributed backend runs static capacities only; remove the dynamics section")
    sim = config.get("simulation", {}) or {}
    alg = config.get("algorithm", {}) or {}
    mode = controller_mode(config)
    notes = []
    if mode in ALGORITHM1_MODES:
        params = {k: float(alg.get(k, v)) for k, v in ALGORITHM1_DEFAULTS.items()}
    else:
        params = dict(ALGORITHM1_DEFAULTS)
        notes.append(f"config is {mode}; Algorithm 1 uses eta={params['eta']}, gamma={params['gamma']}, "
                     f"delta_s={params['delta_s']}")
    params["beta"] = float(alg.get("beta", 0.3))
    params["window"] = float(sim.get("window", 5.0))
    params["safe_step_variant"] = str(alg.get("safe_step_variant", "paper"))
    for key, value in (overrides or {}).items():
        if value is not None:
            params[key] = float(value) if key != "safe_step_variant" else value
            notes.append(f"{key} overridden: {value}")
    if params["safe_step_variant"] not in SAFE_STEP_VARIANTS:
        raise ValueError(f"algorithm.safe_step_variant must be one of {SAFE_STEP_VARIANTS}")
    return {"params": params, "seed": resolve_seed(config), "notes": notes,
            "controller_mode": "capacity_safe_event_driven", "execution_backend": "distributed"}

def resolved_config(config: dict, settings: dict) -> dict:
    """The config the distributed processes actually run (written next to the telemetry)."""
    p = settings["params"]
    return {
        "simulation": {"controller_mode": settings["controller_mode"],
                       "execution_backend": settings["execution_backend"],
                       "window": p["window"], "seed": settings["seed"]},
        "algorithm": {k: p[k] for k in ("eta", "gamma", "delta_s", "eps", "beta", "safe_step_variant")},
        "topology": config["topology"],
        "notes": settings["notes"],
    }
