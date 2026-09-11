"""
The notebook's time-varying capacities (cell 1):

    mu_ij(t) = base_ij * max(0.2, 1 + a_l sin(2 pi t / T_l + phi_ij) + N(0, s_l))
    mu_j(t)  = base_j  * max(0.3, 1 + a_b cos(2 pi t / T_b + phi_j)  + N(0, s_b))

Phases are uniform on [0, 2 pi) with phase_offsets on. The notebook freezes
capacities in its experiments; here they are optional (config
dynamics.capacity_variation: a `combo`, or link/broker {amp, period, noise})
and piecewise constant per window.
"""
from dataclasses import dataclass

import numpy as np

# notebook defaults
DEFAULT_LINK = {"amp": 0.15, "period": 60.0, "noise": 0.02}
DEFAULT_BROKER = {"amp": 0.25, "period": 120.0, "noise": 0.03}
LINK_FLOOR, BROKER_FLOOR = 0.2, 0.3

# the notebook's combos: (link amp, period, noise, broker amp, period, noise)
NOTEBOOK_COMBOS = {
    1: (0.10, 120, 0.02, 0.10, 180, 0.03),   # mild, slow
    2: (0.20, 80, 0.03, 0.25, 120, 0.04),    # moderate
    3: (0.30, 60, 0.05, 0.30, 100, 0.05),    # medium-fast
    4: (0.30, 40, 0.07, 0.30, 80, 0.07),     # faster + moderate noise
    5: (0.40, 30, 0.10, 0.40, 60, 0.10),     # high but still stable-ish
    6: (0.15, 90, 0.12, 0.20, 100, 0.08),    # asymmetric noisy
    7: (0.60, 30, 0.18, 0.60, 40, 0.18),     # strong sinusoid + high noise
    8: (0.80, 30, 0.22, 0.80, 40, 0.22),     # very strong + noisy, fast
    9: (0.70, 20, 0.25, 0.80, 30, 0.25),     # ultra-fast, heavy noise (stress)
    10: (0.65, 25, 0.20, 0.70, 35, 0.20),    # varied but still high dynamics
    11: (0.9, 15, 0.30, 0.9, 20, 0.30),      # extreme oscillation (stress test)
}

def vary_link_capacity(base, t, amp, period, noise, phase, rng):
    """Notebook vary_link_capacity (vectorized over links)."""
    fluct = 1 + amp * np.sin(2 * np.pi * t / period + phase)
    return base * np.maximum(LINK_FLOOR, fluct + rng.normal(0.0, noise, np.shape(base)))

def vary_broker_capacity(base, t, amp, period, noise, phase, rng):
    """Notebook vary_broker_capacity (vectorized over brokers)."""
    fluct = 1 + amp * np.cos(2 * np.pi * t / period + phase)
    return base * np.maximum(BROKER_FLOOR, fluct + rng.normal(0.0, noise, np.shape(base)))

@dataclass
class CapacityVariation:
    """Seeded time-varying capacities around a base topology."""
    base_topology: object
    link: dict
    broker: dict
    link_phase: np.ndarray
    broker_phase: np.ndarray
    rng: np.random.Generator

    @classmethod
    def from_config(cls, config: dict, topology, seed: int):
        """None if the config has no dynamics.capacity_variation section."""
        spec = (config.get("dynamics") or {}).get("capacity_variation")
        if not spec:
            return None
        if "combo" in spec:
            combo = int(spec["combo"])
            if combo not in NOTEBOOK_COMBOS:
                raise ValueError(f"dynamics.capacity_variation.combo must be one of {sorted(NOTEBOOK_COMBOS)}")
            la, lp, ln, ba, bp, bn = NOTEBOOK_COMBOS[combo]
            link, broker = {"amp": la, "period": lp, "noise": ln}, {"amp": ba, "period": bp, "noise": bn}
        else:
            link = {**DEFAULT_LINK, **(spec.get("link") or {})}
            broker = {**DEFAULT_BROKER, **(spec.get("broker") or {})}
        for name, params in (("link", link), ("broker", broker)):
            params = {k: float(v) for k, v in params.items()}
            if params["period"] <= 0 or params["amp"] < 0 or params["noise"] < 0:
                raise ValueError(f"dynamics.capacity_variation.{name}: need period > 0, amp >= 0, noise >= 0")
            (link if name == "link" else broker).update(params)
        # own stream, so capacity noise doesn't touch the traffic draws
        rng = np.random.default_rng([int(seed), 0xCA9])
        n, m = topology.mu_links.shape
        offsets = spec.get("phase_offsets", True)
        link_phase = rng.uniform(0, 2 * np.pi, (n, m)) if offsets else np.zeros((n, m))
        broker_phase = rng.uniform(0, 2 * np.pi, m) if offsets else np.zeros(m)
        return cls(topology, link, broker, link_phase, broker_phase, rng)

    def topology_at(self, t: float):
        """Topology with the capacities in effect at time t (one noise draw per call)."""
        base = self.base_topology
        mu_links = vary_link_capacity(base.mu_links, t, self.link["amp"], self.link["period"],
                                      self.link["noise"], self.link_phase, self.rng)
        mu_brokers = vary_broker_capacity(base.mu_brokers, t, self.broker["amp"], self.broker["period"],
                                          self.broker["noise"], self.broker_phase, self.rng)
        return base.with_capacities(mu_links, mu_brokers)
