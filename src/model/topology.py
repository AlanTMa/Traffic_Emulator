"""
Topology definitions for the traffic allocation problem.

Units (normalized work): every rate in the model is a work rate in the same
unit. The paper derives source rates from message traffic (msg/s x message
size) and expresses rates and capacities in MB/s, i.e. as work rates. The
event simulator treats one event as ONE NORMALIZED UNIT OF WORK: lambda_i is
work units/s, and mu_ij / mu_j are work units served per second, with
exponential service of one unit. For the 5x3 instance one unit = 1 MB (the
reference notebook's PACKET_MB = 1.0). An event is not a message or a packet:
message sizes and per-message service requirements are not modeled, so the
simulated sojourn times are work-unit sojourn times, not message latencies.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Topology:
    """
    Represents the network topology and traffic demands.

    Attributes:
        lambdas_total: (N,) offered work rate of each source (work units/s).
        mu_links: (N, M) access-link service rates (work units/s).
        mu_brokers: (M,) broker service rates (work units/s).
        sources: List of source identifiers.
        brokers: List of broker identifiers.
    """
    lambdas_total: np.ndarray
    mu_links: np.ndarray
    mu_brokers: np.ndarray
    sources: list
    brokers: list

    def __post_init__(self):
        # Validate dimensions
        n_sources = len(self.sources)
        n_brokers = len(self.brokers)
        if self.lambdas_total.shape != (n_sources,):
            raise ValueError(f"lambdas_total shape {self.lambdas_total.shape} must be ({n_sources},)")
        if self.mu_links.shape != (n_sources, n_brokers):
            raise ValueError(f"mu_links shape {self.mu_links.shape} must be ({n_sources}, {n_brokers})")
        if self.mu_brokers.shape != (n_brokers,):
            raise ValueError(f"mu_brokers shape {self.mu_brokers.shape} must be ({n_brokers},)")

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    @property
    def n_brokers(self) -> int:
        return len(self.brokers)
