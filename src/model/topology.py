"""
Topology definitions for the traffic allocation problem.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Topology:
    """
    Represents the network topology and traffic demands.

    Attributes:
        lambdas_total: (N,) array of offered traffic for each source.
        mu_links: (N, M) array of access capacities.
        mu_brokers: (M,) array of broker service capacities.
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
