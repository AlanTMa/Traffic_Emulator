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
        route_mask: (N, M) bool, True where the access link exists (default:
            all True). Unavailable links are stored with mu_ij = 0 and carry
            no flow; every model function treats mu_ij = 0 as "no link".
    """
    lambdas_total: np.ndarray
    mu_links: np.ndarray
    mu_brokers: np.ndarray
    sources: list
    brokers: list
    route_mask: np.ndarray = None

    def __post_init__(self):
        """Validate dimensions, ids and values; raises ValueError on any problem."""
        self.sources, self.brokers = list(self.sources), list(self.brokers)
        n_sources, n_brokers = len(self.sources), len(self.brokers)
        if n_sources == 0 or n_brokers == 0:
            raise ValueError("a topology needs at least one source and one broker")
        for kind, ids in (("source", self.sources), ("broker", self.brokers)):
            if len(set(ids)) != len(ids):
                dupes = sorted({str(x) for x in ids if ids.count(x) > 1})
                raise ValueError(f"duplicate {kind} ids: {dupes}")

        try:
            self.lambdas_total = np.asarray(self.lambdas_total, dtype=float)
            self.mu_links = np.asarray(self.mu_links, dtype=float)
            self.mu_brokers = np.asarray(self.mu_brokers, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(f"topology rates and capacities must be numeric: {e}") from None

        # Dimensions
        if self.route_mask is None:
            self.route_mask = np.ones((n_sources, n_brokers), dtype=bool)
        self.route_mask = np.asarray(self.route_mask, dtype=bool)
        if self.route_mask.shape != (n_sources, n_brokers):
            raise ValueError(f"route_mask shape {self.route_mask.shape} must be ({n_sources}, {n_brokers})")
        if self.lambdas_total.shape != (n_sources,):
            raise ValueError(f"lambdas_total shape {self.lambdas_total.shape} must be ({n_sources},)")
        if self.mu_links.shape != (n_sources, n_brokers):
            raise ValueError(f"mu_links shape {self.mu_links.shape} must be ({n_sources}, {n_brokers})")
        if self.mu_brokers.shape != (n_brokers,):
            raise ValueError(f"mu_brokers shape {self.mu_brokers.shape} must be ({n_brokers},)")
        # Unavailable links have no capacity (the mask is the explicit statement)
        self.mu_links = np.where(self.route_mask, self.mu_links, 0.0)

        # Values: finite; rates >= 0; capacities > 0 (the M/M/1 model needs
        # every access link and broker to serve at a positive rate)
        for name, values in (("lambdas_total", self.lambdas_total), ("mu_links", self.mu_links),
                             ("mu_brokers", self.mu_brokers)):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains NaN or infinite values")
        if np.any(self.lambdas_total < 0):
            bad = [self.sources[i] for i in np.where(self.lambdas_total < 0)[0]]
            raise ValueError(f"source rates must be >= 0 (negative for {bad})")
        if np.any(self.route_mask & (self.mu_links <= 0)):
            bad = [f"{self.sources[i]}->{self.brokers[j]}"
                   for i, j in zip(*np.where(self.route_mask & (self.mu_links <= 0)))]
            raise ValueError(f"access capacities must be > 0 (not for {bad})")
        stranded = [self.sources[i] for i in range(n_sources)
                    if self.lambdas_total[i] > 0 and not self.route_mask[i].any()]
        if stranded:
            raise ValueError(f"sources with traffic but no available access link: {stranded}")
        if np.any(self.mu_brokers <= 0):
            bad = [self.brokers[j] for j in np.where(self.mu_brokers <= 0)[0]]
            raise ValueError(f"broker capacities must be > 0 (not for {bad})")

    def with_capacities(self, mu_links: np.ndarray, mu_brokers: np.ndarray) -> "Topology":
        """Same sources, rates, ids and route mask with new capacities (validated)."""
        return Topology(self.lambdas_total, mu_links, mu_brokers, self.sources, self.brokers,
                        route_mask=self.route_mask)

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    @property
    def n_brokers(self) -> int:
        return len(self.brokers)
