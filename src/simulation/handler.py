"""
Event handler for the discrete-event simulation.
"""
import numpy as np
from src.simulation.events import Event, EventType
from src.simulation.queues import SimulationState
from src.controller.best_response import best_response_mm1
from src.telemetry.metrics import TelemetryBuffer
from src.telemetry.schema import controller_snapshot

class SimulationHandler:
    """
    Processes simulation events and updates the system state.

    Each arrival is one normalized unit of work (see src/model/topology.py):
    Poisson arrivals at lambda_i units/s, exponential service at mu_ij and
    mu_j units/s. Recorded "latency" is the end-to-end sojourn time of a
    work unit (access queue + broker queue), not a message latency.

    Controller mode 'windowed_stochastic': the reference notebook's windowed
    stochastic scheme (WiOpt26JNSC_Extended.ipynb, "Windowed Stochastic
    simulation of the DISTRIBUTED pricing scheme"). At the end of every window:
      1. each broker updates an EWMA estimate of its *measured* arrival rate
         and damps its price toward mu_j / (mu_j - Lambda_hat_j)^2;
      2. after the warm-up windows, every source moves its split toward its
         best response to the new prices (synchronously, with inertia eta).

    This is NOT paper Algorithm 1: there is no common safe step, so planned
    broker loads carry no per-iteration capacity guarantee, and prices track
    noisy measurements rather than C_j(Lambda_j). All sources update at one
    global window barrier; it is not an asynchronous controller.
    """
    def __init__(self, engine, topology, state: SimulationState, x_ij: np.ndarray, telemetry: TelemetryBuffer = None,
                 eta: float = 0.35, gamma: float = 0.5, beta: float = 0.3,
                 window: float = 5.0, warmup_windows: int = 4, eps: float = 1e-9,
                 rng: np.random.Generator = None, latency_log: list = None):
        self.engine = engine
        self.topology = topology
        self.state = state
        self.x_ij = x_ij # Current routing flows (rows sum to lambda_i)
        self.request_id_counter = 0
        # All randomness (arrivals, routing draws, service times) comes from
        # this generator, so a seeded run is reproducible. None: unseeded.
        self.rng = rng if rng is not None else np.random.default_rng()

        # Controller parameters (notebook defaults: ETA_SPLIT, GAMMA_PRICE, BETA_LAMBDA)
        self.eta = eta          # inertia on split updates
        self.gamma = gamma      # smoothing on prices
        self.beta = beta        # EWMA smoothing on measured broker arrival rates
        self.window = window    # seconds per window (one controller iteration)
        self.warmup_windows = warmup_windows  # windows before splits start adapting
        self.eps = eps

        # Controller state; the notebook starts both estimates and prices at zero
        n_brokers = len(self.topology.brokers)
        self.window_arrivals = np.zeros(n_brokers)  # broker arrivals this window
        # End-to-end latencies (source, latency) of requests completed this window
        self.window_latencies = []
        # Optional caller-owned list receiving (completion_time, source, latency)
        # for every completed unit, e.g. for exact percentiles in experiments
        self.latency_log = latency_log
        self.lambda_hat = np.zeros(n_brokers)
        self.prices = np.zeros(n_brokers)
        self.window_idx = 0

        # Telemetry
        self.telemetry = telemetry
        self.prev_x_ij = None
        self.prev_prices = None

        # First controller tick closes the first window
        self.engine.schedule(Event(
            timestamp=self.window,
            event_type=EventType.CONTROLLER_TICK
        ))

    def handle_event(self, event: Event):
        if event.event_type == EventType.SOURCE_ARRIVAL:
            self._handle_source_arrival(event)
        elif event.event_type == EventType.ACCESS_SERVICE_START:
            self._handle_access_start(event)
        elif event.event_type == EventType.ACCESS_SERVICE_COMPLETE:
            self._handle_access_complete(event)
        elif event.event_type == EventType.BROKER_ARRIVAL:
            self._handle_broker_arrival(event)
        elif event.event_type == EventType.BROKER_SERVICE_START:
            self._handle_broker_start(event)
        elif event.event_type == EventType.BROKER_SERVICE_COMPLETE:
            self._handle_broker_complete(event)
        elif event.event_type == EventType.CONTROLLER_TICK:
            self._handle_controller_tick(event)

    def _handle_source_arrival(self, event: Event):
        # 1. Schedule next arrival for this source
        lam_i = self.topology.lambdas_total[event.source_id]
        inter_arrival = self.rng.exponential(1.0 / lam_i)
        self.engine.schedule(Event(
            timestamp=self.engine.now + inter_arrival,
            event_type=EventType.SOURCE_ARRIVAL,
            source_id=event.source_id
        ))

        # 2. Route this work unit
        flows = self.x_ij[event.source_id, :]
        lam_i = np.sum(flows)
        if lam_i > 0:
            probs = flows / lam_i
        else:
            # If no flow is allocated, pick randomly
            probs = np.ones(len(self.topology.brokers)) / len(self.topology.brokers)

        broker_id = int(self.rng.choice(len(self.topology.brokers), p=probs))

        # 3. Work-unit tracking
        req_id = self.request_id_counter
        self.request_id_counter += 1
        self.state.requests[req_id] = {"arrival": self.engine.now, "source": event.source_id}

        # 4. Access queueing
        queue = self.state.access_queues[event.source_id][broker_id]
        if not queue.is_busy:
            self.engine.schedule(Event(
                timestamp=self.engine.now,
                event_type=EventType.ACCESS_SERVICE_START,
                source_id=event.source_id,
                broker_id=broker_id,
                request_id=req_id
            ))
        else:
            queue.push(req_id)

    def _handle_access_start(self, event: Event):
        # Set server to busy
        queue = self.state.access_queues[event.source_id][event.broker_id]
        queue.is_busy = True

        # Service time ~ Exp(mu_ij)
        mu_ij = self.topology.mu_links[event.source_id, event.broker_id]
        service_time = self.rng.exponential(1.0 / mu_ij)

        self.engine.schedule(Event(
            timestamp=self.engine.now + service_time,
            event_type=EventType.ACCESS_SERVICE_COMPLETE,
            source_id=event.source_id,
            broker_id=event.broker_id,
            request_id=event.request_id
        ))

    def _handle_access_complete(self, event: Event):
        # Record completion
        self.state.requests[event.request_id]["access_complete"] = self.engine.now

        # Schedule next request in queue
        queue = self.state.access_queues[event.source_id][event.broker_id]
        if not queue.is_empty():
            next_req_id = queue.pop()
            # Server stays busy
            self.engine.schedule(Event(
                timestamp=self.engine.now,
                event_type=EventType.ACCESS_SERVICE_START,
                source_id=event.source_id,
                broker_id=event.broker_id,
                request_id=next_req_id
            ))
        else:
            queue.is_busy = False

        # Send to broker
        self.engine.schedule(Event(
            timestamp=self.engine.now,
            event_type=EventType.BROKER_ARRIVAL,
            broker_id=event.broker_id,
            request_id=event.request_id
        ))

    def _handle_broker_arrival(self, event: Event):
        self.window_arrivals[event.broker_id] += 1
        queue = self.state.broker_queues[event.broker_id]
        if not queue.is_busy:
            self.engine.schedule(Event(
                timestamp=self.engine.now,
                event_type=EventType.BROKER_SERVICE_START,
                broker_id=event.broker_id,
                request_id=event.request_id
            ))
        else:
            queue.push(event.request_id)

    def _handle_broker_start(self, event: Event):
        # Set server to busy
        queue = self.state.broker_queues[event.broker_id]
        queue.is_busy = True

        # Service time ~ Exp(mu_j)
        mu_j = self.topology.mu_brokers[event.broker_id]
        service_time = self.rng.exponential(1.0 / mu_j)

        self.engine.schedule(Event(
            timestamp=self.engine.now + service_time,
            event_type=EventType.BROKER_SERVICE_COMPLETE,
            broker_id=event.broker_id,
            request_id=event.request_id
        ))

    def _handle_broker_complete(self, event: Event):
        # Record completion
        req = self.state.requests[event.request_id]
        req["broker_complete"] = self.engine.now
        self.state.completed += 1
        latency = self.engine.now - req["arrival"]
        self.state.latency_sum += latency
        self.window_latencies.append((req["source"], latency))
        if self.latency_log is not None:
            self.latency_log.append((self.engine.now, req["source"], latency))
        if not self.state.keep_requests:
            del self.state.requests[event.request_id]

        # Schedule next request in queue
        queue = self.state.broker_queues[event.broker_id]
        if not queue.is_empty():
            next_req_id = queue.pop()
            # Server stays busy
            self.engine.schedule(Event(
                timestamp=self.engine.now,
                event_type=EventType.BROKER_SERVICE_START,
                broker_id=event.broker_id,
                request_id=next_req_id
            ))
        else:
            queue.is_busy = False

    def _handle_controller_tick(self, event: Event):
        mu_j = self.topology.mu_brokers

        # 1. Brokers: EWMA of measured arrival rate -> damped price
        inst_rate = self.window_arrivals / self.window
        self.window_arrivals[:] = 0
        self.lambda_hat = (1.0 - self.beta) * self.lambda_hat + self.beta * inst_rate
        p_inst = mu_j / np.maximum(mu_j - self.lambda_hat, 1e-6)**2
        self.prices = (1.0 - self.gamma) * self.prices + self.gamma * p_inst

        # 2. Sources: after warm-up, move each split toward its best response
        if self.window_idx >= self.warmup_windows:
            for i in range(len(self.topology.sources)):
                try:
                    x_br = best_response_mm1(self.topology.mu_links[i, :], self.prices,
                                             self.topology.lambdas_total[i])
                except RuntimeError:
                    continue  # keep the current split if the solver fails
                self.x_ij[i, :] = (1.0 - self.eta) * self.x_ij[i, :] + self.eta * x_br

        self.window_idx += 1
        if self.telemetry:
            self._record_telemetry()

        self.engine.schedule(Event(
            timestamp=self.engine.now + self.window,
            event_type=EventType.CONTROLLER_TICK
        ))

    def _record_telemetry(self):
        # Relative routing and price change since the last window (same
        # measures as synchronous.iteration_step)
        route_rel = price_rel = np.nan
        if self.prev_x_ij is not None:
            route_rel = np.linalg.norm(self.x_ij - self.prev_x_ij) / max(np.linalg.norm(self.prev_x_ij), self.eps)
            price_rel = np.linalg.norm(self.prices - self.prev_prices) / max(np.linalg.norm(self.prev_prices), self.eps)
        self.prev_x_ij = self.x_ij.copy()
        self.prev_prices = self.prices.copy()

        # Model quantities on the planned split; no safe step in this mode
        self.telemetry.record(controller_snapshot(
            self.topology, self.x_ij, self.prices,
            iteration=self.window_idx, sim_time=self.engine.now, controller_mode="windowed_stochastic",
            route_rel=route_rel, price_rel=price_rel,
            # Measured: EWMA of broker arrival rates, as plotted in the notebook
            lambda_hat_j=self.lambda_hat.copy(),
            util_measured_j=self.lambda_hat / self.topology.mu_brokers,
            **self._queue_and_latency_metrics(),
        ))
        self.window_latencies = []

    def _queue_and_latency_metrics(self) -> dict:
        """
        Queue lengths (number in system: waiting + in service) at the window
        boundary, and end-to-end work-unit sojourn-time ("latency")
        statistics over units that completed during the window (NaN when
        none did).
        """
        n_sources, n_brokers = self.x_ij.shape
        in_system = lambda q: len(q.queue) + int(q.is_busy)
        queue_access = [[in_system(self.state.access_queues[i][j]) for j in range(n_brokers)]
                        for i in range(n_sources)]
        queue_broker = [in_system(self.state.broker_queues[j]) for j in range(n_brokers)]

        sources = np.array([s for s, _ in self.window_latencies], dtype=int)
        latencies = np.array([l for _, l in self.window_latencies], dtype=float)
        if latencies.size:
            p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
            mean = latencies.mean()
        else:
            p50 = p95 = p99 = mean = np.nan
        mean_i = [latencies[sources == i].mean() if np.any(sources == i) else np.nan for i in range(n_sources)]
        return {
            "queue_access_ij": queue_access,
            "queue_broker_j": queue_broker,
            "completed_window": int(latencies.size),
            "completed_total": self.state.completed,
            "latency_mean": mean, "latency_p50": p50, "latency_p95": p95, "latency_p99": p99,
            "latency_mean_i": mean_i,
            "latency_mean_total": self.state.latency_sum / self.state.completed if self.state.completed else np.nan,
        }
