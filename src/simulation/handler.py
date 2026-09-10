"""
Event handler for the discrete-event simulation.
"""
import numpy as np
from src.simulation.events import Event, EventType
from src.simulation.queues import SimulationState
from src.controller.best_response import best_response_available
from src.controller.synchronous import algorithm1_initial_state, iteration_step
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

    Controller mode 'capacity_safe_event_driven': the same event-driven
    queues, but at every window boundary the planned routing is advanced by
    exactly one step of paper Algorithm 1 (synchronous.iteration_step, the
    function static_algorithm1 uses) on the PLANNED rates: prices damp toward
    C_j(planned Lambda_j), sources best-respond, and the common safe step
    keeps planned Lambda_j <= mu_j - delta_s. Measured arrivals, EWMA rates,
    queues and sojourn times are recorded but never enter the controller
    (feeding them in would turn it back into the windowed scheme). There is
    no warm-up (Algorithm 1 starts from its LP initializer and model prices)
    and beta only smooths the measured-rate telemetry. It is synchronous too.
    """
    CONTROLLER_MODES = ("windowed_stochastic", "capacity_safe_event_driven")

    def __init__(self, engine, topology, state: SimulationState, x_ij: np.ndarray, telemetry: TelemetryBuffer = None,
                 eta: float = 0.35, gamma: float = 0.5, beta: float = 0.3,
                 window: float = 5.0, warmup_windows: int = 4, eps: float = 1e-9,
                 rng: np.random.Generator = None, latency_log: list = None,
                 controller_mode: str = "windowed_stochastic", delta_s: float = 1e-8,
                 capacity_model=None):
        self.engine = engine
        # Current topology. With a capacity_model (src/model/dynamics.py) it is
        # replaced at every window boundary by the capacities in effect then;
        # `topology` is the one in effect at t = 0.
        self.topology = topology
        self.capacity_model = capacity_model
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
        self.window_access_arrivals = np.zeros((len(self.topology.sources), n_brokers))  # per access link
        # End-to-end latencies (source, latency) of requests completed this window
        self.window_latencies = []
        # Optional caller-owned list receiving (completion_time, source, latency)
        # for every completed unit, e.g. for exact percentiles in experiments
        self.latency_log = latency_log
        self.lambda_hat = np.zeros(n_brokers)
        self.prices = np.zeros(n_brokers)
        self.window_idx = 0

        if controller_mode not in self.CONTROLLER_MODES:
            raise ValueError(f"controller_mode must be one of {self.CONTROLLER_MODES}, not {controller_mode!r}")
        self.controller_mode = controller_mode
        self.delta_s = delta_s
        self.s_j, self.s_t = None, np.nan            # Algorithm 1 step of the last window
        self.br_failures = []                        # sources whose best response failed last window
        self.br_failures_total = 0
        if controller_mode == "capacity_safe_event_driven":
            # Algorithm 1 state on planned rates; x_ij must satisfy
            # Lambda_j <= mu_j - delta_s (e.g. LP initializer with margin delta_s)
            planned = self.x_ij.sum(axis=0)
            if np.any(planned > self.topology.mu_brokers - delta_s):
                raise ValueError("initial routing violates Lambda_j <= mu_j - delta_s")
            self.controller_state = algorithm1_initial_state(topology, delta_s, eps, initial_lambda=self.x_ij)
            self.x_ij = self.controller_state.lambda_ij
            self.prices = self.controller_state.prices.copy()

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
        self.window_access_arrivals[event.source_id, broker_id] += 1

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
        # Capacities for the window starting now (service, controller, diagnostics)
        if self.capacity_model is not None:
            self.topology = self.capacity_model.topology_at(self.engine.now)

        # Measured EWMA of broker arrival rates over the window just closed
        inst_rate = self.window_arrivals / self.window
        # Raw (unsmoothed) measured rates of the window, recorded in telemetry
        self.last_broker_rate = inst_rate.copy()
        self.last_access_rate = self.window_access_arrivals / self.window
        self.window_arrivals[:] = 0
        self.window_access_arrivals[:] = 0
        self.lambda_hat = (1.0 - self.beta) * self.lambda_hat + self.beta * inst_rate

        if self.controller_mode == "capacity_safe_event_driven":
            self._algorithm1_update()
        else:
            self._windowed_update()
        self.br_failures_total += len(self.br_failures)

        self.window_idx += 1
        if self.telemetry:
            self._record_telemetry()

        self.engine.schedule(Event(
            timestamp=self.engine.now + self.window,
            event_type=EventType.CONTROLLER_TICK
        ))

    def _windowed_update(self):
        """Notebook windowed scheme: prices from the measured EWMA, inertial splits."""
        mu_j = self.topology.mu_brokers
        p_inst = mu_j / np.maximum(mu_j - self.lambda_hat, 1e-6)**2
        self.prices = (1.0 - self.gamma) * self.prices + self.gamma * p_inst

        # After warm-up, move each split toward its best response
        self.br_failures = []
        if self.window_idx >= self.warmup_windows:
            for i in range(len(self.topology.sources)):
                try:
                    x_br = best_response_available(self.topology.mu_links[i, :], self.prices,
                                             self.topology.lambdas_total[i])
                except (RuntimeError, ValueError):
                    # Solver failure, or demand the current access capacities cannot
                    # carry: keep the current split; recorded in telemetry
                    self.br_failures.append(i)
                    continue
                self.x_ij[i, :] = (1.0 - self.eta) * self.x_ij[i, :] + self.eta * x_br

    def _algorithm1_update(self):
        """One exact Algorithm 1 step on the planned rates (shared with static_algorithm1)."""
        state, _, _ = iteration_step(self.controller_state, self.topology, self.eta, self.gamma,
                                     eps=self.eps, delta_s=self.delta_s, on_best_response_failure="hold")
        # New arrivals are routed with probabilities lambda_ij / lambda_i of the updated plan
        self.x_ij = state.lambda_ij
        self.prices = state.prices.copy()
        self.s_j, self.s_t, self.br_failures = state.s_j.copy(), state.s_t, list(state.br_failures)

    def _record_telemetry(self):
        # Relative routing and price change since the last window (same
        # measures as synchronous.iteration_step)
        route_rel = price_rel = np.nan
        if self.prev_x_ij is not None:
            route_rel = np.linalg.norm(self.x_ij - self.prev_x_ij) / max(np.linalg.norm(self.prev_x_ij), self.eps)
            price_rel = np.linalg.norm(self.prices - self.prev_prices) / max(np.linalg.norm(self.prev_prices), self.eps)
        self.prev_x_ij = self.x_ij.copy()
        self.prev_prices = self.prices.copy()

        # Model quantities on the planned split; s_j/s_t only in the Algorithm 1 mode
        self.telemetry.record(controller_snapshot(
            self.topology, self.x_ij, self.prices,
            iteration=self.window_idx, sim_time=self.engine.now, controller_mode=self.controller_mode,
            s_j=self.s_j, s_t=self.s_t, route_rel=route_rel, price_rel=price_rel,
            br_failures=list(self.br_failures), br_failures_total=self.br_failures_total,
            **({"mu_links_t": self.topology.mu_links, "mu_brokers_t": self.topology.mu_brokers}
               if self.capacity_model is not None else {}),
            # Measured: EWMA of broker arrival rates, as plotted in the notebook
            lambda_hat_j=self.lambda_hat.copy(),
            util_measured_j=self.lambda_hat / self.topology.mu_brokers,
            # Raw per-window measured arrival rates and utilizations
            broker_rate_measured_j=self.last_broker_rate,
            access_rate_measured_ij=self.last_access_rate,
            access_util_measured_ij=np.divide(self.last_access_rate, self.topology.mu_links,
                                              out=np.full(self.x_ij.shape, np.nan),
                                              where=self.topology.mu_links > 0),   # NaN: no link
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
