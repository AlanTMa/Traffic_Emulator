"""
Event handler for the discrete-event simulation.
"""
import numpy as np
from src.simulation.events import Event, EventType
from src.simulation.queues import SimulationState
from src.controller.best_response import best_response_mm1
from src.telemetry.metrics import TelemetryBuffer

class SimulationHandler:
    """
    Processes simulation events and updates the system state.

    The routing controller follows the notebook's windowed stochastic scheme
    (WiOpt26JNSC_Extended.ipynb, "Windowed Stochastic simulation of the
    DISTRIBUTED pricing scheme"). At the end of every window:
      1. each broker updates an EWMA estimate of its *measured* arrival rate
         and damps its price toward mu_j / (mu_j - Lambda_hat_j)^2;
      2. after the warm-up windows, every source moves its split toward its
         best response to the new prices (synchronously, with inertia eta).
    """
    def __init__(self, engine, topology, state: SimulationState, x_ij: np.ndarray, telemetry: TelemetryBuffer = None,
                 eta: float = 0.35, gamma: float = 0.5, beta: float = 0.3,
                 window: float = 5.0, warmup_windows: int = 4, eps: float = 1e-9):
        self.engine = engine
        self.topology = topology
        self.state = state
        self.x_ij = x_ij # Current routing flows (rows sum to lambda_i)
        self.request_id_counter = 0

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
        inter_arrival = np.random.exponential(1.0 / lam_i)
        self.engine.schedule(Event(
            timestamp=self.engine.now + inter_arrival,
            event_type=EventType.SOURCE_ARRIVAL,
            source_id=event.source_id
        ))

        # 2. Route the current packet
        flows = self.x_ij[event.source_id, :]
        lam_i = np.sum(flows)
        if lam_i > 0:
            probs = flows / lam_i
        else:
            # If no flow is allocated, pick randomly
            probs = np.ones(len(self.topology.brokers)) / len(self.topology.brokers)

        broker_id = np.random.choice(len(self.topology.brokers), p=probs)

        # 3. Request tracking
        req_id = self.request_id_counter
        self.request_id_counter += 1
        self.state.requests[req_id] = {"arrival": self.engine.now}

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
        service_time = np.random.exponential(1.0 / mu_ij)

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
        service_time = np.random.exponential(1.0 / mu_j)

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
        self.state.latency_sum += self.engine.now - req["arrival"]
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
        mu_j = self.topology.mu_brokers
        planned_j = self.x_ij.sum(axis=0)

        # Analytic M/M/1 objective of the current routing:
        # sum(x_ij / (mu_ij - x_ij)) + sum(lambda_j / (mu_j - lambda_j))
        diff_links = np.maximum(self.topology.mu_links - self.x_ij, 1e-6)
        diff_brokers = np.maximum(mu_j - planned_j, 1e-6)
        obj = np.sum(self.x_ij / diff_links) + np.sum(planned_j / diff_brokers)

        # Controller residual since the last window: max of relative routing
        # and price change (same measure as synchronous.iteration_step)
        rel_change = np.nan
        if self.prev_x_ij is not None:
            route_rel = np.linalg.norm(self.x_ij - self.prev_x_ij) / max(np.linalg.norm(self.prev_x_ij), self.eps)
            price_rel = np.linalg.norm(self.prices - self.prev_prices) / max(np.linalg.norm(self.prev_prices), self.eps)
            rel_change = max(route_rel, price_rel)
        self.prev_x_ij = self.x_ij.copy()
        self.prev_prices = self.prices.copy()

        self.telemetry.record(
            timestamp=self.engine.now,
            iteration=self.window_idx,
            state_data={
                "objective": obj,
                # Measured (EWMA) utilization, as plotted in the notebook
                "max_util": np.max(self.lambda_hat / mu_j),
                "max_util_planned": np.max(planned_j / mu_j),
                "rel_change": rel_change,
            }
        )

        # LIVE UPDATE: Save to CSV every tick so the dashboard can read it
        self.telemetry.save_to_csv("simulation_metrics.csv")
