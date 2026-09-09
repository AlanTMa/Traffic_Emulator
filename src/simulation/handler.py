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
    """
    def __init__(self, engine, topology, state: SimulationState, x_ij: np.ndarray, telemetry: TelemetryBuffer = None):
        self.engine = engine
        self.topology = topology
        self.state = state
        self.x_ij = x_ij # Current routing fractions
        self.request_id_counter = 0

        # Asynchronous Controller State
        self.prices = np.zeros(len(self.topology.brokers))
        self.known_prices = np.zeros((len(self.topology.sources), len(self.topology.brokers)))
        self.price_interval = 5.0 # Broadcast prices every 5 seconds

        # Telemetry
        self.telemetry = telemetry
        self.iteration = 0
        self.prev_objective = None

        # Schedule first controller tick
        self.engine.schedule(Event(
            timestamp=0.0,
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
        elif event.event_type == EventType.PRICE_BROADCAST:
            self._handle_price_broadcast(event)
        elif event.event_type == EventType.PRICE_RECEIVED:
            self._handle_price_received(event)
        elif event.event_type == EventType.ROUTING_UPDATE:
            self._handle_routing_update(event)

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
        self.state.requests[event.request_id]["broker_complete"] = self.engine.now

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
        # --- Telemetry Recording ---
        if self.telemetry:
            # Current broker loads: lambda_j = sum_i x_ij
            lambda_j = np.sum(self.x_ij, axis=0)
            mu_j = self.topology.mu_brokers

            # Objective = sum(x_ij / (mu_ij - x_ij)) + sum(lambda_j / (mu_j - lambda_j))
            diff_links = np.maximum(self.topology.mu_links - self.x_ij, 1e-6)
            diff_brokers = np.maximum(mu_j - lambda_j, 1e-6)

            obj = np.sum(self.x_ij / diff_links) + np.sum(lambda_j / diff_brokers)
            max_util = np.max(lambda_j / mu_j)

            rel_change = 0.0
            if self.prev_objective is not None and self.prev_objective != 0:
                rel_change = abs(obj - self.prev_objective) / self.prev_objective

            self.telemetry.record(
                timestamp=self.engine.now,
                iteration=self.iteration,
                state_data={
                    "objective": obj,
                    "max_util": max_util,
                    "rel_change": rel_change
                }
            )
            self.prev_objective = obj
            self.iteration += 1

            # LIVE UPDATE: Save to CSV every tick so the dashboard can read it
            self.telemetry.save_to_csv("simulation_metrics.csv")

        # Schedule price broadcasts for all brokers
        for j in range(len(self.topology.brokers)):
            self.engine.schedule(Event(
                timestamp=self.engine.now,
                event_type=EventType.PRICE_BROADCAST,
                broker_id=j
            ))

        # Schedule next tick
        self.engine.schedule(Event(
            timestamp=self.engine.now + self.price_interval,
            event_type=EventType.CONTROLLER_TICK
        ))

    def _handle_price_broadcast(self, event: Event):
        broker_id = event.broker_id
        mu_j = self.topology.mu_brokers[broker_id]

        # Current load lambda_j = sum_i x_ij
        lambda_j = np.sum(self.x_ij[:, broker_id])

        # price p_j = mu_j / (mu_j - lambda_j)^2
        # Ensure we don't divide by zero or negative (though x_ij should be safe)
        denom = (mu_j - lambda_j)**2
        price = mu_j / denom if denom > 1e-9 else 1e9

        self.prices[broker_id] = price

        # Broadcast to all sources
        for i in range(len(self.topology.sources)):
            self.engine.schedule(Event(
                timestamp=self.engine.now,
                event_type=EventType.PRICE_RECEIVED,
                source_id=i,
                broker_id=broker_id,
                payload=price
            ))

    def _handle_price_received(self, event: Event):
        source_id = event.source_id
        broker_id = event.broker_id
        price = event.payload

        # Update known prices for this source
        self.known_prices[source_id, broker_id] = price

        # Trigger routing update
        self.engine.schedule(Event(
            timestamp=self.engine.now,
            event_type=EventType.ROUTING_UPDATE,
            source_id=source_id
        ))

    def _handle_routing_update(self, event: Event):
        source_id = event.source_id
        lam_i = self.topology.lambdas_total[source_id]

        # Get mu for this source (row in mu_links)
        mu_row = self.topology.mu_links[source_id, :]

        # Use known prices for this source
        p = self.known_prices[source_id, :]

        # Solve best response
        try:
            new_x = best_response_mm1(mu_row, p, lam_i)
            self.x_ij[source_id, :] = new_x
        except RuntimeError as e:
            # If it fails to converge, keep current routing
            pass
