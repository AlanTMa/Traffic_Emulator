"""
Synchronous implementation of the WiOpt distributed routing algorithm.
"""
import numpy as np
from src.model.marginal_costs import mm1_marginal_cost_vectorized
from src.controller.best_response import best_response_available
from src.controller.diagnostics import compute_diagnostics
from src.controller.feasibility import transportation_feasibility
from src.simulation.state import SystemState

def compute_broker_loads(lambda_ij: np.ndarray) -> np.ndarray:
    """Compute aggregate load at each broker: Lambda_j = sum_i lambda_ij"""
    return lambda_ij.sum(axis=0)

def update_prices(current_prices: np.ndarray, loads: np.ndarray, mu_brokers: np.ndarray, gamma: float, eps: float = 1e-12) -> np.ndarray:
    """
    Update broker prices using price damping.
    p_new = (1-gamma)*p + gamma * (mu / (mu - Lambda)^2)
    """
    # Current marginal prices (p_hat)
    p_hat = mm1_marginal_cost_vectorized(loads, mu_brokers, eps)
    return (1.0 - gamma) * current_prices + gamma * p_hat

def safe_step_bounds(lambda_ij: np.ndarray, lambda_br: np.ndarray, loads: np.ndarray, mu_brokers: np.ndarray,
                     eta: float, delta_s: float = 1e-8) -> np.ndarray:
    """
    Per-broker step bounds s_j of Algorithm 1 (Step 3), with capacity margin delta_s.

    With Delta_j = sum_i (lambda_br_ij - lambda_ij):
        s_j = min(1, (mu_j - delta_s - Lambda_j) / (eta * Delta_j))   if Delta_j > 0
        s_j = 1                                                        otherwise
    """
    load_direction = (lambda_br - lambda_ij).sum(axis=0)
    s_j = np.ones(len(mu_brokers))
    growing = load_direction > 0
    s_j[growing] = np.minimum(1.0, (mu_brokers[growing] - delta_s - loads[growing])
                              / (eta * load_direction[growing]))
    return s_j

def compute_safe_step(lambda_ij: np.ndarray, lambda_br: np.ndarray, loads: np.ndarray, mu_brokers: np.ndarray,
                      eta: float, delta_s: float = 1e-8) -> float:
    """
    Common safe step s_t = min_j s_j of Algorithm 1 (see safe_step_bounds).
    The routing update moves by eta * s_t, so Lambda_j + eta * s_t * Delta_j
    <= mu_j - delta_s for every j.
    """
    return max(0.0, float(np.min(safe_step_bounds(lambda_ij, lambda_br, loads, mu_brokers, eta, delta_s))))

def iteration_step(state: SystemState, topology, eta: float, gamma: float, eps: float = 1e-12,
                   delta_s: float = 1e-8, on_best_response_failure: str = "raise"):
    """
    Perform one iteration of Algorithm 1. Shared by static_algorithm1 and
    capacity_safe_event_driven, so both modes run the same controller step.

    delta_s is the capacity margin kept below every broker's capacity by the
    safe step (paper: delta_s; the reference notebook uses 1e-8). eps only
    guards divisions against zero and is not a modeling parameter.

    on_best_response_failure: "raise" (default) propagates a best-response
    failure; "hold" keeps that source's current split for this step (its
    Delta_ij = 0, so the safe step still bounds the others) and lists the
    source index in state.br_failures, for long event runs that must record,
    not hide, such failures. Failures are solver non-convergence
    (RuntimeError) or a source whose demand its access links cannot carry
    under the current capacities (ValueError, e.g. after a capacity drop).
    """
    if on_best_response_failure not in ("raise", "hold"):
        raise ValueError(f"on_best_response_failure must be 'raise' or 'hold', not {on_best_response_failure!r}")
    l_prev = state.lambda_ij.copy()
    p_prev = state.prices.copy()

    # 1. Calculate current loads
    loads = compute_broker_loads(state.lambda_ij)

    # 2. Update prices
    new_prices = update_prices(state.prices, loads, topology.mu_brokers, gamma, eps)

    # 3. Each source solves best response
    lambda_br = np.zeros_like(state.lambda_ij)
    br_failures = []
    for i in range(topology.n_sources):
        try:
            lambda_br[i, :] = best_response_available(
                topology.mu_links[i, :],
                new_prices,
                topology.lambdas_total[i]
            )
        except (RuntimeError, ValueError):
            if on_best_response_failure == "raise":
                raise
            lambda_br[i, :] = state.lambda_ij[i, :]
            br_failures.append(i)

    # 4. Calculate safe step
    s_j = safe_step_bounds(state.lambda_ij, lambda_br, loads, topology.mu_brokers, eta, delta_s)
    s_t = max(0.0, float(np.min(s_j)))

    # 5. Update routing
    step = eta * s_t
    new_lambda_ij = (1.0 - step) * state.lambda_ij + step * lambda_br

    # Calculate residuals for convergence
    route_rel = np.linalg.norm(new_lambda_ij - l_prev) / max(np.linalg.norm(l_prev), eps)
    price_rel = np.linalg.norm(new_prices - p_prev) / max(np.linalg.norm(p_prev), eps)

    # Update state
    state.lambda_ij = new_lambda_ij
    state.prices = new_prices
    state.iteration += 1
    state.s_j, state.s_t, state.route_rel, state.price_rel = s_j, s_t, route_rel, price_rel
    state.br_failures = br_failures

    return state, s_t, max(route_rel, price_rel)

def algorithm1_initial_state(topology, delta_s: float = 1e-8, eps: float = 1e-12,
                             initial_lambda: np.ndarray = None) -> SystemState:
    """
    Algorithm 1 initialization: an LP-certified routing with
    Lambda_j <= mu_j - delta_s (maximum-headroom transportation LP), unless
    given, and model prices p_j = C_j(Lambda_j) at that routing.
    """
    if initial_lambda is None:
        initial_lambda = transportation_feasibility(topology.lambdas_total, topology.mu_links,
                                                    topology.mu_brokers, margin=delta_s)
    loads = compute_broker_loads(initial_lambda)
    return SystemState(lambda_ij=np.array(initial_lambda, dtype=float),
                       prices=mm1_marginal_cost_vectorized(loads, topology.mu_brokers, eps))

def run_algorithm1(topology, eta: float = 0.25, gamma: float = 0.5, tol: float = 1e-10, max_iter: int = 4000,
                   eps: float = 1e-12, delta_s: float = 1e-8, initial_lambda: np.ndarray = None,
                   require_certificate: bool = False) -> SystemState:
    """
    Run Algorithm 1 from an LP-certified initial routing (maximum-headroom
    transportation LP with margin delta_s) and model prices at that routing.

    Stopping rule:
      require_certificate=False (default): stop once max(route_rel, price_rel)
        < tol, as the reference notebook's distributed_flow_weighted() does.
      require_certificate=True: paper Algorithm 1 Step 5 - additionally require
        the Proposition 1 certificate (price consistency, fixed point, KKT;
        src/controller/diagnostics.py) to pass.
    Either way at most max_iter iterations. Matches the notebook loop except
    for the safe step, which follows the paper (see compute_safe_step).
    """
    state = algorithm1_initial_state(topology, delta_s, eps, initial_lambda)
    for _ in range(max_iter):
        state, _, residual = iteration_step(state, topology, eta, gamma, eps=eps, delta_s=delta_s)
        if residual < tol and (not require_certificate
                               or compute_diagnostics(state.lambda_ij, state.prices, topology, eps=eps)["certified"]):
            break
    return state
