"""Algorithm 1 of the paper (Sec. IV-D)."""
import numpy as np
from src.model.marginal_costs import mm1_marginal_cost_vectorized
from src.controller.best_response import best_response_available
from src.controller.diagnostics import compute_diagnostics
from src.controller.feasibility import transportation_feasibility
from src.simulation.state import SystemState

def compute_broker_loads(lambda_ij: np.ndarray) -> np.ndarray:
    """Lambda_j = sum_i lambda_ij."""
    return lambda_ij.sum(axis=0)

def update_prices(current_prices: np.ndarray, loads: np.ndarray, mu_brokers: np.ndarray, gamma: float, eps: float = 1e-12) -> np.ndarray:
    """p <- (1 - gamma) p + gamma C_j(Lambda_j)."""
    p_hat = mm1_marginal_cost_vectorized(loads, mu_brokers, eps)
    return (1.0 - gamma) * current_prices + gamma * p_hat

SAFE_STEP_VARIANTS = ("paper", "notebook")

def safe_step_bounds(lambda_ij: np.ndarray, lambda_br: np.ndarray, loads: np.ndarray, mu_brokers: np.ndarray,
                     eta: float, delta_s: float = 1e-8, variant: str = "paper") -> np.ndarray:
    """
    Step 3. For Delta_j = sum_i (lambda_br - lambda)_ij > 0,
        paper:    s_j = min(1, (mu_j - delta_s - Lambda_j) / (eta Delta_j))
        notebook: s_j = min(1, (mu_j - delta_s - Lambda_j) / Delta_j)
    else 1. The notebook's form leaves out eta, so it steps shorter when the bound binds.
    """
    if variant not in SAFE_STEP_VARIANTS:
        raise ValueError(f"variant must be one of {SAFE_STEP_VARIANTS}")
    load_direction = (lambda_br - lambda_ij).sum(axis=0)
    s_j = np.ones(len(mu_brokers))
    growing = load_direction > 0
    scale = eta if variant == "paper" else 1.0
    s_j[growing] = np.minimum(1.0, (mu_brokers[growing] - delta_s - loads[growing])
                              / (scale * load_direction[growing]))
    return s_j

def compute_safe_step(lambda_ij: np.ndarray, lambda_br: np.ndarray, loads: np.ndarray, mu_brokers: np.ndarray,
                      eta: float, delta_s: float = 1e-8, variant: str = "paper") -> float:
    """s_t = min_j s_j."""
    return max(0.0, float(np.min(safe_step_bounds(lambda_ij, lambda_br, loads, mu_brokers, eta, delta_s, variant))))

def source_best_response(mu_row: np.ndarray, prices: np.ndarray, lam_i: float) -> np.ndarray:
    """Step 2 for one source. Raises on failure; the caller decides whether to hold."""
    return best_response_available(mu_row, prices, lam_i)

def broker_price(price: float, load: float, mu_j: float, gamma: float, eps: float = 1e-12) -> float:
    """Step 1 for one broker; the same arithmetic as update_prices."""
    return float(update_prices(np.array([price]), np.array([load]), np.array([mu_j]), gamma, eps)[0])

def apply_common_step(state: SystemState, topology, lambda_br: np.ndarray, new_prices: np.ndarray, eta: float,
                      eps: float = 1e-12, delta_s: float = 1e-8, safe_step_variant: str = "paper",
                      br_failures: list = None):
    """
    Steps 3-4: s_j, s_t = min_j s_j, lambda <- (1 - eta s_t) lambda + eta s_t lambda_br.
    Updates state in place; returns (state, s_t, max(route_rel, price_rel)).
    """
    l_prev = state.lambda_ij.copy()
    p_prev = state.prices.copy()
    loads = compute_broker_loads(state.lambda_ij)

    s_j = safe_step_bounds(state.lambda_ij, lambda_br, loads, topology.mu_brokers, eta, delta_s, safe_step_variant)
    s_t = max(0.0, float(np.min(s_j)))
    step = eta * s_t
    new_lambda_ij = (1.0 - step) * state.lambda_ij + step * lambda_br
    route_rel = np.linalg.norm(new_lambda_ij - l_prev) / max(np.linalg.norm(l_prev), eps)
    price_rel = np.linalg.norm(new_prices - p_prev) / max(np.linalg.norm(p_prev), eps)

    state.lambda_ij = new_lambda_ij
    state.prices = new_prices
    state.iteration += 1
    state.s_j, state.s_t, state.route_rel, state.price_rel = s_j, s_t, route_rel, price_rel
    state.br_failures = list(br_failures or [])

    return state, s_t, max(route_rel, price_rel)

def iteration_step(state: SystemState, topology, eta: float, gamma: float, eps: float = 1e-12,
                   delta_s: float = 1e-8, on_best_response_failure: str = "raise",
                   safe_step_variant: str = "paper"):
    """
    One iteration: prices, best responses, common safe step.

    on_best_response_failure: "raise", or "hold" to keep that source's split
    (Delta_ij = 0) and list it in state.br_failures. A failure is solver
    non-convergence (RuntimeError) or a demand the source's links cannot
    carry (ValueError, e.g. after a capacity drop).
    """
    if on_best_response_failure not in ("raise", "hold"):
        raise ValueError(f"on_best_response_failure must be 'raise' or 'hold', not {on_best_response_failure!r}")

    loads = compute_broker_loads(state.lambda_ij)
    new_prices = update_prices(state.prices, loads, topology.mu_brokers, gamma, eps)

    lambda_br = np.zeros_like(state.lambda_ij)
    br_failures = []
    for i in range(topology.n_sources):
        try:
            lambda_br[i, :] = source_best_response(topology.mu_links[i, :], new_prices, topology.lambdas_total[i])
        except (RuntimeError, ValueError):
            if on_best_response_failure == "raise":
                raise
            lambda_br[i, :] = state.lambda_ij[i, :]
            br_failures.append(i)

    return apply_common_step(state, topology, lambda_br, new_prices, eta, eps, delta_s, safe_step_variant,
                             br_failures)

def algorithm1_initial_state(topology, delta_s: float = 1e-8, eps: float = 1e-12,
                             initial_lambda: np.ndarray = None) -> SystemState:
    """The LP routing with Lambda_j <= mu_j - delta_s (unless given) and p_j = C_j(Lambda_j)."""
    if initial_lambda is None:
        initial_lambda = transportation_feasibility(topology.lambdas_total, topology.mu_links,
                                                    topology.mu_brokers, margin=delta_s)
    loads = compute_broker_loads(initial_lambda)
    return SystemState(lambda_ij=np.array(initial_lambda, dtype=float),
                       prices=mm1_marginal_cost_vectorized(loads, topology.mu_brokers, eps))

def run_algorithm1(topology, eta: float = 0.25, gamma: float = 0.5, tol: float = 1e-10, max_iter: int = 4000,
                   eps: float = 1e-12, delta_s: float = 1e-8, initial_lambda: np.ndarray = None,
                   require_certificate: bool = False, safe_step_variant: str = "paper") -> SystemState:
    """
    Iterate until max(route_rel, price_rel) < tol (the notebook's stopping rule)
    or, with require_certificate, until the certificate also passes (Step 5).
    """
    state = algorithm1_initial_state(topology, delta_s, eps, initial_lambda)
    for _ in range(max_iter):
        state, _, residual = iteration_step(state, topology, eta, gamma, eps=eps, delta_s=delta_s,
                                            safe_step_variant=safe_step_variant)
        if residual < tol and (not require_certificate
                               or compute_diagnostics(state.lambda_ij, state.prices, topology, eps=eps)["certified"]):
            break
    return state
