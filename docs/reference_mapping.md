# Paper, notebook, code

Paper: arXiv:2602.03246, extended version; equation and section numbers are
from that version. Notebook: `WiOpt26JNSC_Extended.ipynb` (cells counted
from 0).

## Model

| Paper | Notebook | Code |
|---|---|---|
| D_ij = 1/(μ_ij − λ_ij), D_j = 1/(μ_j − Λ_j) (eqs. 1, 2) | `per_source_mean_e2e` (cell 4) | `diagnostics.marginal_costs` |
| F = Σ λ_ij D_ij + Σ Λ_j D_j (eqs. 4, 9); +∞ outside the open domain | `objective_flow_weighted_at` | `central.system_objective` |
| C_ij = μ_ij/(μ_ij − λ_ij)² (eq. 12) | inline in `verification_residuals` | `diagnostics.marginal_costs` |
| C_j = μ_j/(μ_j − Λ_j)² (eq. 13) | `marginal_prices_mm1` | `marginal_costs.mm1_marginal_cost_vectorized` |
| KKT conditions (eq. 14) | `solve_central_reference`, KKT polish | `central.solve_central` |
| Source problem (eq. 15) | `best_response_mm1_optimizer` (validation only) | `test_notebook_batteries.optimizer_best_response` |
| Threshold best response (eq. 16) | `best_response_mm1` | `best_response.best_response_mm1`; `best_response_available` for sparse rows |
| 5x3 instance (Sec. V-A), seed 42 | cells 0–2 | `config/paper_5x3.yaml`, `tests/regression/data/baseline_5x3.json` |
| Units: msg/s × size → MB/s | `producer_emission_MBps`, `PACKET_MB = 1.0` | one event = one work unit, 1 MB for 5x3 |

## Algorithm 1 (Sec. IV-D)

| Paper | Notebook `distributed_flow_weighted` | Code |
|---|---|---|
| η, γ, margin δ_s | `eta`, `gamma`; one `eps = 1e-8` is both margin and division guard | `delta_s` (margin) and `eps` (guard, 1e-12) are separate |
| Feasible λ⁽⁰⁾ with Λ_j ≤ μ_j − δ_s | `transportation_feasibility`: LP maximizing the minimum headroom | `feasibility.transportation_feasibility`; also checks the returned headroom, since HiGHS's ~1e-7 tolerance can return zero headroom at a critical instance |
| p⁽⁰⁾ | `marginal_prices_mm1(L0)` | `synchronous.algorithm1_initial_state` |
| Step 1: p ← (1−γ)p + γC_j | `p_new = (1-gamma)*p + gamma*p_model` | `synchronous.update_prices` |
| Step 2: best responses | `best_response_mm1` per source | `iteration_step` |
| Step 3: s_j = min(1, (μ_j − δ_s − Λ_j)/(ηΔ_j)) | `(mu-eps-L)/Delta`, then `eta*min(1, ·)`: no η in the bound, so a shorter step when it binds | `safe_step_bounds`; `variant="paper"` (default) follows the paper, `"notebook"` the notebook. They agree on 5x3, where s_t = 1. The notebook's batteries at 0.85 load pass only with the notebook variant |
| Step 4: s_t = min_j s_j; λ ← (1−ηs_t)λ + ηs_t λ^BR | `candidate = l_mat + step*direction` | `apply_common_step` |
| Step 5: stop when changes, price consistency and fixed point are within tolerance | stops on `max(route_rel, price_rel) < tol` | `run_algorithm1(tol)` matches the notebook; `require_certificate=True` is Step 5 |
| η = 0.25, γ = 0.5, tol 1e-10, 4000 iterations | cell 4 | `tests/regression/test_baseline_5x3.py`: 70 iterations, F = 2.0157473649139757 |
| Centralized F* | `solve_central_reference` (SLSQP + KKT polish) | `central.solve_central`: 2.0157473649138 |

Distributed backend (not in the paper or notebook, which run the algorithm
in one process):

| Step | Process | Function |
|---|---|---|
| Λ_j | controller | `compute_broker_loads` |
| 1, price | broker j | `broker_price` |
| 2, best response | source i | `source_best_response` |
| 3–4, safe step and update | controller | `apply_common_step` |
| residuals | controller | `compute_diagnostics` |

## Proposition 1 (Sec. IV-H, Table I)

| Paper | Notebook `verification_residuals` | Code `compute_diagnostics` |
|---|---|---|
| Feasibility | `r_conservation`, `r_nonnegative`, `r_access_capacity`, `r_service_capacity` | same, plus signed `access_margin`, `service_margin` |
| Price consistency (17) | `r_price` | `r_price` |
| Fixed point (18) | ‖λ − BR(C_j(Λ))‖ with model prices | ‖λ − BR(p)‖ with published prices; equal when (17) holds |
| KKT stationarity on used routes | max \|M_ij − mean α_i\| | same, plus per-source spread |
| Unused routes ≥ α_i | against the mean α_i | against the cheapest used route; same at the optimum |
| KKT complementarity | max \|λ_ij(M_ij − min_j M_ij)\| | same |
| Tolerances | `residual_tolerances` | `DEFAULT_TOLERANCES`, same values |
| "Globally optimal" | — | `CERTIFIED_MESSAGE` only when every residual passes; no convergence claim |

## Experiments

| Notebook | Code |
|---|---|
| Windowed stochastic simulation (cell 14): Poisson counts per window, β = 0.3, γ = 0.5, η = 0.35, 4 warm-up windows, 5 s windows, 300 s | `controller_mode: windowed_stochastic`; per-event queues instead of per-window counts; runs until stopped unless `duration` is set |
| — | `controller_mode: capacity_safe_event_driven`: Algorithm 1 steps on planned rates over the event queues |
| Load sweep by scaling μ to service loads 0.2 / 0.55 / 0.85 (cells 5–6) | `scripts/sweep_5x3.py` scales demand instead, λ_i(r) = r·λ_i up to r_max ≈ 4.87; `scripts/compare_modes.py` |
| Verification batteries (cells 6–8) | `tests/regression/test_notebook_batteries.py`, `tests/unit/test_sparse.py`; load sweep and multistart marked `slow` |
| `vary_link_capacity`, `vary_broker_capacity`, the 11 combos (cell 1); frozen in every notebook experiment | `model/dynamics.py`, config `dynamics.capacity_variation`, applied per window |
| — | `model/symmetric.py`: closed-form symmetric optimum, docs/symmetric_case.md |
