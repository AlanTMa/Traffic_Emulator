# Paper → reference notebook → Traffic_Emulator

- **Paper:** arXiv:2602.03246, extended version. Equation, section and
  table numbers are as in that version.
- **Notebook:** `WiOpt26JNSC_Extended.ipynb` from
  [ANRGUSC/JointNetServerCongestion-WiOpt26](https://github.com/ANRGUSC/JointNetServerCongestion-WiOpt26),
  also stored in this repo. "Cell n" means the n-th cell, counting from 0.

## Model

| Paper | Notebook | Traffic_Emulator | Notes |
|---|---|---|---|
| Access delay D_ij = 1/(μ_ij − λ_ij) (eq. 1) | `per_source_mean_e2e` (cell 4) | `model/delays.py`; `controller/diagnostics.py::marginal_costs` (`D_ij`) | |
| Broker delay D_j = 1/(μ_j − Λ_j) (eq. 2) | `per_source_mean_e2e` (cell 4) | same (`D_j`) | |
| Objective F = Σ λ_ij D_ij + Σ Λ_j D_j (eqs. 4, 9) | `objective_flow_weighted_at` (cell 4) | `controller/central.py::system_objective` | Flow-weighted sum, not divided by Σλ; +∞ outside the open domain |
| Access marginal cost C_ij = μ_ij/(μ_ij − λ_ij)² (eq. 12) | inline in `verification_residuals` | `diagnostics.py::marginal_costs` (`C_ij`) | |
| Broker marginal cost / price C_j = μ_j/(μ_j − Λ_j)² (eq. 13) | `marginal_prices_mm1` (cell 4) | `model/marginal_costs.py::mm1_marginal_cost_vectorized`; `diagnostics.py` (`C_j`) | |
| KKT conditions (eq. 14) | `solve_central_reference` KKT polish | `central.py::solve_central` (`kkt_equations`) | |
| Source problem (eq. 15) | `best_response_mm1_optimizer` (validation only) | `tests/regression/test_notebook_batteries.py::optimizer_best_response` | Validation only, as in the notebook |
| Threshold best response (eq. 16) | `best_response_mm1` (cell 4) | `controller/best_response.py::best_response_mm1`; `best_response_available` for sparse rows | Same algorithm and the same input validation as the notebook |
| 5x3 instance (Sec. V-A) | cells 0–2 (seed 42, topics → MB/s) | `config/paper_5x3.yaml`, `config/convergence_5x3.yaml`, `tests/regression/data/baseline_5x3.json` | Exact unrounded values |
| Units: msg/s × size → MB/s | `producer_emission_MBps` (cell 2); `PACKET_MB = 1.0` (cell 14) | `model/topology.py` module docstring | Event = one normalized work unit (1 MB for 5x3) |

## Algorithm 1 (Sec. IV-D)

| Paper | Notebook `distributed_flow_weighted` (cell 4) | Traffic_Emulator | Notes |
|---|---|---|---|
| Inputs η_t, γ_t, margin δ_s | `eta`, `gamma`, `eps` (one `eps = 1e-8` serves as margin and numerical guard) | `iteration_step(..., eta, gamma, eps, delta_s)` | `delta_s` (margin, 1e-8) is separate from `eps` (division guard, 1e-12) |
| Initialization: feasible λ⁽⁰⁾ with Λ_j ≤ μ_j − δ_s | `transportation_feasibility(margin=eps, route_mask)`: LP maximizing minimum headroom | `controller/feasibility.py::transportation_feasibility(margin=delta_s, route_mask)`; `synchronous.py::algorithm1_initial_state` | `route_mask` ported (default: all links that exist). Sparse topologies are also supported beyond the LP, which the notebook does not do. The code also checks the returned routing's actual headroom: HiGHS's ~1e-7 tolerance otherwise lets exactly critical instances through with zero headroom |
| Initial prices p⁽⁰⁾ | `marginal_prices_mm1(L0)` | `run_algorithm1`: `mm1_marginal_cost_vectorized(Λ⁽⁰⁾)` | |
| Step 1: p ← (1−γ)p + γ C_j(Λ_j) | `p_new = (1-gamma)*p + gamma*p_model` | `synchronous.py::update_prices` | |
| Step 2: best responses under p⁽ᵗ⁺¹⁾ | `best_response_mm1` per source | `iteration_step` | |
| Step 3: s_j = min(1, (μ_j − δ_s − Λ_j)/(η Δ_j)) for Δ_j > 0 | `safe_fraction = (mu-eps-L)/Delta`, then `step = eta*min(1, ·)` | `synchronous.py::safe_step_bounds` | **Differs from the notebook.** The notebook omits η in the bound, so it is more conservative whenever the bound binds. The code follows the paper by default; `safe_step_variant="notebook"` reproduces the notebook (used by the ported batteries, where the paper bound fails the 0.85 multistart). The two agree on 5x3, where s_t = 1 |
| Step 4: s_t = min_j s_j; λ ← (1 − η s_t)λ + η s_t λ^BR | `candidate = l_mat + step*direction` | `iteration_step`; `SystemState.s_j`, `s_t` | |
| Step 5: stop when route/price changes, price consistency and fixed point meet their tolerances | stops on `max(route_rel, price_rel) < tol` only | `run_algorithm1(tol)`, which matches the notebook; `run_algorithm1(require_certificate=True)` implements Step 5 | |
| Route/price relative change | `route_rel`, `price_rel` | `SystemState.route_rel`, `price_rel`; telemetry | |
| 5x3 parameters: η = 0.25, γ = 0.5, tol = 1e-10, max 4000 iterations | cell 4, bottom | `tests/regression/test_baseline_5x3.py` | 70 iterations; F_dist = 2.0157473649139757 |
| Centralized reference F* | `solve_central_reference` (SLSQP + KKT polish) | `controller/central.py::solve_central` | F* = 2.0157473649138 |

### Distributed execution backend

| Algorithm 1 step | Runs in | Function (shared with the reference) |
|---|---|---|
| Loads Λ_j = Σ_i λ_ij | controller | `synchronous.compute_broker_loads` |
| Step 1: damped price of broker j | broker j's process | `synchronous.broker_price` |
| Step 2: best response of source i | source i's process | `synchronous.source_best_response` |
| Steps 3–4: s_j, s_t, routing update | controller | `synchronous.apply_common_step` (also called by `iteration_step`) |
| Proposition 1 residuals | controller | `diagnostics.compute_diagnostics` via `telemetry.schema.controller_snapshot` |

Not in the paper or notebook, whose algorithm is described for distributed
agents but implemented in one process. See [distributed.md](distributed.md).
The access stage μ_ij is an explicit queue in each source process, and the
broker stage μ_j an explicit queue in each broker process.

## Proposition 1 and certificates (Sec. IV-H, Table I)

| Paper | Notebook `verification_residuals` (cell 4) | Traffic_Emulator `diagnostics.py::compute_diagnostics` | Notes |
|---|---|---|---|
| Feasibility | `r_conservation`, `r_nonnegative`, `r_access_capacity`, `r_service_capacity` | same names, plus signed `access_margin` and `service_margin` | |
| Price consistency (17) | `r_price` = max \|p − C_j(Λ)\| | `r_price` | |
| Local best response (18): fixed point | `r_fixed_point` = ‖λ − BR(C_j(Λ))‖, using model prices | `r_fixed_point` = ‖λ − BR(p)‖, using published prices | The two coincide when (17) holds |
| KKT stationarity on used routes | `r_active_stationarity` = max \|M_ij − mean α_i\| | same, plus per-source spread (max − min) `active_spread_i` | |
| Unused routes ≥ α_i | `r_inactive_complementarity` against the mean α_i | `r_inactive_complementarity` against the cheapest used route | Same at the optimum |
| KKT complementarity | `r_kkt_complementarity` = max \|λ_ij(M_ij − min_j M_ij)\| | same | Absolute, so it scales with the flows |
| Tolerances | `residual_tolerances` (cell 4) | `DEFAULT_TOLERANCES` | Same values |
| "Globally optimal" statement | — | `CERTIFIED_MESSAGE`, shown only when every residual passes | No convergence claim (paper Appendix A) |

## Stochastic and sweep experiments

| Paper / notebook | Traffic_Emulator | Notes |
|---|---|---|
| Windowed stochastic simulation (cell 14): Poisson counts per window, EWMA β = 0.3, γ = 0.5, split inertia η = 0.35, 4 warm-up windows, 5 s windows over 300 s | `controller_mode: windowed_stochastic`, `simulation/handler.py` | Per-event queues (exponential service, FIFO) instead of Poisson counts per window; notebook `ROUTING_MODE = "random"` corresponds to the per-unit random broker choice; runs until stopped unless `duration` is set |
| — (no counterpart) | `controller_mode: capacity_safe_event_driven` (`simulation/handler.py::_algorithm1_update`) | Not in the paper or notebook: the notebook's event-style experiment uses the windowed scheme. This mode drives the event queues with exact Algorithm 1 steps (`iteration_step`) on planned rates, so the paper's per-iteration capacity guarantee applies to the planned routing |
| Load sweep over service-load targets 0.2 / 0.55 / 0.85 by scaling μ (cells 5–6, `run_load_sweep`) | `scripts/sweep_5x3.py`, `scripts/compare_modes.py` | Different design: scales demand, λ_i(r) = r·λ_i, up to the feasibility limit r_max ≈ 4.87; reports measured rather than target utilization; compares all modes |
| — | `model/symmetric.py`, `docs/symmetric_case.md` | Closed-form symmetric N × M optimum used as an oracle (not in the paper) |
| Verification batteries (cells 6–8): randomized and stressed best-response regressions, finite-difference derivatives, sparse LP regression, capacity-scaled load sweep (0.2 / 0.55 / 0.85), multistart from `random_feasible_initialization` | `tests/regression/test_notebook_batteries.py`, `tests/unit/test_sparse.py`; `feasibility.py::random_feasible_routing` | Ported with the notebook's tolerances. Our residual definitions differ as noted above. The load sweep and multistart are marked `slow` |
| Time-varying capacities (`vary_link_capacity`, `vary_broker_capacity`, the 11 `combos`, phase offsets; cell 1) | `model/dynamics.py`, config `dynamics.capacity_variation` | Same formulas, floors and combos. The notebook freezes them (`FREEZE_CAPS = True`) in all its experiments; here they are optional and applied per window in every mode |
