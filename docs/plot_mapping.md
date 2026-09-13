# Notebook plots and the dashboard

Every figure and table the notebook (`WiOpt26JNSC_Extended.ipynb`) produces,
and where the live dashboard shows the same thing. Cells are counted from 0;
figure names are the notebook's `savefig` names.

Status: done (shown live), partial, replaced (a different quantity serves
the purpose), missing, deferred (dynamics, which come after the static
system), n/a (input data or offline checks, not a live plot).

| Notebook plot/output | What it shows | Current dashboard equivalent | Status | Notes |
|---|---|---|---|---|
| Cell 0, `df_topics` | Topic message rates that give the source rates λ_i | Launch sidebar topology tables | n/a | The rates are `config/paper_5x3.yaml` |
| Cell 1, broker capacities μ_b(t) | Time-varying broker capacities | Brokers, "Broker capacities μ_j(t)" (with a `dynamics` config, in-process only) | done | The notebook freezes capacities in every experiment (`FREEZE_CAPS`) |
| Cell 1, sample link capacities μ̂_{p,b}(t) | Time-varying link capacities | none; `mu_links_t` is in telemetry | deferred | Dynamics scope |
| Cell 1, `df_link_caps`, `df_broker_caps` | Capacity samples | none | n/a | |
| Cell 4, `df_source_br_validation` | Closed-form vs optimizer best response | none | n/a | `tests/regression/test_notebook_batteries.py` |
| Cell 4, `df_sync_residuals.T`, `df_sync_pass` | Proposition 1 residuals and pass/fail, central and distributed | Optimality, residual table with tolerance and pass | done | Live state only; no central column |
| Cell 4, printed F* and F_dist | Objectives | Overview, "Objective" | done | |
| Cells 6 and 8, load sweep and battery tables | Offline verification | none | n/a | `scripts/sweep_5x3.py`, `tests/regression/` |
| Cell 9, `ObjvsIter.png` | F vs iteration | Overview, "Objective F" | done | |
| Cell 9, `RelchangevsIter.png` | Relative routing change, log scale | Overview, "Relative change" | done | The dashboard plots max(route, price) change, the stopping rule's quantity; the notebook plots the routing change only |
| Cell 9, `MaxBrokerUtil.png` | max_j Λ_j/μ_j vs iteration | Overview, "Max broker utilization" | done | Added with this mapping; was a number only |
| Cell 10, `df_util_compare`, `PerBrokerUtils.png` | Per-broker utilization, central vs distributed final | Brokers table (current planned and measured) | missing | Needs the centralized optimum, which no live run computes |
| Cell 10, `PerBrokerUtil_vsIter_DistRef.png` | Per-broker utilization trajectories, dashed final reference | Brokers, "Broker utilization" | done | No final-reference line: unknown while running |
| Cell 11, `df_src` | Per-source mean delay, central vs distributed final | none | missing | Central |
| Cell 11, bar chart (B) | Same, as bars | — | n/a | Disabled in the notebook (inside a string literal) |
| Cell 11, `PerSourceMeanDelay_vsIter.png` | Per-source mean delay trajectories | Routing, "Per-source delay (model)" | done | |
| Cells 12 and 13, Wardrop check, central panel | M_ij per source at the central optimum | none | missing | Central |
| Cells 12 and 13, Wardrop check, distributed panel | M_ij per source, used vs unused routes, max deviation | Optimality, "Marginal costs, latest iteration" (with α_i) and "Total marginal cost" trajectories | done | Per-source deviation is `active_spread_i` in telemetry; its max is `r_kkt_stationarity` in the table. Cell 13 only changes the used-route threshold |
| Cell 15, `Stoch_BrokerUtilEWMA_vsTime.png` | EWMA utilization over time | Brokers, "Broker utilization", measured (dashed) | done | |
| Cell 15, `Stoch_PerSourceMeanDelay_vsTime.png` | Per-source delay proxy (q+1)/μ | Queues & latency, "Per-source sojourn: measured vs model" | replaced | The emulator measures each unit's sojourn time |
| Cell 15, `Stoch_QueueLen_vsTime.png` | Broker queue lengths over time | Queues & latency, "Broker queue lengths" | done | |
| Cell 15, printed max/mean/nonzero q_broker | Queue summary | none | partial | Read off the chart |
| Cell 15, `RateSplit_Dist_vs_Stoch.png` | Deterministic split vs window-averaged split, per source | Routing, "Current split" and "Average split" | partial | Average added with this mapping; the deterministic reference (static Algorithm 1) is not in a live run |
| Cell 16, u_det vs u_stoch mean/std | Utilization statistics after warm-up | Brokers table | partial | Current values, no mean/std; `scripts/compare_modes.py` offline |
| Cell 17, `Stoch_BrokerUtilEWMA_vsTime_DistRef.png` | EWMA utilization with central and distributed references | Brokers, "Broker utilization" | partial | Reference lines missing |

Unmapped: every central-reference item (cells 10, 11, 12/13 central panel,
17). All need the centralized optimum; the smallest change is for the
controller to solve `central.solve_central` at start and write it to
`run.json`. Deferred: the link-capacity plot, with dynamics.
