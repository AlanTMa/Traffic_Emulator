# Traffic Emulator

A standalone Python emulator for **joint access-link and broker congestion
pricing**: N sources route traffic to M brokers (service nodes) over
per-pair access links, brokers publish congestion prices, and sources
best-respond. It implements the model and synchronous distributed algorithm
of the WiOpt'26 paper, adds an event-driven queueing simulation, and
continuously checks the paper's optimality conditions while it runs.

Two execution backends run the same Algorithm 1 code:

- **in-process (reference):** one Python process, analytic or event-driven
  (`python -m src.cli run`). The regression gates run here.
- **distributed:** one controller, N source and M broker processes (Docker
  containers or local processes) connected over TCP, with explicit access
  queues (μ_ij) at the sources and broker queues (μ_j), and a read-only
  dashboard (`traffic-emulator up`). See
  [docs/distributed.md](docs/distributed.md).

- **Paper:** [arXiv:2602.03246](https://arxiv.org/abs/2602.03246)
  (extended version; Algorithm 1, Proposition 1).
- **Reference implementation:**
  [ANRGUSC/JointNetServerCongestion-WiOpt26](https://github.com/ANRGUSC/JointNetServerCongestion-WiOpt26)
  (`WiOpt26JNSC_Extended.ipynb`, also in this repo). The 5x3 regression gate
  is generated from that notebook's own functions.
- **Paper → notebook → code mapping:** [docs/reference_mapping.md](docs/reference_mapping.md).

## Mathematical model

Source *i* offers λ_i and splits it into flows λ_ij ≥ 0 with Σ_j λ_ij = λ_i.
Broker *j* receives Λ_j = Σ_i λ_ij. Access link (i, j) and broker j are
M/M/1 queues with service rates μ_ij and μ_j:

| Quantity | Formula |
|---|---|
| delays | D_ij = 1/(μ_ij − λ_ij), D_j = 1/(μ_j − Λ_j) |
| objective (eq. 4) | F = Σ_ij λ_ij D_ij + Σ_j Λ_j D_j (flow-weighted, not divided by Σλ) |
| marginal costs | C_ij = μ_ij/(μ_ij − λ_ij)², C_j = μ_j/(μ_j − Λ_j)² |
| broker price | p_j = C_j(Λ_j) |
| source best response | min Σ_j λ_ij D_ij + p_j λ_ij s.t. Σ_j λ_ij = λ_i (threshold solution, eq. 16) |

**Algorithm 1** (per iteration t, step sizes η, γ, capacity margin δ_s):
damped prices p ← (1−γ)p + γ C_j(Λ_j); every source computes its best
response λ^BR; per-broker step bounds s_j = min(1, (μ_j − δ_s − Λ_j)/(η Δ_j))
for Δ_j = Σ_i(λ^BR_ij − λ_ij) > 0 (else 1); common step s_t = min_j s_j;
update λ ← (1 − η s_t) λ + η s_t λ^BR.

**Proposition 1:** a *feasible* pair (λ*, p*) with price consistency
p*_j = C_j(Λ*_j) (17) and local best response — C_ij + p*_j = α_i on used
routes, ≥ α_i on unused routes (18) — is globally system-optimal.

### Units

All rates are **normalized work units per second**. The paper derives source
rates from message traffic (msg/s × size) and uses MB/s; the event simulator
treats one event as one unit of work (5x3 instance: 1 unit = 1 MB, the
notebook's `PACKET_MB = 1.0`) with Poisson arrivals and exponential service.
Message sizes and per-message service are not modeled, so simulated
"latency" is the sojourn time of a work unit, not a message latency.

## Controller modes

Set `simulation.controller_mode` in a config, or pass `--controller` to
`python -m src.cli run` (a different mode than the config's runs with that
mode's default parameters, since e.g. η means split inertia in the windowed
scheme and the Algorithm 1 step size otherwise):

| Mode | What runs | Guarantees |
|---|---|---|
| `static_algorithm1` | Paper Algorithm 1 on the analytic model; no events or queues. One iteration per `window` seconds. | Every iterate keeps Λ_j ≤ μ_j − δ_s (safe step). The 5x3 run matches the reference notebook (70 iterations, F = 2.0157473649138, λ within 1e-9). |
| `capacity_safe_event_driven` | The same event-driven queues, but at every window boundary the **planned** routing is advanced by exactly one Algorithm 1 step (`synchronous.iteration_step`, the function `static_algorithm1` uses) on planned rates; arrivals are then routed with probabilities λ_ij / λ_i. Measured rates, queues and sojourn times are recorded but never enter the controller. No warm-up; β only smooths measured-rate telemetry. | Planned Λ_j ≤ μ_j − δ_s and source conservation at every update (safe step). Its planned iterates equal `static_algorithm1`'s bit for bit, independent of the traffic seed. Nothing is guaranteed about *measured* queues, which are stochastic. |
| `windowed_stochastic` | Event-driven queues (Poisson arrivals, exponential service, separate access and broker queues) driven by the notebook's **windowed stochastic** scheme: per window, brokers take an EWMA (β) of *measured* arrival rates, damp prices (γ), and after `warmup` windows every source moves its split toward its best response with inertia η. | None per iteration: there is **no safe step**, so planned broker loads are not kept below capacity by construction, and prices track noisy measurements, so the Proposition 1 certificate is not expected to pass exactly. |

`static_algorithm1` vs `capacity_safe_event_driven` isolates the effect of
stochastic arrivals (same controller); `capacity_safe_event_driven` vs
`windowed_stochastic` isolates the effect of a controller driven by noisy
windowed measurements (same queues). All three update every source at one
global barrier; none is asynchronous (see [Roadmap](#roadmap)).

### What is and is not guaranteed

- **Guaranteed / checked:** exact Algorithm 1 safe step (tests where it
  binds); δ_s separate from the numerical `eps`; source conservation and
  capacity feasibility of static iterates; centralized-vs-distributed
  objective gap on 5x3 (≤ 1e-7, observed 8.7e-14); Proposition 1 residuals
  on every telemetry record.
- **Not claimed:** global convergence of Algorithm 1 (the paper's Appendix A
  does not claim it on the open M/M/1 domain). The dashboard reports
  *"Current state satisfies feasible fixed-point optimality conditions"*
  only when every certificate residual is within tolerance for that state.
  On the symmetric 5x3 instance (broker utilization 0.75) the paper's
  step sizes η = 0.25, γ = 0.5 produce a feasible period-2 cycle that never
  certifies; η = 0.01, γ = 0.02 converged on every tested symmetric case
  ([docs/symmetric_case.md](docs/symmetric_case.md)). Capacity safety held in
  all of these runs; convergence depends on damping.
- **Input validation:** topologies reject NaN/inf, negative rates,
  non-positive capacities, malformed shapes, duplicate ids, and configs with
  missing, unknown, duplicate or malformed access links (no capacity is
  silently zero). The transportation LP rejects routings short of the
  requested headroom (HiGHS's tolerance exceeds the 1e-8 margin).
- **Tolerances** are the notebook's absolute values (e.g. KKT
  complementarity 1e-8). They are scale-dependent: at 95% of the maximum 5x3
  load the notebook stopping rule ends before KKT complementarity meets 1e-8;
  `run_algorithm1(require_certificate=True)` implements the paper's Step 5
  (stop only when certified).
- **Canonical 5x3 objective:** F* = 2.0157473649138 on the notebook's exact
  seed-42 parameters (`config/paper_5x3.yaml`). The value 2.015766 seen in
  earlier versions came from parameters rounded to 3 decimals.

## Architecture

```
src/
  cli.py                 `run` (in-process backend) and `up` (distributed backend)
  distributed/           distributed backend: protocol (NDJSON over TCP),
                         controller, source and broker workers, run settings,
                         launcher (Docker Compose or local processes)
  model/                 topology + units + validation, configs (controller_mode),
                         delays, marginal costs, topology generator, symmetric oracle
  controller/
    synchronous.py       Algorithm 1: iteration_step, safe_step_bounds, run_algorithm1
    best_response.py     exact threshold best response (eq. 16)
    feasibility.py       max-headroom transportation LP (initial routing)
    central.py           centralized solver (SLSQP + KKT polish) and objective F
    diagnostics.py       Proposition 1 certificate: residuals, alpha_i, spreads
  simulation/            event engine (wall-clock pacing), events, deque queues,
                         handler (windowed_stochastic / capacity_safe_event_driven),
                         SystemState
  telemetry/             schema (one JSON record per iteration/window),
                         append-only JSONL writer, incremental reader
  runtime/metadata.py    seeds, git SHA, run.json
  runtime/experiments.py shared static/event experiment runners
  dashboard/app.py       Streamlit: launch/stop in-process runs, live research
                         views; observer-only for distributed runs
scripts/                 load_sweep.py (1x1 M/M/1), sweep_5x3.py (high load),
                         compare_modes.py (three modes side by side)
config/paper_5x3.yaml    the canonical 5x3 instance (exact notebook parameters)
tests/                   unit, integration (incl. distributed vs reference), regression
Dockerfile, docker-compose.yml   one image; controller/broker/source/dashboard services
```

Each run writes `runs/<name>/metrics.jsonl` (per-iteration records: λ_ij,
fractions, Λ_j, per-broker/per-link utilization, p_j, α_i, D_ij, D_j,
per-source delay, C_ij, C_j, C_ij + C_j, objective, s_j, s_t, route/price
changes and all certificate residuals; the event mode adds measured rates,
queue lengths and latency percentiles) and `runs/<name>/run.json` (seed,
config, N, M, rates, capacities, η, γ, β, δ_s, mode, git SHA).

## Running

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

Live dashboard (set up a topology or pick a config, then Launch/Stop):

```bash
streamlit run src/dashboard/app.py
```

Views: Overview, Brokers (per-broker utilization, loads, prices), Routing
(fractions, per-source delay), Optimality (certificate, C_ij + C_j with
used/unused routes, KKT/Wardrop equalization, residuals, s_t/s_j), Queues &
latency (event mode: queue lengths, mean/p50/p95/p99).

Command line (Ctrl+C stops; runs without `duration` continue until stopped):

```bash
python -m src.cli run --config config/paper_5x3.yaml
```

Options: `--no-realtime` (run as fast as possible), `--output-dir runs/NAME`
(default `runs/latest`). Set `simulation.seed` for reproducible event runs;
unseeded runs record the seed they drew in `run.json`.

Generated N x M topology instead of a config (capacities drawn from the
notebook's ranges, source rates scaled to `--load` of total broker capacity):

```bash
python -m src.cli run --sources 10 --brokers 5 --load 0.5 --seed 42 --controller static_algorithm1
```

Also `--controller capacity_safe_event_driven|windowed_stochastic`,
`--window`, `--duration`, `--warmup`, `--eta/--gamma/--beta/--delta-s`. The
complete config is written to `<output-dir>/generated_config.yaml`; passing
it back with `--config` reproduces the run exactly (the seed drives both the
topology and the event RNG).

### Distributed emulator

One command starts the controller, N source and M broker containers and the
dashboard (http://localhost:8501); Ctrl+C stops everything:

```bash
pip install -e .          # optional: provides `traffic-emulator` (else use python -m src.cli)
traffic-emulator up --sources 5 --brokers 3 --config config/paper_5x3.yaml
```

This runs `docker compose --profile distributed up --build --scale source=5 --scale broker=3`
on the resolved run config in `runs/distributed/`. Any N x M topology:

```bash
traffic-emulator up --sources 12 --brokers 4 --load 0.5 --seed 7
```

`--backend local` runs the same processes without Docker. Also `--window`,
`--duration`, `--no-dashboard`, `--dashboard-port`, `--output-dir runs/NAME`,
`--eta/--gamma/--delta-s`. `python -m scripts.verify_distributed_run runs/distributed`
checks a finished run's rounds against the in-process reference. Each round, brokers compute their damped prices,
sources their best responses, and the controller the common safe step and
the Proposition 1 diagnostics. A round equals the reference
`iteration_step` bit for bit (`tests/integration/test_distributed.py`).
Architecture, protocol, where μ_ij and μ_j are modeled, and limitations:
[docs/distributed.md](docs/distributed.md).

### Sparse topologies

Links that do not exist are declared explicitly; every other source->broker
pair still needs a capacity (an omission is an error):

```yaml
topology:
  sources: [{id: P0, rate: 6.0}, {id: P1, rate: 5.0}]
  brokers: [{id: SN1, capacity: 12.0}, {id: SN2, capacity: 9.0}]
  access_capacities: {"P0->SN1": 7.0, "P0->SN2": 6.0, "P1->SN2": 8.0}
  unavailable_links: ["P1->SN1"]
```

Missing links carry no flow in every mode (best responses, LP start,
centralized solver, event routing), and the certificate adds
`r_unavailable_routes`.

### Time-varying capacities

The reference notebook's `vary_link_capacity` / `vary_broker_capacity`
(sinusoid + Gaussian noise, floors 0.2 / 0.3 of base, seeded phase offsets)
are available in every mode, piecewise constant per window:

```yaml
dynamics:
  capacity_variation:
    combo: 3          # one of the notebook's 11 cases, or link/broker {amp, period, noise}
```

Algorithm 1's safe step then protects against the capacities in effect at
each update; it cannot undo a later capacity drop, and telemetry records
`mu_links_t` / `mu_brokers_t`.

### Reproducing the 5x3 results

```bash
python -m src.cli run --config config/paper_5x3.yaml --controller static_algorithm1 --window 1 --duration 100 --no-realtime
```

Runs Algorithm 1 for 100 iterations on the exact notebook instance and ends
at F = 2.0157473649 with the certificate passing (it first passes at
iteration 56). The regression gate
(`tests/regression/test_baseline_5x3.py`) checks iterations, λ_ij, loads,
utilizations, prices, the active route set, the objective, the centralized
comparison and every residual against `tests/regression/data/baseline_5x3.json`.

### Load sweeps

```bash
python -m scripts.sweep_5x3
```

Scales the 5x3 instance by λ_i(r) = r·λ_i with r at 20–95% of the largest
feasible multiplier (r_max ≈ 4.87, set by P2/P3's access links; aggregate
broker load then is only ~63%, SN2 reaches ~85%). For each r it runs static
Algorithm 1, an event simulation of that optimum (queueing validation),
`capacity_safe_event_driven` and `windowed_stochastic`, reporting planned and
actual utilization per broker and per access path, safe-step activity,
headroom, queue growth, p95/p99 and the measured-vs-analytical gap; results
go to `runs/sweep_5x3/`. The 1x1 M/M/1 check is `python -m scripts.load_sweep`.

Side-by-side comparison of the three modes on one topology (same seed for
both event modes; `--config`, `--multiplier`, `--duration`, `--warmup-time`):

```bash
python -m scripts.compare_modes
```

### Tests

```bash
python -m pytest
```

GitHub Actions (`.github/workflows/tests.yml`) runs the full suite, including
the exact 5x3 regression gate, on Python 3.11–3.13 for every push and pull
request. `tests/regression/test_notebook_batteries.py` ports the notebook's
verification batteries with its tolerances (randomized and stressed best
responses, derivatives, capacity-scaled load sweep, multistart); the load
sweep and multistart are marked `slow` (several minutes). For a quick local
run:

```bash
python -m pytest -m "not slow"
```

The `docker-smoke` CI job runs the distributed emulator in containers (5x3,
40 s, via `python -m src.cli up --backend docker`) and checks its telemetry.

### Docker

`Dockerfile` builds one image for every role. `docker-compose.yml` has two
profiles: `distributed` (controller, replicated `broker` and `source`
services, dashboard; normally started by `traffic-emulator up`) and
`reference` (the single-process emulator and the dashboard):

```bash
docker compose --profile reference up --build
```

## Roadmap

1. **Asynchronous operation** (after a literature review): brokers publish
   prices and sources best-respond on their own clocks with stale per-broker
   price views, and no global barrier. The paper leaves this to future
   work.
2. **Dynamic capacities and rates in the distributed backend** (the
   in-process backend already has the notebook's capacity variation), plus
   scheduled scenario events with adaptation, queue build-up and recovery
   measured.
3. **Failures and restarts** of workers and the controller.
4. **Kubernetes** deployment of the same services.

Details: [docs/distributed.md](docs/distributed.md#next-steps).
