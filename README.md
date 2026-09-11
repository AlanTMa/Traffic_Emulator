# Traffic emulator

Emulator for the joint access-link and broker congestion pricing model of
Sarkar and Krishnamachari, WiOpt'26 ([arXiv:2602.03246](https://arxiv.org/abs/2602.03246)).
N sources split their traffic over access links to M brokers, brokers post
congestion prices, sources best-respond. The authors' notebook
(`WiOpt26JNSC_Extended.ipynb`, from
[ANRGUSC/JointNetServerCongestion-WiOpt26](https://github.com/ANRGUSC/JointNetServerCongestion-WiOpt26))
is in the repo, and its 5x3 instance is the default everywhere.

Two ways to run it:

- in one process (`src.cli run`): Algorithm 1 on the analytic model, or an
  event-driven simulation of the queues. This is the reference; the
  regression tests run here.
- as separate processes (`src.cli up`): a controller, N source processes and
  M broker processes, as Docker containers or local processes, talking over
  TCP. Each source runs its own access queues, each broker its own queue,
  and a read-only dashboard follows the run.

## Model

Source i offers λ_i and splits it into λ_ij ≥ 0 with Σ_j λ_ij = λ_i; broker j
sees Λ_j = Σ_i λ_ij. Access link (i, j) and broker j are M/M/1 queues with
rates μ_ij and μ_j:

    F    = Σ_ij λ_ij/(μ_ij − λ_ij) + Σ_j Λ_j/(μ_j − Λ_j)        (eq. 4)
    C_ij = μ_ij/(μ_ij − λ_ij)²,   C_j = μ_j/(μ_j − Λ_j)²,   p_j = C_j(Λ_j)

Algorithm 1, per iteration: p ← (1−γ)p + γC_j(Λ_j); each source computes its
best response λ^BR to p (the threshold solution of eq. 16); for brokers whose
load would grow by Δ_j, s_j = min(1, (μ_j − δ_s − Λ_j)/(ηΔ_j)); s_t = min_j s_j;
λ ← (1 − ηs_t)λ + ηs_t λ^BR. Every iterate keeps Λ_j ≤ μ_j − δ_s.

Proposition 1 says a feasible fixed point with p_j = C_j(Λ_j) and equal
marginal cost C_ij + p_j on every used route is optimal. Its residuals are
computed on every iteration and reported as the certificate. Convergence is
not guaranteed by the paper and not claimed here; [docs/symmetric_case.md](docs/symmetric_case.md)
has a case where the paper's step sizes oscillate.

Rates are work units per second; for the 5x3 instance one unit is 1 MB. A
simulated work unit is not a packet or a message.

## Setup

Python 3.11 or newer.

    pip install -r requirements.txt
    pip install -e .        # optional, provides the traffic-emulator command

## Running

The paper's 5x3 instance as containers, dashboard at http://localhost:8501.
Needs Docker; Ctrl+C stops everything:

    python -m src.cli up

Any N x M (capacities drawn from the notebook's ranges, rates scaled to
`--load` of total broker capacity), or the same processes without Docker:

    python -m src.cli up --sources 12 --brokers 4 --load 0.5 --seed 7
    python -m src.cli up --backend local

Other options: `--window` (seconds between rounds, default 5), `--duration`,
`--no-dashboard`, `--dashboard-port`, `--output-dir`, `--eta`, `--gamma`,
`--delta-s`. `python -m scripts.verify_distributed_run runs/distributed`
replays a finished run against the in-process algorithm. How the processes
work: [docs/distributed.md](docs/distributed.md).

Single process:

    python -m src.cli run --config config/paper_5x3.yaml
    python -m src.cli run --config config/paper_5x3.yaml --controller static_algorithm1 --window 1 --duration 100 --no-realtime
    python -m src.cli run --sources 10 --brokers 5 --load 0.5 --controller capacity_safe_event_driven

The first is the notebook's windowed experiment. The second reproduces the
notebook's Algorithm 1 run: 70 iterations to tolerance, F = 2.0157473649,
certified from iteration 56. The third generates a topology and writes it to
`<output-dir>/generated_config.yaml`. Without `--no-realtime` one window
passes per `window` seconds, so the dashboard can follow:

    streamlit run src/dashboard/app.py

### Controller modes

`simulation.controller_mode` in the config, or `--controller`:

- `static_algorithm1`: Algorithm 1 on the analytic model, one iteration per
  window. No queues.
- `capacity_safe_event_driven`: event-driven queues; the planned routing is
  advanced by one Algorithm 1 step per window and new arrivals are routed
  with it. Measured rates are recorded, never used for control. This is
  what the distributed emulator runs.
- `windowed_stochastic`: the notebook's scheme (cell 14). Per window, prices
  from an EWMA of measured arrival rates; after a warm-up, splits move
  toward the best response with inertia η. No safe step.

Switching a config to a mode from the other family swaps in that mode's
default parameters, since η is the split inertia in the windowed scheme and
the step size in the others.

### Config

    simulation: {controller_mode: capacity_safe_event_driven, window: 5, seed: 1}
    algorithm: {eta: 0.25, gamma: 0.5, delta_s: 1.0e-8, eps: 1.0e-12}
    topology:
      sources: [{id: P0, rate: 6.0}, {id: P1, rate: 5.0}]
      brokers: [{id: SN1, capacity: 12.0}, {id: SN2, capacity: 9.0}]
      access_capacities: {"P0->SN1": 7.0, "P0->SN2": 6.0, "P1->SN2": 8.0}
      unavailable_links: ["P1->SN1"]      # optional; every other pair needs a capacity
    dynamics:                             # optional, in-process only
      capacity_variation: {combo: 3}      # the notebook's time-varying capacities

Every run writes `metrics.jsonl`, one record per iteration (λ_ij, Λ_j,
prices, delays, marginal costs, F, the safe step, every certificate residual,
and in the event modes measured rates, queue lengths and sojourn
percentiles), and `run.json` (config, seed, parameters, git commit).

## Layout

    src/model        topology, config, marginal costs, generator, symmetric oracle, capacity dynamics
    src/controller   Algorithm 1 (synchronous.py), best response, LP initializer, central solver, diagnostics
    src/simulation   event engine, queues, handler for the two event modes
    src/distributed  protocol, controller/source/broker processes, launcher
    src/telemetry    record schema, JSONL writer and tail reader
    src/runtime      run metadata, experiment runners
    src/dashboard    Streamlit app
    scripts          sweep_5x3, compare_modes, verify_distributed_run
    tests            unit, integration, regression (5x3 baseline from the notebook)
    docs             reference_mapping (paper / notebook / code), distributed, symmetric_case

## Tests

    python -m pytest                 # everything, about 7 minutes
    python -m pytest -m "not slow"

The regression tests check the 5x3 run against values produced by the
notebook's own functions, and a live multi-process run against the
in-process algorithm, round by round. CI runs the suite on Python 3.11 to
3.13 and a 40 s Docker run of the 5x3 emulator.

## Docker

One image, one service per role. `src.cli up` runs
`docker compose --profile distributed up --build --scale source=N --scale broker=M`;
`docker compose --profile reference up --build` runs the single-process
emulator with the dashboard.

## Not done

Asynchronous updates (the paper leaves them open), dynamic capacities in the
distributed processes, worker restarts, Kubernetes.
