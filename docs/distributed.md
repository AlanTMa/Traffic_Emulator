# Distributed emulator

The emulator has two **execution backends**. They are orthogonal to the
controller mode (`static_algorithm1`, `capacity_safe_event_driven`,
`windowed_stochastic`), which says *what* is computed:

| Backend | What runs | Use |
|---|---|---|
| `in_process` (reference) | One Python process: analytic Algorithm 1, or the event simulation of the queues (`python -m src.cli run`) | Oracle. The 5x3 regression gate, the notebook batteries, sweeps and comparisons all run here |
| `distributed` | One controller process, N source processes, M broker processes and a read-only dashboard, connected only over TCP (`python -m src.cli up`) | The emulator: real processes/containers with explicit access and broker queues, driven by Algorithm 1 |

The distributed backend runs Algorithm 1 on planned rates over live queues,
the same scheme as the in-process `capacity_safe_event_driven` mode. It
supports static capacities and rates only.

## Processes

```
                    +------------------------------------------+
                    |  controller  (src/distributed/controller) |
                    |  Algorithm 1 state (lambda_ij, p_j)       |
                    |  common safe step, Proposition 1 checks,  |
                    |  telemetry: runs/<run>/metrics.jsonl      |
                    +------------------------------------------+
                      ^  control plane (TCP 7000, NDJSON)    ^
       price, metrics |  price request {p_j, Lambda_j}       | best response, metrics
                      v                                      v  best-response request {p}
  +--------------------------+                    +-------------------------------+
  | broker j (x M)           |  data plane        | source i (x N)                |
  | queue served at Exp(mu_j)| <----------------- | Poisson arrivals at lambda_i  |
  | price p_j (broker_price) |  work units (TCP,  | routing: lambda_ij / lambda_i |
  | latency accounting       |  one connection    | access queue per link (i, j)  |
  +--------------------------+  per pair (i, j))  |   served at Exp(mu_ij)        |
                                                  | best response (source_best_   |
                                                  |   response)                   |
                                                  +-------------------------------+

  dashboard: reads runs/<run>/metrics.jsonl and run.json only (observer mode)
```

No process shares Python objects with another. Everything a process knows
about the others arrives as a message.

| Process | Owns | Code |
|---|---|---|
| controller | topology and parameters from the resolved config; registration and logical ids; the Algorithm 1 state λ_ij and published prices; the common safe step; diagnostics and telemetry; `run.json` | `src/distributed/controller.py`, `settings.py` |
| source i | λ_i; μ_ij for its links; Poisson generation; one FIFO access queue per link (i, j); routing of new units with λ_ij / λ_i; its best response | `src/distributed/source.py` |
| broker j | μ_j; its FIFO queue; service, completion and latency accounting; its damped price | `src/distributed/broker.py` |
| dashboard | nothing: renders the telemetry file (`TRAFFIC_EMULATOR_OBSERVER=1` hides launch/stop) | `src/dashboard/app.py` |

## One Algorithm 1 round

Every `window` seconds (wall clock; `simulation.window`, 5 s for
`config/paper_5x3.yaml`) the controller runs one round. Every step calls the
same function the in-process reference calls
(`src/controller/synchronous.py`):

1. **Controller:** planned loads Λ_j = Σ_i λ_ij from its state
   (`compute_broker_loads`).
2. **Each broker j**, on a `price` request carrying (p_j, Λ_j): the damped
   price p_j ← (1−γ) p_j + γ C_j(Λ_j) (`broker_price`). It replies with the
   new price and its measurements since the last round.
3. **Each source i**, on a `best_response` request carrying the new price
   vector: the threshold best response λ_i^BR (eq. 16) over its available
   links (`source_best_response`). It replies with λ_i^BR and its
   measurements. A source whose best response fails reports it; the
   controller holds that source's split (Δ_ij = 0), as `iteration_step`
   does with `on_best_response_failure="hold"`.
4. **Controller:** per-broker bounds s_j, common step s_t = min_j s_j and
   λ ← (1 − η s_t) λ + η s_t λ^BR (`apply_common_step`, the code
   `iteration_step` runs). It sends every source its new row λ_i·, which
   the source's router uses for new units from then on.
5. **Controller:** Proposition 1 residuals and the certificate for the new
   state (`telemetry.schema.controller_snapshot` → `compute_diagnostics`),
   joined with the aggregated measurements, appended to `metrics.jsonl`.

The round is a global barrier (synchronous Algorithm 1), not an
asynchronous algorithm. If a broker does not answer, the round is skipped
and the state is unchanged (`skipped_rounds` in telemetry). The controller
sends each broker its current published price with the request, so a broker
that missed a round cannot drift.

Because the processes run the shared functions, and JSON carries floats
with Python's round-trip `repr`, a distributed round equals
`synchronous.iteration_step` **bit for bit**: prices, s_j, s_t and λ_ij.
`tests/integration/test_distributed.py` checks this in-process (5 rounds on
5x3, and a 2x2 case where the safe step binds) and on live runs of 9 OS
processes (5x3) and 7 processes (generated 4x2), against
`run_algorithm1` round by round. For any finished run,

```bash
python -m scripts.verify_distributed_run runs/distributed
```

replays the reference from the run's resolved config and compares every
recorded round. Bit-identity holds on one platform: container rounds checked
inside the image differ by exactly 0, while the same rounds checked from a
Windows host differ by about 1e-15, because the Linux and Windows numpy/libm
builds round differently. The script's default tolerance is 1e-12; `--tol 0`
demands bit identity.

`iteration` in telemetry counts Algorithm 1 steps applied. A skipped round
applies no step and writes no record.

The prices and routing are computed from **planned** rates, as in the paper.
Measured arrival rates, queue lengths and latencies are reported and plotted
but never used for control. The certificate therefore describes the planned
routing. The live queues are stochastic, so their measured utilizations
fluctuate around the planned values.

Reaching the certificate does not end the run: traffic keeps flowing under
the converged routing and rounds continue until Ctrl+C (or `--duration`).

## Where the capacities are modeled

- **Access capacity μ_ij:** in source i's process. Each link (i, j) has its
  own FIFO queue (`SourceWorker._serve_access(j)`); a unit waits there, is
  served for Exp(μ_ij), and only then is written to broker j's TCP
  connection. μ_ij is a model parameter from the config. It is not the
  network bandwidth, and the TCP hand-off is not a model of it.
- **Broker capacity μ_j:** in broker j's process
  (`BrokerWorker._serve`): FIFO, Exp(μ_j) service. μ_j is a model
  parameter. It is not the container's CPU speed.

Service is paced on **virtual clocks** tied to the wall clock. A unit starts
service at max(its arrival, the server's free time) and completes Exp(μ)
later, and the process sleeps until that completion time. Late wake-ups
therefore delay only the hand-off, never the server, so each queue is an
M/M/1 queue in wall time. Arrivals are paced the same way.

Each work unit is one normalized unit of work (1 MB for 5x3), not a packet
or a message. It carries its timestamps:

| Field | Meaning |
|---|---|
| `w`, `s`, `b` | unit id, source, broker |
| `g` | generated at source |
| `as`, `ac` | access service start / completion |
| `ba` | arrival at the broker |
| `bs`, `bc` | broker service start / completion |

The broker splits each sojourn into `access_wait`, `access_service`,
`transfer` (= `ba − ac`), `broker_wait` and `broker_service`.
`queueing_sojourn_mean` sums the four modeled stages. `transfer` is
process-to-process transport: TCP plus timer granularity, about 11 ms on
Windows and much less on Linux. The model does not include it, so it is
reported separately. All processes use the host's wall clock (containers
share the host kernel clock), so cross-process timestamps are comparable.

## Protocol

`src/distributed/protocol.py`: newline-delimited JSON over asyncio TCP,
protocol version 1.

- **Control plane** (controller port 7000). A worker connects and sends
  `register` {role, instance, protocol; brokers also send `data_address`}.
  The controller answers `assignment` with a logical index and id, the
  worker's parameters and its seed. Brokers are assigned immediately;
  sources are assigned once every broker has registered, because their
  assignment lists every broker's data address. Sources then connect to
  their brokers and send `ready`. When all N sources are ready, the
  controller sends `start` {t0}, and traffic begins at the same instant
  everywhere. Rounds are correlated request/reply pairs (`req` ids):
  `price` to brokers, `best_response` to sources, then `routing` rows to
  sources. `stop` ends every worker; `shutdown` sent to the controller (the
  launcher's cross-platform Ctrl+C path) stops the whole run.
- **Data plane** (broker port 7100 in Compose; any free port locally). One
  TCP connection per available pair (i, j) carries work units from source
  i's access queue to broker j.
- **Identity.** Registration is idempotent per instance name (container
  hostname and pid by default), and a (N+1)-th source or (M+1)-th broker is
  refused. N and M come from the topology, not from the code. Brokers
  advertise their container IP (`ADVERTISE_HOST` overrides it; the local
  launcher uses 127.0.0.1).
- **Health check.** A bare connect-and-close (Compose's controller health
  check) is ignored.

## Reproducibility

- One root seed (`simulation.seed`, or drawn and recorded) gives each
  process the seed `[root_seed, role (1 = source, 2 = broker), index]`. A
  source spawns separate streams for interarrival times, routing choices
  and each access link's service times, so its draws do not depend on how
  its tasks interleave.
- The Algorithm 1 iterates (λ_ij, p_j, s_j, s_t, residuals) are
  deterministic and equal the reference. The realized traffic is not
  bit-reproducible: when a routing update takes effect relative to the
  arrival stream depends on wall-clock timing.
- `runs/<run>/resolved_config.yaml` is the exact config the processes ran:
  topology, window, seed, η, γ, δ_s, eps, β, safe-step variant, and notes on
  any substitution. `run.json` adds `execution_backend: distributed`, the
  instance → logical id mapping of every process, the seed rule, the
  parameters and the git SHA.
- `config/paper_5x3.yaml` is written for the windowed scheme (its η = 0.35
  is the windowed split inertia). The distributed backend then uses the
  paper's Algorithm 1 values η = 0.25, γ = 0.5, δ_s = 1e-8 and says so in
  the notes. A config written for `static_algorithm1` or
  `capacity_safe_event_driven` supplies its own values; `--eta`, `--gamma`
  and `--delta-s` override either.

## Running

```bash
pip install -e .          # optional: provides the `traffic-emulator` command
```

Canonical 5x3 in containers (1 controller, 3 broker and 5 source
containers, dashboard on http://localhost:8501):

```bash
traffic-emulator up --sources 5 --brokers 3 --config config/paper_5x3.yaml
```

This resolves the run into `runs/distributed/`, then runs
`docker compose --profile distributed up --build --scale source=5 --scale broker=3`.
`--sources`/`--brokers` must match the config; they may be omitted.

Any N x M topology (generated from the notebook's capacity ranges at
`--load` of total broker capacity):

```bash
traffic-emulator up --sources 12 --brokers 4 --load 0.5 --seed 7
```

The same processes without Docker, as child processes on this machine
(also what the integration tests run):

```bash
traffic-emulator up --backend local
```

`python -m src.cli up ...` works without installing. Other options:
`--window` (seconds per round), `--duration` (stop after that many seconds;
by default the run continues until Ctrl+C), `--no-dashboard`,
`--output-dir runs/NAME` (must be under `runs/` for Docker, the mounted
directory), `--dashboard-port`, `--port` (local backend's controller port),
`--eta/--gamma/--delta-s`.

Once the dashboard answers, the launcher prints its address on this machine
and on the local network (below the startup output, so it doesn't scroll away). The second works from other devices only if the network allows
device-to-device connections; campus and enterprise Wi-Fi often doesn't.
Streamlit's own startup lines are misleading here, so the launcher hides
them (`docker compose logs dashboard` shows them). Inside a container, its
"Network URL" is the container's address on Docker's private network, and
its "External URL" is the network's public address, which needs port
forwarding. Don't open the dashboard to the internet: it has no login.

If `docker` isn't on PATH (a terminal opened before Docker Desktop was
installed), the launcher looks in Docker Desktop's install folders, and it
says so when Docker is installed but not running.

On Ctrl+C, Compose sends SIGTERM to every container. Workers stop first
(they depend on the controller), so the controller may log one skipped round
before it writes its final record and exits. Every container exits with
code 0.

Dashboard for an existing or running distributed run:

```bash
TRAFFIC_EMULATOR_OUTPUT_DIR=runs/distributed TRAFFIC_EMULATOR_OBSERVER=1 streamlit run src/dashboard/app.py
```

The reference backend on the same instance:

```bash
python -m src.cli run --config config/paper_5x3.yaml --controller static_algorithm1 --window 1 --duration 100 --no-realtime
```

## What Docker does here

`Dockerfile` builds one image with the code and configs. `docker-compose.yml`
starts a service per role from it: `controller` (health-checked), `broker`
and `source` (replicated with `--scale`, no fixed names or ports) and
`dashboard`. Compose's default network gives them DNS (`controller:7000`)
and container IPs for the data plane. `./runs` is mounted so the
controller's telemetry reaches the host and the dashboard; the controller is
the only writer. The `reference` profile runs the single-process emulator
and the dashboard instead.

The containers have been run with Docker Desktop 29.7 (Compose v5.5) on
Windows 11: the canonical 5x3 (9 containers, 20 rounds) and a generated 7x4
with the dashboard container (13 containers, 23 rounds). Both matched the
reference, all containers exited with code 0, and SIGTERM
(`docker compose stop`) shut every process down cleanly. CI repeats the
5x3 run on every push (`docker-smoke` in `.github/workflows/tests.yml`:
the one-command launcher for 40 s, then `scripts/verify_distributed_run.py`).
The image has no `.git`, so the launcher passes the host's commit to the
controller (`TE_GIT_SHA`, `TE_GIT_DIRTY`) for `run.json`.

## Limitations

- Synchronous rounds with a global barrier: a slow or dead source only
  holds its split; a dead broker stalls the price step (rounds are skipped
  until it returns). Failures are not handled beyond that: a restarted
  source rejoins under its instance name, but sources do not reconnect to
  a restarted broker, a worker under a new instance name is refused, and
  the controller is a single point of failure.
- Static capacities and rates. A config with `dynamics` is refused.
- The controller aggregates the planned loads Λ_j and the prices. A broker
  prices the load the controller sends it rather than one assembled from
  sources' announcements.
- Timestamps come from each host's wall clock; multi-host deployments
  would need clock synchronization.
- The transfer hop is reported but not modeled; at very high rates the
  Python event loops, not μ, can become the bottleneck (check
  `access_rate_measured_ij` and `broker_rate_measured_j` against the plan).
- Global convergence of Algorithm 1 is not claimed (paper, Appendix A). The
  certificate describes the current planned state only.

## Next steps

- **Asynchronous operation:** brokers publish prices on their own
  schedules; sources keep stale per-broker price views and best-respond on
  their own clocks; the global barrier and common step are replaced by a
  per-source or per-broker safety rule. It needs the planned literature
  review, since the paper's step rule relies on the synchronous common
  step. The message protocol already separates the roles, so the controller
  would become a monitor, not a coordinator.
- **Dynamic capacities and rates:** time-varying μ_ij, μ_j and λ_i (the
  in-process `model/dynamics.py` already implements the notebook's
  variation) pushed to workers as `capacity`/`rate` messages, with the
  safe step evaluated against the capacities in effect; scheduled scenario
  events, and measurement of adaptation, queue build-up and recovery.
- **Kubernetes:** one Deployment (or StatefulSet, for stable instance
  names) per role, a Service for the controller, and a headless Service or
  pod IPs for the broker data plane. Telemetry would move from the shared
  `runs/` volume to a PersistentVolume or a telemetry service the dashboard
  reads, plus readiness probes and restart handling (re-registration under
  the same logical id).
