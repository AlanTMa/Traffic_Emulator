# Distributed emulator

`src.cli up` starts one controller, N source processes and M broker
processes (containers with `--backend docker`, child processes with
`--backend local`) and a read-only dashboard. The processes share nothing
but TCP messages and the run directory the controller writes.

    controller   Algorithm 1 state (λ_ij, p_j), the common safe step,
                 Proposition 1 residuals, metrics.jsonl and run.json
    source i     λ_i and its access links; Poisson arrivals, one FIFO queue
                 per link served at Exp(μ_ij), routing by λ_ij/λ_i, its own
                 best response
    broker j     μ_j; a FIFO queue served at Exp(μ_j), completions and
                 latencies, its own damped price
    dashboard    reads metrics.jsonl and run.json

## A round

Every `window` seconds the controller runs one Algorithm 1 iteration with
the same functions `synchronous.iteration_step` calls:

1. Λ_j = Σ_i λ_ij from the planned routing.
2. `price` request to each broker with (p_j, Λ_j). The broker returns
   p_j ← (1−γ)p_j + γC_j(Λ_j) (`broker_price`) and its measurements since
   the last round.
3. `best_response` request to each source with the new prices. The source
   returns λ_i^BR (`source_best_response`) and its measurements. A failed
   best response holds that source's row.
4. `apply_common_step`: s_j, s_t, λ ← (1−ηs_t)λ + ηs_t λ^BR. Each source
   gets its new row (`routing`) and routes new arrivals with it.
5. Diagnostics and the certificate on the new state, with the aggregated
   measurements, go to metrics.jsonl.

Control uses planned rates only; measured rates, queue lengths and sojourn
times are telemetry. The round is a barrier: if a broker does not answer,
the round is skipped (`skipped_rounds`) and the state is unchanged. Floats
cross the wire as `repr`, so a round equals `iteration_step` bit for bit
(`tests/integration/test_distributed.py`). `scripts/verify_distributed_run.py`
replays a finished run; checked inside the containers the difference is 0,
from a Windows host about 1e-15, from the different libm.

The certificate does not end the run. Traffic keeps flowing until Ctrl+C or
`--duration`.

## Queues and timing

μ_ij is in the source process (`SourceWorker._serve_access`): a unit waits
in the link's FIFO, is served for Exp(μ_ij), then is written to the broker's
socket. μ_j is in the broker (`BrokerWorker._serve`). Neither is the network
or the CPU.

Service runs on a virtual clock: start = max(arrival, server free),
completion = start + Exp(μ), and the process sleeps until then. A late
wake-up delays the hand-off, not the server, so each queue is an M/M/1 queue
in wall time. Arrivals are paced the same way.

A work unit carries its timestamps: `g` generated, `as`/`ac` access start
and completion, `ba` broker arrival, `bs`/`bc` broker start and completion.
The broker reports access_wait, access_service, transfer (ba − ac),
broker_wait and broker_service. `queueing_sojourn_mean` is the sum without
transfer, which is TCP plus timer granularity (about 1.7 ms in Docker, 11 ms
for local processes on Windows) and not part of the model.

## Protocol

Newline-delimited JSON over TCP (`src/distributed/protocol.py`).

Control plane, controller port 7000. A worker sends `register` {role,
instance, protocol, and for brokers data_address}; the controller answers
`assignment` with the logical index, id, parameters and seed. Brokers are
assigned at once, sources once every broker is known, because their
assignment lists the brokers' addresses. Sources connect to their brokers
and send `ready`; when all are ready the controller sends `start` {t0}.
Rounds are request/reply pairs matched by `req`; `routing` and `stop` need
no reply. `shutdown` sent to the controller stops the run (the launcher's
Ctrl+C path).

Data plane, broker port 7100 in Compose, any free port locally: one
connection per (source, broker) pair carrying work units.

Registration is idempotent per instance name (hostname-pid by default) and
refuses an (N+1)th source or (M+1)th broker; N and M come from the topology.
Brokers advertise their container IP (`ADVERTISE_HOST` overrides it). A bare
connect-and-close, which is Compose's health check, is ignored.

## Reproducibility

Each process seeds numpy from [root_seed, role (1 source, 2 broker), index];
a source spawns separate streams for arrivals, routing and each link's
service. The Algorithm 1 iterates are deterministic; the realized traffic is
not, since routing updates land at wall-clock times. `resolved_config.yaml`
is the config the processes ran; `run.json` adds the instance → id mapping
and the git commit, which the launcher passes in because the image has no
`.git`.

`config/paper_5x3.yaml` is written for the windowed scheme, so the
distributed backend uses the Algorithm 1 defaults η = 0.25, γ = 0.5,
δ_s = 1e-8 and says so in `notes`. `--eta`, `--gamma` and `--delta-s`
override.

## Running

    python -m src.cli up                                   # paper 5x3, Docker
    python -m src.cli up --sources 12 --brokers 4 --load 0.5 --seed 7
    python -m src.cli up --backend local
    python -m scripts.verify_distributed_run runs/distributed

`Dockerfile` builds one image. `docker-compose.yml` has the controller
(health-checked), the replicated broker and source services and the
dashboard on Compose's default network, with `./runs` mounted for the
telemetry. Ctrl+C sends SIGTERM to every container; workers stop first, so
the controller may log one skipped round before it exits. If `docker` is not
on PATH the launcher looks in Docker Desktop's install folder.

The dashboard link is printed once the first round is recorded. The LAN
address works from other devices only if the network allows it. The
dashboard has no login, so don't expose it beyond that.

## Limitations

- Synchronous rounds with a barrier; a dead broker stalls the price step.
- No failure handling beyond that: sources do not reconnect to a restarted
  broker, and the controller is a single point of failure.
- Static capacities and rates only; a `dynamics` section is refused.
- The controller computes Λ_j and sends it to the brokers, rather than the
  brokers assembling it from source announcements.
- Timestamps come from each host's clock; multi-host runs need synced
  clocks.
- At very high rates the Python event loops, not μ, become the bottleneck.
  Compare the measured rates with the plan.
