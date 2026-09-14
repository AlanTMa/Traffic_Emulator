# Replicating the paper's results

Section V of arXiv:2602.03246 reports numbers for the 5x3 instance in
`config/paper_5x3.yaml`. `scripts/replicate_paper.py` computes the same
quantities with this emulator and prints both:

    python -m scripts.replicate_paper                        # deterministic part and 10 windowed seeds
    python -m scripts.replicate_paper --run runs/distributed  # also a finished distributed run

`tests/regression/test_paper_results.py` runs the same checks. The table
below is the script's output on 2026-09-13 (Python 3.13, numpy 2.3.1, scipy
1.15.3, Windows), with a distributed run of 9 local processes at 1 s rounds
(`python -m src.cli up --backend local --window 1 --duration 85`).

## Centralized benchmark and Algorithm 1 (Sec. V-B to V-E, Table I)

| Quantity | Paper | Emulator |
|---|---|---|
| F* | 2.0157 (29 SLSQP iterations) | 2.0157 (29) |
| Utilizations SN1-SN3 | (0.158136, 0.138642, 0.102983) | same, to six decimals, central and distributed |
| Algorithm 1 iterations to the stopping test | 70 | 70 |
| F_dist | 2.0157 | 2.0157; unrounded 2.0157473649139757, the notebook's stored value |
| \|F_dist − F*\| | 1.76e-13 | 1.76e-13 |
| Largest utilization difference, central vs distributed | 6.10e-13 | 6.10e-13 |
| F along the iteration | 2.0345 at the first plotted point, 2.0158 within ten iterations | 2.0345 after iteration 1, 2.0158 after 10 |
| Utilization trend | SN1 and SN2 fall, SN3 rises | (0.1726, 0.1745, 0.0717) → (0.1581, 0.1386, 0.1030) |
| Per-source mean delay | P4 smallest, P3 largest | P4 smallest, P3 largest |
| Active routes | P0, P1, P4 on SN3; P2, P3 on all three | same |
| Table I, conservation (central / distributed) | 0 / 1.31e-11 | 0 / 1.31e-11 |
| Table I, capacity violation | 0 / 0 | 0 / 0 |
| Table I, price consistency | N/A / 1.86e-14 | 0 / 1.86e-14 |
| Table I, fixed point | 1.23e-11 / 5.86e-9 | 1.23e-11 / 5.86e-9 |
| Table I, KKT complementarity | 6.94e-17 / 1.41e-10 | 1.42e-16 / 1.41e-10 |
| Table I, active stationarity | 6.94e-18 / 5.18e-12 | 6.94e-18 / 5.18e-12 |

The paper's "2.0345 initially" is the first point of its objective plot,
which the notebook records after the first iteration; the LP starting
point itself has F = 2.0517. The central KKT complementarity residual is
the least-squares polish's endpoint and sits at machine precision in both.

## Windowed stochastic experiment (Sec. V-F, Figs. 6 and 8)

The paper reports one 300 s run of its per-window count model. The
emulator's `windowed_stochastic` mode runs the same controller (5 s
windows, β = 0.3, γ = 0.5, η = 0.35, four warm-up windows) over
event-driven queues; ten seeds, post-warm-up statistics.

| Quantity | Paper (one run) | Emulator (mean of 10 seeds) |
|---|---|---|
| Mean EWMA utilization SN1-SN3 | (0.1571, 0.1403, 0.0984) | (0.1586, 0.1407, 0.0997) |
| Relative to the deterministic utilizations | 0.65%, 1.18%, 4.41% | 0.30%, 1.52%, −3.21% |
| Window-averaged split vs deterministic, P2 / P3 | within 0.34 / 0.27 pp | 0.31 / 0.29 pp |
| P4 | exact | exact |
| P0, P1 share left on SN2 | about 3.3% | 3.32% |
| Service-node queues | remain zero (count model) | up to 3 units in a window |

Single seeds land within 1.2%, 4.6% and 6% of the deterministic
utilizations for SN1, SN2, SN3. The 3.3% on SN2 is the same finite-horizon
artifact in both: P0 and P1 start on SN2 from the LP routing and move
toward SN3 with inertia 0.35 after the warm-up, so their window average
keeps a geometric tail.

## The distributed emulator (9 processes, 1 s rounds)

| Quantity | Paper | Live run |
|---|---|---|
| F at round 70 | 2.0157 | 2.0157 (unrounded, the notebook's value) |
| Utilizations at round 70 | (0.158136, 0.138642, 0.102983) | same |
| Certificate | met with the stopping test at 70 | first passes at round 56 |
| Measured EWMA utilization, rounds 5-85 | (0.1571, 0.1403, 0.0984) | (0.1599, 0.1415, 0.0973) |
| Relative to the deterministic utilizations | 0.65%, 1.18%, 4.41% | 1.13%, 2.09%, −5.54% |

The planned iterates of the live processes are the notebook's numbers bit
for bit (see docs/distributed.md); the measured utilizations are the
stochastic part, from 5,125 work units served in 85 s.
