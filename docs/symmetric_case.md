# The fully symmetric N × M case

A closed-form correctness oracle for the model. It is implemented in
`src/model/symmetric.py` and tested in `tests/unit/test_symmetric.py`.

## Setting

- N identical sources, each offering λ_i = λ.
- M identical brokers, each with service rate μ_j = μ_s ("server").
- Every access link (i, j) has service rate μ_ij = μ_a.
- All links are M/M/1 queues, as in the paper.

Routing λ_ij ≥ 0 must satisfy Σ_j λ_ij = λ for every source. The broker loads
are Λ_j = Σ_i λ_ij, and the objective is

F(λ) = Σ_ij f_a(λ_ij) + Σ_j f_s(Λ_j), where f_a(x) = x/(μ_a − x) and f_s(y) = y/(μ_s − y).

## Feasibility

A routing needs λ_ij < μ_a and Λ_j < μ_s for every i and j.

- **Access links:** each source sends λ over M links of capacity μ_a. This is
  possible exactly when λ < M μ_a. The equal split puts λ/M on each link, so it
  needs λ/M < μ_a.
- **Brokers:** the total traffic Nλ is spread over M brokers of capacity μ_s.
  This is possible exactly when Nλ < M μ_s. The equal split puts Nλ/M on each
  broker, so it needs Nλ/M < μ_s.

These are the same two conditions: the instance is feasible exactly when the
equal split is feasible,

**λ/M < μ_a  and  Nλ/M < μ_s.**

If either condition fails, no feasible routing exists (the transportation LP
reports it infeasible), and `symmetric_solution` raises `ValueError`.

## The equal split is the unique optimum

1. **F is strictly convex.** f_a and f_s are strictly convex on [0, μ), since
   f''(x) = 2μ/(μ − x)³ > 0. Every variable λ_ij appears in its own strictly
   convex term f_a(λ_ij), and the broker terms are convex compositions with
   linear maps. So F is strictly convex on the convex feasible set, and it has
   at most one minimizer.
2. **The minimizer is symmetric.** Permuting the brokers, or the sources, maps
   feasible routings to feasible routings and leaves F unchanged. If λ* is the
   unique minimizer, every permutation of λ* is also a minimizer, so λ* equals
   all of its permutations. That forces λ*_ij to be the same for every route:
   **λ*_ij = λ/M.**
3. **It exists.** F → ∞ at the boundary of the open feasible domain, so a
   minimizer exists whenever the domain is non-empty.

## Marginal costs at the optimum

At the equal split, each route carries x = λ/M and each broker carries
Λ = Nλ/M. With the paper's M/M/1 marginal costs (eqs. 12–13):

- **Access marginal cost:** C_a = D_a + x D_a′ = μ_a / (μ_a − λ/M)²
- **Broker price:** p = C_s(Λ) = μ_s / (μ_s − Nλ/M)²
- **Total marginal cost of every route:** α = C_a + p

**Why every route has the same marginal cost:** route (i, j) costs
C_ij(λ_ij) + C_j(Λ_j). Under symmetry, every route has the same flow λ/M and
every broker the same load Nλ/M. So all N·M routes cost exactly the same α.
All routes are used (λ/M > 0), so the KKT conditions (eq. 14) require an
equal marginal cost α_i on the used routes of each source, and that holds here
with α_i = α for every source.

Other quantities at the optimum:

| Quantity | Value |
|---|---|
| Broker utilization | ρ_s = Nλ / (M μ_s) |
| Access utilization | ρ_a = λ / (M μ_a) |
| Mean delay of a work unit on any route | D = 1/(μ_a − λ/M) + 1/(μ_s − Nλ/M) |
| Objective | F* = Nλ · D |

## It is a feasible fixed point of Algorithm 1

Take the prices p_j = p (equal) at the equal split:

- **Price consistency (17)** holds by construction: p_j = C_j(Λ_j).
- **Each source best-responds (18).** A source's problem is to minimize
  Σ_j x_j/(μ_a − x_j) + p x_j subject to Σ_j x_j = λ. This is strictly convex
  and symmetric in j, so its unique solution is x_j = λ/M.

So the equal split is a fixed point of Algorithm 1's price and best-response
map. The safe step does not bind: Δ = 0 at the fixed point. By Proposition 1
the fixed point is globally optimal, which agrees with the direct argument
above.

## Does Algorithm 1 reach it? Observed behavior

Proposition 1 says that a feasible fixed point is optimal. It says nothing
about whether Algorithm 1 gets there, and on these instances the answer
depends on the step sizes. The results below are empirical.

**Starting point.** The transportation-LP initializer often lands exactly on
the equal split already, which makes convergence trivial. It did so for 1×1,
2×3, 4×4 and 3×7. For 5×3 and 10×2 it returns routings that balance the
broker loads but split each source unevenly; for example, 5×3 gets loads
(15, 15, 15) with source split (3, 5, 1).

**Paper step sizes (η = 0.25, γ = 0.5) do not converge on 5×3 or 10×2.** Take
5×3 with λ = 9, μ_a = 10, μ_s = 20, so broker utilization is 0.75:

- Starting from the LP routing, every source reacts to the same prices and
  overshoots at the same time.
- The planned loads settle into a **period-2 cycle**, alternating between
  about (16.58, 15.56, 12.87) and (12.43, 16.00, 16.56).
- The safe step never binds (s_t = 1), every iterate stays feasible (the
  highest load is 16.7 against μ_s = 20), and the certificate never passes.

**η = 0.05, γ = 0.1** converges from the LP routing on 5×3 and 10×2 in about
390 iterations. From the deliberately unbalanced start below, however, it
drifts to loads of about (6.5, 19.3, 19.3), with two brokers at 96% of
capacity. The safe step keeps them below μ − δ_s, but the routing had not
settled after 1,500 iterations.

**η = 0.01, γ = 0.02** (the notebook's load-sweep values) converged to the
certified equal split on every tested case:

| Start | Iterations |
|---|---|
| LP routing, 5×3 | about 2,300 |
| LP routing, 10×2 | about 2,300 |
| Unbalanced start, 5×3 | about 1,500 |
| Unbalanced start, 4×4 | converged |

For the unbalanced start, each source sends as much of its rate to broker 1 as
the broker and link capacities allow, and splits the rest evenly.

The takeaway: capacity safety (the safe step) and convergence are separate
properties. The safe step kept every iterate feasible in all of these runs.
Damping the updates is what determined convergence, and a damped version is
still not a proof.

## What the tests check

`tests/unit/test_symmetric.py` covers several N × M instances, from 1 × 1 to
10 × 2 and 3 × 7, and at loads up to ρ_s = 0.9. It checks that:

- the objective, prices, α and utilizations match the closed form;
- the centralized solver returns λ_ij = λ/M;
- the equal split is a certified best-response fixed point, with the same
  marginal cost α on every route;
- Algorithm 1 with η = 0.01, γ = 0.02 reaches the certified equal split from
  the LP start on every case, and from the unbalanced start on 4×4 and 5×3;
- with η = 0.25, γ = 0.5 on 5×3, Algorithm 1 enters the period-2 cycle above
  and stays feasible (a documented counterexample);
- infeasible and exactly critical parameters are rejected by both the oracle
  and the transportation LP.
