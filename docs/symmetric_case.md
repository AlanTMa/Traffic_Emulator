# The symmetric N × M case

A closed-form optimum to test against (`src/model/symmetric.py`,
`tests/unit/test_symmetric.py`).

N sources each offering λ, M brokers each of rate μ_s, every access link of
rate μ_a. A routing needs λ_ij < μ_a and Λ_j < μ_s. The equal split puts λ/M
on each link and Nλ/M on each broker, so it is feasible iff

    λ/M < μ_a  and  Nλ/M < μ_s,

and if the equal split is infeasible, so is everything else (each source
must send λ over M links of capacity μ_a; the total Nλ must fit in M brokers
of capacity μ_s). `symmetric_solution` raises `ValueError` then, and the
transportation LP reports infeasibility.

## The equal split is the unique optimum

f(x) = x/(μ − x) has f'' = 2μ/(μ − x)³ > 0 on [0, μ), so F is strictly convex
on the convex feasible set and has at most one minimizer. Permuting sources
or brokers leaves F and the feasible set unchanged, so the minimizer equals
all of its permutations: λ*_ij = λ/M. It exists because F → ∞ at the
boundary.

At the equal split, with x = λ/M and Λ = Nλ/M:

    C_a = μ_a/(μ_a − x)²        access marginal cost
    p   = μ_s/(μ_s − Λ)²        broker price
    α   = C_a + p               the same on every route
    D   = 1/(μ_a − x) + 1/(μ_s − Λ)
    F*  = Nλ · D
    ρ_s = Nλ/(Mμ_s),  ρ_a = λ/(Mμ_a)

Every route is used and costs α, so the KKT conditions hold with α_i = α.

It is also a fixed point of Algorithm 1: price consistency holds by
construction, and each source's problem min Σ_j x_j/(μ_a − x_j) + p x_j
s.t. Σ x_j = λ is strictly convex and symmetric in j, so its unique solution
is x_j = λ/M. Proposition 1 then says it is optimal, which agrees with the
direct argument.

## Whether Algorithm 1 gets there

Proposition 1 says nothing about that, and on these instances it depends on
the step sizes. Observed:

- The LP initializer often returns the equal split already (1×1, 2×3, 4×4,
  3×7). For 5×3 and 10×2 it balances the broker loads but splits each source
  unevenly, e.g. loads (15, 15, 15) with a source split (3, 5, 1).
- 5×3 with λ = 9, μ_a = 10, μ_s = 20 (ρ_s = 0.75) at the paper's η = 0.25,
  γ = 0.5: every source reacts to the same prices at once and overshoots.
  The loads settle into a period-2 cycle between about (16.6, 15.6, 12.9)
  and (12.4, 16.0, 16.6). The safe step never binds, every iterate is
  feasible, and the certificate never passes.
- η = 0.05, γ = 0.1 converges from the LP routing on 5×3 and 10×2 in about
  390 iterations, but from an unbalanced start (each source sends as much as
  it can to broker 1) it drifts to loads of about (6.5, 19.3, 19.3) and has
  not settled after 1,500 iterations.
- η = 0.01, γ = 0.02 (the notebook's load-sweep values) reaches the
  certified equal split on every tested case: about 2,300 iterations from
  the LP routing on 5×3 and 10×2, about 1,500 from the unbalanced start.

Capacity safety held in all of these runs; convergence came down to the
damping. Linearizing the iteration at the fixed point explains the cycle:
with ν = N·C_s'(Λ)/C_a'(x) (how strongly all sources together react to a
price change, times how steeply the price reacts to load), the fixed point
is stable iff ν < (2−η)(2−γ)/(ηγ). For the 5×3 case above ν = 27.4; the
threshold is 21 at the paper's steps and 741 at η = 0.05, γ = 0.1. This is a
linearization, not a convergence proof.
