# Cutting Stock

`CuttingStockProblem` (`cutting_stock/standard`) generates a cutting-pattern
model that minimizes the number of stock pieces used to satisfy demand for
required piece lengths, with certificate-backed control of the requested
feasibility status.

## Overview

This generator represents a one-dimensional cutting stock planning problem. A
manufacturer has stock material of a standard length and must cut it into
demanded piece lengths. The model chooses how many times to use each generated
cutting pattern. Pattern usage is continuous, so the model is the LP relaxation
of the pattern-count problem; the category's `integer_patterns` variant covers
the integral formulation.

## Generator Data and Sizing

`target_variables` is the exact number of generated cutting patterns, and
therefore the exact variable count:

```text
n_patterns = target_variables
```

Pattern generation has three stages: one single-piece pattern per piece type
(emitted first, in piece order), a greedy residual-fill sampler for mixed
patterns, and a deterministic enumeration of sub-maximal single-type and
two-type patterns that tops the list up to `n_patterns` even if the greedy
stage stalls. Duplicate patterns are rejected, and the generator errors rather
than returning fewer patterns than requested.

Piece-type counts and distributions scale by target size:

- `target_variables <= 250`: `n_piece_types` from `3:min(15, max(3, target_variables / 10))`, stock length from `Uniform(3, 8)`, demand range from random `5:20` to random `50:200`, common-length probability `0.3-0.6`, waste factor `0.05-0.15`.
- `target_variables <= 1000`: `n_piece_types` from `8:min(50, max(8, target_variables / 20))`, stock length from `Uniform(6, 12)`, demand range from random `20:100` to random `200:1000`, common-length probability `0.4-0.7`, waste factor `0.03-0.10`.
- larger targets: `n_piece_types` from `20:min(200, max(20, target_variables / 50))`, stock length from `Uniform(8, 20)`, demand range from random `100:500` to random `1000:10000`, common-length probability `0.5-0.8`, waste factor `0.02-0.08`.

The sampled count is then raised to at least `clamp(round(target_variables / 4), 20, 200)` so the distinct-pattern pool can reach the target, and capped at
`target_variables` so every piece type can own a single-piece pattern (a piece
type without one could leave its demand row structurally uncoverable). Duplicate
lengths are removed, so the actual piece-type count may shrink slightly.

Piece lengths are either near common catalog sizes, with `Normal(0, 0.02)` variation around a drawn catalog value, or sampled from a transformed `Beta(2, 3)` distribution between `0.1` and about 95 percent of stock length. Lengths are rounded to two digits (catalog) or to precision `0.05` for stock up to 10 and `0.1` above. Which lengths are catalog sizes is tracked as a flag, because the jittered, rounded lengths almost never equal a catalog value exactly.

Base demands are lognormal with parameters quantile-matched to the drawn range:
`[demand_min, demand_max]` is a roughly two-sigma band (`sigma = log(demand_max / demand_min) / 4`, median `sqrt(demand_min * demand_max)`), so the range clamp only trims the outer few percent of draws instead of piling mass on the minimum. Catalog lengths are ordered at 1.25x the volume with three-quarters of the spread; demands are rounded to coarser increments as they grow.

The struct stores:

- `stock_length`
- `piece_lengths`
- `demands`, all positive
- `patterns`, where `patterns[p][i]` is the count of piece type `i` produced by pattern `p`; entries `1:length(piece_lengths)` are the single-piece patterns in piece order
- `stock_limit`, a finite positive cap on total pattern usage (there is no unlimited mode)
- `scenario`, the demand regime the instance narrates
- `feasible_witness`, set for `feasible` requests
- `infeasibility_certificate`, set for `infeasible` requests
- `feasibility_status`, the requested profile

All randomness lives in the constructor behind a constructor-local
`MersenneTwister(seed)`; the caller's global RNG is never read or advanced.

## LP Formulation

Sets:

- `P = {1, ..., number of patterns}` cutting patterns
- `I = {1, ..., number of piece types}` required piece lengths

Decision variable:

- `x_p >= 0`: number of times pattern `p` is used

Objective:

```math
\min \sum_{p \in P} x_p
```

Demand satisfaction:

```math
\sum_{p \in P} a_{pi} x_p \ge d_i \quad \forall i \in I
```

Stock limit (always present, since `stock_limit >= 1`):

```math
\sum_{p \in P} x_p \le S
```

Bounds are nonnegativity only. Although cutting stock is naturally integer, `x_p` is continuous in the implemented model, so this is the LP relaxation of the pattern-count problem.

## Feasibility Controls

Feasibility is decided by the stock limit relative to two elementary bounds
computed from the final pattern list. Let `s_i = floor(L / ℓ_i)` be the yield
of piece `i`'s single-piece pattern and `e_i = max_p a_{pi} >= s_i >= 1` its
best yield over all patterns. Then:

- Any `x >= 0` produces at most `e_i * sum(x)` units of piece `i`, so
  `d_i > S * e_i` for any single `i` proves infeasibility.
- Running each single-piece pattern `cld(d_i, s_i)` times meets every demand
  using `U = sum_i cld(d_i, s_i)` stock pieces, so `S >= U` proves
  feasibility.

The three profiles place `S` relative to `U`:

- `feasible`: `S = U * Uniform(1.3, 1.8)`. The budget is generous but real —
  any plan restricted to single-piece patterns must respect it (the planted
  plan uses 55-77 percent of it) — while the trivial plan itself is stored as
  a `StockPlanWitness` whose integer arithmetic (`s_i * usage[i] >= d_i`,
  `sum(usage) <= S`) needs no tolerances.
- `infeasible`: demands are first scaled by scenario flavor (rush order,
  seasonal spike, backlog clearing, or mixed, with rush orders concentrated on
  catalog lengths), then `S = U * Uniform(0.30, 0.55)`. The piece type with
  the largest demand-per-yield ratio is the bottleneck; its demand is raised,
  if the scenario draw did not already do so, to `ceil(margin * S * e_i)` with
  `margin` in `1.2-1.5`, and the result is stored as a
  `StockShortageCertificate(piece_index, max_yield_per_stock, stock_limit, demand)`
  with the invariant `demand > stock_limit * max_yield_per_stock`. The
  constructor verifies the invariant on the returned fields; the refutation is
  a two-row Farkas argument on the demand and stock rows alone, so it survives
  `relax_integer = true`.
- `unknown`: `S = U * exp(Normal(-0.15, 0.45))`, log-centered slightly below
  the direct plan's need. Whether the sampled mixed patterns close the gap
  depends on the draw; across seeds the profile lands near an even
  OPTIMAL/INFEASIBLE split at every scale. Neither a witness nor a
  certificate is stored.

Unlike earlier versions of this generator, there is no "no-pattern" mode: an
instance whose demand row has no production route at all is trivially
detectable and useless for solver testing, and every infeasible instance is
now built around the stored certificate.

## Model Characteristics

Variable count is exactly `target_variables`. Constraint count is one row per
piece type plus one stock-limit row. The pattern matrix is sparse because each
mixed pattern uses only a subset of piece types, biased toward shorter pieces;
the leading single-piece patterns are very sparse. Coefficients are small
nonnegative integers and the right-hand sides are moderate integers, so the
instances are numerically tame.

## Practical Notes

This generator is useful for testing column-style covering LPs, sparse
nonnegative matrices, and infeasibility that requires combining a covering row
with an aggregate budget row. It does not generate patterns by solving a
pricing problem; it samples a fixed pattern list up front. The `unknown`
profile genuinely depends on how well the sampled patterns pack, rather than
on a hidden coin flip between two committed constructions.

One caveat for dataset work: the infeasible instances are *correct but
presolve-friendly*. Because the certificate is a two-row argument with integer
coefficients, HiGHS detects the contradiction in a handful of simplex
iterations, so `check_quality` rejects them under the default
`min_iterations = 3` when `quality_filter = true` (the same holds, and
strictly worse, for an aggregate all-pieces certificate — scaling every demand
uniformly makes the contradiction even easier to aggregate). The `feasible` and
`unknown` profiles pass the quality filter normally; pair `infeasible`
requests with `quality_filter = false`, or accept that the filter drops them.
