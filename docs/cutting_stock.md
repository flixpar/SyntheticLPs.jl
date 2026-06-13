# Cutting Stock

`CuttingStockProblem` generates a cutting-pattern model that minimizes the number of stock pieces used to satisfy demand for required piece lengths.

## Overview

This generator represents a one-dimensional cutting stock planning problem. A manufacturer has stock material of a standard length and must cut it into demanded piece lengths. The model chooses how many times to use each generated cutting pattern.

## Generator Data and Sizing

`target_variables` is used as a soft target for the number of generated cutting
patterns:

```text
max_patterns = target_variables
```

The actual number of variables in the built model is `length(patterns)`. It can
be less than `target_variables` if duplicate complex patterns are rejected or
the pattern generator exhausts attempts. It can also exceed `target_variables`
when mandatory single-piece patterns are added for more piece types than the
requested pattern target. Piece-type counts and distributions scale by target
size:

- `target_variables <= 250`: `n_piece_types` from `3:min(15, max(3, target_variables / 10))`, stock length from `Uniform(3, 8)`, demand range from random `5:20` to random `50:200`, common-length probability `0.3-0.6`, waste factor `0.05-0.15`.
- `target_variables <= 1000`: `n_piece_types` from `8:min(50, max(8, target_variables / 20))`, stock length from `Uniform(6, 12)`, demand range from random `20:100` to random `200:1000`, common-length probability `0.4-0.7`, waste factor `0.03-0.10`.
- larger targets: `n_piece_types` from `20:min(200, max(20, target_variables / 50))`, stock length from `Uniform(8, 20)`, demand range from random `100:500` to random `1000:10000`, common-length probability `0.5-0.8`, waste factor `0.02-0.08`.

Piece lengths are either near common sizes, with `Normal(0, 0.02)` variation, or sampled from a transformed `Beta(2, 3)` distribution between `0.1` and about `95` percent of stock length. Lengths are rounded to precision `0.05` for smaller stock and `0.1` for stock over 10. Duplicate lengths are removed, so actual piece-type count may shrink.

Demands are sampled from lognormal distributions. Common lengths use `LogNormal(log((demand_min + demand_max) / 1.3), 0.5)`; other lengths use `LogNormal(log((demand_min + demand_max) / 2.0), 0.7)`. Demands are rounded to coarser increments as they grow and clamped to the selected range.

The struct stores:

- `piece_lengths`
- `demands`
- `patterns`, where `patterns[p][i]` is the count of piece type `i` produced by pattern `p`
- `stock_length`
- `stock_limit`, where `0` means unlimited stock pieces

The constructor calls `Random.seed!(seed)`, resetting Julia's global RNG.

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

Optional stock limit:

```math
\sum_{p \in P} x_p \le S \quad \text{if } S > 0
```

Bounds are nonnegativity only. Although cutting stock is naturally integer, `x_p` is continuous in the implemented model, so this is the LP relaxation of the pattern-count problem.

## Feasibility Controls

The constructor maps the requested status to a boolean target:

- `feasible`: target feasible.
- `infeasible`: target infeasible.
- `unknown`: randomly chooses feasible or infeasible with probability 0.5 each.

For feasible targets, the generator creates patterns using `generate_cutting_patterns`, copies base demands, perturbs each demand by `Uniform(0.8, 1.2)`, and sets `stock_limit = 0` for unlimited stock. Pattern generation always starts with single-piece patterns for each piece type, so every piece type has at least one direct production route as long as each length fits in stock.

For infeasible targets, one of three methods is selected:

- Single piece impossibility: picks the smallest piece type, estimates its best pattern efficiency, sets a finite stock limit, and raises that piece's demand above maximum possible production under the stock limit.
- No-pattern scenario: removes every pattern containing one target piece type while keeping positive demand for it, with unlimited stock. That piece cannot be produced at all.
- Combined stock and demand contradiction: generates normal patterns, estimates a lower bound on stock needed, sets stock limit to `40-70` percent of that bound, and scales demands upward.

Infeasible demand scaling is influenced by scenario labels such as rush order, seasonal spike, backlog clearing, and mixed, with scaling factors generally between about `0.8` and `4.0` depending on length commonality and scenario.

## Model Characteristics

Variable count is `length(patterns)`, not necessarily exactly `target_variables`. Constraint count is one row per piece type plus one stock-limit row when `stock_limit > 0`. The pattern matrix is sparse because each pattern uses only a subset of piece types. Single-piece patterns are very sparse; generated complex patterns have a small random number of selected piece types, biased toward shorter pieces.

The model is continuous even though the real cutting decision is integer. Fractional pattern usage is allowed by the built JuMP model.

## Practical Notes

This generator is useful for testing column-style covering LPs, sparse nonnegative matrices, and infeasibility from missing coverage or aggregate resource limits. It does not generate patterns by solving a pricing problem; it samples a fixed pattern list up front. Because duplicate pattern rejection and unique piece lengths can shrink dimensions, expect actual model size to differ from the target, especially for small targets or repeated common lengths.
