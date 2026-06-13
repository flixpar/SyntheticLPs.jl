# Blending

`BlendingProblem` generates a continuous least-cost mixture model whose ingredients must satisfy quality bands, supply limits, and optional usage rules.

## Overview

This generator represents industrial blending problems such as fuel, chemical, food, or material mixing. A planner chooses nonnegative quantities of available ingredients, pays ingredient costs, and must produce at least a target blend amount while keeping the weighted-average quality attributes inside acceptable lower and upper bands.

## Generator Data and Sizing

`target_variables` maps directly to the number of ingredient variables:

```text
n_ingredients = max(3, min(500, target_variables))
```

The generator draws `n_attributes` uniformly from `2:15` and `min_blend_amount` uniformly from integer values `100:20000`, converted to `Float64`. Ingredient costs are integer values from `10:100`. Attribute values are sampled on a `0.01` grid from `0.1:0.01:0.9`, producing an `n_ingredients x n_attributes` matrix.

The struct stores:

- `n_ingredients`, `n_attributes`
- `costs::Vector{Int}`
- `attributes::Matrix{Float64}`, indexed by ingredient and attribute
- `lower_bounds`, `upper_bounds` for average quality attributes
- `supply_limits`, with `Inf` meaning no explicit limit
- `cost_budget`, with `Inf` meaning no explicit budget
- `min_blend_amount`
- `min_usage_required::Dict{Int,Float64}`
- `max_usage_limits::Dict{Int,Float64}`

The constructor calls `Random.seed!(seed)`, so it resets Julia's global RNG. With the same seed, target size, and feasibility status, the generated instance is reproducible in the same Julia/runtime environment.

## LP Formulation

Sets:

- `I = {1, ..., n_ingredients}` ingredients
- `A = {1, ..., n_attributes}` quality attributes

Decision variable:

- `x_i >= 0`: amount of ingredient `i` used in the blend

Objective:

```math
\min \sum_{i \in I} c_i x_i
```

Constraints:

Minimum production:

```math
\sum_{i \in I} x_i \ge B
```

Finite ingredient supply limits:

```math
x_i \le s_i \quad \text{for finite } s_i
```

Finite cost budget:

```math
\sum_{i \in I} c_i x_i \le C
```

Optional ingredient minimum and maximum usage rules:

```math
x_i \ge m_i
```

```math
x_i \le u_i
```

Quality bands are written as linear weighted-average constraints:

```math
\sum_{i \in I} a_{ij} x_i \ge L_j \sum_{i \in I} x_i \quad \forall j \in A
```

```math
\sum_{i \in I} a_{ij} x_i \le U_j \sum_{i \in I} x_i \quad \forall j \in A
```

The model is a continuous LP. No integer or binary variables are used.

## Feasibility Controls

The constructor translates `feasibility_status` into an internal target:

- `feasible` generates a constructed baseline blend and sets constraints around it.
- `infeasible` selects one of four infeasibility scenarios.
- `unknown` randomly chooses feasible or infeasible with probability 0.5 each.

For feasible instances, the generator builds a baseline solution. It scores ingredients by average quality divided by cost, assigns about 60 percent of ingredients as primary ingredients, allocates 80 percent of the blend amount across primaries and 20 percent across secondaries, then rescales to exactly `min_blend_amount`. Achieved average qualities from that blend are used to construct tight quality bands. The tolerance level is drawn from one of three regimes: roughly `0.025-0.05`, `0.04-0.07`, or `0.06-0.08`. Supply limits are set above the baseline amounts, with tighter slack on critical primary ingredients. The cost budget is set above actual baseline cost by about `6-25` percent. Minimum usage rules are added for about one quarter of ingredients at `70-90` percent of baseline usage, and maximum usage rules for about one third of ingredients at `120-150` percent of baseline usage.

For infeasible instances, the generator chooses one of four cases:

- Supply shortage conflict: quality lower bounds are set near the best available values, but supply for critical high-quality ingredients is limited to a small total share of the required blend.
- Budget impossibility: quality lower bounds are set near best qualities and the budget is set to only `60-90` percent of a lower-bound cost estimate.
- Impossible quality conflict: up to three attributes receive lower bounds near the maximum observed attribute value and very tight upper bounds.
- Over-constrained system: early attributes are constrained near maxima, other attributes near midpoints, expensive ingredients are tightly supply-limited, and the budget is set to about `90` percent of average-cost production.

The infeasible cases are designed to create contradictions, but they are heuristic rather than solver-certified at construction time.

## Model Characteristics

Variable count is exactly `n_ingredients`, capped at 500 and floored at 3. The base constraints include one production constraint, two quality constraints per attribute, finite supply constraints, a finite budget constraint, and optional usage bounds. Feasible instances usually have all supply limits finite, one finite budget, and many optional usage rules. The quality constraints are dense across all ingredient variables. Supply and usage constraints are one-variable sparse rows.

The model is continuous. It represents divisible ingredient amounts, not lot-sized procurement or discrete batches.

## Practical Notes

This generator is useful for testing dense weighted-average constraints, cost-budget interactions, and feasibility diagnostics in continuous mixture LPs. The quality constraints are linearized by multiplying the average bounds by total blend amount, so they remain LP constraints. `unknown` does not mean unconstrained random data; it means a random choice between the same feasible and infeasible construction paths. The constructor uses the global RNG, which can affect or be affected by surrounding randomized code.
