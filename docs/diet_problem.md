# Diet Problem

The diet problem generator creates continuous minimum-cost food selection LPs with nutrient requirements, supply limits, budget limits, and optional food-specific consumption bounds.

## Overview

This generator represents nutrition planning under market and dietary restrictions. Foods have costs and nutrient contents; the model chooses nonnegative consumption quantities that meet nutrient minimums while minimizing total cost. Generated instances may also include food supply ceilings, an overall cost budget, minimum consumption requirements for preferred foods, and maximum consumption limits for restricted foods.

## Generator Data and Sizing

`target_variables` maps directly to the number of foods:

```text
target_variables = n_foods
```

The constructor seeds Julia's global RNG with `Random.seed!(seed)`, so foods, nutrients, constraints, and infeasibility scenarios are reproducible for the same inputs.

The number of nutrients and data ranges scale with target size:

| Target variables | Nutrients | Cost endpoint ranges | Nutrient endpoint ranges |
| --- | --- | --- | --- |
| `<= 100` | `5:min(25, max(5, target_variables / 4))` | low from `0.5:0.1:2.0`, high from `3.0:0.5:8.0` | low from `0.05:0.01:0.15`, high from `1.5:0.1:3.0` |
| `<= 1000` | `15:min(75, max(15, target_variables / 8))` | low from `0.1:0.05:1.0`, high from `2.0:0.5:10.0` | low from `0.01:0.005:0.1`, high from `1.0:0.2:4.0` |
| `> 1000` | `25:min(150, max(25, target_variables / 15))` | low from `0.05:0.01:0.5`, high from `1.0:0.2:15.0` | low from `0.005:0.001:0.05`, high from `0.5:0.1:5.0` |

Costs are sampled as `rand(min_cost:0.1:max_cost, n_foods)`. Nutrient contents are sampled as `rand(min_nutrient:0.1:max_nutrient, n_foods, n_nutrients)`. Requirements start at zero, supply limits and cost budget start at `Inf`, and food-specific min/max dictionaries start empty before feasibility logic fills them.

The struct stores:

- `n_foods::Int`
- `n_nutrients::Int`
- `costs::Vector{Float64}`
- `nutrient_content::Matrix{Float64}`
- `requirements::Vector{Float64}`
- `food_supply_limits::Vector{Float64}`
- `cost_budget::Float64`
- `min_food_amounts::Dict{Int, Float64}`
- `max_food_amounts::Dict{Int, Float64}`

## LP Formulation

Sets:

- `F = {1, ..., n_foods}` for foods
- `N = {1, ..., n_nutrients}` for nutrients

Decision variable:

```text
x[i] >= 0 = amount of food i consumed
```

Objective:

```text
minimize sum_{i in F} costs[i] * x[i]
```

Nutrient requirements:

```text
sum_{i in F} nutrient_content[i,j] * x[i] >= requirements[j]
    for each nutrient j
```

Supply limits, when finite:

```text
x[i] <= food_supply_limits[i]
```

Budget limit, when finite:

```text
sum_{i in F} costs[i] * x[i] <= cost_budget
```

Food-specific consumption restrictions:

```text
x[i] >= min_food_amounts[i]    for listed foods
x[i] <= max_food_amounts[i]    for listed foods
```

All variables are continuous and nonnegative.

## Feasibility Controls

`unknown` is first mapped randomly to `feasible` with probability 0.75 or `infeasible` with probability 0.25. The selected actual status controls the rest of generation.

### Feasible

The feasible path constructs a baseline diet rather than solving a verification LP.

1. It scores foods by average nutrient content divided by cost.
2. It allocates 75% of a 100-unit baseline diet across the top 60% of foods by cost-effectiveness, weighted by effectiveness, and spreads the remaining 25% across the rest.
3. It computes achieved nutrient levels from that baseline diet.
4. It sets each nutrient requirement below the baseline achievement using a tolerance scenario: 2% to 5%, 5% to 10%, or 8% to 12%.
5. It creates finite supply limits under one of three scenarios: seasonal availability, market supply, or normal supply. Limits are multiples of baseline amounts, so the baseline remains intended to fit.
6. It sets a finite cost budget based on baseline cost: tight 105% to 115%, moderate 110% to 125%, or generous 150% to 200%.
7. With probability 0.7, it adds minimum amounts for about one-sixth of foods and maximum amounts for about one-fifth of foods, both derived from baseline amounts.

### Infeasible

The infeasible path chooses one of four scenarios and then performs a final verification-style strengthening step.

- Nutrient impossibility: gives foods finite supplies, computes maximum achievable nutrients under those supplies, and sets one nutrient requirement above its maximum.
- Budget impossibility: sets broad food supplies, creates nutrient requirements, estimates a lower bound on cost required to satisfy them, and sets the budget below that estimate.
- Supply shortage: sets requirements and supplies, then reduces supplies for foods contributing to a target nutrient until that nutrient cannot be met.
- Over-constrained system: builds a baseline diet, tightens requirements, supply, budget, minimum consumption of expensive foods, and maximum consumption of nutritious foods, then forces one nutrient above its achievable maximum under the resulting bounds.

After any infeasible scenario, the constructor computes, for each nutrient, the maximum possible amount under food supply and food-specific maximum constraints. It then forces one target nutrient requirement to 200% to 300% of that maximum, or to `100.0` to `200.0` if the maximum is zero. This final step is the strongest infeasibility guarantee in the implementation.

## Model Characteristics

The model has `n_foods` continuous nonnegative variables. It always has `n_nutrients` nutrient constraints. Additional constraints are added for each finite supply limit, for a finite budget, for each minimum food amount, and for each maximum food amount.

Nutrient rows and the budget row are dense across foods. Supply and food-specific bound constraints are one-variable rows. The formulation is a continuous LP; it does not model integer servings or discrete package counts.

## Practical Notes

These instances are useful for testing dense covering-style constraints combined with simple bound rows and an optional budget cap. Feasible instances are constructed around a baseline diet and are meant to be challenging but feasible; infeasible instances include explicit maximum-achievable nutrient checks. One implementation detail to note is that generated nutrient contents use a `0.1` step even for small endpoint ranges, so some scale settings may produce coarser nutrient values than the endpoint precision suggests.
