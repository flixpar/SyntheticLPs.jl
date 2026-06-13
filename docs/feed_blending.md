# Feed Blending

`FeedBlendingProblem` generates a continuous least-cost diet model that mixes ingredients to meet a required batch size, nutrient requirements, ingredient availabilities, and optional nutrient-ratio limits.

## Overview

This generator represents feed formulation or diet blending. A planner chooses how much of each ingredient to include in a fixed-size batch. Ingredients have costs and nutrient content, and the final recipe must meet minimum nutritional requirements while respecting maximum limits for restricted nutrients or anti-nutrients.

## Generator Data and Sizing

`target_variables` maps to ingredient count:

```text
num_ingredients = max(3, target_variables)
```

The number of nutrients and batch size scale by target size:

- `target_variables <= 250`: `num_nutrients` from `4:8`, batch size from `Normal(500, 200)` truncated to `[100, 2000]`
- `target_variables <= 1000`: `num_nutrients` from `6:12`, batch size from `Normal(2000, 800)` truncated to `[500, 10000]`
- larger targets: `num_nutrients` from `8:20`, batch size from `Normal(10000, 5000)` truncated to `[2000, 50000]`

Costs are lognormal and become less dispersed as ingredient count grows:

- up to 250 ingredients: `exp(Normal(log(4.0), 0.8))`
- up to 1000 ingredients: `exp(Normal(log(2.5), 0.6))`
- larger: `exp(Normal(log(1.8), 0.4))`

Each nutrient is assigned a type from `1:4`:

- Type 1 major nutrients: mostly positive `max(0, Normal(20, 7))`, with occasional high (`1.5-2.5x`) or low (`0.2-0.6x`) multipliers.
- Type 2 minor nutrients: present with probability 0.7, sampled as `max(0, Normal(2, 1))`, sometimes multiplied by `2-5x`.
- Type 3 trace nutrients: present with probability 0.3, sampled as `max(0, Normal(0.5, 0.3))`, sometimes multiplied by `3-10x`.
- Type 4 anti-nutrients or upper-limited compounds: present with probability 0.6, sampled as `max(0, Normal(5, 3))`, sometimes multiplied by `1.5-3x`.

The generator repairs all-zero nutrient rows and all-zero ingredient columns by adding positive `Normal(2, 1)` values. Ingredient availability is `Inf` unless an availability limit is drawn. The probability of a finite availability is truncated normal, increasing with size: about `0.1-0.4`, `0.15-0.45`, or `0.2-0.5`. Finite availability values are sampled as fractions of batch size from truncated normal distributions whose centers increase with scale.

The struct stores:

- `num_ingredients`, `num_nutrients`
- `batch_size`
- `costs`
- `nutrient_content::Matrix{Float64}`, indexed as nutrient by ingredient
- `nutrient_types`
- `min_requirements`, `max_limits`
- `availabilities`
- `ratio_constraints::Vector{Tuple}`, storing `(nutrient_idx, target_pct, type_string)`

The constructor uses `rng = Random.MersenneTwister(seed)` and passes it through most random calls, so it does not intentionally reset Julia's global RNG.

## LP Formulation

Sets:

- `I = {1, ..., num_ingredients}` ingredients
- `N = {1, ..., num_nutrients}` nutrients
- `R` ratio constraints

Decision variable:

- `x_i >= 0`: amount of ingredient `i` in the batch

Objective:

```math
\min \sum_{i \in I} c_i x_i
```

Fixed batch size:

```math
\sum_{i \in I} x_i = B
```

Nutrient minimums and maximums:

```math
\sum_{i \in I} a_{ji} x_i \ge r_j \quad \text{for } r_j > 0
```

```math
\sum_{i \in I} a_{ji} x_i \le u_j \quad \text{for finite } u_j
```

Finite ingredient availabilities:

```math
x_i \le A_i
```

Ratio constraints are average-content bounds. For a minimum target percentage `p`:

```math
\sum_{i \in I} (a_{ji} - p) x_i \ge 0
```

For a maximum target percentage `p`:

```math
\sum_{i \in I} (a_{ji} - p) x_i \le 0
```

The implementation chooses the ratio direction by checking whether the type string contains `"min"`. This means injected strings such as `"min_above_achievable"` are treated as minimum constraints.

## Feasibility Controls

For `feasible`, the generator first builds a base recipe `x0`. It samples a Dirichlet allocation across ingredients, clips by finite availabilities, and fills any remaining batch amount by a randomized cheap-ingredient order. If finite availability capacity is too low, it gives the cheapest ingredient enough capacity to cover the batch. Nutrient requirements are then drawn inside achievable minimum/maximum average intervals, with behavior depending on nutrient type. Afterward, the requirements are repaired so the constructed `x0` satisfies every active minimum and maximum. Ratio constraints, if generated, are also biased so `x0` satisfies them.

For `infeasible`, the constructor first skips the feasible requirement construction, generates optional random ratio constraints as it would for non-feasible cases, then injects one infeasibility mechanism:

- A minimum ratio above any ingredient's content for a nutrient with positive content.
- A minimum ratio above the availability-aware achievable maximum average, usually for a major nutrient.
- A maximum ratio below the availability-aware achievable minimum average.
- An availability shortage where total ingredient capacity is forced below batch size.

For `unknown`, nutrient requirements are generated randomly without repair guarantees. Minimum requirement factors come from truncated `Normal(0.4, 0.1)` on `[0.2, 0.6]`; maximum limit factors come from truncated `Normal(1.5, 0.2)` on `[1.1, 2.0]`. Ratio constraints are generated as in non-feasible cases. The result may be feasible or infeasible.

## Model Characteristics

The variable count is `num_ingredients`. Constraint count is driven by one batch equality, active nutrient minimums, active nutrient maximums, finite ingredient availabilities, and ratio constraints. Nutrient and ratio rows are generally dense over all ingredients, though the nutrient matrix itself can contain many zeros, especially for minor and trace nutrients. Availability constraints are sparse one-variable rows.

The model is a continuous LP. Ingredient amounts are divisible; no integer batch, package, or recipe-count restrictions are modeled.

## Practical Notes

This generator is useful for testing diet-style LPs with sparse nutrient matrices, fixed total mass, and feasibility controls based on achievable average nutrient content. Ratio constraints are stored as strings rather than a typed enum, and the model interprets any string containing `"min"` as a lower-bound ratio. Infeasible construction is stronger than random bad data because it uses achievable average calculations or total capacity shortage, but the constructor itself does not solve the instance.
