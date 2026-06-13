# Knapsack

The knapsack generator creates fractional continuous knapsack LPs that maximize selected item value under a capacity limit and optional minimum-value requirement.

## Overview

This generator represents selecting portions of items to fit within a weight budget. Each item has a value and weight, and the decision variable can take any value from 0 to 1. Despite the common integer interpretation of knapsack, this implementation is explicitly fractional.

## Generator Data and Sizing

`target_variables` maps directly to the item count:

```text
target_variables = n_items
```

The constructor seeds Julia's global RNG with `Random.seed!(seed)`, so item values, weights, capacity, and infeasibility threshold are reproducible for the same inputs.

Size-dependent value and weight ranges are sampled in two stages. First the low and high endpoints are drawn, then item data is sampled uniformly from the resulting integer intervals:

| Target variables | Value range endpoints | Weight range endpoints |
| --- | --- | --- |
| `<= 100` | low from `5:20`, high from `80:150` | low from `3:8`, high from `15:25` |
| `<= 1000` | low from `10:30`, high from `100:300` | low from `5:15`, high from `20:40` |
| `> 1000` | low from `20:50`, high from `200:500` | low from `10:25`, high from `30:60` |

Capacity is based on total item weight:

```text
capacity_ratio = 0.3 + rand() * 0.4
capacity = round(Int, sum(weights) * capacity_ratio)
capacity = max(1, capacity + rand(-50:50))
```

The struct stores:

- `n_items::Int`
- `capacity::Int`
- `values::Vector{Int}`
- `weights::Vector{Int}`
- `min_value::Float64`

The docstring lists the first four fields, but the struct also includes `min_value`, which is used to encode infeasible instances.

## LP Formulation

Set:

- `I = {1, ..., n_items}`

Decision variable:

```text
0 <= x[i] <= 1 = fraction of item i selected
```

Objective:

```text
maximize sum_{i in I} values[i] * x[i]
```

Capacity constraint:

```text
sum_{i in I} weights[i] * x[i] <= capacity
```

Optional minimum-value constraint:

```text
sum_{i in I} values[i] * x[i] >= min_value
```

The minimum-value constraint is only added when `min_value > 0`.

## Feasibility Controls

The constructor first samples item data and capacity. It initializes `min_value = 0.0`.

- `feasible`: leaves `min_value` at 0.0, so the all-zero selection is feasible.
- `infeasible`: computes the fractional knapsack optimum by sorting items by value-to-weight ratio and greedily filling capacity. It then sets `min_value` to 110% to 140% of that computed maximum achievable value, making the capacity and minimum-value constraints inconsistent.
- `unknown`: randomly maps to feasible with probability 0.7 or infeasible with probability 0.3, then follows the corresponding behavior.

The infeasibility control is exact for the fractional model because the greedy value-to-weight ordering solves the continuous knapsack relaxation.

## Model Characteristics

The model has `n_items` continuous variables with lower bound 0 and upper bound 1. It always has one capacity constraint and may have one minimum-value constraint. The objective and constraints are dense across items.

This is not a binary knapsack model. It models fractional item selection, so it is an LP relaxation of the common 0-1 knapsack problem. The `feasible` mode is trivially feasible because choosing no items satisfies the capacity constraint and there is no positive value requirement.

## Practical Notes

These instances are useful for testing bounded-variable LPs with dense rows and controlled infeasibility. They are not appropriate when the desired benchmark is combinatorial knapsack hardness, unless the package-level workflow separately imposes or preserves integrality. The implementation's `total_avg_weight` variable is actually the sum of sampled weights.
