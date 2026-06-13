# Transportation

The transportation generator creates continuous minimum-cost shipping LPs that move goods from supply sources to demand destinations.

## Overview

This generator represents a logistics planning problem: each source has a finite supply of a homogeneous good, each destination has a demand requirement, and the model chooses shipment quantities on all source-destination lanes to minimize total transportation cost. It is the classic balanced-or-unbalanced transportation structure, but the implementation uses `<=` supply constraints and `>=` demand constraints, so feasible instances may leave unused supply.

## Generator Data and Sizing

`target_variables` maps to the lane count:

```text
target_variables ~= n_sources * n_destinations
```

The constructor seeds Julia's global RNG with `Random.seed!(seed)`, so the same inputs reproduce the same dimensions, ranges, and data.

Dimensions are chosen by starting near `sqrt(target_variables)`, multiplying the source side by a random ratio in `[0.5, 1.5)`, and setting destinations from `target_variables / n_sources`. Both dimensions are at least 2. A final adjustment tries to keep the product within about 10% of the target.

Generated ranges depend on the final lane count:

| Lane count | Supply range endpoints | Demand range endpoints | Cost range endpoints |
| --- | --- | --- | --- |
| `<= 250` | low from `50:100`, high from `200:500` | low from `30:80`, high from `150:300` | low from `5:15`, high from `25:60` |
| `<= 1000` | low from `100:500`, high from `1000:5000` | low from `80:300`, high from `800:3000` | low from `10:30`, high from `50:150` |
| `> 1000` | low from `500:2000`, high from `5000:50000` | low from `300:1500`, high from `3000:30000` | low from `20:100`, high from `100:500` |

After each range is sampled, supplies, demands, and costs are drawn uniformly from the resulting integer intervals. The struct stores:

- `n_sources::Int`
- `n_destinations::Int`
- `supplies::Vector{Int}`
- `demands::Vector{Int}`
- `costs::Matrix{Int}`

## LP Formulation

Sets:

- `I = {1, ..., n_sources}` for sources
- `J = {1, ..., n_destinations}` for destinations

Decision variable:

```text
x[i,j] >= 0 = amount shipped from source i to destination j
```

Objective:

```text
minimize sum_{i in I, j in J} costs[i,j] * x[i,j]
```

Constraints:

```text
sum_{j in J} x[i,j] <= supplies[i]    for each source i
sum_{i in I} x[i,j] >= demands[j]     for each destination j
```

The source constraints limit outbound shipments by available supply. The destination constraints require every destination's demand to be met or exceeded. There are no upper bounds on individual lanes beyond the source totals.

## Feasibility Controls

The constructor first samples supplies and demands, then compares total supply and total demand.

- `feasible`: if `sum(supplies) < sum(demands)`, it adds the shortage across the supply vector using random weights and integer rounding. This guarantees `total_supply >= total_demand`, which is sufficient for this complete bipartite transportation model.
- `infeasible`: it creates an aggregate shortage by ensuring `total_demand > total_supply` with a random margin equal to 2% to 10% of total supply. If demand is not already high enough, it distributes the missing amount across demands.
- `unknown`: leaves the sampled supplies and demands unchanged.

The helper `distribute_additions!` uses random weights, floors proportional additions, and distributes any integer remainder over a random permutation of entries.

## Model Characteristics

The model has `n_sources * n_destinations` continuous nonnegative variables. It has `n_sources + n_destinations` structural constraints. The coefficient matrix is sparse in the usual transportation sense: each variable appears in exactly one source constraint and one destination constraint, plus the objective.

There are no integer declarations. Although transportation problems often have integral extreme points when supplies and demands are integral, this implementation builds the continuous LP directly.

## Practical Notes

These instances are useful for testing LP solvers on highly structured network-like matrices with predictable row and column sparsity. The generator uses a complete lane set, so it does not model missing routes, lane capacities, or transshipment. Feasibility is controlled only through aggregate supply and demand, which is enough because every source can ship to every destination.
