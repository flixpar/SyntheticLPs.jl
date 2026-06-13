# Production Planning

Production planning generates a continuous profit-maximization LP that chooses production quantities for products subject to shared resource capacities and optional minimum production requirements.

## Overview

This generator represents a classical production planning setting: a firm can manufacture several products, each product earns a per-unit profit, and each unit consumes limited resources such as labor, machine time, materials, or capacity. The optimization chooses how much of each product to make in order to maximize profit without exceeding available resources.

The infeasible variant adds minimum production commitments for selected products and then reduces a critical resource capacity so that those minimum commitments cannot all be met.

## Generator Data and Sizing

`target_variables` maps directly to products:

```text
n_products = max(2, min(2000, target_variables))
```

The number of resources is sampled independently:

```text
n_resources = rand(1:50)
```

Generated random data:

- `profits[i]`: integer profit for product `i`, sampled uniformly from `10:500`.
- `usage[i, j]`: resource `j` consumed per unit of product `i`, sampled uniformly from the float range `0.1:50.0`.
- `resource_factor`: sampled from `0.4:0.1:0.8`.
- `resources[j]`: set to `sum_i usage[i, j] * resource_factor`, so each resource capacity is a fixed fraction of the total consumption that would occur if every product were produced at one unit.
- `min_production[i]`: initialized to zero and only made positive by the infeasible branch.

The stored struct fields are:

- `n_products::Int`
- `n_resources::Int`
- `profits::Vector{Int}`
- `usage::Matrix{Float64}`
- `resources::Vector{Float64}`
- `min_production::Vector{Float64}`

The constructor calls `Random.seed!(seed)`, so the generator resets Julia's global RNG and is reproducible for the same arguments and package version.

## LP Formulation

Sets and indices:

- Products `i in P = {1, ..., n_products}`.
- Resources `j in R = {1, ..., n_resources}`.

Decision variables:

```text
x_i >= 0    quantity of product i to produce
```

Objective:

```math
\max \sum_{i \in P} profit_i x_i
```

Resource constraints:

```math
\sum_{i \in P} usage_{i,j} x_i \le resource_j \quad \forall j \in R
```

Minimum production constraints are added only for products with positive `min_production[i]`:

```math
x_i \ge min\_production_i
```

Bounds:

- All variables are continuous and nonnegative.
- There are no explicit upper bounds on products except those implied by resource capacities.

Interpretation: the model chooses a product mix that consumes no more than each available resource and maximizes total profit. Positive minimum production values model contractual, policy, or demand-floor commitments.

## Feasibility Controls

The constructor first sets `actual_status = feasibility_status`. If `feasibility_status == unknown`, it randomly chooses `feasible` with probability `0.7` and `infeasible` with probability `0.3`.

For `feasible`, the generator leaves `min_production` at all zeros. Because `x = 0` satisfies all resource constraints, the generated LP is feasible.

For `infeasible`, the generator:

1. Computes a per-product single-product maximum:

   ```text
   max_possible[i] = minimum(resources[j] / usage[i, j] for j in 1:n_resources)
   ```

2. Selects `n_constrained` products, where:

   ```text
   n_constrained = max(2, rand(max(1, div(n_products, 4)):max(2, div(n_products, 2))))
   ```

3. Sets each selected minimum production to `max_possible[i] * (0.3 + 0.3 * rand())`.
4. Computes the resource usage required by all minimum productions.
5. Finds the resource with the largest required-to-available ratio.
6. Reduces that critical resource to `required[critical_j] * (0.7 + 0.2 * rand())`.

This final reduction makes the lower-bound requirements exceed the capacity of at least the critical resource.

For `unknown`, the selected actual status follows the same feasible or infeasible path described above.

## Model Characteristics

Variable count:

```text
n_products
```

Constraint count drivers:

- `n_resources` capacity constraints.
- One extra lower-bound constraint for each product with `min_production[i] > 0`.

The resource matrix is dense. Every sampled `usage[i, j]` is positive because the range starts at `0.1`, so each resource constraint includes every product.

The model is a continuous LP. Product quantities are not integer-restricted, so this is a production-rate or divisible-production relaxation rather than a batch/integer production model.

## Practical Notes

This generator is useful for dense continuous LP benchmarks with direct control over the number of variables. The feasible instances are structurally simple because zero production is always feasible and all profits are positive, so the optimum is driven by resource bottlenecks rather than demand satisfaction. The infeasible instances are created by lower-bound commitments and a targeted resource shortage.

The number of resources is independent of `target_variables`, so small product sets can still receive many capacity constraints and large product sets can receive only a few.
