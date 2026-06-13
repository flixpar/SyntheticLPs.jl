# Product Mix

Product mix generates a continuous profit-maximization LP that chooses production levels across many products with resource capacities and optional market lower and upper bounds.

## Overview

This generator represents a product mix planning problem in which a manufacturer chooses quantities for multiple products. Products earn profit and consume resources; optional lower bounds model minimum market commitments, while optional upper bounds model market saturation or sales limits.

The generator includes scale-dependent and industry-dependent data generation. It can produce small, medium, and large operations with different resource counts, profit ranges, usage ranges, sparsity levels, and market-bound frequencies.

## Generator Data and Sizing

`target_variables` maps directly to products:

```text
num_products = max(2, min(10000, target_variables))
```

Resource count and parameter distributions depend on the requested scale.

For `target_variables <= 250`:

- `num_resources`: `DiscreteUniform(3, 8)`.
- `sparsity`: `Beta(2, 6)`.
- `profit_min`: `LogNormal(log(15), 0.4)`.
- `profit_max`: `LogNormal(log(120), 0.3)`.
- `resource_usage_min`: `LogNormal(log(1.0), 0.3)`.
- `resource_usage_max`: `LogNormal(log(5), 0.3)`.
- `market_constraint_prob`: `Beta(4, 6)`.
- `correlation_strength`: `Beta(4, 3)`.

For `250 < target_variables <= 1000`:

- `num_resources`: selected from `5:15` using a `Beta(2, 3)` sample.
- `sparsity`: `Beta(3, 4)`.
- `profit_min`: `LogNormal(log(8), 0.5)`.
- `profit_max`: `LogNormal(log(75), 0.4)`.
- `resource_usage_min`: `LogNormal(log(0.6), 0.4)`.
- `resource_usage_max`: `LogNormal(log(4.5), 0.4)`.
- `market_constraint_prob`: `Beta(5, 5)`.
- `correlation_strength`: `Beta(6, 4)`.

For `target_variables > 1000`:

- `num_resources`: `round(Int, LogNormal(log(18), 0.4))`, clamped to `8:30`.
- `sparsity`: `Beta(2, 3)`.
- `profit_min`: `LogNormal(log(3), 0.6)`.
- `profit_max`: `LogNormal(log(45), 0.5)`.
- `resource_usage_min`: `LogNormal(log(0.3), 0.5)`.
- `resource_usage_max`: `LogNormal(log(4), 0.5)`.
- `market_constraint_prob`: `Beta(6, 4)`.
- `correlation_strength`: `Beta(8, 3)`.

The generator samples an industry type from:

```text
manufacturing, food_processing, electronics, furniture, chemical, automotive
```

with scale-dependent weights. Industry type then modifies profit ranges, resource usage ranges, sparsity, market-bound probability, and/or correlation strength. For example, electronics increases profit and usage maxima and raises sparsity, while automotive strongly increases profit and usage ranges but lowers sparsity and market-bound probability.

Generated data:

- `quality_factors[j]`: `Beta(2, 2)` per product.
- `profits[j]`: log-normal base profit clamped to `[profit_min, profit_max]`, plus a quality-correlated component.
- `usage_matrix[i, j]`: resource `i` usage by product `j`. Entries are zero with probability `sparsity`; otherwise they combine a resource-level base usage, a gamma random component, and quality correlation.
- Each product is forced to use at least one resource.
- Each resource is forced to be used by at least one product.
- `availabilities[i]`: based on average usage per resource times `num_products / 2`, multiplied by log-normal variability clamped to `[0.5, 2.0]` and a factor in roughly `[0.6, 1.2]`.
- `lower_bounds[j]`: optional market floor, initially zero.
- `upper_bounds[j]`: optional market cap, initially `Inf`.

Market bounds are generated from each product's single-product `max_possible[j]`, computed as the minimum availability-to-usage ratio across resources with positive usage. Each product independently receives a lower bound with probability `market_constraint_prob / 2` and an upper bound with the same probability.

The stored struct fields are:

- `num_products::Int`
- `num_resources::Int`
- `profits::Vector{Float64}`
- `usage_matrix::Matrix{Float64}`
- `availabilities::Vector{Float64}`
- `lower_bounds::Vector{Float64}`
- `upper_bounds::Vector{Float64}`

The constructor calls `Random.seed!(seed)`, so generation is reproducible for the same arguments and package version, while also resetting Julia's global RNG.

## LP Formulation

Sets and indices:

- Products `j in P = {1, ..., num_products}`.
- Resources `i in R = {1, ..., num_resources}`.

Decision variables:

```text
x_j >= 0    quantity of product j to produce
```

Objective:

```math
\max \sum_{j \in P} profit_j x_j
```

Resource constraints:

```math
\sum_{j \in P} usage_{i,j} x_j \le availability_i \quad \forall i \in R
```

Market lower-bound constraints are added only when `lower_bounds[j] > 0`:

```math
x_j \ge lower\_bound_j
```

Market upper-bound constraints are added only when `upper_bounds[j] < Inf`:

```math
x_j \le upper\_bound_j
```

Bounds:

- All variables are continuous and nonnegative.
- Upper bounds are optional and product-specific.

Interpretation: the model chooses a profit-maximizing production portfolio while respecting limited resources and product-level market requirements.

## Feasibility Controls

The constructor converts the requested status into:

```text
:feasible      if feasibility_status == feasible
:infeasible    if feasibility_status == infeasible
:all           if feasibility_status == unknown
```

For `feasible`, the generator:

1. Repairs any product where `lower_bounds[j] > upper_bounds[j]` by setting the lower bound to `0.98 * upper_bounds[j]`.
2. Computes aggregate resource usage required by the lower bounds.
3. If any resource requirement exceeds availability, scales all lower bounds down by `minimum(availability / required) * 0.98`.
4. Rechecks lower bounds against finite upper bounds.

This makes the lower-bound point feasible with respect to resource capacities and finite product caps.

For `infeasible`, the generator:

1. Ensures there are positive lower bounds. If none exist, it selects products using a heavily used resource and assigns lower bounds equal to `(0.15 + 0.25 * rand()) * max_possible[j]`.
2. Computes required resource usage from lower bounds.
3. Chooses the resource with the largest required-to-available ratio, or a heavily used fallback resource if requirements are zero.
4. Reduces that resource availability to `required[critical_i] * (1 - shortage_margin)`, where `shortage_margin` is sampled from `0.10` to `0.35`.
5. With probability `0.3`, also reduces another resource to slightly below its required lower-bound usage.

For `unknown`, no repair or forced infeasibility branch is applied after the initial stochastic construction. The instance may be feasible or infeasible depending on the sampled bounds and capacities.

## Model Characteristics

Variable count:

```text
num_products
```

Constraint count drivers:

- `num_resources` capacity constraints.
- One lower-bound constraint for each positive `lower_bounds[j]`.
- One upper-bound constraint for each finite `upper_bounds[j]`.

The usage matrix is sparse by construction: each entry is independently set to zero with probability `sparsity`, followed by repair passes that prevent all-zero product columns and all-zero resource rows.

The model is a continuous LP. Product quantities are divisible; no integer or batch constraints are enforced.

## Practical Notes

This generator is useful for benchmarking product-mix LPs with more varied structure than the simpler production planning generator. It introduces sparse resource consumption, correlated profit/quality/usage patterns, industry-specific regimes, and both lower and upper market bounds.

For `unknown`, the code deliberately leaves the sampled instance uncorrected, so the status is not a randomized 70/30 choice as in some other generators. The imported `LinearAlgebra` module is not used by this file.
