# Product Mix

Product mix generates a continuous profit-maximization LP that chooses production levels across many products with resource capacities and market lower and upper bounds.

## Overview

This generator represents a product mix planning problem in which a manufacturer chooses quantities for multiple products. Products earn profit and consume resources; lower bounds model minimum market commitments, while upper bounds model market saturation or sales limits.

The generator includes scale-dependent and industry-dependent data generation. It can produce small, medium, and large operations with different resource counts, profit ranges, usage ranges, sparsity levels, and market-bound frequencies.

Capacities and market floors are derived from a **planted operating plan** rather than sampled independently of each other. Sampling them independently makes the aggregate floor consumption and the aggregate capacity drift apart as the number of products grows, which used to drive the `unknown` profile to near-certain infeasibility from roughly 500 variables upward. Anchoring both sides on one nominal plan keeps them mutually consistent at every scale.

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
- `volume_center`: `LogNormal(log(140), 0.4)`.

For `250 < target_variables <= 1000`:

- `num_resources`: selected from `5:15` using a `Beta(2, 3)` sample.
- `sparsity`: `Beta(3, 4)`.
- `profit_min`: `LogNormal(log(8), 0.5)`.
- `profit_max`: `LogNormal(log(75), 0.4)`.
- `resource_usage_min`: `LogNormal(log(0.6), 0.4)`.
- `resource_usage_max`: `LogNormal(log(4.5), 0.4)`.
- `market_constraint_prob`: `Beta(5, 5)`.
- `correlation_strength`: `Beta(6, 4)`.
- `volume_center`: `LogNormal(log(90), 0.45)`.

For `target_variables > 1000`:

- `num_resources`: `round(Int, LogNormal(log(18), 0.4))`, clamped to `8:30`.
- `sparsity`: `Beta(2, 3)`.
- `profit_min`: `LogNormal(log(3), 0.6)`.
- `profit_max`: `LogNormal(log(45), 0.5)`.
- `resource_usage_min`: `LogNormal(log(0.3), 0.5)`.
- `resource_usage_max`: `LogNormal(log(4), 0.5)`.
- `market_constraint_prob`: `Beta(6, 4)`.
- `correlation_strength`: `Beta(8, 3)`.
- `volume_center`: `LogNormal(log(50), 0.5)`.

The generator samples an industry type from:

```text
manufacturing, food_processing, electronics, furniture, chemical, automotive
```

with scale-dependent weights (stored as a `Symbol` in the `industry` field). Industry type then modifies profit ranges, resource usage ranges, sparsity, market-bound probability, correlation strength, and/or the nominal production volume. For example, electronics increases profit and usage maxima and raises sparsity, while automotive strongly increases profit and usage ranges but lowers sparsity, market-bound probability, and per-product volume.

Generated data:

- `quality_factors[j]`: `Beta(2, 2)` per product.
- `profits[j]`: log-normal base profit clamped to `[profit_min, profit_max]`, plus a quality-correlated component.
- `usage_matrix[i, j]`: resource `i` usage by product `j`. Entries are zero with probability `sparsity`; otherwise they combine a resource-level base usage, a gamma random component, and quality correlation.
- Each product is forced to use at least one resource (this keeps the maximization bounded).
- Each resource is forced to be used by at least one product.
- `nominal_plan[j]`: the planted operating plan, `LogNormal(log(volume_center), 0.55)` clamped to `[0.1, 10] * volume_center`.
- `consumption = usage_matrix * nominal_plan`: what the plan actually consumes.
- `availabilities[i] = consumption[i] * (1 + headroom[i])`, with `headroom[i]` drawn from `LogNormal(log(0.18), 0.8)` clamped to `[0.02, 2.0]`. Small headroom values give genuinely binding capacity rows.
- `lower_bounds[j]`: with probability `clamp(0.25 + 0.7 * market_constraint_prob, 0.25, 0.95)`, a committed minimum of `(0.2 + 0.7 * Beta(2, 2)) * nominal_plan[j]`; otherwise `0`. At least one product always carries a floor.
- `upper_bounds[j]`: with probability `clamp(0.5 * market_constraint_prob, 0.05, 0.8)`, a sales ceiling of `(1.05 + 1.4 * Beta(2, 2)) * nominal_plan[j]`; otherwise `Inf`.

The stored struct fields are:

- `num_products::Int`
- `num_resources::Int`
- `profits::Vector{Float64}`
- `usage_matrix::Matrix{Float64}`
- `availabilities::Vector{Float64}`
- `lower_bounds::Vector{Float64}`
- `upper_bounds::Vector{Float64}`
- `nominal_plan::Vector{Float64}`
- `floor_utilization::Float64`
- `industry::Symbol`
- `feasible_witness::Union{Nothing,ProductMixPlanWitness}`
- `infeasibility_certificate::Union{Nothing,ResourceOvercommitCertificate}`
- `feasibility_status::FeasibilityStatus`

The constructor draws from a local `MersenneTwister(seed)`, so generation is reproducible for the same arguments and package version and never disturbs Julia's global RNG.

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

Resource constraints (only products with a positive usage coefficient appear in a row):

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
- Market floors and ceilings are explicit rows, not variable bounds.

Interpretation: the model chooses a profit-maximizing production portfolio while respecting limited resources and product-level market requirements.

## Feasibility Controls

Because every usage coefficient is nonnegative, `x = lower_bounds` is the pointwise-smallest candidate point. The constructor always keeps `lower_bounds .<= upper_bounds`, so the instance is feasible **iff**

```math
floor\_utilization \;=\; \max_{i \in R} \frac{\sum_{j \in P} usage_{i,j}\, lower\_bound_j}{availability_i} \;\le\; 1 .
```

That scalar is stored in the `floor_utilization` field, and the three profiles differ only in where they place it.

For `feasible`, nothing is perturbed: floors are at most `0.9 * nominal_plan`, ceilings at least `1.05 * nominal_plan`, and capacities strictly exceed the plan's consumption, so `floor_utilization < 1`. The planted plan is stored as a `ProductMixPlanWitness` (`plan`, `consumption`, `slack = availabilities - consumption`, all strictly positive) and is an actual feasible point of the built model.

For `unknown` and `infeasible`, the generator picks a target utilization and moves the data onto it:

- `infeasible`: `target = 1.15 + 0.45 * rand()`.
- `unknown`: `target = 1 ± margin` with `margin = 0.04 + 0.31 * rand()` and a fair coin for the sign. Feasibility is therefore an explicit coin flip that is independent of problem size.

Let `gap = target / floor_utilization` for the pre-perturbation data. The adjustment is split between tightening capacity and raising commitments so neither is pushed to an extreme:

```text
theta          = 0.35 + 0.3 * rand()
capacity_scale = clamp(gap^(theta - 1), 0.35, 3.0)
floor_scale    = gap * capacity_scale
availabilities .*= capacity_scale
lower_bounds   .*= floor_scale
```

Since the ratio multiplier is `floor_scale / capacity_scale = gap`, the resulting `floor_utilization` equals `target` exactly. Any ceiling that would fall below its floor is raised to `1.05 * lower_bounds[j]`, so infeasibility never degenerates into a per-variable bound clash.

For `infeasible`, the most stressed resource and the products whose floors consume it are stored in a `ResourceOvercommitCertificate` (`resource`, `products`, `required_usage`, `availability`) with `required_usage > availability`. The refutation uses the aggregate capacity row together with the floor rows, so it holds for the LP itself.

## Model Characteristics

Variable count:

```text
num_products
```

Constraint count drivers:

- `num_resources` capacity constraints.
- One lower-bound constraint for each positive `lower_bounds[j]`.
- One upper-bound constraint for each finite `upper_bounds[j]`.

The usage matrix is sparse by construction: each entry is independently set to zero with probability `sparsity`, followed by repair passes that prevent all-zero product columns and all-zero resource rows. Zero coefficients are omitted from the capacity rows.

The model is a continuous LP. Product quantities are divisible; no integer or batch constraints are enforced.

## Practical Notes

This generator is useful for benchmarking product-mix LPs with more varied structure than the simpler production planning generator. It introduces sparse resource consumption, correlated profit/quality/usage patterns, industry-specific regimes, and both lower and upper market bounds.

Observed feasibility mix under HiGHS (30 seeds per cell): `feasible` is 100% `OPTIMAL` and `infeasible` is 100% `INFEASIBLE` at every size, while `unknown` stays close to an even split at 50, 100, 500, 1000, and 5000 variables. The analytic `floor_utilization <= 1` test agreed with HiGHS on all 450 solved instances.
