# Resource Allocation

Resource allocation generates a continuous profit-maximization LP that allocates limited resources among competing activities with optional minimum activity levels.

## Overview

This generator models a generic resource allocation problem: activities earn profit but consume multiple limited resources. The decision is how much of each activity to run. Examples include allocating labor, budget, machine time, or raw materials across departments, campaigns, jobs, or production modes.

The generator can construct feasible instances by building capacities around a synthetic demand plan, or construct infeasible instances by forcing minimum activity levels to exceed selected resource capacities.

## Generator Data and Sizing

`target_variables` maps directly to activities:

```text
n_activities = max(3, min(2000, target_variables))
```

Resource count and parameter ranges are sampled as:

- `n_resources`: `rand(2:50)`.
- `min_resource`: integer from `50:1000`.
- `max_resource`: integer from `200:10000`.
- `min_profit`: from `0.1:0.1:2.0`.
- `max_profit`: from `5.0:5.0:100.0`.
- `min_usage`: from `0.01:0.01:0.5`.
- `max_usage`: from `1.0:1.0:20.0`.
- `correlation_strength`: from `0.5:0.1:0.9`.
- `add_min_constraints`: true with probability `0.7`, always true for requested infeasible instances.
- `min_level_prob`: from `0.2:0.1:0.5`.

Generated data:

- `quality_factors[i]`: uniform random quality in `[0, 1)`.
- `profits[i]`: base uniform profit plus a quality-correlated profit component.
- `usage[i, j]`: dense resource usage; base uniform usage plus a quality-correlated usage component.
- `resources[j]`: generated either stochastically for `unknown` or constructively from a demand plan for requested feasible/infeasible cases.
- `min_levels[i]`: optional activity minimums, initialized to zero.

Two helper calculations are used in constructive modes:

- `compute_demand_plan(...)`: builds a normalized positive activity plan based on profit, average usage, and quality.
- `compute_single_activity_caps(...)`: computes each activity's maximum single-activity level under the current resource capacities.

The stored struct fields are:

- `n_activities::Int`
- `n_resources::Int`
- `profits::Vector{Float64}`
- `usage::Matrix{Float64}`
- `resources::Vector{Float64}`
- `min_levels::Vector{Float64}`

The constructor calls `Random.seed!(seed)`, making instances reproducible for identical inputs while resetting Julia's global RNG.

## LP Formulation

Sets and indices:

- Activities `i in A = {1, ..., n_activities}`.
- Resources `j in R = {1, ..., n_resources}`.

Decision variables:

```text
x_i >= 0    level of activity i
```

Objective:

```math
\max \sum_{i \in A} profit_i x_i
```

Resource constraints:

```math
\sum_{i \in A} usage_{i,j} x_i \le resource_j \quad \forall j \in R
```

Minimum-level constraints are added only when `min_levels[i] > 0`:

```math
x_i \ge min\_level_i
```

Bounds:

- All variables are continuous and nonnegative.
- There are no explicit upper bounds except those implied by resource capacities.

Interpretation: the model allocates scarce resources to the most profitable activity levels while satisfying any required minimum levels.

## Feasibility Controls

The constructor maps the requested status to:

```text
:feasible      if feasibility_status == feasible
:infeasible    if feasibility_status == infeasible
:all           if feasibility_status == unknown
```

For `unknown` (`:all`), the generator uses the original stochastic construction:

1. Computes expected usage per resource.
2. Sets `resources` to a random fraction of expected aggregate usage, then clamps each resource to `[min_resource, max_resource]`.
3. If minimum constraints are enabled, assigns each activity a positive minimum with probability `min_level_prob`; each minimum is `0.1`, `0.15`, `0.2`, `0.25`, or `0.3` times that activity's single-activity maximum.

This branch does not force a particular feasibility outcome.

For `feasible`, the generator:

1. Builds a demand plan from profits, usage, and quality.
2. Sets each resource to demand-plan consumption times a capacity anchor from `1.10` to `1.50`, then clamps to `[min_resource, max_resource]`.
3. Scales the demand plan to fit the post-clamp capacities.
4. Creates a baseline activity vector at roughly `75%` to `95%` of the fitted scale.
5. Optionally assigns minimum levels as `10%` to `30%` of positive baseline levels.

The positive minimum levels are therefore constructed below a known baseline that fits the resource capacities.

For `infeasible`, the generator starts similarly but with tighter capacity anchors from `0.95` to `1.10` and a baseline at roughly `70%` to `90%` of the fitted scale. It then:

1. Ensures minimum constraints are enabled and at least one or two minimum levels are selected when possible.
2. Clips existing positive minimum levels to at most `0.8` times each single-activity cap.
3. Selects one to three resources to violate, biased toward the highest current minimum-consumption ratios.
4. Sets target required consumption for those resources to `resources[r] * (1.05 to 1.25)`.
5. Raises selected minimum levels iteratively until the target violation is reached or candidates are exhausted.
6. Performs final checks and may reduce a violated resource capacity to below the final required minimum consumption.

The intended infeasibility is that the sum of resource usage implied by mandatory minimum activity levels exceeds one or more resource capacities.

## Model Characteristics

Variable count:

```text
n_activities
```

Constraint count drivers:

- `n_resources` dense resource constraints.
- One lower-bound constraint for each positive `min_levels[i]`.

The usage matrix is dense because every entry is assigned a positive continuous value. Constraint density is therefore high: every resource row involves every activity.

The model is a continuous LP. Activities are divisible, with no integer or binary restrictions.

## Practical Notes

This generator is useful for dense resource-allocation LPs where profit and resource usage are positively correlated through a quality factor. Requested feasible and infeasible modes are more constructive than the `unknown` mode, which remains stochastic.

Because both `min_resource` and `max_resource` are sampled independently, it is possible for `min_resource` to exceed `max_resource`. The code then applies `max.(resources, min_resource)` followed by `min.(resources, max_resource)` or equivalent scalar clamps, so the final cap can be driven to `max_resource` in those cases. This is an implementation quirk to keep in mind when interpreting sampled resource scales.
