# Project Selection

Project selection generates a binary portfolio-optimization model that selects projects to maximize return subject to budget, risk, dependency, and high-risk-count constraints.

## Overview

This generator represents capital budgeting or project portfolio selection. Each project has a cost, expected return, risk score, and possible dependencies on other projects. The optimizer chooses which projects to fund.

Although this package is oriented around LP generation, this specific model is a mixed-integer/binary optimization problem because each project is either selected or not selected.

## Generator Data and Sizing

`target_variables` maps directly to projects:

```text
n_projects = target_variables
projects = collect(1:n_projects)
```

There is no lower or upper clamp in this generator.

Scale-dependent parameters are sampled as follows.

For `target_variables <= 250`:

- Cost scale around `log(100_000)` with standard deviation `1.5`.
- `min_cost`: at least `5_000`, rounded from a log-normal sample.
- `max_cost`: at most `500_000`, rounded from a log-normal sample.
- `return_multiplier`: `Uniform(1.5, 4.0)`.
- `min_return`: at least `10_000`, based on `min_cost * Uniform(0.8, 1.2)`.
- `max_return`: at most `1_000_000`, based on `max_cost * return_multiplier`.
- `budget_factor`: `Beta(2, 3) * 0.4 + 0.2`.
- `max_risk_score`: `Uniform(8.0, 12.0)`.
- `dependency_density`: `Beta(2, 8) * 0.1 + 0.05`.

For `250 < target_variables <= 1000`:

- Cost scale around `log(500_000)` with standard deviation `1.8`.
- `min_cost`: at least `10_000`.
- `max_cost`: at most `5_000_000`.
- `return_multiplier`: `Gamma(2, 2) + 2.0`.
- `min_return`: at least `50_000`, based on `min_cost * Uniform(0.7, 1.3)`.
- `max_return`: at most `10_000_000`, based on `max_cost * return_multiplier`.
- `budget_factor`: `Beta(3, 4) * 0.35 + 0.15`.
- `max_risk_score`: `Uniform(12.0, 18.0)`.
- `dependency_density`: `Beta(3, 7) * 0.15 + 0.1`.

For `target_variables > 1000`:

- Cost scale around `log(2_000_000)` with standard deviation `2.0`.
- `min_cost`: at least `50_000`.
- `max_cost`: at most `50_000_000`.
- `return_multiplier`: `Gamma(1.5, 2.5) + 1.5`.
- `min_return`: at least `100_000`, based on `min_cost * Uniform(0.6, 1.4)`.
- `max_return`: at most `100_000_000`, based on `max_cost * return_multiplier`.
- `budget_factor`: `Beta(4, 6) * 0.3 + 0.1`.
- `max_risk_score`: `Uniform(15.0, 25.0)`.
- `dependency_density`: `Beta(2, 5) * 0.15 + 0.15`.

If sampled minimums exceed maximums, the generator repairs them:

- `min_cost = max_cost * 0.3` if `min_cost >= max_cost`.
- `min_return = max_return * 0.4` if `min_return >= max_return`.

Additional generated data:

- `risk_budget = n_projects * Uniform(0.8, 2.5)`.
- `max_high_risk_projects = max(1, ceil(Int, n_projects * high_risk_fraction))`, where `high_risk_fraction = Beta(2, 5) * 0.2 + 0.1`.
- `high_risk_threshold = max_risk_score * Uniform(0.6, 0.8)`.
- Project category is sampled from low-, medium-, and high-risk/return categories with weights `0.4`, `0.4`, and `0.2`.
- `costs[p]` are log-normal-like values between `min_cost` and `max_cost`.
- `returns[p]` are correlated with cost through a sampled ROI and category noise, then clamped to `[min_return, max_return]`.
- `risk_scores[p]` are beta-distributed based on return percentile, with normal noise, and clamped to `[1.0, max_risk_score]`.
- `dependencies` are generated among projects sorted by cost. A dependency `(p1, p2)` means project `p1` depends on project `p2`.
- `budget = sum(costs) * budget_factor`.
- `min_selected` is normally zero and is only set by the infeasible branch.

The stored struct fields are:

- `n_projects::Int`
- `projects::Vector{Int}`
- `costs::Dict{Int,Float64}`
- `returns::Dict{Int,Float64}`
- `risk_scores::Dict{Int,Float64}`
- `dependencies::Vector{Tuple{Int,Int}}`
- `budget::Float64`
- `risk_budget::Float64`
- `max_high_risk_projects::Int`
- `high_risk_threshold::Float64`
- `min_selected::Int`

The constructor calls `Random.seed!(seed)`, so generation is reproducible for identical inputs and resets Julia's global RNG.

## LP Formulation

Sets and indices:

- Projects `p in P`.
- Dependencies `(p1, p2) in D`, where `p1` requires `p2`.
- High-risk projects `H = {p in P : risk_score_p > high_risk_threshold}`.

Decision variables:

```text
x_p in {0, 1}    1 if project p is selected, 0 otherwise
```

Objective:

```math
\max \sum_{p \in P} return_p x_p
```

Budget constraint:

```math
\sum_{p \in P} cost_p x_p \le budget
```

Risk budget constraint:

```math
\sum_{p \in P} risk_p x_p \le risk\_budget
```

Dependency constraints:

```math
x_{p1} \le x_{p2} \quad \forall (p1, p2) \in D
```

High-risk project count, added only if `H` is nonempty:

```math
\sum_{p \in H} x_p \le max\_high\_risk\_projects
```

Minimum selection constraint, added only if `min_selected > 0`:

```math
\sum_{p \in P} x_p \ge min\_selected
```

Interpretation: the model chooses the best-return portfolio that fits aggregate budget and risk limits, respects prerequisite projects, and caps exposure to high-risk projects.

## Feasibility Controls

The constructor sets `actual_status = feasibility_status`. If `feasibility_status == unknown`, it randomly chooses `feasible` with probability `0.7` and `infeasible` with probability `0.3`.

For `feasible`, no extra feasibility repair is applied. Since `min_selected = 0`, the all-zero portfolio is feasible: it satisfies budget, risk, dependency, and high-risk constraints.

For `infeasible`, the generator:

1. Sorts projects by increasing cost.
2. Counts how many cheapest projects can fit within the budget.
3. Sets `min_selected` to more than that affordable count:

   ```text
   min_selected = affordable + max(1, rand(1:max(1, div(n_projects, 4))))
   min_selected = min(min_selected, n_projects)
   ```

The intended infeasibility is that the model requires selecting more projects than the budget can support, even when choosing the cheapest available projects.

For `unknown`, the sampled `actual_status` follows the same feasible or infeasible path above.

## Model Characteristics

Variable count:

```text
n_projects
```

Constraint count drivers:

- One budget constraint.
- One risk budget constraint.
- One constraint per generated dependency.
- Optional one high-risk-count constraint.
- Optional one minimum-selected constraint.

Dependency density grows from pairwise checks over projects sorted by cost, so large instances can generate many dependency constraints.

This is not a continuous LP as built. The JuMP model declares binary variables with `Bin`. A continuous relaxation would replace `x_p in {0, 1}` with `0 <= x_p <= 1`, but the implementation does not do that.

## Practical Notes

This generator is useful for binary portfolio benchmarks with correlated cost, return, risk, and prerequisite structure. It is also a useful caveat case in an LP-focused collection because the generated model is a MIP, not a pure LP.

The constructor does not clamp `target_variables`; if `target_variables` is zero or negative, `projects` can be empty or malformed for downstream assumptions. Normal use should pass a positive project count.
