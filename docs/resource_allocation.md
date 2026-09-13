# Resource Allocation

Resource allocation generates a continuous profit-maximization LP that allocates limited shared resources across competing activities, with optional minimum activity levels.

## Overview

This generator models budget-style allocation: a portfolio of activities earns profit per unit and draws on a sparse set of shared resource pools — machine hours, cloud compute, staff hours, advertising budgets. The decisions are activity levels. One capacity row per pool limits aggregate consumption, and commitment floors impose minimum activity levels on a subset of activities.

Unlike `product_mix`, budgets here are *allocated*, not reserved against market demand. There are no activity ceilings at all, the number of shared pools grows with the portfolio (up to 96 resources versus product mix's 3–30), and each activity touches only a handful of pools. Capacity rows, not per-variable bounds, carry all the tension.

Capacities and commitment floors are derived from a **planted allocation plan** rather than sampled independently of each other. Independent sampling lets aggregate floor demand and aggregate capacity drift apart as the portfolio grows, which silently decides feasibility as a side effect of scale. Anchoring both sides on one nominal plan keeps them mutually consistent at every size.

The generator samples an allocation regime (manufacturing capacity, cloud compute, workforce hours, advertising budget) that shifts profit and usage scales, the profit/usage correlation, and how many pools an activity touches, plus scale-dependent parameter tiers.

## Generator Data and Sizing

`target_variables` maps directly onto activities:

```text
n_activities = max(3, min(100_000, target_variables))
```

Requests above `100_000` raise an `ArgumentError` instead of being silently undersized. The usage matrix is materialised as a dense `n_activities × n_resources` block of `Float64` (at the cap, 100_000 × up to 96 doubles is roughly 77 MB), which is what fixes the limit. The previous silent 2,000-activity cap is gone: a 5,000-variable request realizes 5,000 variables.

Resource counts and parameter distributions depend on the requested scale.

For `target_variables <= 250` (small portfolio):

- `n_resources`: `DiscreteUniform(4, 12)`.
- `usage_min` / `usage_max`: `LogNormal(log(0.8), 0.35)` / `LogNormal(log(6.0), 0.3)`.
- `profit_min` / `profit_max`: `LogNormal(log(12.0), 0.4)` / `LogNormal(log(90.0), 0.3)`.
- `uses_density`: `Beta(3, 5)`.
- `commitment_prob`: `Beta(4, 6)`.
- `correlation_strength`: `Beta(4, 3)`.
- `activity_center`: `LogNormal(log(120.0), 0.4)`.

For `250 < target_variables <= 1000` (departmental allocation):

- `n_resources`: `DiscreteUniform(10, 36)`.
- `usage_min` / `usage_max`: `LogNormal(log(0.5), 0.4)` / `LogNormal(log(4.5), 0.35)`.
- `profit_min` / `profit_max`: `LogNormal(log(8.0), 0.45)` / `LogNormal(log(70.0), 0.35)`.
- `uses_density`: `Beta(2, 5)`.
- `commitment_prob`: `Beta(5, 5)`.
- `correlation_strength`: `Beta(5, 4)`.
- `activity_center`: `LogNormal(log(80.0), 0.45)`.

For `1000 < target_variables <= 5000` (divisional allocation):

- `n_resources`: `DiscreteUniform(20, 64)`.
- `usage_min` / `usage_max`: `LogNormal(log(0.3), 0.45)` / `LogNormal(log(3.5), 0.4)`.
- `profit_min` / `profit_max`: `LogNormal(log(4.0), 0.5)` / `LogNormal(log(50.0), 0.4)`.
- `uses_density`: `Beta(2, 6)`.
- `commitment_prob`: `Beta(6, 4)`.
- `correlation_strength`: `Beta(6, 4)`.
- `activity_center`: `LogNormal(log(50.0), 0.5)`.

For `target_variables > 5000` (enterprise portfolio):

- `n_resources`: `DiscreteUniform(32, 96)`.
- `usage_min` / `usage_max`: `LogNormal(log(0.2), 0.5)` / `LogNormal(log(3.0), 0.45)`.
- `profit_min` / `profit_max`: `LogNormal(log(2.0), 0.55)` / `LogNormal(log(35.0), 0.45)`.
- `uses_density`: `Beta(2, 7)`.
- `commitment_prob`: `Beta(7, 4)`.
- `correlation_strength`: `Beta(8, 3)`.
- `activity_center`: `LogNormal(log(30.0), 0.5)`.

The generator then samples one of four allocation regimes, each stored as a `Symbol` in the `profile` field, and applies its adjustments:

```text
manufacturing_capacity  usage x1.2/x1.5 on min/max, commitment_prob x1.15
cloud_compute           profit x1.6/x2.4, usage_max x1.3, uses_density x0.7
workforce_hours         profit x0.8/x0.7, usage x1.2(min)/x0.8(max),
                        correlation x1.2, commitment_prob x1.3
advertising_budget      profit_max x1.6, usage x0.6/x0.7, uses_density x1.3
```

After adjustments, `uses_density` is clamped to `[0.05, 0.9]`, `commitment_prob` to `[0.05, 0.95]`, and `correlation_strength` to `[0.1, 0.95]`.

Generated data:

- `quality_factors[i]`: `Beta(2, 2)` per activity.
- `profits[i]`: log-normal base profit clamped to `[profit_min, profit_max]`, plus a quality-correlated component. Profits are strictly positive.
- `pool_scale[j]`: per-pool intensity, `LogNormal(log(sqrt(usage_min * usage_max)), 0.4)` clamped to `[usage_min, usage_max]` — some pools are expensive for every activity.
- `usage[i, j]`: sparse by construction. `max_uses = clamp(ceil(n_resources * uses_density), 2, n_resources)`; each activity draws its pool count from `DiscreteUniform(1, max_uses)`, picks that many pools at random, and consumes `pool_scale[j] * (0.4 + correlation_strength * quality[i]) * LogNormal(0, 0.4)` from each. Usage is therefore positively correlated with quality, just like profit: high-quality activities are both lucrative and resource-hungry, so the LP has no obvious winner.
- Repair pass: any pool no activity draws on receives one positive draw from a random activity, so there are no vacuous capacity rows. Every activity uses at least one pool by construction (this is also what keeps the profit-maximization bounded).
- `nominal_plan[i]`: the planted allocation plan, `LogNormal(log(activity_center), 0.55)` clamped to `[0.1, 10] * activity_center`.
- `consumption = usage' * nominal_plan`: what the plan actually consumes.
- `capacities[j] = consumption[j] * (1 + headroom[j])`, with `headroom[j]` from `LogNormal(log(0.2), 0.9)` clamped to `[0.02, 2.5]`. The wide spread leaves a few pools nearly saturated (binding rows at the optimum) and others with slack.
- `min_levels[i]`: with probability `clamp(0.25 + 0.7 * commitment_prob, 0.25, 0.95)`, a commitment floor of `(0.2 + 0.7 * Beta(2, 2)) * nominal_plan[i]`; otherwise `0`. The fraction is capped at `0.9`, so floors always stay strictly below the plan. At least one activity always carries a floor.

The stored struct fields are:

- `n_activities::Int`
- `n_resources::Int`
- `profits::Vector{Float64}`
- `usage::Matrix{Float64}` (`n_activities × n_resources`)
- `capacities::Vector{Float64}`
- `min_levels::Vector{Float64}`
- `nominal_plan::Vector{Float64}`
- `floor_utilization::Float64`
- `profile::Symbol`
- `feasible_witness::Union{Nothing,AllocationPlanWitness}`
- `infeasibility_certificate::Union{Nothing,FloorOvercommitCertificate}`
- `feasibility_status::FeasibilityStatus`

The constructor draws from a local `MersenneTwister(seed)`, so generation is reproducible for the same arguments and package version and never disturbs Julia's global RNG.

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

Resource constraints (only activities with a positive usage coefficient appear in a row):

```math
\sum_{i \in A} usage_{i,j} x_i \le capacity_j \quad \forall j \in R
```

Commitment-floor constraints are added only when `min_levels[i] > 0`, as explicit rows:

```math
x_i \ge min\_level_i
```

Bounds: all variables are continuous and nonnegative; there are no other variable bounds.

Interpretation: the model allocates scarce shared resources to the most profitable activity levels while honouring mandatory minimum commitments.

## Feasibility Controls

Because every usage coefficient is nonnegative and there are no activity ceilings, `x = min_levels` is the pointwise-smallest candidate point. The instance is therefore feasible **iff**

```math
floor\_utilization \;=\; \max_{j \in R} \frac{\sum_{i \in A} usage_{i,j}\, min\_level_i}{capacity_j} \;\le\; 1 .
```

That scalar is stored in the `floor_utilization` field, and the three profiles differ only in where they place it.

For `feasible`, nothing is perturbed: floors are at most `0.9 * nominal_plan`, capacities strictly exceed the plan's consumption, so `floor_utilization < 1` by construction. The plan is stored as an `AllocationPlanWitness` (`plan`, `consumption`, `slack = capacities - consumption`, all strictly positive) and is an actual feasible point of the built model.

For `infeasible`, the generator over-commits specific budget lines. It computes the consumption the floors alone impose, picks the pools that committed activities actually draw on, and cuts one to three of those pools' capacities below the floors' own demand:

```text
capacity_j = floor_consumption_j / (1 + margin),   margin ~ U(0.1, 0.4)
```

The number of violated pools is 1 (55%), 2 (27%), or 3 (18%), clamped to the number of pools with positive floor consumption. Since every `x_i >= min_levels[i]` and usage is nonnegative, resource `j`'s capacity row cannot be satisfied by any allocation — an unconditional refutation with no search or fallbacks, using only aggregate LP rows (so it survives `relax_integer=true`; the model is continuous regardless). The most violated pool and the committed activities that draw on it are stored in a `FloorOvercommitCertificate` (`resource`, `activities`, `floor_consumption`, `capacity`) with `floor_consumption > capacity`, and `floor_utilization` lands in `[1.1, 1.4]`.

For `unknown`, the generator steers the tightest floor utilization onto a target drawn as `1 ± U(0.05, 0.35)` with a fair coin for the sign — a genuine coin flip, independent of problem size. Let `gap = target / floor_utilization` for the pre-perturbation data. The adjustment is split between raising floors and cutting capacity so neither side is pushed to an unrealistic extreme:

```text
theta          = 0.35 + 0.3 * rand()
capacity_scale = clamp(gap^(theta - 1), 0.35, 3.0)
floor_scale    = gap * capacity_scale
capacities    .*= capacity_scale
min_levels    .*= floor_scale
```

Since utilization scales by `floor_scale / capacity_scale = gap` regardless of the clamps, the realized `floor_utilization` equals the target, which sits at least 5% away from 1 on whichever side the coin picked. No witness or certificate is stored.

## Model Characteristics

Variable count:

```text
n_activities
```

Constraint count drivers:

- `n_resources` capacity constraints.
- One commitment-floor constraint for each positive `min_levels[i]` (roughly 35–90% of activities depending on tier and regime).

The usage matrix is stored dense-with-zeros for build simplicity but sampled sparsely: each activity touches `DiscreteUniform(1, max_uses)` pools, so the fraction of nonzero entries is typically 8–30% (denser for small portfolios, sparser at scale). Zero coefficients are omitted from the capacity rows, which are therefore genuinely sparse.

The model is a continuous LP. Activity levels are divisible; no integer or batch restrictions.

## Practical Notes

Use this generator for budget-allocation LPs with many shared resource pools, sparse per-activity consumption, and correlated profit/usage structure. For the market-flavored problem — products with sales ceilings, fewer resources, denser consumption — use `product_mix` instead; the two share the planted-plan/utilization machinery but remain distinct generators.

The LP is always bounded: profits are strictly positive, every activity draws on at least one pool with finite capacity, and capacities are strictly positive, so no variable can grow without bound.

Observed behavior under HiGHS: `feasible` is 100% `OPTIMAL` and `infeasible` is 100% `INFEASIBLE` across targets 50–20,000 (5 seeds each; also verified through `generate_problem(...; optimizer=HiGHS.Optimizer)`, which raises on contract violation). The `unknown` profile stayed close to an even split (20 seeds each: 10/10 at 50, 12/8 at 500, 10/10 at 5,000) and the analytic `floor_utilization <= 1` test agreed with the solver on every instance. A 100,000-variable instance builds and passes solver verification in about 4 seconds.
