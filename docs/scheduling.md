# Scheduling

Scheduling generates a binary workforce scheduling model that assigns workers to shifts while minimizing cost and satisfying staffing, availability, workload, and consecutive-day constraints.

## Overview

This generator represents workforce scheduling for settings such as stores, restaurants, clinics, hospitals, call centers, airlines, and manufacturing operations. It assigns workers to shifts across a multi-day horizon, respecting worker availability, at most one shift per day, minimum and maximum total shifts, and a maximum consecutive working-days rule.

The generator can optionally create skill-based availability: workers have skills, shifts require skills, and availability is removed when a worker lacks a required skill. The build model itself uses the resulting availability matrix; it does not add separate skill constraints.

## Generator Data and Sizing

`target_variables` is intended to approximate `n_workers * n_shifts * n_days`, but the constructor samples dimensions by scale rather than solving exactly for the requested count.

For `target_variables <= 250`:

- `n_workers`: rounded `Uniform(4, 12)`.
- `n_shifts`: rounded `Uniform(2, 4)`.
- `n_days`: rounded `Uniform(3, 7)`.
- `min_staffing`: rounded `Uniform(1, 2)`.
- `max_staffing`: `min_staffing + rounded Uniform(1, 3)`.
- `availability_density`: `Beta(8, 2)`.
- `min_worker_shifts`: rounded `Uniform(2, 4)`.
- `max_worker_shifts`: `min_worker_shifts + rounded Uniform(1, 3)`.
- `max_consecutive_shifts`: rounded `Uniform(2, 3)`.
- `min_cost`: `Uniform(15, 25)`.
- `max_cost`: `min_cost + Uniform(20, 40)`.
- `skill_based`: true with probability `0.2`.

For `250 < target_variables <= 1000`:

- `n_workers`: rounded `Uniform(8, 25)`.
- `n_shifts`: rounded `Uniform(3, 6)`.
- `n_days`: rounded `Uniform(5, 14)`.
- `min_staffing`: rounded `Uniform(2, 4)`.
- `max_staffing`: `min_staffing + rounded Uniform(2, 5)`.
- `availability_density`: `Beta(6, 3)`.
- `min_worker_shifts`: rounded `Uniform(3, 5)`.
- `max_worker_shifts`: `min_worker_shifts + rounded Uniform(2, 4)`.
- `max_consecutive_shifts`: rounded `Uniform(2, 4)`.
- `min_cost`: `Uniform(20, 35)`.
- `max_cost`: `min_cost + Uniform(25, 60)`.
- `skill_based`: true with probability `0.4`.

For `target_variables > 1000`:

- `n_workers`: rounded `Uniform(25, 80)`.
- `n_shifts`: rounded `Uniform(4, 8)`.
- `n_days`: rounded `Uniform(7, 30)`.
- `min_staffing`: rounded `Uniform(3, 6)`.
- `max_staffing`: `min_staffing + rounded Uniform(3, 8)`.
- `availability_density`: `Beta(4, 3)`.
- `min_worker_shifts`: rounded `Uniform(4, 6)`.
- `max_worker_shifts`: `min_worker_shifts + rounded Uniform(2, 5)`.
- `max_consecutive_shifts`: rounded `Uniform(3, 5)`.
- `min_cost`: `Uniform(25, 50)`.
- `max_cost`: `min_cost + Uniform(30, 100)`.
- `skill_based`: true with probability `0.6`.

If skill-based scheduling is enabled:

- `n_skills` is sampled as `2:3`, `3:5`, or `4:8` depending on scale.
- Each worker receives between one and `min(3, n_skills)` skills.
- Each shift requires one skill.
- Worker-shift availability is zeroed out when the worker does not have any required skill for that shift.

Generated data:

- `total_shifts = n_shifts * n_days`.
- `staffing_req[s]`: generated from a Poisson mean affected by shift-in-day peak factor and weekend factor, then clamped to `[min_staffing, max_staffing]`.
- `availability[w, s]`: binary availability generated from full-time or part-time worker patterns, shift timing, and weekend adjustments.
- `costs[w, s]`: generated from worker tier (`junior`, `regular`, `senior`), optional skill premium, shift timing premiums, weekend/month-end premiums, and log-normal noise.

The stored struct fields are:

- `n_workers::Int`
- `n_shifts::Int`
- `n_days::Int`
- `total_shifts::Int`
- `staffing_req::Vector{Int}`
- `availability::Matrix{Int}`
- `costs::Matrix{Float64}`
- `min_worker_shifts::Int`
- `max_worker_shifts::Int`
- `max_consecutive_shifts::Int`
- `skill_based::Bool`
- `worker_skills::Union{Matrix{Int}, Nothing}`
- `shift_skill_req::Union{Matrix{Int}, Nothing}`

The constructor calls `Random.seed!(seed)`, making generated instances reproducible for the same inputs while resetting Julia's global RNG.

## LP Formulation

Sets and indices:

- Workers `w in W = {1, ..., n_workers}`.
- Shifts `s in S = {1, ..., total_shifts}`.
- Days `d in D = {1, ..., n_days}`.
- `S_d`: shifts belonging to day `d`.

Decision variables:

```text
x_{w,s} in {0, 1}    1 if worker w is assigned to shift s, 0 otherwise
```

Objective:

```math
\min \sum_{w \in W} \sum_{s \in S} cost_{w,s} x_{w,s}
```

Staffing requirements:

```math
\sum_{w \in W} x_{w,s} \ge staffing\_req_s \quad \forall s \in S
```

Availability constraints are added for unavailable worker-shift pairs:

```math
x_{w,s} = 0 \quad \text{if availability}_{w,s} = 0
```

At most one shift per worker per day:

```math
\sum_{s \in S_d} x_{w,s} \le 1 \quad \forall w \in W, d \in D
```

Minimum and maximum shifts per worker:

```math
\sum_{s \in S} x_{w,s} \ge min\_worker\_shifts \quad \forall w \in W
```

```math
\sum_{s \in S} x_{w,s} \le max\_worker\_shifts \quad \forall w \in W
```

Maximum consecutive working days:

If `max_consecutive_shifts >= 1` and `n_days > max_consecutive_shifts`, the model creates windows of length `max_consecutive_shifts + 1` and enforces:

```math
\sum_{s \in window} x_{w,s} \le max\_consecutive\_shifts
```

for each worker and each rolling day window. Since there is also at most one shift per day, this prevents working more than `max_consecutive_shifts` days in any such window.

Interpretation: the model finds the minimum-cost assignment of available workers to required shifts while respecting worker workload rules.

## Feasibility Controls

For `feasible`, the generator applies a constructive feasibility process:

1. Converts shift availability into day availability per worker.
2. Computes each worker's maximum assignable days under the consecutive-day rule.
3. Reduces `min_worker_shifts` to at most the minimum worker capacity and keeps `max_worker_shifts` consistent.
4. Caps per-shift staffing by available worker counts with beta-distributed slack.
5. Caps per-day total demand by available day workers with reserve.
6. Caps global demand by total worker capacity with reserve.
7. Uses randomized greedy matching to assign workers to shifts while respecting availability, at most one shift per day, maximum shifts, and consecutive-day windows.
8. Reduces unmet staffing requirements to the covered amounts.
9. Tries to bring each worker up to `min_worker_shifts`, freeing demand slots when possible. If a worker cannot be brought up to the minimum, the global minimum is lowered to that worker's assigned count.

The final `staffing_req` is aligned to a constructed assignment, and worker minimums are adjusted if necessary.

For `infeasible`, the generator randomly chooses one of three modes:

- `shift_blackout`: selects one or two high-demand shifts, ensures each has positive demand, and sets all worker availability for those shifts to zero.
- `day_overload`: picks a day and raises total demand beyond the number of workers available on that day.
- `min_over_cap`: computes each worker's capacity under availability, `max_worker_shifts`, and consecutive-day rules, then sets `min_worker_shifts` above the minimum worker capacity.

For `unknown`, no special feasibility or infeasibility branch is run. The initially sampled staffing, availability, costs, and worker limits are used as-is, so feasibility depends on the random draw.

## Model Characteristics

Variable count:

```text
n_workers * total_shifts
```

Constraint count drivers:

- `total_shifts` staffing constraints.
- One equality constraint for each unavailable worker-shift pair.
- `n_workers * n_days` at-most-one-shift-per-day constraints.
- `2 * n_workers` minimum/maximum workload constraints.
- `n_workers * (n_days - max_consecutive_shifts)` rolling-window constraints when the consecutive-day rule applies.

Availability constraints can dominate the constraint count when availability is sparse, especially after skill filtering.

This is a binary integer model, not a continuous LP. A continuous relaxation would use `0 <= x_{w,s} <= 1`, but the implementation declares `Bin`.

## Practical Notes

This generator is useful for workforce assignment and scheduling benchmarks with rich combinatorial structure. The requested feasible path is unusually involved and actively modifies staffing requirements and minimum shifts to match a constructed schedule.

The field name `max_consecutive_shifts` is implemented as a maximum number of consecutive working days, not consecutive shifts. Skill requirements are enforced indirectly by modifying `availability`; the model builder does not use `worker_skills` or `shift_skill_req` after construction.
