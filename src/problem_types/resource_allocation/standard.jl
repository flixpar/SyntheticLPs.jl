using JuMP
using Random
using Distributions

"""
Planted allocation plan: one effort level per model column (aligned with
[`_resource_allocation_columns`](@ref)), the per-pool, per-period hours it
books, and the output it delivers per activity. For the `feasible` profile this
point satisfies every row of the built model with strictly positive slack on
every pool-capacity row.
"""
struct ResourceAllocationPlanWitness
    allocation::Vector{Float64}
    pool_hours::Matrix{Float64}
    output::Vector{Float64}
end

"""
Department over-commitment certificate. Every activity in `activities` is
committed to deliver at least `floors[a]` units of output and can only draw on
pools of department `department` (`pools`; department `0` stands for the whole
organisation, the fallback when no department has two local activities); one hour of pool `p` delivers at
most `max_efficiency[k]` units for the `k`-th listed activity. Dividing each
commitment row by that best efficiency and summing them shows the activities
need at least `required_hours = Σ floors[a] / max_efficiency[k]` hours from the
department, while summing the department's pool-capacity rows over every period
caps the hours it can supply at `available_hours < required_hours`. The
refutation combines LP rows only (one commitment row per listed activity plus
`|pools| × n_periods` capacity rows), so it survives every transform and no
single row exposes it to presolve.
"""
struct DepartmentOvercommitCertificate
    department::Int
    pools::Vector{Int}
    activities::Vector{Int}
    max_efficiency::Vector{Float64}
    required_hours::Float64
    available_hours::Float64
end

"""
    ResourceAllocationProblem <: ProblemGenerator

Generator for multi-period allocation of skilled resource pools to a portfolio
of activities.

# Overview

An organisation runs a portfolio of activities (engineering projects, cloud
workloads, maintenance work orders) over `n_periods` planning periods. Capacity
comes from resource pools (teams, clusters, crews) grouped into departments;
each pool has a period-dependent number of available hours. Each activity has a
release/deadline window, a home department, and a small set of *eligible* pools
— mostly in its home department, sometimes one cross-trained or contracted
pool elsewhere — each with its own efficiency (output per hour). The decision
`y[a, p, t] >= 0` is the number of hours pool `p` spends on activity `a` in
period `t`, one column per eligible (activity, pool, in-window period) triple.

Rows:

  - pool capacity, per pool and period: `Σ_a y[a,p,t] <= capacity[p,t]`;
  - activity scope, per activity: `floor[a] <= Σ_{p,t} eff[a,p] y[a,p,t] <= workload[a]`
    — a ranged row for committed activities, a plain `<=` otherwise;
  - absorption rate, per activity and in-window period:
    `Σ_p eff[a,p] y[a,p,t] <= rate_cap[a]` (team-size / ramp-up limits;
    emitted as a variable bound when the activity has a single eligible pool).

The objective maximizes delivered value minus pool cost: output in period `t`
is worth `value[a] * discount^(t - release[a])` (earlier delivery is worth
more), and an hour of pool `p` in period `t` costs `pool_cost[p] *
period_cost_factor[t]` (seasonal overtime premiums).

Every column touches three rows with heterogeneous efficiencies, pools are
shared across many activities and periods, and total capacity is well below
total workload, so pool rows, scope rows, and rate rows all bind somewhere —
the LP cannot be decided column-by-column, which is what kept the previous
single-period knapsack formulation collapsing to an empty model under presolve.

# Feasibility control

A nominal plan is planted first (each activity delivers a fraction of its
workload, spread over its window and eligible pools), and capacities are drawn
around the hours it books (`max(plan hours × (1 + headroom), base level ×
availability)`), so the plan is feasible for the `feasible` profile with slack
on every pool row. Commitment floors are fractions of each activity's planned
output.

  - `feasible`: the plan is stored as a [`ResourceAllocationPlanWitness`](@ref).
  - `infeasible`: one department's pool capacities are cut (a team loses staff)
    so that the hours its *department-local* committed activities need at
    their best efficiency exceed the department's total hours by a 12–35%
    margin, certified by a [`DepartmentOvercommitCertificate`](@ref). The
    contradiction needs the sum of many rows, so presolve cannot see it.
  - `unknown`: the same department ratio `required / available` is steered to
    `1 ± U(0.03, 0.30)`. Above 1 the instance is provably infeasible; below 1
    windows, eligibility, and rate caps decide — a genuine two-sided instance
    with no stored witness or certificate.

# Fields

  - `n_periods::Int`, `n_pools::Int`, `n_departments::Int`, `n_activities::Int`
  - `pool_department::Vector{Int}`: department of each pool
  - `pool_cost::Vector{Float64}`: cost per hour of each pool
  - `period_cost_factor::Vector{Float64}`: per-period multiplier on pool cost
  - `capacity::Matrix{Float64}`: available hours, `n_pools × n_periods`
  - `activity_department::Vector{Int}`: home department of each activity
  - `release::Vector{Int}`, `deadline::Vector{Int}`: activity windows (inclusive)
  - `eligible_pools::Vector{Vector{Int}}`: eligible pools per activity (sorted)
  - `efficiency::Vector{Vector{Float64}}`: output per hour, aligned with `eligible_pools`
  - `value::Vector{Float64}`: value per unit of output delivered at release
  - `discount::Float64`: per-period decay of output value
  - `workload::Vector{Float64}`: total output each activity can absorb
  - `rate_cap::Vector{Float64}`: maximum output per period
  - `floors::Vector{Float64}`: committed minimum output (0 if uncommitted)
  - `profile::Symbol`: sampled organisational regime
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct ResourceAllocationProblem <: ProblemGenerator
    n_periods::Int
    n_pools::Int
    n_departments::Int
    n_activities::Int
    pool_department::Vector{Int}
    pool_cost::Vector{Float64}
    period_cost_factor::Vector{Float64}
    capacity::Matrix{Float64}
    activity_department::Vector{Int}
    release::Vector{Int}
    deadline::Vector{Int}
    eligible_pools::Vector{Vector{Int}}
    efficiency::Vector{Vector{Float64}}
    value::Vector{Float64}
    discount::Float64
    workload::Vector{Float64}
    rate_cap::Vector{Float64}
    floors::Vector{Float64}
    profile::Symbol
    feasible_witness::Union{Nothing, ResourceAllocationPlanWitness}
    infeasibility_certificate::Union{Nothing, DepartmentOvercommitCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _resource_allocation_columns(prob) -> (act, pool, period)

The model's column order: activities in index order, then eligible pools in
stored order, then in-window periods ascending. Witness allocations are aligned
with it.
"""
function _resource_allocation_columns(prob::ResourceAllocationProblem)
    act, pool, period = Int[], Int[], Int[]
    for a in 1:prob.n_activities, p in prob.eligible_pools[a], t in prob.release[a]:prob.deadline[a]
        push!(act, a)
        push!(pool, p)
        push!(period, t)
    end
    return act, pool, period
end

"""
    _resource_allocation_profile(rng, profile) -> NamedTuple

Regime parameters: period count range, eligible-pool count range, efficiency
spread, cross-department probability, commitment probability, activities per
pool-period, and value/cost scale.
"""
function _resource_allocation_profile(rng::AbstractRNG, profile::Symbol, target::Int)
    # Longer horizons at larger scale: a 100k-column portfolio is planned over
    # more periods, not just more activities.
    scale_periods = target <= 2_000 ? (4, 8) : (target <= 20_000 ? (6, 13) : (8, 20))
    if profile == :engineering_portfolio
        # Quarters/months; teams with strong skill differences.
        return (
            periods=scale_periods,
            eligible=(2, 5),
            eff_sigma=0.35,
            cross_prob=0.3,
            commit_prob=0.35 + 0.3 * rand(rng),
            per_pool=rand(rng, 6.0:1.0:14.0),
            cost_mu=log(95.0),
            value_ratio=2.2,
        )
    elseif profile == :cloud_capacity
        # Clusters serving workloads; near-interchangeable hardware.
        return (
            periods=(scale_periods[1] + 2, scale_periods[2] + 4),
            eligible=(3, 6),
            eff_sigma=0.18,
            cross_prob=0.45,
            commit_prob=0.25 + 0.3 * rand(rng),
            per_pool=rand(rng, 10.0:1.0:20.0),
            cost_mu=log(3.5),
            value_ratio=1.7,
        )
    else  # :maintenance_crews
        # Weekly crew planning against work orders; craft specialisation.
        return (
            periods=scale_periods,
            eligible=(2, 4),
            eff_sigma=0.28,
            cross_prob=0.2,
            commit_prob=0.45 + 0.3 * rand(rng),
            per_pool=rand(rng, 5.0:1.0:12.0),
            cost_mu=log(60.0),
            value_ratio=2.0,
        )
    end
end

"""
    ResourceAllocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-period resource allocation instance with exactly
`max(target_variables, 4)` columns.
"""
function ResourceAllocationProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    rng = MersenneTwister(seed)
    target = max(target_variables, 4)

    profile = rand(rng, (:engineering_portfolio, :cloud_capacity, :maintenance_crews))
    prm = _resource_allocation_profile(rng, profile, target)

    T = rand(rng, prm.periods[1]:prm.periods[2])
    # Pools sized so each (pool, period) row is shared by ~`per_pool` activities.
    n_pools = max(3, round(Int, target / (T * prm.per_pool)))
    pools_per_dept = rand(rng, 3:7)
    n_departments = max(1, round(Int, n_pools / pools_per_dept))
    # Departments get contiguous pool blocks; every department has >= 1 pool.
    pool_department = [min(n_departments, 1 + (p - 1) * n_departments ÷ n_pools) for p in 1:n_pools]
    dept_pools = [findall(==(d), pool_department) for d in 1:n_departments]

    # Pool economics: cost per hour (contractor-like pools cost more) and a
    # quality level that scales every efficiency drawn on the pool.
    pool_cost = rand(rng, LogNormal(prm.cost_mu, 0.3), n_pools)
    pool_quality = rand(rng, LogNormal(0.0, 0.15), n_pools)
    # Seasonal overtime premium on cost, smooth over the horizon.
    phase = 2π * rand(rng)
    period_cost_factor = [1.0 + 0.12 * sin(2π * t / max(T, 2) + phase) for t in 1:T]

    # --- Activities, generated until the column budget is exactly used -----
    activity_department = Int[]
    release = Int[]
    deadline = Int[]
    eligible_pools = Vector{Int}[]
    efficiency = Vector{Float64}[]
    remaining = target
    while remaining > 0
        d = rand(rng, 1:n_departments)
        home = dept_pools[d]
        k = min(rand(rng, prm.eligible[1]:prm.eligible[2]), length(home))
        chosen = shuffle(rng, home)[1:k]
        if n_departments > 1 && rand(rng) < prm.cross_prob
            other = rand(rng, setdiff(1:n_departments, d))
            push!(chosen, rand(rng, dept_pools[other]))
        end
        # Window length: most activities span a fraction of the horizon.
        len = clamp(round(Int, T * rand(rng, Beta(2.5, 2.0))), 1, T)
        # Never overshoot the column budget: shrink the window, then the pool set.
        if length(chosen) * len > remaining
            len = remaining ÷ length(chosen)
            if len == 0
                chosen = chosen[1:remaining]
                len = 1
            end
        end
        r = rand(rng, 1:(T - len + 1))
        perm = sortperm(chosen)
        chosen = chosen[perm]
        eff = [
            pool_quality[p] * (pool_department[p] == d ? 1.0 : 0.65) *
            rand(rng, LogNormal(0.0, prm.eff_sigma)) for p in chosen
        ]
        push!(activity_department, d)
        push!(release, r)
        push!(deadline, r + len - 1)
        push!(eligible_pools, chosen)
        push!(efficiency, eff)
        remaining -= length(chosen) * len
    end
    n_activities = length(release)

    # --- Planted plan ---------------------------------------------------------
    # Each activity delivers a fraction of its workload, at a per-period output
    # level below its rate cap, split across eligible pools with random shares.
    workload = zeros(n_activities)
    rate_cap = zeros(n_activities)
    value = zeros(n_activities)
    planned_output = zeros(n_activities)
    pool_hours = zeros(n_pools, T)
    plan_hours = Vector{Vector{Float64}}(undef, n_activities)  # per eligible pool, per period
    mean_cost = exp(prm.cost_mu)
    for a in 1:n_activities
        len = deadline[a] - release[a] + 1
        per_period = rand(rng, LogNormal(log(40.0), 0.6))
        fraction = 0.35 + 0.5 * rand(rng)          # share of the workload planned
        planned_output[a] = per_period * len
        workload[a] = planned_output[a] / fraction
        rate_cap[a] = per_period * (1.15 + 0.85 * rand(rng))
        # Value per unit of output: above the cost of an average hour for most
        # activities (profitable work), with a heavy right tail.
        value[a] = mean_cost * prm.value_ratio * rand(rng, LogNormal(0.0, 0.45))
        shares = rand(rng, Dirichlet(length(eligible_pools[a]), 1.5))
        hours = shares .* per_period ./ efficiency[a]
        plan_hours[a] = hours
        for (k, p) in enumerate(eligible_pools[a]), t in release[a]:deadline[a]
            pool_hours[p, t] += hours[k]
        end
    end
    discount = 1.0 - (0.005 + 0.025 * rand(rng))

    # Capacities: around the planned hours, never below them (plus headroom),
    # with a flat base level and availability dips (holidays, outages).
    capacity = zeros(n_pools, T)
    for p in 1:n_pools
        headroom = 0.05 + 0.3 * rand(rng)
        base = (sum(pool_hours[p, :]) / T) * (0.9 + 0.35 * rand(rng))
        for t in 1:T
            availability = rand(rng) < 0.1 ? 0.6 + 0.3 * rand(rng) : 0.95 + 0.1 * rand(rng)
            capacity[p, t] = max(pool_hours[p, t] * (1 + headroom), base * availability, 1.0)
        end
    end

    # Commitment floors: fractions of the planned output.
    floors = zeros(n_activities)
    for a in 1:n_activities
        if rand(rng) < prm.commit_prob
            floors[a] = (0.3 + 0.6 * rand(rng)) * planned_output[a]
        end
    end

    # --- Feasibility profile ---------------------------------------------------
    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        allocation = Float64[]
        for a in 1:n_activities, (k, _) in enumerate(eligible_pools[a]), _ in release[a]:deadline[a]
            push!(allocation, plan_hours[a][k])
        end
        feasible_witness = ResourceAllocationPlanWitness(
            allocation, copy(pool_hours), copy(planned_output)
        )
    else
        # Candidate groups: each department with its department-local
        # activities (every eligible pool inside it), plus group 0 — the whole
        # organisation, every pool and every activity — as a fallback that
        # always exists.
        home_of(a) = pool_department[eligible_pools[a][1]]
        is_local(a) = all(pool_department[p] == home_of(a) for p in eligible_pools[a])
        group_pools(d) = d == 0 ? collect(1:n_pools) : dept_pools[d]
        group_acts(d) = d == 0 ? collect(1:n_activities) :
            [a for a in 1:n_activities if is_local(a) && home_of(a) == d]
        hours_needed(acts) = sum(
            (floors[a] / maximum(efficiency[a]) for a in acts if floors[a] > 0); init=0.0
        )
        hours_available(pools) = sum(capacity[p, t] for p in pools, t in 1:T)

        # The department whose local commitments weigh most on its hours.
        best, best_ratio = 0, -1.0
        for d in 1:n_departments
            acts = group_acts(d)
            length(acts) >= 2 || continue
            r = hours_needed(acts) / hours_available(group_pools(d))
            if r > best_ratio
                best, best_ratio = d, r
            end
        end
        pools_g, acts_g = group_pools(best), group_acts(best)

        # The department's whole local portfolio is put under contract.
        for a in acts_g
            if floors[a] == 0.0
                floors[a] = (0.3 + 0.6 * rand(rng)) * planned_output[a]
            end
        end
        ratio = if feasibility_status == infeasible
            1.12 + 0.23 * rand(rng)
        else
            margin = 0.03 + 0.27 * rand(rng)
            rand(rng) < 0.5 ? 1.0 - margin : 1.0 + margin
        end

        # Split the push between larger commitments and fewer hours (a team
        # losing staff): floors grow by part of the gap, each staying strictly
        # below its own workload and cumulative rate cap so no single
        # activity's rows contradict on their own; capacity absorbs the rest.
        gap = ratio * hours_available(pools_g) / hours_needed(acts_g)
        if gap > 1.0
            grow = gap^(0.3 + 0.3 * rand(rng))
            for a in acts_g
                len = deadline[a] - release[a] + 1
                ceiling = 0.95 * min(workload[a], rate_cap[a] * len)
                floors[a] = min(floors[a] * grow, max(floors[a], ceiling))
            end
        end
        required = hours_needed(acts_g)
        scale = required / (ratio * hours_available(pools_g))
        for p in pools_g, t in 1:T
            capacity[p, t] *= scale
        end
        if feasibility_status == infeasible
            infeasibility_certificate = DepartmentOvercommitCertificate(
                best,
                pools_g,
                acts_g,
                [maximum(efficiency[a]) for a in acts_g],
                required,
                hours_available(pools_g),
            )
        end
    end

    return ResourceAllocationProblem(
        T,
        n_pools,
        n_departments,
        n_activities,
        pool_department,
        pool_cost,
        period_cost_factor,
        capacity,
        activity_department,
        release,
        deadline,
        eligible_pools,
        efficiency,
        value,
        discount,
        workload,
        rate_cap,
        floors,
        profile,
        feasible_witness,
        infeasibility_certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::ResourceAllocationProblem)

Build the multi-period allocation LP. Deterministic — uses only struct data.
Column order follows [`_resource_allocation_columns`](@ref).
"""
function build_model(prob::ResourceAllocationProblem)
    model = Model()
    act, pool, period = _resource_allocation_columns(prob)
    n = length(act)
    @variable(model, y[1:n] >= 0)

    # Efficiency of each column, and per-activity column ranges.
    eff = Vector{Float64}(undef, n)
    first_col = zeros(Int, prob.n_activities + 1)
    j = 0
    for a in 1:prob.n_activities
        first_col[a] = j + 1
        for (k, _) in enumerate(prob.eligible_pools[a]), _ in prob.release[a]:prob.deadline[a]
            j += 1
            eff[j] = prob.efficiency[a][k]
        end
    end
    first_col[end] = n + 1

    # Pool capacity rows.
    pool_cols = [Int[] for _ in 1:(prob.n_pools * prob.n_periods)]
    for c in 1:n
        push!(pool_cols[(period[c] - 1) * prob.n_pools + pool[c]], c)
    end
    for t in 1:prob.n_periods, p in 1:prob.n_pools
        cols = pool_cols[(t - 1) * prob.n_pools + p]
        isempty(cols) && continue
        @constraint(model, sum(y[c] for c in cols) <= prob.capacity[p, t])
    end

    # Activity scope rows (ranged when committed) and absorption-rate rows.
    for a in 1:prob.n_activities
        cols = first_col[a]:(first_col[a + 1] - 1)
        output = @expression(model, sum(eff[c] * y[c] for c in cols))
        if prob.floors[a] > 0
            @constraint(model, prob.floors[a] <= output <= prob.workload[a])
        else
            @constraint(model, output <= prob.workload[a])
        end
        len = prob.deadline[a] - prob.release[a] + 1
        k_pools = length(prob.eligible_pools[a])
        for (i, t) in enumerate(prob.release[a]:prob.deadline[a])
            # Columns of activity a in period t: one per eligible pool, stride len.
            tcols = [first_col[a] + (k - 1) * len + (i - 1) for k in 1:k_pools]
            if k_pools == 1
                set_upper_bound(y[tcols[1]], prob.rate_cap[a] / eff[tcols[1]])
            else
                @constraint(model, sum(eff[c] * y[c] for c in tcols) <= prob.rate_cap[a])
            end
        end
    end

    @objective(
        model,
        Max,
        sum(
            (
                prob.value[act[c]] * prob.discount^(period[c] - prob.release[act[c]]) * eff[c] -
                prob.pool_cost[pool[c]] * prob.period_cost_factor[period[c]]
            ) * y[c] for c in 1:n
        )
    )
    return model
end

register_variant(
    :resource_allocation,
    :standard,
    ResourceAllocationProblem,
    "Multi-period allocation of skilled resource pools to a portfolio of windowed activities: pool-period capacity rows, ranged commitment/scope rows with pool-specific efficiencies, absorption-rate rows, and a department over-commitment infeasibility certificate";
    tags=[:scheduling, :bipartite, :packing],
)
