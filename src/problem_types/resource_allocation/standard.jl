using JuMP
using Random
using Distributions

"""
Largest `target_variables` accepted by `ResourceAllocationProblem`. The usage
matrix is materialised as a dense `n_activities × n_resources` block of
`Float64` (at the cap: 100_000 activities x up to 96 resources, roughly 77 MB),
so larger targets are rejected with an `ArgumentError` instead of being
silently undersized (same convention as `telecom_network_design/standard` and
`supply_chain/network_planning`).
"""
const RESOURCE_ALLOCATION_MAX_VARIABLES = 100_000

"""
Planted allocation plan: the nominal activity levels the instance was designed
around, the resource consumption they induce, and the capacity slack left over.
For the `feasible` profile this point satisfies every row of the built model
(`slack .> 0`, every commitment floor at or below its plan entry).
"""
struct AllocationPlanWitness
    plan::Vector{Float64}
    consumption::Vector{Float64}
    slack::Vector{Float64}
end

"""
Floor over-commitment certificate: the mandatory minimum levels of `activities`
all consume resource `resource`, and together they demand `floor_consumption`
units of it while only `capacity` are on hand. Because every `x_i >= min_levels[i]`
and every usage coefficient is nonnegative, resource `resource`'s capacity row
cannot be satisfied by any allocation. The refutation uses only aggregate LP
rows (the capacity row plus the floor rows), so it survives
`relax_integer=true` — and since this generator has no activity ceilings,
over-commitment is the only infeasibility mode the model can even express.
"""
struct FloorOvercommitCertificate
    resource::Int
    activities::Vector{Int}
    floor_consumption::Float64
    capacity::Float64
end

"""
    ResourceAllocationProblem <: ProblemGenerator

Generator for resource allocation problems.

# Overview

Models continuous allocation of limited shared resources across competing
activities. A portfolio of activities earns profit per unit but draws on a
sparse set of shared resource pools (machine hours, cloud compute, staff
hours, advertising budgets). The decisions are activity levels; the objective
maximizes total profit; one capacity row per pool limits aggregate
consumption; optional commitment floors impose minimum activity levels.

Unlike `product_mix`, budgets here are *allocated*, not reserved against
market demand: there are no activity ceilings, and the number of shared pools
grows with the portfolio (up to 96 resources), so the capacity rows — not
per-variable bounds — carry all the tension.

# Planted allocation plan

Capacities and commitment floors are not sampled independently. At thousands
of activities, independent sampling lets aggregate floor demand and aggregate
capacity drift apart, which silently decides feasibility as a side effect of
scale. Instead a nominal allocation plan is sampled first; capacities are
derived from what that plan consumes (plus per-pool headroom) and floors as
fractions of the plan's own levels, so the two sides stay mutually consistent
at every scale.

Because usage coefficients are nonnegative and there are no upper bounds, the
pointwise-smallest candidate point is `x = min_levels`, so the instance is
feasible **iff**

`floor_utilization = max_j (sum_i usage[i,j] * min_levels[i]) / capacities[j]`

is at most 1. The three profiles place that single scalar deliberately:

  - `feasible`: floors stay below the plan and capacities above its
    consumption, so `floor_utilization < 1`; the plan is stored as an
    [`AllocationPlanWitness`](@ref).
  - `unknown`: floors are raised and capacities tightened onto a target
    utilization of `1 ± U(0.05, 0.35)` — a fair coin flip that lands on either
    side of feasibility at any problem size.
  - `infeasible`: the pools that committed activities actually draw on have
    their capacities cut below the floors' own demand (`capacity = demand /
    (1 + margin)`, `margin ∈ [0.1, 0.4]`), backed by a
    [`FloorOvercommitCertificate`](@ref).

# Fields

  - `n_activities::Int`: Number of activities (one variable each)
  - `n_resources::Int`: Number of shared resource pools (one capacity row each)
  - `profits::Vector{Float64}`: Profit per unit of each activity (strictly positive)
  - `usage::Matrix{Float64}`: Resource usage per unit of activity (`n_activities × n_resources`, stored dense with zeros, sampled sparsely)
  - `capacities::Vector{Float64}`: Available amount of each resource
  - `min_levels::Vector{Float64}`: Mandatory minimum level of each activity (0 when uncommitted)
  - `nominal_plan::Vector{Float64}`: Planted allocation plan the data is built around
  - `floor_utilization::Float64`: Tightest resource's floor-induced utilization
  - `profile::Symbol`: Sampled allocation regime
  - `feasible_witness::Union{Nothing,AllocationPlanWitness}`: set for `feasible`
  - `infeasibility_certificate::Union{Nothing,FloorOvercommitCertificate}`: set for `infeasible`
  - `feasibility_status::FeasibilityStatus`: Requested profile
"""
struct ResourceAllocationProblem <: ProblemGenerator
    n_activities::Int
    n_resources::Int
    profits::Vector{Float64}
    usage::Matrix{Float64}
    capacities::Vector{Float64}
    min_levels::Vector{Float64}
    nominal_plan::Vector{Float64}
    floor_utilization::Float64
    profile::Symbol
    feasible_witness::Union{Nothing, AllocationPlanWitness}
    infeasibility_certificate::Union{Nothing, FloorOvercommitCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _resource_allocation_dimensions(target_variables, rng) -> NTuple{2,Int}

Sample `(n_activities, n_resources)` for a target size. Activities map
one-to-one onto variables, so `n_activities` is the target itself (with a
floor of 3 so tiny requests still form a portfolio). The pool count is
tiered so a bigger portfolio faces proportionally more shared resources —
a 2-resource instance at 10_000 activities would be degenerate — while
staying bounded by the dense-matrix memory budget documented on
`RESOURCE_ALLOCATION_MAX_VARIABLES`.
"""
function _resource_allocation_dimensions(target_variables::Int, rng::AbstractRNG)
    n_activities = max(3, target_variables)
    n_resources = if target_variables <= 250
        # Small portfolio: a handful of shared pools.
        rand(rng, DiscreteUniform(4, 12))
    elseif target_variables <= 1000
        # Departmental allocation.
        rand(rng, DiscreteUniform(10, 36))
    elseif target_variables <= 5000
        # Divisional allocation.
        rand(rng, DiscreteUniform(20, 64))
    else
        # Enterprise portfolio: up to 96 pools (100_000 x 96 doubles ≈ 77 MB).
        rand(rng, DiscreteUniform(32, 96))
    end
    return n_activities, n_resources
end

"""
    _resource_allocation_consumption(usage, levels) -> Vector{Float64}

Resource consumption induced by a given activity-level vector (`usage' * levels`), skipping activities at level zero. With nonnegative usage
coefficients this is also a lower bound on the consumption of any allocation
dominating `levels` componentwise.
"""
function _resource_allocation_consumption(usage::Matrix{Float64}, levels::Vector{Float64})
    n_activities, n_resources = size(usage)
    consumption = zeros(n_resources)
    for i in 1:n_activities
        level = levels[i]
        level > 0.0 || continue
        for j in 1:n_resources
            consumption[j] += usage[i, j] * level
        end
    end
    return consumption
end

"""
    _resource_allocation_utilization(usage, min_levels, capacities) -> Float64

The tightest resource's floor-induced utilization: the ratio of the
consumption the mandatory floors alone impose to the capacity available.
Because usage is nonnegative and there are no upper bounds, the instance is
feasible iff this is `<= 1`.
"""
function _resource_allocation_utilization(
    usage::Matrix{Float64}, min_levels::Vector{Float64}, capacities::Vector{Float64}
)
    required = _resource_allocation_consumption(usage, min_levels)
    return maximum(required[j] / capacities[j] for j in 1:length(capacities))
end

"""
    ResourceAllocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a resource allocation problem instance.

# Arguments

  - `target_variables`: Target number of variables (activities); at most
    `RESOURCE_ALLOCATION_MAX_VARIABLES`
  - `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
  - `seed`: Random seed for reproducibility
"""
function ResourceAllocationProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    if target_variables > RESOURCE_ALLOCATION_MAX_VARIABLES
        throw(
            ArgumentError(
                "resource_allocation/standard supports at most " *
                "$(RESOURCE_ALLOCATION_MAX_VARIABLES) variables (requested $(target_variables))",
            ),
        )
    end

    rng = MersenneTwister(seed)
    n_activities, n_resources = _resource_allocation_dimensions(target_variables, rng)

    # Scale-dependent sampling regime: a bigger portfolio spreads over more
    # pools with finer-grained (smaller volume, cheaper) activities.
    if target_variables <= 250
        usage_min = rand(rng, LogNormal(log(0.8), 0.35))
        usage_max = rand(rng, LogNormal(log(6.0), 0.3))
        profit_min = rand(rng, LogNormal(log(12.0), 0.4))
        profit_max = rand(rng, LogNormal(log(90.0), 0.3))
        uses_density = rand(rng, Beta(3, 5))
        commitment_prob = rand(rng, Beta(4, 6))
        correlation_strength = rand(rng, Beta(4, 3))
        activity_center = rand(rng, LogNormal(log(120.0), 0.4))
    elseif target_variables <= 1000
        usage_min = rand(rng, LogNormal(log(0.5), 0.4))
        usage_max = rand(rng, LogNormal(log(4.5), 0.35))
        profit_min = rand(rng, LogNormal(log(8.0), 0.45))
        profit_max = rand(rng, LogNormal(log(70.0), 0.35))
        uses_density = rand(rng, Beta(2, 5))
        commitment_prob = rand(rng, Beta(5, 5))
        correlation_strength = rand(rng, Beta(5, 4))
        activity_center = rand(rng, LogNormal(log(80.0), 0.45))
    elseif target_variables <= 5000
        usage_min = rand(rng, LogNormal(log(0.3), 0.45))
        usage_max = rand(rng, LogNormal(log(3.5), 0.4))
        profit_min = rand(rng, LogNormal(log(4.0), 0.5))
        profit_max = rand(rng, LogNormal(log(50.0), 0.4))
        uses_density = rand(rng, Beta(2, 6))
        commitment_prob = rand(rng, Beta(6, 4))
        correlation_strength = rand(rng, Beta(6, 4))
        activity_center = rand(rng, LogNormal(log(50.0), 0.5))
    else
        usage_min = rand(rng, LogNormal(log(0.2), 0.5))
        usage_max = rand(rng, LogNormal(log(3.0), 0.45))
        profit_min = rand(rng, LogNormal(log(2.0), 0.55))
        profit_max = rand(rng, LogNormal(log(35.0), 0.45))
        uses_density = rand(rng, Beta(2, 7))
        commitment_prob = rand(rng, Beta(7, 4))
        correlation_strength = rand(rng, Beta(8, 3))
        activity_center = rand(rng, LogNormal(log(30.0), 0.5))
    end

    # Allocation regime: which kind of budget is being split. Regimes shift
    # the profit/usage scales, the correlation, and how many pools an
    # activity touches, so the corpus mixes capacity-, compute-, labor- and
    # spend-flavored instances rather than one undifferentiated soup.
    profile = rand(
        rng, (:manufacturing_capacity, :cloud_compute, :workforce_hours, :advertising_budget)
    )
    if profile == :manufacturing_capacity
        # Machine hours are the binding story: heavy usage, durable commitments.
        usage_min *= 1.2
        usage_max *= 1.5
        commitment_prob *= 1.15
    elseif profile == :cloud_compute
        # Big-ticket workloads with high margins drawing on a few pools.
        profit_min *= 1.6
        profit_max *= 2.4
        usage_max *= 1.3
        uses_density *= 0.7
    elseif profile == :workforce_hours
        # Stable margins, skilled teams on the valuable work, many commitments.
        profit_min *= 0.8
        profit_max *= 0.7
        usage_min *= 1.2
        usage_max *= 0.8
        correlation_strength *= 1.2
        commitment_prob *= 1.3
    else
        # :advertising_budget — cheap impressions spread across many channels.
        profit_max *= 1.6
        usage_min *= 0.6
        usage_max *= 0.7
        uses_density *= 1.3
    end
    uses_density = clamp(uses_density, 0.05, 0.9)
    commitment_prob = clamp(commitment_prob, 0.05, 0.95)
    correlation_strength = clamp(correlation_strength, 0.1, 0.95)

    # Quality factors drive the profit/usage correlation: high-quality
    # activities are both more profitable *and* hungrier per unit, so the LP
    # has no obvious winner (profit density is not monotone in quality).
    quality_factors = rand(rng, Beta(2, 2), n_activities)

    base_profit_scale = (profit_min + profit_max) / 2
    base_profits = rand(rng, LogNormal(log(base_profit_scale), 0.35), n_activities)
    profits = clamp.(base_profits, profit_min, profit_max)
    profits .+= correlation_strength .* quality_factors .* (0.5 * (profit_max - profit_min))

    # Sparse usage: each activity draws on a handful of pools, not all of
    # them (a campaign uses a few channels, a job a few machine types). Every
    # activity uses at least one pool — that is what keeps the profit-max LP
    # bounded, since every strictly profitable activity then hits some finite
    # capacity row.
    usage = zeros(n_activities, n_resources)
    max_uses = clamp(ceil(Int, n_resources * uses_density), 2, n_resources)
    pool_scale = clamp.(
        rand(rng, LogNormal(log(sqrt(usage_min * usage_max)), 0.4), n_resources),
        usage_min,
        usage_max,
    )
    for i in 1:n_activities
        n_uses = rand(rng, DiscreteUniform(1, max_uses))
        pools = shuffle(rng, 1:n_resources)
        for j in pools[1:n_uses]
            jitter = rand(rng, LogNormal(0.0, 0.4))
            quality_multiplier = 0.4 + correlation_strength * quality_factors[i]
            usage[i, j] = pool_scale[j] * quality_multiplier * jitter
        end
    end

    # No dead pools: with few activities a pool can end up unused, which
    # would leave a vacuous capacity row. Give one random activity a positive
    # draw on any empty pool.
    for j in 1:n_resources
        if all(iszero, @view usage[:, j])
            i = rand(rng, 1:n_activities)
            usage[i, j] = pool_scale[j] * (0.5 + rand(rng))
        end
    end

    # --- Planted allocation plan -------------------------------------------
    # A nominal level for every activity; everything downstream (capacities,
    # floors) derives from it, keeping both sides consistent at any scale.
    nominal_plan = clamp.(
        rand(rng, LogNormal(log(activity_center), 0.55), n_activities),
        0.1 * activity_center,
        10.0 * activity_center,
    )
    consumption = _resource_allocation_consumption(usage, nominal_plan)

    # Capacity = what the plan consumes plus per-pool headroom. The wide
    # lognormal headroom (a few pools nearly saturated, others with slack to
    # spare) decides which rows bind at the optimum, and varies per resource.
    headroom = clamp.(rand(rng, LogNormal(log(0.2), 0.9), n_resources), 0.02, 2.5)
    capacities = consumption .* (1.0 .+ headroom)

    # Commitment floors: fractions of the plan's own levels, so they can never
    # conflict with the capacities derived from the same plan.
    floor_prob = clamp(0.25 + 0.7 * commitment_prob, 0.25, 0.95)
    min_levels = zeros(n_activities)
    for i in 1:n_activities
        if rand(rng) < floor_prob
            # Fraction is capped at 0.9 so floors stay strictly below the plan.
            min_levels[i] = (0.2 + 0.7 * rand(rng, Beta(2, 2))) * nominal_plan[i]
        end
    end

    # At least one committed activity, so the utilization scalar below is
    # well defined and the `unknown`/`infeasible` perturbations have
    # something to act on.
    if all(iszero, min_levels)
        i = rand(rng, 1:n_activities)
        min_levels[i] = (0.2 + 0.7 * rand(rng, Beta(2, 2))) * nominal_plan[i]
    end

    # --- Feasibility profile -------------------------------------------------
    feasible_witness = nothing
    infeasibility_certificate = nothing

    if feasibility_status == feasible
        # Nothing to perturb: floors are at most 0.9 * plan and capacities
        # strictly exceed the plan's consumption, so the plan itself is a
        # feasible point with positive slack on every capacity row.
        feasible_witness = AllocationPlanWitness(
            copy(nominal_plan), copy(consumption), capacities .- consumption
        )
    elseif feasibility_status == infeasible
        # Over-commit specific budget lines. The floors alone demand
        # `floor_consumption` of every pool; cutting a pool's capacity below
        # that demand makes the instance infeasible unconditionally — no
        # search, no safety nets. One to three pools are violated (a single
        # blown budget line is the common story; a few at once add variety).
        floor_consumption = _resource_allocation_consumption(usage, min_levels)
        candidates = [j for j in 1:n_resources if floor_consumption[j] > 0.0]
        k_violated = rand(rng) < 0.55 ? 1 : (rand(rng) < 0.6 ? 2 : 3)
        k_violated = min(k_violated, length(candidates))
        violated = shuffle(rng, candidates)[1:k_violated]
        for j in violated
            margin = 0.1 + 0.3 * rand(rng)
            capacities[j] = floor_consumption[j] / (1.0 + margin)
        end

        # Certificate on the single most violated pool.
        critical = argmax([floor_consumption[j] / capacities[j] for j in 1:n_resources])
        committed = [i for i in 1:n_activities if min_levels[i] > 0.0 && usage[i, critical] > 0.0]
        infeasibility_certificate = FloorOvercommitCertificate(
            critical, committed, floor_consumption[critical], capacities[critical]
        )
    else
        # `unknown`: steer the tightest floor utilization onto a target drawn
        # as 1 ± U(0.05, 0.35) — a genuine coin flip at every scale. Splitting
        # the adjustment between raising floors and cutting capacity keeps
        # both sides realistic; only their ratio (which is what decides
        # feasibility) is pinned to `gap`, so the realized utilization equals
        # the target regardless of the clamps below.
        margin = 0.05 + 0.30 * rand(rng)
        target = rand(rng) < 0.5 ? 1.0 - margin : 1.0 + margin
        current = _resource_allocation_utilization(usage, min_levels, capacities)
        gap = target / current
        theta = 0.35 + 0.3 * rand(rng)
        capacity_scale = clamp(gap^(theta - 1.0), 0.35, 3.0)
        floor_scale = gap * capacity_scale
        capacities .*= capacity_scale
        min_levels .*= floor_scale
    end

    floor_utilization = _resource_allocation_utilization(usage, min_levels, capacities)

    return ResourceAllocationProblem(
        n_activities,
        n_resources,
        profits,
        usage,
        capacities,
        min_levels,
        nominal_plan,
        floor_utilization,
        profile,
        feasible_witness,
        infeasibility_certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::ResourceAllocationProblem)

Build a JuMP model for the resource allocation problem.

# Arguments

  - `prob`: ResourceAllocationProblem instance

# Returns

  - `model`: The JuMP model
"""
function build_model(prob::ResourceAllocationProblem)
    model = Model()

    # Decision variables: activity levels.
    @variable(model, x[1:prob.n_activities] >= 0)

    # Resource capacity constraints (only the activities that actually draw
    # on the pool appear in its row; the usage matrix is sparse).
    for j in 1:prob.n_resources
        @constraint(
            model,
            sum(prob.usage[i, j] * x[i] for i in 1:prob.n_activities if prob.usage[i, j] > 0) <=
                prob.capacities[j]
        )
    end

    # Commitment floors as explicit rows (they are commitments, not domain
    # knowledge, and the transform machinery treats rows and bounds
    # differently).
    for i in 1:prob.n_activities
        if prob.min_levels[i] > 0
            @constraint(model, x[i] >= prob.min_levels[i])
        end
    end

    # Objective: maximize total profit.
    @objective(model, Max, sum(prob.profits[i] * x[i] for i in 1:prob.n_activities))

    return model
end

# Register the variant
register_variant(
    :resource_allocation,
    :standard,
    ResourceAllocationProblem,
    "Resource allocation problem that maximizes profit by allocating limited shared resources across competing activities with optional minimum activity levels",
)
