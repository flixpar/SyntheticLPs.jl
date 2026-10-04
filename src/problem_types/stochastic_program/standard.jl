using JuMP
using Random
using Distributions

"""
Planted first- and second-stage plan of a requested-feasible
`StochasticProgramProblem`. `capacity[i]` is the committed capacity; in every
scenario `s` each customer's served share `service_level * demand[j, s]` is
split over its lanes by fixed routing weights (`shipment[l, s]`), and the rest
is `shortfall[j, s]`. `capacity[i]` is the largest load any scenario puts on
facility `i`, so every capacity-linking row holds, every service row holds with
equality, and the capital row holds because the budget was drawn above the
plan's capital use.
"""
struct StochasticProgramWitness
    capacity::Vector{Float64}
    shipment::Matrix{Float64}
    shortfall::Matrix{Float64}
end

"""
Service-level / capital-budget infeasibility certificate. In scenario
`scenario`, the service row forces `Σ_j served ≥ service_level * Σ_j demand =
required_capacity`; every served unit leaves some facility, so summing the
scenario's capacity-linking rows gives `Σ_i x[i] ≥ required_capacity`. With
`existing_capacity ≤ x ≤ capacity_max`, the cheapest way to reach that total in
capital is the greedy fill by increasing `capital_use` (a fractional knapsack),
whose capital is `min_capital`. The capital row allows only
`capital_budget = min_capital - margin`. The argument combines the capital row,
`n_facilities` linking rows, and one service row with the demand equalities —
no single row or bound is contradictory, so presolve does not detect it.
"""
struct StochasticProgramCertificate
    scenario::Int
    required_capacity::Float64
    min_capital::Float64
    capital_budget::Float64
    margin::Float64
end

"""
    StochasticProgramProblem <: ProblemGenerator

Two-stage stochastic capacity-and-distribution planning LP with recourse.

# Overview

Facilities are placed among clustered customers on a 100 × 100 map. In the
**first stage** a capacity `x[i]` is committed at each facility before demand is
known — above the existing installed capacity, below a site limit, and within a
shared capital budget. In the **second stage**, after demand scenario `s` is
revealed, demand is shipped over a sparse set of lanes (each customer is
reachable from its few nearest facilities); unserved demand is a penalized
shortfall, and a scenario-wide **service-level row** caps the total shortfall at
`(1 - service_level)` of the scenario's demand. The objective minimizes
annualized capacity cost plus the probability-weighted shipping and shortfall
cost.

The constraint matrix is the canonical dual block-angular structure of two-stage
stochastic programming: the first-stage capital row plus `S` scenario blocks
(facility linking, customer demand, service level) that interact only through
the capacity columns — the instance class of L-shaped/Benders decomposition.

# Data grounding

Customer demand is lognormal and correlated across scenarios through a global
market factor, a regional factor (customers belong to geographic clusters), and
idiosyncratic noise, so scenarios disagree in both volume and geography.
Shipping cost is a facility handling cost plus a distance-proportional freight
rate. The shortfall penalty is calibrated so that the newsvendor critical ratio
`build_cost / (penalty - freight)` lies around 0.12–0.35: capacity is cheap
enough that the optimal plan serves the large majority of demand in most
scenarios, and the build-versus-shortfall trade-off is real.

# Feasibility control

Statuses differ only in the capital budget; every other datum is drawn
identically. Let `min_capital` be the cheapest capital that lets the
highest-demand scenario meet its service level (see
[`StochasticProgramCertificate`](@ref)) and `witness_capital` the capital of the
planted routing plan.

  - `feasible`: budget `witness_capital * U(1.03, 1.15)`; stores
    [`StochasticProgramWitness`](@ref).
  - `infeasible`: budget `min_capital * U(0.82, 0.92)` (a ≥ 8% margin); stores
    the certificate. Requires simplex work to discover.
  - `unknown`: budget log-uniform between `0.92 * min_capital` and
    `1.08 * witness_capital` — below the provable threshold it is infeasible,
    above the planted plan it is feasible, and in between the lane structure
    decides. Neither witness nor certificate is stored.

# Size

Variables: `n_facilities + n_scenarios * (n_lanes + n_customers)`.
Rows: `1 + n_scenarios * (n_facilities + n_customers + 1)`. The lane count is
tuned after the scenario count is chosen, so the total lands within about
`n_scenarios / 2` of the target.

# Fields

  - `n_facilities`, `n_customers`, `n_scenarios`: dimensions
  - `facility_locations`, `customer_locations`: map coordinates
  - `customer_region::Vector{Int}`: customer cluster index
  - `lanes::Vector{Tuple{Int,Int}}`: `(facility, customer)` lanes, sorted
  - `ship_cost::Vector{Float64}`: per-unit cost of each lane
  - `build_cost::Vector{Float64}`: annualized cost per unit of capacity
  - `capital_use::Vector{Float64}`: capital per unit of capacity
  - `capital_budget::Float64`: first-stage capital budget
  - `existing_capacity`, `capacity_max`: first-stage bounds
  - `shortfall_cost::Vector{Float64}`: per-unit penalty for unmet demand
  - `service_level::Float64`: minimum served fraction of every scenario's demand
  - `scenario_prob::Vector{Float64}`: scenario probabilities (sum to 1)
  - `demand::Matrix{Float64}`: `n_customers × n_scenarios`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct StochasticProgramProblem <: ProblemGenerator
    n_facilities::Int
    n_customers::Int
    n_scenarios::Int
    facility_locations::Vector{Tuple{Float64, Float64}}
    customer_locations::Vector{Tuple{Float64, Float64}}
    customer_region::Vector{Int}
    lanes::Vector{Tuple{Int, Int}}
    ship_cost::Vector{Float64}
    build_cost::Vector{Float64}
    capital_use::Vector{Float64}
    capital_budget::Float64
    existing_capacity::Vector{Float64}
    capacity_max::Vector{Float64}
    shortfall_cost::Vector{Float64}
    service_level::Float64
    scenario_prob::Vector{Float64}
    demand::Matrix{Float64}
    feasible_witness::Union{Nothing, StochasticProgramWitness}
    infeasibility_certificate::Union{Nothing, StochasticProgramCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _stochastic_program_dimensions(target) -> (I, J, S, A)

Facility, customer, scenario, and lane counts. Customers grow like
`target^0.42` (≈22 at 1k, ≈57 at 10k, ≈150 at 100k), facilities are about a
quarter of the customers, the scenario count absorbs the rest of the target at
four lanes per customer, and the lane count is then re-tuned (between two and
six lanes per customer) so `I + S * (A + J)` lands on the target.
"""
function _stochastic_program_dimensions(target::Int)
    t = max(target, 1)
    J = clamp(round(Int, 1.2 * Float64(t)^0.42), 4, 4_000)
    I = clamp(round(Int, J / 4), 3, 1_000)
    k0 = min(I, 4)
    S = max(2, round(Int, (t - I) / (J * (1 + k0))))
    A_lo = J * min(I, 2)
    A_hi = J * min(I, 6)
    A = clamp(round(Int, (t - I) / S) - J, A_lo, A_hi)
    return I, J, S, A
end

"""
    _stochastic_program_min_capital(capital_use, lower, upper, required) -> Float64

Cheapest capital `Σ capital_use[i] x[i]` over `lower ≤ x ≤ upper` with
`Σ x ≥ required` (greedy fill by increasing `capital_use`); `Inf` if even the
upper bounds cannot reach `required`.
"""
function _stochastic_program_min_capital(capital_use, lower, upper, required::Float64)
    capital = sum(capital_use .* lower)
    remaining = required - sum(lower)
    for i in sortperm(capital_use)
        remaining <= 0 && break
        add = min(upper[i] - lower[i], remaining)
        capital += capital_use[i] * add
        remaining -= add
    end
    return remaining > 1e-9 ? Inf : capital
end

function StochasticProgramProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    I, J, S, A = _stochastic_program_dimensions(target_variables)

    # --- Geography: customer clusters (regions) and facilities near them ---
    n_regions = clamp(round(Int, sqrt(J) / 1.5), 2, 12)
    centers = [(rand(rng, Uniform(12, 88)), rand(rng, Uniform(12, 88))) for _ in 1:n_regions]
    region_weight = rand(rng, Dirichlet(fill(2.0, n_regions)))
    customer_region = [rand(rng, Categorical(region_weight)) for _ in 1:J]
    customer_locations = [
        (
            clamp(centers[r][1] + rand(rng, Normal(0, 8)), 0, 100),
            clamp(centers[r][2] + rand(rng, Normal(0, 8)), 0, 100),
        ) for r in customer_region
    ]
    facility_locations = [
        if rand(rng) < 0.7
            r = rand(rng, Categorical(region_weight))
            (
                clamp(centers[r][1] + rand(rng, Normal(0, 12)), 0, 100),
                clamp(centers[r][2] + rand(rng, Normal(0, 12)), 0, 100),
            )
        else
            (rand(rng, Uniform(0, 100)), rand(rng, Uniform(0, 100)))
        end for _ in 1:I
    ]
    dist(i, j) = hypot(
        facility_locations[i][1] - customer_locations[j][1],
        facility_locations[i][2] - customer_locations[j][2],
    )

    # --- Lanes: every customer gets its two nearest facilities, the remaining
    # budget goes to the nearest further (facility, customer) pairs ---
    lane_set = Tuple{Int, Int}[]
    extras = Tuple{Float64, Int, Int}[]
    for j in 1:J
        order = sortperm([dist(i, j) for i in 1:I])
        base = min(2, I)
        for r in 1:base
            push!(lane_set, (order[r], j))
        end
        for r in (base + 1):min(I, 6)
            i = order[r]
            push!(extras, (dist(i, j) * rand(rng, Uniform(0.8, 1.25)), i, j))
        end
    end
    sort!(extras; by=first)
    for (_, i, j) in extras
        length(lane_set) >= A && break
        push!(lane_set, (i, j))
    end
    sort!(lane_set; by=l -> (l[2], l[1]))
    lanes = lane_set
    nL = length(lanes)

    # --- Costs ---
    handling = rand(rng, LogNormal(log(2.0), 0.3), I)
    freight_rate = rand(rng, Uniform(0.08, 0.14))
    ship_cost = [
        handling[i] + freight_rate * dist(i, j) * rand(rng, Uniform(0.9, 1.1)) for (i, j) in lanes
    ]
    capital_use = rand(rng, LogNormal(log(10.0), 0.3), I)          # capex per unit
    build_cost = [0.11 * capital_use[i] + rand(rng, Uniform(0.4, 1.2)) for i in 1:I]
    mean_build = sum(build_cost) / I
    lane_cost_of = [Float64[] for _ in 1:J]
    for (l, (_, j)) in enumerate(lanes)
        push!(lane_cost_of[j], ship_cost[l])
    end
    shortfall_cost = [
        sum(lane_cost_of[j]) / length(lane_cost_of[j]) + mean_build / rand(rng, Uniform(0.12, 0.35)) for j in 1:J
    ]

    # --- Scenarios: correlated lognormal demand ---
    scenario_prob = rand(rng, Uniform(0.5, 1.5), S)
    scenario_prob ./= sum(scenario_prob)
    base_demand = rand(rng, LogNormal(log(100.0), 0.5), J)
    market_sigma = rand(rng, Uniform(0.10, 0.22))
    demand = zeros(Float64, J, S)
    for s in 1:S
        market = rand(rng, LogNormal(0.0, market_sigma))
        regional = rand(rng, LogNormal(0.0, 0.12), n_regions)
        for j in 1:J
            demand[j, s] =
                base_demand[j] * market * regional[customer_region[j]] *
                rand(rng, LogNormal(0.0, 0.10))
        end
    end
    service_level = rand(rng, Uniform(0.85, 0.96))

    # --- Reference plan: fixed routing weights per customer ---
    weight = zeros(Float64, nL)
    first_lane = 1
    for j in 1:J
        last_lane = first_lane + length(lane_cost_of[j]) - 1
        for l in first_lane:last_lane
            weight[l] = rand(rng, Uniform(0.5, 1.5)) / ship_cost[l]
        end
        weight[first_lane:last_lane] ./= sum(@view weight[first_lane:last_lane])
        first_lane = last_lane + 1
    end
    load = zeros(Float64, I, S)
    witness_ship = zeros(Float64, nL, S)
    for s in 1:S, (l, (i, j)) in enumerate(lanes)
        witness_ship[l, s] = service_level * weight[l] * demand[j, s]
        load[i, s] += witness_ship[l, s]
    end
    witness_capacity = vec(maximum(load; dims=2))
    typical_load = sum(witness_capacity) / I
    existing_capacity = [
        min(witness_capacity[i], typical_load) * rand(rng, Uniform(0.0, 0.25)) for i in 1:I
    ]
    witness_capacity .= max.(witness_capacity, existing_capacity)
    capacity_max = [
        max(witness_capacity[i] * rand(rng, Uniform(1.25, 1.8)), 0.5 * typical_load) for i in 1:I
    ]

    witness_capital = sum(capital_use .* witness_capacity)
    totals = vec(sum(demand; dims=1))
    worst = argmax(totals)
    required = service_level * totals[worst]
    min_capital = _stochastic_program_min_capital(
        capital_use, existing_capacity, capacity_max, required
    )
    @assert isfinite(min_capital) && min_capital <= witness_capital + 1e-6

    feasible_witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        capital_budget = witness_capital * rand(rng, Uniform(1.03, 1.15))
        shortfall = [(1 - service_level) * demand[j, s] for j in 1:J, s in 1:S]
        feasible_witness = StochasticProgramWitness(witness_capacity, witness_ship, shortfall)
    elseif feasibility_status == infeasible
        capital_budget = min_capital * rand(rng, Uniform(0.82, 0.92))
        existing_capital = sum(capital_use .* existing_capacity)
        @assert capital_budget > existing_capital
        certificate = StochasticProgramCertificate(
            worst, required, min_capital, capital_budget, min_capital - capital_budget
        )
    else
        lo, hi = log(0.92 * min_capital), log(1.08 * witness_capital)
        capital_budget = exp(lo + rand(rng) * (hi - lo))
    end

    return StochasticProgramProblem(
        I,
        J,
        S,
        facility_locations,
        customer_locations,
        customer_region,
        lanes,
        ship_cost,
        build_cost,
        capital_use,
        capital_budget,
        existing_capacity,
        capacity_max,
        shortfall_cost,
        service_level,
        scenario_prob,
        demand,
        feasible_witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::StochasticProgramProblem)

Build the extensive-form (deterministic-equivalent) LP. Deterministic — uses only
the struct fields.
"""
function build_model(prob::StochasticProgramProblem)
    model = Model()
    I, J, S = prob.n_facilities, prob.n_customers, prob.n_scenarios
    lanes = prob.lanes
    nL = length(lanes)

    @variable(model, prob.existing_capacity[i] <= x[i = 1:I] <= prob.capacity_max[i])
    @variable(model, y[1:nL, 1:S] >= 0)
    @variable(model, z[1:J, 1:S] >= 0)

    @objective(
        model,
        Min,
        sum(prob.build_cost[i] * x[i] for i in 1:I) + sum(
            prob.scenario_prob[s] * (
                sum(prob.ship_cost[l] * y[l, s] for l in 1:nL) +
                sum(prob.shortfall_cost[j] * z[j, s] for j in 1:J)
            ) for s in 1:S
        )
    )

    @constraint(
        model, capital, sum(prob.capital_use[i] * x[i] for i in 1:I) <= prob.capital_budget
    )

    out_lanes = [Int[] for _ in 1:I]
    in_lanes = [Int[] for _ in 1:J]
    for (l, (i, j)) in enumerate(lanes)
        push!(out_lanes[i], l)
        push!(in_lanes[j], l)
    end
    @constraint(model, linking[i = 1:I, s = 1:S], sum(y[l, s] for l in out_lanes[i]) <= x[i])
    @constraint(
        model,
        demand_balance[j = 1:J, s = 1:S],
        sum(y[l, s] for l in in_lanes[j]) + z[j, s] == prob.demand[j, s]
    )
    @constraint(
        model,
        service[s = 1:S],
        sum(z[j, s] for j in 1:J) <= (1 - prob.service_level) * sum(prob.demand[:, s])
    )
    return model
end

register_variant(
    :stochastic_program,
    :standard,
    StochasticProgramProblem,
    "Two-stage stochastic capacity/distribution LP (extensive form): first-stage capacity under a capital budget, sparse-lane recourse with penalized shortfall and per-scenario service levels over correlated demand scenarios — dual block-angular structure";
    default=true,
    tags=[:logistics, :dual_block_angular, :bipartite],
)
