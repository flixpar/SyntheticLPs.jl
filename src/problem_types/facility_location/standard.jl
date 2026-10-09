using JuMP
using Random
using Distributions

"""
    FacilityLocationWitness

Planted feasible plan for [`FacilityLocationProblem`](@ref): the facilities to
open and a shipment plan as `(facility, customer, quantity)` triplets (absent
pairs ship zero). Integral in `y`, so it is feasible for the MIP and its
relaxation.
"""
struct FacilityLocationWitness
    open::Vector{Int}
    shipments::Vector{Tuple{Int, Int, Float64}}
end

"""
    FacilityBudgetCertificate

LP-row infeasibility certificate for [`FacilityLocationProblem`](@ref). Summing
the demand rows and the capacity rows gives
`total_demand ≤ Σ_c Σ_w x[w,c] ≤ Σ_w cap_w y_w`, and the budget row with
`0 ≤ y ≤ 1` caps the right-hand side at the fractional-knapsack optimum
`fundable_capacity = max {Σ cap_w y_w : Σ fixed_w y_w ≤ budget, 0 ≤ y ≤ 1}`.
The generator keeps `fundable_capacity ≤ 0.95 · total_demand`.
"""
struct FacilityBudgetCertificate
    budget::Float64
    fundable_capacity::Float64
    total_demand::Float64
end

"""
    FacilityLocationProblem <: ProblemGenerator

Single-source-free capacitated facility location (CFLP) with an opening budget,
in the strong formulation.

# Overview

A distribution planner chooses which candidate facilities to open and how much
each open facility ships to each customer, minimizing fixed opening cost plus
distance-based shipping cost. Rows:

  - demand `Σ_w x[w,c] ≥ d_c`;
  - aggregate capacity `Σ_c x[w,c] ≤ cap_w · y_w`;
  - strong linking `x[w,c] ≤ d_c · y_w` for every facility–customer pair — the
    disaggregated inequalities of the textbook strong CFLP formulation
    (Cornuéjols, Sridharan & Thizy 1991), which keep the opening decisions
    binding under LP relaxation instead of letting `y_w` shrink to
    `throughput / cap_w`;
  - an opening budget `Σ_w fixed_w y_w ≤ budget`.

Customers are clustered in towns, facilities are spread uniformly, shipping
costs are distance-based with lane noise, and fixed costs grow with capacity
and location (OR-Library `cap`-style instances: tens to hundreds of facilities,
roughly 4–15× as many customers).

# Fields

  - `n_facilities::Int`, `n_customers::Int`
  - `facility_locs`, `customer_locs`: coordinates
  - `demands::Vector{Float64}`, `fixed_costs::Vector{Float64}`,
    `capacities::Vector{Float64}`
  - `shipping_costs::Matrix{Float64}`: `F × C` per-unit shipping cost
  - `budget::Float64`
  - `feasible_witness::Union{Nothing,FacilityLocationWitness}`
  - `infeasibility_certificate::Union{Nothing,FacilityBudgetCertificate}`
"""
struct FacilityLocationProblem <: ProblemGenerator
    n_facilities::Int
    n_customers::Int
    facility_locs::Vector{Tuple{Float64, Float64}}
    customer_locs::Vector{Tuple{Float64, Float64}}
    demands::Vector{Float64}
    fixed_costs::Vector{Float64}
    capacities::Vector{Float64}
    shipping_costs::Matrix{Float64}
    budget::Float64
    feasible_witness::Union{Nothing, FacilityLocationWitness}
    infeasibility_certificate::Union{Nothing, FacilityBudgetCertificate}
end

# Fractional-knapsack capacity reachable within `budget` (open in decreasing
# capacity/cost order, the last one fractionally): the exact LP maximum of
# Σ cap_w y_w subject to Σ fixed_w y_w ≤ budget and 0 ≤ y ≤ 1.
function _fl_fundable_capacity(capacities::Vector{Float64}, fixed::Vector{Float64}, budget::Float64)
    order = sortperm(capacities ./ fixed; rev=true)
    remaining = budget
    total = 0.0
    for w in order
        if fixed[w] <= remaining
            total += capacities[w]
            remaining -= fixed[w]
        else
            total += capacities[w] * max(remaining, 0.0) / fixed[w]
            break
        end
    end
    return total
end

"""
    FacilityLocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a capacitated facility location instance.

# Variable-count formula

    total = F · (C + 1)

(`F` opening variables plus `F·C` shipments). The customer-to-facility ratio
`r ∈ 4..15` is sampled, `F = max(2, round(sqrt(target / r)))` and
`C = max(1, round(target / F) - 1)`, so the count is within `F/2` of the target
(well under 1% above ~1,000 variables). No size cap.

# Feasibility

  - `feasible`: capacities are scaled up when the total falls below 1.05×
    demand; facilities are opened greedily in decreasing capacity/cost order
    until they cover demand, and the budget is at least 1.02–1.25× that
    subset's cost. Customers are then served by their nearest open facilities
    with spare capacity, recorded as a [`FacilityLocationWitness`](@ref).
  - `infeasible`: the budget is set to 75–95% of the fractional-knapsack cost
    of reaching total demand, so even fractional openings fund less capacity
    than demand ([`FacilityBudgetCertificate`](@ref)). The argument aggregates
    every demand and capacity row plus the budget row, so presolve does not
    detect it.
  - `unknown`: the sampled budget (60–95% of the total fixed cost) and
    capacities (1.3–2.0× demand overall) are kept as drawn — usually, but not
    always, enough.
"""
function FacilityLocationProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)

    ratio = rand(rng, 4:15)
    F = max(2, round(Int, sqrt(target_variables / ratio)))
    C = max(1, round(Int, target_variables / F) - 1)

    span = rand(rng, 500.0:100.0:3000.0)
    transport_cost_per_km = rand(rng, 0.8:0.1:2.5)
    min_demand, max_demand = rand(rng, 10.0:5.0:40.0), rand(rng, 100.0:25.0:400.0)
    fixed_cost_min = rand(rng, 50000.0:10000.0:200000.0)
    fixed_cost_max = fixed_cost_min * rand(rng, 3.0:0.5:6.0)
    capacity_factor = rand(rng, 1.3:0.1:2.0)
    budget_factor = rand(rng, 0.6:0.05:0.95)

    facility_locs = [(span * rand(rng), span * rand(rng)) for _ in 1:F]
    n_clusters = max(2, div(C, 20))
    centers = [(span * rand(rng), span * rand(rng)) for _ in 1:n_clusters]
    customer_locs = _fl_clustered_points(rng, C, centers, span / 10, span; rural_fraction=0.0)

    demands = [exp(rand(rng, Normal(log((min_demand + max_demand) / 2), 0.5))) for _ in 1:C]
    total_demand = sum(demands)
    avg_capacity = total_demand / F * capacity_factor

    capacities = Vector{Float64}(undef, F)
    fixed_costs = Vector{Float64}(undef, F)
    for w in 1:F
        capacities[w] = avg_capacity * (0.8 + 0.4 * rand(rng))
        location_factor = 1.0 + 0.2 * (facility_locs[w][1] + facility_locs[w][2]) / span
        fixed_costs[w] = clamp(
            location_factor *
            (fixed_cost_min + capacities[w] / avg_capacity * (fixed_cost_max - fixed_cost_min) / 2),
            fixed_cost_min,
            fixed_cost_max,
        )
    end

    shipping_costs = Matrix{Float64}(undef, F, C)
    for c in 1:C, w in 1:F
        d = _fl_dist(facility_locs[w], customer_locs[c])
        shipping_costs[w, c] = d * transport_cost_per_km * (0.9 + 0.2 * rand(rng))
    end

    budget = sum(fixed_costs) * budget_factor
    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        if sum(capacities) < 1.05 * total_demand
            capacities .*= 1.05 * total_demand / sum(capacities)
        end
        order = sortperm(capacities ./ fixed_costs; rev=true)
        open = Int[]
        covered = 0.0
        for w in order
            push!(open, w)
            covered += capacities[w]
            covered >= total_demand && break
        end
        budget = max(budget, sum(fixed_costs[open]) * (1.02 + 0.23 * rand(rng)))
        # Serve customers (largest first) from their nearest open facilities
        # with spare capacity, splitting where needed. Open capacity covers
        # total demand, so every customer is served in full.
        residual = Dict(w => capacities[w] for w in open)
        shipments = Tuple{Int, Int, Float64}[]
        for c in sortperm(demands; rev=true)
            need = demands[c]
            for w in sort(open; by=w -> shipping_costs[w, c])
                need <= 0 && break
                q = min(need, residual[w])
                q <= 0 && continue
                push!(shipments, (w, c, q))
                residual[w] -= q
                need -= q
            end
        end
        witness = FacilityLocationWitness(sort!(open), shipments)
    elseif feasibility_status == infeasible
        # Smallest budget that funds capacity == total demand fractionally.
        order = sortperm(capacities ./ fixed_costs; rev=true)
        threshold = 0.0
        reached = 0.0
        for w in order
            if reached + capacities[w] >= total_demand
                threshold += fixed_costs[w] * (total_demand - reached) / capacities[w]
                reached = total_demand
                break
            end
            reached += capacities[w]
            threshold += fixed_costs[w]
        end
        if reached < total_demand
            # Even opening everything falls short; any budget is infeasible.
            threshold = sum(fixed_costs)
        end
        budget = min(budget, threshold * rand(rng, 0.75:0.01:0.95))
        fundable = _fl_fundable_capacity(capacities, fixed_costs, budget)
        certificate = FacilityBudgetCertificate(budget, fundable, total_demand)
    end

    return FacilityLocationProblem(
        F,
        C,
        facility_locs,
        customer_locs,
        demands,
        fixed_costs,
        capacities,
        shipping_costs,
        budget,
        witness,
        certificate,
    )
end

"""
    build_model(prob::FacilityLocationProblem)

Build the strong capacitated facility location model. Deterministic — uses only
data from the struct fields.
"""
function build_model(prob::FacilityLocationProblem)
    model = Model()
    F, C = prob.n_facilities, prob.n_customers

    @variable(model, y[1:F], Bin)
    @variable(model, x[1:F, 1:C] >= 0)

    @objective(
        model,
        Min,
        sum(prob.fixed_costs[w] * y[w] for w in 1:F) +
            sum(prob.shipping_costs[w, c] * x[w, c] for w in 1:F, c in 1:C)
    )
    for c in 1:C
        @constraint(model, sum(x[w, c] for w in 1:F) >= prob.demands[c])
    end
    for w in 1:F
        @constraint(model, sum(x[w, c] for c in 1:C) <= prob.capacities[w] * y[w])
    end
    for c in 1:C, w in 1:F
        @constraint(model, x[w, c] <= prob.demands[c] * y[w])
    end
    @constraint(model, sum(prob.fixed_costs[w] * y[w] for w in 1:F) <= prob.budget)
    return model
end

register_variant(
    :facility_location,
    :standard,
    FacilityLocationProblem,
    "Budgeted capacitated facility location in the strong formulation: open facilities and ship to clustered customers with disaggregated x ≤ d·y linking";
    default=true,
    tags=[:location, :bipartite, :big_m],
)
