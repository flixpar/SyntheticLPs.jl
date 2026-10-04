using JuMP
using Random
using Distributions

"""
    TwoEchelonWitness

Planted feasible plan for [`TwoEchelonFacilityLocationProblem`](@ref): the open
warehouses, the size installed at each (`0` when closed), and one flow per
inbound (`supply_flow`, aligned with `in_supplier`/`in_warehouse`) and outbound
(`delivery_flow`, aligned with `out_warehouse`/`out_customer`) lane. Integral in
`y`/`z`, so it is feasible for the MIP and for every relaxation.
"""
struct TwoEchelonWitness
    open::Vector{Int}
    size_choice::Vector{Int}
    supply_flow::Vector{Float64}
    delivery_flow::Vector{Float64}
end

"""
    TwoEchelonRegionalDeficit

LP-row infeasibility certificate for [`TwoEchelonFacilityLocationProblem`](@ref).
`customers` is a demand region `R` and `warehouses` is `N(R)`, every warehouse
with a delivery lane into `R`. Summing the demand rows of `R`, the delivery
lanes are bounded by the outflow of `N(R)`; conservation equates outflow with
inflow, the throughput rows bound inflow by `Σ_k cap[w,k] z[w,k]`, and the size
rows with `y ≤ 1` bound that by `max_k cap[w,k]`:

    region_demand = Σ_{c∈R} d_c ≤ Σ_{w∈N(R)} max_k cap[w,k] = max_capacity

The generator makes `region_demand ≥ 1.1 · max_capacity`, so the model is
infeasible for fractional `y`/`z` too. The argument aggregates `|R|` demand rows
and `3|N(R)|` warehouse rows, so presolve does not see it in a single row.
"""
struct TwoEchelonRegionalDeficit
    customers::Vector{Int}
    warehouses::Vector{Int}
    region_demand::Float64
    max_capacity::Float64
end

"""
    TwoEchelonFacilityLocationProblem <: ProblemGenerator

Two-echelon capacitated facility location with discrete warehouse sizing over a
sparse plant → distribution-center → customer network.

# Overview

Plants (suppliers) ship to candidate distribution centers (warehouses), which
deliver to customers. Decisions: which warehouses to open (`y`), which discrete
size to build at each open one (`z`), and the flows on the sparse lane sets —
each warehouse is sourced from its `L` nearest plants (`f1`) and each customer
can be served from its `K_c` nearest warehouses (`f2`). Lanes follow real
practice: nobody ships pallets across the continent to a customer when a closer
DC exists, which keeps the model sparse and lets it scale to millions of lanes.

Constraints:

  - size choice `Σ_k z[w,k] = y[w]`;
  - plant capacity `Σ_{lanes out of s} f1 ≤ supply_s`;
  - customer demand `Σ_{lanes into c} f2 ≥ d_c`;
  - throughput `Σ_{lanes into w} f1 ≤ Σ_k cap[w,k] z[w,k]`;
  - cross-dock conservation `Σ_{lanes into w} f1 = Σ_{lanes out of w} f2`;
  - strong (disaggregated) linking `f2[w→c] ≤ d_c · y[w]` on every delivery
    lane, the textbook strengthening that keeps the opening decisions binding in
    the LP relaxation (the aggregate form lets `y` go to `throughput / cap`).

Size options have concave-with-noise installation costs (economies of scale
with site-specific construction costs), so several sizes stay on the lower
envelope rather than one dominating.

# Fields

  - `n_warehouses`, `n_suppliers`, `n_customers`
  - `warehouse_locations`, `supplier_locations`, `customer_locations`
  - `supplier_capacities::Vector{Float64}`, `customer_demands::Vector{Float64}`
  - `warehouse_fixed_costs::Vector{Float64}`
  - `size_capacity::Matrix{Float64}`, `size_cost::Matrix{Float64}`: `W × K`
    per-warehouse size options (sorted increasing in capacity)
  - `handling_costs::Vector{Float64}`: per-unit cross-dock cost at each warehouse
  - `in_supplier`, `in_warehouse`, `in_cost`: inbound lanes (plant → DC)
  - `out_warehouse`, `out_customer`, `out_cost`: delivery lanes (DC → customer),
    grouped by customer
  - `feasible_witness::Union{Nothing,TwoEchelonWitness}`
  - `infeasibility_certificate::Union{Nothing,TwoEchelonRegionalDeficit}`
"""
struct TwoEchelonFacilityLocationProblem <: ProblemGenerator
    n_warehouses::Int
    n_suppliers::Int
    n_customers::Int
    warehouse_locations::Vector{Tuple{Float64, Float64}}
    supplier_locations::Vector{Tuple{Float64, Float64}}
    customer_locations::Vector{Tuple{Float64, Float64}}
    supplier_capacities::Vector{Float64}
    customer_demands::Vector{Float64}
    warehouse_fixed_costs::Vector{Float64}
    size_capacity::Matrix{Float64}
    size_cost::Matrix{Float64}
    handling_costs::Vector{Float64}
    in_supplier::Vector{Int}
    in_warehouse::Vector{Int}
    in_cost::Vector{Float64}
    out_warehouse::Vector{Int}
    out_customer::Vector{Int}
    out_cost::Vector{Float64}
    feasible_witness::Union{Nothing, TwoEchelonWitness}
    infeasibility_certificate::Union{Nothing, TwoEchelonRegionalDeficit}
end

const _TWO_ECHELON_SIZE_MULTS = Dict(3 => [0.5, 1.0, 1.7], 4 => [0.4, 0.75, 1.2, 1.8])

# Dimension plan: returns (W, K_sizes, L, lanes_per_customer_vector).
# Variable count = W*(1 + K) + W*L + Σ_c lanes_c, hit exactly whenever the
# target leaves room for at least one customer.
function _two_echelon_dimensions(rng::AbstractRNG, target::Int)
    n_sizes = rand(rng, 3:4)
    n_inbound = rand(rng, 2:3)
    lanes = rand(rng, 3:6)
    customers_per_dc = rand(rng, 8:16)
    W = max(2, round(Int, target / (1 + n_sizes + n_inbound + customers_per_dc * lanes)))
    n_suppliers = max(1, round(Int, W / rand(rng, 4:8)))
    n_inbound = min(n_inbound, n_suppliers)
    lanes = min(lanes, W)
    remaining = target - W * (1 + n_sizes + n_inbound)
    if remaining < 2
        # Tiny targets: shrink the per-warehouse blocks, accept a small overshoot.
        n_sizes, n_inbound = 3, 1
        lanes = min(lanes, 2)
        remaining = max(lanes, target - W * (1 + n_sizes + n_inbound))
    end
    n_customers = cld(remaining, lanes)
    lanes_per_customer = fill(lanes, n_customers)
    # Trim lanes (keeping at least one per customer) to land exactly.
    deficit = n_customers * lanes - remaining
    c = 1
    while deficit > 0 && any(>(1), lanes_per_customer)
        if lanes_per_customer[c] > 1
            lanes_per_customer[c] -= 1
            deficit -= 1
        end
        c = mod1(c + 1, n_customers)
    end
    return W, n_suppliers, n_sizes, n_inbound, lanes_per_customer
end

"""
    TwoEchelonFacilityLocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a sparse two-echelon facility location and sizing instance.

# Variable-count formula

    total = W·(1 + K) + W·L + Σ_c K_c

(`y`, `z`, inbound lanes, delivery lanes) with `K ∈ {3,4}` size options,
`L ∈ {2,3}` inbound lanes per warehouse, `8–16` customers per warehouse and
`K_c ∈ 3..6` delivery lanes per customer. The customer count and per-customer
lane counts are chosen so `total == target_variables` exactly for every target
above ~15 (tiny targets overshoot by a few variables). Build time is
near-linear (nearest-site queries use a bucket grid), so there is no size cap.

# Feasibility

  - `feasible`: a plan is planted — a random 50–80% of warehouses open, every
    customer is served in full by its nearest open admissible warehouse, each
    open warehouse buys the smallest size covering its throughput with a 5–15%
    margin (its size ladder is raised when even the largest size is short), and
    each open warehouse is sourced from its nearest plant, whose capacity is
    raised to 1.1–1.3× its planted load where needed. Stored as a
    [`TwoEchelonWitness`](@ref).
  - `infeasible`: natural data plus a regional deficit: a compact region `R`
    (4–12% of customers around a random customer) sees a 30–80% demand surge
    and zoning limits on the DCs `N(R)` that can reach it, leaving
    `Σ_R d ≥ 1.1–1.3 × Σ_{N(R)} max_k cap[w,k]`. Stored as a
    [`TwoEchelonRegionalDeficit`](@ref); valid for the LP relaxation.
  - `unknown`: natural data — size ladders and plant capacities are planned
    against a demand *forecast* (ladders top out at ~1.5–2.3× each DC's
    catchment; plants at 1.0–1.4× their planned load), while realized demand
    carries 1–3 spatially correlated regional shocks of 1.2–2.8×. Whether
    neighbouring DCs and plants can absorb a shock is left to the data, so the
    instance is genuinely two-sided (about 80% feasible across seeds and sizes).
"""
function TwoEchelonFacilityLocationProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    W, S, n_sizes, n_inbound, lanes_per_customer = _two_echelon_dimensions(rng, target_variables)
    C = length(lanes_per_customer)
    max_lanes = maximum(lanes_per_customer)

    # --- Geography: customers in metro clusters, DC sites near metros,
    # plants in a few manufacturing zones ---
    span = rand(rng, 600.0:100.0:1500.0)
    n_metros = clamp(round(Int, sqrt(W) * rand(rng, 0.8:0.1:1.5)), 2, 60)
    metros = [(span * rand(rng), span * rand(rng)) for _ in 1:n_metros]
    metro_spread = span / (2.5 * sqrt(n_metros))
    customer_locations = _fl_clustered_points(rng, C, metros, metro_spread, span)
    warehouse_locations = _fl_clustered_points(
        rng, W, metros, 1.5 * metro_spread, span; rural_fraction=0.35
    )
    n_zones = clamp(round(Int, sqrt(S)), 1, 12)
    zones = [(span * rand(rng), span * rand(rng)) for _ in 1:n_zones]
    supplier_locations = _fl_clustered_points(
        rng, S, zones, span / 12, span; rural_fraction=0.25
    )

    # --- Sparse lanes ---
    nearest_dc = _fl_nearest_sites(warehouse_locations, customer_locations, max_lanes)
    nearest_plant = _fl_nearest_sites(supplier_locations, warehouse_locations, n_inbound)
    inbound_rate = rand(rng, 0.015:0.005:0.04)      # full-truckload, per unit-km
    delivery_rate = rand(rng, 0.06:0.01:0.15)       # less-than-truckload, per unit-km
    out_warehouse = Int[]
    out_customer = Int[]
    out_cost = Float64[]
    sizehint!(out_warehouse, sum(lanes_per_customer))
    for c in 1:C, w in nearest_dc[c][1:lanes_per_customer[c]]
        push!(out_warehouse, w)
        push!(out_customer, c)
        d = _fl_dist(customer_locations[c], warehouse_locations[w])
        push!(out_cost, round(delivery_rate * (5.0 + d) * (0.9 + 0.2 * rand(rng)); digits=3))
    end
    in_supplier = Int[]
    in_warehouse = Int[]
    in_cost = Float64[]
    for w in 1:W, s in nearest_plant[w]
        push!(in_supplier, s)
        push!(in_warehouse, w)
        d = _fl_dist(supplier_locations[s], warehouse_locations[w])
        push!(in_cost, round(inbound_rate * (10.0 + d) * (0.9 + 0.2 * rand(rng)); digits=3))
    end

    # --- Demand forecast (lognormal, heavier in big metros) ---
    forecast = round.(exp.(rand(rng, Normal(log(120.0), 0.7), C)); digits=2)
    forecast .= max.(forecast, 1.0)

    # --- Size ladders planned against the forecast catchment ---
    catchment = zeros(W)
    for c in 1:C
        catchment[nearest_dc[c][1]] += forecast[c]
    end
    avg_catchment = sum(forecast) / W
    mults = _TWO_ECHELON_SIZE_MULTS[n_sizes]
    base = [max(catchment[w], 0.3 * avg_catchment) * rand(rng, Uniform(0.85, 1.25)) for w in 1:W]
    size_capacity = [round(base[w] * mults[k]; digits=2) for w in 1:W, k in 1:n_sizes]

    # --- Realized demand: `unknown` adds a few spatially correlated regional
    # shocks (a fixed handful of hot spots, so the chance that one of them
    # overwhelms its local capacity does not drift with instance size) ---
    demands = copy(forecast)
    if feasibility_status == unknown
        for _ in 1:rand(rng, 1:3)
            center = customer_locations[rand(rng, 1:C)]
            radius = metro_spread * rand(rng, Uniform(0.5, 1.5))
            surge = rand(rng, Uniform(1.2, 2.8))
            for c in 1:C
                _fl_dist(customer_locations[c], center) <= radius && (demands[c] *= surge)
            end
        end
        demands .= round.(demands; digits=2)
    end

    # --- Plant capacity, planned against the forecast load each plant would
    # carry if every DC were replenished from its nearest plant ---
    planned_load = zeros(S)
    for w in 1:W
        planned_load[nearest_plant[w][1]] += max(catchment[w], 0.3 * avg_catchment)
    end
    supply_factor = feasibility_status == unknown ? rand(rng, Uniform(1.0, 1.4)) : 1.25
    supplier_capacities = round.(
        max.(planned_load, 0.3 * sum(forecast) / S) .* supply_factor .*
        rand(rng, Uniform(0.9, 1.2), S);
        digits=2,
    )

    # --- Costs: fixed opening, size installation, handling ---
    spacing = span / sqrt(W)
    site_factor = rand(rng, Uniform(0.7, 1.4), W)
    warehouse_fixed_costs = round.(
        site_factor .* delivery_rate .* spacing .* max.(base, 1.0) .* rand(rng, Uniform(0.4, 1.2));
        digits=2,
    )
    handling_costs = round.(rand(rng, Uniform(0.5, 2.5), W); digits=3)

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        p_open = rand(rng, Uniform(0.5, 0.8))
        is_open = rand(rng, W) .< p_open
        # Every customer needs an open admissible DC; open the nearest if not.
        lane_start = cumsum(vcat(1, lanes_per_customer))
        for c in 1:C
            any(is_open[out_warehouse[l]] for l in lane_start[c]:(lane_start[c + 1] - 1)) ||
                (is_open[out_warehouse[lane_start[c]]] = true)
        end
        delivery_flow = zeros(length(out_warehouse))
        throughput = zeros(W)
        for c in 1:C
            # Lanes are sorted by distance, so the first open one is nearest.
            l = findfirst(l -> is_open[out_warehouse[l]], lane_start[c]:(lane_start[c + 1] - 1))
            lane = lane_start[c] + l - 1
            delivery_flow[lane] = demands[c]
            throughput[out_warehouse[lane]] += demands[c]
        end
        size_choice = zeros(Int, W)
        for w in 1:W
            is_open[w] || continue
            need = throughput[w] * rand(rng, Uniform(1.05, 1.15))
            if size_capacity[w, end] < need
                scale = need * rand(rng, Uniform(1.0, 1.2)) / size_capacity[w, end]
                size_capacity[w, :] .= round.(size_capacity[w, :] .* scale; digits=2)
                size_capacity[w, end] < need && (size_capacity[w, end] = ceil(need))
            end
            size_choice[w] = findfirst(k -> size_capacity[w, k] >= need, 1:n_sizes)
        end
        # Each open DC is replenished from its nearest plant (first inbound lane).
        supply_flow = zeros(length(in_supplier))
        plant_load = zeros(S)
        for w in 1:W
            is_open[w] || continue
            lane = (w - 1) * n_inbound + 1
            supply_flow[lane] = throughput[w]
            plant_load[in_supplier[lane]] += throughput[w]
        end
        for s in 1:S
            need = plant_load[s] * rand(rng, Uniform(1.1, 1.3))
            supplier_capacities[s] < need && (supplier_capacities[s] = round(need + 0.01; digits=2))
        end
        witness = TwoEchelonWitness(findall(is_open), size_choice, supply_flow, delivery_flow)
    elseif feasibility_status == infeasible
        center = customer_locations[rand(rng, 1:C)]
        m = clamp(round(Int, rand(rng, Uniform(0.04, 0.12)) * C), 1, C)
        region = sort!(partialsortperm([_fl_dist(p, center) for p in customer_locations], 1:m))
        in_region = falses(C)
        in_region[region] .= true
        reach = sort!(unique(out_warehouse[l] for l in eachindex(out_customer) if in_region[out_customer[l]]))
        surge = rand(rng, Uniform(1.3, 1.8))
        for c in region
            demands[c] = round(demands[c] * surge; digits=2)
        end
        region_demand = sum(demands[region])
        margin = rand(rng, Uniform(1.1, 1.3))
        max_capacity = sum(size_capacity[w, end] for w in reach)
        if region_demand < margin * max_capacity
            # Zoning limits on the DC footprint in the region's catchment.
            g = region_demand / (margin * max_capacity)
            for w in reach
                size_capacity[w, :] .= floor.(size_capacity[w, :] .* g; digits=2)
            end
            max_capacity = sum(size_capacity[w, end] for w in reach)
        end
        certificate = TwoEchelonRegionalDeficit(region, reach, region_demand, max_capacity)
    end

    # Installation cost: concave in capacity with site-specific noise.
    size_cost = [
        round(
            0.5 * site_factor[w] * delivery_rate * spacing * size_capacity[w, k]^0.85 *
            max(base[w], 1.0)^0.15 * rand(rng, Uniform(0.9, 1.15));
            digits=2,
        ) for w in 1:W, k in 1:n_sizes
    ]

    return TwoEchelonFacilityLocationProblem(
        W,
        S,
        C,
        warehouse_locations,
        supplier_locations,
        customer_locations,
        supplier_capacities,
        demands,
        warehouse_fixed_costs,
        size_capacity,
        size_cost,
        handling_costs,
        in_supplier,
        in_warehouse,
        in_cost,
        out_warehouse,
        out_customer,
        out_cost,
        witness,
        certificate,
    )
end

"""
    build_model(prob::TwoEchelonFacilityLocationProblem)

Build the sparse two-echelon facility location and sizing model. Deterministic —
uses only data from the struct fields.
"""
function build_model(prob::TwoEchelonFacilityLocationProblem)
    model = Model()
    W, S, C = prob.n_warehouses, prob.n_suppliers, prob.n_customers
    K = size(prob.size_capacity, 2)
    n_in = length(prob.in_supplier)
    n_out = length(prob.out_customer)

    @variable(model, y[1:W], Bin)
    @variable(model, z[1:W, 1:K], Bin)
    @variable(model, f1[1:n_in] >= 0)
    @variable(model, f2[1:n_out] >= 0)

    inbound_of = [Int[] for _ in 1:W]
    outbound_of = [Int[] for _ in 1:W]
    supplier_lanes = [Int[] for _ in 1:S]
    customer_lanes = [Int[] for _ in 1:C]
    for l in 1:n_in
        push!(inbound_of[prob.in_warehouse[l]], l)
        push!(supplier_lanes[prob.in_supplier[l]], l)
    end
    for l in 1:n_out
        push!(outbound_of[prob.out_warehouse[l]], l)
        push!(customer_lanes[prob.out_customer[l]], l)
    end

    @objective(
        model,
        Min,
        sum(prob.warehouse_fixed_costs[w] * y[w] for w in 1:W) +
            sum(prob.size_cost[w, k] * z[w, k] for w in 1:W, k in 1:K) +
            sum(prob.in_cost[l] * f1[l] for l in 1:n_in) +
            sum((prob.out_cost[l] + prob.handling_costs[prob.out_warehouse[l]]) * f2[l] for l in 1:n_out)
    )

    for w in 1:W
        @constraint(model, sum(z[w, k] for k in 1:K) == y[w])
        @constraint(
            model,
            sum(f1[l] for l in inbound_of[w]) <=
                sum(prob.size_capacity[w, k] * z[w, k] for k in 1:K)
        )
        @constraint(model, sum(f1[l] for l in inbound_of[w]) == sum(f2[l] for l in outbound_of[w]))
    end
    for s in 1:S
        isempty(supplier_lanes[s]) && continue
        @constraint(model, sum(f1[l] for l in supplier_lanes[s]) <= prob.supplier_capacities[s])
    end
    for c in 1:C
        @constraint(model, sum(f2[l] for l in customer_lanes[c]) >= prob.customer_demands[c])
    end
    for l in 1:n_out
        c = prob.out_customer[l]
        @constraint(model, f2[l] <= prob.customer_demands[c] * y[prob.out_warehouse[l]])
    end
    return model
end

register_variant(
    :facility_location,
    :two_echelon,
    TwoEchelonFacilityLocationProblem,
    "Two-echelon plant→DC→customer facility location with discrete DC sizing, sparse nearest-site lanes, and strong delivery-lane linking";
    tags=[:location, :network, :big_m],
)
