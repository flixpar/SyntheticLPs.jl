using JuMP
using Random
using StatsBase
using Distributions

"""
    CarbonSupplyChainProblem <: ProblemGenerator

Generator for supply chain optimization problems with a global transportation
carbon budget on top of the standard facility-location/transport formulation.

This problem models realistic supply chain networks with:
- Geographic clustering of customers and facilities
- Multiple transportation modes with infrastructure availability
- K-nearest connectivity guarantees for feasible instances
- Facility opening costs and capacities
- Mode-specific capacity constraints
- A single global carbon-emission budget on all shipments

# Overview
Models strategic supply-chain network design under a carbon cap. The decisions
open facilities and ship customer demand from open facilities over available
transportation modes. The objective minimizes fixed facility cost plus
mode-specific transportation cost. Constraints satisfy customer demand, gate
shipments by open facility capacity, limit aggregate shipment volume by
transportation mode, and bound total transportation carbon emissions by a global
carbon budget.

Per-arc emissions are `emission_rate(mode) * distance`, where the emission rate
is mode-dependent (truck 0.1, rail 0.03, ship 0.02, air 0.5). Because cleaner
modes (rail/ship) tend to be more expensive or less available than trucking, the
carbon budget is deliberately derived from a feasible reference flow so that the
cap is reliably *active*: the cost-minimizing solution must shift volume toward
lower-emission arcs to respect it.

# Fields
All data generated in constructor based on target_variables and feasibility_status:
- `n_facilities::Int`: Number of potential facility locations
- `n_customers::Int`: Number of customer locations
- `transport_modes::Vector{String}`: Selected transport modes
- `facility_locs::Vector{Tuple{Float64,Float64}}`: Geographic facility locations
- `customer_locs::Vector{Tuple{Float64,Float64}}`: Geographic customer locations
- `cluster_centers::Vector{Tuple{Float64,Float64}}`: Cluster centers for customer distribution
- `cluster_weights::Vector{Float64}`: Weights for cluster importance
- `fixed_costs::Dict{Int, Float64}`: Fixed cost to open each facility
- `demands::Dict{Int, Float64}`: Demand at each customer location
- `capacities::Dict{Int, Float64}`: Capacity of each facility
- `transport_costs::Dict{Tuple{Int,Int,String}, Float64}`: Transport cost per (facility, customer, mode)
- `mode_capacities::Dict{String, Float64}`: Total capacity available for each transport mode
- `carbon_emissions::Dict{Tuple{Int,Int,String}, Float64}`: Carbon emission per unit per (facility, customer, mode)
- `carbon_limit::Float64`: Global transportation carbon budget
- `total_demand::Float64`: Total demand across all customers
"""
struct CarbonSupplyChainProblem <: ProblemGenerator
    n_facilities::Int
    n_customers::Int
    transport_modes::Vector{String}
    facility_locs::Vector{Tuple{Float64,Float64}}
    customer_locs::Vector{Tuple{Float64,Float64}}
    cluster_centers::Vector{Tuple{Float64,Float64}}
    cluster_weights::Vector{Float64}
    fixed_costs::Dict{Int, Float64}
    demands::Dict{Int, Float64}
    capacities::Dict{Int, Float64}
    transport_costs::Dict{Tuple{Int,Int,String}, Float64}
    mode_capacities::Dict{String, Float64}
    carbon_emissions::Dict{Tuple{Int,Int,String}, Float64}
    carbon_limit::Float64
    total_demand::Float64
end

"""
    CarbonSupplyChainProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a carbon-constrained supply chain problem instance with sophisticated
geographic clustering and connectivity logic, plus a global transport carbon
budget.

# Variable count
Same arc set as the standard supply-chain variant:
`n_facilities` binary `y` variables + one continuous `x` per available
`(facility, customer, mode)` arc. The arc set is sized via `n_facilities`,
`n_customers`, `n_transport_modes`, and `infrastructure_density` so that the
total approximates `target_variables`.

# Carbon budget derivation
The carbon budget is anchored to a feasible reference flow (each customer served
by its lowest-emission available arc). The cap is set just below that reference's
emissions so the budget is reliably active without precluding feasibility, since
a feasible (lower-emission) assignment always exists by construction.

# Feasibility logic
- `feasible`: K-nearest connectivity and capacity smoothing as in the standard
  variant, plus a carbon budget guaranteed to admit the minimum-emission flow.
- `infeasible`: carbon budget set strictly below a valid lower bound on the
  emissions of *any* feasible flow (sum over customers of demand times the
  cheapest available per-unit emission), guaranteeing a deterministic
  contradiction with margin.
- `unknown`: natural instance with the reference-anchored budget; no forced
  infeasibility.

# Arguments
- `target_variables`: Target number of variables (approximately n_facilities × n_customers × n_transport_modes × infrastructure_density, plus n_facilities binaries)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function CarbonSupplyChainProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem dimensions based on target variables.
    # VARIABLE COUNT FORMULA:
    #   vars = n_facilities (binary y) + (# available arcs) (continuous x)
    #   # available arcs ≈ n_facilities * n_customers * n_transport_modes * density
    # We size dimensions analytically to hit target. Because per-arc availability
    # is high for truck (0.98) and we multiply by infrastructure_density, the
    # effective fill of the arc tensor is ≈ eff_density below; nf and nc are
    # chosen so that nf + nf*nc*nm*eff_density ≈ target.
    if target_variables <= 250
        n_transport_modes = 1
        grid_width = rand(Uniform(200.0, 800.0))
        grid_height = rand(Uniform(200.0, 800.0))
        infrastructure_density = rand(Beta(5, 2)) * 0.2 + 0.8  # 0.8-1.0
        clustering_factor = rand(Beta(3, 2)) * 0.6 + 0.25
        min_fixed_cost = max(100000.0, rand(LogNormal(log(300000), 0.5)))
        max_fixed_cost = min_fixed_cost * rand(Uniform(1.8, 3.5))
        base_demand = rand(Uniform(80.0, 150.0))
    elseif target_variables <= 1000
        n_transport_modes = 2
        grid_width = rand(Uniform(800.0, 2000.0))
        grid_height = rand(Uniform(800.0, 2000.0))
        infrastructure_density = rand(Beta(3, 2)) * 0.3 + 0.6  # 0.6-0.9
        clustering_factor = rand(Beta(2, 3)) * 0.5 + 0.2
        min_fixed_cost = max(300000.0, rand(LogNormal(log(800000), 0.6)))
        max_fixed_cost = min_fixed_cost * rand(Uniform(2.0, 4.0))
        base_demand = rand(Uniform(150.0, 300.0))
    else
        n_transport_modes = 3
        grid_width = rand(Uniform(2000.0, 5000.0))
        grid_height = rand(Uniform(2000.0, 5000.0))
        infrastructure_density = rand(Beta(2, 3)) * 0.3 + 0.5  # 0.5-0.8
        clustering_factor = rand(Beta(1, 3)) * 0.4 + 0.15
        min_fixed_cost = max(500000.0, rand(LogNormal(log(1500000), 0.7)))
        max_fixed_cost = min_fixed_cost * rand(Uniform(2.5, 5.0))
        base_demand = rand(Uniform(300.0, 600.0))
    end
    min_demand = base_demand
    max_demand = base_demand * rand(Uniform(3.0, 8.0))

    # Estimated effective fill of the (f,c,mode) arc tensor (empirically
    # calibrated). Truck (always present) fills ~0.55*density of its slice;
    # additional modes are much sparser. eff_density is the average fill across
    # the n_transport_modes slices, used to predict the available-arc count.
    truck_fill = 0.95 * infrastructure_density
    extra_fill = 0.30 * infrastructure_density
    eff_density = (truck_fill + (n_transport_modes - 1) * extra_fill) / n_transport_modes

    # Solve nf + nf*nc*nm*eff_density = target with a balanced aspect ratio.
    # Choose nf ≈ sqrt(target / (nm*eff_density)) * ratio, then nc from target.
    arc_target = max(1.0, target_variables - 0.0)  # binaries are a small share
    ratio = 0.5 + rand() * 0.6  # 0.5-1.1
    n_facilities = max(2, round(Int, sqrt(arc_target / (n_transport_modes * eff_density)) * ratio))
    n_customers = max(3, round(Int,
        (target_variables - n_facilities) / (n_facilities * n_transport_modes * eff_density)))
    n_facilities = min(n_facilities, 60)
    n_customers = min(n_customers, 400)

    # Additional parameters
    capacity_factor = rand(Uniform(1.2, 2.2))
    mode_capacity_factor = rand(Uniform(0.25, 0.65))

    # Mode-specific carbon emission rates (per unit per distance).
    emission_rate(mode) = mode == "truck" ? 0.1 :
                          mode == "rail" ? 0.03 :
                          mode == "ship" ? 0.02 : 0.5  # air

    # Transport modes and costs
    all_transport_modes = ["truck", "rail", "ship", "air"]
    transport_base_costs = Dict(
        "truck" => rand(Gamma(4, 0.25)),
        "rail" => rand(Gamma(3, 0.2)),
        "ship" => rand(Gamma(2, 0.15)),
        "air" => rand(Gamma(6, 0.5))
    )

    # Select transport modes. Truck (the high-availability backbone) is always
    # included so the available-arc count is predictable; additional, sparser
    # modes are sampled from the rest.
    extra_modes = sample(["rail", "ship", "air"], min(n_transport_modes - 1, 3), replace=false)
    transport_modes = vcat(["truck"], extra_modes)

    # Geographic clusters
    n_clusters = max(2, round(Int, sqrt(n_customers) * clustering_factor))
    cluster_centers = [(grid_width * rand(), grid_height * rand()) for _ in 1:n_clusters]

    # Facility locations (more dispersed than customers)
    facility_locs = Vector{Tuple{Float64,Float64}}()
    for _ in 1:n_facilities
        if rand() < 0.4
            center = rand(cluster_centers)
            spread_x = grid_width * 0.12
            spread_y = grid_height * 0.12
            x = clamp(center[1] + rand(Normal(0, spread_x)), 0, grid_width)
            y = clamp(center[2] + rand(Normal(0, spread_y)), 0, grid_height)
        else
            x = grid_width * rand(Beta(1.5, 1.5))
            y = grid_height * rand(Beta(1.5, 1.5))
        end
        push!(facility_locs, (x, y))
    end

    # Customer locations (more clustered)
    customer_locs = Vector{Tuple{Float64,Float64}}()
    cluster_weights = rand(Dirichlet(ones(n_clusters)))
    for _ in 1:n_customers
        cluster_idx = sample(1:n_clusters, Weights(cluster_weights))
        center = cluster_centers[cluster_idx]
        base_spread = grid_width * (1 - clustering_factor) * 0.08
        spread = rand(LogNormal(log(base_spread), 0.3))
        x = clamp(center[1] + rand(Normal(0, spread)), 0, grid_width)
        y = clamp(center[2] + rand(Normal(0, spread)), 0, grid_height)
        push!(customer_locs, (x, y))
    end

    # Distance helper
    dist(f, c) = sqrt((facility_locs[f][1] - customer_locs[c][1])^2 +
                      (facility_locs[f][2] - customer_locs[c][2])^2)

    # Facility fixed costs (correlated with location and market size)
    fixed_costs = Dict{Int, Float64}()
    for f in 1:n_facilities
        distances_to_customers = [dist(f, c) for c in 1:n_customers]
        market_potential = sum(exp.(-distances_to_customers ./ (grid_width * 0.2)))
        location_factor = (facility_locs[f][1] / grid_width + facility_locs[f][2] / grid_height) / 2
        base_cost = min_fixed_cost + (max_fixed_cost - min_fixed_cost) *
                   (0.2 + 0.5 * market_potential / n_customers + 0.3 * location_factor)
        fixed_costs[f] = base_cost * rand(LogNormal(log(1.0), 0.25))
    end

    # Customer demands (correlated with cluster size)
    demands = Dict{Int, Float64}()
    for c in 1:n_customers
        distances_to_clusters = [
            sqrt((customer_locs[c][1] - center[1])^2 + (customer_locs[c][2] - center[2])^2)
            for center in cluster_centers
        ]
        _, cluster_idx = findmin(distances_to_clusters)
        cluster_influence = cluster_weights[cluster_idx]
        base_demand_val = min_demand + (max_demand - min_demand) * (0.2 + 0.8 * cluster_influence)
        demands[c] = base_demand_val * rand(LogNormal(log(1.0), 0.4))
    end

    # Facility capacities
    total_demand = sum(values(demands))
    avg_capacity = (total_demand / n_facilities) * capacity_factor
    capacities = Dict{Int, Float64}()
    for f in 1:n_facilities
        relative_cost = (fixed_costs[f] - minimum(values(fixed_costs))) /
                       max(1.0, maximum(values(fixed_costs)) - minimum(values(fixed_costs)))
        base_capacity = avg_capacity * (0.6 + 0.8 * relative_cost)
        capacities[f] = base_capacity * rand(Gamma(3, 1/3))
    end

    # Transport costs, emissions, and infrastructure availability
    transport_costs = Dict{Tuple{Int,Int,String}, Float64}()
    carbon_emissions = Dict{Tuple{Int,Int,String}, Float64}()
    infrastructure = Dict{Tuple{Int,Int,String}, Bool}()
    max_demand_val = maximum(values(demands))

    for f in 1:n_facilities
        for c in 1:n_customers
            distance = dist(f, c)
            for mode in transport_modes
                prob_available = if mode == "truck"
                    0.98
                elseif mode == "rail"
                    min(0.8, 0.3 + 0.5 * (distance / sqrt(grid_width^2 + grid_height^2)))
                elseif mode == "ship"
                    any(loc -> abs(loc[2]) < grid_height * 0.1, [facility_locs[f], customer_locs[c]]) ? 0.8 : 0.0
                else  # air
                    distance > sqrt(grid_width^2 + grid_height^2) * 0.3 ? 0.7 : 0.2
                end

                infrastructure[(f,c,mode)] = rand() < prob_available * infrastructure_density

                if infrastructure[(f,c,mode)]
                    base_cost = get(transport_base_costs, mode, 1.0)
                    terrain_factor = rand(LogNormal(log(1.0), 0.15))
                    volume_factor = 1.0 - 0.25 * (demands[c] / max_demand_val)
                    efficiency_factor = rand(Beta(3, 2)) * 0.4 + 0.8
                    transport_costs[(f,c,mode)] = base_cost * distance * terrain_factor * volume_factor * efficiency_factor
                    carbon_emissions[(f,c,mode)] = emission_rate(mode) * distance * rand(Uniform(0.9, 1.1))
                end
            end
        end
    end

    # Mode capacities
    mode_capacities = Dict{String, Float64}()
    for mode in transport_modes
        base_capacity = total_demand * mode_capacity_factor
        capacity_multiplier = if mode == "truck"
            rand(Gamma(4, 0.25))
        elseif mode == "rail"
            rand(Gamma(6, 0.33))
        elseif mode == "ship"
            rand(Gamma(9, 0.33))
        else  # air
            rand(Gamma(2, 0.25))
        end
        mode_capacities[mode] = base_capacity * capacity_multiplier
    end

    # Keep only available routes
    transport_costs = Dict(k => v for (k,v) in transport_costs if get(infrastructure, k, false))
    carbon_emissions = Dict(k => v for (k,v) in carbon_emissions if get(infrastructure, k, false))

    # FEASIBILITY ENFORCEMENT (mirrors the standard variant for connectivity/capacity)
    if feasibility_status == feasible
        fallback_mode = ("truck" in transport_modes) ? "truck" : transport_modes[1]
        K = min(max(3, ceil(Int, n_facilities ÷ 3)), n_facilities)

        customers_linked_to_facility = [Int[] for _ in 1:n_facilities]
        for c in 1:n_customers
            dvec = [dist(f, c) for f in 1:n_facilities]
            nearest_idxs = sortperm(dvec)[1:K]
            for f in nearest_idxs
                if !haskey(transport_costs, (f, c, fallback_mode))
                    distance = dvec[f]
                    base_cost = get(transport_base_costs, fallback_mode, 1.0)
                    terrain_factor = rand(LogNormal(log(1.0), 0.15))
                    volume_factor = 1.0 - 0.25 * (demands[c] / max_demand_val)
                    efficiency_factor = rand(Beta(3, 2)) * 0.4 + 0.8
                    infrastructure[(f, c, fallback_mode)] = true
                    transport_costs[(f, c, fallback_mode)] = base_cost * distance * terrain_factor * volume_factor * efficiency_factor
                    carbon_emissions[(f, c, fallback_mode)] = emission_rate(fallback_mode) * distance * rand(Uniform(0.9, 1.1))
                end
                push!(customers_linked_to_facility[f], c)
            end
        end

        # Capacity smoothing across linked customers
        approx_share = zeros(Float64, n_facilities)
        for f in 1:n_facilities
            for c in customers_linked_to_facility[f]
                n_links = length([ff for ff in 1:n_facilities if c in customers_linked_to_facility[ff]])
                approx_share[f] += demands[c] / n_links
            end
        end
        for f in 1:n_facilities
            if capacities[f] < 1.05 * approx_share[f]
                capacities[f] = 1.05 * approx_share[f]
            end
        end

        # Aggregate capacity guarantees
        if mode_capacities[fallback_mode] < 1.05 * total_demand
            mode_capacities[fallback_mode] = 1.05 * total_demand
        end
        total_mode_capacity = sum(mode_capacities[m] for m in transport_modes)
        if total_mode_capacity < total_demand
            scale = 1.05 * total_demand / max(total_mode_capacity, eps())
            for m in transport_modes
                mode_capacities[m] *= scale
            end
        end
        total_facility_capacity = sum(values(capacities))
        if total_facility_capacity < total_demand
            scale = 1.05 * total_demand / max(total_facility_capacity, eps())
            for f in 1:n_facilities
                capacities[f] *= scale
            end
        end
    elseif feasibility_status == infeasible
        # Make facilities/modes generous so the binding contradiction is the carbon cap.
        if sum(values(capacities)) < 1.1 * total_demand
            scale = 1.1 * total_demand / max(sum(values(capacities)), eps())
            for f in 1:n_facilities
                capacities[f] *= scale
            end
        end
        for m in transport_modes
            if mode_capacities[m] < 1.1 * total_demand
                mode_capacities[m] = 1.1 * total_demand
            end
        end
    end

    # --- Carbon budget derivation ---
    # Group available arcs by customer (with per-unit emission and mode).
    arcs_by_customer = Dict{Int, Vector{Tuple{Int,String,Float64}}}()
    for k in keys(carbon_emissions)
        f, c, m = k
        push!(get!(arcs_by_customer, c, Tuple{Int,String,Float64}[]), (f, m, carbon_emissions[k]))
    end

    # Lower bound on total emissions for ANY feasible flow: each customer's full
    # demand carried on its single cheapest-emission available arc.
    emission_lower_bound = 0.0
    for c in 1:n_customers
        if haskey(arcs_by_customer, c)
            emission_lower_bound += demands[c] * minimum(t -> t[3], arcs_by_customer[c])
        end
    end

    if feasibility_status == infeasible
        # Strictly below the lower bound (with margin) -> no feasible flow can
        # respect the budget, so the carbon constraint is the binding
        # contradiction (facility/mode capacities were already relaxed above).
        carbon_limit = emission_lower_bound * rand(Uniform(0.5, 0.8))
    elseif feasibility_status == feasible
        # Make the MINIMUM-EMISSION single-source assignment (each customer on its
        # globally cleanest arc) capacity-feasible by relaxing facility and mode
        # capacities to fit it. That assignment attains the emission lower bound,
        # so a flow meeting a cap of (lower_bound * small_slack) provably exists.
        # The small slack keeps the budget tight, so it stays ACTIVE relative to
        # the cost-minimizing (truck-heavy, higher-emission) solution.
        fac_need = Dict(f => 0.0 for f in 1:n_facilities)
        mode_need = Dict(m => 0.0 for m in transport_modes)
        for c in 1:n_customers
            haskey(arcs_by_customer, c) || continue
            f, m, _ = argmin(t -> t[3], arcs_by_customer[c])
            fac_need[f] += demands[c]
            mode_need[m] += demands[c]
        end
        for f in 1:n_facilities
            if capacities[f] < fac_need[f]
                capacities[f] = fac_need[f] * 1.05
            end
        end
        for m in transport_modes
            if mode_capacities[m] < mode_need[m]
                mode_capacities[m] = mode_need[m] * 1.05
            end
        end
        carbon_limit = emission_lower_bound * rand(Uniform(1.01, 1.05))
    else
        # unknown: natural instance, no forced infeasibility. Anchor the cap on a
        # capacity-respecting greedy reference flow and leave generous slack so the
        # instance is typically (but not guaranteed) feasible.
        remaining_fac = Dict(f => capacities[f] for f in 1:n_facilities)
        remaining_mode = Dict(m => mode_capacities[m] for m in transport_modes)
        ref_emissions = 0.0
        for c in 1:n_customers
            haskey(arcs_by_customer, c) || continue
            d = demands[c]
            cands = sort(arcs_by_customer[c], by = t -> t[3])
            idx = findfirst(t -> remaining_fac[t[1]] >= d && remaining_mode[t[2]] >= d, cands)
            f, m, e = idx === nothing ? cands[1] : cands[idx]
            remaining_fac[f] = max(0.0, remaining_fac[f] - d)
            remaining_mode[m] = max(0.0, remaining_mode[m] - d)
            ref_emissions += d * e
        end
        carbon_limit = max(emission_lower_bound, ref_emissions) * rand(Uniform(1.3, 2.5))
    end

    return CarbonSupplyChainProblem(
        n_facilities,
        n_customers,
        transport_modes,
        facility_locs,
        customer_locs,
        cluster_centers,
        cluster_weights,
        fixed_costs,
        demands,
        capacities,
        transport_costs,
        mode_capacities,
        carbon_emissions,
        carbon_limit,
        total_demand
    )
end

"""
    build_model(prob::CarbonSupplyChainProblem)

Build a JuMP model for the carbon-constrained supply chain problem (deterministic).

# Arguments
- `prob`: CarbonSupplyChainProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::CarbonSupplyChainProblem)
    model = Model()

    # Facility-open decisions
    @variable(model, y[1:prob.n_facilities], Bin)

    # Available (facility, customer, mode) arcs
    valid_combinations = [(f,c,m) for f in 1:prob.n_facilities, c in 1:prob.n_customers, m in prob.transport_modes
                          if haskey(prob.transport_costs, (f,c,m))]

    @variable(model, x[valid_combinations] >= 0)

    # Objective: minimize fixed facility cost plus transportation cost
    @objective(model, Min,
        sum(prob.fixed_costs[f] * y[f] for f in 1:prob.n_facilities) +
        sum(prob.transport_costs[combo] * x[combo] for combo in valid_combinations)
    )

    # Customer demand satisfaction
    for c in 1:prob.n_customers
        combos_for_customer = filter(combo -> combo[2] == c, valid_combinations)
        @constraint(model, sum(x[combo] for combo in combos_for_customer) >= prob.demands[c])
    end

    # Facility capacity (gated by open decision)
    for f in 1:prob.n_facilities
        combos_for_facility = filter(combo -> combo[1] == f, valid_combinations)
        @constraint(model, sum(x[combo] for combo in combos_for_facility) <= prob.capacities[f] * y[f])
    end

    # Per-mode transport capacity
    for m in prob.transport_modes
        combos_for_mode = filter(combo -> combo[3] == m, valid_combinations)
        @constraint(model, sum(x[combo] for combo in combos_for_mode) <= prob.mode_capacities[m])
    end

    # Global transportation carbon budget
    @constraint(model,
        sum(prob.carbon_emissions[combo] * x[combo] for combo in valid_combinations) <= prob.carbon_limit
    )

    return model
end

# Register the variant
register_variant(
    :supply_chain,
    :carbon,
    CarbonSupplyChainProblem,
    "Supply chain optimization with a global transportation carbon budget on shipments across truck/rail/ship/air modes",
)
