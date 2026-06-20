using JuMP
using Random
using StatsBase
using Distributions

"""
    SingleSourceSupplyChainProblem <: ProblemGenerator

Generator for single-source capacitated supply chain (facility location) problems.

Each customer must be served in full by exactly one open facility (a single-source
assignment), in contrast to the standard supply chain model where a customer's
demand can be split across several facilities. This adds a binary assignment
matrix `z[f, c]` whose `sum_f z[f, c] == 1` enforces single sourcing, together
with flow-gating constraints `x[(f,c,m)] <= demand[c] * z[f, c]` that allow
shipping on a lane only from the facility a customer is assigned to.

This problem models realistic supply chain networks with:
- Geographic clustering of customers and facilities
- Multiple transportation modes with infrastructure availability
- K-nearest connectivity guarantees for feasible instances
- Facility opening costs and capacities
- Mode-specific capacity constraints
- Single-source assignment of every customer to one facility

# Overview
Models single-source strategic supply-chain network design. The decisions open
facilities (`y`), assign each customer to exactly one facility (`z`), and ship
that customer's demand from its assigned facility over available transportation
modes (`x`). The objective minimizes fixed facility cost plus mode-specific
transportation cost. Constraints assign each customer to one facility, gate lane
flow by the assignment, satisfy customer demand, gate shipments by open-facility
capacity, and limit aggregate shipment volume by transportation mode.

# Fields
All data generated in the constructor based on `target_variables` and `feasibility_status`:
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
- `total_demand::Float64`: Total demand across all customers
"""
struct SingleSourceSupplyChainProblem <: ProblemGenerator
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
    total_demand::Float64
end

"""
    SingleSourceSupplyChainProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a single-source supply chain problem instance with geographic clustering
and connectivity logic.

# Variable-count formula
The built model has three variable blocks:
- `y[1:n_facilities]`                            -> `n_facilities`
- `x[valid_combinations]`                        -> `≈ n_facilities * n_customers * n_modes * density`
- `z[1:n_facilities, 1:n_customers]` (Bin)       -> `n_facilities * n_customers`

Total `≈ n_facilities + n_facilities * n_customers * (1 + n_modes * density)`.
The O(n_facilities * n_customers) binary `z` block is the dominant extra term and
is accounted for here so the instance hits `target_variables`. Given a chosen
`n_facilities`, `n_modes`, and an estimated infrastructure `density`, the number of
customers is solved from the formula above.

# Sophisticated Feasibility Logic
- **Geographic clustering**: Customers clustered using Dirichlet-weighted clusters with log-normal spread
- **Facility placement**: Beta-distributed strategic placement with market-access consideration
- **K-nearest connectivity**: For feasible instances, ensures each customer connects to K nearest facilities
- **Single-source feasibility**: For feasible instances, builds an explicit capacity-respecting
  assignment of each customer to one facility, guaranteeing a single-source solution exists
- **Capacity deficit**: For infeasible instances, drives total facility capacity below total
  demand with a margin, so no assignment can be served

# Arguments
- `target_variables`: Target number of variables (see variable-count formula above)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function SingleSourceSupplyChainProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # Total vars ≈ n_facilities + n_facilities * n_customers * (1 + n_modes * density)
    # Choose n_facilities, n_modes, and an estimated density, then solve for n_customers.
    if target_variables <= 250
        n_facilities = rand(DiscreteUniform(3, 6))
        n_transport_modes = rand(DiscreteUniform(1, 2))
        grid_width = rand(Uniform(200.0, 800.0))
        grid_height = rand(Uniform(200.0, 800.0))
        infrastructure_density = rand(Beta(5, 2)) * 0.3 + 0.7  # 0.7-1.0
        clustering_factor = rand(Beta(3, 2)) * 0.6 + 0.25  # 0.25-0.85
        min_fixed_cost = max(100000.0, rand(LogNormal(log(300000), 0.5)))
        max_fixed_cost = min_fixed_cost * rand(Uniform(1.8, 3.5))
        base_demand = rand(Uniform(80.0, 150.0))
        min_demand = base_demand
        max_demand = base_demand * rand(Uniform(3.0, 8.0))
    elseif target_variables <= 1000
        n_facilities = rand(DiscreteUniform(5, 12))
        n_transport_modes = rand(DiscreteUniform(2, 3))
        grid_width = rand(Uniform(800.0, 2000.0))
        grid_height = rand(Uniform(800.0, 2000.0))
        infrastructure_density = rand(Beta(3, 2)) * 0.4 + 0.5  # 0.5-0.9
        clustering_factor = rand(Beta(2, 3)) * 0.5 + 0.2  # 0.2-0.7
        min_fixed_cost = max(300000.0, rand(LogNormal(log(800000), 0.6)))
        max_fixed_cost = min_fixed_cost * rand(Uniform(2.0, 4.0))
        base_demand = rand(Uniform(150.0, 300.0))
        min_demand = base_demand
        max_demand = base_demand * rand(Uniform(4.0, 12.0))
    else
        n_facilities = rand(DiscreteUniform(8, 20))
        n_transport_modes = rand(DiscreteUniform(3, 4))
        grid_width = rand(Uniform(2000.0, 5000.0))
        grid_height = rand(Uniform(2000.0, 5000.0))
        infrastructure_density = rand(Beta(2, 3)) * 0.4 + 0.4  # 0.4-0.8
        clustering_factor = rand(Beta(1, 3)) * 0.4 + 0.15  # 0.15-0.55
        min_fixed_cost = max(500000.0, rand(LogNormal(log(1500000), 0.7)))
        max_fixed_cost = min_fixed_cost * rand(Uniform(2.5, 5.0))
        base_demand = rand(Uniform(300.0, 600.0))
        min_demand = base_demand
        max_demand = base_demand * rand(Uniform(6.0, 20.0))
    end

    # Solve for n_customers from the variable-count formula.
    # target ≈ n_facilities + n_facilities * n_customers * (1 + eff)
    # where the z block contributes exactly n_facilities per customer and the x block
    # contributes ≈ n_facilities * eff per customer. The realized lane density is well
    # below n_modes * infrastructure_density (non-truck modes are often unavailable),
    # so a calibrated effective coefficient is used.
    eff = 0.6 * n_transport_modes * infrastructure_density
    per_customer_vars = n_facilities * (1.0 + eff)
    n_customers = max(4, round(Int, (target_variables - n_facilities) / per_customer_vars))

    # Additional parameters
    capacity_factor = rand(Uniform(1.2, 2.2))
    mode_capacity_factor = rand(Uniform(0.25, 0.65))

    # Transport modes and costs
    all_transport_modes = ["truck", "rail", "ship", "air"]
    transport_base_costs = Dict(
        "truck" => rand(Gamma(4, 0.25)),
        "rail" => rand(Gamma(3, 0.2)),
        "ship" => rand(Gamma(2, 0.15)),
        "air" => rand(Gamma(6, 0.5))
    )

    transport_modes = sample(all_transport_modes, min(n_transport_modes, length(all_transport_modes)), replace=false)

    # Geographic clusters
    n_clusters = max(2, round(Int, sqrt(n_customers) * clustering_factor))
    cluster_centers = [(grid_width * rand(), grid_height * rand()) for _ in 1:n_clusters]

    # Facility locations (more dispersed)
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

    # Customer locations (clustered)
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

    # Facility fixed costs (correlated with market access and location)
    fixed_costs = Dict{Int, Float64}()
    for f in 1:n_facilities
        distances_to_customers = [
            sqrt((facility_locs[f][1] - c[1])^2 + (facility_locs[f][2] - c[2])^2)
            for c in customer_locs
        ]
        market_potential = sum(exp.(-distances_to_customers ./ (grid_width * 0.2)))
        location_factor = (facility_locs[f][1] / grid_width + facility_locs[f][2] / grid_height) / 2
        base_cost = min_fixed_cost + (max_fixed_cost - min_fixed_cost) *
                   (0.2 + 0.5 * market_potential / n_customers + 0.3 * location_factor)
        cost_multiplier = rand(LogNormal(log(1.0), 0.25))
        fixed_costs[f] = base_cost * cost_multiplier
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
        demand_multiplier = rand(LogNormal(log(1.0), 0.4))
        demands[c] = base_demand_val * demand_multiplier
    end

    # Facility capacities
    total_demand = sum(values(demands))
    avg_capacity = (total_demand / n_facilities) * capacity_factor
    capacities = Dict{Int, Float64}()
    for f in 1:n_facilities
        relative_cost = (fixed_costs[f] - minimum(values(fixed_costs))) /
                       max(eps(), (maximum(values(fixed_costs)) - minimum(values(fixed_costs))))
        base_capacity = avg_capacity * (0.6 + 0.8 * relative_cost)
        capacity_multiplier = rand(Gamma(3, 1/3))
        capacities[f] = base_capacity * capacity_multiplier
    end

    # Transport costs and infrastructure availability
    transport_costs = Dict{Tuple{Int,Int,String}, Float64}()
    infrastructure = Dict{Tuple{Int,Int,String}, Bool}()

    for f in 1:n_facilities
        for c in 1:n_customers
            distance = sqrt(
                (facility_locs[f][1] - customer_locs[c][1])^2 +
                (facility_locs[f][2] - customer_locs[c][2])^2
            )
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
                    volume_factor = 1.0 - 0.25 * (demands[c] / maximum(values(demands)))
                    efficiency_factor = rand(Beta(3, 2)) * 0.4 + 0.8
                    transport_costs[(f,c,mode)] = base_cost * distance * terrain_factor * volume_factor * efficiency_factor
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
    transport_costs = Dict(k => v for (k,v) in transport_costs if infrastructure[k])

    # --- Feasibility enforcement ---
    if feasibility_status == feasible
        # Pick a single fallback mode and guarantee every customer has a route to its
        # K nearest facilities on that mode (so a single-source assignment is buildable).
        fallback_mode = ("truck" in transport_modes) ? "truck" : transport_modes[1]
        K = min(max(3, ceil(Int, n_facilities ÷ 3)), n_facilities)

        for c in 1:n_customers
            dvec = [sqrt((facility_locs[f][1] - customer_locs[c][1])^2 +
                        (facility_locs[f][2] - customer_locs[c][2])^2) for f in 1:n_facilities]
            nearest_idxs = sortperm(dvec)[1:K]
            for f in nearest_idxs
                if !haskey(transport_costs, (f, c, fallback_mode))
                    distance = dvec[f]
                    base_cost = get(transport_base_costs, fallback_mode, 1.0)
                    terrain_factor = rand(LogNormal(log(1.0), 0.15))
                    volume_factor = 1.0 - 0.25 * (demands[c] / maximum(values(demands)))
                    efficiency_factor = rand(Beta(3, 2)) * 0.4 + 0.8
                    transport_costs[(f, c, fallback_mode)] = base_cost * distance * terrain_factor * volume_factor * efficiency_factor
                end
            end
        end

        # Build an explicit single-source assignment: greedily assign each customer
        # (largest demand first) to the nearest facility that still has residual
        # capacity and a valid route. Bump capacities/mode capacity so this fits.
        residual = Dict(f => capacities[f] for f in 1:n_facilities)
        assigned_to = Dict{Int, Int}()
        customer_order = sort(1:n_customers, by = c -> -demands[c])
        for c in customer_order
            dvec = [sqrt((facility_locs[f][1] - customer_locs[c][1])^2 +
                        (facility_locs[f][2] - customer_locs[c][2])^2) for f in 1:n_facilities]
            order = sortperm(dvec)
            # facilities with a route to c on the fallback mode
            routed = [f for f in order if haskey(transport_costs, (f, c, fallback_mode))]
            chosen = nothing
            for f in routed
                if residual[f] >= demands[c]
                    chosen = f
                    break
                end
            end
            if chosen === nothing
                # No facility has room: assign to nearest routed facility and grow its capacity.
                chosen = isempty(routed) ? order[1] : routed[1]
                # ensure a route exists for the chosen facility
                if !haskey(transport_costs, (chosen, c, fallback_mode))
                    base_cost = get(transport_base_costs, fallback_mode, 1.0)
                    transport_costs[(chosen, c, fallback_mode)] = base_cost * dvec[chosen] * rand(Uniform(0.9, 1.1))
                end
                capacities[chosen] += 1.05 * demands[c]
                residual[chosen] = capacities[chosen] - sum(demands[cc] for (cc, ff) in assigned_to if ff == chosen; init=0.0)
            end
            assigned_to[c] = chosen
            residual[chosen] -= demands[c]
        end

        # Guarantee each facility's capacity covers its assigned single-source load.
        assigned_load = Dict(f => 0.0 for f in 1:n_facilities)
        for (c, f) in assigned_to
            assigned_load[f] += demands[c]
        end
        for f in 1:n_facilities
            if capacities[f] < 1.05 * assigned_load[f]
                capacities[f] = 1.05 * assigned_load[f]
            end
        end

        # Ensure the fallback mode alone can move all demand.
        if mode_capacities[fallback_mode] < 1.05 * total_demand
            mode_capacities[fallback_mode] = 1.05 * total_demand
        end

    elseif feasibility_status == infeasible
        # CAPACITY DEFICIT: drive total facility capacity strictly below total demand
        # with a margin, so no assignment (single-source or otherwise) can satisfy demand.
        margin = rand(Uniform(0.7, 0.9))  # target total capacity = margin * total_demand
        total_capacity = sum(values(capacities))
        target_total = margin * total_demand
        if total_capacity > target_total
            scale = target_total / total_capacity
            for f in 1:n_facilities
                capacities[f] *= scale
            end
        end
    end
    # For unknown, leave the natural instance unchanged.

    return SingleSourceSupplyChainProblem(
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
        total_demand
    )
end

"""
    build_model(prob::SingleSourceSupplyChainProblem)

Build a JuMP model for the single-source supply chain problem (deterministic).

# Arguments
- `prob`: SingleSourceSupplyChainProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::SingleSourceSupplyChainProblem)
    model = Model()

    # Open-facility decisions
    @variable(model, y[1:prob.n_facilities], Bin)

    # Valid (facility, customer, mode) lanes with an available route
    valid_combinations = [(f,c,m) for f in 1:prob.n_facilities, c in 1:prob.n_customers, m in prob.transport_modes
                          if haskey(prob.transport_costs, (f,c,m))]

    # Flow decisions on valid lanes
    @variable(model, x[valid_combinations] >= 0)

    # Single-source assignment decisions
    @variable(model, z[1:prob.n_facilities, 1:prob.n_customers], Bin)

    # Objective: minimize fixed facility cost + transportation cost
    @objective(model, Min,
        sum(prob.fixed_costs[f] * y[f] for f in 1:prob.n_facilities) +
        sum(prob.transport_costs[combo] * x[combo] for combo in valid_combinations)
    )

    # Each customer assigned to exactly one facility
    for c in 1:prob.n_customers
        @constraint(model, sum(z[f, c] for f in 1:prob.n_facilities) == 1)
    end

    # Flow gating: a lane (f,c,m) can carry flow only if customer c is assigned to f
    for (f, c, m) in valid_combinations
        @constraint(model, x[(f, c, m)] <= prob.demands[c] * z[f, c])
    end

    # Customer demand satisfaction
    for c in 1:prob.n_customers
        combos_for_customer = filter(combo -> combo[2] == c, valid_combinations)
        @constraint(model, sum(x[combo] for combo in combos_for_customer) >= prob.demands[c])
    end

    # Facility capacity (also links opening decision y)
    for f in 1:prob.n_facilities
        combos_for_facility = filter(combo -> combo[1] == f, valid_combinations)
        @constraint(model, sum(x[combo] for combo in combos_for_facility) <= prob.capacities[f] * y[f])
    end

    # Transport mode capacity (restored for consistency with the standard SC model)
    for m in prob.transport_modes
        combos_for_mode = filter(combo -> combo[3] == m, valid_combinations)
        @constraint(model, sum(x[combo] for combo in combos_for_mode) <= prob.mode_capacities[m])
    end

    return model
end

# Register the variant
register_variant(
    :supply_chain,
    :single_source,
    SingleSourceSupplyChainProblem,
    "Single-source capacitated supply chain where each customer is served in full by exactly one facility",
)
