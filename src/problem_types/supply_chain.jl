using JuMP
using Random
using StatsBase
using Distributions

"""
    SupplyChainProblem <: ProblemGenerator

Generator for supply chain optimization problems that minimize facility and transportation costs while
meeting customer demands and respecting capacity constraints.

This problem models realistic supply chain networks with:
- Geographic clustering of customers and facilities
- Multiple transportation modes with infrastructure availability
- K-nearest connectivity guarantees for feasible instances
- Facility opening costs and capacities
- Mode-specific capacity constraints

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
- `total_demand::Float64`: Total demand across all customers
"""
struct SupplyChainProblem <: ProblemGenerator
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
    SupplyChainProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a supply chain problem instance with sophisticated geographic clustering and connectivity logic.

# Sophisticated Feasibility Logic Preserved:
- **Geographic clustering**: Customers clustered using Dirichlet-weighted clusters with log-normal spread
- **Facility placement**: Beta-distributed strategic placement with market-access consideration
- **K-nearest connectivity**: For feasible instances, ensures each customer connects to K nearest facilities
- **Market potential**: Fixed costs correlated with proximity to customer clusters
- **Infrastructure availability**: Mode-specific availability based on distance and characteristics
- **Capacity balancing**: For feasible instances, adjusts facility and mode capacities to ensure feasibility

# Arguments
- `target_variables`: Target number of variables (approximately n_facilities × n_customers × n_transport_modes)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function SupplyChainProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem dimensions based on target variables
    # Variables = n_facilities (binary) + n_facilities * n_customers * n_transport_modes * infrastructure_density (continuous)
    # Approximate: infrastructure_density ≈ 0.5-0.8 on average
    avg_density = 0.65

    if target_variables <= 250
        # Small: local/regional
        n_facilities = rand(DiscreteUniform(3, 8))
        n_customers = rand(DiscreteUniform(15, 35))
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
        # Medium: regional/national
        n_facilities = rand(DiscreteUniform(6, 18))
        n_customers = rand(DiscreteUniform(25, 65))
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
        # Large: national/global
        n_facilities = rand(DiscreteUniform(12, 40))
        n_customers = rand(DiscreteUniform(60, 200))
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

    # Select transport modes
    transport_modes = sample(all_transport_modes, min(n_transport_modes, length(all_transport_modes)), replace=false)

    # Generate geographic clusters for realistic location distribution
    n_clusters = max(2, round(Int, sqrt(n_customers) * clustering_factor))
    cluster_centers = [(grid_width * rand(), grid_height * rand()) for _ in 1:n_clusters]

    # Generate facility locations (more dispersed than customers)
    facility_locs = Vector{Tuple{Float64,Float64}}()
    for _ in 1:n_facilities
        # Use Beta distribution to create more realistic facility placement
        if rand() < 0.4  # 40% chance to be near a cluster center (strategic placement)
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

    # Generate customer locations (more clustered)
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

    # Generate facility fixed costs (correlated with location and market size)
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

    # Generate customer demands (correlated with cluster size)
    demands = Dict{Int, Float64}()

    for c in 1:n_customers
        distances_to_clusters = [
            sqrt((customer_locs[c][1] - center[1])^2 + (customer_locs[c][2] - center[2])^2)
            for center in cluster_centers
        ]
        _, cluster_idx = findmin(distances_to_clusters)

        cluster_influence = cluster_weights[cluster_idx]
        base_demand_val = min_demand + (max_demand - min_demand) *
                     (0.2 + 0.8 * cluster_influence)

        demand_multiplier = rand(LogNormal(log(1.0), 0.4))
        demands[c] = base_demand_val * demand_multiplier
    end

    # Generate facility capacities
    total_demand = sum(values(demands))
    avg_capacity = (total_demand / n_facilities) * capacity_factor
    capacities = Dict{Int, Float64}()

    for f in 1:n_facilities
        relative_cost = (fixed_costs[f] - minimum(values(fixed_costs))) /
                       (maximum(values(fixed_costs)) - minimum(values(fixed_costs)))

        base_capacity = avg_capacity * (0.6 + 0.8 * relative_cost)
        capacity_multiplier = rand(Gamma(3, 1/3))
        capacities[f] = base_capacity * capacity_multiplier
    end

    # Generate transport costs and infrastructure availability
    transport_costs = Dict{Tuple{Int,Int,String}, Float64}()
    infrastructure = Dict{Tuple{Int,Int,String}, Bool}()

    for f in 1:n_facilities
        for c in 1:n_customers
            distance = sqrt(
                (facility_locs[f][1] - customer_locs[c][1])^2 +
                (facility_locs[f][2] - customer_locs[c][2])^2
            )

            for mode in transport_modes
                # Determine if this transport mode is available
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

    # Generate mode capacities
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

    # Clean up transport costs to only include available routes
    transport_costs = Dict(k => v for (k,v) in transport_costs if infrastructure[k])

    # SOPHISTICATED FEASIBILITY ENFORCEMENT
    if feasibility_status == feasible
        # K-NEAREST CONNECTIVITY: Select fallback mode and ensure connectivity to K nearest facilities
        fallback_mode = ("truck" in transport_modes) ? "truck" : transport_modes[1]
        K = min(max(3, ceil(Int, n_facilities ÷ 3)), n_facilities)

        customers_linked_to_facility = [Int[] for _ in 1:n_facilities]
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
                    infrastructure[(f, c, fallback_mode)] = true
                    transport_costs[(f, c, fallback_mode)] = base_cost * distance * terrain_factor * volume_factor * efficiency_factor
                end
                push!(customers_linked_to_facility[f], c)
            end
        end

        # CAPACITY SMOOTHING: Adjust facility capacities to cover nearby demand share
        approx_share = zeros(Float64, n_facilities)
        for f in 1:n_facilities
            linked_customers = customers_linked_to_facility[f]
            for c in linked_customers
                approx_share[f] += demands[c] / length([ff for ff in 1:n_facilities if c in customers_linked_to_facility[ff]])
            end
        end
        for f in 1:n_facilities
            if capacities[f] < 1.05 * approx_share[f]
                capacities[f] = 1.05 * approx_share[f]
            end
        end

        # AGGREGATE CAPACITY GUARANTEES
        total_mode_capacity = sum(mode_capacities[m] for m in transport_modes)
        total_facility_capacity = sum(values(capacities))

        # Ensure fallback mode can alone move all demand
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
        if total_facility_capacity < total_demand
            scale = 1.05 * total_demand / max(total_facility_capacity, eps())
            for f in 1:n_facilities
                capacities[f] *= scale
            end
        end

    elseif feasibility_status == infeasible
        # REALISTIC INFEASIBILITY: Transport capacity shortfall
        desired_ratio = rand(Uniform(0.7, 0.95))
        desired_total_mode_capacity = desired_ratio * total_demand
        total_mode_capacity = sum(mode_capacities[m] for m in transport_modes)
        if total_mode_capacity > desired_total_mode_capacity
            scale = desired_total_mode_capacity / total_mode_capacity
            for m in transport_modes
                mode_capacities[m] *= scale
            end
        end
    end

    return SupplyChainProblem(
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
    build_model(prob::SupplyChainProblem)

Build a JuMP model for the supply chain problem (deterministic).

# Arguments
- `prob`: SupplyChainProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::SupplyChainProblem)
    model = Model()

    # Decision variables
    @variable(model, y[1:prob.n_facilities], Bin)

    # Create valid combinations
    valid_combinations = [(f,c,m) for f in 1:prob.n_facilities, c in 1:prob.n_customers, m in prob.transport_modes
                          if haskey(prob.transport_costs, (f,c,m))]

    @variable(model, x[valid_combinations] >= 0)

    # Objective: Minimize total cost
    @objective(model, Min,
        sum(prob.fixed_costs[f] * y[f] for f in 1:prob.n_facilities) +
        sum(prob.transport_costs[combo] * x[combo] for combo in valid_combinations)
    )

    # Customer demand satisfaction
    for c in 1:prob.n_customers
        combos_for_customer = filter(combo -> combo[2] == c, valid_combinations)
        @constraint(model,
            sum(x[combo] for combo in combos_for_customer) >= prob.demands[c]
        )
    end

    # Facility capacity constraints
    for f in 1:prob.n_facilities
        combos_for_facility = filter(combo -> combo[1] == f, valid_combinations)
        @constraint(model,
            sum(x[combo] for combo in combos_for_facility) <= prob.capacities[f] * y[f]
        )
    end

    # Transport mode capacity constraints
    for m in prob.transport_modes
        combos_for_mode = filter(combo -> combo[3] == m, valid_combinations)
        @constraint(model,
            sum(x[combo] for combo in combos_for_mode) <= prob.mode_capacities[m]
        )
    end

    return model
end

# Register the problem type
register_problem(
    :supply_chain,
    SupplyChainProblem,
    "Supply chain optimization problem that minimizes facility and transportation costs while meeting customer demands and respecting capacity constraints"
)
