using JuMP
using Random
using Distributions

"""
    HubLocation <: ProblemGenerator

Generator for hub location problems in hub-and-spoke networks.

This problem selects hub locations and routes origin-destination flows through
the hub network, exploiting economies of scale on hub-to-hub connections.

# Fields
- `n_nodes::Int`: Number of nodes (potential origins/destinations)
- `n_potential_hubs::Int`: Number of potential hub locations
- `node_locations::Vector{Tuple{Float64,Float64}}`: Node coordinates
- `demands::Dict{Tuple{Int,Int},Float64}`: Demand from origin i to destination j
- `direct_costs::Matrix{Float64}`: Cost of direct connection between nodes
- `hub_fixed_costs::Vector{Float64}`: Fixed cost to open each hub
- `hub_capacities::Vector{Float64}`: Capacity of each potential hub
- `collection_costs::Matrix{Float64}`: Cost from origin to hub (collection)
- `transfer_costs::Matrix{Float64}`: Cost between hubs (discounted)
- `distribution_costs::Matrix{Float64}`: Cost from hub to destination
- `discount_factor::Float64`: Discount on inter-hub connections (< 1.0)
"""
struct HubLocation <: ProblemGenerator
    n_nodes::Int
    n_potential_hubs::Int
    node_locations::Vector{Tuple{Float64,Float64}}
    demands::Dict{Tuple{Int,Int},Float64}
    direct_costs::Matrix{Float64}
    hub_fixed_costs::Vector{Float64}
    hub_capacities::Vector{Float64}
    collection_costs::Matrix{Float64}
    transfer_costs::Matrix{Float64}
    distribution_costs::Matrix{Float64}
    discount_factor::Float64
end

"""
    HubLocation(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a hub location problem instance.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Details
Variables: hub selection + flow routing
Target: n_hubs + n_nodes² × routing_options
"""
function HubLocation(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine scale
    if target_variables <= 150
        min_nodes, max_nodes = 5, 15
        hub_ratio = rand(0.3:0.05:0.6)
        grid_size = rand(200.0:50.0:500.0)
        demand_range = (5.0, 50.0)
        cost_per_km = rand(1.0:0.2:2.0)
        discount_factor = rand(0.6:0.05:0.8)
    elseif target_variables <= 800
        min_nodes, max_nodes = 10, 30
        hub_ratio = rand(0.25:0.05:0.5)
        grid_size = rand(300.0:100.0:1000.0)
        demand_range = (10.0, 100.0)
        cost_per_km = rand(1.5:0.2:3.0)
        discount_factor = rand(0.5:0.05:0.75)
    else
        min_nodes, max_nodes = 15, 50
        hub_ratio = rand(0.2:0.05:0.4)
        grid_size = rand(500.0:200.0:2000.0)
        demand_range = (20.0, 200.0)
        cost_per_km = rand(2.0:0.5:5.0)
        discount_factor = rand(0.4:0.05:0.7)
    end

    # Solve for dimensions
    # Variables: n_hubs (binary) + flow variables
    # Flow variables: for each O-D pair, we have routing through hubs
    # Approximate: n_nodes² × (n_hubs + n_hubs²) for routing options

    best_n_nodes = min_nodes
    best_error = Inf

    for n_nodes in min_nodes:max_nodes
        n_hubs = max(2, round(Int, n_nodes * hub_ratio))
        # Flow variables: direct + via hubs
        # Simplified: n_nodes² × n_hubs (assignment to first hub)
        approx_vars = n_hubs + n_nodes * n_nodes * n_hubs
        error = abs(approx_vars - target_variables) / target_variables

        if error < best_error
            best_error = error
            best_n_nodes = n_nodes
        end
    end

    n_nodes = best_n_nodes
    n_potential_hubs = max(2, round(Int, n_nodes * hub_ratio))

    # Generate node locations
    # Create geographic clusters for realism
    n_clusters = max(2, n_nodes ÷ 6)
    cluster_centers = [(grid_size * rand(), grid_size * rand()) for _ in 1:n_clusters]

    node_locations = Tuple{Float64,Float64}[]
    for i in 1:n_nodes
        center = rand(cluster_centers)
        cluster_radius = grid_size / (2 * sqrt(n_clusters))
        x = clamp(center[1] + randn() * cluster_radius, 0.0, grid_size)
        y = clamp(center[2] + randn() * cluster_radius, 0.0, grid_size)
        push!(node_locations, (x, y))
    end

    # Generate demands (sparse O-D matrix)
    # Not all pairs have demand
    min_demand, max_demand = demand_range
    log_mean = log(sqrt(min_demand * max_demand))
    log_std = log(max_demand / min_demand) / 3

    demands = Dict{Tuple{Int,Int},Float64}()
    demand_density = if target_variables <= 150
        rand(0.3:0.05:0.6)
    elseif target_variables <= 800
        rand(0.2:0.05:0.5)
    else
        rand(0.15:0.05:0.4)
    end

    for i in 1:n_nodes
        for j in 1:n_nodes
            if i != j && rand() < demand_density
                demand = exp(rand(Normal(log_mean, log_std)))
                demand = clamp(demand, min_demand, max_demand)
                demands[(i, j)] = round(demand, digits=2)
            end
        end
    end

    # Calculate distances
    function calc_distance(loc1, loc2)
        return sqrt((loc1[1] - loc2[1])^2 + (loc1[2] - loc2[2])^2)
    end

    direct_costs = zeros(n_nodes, n_nodes)
    for i in 1:n_nodes
        for j in 1:n_nodes
            if i != j
                dist = calc_distance(node_locations[i], node_locations[j])
                direct_costs[i, j] = round(dist * cost_per_km, digits=2)
            end
        end
    end

    # Hub costs and capacities
    total_demand = sum(values(demands))
    avg_hub_capacity = (total_demand / n_potential_hubs) * rand(1.2:0.1:2.0)

    hub_capacities = [round(avg_hub_capacity * (0.8 + 0.4 * rand()), digits=2)
                      for _ in 1:n_potential_hubs]

    # Fixed costs for hubs (economies of scale with capacity)
    base_hub_cost = rand(50000.0:10000.0:200000.0)
    hub_fixed_costs = [round(base_hub_cost * (hub_capacities[h] / avg_hub_capacity)^0.9, digits=2)
                       for h in 1:n_potential_hubs]

    # Collection costs (node to hub) - same as direct costs
    collection_costs = copy(direct_costs)

    # Transfer costs (hub to hub) - discounted due to economies of scale
    transfer_costs = direct_costs * discount_factor

    # Distribution costs (hub to node) - same as direct costs
    distribution_costs = copy(direct_costs)

    # Adjust for feasibility
    if feasibility_status == feasible
        # Ensure sufficient hub capacity
        if sum(hub_capacities) < total_demand
            scale = (total_demand * 1.2) / sum(hub_capacities)
            hub_capacities .*= scale
            hub_capacities = round.(hub_capacities, digits=2)
        end

    elseif feasibility_status == infeasible
        # Create capacity infeasibility
        if rand() < 0.6
            # Reduce hub capacities
            scale = rand(0.4:0.05:0.75)
            hub_capacities .*= scale
            hub_capacities = round.(hub_capacities, digits=2)
        else
            # Increase demands
            scale = rand(1.5:0.1:2.5)
            for key in keys(demands)
                demands[key] = round(demands[key] * scale, digits=2)
            end
        end
    end

    return HubLocation(
        n_nodes,
        n_potential_hubs,
        node_locations,
        demands,
        direct_costs,
        hub_fixed_costs,
        hub_capacities,
        collection_costs,
        transfer_costs,
        distribution_costs,
        discount_factor
    )
end

"""
    build_model(prob::HubLocation)

Build a JuMP model for the hub location problem.

# Arguments
- `prob`: HubLocation instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::HubLocation)
    model = Model()

    N = prob.n_nodes
    H = prob.n_potential_hubs
    OD_pairs = collect(keys(prob.demands))

    # Decision variables
    # y[h] = 1 if hub h is opened
    @variable(model, y[1:H], Bin)

    # x[i,j,h1,h2] = flow from i to j routed via hubs h1 (collection) and h2 (distribution)
    # For LP, we use a simplified formulation:
    # z[i,j,h] = 1 if flow from i to j uses hub h as the collection hub
    @variable(model, 0 <= z[od in OD_pairs, 1:H] <= 1)

    # w[i,j,h1,h2] = fraction of flow from i to j going through hub pair (h1, h2)
    @variable(model, 0 <= w[od in OD_pairs, 1:H, 1:H] <= 1)

    # Objective: minimize total cost
    @objective(model, Min,
        # Fixed costs for opening hubs
        sum(prob.hub_fixed_costs[h] * y[h] for h in 1:H) +
        # Routing costs through hubs
        sum(
            prob.demands[od] * (
                prob.collection_costs[od[1], h1] +
                prob.transfer_costs[h1, h2] +
                prob.distribution_costs[h2, od[2]]
            ) * w[od, h1, h2]
            for od in OD_pairs, h1 in 1:H, h2 in 1:H
        )
    )

    # Constraints

    # Each O-D flow must be fully routed
    for od in OD_pairs
        @constraint(model, sum(w[od, h1, h2] for h1 in 1:H, h2 in 1:H) == 1)
    end

    # Can only route through open hubs
    for od in OD_pairs
        for h1 in 1:H
            @constraint(model, sum(w[od, h1, h2] for h2 in 1:H) <= y[h1])
        end
        for h2 in 1:H
            @constraint(model, sum(w[od, h1, h2] for h1 in 1:H) <= y[h2])
        end
    end

    # Hub capacity constraints (throughput)
    # Each hub has limited capacity for flows passing through it
    for h in 1:H
        # Outbound capacity (collection hub)
        @constraint(model,
            sum(prob.demands[od] * sum(w[od, h, h2] for h2 in 1:H) for od in OD_pairs) <=
            prob.hub_capacities[h]
        )
        # Inbound capacity (distribution hub)
        @constraint(model,
            sum(prob.demands[od] * sum(w[od, h1, h] for h1 in 1:H) for od in OD_pairs) <=
            prob.hub_capacities[h]
        )
    end

    return model
end

# Register the problem type
register_problem(
    :hub_location,
    HubLocation,
    "Hub location problem optimizing hub-and-spoke network with economies of scale"
)
