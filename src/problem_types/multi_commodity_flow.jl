using JuMP
using Random
using Distributions

"""
    MultiCommodityFlow <: ProblemGenerator

Generator for multi-commodity flow problems.

Multi-commodity flow extends single-commodity network flow by allowing multiple
different commodities (e.g., different products, message types, or freight classes)
to share the same network infrastructure. Each commodity has its own source-sink
pairs and demands, but all commodities compete for limited arc capacity.

# Fields
- `n_nodes::Int`: Number of nodes in the network
- `n_arcs::Int`: Number of arcs in the network
- `n_commodities::Int`: Number of commodities
- `arcs::Vector{Tuple{Int,Int}}`: List of directed arcs (i,j)
- `capacities::Dict{Tuple{Int,Int},Float64}`: Capacity for each arc
- `demands::Dict{Int,Float64}`: Demand for each commodity
- `costs::Dict{Tuple{Int,Int},Float64}`: Cost per unit flow on each arc
- `commodities::Vector{Tuple{Int,Int}}`: Source-sink pairs for each commodity
"""
struct MultiCommodityFlow <: ProblemGenerator
    n_nodes::Int
    n_arcs::Int
    n_commodities::Int
    arcs::Vector{Tuple{Int,Int}}
    capacities::Dict{Tuple{Int,Int},Float64}
    demands::Dict{Int,Float64}
    costs::Dict{Tuple{Int,Int},Float64}
    commodities::Vector{Tuple{Int,Int}}
end

"""
    MultiCommodityFlow(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-commodity flow problem instance.

# Arguments
- `target_variables`: Target number of variables in the LP formulation
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Details
For multi-commodity flow: target_variables = n_commodities × n_arcs
We optimize for realistic combinations of commodities and arcs that yield the target.
"""
function MultiCommodityFlow(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Sample parameters based on target_variables
    params = sample_parameters_mcf(target_variables, seed)

    n_nodes = params[:n_nodes]
    n_arcs = params[:n_arcs]
    n_commodities = params[:n_commodities]
    capacity_range = params[:capacity_range]
    demand_range = params[:demand_range]
    cost_range = params[:cost_range]
    capacity_utilization = params[:capacity_utilization]

    # Validate parameters
    if n_arcs > n_nodes * (n_nodes - 1)
        n_arcs = n_nodes * (n_nodes - 1)  # Maximum possible arcs without self-loops
    end

    # Generate network topology ensuring connectivity
    arcs = generate_connected_mcf_network(n_nodes, n_arcs)

    # Generate arc capacities using log-normal distribution for realism
    min_capacity, max_capacity = capacity_range
    capacities = Dict{Tuple{Int,Int},Float64}()

    # Use log-normal distribution for more realistic capacity variation
    # (some high-capacity backbone links, many medium-capacity links)
    log_mean = log(sqrt(min_capacity * max_capacity))
    log_std = log(max_capacity / min_capacity) / 4

    for arc in arcs
        # Generate capacity with log-normal distribution
        capacity = exp(rand(Normal(log_mean, log_std)))
        capacity = clamp(capacity, min_capacity, max_capacity)
        capacities[arc] = round(capacity, digits=2)
    end

    # Generate commodity source-sink pairs ensuring diversity
    commodities = generate_commodity_pairs(n_nodes, n_commodities, arcs)

    # Generate commodity demands using log-normal distribution
    # (realistic: some high-demand commodities, many lower-demand ones)
    min_demand, max_demand = demand_range
    demands = Dict{Int,Float64}()

    log_demand_mean = log(sqrt(min_demand * max_demand))
    log_demand_std = log(max_demand / min_demand) / 3

    for k in 1:n_commodities
        demand = exp(rand(Normal(log_demand_mean, log_demand_std)))
        demand = clamp(demand, min_demand, max_demand)
        demands[k] = round(demand, digits=2)
    end

    # Generate arc costs (distance-based with variation)
    min_cost, max_cost = cost_range
    costs = Dict{Tuple{Int,Int},Float64}()

    for arc in arcs
        base_cost = rand() * (max_cost - min_cost) + min_cost
        # Add congestion factor based on capacity (lower capacity = higher cost)
        congestion_factor = 1.0 + 0.3 * (1.0 - (capacities[arc] - min_capacity) / (max_capacity - min_capacity))
        costs[arc] = round(base_cost * congestion_factor, digits=2)
    end

    # Adjust capacities and demands to ensure desired feasibility
    total_demand = sum(values(demands))
    total_capacity = sum(values(capacities))

    solution_status = feasibility_status == feasible ? :feasible :
                     feasibility_status == infeasible ? :infeasible : :all

    if solution_status == :feasible
        # Ensure network can handle demand with target utilization
        # We need total capacity >= total demand / capacity_utilization
        required_total_capacity = total_demand / capacity_utilization

        if total_capacity < required_total_capacity
            # Scale up capacities proportionally
            scale_factor = required_total_capacity / total_capacity
            for arc in arcs
                capacities[arc] = round(capacities[arc] * scale_factor, digits=2)
            end
        end

        # Ensure connectivity: verify each commodity can reach its destination
        # Add strategic arcs if needed to ensure basic connectivity
        for k in 1:n_commodities
            source, sink = commodities[k]
            if !has_path(arcs, source, sink, n_nodes)
                # Add a direct or indirect path
                new_arcs = create_path(source, sink, arcs, n_nodes)
                for new_arc in new_arcs
                    if new_arc ∉ arcs
                        push!(arcs, new_arc)
                        # Add capacity and cost for new arc
                        capacities[new_arc] = round(rand() * (max_capacity - min_capacity) + min_capacity, digits=2)
                        costs[new_arc] = round(rand() * (max_cost - min_cost) + min_cost, digits=2)
                    end
                end
            end
        end

    elseif solution_status == :infeasible
        # Create infeasibility by reducing capacity or increasing demand
        if rand() < 0.5
            # Option 1: Reduce arc capacities to create bottleneck
            scale_factor = rand(Uniform(0.3, 0.7))  # Reduce to 30-70% of current
            for arc in arcs
                capacities[arc] = round(capacities[arc] * scale_factor, digits=2)
            end
        else
            # Option 2: Increase demands beyond network capacity
            scale_factor = rand(Uniform(1.5, 2.5))  # Increase to 150-250%
            for k in 1:n_commodities
                demands[k] = round(demands[k] * scale_factor, digits=2)
            end
        end

        # Add targeted disruptions so at least one commodity cannot be satisfied
        enforce_infeasibility_mcf!(commodities, arcs, capacities, demands)
    end
    # else :all - keep natural randomness

    return MultiCommodityFlow(n_nodes, n_arcs, n_commodities, arcs, capacities, demands, costs, commodities)
end

"""
    build_model(prob::MultiCommodityFlow)

Build a JuMP model for the multi-commodity flow problem.

# Arguments
- `prob`: MultiCommodityFlow instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::MultiCommodityFlow)
    model = Model()

    # Decision variables: flow of each commodity on each arc
    @variable(model, flow[1:prob.n_commodities, arc in prob.arcs] >= 0)

    # Objective: Minimize total cost
    @objective(model, Min,
        sum(prob.costs[arc] * sum(flow[k, arc] for k in 1:prob.n_commodities) for arc in prob.arcs)
    )

    # Capacity constraints: total flow on each arc cannot exceed capacity
    for arc in prob.arcs
        @constraint(model,
            sum(flow[k, arc] for k in 1:prob.n_commodities) <= prob.capacities[arc]
        )
    end

    # Flow conservation constraints for each commodity
    for k in 1:prob.n_commodities
        source, sink = prob.commodities[k]

        for node in 1:prob.n_nodes
            # Arcs entering this node
            inflow_arcs = [arc for arc in prob.arcs if arc[2] == node]
            # Arcs leaving this node
            outflow_arcs = [arc for arc in prob.arcs if arc[1] == node]

            inflow = isempty(inflow_arcs) ? 0 : sum(flow[k, arc] for arc in inflow_arcs)
            outflow = isempty(outflow_arcs) ? 0 : sum(flow[k, arc] for arc in outflow_arcs)

            if node == source
                # At source: outflow - inflow = demand
                @constraint(model, outflow - inflow == prob.demands[k])
            elseif node == sink
                # At sink: inflow - outflow = demand
                @constraint(model, inflow - outflow == prob.demands[k])
            else
                # At intermediate nodes: flow conservation
                @constraint(model, inflow == outflow)
            end
        end
    end

    return model
end

# Helper functions

"""
    sample_parameters_mcf(target_variables::Int, seed::Int)

Sample realistic parameters for a multi-commodity flow problem targeting approximately
the specified number of variables.

For multi-commodity flow: target_variables = n_commodities × n_arcs
We optimize for realistic combinations of commodities and arcs that yield the target.
"""
function sample_parameters_mcf(target_variables::Int, seed::Int)
    Random.seed!(seed)

    params = Dict{Symbol,Any}()

    # Set realistic ranges based on target size
    if target_variables <= 100
        # Small networks
        min_commodities = 2
        max_commodities = 5
        min_nodes = 5
        max_nodes = 15
        min_density = 0.2
        max_density = 0.6
    elseif target_variables <= 500
        # Medium networks
        min_commodities = 3
        max_commodities = 15
        min_nodes = 10
        max_nodes = 30
        min_density = 0.15
        max_density = 0.5
    else
        # Large networks
        min_commodities = 8
        max_commodities = 50
        min_nodes = 15
        max_nodes = 100
        min_density = 0.1
        max_density = 0.4
    end

    best_n_commodities = min_commodities
    best_n_arcs = 10
    best_n_nodes = min_nodes
    best_error = Inf

    # Search for optimal combination
    for n_commodities in min_commodities:max_commodities
        # For each commodity count, find best arc count
        target_arcs = round(Int, target_variables / n_commodities)

        # Find reasonable node count for this arc count
        for n_nodes in min_nodes:max_nodes
            max_possible_arcs = n_nodes * (n_nodes - 1)

            # Skip if even minimum density would exceed our target
            if max_possible_arcs * min_density > target_arcs * 1.1
                continue
            end

            # Calculate required density
            required_density = target_arcs / max_possible_arcs

            # Check if this density is realistic
            if required_density >= min_density && required_density <= max_density
                actual_arcs = round(Int, max_possible_arcs * required_density)
                actual_vars = n_commodities * actual_arcs
                error = abs(actual_vars - target_variables) / target_variables

                if error < best_error
                    best_error = error
                    best_n_commodities = n_commodities
                    best_n_arcs = actual_arcs
                    best_n_nodes = n_nodes
                end
            end
        end
    end

    # If we couldn't find a good solution, use heuristic approach
    if best_error > 0.1
        # Use square root heuristic
        if target_variables <= 100
            n_commodities = rand(2:5)
        elseif target_variables <= 500
            n_commodities = rand(5:15)
        else
            n_commodities = rand(10:50)
        end

        target_arcs = round(Int, target_variables / n_commodities)

        # Estimate nodes from arcs assuming moderate density
        target_density = if target_variables <= 100
            0.4
        elseif target_variables <= 500
            0.3
        else
            0.2
        end

        n_nodes = max(min_nodes, min(max_nodes, round(Int, sqrt(target_arcs / target_density))))
        n_arcs = min(target_arcs, n_nodes * (n_nodes - 1))

        best_n_commodities = n_commodities
        best_n_arcs = n_arcs
        best_n_nodes = n_nodes
    end

    params[:n_commodities] = best_n_commodities
    params[:n_arcs] = best_n_arcs
    params[:n_nodes] = best_n_nodes

    # Set capacity and demand ranges based on problem size
    if target_variables <= 100
        # Small problem
        params[:capacity_range] = (20.0, 200.0)
        params[:demand_range] = (5.0, 50.0)
        params[:cost_range] = (1.0, 15.0)
    elseif target_variables <= 500
        # Medium problem
        params[:capacity_range] = (50.0, 500.0)
        params[:demand_range] = (10.0, 100.0)
        params[:cost_range] = (1.0, 25.0)
    else
        # Large problem
        params[:capacity_range] = (100.0, 2000.0)
        params[:demand_range] = (20.0, 500.0)
        params[:cost_range] = (1.0, 50.0)
    end

    # Capacity utilization: higher for smaller problems (more efficient), lower for larger (more complex)
    params[:capacity_utilization] = if target_variables <= 100
        rand(Uniform(0.6, 0.8))
    elseif target_variables <= 500
        rand(Uniform(0.5, 0.7))
    else
        rand(Uniform(0.4, 0.6))
    end

    return params
end

"""
    generate_connected_mcf_network(n_nodes::Int, n_arcs::Int)

Generate a connected directed network with the specified number of nodes and arcs.
Ensures strong connectivity for multi-commodity flow.
"""
function generate_connected_mcf_network(n_nodes::Int, n_arcs::Int)
    arcs = Set{Tuple{Int,Int}}()

    # Create a cycle to ensure strong connectivity (every node can reach every other node)
    for i in 1:n_nodes
        next_node = (i % n_nodes) + 1
        push!(arcs, (i, next_node))
    end

    # Add some reverse arcs for bidirectional flow
    n_reverse = min(n_nodes ÷ 2, max(0, n_arcs - n_nodes))
    reverse_candidates = [((i % n_nodes) + 1, i) for i in 1:n_nodes]
    shuffle!(reverse_candidates)

    for i in 1:min(n_reverse, length(reverse_candidates))
        arc = (reverse_candidates[i][1], reverse_candidates[i][2])
        if arc ∉ arcs
            push!(arcs, arc)
        end
    end

    # Add shortcut connections for realism (express routes)
    n_shortcuts = min(n_nodes ÷ 3, n_arcs - length(arcs))
    for _ in 1:n_shortcuts
        if length(arcs) >= n_arcs
            break
        end
        # Create shortcuts between distant nodes
        i = rand(1:n_nodes)
        j = (i + rand(2:max(2, n_nodes-1))) % n_nodes + 1
        arc = (i, j)
        if arc ∉ arcs && i != j
            push!(arcs, arc)
        end
    end

    # Add remaining random arcs to reach target count
    all_possible_arcs = [(i, j) for i in 1:n_nodes for j in 1:n_nodes if i != j]
    remaining_arcs = [arc for arc in all_possible_arcs if arc ∉ arcs]

    shuffle!(remaining_arcs)
    for arc in remaining_arcs
        if length(arcs) >= n_arcs
            break
        end
        push!(arcs, arc)
    end

    return collect(arcs)
end

"""
    generate_commodity_pairs(n_nodes::Int, n_commodities::Int, arcs::Vector{Tuple{Int,Int}})

Generate diverse source-sink pairs for commodities.
Ensures variety in distances and different traffic patterns.
"""
function generate_commodity_pairs(n_nodes::Int, n_commodities::Int, arcs::Vector{Tuple{Int,Int}})
    commodities = Vector{Tuple{Int,Int}}()
    used_pairs = Set{Tuple{Int,Int}}()

    for k in 1:n_commodities
        # Try to find a unique source-sink pair
        max_attempts = 100
        for attempt in 1:max_attempts
            source = rand(1:n_nodes)

            # Choose sink with varying distances for realism
            # Mix of short-haul (nearby) and long-haul (distant) commodities
            if rand() < 0.4  # 40% short-haul
                # Nearby destination (within 30% of network)
                offset = rand(1:max(1, n_nodes ÷ 3))
                sink = ((source + offset - 1) % n_nodes) + 1
            else  # 60% long-haul
                # More distant destination
                offset = rand((n_nodes ÷ 3):n_nodes)
                sink = ((source + offset - 1) % n_nodes) + 1
            end

            if source != sink && (source, sink) ∉ used_pairs
                push!(commodities, (source, sink))
                push!(used_pairs, (source, sink))
                break
            end
        end
    end

    # If we couldn't generate enough unique pairs, just create simple pairs
    while length(commodities) < n_commodities
        source = rand(1:n_nodes)
        sink = rand(1:n_nodes)
        if source != sink
            push!(commodities, (source, sink))
        end
    end

    return commodities
end

"""
    has_path(arcs::Vector{Tuple{Int,Int}}, source::Int, sink::Int, n_nodes::Int)

Check if there's a path from source to sink using BFS.
"""
function has_path(arcs::Vector{Tuple{Int,Int}}, source::Int, sink::Int, n_nodes::Int)
    if source == sink
        return true
    end

    # Build adjacency list
    adj = [Int[] for _ in 1:n_nodes]
    for (i, j) in arcs
        push!(adj[i], j)
    end

    # BFS
    visited = Set{Int}([source])
    queue = [source]

    while !isempty(queue)
        current = popfirst!(queue)

        for neighbor in adj[current]
            if neighbor == sink
                return true
            end

            if neighbor ∉ visited
                push!(visited, neighbor)
                push!(queue, neighbor)
            end
        end
    end

    return false
end

"""
    create_path(source::Int, sink::Int, existing_arcs::Vector{Tuple{Int,Int}}, n_nodes::Int)

Create a simple path from source to sink.
"""
function create_path(source::Int, sink::Int, existing_arcs::Vector{Tuple{Int,Int}}, n_nodes::Int)
    # Create a simple 1 or 2-hop path
    if rand() < 0.5
        # Direct path
        return [(source, sink)]
    else
        # 2-hop path through intermediate node
        intermediate = rand(1:n_nodes)
        if intermediate != source && intermediate != sink
            return [(source, intermediate), (intermediate, sink)]
        else
            return [(source, sink)]
        end
    end
end

"""
    enforce_infeasibility_mcf!(commodities, arcs, capacities, demands)

Introduce bottlenecks so that at least one commodity cannot satisfy its demand.
Keeps randomness by selecting different commodities and disruption styles while
guaranteeing an infeasible configuration.
"""
function enforce_infeasibility_mcf!(
    commodities::Vector{Tuple{Int,Int}},
    arcs::Vector{Tuple{Int,Int}},
    capacities::Dict{Tuple{Int,Int},Float64},
    demands::Dict{Int,Float64}
)
    if isempty(commodities)
        return
    end

    commodity_indices = shuffle(collect(eachindex(commodities)))
    max_targets = max(1, min(length(commodity_indices), 3))
    n_targets = rand(1:max_targets)
    selected = commodity_indices[1:n_targets]

    success = false

    for idx in selected
        choice = rand()
        if choice < 0.45
            success |= enforce_source_bottleneck!(idx, commodities, arcs, capacities, demands)
        elseif choice < 0.9
            success |= enforce_sink_bottleneck!(idx, commodities, arcs, capacities, demands)
        else
            source_hit = enforce_source_bottleneck!(idx, commodities, arcs, capacities, demands)
            sink_hit = enforce_sink_bottleneck!(idx, commodities, arcs, capacities, demands)
            success |= (source_hit || sink_hit)
        end

        if success && rand() < 0.6
            break
        end
    end

    if !success
        # Deterministic fallback: cripple the first commodity
        idx = commodity_indices[1]
        source_hit = enforce_source_bottleneck!(idx, commodities, arcs, capacities, demands)
        sink_hit = enforce_sink_bottleneck!(idx, commodities, arcs, capacities, demands)
        success = source_hit || sink_hit

        if !success
            # As a last resort, inflate demand beyond total network capacity
            total_capacity = sum(values(capacities))
            boost = total_capacity <= 0 ? 50.0 : total_capacity * rand(Uniform(1.2, 1.6))
            demands[idx] = round(max(demands[idx], boost), digits=2)
        end
    end
end

function enforce_source_bottleneck!(
    commodity_index::Int,
    commodities::Vector{Tuple{Int,Int}},
    arcs::Vector{Tuple{Int,Int}},
    capacities::Dict{Tuple{Int,Int},Float64},
    demands::Dict{Int,Float64}
)
    source = commodities[commodity_index][1]
    outgoing = [arc for arc in arcs if arc[1] == source]

    if isempty(outgoing)
        return false
    end

    current_total = sum(capacities[arc] for arc in outgoing)
    if current_total <= 1e-6
        return true
    end

    target_total = min(
        demands[commodity_index] * rand(Uniform(0.1, 0.6)),
        current_total * rand(Uniform(0.15, 0.5))
    )

    redistribute_capacity!(outgoing, capacities, target_total)

    new_total = sum(capacities[arc] for arc in outgoing)
    if new_total <= 1e-6
        return true
    end

    if new_total >= demands[commodity_index]
        scale = min(0.95, (demands[commodity_index] * rand(Uniform(0.2, 0.6))) / new_total)
        for arc in outgoing
            capacities[arc] = round(capacities[arc] * scale, digits=2)
        end
        new_total = sum(capacities[arc] for arc in outgoing)
    end

    return new_total + 1e-6 < demands[commodity_index]
end

function enforce_sink_bottleneck!(
    commodity_index::Int,
    commodities::Vector{Tuple{Int,Int}},
    arcs::Vector{Tuple{Int,Int}},
    capacities::Dict{Tuple{Int,Int},Float64},
    demands::Dict{Int,Float64}
)
    sink = commodities[commodity_index][2]
    incoming = [arc for arc in arcs if arc[2] == sink]

    if isempty(incoming)
        return false
    end

    current_total = sum(capacities[arc] for arc in incoming)
    if current_total <= 1e-6
        return true
    end

    target_total = min(
        demands[commodity_index] * rand(Uniform(0.1, 0.6)),
        current_total * rand(Uniform(0.15, 0.5))
    )

    redistribute_capacity!(incoming, capacities, target_total)

    new_total = sum(capacities[arc] for arc in incoming)
    if new_total <= 1e-6
        return true
    end

    if new_total >= demands[commodity_index]
        scale = min(0.95, (demands[commodity_index] * rand(Uniform(0.2, 0.6))) / new_total)
        for arc in incoming
            capacities[arc] = round(capacities[arc] * scale, digits=2)
        end
        new_total = sum(capacities[arc] for arc in incoming)
    end

    return new_total + 1e-6 < demands[commodity_index]
end

function redistribute_capacity!(
    arc_subset::Vector{Tuple{Int,Int}},
    capacities::Dict{Tuple{Int,Int},Float64},
    target_total::Float64
)
    if isempty(arc_subset)
        return
    end

    if target_total <= 1e-6
        for arc in arc_subset
            capacities[arc] = 0.0
        end
        return
    end

    if length(arc_subset) == 1
        arc = arc_subset[1]
        capacities[arc] = round(min(capacities[arc], target_total), digits=2)
        return
    end

    weights = rand(Dirichlet(fill(1.5, length(arc_subset))))
    for (arc, weight) in zip(arc_subset, weights)
        capacities[arc] = round(min(capacities[arc], target_total * weight), digits=2)
    end

    total = sum(capacities[arc] for arc in arc_subset)
    if total > target_total
        scale = target_total / total
        for arc in arc_subset
            capacities[arc] = round(capacities[arc] * scale, digits=2)
        end
    end
end

# Register the problem type
register_problem(
    :multi_commodity_flow,
    MultiCommodityFlow,
    "Multi-commodity flow problem that routes multiple commodities through a shared network with capacity constraints"
)
