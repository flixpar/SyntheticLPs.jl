using JuMP
using Random
using Distributions

const CAPACITY_EPS = 1e-6

@inline function canonical_arc(i::Int, j::Int)::Tuple{Int,Int}
    return i < j ? (i, j) : (j, i)
end

function build_adjacency(arcs_subset::AbstractVector{Tuple{Int,Int}}, n_nodes::Int)
    adjacency = [Int[] for _ in 1:n_nodes]
    for (i, j) in arcs_subset
        push!(adjacency[i], j)
        push!(adjacency[j], i)
    end
    return adjacency
end

function shortest_path_with_capacity(
    source::Int,
    sink::Int,
    adjacency::Vector{Vector{Int}},
    remaining_capacity::Dict{Tuple{Int,Int},Float64},
    flow_costs::Dict{Tuple{Int,Int},Float64}
)
    n_nodes = length(adjacency)
    dist = fill(Inf, n_nodes)
    prev = fill(0, n_nodes)
    visited = falses(n_nodes)

    dist[source] = 0.0

    while true
        u = 0
        best = Inf
        for node in 1:n_nodes
            if !visited[node] && dist[node] < best
                best = dist[node]
                u = node
            end
        end

        if u == 0 || best == Inf
            return nothing
        end

        if u == sink
            break
        end

        visited[u] = true

        for v in adjacency[u]
            arc = canonical_arc(u, v)
            cap = get(remaining_capacity, arc, 0.0)
            if cap <= CAPACITY_EPS
                continue
            end

            cost = flow_costs[(u, v)]
            alt = dist[u] + cost
            if alt < dist[v]
                dist[v] = alt
                prev[v] = u
            end
        end
    end

    if dist[sink] == Inf
        return nothing
    end

    path_nodes = Int[]
    node = sink
    while node != 0
        push!(path_nodes, node)
        if node == source
            break
        end
        node = prev[node]
    end

    if isempty(path_nodes) || last(path_nodes) != source
        return nothing
    end

    reverse!(path_nodes)

    path_arcs = Tuple{Int,Int}[]
    for idx in 1:(length(path_nodes) - 1)
        push!(path_arcs, canonical_arc(path_nodes[idx], path_nodes[idx + 1]))
    end

    return path_arcs
end

function can_route_demands(
    arcs_subset,
    link_capacities::Dict{Tuple{Int,Int},Float64},
    commodities::Vector{Dict{Symbol,Any}},
    n_nodes::Int,
    flow_costs::Dict{Tuple{Int,Int},Float64}
)
    arc_list = collect(arcs_subset)
    if isempty(arc_list)
        return false
    end

    remaining_capacity = Dict{Tuple{Int,Int}, Float64}()
    for arc in arc_list
        remaining_capacity[arc] = get(link_capacities, arc, 0.0)
    end

    adjacency = build_adjacency(arc_list, n_nodes)

    # If any node lacks incident arcs, routing is impossible
    if any(isempty(adjacency[node]) for node in 1:n_nodes)
        return false
    end

    commodity_indices = sortperm(1:length(commodities), by=i -> commodities[i][:demand], rev=true)

    for idx in commodity_indices
        commodity = commodities[idx]
        source = commodity[:source]
        sink = commodity[:sink]
        remaining_demand = commodity[:demand]

        attempts = 0
        while remaining_demand > CAPACITY_EPS
            path = shortest_path_with_capacity(source, sink, adjacency, remaining_capacity, flow_costs)
            if path === nothing
                return false
            end

            min_cap = minimum(get(remaining_capacity, arc, 0.0) for arc in path)
            flow = min(remaining_demand, min_cap)

            if flow <= CAPACITY_EPS
                return false
            end

            for arc in path
                remaining_capacity[arc] -= flow
            end

            remaining_demand -= flow
            attempts += 1
            if attempts > length(arc_list) * 5
                return false
            end
        end
    end

    return true
end

"""
    generate_telecom_network_design_problem(params::Dict=Dict(); seed::Int=0)

Generate a telecommunication network design problem instance.

This problem models the design of a telecommunications network by deciding which links to install
and how to route multiple traffic demands (commodities) to minimize total cost while satisfying
capacity constraints. It is a multicommodity network design problem with discrete capacity installation.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_nodes`: Number of nodes in the network (default: 8)
  - `:n_arcs`: Number of potential arcs/links (default: 15)
  - `:n_commodities`: Number of traffic demands (origin-destination pairs) (default: 10)
  - `:grid_width`: Width of geographic area for node placement (default: 1000.0)
  - `:grid_height`: Height of geographic area for node placement (default: 1000.0)
  - `:capacity_modules`: Available capacity modules in Mbps (default: [155.0, 622.0, 2488.0, 9953.0])
  - `:base_installation_cost`: Base cost for installing a link (default: 50000.0)
  - `:cost_per_km`: Additional installation cost per kilometer (default: 100.0)
  - `:flow_cost_per_unit`: Cost per unit of traffic per kilometer (default: 0.01)
  - `:demand_range`: Range for commodity demands in Mbps (default: (1.0, 100.0))
  - `:budget_factor`: Budget as fraction of total installation cost (default: 0.6)
  - `:solution_status`: Desired feasibility status (:feasible [default], :infeasible, :all)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)

# Model Details
Variables:
    - y[arc]: Binary variable, 1 if link is installed on arc
    - f[k,(i,j)]: Continuous flow variable for commodity k on directed arc (i → j)

Objective:
    - Minimize: installation costs + routing costs

Constraints:
    - Flow conservation: at each node, inflow = outflow (except source/sink)
    - Capacity: total flow in both directions on a link ≤ installed capacity * y[arc]
    - Demand satisfaction: each commodity routed from source to destination
    - Budget: total installation cost ≤ budget
"""
function generate_telecom_network_design_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)

    # Extract parameters with defaults
    n_nodes = get(params, :n_nodes, 8)
    n_arcs = get(params, :n_arcs, 15)
    n_commodities = get(params, :n_commodities, 10)
    grid_width = get(params, :grid_width, 1000.0)
    grid_height = get(params, :grid_height, 1000.0)
    capacity_modules = get(params, :capacity_modules, [155.0, 622.0, 2488.0, 9953.0])  # OC-3, OC-12, OC-48, OC-192 in Mbps
    base_installation_cost = get(params, :base_installation_cost, 50000.0)
    cost_per_km = get(params, :cost_per_km, 100.0)
    flow_cost_per_unit = get(params, :flow_cost_per_unit, 0.01)
    demand_range = get(params, :demand_range, (1.0, 100.0))
    budget_factor = get(params, :budget_factor, 0.6)
    solution_status = get(params, :solution_status, :feasible)

    if solution_status isa String
        solution_status = Symbol(lowercase(solution_status))
    end
    if !(solution_status in (:feasible, :infeasible, :all))
        error("Unknown solution_status=$(solution_status). Use :feasible, :infeasible, or :all")
    end

    @assert length(capacity_modules) > 1 "At least two capacity modules are required."

    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_nodes => n_nodes,
        :n_arcs => n_arcs,
        :n_commodities => n_commodities,
        :grid_width => grid_width,
        :grid_height => grid_height,
        :capacity_modules => capacity_modules,
        :base_installation_cost => base_installation_cost,
        :cost_per_km => cost_per_km,
        :flow_cost_per_unit => flow_cost_per_unit,
        :demand_range => demand_range,
        :budget_factor => budget_factor,
        :solution_status => solution_status
    )

    # Generate node locations with realistic geographic distribution
    node_locations = generate_node_locations(n_nodes, grid_width, grid_height)
    actual_params[:node_locations] = node_locations

    # Generate network topology (arcs) based on proximity and connectivity
    arcs = generate_telecom_topology(n_nodes, n_arcs, node_locations)
    unique!(arcs)
    actual_params[:arcs] = arcs

    # Construct directed arcs to represent both orientations of each physical link
    directed_arcs = Vector{Tuple{Int,Int}}()
    for (i, j) in arcs
        push!(directed_arcs, (i, j))
        push!(directed_arcs, (j, i))
    end
    actual_params[:directed_arcs] = directed_arcs

    # Calculate distances for each physical arc
    distances = Dict{Tuple{Int,Int}, Float64}()
    for arc in arcs
        i, j = arc
        loc_i, loc_j = node_locations[i], node_locations[j]
        dist = sqrt((loc_i[1] - loc_j[1])^2 + (loc_i[2] - loc_j[2])^2)
        distances[(i, j)] = dist
        distances[(j, i)] = dist
    end
    actual_params[:distances] = distances

    # Generate installation costs based on distance
    installation_costs = Dict{Tuple{Int,Int}, Float64}()
    link_capacities = Dict{Tuple{Int,Int}, Float64}()

    for arc in arcs
        distance = distances[arc]
        # Select capacity module based on distance (longer links get higher capacity modules)
        capacity_idx = if distance < grid_width * 0.2
            rand(1:length(capacity_modules))
        elseif distance < grid_width * 0.5
            rand(2:length(capacity_modules))
        else
            rand(max(2, length(capacity_modules)-1):length(capacity_modules))
        end
        link_capacities[arc] = capacity_modules[capacity_idx]

        # Installation cost = base + distance-based + capacity-based
        capacity_factor = capacity_idx / length(capacity_modules)
        installation_costs[arc] = base_installation_cost * (0.8 + 0.4 * capacity_factor) +
                                   distance * cost_per_km * (0.9 + 0.2 * rand())
    end
    actual_params[:installation_costs] = installation_costs
    actual_params[:link_capacities] = link_capacities

    # Generate flow costs (proportional to distance) for both directions
    flow_costs = Dict{Tuple{Int,Int}, Float64}()
    for arc in arcs
        i, j = arc
        base_cost = distances[arc] * flow_cost_per_unit * (0.9 + 0.2 * rand())
        flow_costs[(i, j)] = base_cost
        flow_costs[(j, i)] = base_cost
    end
    actual_params[:flow_costs] = flow_costs

    # Build adjacency lists for flow conservation
    outgoing_arcs = Dict{Int, Vector{Tuple{Int,Int}}}()
    incoming_arcs = Dict{Int, Vector{Tuple{Int,Int}}}()
    for node in 1:n_nodes
        outgoing_arcs[node] = Tuple{Int,Int}[]
        incoming_arcs[node] = Tuple{Int,Int}[]
    end
    for arc in directed_arcs
        push!(outgoing_arcs[arc[1]], arc)
        push!(incoming_arcs[arc[2]], arc)
    end

    # Generate commodities (traffic demands) with realistic patterns
    commodities = generate_traffic_commodities(n_nodes, n_commodities, demand_range)
    actual_params[:commodities] = commodities
    total_demand = sum(c[:demand] for c in commodities)
    actual_params[:total_demand] = total_demand

    node_out_demands = zeros(Float64, n_nodes)
    node_in_demands = zeros(Float64, n_nodes)
    for commodity in commodities
        node_out_demands[commodity[:source]] += commodity[:demand]
        node_in_demands[commodity[:sink]] += commodity[:demand]
    end
    actual_params[:node_out_demands] = node_out_demands
    actual_params[:node_in_demands] = node_in_demands

    # Calculate total potential installation cost
    total_installation_cost = sum(values(installation_costs))
    budget = total_installation_cost * budget_factor
    original_budget = budget
    actual_params[:total_installation_cost] = total_installation_cost

    # Enforce feasibility/infeasibility based on solution_status
    if solution_status == :feasible

        # Scale capacities until the full network can explicitly route all traffic
        capacity_scale_factor = 1.0
        max_scale_factor = 50.0
        capacity_feasible = can_route_demands(arcs, link_capacities, commodities, n_nodes, flow_costs)

        while !capacity_feasible && capacity_scale_factor < max_scale_factor
            scale_step = min(1.5, max_scale_factor / capacity_scale_factor)
            capacity_scale_factor *= scale_step
            for arc in arcs
                link_capacities[arc] *= scale_step
            end
            capacity_feasible = can_route_demands(arcs, link_capacities, commodities, n_nodes, flow_costs)
        end

        if !capacity_feasible
            for arc in arcs
                link_capacities[arc] *= 2.0
            end
            capacity_feasible = can_route_demands(arcs, link_capacities, commodities, n_nodes, flow_costs)
        end

        actual_params[:capacity_scale_factor] = capacity_scale_factor
        actual_params[:capacity_feasible_with_all_links] = capacity_feasible

        # Greedily select a subset of links that still supports all traffic
        arc_ratios = [(arc, link_capacities[arc] / installation_costs[arc]) for arc in arcs]
        sort!(arc_ratios, by=x -> x[2], rev=true)
        arc_order = [entry[1] for entry in arc_ratios]

        selected_arcs = Set{Tuple{Int,Int}}()
        selected_cost = 0.0

        # Build a spanning tree to ensure basic connectivity
        nodes_connected = Set([1])
        remaining_nodes = Set(2:n_nodes)
        arc_ratios_copy = copy(arc_ratios)

        while !isempty(remaining_nodes) && !isempty(arc_ratios_copy)
            found_arc = false
            for (idx, (arc, _)) in enumerate(arc_ratios_copy)
                i, j = arc
                if (i in nodes_connected && j in remaining_nodes) || (j in nodes_connected && i in remaining_nodes)
                    push!(selected_arcs, arc)
                    selected_cost += installation_costs[arc]
                    push!(nodes_connected, i, j)
                    delete!(remaining_nodes, i)
                    delete!(remaining_nodes, j)
                    deleteat!(arc_ratios_copy, idx)
                    found_arc = true
                    break
                end
            end
            if !found_arc
                break
            end
        end

        # Ensure we have at least a spanning tree
        if length(selected_arcs) < n_nodes - 1
            for (arc, _) in arc_ratios_copy
                if arc ∉ selected_arcs
                    push!(selected_arcs, arc)
                    selected_cost += installation_costs[arc]
                end
                if length(selected_arcs) >= n_nodes - 1
                    break
                end
            end
        end

        routing_feasible = can_route_demands(selected_arcs, link_capacities, commodities, n_nodes, flow_costs)
        idx = 1
        while !routing_feasible && idx <= length(arc_order)
            arc_to_add = arc_order[idx]
            idx += 1
            if arc_to_add ∈ selected_arcs
                continue
            end
            push!(selected_arcs, arc_to_add)
            selected_cost += installation_costs[arc_to_add]
            routing_feasible = can_route_demands(selected_arcs, link_capacities, commodities, n_nodes, flow_costs)
        end

        if !routing_feasible
            selected_arcs = Set(arcs)
            selected_cost = sum(installation_costs[arc] for arc in arcs)
            routing_feasible = true
        end

        selected_capacity = sum(link_capacities[arc] for arc in selected_arcs)

        # Set budget to allow this feasible solution with some slack
        budget = max(original_budget, selected_cost * (1.05 + 0.15 * rand()))
        actual_params[:feasible_arcs] = collect(selected_arcs)
        actual_params[:feasible_cost] = selected_cost
        actual_params[:feasible_capacity] = selected_capacity
        actual_params[:routing_feasible] = routing_feasible

    elseif solution_status == :infeasible
        # Make problem infeasible by tightening both budget and link capacities around high-demand nodes

        # Estimate minimum connectivity cost via a greedy spanning tree approximation
        arc_costs = sort([(arc, installation_costs[arc]) for arc in arcs], by=x -> x[2])
        nodes_connected = Set([1])
        remaining_nodes = Set(2:n_nodes)
        mst_cost = 0.0

        for (arc, cost) in arc_costs
            if isempty(remaining_nodes)
                break
            end
            i, j = arc
            if (i in nodes_connected && j in remaining_nodes) || (j in nodes_connected && i in remaining_nodes)
                mst_cost += cost
                push!(nodes_connected, i, j)
                delete!(remaining_nodes, i)
                delete!(remaining_nodes, j)
            end
        end

    budget_multiplier = rand(0.45:0.05:0.75)
    cost_reference = mst_cost > CAPACITY_EPS ? mst_cost : total_installation_cost
    budget = min(original_budget, cost_reference * budget_multiplier)
        actual_params[:mst_cost] = mst_cost
        actual_params[:budget_reduction_factor] = budget / max(original_budget, 1.0)

        # Create targeted capacity bottlenecks around the busiest sources and sinks
        original_capacities = copy(link_capacities)

        n_sources_positive = count(d -> d > CAPACITY_EPS, node_out_demands)
        max_sources_to_target = max(1, min(2, n_sources_positive))
        source_order = sortperm(node_out_demands; rev=true)
        targeted_sources = Int[]
        for node in source_order
            if node_out_demands[node] > CAPACITY_EPS
                push!(targeted_sources, node)
            end
            if length(targeted_sources) >= max_sources_to_target
                break
            end
        end
        if isempty(targeted_sources) && !isempty(commodities)
            push!(targeted_sources, commodities[1][:source])
        end

        n_sinks_positive = count(d -> d > CAPACITY_EPS, node_in_demands)
        max_sinks_to_target = max(1, min(2, n_sinks_positive))
        sink_order = sortperm(node_in_demands; rev=true)
        targeted_sinks = Int[]
        for node in sink_order
            if node_in_demands[node] > CAPACITY_EPS
                push!(targeted_sinks, node)
            end
            if length(targeted_sinks) >= max_sinks_to_target
                break
            end
        end
        if isempty(targeted_sinks) && !isempty(commodities)
            push!(targeted_sinks, commodities[1][:sink])
        end

        source_capacity_totals = Dict{Int, Float64}()
        for node in targeted_sources
            incident_arcs = [arc for arc in arcs if arc[1] == node || arc[2] == node]
            if isempty(incident_arcs)
                continue
            end
            demand_total = node_out_demands[node]
            if demand_total <= CAPACITY_EPS
                continue
            end

            desired_total = demand_total * rand(0.25:0.05:0.6)
            desired_total = max(desired_total, CAPACITY_EPS * length(incident_arcs))

            weights = rand(length(incident_arcs))
            weight_sum = sum(weights)
            if weight_sum <= CAPACITY_EPS
                weights .= 1.0
                weight_sum = length(weights)
            end
            weights ./= weight_sum

            updated_total = 0.0
            for (idx, arc) in enumerate(incident_arcs)
                cap_target = max(CAPACITY_EPS, desired_total * weights[idx])
                link_capacities[arc] = min(link_capacities[arc], cap_target)
                updated_total += link_capacities[arc]
            end
            source_capacity_totals[node] = updated_total
        end

        sink_capacity_totals = Dict{Int, Float64}()
        for node in targeted_sinks
            incident_arcs = [arc for arc in arcs if arc[1] == node || arc[2] == node]
            if isempty(incident_arcs)
                continue
            end
            demand_total = node_in_demands[node]
            if demand_total <= CAPACITY_EPS
                continue
            end

            desired_total = demand_total * rand(0.25:0.05:0.6)
            desired_total = max(desired_total, CAPACITY_EPS * length(incident_arcs))

            weights = rand(length(incident_arcs))
            weight_sum = sum(weights)
            if weight_sum <= CAPACITY_EPS
                weights .= 1.0
                weight_sum = length(weights)
            end
            weights ./= weight_sum

            updated_total = 0.0
            for (idx, arc) in enumerate(incident_arcs)
                cap_target = max(CAPACITY_EPS, desired_total * weights[idx])
                link_capacities[arc] = min(link_capacities[arc], cap_target)
                updated_total += link_capacities[arc]
            end
            sink_capacity_totals[node] = updated_total
        end

        routing_possible_after_reduction = can_route_demands(arcs, link_capacities, commodities, n_nodes, flow_costs)

        fallback_scaling_applied = false
        if routing_possible_after_reduction
            # Apply a global capacity contraction and re-test
            for _ in 1:3
                if !routing_possible_after_reduction
                    break
                end
                for arc in arcs
                    link_capacities[arc] = max(CAPACITY_EPS, link_capacities[arc] * 0.5)
                end
                routing_possible_after_reduction = can_route_demands(arcs, link_capacities, commodities, n_nodes, flow_costs)
            end

            if routing_possible_after_reduction
                # Directly cap capacity around the busiest source to force infeasibility
                critical_node = isempty(targeted_sources) ? commodities[1][:source] : targeted_sources[1]
                incident_arcs = [arc for arc in arcs if arc[1] == critical_node || arc[2] == critical_node]
                if !isempty(incident_arcs)
                    desired_total = node_out_demands[critical_node] * 0.1
                    desired_total = max(desired_total, CAPACITY_EPS * length(incident_arcs))
                    per_arc_cap = desired_total / length(incident_arcs)
                    for arc in incident_arcs
                        cap_target = max(CAPACITY_EPS, per_arc_cap)
                        link_capacities[arc] = min(link_capacities[arc], cap_target)
                    end
                end
                routing_possible_after_reduction = can_route_demands(arcs, link_capacities, commodities, n_nodes, flow_costs)
                fallback_scaling_applied = true
            end
        end

        actual_params[:infeasible_source_nodes] = targeted_sources
        actual_params[:infeasible_sink_nodes] = targeted_sinks
        actual_params[:source_capacity_totals] = source_capacity_totals
        actual_params[:sink_capacity_totals] = sink_capacity_totals
        actual_params[:routing_feasible_after_infeasible_adjustment] = routing_possible_after_reduction
        actual_params[:infeasible_fallback_scaling_applied] = fallback_scaling_applied
    end

    actual_params[:budget] = budget

    # Build the optimization model
    model = Model()

    # Decision variables
    @variable(model, y[arc in arcs], Bin)  # 1 if link is installed
    @variable(model, f[k=1:n_commodities, arc in directed_arcs] >= 0)  # flow of commodity k on directed arc

    # Objective: Minimize total cost (installation + routing)
    @objective(model, Min,
        sum(installation_costs[arc] * y[arc] for arc in arcs) +
        sum(flow_costs[arc] * sum(f[k, arc] for k in 1:n_commodities) for arc in directed_arcs)
    )

    # Flow conservation constraints for each commodity at each node
    for k in 1:n_commodities
        commodity = commodities[k]
        source = commodity[:source]
        sink = commodity[:sink]
        demand = commodity[:demand]

        for node in 1:n_nodes
            # Arcs leaving and entering the node (directed)
            out_arcs = outgoing_arcs[node]
            in_arcs = incoming_arcs[node]

            # Flow balance
            out_flow = isempty(out_arcs) ? 0.0 : sum(f[k, arc] for arc in out_arcs)
            in_flow = isempty(in_arcs) ? 0.0 : sum(f[k, arc] for arc in in_arcs)

            if node == source
                @constraint(model, out_flow - in_flow == demand)
            elseif node == sink
                @constraint(model, out_flow - in_flow == -demand)
            else
                @constraint(model, out_flow - in_flow == 0)
            end
        end
    end

    # Capacity constraints: total flow on each physical link (both directions) ≤ capacity if installed
    for arc in arcs
        forward_arc = arc
        reverse_arc = (arc[2], arc[1])
        @constraint(model,
            sum(f[k, forward_arc] + f[k, reverse_arc] for k in 1:n_commodities) <= link_capacities[arc] * y[arc]
        )
    end

    # Budget constraint
    @constraint(model,
        sum(installation_costs[arc] * y[arc] for arc in arcs) <= budget
    )

    return model, actual_params
end

"""
    generate_node_locations(n_nodes::Int, width::Float64, height::Float64)

Generate realistic node locations with clustering (representing cities/population centers).
"""
function generate_node_locations(n_nodes::Int, width::Float64, height::Float64)
    locations = Vector{Tuple{Float64, Float64}}(undef, n_nodes)

    # Create clusters for realistic geographic distribution
    n_clusters = max(1, div(n_nodes, 3))
    cluster_centers = [(width * rand(), height * rand()) for _ in 1:n_clusters]

    for i in 1:n_nodes
        center = rand(cluster_centers)
        # Add normally distributed offset from cluster center
        x = clamp(center[1] + randn() * (width/8), 0, width)
        y = clamp(center[2] + randn() * (height/8), 0, height)
        locations[i] = (x, y)
    end

    return locations
end

"""
    generate_telecom_topology(n_nodes::Int, n_arcs::Int, locations::Vector{Tuple{Float64,Float64}})

Generate network topology based on proximity and realistic connectivity patterns.
Creates a network that prioritizes nearby connections and ensures basic connectivity.
"""
function generate_telecom_topology(n_nodes::Int, n_arcs::Int, locations::Vector{Tuple{Float64,Float64}})
    arcs = Set{Tuple{Int,Int}}()

    # Calculate all pairwise distances
    distances = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:n_nodes
        for j in (i+1):n_nodes
            loc_i, loc_j = locations[i], locations[j]
            dist = sqrt((loc_i[1] - loc_j[1])^2 + (loc_i[2] - loc_j[2])^2)
            distances[(i,j)] = dist
            distances[(j,i)] = dist
        end
    end

    # Start with a minimum spanning tree to ensure connectivity
    nodes_connected = Set([1])
    remaining_nodes = Set(2:n_nodes)

    while !isempty(remaining_nodes) && length(arcs) < n_arcs
        # Find nearest unconnected node
        best_arc = nothing
        best_dist = Inf

        for i in nodes_connected
            for j in remaining_nodes
                if haskey(distances, (i,j)) && distances[(i,j)] < best_dist
                    best_dist = distances[(i,j)]
                    best_arc = (i, j)
                end
            end
        end

        if best_arc !== nothing
            i, j = best_arc
            canonical_arc = i < j ? (i, j) : (j, i)
            push!(arcs, canonical_arc)
            push!(nodes_connected, i, j)
            delete!(remaining_nodes, j)
        else
            break
        end
    end

    # Add additional arcs based on distance (preferring shorter connections)
    all_possible_arcs = [(i, j) for i in 1:n_nodes for j in 1:n_nodes if i < j]
    sort!(all_possible_arcs, by=arc -> distances[arc])

    for arc in all_possible_arcs
        if length(arcs) >= n_arcs
            break
        end
        if arc ∉ arcs
            # Probability decreases with distance (prefer short links)
            if rand() < 0.7  # 70% chance to add shorter links
                push!(arcs, arc)
            end
        end
    end

    # If still need more arcs, add remaining ones randomly
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
    generate_traffic_commodities(n_nodes::Int, n_commodities::Int, demand_range::Tuple{Float64,Float64})

Generate traffic demands with realistic patterns (log-normal distribution, hub traffic).
"""
function generate_traffic_commodities(n_nodes::Int, n_commodities::Int, demand_range::Tuple{Float64,Float64})
    commodities = Vector{Dict{Symbol,Any}}(undef, n_commodities)

    min_demand, max_demand = demand_range

    # Identify hub nodes (lower numbered nodes are more important)
    hub_threshold = max(1, div(n_nodes, 4))

    for k in 1:n_commodities
        # Select source and sink
        source = rand(1:n_nodes)
        sink = rand(1:n_nodes)
        while sink == source
            sink = rand(1:n_nodes)
        end

        # Generate demand with log-normal distribution
        # Hub-to-hub traffic has higher demands
        is_hub_traffic = (source <= hub_threshold && sink <= hub_threshold)

        if is_hub_traffic
            # High traffic between hubs (upper range)
            mean_log = log((min_demand + max_demand) / 2)
            demand = exp(rand(Normal(mean_log + 0.5, 0.4)))
        else
            # Lower traffic for non-hub pairs
            mean_log = log((min_demand + max_demand) / 2)
            demand = exp(rand(Normal(mean_log - 0.3, 0.6)))
        end

        # Clamp to reasonable range
        demand = clamp(demand, min_demand, max_demand * 1.5)

        commodities[k] = Dict{Symbol,Any}(
            :source => source,
            :sink => sink,
            :demand => round(demand, digits=2)
        )
    end

    return commodities
end

"""
    sample_telecom_network_design_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a telecommunication network design problem targeting
approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters

# Details
For telecom network design: target_variables = n_arcs × (2 × n_commodities + 1)
We optimize for realistic n_arcs and n_commodities values that yield the target.
"""
function sample_telecom_network_design_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)

    params = Dict{Symbol, Any}()

    # Target: n_arcs × (2 × n_commodities + 1) = target_variables
    # We need to find realistic values of n_arcs and n_commodities

    # Set realistic ranges based on problem scale
    if target_variables <= 100
        # Small problems (metro networks)
        min_nodes = 4
        max_nodes = 12
        min_arcs = 5
        max_arcs = 25
        min_commodities = 3
        max_commodities = 15
    elseif target_variables <= 1000
        # Medium problems (regional networks)
        min_nodes = 8
        max_nodes = 35
        min_arcs = 15
        max_arcs = 120
        min_commodities = 10
        max_commodities = 80
    else
        # Large problems (national/international networks)
        min_nodes = 20
        max_nodes = 120
        min_arcs = 50
        max_arcs = 600
        min_commodities = 20
        max_commodities = 300
    end

    best_n_arcs = min_arcs
    best_n_commodities = min_commodities
    best_n_nodes = min_nodes
    best_error = Inf

    # Search for optimal n_arcs and n_commodities
    for n_arcs in min_arcs:max_arcs
        # Given n_arcs, solve for n_commodities
        # target_variables = n_arcs × (2 × n_commodities + 1)
        # n_commodities = ((target_variables / n_arcs) - 1) / 2
        n_commodities_exact = ((target_variables / n_arcs) - 1) / 2

        # Check if this gives reasonable n_commodities
        if n_commodities_exact >= min_commodities && n_commodities_exact <= max_commodities
            n_commodities = clamp(round(Int, n_commodities_exact), min_commodities, max_commodities)

            # Calculate actual variables with this combination
            actual_vars = n_arcs * (2 * n_commodities + 1)
            error = abs(actual_vars - target_variables) / target_variables

            if error < best_error
                best_error = error
                best_n_arcs = n_arcs
                best_n_commodities = n_commodities
            end
        end
    end

    # If we couldn't find a good solution within 10% error, use a heuristic approach
    if best_error > 0.1
        # Use square root heuristic as fallback
        # Assume n_commodities ≈ n_arcs / 2 (typical ratio)
        # target_variables ≈ n_arcs × (n_arcs + 1) ≈ n_arcs²
        n_arcs_approx = max(min_arcs, min(max_arcs, round(Int, sqrt(target_variables))))
        n_commodities_approx = clamp(round(Int, ((target_variables / n_arcs_approx) - 1) / 2), min_commodities, max_commodities)

        best_n_arcs = n_arcs_approx
        best_n_commodities = n_commodities_approx
        best_error = abs(best_n_arcs * (2 * best_n_commodities + 1) - target_variables) / target_variables
    end

    # Determine n_nodes based on n_arcs (realistic topology)
    # For telecom networks, typical density: n_arcs ≈ 1.5 × n_nodes to 2.5 × n_nodes
    density_factor = 1.5 + rand() * 1.0  # Between 1.5 and 2.5
    best_n_nodes = max(min_nodes, min(max_nodes, round(Int, best_n_arcs / density_factor)))

    # Ensure we have enough nodes for the number of arcs
    max_possible_arcs = div(best_n_nodes * (best_n_nodes - 1), 2)
    while max_possible_arcs < best_n_arcs && best_n_nodes < max_nodes
        best_n_nodes += 1
        max_possible_arcs = div(best_n_nodes * (best_n_nodes - 1), 2)
    end

    params[:n_nodes] = best_n_nodes
    params[:n_arcs] = best_n_arcs
    params[:n_commodities] = best_n_commodities

    # Set realistic parameters based on problem size
    if target_variables <= 100
        # Small metro networks
        params[:grid_width] = rand(100.0:50.0:500.0)
        params[:grid_height] = rand(100.0:50.0:500.0)
        params[:base_installation_cost] = rand(10000.0:5000.0:50000.0)
        params[:cost_per_km] = rand(50.0:10.0:150.0)
        params[:flow_cost_per_unit] = rand(0.005:0.001:0.02)
        params[:demand_range] = (rand(1.0:5.0), rand(50.0:10.0:150.0))
        params[:budget_factor] = rand(0.4:0.05:0.7)
        params[:capacity_modules] = [155.0, 622.0, 2488.0]  # OC-3, OC-12, OC-48
    elseif target_variables <= 1000
        # Medium regional networks
        params[:grid_width] = rand(500.0:100.0:2000.0)
        params[:grid_height] = rand(500.0:100.0:2000.0)
        params[:base_installation_cost] = rand(30000.0:10000.0:100000.0)
        params[:cost_per_km] = rand(80.0:20.0:200.0)
        params[:flow_cost_per_unit] = rand(0.01:0.005:0.05)
        params[:demand_range] = (rand(5.0:15.0), rand(100.0:20.0:300.0))
        params[:budget_factor] = rand(0.5:0.05:0.8)
        params[:capacity_modules] = [155.0, 622.0, 2488.0, 9953.0]  # OC-3, OC-12, OC-48, OC-192
    else
        # Large national/international networks
        params[:grid_width] = rand(2000.0:500.0:10000.0)
        params[:grid_height] = rand(2000.0:500.0:10000.0)
        params[:base_installation_cost] = rand(100000.0:50000.0:500000.0)
        params[:cost_per_km] = rand(150.0:50.0:500.0)
        params[:flow_cost_per_unit] = rand(0.02:0.01:0.1)
        params[:demand_range] = (rand(10.0:30.0), rand(200.0:100.0:1000.0))
        params[:budget_factor] = rand(0.6:0.05:0.9)
        params[:capacity_modules] = [622.0, 2488.0, 9953.0, 39813.0]  # OC-12, OC-48, OC-192, OC-768
    end

    return params
end

"""
    sample_telecom_network_design_parameters(size::Symbol; seed::Int=0)

Legacy function for backward compatibility. Sample realistic parameters for a
telecommunication network design problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_telecom_network_design_parameters(size::Symbol; seed::Int=0)
    Random.seed!(seed)

    target_variables = if size == :small
        rand(50:250)
    elseif size == :medium
        rand(250:1000)
    elseif size == :large
        rand(1000:10000)
    else
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end

    return sample_telecom_network_design_parameters(target_variables; seed=seed)
end

"""
    calculate_telecom_network_design_variable_count(params::Dict)

Calculate the total number of variables for a telecommunication network design problem.

# Arguments
- `params`: Dictionary of problem parameters containing :n_arcs and :n_commodities

# Returns
- Total number of variables in the problem
"""
function calculate_telecom_network_design_variable_count(params::Dict)
    # Extract parameters with defaults
    n_arcs = get(params, :n_arcs, 15)
    n_commodities = get(params, :n_commodities, 10)

    # Variables:
    # - Binary variables y[arc] for each arc: n_arcs variables
    # - Continuous variables f[k, (i,j)] for each commodity on each directed arc: 2 × n_arcs × n_commodities variables
    # Total: n_arcs + (2 × n_arcs × n_commodities) = n_arcs × (2 × n_commodities + 1)

    return n_arcs * (2 * n_commodities + 1)
end

# Register the problem type
register_problem(
    :telecom_network_design,
    generate_telecom_network_design_problem,
    sample_telecom_network_design_parameters,
    "Telecommunication network design problem that minimizes installation and routing costs while satisfying capacity constraints and traffic demands"
)
