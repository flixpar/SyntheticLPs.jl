using JuMP
using Random


"""
    generate_network_flow_problem(params::Dict=Dict(); seed::Int=0)

Generate a network flow problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_nodes`: Number of nodes in the network (default: 6)
  - `:n_arcs`: Number of arcs in the network (default: 10)
  - `:capacity_range`: Tuple (min, max) for arc capacities (default: (10.0, 100.0))
  - `:cost_range`: Tuple (min, max) for arc costs (default: (1.0, 20.0))
  - `:source_node`: Source node (default: 1)
  - `:sink_node`: Sink node (default: n_nodes)
  - `:flow_objective`: :max_flow or :min_cost (default: :max_flow)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_network_flow_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_nodes = get(params, :n_nodes, 6)
    n_arcs = get(params, :n_arcs, 10)
    capacity_range = get(params, :capacity_range, (10.0, 100.0))
    cost_range = get(params, :cost_range, (1.0, 20.0))
    source_node = get(params, :source_node, 1)
    sink_node = get(params, :sink_node, n_nodes)
    flow_objective = get(params, :flow_objective, :max_flow)
    
    # Validate parameters
    if n_arcs > n_nodes * (n_nodes - 1)
        n_arcs = n_nodes * (n_nodes - 1)  # Maximum possible arcs without self-loops
    end
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_nodes => n_nodes,
        :n_arcs => n_arcs,
        :capacity_range => capacity_range,
        :cost_range => cost_range,
        :source_node => source_node,
        :sink_node => sink_node,
        :flow_objective => flow_objective
    )
    
    # Generate network topology ensuring connectivity
    arcs = generate_connected_network(n_nodes, n_arcs, source_node, sink_node)
    
    # Generate capacities and costs for each arc
    min_capacity, max_capacity = capacity_range
    min_cost, max_cost = cost_range
    capacities = Dict{Tuple{Int, Int}, Float64}()
    costs = Dict{Tuple{Int, Int}, Float64}()
    
    for arc in arcs
        capacities[arc] = round(rand() * (max_capacity - min_capacity) + min_capacity, digits=2)
        costs[arc] = round(rand() * (max_cost - min_cost) + min_cost, digits=2)
    end
    
    # Store generated data in params
    actual_params[:arcs] = arcs
    actual_params[:capacities] = capacities
    actual_params[:costs] = costs
    
    # Model
    model = Model()
    
    # Variables
    @variable(model, flow[arc in arcs] >= 0)
    
    # Objective function
    if flow_objective == :max_flow
        # Maximize total flow out of source
        source_out_arcs = [(i, j) for (i, j) in arcs if i == source_node]
        if !isempty(source_out_arcs)
            @objective(model, Max, sum(flow[arc] for arc in source_out_arcs))
        else
            @objective(model, Max, 0)
        end
    else  # :min_cost
        # Minimize total cost of flow
        @objective(model, Min, sum(costs[arc] * flow[arc] for arc in arcs))
        
        # For min cost flow, we need to specify a target flow amount
        # Set target flow as a fraction of minimum bottleneck capacity
        target_flow = get(params, :target_flow, minimum(values(capacities)) * 0.8)
        
        # Flow requirement: total flow out of source equals target
        source_out_arcs = [(i, j) for (i, j) in arcs if i == source_node]
        if !isempty(source_out_arcs)
            @constraint(model, sum(flow[arc] for arc in source_out_arcs) == target_flow)
        end
        
        actual_params[:target_flow] = target_flow
    end
    
    # Capacity constraints
    for arc in arcs
        @constraint(model, flow[arc] <= capacities[arc])
    end
    
    # Flow conservation constraints
    for i in 1:n_nodes
        if i != source_node && i != sink_node
            # Flow in = Flow out for intermediate nodes
            flow_in = [flow[arc] for arc in arcs if arc[2] == i]
            flow_out = [flow[arc] for arc in arcs if arc[1] == i]
            
            if !isempty(flow_in) || !isempty(flow_out)
                @constraint(model, sum(flow_in) == sum(flow_out))
            end
        end
    end
    
    return model, actual_params
end

"""
    generate_connected_network(n_nodes::Int, n_arcs::Int, source::Int, sink::Int)

Generate a connected network with the specified number of nodes and arcs,
ensuring connectivity from source to sink.
"""
function generate_connected_network(n_nodes::Int, n_arcs::Int, source::Int, sink::Int)
    # Start with a spanning tree to ensure connectivity
    arcs = Set{Tuple{Int, Int}}()
    
    # Create a simple path from source to sink
    for i in source:(sink-1)
        push!(arcs, (i, i+1))
    end
    
    # Add some reverse connections and cross connections for realism
    # Connect some nodes back to earlier nodes (but not creating cycles naively)
    for i in 2:n_nodes
        if i != source && i != sink && rand() < 0.3  # 30% chance
            # Add connection to source
            push!(arcs, (source, i))
        end
        if i != source && i != sink && rand() < 0.3  # 30% chance
            # Add connection to sink
            push!(arcs, (i, sink))
        end
    end
    
    # Add additional random arcs to reach target count
    all_possible_arcs = [(i, j) for i in 1:n_nodes for j in 1:n_nodes if i != j]
    remaining_arcs = [arc for arc in all_possible_arcs if arc ∉ arcs]
    
    # Shuffle and add arcs until we reach the target
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
    sample_network_flow_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a network flow problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters

# Details
For network flow: target_variables = n_arcs
We optimize for realistic n_nodes and density values that yield the target number of arcs.
"""
function sample_network_flow_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Target: n_arcs = target_variables
    # For network flow problems, we want realistic network topologies
    
    # Set realistic ranges based on target size
    if target_variables <= 100
        # Small networks
        min_nodes = 4
        max_nodes = 15
        min_density = 0.2
        max_density = 0.8
    elseif target_variables <= 500
        # Medium networks  
        min_nodes = 10
        max_nodes = 35
        min_density = 0.1
        max_density = 0.6
    else
        # Large networks
        min_nodes = 20
        max_nodes = 100
        min_density = 0.05
        max_density = 0.3
    end
    
    best_n_nodes = min_nodes
    best_density = min_density
    best_error = Inf
    
    # Search for optimal n_nodes and density
    for n_nodes in min_nodes:max_nodes
        max_possible_arcs = n_nodes * (n_nodes - 1)
        
        # Skip if even minimum density would exceed our target
        if max_possible_arcs * min_density > target_variables * 1.1
            continue
        end
        
        # Calculate required density for this n_nodes
        required_density = target_variables / max_possible_arcs
        
        # Check if this density is realistic
        if required_density >= min_density && required_density <= max_density
            # Calculate actual arcs with this combination
            actual_arcs = round(Int, max_possible_arcs * required_density)
            error = abs(actual_arcs - target_variables) / target_variables
            
            if error < best_error
                best_error = error
                best_n_nodes = n_nodes
                best_density = required_density
            end
        end
    end
    
    # If we couldn't find a good solution within 10% error, use a heuristic approach
    if best_error > 0.1
        # Use square root heuristic as fallback
        # For network flow, assume moderate density and solve for n_nodes
        target_density = if target_variables <= 100
            0.4  # Higher density for small networks
        elseif target_variables <= 500
            0.2  # Medium density for medium networks
        else
            0.1  # Lower density for large networks
        end
        
        n_nodes_approx = max(min_nodes, min(max_nodes, round(Int, sqrt(target_variables / target_density))))
        max_possible_arcs = n_nodes_approx * (n_nodes_approx - 1)
        
        if max_possible_arcs > 0
            density_approx = max(min_density, min(max_density, target_variables / max_possible_arcs))
        else
            density_approx = target_density
        end
        
        best_n_nodes = n_nodes_approx
        best_density = density_approx
    end
    
    params[:n_nodes] = best_n_nodes
    params[:n_arcs] = min(target_variables, round(Int, best_n_nodes * (best_n_nodes - 1) * best_density))
    
    # Set capacity and cost ranges based on problem size
    if target_variables <= 100
        # Small problem
        params[:capacity_range] = (5.0, 100.0)
        params[:cost_range] = (1.0, 10.0)
    elseif target_variables <= 500
        # Medium problem
        params[:capacity_range] = (10.0, 500.0)
        params[:cost_range] = (1.0, 25.0)
    else
        # Large problem
        params[:capacity_range] = (50.0, 2000.0)
        params[:cost_range] = (1.0, 50.0)
    end
    
    # Choose objective type based on problem size (larger problems more likely to be min-cost)
    if target_variables <= 100
        params[:flow_objective] = rand() < 0.7 ? :max_flow : :min_cost
    else
        params[:flow_objective] = rand() < 0.4 ? :max_flow : :min_cost
    end
    
    params[:source_node] = 1
    params[:sink_node] = params[:n_nodes]
    
    return params
end

"""
    sample_network_flow_parameters(size::Symbol; seed::Int=0)

Legacy function for backward compatibility. Sample realistic parameters for a network flow problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_network_flow_parameters(size::Symbol; seed::Int=0)
    Random.seed!(seed)
    
    # Map size categories to realistic target variable counts for network flow
    target_map = Dict(
        :small => rand(50:250),
        :medium => rand(250:1000),
        :large => rand(1000:10000)
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_network_flow_parameters(target_map[size]; seed=seed)
end

"""
    calculate_network_flow_variable_count(params::Dict)

Calculate the total number of variables for a network flow problem.

# Arguments
- `params`: Dictionary of problem parameters containing :n_arcs

# Returns
- Total number of variables in the problem
"""
function calculate_network_flow_variable_count(params::Dict)
    # Extract parameters with defaults
    n_arcs = get(params, :n_arcs, 10)
    
    # Variables:
    # - Flow variables flow[arc] for each arc in arcs: n_arcs variables
    # Total: n_arcs
    
    return n_arcs
end

# Register the problem type
register_problem(
    :network_flow,
    generate_network_flow_problem,
    sample_network_flow_parameters,
    "Network flow problem that maximizes flow from source to sink or minimizes cost subject to capacity constraints"
)