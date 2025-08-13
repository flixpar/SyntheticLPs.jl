using JuMP
using Random
using Distributions

"""
    generate_load_balancing_problem(params::Dict=Dict(); seed::Int=0)

Generate a load balancing problem instance for network traffic optimization.

This models a network load balancing problem where the goal is to minimize the maximum
link utilization while satisfying all traffic demands. The problem scales from small
edge networks to large cloud infrastructure.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_nodes`: Number of nodes in the network (default: 6)
  - `:n_links`: Number of links in the network (default: 10)
  - `:n_demands`: Number of traffic demands (default: 5)
  - `:min_capacity`: Minimum link capacity (default: 100.0)
  - `:max_capacity`: Maximum link capacity (default: 500.0)
  - `:min_demand`: Minimum demand amount (default: 20.0)
  - `:max_demand`: Maximum demand amount (default: 100.0)
  - `:max_path_length`: Maximum length of a path for a demand (default: 4)
  - `:link_density`: Probability of creating a link between two nodes (default: 0.4)
- `seed`: Random seed for reproducibility (default: 0)

# Distribution Usage
- Link capacities: Truncated normal distribution centered around capacity mean
- Traffic demands: Gamma distribution (realistic for network traffic patterns)
- Network topology: Ensures connectivity via spanning tree, then adds random links

# Returns
- `model`: The JuMP model with objective to minimize maximum link utilization
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_load_balancing_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_nodes = get(params, :n_nodes, 6)
    n_links = get(params, :n_links, 10)
    n_demands = get(params, :n_demands, 5)
    min_capacity = get(params, :min_capacity, 100.0)
    max_capacity = get(params, :max_capacity, 500.0)
    min_demand = get(params, :min_demand, 20.0)
    max_demand = get(params, :max_demand, 100.0)
    max_path_length = get(params, :max_path_length, 4)
    link_density = get(params, :link_density, 0.4)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_nodes => n_nodes,
        :n_links => n_links,
        :n_demands => n_demands,
        :min_capacity => min_capacity,
        :max_capacity => max_capacity,
        :min_demand => min_demand,
        :max_demand => max_demand,
        :max_path_length => max_path_length,
        :link_density => link_density
    )
    
    # Generate network topology
    # Start with all possible links
    possible_links = [(i, j) for i in 1:n_nodes for j in 1:n_nodes if i != j]
    
    # Select links based on density parameter or fixed number
    if n_links >= length(possible_links)
        # If requested links exceeds possible links, use all possible links
        links = possible_links
    else
        # Randomly select links
        links = []
        # First ensure the network is connected by adding a spanning tree
        connected_nodes = [1]  # Start with node 1
        remaining_nodes = collect(2:n_nodes)
        
        while !isempty(remaining_nodes)
            # Choose a random node from connected nodes
            from_node = rand(connected_nodes)
            # Choose a random node from remaining nodes
            to_node_idx = rand(1:length(remaining_nodes))
            to_node = remaining_nodes[to_node_idx]
            
            # Add link in both directions to ensure connectivity
            push!(links, (from_node, to_node))
            push!(links, (to_node, from_node))
            
            # Update node sets
            push!(connected_nodes, to_node)
            deleteat!(remaining_nodes, to_node_idx)
        end
        
        # Add more random links up to n_links
        remaining_links = setdiff(possible_links, links)
        shuffle!(remaining_links)
        additional_links = min(n_links - length(links), length(remaining_links))
        append!(links, remaining_links[1:additional_links])
    end
    
    # Ensure uniqueness
    links = unique(links)
    
    # Generate link capacities using normal distribution
    capacities = Dict{Tuple{Int, Int}, Float64}()
    capacity_mean = (min_capacity + max_capacity) / 2
    capacity_std = (max_capacity - min_capacity) / 6  # 99.7% within bounds
    capacity_dist = Truncated(Normal(capacity_mean, capacity_std), min_capacity, max_capacity)
    
    for link in links
        capacities[link] = rand(capacity_dist)
    end
    
    # Generate demands
    possible_demands = [(i, j) for i in 1:n_nodes for j in 1:n_nodes if i != j]
    if n_demands > length(possible_demands)
        n_demands = length(possible_demands)
    end
    
    demand_pairs = sample(possible_demands, n_demands, replace=false)
    demands = Dict{Tuple{Int, Int}, Float64}()
    
    # Use gamma distribution for demands (common for network traffic)
    demand_mean = (min_demand + max_demand) / 2
    demand_std = (max_demand - min_demand) / 6
    # Convert to gamma distribution parameters
    demand_scale = demand_std^2 / demand_mean
    demand_shape = demand_mean / demand_scale
    demand_dist = Truncated(Gamma(demand_shape, demand_scale), min_demand, max_demand)
    
    for demand_pair in demand_pairs
        demands[demand_pair] = rand(demand_dist)
    end
    
    # Generate paths for each demand
    paths = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    
    for demand_pair in keys(demands)
        source, target = demand_pair
        
        # Generate a random path from source to target
        current = source
        path = []
        visited = Set([source])
        path_length = 0
        
        while current != target && path_length < max_path_length
            # Find all possible next hops from current node
            next_hops = [link[2] for link in links if link[1] == current]
            
            # Filter out already visited nodes if possible
            unvisited_hops = setdiff(next_hops, visited)
            if !isempty(unvisited_hops)
                next_hop = rand(unvisited_hops)
            elseif !isempty(next_hops)
                next_hop = rand(next_hops)
            else
                # Dead end, no valid path
                break
            end
            
            # Add link to path
            push!(path, (current, next_hop))
            path_length += 1
            
            # Update state
            push!(visited, next_hop)
            current = next_hop
        end
        
        # If we reached the target, save the path
        if current == target
            paths[demand_pair] = path
        else
            # If we couldn't reach the target, create a direct link if it exists
            direct_link = (source, target)
            if direct_link in links
                paths[demand_pair] = [direct_link]
            else
                # If no direct link, remove this demand
                delete!(demands, demand_pair)
            end
        end
    end
    
    # Store generated data in params
    actual_params[:links] = links
    actual_params[:capacities] = capacities
    actual_params[:demands] = demands
    actual_params[:paths] = paths
    
    # Create model
    model = Model()
    
    # Variables
    @variable(model, u >= 0)  # Maximum link utilization
    @variable(model, f[links] >= 0)  # Flow on each link
    
    # Objective: minimize maximum link utilization
    @objective(model, Min, u)
    
    # Constraints: link utilization
    for link in links
        @constraint(model, f[link] <= u * capacities[link])
    end
    
    # Constraints: flow conservation
    for (demand_pair, demand_amount) in demands
        if haskey(paths, demand_pair)
            path_links = paths[demand_pair]
            for link in path_links
                @constraint(model, f[link] >= demand_amount)
            end
        end
    end
    
    return model, actual_params
end

"""
    sample_load_balancing_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a load balancing problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_load_balancing_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Determine scale based on target variables
    if target_variables <= 250
        # Small scale: edge network or small cluster
        scale = :small
        base_nodes = round(Int, rand(Uniform(5, 20)))
        density_dist = Uniform(0.4, 0.7)  # Higher density for small networks
        capacity_mean = 500.0
        capacity_std = 150.0
        demand_mean = 50.0
        demand_std = 20.0
    elseif target_variables <= 1000
        # Medium scale: data center or regional network  
        scale = :medium
        base_nodes = round(Int, rand(Uniform(20, 60)))
        density_dist = Uniform(0.25, 0.5)  # Moderate density
        capacity_mean = 2000.0
        capacity_std = 600.0
        demand_mean = 150.0
        demand_std = 60.0
    else
        # Large scale: cloud infrastructure or global network
        scale = :large
        base_nodes = round(Int, rand(Uniform(50, 150)))
        density_dist = Uniform(0.15, 0.35)  # Lower density for large networks
        capacity_mean = 8000.0
        capacity_std = 2000.0
        demand_mean = 500.0
        demand_std = 200.0
    end
    
    # Start with node count and density
    params[:n_nodes] = base_nodes
    params[:link_density] = rand(density_dist)
    
    # Start with defaults - target_variables = 1 + n_links
    target_links = max(1, target_variables - 1)
    params[:n_links] = target_links
    
    # Iteratively adjust to get closer to target
    for iteration in 1:15
        current_vars = calculate_load_balancing_variable_count(params)
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust n_links directly since it's the main parameter affecting variable count
        if current_vars < target_variables
            max_possible = params[:n_nodes] * (params[:n_nodes] - 1)
            params[:n_links] = min(max_possible, params[:n_links] + max(1, round(Int, (target_variables - current_vars) * 0.8)))
        elseif current_vars > target_variables
            params[:n_links] = max(params[:n_nodes] - 1, params[:n_links] - max(1, round(Int, (current_vars - target_variables) * 0.8)))
        end
    end
    
    # Set number of demands based on network size - typically 30-70% of possible node pairs
    max_demands = params[:n_nodes] * (params[:n_nodes] - 1)
    demand_ratio = rand(Uniform(0.3, 0.7))
    params[:n_demands] = max(1, round(Int, max_demands * demand_ratio))
    
    # Set capacity parameters using distributions
    capacity_lower = max(10.0, rand(Truncated(Normal(capacity_mean * 0.3, capacity_std * 0.2), 10.0, capacity_mean)))
    capacity_upper = capacity_lower + rand(Truncated(Normal(capacity_mean * 1.2, capacity_std), capacity_mean * 0.5, capacity_mean * 3.0))
    params[:min_capacity] = capacity_lower
    params[:max_capacity] = capacity_upper
    
    # Set demand parameters using distributions
    demand_lower = max(1.0, rand(Truncated(Normal(demand_mean * 0.2, demand_std * 0.1), 1.0, demand_mean * 0.5)))
    demand_upper = demand_lower + rand(Truncated(Normal(demand_mean * 0.8, demand_std), demand_mean * 0.3, demand_mean * 2.0))
    params[:min_demand] = demand_lower
    params[:max_demand] = demand_upper
    
    # Path length varies with network scale
    if scale == :small
        params[:max_path_length] = rand(DiscreteUniform(2, 4))
    elseif scale == :medium
        params[:max_path_length] = rand(DiscreteUniform(3, 6))
    else
        params[:max_path_length] = rand(DiscreteUniform(4, 8))
    end
    
    return params
end

"""
    sample_load_balancing_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a load balancing problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_load_balancing_parameters(size::Symbol=:medium; seed::Int=0)
    Random.seed!(seed)
    
    # Map size categories to realistic target variable ranges
    target_map = Dict(
        :small => rand(50:250),      # Small cluster/edge network
        :medium => rand(250:1000),   # Data center/regional network
        :large => rand(1000:10000)   # Cloud infrastructure/global network
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_load_balancing_parameters(target_map[size]; seed=seed)
end

"""
    calculate_load_balancing_variable_count(params::Dict)

Calculate the number of variables in a load balancing problem.

# Arguments
- `params`: Dictionary of problem parameters containing :n_nodes, :n_links, :link_density

# Returns
- Number of variables (1 for maximum utilization + number of links for flow variables)
"""
function calculate_load_balancing_variable_count(params::Dict)
    n_nodes = get(params, :n_nodes, 6)
    n_links = get(params, :n_links, 10)
    link_density = get(params, :link_density, 0.4)
    
    # Calculate number of actual links that would be generated
    # Start with all possible links
    possible_links = n_nodes * (n_nodes - 1)  # All directed pairs excluding self-loops
    
    # If n_links >= possible_links, use all possible links
    if n_links >= possible_links
        actual_links = possible_links
    else
        # Otherwise use the specified n_links
        actual_links = n_links
    end
    
    # Variables: 1 for maximum utilization (u) + flow variables for each link
    return 1 + actual_links
end

# Register the problem type
register_problem(
    :load_balancing,
    generate_load_balancing_problem,
    sample_load_balancing_parameters,
    "Load balancing problem that minimizes maximum link utilization in a network while satisfying traffic demands. Scales from small edge networks (50-250 vars) to large cloud infrastructure (1000-10000 vars) with realistic capacity and demand distributions."
)