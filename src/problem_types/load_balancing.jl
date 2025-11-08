using JuMP
using Random
using Distributions

"""
    LoadBalancingProblem <: ProblemGenerator

Generator for load balancing problems in network traffic optimization.

# Fields
- `n_nodes::Int`: Number of nodes in the network
- `links::Vector{Tuple{Int,Int}}`: List of directed links in the network
- `capacities::Dict{Tuple{Int,Int},Float64}`: Capacity of each link
- `demands::Dict{Tuple{Int,Int},Float64}`: Traffic demand between node pairs
- `paths::Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}`: Path (sequence of links) for each demand
"""
struct LoadBalancingProblem <: ProblemGenerator
    n_nodes::Int
    links::Vector{Tuple{Int,Int}}
    capacities::Dict{Tuple{Int,Int},Float64}
    demands::Dict{Tuple{Int,Int},Float64}
    paths::Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}
end

"""
    LoadBalancingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a load balancing problem instance.

# Arguments
- `target_variables`: Target number of variables (1 + number of links)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function LoadBalancingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine scale
    if target_variables <= 250
        # Small scale
        base_nodes = round(Int, rand(Uniform(5, 20)))
        density_dist = Uniform(0.4, 0.7)
        capacity_mean = 500.0
        capacity_std = 150.0
        demand_mean = 50.0
        demand_std = 20.0
        max_path_length = rand(DiscreteUniform(2, 4))
    elseif target_variables <= 1000
        # Medium scale
        base_nodes = round(Int, rand(Uniform(20, 60)))
        density_dist = Uniform(0.25, 0.5)
        capacity_mean = 2000.0
        capacity_std = 600.0
        demand_mean = 150.0
        demand_std = 60.0
        max_path_length = rand(DiscreteUniform(3, 6))
    else
        # Large scale
        base_nodes = round(Int, rand(Uniform(50, 150)))
        density_dist = Uniform(0.15, 0.35)
        capacity_mean = 8000.0
        capacity_std = 2000.0
        demand_mean = 500.0
        demand_std = 200.0
        max_path_length = rand(DiscreteUniform(4, 8))
    end

    n_nodes = base_nodes
    link_density = rand(density_dist)

    # Calculate target links
    target_links = max(1, target_variables - 1)
    n_links = target_links

    # Generate network topology
    possible_links = [(i, j) for i in 1:n_nodes for j in 1:n_nodes if i != j]

    if n_links >= length(possible_links)
        links = possible_links
    else
        links = []
        # Ensure connectivity with spanning tree
        connected_nodes = [1]
        remaining_nodes = collect(2:n_nodes)

        while !isempty(remaining_nodes)
            from_node = rand(connected_nodes)
            to_node_idx = rand(1:length(remaining_nodes))
            to_node = remaining_nodes[to_node_idx]

            push!(links, (from_node, to_node))
            push!(links, (to_node, from_node))

            push!(connected_nodes, to_node)
            deleteat!(remaining_nodes, to_node_idx)
        end

        # Add random links up to n_links
        remaining_links = setdiff(possible_links, links)
        shuffle!(remaining_links)
        additional_links = min(n_links - length(links), length(remaining_links))
        append!(links, remaining_links[1:additional_links])
    end

    links = unique(links)

    # Generate capacities
    capacities = Dict{Tuple{Int, Int}, Float64}()
    min_capacity = max(10.0, rand(Truncated(Normal(capacity_mean * 0.3, capacity_std * 0.2), 10.0, capacity_mean)))
    max_capacity = min_capacity + rand(Truncated(Normal(capacity_mean * 1.2, capacity_std), capacity_mean * 0.5, capacity_mean * 3.0))

    capacity_dist = Truncated(Normal((min_capacity + max_capacity) / 2, (max_capacity - min_capacity) / 6), min_capacity, max_capacity)

    for link in links
        capacities[link] = rand(capacity_dist)
    end

    # Generate demands
    possible_demands = [(i, j) for i in 1:n_nodes for j in 1:n_nodes if i != j]
    max_demands = n_nodes * (n_nodes - 1)
    demand_ratio = rand(Uniform(0.3, 0.7))
    n_demands = max(1, round(Int, max_demands * demand_ratio))

    if n_demands > length(possible_demands)
        n_demands = length(possible_demands)
    end

    demand_pairs = sample(possible_demands, n_demands, replace=false)
    demands = Dict{Tuple{Int, Int}, Float64}()

    min_demand = max(1.0, rand(Truncated(Normal(demand_mean * 0.2, demand_std * 0.1), 1.0, demand_mean * 0.5)))
    max_demand = min_demand + rand(Truncated(Normal(demand_mean * 0.8, demand_std), demand_mean * 0.3, demand_mean * 2.0))

    demand_mean_calc = (min_demand + max_demand) / 2
    demand_std_calc = (max_demand - min_demand) / 6
    demand_scale = demand_std_calc^2 / demand_mean_calc
    demand_shape = demand_mean_calc / demand_scale
    demand_dist = Truncated(Gamma(demand_shape, demand_scale), min_demand, max_demand)

    for demand_pair in demand_pairs
        demands[demand_pair] = rand(demand_dist)
    end

    # Generate paths for each demand
    paths = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()

    for demand_pair in keys(demands)
        source, target = demand_pair

        # Generate random path
        current = source
        path = []
        visited = Set([source])
        path_length = 0

        while current != target && path_length < max_path_length
            next_hops = [link[2] for link in links if link[1] == current]

            unvisited_hops = setdiff(next_hops, visited)
            if !isempty(unvisited_hops)
                next_hop = rand(unvisited_hops)
            elseif !isempty(next_hops)
                next_hop = rand(next_hops)
            else
                break
            end

            push!(path, (current, next_hop))
            path_length += 1

            push!(visited, next_hop)
            current = next_hop
        end

        if current == target
            paths[demand_pair] = path
        else
            direct_link = (source, target)
            if direct_link in links
                paths[demand_pair] = [direct_link]
            else
                delete!(demands, demand_pair)
            end
        end
    end

    return LoadBalancingProblem(n_nodes, links, capacities, demands, paths)
end

"""
    build_model(prob::LoadBalancingProblem)

Build a JuMP model for the load balancing problem.

# Arguments
- `prob`: LoadBalancingProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::LoadBalancingProblem)
    model = Model()

    # Variables
    @variable(model, u >= 0)  # Maximum link utilization
    @variable(model, f[prob.links] >= 0)  # Flow on each link

    # Objective: minimize maximum link utilization
    @objective(model, Min, u)

    # Constraints: link utilization
    for link in prob.links
        @constraint(model, f[link] <= u * prob.capacities[link])
    end

    # Constraints: flow conservation
    for (demand_pair, demand_amount) in prob.demands
        if haskey(prob.paths, demand_pair)
            path_links = prob.paths[demand_pair]
            for link in path_links
                @constraint(model, f[link] >= demand_amount)
            end
        end
    end

    return model
end

# Register the problem type
register_problem(
    :load_balancing,
    LoadBalancingProblem,
    "Load balancing problem that minimizes maximum link utilization in a network while satisfying traffic demands"
)
