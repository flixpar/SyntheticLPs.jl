using JuMP
using Random

"""
    NetworkFlowProblem <: ProblemGenerator

Generator for network flow problems that optimize flow from source to sink.

# Fields
- `n_nodes::Int`: Number of nodes in the network
- `source_node::Int`: Source node
- `sink_node::Int`: Sink node
- `arcs::Vector{Tuple{Int,Int}}`: List of arcs in the network
- `capacities::Dict{Tuple{Int,Int},Float64}`: Capacity of each arc
- `costs::Dict{Tuple{Int,Int},Float64}`: Cost per unit of flow on each arc
- `flow_objective::Symbol`: Objective type (:max_flow or :min_cost)
- `target_flow::Union{Float64,Nothing}`: Target flow for min_cost objective
"""
struct NetworkFlowProblem <: ProblemGenerator
    n_nodes::Int
    source_node::Int
    sink_node::Int
    arcs::Vector{Tuple{Int,Int}}
    capacities::Dict{Tuple{Int,Int},Float64}
    costs::Dict{Tuple{Int,Int},Float64}
    flow_objective::Symbol
    target_flow::Union{Float64,Nothing}
end

"""
    NetworkFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a network flow problem instance.

# Arguments
- `target_variables`: Target number of variables (arcs)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function NetworkFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem scale
    if target_variables <= 100
        min_nodes, max_nodes = 4, 15
        capacity_range = (10.0, 100.0)
        cost_range = (1.0, 10.0)
    elseif target_variables <= 500
        min_nodes, max_nodes = 10, 35
        capacity_range = (10.0, 500.0)
        cost_range = (1.0, 25.0)
    else
        min_nodes, max_nodes = 20, 100
        capacity_range = (50.0, 2000.0)
        cost_range = (1.0, 50.0)
    end

    # Calculate appropriate number of nodes
    n_nodes = min_nodes + 2
    target_density = target_variables <= 100 ? 0.4 : target_variables <= 500 ? 0.2 : 0.1

    # Iteratively find good n_nodes
    for n in min_nodes:max_nodes
        possible_arcs = n * (n - 1)
        if round(Int, possible_arcs * target_density) >= target_variables * 0.9
            n_nodes = n
            break
        end
    end

    source_node = 1
    sink_node = n_nodes

    # Generate network topology ensuring connectivity
    arcs = generate_connected_network(n_nodes, target_variables, source_node, sink_node)

    # Generate capacities and costs
    min_capacity, max_capacity = capacity_range
    min_cost, max_cost = cost_range
    capacities = Dict{Tuple{Int,Int}, Float64}()
    costs = Dict{Tuple{Int,Int}, Float64}()

    for arc in arcs
        capacities[arc] = round(rand() * (max_capacity - min_capacity) + min_capacity, digits=2)
        costs[arc] = round(rand() * (max_cost - min_cost) + min_cost, digits=2)
    end

    # Determine objective type
    flow_objective = target_variables <= 100 ? (rand() < 0.7 ? :max_flow : :min_cost) : (rand() < 0.4 ? :max_flow : :min_cost)

    # Set target flow for min_cost objective
    target_flow = nothing
    if flow_objective == :min_cost
        target_flow = minimum(values(capacities)) * 0.8
    end

    return NetworkFlowProblem(n_nodes, source_node, sink_node, arcs, capacities, costs, flow_objective, target_flow)
end

"""
    generate_connected_network(n_nodes::Int, n_arcs::Int, source::Int, sink::Int)

Generate a connected network with the specified number of nodes and arcs.
"""
function generate_connected_network(n_nodes::Int, n_arcs::Int, source::Int, sink::Int)
    arcs = Set{Tuple{Int,Int}}()

    # Create path from source to sink
    for i in source:(sink-1)
        push!(arcs, (i, i+1))
    end

    # Add some connections back to source and to sink for realism
    for i in 2:n_nodes
        if i != source && i != sink && rand() < 0.3
            push!(arcs, (source, i))
        end
        if i != source && i != sink && rand() < 0.3
            push!(arcs, (i, sink))
        end
    end

    # Add additional random arcs
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
    build_model(prob::NetworkFlowProblem)

Build a JuMP model for the network flow problem.
"""
function build_model(prob::NetworkFlowProblem)
    model = Model()

    # Variables
    @variable(model, flow[arc in prob.arcs] >= 0)

    # Objective
    if prob.flow_objective == :max_flow
        source_out_arcs = [(i, j) for (i, j) in prob.arcs if i == prob.source_node]
        if !isempty(source_out_arcs)
            @objective(model, Max, sum(flow[arc] for arc in source_out_arcs))
        else
            @objective(model, Max, 0)
        end
    else  # :min_cost
        @objective(model, Min, sum(prob.costs[arc] * flow[arc] for arc in prob.arcs))

        # Flow requirement
        source_out_arcs = [(i, j) for (i, j) in prob.arcs if i == prob.source_node]
        if !isempty(source_out_arcs) && prob.target_flow !== nothing
            @constraint(model, sum(flow[arc] for arc in source_out_arcs) == prob.target_flow)
        end
    end

    # Capacity constraints
    for arc in prob.arcs
        @constraint(model, flow[arc] <= prob.capacities[arc])
    end

    # Flow conservation
    for i in 1:prob.n_nodes
        if i != prob.source_node && i != prob.sink_node
            flow_in = [flow[arc] for arc in prob.arcs if arc[2] == i]
            flow_out = [flow[arc] for arc in prob.arcs if arc[1] == i]

            if !isempty(flow_in) || !isempty(flow_out)
                @constraint(model, sum(flow_in) == sum(flow_out))
            end
        end
    end

    return model
end

# Register the problem type
register_problem(
    :network_flow,
    NetworkFlowProblem,
    "Network flow problem that maximizes flow from source to sink or minimizes cost subject to capacity constraints"
)
