using JuMP
using Random
using Distributions

"""
Network flow problem variants.

# Variants
- `flow_max_flow`: Maximize flow from source to sink
- `flow_min_cost`: Minimize cost for required flow amount
- `flow_multi_source_sink`: Multiple sources and sinks
- `flow_node_capacities`: Nodes have capacity limits (not just arcs)
- `flow_gain_loss`: Arcs can have gain/loss multipliers (leaky pipes)
- `flow_time_expanded`: Dynamic flow over time periods
- `flow_reliable`: Require k arc-disjoint paths for redundancy
- `flow_min_max_util`: Minimize maximum arc utilization (load balancing)
"""
@enum NetworkFlowVariant begin
    flow_max_flow
    flow_min_cost
    flow_multi_source_sink
    flow_node_capacities
    flow_gain_loss
    flow_time_expanded
    flow_reliable
    flow_min_max_util
end

"""
    NetworkFlowProblem <: ProblemGenerator

Generator for network flow problems with multiple variants.

# Fields
- `n_nodes::Int`: Number of nodes in the network
- `source_node::Int`: Source node (primary)
- `sink_node::Int`: Sink node (primary)
- `arcs::Vector{Tuple{Int,Int}}`: List of arcs
- `capacities::Dict{Tuple{Int,Int},Float64}`: Arc capacities
- `costs::Dict{Tuple{Int,Int},Float64}`: Arc costs
- `variant::NetworkFlowVariant`: Problem variant
- `target_flow::Union{Float64,Nothing}`: Required flow amount
- Plus variant-specific fields
"""
struct NetworkFlowProblem <: ProblemGenerator
    n_nodes::Int
    source_node::Int
    sink_node::Int
    arcs::Vector{Tuple{Int,Int}}
    capacities::Dict{Tuple{Int,Int},Float64}
    costs::Dict{Tuple{Int,Int},Float64}
    variant::NetworkFlowVariant
    target_flow::Union{Float64,Nothing}
    # Multi source/sink variant
    sources::Vector{Int}
    sinks::Vector{Int}
    source_supplies::Dict{Int,Float64}
    sink_demands::Dict{Int,Float64}
    # Node capacities variant
    node_capacities::Union{Dict{Int,Float64}, Nothing}
    # Gain/loss variant
    gain_factors::Union{Dict{Tuple{Int,Int},Float64}, Nothing}
    # Time-expanded variant
    n_periods::Int
    time_arc_capacities::Union{Dict{Tuple{Int,Int,Int},Float64}, Nothing}
    storage_capacities::Union{Dict{Int,Float64}, Nothing}
    # Reliable flow variant
    n_disjoint_paths::Int
    # Min-max utilization variant
    max_utilization::Float64
end

# Backwards compatibility
function NetworkFlowProblem(n_nodes::Int, source_node::Int, sink_node::Int,
                            arcs::Vector{Tuple{Int,Int}},
                            capacities::Dict{Tuple{Int,Int},Float64},
                            costs::Dict{Tuple{Int,Int},Float64},
                            flow_objective::Symbol,
                            target_flow::Union{Float64,Nothing})
    variant = flow_objective == :max_flow ? flow_max_flow : flow_min_cost
    NetworkFlowProblem(
        n_nodes, source_node, sink_node, arcs, capacities, costs, variant, target_flow,
        [source_node], [sink_node], Dict{Int,Float64}(), Dict{Int,Float64}(),
        nothing, nothing, 1, nothing, nothing, 1, 1.0
    )
end

"""
    NetworkFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                       variant::NetworkFlowVariant=flow_max_flow)

Construct a network flow problem instance with the specified variant.
"""
function NetworkFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                            variant::NetworkFlowVariant=flow_max_flow)
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

    for n in min_nodes:max_nodes
        possible_arcs = n * (n - 1)
        if round(Int, possible_arcs * target_density) >= target_variables * 0.9
            n_nodes = n
            break
        end
    end

    source_node = 1
    sink_node = n_nodes

    # Generate network topology
    arcs = generate_connected_network(n_nodes, target_variables, source_node, sink_node)

    # Generate capacities and costs
    min_capacity, max_capacity = capacity_range
    min_cost, max_cost = cost_range
    capacities = Dict{Tuple{Int,Int}, Float64}()
    costs = Dict{Tuple{Int,Int}, Float64}()

    for arc in arcs
        capacities[arc] = rand(Uniform(min_capacity, max_capacity))
        costs[arc] = rand(Uniform(min_cost, max_cost))
    end

    # Initialize variant-specific fields
    target_flow = nothing
    sources = [source_node]
    sinks = [sink_node]
    source_supplies = Dict{Int,Float64}()
    sink_demands = Dict{Int,Float64}()
    node_capacities = nothing
    gain_factors = nothing
    n_periods = 1
    time_arc_capacities = nothing
    storage_capacities = nothing
    n_disjoint_paths = 1
    max_utilization = 1.0

    # Generate variant-specific data
    if variant == flow_min_cost
        # Set target flow for min cost objective
        min_cap = minimum(values(capacities))
        target_flow = min_cap * rand(Uniform(0.5, 0.9))

    elseif variant == flow_multi_source_sink
        # Multiple sources and sinks
        n_sources = rand(2:max(2, n_nodes ÷ 4))
        n_sinks = rand(2:max(2, n_nodes ÷ 4))

        # Select sources (near beginning) and sinks (near end)
        all_nodes = collect(1:n_nodes)
        sources = sort(sample(all_nodes[1:n_nodes÷2], min(n_sources, n_nodes÷2), replace=false))
        sinks = sort(sample(all_nodes[n_nodes÷2+1:end], min(n_sinks, n_nodes÷2), replace=false))

        # Assign supplies and demands
        total_capacity = sum(values(capacities)) / 2
        for s in sources
            source_supplies[s] = total_capacity / length(sources) * rand(Uniform(0.8, 1.2))
        end
        for t in sinks
            sink_demands[t] = total_capacity / length(sinks) * rand(Uniform(0.8, 1.2))
        end

        # Balance supply and demand
        total_supply = sum(values(source_supplies))
        total_demand = sum(values(sink_demands))
        if total_supply > total_demand
            for t in sinks
                sink_demands[t] *= total_supply / total_demand
            end
        else
            for s in sources
                source_supplies[s] *= total_demand / total_supply
            end
        end

    elseif variant == flow_node_capacities
        # Add capacity limits on nodes
        node_capacities = Dict{Int,Float64}()
        for i in 1:n_nodes
            if i != source_node && i != sink_node
                # Node capacity based on adjacent arc capacities
                in_cap = sum(capacities[arc] for arc in arcs if arc[2] == i; init=0.0)
                out_cap = sum(capacities[arc] for arc in arcs if arc[1] == i; init=0.0)
                node_capacities[i] = max(in_cap, out_cap) * rand(Uniform(0.5, 1.0))
            end
        end

    elseif variant == flow_gain_loss
        # Add gain/loss factors (modeling leaky pipes, chemical reactions, etc.)
        gain_factors = Dict{Tuple{Int,Int},Float64}()
        for arc in arcs
            if rand() < 0.3  # 30% of arcs have gain/loss
                gain_factors[arc] = rand(Uniform(0.85, 1.15))  # ±15% gain/loss
            else
                gain_factors[arc] = 1.0
            end
        end

    elseif variant == flow_time_expanded
        # Time-expanded network
        n_periods = rand(3:min(8, max(3, target_variables ÷ length(arcs))))

        time_arc_capacities = Dict{Tuple{Int,Int,Int},Float64}()
        for t in 1:n_periods, arc in arcs
            # Time-varying capacities
            base_cap = capacities[arc]
            time_factor = 0.7 + 0.6 * sin(2π * t / n_periods)
            time_arc_capacities[(arc[1], arc[2], t)] = base_cap * time_factor
        end

        # Storage at nodes
        storage_capacities = Dict{Int,Float64}()
        for i in 2:n_nodes-1
            if rand() < 0.5
                storage_capacities[i] = sum(capacities[arc] for arc in arcs if arc[2] == i; init=0.0) *
                                        rand(Uniform(0.3, 0.8))
            end
        end

    elseif variant == flow_reliable
        # Require multiple disjoint paths
        n_disjoint_paths = rand(2:min(3, length(arcs) ÷ 5))

    elseif variant == flow_min_max_util
        # Minimize maximum utilization
        max_utilization = rand(Uniform(0.5, 0.9))
    end

    # Handle feasibility
    if feasibility_status == infeasible
        if variant == flow_min_cost
            # Require more flow than capacity allows
            source_out_cap = sum(capacities[arc] for arc in arcs if arc[1] == source_node; init=0.0)
            sink_in_cap = sum(capacities[arc] for arc in arcs if arc[2] == sink_node; init=0.0)
            max_possible_flow = min(source_out_cap, sink_in_cap)
            target_flow = max_possible_flow * 1.5
        elseif variant == flow_multi_source_sink
            # Demand exceeds supply
            for s in sources
                source_supplies[s] *= 0.3
            end
        elseif variant == flow_node_capacities
            # Node capacities too tight
            for i in keys(node_capacities)
                node_capacities[i] *= 0.1
            end
        elseif variant == flow_reliable
            # Require more disjoint paths than network can support
            n_disjoint_paths = length(arcs)
        elseif variant == flow_min_max_util
            # Impossible utilization limit
            max_utilization = 0.01
        else
            # Reduce capacities to make flow impossible
            for arc in arcs
                capacities[arc] *= 0.01
            end
        end
    elseif feasibility_status == feasible
        if variant == flow_min_cost
            # Ensure achievable target flow
            source_out_cap = sum(capacities[arc] for arc in arcs if arc[1] == source_node; init=Inf)
            sink_in_cap = sum(capacities[arc] for arc in arcs if arc[2] == sink_node; init=Inf)
            min_cap = min(source_out_cap, sink_in_cap)
            target_flow = min(target_flow !== nothing ? target_flow : min_cap * 0.5, min_cap * 0.8)
        elseif variant == flow_reliable
            n_disjoint_paths = min(n_disjoint_paths, 2)
        end
    end

    return NetworkFlowProblem(
        n_nodes, source_node, sink_node, arcs, capacities, costs, variant, target_flow,
        sources, sinks, source_supplies, sink_demands,
        node_capacities, gain_factors, n_periods, time_arc_capacities, storage_capacities,
        n_disjoint_paths, max_utilization
    )
end

"""
Generate a connected network with the specified number of nodes and arcs.
"""
function generate_connected_network(n_nodes::Int, n_arcs::Int, source::Int, sink::Int)
    arcs = Set{Tuple{Int,Int}}()

    # Create path from source to sink
    for i in source:(sink-1)
        push!(arcs, (i, i+1))
    end

    # Add connections from source and to sink
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

Build a JuMP model for the network flow problem based on its variant.
"""
function build_model(prob::NetworkFlowProblem)
    model = Model()

    if prob.variant == flow_max_flow
        @variable(model, flow[arc in prob.arcs] >= 0)

        source_out_arcs = [(i, j) for (i, j) in prob.arcs if i == prob.source_node]
        if !isempty(source_out_arcs)
            @objective(model, Max, sum(flow[arc] for arc in source_out_arcs))
        else
            @objective(model, Max, 0)
        end

        for arc in prob.arcs
            @constraint(model, flow[arc] <= prob.capacities[arc])
        end

        for i in 1:prob.n_nodes
            if i != prob.source_node && i != prob.sink_node
                flow_in = [flow[arc] for arc in prob.arcs if arc[2] == i]
                flow_out = [flow[arc] for arc in prob.arcs if arc[1] == i]
                if !isempty(flow_in) || !isempty(flow_out)
                    @constraint(model, sum(flow_in) == sum(flow_out))
                end
            end
        end

    elseif prob.variant == flow_min_cost
        @variable(model, flow[arc in prob.arcs] >= 0)

        @objective(model, Min, sum(prob.costs[arc] * flow[arc] for arc in prob.arcs))

        for arc in prob.arcs
            @constraint(model, flow[arc] <= prob.capacities[arc])
        end

        for i in 1:prob.n_nodes
            if i != prob.source_node && i != prob.sink_node
                flow_in = [flow[arc] for arc in prob.arcs if arc[2] == i]
                flow_out = [flow[arc] for arc in prob.arcs if arc[1] == i]
                if !isempty(flow_in) || !isempty(flow_out)
                    @constraint(model, sum(flow_in) == sum(flow_out))
                end
            end
        end

        # Flow requirement
        source_out_arcs = [(i, j) for (i, j) in prob.arcs if i == prob.source_node]
        if !isempty(source_out_arcs) && prob.target_flow !== nothing
            @constraint(model, sum(flow[arc] for arc in source_out_arcs) >= prob.target_flow)
        end

    elseif prob.variant == flow_multi_source_sink
        @variable(model, flow[arc in prob.arcs] >= 0)

        @objective(model, Min, sum(prob.costs[arc] * flow[arc] for arc in prob.arcs))

        for arc in prob.arcs
            @constraint(model, flow[arc] <= prob.capacities[arc])
        end

        # Source supply constraints
        for s in prob.sources
            out_arcs = [arc for arc in prob.arcs if arc[1] == s]
            if !isempty(out_arcs) && haskey(prob.source_supplies, s)
                @constraint(model, sum(flow[arc] for arc in out_arcs) <= prob.source_supplies[s])
            end
        end

        # Sink demand constraints
        for t in prob.sinks
            in_arcs = [arc for arc in prob.arcs if arc[2] == t]
            if !isempty(in_arcs) && haskey(prob.sink_demands, t)
                @constraint(model, sum(flow[arc] for arc in in_arcs) >= prob.sink_demands[t])
            end
        end

        # Flow conservation at intermediate nodes
        for i in 1:prob.n_nodes
            if i ∉ prob.sources && i ∉ prob.sinks
                flow_in = [flow[arc] for arc in prob.arcs if arc[2] == i]
                flow_out = [flow[arc] for arc in prob.arcs if arc[1] == i]
                if !isempty(flow_in) || !isempty(flow_out)
                    @constraint(model, sum(flow_in) == sum(flow_out))
                end
            end
        end

    elseif prob.variant == flow_node_capacities
        @variable(model, flow[arc in prob.arcs] >= 0)
        @variable(model, node_flow[1:prob.n_nodes] >= 0)

        source_out_arcs = [(i, j) for (i, j) in prob.arcs if i == prob.source_node]
        @objective(model, Max, sum(flow[arc] for arc in source_out_arcs))

        for arc in prob.arcs
            @constraint(model, flow[arc] <= prob.capacities[arc])
        end

        # Node capacity constraints
        for i in 1:prob.n_nodes
            flow_in = [flow[arc] for arc in prob.arcs if arc[2] == i]
            flow_out = [flow[arc] for arc in prob.arcs if arc[1] == i]

            if !isempty(flow_in)
                @constraint(model, node_flow[i] == sum(flow_in))
            end

            if haskey(prob.node_capacities, i)
                @constraint(model, node_flow[i] <= prob.node_capacities[i])
            end

            if i != prob.source_node && i != prob.sink_node
                if !isempty(flow_in) || !isempty(flow_out)
                    @constraint(model, sum(flow_in) == sum(flow_out))
                end
            end
        end

    elseif prob.variant == flow_gain_loss
        @variable(model, flow[arc in prob.arcs] >= 0)

        source_out_arcs = [(i, j) for (i, j) in prob.arcs if i == prob.source_node]
        @objective(model, Max, sum(flow[arc] for arc in source_out_arcs))

        for arc in prob.arcs
            @constraint(model, flow[arc] <= prob.capacities[arc])
        end

        # Flow conservation with gain/loss
        for i in 1:prob.n_nodes
            if i != prob.source_node && i != prob.sink_node
                in_arcs = [arc for arc in prob.arcs if arc[2] == i]
                out_arcs = [arc for arc in prob.arcs if arc[1] == i]

                if !isempty(in_arcs) || !isempty(out_arcs)
                    # Flow in (adjusted by gain factors) = flow out
                    flow_in_adjusted = isempty(in_arcs) ? 0.0 :
                        sum(prob.gain_factors[arc] * flow[arc] for arc in in_arcs)
                    flow_out_sum = isempty(out_arcs) ? 0.0 : sum(flow[arc] for arc in out_arcs)
                    @constraint(model, flow_in_adjusted == flow_out_sum)
                end
            end
        end

    elseif prob.variant == flow_time_expanded
        # Time-expanded network
        @variable(model, flow[arc in prob.arcs, 1:prob.n_periods] >= 0)
        @variable(model, storage[i in 2:prob.n_nodes-1, 0:prob.n_periods] >= 0)

        # Maximize total flow reaching sink over all time periods
        sink_in_arcs = [arc for arc in prob.arcs if arc[2] == prob.sink_node]
        @objective(model, Max, sum(flow[arc, t] for arc in sink_in_arcs, t in 1:prob.n_periods))

        # Time-varying capacity constraints
        for arc in prob.arcs, t in 1:prob.n_periods
            cap = get(prob.time_arc_capacities, (arc[1], arc[2], t), prob.capacities[arc])
            @constraint(model, flow[arc, t] <= cap)
        end

        # Initial storage
        for i in 2:prob.n_nodes-1
            @constraint(model, storage[i, 0] == 0)
        end

        # Storage capacity
        for i in 2:prob.n_nodes-1, t in 1:prob.n_periods
            if haskey(prob.storage_capacities, i)
                @constraint(model, storage[i, t] <= prob.storage_capacities[i])
            end
        end

        # Flow conservation over time
        for i in 2:prob.n_nodes-1, t in 1:prob.n_periods
            in_arcs = [arc for arc in prob.arcs if arc[2] == i]
            out_arcs = [arc for arc in prob.arcs if arc[1] == i]

            flow_in = isempty(in_arcs) ? 0.0 : sum(flow[arc, t] for arc in in_arcs)
            flow_out = isempty(out_arcs) ? 0.0 : sum(flow[arc, t] for arc in out_arcs)

            @constraint(model, storage[i, t-1] + flow_in == storage[i, t] + flow_out)
        end

    elseif prob.variant == flow_reliable
        # Require k disjoint paths - model with path-indexed flows
        @variable(model, flow[arc in prob.arcs, 1:prob.n_disjoint_paths] >= 0)
        @variable(model, path_flow[1:prob.n_disjoint_paths] >= 0)

        @objective(model, Max, sum(path_flow[k] for k in 1:prob.n_disjoint_paths))

        # Capacity constraints (shared across paths)
        for arc in prob.arcs
            @constraint(model, sum(flow[arc, k] for k in 1:prob.n_disjoint_paths) <= prob.capacities[arc])
        end

        # Path flow constraints
        for k in 1:prob.n_disjoint_paths
            source_out = [arc for arc in prob.arcs if arc[1] == prob.source_node]
            sink_in = [arc for arc in prob.arcs if arc[2] == prob.sink_node]

            if !isempty(source_out)
                @constraint(model, sum(flow[arc, k] for arc in source_out) == path_flow[k])
            end
            if !isempty(sink_in)
                @constraint(model, sum(flow[arc, k] for arc in sink_in) == path_flow[k])
            end

            # Conservation per path
            for i in 1:prob.n_nodes
                if i != prob.source_node && i != prob.sink_node
                    in_arcs = [arc for arc in prob.arcs if arc[2] == i]
                    out_arcs = [arc for arc in prob.arcs if arc[1] == i]
                    if !isempty(in_arcs) || !isempty(out_arcs)
                        @constraint(model, sum(flow[arc, k] for arc in in_arcs) ==
                                          sum(flow[arc, k] for arc in out_arcs))
                    end
                end
            end
        end

    elseif prob.variant == flow_min_max_util
        @variable(model, flow[arc in prob.arcs] >= 0)
        @variable(model, max_util >= 0)

        @objective(model, Min, max_util)

        # Capacity constraints with utilization tracking
        for arc in prob.arcs
            @constraint(model, flow[arc] <= prob.capacities[arc])
            @constraint(model, flow[arc] <= max_util * prob.capacities[arc])
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

        # Minimum flow requirement
        source_out_arcs = [(i, j) for (i, j) in prob.arcs if i == prob.source_node]
        min_flow = sum(prob.capacities[arc] for arc in prob.arcs) * 0.1
        if !isempty(source_out_arcs)
            @constraint(model, sum(flow[arc] for arc in source_out_arcs) >= min_flow)
        end

        # Maximum utilization limit
        @constraint(model, max_util <= prob.max_utilization)
    end

    return model
end

# Register the problem type
register_problem(
    :network_flow,
    NetworkFlowProblem,
    "Network flow problem with variants including max flow, min cost, multi-source/sink, node capacities, gain/loss, time-expanded, reliable paths, and min-max utilization"
)
