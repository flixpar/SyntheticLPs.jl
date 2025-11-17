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

const NETWORK_FLOW_VARIANT_ORDER = [:logistics, :urban_water, :power_transmission, :generic]
const NETWORK_FLOW_VARIANT_WEIGHTS = Dict(
    :logistics => 0.45,
    :urban_water => 0.25,
    :power_transmission => 0.2,
    :generic => 0.1,
)

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

    params = base_network_parameters(target_variables)
    variant = select_network_flow_variant()
    generator = NETWORK_FLOW_VARIANT_GENERATORS[variant]
    return generator(target_variables, feasibility_status, params)
end

"""
    base_network_parameters(target_variables)

Return common parameter ranges used across all network flow variants.
"""
function base_network_parameters(target_variables::Int)
    if target_variables <= 100
        return (;
            min_nodes = 4,
            max_nodes = 15,
            capacity_range = (10.0, 100.0),
            cost_range = (1.0, 10.0),
            target_density = 0.4,
        )
    elseif target_variables <= 500
        return (;
            min_nodes = 10,
            max_nodes = 35,
            capacity_range = (10.0, 500.0),
            cost_range = (1.0, 25.0),
            target_density = 0.2,
        )
    else
        return (;
            min_nodes = 20,
            max_nodes = 100,
            capacity_range = (50.0, 2000.0),
            cost_range = (1.0, 50.0),
            target_density = 0.1,
        )
    end
end

function select_network_flow_variant()
    total_weight = sum(values(NETWORK_FLOW_VARIANT_WEIGHTS))
    draw = rand() * total_weight
    cumulative = 0.0
    for variant in NETWORK_FLOW_VARIANT_ORDER
        cumulative += NETWORK_FLOW_VARIANT_WEIGHTS[variant]
        if draw <= cumulative
            return variant
        end
    end
    return :generic
end

function determine_node_count(params, target_variables)
    n_nodes = params.min_nodes + 2
    for n in params.min_nodes:params.max_nodes
        possible_arcs = n * (n - 1)
        if round(Int, possible_arcs * params.target_density) >= target_variables * 0.9
            n_nodes = n
            break
        end
    end
    return n_nodes
end

function sample_range_value(range::Tuple{Float64,Float64}; bias::Symbol = :uniform)
    lo, hi = range
    span = hi - lo
    u = rand()
    if bias == :high
        u = sqrt(u)
    elseif bias == :low
        u = u^2
    elseif bias == :mid
        u = clamp(0.5 * u + 0.25, 0.0, 1.0)
    end
    return round(lo + u * span, digits=2)
end

function connect_layers!(arcs::Set{Tuple{Int,Int}}, arc_category::Dict{Tuple{Int,Int},Symbol}, from_nodes, to_nodes, probability, category)
    for i in from_nodes
        for j in to_nodes
            if i == j
                continue
            end
            if rand() < probability
                arc = (i, j)
                push!(arcs, arc)
                arc_category[arc] = category
            end
        end
    end
end

function ensure_downstream_incoming!(arcs::Set{Tuple{Int,Int}}, arc_category::Dict{Tuple{Int,Int},Symbol}, from_nodes, to_nodes, category)
    if isempty(from_nodes) || isempty(to_nodes)
        return
    end
    for target in to_nodes
        has_incoming = false
        for source in from_nodes
            if (source, target) in arcs
                has_incoming = true
                break
            end
        end
        if !has_incoming
            source = rand(from_nodes)
            arc = (source, target)
            push!(arcs, arc)
            arc_category[arc] = category
        end
    end
end

function ensure_target_arc_count!(arcs::Set{Tuple{Int,Int}}, n_nodes::Int, target::Int, arc_category::Dict{Tuple{Int,Int},Symbol}; supplemental_category::Symbol = :supplemental)
    if length(arcs) >= target
        return
    end
    candidates = [(i, j) for i in 1:n_nodes for j in 1:n_nodes if i != j && (i, j) ∉ arcs]
    shuffle!(candidates)
    for arc in candidates
        push!(arcs, arc)
        arc_category[arc] = supplemental_category
        if length(arcs) >= target
            break
        end
    end
end

function target_flow_from_source(flow_objective::Symbol, arcs, capacities::Dict{Tuple{Int,Int},Float64}, source_node::Int; ratio::Float64 = 0.7)
    if flow_objective != :min_cost
        return nothing
    end
    source_arcs = [arc for arc in arcs if arc[1] == source_node]
    if isempty(source_arcs)
        return nothing
    end
    total_capacity = sum(capacities[arc] for arc in source_arcs)
    return round(total_capacity * ratio, digits=2)
end

function partition_nodes(nodes::Vector{Int}, n_parts::Int)
    n = length(nodes)
    n_parts = max(1, n_parts)
    partitions = Vector{Vector{Int}}(undef, n_parts)
    start_idx = 1
    for part in 1:n_parts
        remaining_parts = n_parts - part + 1
        remaining_nodes = max(0, n - start_idx + 1)
        size = remaining_parts == 0 ? remaining_nodes : (remaining_nodes == 0 ? 0 : max(1, round(Int, remaining_nodes / remaining_parts)))
        if part == n_parts
            size = remaining_nodes
        end
        if size > 0
            partitions[part] = nodes[start_idx:start_idx + size - 1]
        else
            partitions[part] = Int[]
        end
        start_idx += size
    end
    return partitions
end

function choose_flow_objective(variant::Symbol, target_variables::Int)
    base_prob = Dict(
        :logistics => 0.75,
        :urban_water => 0.55,
        :power_transmission => 0.35,
        :generic => 0.5,
    )[variant]
    adjustment = target_variables <= 150 ? 0.1 : 0.0
    probability = clamp(base_prob + adjustment, 0.05, 0.95)
    return rand() < probability ? :min_cost : :max_flow
end

function generate_logistics_flow_problem(target_variables::Int, feasibility_status::FeasibilityStatus, params)
    n_nodes = determine_node_count(params, target_variables)
    source_node = 1
    sink_node = n_nodes
    interior_nodes = collect(2:(n_nodes - 1))
    if length(interior_nodes) < 3
        return generate_generic_network_problem(target_variables, feasibility_status, params)
    end

    layers = partition_nodes(interior_nodes, 3)
    factories = !isempty(layers[1]) ? layers[1] : Int[]
    warehouses = length(layers) >= 2 ? layers[2] : Int[]
    retailers = length(layers) >= 3 ? layers[3] : Int[]
    if isempty(warehouses)
        warehouses = factories
    end
    if isempty(retailers)
        retailers = !isempty(warehouses) ? warehouses : factories
    end

    arcs = Set{Tuple{Int,Int}}()
    arc_category = Dict{Tuple{Int,Int},Symbol}()

    for node in factories
        arc = (source_node, node)
        push!(arcs, arc)
        arc_category[arc] = :supply
    end

    connect_layers!(arcs, arc_category, factories, warehouses, 0.6, :factory_to_warehouse)
    connect_layers!(arcs, arc_category, warehouses, retailers, 0.5, :warehouse_to_retail)
    connect_layers!(arcs, arc_category, factories, retailers, 0.2, :direct_shipping)
    ensure_downstream_incoming!(arcs, arc_category, factories, warehouses, :factory_to_warehouse)
    ensure_downstream_incoming!(arcs, arc_category, warehouses, retailers, :warehouse_to_retail)

    for node in retailers
        arc = (node, sink_node)
        push!(arcs, arc)
        arc_category[arc] = :retail_to_sink
    end

    connect_layers!(arcs, arc_category, warehouses, warehouses, 0.15, :cross_docking)
    connect_layers!(arcs, arc_category, retailers, retailers, 0.1, :last_mile_rebalance)

    ensure_target_arc_count!(arcs, n_nodes, target_variables, arc_category)
    arc_list = collect(arcs)

    capacities = Dict{Tuple{Int,Int},Float64}()
    costs = Dict{Tuple{Int,Int},Float64}()
    for arc in arc_list
        category = get(arc_category, arc, :supplemental)
        cap_bias = category in (:supply, :factory_to_warehouse) ? :high : category == :warehouse_to_retail ? :mid : :uniform
        cost_bias = category in (:supply, :retail_to_sink) ? :low : category in (:cross_docking, :last_mile_rebalance) ? :high : :mid
        capacities[arc] = sample_range_value(params.capacity_range; bias=cap_bias)
        costs[arc] = sample_range_value(params.cost_range; bias=cost_bias)
    end

    flow_objective = choose_flow_objective(:logistics, target_variables)
    target_flow = target_flow_from_source(flow_objective, arc_list, capacities, source_node; ratio=0.8)

    return NetworkFlowProblem(n_nodes, source_node, sink_node, arc_list, capacities, costs, flow_objective, target_flow)
end

function generate_urban_water_flow_problem(target_variables::Int, feasibility_status::FeasibilityStatus, params)
    n_nodes = determine_node_count(params, target_variables)
    source_node = 1
    sink_node = n_nodes
    interior_nodes = collect(2:(n_nodes - 1))
    n_layers = clamp(rand(3:5), 2, max(2, length(interior_nodes)))
    raw_layers = partition_nodes(interior_nodes, n_layers)
    layers = [layer for layer in raw_layers if !isempty(layer)]
    if isempty(layers)
        return generate_generic_network_problem(target_variables, feasibility_status, params)
    end

    arcs = Set{Tuple{Int,Int}}()
    arc_category = Dict{Tuple{Int,Int},Symbol}()

    first_layer = layers[1]
    last_layer = layers[end]

    connect_layers!(arcs, arc_category, [source_node], first_layer, 0.9, :reservoir_to_pump)
    ensure_downstream_incoming!(arcs, arc_category, [source_node], first_layer, :reservoir_to_pump)
    for idx in 1:(length(layers) - 1)
        layer = layers[idx]
        next_layer = layers[idx + 1]
        connect_layers!(arcs, arc_category, layer, next_layer, 0.55, :mainline)
        ensure_downstream_incoming!(arcs, arc_category, layer, next_layer, :mainline)
    end
    if length(layers) > 2
        for idx in 1:length(layers) - 2
            connect_layers!(arcs, arc_category, layers[idx], layers[idx + 2], 0.25, :bypass)
        end
    end
    connect_layers!(arcs, arc_category, last_layer, [sink_node], 0.85, :distribution_to_sink)
    for node in last_layer
        arc = (node, sink_node)
        arc_category[arc] = :distribution_to_sink
        push!(arcs, arc)
    end

    for layer in layers
        connect_layers!(arcs, arc_category, layer, layer, 0.2, :loop)
    end

    ensure_target_arc_count!(arcs, n_nodes, target_variables, arc_category)
    arc_list = collect(arcs)

    capacities = Dict{Tuple{Int,Int},Float64}()
    costs = Dict{Tuple{Int,Int},Float64}()
    for arc in arc_list
        category = get(arc_category, arc, :supplemental)
        cap_bias = category in (:reservoir_to_pump, :mainline) ? :high : category == :distribution_to_sink ? :mid : :low
        cost_bias = category == :reservoir_to_pump ? :low : category in (:mainline, :bypass) ? :mid : :high
        capacities[arc] = sample_range_value(params.capacity_range; bias=cap_bias)
        costs[arc] = sample_range_value(params.cost_range; bias=cost_bias)
    end

    flow_objective = choose_flow_objective(:urban_water, target_variables)
    target_flow = target_flow_from_source(flow_objective, arc_list, capacities, source_node; ratio=0.65)

    return NetworkFlowProblem(n_nodes, source_node, sink_node, arc_list, capacities, costs, flow_objective, target_flow)
end

function generate_power_transmission_flow_problem(target_variables::Int, feasibility_status::FeasibilityStatus, params)
    n_nodes = determine_node_count(params, target_variables)
    source_node = 1
    sink_node = n_nodes
    interior_nodes = collect(2:(n_nodes - 1))
    if length(interior_nodes) < 4
        return generate_generic_network_problem(target_variables, feasibility_status, params)
    end

    raw_layers = partition_nodes(interior_nodes, 3)
    generation_nodes = !isempty(raw_layers[1]) ? raw_layers[1] : Int[]
    transmission_nodes = length(raw_layers) >= 2 ? raw_layers[2] : Int[]
    load_nodes = length(raw_layers) >= 3 ? raw_layers[3] : Int[]
    if isempty(transmission_nodes)
        transmission_nodes = generation_nodes
    end
    if isempty(load_nodes)
        load_nodes = transmission_nodes
    end

    arcs = Set{Tuple{Int,Int}}()
    arc_category = Dict{Tuple{Int,Int},Symbol}()

    connect_layers!(arcs, arc_category, [source_node], generation_nodes, 0.95, :generation)
    connect_layers!(arcs, arc_category, generation_nodes, transmission_nodes, 0.7, :backbone)
    connect_layers!(arcs, arc_category, transmission_nodes, transmission_nodes, 0.4, :mesh)
    connect_layers!(arcs, arc_category, transmission_nodes, load_nodes, 0.6, :distribution)
    connect_layers!(arcs, arc_category, generation_nodes, load_nodes, 0.3, :direct_supply)
    connect_layers!(arcs, arc_category, load_nodes, [sink_node], 0.9, :load_to_sink)
    ensure_downstream_incoming!(arcs, arc_category, [source_node], generation_nodes, :generation)
    ensure_downstream_incoming!(arcs, arc_category, transmission_nodes, load_nodes, :distribution)
    for node in load_nodes
        arc = (node, sink_node)
        arc_category[arc] = :load_to_sink
        push!(arcs, arc)
    end

    ensure_target_arc_count!(arcs, n_nodes, target_variables, arc_category)
    arc_list = collect(arcs)

    capacities = Dict{Tuple{Int,Int},Float64}()
    costs = Dict{Tuple{Int,Int},Float64}()
    for arc in arc_list
        category = get(arc_category, arc, :supplemental)
        cap_bias = category in (:generation, :backbone, :mesh) ? :high : :mid
        cost_bias = category in (:generation, :backbone, :mesh) ? :low : :mid
        capacities[arc] = sample_range_value(params.capacity_range; bias=cap_bias)
        costs[arc] = sample_range_value(params.cost_range; bias=cost_bias)
    end

    flow_objective = rand() < 0.6 ? :max_flow : :min_cost
    target_flow = target_flow_from_source(flow_objective, arc_list, capacities, source_node; ratio=0.7)

    return NetworkFlowProblem(n_nodes, source_node, sink_node, arc_list, capacities, costs, flow_objective, target_flow)
end

function assign_generic_capacities_costs(arcs, capacity_range, cost_range)
    capacities = Dict{Tuple{Int,Int},Float64}()
    costs = Dict{Tuple{Int,Int},Float64}()
    for arc in arcs
        capacities[arc] = sample_range_value(capacity_range)
        costs[arc] = sample_range_value(cost_range)
    end
    return capacities, costs
end

function generate_generic_network_problem(target_variables::Int, feasibility_status::FeasibilityStatus, params)
    n_nodes = determine_node_count(params, target_variables)
    source_node = 1
    sink_node = n_nodes
    arcs = generate_connected_network(n_nodes, target_variables, source_node, sink_node)
    capacities, costs = assign_generic_capacities_costs(arcs, params.capacity_range, params.cost_range)
    flow_objective = choose_flow_objective(:generic, target_variables)
    target_flow = target_flow_from_source(flow_objective, arcs, capacities, source_node; ratio=0.8)
    return NetworkFlowProblem(n_nodes, source_node, sink_node, arcs, capacities, costs, flow_objective, target_flow)
end

const NETWORK_FLOW_VARIANT_GENERATORS = Dict(
    :logistics => generate_logistics_flow_problem,
    :urban_water => generate_urban_water_flow_problem,
    :power_transmission => generate_power_transmission_flow_problem,
    :generic => generate_generic_network_problem,
)

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
