using JuMP
using Random
using Distributions

"""
Ground truth for a controlled layered inverse-shortest-path instance.
"""
struct LayeredPathWitness
    potentials::Vector{Float64}
    cost::Vector{Float64}
end

"""
A chord is cheaper than the observed segment under every admissible cost.
"""
struct LayeredShortcutCertificate
    shortcut_arc::Int
    bypassed_arcs::Vector{Int}
    shortcut_upper::Float64
    bypassed_lower_sum::Float64
end

"""
Single-observation inverse shortest path on a layered DAG. This controlled
family complements the spatial multi-route variant: topology, route length,
reduced-cost margins, and a native shortcut contradiction are transparent.
"""
struct LayeredInverseShortestPathProblem <: ProblemGenerator
    num_nodes::Int
    num_arcs::Int
    num_layers::Int
    width::Int
    tail::Vector{Int}
    head::Vector{Int}
    layer::Vector{Int}
    source::Int
    sink::Int
    path_arcs::Vector{Int}
    prior_cost::Vector{Float64}
    cost_lower::Vector{Float64}
    cost_upper::Vector{Float64}
    deviation_weights::Vector{Float64}
    feasible_witness::Union{Nothing, LayeredPathWitness}
    infeasibility_certificate::Union{Nothing, LayeredShortcutCertificate}
    feasibility_status::FeasibilityStatus
end

function _layered_arc_capacity(layers::Int, width::Int)
    adjacent = 2 * width + (layers - 3) * width^2
    skips = layers >= 4 ? 2 * width + max(0, layers - 4) * width^2 : 0
    return adjacent + skips + 1 # source-to-sink chord
end

function _layered_inverse_dims(target::Int)
    layers0 = clamp(round(Int, sqrt(target / 10.0)), 4, 80)
    width0 = clamp(round(Int, target / (10.0 * layers0)), 2, 60)
    for layers in _around(layers0, 4), width in _around(width0, 2)
        nodes = 2 + width * (layers - 2)
        # Route, chord, and one outgoing arc per off-route intermediate node.
        mandatory = (layers - 1) + 1 + (width - 1) * (layers - 2)
        if nodes + 3 * mandatory <= target <= nodes + 3 * _layered_arc_capacity(layers, width) &&
            (target - nodes) % 3 == 0
            return layers, width
        end
    end
    return 4, 2
end

function _layered_prior_is_optimal(
    num_nodes::Int,
    source::Int,
    sink::Int,
    tail::Vector{Int},
    head::Vector{Int},
    costs::Vector{Float64},
    path_arcs::Vector{Int},
)
    distance = fill(Inf, num_nodes)
    distance[source] = 0.0
    outgoing = [Int[] for _ in 1:num_nodes]
    for e in eachindex(tail)
        push!(outgoing[tail[e]], e)
    end
    for u in 1:num_nodes
        isfinite(distance[u]) || continue
        for e in outgoing[u]
            distance[head[e]] = min(distance[head[e]], distance[u] + costs[e])
        end
    end
    path_cost = sum(costs[e] for e in path_arcs)
    return isapprox(path_cost, distance[sink]; atol=1.0e-10, rtol=1.0e-10)
end

function LayeredInverseShortestPathProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    _check_inverse_target(target_variables)
    rng = MersenneTwister(seed)
    layers, width = _layered_inverse_dims(target_variables)
    num_nodes = 2 + width * (layers - 2)
    source, sink = 1, num_nodes
    layer_of = vcat([1], repeat(2:(layers - 1); inner=width), [layers])
    nodes_in_layer(i) =
        if i == 1
            [source]
        elseif i == layers
            [sink]
        else
            collect((2 + (i - 2) * width):(1 + (i - 1) * width))
        end

    route = Vector{Int}(undef, layers)
    route[1], route[end] = source, sink
    for layer in 2:(layers - 1)
        route[layer] = rand(rng, nodes_in_layer(layer))
    end

    tail = Int[]
    head = Int[]
    arc_index = Dict{Tuple{Int, Int}, Int}()
    function add_arc!(u::Int, v::Int)
        haskey(arc_index, (u, v)) && return arc_index[(u, v)]
        push!(tail, u)
        push!(head, v)
        arc_index[(u, v)] = length(tail)
        return length(tail)
    end

    path_arcs = [add_arc!(route[layer], route[layer + 1]) for layer in 1:(layers - 1)]
    chord_arc = add_arc!(source, sink)

    # Give every intermediate node a route to the sink before adding fillers.
    for layer in 2:(layers - 1), u in nodes_in_layer(layer)
        any(tail[e] == u for e in eachindex(tail)) ||
            add_arc!(u, rand(rng, nodes_in_layer(layer + 1)))
    end

    candidates = Tuple{Int, Int}[]
    for layer in 1:(layers - 1), u in nodes_in_layer(layer), v in nodes_in_layer(layer + 1)
        haskey(arc_index, (u, v)) || push!(candidates, (u, v))
    end
    for layer in 1:(layers - 2), u in nodes_in_layer(layer), v in nodes_in_layer(layer + 2)
        haskey(arc_index, (u, v)) || push!(candidates, (u, v))
    end
    shuffle!(rng, candidates)

    arc_budget = if target_variables > num_nodes
        max(length(tail), (target_variables - num_nodes) ÷ 3)
    else
        length(tail)
    end
    arc_budget = min(arc_budget, _layered_arc_capacity(layers, width))

    # Reachability from the source, then arbitrary sparse fillers.
    has_incoming = Set(head)
    for (u, v) in candidates
        length(tail) >= arc_budget && break
        v in has_incoming && continue
        add_arc!(u, v)
        push!(has_incoming, v)
    end
    for (u, v) in candidates
        length(tail) >= arc_budget && break
        add_arc!(u, v)
    end
    num_arcs = length(tail)

    potentials = zeros(num_nodes)
    increments = round.(rand(rng, Uniform(4.0, 9.0), layers - 1); digits=1)
    for layer in 2:layers
        potentials[route[layer]] = potentials[route[layer - 1]] + increments[layer - 1]
    end
    for layer in 2:(layers - 1), node in nodes_in_layer(layer)
        node == route[layer] && continue
        potentials[node] = potentials[route[layer]] - rand(rng, Uniform(0.5, 3.5))
    end

    path_set = Set(path_arcs)
    true_cost = Vector{Float64}(undef, num_arcs)
    for e in 1:num_arcs
        reduced = potentials[head[e]] - potentials[tail[e]]
        true_cost[e] = e in path_set ? reduced : max(reduced, 0.0) + rand(rng, Uniform(2.0, 12.0))
    end

    # Condition the prior on requiring a genuine inverse adjustment.
    prior_cost = copy(true_cost)
    for _ in 1:100
        prior_cost = true_cost .* rand(rng, LogNormal(0.0, 0.24), num_arcs)
        _layered_prior_is_optimal(num_nodes, source, sink, tail, head, prior_cost, path_arcs) ||
            break
    end
    if _layered_prior_is_optimal(num_nodes, source, sink, tail, head, prior_cost, path_arcs)
        prior_cost[chord_arc] = 0.92 * sum(prior_cost[e] for e in path_arcs)
    end

    radius = feasibility_status == feasible ? 0.85 : 0.65
    cost_lower = prior_cost .* exp(-radius)
    cost_upper = prior_cost .* exp(radius)
    if feasibility_status == feasible
        cost_lower = min.(cost_lower, 0.98 .* true_cost)
        cost_upper = max.(cost_upper, 1.02 .* true_cost)
    end

    witness = if feasibility_status == feasible
        LayeredPathWitness(copy(potentials), copy(true_cost))
    else
        nothing
    end
    certificate = nothing
    if feasibility_status == infeasible
        bypassed_floor = sum(cost_lower[e] for e in path_arcs)
        cost_upper[chord_arc] = rand(rng, Uniform(0.55, 0.82)) * bypassed_floor
        cost_lower[chord_arc] = 0.35 * cost_upper[chord_arc]
        prior_cost[chord_arc] = 0.70 * cost_upper[chord_arc]
        certificate = LayeredShortcutCertificate(
            chord_arc, copy(path_arcs), cost_upper[chord_arc], bypassed_floor
        )
    elseif feasibility_status == unknown && rand(rng) < 0.70
        bypassed_floor = sum(cost_lower[e] for e in path_arcs)
        cost_upper[chord_arc] = rand(rng, Uniform(0.75, 1.30)) * bypassed_floor
        cost_lower[chord_arc] = 0.35 * cost_upper[chord_arc]
        prior_cost[chord_arc] = 0.70 * cost_upper[chord_arc]
    end

    deviation_weights = 1.0 ./ max.(prior_cost, 1.0e-8)
    deviation_weights .*= num_arcs / sum(deviation_weights)
    return LayeredInverseShortestPathProblem(
        num_nodes,
        num_arcs,
        layers,
        width,
        tail,
        head,
        layer_of,
        source,
        sink,
        path_arcs,
        prior_cost,
        cost_lower,
        cost_upper,
        deviation_weights,
        witness,
        certificate,
        feasibility_status,
    )
end

function build_model(prob::LayeredInverseShortestPathProblem)
    model = Model()
    V, E = prob.num_nodes, prob.num_arcs
    @variable(model, potential[1:V])
    @variable(model, prob.cost_lower[e] <= arc_cost[e = 1:E] <= prob.cost_upper[e])
    @variable(model, deviation_positive[1:E] >= 0)
    @variable(model, deviation_negative[1:E] >= 0)
    @constraint(model, potential[prob.source] == 0.0)
    path_set = Set(prob.path_arcs)
    for e in 1:E
        @constraint(model, potential[prob.head[e]] - potential[prob.tail[e]] <= arc_cost[e])
        if e in path_set
            @constraint(model, potential[prob.head[e]] - potential[prob.tail[e]] == arc_cost[e])
        end
        @constraint(
            model, arc_cost[e] - prob.prior_cost[e] == deviation_positive[e] - deviation_negative[e]
        )
    end
    @objective(
        model,
        Min,
        sum(
            prob.deviation_weights[e] * (deviation_positive[e] + deviation_negative[e]) for e in 1:E
        )
    )
    return model
end

function _layered_shortcut_certificate_is_valid(prob::LayeredInverseShortestPathProblem)
    certificate = prob.infeasibility_certificate
    certificate isa LayeredShortcutCertificate || return false
    return certificate.shortcut_arc ∉ prob.path_arcs &&
           isapprox(certificate.shortcut_upper, prob.cost_upper[certificate.shortcut_arc]) &&
           isapprox(
               certificate.bypassed_lower_sum,
               sum(prob.cost_lower[e] for e in certificate.bypassed_arcs),
           ) &&
           certificate.shortcut_upper < certificate.bypassed_lower_sum
end

register_variant(
    :inverse_optimization,
    :shortest_path_layered,
    LayeredInverseShortestPathProblem,
    "Controlled single-observation inverse shortest path on a layered DAG with informative priors and shortcut certificates",
)
