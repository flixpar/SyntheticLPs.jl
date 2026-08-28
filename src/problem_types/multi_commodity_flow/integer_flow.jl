using JuMP
using Random

"""
    IntegerMultiCommodityFlowProblem <: ProblemGenerator

An unsplittable-unit-style multicommodity min-cost flow model whose arc flows
are nonnegative general-integer variables. Commodity demands and arc capacities
are integral, and shared capacities couple all commodities on each arc.
"""
struct IntegerMultiCommodityFlowProblem <: ProblemGenerator
    n_nodes::Int
    n_arcs::Int
    n_commodities::Int
    arcs::Vector{Tuple{Int,Int}}
    sources::Vector{Int}
    sinks::Vector{Int}
    demands::Vector{Int}
    capacities::Vector{Int}
    costs::Matrix{Float64}
end

function _integer_mcf_path(arcs::Vector{Tuple{Int,Int}}, n_nodes::Int,
                           source::Int, sink::Int)
    outgoing = [Int[] for _ in 1:n_nodes]
    for (a, (i, _)) in enumerate(arcs)
        push!(outgoing[i], a)
    end
    parent_node = zeros(Int, n_nodes)
    parent_arc = zeros(Int, n_nodes)
    visited = falses(n_nodes)
    queue = Int[source]
    visited[source] = true
    head = 1
    while head <= length(queue) && !visited[sink]
        node = queue[head]
        head += 1
        for a in outgoing[node]
            next = arcs[a][2]
            visited[next] && continue
            visited[next] = true
            parent_node[next] = node
            parent_arc[next] = a
            push!(queue, next)
        end
    end
    visited[sink] || error("integer-flow MCF topology is not strongly connected")
    path = Int[]
    node = sink
    while node != source
        push!(path, parent_arc[node])
        node = parent_node[node]
    end
    reverse!(path)
    return path
end

"""
    IntegerMultiCommodityFlowProblem(target_variables, feasibility_status, seed)

The formulation creates `n_arcs * n_commodities` general-integer variables.
Feasible requests plant integral path flows and widen shared capacities to hold
their aggregate load. Infeasible requests make one demand exceed the total
capacity entering its sink, which is also a valid infeasibility certificate for
the default continuous relaxation. Unknown requests keep natural independently
sampled capacities.
"""
function IntegerMultiCommodityFlowProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = MersenneTwister(seed)

    target = max(target_variables, 1)
    n_commodities = clamp(round(Int, sqrt(target) / 2), 2, 28)
    n_arcs = max(4, round(Int, target / n_commodities))
    n_nodes = max(4, min(n_arcs, round(Int, n_arcs / 2)))
    while n_nodes * (n_nodes - 1) < n_arcs
        n_nodes += 1
    end

    arcs = _discrete_mcf_topology(rng, n_nodes, n_arcs)
    n_arcs = length(arcs)
    sources = Vector{Int}(undef, n_commodities)
    sinks = Vector{Int}(undef, n_commodities)
    for k in 1:n_commodities
        sources[k] = rand(rng, 1:n_nodes)
        sink = rand(rng, 1:(n_nodes - 1))
        sinks[k] = sink >= sources[k] ? sink + 1 : sink
    end

    demands = rand(rng, 1:9, n_commodities)
    capacities = rand(rng, 7:35, n_arcs)
    costs = Matrix{Float64}(undef, n_commodities, n_arcs)
    for k in 1:n_commodities, a in 1:n_arcs
        # Commodity-specific perturbations avoid duplicate objective rows while
        # retaining a common per-arc distance/cost scale.
        costs[k, a] = round(0.5 + 10.0 * rand(rng), digits=3)
    end

    if feasibility_status == feasible
        planted_load = zeros(Int, n_arcs)
        for k in 1:n_commodities
            for a in _integer_mcf_path(arcs, n_nodes, sources[k], sinks[k])
                planted_load[a] += demands[k]
            end
        end
        for a in 1:n_arcs
            capacities[a] = max(capacities[a], planted_load[a] + rand(rng, 1:5))
        end
    elseif feasibility_status == infeasible
        sink = sinks[1]
        incoming = [a for (a, (_, j)) in enumerate(arcs) if j == sink]
        demands[1] = sum(capacities[a] for a in incoming) + rand(rng, 1:6)
    end

    return IntegerMultiCommodityFlowProblem(
        n_nodes, n_arcs, n_commodities, arcs,
        sources, sinks, demands, capacities, costs,
    )
end

"""Build the deterministic general-integer multicommodity-flow formulation."""
function build_model(prob::IntegerMultiCommodityFlowProblem)
    model = Model()
    K = prob.n_commodities
    A = prob.n_arcs

    @variable(model, flow[1:K, 1:A] >= 0, Int)
    @objective(model, Min,
        sum(prob.costs[k, a] * flow[k, a] for k in 1:K, a in 1:A)
    )

    for a in 1:A
        @constraint(model, sum(flow[k, a] for k in 1:K) <= prob.capacities[a])
    end

    outgoing = [Int[] for _ in 1:prob.n_nodes]
    incoming = [Int[] for _ in 1:prob.n_nodes]
    for (a, (i, j)) in enumerate(prob.arcs)
        push!(outgoing[i], a)
        push!(incoming[j], a)
    end
    for k in 1:K, node in 1:prob.n_nodes
        rhs = node == prob.sources[k] ? prob.demands[k] :
              node == prob.sinks[k] ? -prob.demands[k] : 0
        @constraint(model,
            sum(flow[k, a] for a in outgoing[node]) -
            sum(flow[k, a] for a in incoming[node]) == rhs
        )
    end

    return model
end

register_variant(
    :multi_commodity_flow,
    :integer_flow,
    IntegerMultiCommodityFlowProblem,
    "Shared-capacity multicommodity min-cost flow with nonnegative general-integer arc flows",
)
