using JuMP
using Random

"""
    BinaryCapacityMultiCommodityFlowProblem <: ProblemGenerator

Discrete-capacity multicommodity network design. Each directed arc offers two
mutually exclusive capacity modules, represented by binary installation
variables, while commodity routing remains continuous. The objective combines
fixed module costs and per-unit routing costs.
"""
struct BinaryCapacityMultiCommodityFlowProblem <: ProblemGenerator
    n_nodes::Int
    n_arcs::Int
    n_commodities::Int
    n_modules::Int
    arcs::Vector{Tuple{Int,Int}}
    sources::Vector{Int}
    sinks::Vector{Int}
    demands::Vector{Int}
    routing_cost::Vector{Float64}
    module_capacity::Matrix{Int}
    module_cost::Matrix{Float64}
end

function _binary_mcf_path(arcs::Vector{Tuple{Int,Int}}, n_nodes::Int,
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
    visited[sink] || error("binary-capacity MCF topology is not strongly connected")
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
    BinaryCapacityMultiCommodityFlowProblem(target_variables, feasibility_status, seed)

The variable count is `n_arcs * (n_commodities + 2)`: one flow per
commodity-arc pair and two binary module choices per arc. Feasible instances
plant a path routing and make one module on every used arc large enough for its
aggregate load. Infeasible instances make commodity 1's demand exceed the sum
of the largest capacity modules entering its sink; this remains a certificate
after the binary variables are relaxed.
"""
function BinaryCapacityMultiCommodityFlowProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = MersenneTwister(seed)

    target = max(target_variables, 1)
    n_modules = 2
    n_commodities = clamp(round(Int, sqrt(target) / 2), 2, 24)
    n_arcs = max(4, round(Int, target / (n_commodities + n_modules)))
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
    demands = rand(rng, 2:12, n_commodities)
    routing_cost = [round(0.5 + 8.0 * rand(rng), digits=3) for _ in 1:n_arcs]

    module_capacity = Matrix{Int}(undef, n_arcs, n_modules)
    module_cost = Matrix{Float64}(undef, n_arcs, n_modules)
    for a in 1:n_arcs
        small = rand(rng, 5:18)
        large = small + rand(rng, 12:45)
        module_capacity[a, 1] = small
        module_capacity[a, 2] = large
        base_cost = 8.0 + 30.0 * rand(rng)
        module_cost[a, 1] = round(base_cost, digits=2)
        module_cost[a, 2] = round(base_cost * rand(rng, 1.35:0.05:1.90), digits=2)
    end

    if feasibility_status == feasible
        planted_load = zeros(Int, n_arcs)
        for k in 1:n_commodities
            for a in _binary_mcf_path(arcs, n_nodes, sources[k], sinks[k])
                planted_load[a] += demands[k]
            end
        end
        for a in 1:n_arcs
            planted_load[a] == 0 && continue
            required = ceil(Int, 1.15 * planted_load[a]) + 1
            module_capacity[a, 2] = max(module_capacity[a, 2], required)
            module_cost[a, 2] = max(module_cost[a, 2], 1.25 * module_cost[a, 1])
        end
    elseif feasibility_status == infeasible
        # Even fractional module selection with sum(y) <= 1 supplies at most the
        # largest module on each incoming arc.
        sink = sinks[1]
        incoming = [a for (a, (_, j)) in enumerate(arcs) if j == sink]
        maximum_sink_inflow = sum(maximum(module_capacity[a, :]) for a in incoming)
        demands[1] = maximum_sink_inflow + rand(rng, 2:8)
    end

    return BinaryCapacityMultiCommodityFlowProblem(
        n_nodes, n_arcs, n_commodities, n_modules, arcs,
        sources, sinks, demands, routing_cost, module_capacity, module_cost,
    )
end

"""Build the deterministic binary-capacity multicommodity-flow formulation."""
function build_model(prob::BinaryCapacityMultiCommodityFlowProblem)
    model = Model()
    K = prob.n_commodities
    A = prob.n_arcs
    M = prob.n_modules

    @variable(model, flow[1:K, 1:A] >= 0)
    @variable(model, install[1:A, 1:M], Bin)

    @objective(model, Min,
        sum(prob.routing_cost[a] * flow[k, a] for k in 1:K, a in 1:A) +
        sum(prob.module_cost[a, m] * install[a, m] for a in 1:A, m in 1:M)
    )

    for a in 1:A
        @constraint(model, sum(install[a, m] for m in 1:M) <= 1)
        @constraint(model,
            sum(flow[k, a] for k in 1:K) <=
            sum(prob.module_capacity[a, m] * install[a, m] for m in 1:M)
        )
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
    :binary_capacity,
    BinaryCapacityMultiCommodityFlowProblem,
    "Multicommodity network design with continuous routing and mutually exclusive binary capacity modules",
)
