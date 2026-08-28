using JuMP
using Random

"""
    ResilientNetworkDesignProblem <: ProblemGenerator

Choose which undirected edges to build and harden, then route a required flow
under each generated failure scenario. A failed edge has no capacity unless it
was hardened; surviving edges require only the shared first-stage build choice.
"""
struct ResilientNetworkDesignProblem <: ProblemGenerator
    n_nodes::Int
    n_edges::Int
    n_scenarios::Int
    edges::Vector{Tuple{Int,Int}}
    sources::Vector{Int}
    sinks::Vector{Int}
    demands::Vector{Float64}
    capacities::Vector{Float64}
    build_cost::Vector{Float64}
    hardening_cost::Vector{Float64}
    routing_cost::Vector{Float64}
    failed::Matrix{Bool}
    design_budget::Float64
end

function _resilient_topology(rng::AbstractRNG, n_nodes::Int, n_edges::Int)
    edges = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()
    order = randperm(rng, n_nodes)
    for idx in 2:n_nodes
        parent = order[rand(rng, 1:(idx - 1))]
        child = order[idx]
        edge = parent < child ? (parent, child) : (child, parent)
        push!(edges, edge)
        push!(seen, edge)
    end
    candidates = [(i, j) for i in 1:n_nodes for j in (i + 1):n_nodes]
    shuffle!(rng, candidates)
    for edge in candidates
        length(edges) >= n_edges && break
        edge in seen && continue
        push!(seen, edge)
        push!(edges, edge)
    end
    return edges
end

function ResilientNetworkDesignProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)
    n_scenarios = clamp(round(Int, sqrt(target) / 3), 2, 8)
    n_edges = max(4, round(Int, target / (2 * (n_scenarios + 1))))
    n_nodes = max(4, min(n_edges + 1, round(Int, n_edges / 1.8)))
    while n_nodes * (n_nodes - 1) ÷ 2 < n_edges
        n_nodes += 1
    end
    n_nodes = min(n_nodes, n_edges + 1)

    edges = _resilient_topology(rng, n_nodes, n_edges)
    n_edges = length(edges)
    sources = Vector{Int}(undef, n_scenarios)
    sinks = Vector{Int}(undef, n_scenarios)
    for s in 1:n_scenarios
        sources[s] = rand(rng, 1:n_nodes)
        sink = rand(rng, 1:(n_nodes - 1))
        sinks[s] = sink >= sources[s] ? sink + 1 : sink
    end
    demands = Float64.(rand(rng, 5:20, n_scenarios))
    capacities = Float64.(rand(rng, 12:40, n_edges))
    build_cost = [round(20.0 + 80.0 * rand(rng), digits=2) for _ in 1:n_edges]
    hardening_cost = [round(8.0 + 45.0 * rand(rng), digits=2) for _ in 1:n_edges]
    routing_cost = [round(0.5 + 6.0 * rand(rng), digits=3) for _ in 1:n_edges]
    failed = Matrix{Bool}(undef, n_edges, n_scenarios)
    for e in 1:n_edges, s in 1:n_scenarios
        failed[e, s] = rand(rng) < 0.25
    end
    for s in 1:n_scenarios
        failed[rand(rng, 1:n_edges), s] = true
    end

    tree_edges = 1:(n_nodes - 1)
    natural_budget = 0.55 * (sum(build_cost) + sum(hardening_cost))
    design_budget = natural_budget
    if feasibility_status == feasible
        # The first n-1 edges form a spanning tree. Building and hardening that
        # tree restores it in every scenario; sizing each tree edge for the
        # largest demand gives an explicit scenario-flow witness.
        max_demand = maximum(demands)
        for e in tree_edges
            capacities[e] = max(capacities[e], 1.15 * max_demand + 1.0)
        end
        planted_cost = sum(build_cost[e] + hardening_cost[e] for e in tree_edges)
        design_budget = 1.05 * planted_cost + 1.0
    elseif feasibility_status == infeasible
        # Scenario 1 needs more net inflow at its sink than every incident edge
        # could carry even if all were built and hardened. This certificate is
        # independent of the binary domains and of the design budget.
        sink = sinks[1]
        incident = [e for (e, (i, j)) in enumerate(edges) if i == sink || j == sink]
        demands[1] = sum(capacities[e] for e in incident) + rand(rng, 2.0:1.0:8.0)
        design_budget = sum(build_cost) + sum(hardening_cost) + 1.0
    end

    return ResilientNetworkDesignProblem(
        n_nodes, n_edges, n_scenarios, edges, sources, sinks, demands,
        capacities, build_cost, hardening_cost, routing_cost, failed, design_budget,
    )
end

function build_model(prob::ResilientNetworkDesignProblem)
    model = Model()
    E = prob.n_edges
    S = prob.n_scenarios

    @variable(model, build[1:E], Bin)
    @variable(model, harden[1:E], Bin)
    @variable(model, forward[1:E, 1:S] >= 0)
    @variable(model, reverse[1:E, 1:S] >= 0)

    @objective(model, Min,
        sum(prob.build_cost[e] * build[e] + prob.hardening_cost[e] * harden[e]
            for e in 1:E) +
        sum(prob.routing_cost[e] * (forward[e, s] + reverse[e, s]) / S
            for e in 1:E, s in 1:S)
    )
    @constraint(model,
        sum(prob.build_cost[e] * build[e] + prob.hardening_cost[e] * harden[e]
            for e in 1:E) <= prob.design_budget
    )
    for e in 1:E
        @constraint(model, harden[e] <= build[e])
        for s in 1:S
            available_capacity = prob.failed[e, s] ? harden[e] : build[e]
            @constraint(model,
                forward[e, s] + reverse[e, s] <= prob.capacities[e] * available_capacity
            )
        end
    end

    incident = [Int[] for _ in 1:prob.n_nodes]
    for (e, (i, j)) in enumerate(prob.edges)
        push!(incident[i], e)
        push!(incident[j], e)
    end
    for s in 1:S, node in 1:prob.n_nodes
        balance = AffExpr(0.0)
        for e in incident[node]
            i, j = prob.edges[e]
            if node == i
                add_to_expression!(balance, 1.0, forward[e, s])
                add_to_expression!(balance, -1.0, reverse[e, s])
            else
                add_to_expression!(balance, 1.0, reverse[e, s])
                add_to_expression!(balance, -1.0, forward[e, s])
            end
        end
        rhs = node == prob.sources[s] ? prob.demands[s] :
              node == prob.sinks[s] ? -prob.demands[s] : 0.0
        @constraint(model, balance == rhs)
    end

    return model
end

register_variant(
    :resilient_network_design,
    :standard,
    ResilientNetworkDesignProblem,
    "Two-stage network build and hardening with capacity-feasible flows under explicit edge-failure scenarios",
)
