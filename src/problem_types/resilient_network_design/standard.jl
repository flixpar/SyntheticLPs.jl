using JuMP
using Random
using Distributions

"""
Planted design of a requested-feasible `ResilientNetworkDesignProblem`: the
spanning tree (the first `n_nodes - 1` edges) is built and hardened, so it
survives every scenario, and every tree edge carries at least 1.15 × the
largest demand. `forward[e, s]` / `reverse[e, s]` route each scenario's demand
along the unique tree path; the budget was drawn above the tree's cost.
"""
struct ResilientNetworkWitness
    build::Vector{Float64}
    harden::Vector{Float64}
    forward::Matrix{Float64}
    reverse::Matrix{Float64}
end

"""
Regional cut certificate. `region` contains scenario `scenario`'s sink but not
its source; summing the region's flow-balance rows shows that the scenario's
`demand` must cross `cut_edges`, each of which carries at most
`capacity[e] * max(build[e], harden[e]) ≤ capacity[e]` (`build`, `harden` ≤ 1).
Their total `cut_capacity` is below `demand` by `margin`. The argument sums
`length(region)` (about `sqrt(n_nodes)`, at least 4 when the source allows)
balance rows with the cut's capacity rows, so presolve does
not detect it, and it holds for every value of the (relaxed) design variables.
"""
struct ResilientNetworkCutCertificate
    scenario::Int
    region::Vector{Int}
    cut_edges::Vector{Int}
    demand::Float64
    cut_capacity::Float64
    margin::Float64
end

"""
    ResilientNetworkDesignProblem <: ProblemGenerator

Two-stage resilient network design: choose which candidate links to build and
which to harden, then route each failure scenario's required flow.

# Overview

Nodes are sites on a 100 × 100 map; candidate links join each node to a nearby
earlier node (a spanning tree) plus nearest-neighbour shortcuts. Each scenario
is a regional hazard (a disaster center and radius): links near the center fail
with high probability, distant links rarely. A failed link carries flow only if
it was hardened; a surviving link only needs to be built. First stage:
`build[e]`, `harden[e]` (binary; relaxed by default) under a design budget,
with `harden ≤ build`. Second stage per scenario: directed flows on both link
orientations, link capacity `forward + reverse ≤ capacity × (harden if failed
else build)`, and flow conservation sending `demand[s]` from `sources[s]` to
`sinks[s]`. The objective is design cost plus average routing cost.

# Feasibility control

  - `feasible`: the spanning tree is built and hardened with capacities raised to
    1.15 × the largest demand + 1, and the budget is 1.05 × its cost + 1
    ([`ResilientNetworkWitness`](@ref)).
  - `infeasible`: a region (a breadth-first ball of about `sqrt(n_nodes)` nodes
    around scenario 1's sink, excluding its source) is weakly connected: the
    capacities of its boundary links are scaled so they total
    `demand / U(1.08, 1.20)` ([`ResilientNetworkCutCertificate`](@ref)). The
    budget is unlimited, so the cut — not the budget — is the obstruction, and
    it survives relaxation. The certificate needs a whole region's balance
    rows; HiGHS presolve still refutes some instances (about half at 1k, a
    quarter at 10k in the audit) through its propagation, the rest need simplex.
  - `unknown`: natural capacities and a budget `U(0.45, 1.1) ×` the planted
    tree's cost: whether a design within budget routes every scenario is left
    to the instance.

# Size

Variables `2 n_edges (1 + n_scenarios)`; rows `1 + n_edges (1 + n_scenarios) +
n_nodes n_scenarios`.
"""
struct ResilientNetworkDesignProblem <: ProblemGenerator
    n_nodes::Int
    n_edges::Int
    n_scenarios::Int
    positions::Vector{Tuple{Float64, Float64}}
    edges::Vector{Tuple{Int, Int}}
    sources::Vector{Int}
    sinks::Vector{Int}
    demands::Vector{Float64}
    capacities::Vector{Float64}
    build_cost::Vector{Float64}
    hardening_cost::Vector{Float64}
    routing_cost::Vector{Float64}
    hazard_center::Vector{Tuple{Float64, Float64}}
    hazard_radius::Vector{Float64}
    failed::Matrix{Bool}
    design_budget::Float64
    feasible_witness::Union{Nothing, ResilientNetworkWitness}
    infeasibility_certificate::Union{Nothing, ResilientNetworkCutCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _resilient_dimensions(target) -> (n_scenarios, n_edges, n_nodes)

`2 E (S + 1)` variables: scenarios grow like `sqrt(target) / 3` (2..8), edges
absorb the rest, and nodes are about `E / 1.8` (average degree ≈ 3.6).
"""
function _resilient_dimensions(target::Int)
    t = max(target, 1)
    S = clamp(round(Int, sqrt(t) / 3), 2, 8)
    E = max(4, round(Int, t / (2 * (S + 1))))
    N = clamp(round(Int, E / 1.8), 4, E + 1)
    return S, E, N
end

"""
    _resilient_topology(rng, positions, n_edges)

Geometric candidate links: node `order[i]` joins its nearest node among
`order[1:i-1]` (a spanning tree listed first), then shortcuts are added from
each node's nearest neighbours, shortest first, until `n_edges` exist.
"""
function _resilient_topology(rng::AbstractRNG, positions, n_edges::Int)
    n = length(positions)
    dist(i, j) = hypot(positions[i][1] - positions[j][1], positions[i][2] - positions[j][2])
    order = randperm(rng, n)
    edges = Tuple{Int, Int}[]
    seen = Set{Tuple{Int, Int}}()
    for idx in 2:n
        child = order[idx]
        parent = order[argmin([dist(child, order[k]) for k in 1:(idx - 1)])]
        edge = minmax(parent, child)
        push!(edges, edge)
        push!(seen, edge)
    end
    k = min(n - 1, max(4, ceil(Int, 2 * n_edges / n) + 2))
    candidates = Tuple{Float64, Int, Int}[]
    for i in 1:n
        nearest = partialsortperm([j == i ? Inf : dist(i, j) for j in 1:n], 1:k)
        for j in nearest
            edge = minmax(i, j)
            edge in seen && continue
            push!(candidates, (dist(i, j) * rand(rng, Uniform(0.9, 1.1)), edge...))
        end
    end
    sort!(candidates; by=first)
    for (_, i, j) in candidates
        length(edges) >= n_edges && break
        edge = (i, j)
        edge in seen && continue
        push!(seen, edge)
        push!(edges, edge)
    end
    return edges
end

"""
    _resilient_tree_path(n_nodes, edges, tree_count, source, sink) -> Vector{Tuple{Int,Bool}}

Edges of the unique spanning-tree path from `source` to `sink`, each with
`true` when traversed in its stored `(i, j)` orientation.
"""
function _resilient_tree_path(n_nodes::Int, edges, tree_count::Int, source::Int, sink::Int)
    adjacency = [Tuple{Int, Int}[] for _ in 1:n_nodes]
    for e in 1:tree_count
        i, j = edges[e]
        push!(adjacency[i], (j, e))
        push!(adjacency[j], (i, e))
    end
    parent_edge = zeros(Int, n_nodes)
    parent_node = zeros(Int, n_nodes)
    visited = falses(n_nodes)
    queue = [source]
    visited[source] = true
    while !isempty(queue)
        u = popfirst!(queue)
        for (v, e) in adjacency[u]
            visited[v] && continue
            visited[v] = true
            parent_node[v] = u
            parent_edge[v] = e
            push!(queue, v)
        end
    end
    path = Tuple{Int, Bool}[]
    v = sink
    while v != source
        u = parent_node[v]
        e = parent_edge[v]
        push!(path, (e, edges[e] == (u, v)))
        v = u
    end
    return reverse!(path)
end

function ResilientNetworkDesignProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    S, E_target, N = _resilient_dimensions(target_variables)
    positions = [(rand(rng, Uniform(0, 100)), rand(rng, Uniform(0, 100))) for _ in 1:N]
    edges = _resilient_topology(rng, positions, E_target)
    E = length(edges)
    tree_count = N - 1
    len = [hypot(positions[i][1] - positions[j][1], positions[i][2] - positions[j][2]) for (i, j) in edges]

    sources = zeros(Int, S)
    sinks = zeros(Int, S)
    for s in 1:S
        sources[s] = rand(rng, 1:N)
        sink = rand(rng, 1:(N - 1))
        sinks[s] = sink >= sources[s] ? sink + 1 : sink
    end
    demands = [round(rand(rng, LogNormal(log(12.0), 0.35)); digits=2) for _ in 1:S]
    capacities = [round(rand(rng, Uniform(12.0, 40.0)); digits=1) for _ in 1:E]
    build_cost = [round(5.0 + 2.0 * len[e] * rand(rng, Uniform(0.8, 1.25)); digits=2) for e in 1:E]
    hardening_cost = [round(build_cost[e] * rand(rng, Uniform(0.3, 0.9)); digits=2) for e in 1:E]
    routing_cost = [round(0.05 + 0.08 * len[e] * rand(rng, Uniform(0.9, 1.1)); digits=3) for e in 1:E]

    # Regional hazards: failure probability decays with distance to the center.
    hazard_center = [(rand(rng, Uniform(0, 100)), rand(rng, Uniform(0, 100))) for _ in 1:S]
    hazard_radius = [rand(rng, Uniform(15.0, 35.0)) for _ in 1:S]
    failed = falses(E, S)
    for s in 1:S, (e, (i, j)) in enumerate(edges)
        mid = ((positions[i][1] + positions[j][1]) / 2, (positions[i][2] + positions[j][2]) / 2)
        r = hypot(mid[1] - hazard_center[s][1], mid[2] - hazard_center[s][2]) / hazard_radius[s]
        failed[e, s] = rand(rng) < 0.85 * exp(-r^2) + 0.03
    end
    for s in 1:S
        any(@view failed[:, s]) || (failed[rand(rng, 1:E), s] = true)
    end

    tree_cost = sum(build_cost[e] + hardening_cost[e] for e in 1:tree_count)
    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        max_demand = maximum(demands)
        for e in 1:tree_count
            capacities[e] = max(capacities[e], round(1.15 * max_demand + 1.0; digits=1))
        end
        design_budget = 1.05 * tree_cost + 1.0
        forward = zeros(Float64, E, S)
        reverse = zeros(Float64, E, S)
        for s in 1:S
            for (e, along) in _resilient_tree_path(N, edges, tree_count, sources[s], sinks[s])
                along ? (forward[e, s] = demands[s]) : (reverse[e, s] = demands[s])
            end
        end
        build = [e <= tree_count ? 1.0 : 0.0 for e in 1:E]
        witness = ResilientNetworkWitness(build, copy(build), forward, reverse)
    elseif feasibility_status == infeasible
        design_budget = sum(build_cost) + sum(hardening_cost) + 1.0
        adjacency = [Int[] for _ in 1:N]
        for (i, j) in edges
            push!(adjacency[i], j)
            push!(adjacency[j], i)
        end
        # Breadth-first ball around the sink that stops before the source.
        source, sink = sources[1], sinks[1]
        ball_size = clamp(round(Int, sqrt(N)), 4, max(4, N ÷ 3))
        region = [sink]
        in_region = falses(N)
        in_region[sink] = true
        queue = [sink]
        while !isempty(queue) && length(region) < ball_size
            u = popfirst!(queue)
            for v in shuffle(rng, adjacency[u])
                (in_region[v] || v == source || length(region) >= ball_size) && continue
                in_region[v] = true
                push!(region, v)
                push!(queue, v)
            end
        end
        cut_edges = [e for (e, (i, j)) in enumerate(edges) if in_region[i] != in_region[j]]
        goal = demands[1] / rand(rng, Uniform(1.08, 1.20))
        current = sum(capacities[e] for e in cut_edges)
        for e in cut_edges
            capacities[e] *= goal / current
        end
        cut_capacity = sum(capacities[e] for e in cut_edges)
        certificate = ResilientNetworkCutCertificate(
            1, sort(region), cut_edges, demands[1], cut_capacity, demands[1] - cut_capacity
        )
    else
        design_budget = tree_cost * rand(rng, Uniform(0.45, 1.1))
    end

    return ResilientNetworkDesignProblem(
        N,
        E,
        S,
        positions,
        edges,
        sources,
        sinks,
        demands,
        capacities,
        build_cost,
        hardening_cost,
        routing_cost,
        hazard_center,
        hazard_radius,
        failed,
        design_budget,
        witness,
        certificate,
        feasibility_status,
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

    @objective(
        model,
        Min,
        sum(prob.build_cost[e] * build[e] + prob.hardening_cost[e] * harden[e] for e in 1:E) +
            sum(prob.routing_cost[e] * (forward[e, s] + reverse[e, s]) / S for e in 1:E, s in 1:S)
    )
    @constraint(
        model,
        design_budget,
        sum(prob.build_cost[e] * build[e] + prob.hardening_cost[e] * harden[e] for e in 1:E) <=
            prob.design_budget
    )
    @constraint(model, harden_requires_build[e = 1:E], harden[e] <= build[e])
    @constraint(
        model,
        link_capacity[e = 1:E, s = 1:S],
        forward[e, s] + reverse[e, s] <=
        prob.capacities[e] * (prob.failed[e, s] ? harden[e] : build[e])
    )

    incident = [Int[] for _ in 1:prob.n_nodes]
    for (e, (i, j)) in enumerate(prob.edges)
        push!(incident[i], e)
        push!(incident[j], e)
    end
    balance = [AffExpr(0.0) for _ in 1:prob.n_nodes, _ in 1:S]
    for s in 1:S, (e, (i, j)) in enumerate(prob.edges)
        add_to_expression!(balance[i, s], 1.0, forward[e, s])
        add_to_expression!(balance[i, s], -1.0, reverse[e, s])
        add_to_expression!(balance[j, s], 1.0, reverse[e, s])
        add_to_expression!(balance[j, s], -1.0, forward[e, s])
    end
    rhs(node, s) =
        node == prob.sources[s] ? prob.demands[s] : (node == prob.sinks[s] ? -prob.demands[s] : 0.0)
    @constraint(model, flow_balance[v = 1:prob.n_nodes, s = 1:S], balance[v, s] == rhs(v, s))
    return model
end

register_variant(
    :resilient_network_design,
    :standard,
    ResilientNetworkDesignProblem,
    "Two-stage network build and hardening under spatially correlated hazard scenarios: geometric candidate links, scenario flow routing with failed links usable only if hardened, and a design budget";
    tags=[:telecom, :network, :dual_block_angular, :big_m],
)
