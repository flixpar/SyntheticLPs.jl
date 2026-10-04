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
Hardening-budget certificate: a lower bound on design spend, built from LP
rows only, that exceeds the design budget.

 1. *Bridges.* Removing a bridge link (`bridges`) splits the network, so a
    scenario whose source and sink fall on different sides routes its whole
    demand over it (sum one side's balance rows); its capacity row forces
    `harden ≥ demand / capacity` when the link fails in that scenario, else
    `build ≥ demand / capacity`, and `build ≥ harden`. The largest such levels
    are `bridge_build` / `bridge_harden`, costing `forced_spend`.
 2. *District.* `region` contains scenario `scenario`'s sink but not its
    source, and that scenario's hazard takes out every link crossing its
    boundary (`cut_edges`). Summing the region's balance rows, the scenario's
    `demand` must cross the boundary, where a failed link carries at most
    `capacity[e] * harden[e]`: `Σ_cut capacity[e] harden[e] ≥ demand`
    (`cut_capacity` exceeds `demand`, so fully hardening the boundary would
    do). Raising `harden[e]` above its forced level costs `hardening_cost[e]`
    per unit, plus `build_cost[e]` once it passes the forced build level
    (`harden ≤ build`); the cheapest fractional way to close the district's
    shortfall (a knapsack by cost per unit of capacity) costs `cut_spend`.

Bridges and boundary pieces are priced on disjoint increments, so every
solution of the LP relaxation spends at least
`implied_minimum = forced_spend + cut_spend`, which exceeds `budget` by
`margin`. The budget still covers the forced spend plus most of the district,
so no single row's bound propagation exposes the shortfall.
"""
struct ResilientHardeningBudgetCertificate
    scenario::Int
    region::Vector{Int}
    cut_edges::Vector{Int}
    demand::Float64
    cut_capacity::Float64
    bridges::Vector{Int}
    bridge_build::Vector{Float64}
    bridge_harden::Vector{Float64}
    forced_spend::Float64
    cut_spend::Float64
    implied_minimum::Float64
    budget::Float64
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
  - `infeasible`: scenario 1's hazard is centred on a district (a
    breadth-first ball of about `sqrt(n_nodes)` nodes around its sink,
    excluding its source) and takes out every access link into it. The
    access links' capacities total `U(1.25, 1.45) ×` the demand, so hardening
    them would carry it, but the design budget is set `U(1.08, 1.20)` times
    below the cheapest fractional hardening of enough boundary capacity
    ([`ResilientHardeningBudgetCertificate`](@ref)). The budget still covers
    any single link, so presolve's bound propagation cannot see the
    shortfall: it is a knapsack over the boundary that needs simplex work.
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
    infeasibility_certificate::Union{Nothing, ResilientHardeningBudgetCertificate}
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

"""
    _resilient_district(rng, n_nodes, edges, adjacency, excluded, sink) -> (region, cut_edges)

Breadth-first ball of about `sqrt(n_nodes)` nodes (at least 4, at most a third
of the nodes) around `sink` that never enters an `excluded` node (the
scenario's source and every other scenario's endpoints, so no other scenario
has to reach into the district), grown further while its boundary has fewer
than six links. Returns the region's nodes and the
indices of the links crossing its boundary.
"""
function _resilient_district(
    rng::AbstractRNG, n_nodes::Int, edges, adjacency, excluded::AbstractVector{Bool}, sink::Int
)
    ball_size = clamp(round(Int, sqrt(n_nodes)), 4, max(4, n_nodes ÷ 3))
    region = [sink]
    in_region = falses(n_nodes)
    in_region[sink] = true
    queue = [sink]
    boundary() = [e for (e, (i, j)) in enumerate(edges) if in_region[i] != in_region[j]]
    while !isempty(queue)
        u = popfirst!(queue)
        for v in shuffle(rng, adjacency[u])
            (in_region[v] || excluded[v]) && continue
            if length(region) >= ball_size
                # A district reached by only a handful of links needs too
                # little hardening for the budget to bind without also
                # capping single links; keep growing (up to half the map).
                (length(boundary()) >= 6 || 2 * length(region) >= n_nodes) && break
            end
            in_region[v] = true
            push!(region, v)
            push!(queue, v)
        end
        length(region) >= ball_size && length(boundary()) >= 6 && break
        2 * length(region) >= n_nodes && break
    end
    # Fill holes: a node whose every neighbour is in the district (unless
    # excluded) joins it - left outside, all its links would be
    # resized boundary links, a one-row bottleneck for its own traffic.
    changed = true
    while changed
        changed = false
        for v in 1:n_nodes
            (in_region[v] || excluded[v]) && continue
            if all(in_region[w] for w in adjacency[v])
                in_region[v] = true
                push!(region, v)
                changed = true
            end
        end
    end
    return region, boundary()
end

"""
    _resilient_bridges(n_nodes, arcs) -> (bridges, child, tin, tout)

Bridge links of the (connected) topology by an iterative Tarjan DFS from node
1. `bridges[t]` is a link index whose removal splits off the DFS subtree of
node `child[t]`; node `v` lies in that subtree iff
`tin[child[t]] <= tin[v] <= tout[child[t]]`.
"""
function _resilient_bridges(n_nodes::Int, arcs::Vector{Tuple{Int, Int}})
    adjacency = [Tuple{Int, Int}[] for _ in 1:n_nodes]
    for (a, (i, j)) in enumerate(arcs)
        push!(adjacency[i], (j, a))
        push!(adjacency[j], (i, a))
    end
    tin = zeros(Int, n_nodes)
    tout = zeros(Int, n_nodes)
    low = zeros(Int, n_nodes)
    parent_link = zeros(Int, n_nodes)
    next_edge = ones(Int, n_nodes)
    bridges = Int[]
    child = Int[]
    timer = 0
    for root in 1:n_nodes
        tin[root] != 0 && continue
        timer += 1
        tin[root] = low[root] = timer
        stack = [root]
        while !isempty(stack)
            u = stack[end]
            if next_edge[u] <= length(adjacency[u])
                v, a = adjacency[u][next_edge[u]]
                next_edge[u] += 1
                a == parent_link[u] && continue
                if tin[v] == 0
                    timer += 1
                    tin[v] = low[v] = timer
                    parent_link[v] = a
                    push!(stack, v)
                else
                    low[u] = min(low[u], tin[v])
                end
            else
                pop!(stack)
                tout[u] = timer
                if !isempty(stack)
                    w = stack[end]
                    low[w] = min(low[w], low[u])
                    if low[u] > tin[w]
                        push!(bridges, parent_link[u])
                        push!(child, u)
                    end
                end
            end
        end
    end
    return bridges, child, tin, tout
end

"""
    _resilient_forced_levels(n_nodes, edges, capacities, failed, sources, sinks, demands)
    -> (forced_build, forced_harden, bridges, child, tin, tout)

Design levels every solution must reach on bridge links. Removing bridge `b`
separates the network, so a scenario whose source and sink fall on different
sides routes all its demand over `b` (sum one side's balance rows); the
capacity row then forces `harden[b] ≥ demand / capacity[b]` if `b` fails in
that scenario, else `build[b] ≥ demand / capacity[b]` (and `build ≥ harden`).
Non-bridge links get zeros. The caller keeps bridge capacities above every
crossing demand, so the levels stay below 1.
"""
function _resilient_forced_levels(n_nodes::Int, edges, capacities, failed, sources, sinks, demands)
    E = length(edges)
    forced_build = zeros(E)
    forced_harden = zeros(E)
    bridges, child, tin, tout = _resilient_bridges(n_nodes, edges)
    for (t, b) in enumerate(bridges)
        inside(v) = tin[child[t]] <= tin[v] <= tout[child[t]]
        for s in eachindex(sources)
            inside(sources[s]) == inside(sinks[s]) && continue
            level = demands[s] / capacities[b]
            failed[b, s] && (forced_harden[b] = max(forced_harden[b], level))
            forced_build[b] = max(forced_build[b], level)
        end
    end
    return forced_build, forced_harden, bridges, child, tin, tout
end

"""
    _resilient_hardening_spend(capacity, build_cost, hardening_cost,
                               forced_build, forced_harden, required) -> Float64

Cheapest fractional way to add `required` units of hardened capacity on
boundary links already at the forced levels: raising `harden[e]` from
`forced_harden[e]` to `forced_build[e]` costs `hardening_cost[e]` per unit,
beyond it also `build_cost[e]` (`harden ≤ build`). The cost is convex in each
link's level, so the greedy cost-per-capacity order over these pieces is
optimal (`Inf` if even full hardening falls short, `0` if `required ≤ 0`).
"""
function _resilient_hardening_spend(
    capacity::Vector{Float64},
    build_cost::Vector{Float64},
    hardening_cost::Vector{Float64},
    forced_build::Vector{Float64},
    forced_harden::Vector{Float64},
    required::Float64,
)
    required <= 0 && return 0.0
    pieces = Tuple{Float64, Float64, Float64}[]  # (cost per capacity, capacity, unit cost)
    for e in eachindex(capacity)
        cheap = forced_build[e] - forced_harden[e]
        cheap > 0 && push!(pieces, (hardening_cost[e] / capacity[e], capacity[e] * cheap, hardening_cost[e]))
        full = 1.0 - max(forced_build[e], forced_harden[e])
        unit = build_cost[e] + hardening_cost[e]
        full > 0 && push!(pieces, (unit / capacity[e], capacity[e] * full, unit))
    end
    sort!(pieces; by=first)
    spend = 0.0
    remaining = required
    for (ratio, cap, _) in pieces
        take = min(cap, remaining)
        spend += ratio * take
        remaining -= take
        remaining <= 1e-12 * required && return spend
    end
    return Inf
end

"""
    _resilient_max_flow(n_nodes, edges, capacities, source, sink) -> (value, source_side)

Maximum `source`-`sink` flow when every link is built and hardened (an
undirected link carries up to its capacity in either direction), by Dinic's
algorithm, and the source side of a minimum cut.
"""
function _resilient_max_flow(n_nodes::Int, edges, capacities, source::Int, sink::Int)
    # Arc 2e-1 is i -> j, arc 2e is j -> i; each starts with the full capacity.
    m = length(edges)
    head = Vector{Int}(undef, 2m)
    residual = Vector{Float64}(undef, 2m)
    out = [Int[] for _ in 1:n_nodes]
    for (e, (i, j)) in enumerate(edges)
        head[2e - 1], head[2e] = j, i
        residual[2e - 1] = residual[2e] = capacities[e]
        push!(out[i], 2e - 1)
        push!(out[j], 2e)
    end
    partner(a) = isodd(a) ? a + 1 : a - 1
    tail(a) = head[partner(a)]
    level = zeros(Int, n_nodes)
    pointer = ones(Int, n_nodes)
    tol = 1e-12 * (1.0 + sum(capacities))
    value = 0.0
    function bfs!()
        fill!(level, 0)
        level[source] = 1
        queue = [source]
        while !isempty(queue)
            u = popfirst!(queue)
            for a in out[u]
                v = head[a]
                if level[v] == 0 && residual[a] > tol
                    level[v] = level[u] + 1
                    push!(queue, v)
                end
            end
        end
        return level[sink] != 0
    end
    while bfs!()
        fill!(pointer, 1)
        while true
            # Iterative blocking-flow DFS: advance along level-increasing arcs.
            path = Int[]
            u = source
            while u != sink
                advanced = false
                while pointer[u] <= length(out[u])
                    a = out[u][pointer[u]]
                    v = head[a]
                    if residual[a] > tol && level[v] == level[u] + 1
                        push!(path, a)
                        u = v
                        advanced = true
                        break
                    end
                    pointer[u] += 1
                end
                if !advanced
                    u == source && break
                    level[u] = 0  # dead end: retreat
                    a = pop!(path)
                    u = tail(a)
                    pointer[u] += 1
                end
            end
            u == sink || break
            push_amount = minimum(residual[a] for a in path)
            for a in path
                residual[a] -= push_amount
                residual[partner(a)] += push_amount
            end
            value += push_amount
        end
    end
    source_side = falses(n_nodes)
    source_side[source] = true
    queue = [source]
    while !isempty(queue)
        u = popfirst!(queue)
        for a in out[u]
            v = head[a]
            if !source_side[v] && residual[a] > tol
                source_side[v] = true
                push!(queue, v)
            end
        end
    end
    return value, source_side
end

"""
    _resilient_budget_plan(rng, ...; star) -> NamedTuple

Infeasible-mode data for scenario `star`: its hazard is recentred on a
district around its sink (redrawing that scenario's failures) and fails every
access link; the access links are resized to total `U(1.25, 1.45) ×` the
demand; bridges are kept at least 1.25 × the largest demand crossing them;
then the bridge-forced design levels and spend, and the cheapest fractional
hardening of the district's remaining shortfall (`cut_spend`), are computed.
"""
function _resilient_budget_plan(
    rng::AbstractRNG, N, edges, adjacency, positions, sources, sinks, demands, capacities,
    build_cost, hardening_cost, failed, hazard_radius, star::Int,
)
    excluded = falses(N)
    for s in eachindex(sources)
        excluded[sources[s]] = true
        excluded[sinks[s]] = true
    end
    excluded[sinks[star]] = false
    excluded[sources[star]] = true
    region, cut_edges = _resilient_district(rng, N, edges, adjacency, excluded, sinks[star])
    cx = sum(positions[v][1] for v in region) / length(region)
    cy = sum(positions[v][2] for v in region) / length(region)
    for (e, (i, j)) in enumerate(edges)
        mid = ((positions[i][1] + positions[j][1]) / 2, (positions[i][2] + positions[j][2]) / 2)
        r = hypot(mid[1] - cx, mid[2] - cy) / hazard_radius[star]
        failed[e, star] = rand(rng) < 0.85 * exp(-r^2) + 0.03
    end
    failed[cut_edges, star] .= true
    # Access links sized so hardening the whole boundary would carry the
    # demand with 25-45% to spare: capacity alone is not the obstruction -
    # for any scenario that has to cross the district's boundary.
    in_district = falses(N)
    in_district[region] .= true
    crossing = maximum(
        demands[s] for s in eachindex(demands) if in_district[sources[s]] != in_district[sinks[s]]
    )
    goal = crossing * rand(rng, Uniform(1.25, 1.45))
    current = sum(capacities[e] for e in cut_edges)
    for e in cut_edges
        capacities[e] = round(capacities[e] * goal / current; digits=3)
    end
    # Headroom guard: with every link built and hardened, each scenario's
    # maximum flow must exceed its demand by 25%, so the budget is the only
    # obstruction and no small bottleneck (a node hemmed in by the resized
    # boundary, say) lets presolve refute the instance on its own. Links of a
    # short minimum cut are widened - off the district boundary when possible.
    on_cut = falses(length(edges))
    on_cut[cut_edges] .= true
    for s in eachindex(sources)
        for _ in 1:50
            value, side = _resilient_max_flow(N, edges, capacities, sources[s], sinks[s])
            value >= 1.25 * demands[s] && break
            crossing = [e for (e, (i, j)) in enumerate(edges) if side[i] != side[j]]
            widen = [e for e in crossing if !on_cut[e]]
            isempty(widen) && (widen = crossing)
            grow = (1.25 * demands[s] - value) / sum(capacities[e] for e in widen) + 1.0
            for e in widen
                capacities[e] = round(capacities[e] * grow + 1e-3; digits=3)
            end
        end
    end
    # (Bridges are minimum cuts too, so every forced level is at most 0.8.)
    forced_build, forced_harden, bridges, _, _, _ = _resilient_forced_levels(
        N, edges, capacities, failed, sources, sinks, demands
    )
    forced_spend = sum(
        build_cost[b] * forced_build[b] + hardening_cost[b] * forced_harden[b] for b in bridges;
        init=0.0,
    )
    required = demands[star] - sum(capacities[e] * forced_harden[e] for e in cut_edges)
    cut_spend = _resilient_hardening_spend(
        capacities[cut_edges],
        build_cost[cut_edges],
        hardening_cost[cut_edges],
        forced_build[cut_edges],
        forced_harden[cut_edges],
        required,
    )
    return (
        scenario=star,
        region=region,
        cut_edges=cut_edges,
        center=(cx, cy),
        capacities=capacities,
        failed=failed,
        bridges=bridges,
        forced_build=forced_build[bridges],
        forced_harden=forced_harden[bridges],
        forced_spend=forced_spend,
        cut_spend=cut_spend,
    )
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

    adjacency = [Int[] for _ in 1:N]
    for (i, j) in edges
        push!(adjacency[i], j)
        push!(adjacency[j], i)
    end

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
        # Plan a district for every scenario and keep the one whose hardening
        # shortfall (beyond what bridges already force) is largest: a large
        # `cut_spend` leaves the budget room for every single link.
        best = nothing
        for star in 1:S
            plan = _resilient_budget_plan(
                rng, N, edges, adjacency, positions, sources, sinks, demands, copy(capacities),
                build_cost, hardening_cost, copy(failed), hazard_radius, star,
            )
            if best === nothing || plan.cut_spend > best.cut_spend
                best = plan
            end
        end
        capacities = best.capacities
        failed = best.failed
        hazard_center[best.scenario] = best.center
        implied = best.forced_spend + best.cut_spend
        shrink = rand(rng, Uniform(1.08, 1.20))
        # The shortfall sits on the district's knapsack; only on degenerate
        # tiny networks, where bridges already force most of the spend, is
        # the whole bound shrunk instead (still a strict, certified margin).
        design_budget = if best.cut_spend >= 0.25 * implied
            best.forced_spend + best.cut_spend / shrink
        else
            implied / shrink
        end
        certificate = ResilientHardeningBudgetCertificate(
            best.scenario,
            sort(best.region),
            best.cut_edges,
            demands[best.scenario],
            sum(capacities[e] for e in best.cut_edges),
            best.bridges,
            best.forced_build,
            best.forced_harden,
            best.forced_spend,
            best.cut_spend,
            implied,
            design_budget,
            implied - design_budget,
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
