using JuMP
using Random
using Distributions

"""
Largest `target_variables` accepted by `NetworkFlowProblem`. Variables are the
directed arcs, and the constructor materialises every forward candidate arc
(`n * (n - 1) / 2` of them, about two million at the cap) while shuffling the
fill set, so larger targets are rejected with an `ArgumentError` instead of
being silently undersized (same convention as `telecom_network_design/standard`
and `supply_chain/network_planning`).
"""
const NETWORK_FLOW_MAX_ARCS = 1_000_000

"""
Planted flow plan: a genuine feasible point of the built model, derived from the
exact maximum flow computed on the sampled capacities. For `:min_cost` instances
the max-flow assignment is scaled by `target_flow / max_flow_value`; scaling
shrinks every arc flow, so capacity rows stay satisfied, conservation survives
because the rows are linear, and the source outflow hits the contracted volume.
For `:max_flow` instances the unscaled assignment is stored — a feasible point
that is in fact optimal. `source_outflow` is the plan's total flow leaving the
source node.
"""
struct NetworkFlowWitness
    arc_flows::Vector{Float64}
    source_outflow::Float64
end

"""
Min-cut infeasibility certificate: `cut_arcs` are exactly the stored arcs whose
tail lies on `source_side` (which contains the source, not the sink) and whose
head does not, and their total capacity is `cut_capacity`. In a forward-arc DAG
nothing re-enters the source, so any flow meeting the contracted volume sends
`target_flow` plus any back-crossing flow — at least `target_flow` — across the
cut, while the crossing arcs can carry at most `cut_capacity < target_flow` in
total. That contradicts the model's source-outflow equality using LP rows alone;
by max-flow/min-cut the stored cut is a minimum cut, so `cut_capacity` equals
`max_flow_value`.
"""
struct NetworkFlowCutCertificate
    source_side::Vector{Int}
    cut_arcs::Vector{Int}
    cut_capacity::Float64
end

"""
    NetworkFlowProblem <: ProblemGenerator

Generator for single-commodity network flow problems over a directed acyclic
network, from a source (node 1) to a sink (node `n_nodes`).

# Overview

The decisions are nonnegative arc flows. Arc rows enforce capacities,
intermediate-node rows conserve flow, and the network is a forward-arc DAG
(`i < j`) with a guaranteed `1 -> 2 -> ... -> n` backbone, so nothing can
re-enter the source or leave the sink. Depending on the sampled objective the
model either maximizes source outflow (`:max_flow`) or minimizes routing cost
for a contracted volume (`:min_cost`), enforced as an equality on the source's
out-arcs.

# Data grounding

Nodes are scattered over a 100 x 100 region in one of three geography shapes
(`:corridor` — a bent transport corridor, in index order, so the backbone
follows it; `:clustered` — population clusters; `:uniform` — mesh-like
coverage). Per-unit routing cost is proportional to the Euclidean distance
between the arc's endpoints, multiplied by a lognormal route factor whose spread
is drawn once per instance — long-haul arcs cost more, with route-dependent
heterogeneity on top. Capacities are lognormal, in tiers that grow with the
requested scale (feeder networks vs. trunk lines vs. backbone grids).

# Feasibility control

The constructor computes the TRUE maximum source-sink flow (Dinic's algorithm,
deterministic) on the sampled capacities, so every profile is placed relative to
an exact boundary rather than an estimate:

  - `feasible`: `target_flow` is 25%-85% of the max flow; the stored witness is
    the scaled max-flow assignment, a feasible point by construction.
  - `infeasible`: the request becomes a min-cost contract 115%-160% of the max
    flow (a max-flow objective is always feasible — the zero flow is admissible
    — so it is switched to `:min_cost`); the stored certificate is a minimum cut
    whose capacity equals the max flow and is therefore strictly below the
    contract.
  - `unknown`: for `:min_cost` the contracted volume is 60%-140% of the max
    flow — a genuine coin flip on either side of the boundary; `:max_flow`
    instances stay unconstrained (trivially feasible, no claim either way).

The instance is a pure continuous LP; the certificate refutes the LP itself, so
it survives the default `relax_integer` (a no-op here) and every transform.

# Fields

  - `n_nodes::Int`: Number of nodes (node 1 = source, node `n_nodes` = sink)
  - `source_node::Int`: Source node index (always 1)
  - `sink_node::Int`: Sink node index (always `n_nodes`)
  - `arcs::Vector{Tuple{Int,Int}}`: Forward arcs `(i, j)`, `i < j`, sorted
  - `capacities::Vector{Float64}`: Per-arc capacity, aligned with `arcs`
  - `costs::Vector{Float64}`: Per-unit routing cost, aligned with `arcs`
  - `positions::Vector{Tuple{Float64,Float64}}`: Node coordinates in `[0, 100]^2`
  - `geography::Symbol`: Sampled geography shape
  - `flow_objective::Symbol`: `:max_flow` or `:min_cost`
  - `target_flow::Union{Float64,Nothing}`: Contracted volume (`:min_cost` only)
  - `max_flow_value::Float64`: Exact max source-sink flow of the sampled arcs
  - `feasible_witness::Union{Nothing,NetworkFlowWitness}`: set for `feasible`
  - `infeasibility_certificate::Union{Nothing,NetworkFlowCutCertificate}`: set
    for `infeasible`
  - `feasibility_status::FeasibilityStatus`: Requested profile
"""
struct NetworkFlowProblem <: ProblemGenerator
    n_nodes::Int
    source_node::Int
    sink_node::Int
    arcs::Vector{Tuple{Int, Int}}
    capacities::Vector{Float64}
    costs::Vector{Float64}
    positions::Vector{Tuple{Float64, Float64}}
    geography::Symbol
    flow_objective::Symbol
    target_flow::Union{Float64, Nothing}
    max_flow_value::Float64
    feasible_witness::Union{Nothing, NetworkFlowWitness}
    infeasibility_certificate::Union{Nothing, NetworkFlowCutCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _network_flow_positions(rng, n, shape; span=100.0) -> Vector{Tuple{Float64,Float64}}

Scatter `n` nodes over a `span` x `span` region in one of three geography
shapes:

  - `:corridor`  – nodes strung along a slightly bent transport corridor *in
    index order*, so source and sink sit at opposite ends and the guaranteed
    backbone path follows the corridor (pipeline, rail, river navigation);
  - `:clustered` – nodes grouped around a handful of well-separated cluster
    centers with Gaussian spread (regional road networks); index order is
    randomised so the source and sink land anywhere;
  - `:uniform`   – uniform coverage of the region (mesh-like distribution
    grids).

The positions ground the cost model — per-unit cost is distance-proportional —
so the shape decides whether short-haul or long-haul arcs dominate an instance.
"""
function _network_flow_positions(rng::AbstractRNG, n::Int, shape::Symbol; span::Float64=100.0)
    if shape == :corridor
        lateral = rand(rng, Uniform(0.2span, 0.8span))
        slope = rand(rng, Uniform(-0.6, 0.6))
        jitter = 0.07span
        return [
            (
                clamp(u + rand(rng, Normal(0.0, jitter)), 0.0, span),
                clamp(lateral + slope * (u - span / 2) + rand(rng, Normal(0.0, jitter)), 0.0, span),
            ) for u in range(0.0, span; length=n)
        ]
    end

    positions = if shape == :clustered
        n_groups = clamp(round(Int, sqrt(n)), 2, 6)
        centers = _network_flow_centers(rng, n_groups, 0.3span; span=span)
        spread = 0.08span
        # One anchor node per group sits exactly on its center; the rest are
        # Gaussian scatter around a uniformly drawn group.
        placed = [centers[g] for g in 1:n_groups]
        while length(placed) < n
            g = rand(rng, 1:n_groups)
            push!(
                placed,
                (
                    clamp(centers[g][1] + rand(rng, Normal(0.0, spread)), 0.0, span),
                    clamp(centers[g][2] + rand(rng, Normal(0.0, spread)), 0.0, span),
                ),
            )
        end
        placed
    else  # :uniform
        [(span * rand(rng), span * rand(rng)) for _ in 1:n]
    end
    shuffle!(rng, positions)
    return positions
end

"""
    _network_flow_centers(rng, q, min_sep; span=100.0, tries=400)

Rejection-sample `q` cluster centers in `[0, span]^2` that are pairwise at least
`min_sep` apart, relaxing the separation geometrically when it cannot be met so
a valid configuration is always returned.
"""
function _network_flow_centers(
    rng::AbstractRNG, q::Int, min_sep::Float64; span::Float64=100.0, tries::Int=400
)
    centers = Tuple{Float64, Float64}[]
    sep = min_sep
    while length(centers) < q
        proposed = (span * rand(rng), span * rand(rng))
        if all(hypot(proposed[1] - c[1], proposed[2] - c[2]) >= sep for c in centers)
            push!(centers, proposed)
        elseif rand(rng) < 1.0 / tries
            sep = max(sep * 0.95, 1e-6 * span)
        end
    end
    return centers
end

"""
    _network_flow_topology(rng, n_nodes, n_arcs) -> Vector{Tuple{Int,Int}}

Build a directed network on `1:n_nodes` with exactly `n_arcs` forward arcs
`(i, j)`, `i < j` (targets below the 3-arc backbone of the minimum 4-node
network round up to it): the backbone path `1 -> 2 -> ... -> n_nodes` is always
present, shortcut arcs from the source and to the sink are added with
probability 0.3 each while budget remains (feeder arcs bypassing intermediate
relays, as the `generalized_flow` sibling does), and the remainder is a
shuffled fill of the forward candidates. Forward-only arcs keep the graph a DAG ordered by node
index — nothing re-enters the source or leaves the sink — which is what makes
source outflow, sink inflow, and cut crossings coincide, so the max-flow /
min-cut theory behind the feasibility control applies verbatim to the built LP.

The returned list is sorted lexicographically: the fill set is shuffled, so
sorting (not `Set` iteration order) is what makes the struct reproducible.
"""
function _network_flow_topology(rng::AbstractRNG, n_nodes::Int, n_arcs::Int)
    arcs = Set{Tuple{Int, Int}}()

    # Backbone path (guarantees source -> sink connectivity).
    for i in 1:(n_nodes - 1)
        push!(arcs, (i, i + 1))
    end

    # Source shortcuts and sink shortcuts, capped at the target arc count:
    # small requests leave little or no budget beyond the backbone, and
    # uncapped shortcuts could return more arcs than requested (the fill stage
    # below only tops up, it never truncates).
    for i in 2:(n_nodes - 1)
        if rand(rng) < 0.3 && length(arcs) < n_arcs
            push!(arcs, (1, i))
        end
        if rand(rng) < 0.3 && length(arcs) < n_arcs
            push!(arcs, (i, n_nodes))
        end
    end

    # Shuffled fill of the remaining forward candidates, up to the target.
    if length(arcs) < n_arcs
        candidates = [(i, j) for i in 1:n_nodes for j in (i + 1):n_nodes if (i, j) ∉ arcs]
        shuffle!(rng, candidates)
        for arc in candidates
            length(arcs) >= n_arcs && break
            push!(arcs, arc)
        end
    end

    return sort!(collect(arcs))
end

"""
    _network_flow_max_flow(n_nodes, source, sink, arcs, capacities)
        -> (value, arc_flows, source_side, cut_arcs, cut_capacity)

Exact maximum `source`-`sink` flow (Dinic's algorithm: BFS level graphs plus an
iterative blocking-flow DFS with current-arc pointers), together with the
achieving per-arc flow assignment and a minimum cut read off the residual
graph's source-reachable set. Residual capacities are compared with a `1e-9`
tolerance so floating-point residues of saturated edges are never mistaken for
usable slack; the smallest real capacity is `0.01`, seven orders of magnitude
above it, so the computed value is exact up to rounding noise.

Deterministic — no RNG. Called from the constructor so the stored
`max_flow_value` (and the witness/certificate derived from it) can be relied on
by the deterministic `build_model`.
"""
function _network_flow_max_flow(
    n_nodes::Int, source::Int, sink::Int, arcs::Vector{Tuple{Int, Int}}, capacities::Vector{Float64}
)
    m = length(arcs)
    tol = 1e-9

    # Residual graph: arc k contributes forward edge 2k-1 (capacity) and reverse
    # edge 2k (initially zero); `edge_rev` maps an edge to its counterpart.
    neighbors = [Int[] for _ in 1:n_nodes]
    edge_head = Vector{Int}(undef, 2m)
    edge_cap = Vector{Float64}(undef, 2m)
    edge_rev = Vector{Int}(undef, 2m)
    for k in 1:m
        u, v = arcs[k]
        f, r = 2k - 1, 2k
        push!(neighbors[u], f)
        edge_head[f], edge_cap[f], edge_rev[f] = v, capacities[k], r
        push!(neighbors[v], r)
        edge_head[r], edge_cap[r], edge_rev[r] = u, 0.0, f
    end

    value = 0.0
    level = Vector{Int}(undef, n_nodes)
    next_arc = Vector{Int}(undef, n_nodes)
    while true
        # BFS: level graph over edges with usable residual capacity.
        fill!(level, -1)
        level[source] = 0
        queue = [source]
        head = 1
        while head <= length(queue)
            u = queue[head]
            head += 1
            for e in neighbors[u]
                v = edge_head[e]
                if edge_cap[e] > tol && level[v] < 0
                    level[v] = level[u] + 1
                    push!(queue, v)
                end
            end
        end
        level[sink] < 0 && break

        # Iterative blocking-flow DFS with current-arc pointers.
        fill!(next_arc, 1)
        path_nodes = [source]
        path_edges = Int[]
        while true
            if path_nodes[end] == sink
                # Augment along the whole path by its bottleneck ...
                bottleneck = minimum(edge_cap[e] for e in path_edges)
                value += bottleneck
                for e in path_edges
                    edge_cap[e] -= bottleneck
                    edge_cap[edge_rev[e]] += bottleneck
                end
                # ... then retreat to just before the first saturated edge.
                saturated = findfirst(e -> edge_cap[e] <= tol, path_edges)
                resize!(path_edges, saturated - 1)
                resize!(path_nodes, saturated)
            else
                u = path_nodes[end]
                advanced = false
                while next_arc[u] <= length(neighbors[u])
                    e = neighbors[u][next_arc[u]]
                    v = edge_head[e]
                    if edge_cap[e] > tol && level[v] == level[u] + 1
                        push!(path_edges, e)
                        push!(path_nodes, v)
                        advanced = true
                        break
                    end
                    next_arc[u] += 1
                end
                if !advanced
                    # Dead end: retire the node and step back over its in-edge.
                    level[u] = -1
                    pop!(path_nodes)
                    isempty(path_nodes) && break
                    e = pop!(path_edges)
                    next_arc[path_nodes[end]] += 1
                end
            end
        end
    end

    # Minimum cut: nodes still reachable from the source in the residual graph.
    side = falses(n_nodes)
    side[source] = true
    stack = [source]
    while !isempty(stack)
        u = pop!(stack)
        for e in neighbors[u]
            v = edge_head[e]
            if edge_cap[e] > tol && !side[v]
                side[v] = true
                push!(stack, v)
            end
        end
    end

    arc_flows = [capacities[k] - edge_cap[2k - 1] for k in 1:m]
    cut_arcs = [k for k in 1:m if side[arcs[k][1]] && !side[arcs[k][2]]]
    cut_capacity = sum(capacities[k] for k in cut_arcs; init=0.0)
    return value, arc_flows, findall(side), cut_arcs, cut_capacity
end

"""
    NetworkFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a network flow problem instance.

# Arguments

  - `target_variables`: Target number of variables (= arcs). Values above
    `NETWORK_FLOW_MAX_ARCS` raise an `ArgumentError`; targets below 3 round up
    to the 3-arc backbone of the minimum 4-node network.
  - `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
  - `seed`: Random seed for reproducibility
"""
function NetworkFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= NETWORK_FLOW_MAX_ARCS || throw(
        ArgumentError(
            "network_flow/standard supports at most $NETWORK_FLOW_MAX_ARCS arcs; " *
            "requested $target_variables.",
        ),
    )
    rng = MersenneTwister(seed)

    # Capacity tier: larger networks model higher-tier infrastructure (feeder
    # roads -> trunk lines -> backbone grids) whose links are fatter and whose
    # capacity spread is wider.
    cap_mu, cap_sigma = if target_variables <= 100
        (log(40.0), 0.5)
    elseif target_variables <= 500
        (log(120.0), 0.6)
    elseif target_variables <= 5000
        (log(300.0), 0.65)
    else
        (log(600.0), 0.7)
    end

    # Size n_nodes so the forward-arc DAG offers n*(n-1)/2 >= target candidate
    # arcs; combined with the topology fill this makes the arc count exactly the
    # target rather than a silent undersize.
    n_nodes = max(4, ceil(Int, (1 + sqrt(1 + 8 * target_variables)) / 2))
    source_node, sink_node = 1, n_nodes

    geography = let r = rand(rng)
        r < 0.35 ? :corridor : (r < 0.75 ? :clustered : :uniform)
    end
    positions = _network_flow_positions(rng, n_nodes, geography)

    arcs = _network_flow_topology(rng, n_nodes, target_variables)

    # Grounded arc data: lognormal capacity, distance-proportional lognormal
    # per-unit cost (route heterogeneity drawn once per instance).
    cost_spread = 0.25 + 0.3 * rand(rng)
    capacities = Vector{Float64}(undef, length(arcs))
    costs = Vector{Float64}(undef, length(arcs))
    for (k, (u, v)) in enumerate(arcs)
        capacities[k] = max(round(rand(rng, LogNormal(cap_mu, cap_sigma)); digits=2), 0.01)
        d = hypot(positions[v][1] - positions[u][1], positions[v][2] - positions[u][2])
        costs[k] = max(round(d * rand(rng, LogNormal(0.0, cost_spread)); digits=3), 0.01)
    end

    # Objective mix: small instances lean toward max flow, larger ones toward
    # min-cost routing with a contracted volume.
    flow_objective = rand(rng) < (target_variables <= 100 ? 0.7 : 0.4) ? :max_flow : :min_cost

    # Exact max flow and min cut on the sampled capacities (deterministic).
    max_flow_value, arc_flows, source_side, cut_arcs, cut_capacity = _network_flow_max_flow(
        n_nodes, source_node, sink_node, arcs, capacities
    )

    target_flow = nothing
    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        if flow_objective == :max_flow
            feasible_witness = NetworkFlowWitness(copy(arc_flows), max_flow_value)
        else
            fraction = 0.25 + 0.6 * rand(rng)
            target_flow = max_flow_value * fraction
            feasible_witness = NetworkFlowWitness(fraction .* arc_flows, target_flow)
        end
    elseif feasibility_status == infeasible
        # A max-flow objective cannot be infeasible (the zero flow is
        # admissible), so an infeasible request becomes a min-cost instance
        # whose contracted volume provably cannot be routed: the target exceeds
        # the exact max flow, certified by the minimum cut.
        flow_objective = :min_cost
        target_flow = max_flow_value * (1.15 + 0.45 * rand(rng))
        infeasibility_certificate = NetworkFlowCutCertificate(source_side, cut_arcs, cut_capacity)
    else  # unknown
        if flow_objective == :min_cost
            # Contract drawn as 60%-140% of the max flow: a genuine coin flip on
            # either side of the feasibility boundary, at every problem size.
            target_flow = max_flow_value * (0.6 + 0.8 * rand(rng))
        end
        # :max_flow stays unconstrained: trivially feasible, no claim either way.
    end

    return NetworkFlowProblem(
        n_nodes,
        source_node,
        sink_node,
        arcs,
        capacities,
        costs,
        positions,
        geography,
        flow_objective,
        target_flow,
        max_flow_value,
        feasible_witness,
        infeasibility_certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::NetworkFlowProblem)

Build the JuMP model for the network flow problem. Deterministic — uses only
data from the struct fields.

Decision variables:

  - `flow[k] >= 0`: flow on arc `k` (one per arc, so variables == arcs).

# Returns

  - `model`: The JuMP model
"""
function build_model(prob::NetworkFlowProblem)
    model = Model()
    m = length(prob.arcs)
    n = prob.n_nodes

    @variable(model, flow[1:m] >= 0)

    # Per-arc capacity rows.
    @constraint(model, flow .<= prob.capacities)

    # Adjacency built in one pass (building per-node lists by scanning all arcs
    # for every node is O(n*m) and needlessly slow at large sizes).
    out_arcs = [Int[] for _ in 1:n]
    in_arcs = [Int[] for _ in 1:n]
    for (k, (u, v)) in enumerate(prob.arcs)
        push!(out_arcs[u], k)
        push!(in_arcs[v], k)
    end

    # Conservation at intermediate nodes. The backbone guarantees every
    # intermediate node has both in- and out-arcs, so exactly n - 2 rows exist.
    for v in 1:n
        (v == prob.source_node || v == prob.sink_node) && continue
        @constraint(model, sum(flow[k] for k in in_arcs[v]) == sum(flow[k] for k in out_arcs[v]))
    end

    if prob.flow_objective == :max_flow
        @objective(model, Max, sum(flow[k] for k in out_arcs[prob.source_node]))
    else  # :min_cost
        @objective(model, Min, sum(prob.costs[k] * flow[k] for k in 1:m))
        if prob.target_flow !== nothing
            @constraint(model, sum(flow[k] for k in out_arcs[prob.source_node]) == prob.target_flow)
        end
    end

    return model
end

# Register the variant
register_variant(
    :network_flow,
    :standard,
    NetworkFlowProblem,
    "Single-commodity flow LP over a directed acyclic network: maximize source-to-sink flow or route a contracted volume at minimum cost, with feasibility certified by an exact max-flow computation and a min-cut infeasibility certificate",
)
