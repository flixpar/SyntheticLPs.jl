# Shared geographic-network machinery for the flow family (`network_flow`,
# `transportation`, `multi_commodity_flow`, `load_balancing`, `assignment`).
#
# Everything here is plain functions over plain vectors (no new types), so any
# category can call it regardless of include order: Julia resolves the calls at
# run time. Every helper that draws randomness takes the `rng` first; the rest
# are deterministic.
#
# The pieces:
#   - `_geo_positions`      node coordinates + heavy-tailed activity weights in
#                           one of three geography shapes;
#   - `_geo_knn`            grid-accelerated k-nearest-neighbour lists (near
#                           linear, so 250k-node networks build in well under a
#                           second);
#   - `_geo_network`        a sparse, strongly connected road/pipeline-like
#                           digraph with EXACTLY the requested number of arcs: a
#                           geometric spanning tree (Kruskal on the kNN graph,
#                           components joined by trunk links) in both directions
#                           plus short-biased extra links;
#   - `_geo_adjacency`, `_geo_dijkstra`, `_geo_tree_flows` shortest-path trees
#                           and the arc flows of routing demand along them
#                           (optionally with multiplicative arc gains);
#   - `_flow_max_flow`      exact Dinic max flow + min cut.

using Random
using Distributions

"""
    _geo_positions(rng, n, shape; span=100.0) -> (positions, weights)

Scatter `n` nodes over a `span` x `span` region and give each a heavy-tailed
activity weight (population, production, traffic generation). Shapes:

  - `:uniform`   – even coverage (rural grids, mesh distribution systems);
  - `:clustered` – metropolitan clusters with Zipf-like sizes and spreads that
    grow with the cluster's share; each cluster's anchor node (its "city
    centre") carries a 5-20x weight boost;
  - `:corridor`  – a bent band across the region (river valley, coastal strip,
    interstate corridor).

Weights are lognormal (`sigma = 0.9`) for every shape, so a few nodes dominate
supply/demand the way real cities and plants do. Coordinates are clamped to the
region.
"""
function _geo_positions(rng::AbstractRNG, n::Int, shape::Symbol; span::Float64=100.0)
    n >= 1 || throw(ArgumentError("need at least one node (got $n)"))
    weights = [rand(rng, LogNormal(0.0, 0.9)) for _ in 1:n]
    clampc(v) = clamp(v, 0.0, span)
    if shape == :uniform
        positions = [(span * rand(rng), span * rand(rng)) for _ in 1:n]
    elseif shape == :corridor
        lateral = rand(rng, Uniform(0.3span, 0.7span))
        slope = rand(rng, Uniform(-0.5, 0.5))
        bend = rand(rng, Uniform(-0.15span, 0.15span))
        width = rand(rng, Uniform(0.05span, 0.12span))
        positions = Vector{Tuple{Float64, Float64}}(undef, n)
        for i in 1:n
            u = span * rand(rng)
            center = lateral + slope * (u - span / 2) + bend * sin(pi * u / span)
            positions[i] = (u, clampc(center + width * randn(rng)))
        end
    elseif shape == :clustered
        n_clusters = clamp(round(Int, n^0.4), 2, 60)
        n_clusters = min(n_clusters, n)
        centers = _geo_centers(rng, n_clusters, 0.6span / sqrt(n_clusters); span=span)
        # Zipf-like cluster sizes (rank-size rule), shuffled over the centers.
        share = [1.0 / c^0.9 for c in 1:n_clusters]
        shuffle!(rng, share)
        share ./= sum(share)
        spread = [span * (0.025 + 0.12 * sqrt(s)) for s in share]
        cum = cumsum(share)
        positions = Vector{Tuple{Float64, Float64}}(undef, n)
        for i in 1:n
            if i <= n_clusters
                # One anchor per cluster sits on its centre: the city hub.
                positions[i] = centers[i]
                weights[i] *= rand(rng, Uniform(5.0, 20.0))
            else
                g = min(searchsortedfirst(cum, rand(rng)), n_clusters)
                positions[i] = (
                    clampc(centers[g][1] + spread[g] * randn(rng)),
                    clampc(centers[g][2] + spread[g] * randn(rng)),
                )
            end
        end
        # Anchors are the first nodes; shuffle jointly so node indices carry no
        # geography.
        perm = randperm(rng, n)
        positions = positions[perm]
        weights = weights[perm]
    else
        throw(ArgumentError("unknown geography shape $shape"))
    end
    return positions, weights
end

"""
    _geo_centers(rng, q, min_sep; span=100.0, tries=400)

Rejection-sample `q` points in `[0, span]^2` pairwise at least `min_sep`
apart, relaxing the separation geometrically when it cannot be met.
"""
function _geo_centers(rng::AbstractRNG, q::Int, min_sep::Float64; span::Float64=100.0, tries::Int=400)
    centers = Tuple{Float64, Float64}[]
    sep = min_sep
    margin = 0.05span
    while length(centers) < q
        p = (margin + (span - 2margin) * rand(rng), margin + (span - 2margin) * rand(rng))
        if all(hypot(p[1] - c[1], p[2] - c[2]) >= sep for c in centers)
            push!(centers, p)
        elseif rand(rng) < 1.0 / tries
            sep = max(sep * 0.95, 1e-6 * span)
        end
    end
    return centers
end

_geo_dist(positions, i::Int, j::Int) =
    hypot(positions[i][1] - positions[j][1], positions[i][2] - positions[j][2])

"""
    _geo_knn(positions, k) -> Vector{Vector{Int}}

The `k` nearest other nodes of every node (ascending distance; ties broken by
index), found with a uniform bucket grid of about two points per cell and ring
search, so the cost is near-linear in the number of nodes. Deterministic.
"""
function _geo_knn(positions::Vector{Tuple{Float64, Float64}}, k::Int)
    n = length(positions)
    k = min(k, n - 1)
    k <= 0 && return [Int[] for _ in 1:n]
    xmin = minimum(p[1] for p in positions)
    ymin = minimum(p[2] for p in positions)
    xmax = maximum(p[1] for p in positions)
    ymax = maximum(p[2] for p in positions)
    extent = max(xmax - xmin, ymax - ymin, 1e-9)
    G = max(1, floor(Int, sqrt(n / 2)))
    h = extent / G * (1 + 1e-9)
    cell(p) = (
        clamp(floor(Int, (p[1] - xmin) / h) + 1, 1, G),
        clamp(floor(Int, (p[2] - ymin) / h) + 1, 1, G),
    )
    buckets = [Int[] for _ in 1:G, _ in 1:G]
    for i in 1:n
        cx, cy = cell(positions[i])
        push!(buckets[cx, cy], i)
    end

    result = Vector{Vector{Int}}(undef, n)
    best_d = Float64[]
    best_i = Int[]
    for i in 1:n
        empty!(best_d)
        empty!(best_i)
        cx, cy = cell(positions[i])
        r = 0
        while true
            for gx in (cx - r):(cx + r), gy in (cy - r):(cy + r)
                (1 <= gx <= G && 1 <= gy <= G) || continue
                max(abs(gx - cx), abs(gy - cy)) == r || continue
                for j in buckets[gx, gy]
                    j == i && continue
                    d = _geo_dist(positions, i, j)
                    if length(best_d) < k || (d, j) < (best_d[end], best_i[end])
                        # Sorted insertion into the bounded candidate list.
                        pos = length(best_d) + 1
                        while pos > 1 && (d, j) < (best_d[pos - 1], best_i[pos - 1])
                            pos -= 1
                        end
                        insert!(best_d, pos, d)
                        insert!(best_i, pos, j)
                        if length(best_d) > k
                            pop!(best_d)
                            pop!(best_i)
                        end
                    end
                end
            end
            # Every point outside rings 0..r lies at least r*h away.
            if (length(best_d) == k && best_d[end] <= r * h) || r > G
                break
            end
            r += 1
        end
        result[i] = copy(best_i)
    end
    return result
end

"""
    _geo_knn_query(ref_positions, query_positions, k) -> Vector{Vector{Int}}

For every query point, the indices of its `k` nearest REFERENCE points
(ascending distance, ties by index) — e.g. the nearest plants of each customer.
Same bucket-grid ring search as `_geo_knn`, built over the reference set.
Deterministic.
"""
function _geo_knn_query(
    ref_positions::Vector{Tuple{Float64, Float64}},
    query_positions::Vector{Tuple{Float64, Float64}},
    k::Int,
)
    nr = length(ref_positions)
    k = min(k, nr)
    k <= 0 && return [Int[] for _ in query_positions]
    xs = vcat([p[1] for p in ref_positions], [p[1] for p in query_positions])
    ys = vcat([p[2] for p in ref_positions], [p[2] for p in query_positions])
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    extent = max(xmax - xmin, ymax - ymin, 1e-9)
    G = max(1, floor(Int, sqrt(nr / 2)))
    h = extent / G * (1 + 1e-9)
    cell(p) = (
        clamp(floor(Int, (p[1] - xmin) / h) + 1, 1, G),
        clamp(floor(Int, (p[2] - ymin) / h) + 1, 1, G),
    )
    buckets = [Int[] for _ in 1:G, _ in 1:G]
    for i in 1:nr
        cx, cy = cell(ref_positions[i])
        push!(buckets[cx, cy], i)
    end
    result = Vector{Vector{Int}}(undef, length(query_positions))
    best_d = Float64[]
    best_i = Int[]
    for (q, qp) in enumerate(query_positions)
        empty!(best_d)
        empty!(best_i)
        cx, cy = cell(qp)
        r = 0
        while true
            for gx in (cx - r):(cx + r), gy in (cy - r):(cy + r)
                (1 <= gx <= G && 1 <= gy <= G) || continue
                max(abs(gx - cx), abs(gy - cy)) == r || continue
                for j in buckets[gx, gy]
                    d = hypot(qp[1] - ref_positions[j][1], qp[2] - ref_positions[j][2])
                    if length(best_d) < k || (d, j) < (best_d[end], best_i[end])
                        pos = length(best_d) + 1
                        while pos > 1 && (d, j) < (best_d[pos - 1], best_i[pos - 1])
                            pos -= 1
                        end
                        insert!(best_d, pos, d)
                        insert!(best_i, pos, j)
                        if length(best_d) > k
                            pop!(best_d)
                            pop!(best_i)
                        end
                    end
                end
            end
            if (length(best_d) == k && best_d[end] <= r * h) || r > G
                break
            end
            r += 1
        end
        result[q] = copy(best_i)
    end
    return result
end

# Union-find with path halving.
function _geo_find!(parent::Vector{Int}, i::Int)
    while parent[i] != i
        parent[i] = parent[parent[i]]
        i = parent[i]
    end
    return i
end

"""
    _geo_spanning_edges(positions, knn) -> Vector{Tuple{Int,Int}}

A geometric spanning tree: Kruskal over the kNN candidate edges by length
(an approximate Euclidean MST), then — if the kNN graph is disconnected, e.g.
well-separated metropolitan clusters — the components are joined along a Prim
MST over their representative nodes, each link realised between the closest
pair found by scanning the two components (inter-regional trunk lines).
Returns `n - 1` undirected edges `(i, j)`, `i < j`. Deterministic.
"""
function _geo_spanning_edges(positions::Vector{Tuple{Float64, Float64}}, knn::Vector{Vector{Int}})
    n = length(positions)
    cand = Tuple{Float64, Int, Int}[]
    for i in 1:n, j in knn[i]
        i < j ? push!(cand, (_geo_dist(positions, i, j), i, j)) :
        (i ∉ knn[j] && push!(cand, (_geo_dist(positions, i, j), j, i)))
    end
    sort!(cand)
    parent = collect(1:n)
    tree = Tuple{Int, Int}[]
    for (_, i, j) in cand
        ri, rj = _geo_find!(parent, i), _geo_find!(parent, j)
        ri == rj && continue
        parent[ri] = rj
        push!(tree, (i, j))
    end
    length(tree) == n - 1 && return tree

    # Join the remaining components.
    roots = [_geo_find!(parent, i) for i in 1:n]
    comp_ids = unique(roots)
    C = length(comp_ids)
    index = Dict(r => c for (c, r) in enumerate(comp_ids))
    members = [Int[] for _ in 1:C]
    for i in 1:n
        push!(members[index[roots[i]]], i)
    end
    reps = Vector{Int}(undef, C)
    for c in 1:C
        mx = sum(positions[i][1] for i in members[c]) / length(members[c])
        my = sum(positions[i][2] for i in members[c]) / length(members[c])
        reps[c] = argmin(i -> (hypot(positions[i][1] - mx, positions[i][2] - my), i), members[c])
    end
    # Prim over component representatives.
    in_tree = falses(C)
    best = fill(Inf, C)
    link = zeros(Int, C)
    in_tree[1] = true
    for c in 2:C
        best[c] = _geo_dist(positions, reps[1], reps[c])
        link[c] = 1
    end
    for _ in 2:C
        c = 0
        bd = Inf
        for q in 1:C
            if !in_tree[q] && best[q] < bd
                bd = best[q]
                c = q
            end
        end
        in_tree[c] = true
        # Realise the link between components `link[c]` and `c`.
        p = link[c]
        a = argmin(i -> (_geo_dist(positions, i, reps[c]), i), members[p])
        b = argmin(j -> (_geo_dist(positions, a, j), j), members[c])
        push!(tree, (min(a, b), max(a, b)))
        for q in 1:C
            if !in_tree[q]
                d = _geo_dist(positions, reps[c], reps[q])
                if d < best[q]
                    best[q] = d
                    link[q] = c
                end
            end
        end
    end
    return tree
end

"""
    _geo_network(rng, positions, n_arcs; k_cand=8, bidirectional=0.85)
        -> (arcs, trunk)

A sparse, geographically embedded digraph on `length(positions)` nodes with
EXACTLY `n_arcs` distinct directed arcs (no self-loops), sorted
lexicographically:

  1. a geometric spanning tree (`_geo_spanning_edges`) in both directions, so
     the network is strongly connected (`trunk[k] = true` on these arcs);
  2. extra local links drawn from the `k_cand`-nearest-neighbour candidate
     edges in order of noisy length (short links first, lognormal noise so the
     pattern is not purely metric), each two-way with probability
     `bidirectional` and one-way otherwise — the last odd unit of budget is a
     one-way arc;
  3. only if the candidate pool is exhausted, random long-range express links.

Requires `2 * (n - 1) <= n_arcs <= n * (n - 1)`; callers size `n` from the
arc budget (typically 3-5 arcs per node, like road and pipeline networks).
"""
function _geo_network(
    rng::AbstractRNG,
    positions::Vector{Tuple{Float64, Float64}},
    n_arcs::Int;
    k_cand::Int=8,
    bidirectional::Float64=0.85,
)
    n = length(positions)
    n >= 2 || throw(ArgumentError("a network needs at least 2 nodes (got $n)"))
    2 * (n - 1) <= n_arcs <= n * (n - 1) || throw(
        ArgumentError("n_arcs=$n_arcs outside [$(2 * (n - 1)), $(n * (n - 1))] for $n nodes")
    )
    knn = _geo_knn(positions, k_cand)
    tree = _geo_spanning_edges(positions, knn)

    present = Set{Tuple{Int, Int}}()
    arcs = Tuple{Int, Int}[]
    trunk_set = Set{Tuple{Int, Int}}()
    for (i, j) in tree
        for a in ((i, j), (j, i))
            push!(present, a)
            push!(arcs, a)
            push!(trunk_set, a)
        end
    end

    # Candidate extra links: kNN edges not in the tree, short-first with noise.
    undirected = Set{Tuple{Int, Int}}(tree)
    pool = Tuple{Float64, Int, Int}[]
    for i in 1:n, j in knn[i]
        e = (min(i, j), max(i, j))
        e in undirected && continue
        push!(undirected, e)
        push!(pool, (_geo_dist(positions, e[1], e[2]) * exp(0.6 * randn(rng)), e[1], e[2]))
    end
    sort!(pool)
    for (_, i, j) in pool
        remaining = n_arcs - length(arcs)
        remaining <= 0 && break
        if remaining >= 2 && rand(rng) < bidirectional
            for a in ((i, j), (j, i))
                push!(present, a)
                push!(arcs, a)
            end
        else
            a = rand(rng) < 0.5 ? (i, j) : (j, i)
            push!(present, a)
            push!(arcs, a)
        end
    end
    # Express links if the local pool ran dry (only on very dense requests).
    while length(arcs) < n_arcs
        i, j = rand(rng, 1:n), rand(rng, 1:n)
        (i == j || (i, j) in present) && continue
        push!(present, (i, j))
        push!(arcs, (i, j))
    end

    sort!(arcs)
    trunk = [a in trunk_set for a in arcs]
    return arcs, trunk
end

"""
    _geo_adjacency(n, arcs) -> (out_adj, in_adj)

Per-node lists of outgoing and incoming arc indices, built in one pass.
"""
function _geo_adjacency(n::Int, arcs::Vector{Tuple{Int, Int}})
    out_adj = [Int[] for _ in 1:n]
    in_adj = [Int[] for _ in 1:n]
    for (k, (u, v)) in enumerate(arcs)
        push!(out_adj[u], k)
        push!(in_adj[v], k)
    end
    return out_adj, in_adj
end

# Minimal binary min-heap of (key, node) pairs for Dijkstra (lazy deletion).
function _geo_heap_push!(heap::Vector{Tuple{Float64, Int}}, item::Tuple{Float64, Int})
    push!(heap, item)
    i = length(heap)
    while i > 1
        p = i >> 1
        heap[p] <= heap[i] && break
        heap[p], heap[i] = heap[i], heap[p]
        i = p
    end
    return heap
end

function _geo_heap_pop!(heap::Vector{Tuple{Float64, Int}})
    top = heap[1]
    last = pop!(heap)
    if !isempty(heap)
        heap[1] = last
        i = 1
        len = length(heap)
        while true
            l = 2i
            l > len && break
            c = (l + 1 <= len && heap[l + 1] < heap[l]) ? l + 1 : l
            heap[c] < heap[i] || break
            heap[c], heap[i] = heap[i], heap[c]
            i = c
        end
    end
    return top
end

"""
    _geo_dijkstra(n, arcs, out_adj, lengths, sources) -> (dist, pred)

Multi-source Dijkstra over nonnegative arc `lengths`: `dist[v]` is the shortest
distance from the nearest source (`Inf` if unreachable) and `pred[v]` the arc
index entering `v` on a shortest path (`0` at sources and unreached nodes).
Ties are broken deterministically (by node index through the heap order).
"""
function _geo_dijkstra(
    n::Int,
    arcs::Vector{Tuple{Int, Int}},
    out_adj::Vector{Vector{Int}},
    lengths::AbstractVector{<:Real},
    sources::AbstractVector{Int},
)
    dist = fill(Inf, n)
    pred = zeros(Int, n)
    done = falses(n)
    heap = Tuple{Float64, Int}[]
    for s in sources
        dist[s] = 0.0
        _geo_heap_push!(heap, (0.0, s))
    end
    while !isempty(heap)
        d, u = _geo_heap_pop!(heap)
        done[u] && continue
        done[u] = true
        for k in out_adj[u]
            v = arcs[k][2]
            nd = d + lengths[k]
            if nd < dist[v]
                dist[v] = nd
                pred[v] = k
                _geo_heap_push!(heap, (nd, v))
            end
        end
    end
    return dist, pred
end

"""
    _geo_tree_flows(n, arcs, dist, pred, demand; gains=nothing) -> Vector{Float64}

Arc flows of routing `demand[v]` to every node `v` along the shortest-path tree
`(dist, pred)` returned by `_geo_dijkstra` (rooted at its sources). Nodes are
processed farthest-first, so each tree arc carries its whole subtree's demand.
With `gains` (multiplicative arc gains in `(0, 1]`), the flow SENT on an arc is
what must arrive at its head divided by the gain, and that sent amount is what
the tail must receive — the lossy (generalized-flow) routing. Unreached nodes
must have zero demand. Deterministic.
"""
function _geo_tree_flows(
    n::Int,
    arcs::Vector{Tuple{Int, Int}},
    dist::Vector{Float64},
    pred::Vector{Int},
    demand::AbstractVector{<:Real};
    gains::Union{Nothing, AbstractVector{<:Real}}=nothing,
)
    flows = zeros(Float64, length(arcs))
    acc = Float64.(demand)
    order = sortperm(dist; rev=true)
    for v in order
        if !isfinite(dist[v])
            acc[v] == 0 || error("demand at unreachable node $v")
            continue
        end
        k = pred[v]
        k == 0 && continue
        sent = gains === nothing ? acc[v] : acc[v] / gains[k]
        flows[k] += sent
        acc[arcs[k][1]] += sent
    end
    return flows
end

"""
    _flow_max_flow(n_nodes, source, sink, arcs, capacities)
        -> (value, arc_flows, source_side, cut_arcs, cut_capacity)

Exact maximum `source`-`sink` flow (Dinic's algorithm: BFS level graphs plus an
iterative blocking-flow DFS with current-arc pointers), the achieving per-arc
flow assignment, and a minimum cut read off the residual graph's
source-reachable set. Residual capacities are compared with a `1e-9`
tolerance so floating-point residues of saturated edges are never mistaken for
usable slack. Works on any digraph (cycles and antiparallel arcs included).
Deterministic — no RNG.
"""
function _flow_max_flow(
    n_nodes::Int, source::Int, sink::Int, arcs::Vector{Tuple{Int, Int}}, capacities::Vector{Float64}
)
    m = length(arcs)
    n = n_nodes
    tol = 1e-9

    # CSR residual graph: arc k contributes forward edge 2k-1 (capacity) and
    # reverse edge 2k (initially zero); an edge's partner is `e ± 1`.
    deg = zeros(Int, n)
    for (u, v) in arcs
        deg[u] += 1
        deg[v] += 1
    end
    first = Vector{Int}(undef, n + 1)
    first[1] = 1
    for v in 1:n
        first[v + 1] = first[v] + deg[v]
    end
    fillpos = first[1:n]
    adj = Vector{Int}(undef, 2m)
    edge_head = Vector{Int}(undef, 2m)
    edge_cap = Vector{Float64}(undef, 2m)
    for k in 1:m
        u, v = arcs[k]
        f, r = 2k - 1, 2k
        edge_head[f], edge_cap[f] = v, capacities[k]
        edge_head[r], edge_cap[r] = u, 0.0
        adj[fillpos[u]] = f
        fillpos[u] += 1
        adj[fillpos[v]] = r
        fillpos[v] += 1
    end
    partner(e) = isodd(e) ? e + 1 : e - 1

    value = 0.0
    level = Vector{Int}(undef, n)
    next_arc = Vector{Int}(undef, n)
    queue = Vector{Int}(undef, n)
    path_nodes = Int[]
    path_edges = Int[]
    while true
        # BFS level graph over edges with usable residual capacity; nodes at or
        # beyond the sink's level are never expanded (they cannot be on a
        # shortest augmenting path).
        fill!(level, -1)
        level[source] = 0
        qh, qt = 1, 1
        queue[1] = source
        while qh <= qt
            u = queue[qh]
            qh += 1
            level[sink] >= 0 && level[u] >= level[sink] && break
            for i in first[u]:(first[u + 1] - 1)
                e = adj[i]
                v = edge_head[e]
                if level[v] < 0 && edge_cap[e] > tol
                    level[v] = level[u] + 1
                    qt += 1
                    queue[qt] = v
                end
            end
        end
        level[sink] < 0 && break

        # Iterative blocking-flow DFS with current-arc pointers.
        for v in 1:n
            next_arc[v] = first[v]
        end
        empty!(path_nodes)
        empty!(path_edges)
        push!(path_nodes, source)
        while true
            if path_nodes[end] == sink
                # Augment along the whole path by its bottleneck ...
                bottleneck = Inf
                for e in path_edges
                    bottleneck = min(bottleneck, edge_cap[e])
                end
                value += bottleneck
                for e in path_edges
                    edge_cap[e] -= bottleneck
                    edge_cap[partner(e)] += bottleneck
                end
                # ... then retreat to just before the first saturated edge.
                saturated = findfirst(e -> edge_cap[e] <= tol, path_edges)
                resize!(path_edges, saturated - 1)
                resize!(path_nodes, saturated)
            else
                u = path_nodes[end]
                advanced = false
                while next_arc[u] < first[u + 1]
                    e = adj[next_arc[u]]
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
                    pop!(path_edges)
                    next_arc[path_nodes[end]] += 1
                end
            end
        end
    end

    # Minimum cut: nodes still reachable from the source in the residual graph.
    side = falses(n)
    side[source] = true
    stack = [source]
    while !isempty(stack)
        u = pop!(stack)
        for i in first[u]:(first[u + 1] - 1)
            e = adj[i]
            v = edge_head[e]
            if edge_cap[e] > tol && !side[v]
                side[v] = true
                push!(stack, v)
            end
        end
    end

    # Net flow per arc (antiparallel pairs are separate arcs, so this is exact).
    arc_flows = [max(capacities[k] - edge_cap[2k - 1], 0.0) for k in 1:m]
    cut_arcs = [k for k in 1:m if side[arcs[k][1]] && !side[arcs[k][2]]]
    cut_capacity = sum(capacities[k] for k in cut_arcs; init=0.0)
    return value, arc_flows, findall(side), cut_arcs, cut_capacity
end
