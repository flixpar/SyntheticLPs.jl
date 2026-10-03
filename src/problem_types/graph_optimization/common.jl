using Random

# Shared graph machinery for the graph_optimization family.
#
# Every helper that draws randomness takes the caller's `rng` first, so call
# them from constructors only — never from `build_model`. All constructions
# are near-linear in the number of vertices and edges (grid bucketing instead
# of all-pairs scans), so the family builds 100k-variable instances in seconds.

# -----------------------------------------------------------------------------
# Geometric (unit-disk) graphs
# -----------------------------------------------------------------------------

"""
    _graph_geometric_points(rng, n, avg_degree; radius=1.0, hotspot_share=0.55)
        -> (xs, ys, side)

Scatter `n` sites over a square whose side makes a radius-`radius` disk hold
`avg_degree` other sites on average under uniform density
(`side = radius * sqrt(pi * n / avg_degree)`). A `hotspot_share` of the sites
concentrates in Gaussian hotspots (venues, towns, dense office floors) whose peak
density is a few times the background, so degrees and clique sizes are
heterogeneous, as in real interference and siting graphs.
"""
function _graph_geometric_points(
    rng::AbstractRNG, n::Int, avg_degree::Real; radius::Float64=1.0, hotspot_share::Float64=0.55
)
    side = radius * sqrt(pi * n / avg_degree)
    density = n / side^2
    n_hot = round(Int, hotspot_share * n)
    n_hotspots = max(1, round(Int, n / 350))
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    centers = [(side * rand(rng), side * rand(rng)) for _ in 1:n_hotspots]
    per_hotspot = max(1.0, n_hot / n_hotspots)
    for i in 1:n
        if i <= n_hot
            cx, cy = centers[rand(rng, 1:n_hotspots)]
            # Peak density `kappa` times the background: m / (2 pi sigma^2) = kappa * density.
            kappa = 2.0 + 3.0 * rand(rng)
            sigma = sqrt(per_hotspot / (2pi * kappa * density))
            xs[i] = clamp(cx + sigma * randn(rng), 0.0, side)
            ys[i] = clamp(cy + sigma * randn(rng), 0.0, side)
        else
            xs[i] = side * rand(rng)
            ys[i] = side * rand(rng)
        end
    end
    # Shuffle so hotspot membership is not encoded in the vertex index.
    perm = randperm(rng, n)
    return xs[perm], ys[perm], side
end

# Bucket point indices into square cells of side `cell`; buckets keep indices in
# ascending order, so every downstream traversal is deterministic.
function _graph_buckets(xs::Vector{Float64}, ys::Vector{Float64}, cell::Float64)
    buckets = Dict{Tuple{Int, Int}, Vector{Int}}()
    for i in eachindex(xs)
        push!(get!(buckets, (floor(Int, xs[i] / cell), floor(Int, ys[i] / cell)), Int[]), i)
    end
    return buckets
end

"""
    _graph_pairs_within(xs, ys, radius; min_radius=0.0) -> Vector{Tuple{Int,Int}}

Sorted pairs `(i, j)`, `i < j`, at Euclidean distance in `[min_radius, radius]`.
Grid bucketing keeps this `O(n + pairs)`.
"""
function _graph_pairs_within(
    xs::Vector{Float64}, ys::Vector{Float64}, radius::Float64; min_radius::Float64=0.0
)
    buckets = _graph_buckets(xs, ys, radius)
    r2 = radius^2
    lo2 = min_radius^2
    pairs = Tuple{Int, Int}[]
    for i in eachindex(xs)
        cx, cy = floor(Int, xs[i] / radius), floor(Int, ys[i] / radius)
        for dx in -1:1, dy in -1:1
            bucket = get(buckets, (cx + dx, cy + dy), nothing)
            bucket === nothing && continue
            for j in bucket
                j > i || continue
                d2 = (xs[i] - xs[j])^2 + (ys[i] - ys[j])^2
                (lo2 <= d2 <= r2) && push!(pairs, (i, j))
            end
        end
    end
    return sort!(pairs)
end

"""
    _graph_cell_groups(xs, ys, radius) -> Vector{Vector{Int}}

Partition the sites into square cells of side `radius / sqrt(2)`. Two sites in
the same cell are at most `radius` apart, so every cell is a clique of the
radius-`radius` unit-disk graph. Groups are returned in a deterministic order.
"""
function _graph_cell_groups(xs::Vector{Float64}, ys::Vector{Float64}, radius::Float64)
    buckets = _graph_buckets(xs, ys, radius / sqrt(2.0) * (1 - 1e-9))
    return sort!(collect(values(buckets)); by=first)
end

# -----------------------------------------------------------------------------
# Generic graph utilities
# -----------------------------------------------------------------------------

function _graph_adjacency(n::Int, edges::Vector{Tuple{Int, Int}})
    adj = [Int[] for _ in 1:n]
    for (u, v) in edges
        push!(adj[u], v)
        push!(adj[v], u)
    end
    foreach(sort!, adj)
    return adj
end

_graph_adjacent(adj::Vector{Vector{Int}}, u::Int, v::Int) = insorted(v, adj[u])

"""
    _graph_clique_cover(adj, seeds) -> Vector{Vector{Int}}

Greedy edge clique cover: a list of cliques such that every edge lies inside at
least one of them. Each `seed` (a known clique, e.g. a grid cell or a map
feature's candidate set) is grown to a maximal clique and emitted first; every
edge still uncovered afterwards starts a new clique grown from its endpoints'
common neighbours. Packing rows `sum(x[K]) <= 1` over these cliques imply every
edge row `x_u + x_v <= 1`, and give the much stronger *clique formulation* whose
LP relaxation is not the trivially half-integral edge relaxation.
"""
function _graph_clique_cover(adj::Vector{Vector{Int}}, seeds::Vector{Vector{Int}})
    covered = Set{Tuple{Int, Int}}()
    cliques = Vector{Vector{Int}}()

    function grow_and_emit!(clique::Vector{Int})
        # Candidates: common neighbours of the whole seed, ascending.
        candidates = copy(adj[clique[1]])
        for k in clique
            filter!(w -> _graph_adjacent(adj, k, w), candidates)
        end
        for w in candidates
            w in clique && continue
            all(k -> _graph_adjacent(adj, k, w), clique) && push!(clique, w)
        end
        sort!(clique)
        for a in 1:(length(clique) - 1), b in (a + 1):length(clique)
            push!(covered, (clique[a], clique[b]))
        end
        push!(cliques, clique)
        return nothing
    end

    for seed in seeds
        length(seed) >= 2 || continue
        # A seed whose pairs are all covered adds nothing new.
        fresh = any(
            !((seed[a], seed[b]) in covered) for a in 1:(length(seed) - 1) for
            b in (a + 1):length(seed)
        )
        fresh && grow_and_emit!(sort(seed))
    end
    for u in eachindex(adj), v in adj[u]
        v > u || continue
        (u, v) in covered && continue
        grow_and_emit!([u, v])
    end
    return cliques
end

"""
    _graph_clique_partition(n, cliques) -> (parts, part_rows)

Greedy partition of `1:n` into cliques drawn from `cliques` (largest first):
each part is a subset of the clique row `cliques[part_rows[p]]`, or a singleton
with `part_rows[p] == 0`. Summing the packing rows of the parts proves
`sum(x) <= length(parts)` for every LP-feasible `x` in `[0,1]^n`.
"""
function _graph_clique_partition(n::Int, cliques::Vector{Vector{Int}})
    assigned = falses(n)
    parts = Vector{Vector{Int}}()
    part_rows = Int[]
    order = sortperm(length.(cliques); rev=true)
    for r in order
        members = [v for v in cliques[r] if !assigned[v]]
        length(members) >= 2 || continue
        assigned[members] .= true
        push!(parts, members)
        push!(part_rows, r)
    end
    for v in 1:n
        assigned[v] && continue
        push!(parts, [v])
        push!(part_rows, 0)
    end
    return parts, part_rows
end

"""
    _graph_greedy_independent_set(adj, weights) -> Vector{Int}

Greedy weighted independent set: scan vertices by `weight / (degree + 1)`
descending and keep each one with no kept neighbour. Deterministic.
"""
function _graph_greedy_independent_set(adj::Vector{Vector{Int}}, weights::Vector{Float64})
    n = length(adj)
    order = sortperm([weights[v] / (length(adj[v]) + 1) for v in 1:n]; rev=true)
    blocked = falses(n)
    chosen = Int[]
    for v in order
        blocked[v] && continue
        push!(chosen, v)
        blocked[v] = true
        blocked[adj[v]] .= true
    end
    return sort!(chosen)
end

"""
    _graph_lognormal_weights(rng, n; median=50.0, sigma=0.6, digits=2)

Right-skewed positive weights (traffic demand, population, wind resource).
"""
function _graph_lognormal_weights(
    rng::AbstractRNG, n::Int; median::Float64=50.0, sigma::Float64=0.6, digits::Int=2
)
    return round.(median .* exp.(sigma .* randn(rng, n)); digits=digits)
end

"""
    _graph_floor_between(rng, low, high; reach=0.65) -> Int

A cardinality floor for `unknown` requests, drawn uniformly from the first
`reach` share of the gap between an integral lower reference (a greedy solution
that proves feasibility up to it) and an LP-valid upper bound (a certificate
that proves infeasibility above it). The LP optimum lies strictly inside that
gap (measured at 25–45% of it for the unit-disk families), so the request is
genuinely two-sided.
"""
function _graph_floor_between(rng::AbstractRNG, low::Int, high::Int; reach::Float64=0.65)
    high <= low && return low
    return low + round(Int, reach * rand(rng) * (high - low))
end

# -----------------------------------------------------------------------------
# Scale-free and community graphs
# -----------------------------------------------------------------------------

"""
    _graph_preferential_attachment(rng, n, m) -> Vector{Tuple{Int,Int}}

A connected scale-free graph with exactly `m` distinct edges on `n` vertices
(`n - 1 <= m <= n(n-1)/2`): a small seed clique, then each arriving vertex
attaches to existing vertices with probability proportional to degree (80%) or
uniformly (20%), as in router-level and peering topologies. Per-vertex
attachment counts are spread so the edge total is exact.
"""
function _graph_preferential_attachment(rng::AbstractRNG, n::Int, m::Int)
    (n - 1 <= m <= n * (n - 1) ÷ 2) ||
        throw(ArgumentError("preferential attachment needs n-1 <= m <= n(n-1)/2"))
    base = max(1, m ÷ max(1, n))
    s0 = min(n, base + 2)
    while s0 > 2 && s0 * (s0 - 1) ÷ 2 > m - (n - s0)
        s0 -= 1
    end
    edges = Set{Tuple{Int, Int}}()
    endpoints = Int[]
    for u in 1:(s0 - 1), v in (u + 1):s0
        push!(edges, (u, v))
        push!(endpoints, u)
        push!(endpoints, v)
    end
    remaining = m - length(edges)
    arrivals = n - s0
    for (idx, t) in enumerate((s0 + 1):n)
        # Spread the remaining edges evenly over the arrivals still to come;
        # every arrival keeps at least one edge so the graph stays connected.
        left = arrivals - idx + 1
        quota = clamp(round(Int, remaining / left + (rand(rng) - 0.5)), 1, t - 1)
        quota = min(quota, remaining - (left - 1))
        chosen = Int[]
        while length(chosen) < quota
            u = if !isempty(endpoints) && rand(rng) < 0.8
                endpoints[rand(rng, 1:length(endpoints))]
            else
                rand(rng, 1:(t - 1))
            end
            u in chosen || push!(chosen, u)
        end
        for u in chosen
            push!(edges, (u, t))
            push!(endpoints, u)  # (multi-item push! is O(n) on Julia 1.12)
            push!(endpoints, t)
        end
        remaining -= quota
    end
    # Top up any shortfall (tiny graphs) with uniform non-edges.
    while length(edges) < m
        u, v = rand(rng, 1:n), rand(rng, 1:n)
        u == v && continue
        push!(edges, minmax(u, v))
    end
    return sort!(collect(edges))
end

"""
    _graph_peeling(n, adj) -> (order, rank, core)

Batagelj–Zaversnik minimum-degree peeling in `O(n + m)`. `order` is the removal
order, `rank[v]` the position of `v` in it, and `core[v]` its core number.
Orienting every edge toward its earlier-removed endpoint (smaller `rank`) makes
each vertex the head of at most `maximum(core)` (the degeneracy) edges. The
last `k` vertices of `order` form the greedy-peeled dense `k`-subgraph.
"""
function _graph_peeling(n::Int, adj::Vector{Vector{Int}})
    deg = length.(adj)
    maxdeg = n == 0 ? 0 : maximum(deg)
    bins = zeros(Int, maxdeg + 1)
    for d in deg
        bins[d + 1] += 1
    end
    start = 1
    for d in 0:maxdeg
        count = bins[d + 1]
        bins[d + 1] = start
        start += count
    end
    pos = zeros(Int, n)
    vert = zeros(Int, n)
    for v in 1:n
        pos[v] = bins[deg[v] + 1]
        vert[pos[v]] = v
        bins[deg[v] + 1] += 1
    end
    for d in maxdeg:-1:1
        bins[d + 1] = bins[d]
    end
    bins[1] = 1
    removed = falses(n)
    for i in 1:n
        v = vert[i]
        removed[v] = true
        for u in adj[v]
            removed[u] && continue
            if deg[u] > deg[v]
                du = deg[u]
                pu = pos[u]
                pw = bins[du + 1]
                w = vert[pw]
                if u != w
                    pos[u] = pw
                    vert[pu] = w
                    pos[w] = pu
                    vert[pw] = u
                end
                bins[du + 1] += 1
                deg[u] -= 1
            end
        end
    end
    order = vert
    rank = zeros(Int, n)
    for (i, v) in enumerate(order)
        rank[v] = i
    end
    return order, rank, deg
end

"""
    _graph_community_edges(rng, n, m; planted=Int[], planted_edges=0)

Exactly `m` distinct edges of a social/interaction network on `n` vertices:
vertices belong to communities of heavy-tailed size, ~70% of edges fall inside
communities and the rest connect random vertices with degree-propensity
(Chung–Lu style) weighting, giving hubs. If `planted` is nonempty,
`planted_edges` distinct pairs inside it are placed first (a dense hidden
community).
"""
function _graph_community_edges(
    rng::AbstractRNG, n::Int, m::Int; planted::Vector{Int}=Int[], planted_edges::Int=0
)
    m <= n * (n - 1) ÷ 2 || throw(ArgumentError("too many edges requested"))
    edges = Set{Tuple{Int, Int}}()
    if !isempty(planted)
        pairs = [(planted[a], planted[b]) for a in 1:(length(planted) - 1) for
                 b in (a + 1):length(planted)]
        shuffle!(rng, pairs)
        for pair in pairs[1:planted_edges]
            push!(edges, minmax(pair...))
        end
    end
    # Communities: heavy-tailed sizes between 8 and 120 over a random order.
    order = randperm(rng, n)
    communities = Vector{Vector{Int}}()
    cursor = 1
    while cursor <= n
        size = clamp(round(Int, 8 * exp(1.2 * rand(rng) + 0.8 * abs(randn(rng)))), 8, 120)
        push!(communities, order[cursor:min(n, cursor + size - 1)])
        cursor += size
    end
    # Degree propensities (lognormal) drive both the community choice and the
    # cross-community endpoints.
    theta = exp.(0.8 .* randn(rng, n))
    cumulative = cumsum(theta)
    pick_weighted() = searchsortedfirst(cumulative, rand(rng) * cumulative[end])
    comm_weight = cumsum([Float64(length(c))^2 for c in communities])
    intra_share = 0.6 + 0.2 * rand(rng)
    attempts = 0
    while length(edges) < m
        attempts += 1
        if rand(rng) < intra_share && attempts < 50 * m
            c = communities[searchsortedfirst(comm_weight, rand(rng) * comm_weight[end])]
            length(c) >= 2 || continue
            u, v = c[rand(rng, 1:length(c))], c[rand(rng, 1:length(c))]
        else
            u, v = pick_weighted(), rand(rng) < 0.5 ? pick_weighted() : rand(rng, 1:n)
        end
        u == v && continue
        push!(edges, minmax(u, v))
    end
    return sort!(collect(edges))
end
