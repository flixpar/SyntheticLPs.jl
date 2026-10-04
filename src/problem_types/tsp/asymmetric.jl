using JuMP
using Random

"""
    TSPAsymmetricProblem <: ProblemGenerator

Generator for the **sparse asymmetric travelling-salesman problem** (ATSP) — a
large urban courier route over a hilly street network with one-way streets —
formulated with lifted Miller–Tucker–Zemlin (MTZ) constraints over a sparse
candidate-arc graph.

# Overview

The dense variants (`standard`, `flow`, `precedence`, …) price every ordered
pair of a few hundred stops. Real large-scale routing instead works on a
**candidate-arc graph**: each stop keeps only the few onward legs a dispatcher
would ever consider (its cheapest directed connections to nearby stops). This
variant generates that structure directly, so at a given variable budget it has
roughly `target / (m + 1)` stops (≈ 10,000 at 100k variables, versus ≈ 316 for
`standard`), very sparse degree rows, and an MTZ block whose big-M equals the
large stop count — a genuinely different LP from `standard` rather than the
same dense model with another cost matrix.

Travel times are direction-dependent for physical reasons:

  - **Elevation**: a smooth terrain of a few hills; climbing costs extra time
    (`κ` minutes per metre of ascent), descending does not.
  - **One-way streets**: about a quarter of neighbouring stop pairs are joined by
    a one-way street; driving against it means a detour around the block
    (`1.3–2.0×` the direct time).

Candidate arcs: every stop keeps its `m` cheapest outgoing legs (`m ∈ 6:10`,
by directed travel time) among its `2m` geometrically nearest neighbours, so
the support itself is asymmetric. Stops left with fewer than two incoming
candidates receive arcs from their nearest neighbours, so no degree row is a
singleton.

The formulation is the lifted MTZ model restricted to candidate arcs:

  - Binary `x[a]` per candidate arc, continuous order `u[j] ∈ [1, n-1]` per stop.
  - Degree rows (one in-arc and one out-arc per node).
  - `u_i − u_j + (n−1)x_ij + (n−3)x_ji ≤ n−2` when both directions are
    candidates, `u_i − u_j + (n−1)x_ij ≤ n−2` otherwise (stop-to-stop arcs).

This is a MIP whose continuous relaxation is a sparse tour relaxation: a useful
LP test instance, but a fractional `x` is not an implementable tour.

# Fields

  - `n_stops::Int`: node count `n` (node 1 = home base / depot)
  - `out_degree::Int`: candidate out-arcs kept per stop (`m`)
  - `locations::Vector{Tuple{Float64,Float64}}`: coordinates (km)
  - `elevation::Vector{Float64}`: terrain height at each node (m)
  - `arcs::Vector{Tuple{Int,Int}}`: candidate arcs `(i, j)`, sorted, no loops
  - `travel_time::Vector{Float64}`: directed travel time (min) per arc
  - `planted_tour::Vector{Int}`: the planted `[1, …, 1]` tour whose arcs are all
    candidates (empty for `unknown`)
  - `blocked_set::Vector{Int}`: Hall-deficit district `S` (empty unless infeasible)
  - `gate_set::Vector{Int}`: gateway stops `T`, `|T| = |S| − 1` (empty unless infeasible)
"""
struct TSPAsymmetricProblem <: ProblemGenerator
    n_stops::Int
    out_degree::Int
    locations::Vector{Tuple{Float64, Float64}}
    elevation::Vector{Float64}
    arcs::Vector{Tuple{Int, Int}}
    travel_time::Vector{Float64}
    planted_tour::Vector{Int}
    blocked_set::Vector{Int}
    gate_set::Vector{Int}
end

# `K` nearest neighbours of every point (excluding itself), by uniform-grid
# bucketing with expanding rings: O(n·K) expected for the clustered stop
# layouts used here, instead of the O(n²) all-pairs scan.
function _tsp_nearest_neighbors(locations::Vector{Tuple{Float64, Float64}}, K::Int)
    n = length(locations)
    K = min(K, n - 1)
    xs = first.(locations)
    ys = last.(locations)
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    side = max(1, floor(Int, sqrt(n / 2)))
    cell = span / side * (1 + 1e-9)
    cell_of(i) = (
        clamp(floor(Int, (xs[i] - xmin) / cell) + 1, 1, side),
        clamp(floor(Int, (ys[i] - ymin) / cell) + 1, 1, side),
    )
    buckets = [Int[] for _ in 1:side, _ in 1:side]
    for i in 1:n
        cx, cy = cell_of(i)
        push!(buckets[cx, cy], i)
    end
    neighbors = Vector{Vector{Int}}(undef, n)
    cand = Int[]
    for i in 1:n
        cx, cy = cell_of(i)
        empty!(cand)
        ring = 0
        while true
            for gx in (cx - ring):(cx + ring), gy in (cy - ring):(cy + ring)
                (max(abs(gx - cx), abs(gy - cy)) == ring) || continue
                (1 <= gx <= side && 1 <= gy <= side) || continue
                for j in buckets[gx, gy]
                    j != i && push!(cand, j)
                end
            end
            # Every point within `ring * cell` of i lies in rings 0..ring, so
            # once K candidates are no farther than that, the search is exact.
            if length(cand) >= K
                d2(j) = (xs[j] - xs[i])^2 + (ys[j] - ys[i])^2
                partialsort!(cand, K; by=j -> (d2(j), j))
                d2(cand[K]) <= (ring * cell)^2 && break
            end
            ring > 2 * side && break
            ring += 1
        end
        d2b(j) = (xs[j] - xs[i])^2 + (ys[j] - ys[i])^2
        sort!(cand; by=j -> (d2b(j), j))
        neighbors[i] = cand[1:min(K, length(cand))]
    end
    return neighbors
end

# Order points along a Hilbert curve (resolution 2^10): consecutive points are
# spatially close, so the order is a plausible, short-legged planted tour.
function _tsp_hilbert_order(locations::Vector{Tuple{Float64, Float64}}, idx::Vector{Int})
    xs = [locations[i][1] for i in idx]
    ys = [locations[i][2] for i in idx]
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    order_bits = 10
    side = 2^order_bits
    function hilbert_d(x::Int, y::Int)
        d = 0
        s = side ÷ 2
        while s > 0
            rx = (x & s) > 0 ? 1 : 0
            ry = (y & s) > 0 ? 1 : 0
            d += s * s * ((3 * rx) ⊻ ry)
            if ry == 0
                if rx == 1
                    x = s - 1 - x
                    y = s - 1 - y
                end
                x, y = y, x
            end
            s ÷= 2
        end
        return d
    end
    hkey = [
        hilbert_d(
            clamp(floor(Int, (xs[t] - xmin) / span * (side - 1)), 0, side - 1),
            clamp(floor(Int, (ys[t] - ymin) / span * (side - 1)), 0, side - 1),
        ) for t in eachindex(idx)
    ]
    return idx[sortperm(collect(zip(hkey, idx)))]
end

# Make the candidate graph admit a perfect "successor assignment" (a
# bipartite matching of every node's out-side to a distinct in-side), the
# condition for the degree rows — and hence the LP relaxation's assignment
# core — to be feasible. A sparse nearest-neighbour graph occasionally has a
# small Hall violation (a few stops whose in-arcs all come from fewer stops),
# which makes an `unknown` instance trivially infeasible. Greedy matching plus
# BFS augmenting paths finds a maximum matching; each tail left unmatched is
# then joined to the nearest unmatched head. Deterministic (sorted adjacency).
function _tsp_repair_assignment!(
    arcset::Set{Tuple{Int, Int}}, n::Int, xs::Vector{Float64}, ys::Vector{Float64}
)
    adj = [Int[] for _ in 1:n]
    for (i, j) in sort!(collect(arcset))
        push!(adj[i], j)
    end
    match_tail = zeros(Int, n)   # tail -> head
    match_head = zeros(Int, n)   # head -> tail
    for i in 1:n, j in adj[i]
        if match_head[j] == 0
            match_tail[i] = j
            match_head[j] = i
            break
        end
    end
    parent_tail = zeros(Int, n)   # head -> tail that reached it in the BFS
    queue = Int[]
    seen = falses(n)
    for root in 1:n
        match_tail[root] == 0 || continue
        fill!(seen, false)
        empty!(queue)
        push!(queue, root)
        free_head = 0
        qi = 1
        while qi <= length(queue) && free_head == 0
            u = queue[qi]
            qi += 1
            for j in adj[u]
                seen[j] && continue
                seen[j] = true
                parent_tail[j] = u
                if match_head[j] == 0
                    free_head = j
                    break
                end
                push!(queue, match_head[j])
            end
        end
        free_head == 0 && continue
        j = free_head
        while j != 0
            u = parent_tail[j]
            previous = match_tail[u]
            match_tail[u] = j
            match_head[j] = u
            j = previous
        end
    end
    open_heads = [j for j in 1:n if match_head[j] == 0]
    for i in 1:n
        match_tail[i] == 0 || continue
        candidates = [j for j in open_heads if j != i]
        if isempty(candidates)
            # Only i's own head is free: splice i into a matched pair a -> b.
            a = findfirst(t -> t != i && match_tail[t] != 0 && match_tail[t] != i, 1:n)
            b = match_tail[a]
            push!(arcset, (i, b), (a, i))
            match_tail[i], match_head[b] = b, i
            match_tail[a], match_head[i] = i, a
        else
            j = candidates[argmin([((xs[j] - xs[i])^2 + (ys[j] - ys[i])^2, j) for j in candidates])]
            push!(arcset, (i, j))
            match_tail[i], match_head[j] = j, i
        end
        filter!(h -> match_head[h] == 0, open_heads)
    end
    return arcset
end

"""
    TSPAsymmetricProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a sparse ATSP instance.

# Variable count

One binary per candidate arc plus one order variable per stop:

    total = |arcs| + (n - 1)

With `m` out-candidates per stop, `|arcs| ≈ m·n` (plus a few planted-tour and
in-degree repair arcs), so `n = round(target / (m + 1.15))`; the delivered
count lands within a few percent of the target. Tiny targets clamp to `n = 5`
with complete support.

# Feasibility

  - `feasible`: a Hilbert-curve tour through all stops (starting at the depot)
    is planted into the candidate set, so it is an integer witness; with
    `u_j` = visit position it satisfies every lifted MTZ row, and it survives
    relaxation verbatim.
  - `infeasible`: the planted tour is added as for `feasible`, then a
    Hall-deficit district (`S`: the `k ≈ 0.4–0.8·√n` stops nearest an anchor;
    `T`: the next `k−1` nearest, the gateways) loses every in-arc whose tail is
    not a gateway, and each district stop receives in-arcs from its three
    nearest gateways and out-arcs to its three nearest stops outside the
    district (its candidate legs mostly pointed inside it). The in-degree rows of `S` sum to `k` but draw only on the
    `k−1` unit out-degrees of `T` — infeasible from the degree rows alone, so
    also in the LP relaxation; the deficit is spread over `2k−1` rows, which
    presolve does not aggregate.
  - `unknown`: the bare candidate graph (no planted tour), repaired so its
    degree rows admit a successor assignment (a maximum bipartite matching
    plus nearest-stop arcs for any stop left unmatched — otherwise a small Hall
    violation in the sparse graph occasionally made the instance trivially
    infeasible). Whether it contains a Hamiltonian cycle is not known, and the
    relaxation is genuinely two-sided: the sparse lifted-MTZ rows sometimes
    cut off every fractional assignment (observed in roughly a quarter of
    large instances), a refutation that takes thousands of simplex iterations.
"""
function TSPAsymmetricProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)

    # --- Dimension sizing ---
    m = rand(rng, 6:10)
    n = max(5, round(Int, target_variables / (m + 1.15)))
    m = min(m, n - 1)

    # --- Geography: clustered stops, smooth hilly terrain ---
    locations = _tsp_stops(rng, n)
    xs = first.(locations)
    ys = last.(locations)
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    span = max(xmax - xmin, ymax - ymin, 1.0)
    hills = [
        (
            xmin + span * rand(rng),
            ymin + span * rand(rng),
            20.0 + 60.0 * rand(rng),            # height (m)
            span * (0.08 + 0.2 * rand(rng)),     # width (km)
        ) for _ in 1:rand(rng, 2:4)
    ]
    elevation = [
        round(
            sum(
                H * exp(-((x - cx)^2 + (y - cy)^2) / (2 * w^2)) for (cx, cy, H, w) in hills
            );
            digits=1,
        ) for (x, y) in locations
    ]
    circuity = 1.2 + 0.2 * rand(rng)
    pace = 1.2 + 0.8 * rand(rng)               # min per km
    climb = 0.02 + 0.03 * rand(rng)            # min per metre of ascent
    one_way_share = 0.2 + 0.1 * rand(rng)

    # One-way streets are a property of the unordered pair: the pair (i, j)
    # with i < j is one-way with probability `one_way_share`, in the direction
    # `forward` (i -> j) or backward; driving against it costs a detour factor.
    # Pairs are evaluated lazily and memoised so every pair gets exactly one
    # draw, in a deterministic (sorted) order.
    K = min(2 * m, n - 1)
    neighbors = _tsp_nearest_neighbors(locations, K)
    pairs = Set{Tuple{Int, Int}}()
    for i in 1:n, j in neighbors[i]
        push!(pairs, minmax(i, j))
    end
    street = Dict{Tuple{Int, Int}, Tuple{Bool, Bool, Float64}}()  # (one_way, forward, detour)
    for pr in sort!(collect(pairs))
        street[pr] = (rand(rng) < one_way_share, rand(rng) < 0.5, 1.3 + 0.7 * rand(rng))
    end
    function leg_time(i::Int, j::Int)
        d = hypot(xs[i] - xs[j], ys[i] - ys[j])
        t = max(circuity * d, 0.05) * pace + climb * max(0.0, elevation[j] - elevation[i])
        pr = minmax(i, j)
        info = get(street, pr, (false, true, 1.0))
        if info[1]
            with_flow = (i < j) == info[2]
            with_flow || (t *= info[3])
        end
        return round(t; digits=2)
    end

    # --- Candidate arcs: m cheapest outgoing legs among the 2m nearest ---
    arcset = Set{Tuple{Int, Int}}()
    for i in 1:n
        cands = neighbors[i]
        order = sortperm([(leg_time(i, j), j) for j in cands])
        for t in order[1:min(m, length(order))]
            push!(arcset, (i, cands[t]))
        end
    end

    # --- Planted tour (feasible and infeasible requests) ---
    planted = Int[]
    if feasibility_status != unknown
        planted = vcat(1, _tsp_hilbert_order(locations, collect(2:n)), 1)
        for t in 2:length(planted)
            push!(arcset, (planted[t - 1], planted[t]))
        end
    end

    # --- In-degree repair: no stop with fewer than two incoming candidates ---
    indeg = zeros(Int, n)
    for (_, j) in arcset
        indeg[j] += 1
    end
    for j in 1:n
        for i in neighbors[j]
            indeg[j] >= 2 && break
            if !((i, j) in arcset)
                push!(arcset, (i, j))
                indeg[j] += 1
            end
        end
    end

    # --- Assignment repair (unknown): no trivial Hall violation ---
    feasibility_status == unknown && _tsp_repair_assignment!(arcset, n, xs, ys)

    # --- Hall-deficit district (infeasible) ---
    S = Int[]
    T = Int[]
    if feasibility_status == infeasible
        f = 0.4 + 0.4 * rand(rng)
        k = n < 8 ? 2 : clamp(round(Int, sqrt(n) * f), 3, (n - 1) ÷ 2)
        anchor = rand(rng, 2:n)
        ax, ay = locations[anchor]
        by_distance = sort(collect(2:n); by=j -> ((xs[j] - ax)^2 + (ys[j] - ay)^2, j))
        S = sort(by_distance[1:k])
        T = sort(by_distance[(k + 1):(2k - 1)])
        in_S = falses(n)
        in_S[S] .= true
        in_T = falses(n)
        in_T[T] .= true
        filter!(a -> !(in_S[a[2]] && !in_T[a[1]]), arcset)
        outside = [v for v in 1:n if !in_S[v]]
        for j in S
            near(v) = ((xs[v] - xs[j])^2 + (ys[v] - ys[j])^2, v)
            gates = sort(T; by=near)
            for t in gates[1:min(3, length(gates))]
                push!(arcset, (t, j))
            end
            # Most of an interior district stop's candidate legs pointed at
            # its district neighbours and were just deleted; give it legs out
            # to the three nearest stops outside the district so no out-degree
            # row is left empty or a singleton.
            for v in partialsort(outside, 1:min(3, length(outside)); by=near)
                push!(arcset, (j, v))
            end
        end
    end

    arcs = sort!(collect(arcset))
    travel_time = [leg_time(i, j) for (i, j) in arcs]
    return TSPAsymmetricProblem(n, m, locations, elevation, arcs, travel_time, planted, S, T)
end

"""
    build_model(prob::TSPAsymmetricProblem)

Build the sparse lifted-MTZ ATSP model over the candidate arcs. Deterministic —
uses only data from the struct fields.

  - `x[(i,j)] ∈ {0,1}` per candidate arc, `u[j] ∈ [1, n-1]` per stop `j = 2..n`
  - degree rows: one in-arc and one out-arc per node
  - lifted MTZ rows on stop-to-stop arcs (using the reverse arc when it is also
    a candidate)
"""
function build_model(prob::TSPAsymmetricProblem)
    model = Model()
    n = prob.n_stops
    arcs = prob.arcs
    stops = 2:n

    @variable(model, x[arcs], Bin)
    @variable(model, 1 <= u[j in stops] <= n - 1)
    @objective(model, Min, sum(prob.travel_time[a] * x[arcs[a]] for a in eachindex(arcs)))

    out_arcs = [Tuple{Int, Int}[] for _ in 1:n]
    in_arcs = [Tuple{Int, Int}[] for _ in 1:n]
    for a in arcs
        push!(out_arcs[a[1]], a)
        push!(in_arcs[a[2]], a)
    end
    for v in 1:n
        @constraint(model, sum(x[a] for a in in_arcs[v]; init=0.0) == 1)
        @constraint(model, sum(x[a] for a in out_arcs[v]; init=0.0) == 1)
    end

    arc_lookup = Set(arcs)
    for (i, j) in arcs
        (i == 1 || j == 1) && continue
        if (j, i) in arc_lookup
            @constraint(model, u[i] - u[j] + (n - 1) * x[(i, j)] + (n - 3) * x[(j, i)] <= n - 2)
        else
            @constraint(model, u[i] - u[j] + (n - 1) * x[(i, j)] <= n - 2)
        end
    end
    return model
end

# Register the variant (standard remains the category default).
register_variant(
    :tsp,
    :asymmetric,
    TSPAsymmetricProblem,
    "Sparse asymmetric TSP for large urban courier routes: candidate-arc graph over thousands of stops with one-way detours and uphill penalties, lifted MTZ subtour elimination";
    tags=[:routing, :big_m],
)
