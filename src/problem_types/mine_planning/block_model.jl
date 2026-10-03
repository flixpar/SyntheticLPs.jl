using JuMP
using Random
using Distributions

# ---------------------------------------------------------------------------
# Shared machinery for the mine_planning category: block-model generation,
# precedence arcs, economics, the planted-schedule witness, and the
# precedence-closure flow bound behind every infeasibility certificate.
# ---------------------------------------------------------------------------

"""
Largest `target_variables` accepted by every `mine_planning` variant. Each
schedule variable carries 5-10 precedence/chain rows (the 1-5 or 1-9 slope
pattern times the period count), so beyond a million variables the JuMP model
would exceed ten million rows; larger requests raise an `ArgumentError` instead
of being silently undersized (same convention as `network_flow/standard`).
"""
const MINE_PLANNING_MAX_VARIABLES = 1_000_000

"""
    MineEconomics

Commodity price and processing/mining economics shared by an instance (copper
porphyry units: grades in % Cu, tonnages in kt, money in k\$).

  - `price`: net metal price, \$ per tonne of contained copper
  - `mill_recovery_sulfide`, `mill_recovery_oxide`: flotation-mill recovery by
    rock type (oxide copper floats poorly)
  - `mill_cost`: milling cost, \$ per tonne of feed
  - `leach_recovery`: heap-leach recovery of oxide copper (sulfides do not leach)
  - `leach_cost`: heap-leach cost, \$ per tonne stacked
  - `mining_cost_surface`: mining cost of a bench-1 tonne, \$/t
  - `mining_cost_per_bench`: haulage increment per bench of depth, \$/t
  - `discount_rate`: per-period discount rate
"""
struct MineEconomics
    price::Float64
    mill_recovery_sulfide::Float64
    mill_recovery_oxide::Float64
    mill_cost::Float64
    leach_recovery::Float64
    leach_cost::Float64
    mining_cost_surface::Float64
    mining_cost_per_bench::Float64
    discount_rate::Float64
end

"""
    _mine_economics(rng) -> MineEconomics

Draw copper-porphyry economics in realistic ranges: price 6,000-10,000 \$/t Cu,
flotation recovery 85-92% (sulfide) / 45-60% (oxide), milling 7-13 \$/t, heap
leach 55-75% recovery at 2.5-5 \$/t, mining 1.4-2.6 \$/t at surface plus
2-5 cents per bench of haul depth, and an 6-12% discount rate.
"""
function _mine_economics(rng::AbstractRNG)
    return MineEconomics(
        rand(rng, Uniform(6000.0, 10000.0)),
        rand(rng, Uniform(0.85, 0.92)),
        rand(rng, Uniform(0.45, 0.60)),
        rand(rng, Uniform(7.0, 13.0)),
        rand(rng, Uniform(0.55, 0.75)),
        rand(rng, Uniform(2.5, 5.0)),
        rand(rng, Uniform(1.4, 2.6)),
        rand(rng, Uniform(0.02, 0.05)),
        rand(rng, Uniform(0.06, 0.12)),
    )
end

"""
Mill recovery of block `b` given its rock type.
"""
_mine_mill_recovery(econ::MineEconomics, oxide::Bool) =
    oxide ? econ.mill_recovery_oxide : econ.mill_recovery_sulfide

"""
Breakeven milling grade (% Cu) for a rock type: revenue per tonne
`grade/100 * recovery * price` equals the milling cost.
"""
_mine_mill_cutoff(econ::MineEconomics, oxide::Bool) =
    100.0 * econ.mill_cost / (econ.price * _mine_mill_recovery(econ, oxide))

"""
Breakeven heap-leach grade (% Cu) for oxide rock.
"""
_mine_leach_cutoff(econ::MineEconomics) = 100.0 * econ.leach_cost / (econ.price * econ.leach_recovery)

"""
    MineBlockModel

A precedence-closed open-pit block model. Blocks are indexed in a topological
order of the precedence DAG (every predecessor has a smaller index than its
successors), which is also the order of nested pit shells used by the planted
schedules: mining any prefix `1:k` of the blocks is a precedence-feasible pit.

# Fields

  - `nx`, `ny`, `nz`: grid dimensions (bench `z = 1` is the surface)
  - `block_size`, `bench_height`: block footprint and height, metres
  - `pattern`: slope pattern, `:five` (block needs the 5 blocks above it — the
    one directly above and its 4 edge neighbours) or `:nine` (the 3 x 3 above)
  - `i`, `j`, `z`: grid coordinates of every block
  - `tonnage`: block tonnage, kt (block volume times rock density)
  - `grade`: copper grade, % Cu
  - `oxide`: true above the oxidation surface (oxide rock), false for sulfide
  - `contaminant`: arsenic, ppm (concentrate penalty element)
  - `mining_cost`: cost of mining the whole block, k\$ (deeper benches haul
    farther; oxide is softer)
  - `arc_succ`, `arc_pred`: precedence arcs, sorted by successor: block
    `arc_succ[k]` cannot be mined before `arc_pred[k]` (`arc_pred[k] < arc_succ[k]`,
    and the predecessor always lies one bench higher, so the arc set is
    transitively reduced)
"""
struct MineBlockModel
    nx::Int
    ny::Int
    nz::Int
    block_size::Float64
    bench_height::Float64
    pattern::Symbol
    i::Vector{Int}
    j::Vector{Int}
    z::Vector{Int}
    tonnage::Vector{Float64}
    grade::Vector{Float64}
    oxide::Vector{Bool}
    contaminant::Vector{Float64}
    mining_cost::Vector{Float64}
    arc_succ::Vector{Int}
    arc_pred::Vector{Int}
end

Base.length(bm::MineBlockModel) = length(bm.tonnage)

"""
Horizontal offsets of the predecessors one bench up for a slope pattern.
"""
function _mine_pattern_offsets(pattern::Symbol)
    pattern == :five && return ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    pattern == :nine && return Tuple((di, dj) for di in -1:1 for dj in -1:1)
    throw(ArgumentError("unknown slope pattern $pattern"))
end

"""
    _mine_horizon(rng, n) -> Int

Number of scheduling periods (years) for a `target_variables` of `n`: grows
logarithmically with size (about 3 at n = 50, 6-7 at 1k, 9 at 10k, 11-12 at
100k, 14 at 1M) with a +-1 jitter, clamped to `3:24`. Larger mines are
scheduled over longer lives, as in the MineLib instances.
"""
function _mine_horizon(rng::AbstractRNG, n::Int)
    base = 2.5 * log10(max(n, 10)) - 1.0
    return clamp(round(Int, base + rand(rng, -1:1)), 3, 24)
end

"""
    _mine_block_model(rng, n_blocks, econ) -> MineBlockModel

Generate a precedence-closed pit of exactly `n_blocks` blocks (at least 4).

Construction:

 1. A grid of about `3 * n_blocks` candidate blocks, with `nz ~ n_blocks^(1/3)`
    benches and an elongated footprint.
 2. One to four copper ore bodies: anisotropic, rotated 3D Gaussian kernels
    buried under an overburden of barren rock (centres at 35-75% of the grid
    depth). The raw grade field is the kernel sum plus a low background, times
    a lognormal nugget.
 3. A pit envelope `f(i, j) = max_k (H_k - s_k * ||R_k (i - cx_k, j - cy_k)||)`:
    cones centred on the ore bodies with overall wall slope `s_k <= 0.68`, so
    `f` changes by less than 1 between any two horizontally adjacent (including
    diagonal) columns.
 4. Every candidate gets the score `f(i, j) - z`; the pit is the `n_blocks`
    highest-scoring candidates, ties broken toward shallower benches. Because
    `f` is 1-Lipschitz across neighbouring columns, every in-grid predecessor of
    a block (one bench up, horizontally adjacent) scores at least as high and
    is shallower, so the selected set is precedence-closed for both slope
    patterns, and the score order is a topological order whose prefixes are
    nested pit shells (pushbacks).
 5. Grades are calibrated so 15-35% of the pit is above the sulfide milling
    cutoff of `econ`; rock above a smooth oxidation surface is oxide; arsenic
    is concentrated around one ore body (an enargite zone); density, tonnage,
    and depth-dependent mining cost follow.
"""
function _mine_block_model(rng::AbstractRNG, n_blocks::Int, econ::MineEconomics)
    K = max(n_blocks, 4)

    # --- Grid -------------------------------------------------------------
    nz = clamp(round(Int, 0.95 * cbrt(K) * rand(rng, Uniform(0.85, 1.15))), 2, 60)
    area = 3.0 * K / nz
    aspect = rand(rng, Uniform(1.0, 1.6))
    nx = max(3, ceil(Int, sqrt(area * aspect)))
    ny = max(3, ceil(Int, area / nx))
    while nx * ny * nz < 2K
        ny += 1
    end
    block_size = rand(rng, (10.0, 12.5, 15.0, 20.0, 25.0))
    bench_height = rand(rng, (10.0, 12.0, 15.0))
    pattern = rand(rng) < 0.6 ? :five : :nine

    # --- Ore bodies -------------------------------------------------------
    n_bodies = rand(rng, 1:3) + (K > 20_000 ? 1 : 0)
    hmin = min(nx, ny)
    cx = [rand(rng, Uniform(0.3, 0.7)) * nx for _ in 1:n_bodies]
    cy = [rand(rng, Uniform(0.3, 0.7)) * ny for _ in 1:n_bodies]
    cz = [rand(rng, Uniform(0.35, 0.75)) * nz for _ in 1:n_bodies]
    ax = [max(0.8, rand(rng, Uniform(0.08, 0.22)) * hmin) for _ in 1:n_bodies]
    ay = [max(0.8, rand(rng, Uniform(0.08, 0.22)) * hmin) for _ in 1:n_bodies]
    az = [max(0.8, rand(rng, Uniform(0.12, 0.30)) * nz) for _ in 1:n_bodies]
    rot = [rand(rng, Uniform(0.0, pi)) for _ in 1:n_bodies]
    amp = [rand(rng, Uniform(0.6, 1.4)) for _ in 1:n_bodies]
    background = rand(rng, Uniform(0.02, 0.06))
    nugget = rand(rng, Uniform(0.2, 0.35))

    # --- Pit envelope (cones on the ore bodies) ----------------------------
    H = [cz[k] + az[k] + rand(rng, Uniform(0.0, 0.25)) * nz for k in 1:n_bodies]
    slope = [rand(rng, Uniform(0.45, 0.68)) for _ in 1:n_bodies]
    squash = [rand(rng, Uniform(0.65, 1.0)) for _ in 1:n_bodies]
    envelope = Matrix{Float64}(undef, nx, ny)
    for jj in 1:ny, ii in 1:nx
        best = -Inf
        for k in 1:n_bodies
            dx, dy = ii - cx[k], jj - cy[k]
            u = cos(rot[k]) * dx + sin(rot[k]) * dy
            v = squash[k] * (-sin(rot[k]) * dx + cos(rot[k]) * dy)
            best = max(best, H[k] - slope[k] * sqrt(u^2 + v^2))
        end
        envelope[ii, jj] = best
    end

    # --- Score order and pit selection --------------------------------------
    n_cand = nx * ny * nz
    cand_i = Vector{Int}(undef, n_cand)
    cand_j = Vector{Int}(undef, n_cand)
    cand_z = Vector{Int}(undef, n_cand)
    score = Vector{Float64}(undef, n_cand)
    c = 0
    for zz in 1:nz, jj in 1:ny, ii in 1:nx
        c += 1
        cand_i[c], cand_j[c], cand_z[c] = ii, jj, zz
        score[c] = envelope[ii, jj] - zz
    end
    # Highest score first; ties toward shallower benches, then by column.
    order = sortperm(1:n_cand; by=p -> (-score[p], cand_z[p], cand_j[p], cand_i[p]))
    pit = order[1:K]
    bi, bj, bz = cand_i[pit], cand_j[pit], cand_z[pit]

    position = zeros(Int, nx, ny, nz)
    for b in 1:K
        position[bi[b], bj[b], bz[b]] = b
    end

    # --- Precedence arcs (one bench up, slope pattern) ----------------------
    offsets = _mine_pattern_offsets(pattern)
    arc_succ = Int[]
    arc_pred = Int[]
    sizehint!(arc_succ, length(offsets) * K)
    sizehint!(arc_pred, length(offsets) * K)
    for b in 1:K
        bz[b] == 1 && continue
        for (di, dj) in offsets
            ii, jj = bi[b] + di, bj[b] + dj
            (1 <= ii <= nx && 1 <= jj <= ny) || continue
            a = position[ii, jj, bz[b] - 1]
            # Closure by construction: the predecessor scores at least as high
            # and is shallower, so it precedes `b` in the score order.
            0 < a < b || error("mine_planning: pit selection is not precedence-closed")
            push!(arc_succ, b)
            push!(arc_pred, a)
        end
    end

    # --- Grade field ---------------------------------------------------------
    raw = Vector{Float64}(undef, K)
    for b in 1:K
        s = background
        for k in 1:n_bodies
            dx, dy = bi[b] - cx[k], bj[b] - cy[k]
            u = cos(rot[k]) * dx + sin(rot[k]) * dy
            v = -sin(rot[k]) * dx + cos(rot[k]) * dy
            d2 = (u / ax[k])^2 + (v / ay[k])^2 + ((bz[b] - cz[k]) / az[k])^2
            s += amp[k] * exp(-0.5 * d2)
        end
        raw[b] = s * rand(rng, LogNormal(0.0, nugget))
    end
    ore_fraction = rand(rng, Uniform(0.15, 0.35))
    sulfide_cutoff = _mine_mill_cutoff(econ, false)
    threshold = sort(raw)[clamp(ceil(Int, (1 - ore_fraction) * K), 1, K)]
    alpha = rand(rng, Uniform(0.8, 1.2))
    grade = [min(sulfide_cutoff * (r / threshold)^alpha, 12.0 * sulfide_cutoff) for r in raw]

    # --- Oxidation surface ----------------------------------------------------
    ox_base = 0.5 + rand(rng, Uniform(0.08, 0.22)) * nz
    ox_amp = rand(rng, Uniform(0.5, 2.0))
    fx, fy = rand(rng, Uniform(0.5, 1.5)), rand(rng, Uniform(0.5, 1.5))
    phx, phy = rand(rng, Uniform(0.0, 2pi)), rand(rng, Uniform(0.0, 2pi))
    oxide = [
        bz[b] <= ox_base + ox_amp * sin(2pi * fx * bi[b] / nx + phx) * sin(2pi * fy * bj[b] / ny + phy)
        for b in 1:K
    ]

    # Guarantee at least one block worth milling (tiny pits may sit entirely in
    # the oxide cap, whose milling cutoff is higher).
    if !any(grade[b] > _mine_mill_cutoff(econ, oxide[b]) for b in 1:K)
        b = argmax(raw)
        grade[b] = 1.25 * _mine_mill_cutoff(econ, oxide[b])
    end

    # --- Arsenic (enargite zone around one body) ------------------------------
    as_body = rand(rng, 1:n_bodies)
    as_background = rand(rng, Uniform(30.0, 120.0))
    as_peak = rand(rng, Uniform(500.0, 2500.0))
    contaminant = Vector{Float64}(undef, K)
    for b in 1:K
        dx, dy = bi[b] - cx[as_body], bj[b] - cy[as_body]
        d2 = (dx / (1.3 * ax[as_body]))^2 + (dy / (1.3 * ay[as_body]))^2 +
             ((bz[b] - cz[as_body]) / az[as_body])^2
        contaminant[b] = as_background * rand(rng, LogNormal(0.0, 0.4)) + as_peak * exp(-0.5 * d2)
    end

    # --- Tonnage and mining cost ------------------------------------------------
    volume = block_size^2 * bench_height                     # m^3
    rho_oxide = rand(rng, Uniform(2.3, 2.5))
    rho_sulfide = rand(rng, Uniform(2.6, 2.8))
    tonnage = [
        volume * (oxide[b] ? rho_oxide : rho_sulfide) * rand(rng, Uniform(0.97, 1.03)) / 1000.0 for
        b in 1:K
    ]
    mining_cost = [
        tonnage[b] * (econ.mining_cost_surface + econ.mining_cost_per_bench * (bz[b] - 1)) *
        (oxide[b] ? 0.9 : 1.0) for b in 1:K
    ]

    return MineBlockModel(
        nx,
        ny,
        nz,
        block_size,
        bench_height,
        pattern,
        bi,
        bj,
        bz,
        tonnage,
        grade,
        oxide,
        contaminant,
        mining_cost,
        arc_succ,
        arc_pred,
    )
end

"""
    _mine_truncate(bm, K) -> MineBlockModel

The pit made of the first `K` blocks of `bm`. Blocks are in topological order,
so every prefix is precedence-closed, and arcs are sorted by successor, so the
retained arcs are a prefix of the arc list.
"""
function _mine_truncate(bm::MineBlockModel, K::Int)
    K == length(bm) && return bm
    m = searchsortedlast(bm.arc_succ, K)
    return MineBlockModel(
        bm.nx,
        bm.ny,
        bm.nz,
        bm.block_size,
        bm.bench_height,
        bm.pattern,
        bm.i[1:K],
        bm.j[1:K],
        bm.z[1:K],
        bm.tonnage[1:K],
        bm.grade[1:K],
        bm.oxide[1:K],
        bm.contaminant[1:K],
        bm.mining_cost[1:K],
        bm.arc_succ[1:m],
        bm.arc_pred[1:m],
    )
end

"""
    _mine_prefix_size(counts, T, offset, n) -> K

Smallest-error pit size: the `K >= 4` minimising `|T * sum(counts[1:K]) + offset - n|`
for per-block variable counts `counts` (1 + number of destination variables).
"""
function _mine_prefix_size(counts::Vector{Int}, T::Int, offset::Int, n::Int)
    best_K, best_err = 4, typemax(Int)
    total = 0
    for K in 1:length(counts)
        total += counts[K]
        err = abs(T * total + offset - n)
        if K >= 4 && err < best_err
            best_K, best_err = K, err
        end
        T * total + offset > n && K >= 4 && break
    end
    return best_K
end

"""
Discount factors `(1 + r)^-t`, `t = 1..T`.
"""
_mine_discount(r::Float64, T::Int) = [(1.0 + r)^(-t) for t in 1:T]

"""
    MinePlanWitness

Planted whole-block schedule — an integral feasible point of the built model,
so it certifies the MIP and its LP relaxation alike.

  - `mining_period[b]`: period in which block `b` is mined, `0` if never; mined
    blocks always form a prefix `1:k` of the topological block order
  - `destination[b]`: where the mined block goes (variant-specific code,
    `0` = waste dump)
  - `reclaim`, `inventory`: per stockpile bin and period (`stockpile` variant
    only; empty `0 x T` matrices otherwise), reclaimed tonnage and end-of-period
    inventory, kt
"""
struct MinePlanWitness
    mining_period::Vector{Int}
    destination::Vector{Int}
    reclaim::Matrix{Float64}
    inventory::Matrix{Float64}
end

"""
    MineClosureCertificate

Infeasibility certificate from LP rows alone (it survives `relax_integer`).
Writing `x_b = x[b, k]` for the cumulative extraction at the end of period `k`:

 1. Precedence rows at period `k` and the bounds `0 <= x <= 1` (via the chain
    rows) put `x` in the closure polytope of the precedence DAG, and summing
    the mining-capacity rows of periods `1..k` gives `sum_b w_b x_b <=
    mining_budget`.
 2. Summing the variant's mill rows over periods `1..k` (minimum feed times
    `feed_multiplier`, plus the head-grade rows in `:head_grade` mode) and
    using the linking/inventory rows yields `sum_b weights[b] x_b >= requirement`.
 3. For any `lambda >= 0`, `sum_b weights[b] x_b <= lambda * budget +
    max_{closed S} sum_{b in S} (weights[b] - lambda * w_b)`. The flow
    (`source_flow`, `arc_flow` along precedence arcs from successor to
    predecessor, `sink_flow`) is a feasible flow in the max-closure network
    with source capacities `max(c_b, 0)` and sink capacities `max(-c_b, 0)`,
    `c_b = weights[b] - lambda * w_b`, so every closed set satisfies
    `c(S) <= sum_{c_b > 0} (c_b - source_flow[b])`. Hence `bound =
    lambda * budget + sum_{c_b > 0} (c_b - source_flow[b])` caps the left side
    of 2, and `requirement > bound` (by a planted margin) is the contradiction.

`mode` is `:ramp_up` (a minimum mill feed the reachable ore cannot supply in
the first `periods` periods), `:exhaustion` (the same over the whole horizon:
the pit cannot supply the contracted feed at all), or `:head_grade` (minimum
mill feed at a head grade above what the reachable ore can sustain;
`weights[b] = w_b * max(grade_b - grade_threshold, 0)` on mill-eligible blocks
and `feed_multiplier = head_grade_min - grade_threshold`).
"""
struct MineClosureCertificate
    mode::Symbol
    periods::Int
    mining_budget::Float64
    grade_threshold::Float64
    feed_multiplier::Float64
    weights::Vector{Float64}
    lambda::Float64
    source_flow::Vector{Float64}
    sink_flow::Vector{Float64}
    arc_flow::Vector{Float64}
    bound::Float64
    requirement::Float64
end

"""
    _mine_closure_maxflow(n, succ, pred, c; max_phases=40)
        -> (flow_value, source_flow, sink_flow, arc_flow)

Flow in the max-closure network of the precedence DAG on `n` blocks: source ->
b with capacity `c_b` for `c_b > 0`, b -> sink with capacity `-c_b` for
`c_b < 0`, and an uncapacitated edge `succ[k] -> pred[k]` per precedence arc.
Dinic's algorithm (BFS level graph plus iterative blocking-flow DFS with
current-arc pointers); deterministic. By max-flow/min-cut, the maximum-weight
closure has value `sum_{c_b > 0} c_b - (max flow)`.

The phase count is capped at `max_phases`: the early, short-path phases carry
almost all of the flow, while the residual graph of a deep pit can need
hundreds of late phases that each move little. The returned flow is always
feasible (conserved and within capacities), which is all a certificate needs —
`sum_{c_b > 0} (c_b - source_flow_b)` bounds every closure from above, just
slightly less tightly when the cap binds.
"""
function _mine_closure_maxflow(
    n::Int, succ::Vector{Int}, pred::Vector{Int}, c::Vector{Float64}; max_phases::Int=40
)
    m = length(succ)
    s, t = n + 1, n + 2
    N = n + 2
    head = Int[]
    cap = Float64[]
    sizehint!(head, 2 * (m + n))
    sizehint!(cap, 2 * (m + n))
    adj = [Int[] for _ in 1:N]
    function add_edge!(u, v, cu)
        push!(head, v)
        push!(cap, cu)
        push!(adj[u], length(head))
        push!(head, u)
        push!(cap, 0.0)
        push!(adj[v], length(head))
        return nothing
    end
    for k in 1:m
        add_edge!(succ[k], pred[k], Inf)              # edges 2k-1 / 2k
    end
    src_edge = zeros(Int, n)
    snk_edge = zeros(Int, n)
    for b in 1:n
        if c[b] > 0
            add_edge!(s, b, c[b])
            src_edge[b] = length(head) - 1
        elseif c[b] < 0
            add_edge!(b, t, -c[b])
            snk_edge[b] = length(head) - 1
        end
    end
    rev(e) = isodd(e) ? e + 1 : e - 1

    tol = 1e-12 * max(1.0, maximum(abs, c; init=0.0))
    value = 0.0
    level = Vector{Int}(undef, N)
    next_arc = Vector{Int}(undef, N)
    queue = Vector{Int}(undef, N)
    phases = 0
    while phases < max_phases
        phases += 1
        fill!(level, -1)
        level[s] = 0
        queue[1] = s
        qh, qt = 1, 1
        while qh <= qt
            u = queue[qh]
            qh += 1
            # Nodes at or beyond the sink's level cannot lie on a shortest path.
            level[t] >= 0 && level[u] >= level[t] && break
            for e in adj[u]
                v = head[e]
                if cap[e] > tol && level[v] < 0
                    level[v] = level[u] + 1
                    qt += 1
                    queue[qt] = v
                end
            end
        end
        level[t] < 0 && break

        fill!(next_arc, 1)
        path_nodes = [s]
        path_edges = Int[]
        while true
            if path_nodes[end] == t
                bottleneck = minimum(cap[e] for e in path_edges)
                value += bottleneck
                for e in path_edges
                    cap[e] -= bottleneck
                    cap[rev(e)] += bottleneck
                end
                saturated = findfirst(e -> cap[e] <= tol, path_edges)
                resize!(path_edges, saturated - 1)
                resize!(path_nodes, saturated)
            else
                u = path_nodes[end]
                advanced = false
                while next_arc[u] <= length(adj[u])
                    e = adj[u][next_arc[u]]
                    v = head[e]
                    if cap[e] > tol && level[v] == level[u] + 1
                        push!(path_edges, e)
                        push!(path_nodes, v)
                        advanced = true
                        break
                    end
                    next_arc[u] += 1
                end
                if !advanced
                    level[u] = -1
                    pop!(path_nodes)
                    isempty(path_nodes) && break
                    pop!(path_edges)
                    next_arc[path_nodes[end]] += 1
                end
            end
        end
    end

    # Flows are read off the reverse residuals (forward capacities may be Inf).
    arc_flow = [cap[2k] for k in 1:m]
    source_flow = [src_edge[b] > 0 ? cap[src_edge[b] + 1] : 0.0 for b in 1:n]
    sink_flow = [snk_edge[b] > 0 ? cap[snk_edge[b] + 1] : 0.0 for b in 1:n]
    return value, source_flow, sink_flow, arc_flow
end

"""
    _mine_closure_bound_at(bm, weights, budget, lambda) -> (bound, source_flow, sink_flow, arc_flow)

The Lagrangian closure bound for one `lambda` (see [`MineClosureCertificate`](@ref)).
"""
function _mine_closure_bound_at(bm::MineBlockModel, weights::Vector{Float64}, budget::Float64, lambda::Float64)
    n = length(bm)
    cvec = weights .- lambda .* bm.tonnage
    _, src, snk, arcf = _mine_closure_maxflow(n, bm.arc_succ, bm.arc_pred, cvec)
    bound = lambda * budget + sum((cvec[b] - src[b] for b in 1:n if cvec[b] > 0); init=0.0)
    return bound, src, snk, arcf
end

"""
    _mine_closure_bound(bm, weights, budget; iterations=14)
        -> (bound, lambda, source_flow, sink_flow, arc_flow)

Upper bound on `max { weights' x : tonnage' x <= budget, x in the closure
polytope }`. The bound `lambda * budget + maxclosure(weights - lambda * tonnage)`
is convex in `lambda` and exact at its minimiser (LP duality over the
integral closure polytope), so a golden-section search over
`lambda in [0, max weights/tonnage]` returns a nearly tight bound; any
`lambda` it reports, together with the stored flow, is a valid certificate.
Pits above 20,000 blocks use 8 golden-section steps instead of 14 (each step
is one max-flow).
"""
function _mine_closure_bound(
    bm::MineBlockModel,
    weights::Vector{Float64},
    budget::Float64;
    iterations::Int=(length(bm) > 20_000 ? 8 : 14),
)
    lam_max = maximum(weights[b] / bm.tonnage[b] for b in 1:length(bm))
    best = (Inf, 0.0, Float64[], Float64[], Float64[])
    function evaluate(lam)
        bound, src, snk, arcf = _mine_closure_bound_at(bm, weights, budget, lam)
        if bound < best[1]
            best = (bound, lam, src, snk, arcf)
        end
        return bound
    end
    evaluate(0.0)
    lam_max <= 0 && return best
    golden = (sqrt(5.0) - 1.0) / 2.0
    a, b = 0.0, lam_max
    x1, x2 = b - golden * (b - a), a + golden * (b - a)
    f1, f2 = evaluate(x1), evaluate(x2)
    for _ in 1:iterations
        if f1 <= f2
            b, x2, f2 = x2, x1, f1
            x1 = b - golden * (b - a)
            f1 = evaluate(x1)
        else
            a, x1, f1 = x1, x2, f2
            x2 = a + golden * (b - a)
            f2 = evaluate(x2)
        end
    end
    return best
end

"""
    _mine_rebudget(closure, old_budget, new_budget)

The same certificate flow re-priced for another mining budget: at a fixed
`lambda` the bound is affine in the budget, so it moves by
`lambda * (new_budget - old_budget)` with no new max-flow.
"""
function _mine_rebudget(closure, old_budget::Float64, new_budget::Float64)
    bound, lam, src, snk, arcf = closure
    return (bound + lam * (new_budget - old_budget), lam, src, snk, arcf)
end

"""
    _mine_certificate(mode, k, budget, weights, threshold, multiplier, closure, requirement)

Build a [`MineClosureCertificate`](@ref) for periods `1..k` from a
`closure = (bound, lambda, source_flow, sink_flow, arc_flow)` tuple, erroring
if the requirement does not exceed the bound.
"""
function _mine_certificate(
    mode::Symbol,
    k::Int,
    budget::Float64,
    weights::Vector{Float64},
    threshold::Float64,
    multiplier::Float64,
    closure,
    requirement::Float64,
)
    bound, lam, src, snk, arcf = closure
    requirement > bound || error("mine_planning: certificate requirement does not exceed its bound")
    return MineClosureCertificate(mode, k, budget, threshold, multiplier, weights, lam, src, snk, arcf, bound, requirement)
end

"""
    _mine_add_chain_and_precedence!(model, x, bm, T)

Add the cumulative-extraction rows shared by every variant:
`x[b, t-1] <= x[b, t]` (a block mined by period `t-1` stays mined) and
`x[b, t] <= x[a, t]` for every precedence arc `b -> a` and period `t`.
"""
function _mine_add_chain_and_precedence!(model::Model, x, bm::MineBlockModel, T::Int)
    B = length(bm)
    for t in 2:T, b in 1:B
        @constraint(model, x[b, t - 1] - x[b, t] <= 0)
    end
    for t in 1:T, k in eachindex(bm.arc_succ)
        @constraint(model, x[bm.arc_succ[k], t] - x[bm.arc_pred[k], t] <= 0)
    end
    return nothing
end

"""
    _mine_period_tonnage(x, coeffs, blocks, t)

Affine expression `sum_b coeffs[b] * (x[b, t] - x[b, t-1])` over `blocks`
(with `x[b, 0] = 0`): the tonnage extracted in period `t`.
"""
function _mine_period_tonnage(x, coeffs::Vector{Float64}, blocks, t::Int)
    expr = AffExpr(0.0)
    for b in blocks
        add_to_expression!(expr, coeffs[b], x[b, t])
        t > 1 && add_to_expression!(expr, -coeffs[b], x[b, t - 1])
    end
    return expr
end

"""
    _mine_add_feed_row!(model, expr, lower, upper)

Add `lower <= expr <= upper` (a ranged row) or just `expr <= upper` when the
lower requirement is zero (it is then implied by the chain rows).
"""
function _mine_add_feed_row!(model::Model, expr, lower::Float64, upper::Float64)
    if lower > 0
        return @constraint(model, lower <= expr <= upper)
    else
        return @constraint(model, expr <= upper)
    end
end

"""
Coefficient of `x[b, t]` for a per-block undiscounted cash flow `v` realised in
the period the block is mined: `v * (delta_t - delta_{t+1})` with
`delta_{T+1} = 0` (the "by" telescoping of `sum_t delta_t (x[b,t] - x[b,t-1])`).
"""
_mine_by_coefficient(v::Float64, delta::Vector{Float64}, t::Int) =
    v * (delta[t] - (t < length(delta) ? delta[t + 1] : 0.0))

"""
    _mine_ramp_up!(rng, mining_capacity, feed_capacity)

Start-up profile shared by every profile and variant: the fleet ramps up over
the first two periods (period 1 at 35-100% of steady-state capacity, period 2
at 70-100%) and the mill is commissioned during period 1 (60-100%).
"""
function _mine_ramp_up!(rng::AbstractRNG, mining_capacity::Vector{Float64}, feed_capacity::Vector{Float64})
    T = length(mining_capacity)
    mining_capacity[1] *= rand(rng, Uniform(0.35, 1.0))
    T >= 2 && (mining_capacity[2] *= rand(rng, Uniform(0.7, 1.0)))
    feed_capacity[1] *= rand(rng, Uniform(0.6, 1.0))
    return nothing
end

"""
    _mine_feed_infeasibility!(rng, bm, weights, mining_capacity, feed_capacity,
                              min_feed, margin) -> MineClosureCertificate

Make the mill-feed contract provably unachievable, mutating the capacity and
feed vectors in place.

`:ramp_up` (tried first, for start-up horizons `k in 1:min(3, T-1)` in random
order): the minimum feed of periods `1..k` is raised to 80-95% of the mill
capacity, and if the closure bound on the mill-eligible tonnage (`weights`)
reachable with the start-up fleet still exceeds that requirement divided by
`1 + margin`, the start-up mining capacities of periods `1..k` are scaled down
(never below 30% of the steady-state capacity, the low end of the natural
ramp-up) until it does not: the contract demands full feed before the fleet
can strip enough overburden. Scaling uses the certificate's own linearity in
the budget at the `lambda` optimised for the unscaled fleet, so one
`lambda` search per horizon suffices. If the fleet floor
is reached first, the mill (all periods) and the contracted feed are enlarged
together by up to 1.6x instead — an overbuilt plant.

`:exhaustion` (drawn directly 20% of the time, and the fallback when no
start-up horizon qualifies): over the whole horizon, the contracted feed exceeds
`1 + margin` times everything the pit can supply within its total mining
capacity, enlarging the mill where needed so the per-period feed stays at 90%
of capacity.

Every per-period minimum stays at or below 95% of its mill capacity, so no
single row is contradictory.
"""
function _mine_feed_infeasibility!(
    rng::AbstractRNG,
    bm::MineBlockModel,
    weights::Vector{Float64},
    mining_capacity::Vector{Float64},
    feed_capacity::Vector{Float64},
    min_feed::Vector{Float64},
    margin::Float64,
)
    T = length(mining_capacity)
    horizons = rand(rng) < 0.2 ? Int[] : shuffle(rng, collect(1:min(3, T - 1)))
    for k in horizons
        theta_hi = rand(rng, Uniform(0.8, 0.95))
        level = [max(min_feed[t], theta_hi * feed_capacity[t]) for t in 1:k]
        requirement = sum(level)
        target = requirement / (1 + margin)
        W0 = sum(mining_capacity[1:k])
        # Smallest admissible scale: no start-up period below 30% of the
        # steady-state fleet (the natural ramp-up draws go down to 35%).
        min_scale = 0.3 * maximum(mining_capacity) / minimum(mining_capacity[1:k])
        closure = _mine_closure_bound(bm, weights, W0)
        bound, lam = closure[1], closure[2]
        offset = bound - lam * W0          # budget-independent part of the bound
        scale = if bound <= target
            1.0
        elseif lam > 0 && offset < target
            0.97 * (target - offset) / lam / W0
        else
            -Inf
        end
        enlarge = 1.0
        if scale < min_scale
            # Fleet floor reached: the contract is for an overbuilt mill instead
            # (feed and mill capacity scaled up together, at most 1.6x).
            scale = min_scale
            enlarge = max(1.0, (1 + margin) * (offset + lam * scale * W0) / requirement)
            enlarge <= 1.6 || continue
        end
        mining_capacity[1:k] .*= scale
        feed_capacity .*= enlarge
        min_feed[1:k] .= enlarge .* level
        closure = _mine_rebudget(closure, W0, scale * W0)
        return _mine_certificate(:ramp_up, k, scale * W0, weights, 0.0, 1.0, closure, enlarge * requirement)
    end
    budget = sum(mining_capacity)
    closure = _mine_closure_bound(bm, weights, budget)
    level = max((1 + margin) * closure[1] / T, 0.6 * minimum(feed_capacity))
    for t in 1:T
        feed_capacity[t] = max(feed_capacity[t], level / 0.9)
        min_feed[t] = max(min_feed[t], level)
    end
    return _mine_certificate(:exhaustion, T, budget, weights, 0.0, 1.0, closure, sum(min_feed))
end

"""
    _mine_feed_contract(rng, feed_capacity, reserve) -> Vector{Float64}

Natural minimum mill feed per period (an offtake contract): over a window
starting in period 1 or 2 and ending in period `T - 1` or `T`, the contract
spreads 60-125% of the estimated mill-eligible `reserve` evenly, capped at
50-90% of the mill capacity of each period; zero outside the window. Contracts
sized off the reserve estimate are the natural two-sided risk: an optimistic
estimate exhausts the pit before the contract ends, and a start-up contract
can outrun pre-stripping.
"""
function _mine_feed_contract(rng::AbstractRNG, feed_capacity::Vector{Float64}, reserve::Float64)
    T = length(feed_capacity)
    t_start = rand(rng, 1:min(2, T))
    t_end = T - rand(rng, 0:1)
    phi = rand(rng, Uniform(0.6, 1.25))
    cap_frac = rand(rng, Uniform(0.5, 0.9))
    rate = phi * reserve / (t_end - t_start + 1)
    return [t_start <= t <= t_end ? min(rate, cap_frac * feed_capacity[t]) : 0.0 for t in 1:T]
end

"""
    _mine_unknown_startup!(rng, bm, weights, mining_capacity, feed_capacity, min_feed)

`unknown` profile: the start-up contract (periods `1..k0`, `k0 in 1:2`) is
drawn at 60-120% of the closure bound on the mill-eligible tonnage reachable
with the start-up fleet, per period, capped at 95% of the mill capacity. The
bound is exact for the cumulative relaxation (per-period capacities only make
the schedule harder), so draws above it are infeasible and draws below it
usually but not always feasible — a genuine two-sided boundary, like
`network_flow`'s contract at 60-140% of the exact max flow. No claim is stored.
"""
function _mine_unknown_startup!(
    rng::AbstractRNG,
    bm::MineBlockModel,
    weights::Vector{Float64},
    mining_capacity::Vector{Float64},
    feed_capacity::Vector{Float64},
    min_feed::Vector{Float64},
)
    T = length(mining_capacity)
    k0 = rand(rng, 1:min(2, T - 1))
    phi = rand(rng, Uniform(0.6, 1.2))
    bound = _mine_closure_bound(bm, weights, sum(mining_capacity[1:k0]))[1]
    for t in 1:k0
        min_feed[t] = min(phi * bound / k0, 0.95 * feed_capacity[t])
    end
    return nothing
end
