# tsp category
#
# Entry point for the `tsp` problem category. A category groups one or more
# variant formulations; the category is registered lazily from its first
# variant's `register_variant` call (or call `register_category` explicitly to
# give the category its own description). Add a variant by creating a file in
# this folder and including it below.

using Random

# Category-level description (it groups several formulations).
register_category(:tsp,
    "Travelling-salesman routing: symmetric and one-way-street tours, alternative LP formulations, time windows, prize collection, multiple salespersons, and precedence constraints")

# --- Shared data helpers -----------------------------------------------------
# Used by every tsp variant; names live in the module's namespace, hence the
# `_tsp_` prefix. Each randomised helper takes the caller's `rng` explicitly,
# so call them from constructors only — never from build_model.

# Sample `n` stop locations (index 1 = home base / depot): a depot near the
# centre of a scale-tiered service region, most stops clustered in a few towns
# (Gaussian spread), and ~20% uniform rural scatter between them.
function _tsp_stops(rng::AbstractRNG, n::Int)
    if n <= 9
        grid_size = rand(rng, 15.0:5.0:40.0)
        n_clusters = rand(rng, 1:2)
    elseif n <= 25
        grid_size = rand(rng, 30.0:10.0:80.0)
        n_clusters = rand(rng, 2:4)
    else
        grid_size = rand(rng, 60.0:20.0:150.0)
        n_clusters = rand(rng, 3:6)
    end
    depot = (grid_size * (0.4 + 0.2 * rand(rng)), grid_size * (0.4 + 0.2 * rand(rng)))
    cluster_centers = [(grid_size * rand(rng), grid_size * rand(rng)) for _ in 1:n_clusters]
    cluster_spread = grid_size / (4.0 * n_clusters)
    n_customers = n - 1
    n_rural = max(1, round(Int, 0.2 * n_customers))   # ~20% rural scatter
    n_town = n_customers - n_rural
    stops = Tuple{Float64,Float64}[]
    for _ in 1:n_town
        center = rand(rng, cluster_centers)
        x = clamp(center[1] + randn(rng) * cluster_spread, 0.0, grid_size)
        y = clamp(center[2] + randn(rng) * cluster_spread, 0.0, grid_size)
        push!(stops, (x, y))
    end
    for _ in 1:n_rural
        push!(stops, (grid_size * rand(rng), grid_size * rand(rng)))
    end
    return vcat([depot], stops)
end

# Symmetric road-distance matrix from stop locations: Euclidean distance times
# ONE per-instance road-circuity factor (roads are longer than straight lines).
# Keeping the factor per instance rather than per arc keeps `dist` a true metric
# up to 2-digit rounding (unlike the CVRP's deliberate per-arc asymmetry) and
# makes symmetry exact. `min_dist` floors distinct-stop distances so clamped or
# coincident coordinates can never produce a zero-length leg.
function _tsp_distance(rng::AbstractRNG, locations::Vector{Tuple{Float64,Float64}},
                       min_dist::Float64 = 0.5)
    n = length(locations)
    circuity = rand(rng, 1.15:0.05:1.45)
    dist = zeros(n, n)
    for i in 1:n, j in i+1:n
        a, b = locations[i], locations[j]
        d = sqrt((a[1] - b[1])^2 + (a[2] - b[2])^2)
        dist[i, j] = dist[j, i] = round(max(circuity * d, min_dist), digits = 2)
    end
    return dist
end

# Complete arc support (no self-loops): every directed arc between distinct
# nodes is allowed.
function _tsp_full_support(n::Int)
    arc_ok = trues(n, n)
    for i in 1:n
        arc_ok[i, i] = false
    end
    return arc_ok
end

# Hall-deficit arc block ("road closures cut off a district"): pick a set `S` of
# `k` stops and `k-1` gate nodes `T` outside it, then forbid every arc into `S`
# whose tail is not in `T` (including arcs within `S`). The in-degree rows of
# `S` must then sum to `k` yet can draw only on the `k-1` unit out-degrees of
# `T`: `k = Σ_{j∈S} indeg(j) ≤ Σ_{i∈T} outdeg(i) = k-1`. This contradiction
# uses only the degree rows, so it makes the model infeasible even in the LP
# relaxation, identically for every tsp formulation (the MTZ / flow / subtour
# rows only shrink the feasible set further).
#
# Plain disconnection is NOT a safe alternative: a depot-free component with
# ≥ 3 nodes admits fractional points that satisfy the relaxed degree rows and
# the relaxed MTZ rows (`x = 1/(a-1)` inside a component of size `a`, equal
# `u`), so the relaxed model would stay feasible.
#
# Requires `2k - 1 ≤ n - 1` (the sets must fit in the non-depot nodes).
function _tsp_hall_block(rng::AbstractRNG, n::Int, k::Int)
    @assert 2 * k - 1 <= n - 1 "Hall block needs 2k-1 <= n-1 (got k=$k, n=$n)"
    order = shuffle(rng, collect(2:n))
    S = sort(order[1:k])
    T = sort(order[k+1:2k-1])
    arc_ok = _tsp_full_support(n)
    for j in S, i in 1:n
        if i != j && !(i in T)
            arc_ok[i, j] = false
        end
    end
    return arc_ok, S, T
end

# Pick `n` among `{n0-1, n0, n0+1}` to best match the *delivered* variable
# count after the Hall block deletes arcs, given `delivered(n)` — the count of
# variables the variant's model actually creates at dimension `n`. Candidates
# must still leave room for the block: `n ≥ max(5, 2k+1)`.
function _tsp_pick_n(n0::Int, target::Int, k::Int, delivered::Function)
    lo = max(5, 2 * k + 1)
    best = n0
    for cand in (n0 - 1, n0, n0 + 1)
        cand >= lo || continue
        if abs(delivered(cand) - target) < abs(delivered(best) - target)
            best = cand
        end
    end
    return best
end

# Dimension plan shared by the Hall-block variants: draw the block size `k`
# unconditionally so the RNG stream stays aligned across statuses (ignored
# unless infeasible), and for an `infeasible` request size `n` against the
# *delivered* variable count — `delivered(n, k)` is the variant's variable
# count at dimension `n` once the block has deleted `k*(n-k)` arcs.
function _tsp_plan_dimensions(rng::AbstractRNG, n0::Int, target::Int,
                              status::FeasibilityStatus, delivered::Function)
    k = n0 >= 8 ? rand(rng, 2:3) : 2
    n = status == infeasible ?
        _tsp_pick_n(n0, target, k, m -> delivered(m, k)) : n0
    return n, k
end

# Arc support for the requested status: complete support unless infeasible, in
# which case the Hall-deficit block (see `_tsp_hall_block`).
function _tsp_arc_support(rng::AbstractRNG, n::Int, k::Int, status::FeasibilityStatus)
    status == infeasible && return _tsp_hall_block(rng, n, k)
    return _tsp_full_support(n), Int[], Int[]
end

include("assignment_relaxation.jl")
include("asymmetric.jl")
include("flow.jl")
include("multiple_salespersons.jl")
include("precedence.jl")
include("prize_collecting.jl")
include("standard.jl")
include("time_windows.jl")
