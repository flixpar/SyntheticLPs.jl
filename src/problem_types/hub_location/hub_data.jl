# Shared data generation for the hub_location family.
#
# The helpers below are grounded in the two classical hub-location benchmark
# datasets, whose conventions were verified against the published files:
#
# * CAB (Civil Aeronautics Board; O'Kelly 1987, OR-Library file `phub4`):
#   25 US cities, fully symmetric passenger flows spanning roughly three
#   orders of magnitude (565..205,088, median ~7,000) and *network* distances
#   (they do not come from coordinates, and obey the triangle inequality only
#   approximately). Discount factors alpha in [0.2, 1.0], p in 2..5, and
#   collection/distribution multipliers chi = delta = 1 are the standard
#   experimental grid.
# * AP (Australia Post; Ernst & Krishnamoorthy 1996, OR-Library files
#   `phub1..3`): 200 postal districts on a ~57 km x 55 km plane, Euclidean
#   distances, *asymmetric* flows (63% of ordered pairs) with heavy right skew
#   (mean/median ~ 5.2, coefficient of variation ~ 5), nonzero self-flows on
#   every node, per-unit costs "per unit (euclidean) distance, per unit flow
#   volume divided by 1000", and the published parameters
#   chi = 3 (collection), alpha = 0.75 (transfer), delta = 2 (distribution).
#   Hub capacities ("T" tight / "L" loose files) bound the total flow *into*
#   a hub, including flow originating at the node itself.

using Random
using Distributions
using LinearAlgebra: diagind

"""
    _hub_city_locations(rng, n, shape; span=100.0) -> Vector{Tuple{Float64,Float64}}

Scatter `n` cities over a `span` x `span` region in one of three settlement
shapes:

* `:clustered`   – a handful of population regions (the AP postal geography);
* `:corridor`    – cities strung along a linear freight/rail corridor;
* `:archipelago` – well-separated island groups (airline / backbone settings
  where inter-group distance dominates).

Returns coordinates in `[0, span]^2`.
"""
function _hub_city_locations(rng::AbstractRNG, n::Int, shape::Symbol;
                             span::Float64=100.0)
    shape in (:clustered, :corridor, :archipelago) ||
        error("Unknown hub geography shape $shape.")
    if shape == :corridor
        # Cities along a slightly bent corridor with Gaussian lateral scatter.
        m = rand(rng, Uniform(0.15span, 0.35span))
        b = rand(rng, Uniform(0.2span, 0.8span))
        slope = rand(rng, Uniform(-0.6, 0.6))
        jitter = 0.06span
        return [(clamp(u + rand(rng, Normal(0.0, jitter)), 0.0, span),
                 clamp(m + slope * (u - b) + rand(rng, Normal(0.0, jitter)),
                       0.0, span))
                for u in range(0.0, span; length=n)]
    end

    if shape == :archipelago
        n_groups = clamp(round(Int, sqrt(n) / 1.6), 2, 5)
        min_sep = span * 0.55
        centers = _hub_separated_centers(rng, n_groups, min_sep; span=span)
    else
        n_groups = clamp(round(Int, sqrt(n)), 2, 6)
        min_sep = span * 0.28
        centers = _hub_separated_centers(rng, n_groups, min_sep; span=span)
    end

    group_radius = 0.16span
    # Every group keeps one anchor city exactly at its center; this guarantees
    # each group contributes a reachable hub candidate for reach windows.
    locations = Tuple{Float64,Float64}[]
    for g in 1:n_groups
        push!(locations, centers[g])
    end
    groups = rand(rng, 1:n_groups, n)
    while length(locations) < n
        g = rand(rng, groups)
        push!(locations,
              (clamp(centers[g][1] + rand(rng, Normal(0.0, group_radius / 2)),
                     0.0, span),
               clamp(centers[g][2] + rand(rng, Normal(0.0, group_radius / 2)),
                     0.0, span)))
    end
    return locations
end

"""
    _hub_separated_centers(rng, q, min_sep; span=100.0, tries=800)

Rejection-sample `q` group centers in `[0, span]^2` that are pairwise at least
`min_sep` apart. If the separation cannot be met within `tries` proposals the
requirement is relaxed geometrically so a valid configuration is always
returned - the returned centers therefore only *approximately* honour
`min_sep`. Callers that need a guaranteed separation (the disjoint-region
infeasibility certificates) use [`_hub_ring_centers`](@ref) instead.
"""
function _hub_separated_centers(rng::AbstractRNG, q::Int, min_sep::Float64;
                                span::Float64=100.0, tries::Int=800)
    centers = Tuple{Float64,Float64}[]
    sep = min_sep
    while length(centers) < q
        proposed = (span * rand(rng), span * rand(rng))
        if all(hypot(proposed[1] - c[1], proposed[2] - c[2]) >= sep
               for c in centers)
            push!(centers, proposed)
            continue
        end
        # Restart with a looser separation rather than looping forever.
        if length(centers) == 0 || rand(rng) < 1.0 / tries
            sep *= 0.95
            if sep < 1e-6 * span
                sep = 1e-6 * span
            end
        end
    end
    return centers
end

"""
    _hub_distance_matrix(locations) -> Matrix{Float64}

Euclidean distance matrix with a zero diagonal. Distances stay metric, which
the multicommodity-flow variants rely on: with metric costs and a uniform
inter-hub discount, routing through additional hubs is never cheaper than the
direct inter-hub leg, so optimal flow paths visit at most two hubs.
"""
function _hub_distance_matrix(locations::Vector{Tuple{Float64,Float64}})
    n = length(locations)
    d = zeros(n, n)
    for i in 1:n, j in (i + 1):n
        d[i, j] = d[j, i] = hypot(locations[i][1] - locations[j][1],
                                  locations[i][2] - locations[j][2])
    end
    return d
end

"""
    _hub_detour_cost_matrix(rng, dist, lo, hi) -> Matrix{Float64}

CAB-style *network* costs: symmetric per-pair detour factors multiply the
Euclidean distance, mimicking road/route distances that only approximately
obey the triangle inequality (as in the published CAB cost matrix). Used only
by the 4-index path-flow variants, whose fixed i-k-m-j paths need no metric
assumption.
"""
function _hub_detour_cost_matrix(rng::AbstractRNG, dist::Matrix{Float64},
                                 lo::Float64, hi::Float64)
    n = size(dist, 1)
    c = zeros(n, n)
    for i in 1:n, j in (i + 1):n
        factor = exp(rand(rng, Uniform(log(lo), log(hi))))
        c[i, j] = c[j, i] = dist[i, j] * factor
    end
    return c
end

"""
    _hub_populations(rng, n) -> Vector{Float64}

City masses: lognormal (the AP and CAB flow totals span two-plus orders of
magnitude between major sorting centers and small offices).
"""
_hub_populations(rng::AbstractRNG, n::Int) =
    exp.(rand(rng, Normal(log(60.0), 1.1), n))

"""
    _hub_gravity_flows(rng, n, populations, dist, decay, noise;
                       symmetric=false, scale=1.0) -> Matrix{Float64}

Production/attraction gravity flows with lognormal residual scatter:

    w_ij = scale * O_i * D_j / max(d_ij, d_floor)^decay * LogNormal(0, noise)

`O_i`/`D_j` are origin/destination potentials proportional to population with
independent lognormal jitter. With `symmetric=true` the matrix is symmetrized
(CAB airline passengers travel both directions on the same route); otherwise
directions are independent (AP postal volumes are 63% asymmetric). The diagonal
is zero: same-node flow never enters the routing models.
"""
function _hub_gravity_flows(rng::AbstractRNG, n::Int,
                            populations::Vector{Float64},
                            dist::Matrix{Float64}, decay::Float64,
                            noise::Float64; symmetric::Bool=false,
                            scale::Float64=1.0)
    origins = populations .* exp.(rand(rng, Normal(0.0, 0.45), n))
    destinations = populations .* exp.(rand(rng, Normal(0.0, 0.45), n))
    floor_dist = 0.05 * maximum(dist)
    w = zeros(n, n)
    for i in 1:n, j in 1:n
        i == j && continue
        base = origins[i] * destinations[j] /
               max(dist[i, j], floor_dist)^decay
        w[i, j] = base * exp(rand(rng, Normal(0.0, noise)))
    end
    if symmetric
        w = (w .+ w') ./ 2
    end
    # Normalise the overall volume so downstream capacity scales are stable.
    total = sum(w)
    total > 0 && (w .*= scale * n^2 / total)
    # Keep every ordered pair positive (the AP matrix has no zero entries) but
    # retain the heavy right skew: round to three decimals *first*, then floor,
    # so the rounding cannot push a tiny volume back to exactly zero (a zero
    # w_ij drops that pair's supply row and the hub-opening it forces).
    w .= max.(round.(w; digits=3), 0.001)
    w[diagind(w)] .= 0.0
    return w
end

"""
    _hub_reach_admissible(dist, reach; candidates=Int[], include_self=true)
    -> Vector{Vector{Int}}

Admissible hub lists `A_i` under a reach window: node `i` may only be served by
candidates within `reach` of it. Returns, per node, the sorted list of
admissible candidates. `candidates` restricts the hub sites (empty means every
node is a candidate).

With every node a candidate each list contains the node itself (distance zero)
and is therefore nonempty. Under a restricted `candidates` set that guarantee
is gone: a node farther than `reach` from every candidate gets an empty list,
which makes its allocation/supply rows unsatisfiable. Callers that need a
feasible instance must size `reach` above the covering radius of `candidates`.
"""
function _hub_reach_admissible(dist::Matrix{Float64}, reach::Float64;
                               candidates::AbstractVector{Int}=Int[],
                               include_self::Bool=true)
    n = size(dist, 1)
    cand = isempty(candidates) ? collect(1:n) : collect(candidates)
    admissible = [Int[] for _ in 1:n]
    for i in 1:n
        for k in cand
            (k == i && !include_self) && continue
            dist[i, k] <= reach && push!(admissible[i], k)
        end
    end
    return admissible
end

"""
    _hub_greedy_hubs(dist, weight, p) -> Vector{Int}

Greedy p-median style hub placement: add the candidate that most reduces the
weight-loaded distance to the nearest chosen hub. `weight[i]` is the total
flow originating and terminating at node `i`, so heavy cities pull hubs toward
them, as in real networks.
"""
function _hub_greedy_hubs(dist::Matrix{Float64}, weight::Vector{Float64}, p::Int)
    n = size(dist, 1)
    chosen = Int[]
    # Start from the diameter rather than Inf: with an infinite incumbent every
    # candidate scores an infinite gain in the first round, so the first (and
    # most important) hub would always fall on node 1 regardless of weights.
    best = fill(maximum(dist), n)
    for _ in 1:p
        best_gain = -Inf
        best_k = 0
        for k in 1:n
            k in chosen && continue
            gain = 0.0
            for i in 1:n
                gain += weight[i] * max(best[i] - dist[i, k], 0.0)
            end
            if gain > best_gain
                best_gain = gain
                best_k = k
            end
        end
        best_k == 0 && break
        push!(chosen, best_k)
        for i in 1:n
            best[i] = min(best[i], dist[i, best_k])
        end
    end
    return chosen
end

"""
    _hub_nearest_assignment(dist, hubs) -> Vector{Int}

Assign every node to its nearest hub (ties to the smallest index).
"""
function _hub_nearest_assignment(dist::Matrix{Float64}, hubs::Vector{Int})
    n = size(dist, 1)
    return [hubs[argmin([dist[i, k] for k in hubs])] for i in 1:n]
end
