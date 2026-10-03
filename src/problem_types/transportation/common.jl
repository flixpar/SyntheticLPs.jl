# Shared geography and lane machinery for the transportation variants.
#
# Plants/warehouses and customers are scattered over one region (sources sit
# among the population, as real DCs do), and each customer is served by a
# sparse set of lanes to its nearest sources plus occasional long-haul lanes —
# the lane sets real freight networks contract, instead of a complete bipartite
# graph between a few dozen nodes. Exact max flows on the lane network (the
# `network_flow` helpers `_network_flow_max_scale` / `_network_flow_extended`)
# place every feasibility profile against an exact boundary.

using Random
using Distributions

"""
    _tp_dimensions(rng, target_lanes, lanes_per_dest, dest_per_source) -> (n_sources, n_dests)

Customer and source counts for a lane budget: about `lanes_per_dest` lanes per
customer and `dest_per_source` customers per source, adjusted so the complete
bipartite graph can hold the budget.
"""
function _tp_dimensions(rng::AbstractRNG, target_lanes::Int, lanes_per_dest::Float64, dest_per_source::Float64)
    n_dests = max(1, round(Int, target_lanes / lanes_per_dest))
    n_sources = max(2, round(Int, n_dests / dest_per_source))
    while n_sources * n_dests < target_lanes
        n_sources <= 2 * n_dests ? (n_sources += 1) : (n_dests += 1)
    end
    return n_sources, n_dests
end

"""
    _tp_geography(rng, n_sources, n_dests) -> (src_pos, dst_pos, src_weight, dst_weight, shape)

Positions and heavy-tailed activity weights for sources and customers drawn
from one `_geo_positions` population (region side grows with `sqrt` of the
node count), split at random so sources sit among the customers.
"""
function _tp_geography(rng::AbstractRNG, n_sources::Int, n_dests::Int)
    n = n_sources + n_dests
    shape = let r = rand(rng)
        r < 0.45 ? :clustered : (r < 0.8 ? :uniform : :corridor)
    end
    positions, weights = _geo_positions(rng, n, shape; span=12.0 * sqrt(n))
    perm = randperm(rng, n)
    src = perm[1:n_sources]
    dst = perm[(n_sources + 1):end]
    return positions[src], positions[dst], weights[src], weights[dst], shape
end

"""
    _tp_lanes(rng, src_pos, dst_pos, n_lanes; mean_lanes, long_haul=0.25) -> (lanes, primary)

Exactly `n_lanes` distinct lanes `(source, customer)`, sorted, with every
customer served by between 2 (1 only when the budget is below two per customer)
and `n_sources` lanes: a lognormal per-customer lane count around `mean_lanes`,
scaled by the customer's activity `weights`^0.3 (big customers contract more
carriers)
(adjusted by +-1 steps to hit the budget), filled with the customer's nearest
sources and — with probability `long_haul` — one long-haul lane to a farther
source among its `2k + 3` nearest. Also returns each customer's primary
(nearest) source.
"""
function _tp_lanes(
    rng::AbstractRNG,
    src_pos::Vector{Tuple{Float64, Float64}},
    dst_pos::Vector{Tuple{Float64, Float64}},
    n_lanes::Int;
    mean_lanes::Float64,
    long_haul::Float64=0.25,
    weights::Vector{Float64}=ones(length(dst_pos)),
)
    nS, nD = length(src_pos), length(dst_pos)
    nS * nD >= n_lanes >= nD || throw(ArgumentError("lane budget $n_lanes infeasible for $nS x $nD"))
    kmin = n_lanes >= 2nD ? 2 : 1
    # Bigger customers contract more carriers (lane count ~ weight^0.3).
    rel = (weights ./ (sum(weights) / nD)) .^ 0.3
    rel ./= sum(rel) / nD
    k = [
        clamp(round(Int, mean_lanes * rel[j] * rand(rng, LogNormal(0.0, 0.3))), kmin, nS) for j in 1:nD
    ]
    diff = n_lanes - sum(k)
    while diff != 0
        j = rand(rng, 1:nD)
        if diff > 0 && k[j] < nS
            k[j] += 1
            diff -= 1
        elseif diff < 0 && k[j] > kmin
            k[j] -= 1
            diff += 1
        end
    end
    near = _geo_knn_query(src_pos, dst_pos, min(nS, 2 * maximum(k) + 3))
    lanes = Vector{Tuple{Int, Int}}(undef, 0)
    sizehint!(lanes, n_lanes)
    primary = [near[j][1] for j in 1:nD]
    for j in 1:nD
        cand = near[j]
        kj = k[j]
        if kj >= 2 && length(cand) > kj && rand(rng) < long_haul
            for i in cand[1:(kj - 1)]
                push!(lanes, (i, j))
            end
            push!(lanes, (cand[rand(rng, (kj + 1):min(length(cand), 2kj + 3))], j))
        else
            for i in cand[1:kj]
                push!(lanes, (i, j))
            end
        end
    end
    return sort!(lanes), primary
end

"""
    _tp_maxflow_network(n_sources, lanes, lane_caps, big) -> (arcs, caps)

The lane network in the node numbering of the `network_flow` max-flow helpers:
sources `1:n_sources`, customer `j` at `n_sources + j`, uncapacitated lanes
(`Inf`) at capacity `big`.
"""
function _tp_maxflow_network(n_sources::Int, lanes::Vector{Tuple{Int, Int}}, lane_caps::Vector{Float64}, big::Float64)
    arcs = [(i, n_sources + j) for (i, j) in lanes]
    caps = [isfinite(u) ? u : big for u in lane_caps]
    return arcs, caps
end

"""
    _tp_market_supply(rng, src_pos, lanes, primary, d0; build_median=1.3, build_sd=0.2)
        -> Vector{Float64}

Source capacities sized to their historical market: every customer's nominal
demand is attributed half to its primary (nearest) source and half spread over
its other lanes. Each source's capacity is that market times a lognormal build
factor (median `build_median`, default 1.5, log-sd `build_sd`) times REGIONAL under-build
shocks: 2-6 discs (centred on random sources, radius 10%-25% of the region)
inside which plants were built at only 45%-80% of their market (a closed
plant, a lagging region). Total supply comfortably exceeds demand, so the
binding limits are regional — whole areas whose own plants plus the lanes
reaching in from outside cannot cover them — rather than an aggregate
shortage or a single starved customer. A 5% floor of the mean market keeps
every source meaningful.
"""
function _tp_market_supply(
    rng::AbstractRNG,
    src_pos::Vector{Tuple{Float64, Float64}},
    lanes::Vector{Tuple{Int, Int}},
    primary::Vector{Int},
    d0::Vector{Float64};
    build_median::Float64=1.5,
    build_sd::Float64=0.2,
)
    n_sources = length(src_pos)
    nD = length(d0)
    deg = zeros(Int, nD)
    for (_, j) in lanes
        deg[j] += 1
    end
    market = zeros(n_sources)
    for (i, j) in lanes
        if i == primary[j]
            market[i] += deg[j] == 1 ? d0[j] : 0.5 * d0[j]
        else
            market[i] += 0.5 * d0[j] / (deg[j] - 1)
        end
    end
    xs = [p[1] for p in src_pos]
    ys = [p[2] for p in src_pos]
    span = max(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys), 1e-9)
    factor = [rand(rng, LogNormal(log(build_median), build_sd)) for _ in 1:n_sources]
    for _ in 1:rand(rng, 2:6)
        c = src_pos[rand(rng, 1:n_sources)]
        radius = span * (0.1 + 0.15 * rand(rng))
        shock = 0.45 + 0.35 * rand(rng)
        for i in 1:n_sources
            hypot(src_pos[i][1] - c[1], src_pos[i][2] - c[2]) <= radius && (factor[i] *= shock)
        end
    end
    floor_supply = 0.05 * sum(market) / n_sources
    return [round(market[i] * factor[i] + floor_supply; digits=2) for i in 1:n_sources]
end
