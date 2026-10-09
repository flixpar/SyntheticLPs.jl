# multi_commodity_flow category
#
# Entry point for the `multi_commodity_flow` problem category: commodities
# competing for shared arc capacity on sparse geographic networks (the shared
# `network_flow/geo_network.jl` machinery). This file holds the gravity-model
# commodity sampling and the metric-inequality certificate arithmetic shared by
# the variants; each included file is one variant.

using Random
using Distributions
using StatsBase

"""
    _mcf_gravity_destinations(rng, n, origin, positions, weights, n_dest, length_scale)
        -> (destinations, raw_demand)

Destination set of a commodity shipping out of `origin`: `n_dest` distinct
other nodes drawn without replacement with gravity weights
`weight[v] / (1 + dist(origin, v) / length_scale)^1.5`, and their raw (unscaled)
gravity demands `weight[origin]^0.5 * weight[v] / (1 + dist/length_scale)^1.5`.
Destinations are returned sorted.
"""
function _mcf_gravity_destinations(
    rng::AbstractRNG,
    n::Int,
    origin::Int,
    positions::Vector{Tuple{Float64, Float64}},
    weights::Vector{Float64},
    n_dest::Int,
    length_scale::Float64,
)
    others = [v for v in 1:n if v != origin]
    decay = [(1 + _geo_dist(positions, origin, v) / length_scale)^1.5 for v in others]
    g = [weights[others[i]] / decay[i] for i in eachindex(others)]
    picked = sample(rng, eachindex(others), Weights(g), min(n_dest, length(others)); replace=false)
    sort!(picked)
    return others[picked], [sqrt(weights[origin]) * g[i] for i in picked]
end

"""
    _mcf_metric_requirement(n, arcs, out_adj, lengths, origins, destinations, demands)
        -> Float64

Right-hand side of the metric (Japanese-theorem) inequality for arc lengths
`lengths >= 0`: `sum_k sum_v demand[k][v] * dist(origin[k], v)`, with
shortest-path distances from each commodity's origin. Any feasible
multicommodity routing satisfies
`sum_a capacity[a] * lengths[a] >= sum_a load[a] * lengths[a] >= requirement`,
so a capacity-length product below it certifies infeasibility from the
capacity and conservation rows alone (relaxation-proof).
"""
function _mcf_metric_requirement(
    n::Int,
    arcs::Vector{Tuple{Int, Int}},
    out_adj::Vector{Vector{Int}},
    lengths::Vector{Float64},
    origins::Vector{Int},
    destinations::Vector{Vector{Int}},
    demands::Vector{Vector{Float64}},
)
    total = 0.0
    for k in eachindex(origins)
        dist, _ = _geo_dijkstra(n, arcs, out_adj, lengths, [origins[k]])
        for (i, v) in enumerate(destinations[k])
            total += demands[k][i] * dist[v]
        end
    end
    return total
end

include("standard.jl")
include("binary_capacity.jl")
