using JuMP
using Random
using Distributions
using StatsBase
using Statistics

"""
Largest `target_variables` accepted by `load_balancing/standard`; the
constructor stores every candidate path explicitly, so larger requests raise an
`ArgumentError`.
"""
const LOAD_BALANCING_MAX_VARIABLES = 1_000_000

"""
Standard router-port capacities (in units of 1 Gb/s): every link is provisioned
with one of these, and very fat links as a bundle (LAG) of the largest.
"""
const _LB_PORT_SIZES = (1.0, 2.5, 10.0, 40.0, 100.0, 400.0)

"""
Planted traffic-engineering solution: every OD demand on its first (shortest)
candidate path; `utilization` is the resulting maximum link utilization, at
most the SLA bound.
"""
struct LoadBalancingWitness
    path_flows::Vector{Float64}
    utilization::Float64
end

"""
Metric certificate over the candidate paths. For link lengths `lengths >= 0`
(here: link latencies on the links present in the model, 0 elsewhere), any
feasible point routes every OD demand on paths no
shorter than its shortest candidate, while each link carries at most
`max_utilization * capacity`:

    sum_k demand[k] * min_{p in P(k)} length(p)  <=  sum_a length[a] * load[a]
                                                <=  max_utilization * sum_a length[a] * capacity[a]

The certificate stores `required` (left side) and `capacity_length` (right
side) with `capacity_length < required` — the network cannot carry the grown
traffic matrix within the SLA. It combines every demand row with every link
row, so presolve cannot see it.
"""
struct LoadBalancingCertificate
    lengths::Vector{Float64}
    capacity_length::Float64
    required::Float64
end

"""
    LoadBalancingProblem <: ProblemGenerator

Path-based traffic engineering on an ISP-style backbone: route a gravity
traffic matrix over precomputed candidate paths to minimise the maximum link
utilization (plus a small latency term).

# Overview

    minimize    U + latency_weight * sum_p latency[p] * x[p]
    subject to  sum_{p in P(k)} x[p] = demand[k]               every OD pair k
                sum_{p uses a} x[p] - capacity[a] * U <= 0     every used link a
                0 <= U <= max_utilization,  x >= 0

Each OD pair has 1-8 diverse candidate paths (successive shortest-path trees
from its origin with lengths penalised on links the earlier trees used, as
k-path TE tools produce). The path-link incidence makes the matrix very
different from arc-based multicommodity flow: long dense columns, few rows per
column, and one coupling column `U` in every link row.

# Data grounding

PoPs come from `_geo_positions` (metro clusters dominate); links from
`_geo_network` (4.5-6.5 directed links per PoP, strongly connected). Link
latency = distance / 200 + 0.1 per hop (ms-like units). Traffic is a gravity
matrix `w_o w_d / (1 + dist / L)` over OD pairs drawn by gravity weight.
Capacities are standard router ports (`_LB_PORT_SIZES`, bundles of 400 for
fat links), provisioned for the planted routing at a target utilization.
`max_utilization` (the SLA) is 0.8, 0.9 or 1.0.

# Feasibility control

The planted routing puts each demand on its first path; ports are provisioned
so its utilization is at most a target in [0.45, 0.8] of the SLA.

  - `feasible`: the planted routing is the witness.
  - `infeasible`: the traffic matrix grows until the latency-metric
    certificate separates (capacity-length 80%-93% of the requirement); every
    OD pair keeps at least 1.15x its demand of path bottleneck capacity, so no
    single demand row is refutable on its own.
  - `unknown`: traffic grows by 0.6-1.2x the factor that would bring the
    planted routing to the SLA (never shrinking), with the same local repair:
    rerouting over the alternative paths may or may not absorb it.

# Sizing

Variables = candidate paths + 1, within 2 of `target_variables` (paths are
trimmed, never below three per pair, or OD pairs added until the count
matches; networks too small to offer that many distinct paths stay below).
Rows = OD pairs + links carrying a path or background traffic.
"""
struct LoadBalancingProblem <: ProblemGenerator
    n_nodes::Int
    links::Vector{Tuple{Int, Int}}
    capacities::Vector{Float64}
    link_latency::Vector{Float64}
    od_pairs::Vector{Tuple{Int, Int}}
    demands::Vector{Float64}
    background::Vector{Float64}
    paths::Vector{Vector{Int}}
    path_od::Vector{Int}
    max_utilization::Float64
    latency_weight::Float64
    positions::Vector{Tuple{Float64, Float64}}
    geography::Symbol
    feasible_witness::Union{Nothing, LoadBalancingWitness}
    infeasibility_certificate::Union{Nothing, LoadBalancingCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _lb_port(need) -> Float64

Smallest standard port at least `need`, or the smallest bundle of 400s.
"""
function _lb_port(need::Float64)
    for s in _LB_PORT_SIZES
        s >= need && return s
    end
    return 400.0 * ceil(need / 400.0)
end

"""
    _lb_candidate_paths(rng, n, links, out_adj, latency, origin, dests, n_iter)
        -> Vector{Vector{Vector{Int}}}

Up to `n_iter` distinct candidate paths (link-index lists) from `origin` to
each destination: successive shortest-path trees with lengths multiplied by
`exp(0.7 * uses)` on links earlier trees from this origin used (plus 10%
lognormal noise), deduplicated per destination.
"""
function _lb_candidate_paths(
    rng::AbstractRNG,
    n::Int,
    links::Vector{Tuple{Int, Int}},
    out_adj::Vector{Vector{Int}},
    latency::Vector{Float64},
    origin::Int,
    dests::Vector{Int},
    n_iter::Int,
)
    uses = zeros(Int, length(links))
    found = [Vector{Vector{Int}}() for _ in dests]
    seen = [Set{Vector{Int}}() for _ in dests]
    for _ in 1:n_iter
        lens = [latency[a] * exp(0.7 * uses[a]) * rand(rng, LogNormal(0.0, 0.1)) for a in eachindex(links)]
        _, pred = _geo_dijkstra(n, links, out_adj, lens, [origin])
        tree_links = Set{Int}()
        for (i, d) in enumerate(dests)
            path = Int[]
            v = d
            while v != origin
                a = pred[v]
                push!(path, a)
                v = links[a][1]
            end
            reverse!(path)
            if !(path in seen[i])
                push!(seen[i], path)
                push!(found[i], path)
            end
            union!(tree_links, path)
        end
        for a in tree_links
            uses[a] += 1
        end
    end
    return found
end

function LoadBalancingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= LOAD_BALANCING_MAX_VARIABLES || throw(
        ArgumentError(
            "load_balancing/standard supports at most $LOAD_BALANCING_MAX_VARIABLES variables; " *
            "requested $target_variables.",
        ),
    )
    rng = MersenneTwister(seed)
    n_paths_target = max(target_variables - 1, 1)

    # Backbone.
    n = clamp(round(Int, 1.2 * sqrt(n_paths_target / 3)), 4, 1500)
    n_links = clamp(round(Int, n * (4.5 + 2.0 * rand(rng))), 2 * (n - 1), n * (n - 1))
    geography = let r = rand(rng)
        r < 0.5 ? :clustered : (r < 0.8 ? :uniform : :corridor)
    end
    positions, weights = _geo_positions(rng, n, geography; span=40.0 * sqrt(n))
    links, _ = _geo_network(rng, positions, n_links)
    L = length(links)
    out_adj, _ = _geo_adjacency(n, links)
    dist = [_geo_dist(positions, u, v) for (u, v) in links]
    latency = round.(dist ./ 200.0 .+ 0.1; digits=4)
    length_scale = 5.0 * median(dist)

    # OD pairs by gravity, in batches until enough candidate paths exist. An
    # OD pair whose candidate generation finds fewer than three paths has
    # (almost) nothing to balance: it stays on its shortest path as fixed
    # background traffic (with two paths its demand row would just be a
    # doubleton that presolve substitutes away).
    all_pairs = [(o, d) for o in 1:n for d in 1:n if o != d]
    gravity = [weights[o] * weights[d] / (1 + _geo_dist(positions, o, d) / length_scale) for (o, d) in all_pairs]
    order = sample(rng, eachindex(all_pairs), Weights(gravity), length(all_pairs); replace=false)
    n_iter = rand(rng, 5:8)
    od_pairs = Tuple{Int, Int}[]
    od_paths = Vector{Vector{Int}}[]
    bg_pairs = Tuple{Int, Int}[]
    bg_paths = Vector{Int}[]
    next = 1
    total = 0
    while total < n_paths_target && next <= length(order)
        batch_size = max(1, ceil(Int, (n_paths_target - total) / 2.5))
        batch = order[next:min(next + batch_size - 1, length(order))]
        next += length(batch)
        by_origin = Dict{Int, Vector{Int}}()
        for idx in batch
            push!(get!(by_origin, all_pairs[idx][1], Int[]), all_pairs[idx][2])
        end
        for o in sort!(collect(keys(by_origin)))
            dests = sort!(by_origin[o])
            found = _lb_candidate_paths(rng, n, links, out_adj, latency, o, dests, n_iter)
            for (i, d) in enumerate(dests)
                if length(found[i]) >= 3
                    push!(od_pairs, (o, d))
                    push!(od_paths, found[i])
                    total += length(found[i])
                else
                    push!(bg_pairs, (o, d))
                    push!(bg_paths, found[i][1])
                end
            end
        end
    end
    # Trim surplus paths (never below three per OD pair) to hit the target.
    trimmable = [k for k in eachindex(od_paths) if length(od_paths[k]) > 3]
    while total > n_paths_target && !isempty(trimmable)
        i = rand(rng, eachindex(trimmable))
        k = trimmable[i]
        deleteat!(od_paths[k], rand(rng, 2:length(od_paths[k])))
        total -= 1
        if length(od_paths[k]) <= 3
            trimmable[i] = trimmable[end]
            pop!(trimmable)
        end
    end
    while total > n_paths_target  # only pairs left: drop whole pairs
        k = rand(rng, eachindex(od_paths))
        total -= length(od_paths[k])
        deleteat!(od_pairs, k)
        deleteat!(od_paths, k)
    end
    perm = sortperm(od_pairs)
    od_pairs = od_pairs[perm]
    od_paths = od_paths[perm]
    K = length(od_pairs)

    gravity_of(o, d) = weights[o] * weights[d] / (1 + _geo_dist(positions, o, d) / length_scale)
    raw = [gravity_of(o, d) for (o, d) in od_pairs]
    scale = 10.0 / (sum(raw; init=0.0) / max(K, 1))
    demands = max.(round.(scale .* raw .* rand(rng, LogNormal(0.0, 0.3), K); digits=3), 0.001)
    background = zeros(L)
    for (i, (o, d)) in enumerate(bg_pairs)
        v = max(round(scale * gravity_of(o, d) * rand(rng, LogNormal(0.0, 0.3)); digits=3), 0.001)
        for a in bg_paths[i]
            background[a] += v
        end
    end

    paths = Vector{Int}[]
    path_od = Int[]
    for k in 1:K, path in od_paths[k]
        push!(paths, path)
        push!(path_od, k)
    end
    paths_of = [Int[] for _ in 1:K]
    for (p, k) in enumerate(path_od)
        push!(paths_of[k], p)
    end
    first_path = [paths_of[k][1] for k in 1:K]
    # Links every candidate path of an OD pair uses (its forced links).
    forced = [reduce(intersect, (Set(paths[p]) for p in paths_of[k])) for k in 1:K]

    # Planted routing: everything on the first path; ports provisioned.
    max_utilization = rand(rng, (0.8, 0.9, 1.0))
    target_util = max_utilization * (0.45 + 0.35 * rand(rng))
    load = copy(background)
    for k in 1:K, a in paths[first_path[k]]
        load[a] += demands[k]
    end
    capacities = [
        load[a] > 0 ? _lb_port(load[a] / target_util) : _LB_PORT_SIZES[rand(rng, 1:4)] for a in 1:L
    ]
    planted_util = maximum(load[a] / capacities[a] for a in 1:L)
    latency_weight = round(
        (0.05 + 0.25 * rand(rng)) / ((sum(demands) + sum(background) / 5) * mean(latency) * 5); sigdigits=4
    )

    # Local repair at the SLA, with 15% slack: every link carries its forced
    # load (background plus the demands of OD pairs all of whose paths use
    # it), and every OD pair has 1.15x its demand of summed path bottlenecks.
    # Otherwise one link row or one demand row could refute the model.
    function local_repair!(caps, dem; slack=1.15)
        slack > 0 || return nothing
        forced_load = copy(background)
        for k in 1:K, a in forced[k]
            forced_load[a] += dem[k]
        end
        for a in 1:L
            need = slack * forced_load[a] / max_utilization
            caps[a] < need && (caps[a] = _lb_port(need))
        end
        for k in 1:K
            ps = [paths[p] for p in paths_of[k]]
            have = sum(minimum(caps[a] for a in p) for p in ps) * max_utilization
            have >= slack * dem[k] && continue
            g = slack * dem[k] / have
            # Lift only each path's bottleneck links to g x its bottleneck.
            for p in ps
                lift = g * minimum(caps[a] for a in p)
                for a in p
                    caps[a] < lift && (caps[a] = _lb_port(lift))
                end
            end
        end
    end

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        flows = zeros(length(paths))
        for k in 1:K
            flows[first_path[k]] = demands[k]
        end
        feasible_witness = LoadBalancingWitness(flows, planted_util)
    elseif feasibility_status == infeasible
        # Latency on the links the model actually has (unused links carry no
        # row, so their capacity must not count).
        in_model = copy(background .> 0)
        for path in paths, a in path
            in_model[a] = true
        end
        lengths = [in_model[a] ? latency[a] : 0.0 for a in 1:L]
        min_len = [minimum(sum(lengths[a] for a in paths[p]) for p in paths_of[k]) for k in 1:K]
        ratio = 0.8 + 0.13 * rand(rng)
        required_of() = sum(demands .* min_len; init=0.0) + sum(background .* lengths)
        caps0, dem0 = copy(capacities), copy(demands)
        # On tiny backbones the repair can keep pace with the growth; relax its
        # slack (and finally drop it) until the certificate separates.
        for slack in (1.15, 1.0, 0.0)
            capacities .= caps0
            demands .= dem0
            for _ in 1:40
                cap_len = max_utilization * sum(capacities .* lengths)
                cap_len < ratio * required_of() && break
                # Grow TE traffic (background is fixed) toward the target ratio.
                te = sum(demands .* min_len; init=0.0)
                g = (cap_len / ratio - sum(background .* lengths)) / te
                demands .= round.(demands .* max(g, 1.01); digits=3)
                local_repair!(capacities, demands; slack=slack)
            end
            max_utilization * sum(capacities .* lengths) < required_of() && break
        end
        cap_len = max_utilization * sum(capacities .* lengths)
        required = required_of()
        cap_len < required || error("load_balancing/standard: certificate failed to separate (seed $seed)")
        infeasibility_certificate = LoadBalancingCertificate(lengths, cap_len, required)
    else
        growth = (max_utilization / planted_util) * (0.6 + 0.6 * rand(rng))
        demands .= round.(demands .* max(growth, 1.0); digits=3)
        local_repair!(capacities, demands)
    end

    return LoadBalancingProblem(
        n,
        links,
        capacities,
        latency,
        od_pairs,
        demands,
        background,
        paths,
        path_od,
        max_utilization,
        latency_weight,
        positions,
        geography,
        feasible_witness,
        infeasibility_certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::LoadBalancingProblem)

Build the path-based min-max-utilization TE LP. Deterministic — uses only the
struct fields.
"""
function build_model(prob::LoadBalancingProblem)
    model = Model()
    P = length(prob.paths)
    K = length(prob.od_pairs)
    @variable(model, x[1:P] >= 0)
    @variable(model, 0 <= U <= prob.max_utilization)
    path_latency = [sum(prob.link_latency[a] for a in prob.paths[p]) for p in 1:P]
    @objective(model, Min, U + prob.latency_weight * sum(path_latency[p] * x[p] for p in 1:P))
    of_od = [Int[] for _ in 1:K]
    on_link = [Int[] for _ in eachindex(prob.links)]
    for p in 1:P
        push!(of_od[prob.path_od[p]], p)
        for a in prob.paths[p]
            push!(on_link[a], p)
        end
    end
    for k in 1:K
        @constraint(model, sum(x[p] for p in of_od[k]) == prob.demands[k])
    end
    for a in eachindex(prob.links)
        (isempty(on_link[a]) && prob.background[a] == 0) && continue
        @constraint(
            model,
            sum(x[p] for p in on_link[a]; init=AffExpr(0.0)) - prob.capacities[a] * U <= -prob.background[a]
        )
    end
    return model
end

register_variant(
    :load_balancing,
    :standard,
    LoadBalancingProblem,
    "Path-based traffic engineering on an ISP-style geographic backbone: route a gravity traffic matrix over diverse candidate paths to minimise maximum link utilization under an SLA bound, with standard router-port capacities, a planted routing witness and a latency-metric infeasibility certificate",
)
