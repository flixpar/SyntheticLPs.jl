using JuMP
using Random
using Distributions
using Statistics

"""
Largest `target_variables` accepted by the multi_commodity_flow variants; the
constructors hold arc x commodity matrices and per-commodity shortest-path
trees, so larger requests raise an `ArgumentError` instead of undersizing.
"""
const MCF_MAX_VARIABLES = 1_000_000

"""
Planted routing: `flows[a, k]` routes every commodity's demands along a
shortest-path tree (commodity-specific noisy lengths) from its origin, so
conservation holds exactly for every commodity, and the installed capacities
cover the aggregate load on every arc.
"""
struct MultiCommodityFlowWitness
    flows::Matrix{Float64}
end

"""
Metric-inequality (Onaga-Kakusho / "Japanese theorem") certificate. For arc
lengths `lengths >= 0`, any feasible routing satisfies

    sum_a capacity[a] * lengths[a]  >=  sum_k sum_v demand[k][v] * dist_lengths(origin[k], v)

(each commodity's flow decomposes into origin-destination paths no shorter
than the shortest one, and the bundle rows cap the aggregate). The certificate
stores the lengths and both sides with `capacity_length < required`.

  - `mode = :length`: lengths are geographic arc lengths — a network-wide
    capacity shortage that only shows when all commodities' path lengths are
    weighed together.
  - `mode = :regional_cut`: lengths are 1 on the arcs leaving `region` (a
    geographic area around a major origin) and 0 elsewhere — the exit
    capacity of the region is below the demand that must leave it.

Neither involves a single row, so presolve cannot detect it.
"""
struct MultiCommodityFlowMetricCertificate
    mode::Symbol
    lengths::Vector{Float64}
    region::Vector{Int}
    capacity_length::Float64
    required::Float64
end

"""
    MultiCommodityFlow <: ProblemGenerator

Generator for multicommodity minimum-cost flow problems (source-aggregated
commodities sharing arc capacities) on sparse geographic networks.

# Overview

Commodity `k` ships from its origin `origins[k]` (a port, plant or hub) to a
gravity-sampled set of destinations with demands `demands[k]`. All commodities
share the arc capacities.

    minimize    sum_{a,k} cost[a,k] * x[a,k]
    subject to  sum_k x[a,k] <= capacity[a]                          every arc a
                sum_out x[.,k] - sum_in x[.,k] = b[v,k]              every node v, commodity k
                x >= 0

with `b[origin_k, k] = total demand of k`, `b[v, k] = -demand` at its
destinations and 0 elsewhere. The bundle rows couple every commodity: the LP
is not a network LP and simplex must work across commodities.

# Data grounding

The network is `_geo_network` (strongly connected, 3.2-4.6 arcs per node, three
geography shapes, region side `12 sqrt(n)`). Origins are drawn by activity
weight; each commodity serves 10%-35% of the nodes with gravity demands
`w_o^0.5 * w_v / (1 + dist / L)^1.5` (mean 50 units). Per-unit cost =
arc length x commodity value factor (lognormal) x arc noise, 15% cheaper on
trunk arcs.

# Feasibility control

A planted routing sends every commodity along a shortest-path tree on its own
noisy lengths (`_geo_dijkstra` + `_geo_tree_flows`), giving per-arc loads.
Capacities are `max(floor, rho * load)` with a tiered lognormal floor:

  - `feasible`: `rho` in [1.05, 1.5]; the planted routing is the witness.
  - `infeasible`: as `feasible`, then a metric certificate is enforced —
    `:length` mode (60%): all capacities scaled down so
    `sum capacity * length` is 80%-93% of the length-weighted demand
    requirement; `:regional_cut` (40%): only the arcs leaving a region around a
    major origin are squeezed to 80%-93% of the demand that must exit it. In
    both modes no node is left starved behind its own arcs
    (`_mcf_local_repair!`), so presolve cannot refute it from one row.
  - `unknown`: capacities as `feasible`, then every demand grows by a common
    factor in [1.0, 1.15] (traffic growth since the network was provisioned);
    whether the commodities can reroute through the remaining spare capacity
    is left open — natural, either side (`_mcf_unknown_growth!`, which keeps
    every node open).

# Fields

  - `n_nodes`, `arcs` (sorted), `trunk`, `positions`, `geography`
  - `capacities::Vector{Float64}`, `costs::Matrix{Float64}` (arc x commodity)
  - `origins::Vector{Int}`, `destinations::Vector{Vector{Int}}`,
    `demands::Vector{Vector{Float64}}` (aligned with `destinations`)
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct MultiCommodityFlow <: ProblemGenerator
    n_nodes::Int
    arcs::Vector{Tuple{Int, Int}}
    trunk::Vector{Bool}
    capacities::Vector{Float64}
    costs::Matrix{Float64}
    origins::Vector{Int}
    destinations::Vector{Vector{Int}}
    demands::Vector{Vector{Float64}}
    positions::Vector{Tuple{Float64, Float64}}
    geography::Symbol
    feasible_witness::Union{Nothing, MultiCommodityFlowWitness}
    infeasibility_certificate::Union{Nothing, MultiCommodityFlowMetricCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _mcf_region(positions, center, size) -> Vector{Int}

The `size` nodes nearest to `center` (sorted indices): a geographic region.
"""
function _mcf_region(positions::Vector{Tuple{Float64, Float64}}, center::Int, size::Int)
    order = sortperm([(_geo_dist(positions, center, v), v) for v in eachindex(positions)])
    return sort(order[1:size])
end

"""
    _mcf_local_repair!(capacities, frozen, n, arcs, origins, destinations, demands; slack=1.15)
        -> Bool

Keep every node locally routable with `slack` (default 15%): each node's
in-arcs must carry `slack` times the total demand ending there and each
origin's out-arcs `slack` times the demand leaving it (`slack = 0` disables
the repair). Short nodes get their arcs enlarged (and marked `frozen`). A node
starved behind its own arcs is exactly what presolve refutes from one
conservation row (or, at a dead end, a doubleton substitution plus one bundle
row); with every node open — and `_geo_network` closing dead ends —
infeasibility is left to regional and network-wide shortages that need real
simplex work. Returns whether anything was repaired.
"""
function _mcf_local_repair!(
    capacities::Vector{Float64},
    frozen::AbstractVector{Bool},
    n::Int,
    arcs::Vector{Tuple{Int, Int}},
    origins::Vector{Int},
    destinations::Vector{Vector{Int}},
    demands::Vector{Vector{Float64}};
    slack::Float64=1.15,
)
    slack > 0 || return false
    repaired = false
    A = length(arcs)
    # Node level: every node's in-arcs carry 1.15x the total demand ending
    # there and every origin's out-arcs 1.15x the demand leaving it (a
    # dead-end node behind one arc would otherwise be refuted by presolve's
    # doubleton substitution plus one bundle row).
    out_adj, in_adj = _geo_adjacency(n, arcs)
    need_in = zeros(n)
    need_out = zeros(n)
    for k in eachindex(origins)
        need_out[origins[k]] += sum(demands[k])
        for (i, v) in enumerate(destinations[k])
            need_in[v] += demands[k][i]
        end
    end
    for v in 1:n, (need, adj) in ((need_in[v], in_adj[v]), (need_out[v], out_adj[v]))
        need > 0 || continue
        have = sum(capacities[a] for a in adj)
        have >= slack * need && continue
        g = slack * need / have
        for a in adj
            capacities[a] = ceil(capacities[a] * g; digits=2)
            frozen[a] = true
        end
        repaired = true
    end
    return repaired
end

"""
    _mcf_squeeze!(capacities, scalable, lengths, target, n, arcs, origins,
                  destinations, demands; slack=1.15) -> Float64

Scale the `scalable` arcs' capacities down until `sum(capacities .* lengths)`
reaches `target`, then re-open every starved node (`_mcf_local_repair!`,
which freezes the arcs it enlarges) and squeeze the remaining free arcs again
(at most 30 rounds). Returns the final
capacity-length product.
"""
function _mcf_squeeze!(
    capacities::Vector{Float64},
    scalable::AbstractVector{Bool},
    lengths::Vector{Float64},
    target::Float64,
    n::Int,
    arcs::Vector{Tuple{Int, Int}},
    origins::Vector{Int},
    destinations::Vector{Vector{Int}},
    demands::Vector{Vector{Float64}};
    slack::Float64=1.15,
)
    frozen = .!collect(scalable)
    for _ in 1:30
        fixed_part = sum(capacities[a] * lengths[a] for a in eachindex(arcs) if frozen[a]; init=0.0)
        free_part = sum(capacities[a] * lengths[a] for a in eachindex(arcs) if !frozen[a]; init=0.0)
        if fixed_part + free_part > target && free_part > 0
            f = max((target - fixed_part) / free_part, 0.0)
            for a in eachindex(arcs)
                frozen[a] && continue
                capacities[a] = max(floor(capacities[a] * f; digits=2), 0.01)
            end
        end
        _mcf_local_repair!(capacities, frozen, n, arcs, origins, destinations, demands; slack=slack) ||
            break
    end
    return sum(capacities .* lengths)
end

"""
    _mcf_enforce_metric!(rng, capacities, n, arcs, out_adj, positions, origins,
                         destinations, demands) -> certificate

Squeeze capacities until a metric certificate separates (capacity-length
product 80%-93% of the requirement) while no node is starved outright
(`_mcf_squeeze!`). `:regional_cut` mode (40%) squeezes only the
arcs leaving a region of 5%-25% of the nodes around the origin with the
largest demand; `:length` mode (otherwise, or as fallback when the region
cannot separate) squeezes every arc against geographic lengths. Returns the
certificate.
"""
function _mcf_enforce_metric!(
    rng::AbstractRNG,
    capacities::Vector{Float64},
    n::Int,
    arcs::Vector{Tuple{Int, Int}},
    out_adj::Vector{Vector{Int}},
    positions::Vector{Tuple{Float64, Float64}},
    origins::Vector{Int},
    destinations::Vector{Vector{Int}},
    demands::Vector{Vector{Float64}},
)
    ratio = 0.8 + 0.13 * rand(rng)
    use_cut = rand(rng) < 0.4 && n >= 10
    if use_cut
        k_big = argmax([sum(d) for d in demands])
        size = clamp(round(Int, n * (0.05 + 0.2 * rand(rng))), 5, n - 1)
        region = _mcf_region(positions, origins[k_big], size)
        inside = falses(n)
        inside[region] .= true
        lengths = [(inside[u] && !inside[v]) ? 1.0 : 0.0 for (u, v) in arcs]
        required = _mcf_metric_requirement(n, arcs, out_adj, lengths, origins, destinations, demands)
        if required > 0
            trial = copy(capacities)
            cap_len = _mcf_squeeze!(
                trial, lengths .> 0, lengths, ratio * required, n, arcs, origins, destinations, demands
            )
            if cap_len < required
                capacities .= trial
                return MultiCommodityFlowMetricCertificate(:regional_cut, lengths, region, cap_len, required)
            end
        end
    end
    lengths = [_geo_dist(positions, u, v) for (u, v) in arcs]
    required = _mcf_metric_requirement(n, arcs, out_adj, lengths, origins, destinations, demands)
    # Local repairs can eat into the squeeze; aim lower, and on tiny networks
    # (destinations a hop or two from their origins) relax the repair slack,
    # until it separates.
    original = copy(capacities)
    for slack in (1.15, 1.02, 0.0), attempt in 1:4
        capacities .= original
        cap_len = _mcf_squeeze!(
            capacities,
            trues(length(arcs)),
            lengths,
            ratio * 0.85^(attempt - 1) * required,
            n,
            arcs,
            origins,
            destinations,
            demands;
            slack=slack,
        )
        cap_len < required &&
            return MultiCommodityFlowMetricCertificate(:length, lengths, Int[], cap_len, required)
    end
    error("multi_commodity_flow: metric certificate failed to separate")
end

"""
    _mcf_unknown_growth!(rng, capacities, n, arcs, origins, destinations, demands;
                         max_growth) -> Float64

`unknown` profile: the capacities were provisioned (1.05-1.5x) for the
planted routing of the current demand; demand has since grown by a common
factor drawn from [1.0, max_growth] (rounded to cents, in place). Up to 1.05x the
planted routing still fits; beyond that the commodities must reroute through
whatever spare capacity the network has — a natural instance on either side.
The same local repair as the infeasible profile keeps every node open.
Returns the growth factor.
"""
function _mcf_unknown_growth!(
    rng::AbstractRNG,
    capacities::Vector{Float64},
    n::Int,
    arcs::Vector{Tuple{Int, Int}},
    origins::Vector{Int},
    destinations::Vector{Vector{Int}},
    demands::Vector{Vector{Float64}};
    max_growth::Float64,
)
    growth = 1.0 + (max_growth - 1.0) * rand(rng)
    for d in demands
        d .= round.(d .* growth; digits=2)
    end
    _mcf_local_repair!(capacities, falses(length(arcs)), n, arcs, origins, destinations, demands)
    return growth
end

"""
    _mcf_instance(rng, target_arcs, n_commodities; dest_share) -> NamedTuple

Shared geographic network and gravity commodities: network sized from the arc
budget, origins drawn by activity weight, destinations and demands from
`_mcf_gravity_destinations`.
"""
function _mcf_instance(rng::AbstractRNG, target_arcs::Int, n_commodities::Int; dest_share=(0.1, 0.35))
    n, n_arcs = _network_flow_dimensions(rng, target_arcs)
    while n < n_commodities + 1
        n += 1
        n_arcs = max(n_arcs, 2 * (n - 1))
    end
    geography = let r = rand(rng)
        r < 0.4 ? :clustered : (r < 0.75 ? :uniform : :corridor)
    end
    positions, weights = _geo_positions(rng, n, geography; span=12.0 * sqrt(n))
    arcs, trunk = _geo_network(rng, positions, n_arcs)
    dist = [_geo_dist(positions, u, v) for (u, v) in arcs]
    length_scale = 5.0 * median(dist)
    origins = sort(sample(rng, 1:n, Weights(weights), n_commodities; replace=false))
    destinations = Vector{Vector{Int}}(undef, n_commodities)
    raw = Vector{Vector{Float64}}(undef, n_commodities)
    for k in 1:n_commodities
        n_dest = clamp(round(Int, (n - 1) * (dest_share[1] + (dest_share[2] - dest_share[1]) * rand(rng))), 1, n - 1)
        destinations[k], raw[k] = _mcf_gravity_destinations(
            rng, n, origins[k], positions, weights, n_dest, length_scale
        )
    end
    mean_raw = sum(sum, raw) / sum(length, raw)
    demands = [max.(round.(50.0 .* r ./ mean_raw; digits=2), 0.01) for r in raw]
    return (; n, arcs, trunk, positions, weights, geography, dist, origins, destinations, demands)
end

"""
    _mcf_planted_loads(rng, n, arcs, out_adj, dist, origins, destinations, demands)
        -> (flows::Matrix, load::Vector)

Route each commodity along a shortest-path tree on its own noisy lengths.
"""
function _mcf_planted_loads(
    rng::AbstractRNG,
    n::Int,
    arcs::Vector{Tuple{Int, Int}},
    out_adj::Vector{Vector{Int}},
    dist::Vector{Float64},
    origins::Vector{Int},
    destinations::Vector{Vector{Int}},
    demands::Vector{Vector{Float64}},
)
    K = length(origins)
    flows = zeros(length(arcs), K)
    node_demand = zeros(n)
    for k in 1:K
        noisy = [dist[a] * rand(rng, LogNormal(0.0, 0.35)) + 1e-6 for a in eachindex(arcs)]
        d, pred = _geo_dijkstra(n, arcs, out_adj, noisy, [origins[k]])
        fill!(node_demand, 0.0)
        node_demand[destinations[k]] .= demands[k]
        flows[:, k] .= _geo_tree_flows(n, arcs, d, pred, node_demand)
    end
    return flows, vec(sum(flows; dims=2))
end

"""
    MultiCommodityFlow(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Variables `n_arcs * n_commodities` with `n_commodities =
clamp(round(0.4 * target^0.35), 2, 60)` (about 4 at 1k, 10 at 10k, 22 at 100k)
and `n_arcs ~ target / n_commodities` (within half a commodity of the target).
Rows `n_nodes * n_commodities + n_arcs`. Values above `MCF_MAX_VARIABLES`
raise an `ArgumentError`.
"""
function MultiCommodityFlow(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= MCF_MAX_VARIABLES || throw(
        ArgumentError(
            "multi_commodity_flow/standard supports at most $MCF_MAX_VARIABLES variables; " *
            "requested $target_variables.",
        ),
    )
    rng = MersenneTwister(seed)
    K = clamp(round(Int, 0.4 * target_variables^0.35), 2, 60)
    inst = _mcf_instance(rng, max(round(Int, target_variables / K), 2), K)
    n, arcs, trunk, positions, dist = inst.n, inst.arcs, inst.trunk, inst.positions, inst.dist
    origins, destinations, demands = inst.origins, inst.destinations, inst.demands
    A = length(arcs)
    out_adj, _ = _geo_adjacency(n, arcs)

    value_factor = [rand(rng, LogNormal(0.0, 0.35)) for _ in 1:K]
    arc_noise = [rand(rng, LogNormal(0.0, 0.2)) * (trunk[a] ? 0.85 : 1.0) for a in 1:A]
    costs = [round((dist[a] * arc_noise[a] + 0.1) * value_factor[k]; digits=3) for a in 1:A, k in 1:K]

    flows, load = _mcf_planted_loads(rng, n, arcs, out_adj, dist, origins, destinations, demands)
    used = filter(>(0.0), load)
    floor_scale = (0.1 + 0.3 * rand(rng)) * (isempty(used) ? 50.0 : median(used))
    rho_range = (1.05, 1.5)
    capacities = [
        ceil(
            max(
                floor_scale * rand(rng, LogNormal(0.0, 0.6)) * (trunk[a] ? 1.6 : 1.0),
                (rho_range[1] + (rho_range[2] - rho_range[1]) * rand(rng)) * load[a],
                0.01,
            );
            digits=2,
        ) for a in 1:A
    ]

    if feasibility_status == unknown
        _mcf_unknown_growth!(rng, capacities, n, arcs, origins, destinations, demands; max_growth=1.15)
    end

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        feasible_witness = MultiCommodityFlowWitness(flows)
    elseif feasibility_status == infeasible
        infeasibility_certificate = _mcf_enforce_metric!(
            rng, capacities, n, arcs, out_adj, positions, origins, destinations, demands
        )
    end

    return MultiCommodityFlow(
        n,
        arcs,
        trunk,
        capacities,
        costs,
        origins,
        destinations,
        demands,
        positions,
        inst.geography,
        feasible_witness,
        infeasibility_certificate,
        feasibility_status,
    )
end

"""
    _mcf_supply_matrix(n, origins, destinations, demands) -> Matrix{Float64}

Node x commodity net supplies `b[v, k]` (out minus in).
"""
function _mcf_supply_matrix(n::Int, origins, destinations, demands)
    b = zeros(n, length(origins))
    for k in eachindex(origins)
        b[origins[k], k] += sum(demands[k])
        for (i, v) in enumerate(destinations[k])
            b[v, k] -= demands[k][i]
        end
    end
    return b
end

"""
    build_model(prob::MultiCommodityFlow)

Build the multicommodity min-cost flow LP. Deterministic — uses only the
struct fields.
"""
function build_model(prob::MultiCommodityFlow)
    model = Model()
    A = length(prob.arcs)
    K = length(prob.origins)
    n = prob.n_nodes
    @variable(model, x[1:A, 1:K] >= 0)
    @objective(model, Min, sum(prob.costs[a, k] * x[a, k] for a in 1:A, k in 1:K))
    for a in 1:A
        @constraint(model, sum(x[a, k] for k in 1:K) <= prob.capacities[a])
    end
    out_adj, in_adj = _geo_adjacency(n, prob.arcs)
    b = _mcf_supply_matrix(n, prob.origins, prob.destinations, prob.demands)
    for k in 1:K, v in 1:n
        @constraint(
            model,
            sum(x[a, k] for a in out_adj[v]; init=AffExpr(0.0)) -
            sum(x[a, k] for a in in_adj[v]; init=AffExpr(0.0)) == b[v, k]
        )
    end
    return model
end

register_variant(
    :multi_commodity_flow,
    :standard,
    MultiCommodityFlow,
    "Multicommodity min-cost flow on a sparse geographic network: origin-aggregated gravity commodities sharing arc capacities through bundle rows, planted shortest-path routing as witness, and metric-inequality (length or regional-cut) infeasibility certificates";
    default=true,
)
