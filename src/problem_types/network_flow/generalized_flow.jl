using JuMP
using Random
using Distributions
using StatsBase

"""
Planted lossy routing: a genuine feasible point of the built model. Every demand
node is served from a supply node along a shortest-path tree of near-most-
efficient routes; the flow SENT on each tree arc is what must arrive at its head divided
by the arc gain, so generalized conservation holds exactly at every transit and
demand node, `source_outflow[i]` is what supply node `supply_nodes[i]` ships
(at most its supply), and every arc carries at most its capacity.
"""
struct GeneralizedFlowWitness
    arc_flows::Vector{Float64}
    source_outflow::Vector{Float64}
end

"""
Loss-adjusted supply-adequacy certificate (a Farkas certificate built from node
potentials). `efficiency[v]` is the best achievable delivery efficiency from any
supply node to `v`: the largest product of arc gains over a path, computed by
Dijkstra on `-log(gain)` lengths, so `efficiency[s] = 1` at supply nodes and
`efficiency[j] >= efficiency[i] * gain[a]` for every arc `a = (i, j)`.

Multiply node `v`'s balance row by `1 / efficiency[v]` and add: every arc's
column gets coefficient `gain[a] / efficiency[j] - 1 / efficiency[i] <= 0`, so
the weighted sum of all node rows says

    sum_v demand[v] / efficiency[v]  <=  sum_s supply[s]

for any feasible flow, whatever the capacities. The certificate stores both
sides, with `required_supply > total_supply` by at least 20% of the loss volume
`required_supply - total_demand`: the network loses more in transit than the supply surplus can cover.
No single row or bound shows this, so presolve cannot detect it.
"""
struct GeneralizedFlowLossCertificate
    efficiency::Vector{Float64}
    required_supply::Float64
    total_supply::Float64
end

"""
    GeneralizedFlowProblem <: ProblemGenerator

Generator for generalized (lossy) minimum-cost flow problems on sparse
geographic networks.

# Overview

Each arc `a = (i, j)` has a gain `gain[a] in (0, 1)`: of `flow[a]` units sent,
only `gain[a] * flow[a]` arrive at `j` (transmission/line losses, pipeline
leakage, evaporation, spoilage). Balance rows:

    supply node v:   sum_out flow - sum_in gain*flow <= supply[v]
    other node v:    sum_in gain*flow - sum_out flow  = demand[v]   (0 at transit)

with `0 <= flow[a] <= capacity[a]` as variable bounds and minimum total routing
cost. Gains below one destroy total unimodularity, so vertices are genuinely
fractional and simplex must do real work (the classic generalized-flow family).
Antiparallel arc pairs never form gain-amplifying cycles (every gain is < 1).

# Data grounding

The same sparse, strongly connected geographic networks as `network_flow/standard`
(`_geo_network`, about 3-5 arcs per node, three geography shapes). Gains decay
exponentially with arc length times a lognormal per-arc factor (line quality),
calibrated per instance so the median best-route delivery efficiency to demand
nodes is 72%-90%; gains are stored to 4 digits and capped at 0.9995. Costs are
distance-proportional with lognormal route noise (so cheap routes and
efficient routes disagree); capacities are sized from the planted lossy routing
with a provisioning factor of 1.05-1.55 plus a tiered lognormal floor.

# Feasibility control

The planted routing (a shortest-path tree from all supply nodes on loss lengths
`-log(gain)` with mild lognormal noise, `_geo_tree_flows` with gains) is a
concrete lossy flow; capacities always cover it, and its per-site supply draw
is close to the loss-adjusted minimum.

  - `feasible`: each supply node gets at least 1.05-1.6x what the planted routing
    draws from it; the planted routing is stored as the witness.
  - `infeasible`: total supply is placed 20%-80% of the way from the lossless
    total demand to the loss-adjusted requirement `sum demand/efficiency` (so
    the naive supply >= demand check passes), spread over sites in proportion
    to their planted draw with mild noise; the efficiency potentials are the
    certificate. No region is short in isolation, so presolve cannot see it.
  - `unknown`: total supply is 1.0-1.35x the loss-adjusted requirement, spread
    by planted draw with lognormal site noise (sigma 0.15): some sites run
    short and must be relieved through the network, which may or may not have
    the capacity — a natural instance on either side.

# Fields

  - `n_nodes`, `arcs` (sorted), `trunk`, `positions`, `geography`
  - `capacities`, `costs`, `gains`: aligned with `arcs`
  - `supplies`, `demands`: per node; `supply_nodes`, `demand_nodes`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct GeneralizedFlowProblem <: ProblemGenerator
    n_nodes::Int
    arcs::Vector{Tuple{Int, Int}}
    trunk::Vector{Bool}
    capacities::Vector{Float64}
    costs::Vector{Float64}
    gains::Vector{Float64}
    supplies::Vector{Float64}
    demands::Vector{Float64}
    supply_nodes::Vector{Int}
    demand_nodes::Vector{Int}
    positions::Vector{Tuple{Float64, Float64}}
    geography::Symbol
    feasible_witness::Union{Nothing, GeneralizedFlowWitness}
    infeasibility_certificate::Union{Nothing, GeneralizedFlowLossCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _generalized_flow_efficiency(n, arcs, out_adj, gains, supply_nodes) -> Vector{Float64}

Best delivery efficiency from any supply node to every node: the maximum
product of gains over a path, via Dijkstra on `-log(gain)`.
"""
function _generalized_flow_efficiency(
    n::Int,
    arcs::Vector{Tuple{Int, Int}},
    out_adj::Vector{Vector{Int}},
    gains::Vector{Float64},
    supply_nodes::Vector{Int},
)
    d, _ = _geo_dijkstra(n, arcs, out_adj, [-log(g) for g in gains], supply_nodes)
    return exp.(-d)
end

"""
    GeneralizedFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a generalized-flow instance with exactly `target_variables` arcs
(= variables; targets below 2 round up to 2 and a target of 3 to 4). Values
above `NETWORK_FLOW_MAX_ARCS` raise an `ArgumentError`. Rows: one balance row
per node (about a quarter of the arcs).
"""
function GeneralizedFlowProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= NETWORK_FLOW_MAX_ARCS || throw(
        ArgumentError(
            "network_flow/generalized_flow supports at most $NETWORK_FLOW_MAX_ARCS arcs; " *
            "requested $target_variables.",
        ),
    )
    rng = MersenneTwister(seed)

    n, n_arcs = _network_flow_dimensions(rng, target_variables)
    geography = let r = rand(rng)
        r < 0.4 ? :clustered : (r < 0.75 ? :uniform : :corridor)
    end
    positions, weights = _geo_positions(rng, n, geography; span=12.0 * sqrt(n))
    arcs, trunk = _geo_network(rng, positions, n_arcs)
    m = length(arcs)
    dist = [_geo_dist(positions, u, v) for (u, v) in arcs]
    out_adj, _ = _geo_adjacency(n, arcs)

    n_supply = clamp(round(Int, n * (0.04 + 0.08 * rand(rng))), 1, n - 1)
    supply_nodes = sort(sample(rng, 1:n, n_supply; replace=false))
    rest = setdiff(1:n, supply_nodes)
    n_demand = clamp(round(Int, n * (0.25 + 0.25 * rand(rng))), 1, length(rest))
    demand_nodes = sort(sample(rng, rest, Weights(weights[rest]), n_demand; replace=false))
    wd = weights[demand_nodes]
    demand_vals = round.(50.0 .* wd ./ (sum(wd) / length(wd)); digits=2)
    demand_vals = max.(demand_vals, 0.01)
    demands = zeros(n)
    demands[demand_nodes] .= demand_vals
    supply_weight = [rand(rng, LogNormal(0.0, 0.6)) for _ in supply_nodes]

    # Gains: exponential decay in length x line-quality factor, calibrated so
    # the median best-route efficiency to the demand nodes hits a target.
    loss_length = [max(dist[k], 1e-3) * rand(rng, LogNormal(0.0, 0.35)) for k in 1:m]
    raw_dist, _ = _geo_dijkstra(n, arcs, out_adj, loss_length, supply_nodes)
    target_eff = 0.72 + 0.18 * rand(rng)
    ref = median(raw_dist[demand_nodes])
    alpha = ref > 0 ? -log(target_eff) / ref : 0.01
    gains = [clamp(round(exp(-alpha * loss_length[k]); digits=4), 0.5, 0.9995) for k in 1:m]

    route_spread = 0.2 + 0.25 * rand(rng)
    costs = [
        round(
            max(dist[k], 0.05) * (trunk[k] ? 0.8 : 1.0) * rand(rng, LogNormal(0.0, route_spread)) +
            0.05;
            digits=3,
        ) for k in 1:m
    ]

    # Historical lossy routing over noisy lengths: the planted plan.
    # Operators route along near-most-efficient paths (loss length with mild
    # noise), so the plan's supply draw is close to the loss-adjusted minimum.
    hist_len = [-log(gains[k]) * rand(rng, LogNormal(0.0, 0.15)) + 1e-9 for k in 1:m]
    hdist, hpred = _geo_dijkstra(n, arcs, out_adj, hist_len, supply_nodes)
    plan = _geo_tree_flows(n, arcs, hdist, hpred, demands; gains=gains)
    source_draw = zeros(length(supply_nodes))
    supply_index = Dict(s => i for (i, s) in enumerate(supply_nodes))
    for (k, (u, _)) in enumerate(arcs)
        haskey(supply_index, u) && (source_draw[supply_index[u]] += plan[k])
    end

    used = filter(>(0.0), plan)
    floor_scale = (0.3 + 0.5 * rand(rng)) * (isempty(used) ? 50.0 : median(used))
    capacities = Vector{Float64}(undef, m)
    for k in 1:m
        floor_cap = floor_scale * rand(rng, LogNormal(0.0, 0.6)) * (trunk[k] ? 1.6 : 1.0)
        # ceil to 2 digits keeps capacity >= provision * plan exactly.
        capacities[k] = ceil(max(floor_cap, (1.05 + 0.5 * rand(rng)) * plan[k], 0.01); digits=2)
    end

    efficiency = _generalized_flow_efficiency(n, arcs, out_adj, gains, supply_nodes)
    required = sum(demands[v] / efficiency[v] for v in demand_nodes)
    lossless = sum(demand_vals)

    feasible_witness = nothing
    infeasibility_certificate = nothing
    supply_caps = if feasibility_status == feasible
        caps = [
            ceil(max(source_draw[i] * (1.05 + 0.55 * rand(rng)), 0.01); digits=2) for
            i in eachindex(supply_nodes)
        ]
        feasible_witness = GeneralizedFlowWitness(plan, source_draw)
        caps
    else
        total = if feasibility_status == infeasible
            lossless + (0.2 + 0.6 * rand(rng)) * (required - lossless)
        else
            required * (1.0 + 0.35 * rand(rng))
        end
        # Spread by where the historical routing drew supply (sites are sized
        # for their market), with lognormal site noise.
        noise = feasibility_status == infeasible ? 0.1 : 0.15
        share = [
            (source_draw[i] + 1e-9) * supply_weight[i]^(noise / 0.6) for i in eachindex(supply_nodes)
        ]
        caps = floor.(total .* share ./ sum(share); digits=2)
        max.(caps, 0.01)
    end
    supplies = zeros(n)
    supplies[supply_nodes] .= supply_caps

    if feasibility_status == infeasible
        total_supply = sum(supply_caps)
        total_supply < required ||
            error("generalized_flow: loss certificate failed to separate (seed $seed)")
        infeasibility_certificate = GeneralizedFlowLossCertificate(efficiency, required, total_supply)
    end

    return GeneralizedFlowProblem(
        n,
        arcs,
        trunk,
        capacities,
        costs,
        gains,
        supplies,
        demands,
        supply_nodes,
        demand_nodes,
        positions,
        geography,
        feasible_witness,
        infeasibility_certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::GeneralizedFlowProblem)

Build the generalized min-cost flow LP. Deterministic — uses only the struct
fields.

  - `flow[k] in [0, capacities[k]]`: flow SENT on arc `k` (variables == arcs);
    `gains[k] * flow[k]` arrives at the head
  - one balance row per node (supply rows `<=`, all others `==`)
"""
function build_model(prob::GeneralizedFlowProblem)
    model = Model()
    m = length(prob.arcs)
    n = prob.n_nodes

    @variable(model, 0 <= flow[k = 1:m] <= prob.capacities[k])
    @objective(model, Min, sum(prob.costs[k] * flow[k] for k in 1:m))

    out_adj, in_adj = _geo_adjacency(n, prob.arcs)
    is_supply = falses(n)
    is_supply[prob.supply_nodes] .= true
    for v in 1:n
        net_in = sum(prob.gains[k] * flow[k] for k in in_adj[v]; init=AffExpr(0.0)) -
            sum(flow[k] for k in out_adj[v]; init=AffExpr(0.0))
        if is_supply[v]
            @constraint(model, -net_in <= prob.supplies[v])
        else
            @constraint(model, net_in == prob.demands[v])
        end
    end
    return model
end

register_variant(
    :network_flow,
    :generalized_flow,
    GeneralizedFlowProblem,
    "Generalized (lossy) min-cost flow on a sparse geographic network: distance-decaying arc gains below one, supply/demand/transit balance rows, a planted lossy routing as witness, and a loss-adjusted supply-adequacy (node-potential Farkas) certificate",
)
