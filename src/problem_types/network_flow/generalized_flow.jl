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
sides, with `total_supply` 3%-8% below `required_supply` (and, whenever the
losses allow, above the lossless total demand): the network loses more in transit than the supply surplus can cover.
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
  - `infeasible`: total supply is 3%-8% below the loss-adjusted requirement
    `sum demand/efficiency` (and above the lossless total demand whenever the
    losses allow, so the naive supply >= demand check passes): every site
    starts at its planted draw and the shortfall is taken in proportion to
    draw x out-degree^2 (the best-connected hubs run short, not a district's
    only source), then locally repaired (`_generalized_flow_local_repair!`:
    no demand node refutable by one step of bound propagation, total
    unchanged); the efficiency potentials are the certificate.
  - `unknown`: every site holds a common reserve factor in [0.9, 1.1] of its
    planted draw (4% site noise), starved sites topped up by the same local
    check: below 1 the routing must beat the planted (near-efficient, noisy)
    paths, above 1 it has slack — a natural instance on either side.

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
    _generalized_flow_local_repair!(supplies, n, arcs, capacities, gains, demands, supply_nodes;
                                    keep_total=true)

Keep every demand node deliverable under ONE step of bound propagation, the
reasoning presolve applies: an arc can carry at most its capacity and at most
what its tail can pass on (a supply site: its supply plus its lossy inflow
capacity; a transit node: its inflow capacity; a demand node: that minus its
own demand). Where a demand node's post-gain intake under these bounds falls
below 1.3x its demand (slack for sites feeding several neighbours) because a
supplying site is short, that site's supply is
raised and (with `keep_total`) the same total is taken back proportionally
from the sites not involved (at most 20 rounds), so the total — and hence the
loss-adjusted certificate — is unchanged while no local pair of rows refutes
the model.
"""
function _generalized_flow_local_repair!(
    supplies::Vector{Float64},
    n::Int,
    arcs::Vector{Tuple{Int, Int}},
    capacities::Vector{Float64},
    gains::Vector{Float64},
    demands::Vector{Float64},
    supply_nodes::Vector{Int};
    keep_total::Bool=true,
)
    out_adj, in_adj = _geo_adjacency(n, arcs)
    is_supply = falses(n)
    is_supply[supply_nodes] .= true
    total = sum(supplies)
    inflow_cap = [sum(gains[a] * capacities[a] for a in in_adj[v]; init=0.0) for v in 1:n]
    for _ in 1:20
        touched = falses(n)
        for v in 1:n
            demands[v] > 0 || continue
            avail(u) =
                if is_supply[u]
                    supplies[u] + inflow_cap[u]
                else
                    (demands[u] > 0 ? max(inflow_cap[u] - demands[u], 0.0) : inflow_cap[u])
                end
            intake = sum(
                gains[a] * min(capacities[a], avail(arcs[a][1])) for a in in_adj[v]; init=0.0
            )
            deficit = 1.3 * demands[v] - intake
            deficit > 0 || continue
            for a in in_adj[v]
                u = arcs[a][1]
                is_supply[u] || continue
                room = capacities[a] - avail(u)
                room > 0 || continue
                raise = min(room, deficit / gains[a])
                supplies[u] = ceil(supplies[u] + raise; digits=2)
                touched[u] = true
                deficit -= gains[a] * raise
                deficit <= 0 && break
            end
        end
        any(touched) || break
        keep_total || continue
        # Give the total back from the untouched sites.
        free = [u for u in supply_nodes if !touched[u]]
        isempty(free) && break
        excess = sum(supplies) - total
        pool = sum(supplies[free])
        excess >= pool && break
        for u in free
            supplies[u] = max(floor(supplies[u] * (1 - excess / pool); digits=2), 0.01)
        end
    end
    return supplies
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
            # 3%-8% below the loss-adjusted requirement, and when possible
            # above the lossless demand (the naive check passes).
            t = required * (0.92 + 0.05 * rand(rng))
            t < 1.005 * lossless < 0.99 * required ? 1.005 * lossless : t
        else
            0.0  # unused: unknown supplies are set per site below
        end
        if feasibility_status == infeasible
            # Every site starts at its planted draw; the shortfall is taken
            # from sites in proportion to draw x out-degree^2, so the
            # best-connected hubs (whose customers have alternatives) run
            # short rather than a site that is some district's only source.
            deg = [length(out_adj[u]) for u in supply_nodes]
            w = [
                source_draw[i] * deg[i]^2 * supply_weight[i]^(0.1 / 0.6) for
                i in eachindex(supply_nodes)
            ]
            shortfall = sum(source_draw) - total
            caps = [source_draw[i] - shortfall * w[i] / sum(w) for i in eachindex(supply_nodes)]
            if minimum(caps) < 0.05 * maximum(source_draw)
                # Rare: fall back to a plain proportional cut.
                caps = total .* source_draw ./ sum(source_draw)
            end
            max.(floor.(caps; digits=2), 0.01)
        else
            # Each site sized at a common reserve factor of its planted draw
            # (with 4% site noise): below 1 the routing must find more
            # efficient paths than the planted ones, above 1 it has slack.
            phi = 0.9 + 0.2 * rand(rng)
            [
                max(floor(source_draw[i] * phi * rand(rng, LogNormal(0.0, 0.04)); digits=2), 0.01) for i in eachindex(supply_nodes)
            ]
        end
    end
    supplies = zeros(n)
    supplies[supply_nodes] .= supply_caps
    if feasibility_status != feasible
        # Infeasible: keep the total (the certificate depends on it). Unknown:
        # just top up starved sites (a natural instance).
        repaired = _generalized_flow_local_repair!(
            copy(supplies),
            n,
            arcs,
            capacities,
            gains,
            demands,
            supply_nodes;
            keep_total=feasibility_status == infeasible,
        )
        # The repair keeps the total only approximately (it rounds to cents,
        # floors sites at 0.01 and stops early when nothing is left to give
        # back), so it must never lift an infeasible instance over the
        # loss-adjusted requirement; small networks keep the plain cut then.
        if feasibility_status == unknown || sum(repaired[supply_nodes]) < required
            supplies = repaired
        end
    end

    if feasibility_status == infeasible
        # The certificate is about the supplies the model actually carries.
        total_supply = sum(supplies[supply_nodes])
        total_supply < required ||
            error("generalized_flow: loss certificate failed to separate (seed $seed)")
        infeasibility_certificate = GeneralizedFlowLossCertificate(
            efficiency, required, total_supply
        )
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
        net_in =
            sum(prob.gains[k] * flow[k] for k in in_adj[v]; init=AffExpr(0.0)) -
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
    "Generalized (lossy) min-cost flow on a sparse geographic network: distance-decaying arc gains below one, supply/demand/transit balance rows, a planted lossy routing as witness, and a loss-adjusted supply-adequacy (node-potential Farkas) certificate";
    tags=[:energy, :network],
    max_target_variables=1_000_000,
)
