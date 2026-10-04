using JuMP
using Random
using Distributions
using StatsBase
using Statistics

"""
Largest `target_variables` accepted by `NetworkFlowProblem` (variables are the
directed arcs). The constructor keeps several arc-indexed structures (the
network, the residual graph of the exact max-flow computation, shortest-path
trees), so larger targets are rejected with an `ArgumentError` instead of being
silently undersized (same convention as `telecom_network_design/standard` and
`supply_chain/network_planning`).
"""
const NETWORK_FLOW_MAX_ARCS = 1_000_000

"""
Planted flow plan: a genuine feasible point of the built model. It is the arc
part of an exact maximum flow on the extended network (super source -> supply
nodes with the supply caps, demand nodes -> super sink with the demands); the
max flow equals total demand, so every demand arc is saturated, every supply
node ships at most its supply, and every arc carries at most its capacity.
"""
struct NetworkFlowWitness
    arc_flows::Vector{Float64}
end

"""
Gale/Hoffman cut certificate of infeasibility. `region` is a node set `T` (read
off the residual graph of an exact max flow) whose demand cannot be met even if
every unit of supply located inside `T` and every inbound arc's full capacity
were devoted to it:

    region_demand > region_supply + inbound_capacity

`inbound_arcs` are exactly the arcs with tail outside `T` and head inside it.
Summing the node rows over `T` (demand rows `in - out = d`, supply rows
`out - in <= s`, transit rows `in - out = 0`) gives
`inflow(T) - outflow(T) >= region_demand - region_supply`, while the arc bounds
give `inflow(T) - outflow(T) <= inbound_capacity` — a contradiction from LP rows
and bounds alone. `T` is typically a whole under-connected region, not a single
node, so presolve's single-row reasoning does not see it.
"""
struct NetworkFlowCutCertificate
    region::Vector{Int}
    inbound_arcs::Vector{Int}
    inbound_capacity::Float64
    region_demand::Float64
    region_supply::Float64
end

"""
    NetworkFlowProblem <: ProblemGenerator

Generator for single-commodity minimum-cost flow problems (the NETGEN-style
transshipment LP) on sparse geographic networks.

# Overview

A strongly connected road/pipeline-like digraph (about 3-5 arcs per node, built
by `_geo_network`) carries one commodity from supply nodes (plants, terminals,
reservoirs) to demand nodes (cities, depots) through transit junctions. Every
node has one balance row:

    supply node v:   sum_out flow - sum_in flow <= supply[v]
    other node v:    sum_in flow - sum_out flow  = demand[v]      (0 at transit)

and every arc a bounded flow `0 <= flow[a] <= capacity[a]` (capacities are
variable bounds, not rows). The objective minimizes total routing cost.

# Data grounding

Nodes are scattered over a region whose side grows with `sqrt(n_nodes)` (so arc
lengths keep realistic magnitudes at every size) in one of three geography
shapes (`:uniform`, `:clustered` metropolitan clusters with Zipf-like sizes,
`:corridor`). Node activity weights are heavy-tailed; demand nodes are drawn in
proportion to them and their demands follow them. Per-unit cost is distance
times a lognormal route factor (trunk links cheaper per km). Capacities are
sized from a "historical" routing of the nominal demand on noisy route lengths
(infrastructure is built where traffic used to go) with a wide lognormal
provisioning factor, plus a tiered lognormal floor — so the cost-optimal routing
conflicts with the installed capacities and many arcs bind.

# Feasibility control

The constructor computes the EXACT largest uniform demand scale `lambda*` the
network can serve (Dinkelbach min-ratio-cut iterations over exact Dinic max flows on the extended
network), first with unlimited supplies (to size total supply above the network
limit) and then with the sampled supplies. Demands are the nominal profile times
`lambda = load_factor * lambda*`:

  - `feasible`: `load_factor` in [0.55, 0.9]; witness = exact max-flow plan.
  - `infeasible`: `load_factor` in [1.08, 1.3]; certificate = the min-cut region
    `T` (a Gale/Hoffman violation) read off the residual graph.
  - `unknown`: `load_factor` in [0.85, 1.15]: a natural instance on either side
    of the exact boundary; the stored `max_flow_value` vs `total_demand`
    decides it.

# Fields

  - `n_nodes::Int`, `arcs::Vector{Tuple{Int,Int}}` (sorted), `trunk::Vector{Bool}`
    (spanning-tree arcs), `positions`, `geography::Symbol`
  - `capacities::Vector{Float64}`, `costs::Vector{Float64}`: aligned with `arcs`
  - `supplies::Vector{Float64}`, `demands::Vector{Float64}`: per node (zero where
    the node has no such role); `supply_nodes`, `demand_nodes` (sorted, disjoint)
  - `load_factor::Float64`: `lambda / lambda*`
  - `max_flow_value::Float64`: exact max deliverable total for the final data
  - `total_demand::Float64`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct NetworkFlowProblem <: ProblemGenerator
    n_nodes::Int
    arcs::Vector{Tuple{Int, Int}}
    trunk::Vector{Bool}
    capacities::Vector{Float64}
    costs::Vector{Float64}
    supplies::Vector{Float64}
    demands::Vector{Float64}
    supply_nodes::Vector{Int}
    demand_nodes::Vector{Int}
    positions::Vector{Tuple{Float64, Float64}}
    geography::Symbol
    load_factor::Float64
    max_flow_value::Float64
    total_demand::Float64
    feasible_witness::Union{Nothing, NetworkFlowWitness}
    infeasibility_certificate::Union{Nothing, NetworkFlowCutCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _network_flow_dimensions(rng, target) -> (n_nodes, n_arcs)

Node count for a sparse network with about 3.2-4.6 arcs per node and an arc
budget equal to `target`, adjusted so `2(n-1) <= n_arcs <= n(n-1)` (the spanning
tree in both directions must fit). Only tiny targets (below 2, or 3) round up.
"""
function _network_flow_dimensions(rng::AbstractRNG, target::Int)
    per_node = 3.2 + 1.4 * rand(rng)
    n = clamp(round(Int, target / per_node), 2, max(2, target ÷ 2 + 1))
    while n * (n - 1) < target
        n += 1
    end
    return n, max(target, 2 * (n - 1))
end

"""
    _network_flow_extended(n, arcs, caps, supply_nodes, supply_caps, demand_nodes, demand_caps)

Max flow on the extended network (super source `n+1` feeding the supply nodes,
super sink `n+2` fed by the demand nodes). Returns the full `_flow_max_flow`
tuple; arc indices `1:length(arcs)` are the real arcs.
"""
function _network_flow_extended(
    n::Int,
    arcs::Vector{Tuple{Int, Int}},
    caps::Vector{Float64},
    supply_nodes::Vector{Int},
    supply_caps::Vector{Float64},
    demand_nodes::Vector{Int},
    demand_caps::Vector{Float64},
)
    ext_arcs = vcat(arcs, [(n + 1, s) for s in supply_nodes], [(v, n + 2) for v in demand_nodes])
    ext_caps = vcat(caps, supply_caps, demand_caps)
    return _flow_max_flow(n + 2, n + 1, n + 2, ext_arcs, ext_caps)
end

"""
    _network_flow_max_scale(n, arcs, caps, supply_nodes, supply_caps, demand_nodes, d0)

EXACT largest uniform scale `lambda*` such that demands `lambda * d0` are fully
deliverable. By Gale's theorem `lambda* = min_T (supply(T) + cap_in(T)) / d0(T)`
over node sets `T`, and a max flow at any `lambda` exposes the most violated `T`
as its min cut, so Dinkelbach's iteration `lambda <- ratio(T)` from an upper
bound decreases monotonically to `lambda*` in a handful of max flows. Returns
`lambda*` shaved by `1e-9` (relative) so it is deliverable despite rounding.
"""
function _network_flow_max_scale(
    n::Int,
    arcs::Vector{Tuple{Int, Int}},
    caps::Vector{Float64},
    supply_nodes::Vector{Int},
    supply_caps::Vector{Float64},
    demand_nodes::Vector{Int},
    d0::Vector{Float64},
)
    D0 = sum(d0)
    supply_of = zeros(n)
    supply_of[supply_nodes] .= supply_caps
    d0_of = zeros(n)
    d0_of[demand_nodes] .= d0
    inbound = zeros(n)
    for (k, (_, v)) in enumerate(arcs)
        inbound[v] += caps[k]
    end
    # Upper bound from single-node cuts and total supply.
    lambda = min(sum(supply_caps) / D0, minimum(inbound[v] / d0_of[v] for v in demand_nodes))
    for _ in 1:100
        value, _, side, _, _ = _network_flow_extended(
            n, arcs, caps, supply_nodes, supply_caps, demand_nodes, lambda .* d0
        )
        value >= lambda * D0 * (1 - 1e-9) && break
        on_side = falses(n + 2)
        on_side[side] .= true
        num = 0.0
        den = 0.0
        for v in 1:n
            on_side[v] && continue
            num += supply_of[v]
            den += d0_of[v]
        end
        for (k, (u, v)) in enumerate(arcs)
            (on_side[u] && !on_side[v]) && (num += caps[k])
        end
        ratio = num / den
        # Dinkelbach strictly decreases lambda; guard against float stalls.
        lambda = min(ratio, lambda * (1 - 1e-9))
    end
    return lambda * (1 - 1e-9)
end

"""
    NetworkFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a geographic min-cost flow instance with exactly `target_variables`
arcs (= variables; targets below 2 round up to the 2-arc minimum network, and a
target of 3 to 4). Values above `NETWORK_FLOW_MAX_ARCS` raise an
`ArgumentError`. Rows: one balance row per node (about a quarter of the arcs).
"""
function NetworkFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= NETWORK_FLOW_MAX_ARCS || throw(
        ArgumentError(
            "network_flow/standard supports at most $NETWORK_FLOW_MAX_ARCS arcs; " *
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

    # Node roles: supply sites are resource locations (uniform draw), demand
    # sites follow activity weight (cities consume).
    n_supply = clamp(round(Int, n * (0.04 + 0.08 * rand(rng))), 1, n - 1)
    supply_nodes = sort(sample(rng, 1:n, n_supply; replace=false))
    rest = setdiff(1:n, supply_nodes)
    n_demand = clamp(round(Int, n * (0.25 + 0.25 * rand(rng))), 1, length(rest))
    demand_nodes = sort(sample(rng, rest, Weights(weights[rest]), n_demand; replace=false))

    # Nominal demand profile (mean 50 units per demand node, heavy-tailed).
    wd = weights[demand_nodes]
    d0 = 50.0 .* wd ./ (sum(wd) / length(wd))
    supply_weight = [rand(rng, LogNormal(0.0, 0.6)) for _ in supply_nodes]

    # Costs: distance x route factor, trunk links cheaper per km.
    route_spread = 0.2 + 0.25 * rand(rng)
    costs = [
        round(
            max(dist[k], 0.05) * (trunk[k] ? 0.8 : 1.0) * rand(rng, LogNormal(0.0, route_spread)) +
            0.05;
            digits=3,
        ) for k in 1:m
    ]

    # Capacities from a historical routing of the nominal demand over noisy
    # route lengths, with a wide provisioning factor (some arcs under-built)
    # and a tiered floor.
    out_adj, _ = _geo_adjacency(n, arcs)
    hist_len = [dist[k] * rand(rng, LogNormal(0.0, 0.5)) + 1e-6 for k in 1:m]
    hdist, hpred = _geo_dijkstra(n, arcs, out_adj, hist_len, supply_nodes)
    node_demand = zeros(n)
    node_demand[demand_nodes] .= d0
    hist_load = _geo_tree_flows(n, arcs, hdist, hpred, node_demand)
    used = filter(>(0.0), hist_load)
    floor_scale = (0.15 + 0.35 * rand(rng)) * (isempty(used) ? 50.0 : median(used))
    capacities = Vector{Float64}(undef, m)
    for k in 1:m
        provision = rand(rng, LogNormal(log(1.2), 0.45))
        floor_cap = floor_scale * rand(rng, LogNormal(0.0, 0.6)) * (trunk[k] ? 1.6 : 1.0)
        capacities[k] = round(max(floor_cap, provision * hist_load[k], 0.01); digits=2)
    end

    # Exact network limit with unlimited supply, then size supply above it so
    # the binding limits are the network's, then the exact limit with supplies.
    unlimited = fill(sum(capacities) + 1.0, n_supply)
    lambda_net = _network_flow_max_scale(
        n, arcs, capacities, supply_nodes, unlimited, demand_nodes, d0
    )
    total_supply = (1.15 + 0.45 * rand(rng)) * lambda_net * sum(d0)
    supply_caps = round.(total_supply .* supply_weight ./ sum(supply_weight); digits=2)
    supply_caps = max.(supply_caps, 0.01)
    lambda_star = _network_flow_max_scale(
        n, arcs, capacities, supply_nodes, supply_caps, demand_nodes, d0
    )

    load_factor = if feasibility_status == feasible
        0.55 + 0.35 * rand(rng)
    elseif feasibility_status == infeasible
        1.08 + 0.22 * rand(rng)
    else
        0.85 + 0.3 * rand(rng)
    end
    demand_vals = max.(round.(load_factor * lambda_star .* d0; digits=2), 0.01)

    value, ext_flows, source_side, _, _ = _network_flow_extended(
        n, arcs, capacities, supply_nodes, supply_caps, demand_nodes, demand_vals
    )
    total_demand = sum(demand_vals)

    supplies = zeros(n)
    supplies[supply_nodes] .= supply_caps
    demands = zeros(n)
    demands[demand_nodes] .= demand_vals

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        value >= total_demand * (1 - 1e-9) ||
            error("network_flow/standard: planted feasible load is not deliverable (seed $seed)")
        feasible_witness = NetworkFlowWitness(ext_flows[1:m])
    elseif feasibility_status == infeasible
        value < total_demand * (1 - 1e-6) ||
            error("network_flow/standard: infeasible load is deliverable (seed $seed)")
        on_source_side = falses(n + 2)
        on_source_side[source_side] .= true
        region = [v for v in 1:n if !on_source_side[v]]
        inbound_arcs = [
            k for (k, (u, v)) in enumerate(arcs) if on_source_side[u] && !on_source_side[v]
        ]
        infeasibility_certificate = NetworkFlowCutCertificate(
            region,
            inbound_arcs,
            sum(capacities[k] for k in inbound_arcs; init=0.0),
            sum(demands[v] for v in region; init=0.0),
            sum(supplies[v] for v in region; init=0.0),
        )
    end

    return NetworkFlowProblem(
        n,
        arcs,
        trunk,
        capacities,
        costs,
        supplies,
        demands,
        supply_nodes,
        demand_nodes,
        positions,
        geography,
        load_factor,
        value,
        total_demand,
        feasible_witness,
        infeasibility_certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::NetworkFlowProblem)

Build the min-cost flow LP. Deterministic — uses only the struct fields.

  - `flow[k] in [0, capacities[k]]`: flow on arc `k` (variables == arcs)
  - one balance row per node (supply rows `<=`, all others `==`)
"""
function build_model(prob::NetworkFlowProblem)
    model = Model()
    m = length(prob.arcs)
    n = prob.n_nodes

    @variable(model, 0 <= flow[k = 1:m] <= prob.capacities[k])
    @objective(model, Min, sum(prob.costs[k] * flow[k] for k in 1:m))

    out_adj, in_adj = _geo_adjacency(n, prob.arcs)
    is_supply = falses(n)
    is_supply[prob.supply_nodes] .= true
    for v in 1:n
        net_out =
            sum(flow[k] for k in out_adj[v]; init=AffExpr(0.0)) -
            sum(flow[k] for k in in_adj[v]; init=AffExpr(0.0))
        if is_supply[v]
            @constraint(model, net_out <= prob.supplies[v])
        else
            @constraint(model, -net_out == prob.demands[v])
        end
    end
    return model
end

register_variant(
    :network_flow,
    :standard,
    NetworkFlowProblem,
    "Single-commodity min-cost flow (NETGEN-style transshipment) on a sparse geographic network with supply, demand and transit nodes, arc capacities as bounds, and feasibility placed by an exact max-flow boundary with a min-cut region certificate";
    tags=[:logistics, :network, :unimodular],
    max_target_variables=1_000_000,
)
