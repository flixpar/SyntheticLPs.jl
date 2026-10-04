using JuMP
using Random
using Distributions

"""
Largest `target_variables` accepted by the transportation variants. The
constructors keep several lane-indexed structures (lane lists, nearest-source
lists, the residual graph of exact max flows), so larger targets raise an
`ArgumentError` instead of being silently undersized.
"""
const TRANSPORTATION_MAX_VARIABLES = 1_000_000

function _tp_check_target(target_variables::Int, name::AbstractString)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= TRANSPORTATION_MAX_VARIABLES || throw(
        ArgumentError(
            "transportation/$name supports at most $TRANSPORTATION_MAX_VARIABLES variables; " *
            "requested $target_variables.",
        ),
    )
    return nothing
end

"""
Planted shipment plan: per-lane flows from an exact max flow on the lane
network whose value equals total demand, so every customer receives its demand,
every source ships at most its supply, and every capacitated lane carries at
most its capacity.
"""
struct TransportationWitness
    flows::Vector{Float64}
end

"""
Hall/Gale region certificate. `sources` and `customers` form a region `T` (read
off the residual graph of an exact max flow) such that

    region_demand > region_supply + inbound_capacity

where `region_demand` is the demand of the customers in `T`, `region_supply`
the supply of the sources in `T`, and `inbound_lanes` exactly the lanes from
sources OUTSIDE `T` into customers in `T` (all capacitated; their capacities
sum to `inbound_capacity`). The customers' demand rows, minus the region's
supply rows, bound the region's demand by its own supply plus what can come in
over the inbound lanes — a contradiction from rows and bounds alone, involving
many customers and sources at once (not a single starved customer), so presolve
does not detect it.
"""
struct TransportationCutCertificate
    sources::Vector{Int}
    customers::Vector{Int}
    inbound_lanes::Vector{Int}
    inbound_capacity::Float64
    region_demand::Float64
    region_supply::Float64
end

"""
    TransportationProblem <: ProblemGenerator

Generator for capacitated transportation problems on sparse geographic lane
networks.

# Overview

Sources (plants, distribution centres) ship to customers over a sparse set of
contracted lanes: each customer is served by a lognormal number (mean 4-9, at
least 2) of lanes to its nearest sources, sometimes one long-haul lane to a
farther source.
Many lanes carry a truck-allotment capacity (a variable bound); the rest are
uncapacitated.

    minimize    sum_l cost[l] * x[l]
    subject to  sum_{l from i} x[l] <= supply[i]      for every source i
                sum_{l to j}   x[l] >= demand[j]      for every customer j
                0 <= x[l] <= capacity[l]              (bound only when finite)

Rows grow with the instance (about one per 4-9 lanes) instead of saturating at
a few hundred as a complete bipartite graph between a few dozen nodes would.

# Data grounding

Sources and customers are drawn from one geographic population (clustered
metropolitan areas, uniform coverage, or a corridor; region side grows with
`sqrt` of the node count). Customer demand follows a heavy-tailed activity
weight (mean 100 units nominal); each source's capacity is its historical
market (half of each nearby customer's demand goes to its nearest source, half
spread over its other lanes) times a lognormal build factor (median 1.5) and
regional under-build shocks (`_tp_market_supply`), so shortages arise in whole
regions while total supply comfortably exceeds demand. Landed cost per
unit = source production cost (lognormal, mean ~20) + distance x freight rate x
lognormal lane noise + handling. Each customer's primary lane (to its nearest
source) is uncapped; 30%-70% of the other lanes carry truck allotments of
25%-100% of the customer's nominal demand.

# Feasibility control

The constructor computes the EXACT largest uniform demand scale `lambda*` the
lanes and supplies can serve (`_network_flow_max_scale`: Dinkelbach
min-ratio-cut iterations over exact max flows) and sets demands to
`load_factor * lambda*` times the nominal profile:

  - `feasible`: `load_factor` in [0.6, 0.92]; witness = exact max-flow plan.
  - `infeasible`: `load_factor` in [1.06, 1.25]; certificate = min-cut region
    (Hall/Gale violation over a set of customers and sources).
  - `unknown`: `load_factor` in [0.85, 1.15]: a natural instance on either side
    of the exact boundary; `max_flow_value >= total_demand` decides it.

# Fields

  - `n_sources`, `n_customers`, `lanes::Vector{Tuple{Int,Int}}` (sorted
    `(source, customer)`), `lane_capacity` (`Inf` = uncapacitated), `costs`
  - `supplies`, `demands`, `source_positions`, `customer_positions`, `geography`
  - `load_factor`, `max_flow_value`, `total_demand`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct TransportationProblem <: ProblemGenerator
    n_sources::Int
    n_customers::Int
    lanes::Vector{Tuple{Int, Int}}
    lane_capacity::Vector{Float64}
    costs::Vector{Float64}
    supplies::Vector{Float64}
    demands::Vector{Float64}
    source_positions::Vector{Tuple{Float64, Float64}}
    customer_positions::Vector{Tuple{Float64, Float64}}
    geography::Symbol
    load_factor::Float64
    max_flow_value::Float64
    total_demand::Float64
    feasible_witness::Union{Nothing, TransportationWitness}
    infeasibility_certificate::Union{Nothing, TransportationCutCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    TransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a sparse capacitated transportation instance with exactly
`target_variables` lanes (= variables; a target of 1 rounds up to 2). Values
above `TRANSPORTATION_MAX_VARIABLES` raise an `ArgumentError`.
"""
function TransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    _tp_check_target(target_variables, "standard")
    rng = MersenneTwister(seed)
    n_lanes = max(target_variables, 2)

    mean_lanes = 4.0 + 5.0 * rand(rng)
    nS, nD = _tp_dimensions(rng, n_lanes, mean_lanes, 6.0 + 9.0 * rand(rng))
    src_pos, dst_pos, src_w, dst_w, geography = _tp_geography(rng, nS, nD)
    lanes, primary = _tp_lanes(rng, src_pos, dst_pos, n_lanes; mean_lanes=mean_lanes, weights=dst_w)
    L = length(lanes)

    d0 = 100.0 .* dst_w ./ (sum(dst_w) / nD)
    D0 = sum(d0)
    supplies = _tp_market_supply(rng, src_pos, lanes, primary, d0)

    production = [20.0 * rand(rng, LogNormal(0.0, 0.25)) for _ in 1:nS]
    rate = 0.8 + 0.8 * rand(rng)
    capacitated_share = 0.3 + 0.4 * rand(rng)
    costs = Vector{Float64}(undef, L)
    lane_capacity = Vector{Float64}(undef, L)
    for (l, (i, j)) in enumerate(lanes)
        d = hypot(src_pos[i][1] - dst_pos[j][1], src_pos[i][2] - dst_pos[j][2])
        costs[l] = round(production[i] + rate * d * rand(rng, LogNormal(0.0, 0.15)) + 1.0; digits=3)
        # The primary (nearest-source) lane is the customer's own contract
        # carrier and is never capped; other lanes are truck allotments.
        lane_capacity[l] = (i != primary[j] && rand(rng) < capacitated_share) ?
            round(max(d0[j] * (0.25 + 0.75 * rand(rng)), 0.01); digits=2) : Inf
    end

    big = 4.0 * (sum(supplies) + 2.0 * D0)
    arcs, caps = _tp_maxflow_network(nS, lanes, lane_capacity, big)
    src_nodes = collect(1:nS)
    dst_nodes = collect((nS + 1):(nS + nD))
    lambda_star = _network_flow_max_scale(nS + nD, arcs, caps, src_nodes, supplies, dst_nodes, d0)

    load_factor = if feasibility_status == feasible
        0.6 + 0.32 * rand(rng)
    elseif feasibility_status == infeasible
        1.06 + 0.19 * rand(rng)
    else
        0.85 + 0.3 * rand(rng)
    end
    demands = max.(round.(load_factor * lambda_star .* d0; digits=2), 0.01)
    total_demand = sum(demands)
    value, ext_flows, source_side, _, _ =
        _network_flow_extended(nS + nD, arcs, caps, src_nodes, supplies, dst_nodes, demands)

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        value >= total_demand * (1 - 1e-9) ||
            error("transportation/standard: planted load not deliverable (seed $seed)")
        feasible_witness = TransportationWitness(ext_flows[1:L])
    elseif feasibility_status == infeasible
        value < total_demand * (1 - 1e-6) ||
            error("transportation/standard: infeasible load deliverable (seed $seed)")
        outside = falses(nS + nD + 2)
        outside[source_side] .= true
        in_src = [i for i in 1:nS if !outside[i]]
        in_dst = [j for j in 1:nD if !outside[nS + j]]
        inbound = [l for (l, (i, j)) in enumerate(lanes) if outside[i] && !outside[nS + j]]
        all(isfinite(lane_capacity[l]) for l in inbound) ||
            error("transportation/standard: uncapacitated lane in min cut (seed $seed)")
        infeasibility_certificate = TransportationCutCertificate(
            in_src,
            in_dst,
            inbound,
            sum(lane_capacity[l] for l in inbound; init=0.0),
            sum(demands[j] for j in in_dst; init=0.0),
            sum(supplies[i] for i in in_src; init=0.0),
        )
    end

    return TransportationProblem(
        nS,
        nD,
        lanes,
        lane_capacity,
        costs,
        supplies,
        demands,
        src_pos,
        dst_pos,
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
    build_model(prob::TransportationProblem)

Build the sparse capacitated transportation LP. Deterministic — uses only the
struct fields. Variables `x[l] >= 0` per lane (upper bound when capacitated);
one supply row per source and one demand row per customer.
"""
function build_model(prob::TransportationProblem)
    model = Model()
    L = length(prob.lanes)
    @variable(model, x[1:L] >= 0)
    for l in 1:L
        isfinite(prob.lane_capacity[l]) && set_upper_bound(x[l], prob.lane_capacity[l])
    end
    @objective(model, Min, sum(prob.costs[l] * x[l] for l in 1:L))
    out_lanes = [Int[] for _ in 1:(prob.n_sources)]
    in_lanes = [Int[] for _ in 1:(prob.n_customers)]
    for (l, (i, j)) in enumerate(prob.lanes)
        push!(out_lanes[i], l)
        push!(in_lanes[j], l)
    end
    for i in 1:(prob.n_sources)
        @constraint(model, sum(x[l] for l in out_lanes[i]; init=AffExpr(0.0)) <= prob.supplies[i])
    end
    for j in 1:(prob.n_customers)
        @constraint(model, sum(x[l] for l in in_lanes[j]) >= prob.demands[j])
    end
    return model
end

register_variant(
    :transportation,
    :standard,
    TransportationProblem,
    "Capacitated transportation LP on a sparse geographic lane network (customers served by their nearest sources plus long-haul lanes, truck-allotment lane capacities as bounds), with feasibility placed by an exact max-flow boundary and a Hall-region certificate";
    default=true,
    tags=[:logistics, :bipartite, :unimodular],
    max_target_variables=1_000_000,
)
