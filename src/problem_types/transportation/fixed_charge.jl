using JuMP
using Random
using Distributions

"""
Planted shipment plan with its lanes opened: `flows` is an exact max-flow plan
meeting every demand, and `open[l] = 1` exactly on the lanes it uses. It is a
feasible point of the MIP itself (not only of the relaxation): supplies hold,
every used lane satisfies `x <= link_bound * 1`, and every source's
`max_lanes` is at least its number of used lanes.
"""
struct FixedChargeTransportationWitness
    flows::Vector{Float64}
    open::Vector{Float64}
end

"""
    FixedChargeTransportationProblem <: ProblemGenerator

Fixed-charge transportation with lane-opening decisions and per-source
lane-count limits, on sparse geographic lane networks.

# Overview

Opening lane `l = (i, j)` (contracting a carrier, dedicating equipment) costs
`fixed_cost[l]`; shipments cost `unit_cost[l]` per unit. Each source can run at
most `max_lanes[i]` open lanes (dock doors, dispatch capacity, carrier
contracts).

    minimize    sum_l unit_cost[l] x[l] + fixed_cost[l] y[l]
    subject to  sum_{l from i} x[l] <= supply[i]            every source i
                sum_{l to j}   x[l] >= demand[j]            every customer j
                x[l] <= link_bound[l] * y[l]                 every lane l
                sum_{l from i} y[l] <= max_lanes[i]          every source i
                x >= 0,  y in {0,1}

`link_bound[l] = min(supply[i], demand[j], lane_capacity[l])` is the STRONG
linking coefficient (not a big-M), so the LP relaxation stays meaningful: with
`y` relaxed every lane pays its fixed charge per unit at rate
`fixed_cost/link_bound`, and the lane-count rows become weighted capacity rows
`sum_l x[l]/link_bound[l] <= max_lanes[i]` with heterogeneous coefficients — a
non-unimodular LP that does not collapse to a plain transportation problem,
and in which every `y` sits in two rows (never a removable column singleton).

# Data grounding

Geography, lanes (mean 3-6 per customer: nearest sources plus long-haul),
heavy-tailed demand and market-sized supplies with regional under-build shocks
as in `transportation/standard`. Unit cost = production + distance x rate x
noise + handling; fixed cost = lane setup (lognormal, median 150) + a
distance-proportional deadhead component. Each customer's primary lane is
uncapped; 30%-60% of the others carry an allotment cap.

# Feasibility control

As in `transportation/standard`, the EXACT largest deliverable demand scale
`lambda*` of the lane network is computed (lane capacities bound shipments
through `x <= link_bound * y <= lane_capacity`), demands are
`load_factor * lambda*` times the nominal profile, and an exact max flow at the
final demands gives the plan; `max_lanes[i]` is sized from the number of lanes
that plan uses at `i`:

  - `feasible`: `load_factor` in [0.6, 0.88]; `max_lanes` 1.1-1.5x the plan's
    usage; the plan with its lanes opened is the witness (MIP-feasible).
  - `infeasible`: `load_factor` in [1.06, 1.25]; certificate = the min-cut
    region (`TransportationCutCertificate`): its customers' demand exceeds its
    sources' supply plus the caps of the lanes reaching in, where each cap
    binds through `x <= link_bound * y`, `y <= 1` — valid in the relaxation.
  - `unknown`: `load_factor` in [0.85, 1.1] and `max_lanes` 0.75-1.3x the
    plan's usage: the lane budgets may or may not leave enough routing
    flexibility — natural, either side.

# Fields

  - `n_sources`, `n_customers`, `lanes`, `unit_cost`, `fixed_cost`,
    `lane_capacity` (`Inf` = uncapped), `link_bound`, `max_lanes`
  - `supplies`, `demands`, positions, `geography`, `load_factor`,
    `max_flow_value`, `total_demand`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct FixedChargeTransportationProblem <: ProblemGenerator
    n_sources::Int
    n_customers::Int
    lanes::Vector{Tuple{Int, Int}}
    unit_cost::Vector{Float64}
    fixed_cost::Vector{Float64}
    lane_capacity::Vector{Float64}
    link_bound::Vector{Float64}
    max_lanes::Vector{Int}
    supplies::Vector{Float64}
    demands::Vector{Float64}
    source_positions::Vector{Tuple{Float64, Float64}}
    customer_positions::Vector{Tuple{Float64, Float64}}
    geography::Symbol
    load_factor::Float64
    max_flow_value::Float64
    total_demand::Float64
    feasible_witness::Union{Nothing, FixedChargeTransportationWitness}
    infeasibility_certificate::Union{Nothing, TransportationCutCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    FixedChargeTransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Variables are `2 * n_lanes` (a shipment and an opening decision per lane), with
`n_lanes = max(round(target_variables / 2), 2)`. Rows:
`2 * n_sources + n_customers + n_lanes`. Values above
`TRANSPORTATION_MAX_VARIABLES` raise an `ArgumentError`.
"""
function FixedChargeTransportationProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    _tp_check_target(target_variables, "fixed_charge")
    rng = MersenneTwister(seed)
    n_lanes = max(round(Int, target_variables / 2), 2)

    mean_lanes = 3.0 + 3.0 * rand(rng)
    nS, nD = _tp_dimensions(rng, n_lanes, mean_lanes, 5.0 + 7.0 * rand(rng))
    src_pos, dst_pos, src_w, dst_w, geography = _tp_geography(rng, nS, nD)
    lanes, primary = _tp_lanes(rng, src_pos, dst_pos, n_lanes; mean_lanes=mean_lanes, weights=dst_w)
    L = length(lanes)

    d0 = 100.0 .* dst_w ./ (sum(dst_w) / nD)
    D0 = sum(d0)
    supplies = _tp_market_supply(rng, src_pos, lanes, primary, d0)

    production = [20.0 * rand(rng, LogNormal(0.0, 0.25)) for _ in 1:nS]
    rate = 0.8 + 0.8 * rand(rng)
    capped_share = 0.3 + 0.3 * rand(rng)
    unit_cost = Vector{Float64}(undef, L)
    fixed_cost = Vector{Float64}(undef, L)
    lane_capacity = Vector{Float64}(undef, L)
    for (l, (i, j)) in enumerate(lanes)
        d = hypot(src_pos[i][1] - dst_pos[j][1], src_pos[i][2] - dst_pos[j][2])
        unit_cost[l] = round(production[i] + rate * d * rand(rng, LogNormal(0.0, 0.15)) + 1.0; digits=3)
        fixed_cost[l] = round(150.0 * rand(rng, LogNormal(0.0, 0.4)) + 8.0 * rate * d; digits=2)
        lane_capacity[l] = (i != primary[j] && rand(rng) < capped_share) ?
            round(max(d0[j] * (0.3 + 0.9 * rand(rng)), 0.01); digits=2) : Inf
    end

    big = 4.0 * (sum(supplies) + 2.0 * D0)
    arcs, caps = _tp_maxflow_network(nS, lanes, lane_capacity, big)
    src_nodes = collect(1:nS)
    dst_nodes = collect((nS + 1):(nS + nD))
    lambda_star = _network_flow_max_scale(nS + nD, arcs, caps, src_nodes, supplies, dst_nodes, d0)
    load_factor = if feasibility_status == feasible
        0.6 + 0.28 * rand(rng)
    elseif feasibility_status == infeasible
        1.06 + 0.19 * rand(rng)
    else
        0.85 + 0.25 * rand(rng)
    end
    demands = max.(round.(load_factor * lambda_star .* d0; digits=2), 0.01)
    total_demand = sum(demands)
    value, ext_flows, source_side, _, _ =
        _network_flow_extended(nS + nD, arcs, caps, src_nodes, supplies, dst_nodes, demands)
    flows = ext_flows[1:L]

    link_bound = [min(supplies[i], demands[j], lane_capacity[l]) for (l, (i, j)) in enumerate(lanes)]
    used = zeros(Int, nS)
    for (l, (i, _)) in enumerate(lanes)
        flows[l] > 1e-9 && (used[i] += 1)
    end
    budget_range = feasibility_status == unknown ? (0.75, 1.3) : (1.1, 1.5)
    max_lanes = [
        max(1, ceil(Int, used[i] * (budget_range[1] + (budget_range[2] - budget_range[1]) * rand(rng))))
        for i in 1:nS
    ]

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        value >= total_demand * (1 - 1e-9) ||
            error("transportation/fixed_charge: planted load not deliverable (seed $seed)")
        open = [flows[l] > 1e-9 ? 1.0 : 0.0 for l in 1:L]
        flows = [open[l] > 0 ? flows[l] : 0.0 for l in 1:L]
        feasible_witness = FixedChargeTransportationWitness(flows, open)
    elseif feasibility_status == infeasible
        value < total_demand * (1 - 1e-6) ||
            error("transportation/fixed_charge: infeasible load deliverable (seed $seed)")
        outside = falses(nS + nD + 2)
        outside[source_side] .= true
        in_src = [i for i in 1:nS if !outside[i]]
        in_dst = [j for j in 1:nD if !outside[nS + j]]
        inbound = [l for (l, (i, j)) in enumerate(lanes) if outside[i] && !outside[nS + j]]
        all(isfinite(lane_capacity[l]) for l in inbound) ||
            error("transportation/fixed_charge: uncapacitated lane in min cut (seed $seed)")
        infeasibility_certificate = TransportationCutCertificate(
            in_src,
            in_dst,
            inbound,
            sum(lane_capacity[l] for l in inbound; init=0.0),
            sum(demands[j] for j in in_dst; init=0.0),
            sum(supplies[i] for i in in_src; init=0.0),
        )
    end

    return FixedChargeTransportationProblem(
        nS,
        nD,
        lanes,
        unit_cost,
        fixed_cost,
        lane_capacity,
        link_bound,
        max_lanes,
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
    build_model(prob::FixedChargeTransportationProblem)

Build the fixed-charge transportation MIP with strong linking and lane-count
limits. Deterministic — uses only the struct fields.
"""
function build_model(prob::FixedChargeTransportationProblem)
    model = Model()
    L = length(prob.lanes)
    @variable(model, x[1:L] >= 0)
    @variable(model, y[1:L], Bin)
    @objective(
        model, Min, sum(prob.unit_cost[l] * x[l] + prob.fixed_cost[l] * y[l] for l in 1:L)
    )
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
    for l in 1:L
        @constraint(model, x[l] <= prob.link_bound[l] * y[l])
    end
    for i in 1:(prob.n_sources)
        @constraint(model, sum(y[l] for l in out_lanes[i]; init=AffExpr(0.0)) <= prob.max_lanes[i])
    end
    return model
end

register_variant(
    :transportation,
    :fixed_charge,
    FixedChargeTransportationProblem,
    "Fixed-charge transportation on sparse geographic lanes with strong linking x <= min(supply, demand, capacity) * y and per-source lane-count limits, so the LP relaxation keeps non-unimodular lane-budget structure; exact max-flow placement, MIP-feasible planted plan, Hall-region certificate";
    tags=[:logistics, :bipartite, :big_m],
    max_target_variables=1_000_000,
)
