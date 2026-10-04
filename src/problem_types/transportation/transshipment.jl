using JuMP
using Random
using Distributions
using StatsBase

"""
Planted two-echelon plan: per-lane flows of an exact max flow (value = total
demand) on the plant -> DC -> customer network with DC throughputs as node
capacities: plant supplies, DC conservation and throughput, lane caps and
customer demands all hold.
"""
struct TransshipmentWitness
    inbound::Vector{Float64}
    outbound::Vector{Float64}
    direct::Vector{Float64}
end

"""
Min-cut region certificate on the DC-split network. The region holds the
`plants`, the DCs whose receiving side (`dcs_in`) and shipping side
(`dcs_out`) lie in it, and the `customers`. Every way into the region is
listed and capacitated: inbound linehaul lanes from plants outside into
receiving sides inside (`inbound_lanes`), direct lanes from plants outside into
customers inside (`direct_lanes`), and DCs whose receiving side is outside but
shipping side inside (`throughput_dcs`, bounded by their throughput rows).
Summing the customers' demand rows, the region's DC conservation and
throughput rows and its plants' supply rows gives

    region_demand <= region_supply + entry_capacity

which the certificate violates. Relaxation-proof (a pure LP).
"""
struct TransshipmentCutCertificate
    plants::Vector{Int}
    dcs_in::Vector{Int}
    dcs_out::Vector{Int}
    customers::Vector{Int}
    inbound_lanes::Vector{Int}
    direct_lanes::Vector{Int}
    throughput_dcs::Vector{Int}
    entry_capacity::Float64
    region_demand::Float64
    region_supply::Float64
end

"""
    TransshipmentProblem <: ProblemGenerator

Two-echelon distribution LP: plants -> distribution centres (DCs) ->
customers, plus direct plant -> customer lanes for large customers, on sparse
geographic lane sets.

# Overview

    minimize    sum cost * flow over inbound, outbound and direct lanes
    subject to  sum_{inbound+direct from p} flow <= supply[p]          plants
                sum_{inbound to h} flow = sum_{outbound from h} flow    DCs
                sum_{inbound to h} flow <= throughput[h]                DCs
                sum_{outbound+direct to c} flow >= demand[c]            customers
                0 <= flow <= lane cap (linehaul allotments, direct FTL caps)

Each DC is fed by 2-5 of its nearest plants; each customer is served by 2-4 of
its nearest DCs; the largest customers also get direct full-truckload lanes
from their nearest plants. DC throughput rows (dock/labour capacity) and DC
conservation rows give the LP genuinely coupled node-capacity structure
beyond a bipartite transportation problem.

# Data grounding

One geographic population (clustered/uniform/corridor); DCs are placed at
high-activity (urban) nodes, plants at random ones. DC throughput is sized
from each DC's historical market times a lognormal build factor with regional
under-build shocks (`_tp_market_supply`); plants hold 1.3-1.6x total nominal
demand. Costs: production + linehaul at half the per-km rate, final mile at
1.4x the rate plus a DC handling fee, direct lanes at the base rate plus a
surcharge. 40% of linehaul lanes carry allotments of 30%-80% of the DC's
throughput; direct lanes are capped at 20%-60% of the customer's demand.

# Feasibility control

The EXACT largest deliverable demand scale `lambda*` is computed on the
DC-split network (Dinkelbach iterations over exact max flows) and demands set
to `load_factor * lambda*` times the nominal profile:

  - `feasible`: `load_factor` in [0.6, 0.92]; witness = exact max-flow plan.
  - `infeasible`: `load_factor` in [1.06, 1.25]; certificate = min-cut region
    (typically a region whose DCs are under-built and whose plants and
    linehaul cannot make up for it).
  - `unknown`: `load_factor` in [0.85, 1.15]; `max_flow_value` decides.

# Fields

  - `n_plants`, `n_dcs`, `n_customers`; lane lists `inbound` (plant, DC),
    `outbound` (DC, customer), `direct` (plant, customer), each with aligned
    costs and capacities (`Inf` = uncapped)
  - `supplies`, `throughput`, `demands`, positions, `geography`
  - `load_factor`, `max_flow_value`, `total_demand`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct TransshipmentProblem <: ProblemGenerator
    n_plants::Int
    n_dcs::Int
    n_customers::Int
    inbound::Vector{Tuple{Int, Int}}
    outbound::Vector{Tuple{Int, Int}}
    direct::Vector{Tuple{Int, Int}}
    inbound_cost::Vector{Float64}
    outbound_cost::Vector{Float64}
    direct_cost::Vector{Float64}
    inbound_capacity::Vector{Float64}
    direct_capacity::Vector{Float64}
    supplies::Vector{Float64}
    throughput::Vector{Float64}
    demands::Vector{Float64}
    plant_positions::Vector{Tuple{Float64, Float64}}
    dc_positions::Vector{Tuple{Float64, Float64}}
    customer_positions::Vector{Tuple{Float64, Float64}}
    geography::Symbol
    load_factor::Float64
    max_flow_value::Float64
    total_demand::Float64
    feasible_witness::Union{Nothing, TransshipmentWitness}
    infeasibility_certificate::Union{Nothing, TransshipmentCutCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _ts_maxflow_network(prob_dims, inbound, outbound, direct, in_cap, throughput, dir_cap, big)

DC-split max-flow network: plants `1:P`, DC receiving sides `P+1:P+H`, DC
shipping sides `P+H+1:P+2H`, customers after. Arc order: inbound lanes, DC
throughput arcs, outbound lanes, direct lanes.
"""
function _ts_maxflow_network(
    P::Int,
    H::Int,
    inbound::Vector{Tuple{Int, Int}},
    outbound::Vector{Tuple{Int, Int}},
    direct::Vector{Tuple{Int, Int}},
    in_cap::Vector{Float64},
    throughput::Vector{Float64},
    dir_cap::Vector{Float64},
    big::Float64,
)
    arcs = vcat(
        [(p, P + h) for (p, h) in inbound],
        [(P + h, P + H + h) for h in 1:H],
        [(P + H + h, P + 2H + c) for (h, c) in outbound],
        [(p, P + 2H + c) for (p, c) in direct],
    )
    caps = vcat(
        [isfinite(u) ? u : big for u in in_cap],
        throughput,
        fill(big, length(outbound)),
        [isfinite(u) ? u : big for u in dir_cap],
    )
    return arcs, caps
end

"""
    TransshipmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Variables are the lanes, exactly `max(target_variables, 6)`: about 7%
inbound, 8% direct and the rest outbound. Rows:
`n_plants + 2 * n_dcs + n_customers`. Values above
`TRANSPORTATION_MAX_VARIABLES` raise an `ArgumentError`.
"""
function TransshipmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    _tp_check_target(target_variables, "transshipment")
    rng = MersenneTwister(seed)
    target = max(target_variables, 6)

    # Dimensions.
    mean_out = 2.0 + 2.0 * rand(rng)
    C = max(2, round(Int, 0.85 * target / mean_out))
    H = max(2, round(Int, C / (20.0 + 30.0 * rand(rng))))
    P = max(2, round(Int, H / (1.5 + 2.5 * rand(rng))))
    n_in = clamp(round(Int, H * (2.0 + 3.0 * rand(rng))), H, P * H)
    n_dir = clamp(round(Int, 0.08 * target), 0, P * C)
    n_out = target - n_in - n_dir
    while n_out > H * C
        H += 1
        n_in = clamp(n_in, H, P * H)
        n_out = target - n_in - n_dir
    end
    if n_out < C
        C = n_out
    end

    # Geography: DCs at high-activity nodes, plants at random ones.
    n = P + H + C
    shape = let r = rand(rng)
        r < 0.45 ? :clustered : (r < 0.8 ? :uniform : :corridor)
    end
    positions, weights = _geo_positions(rng, n, shape; span=12.0 * sqrt(n))
    order = randperm(rng, n)
    dc_idx = sample(rng, order, Weights(weights[order]), H; replace=false)
    rest = setdiff(order, dc_idx)
    plant_idx = rest[1:P]
    cust_idx = rest[(P + 1):end]
    plant_pos, dc_pos, cust_pos = positions[plant_idx], positions[dc_idx], positions[cust_idx]
    cust_w = weights[cust_idx]

    inbound, _ = _tp_lanes(rng, plant_pos, dc_pos, n_in; mean_lanes=n_in / H, long_haul=0.15)
    outbound, primary_dc = _tp_lanes(rng, dc_pos, cust_pos, n_out; mean_lanes=n_out / C, weights=cust_w)
    # Direct lanes: largest customers to their nearest plants.
    direct = Tuple{Int, Int}[]
    if n_dir > 0
        per = cld(n_dir, C)
        near = _geo_knn_query(plant_pos, cust_pos, min(P, per))
        for c in sortperm(cust_w; rev=true), p in near[c]
            length(direct) >= n_dir && break
            push!(direct, (p, c))
        end
        sort!(direct)
    end

    d0 = 100.0 .* cust_w ./ (sum(cust_w) / C)
    D0 = sum(d0)
    throughput = _tp_market_supply(rng, dc_pos, outbound, primary_dc, d0)
    plant_w = [weights[i] * rand(rng, LogNormal(0.0, 0.3)) for i in plant_idx]
    supplies = round.((1.3 + 0.3 * rand(rng)) * D0 .* plant_w ./ sum(plant_w); digits=2)

    rate = 0.8 + 0.8 * rand(rng)
    production = [20.0 * rand(rng, LogNormal(0.0, 0.25)) for _ in 1:P]
    handling = [3.0 * rand(rng, LogNormal(0.0, 0.3)) for _ in 1:H]
    pd(a, b) = hypot(a[1] - b[1], a[2] - b[2])
    inbound_cost = [
        round(production[p] + 0.5 * rate * pd(plant_pos[p], dc_pos[h]) * rand(rng, LogNormal(0.0, 0.15)) + 1.0; digits=3)
        for (p, h) in inbound
    ]
    inbound_capacity = [
        rand(rng) < 0.4 ? round(throughput[h] * (0.3 + 0.5 * rand(rng)); digits=2) : Inf for
        (_, h) in inbound
    ]
    outbound_cost = [
        round(1.4 * rate * pd(dc_pos[h], cust_pos[c]) * rand(rng, LogNormal(0.0, 0.15)) + handling[h] + 1.0; digits=3)
        for (h, c) in outbound
    ]
    direct_cost = [
        round(production[p] + rate * pd(plant_pos[p], cust_pos[c]) * rand(rng, LogNormal(0.0, 0.15)) + 4.0; digits=3)
        for (p, c) in direct
    ]
    direct_capacity = [round(max(d0[c] * (0.2 + 0.4 * rand(rng)), 0.01); digits=2) for (_, c) in direct]

    big = 4.0 * (sum(supplies) + 2.0 * D0)
    arcs, caps = _ts_maxflow_network(
        P, H, inbound, outbound, direct, inbound_capacity, throughput, direct_capacity, big
    )
    N = P + 2H + C
    plant_nodes = collect(1:P)
    cust_nodes = collect((P + 2H + 1):N)
    lambda_star = _network_flow_max_scale(N, arcs, caps, plant_nodes, supplies, cust_nodes, d0)
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
        _network_flow_extended(N, arcs, caps, plant_nodes, supplies, cust_nodes, demands)

    feasible_witness = nothing
    infeasibility_certificate = nothing
    n_i, n_o = length(inbound), length(outbound)
    if feasibility_status == feasible
        value >= total_demand * (1 - 1e-9) ||
            error("transportation/transshipment: planted load not deliverable (seed $seed)")
        feasible_witness = TransshipmentWitness(
            ext_flows[1:n_i],
            ext_flows[(n_i + H + 1):(n_i + H + n_o)],
            ext_flows[(n_i + H + n_o + 1):(n_i + H + n_o + length(direct))],
        )
    elseif feasibility_status == infeasible
        value < total_demand * (1 - 1e-6) ||
            error("transportation/transshipment: infeasible load deliverable (seed $seed)")
        out = falses(N + 2)
        out[source_side] .= true
        plants_T = [p for p in 1:P if !out[p]]
        dcs_in = [h for h in 1:H if !out[P + h]]
        dcs_out = [h for h in 1:H if !out[P + H + h]]
        customers_T = [c for c in 1:C if !out[P + 2H + c]]
        in_lanes = [l for (l, (p, h)) in enumerate(inbound) if out[p] && !out[P + h]]
        dir_lanes = [l for (l, (p, c)) in enumerate(direct) if out[p] && !out[P + 2H + c]]
        thr = [h for h in 1:H if out[P + h] && !out[P + H + h]]
        any(out[P + H + h] && !out[P + 2H + c] for (h, c) in outbound) &&
            error("transportation/transshipment: uncapacitated lane in min cut (seed $seed)")
        all(isfinite(inbound_capacity[l]) for l in in_lanes) ||
            error("transportation/transshipment: uncapacitated lane in min cut (seed $seed)")
        entry = sum(inbound_capacity[l] for l in in_lanes; init=0.0) +
            sum(direct_capacity[l] for l in dir_lanes; init=0.0) +
            sum(throughput[h] for h in thr; init=0.0)
        infeasibility_certificate = TransshipmentCutCertificate(
            plants_T,
            dcs_in,
            dcs_out,
            customers_T,
            in_lanes,
            dir_lanes,
            thr,
            entry,
            sum(demands[c] for c in customers_T; init=0.0),
            sum(supplies[p] for p in plants_T; init=0.0),
        )
    end

    return TransshipmentProblem(
        P,
        H,
        C,
        inbound,
        outbound,
        direct,
        inbound_cost,
        outbound_cost,
        direct_cost,
        inbound_capacity,
        direct_capacity,
        supplies,
        throughput,
        demands,
        plant_pos,
        dc_pos,
        cust_pos,
        shape,
        load_factor,
        value,
        total_demand,
        feasible_witness,
        infeasibility_certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::TransshipmentProblem)

Build the two-echelon transshipment LP. Deterministic — uses only the struct
fields. Variables: `x_in` (inbound lanes), `x_out` (outbound lanes), `x_dir`
(direct lanes); lane caps are variable bounds.
"""
function build_model(prob::TransshipmentProblem)
    model = Model()
    P, H, C = prob.n_plants, prob.n_dcs, prob.n_customers
    ni, no, nd = length(prob.inbound), length(prob.outbound), length(prob.direct)
    @variable(model, x_in[1:ni] >= 0)
    @variable(model, x_out[1:no] >= 0)
    @variable(model, x_dir[1:nd] >= 0)
    for l in 1:ni
        isfinite(prob.inbound_capacity[l]) && set_upper_bound(x_in[l], prob.inbound_capacity[l])
    end
    for l in 1:nd
        set_upper_bound(x_dir[l], prob.direct_capacity[l])
    end
    @objective(
        model,
        Min,
        sum(prob.inbound_cost[l] * x_in[l] for l in 1:ni) +
            sum(prob.outbound_cost[l] * x_out[l] for l in 1:no) +
            sum(prob.direct_cost[l] * x_dir[l] for l in 1:nd; init=0.0)
    )
    plant_out = [AffExpr(0.0) for _ in 1:P]
    dc_in = [AffExpr(0.0) for _ in 1:H]
    dc_out = [AffExpr(0.0) for _ in 1:H]
    cust_in = [AffExpr(0.0) for _ in 1:C]
    for (l, (p, h)) in enumerate(prob.inbound)
        add_to_expression!(plant_out[p], x_in[l])
        add_to_expression!(dc_in[h], x_in[l])
    end
    for (l, (h, c)) in enumerate(prob.outbound)
        add_to_expression!(dc_out[h], x_out[l])
        add_to_expression!(cust_in[c], x_out[l])
    end
    for (l, (p, c)) in enumerate(prob.direct)
        add_to_expression!(plant_out[p], x_dir[l])
        add_to_expression!(cust_in[c], x_dir[l])
    end
    for p in 1:P
        @constraint(model, plant_out[p] <= prob.supplies[p])
    end
    for h in 1:H
        @constraint(model, dc_in[h] == dc_out[h])
        @constraint(model, dc_in[h] <= prob.throughput[h])
    end
    for c in 1:C
        @constraint(model, cust_in[c] >= prob.demands[c])
    end
    return model
end

register_variant(
    :transportation,
    :transshipment,
    TransshipmentProblem,
    "Two-echelon plant -> DC -> customer distribution LP with direct lanes, DC conservation and throughput rows, and capped linehaul/direct lanes on sparse geographic lane sets; exact max-flow placement with a DC-split min-cut region certificate";
    tags=[:logistics, :network, :unimodular],
    max_target_variables=1_000_000,
)
