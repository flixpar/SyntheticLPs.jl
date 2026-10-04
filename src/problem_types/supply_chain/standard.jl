using JuMP
using Random
using Distributions

"""
Regional throughput certificate. Every customer in `customers` (one geographic
region) is served only from the DCs in `dcs` (the union of their delivery
arcs). Summing their demand rows in `period` requires `demand` units out of
those DCs, while summing those DCs' throughput rows allows at most
`Σ throughput[d] * open[d] ≤ throughput` (`open ≤ 1`). `margin = demand -
throughput > 0`. The argument aggregates many demand and throughput rows, so
presolve does not see it.
"""
struct SupplyChainRegionalCertificate
    region::Int
    customers::Vector{Int}
    dcs::Vector{Int}
    period::Int
    demand::Float64
    throughput::Float64
    margin::Float64
end

"""
    SupplyChainProblem <: ProblemGenerator

Multi-echelon, multi-period supply-chain network design: plants ship over
truck/rail/intermodal linehaul lanes to candidate distribution centers (DCs),
which hold inventory and deliver to clustered customers.

# Overview

Binary `open[d]` decides which DCs to operate (fixed cost over the horizon).
Continuous flows: lane shipments `ship[lane, product, period]` for every
product the plant makes, DC inventories `stock[d, product, period]`, and
deliveries `deliver[arc, product, period]` on each customer's 2–5 nearest DCs.
Rows (see [`_scn_build_model`](@ref)): plant resource capacity per period;
rail and intermodal capacity per period; DC inventory balance per
DC/product/period; DC throughput and storage per period gated by `open[d]`;
customer demand per customer/product/period; and per-arc, per-period linking
`Σ_k deliver ≤ demand × open[d]` — the disaggregated formulation, so the LP
relaxation (the default `relax_integer=true`) still has to price DC capacity
fractionally instead of letting one throughput row absorb it.

Rows grow with the instance (≈ 0.6–0.8 per column): facilities × products ×
periods balance rows, customers × products × periods demand rows, and arcs ×
periods linking rows.

# Data grounding

Customers cluster in regions on a 100 × 100 map with lognormal sizes; DCs sit
near regions; plants are spread out. Demand has product-specific seasonality.
Linehaul cost is production cost plus a mode terminal charge and
distance-proportional rate (rail and intermodal are cheaper per km but carry a
terminal cost and are not available on short or every lane); last-mile cost is
distance-based; DC fixed costs scale with DC size.

# Feasibility control

A feasible plan is always planted: a set of open DCs covering every customer,
distance-weighted deliveries, a cover-stock inventory policy, and
weight-split lane shipments; every capacity is sized 8–35% above the plan's
usage.

  - `feasible`: stores the plan as [`SupplyChainNetworkWitness`](@ref).
  - `infeasible`: one region's DCs lose throughput so that their combined
    capacity is 8–18% below the region's peak-period demand
    ([`SupplyChainRegionalCertificate`](@ref)); needs simplex to discover.
  - `unknown`: every capacity is multiplied by one network-wide supply factor
    `U(0.60, 1.05)` (stored as `capacity_factor`); below the plan's slack the
    network may or may not cope by rerouting, opening DCs, and prebuilding stock.

# Size

`n_dcs + Σ_lanes |products(plant)| T + n_dcs K T + Σ_arcs |products(customer)| T`
variables ([`_scn_num_variables`](@ref)); extra last-mile arcs are added until
the count is within half an arc's worth (`K T / 2`) of the target. Targets
below 30 are treated as 30.
"""
struct SupplyChainProblem <: ProblemGenerator
    network::SupplyChainNetwork
    capacity_factor::Float64
    feasible_witness::Union{Nothing, SupplyChainNetworkWitness}
    infeasibility_certificate::Union{Nothing, SupplyChainRegionalCertificate}
    feasibility_status::FeasibilityStatus
end

function SupplyChainProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    net, witness, regions = _scn_instance(rng, target_variables, :standard)
    capacity_factor = 1.0
    certificate = nothing
    if feasibility_status == infeasible
        populated = [r for r in eachindex(regions) if !isempty(regions[r])]
        r = populated[rand(rng, 1:length(populated))]
        customers = regions[r]
        dcs = sort!(unique(d for (d, c) in net.arcs if net.customer_region[c] == r))
        load = [sum(net.demand[c, k, t] for c in customers for k in net.customer_products[c]) for t in 1:net.n_periods]
        period = argmax(load)
        required = load[period]
        target = required * rand(rng, Uniform(0.82, 0.92))
        current = sum(net.dc_throughput[d] for d in dcs)
        if current > target
            for d in dcs
                net.dc_throughput[d] *= target / current
            end
        end
        available = sum(net.dc_throughput[d] for d in dcs)
        certificate = SupplyChainRegionalCertificate(
            r, customers, dcs, period, required, available, required - available
        )
        @assert certificate.margin > 0.05 * required
    elseif feasibility_status == unknown
        capacity_factor = rand(rng, Uniform(0.60, 1.05))
        _scn_scale_capacities!(net, capacity_factor)
    end
    return SupplyChainProblem(
        net,
        capacity_factor,
        feasibility_status == feasible ? witness : nothing,
        certificate,
        feasibility_status,
    )
end

function build_model(prob::SupplyChainProblem)
    model, _, _ = _scn_build_model(prob.network)
    return model
end

register_variant(
    :supply_chain,
    :standard,
    SupplyChainProblem,
    "Multi-echelon, multi-period supply-chain network design: plants -> candidate DCs -> customers with DC opening decisions and disaggregated linking, truck/rail/intermodal linehaul with modal capacity, DC inventory, throughput and storage limits";
    default=true,
    tags=[:logistics, :network, :staircase, :big_m],
)
