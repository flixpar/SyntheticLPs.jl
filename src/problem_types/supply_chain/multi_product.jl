using JuMP
using Random
using Distributions

"""
Cumulative product-supply certificate. Through period `period`, deliveries of
`product` must cover `demand` (its demand rows); DC balance rows with
nonnegative stock limit deliveries to `initial_stock` plus linehaul inflow; and
every capable plant's shipments of the product are limited each period by its
product-line row and by its shared resource row divided by the product's
resource use. Summing gives at most `supply_bound`, below `demand` by `margin`.
"""
struct SupplyChainProductCertificate
    product::Int
    period::Int
    plants::Vector{Int}
    demand::Float64
    initial_stock::Float64
    supply_bound::Float64
    margin::Float64
end

"""
    MultiProductSupplyChainProblem <: ProblemGenerator

Multi-commodity, multi-echelon, multi-period production–distribution LP over an
existing plant -> DC -> customer network (no design decisions).

# Overview

3–12 products; plants are specialized (each makes a random subset, every
product has at least one source) and each DC is linked to its nearest plants
plus the nearest source of any product they miss. Variables: truck shipments
per lane × product × period, DC stock per DC × product × period, and
deliveries per arc × ordered product × period (customers order ~60% of the
products). Coupling rows beyond the shared balance backbone
([`_scn_build_model`](@ref)):

  - product-line capacity: `Σ_{lanes from p} ship[·, k, t] ≤ line_capacity[p, k]`;
  - shared plant resource: `Σ_k resource_use[p,k] Σ ship ≤ plant_capacity[p, t]`;
  - **lane bundle capacity**: `Σ_k ship[lane, k, t] ≤ lane_capacity[lane]` — the
    multicommodity coupling, since products compete for the same trucks;
  - DC throughput and storage shared across products.

# Feasibility control

  - `feasible`: the planted plan (see `standard`) with every capacity 8–40%
    above its usage ([`SupplyChainNetworkWitness`](@ref)).
  - `infeasible`: the product lines of one product are cut so its cumulative
    supply through a late period is 10–20% short of cumulative demand
    ([`SupplyChainProductCertificate`](@ref)); a sum over many rows.
  - `unknown`: all capacities scaled by one supply factor `U(0.60, 1.05)`
    (`capacity_factor`).

# Size

`Σ_lanes |products(plant)| T + n_dcs K T + Σ_arcs |products(customer)| T`
variables, within half an arc's worth of the target (targets below 30 → 30).
"""
struct MultiProductSupplyChainProblem <: ProblemGenerator
    network::SupplyChainNetwork
    capacity_factor::Float64
    feasible_witness::Union{Nothing, SupplyChainNetworkWitness}
    infeasibility_certificate::Union{Nothing, SupplyChainProductCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _multi_product_supply_bound(net, k, tau)

Upper bound on cumulative supply of product `k` through `tau`: initial DC stock
plus, for each capable plant and period, `min(line capacity, plant capacity /
resource use)`.
"""
function _multi_product_supply_bound(net::SupplyChainNetwork, k::Int, tau::Int)
    plants = [p for p in 1:net.n_plants if k in net.plant_products[p]]
    stock = sum(net.initial_stock[:, k])
    production = sum(
        min(net.line_capacity[p, k], net.plant_capacity[p, t] / net.resource_use[p, k]) for
        p in plants, t in 1:tau
    )
    return plants, stock, stock + production
end

function MultiProductSupplyChainProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    net, witness, _ = _scn_instance(rng, target_variables, :multi_product)
    capacity_factor = 1.0
    certificate = nothing
    if feasibility_status == infeasible
        T = net.n_periods
        # Only a product someone orders can be short: on tiny networks (two
        # customers ordering ~60% of the products each) a product may have no
        # demand at all, which would leave nothing to cut.
        tau_range = max(2, T - 2):T
        ordered = [k for k in 1:net.n_products if sum(net.demand[:, k, 1:first(tau_range)]) > 0]
        k = rand(rng, ordered)
        tau = rand(rng, tau_range)
        demand = sum(net.demand[:, k, 1:tau])
        plants, stock, _ = _multi_product_supply_bound(net, k, tau)
        goal = demand * rand(rng, Uniform(0.80, 0.90))
        # Opening stock covers at most 45% of period-1 outflow, so it is always
        # below 80% of a positive cumulative demand.
        @assert goal > stock
        # Per-plant effective rate (never above the plant's resource limit), then
        # one common scale so the cumulative bound lands on the goal.
        rate = [
            min(
                net.line_capacity[p, k],
                minimum(net.plant_capacity[p, t] for t in 1:tau) / net.resource_use[p, k],
            ) for p in plants
        ]
        scale = (goal - stock) / (tau * sum(rate))
        for (i, p) in enumerate(plants)
            net.line_capacity[p, k] = scale * rate[i]
        end
        _, _, bound = _multi_product_supply_bound(net, k, tau)
        certificate = SupplyChainProductCertificate(
            k, tau, plants, demand, stock, bound, demand - bound
        )
        @assert certificate.margin > 0.05 * demand
    elseif feasibility_status == unknown
        capacity_factor = rand(rng, Uniform(0.60, 1.05))
        _scn_scale_capacities!(net, capacity_factor)
    end
    return MultiProductSupplyChainProblem(
        net,
        capacity_factor,
        feasibility_status == feasible ? witness : nothing,
        certificate,
        feasibility_status,
    )
end

function build_model(prob::MultiProductSupplyChainProblem)
    model, _, _ = _scn_build_model(prob.network)
    return model
end

register_variant(
    :supply_chain,
    :multi_product,
    MultiProductSupplyChainProblem,
    "Multi-commodity, multi-echelon, multi-period production-distribution LP: specialized plants with product-line and shared resource capacity, lane bundle capacity shared by products, DC inventory, throughput and storage limits";
    tags=[:logistics, :network, :multicommodity, :staircase],
)
