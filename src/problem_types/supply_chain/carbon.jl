using JuMP
using Random
using Distributions

"""
Carbon-budget infeasibility certificate. Let `dc_inbound_min[d]` be the
cleanest lane into DC `d` (production intensity plus linehaul) and
`customer_unit_min[c] = min over c's arcs (d, c) of arc_emission + dc_inbound_min[d]`.
Summing DC balance rows over the horizon, a DC's linehaul inflow is at least
its deliveries minus its initial stock (final stock is nonnegative), so total
emissions are at least `Σ_a (arc_emission[a] + dc_inbound_min[d(a)]) deliver[a]
- Σ_d dc_inbound_min[d] initial_stock[d]`, and with the demand rows at least
`lower_bound = demand_bound - stock_credit`. The budget row allows
`budget = lower_bound - margin`; summing the per-period cap rows gives the
contradiction.
"""
struct SupplyChainCarbonCertificate
    dc_inbound_min::Vector{Float64}
    customer_unit_min::Vector{Float64}
    demand_bound::Float64
    stock_credit::Float64
    lower_bound::Float64
    budget::Float64
    margin::Float64
end

"""
    CarbonSupplyChainProblem <: ProblemGenerator

The multi-echelon network-design model of `supply_chain/standard` (plants ->
candidate DCs -> customers, products, periods, DC opening with disaggregated
linking, truck/rail/intermodal linehaul with modal capacity) under a horizon
**carbon budget**.

# Emissions

  - production: plant-specific intensity per unit (`plant_intensity`, cleaner
    and dirtier sites);
  - linehaul: mode factor (truck 0.10, intermodal 0.05, rail 0.03 per unit-km)
    times distance, so `lane_emission = intensity + factor × distance`;
  - last mile: truck factor times distance (`arc_emission`).

The allowance `carbon_budget` is issued per compliance period
(`period_budget`, proportional to the planted plan's emission profile, no
banking): one row per period `Σ lane_emission × ship[·,·,t] + Σ arc_emission ×
deliver[·,·,t] ≤ period_budget[t]` couples every flow of that period. Because
rail and intermodal are cheaper per km but carry terminal costs and limited
capacity, the cost-minimal plan is truck-heavy and the caps force a modal
shift and sourcing from cleaner plants. (A single horizon row made HiGHS hit
numerical trouble near the feasibility threshold.)

# Feasibility control

The planted plan (see `standard`) routes three times as much weight on rail
and intermodal lanes as on trucks, so it is a low-emission plan.

  - `feasible`: budget `U(1.00, 1.05) ×` the plan's emissions (active against
    the truck-heavy cost optimum); stores [`SupplyChainNetworkWitness`](@ref).
  - `infeasible`: budget `U(0.85, 0.93) ×` a valid emission lower bound
    ([`SupplyChainCarbonCertificate`](@ref)); presolve cannot see it.
  - `unknown`: an active cap `U(0.97, 1.05) ×` the plan's emissions with every
    capacity scaled by one supply factor `U(0.60, 1.05)` (`capacity_factor`):
    tighter capacity also starves the clean rail/intermodal options, so cap
    and capacities interact. (Caps drawn near the minimum achievable emissions
    made HiGHS's dual simplex stall on the barely-infeasible side.)

Variables and sizing are those of `standard` (one extra row, no extra column).
"""
struct CarbonSupplyChainProblem <: ProblemGenerator
    network::SupplyChainNetwork
    plant_intensity::Vector{Float64}
    lane_emission::Vector{Float64}
    arc_emission::Vector{Float64}
    carbon_budget::Float64
    period_budget::Vector{Float64}
    plan_emissions::Float64
    capacity_factor::Float64
    feasible_witness::Union{Nothing, SupplyChainNetworkWitness}
    infeasibility_certificate::Union{Nothing, SupplyChainCarbonCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _carbon_emission_lower_bound(net, lane_emission, arc_emission)

Valid lower bound on the total emissions of any feasible plan (see
[`SupplyChainCarbonCertificate`](@ref)); returns
`(dc_inbound_min, customer_unit_min, demand_bound, stock_credit, bound)`.
"""
function _carbon_emission_lower_bound(net::SupplyChainNetwork, lane_emission, arc_emission)
    dc_inbound_min = fill(Inf, net.n_dcs)
    for (l, (_, d, _)) in enumerate(net.lanes)
        dc_inbound_min[d] = min(dc_inbound_min[d], lane_emission[l])
    end
    customer_unit_min = fill(Inf, net.n_customers)
    for (a, (d, c)) in enumerate(net.arcs)
        customer_unit_min[c] = min(customer_unit_min[c], arc_emission[a] + dc_inbound_min[d])
    end
    demand_bound = 0.0
    for c in 1:net.n_customers, k in net.customer_products[c], t in 1:net.n_periods
        demand_bound += net.demand[c, k, t] * customer_unit_min[c]
    end
    stock_credit = sum(dc_inbound_min[d] * net.initial_stock[d, k] for d in 1:net.n_dcs, k in 1:net.n_products)
    return dc_inbound_min, customer_unit_min, demand_bound, stock_credit, demand_bound - stock_credit
end

function CarbonSupplyChainProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    net, witness, _ = _scn_instance(rng, target_variables, :carbon)
    plant_intensity = rand(rng, LogNormal(log(1.5), 0.5), net.n_plants)
    lane_emission = [
        plant_intensity[p] + _SCN_MODES[net.modes[mi]].emission * net.lane_distance[l] for
        (l, (p, _, mi)) in enumerate(net.lanes)
    ]
    arc_emission = [_SCN_MODES.truck.emission * dist for dist in net.arc_distance]

    T = net.n_periods
    plan_by_period = zeros(Float64, T)
    for (i, (l, _, t)) in enumerate(_scn_ship_keys(net))
        plan_by_period[t] += lane_emission[l] * witness.ship[i]
    end
    for (i, (a, _, t)) in enumerate(_scn_deliver_keys(net))
        plan_by_period[t] += arc_emission[a] * witness.deliver[i]
    end
    plan = sum(plan_by_period)
    inbound_min, unit_min, demand_bound, credit, bound = _carbon_emission_lower_bound(
        net, lane_emission, arc_emission
    )
    @assert bound <= plan + 1e-6

    certificate = nothing
    capacity_factor = 1.0
    budget = if feasibility_status == feasible
        plan * rand(rng, Uniform(1.00, 1.05))
    elseif feasibility_status == infeasible
        b = bound * rand(rng, Uniform(0.85, 0.93))
        certificate = SupplyChainCarbonCertificate(
            inbound_min, unit_min, demand_bound, credit, bound, b, bound - b
        )
        b
    else
        # Natural uncertainty comes from a network-wide supply condition (as in
        # `standard`) under an active cap. Budgets drawn close to the minimum
        # achievable emissions made HiGHS's dual simplex stall or return an
        # unknown status on the barely-infeasible side, so the cap itself is
        # kept near the plan.
        capacity_factor = rand(rng, Uniform(0.60, 1.05))
        _scn_scale_capacities!(net, capacity_factor)
        plan * rand(rng, Uniform(0.97, 1.05))
    end
    # The horizon allowance is issued per compliance period in proportion to
    # the plan's emission profile (no banking between periods).
    period_budget = budget .* plan_by_period ./ plan
    return CarbonSupplyChainProblem(
        net,
        plant_intensity,
        lane_emission,
        arc_emission,
        budget,
        period_budget,
        plan,
        capacity_factor,
        feasibility_status == feasible ? witness : nothing,
        certificate,
        feasibility_status,
    )
end

function build_model(prob::CarbonSupplyChainProblem)
    net = prob.network
    model, ship, deliver = _scn_build_model(net)
    emissions = [AffExpr(0.0) for _ in 1:net.n_periods]
    for (i, (l, _, t)) in enumerate(_scn_ship_keys(net))
        add_to_expression!(emissions[t], prob.lane_emission[l], ship[i])
    end
    for (i, (a, _, t)) in enumerate(_scn_deliver_keys(net))
        add_to_expression!(emissions[t], prob.arc_emission[a], deliver[i])
    end
    @constraint(model, carbon_cap[t = 1:net.n_periods], emissions[t] <= prob.period_budget[t])
    return model
end

register_variant(
    :supply_chain,
    :carbon,
    CarbonSupplyChainProblem,
    "Multi-echelon, multi-period supply-chain network design under a horizon carbon budget on production, linehaul (truck/rail/intermodal), and last-mile emissions",
)
