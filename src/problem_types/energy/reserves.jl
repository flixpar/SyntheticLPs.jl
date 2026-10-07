using JuMP
using Random
using Distributions

"""
Planted feasible point of the reserves variant: the dispatch-core witness plus
spinning reserve `spin[g, t]`, non-spinning reserve `nonspin[g, t]` (zero for
units without the product) and the largest-contingency level `contingency[t]`.
"""
struct ReserveDispatchWitness
    dispatch::EnergyDispatchWitness
    spin::Matrix{Float64}
    nonspin::Matrix{Float64}
    contingency::Vector{Float64}
end

"""
    ReservesDispatchProblem <: ProblemGenerator

Multi-area economic dispatch co-optimized with operating reserves.

# Overview

The dispatch core of `energy/standard` (zones, lossy tie-lines, a committed
fleet with must-run floors and ramp limits, curtailable renewables; see
`EnergyDispatchCore`) without the emissions budget, plus energy–reserve
co-optimization as cleared by US ISOs (PJM, MISO, ERCOT):

  - **Products**: 10-minute spinning reserve `spin[g,t]` (synchronized
    thermal and hydro units, capped by 10-minute ramp capability) and 30-minute
    non-spinning reserve `nonspin[g,t]` (capped by 30-minute capability; peakers
    can offer their full capacity). Wind, solar and nuclear offer neither.
  - **Headroom**: `x + spin + nonspin ≤ available capacity` per unit-period.
  - **Deliverability within the ramp**: `x[g,t] + spin[g,t] − x[g,t−1] ≤ RU_g`
    — reserve held must be deployable on top of the scheduled ramp.
  - **Dynamic largest contingency**: `contingency[t] ≥ x[h,t] + spin[h,t]` for
    the largest units `h`, and `Σ_g spin[g,t] ≥ contingency[t]` — spinning
    reserve must cover the loss of the largest online unit (and the reserve it
    was carrying), an N-1 rule that ties reserve to dispatch.
  - **Operating reserve**: `Σ_g (spin + nonspin)[g,t] ≥ R_t` (≈ 5–12 % of load).
  - **Zonal deliverability**: each zone holds a minimum local spinning share.

Columns per period: units + 2·ties + spin- and non-spin-capable units + 1;
the fleet is sampled against that budget so the total matches the target.

# Feasibility control

  - `feasible`: the planted dispatch witness (see `energy/standard`) is extended
    with reserves carved from each unit's headroom and ramp slack (70 % of the
    room). Requirements are set at or below what the witness carries, and only
    units whose loss the witness's spinning reserve covers (with a 10 % margin)
    enter the contingency set.
  - `infeasible`: `:reserve_scarcity` — in a stressed hour the operating
    reserve requirement (at most 90 % of all reserve offers, so its row alone is
    satisfiable) plus the load (raised if needed by the same share of every
    zone's local headroom, so load alone fits in every zone) exceeds total
    available capacity by 4–10 %. Exposing it needs the balance rows of every
    zone, every headroom row and the requirement row together.
  - `unknown`: natural load with a tighter random planning margin, natural
    requirements (5–12 % operating, 1–3 % local spinning) and, as contingencies,
    the largest units whose loss the fleet's total spinning capability could
    cover.

# Fields

  - `core::EnergyDispatchCore`
  - `spin_max`, `nonspin_max::Vector{Float64}`: per-unit product caps (MW; 0 =
    product not offered)
  - `spin_cost`, `nonspin_cost::Vector{Float64}`: reserve offer prices, \$/MW-h
  - `contingency_units::Vector{Int}`: units whose loss spinning reserve covers
  - `operating_requirement::Vector{Float64}`: system 30-minute requirement per period
  - `zone_spin_requirement::Matrix{Float64}`: local spinning requirement (zone × period)
  - `feasibility_status`, `feasible_witness`, `infeasibility_certificate`
"""
struct ReservesDispatchProblem <: ProblemGenerator
    core::EnergyDispatchCore
    spin_max::Vector{Float64}
    nonspin_max::Vector{Float64}
    spin_cost::Vector{Float64}
    nonspin_cost::Vector{Float64}
    contingency_units::Vector{Int}
    operating_requirement::Vector{Float64}
    zone_spin_requirement::Matrix{Float64}
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, ReserveDispatchWitness}
    infeasibility_certificate::Union{Nothing, EnergyAggregateCertificate}
end

_reserve_cols(tech::Symbol) =
    1 + (ENERGY_TECHNOLOGIES[tech].spin10 > 0) + (ENERGY_TECHNOLOGIES[tech].nonspin30 > 0)

"""Reserve-product caps of every unit from its technology and ramp capability."""
function _reserve_caps(c::EnergyDispatchCore)
    G = _ed_n_units(c)
    spin = zeros(G)
    nonspin = zeros(G)
    for g in 1:G
        spec = ENERGY_TECHNOLOGIES[c.unit_tech[g]]
        base = isfinite(c.ramp_up[g]) ? c.ramp_up[g] : c.capacity[g]
        spin[g] = min(spec.spin10 * base, c.capacity[g])
        nonspin[g] = min(spec.nonspin30 * base, c.capacity[g])
    end
    return spin, nonspin
end

"""
Largest operating reserve the fleet can offer in period `t`: each unit's two
products, capped by its room between the output floor and the available
capacity (what the headroom rows allow even at minimum output).
"""
_reserve_offer(c::EnergyDispatchCore, spin_max, nonspin_max, t::Int) = sum(
    min(spin_max[g] + nonspin_max[g], max(0.0, _ed_upper(c, g, t) - _ed_lower(c, g, t))) for
    g in 1:_ed_n_units(c)
)

"""
    ReservesDispatchProblem(target_variables, feasibility_status, seed)

Build a reserve co-optimization dispatch instance with about `target_variables`
columns. See the type docstring.
"""
function ReservesDispatchProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    T, Z, layout, per_period = _ed_dimensions(rng, target_variables)
    L = length(layout[3])
    techs, zones = _ed_sample_fleet(rng, Z, max(Z, per_period - 2L - 1), _reserve_cols)
    margin = feasibility_status == unknown ? _e_unif(rng, (0.98, 1.25)) : _e_unif(rng, (1.15, 1.40))
    nt = _ed_sample_core(rng, layout, T, techs, zones; margin=margin)
    G = length(techs)

    spin_cost = zeros(G)
    nonspin_cost = zeros(G)
    op_frac = _e_unif(rng, (0.05, 0.12))
    local_frac = _e_unif(rng, (0.01, 0.03))
    witness = nothing
    certificate = nothing

    if feasibility_status == unknown
        demand, x0 = _ed_natural(rng, nt, techs, zones, Z, T)
        core = _ed_assemble(nt, techs, zones, Z, T, demand, x0)
    else
        demand, x0, dw = _ed_plant(rng, nt, techs, zones, Z, T)
        core = _ed_assemble(nt, techs, zones, Z, T, demand, x0)
    end
    spin_max, nonspin_max = _reserve_caps(core)
    for g in 1:G
        # Offer prices: opportunity cost rises with the energy price; peakers'
        # spinning reserve is dear (they must run to provide it).
        spin_cost[g] = spin_max[g] > 0 ? _e_unif(rng, (2.0, 9.0)) + 0.05 * core.cost[g] : 0.0
        nonspin_cost[g] = nonspin_max[g] > 0 ? _e_unif(rng, (0.5, 4.0)) : 0.0
    end
    ranked = sort(
        [g for g in 1:G if !(core.unit_tech[g] in (:wind, :solar))]; by=g -> -core.capacity[g]
    )
    k = clamp(round(Int, 0.04 * G), 1, 25)

    if feasibility_status == unknown
        # The N-1 rule covers the largest units whose loss the fleet's spinning
        # capability could ever cover; requirements never exceed what the fleet
        # can offer at all (a requirement no offer could meet is not a natural
        # instance, just a single infeasible row).
        spin_capability = sum(spin_max)
        contingency_units = [g for g in ranked if core.capacity[g] <= 0.8 * spin_capability]
        contingency_units = contingency_units[1:min(k, length(contingency_units))]
        op_req = [
            min(
                op_frac * _ed_system_demand(core, t),
                0.9 * _reserve_offer(core, spin_max, nonspin_max, t),
            ) for t in 1:T
        ]
        zone_req = zeros(Z, T)
        for z in 1:Z
            cap_z = sum((spin_max[g] for g in 1:G if zones[g] == z); init=0.0)
            for t in 1:T
                req = local_frac * core.demand[z, t]
                zone_req[z, t] = req <= 0.5 * cap_z ? req : 0.0
            end
        end
    else
        # Carve reserves from the witness's headroom and ramp slack.
        x = dw.output
        spin = zeros(G, T)
        nonspin = zeros(G, T)
        for g in 1:G, t in 1:T
            head = _ed_upper(core, g, t) - x[g, t]
            head <= 0 && continue
            ramp_room =
                (_ed_ramped(core, g) && t >= 2) ? core.ramp_up[g] - (x[g, t] - x[g, t - 1]) : Inf
            spin[g, t] = 0.7 * max(0.0, min(spin_max[g], head, ramp_room))
            nonspin[g, t] = 0.7 * max(0.0, min(nonspin_max[g], head - spin[g, t]))
        end
        total_spin = vec(sum(spin; dims=1))
        contingency_units = Int[]
        for h in ranked
            length(contingency_units) >= k && break
            if all(x[h, t] + spin[h, t] <= 0.9 * total_spin[t] for t in 1:T)
                push!(contingency_units, h)
            end
        end
        contingency = [
            if isempty(contingency_units)
                0.0
            else
                maximum(x[h, t] + spin[h, t] for h in contingency_units)
            end for t in 1:T
        ]
        carried = vec(sum(spin .+ nonspin; dims=1))
        op_req = [min(op_frac * _ed_system_demand(core, t), 0.9 * carried[t]) for t in 1:T]
        zone_req = zeros(Z, T)
        for g in 1:G, t in 1:T
            zone_req[zones[g], t] += spin[g, t]
        end
        for z in 1:Z, t in 1:T
            zone_req[z, t] = min(local_frac * core.demand[z, t], 0.85 * zone_req[z, t])
        end
        witness = ReserveDispatchWitness(dw, spin, nonspin, contingency)

        if feasibility_status == infeasible
            witness = nothing
            m = _e_unif(rng, (0.04, 0.10))
            # A full outage has zero demand and capacity; it cannot certify a
            # deficit and its 0/0 stress would otherwise sort ahead of real hours.
            capacity = [_ed_system_upper(core, t) for t in 1:T]
            stress = [
                capacity[t] > 0 ? (_ed_system_demand(core, t) + op_req[t]) / capacity[t] : -Inf for
                t in 1:T
            ]
            for t in sortperm(stress; rev=true)
                cap_t = capacity[t]
                cap_t > 0 || continue
                offer = _reserve_offer(core, spin_max, nonspin_max, t)
                # Requirement at most 90 % of all offers (its row alone stays
                # satisfiable); load makes up the rest, raised (if needed) by
                # giving every zone the same share of its local headroom, so
                # load alone still fits in every zone.
                req = min(0.9 * offer, (1 + m) * cap_t - _ed_system_demand(core, t))
                load = (1 + m) * cap_t - req
                load <= 0.97 * cap_t || continue
                _ed_raise_system_demand!(core, t, load) || continue
                requirement = _ed_system_demand(core, t) + req
                requirement > cap_t || continue
                op_req[t] = req
                certificate = EnergyAggregateCertificate(
                    :reserve_scarcity, collect(1:Z), [t], cap_t, requirement
                )
                break
            end
            if certificate === nothing
                certificate = _ed_plant_system_shortage!(rng, core; margin=m)
            end
        end
    end
    return ReservesDispatchProblem(
        core,
        spin_max,
        nonspin_max,
        spin_cost,
        nonspin_cost,
        contingency_units,
        op_req,
        zone_req,
        feasibility_status,
        witness,
        certificate,
    )
end

"""
    build_model(prob::ReservesDispatchProblem)

Dispatch core + reserve products, headroom rows, reserve-aware up-ramp rows,
largest-contingency rows, the operating-reserve requirement and zonal local
spinning requirements; minimize energy + wheeling + reserve offer cost.
"""
function build_model(prob::ReservesDispatchProblem)
    model = Model()
    c = prob.core
    G = _ed_n_units(c)
    T = c.n_periods
    x, _, _, balance, objective = _ed_core_variables!(model, c)

    spin_units = [g for g in 1:G if prob.spin_max[g] > 0]
    nonspin_units = [g for g in 1:G if prob.nonspin_max[g] > 0]
    @variable(model, 0 <= spin[g in spin_units, t = 1:T] <= prob.spin_max[g])
    @variable(model, 0 <= nonspin[g in nonspin_units, t = 1:T] <= prob.nonspin_max[g])
    @variable(model, contingency[t = 1:T] >= 0)
    has_spin = falses(G)
    has_spin[spin_units] .= true
    has_nonspin = falses(G)
    has_nonspin[nonspin_units] .= true

    for g in spin_units, t in 1:T
        add_to_expression!(objective, prob.spin_cost[g], spin[g, t])
    end
    for g in nonspin_units, t in 1:T
        add_to_expression!(objective, prob.nonspin_cost[g], nonspin[g, t])
    end

    spin_rows = [has_spin[g] ? [spin[g, t] for t in 1:T] : nothing for g in 1:G]
    _ed_core_rows!(model, c, x, balance; spin=spin_rows)

    # Headroom: energy and both reserve products share the available capacity.
    for g in 1:G
        (has_spin[g] || has_nonspin[g]) || continue
        for t in 1:T
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, x[g, t])
            has_spin[g] && add_to_expression!(expr, 1.0, spin[g, t])
            has_nonspin[g] && add_to_expression!(expr, 1.0, nonspin[g, t])
            @constraint(model, expr <= _ed_upper(c, g, t))
        end
    end

    # Dynamic largest contingency covered by spinning reserve.
    for h in prob.contingency_units, t in 1:T
        if has_spin[h]
            @constraint(model, contingency[t] - x[h, t] - spin[h, t] >= 0)
        else
            @constraint(model, contingency[t] - x[h, t] >= 0)
        end
    end
    for t in 1:T
        @constraint(
            model, sum(spin[g, t] for g in spin_units; init=AffExpr(0.0)) - contingency[t] >= 0
        )
        op = AffExpr(0.0)
        for g in spin_units
            add_to_expression!(op, 1.0, spin[g, t])
        end
        for g in nonspin_units
            add_to_expression!(op, 1.0, nonspin[g, t])
        end
        @constraint(model, op >= prob.operating_requirement[t])
    end

    # Zonal local spinning requirement (deliverability).
    by_zone = _ed_units_by_zone(c)
    for z in 1:c.n_zones, t in 1:T
        prob.zone_spin_requirement[z, t] > 0 || continue
        expr = AffExpr(0.0)
        for g in by_zone[z]
            has_spin[g] && add_to_expression!(expr, 1.0, spin[g, t])
        end
        @constraint(model, expr >= prob.zone_spin_requirement[z, t])
    end

    @objective(model, Min, objective)
    return model
end

register_variant(
    :energy,
    :reserves,
    ReservesDispatchProblem,
    "Multi-area dispatch co-optimized with spinning and non-spinning reserves: shared headroom, reserve-aware ramping, a dynamic largest-unit contingency rule, and system and zonal requirements";
    tags=[:energy, :network],
)
