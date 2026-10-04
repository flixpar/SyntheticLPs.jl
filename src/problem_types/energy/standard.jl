using JuMP
using Random
using Distributions

"""
    EconomicDispatchProblem <: ProblemGenerator

Multi-area, multi-period economic dispatch with ramping and a horizon emissions
budget — the base of the energy dispatch family.

# Overview

A look-ahead security-constrained economic dispatch (SCED) run after unit
commitment: the committed fleet of every balancing zone is dispatched hour by
hour over a horizon of 4–168 periods to serve zonal load at minimum cost.

  - **Fleet**: technology-grounded units (nuclear, coal, CCGT, gas CT, oil,
    biomass, hydro, wind, solar; see `ENERGY_TECHNOLOGIES`) with heterogeneous
    sizes, costs and emission rates, placed in zones with regional resource
    mixes. Committed thermal units carry must-run floors; wind, solar and hydro
    follow zone-correlated weather profiles and may be curtailed.
  - **Ramping**: every ramp-limited unit has `x[g,t] − x[g,t−1] ≤ RU_g` and
    `x[g,t−1] − x[g,t] ≤ RD_g` (nuclear 2–5 %/h, coal 8–18 %/h, CCGT 30–60 %/h);
    the first period's window around the pre-horizon output is folded into the
    bounds.
  - **Network**: zones sit on a map joined by lossy tie-lines (a spanning tree
    plus meshing ties). Each tie has two directional flows bounded by its
    transfer capability; the receiving zone gets `(1 − loss)` of what is sent.
  - **Emissions budget**: one horizon-wide cap-and-trade row
    `Σ_{g,t} e_g x[g,t] ≤ emission_cap` couples every emitting unit in every
    period.

Size grows with the target through the horizon (snapped to natural planning
lengths), the number of zones (≈ per-period columns / 16), and the fleet — not by
replicating identical blocks. Variables: `n_periods × (n_units + 2·n_ties)`,
matching the target to within one unit per period.

# Feasibility control

  - `feasible`: tie flows are planted first; each zone's controllable units then
    track its natural residual load inside their ramp windows, and zonal demand
    is defined as the resulting supply plus net imports, so the planted
    `EnergyDispatchWitness` satisfies every row exactly. The emissions budget is
    3–15 % above the witness's emissions.
  - `infeasible`: the same planted data, then one of two contradictions that need
    many rows to expose (presolve sees no single violated row):
    `:emissions_budget` — the cap is set below a lower bound on horizon emissions
    implied by the balance rows and the clean fleet's capacity; or
    `:import_pocket` — a connected set of ≥ 2 zones has its peak-hour demand
    raised above local capacity plus delivered import capability (no zone and no
    proper subset of the pocket is short on its own). The typed
    `EnergyAggregateCertificate` records the bound and the requirement.
  - `unknown`: natural load with a tighter random planning margin, natural
    initial state, and an emissions cap of 60–100 % of the business-as-usual
    (load-tracking) emissions — genuinely undetermined.

# Fields

  - `core::EnergyDispatchCore`: zones, ties, fleet, profiles, demand
  - `emission_cap::Float64`: horizon emissions budget (tCO2)
  - `feasibility_status::FeasibilityStatus`: requested status
  - `feasible_witness::Union{Nothing,EnergyDispatchWitness}`
  - `infeasibility_certificate::Union{Nothing,EnergyAggregateCertificate}`
"""
struct EconomicDispatchProblem <: ProblemGenerator
    core::EnergyDispatchCore
    emission_cap::Float64
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, EnergyDispatchWitness}
    infeasibility_certificate::Union{Nothing, EnergyAggregateCertificate}
end

"""
    _ed_plant_emissions(rng, c) -> Union{Nothing, Tuple{Float64,EnergyAggregateCertificate}}

Pick an emissions budget strictly between the emission row's own minimum
activity (what presolve sees) and 93 % of the balance-implied lower bound.
"""
function _ed_plant_emissions(rng::AbstractRNG, c::EnergyDispatchCore)
    bound, row_min = _ed_emission_lower_bound(c)
    hi = 0.93 * bound
    lo = 1.04 * row_min + 1e-6 * bound
    hi > lo || return nothing
    cap = lo + _e_unif(rng, (0.3, 0.8)) * (hi - lo)
    cert = EnergyAggregateCertificate(:emissions_budget, collect(1:c.n_zones), collect(1:c.n_periods), cap, bound)
    return cap, cert
end

"""
    EconomicDispatchProblem(target_variables, feasibility_status, seed)

Build a multi-area economic dispatch instance with about `target_variables`
columns (`n_periods × (n_units + 2·n_ties)`). See the type docstring.
"""
function EconomicDispatchProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    return _economic_dispatch(target_variables, feasibility_status, seed)
end

"""
    _economic_dispatch(target, status, seed; infeasible_mode=nothing)

Constructor body. `infeasible_mode` (`:emissions_budget` or `:import_pocket`)
forces the contradiction planted for `infeasible` requests (tests use it);
`nothing` samples it (pocket 15 % when there are ≥ 3 zones, else the emissions
budget, which presolve cannot see).
"""
function _economic_dispatch(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int; infeasible_mode=nothing
)
    rng = MersenneTwister(seed)
    T, Z, layout, per_period = _ed_dimensions(rng, target_variables)
    L = length(layout[3])
    techs, zones = _ed_sample_fleet(rng, Z, max(Z, per_period - 2L), _ -> 1)
    margin = feasibility_status == unknown ? _e_unif(rng, (1.02, 1.30)) : _e_unif(rng, (1.12, 1.35))
    nt = _ed_sample_core(rng, layout, T, techs, zones; margin=margin)

    witness = nothing
    certificate = nothing
    if feasibility_status == unknown
        demand, x0 = _ed_natural(rng, nt, techs, zones, Z, T)
        core = _ed_assemble(nt, techs, zones, Z, T, demand, x0)
        # Business-as-usual emissions of a load-tracking dispatch without trade.
        L0 = length(nt.tie_from)
        xb, _, _ = _ed_tracking_dispatch(rng, nt, techs, zones, T, demand, zeros(L0, T), zeros(L0, T))
        cap = _ed_emissions(core, xb) * _e_unif(rng, (0.60, 1.00))
    else
        demand, x0, witness = _ed_plant(rng, nt, techs, zones, Z, T)
        core = _ed_assemble(nt, techs, zones, Z, T, demand, x0)
        cap = _ed_emissions(core, witness.output) * _e_unif(rng, (1.03, 1.15))
        if feasibility_status == infeasible
            witness = nothing
            margin_inf = _e_unif(rng, (0.06, 0.15))
            draw = rand(rng)
            prefer_pocket = infeasible_mode === nothing ? (Z >= 3 && draw < 0.15) : infeasible_mode == :import_pocket
            if prefer_pocket
                certificate = _ed_plant_pocket!(rng, core; margin=margin_inf)
            end
            if certificate === nothing
                planted = _ed_plant_emissions(rng, core)
                if planted !== nothing
                    cap, certificate = planted
                end
            end
            if certificate === nothing
                certificate = _ed_plant_pocket!(rng, core; margin=margin_inf)
            end
            if certificate === nothing
                certificate = _ed_plant_system_shortage!(rng, core; margin=margin_inf)
            end
        end
    end
    return EconomicDispatchProblem(core, cap, feasibility_status, witness, certificate)
end

"""
    build_model(prob::EconomicDispatchProblem)

Dispatch core (bounded unit outputs, directional tie flows, zonal balance
equalities, ramp rows) plus the horizon emissions-budget row; minimize energy
cost plus wheeling charges.
"""
function build_model(prob::EconomicDispatchProblem)
    model = Model()
    c = prob.core
    x, _, _, balance, objective = _ed_core_variables!(model, c)
    _ed_core_rows!(model, c, x, balance)
    emissions = AffExpr(0.0)
    for g in 1:_ed_n_units(c)
        e = c.emission_rate[g]
        e > 0 || continue
        for t in 1:c.n_periods
            add_to_expression!(emissions, e, x[g, t])
        end
    end
    @constraint(model, emissions_budget, emissions <= prob.emission_cap)
    @objective(model, Min, objective)
    return model
end

register_variant(
    :energy,
    :standard,
    EconomicDispatchProblem,
    "Multi-area multi-period economic dispatch: technology-grounded fleet with must-run floors and ramp limits, curtailable renewables, lossy tie-lines between zones, and a horizon emissions budget";
    default=true,
    tags=[:energy, :network],
)
