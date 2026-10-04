using JuMP
using Random
using Distributions

"""
Planted feasible point of the storage variant: the dispatch-core witness plus
per-device charging `charge[s, t]`, discharging `discharge[s, t]` and state of
charge `soc[s, t]` (MWh at the end of period `t`).
"""
struct StorageDispatchWitness
    dispatch::EnergyDispatchWitness
    charge::Matrix{Float64}
    discharge::Matrix{Float64}
    soc::Matrix{Float64}
end

"""
    StorageDispatchProblem <: ProblemGenerator

Multi-area economic dispatch with grid-scale energy storage.

# Overview

The dispatch core of `energy/standard` (zones, lossy tie-lines, a committed
fleet with must-run floors and ramp limits, curtailable wind and solar; see
`EnergyDispatchCore`) without the emissions budget, plus a fleet of storage
devices sited in the zones:

  - **Devices**: lithium-ion batteries (2–4 h duration, ≈ 85–92 % round trip,
    small degradation cost) and pumped hydro (6–12 h, ≈ 72–80 % round trip).
    Charging power, discharging power and energy capacity are separate limits.
  - **State of charge**: `soc[s,t] = soc[s,t−1] + η_ch·charge[s,t] −
    discharge[s,t]/η_dis` (the initial level is data), bounded by
    `[soc_min, soc_max]`; the end-of-horizon level must be at least the initial
    one (no free draindown), folded into the last period's bound.
  - Storage net discharge enters its zone's balance, so storage arbitrages
    across hours (cheap night/solar energy into the evening peak), absorbs
    curtailable renewables, and couples periods far more than ramp rows do.

Columns per period: units + 2·ties + 3·devices (≈ one device per 25 per-period
columns); the fleet absorbs the rest of the budget exactly.

# Feasibility control

  - `feasible`: the planted dispatch witness tracks load net of a planted daily
    storage cycle (charge in the lowest-load hours of each day, discharge the
    same energy times the round-trip efficiency in the highest-load hours,
    returning exactly to the initial level). The `StorageDispatchWitness` meets
    every row.
  - `infeasible`: `:energy_limited_peak` — over a window of consecutive peak
    hours the load is raised above available generation by more than the
    storage fleet's usable energy (`Σ η_dis·(soc_max − soc_min)`, plus a 6–15 %
    margin), while every single hour stays within generation plus storage
    *power*. Only the sum over the window of the balance and state-of-charge
    rows exposes it; no single row does.
  - `unknown`: natural load and a tighter random planning margin, natural
    initial storage levels.

# Fields

  - `core::EnergyDispatchCore`
  - `storage_zone::Vector{Int}`, `storage_kind::Vector{Symbol}` (`:battery` /
    `:pumped_hydro`)
  - `charge_max`, `discharge_max` (MW), `soc_min`, `soc_max`, `soc_initial`
    (MWh), `eta_charge`, `eta_discharge`, `cycle_cost` (\$/MWh discharged)
  - `feasibility_status`, `feasible_witness`, `infeasibility_certificate`
"""
struct StorageDispatchProblem <: ProblemGenerator
    core::EnergyDispatchCore
    storage_zone::Vector{Int}
    storage_kind::Vector{Symbol}
    charge_max::Vector{Float64}
    discharge_max::Vector{Float64}
    soc_min::Vector{Float64}
    soc_max::Vector{Float64}
    soc_initial::Vector{Float64}
    eta_charge::Vector{Float64}
    eta_discharge::Vector{Float64}
    cycle_cost::Vector{Float64}
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, StorageDispatchWitness}
    infeasibility_certificate::Union{Nothing, EnergyAggregateCertificate}
end

"""
    _storage_cycle(T, sysload, P_ch, P_dis, η_ch, η_dis, e0, emin, emax)

Planted daily cycle for one device: in each 24-hour block (or the whole horizon
when shorter) charge in the `k` lowest-load hours and discharge the stored
energy in the `k` highest-load hours, scaled so the state of charge stays inside
its bounds and returns to `e0` at the end of every block.
"""
function _storage_cycle(T, sysload, P_ch, P_dis, η_ch, η_dis, e0, emin, emax)
    charge = zeros(T)
    discharge = zeros(T)
    block = min(T, 24)
    k = max(1, block ÷ 6)
    start = 1
    while start + block - 1 <= T
        hours = start:(start + block - 1)
        order = sortperm(sysload[hours])
        low = hours[order[1:k]]
        high = hours[order[(end - k + 1):end]]
        # Per-hour charge level a: energy in = η_ch·a·k, out = η_dis·η_ch·a·k
        # spread over k discharge hours.
        a = min(P_ch, P_dis / η_dis / η_ch)
        # Keep the SoC trajectory inside its band (worst case: all charging or
        # all discharging happens first).
        a = min(a, 0.9 * (emax - e0) / (η_ch * k), 0.9 * (e0 - emin) / (η_ch * k))
        a = 0.8 * max(a, 0.0)
        charge[low] .= a
        discharge[high] .= η_dis * η_ch * a
        start += block
    end
    soc = zeros(T)
    e = e0
    for t in 1:T
        e = e + η_ch * charge[t] - discharge[t] / η_dis
        soc[t] = e
    end
    return charge, discharge, soc
end

"""
    StorageDispatchProblem(target_variables, feasibility_status, seed)

Build a storage-coupled dispatch instance with about `target_variables`
columns. See the type docstring.
"""
function StorageDispatchProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    T, Z, layout, per_period = _ed_dimensions(rng, target_variables)
    L = length(layout[3])
    S = max(1, round(Int, per_period / 25))
    techs, zones = _ed_sample_fleet(rng, Z, max(Z, per_period - 2L - 3S), _ -> 1)
    margin = feasibility_status == unknown ? _e_unif(rng, (0.80, 1.12)) : _e_unif(rng, (1.12, 1.35))
    nt = _ed_sample_core(rng, layout, T, techs, zones; margin=margin)

    # Storage fleet, sited in proportion to zonal peak load.
    zone_peak = [maximum(nt.natural_demand[z, :]) for z in 1:Z]
    zcdf = cumsum(zone_peak ./ sum(zone_peak))
    storage_zone = [min(Z, searchsortedfirst(zcdf, rand(rng))) for _ in 1:S]
    storage_kind = [rand(rng) < 0.7 ? :battery : :pumped_hydro for _ in 1:S]
    per_device = sum(zone_peak) * _e_unif(rng, (0.10, 0.25)) / S
    charge_max = zeros(S)
    discharge_max = zeros(S)
    soc_min = zeros(S)
    soc_max = zeros(S)
    soc_initial = zeros(S)
    eta_ch = zeros(S)
    eta_dis = zeros(S)
    cycle_cost = zeros(S)
    for s in 1:S
        P = per_device * exp(0.4 * randn(rng))
        if storage_kind[s] == :battery
            hours = _e_unif(rng, (2.0, 4.0))
            rt = _e_unif(rng, (0.85, 0.92))
            discharge_max[s] = P
            charge_max[s] = P
            soc_min[s] = 0.05 * P * hours
            cycle_cost[s] = _e_unif(rng, (2.0, 8.0))
        else
            hours = _e_unif(rng, (6.0, 12.0))
            rt = _e_unif(rng, (0.72, 0.80))
            discharge_max[s] = P
            charge_max[s] = P * _e_unif(rng, (0.85, 1.0))
            soc_min[s] = 0.10 * P * hours
            cycle_cost[s] = _e_unif(rng, (0.5, 2.0))
        end
        soc_max[s] = P * hours
        eta_ch[s] = sqrt(rt)
        eta_dis[s] = sqrt(rt)
        soc_initial[s] = soc_min[s] + _e_unif(rng, (0.3, 0.6)) * (soc_max[s] - soc_min[s])
    end

    witness = nothing
    certificate = nothing
    if feasibility_status == unknown
        demand, x0 = _ed_natural(rng, nt, techs, zones, Z, T)
        core = _ed_assemble(nt, techs, zones, Z, T, demand, x0)
    else
        sysload = vec(sum(nt.natural_demand; dims=1))
        charge = zeros(S, T)
        discharge = zeros(S, T)
        soc = zeros(S, T)
        net = zeros(Z, T)
        for s in 1:S
            c_s, d_s, e_s = _storage_cycle(
                T, sysload, charge_max[s], discharge_max[s], eta_ch[s], eta_dis[s], soc_initial[s],
                soc_min[s], soc_max[s],
            )
            charge[s, :] .= c_s
            discharge[s, :] .= d_s
            soc[s, :] .= e_s
            net[storage_zone[s], :] .+= d_s .- c_s
        end
        demand, x0, dw = _ed_plant(rng, nt, techs, zones, Z, T; extra=net)
        core = _ed_assemble(nt, techs, zones, Z, T, demand, x0)
        witness = StorageDispatchWitness(dw, charge, discharge, soc)

        if feasibility_status == infeasible
            witness = nothing
            m = _e_unif(rng, (0.06, 0.15))
            energy = sum(eta_dis[s] * (soc_max[s] - soc_min[s]) for s in 1:S)
            power = sum(discharge_max)
            width = clamp(ceil(Int, (1 + m) * energy / (0.8 * power)), 1, T)
            stress = [_ed_system_demand(core, t) - _ed_system_upper(core, t) for t in 1:T]
            # Window of `width` consecutive hours with the largest total stress.
            best = 1
            bestv = -Inf
            for s0 in 1:(T - width + 1)
                v = sum(stress[s0:(s0 + width - 1)])
                v > bestv && ((bestv, best) = (v, s0))
            end
            W = collect(best:(best + width - 1))
            delta = (1 + m) * energy / width
            local_dis = zeros(Z)
            for s in 1:S
                local_dis[storage_zone[s]] += discharge_max[s]
            end
            ok = true
            for t in W
                target = _ed_system_upper(core, t) + delta
                ok &= _ed_spread_demand!(core, collect(1:Z), t, target; extra=local_dis)
            end
            ok || error("energy/storage: could not plant the energy-limited peak")
            bound = sum(_ed_system_upper(core, t) for t in W) + energy
            req = sum(_ed_system_demand(core, t) for t in W)
            certificate = EnergyAggregateCertificate(:energy_limited_peak, collect(1:Z), W, bound, req)
        end
    end

    return StorageDispatchProblem(
        core,
        storage_zone,
        storage_kind,
        charge_max,
        discharge_max,
        soc_min,
        soc_max,
        soc_initial,
        eta_ch,
        eta_dis,
        cycle_cost,
        feasibility_status,
        witness,
        certificate,
    )
end

"""
    build_model(prob::StorageDispatchProblem)

Dispatch core + storage charge/discharge/state-of-charge columns, the
state-of-charge recursion, and storage net discharge in the zonal balances;
minimize energy + wheeling + cycling cost.
"""
function build_model(prob::StorageDispatchProblem)
    model = Model()
    c = prob.core
    T = c.n_periods
    S = length(prob.storage_zone)
    x, _, _, balance, objective = _ed_core_variables!(model, c)

    @variable(model, 0 <= charge[s=1:S, t=1:T] <= prob.charge_max[s])
    @variable(model, 0 <= discharge[s=1:S, t=1:T] <= prob.discharge_max[s])
    # Terminal level ≥ initial level, folded into the last period's bound.
    soc_lb = [t == T ? max(prob.soc_min[s], prob.soc_initial[s]) : prob.soc_min[s] for s in 1:S, t in 1:T]
    @variable(model, soc_lb[s, t] <= soc[s=1:S, t=1:T] <= prob.soc_max[s])

    for s in 1:S, t in 1:T
        z = prob.storage_zone[s]
        add_to_expression!(balance[z, t], 1.0, discharge[s, t])
        add_to_expression!(balance[z, t], -1.0, charge[s, t])
        add_to_expression!(objective, prob.cycle_cost[s], discharge[s, t])
    end
    _ed_core_rows!(model, c, x, balance)

    for s in 1:S, t in 1:T
        prev = t == 1 ? prob.soc_initial[s] : soc[s, t - 1]
        @constraint(
            model,
            soc[s, t] - prob.eta_charge[s] * charge[s, t] + discharge[s, t] / prob.eta_discharge[s] == prev
        )
    end

    @objective(model, Min, objective)
    return model
end

register_variant(
    :energy,
    :storage,
    StorageDispatchProblem,
    "Multi-area dispatch with batteries and pumped hydro: charge/discharge limits, state-of-charge recursion with round-trip losses, terminal level floor, and storage arbitrage across zones and hours",
)
