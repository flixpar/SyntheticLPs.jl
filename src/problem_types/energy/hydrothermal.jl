using JuMP
using Random
using Distributions

"""
Planted feasible point of the hydrothermal variant: the dispatch-core witness
plus per-plant turbined release `release[r, t]`, spill `spill[r, t]` (m³/s) and
end-of-period reservoir content `volume[r, t]` (in flow-hours, m³/s·h).
"""
struct HydrothermalWitness
    dispatch::EnergyDispatchWitness
    release::Matrix{Float64}
    spill::Matrix{Float64}
    volume::Matrix{Float64}
end

"""
Drought certificate for one river basin. Summing the water-balance rows of the
basin's plants over the horizon, every unit of water that leaves the basin
through the outlet `outlet` (turbined or spilled) must come from the basin's
initial storage above the end-of-horizon floors, from natural inflows, or from
releases made before the horizon that arrive during it:
`Σ_t (release + spill)[outlet, t] ≤ available_water`. The outlet's
minimum-release rows demand `required_release = T·min_release[outlet] >
available_water`.
"""
struct HydroDroughtCertificate
    basin::Vector{Int}
    outlet::Int
    available_water::Float64
    required_release::Float64
end

"""
    HydrothermalDispatchProblem <: ProblemGenerator

Short-term hydrothermal scheduling: multi-area dispatch with cascaded
reservoirs.

# Overview

The dispatch core of `energy/standard` (zones, lossy tie-lines, a committed
thermal and renewable fleet with ramp limits; see `EnergyDispatchCore`) without
the emissions budget, where the hydro fleet is a set of river basins:

  - **Cascades**: each basin is a chain of 2–5 reservoir plants (occasionally a
    tributary joins further downstream). Water released or spilled by a plant
    reaches the next plant after a travel delay of 0–3 hours; releases made
    before the horizon still arrive during it (data).
  - **Water balance**: `volume[r,t] = volume[r,t−1] + inflow[r,t] + Σ_{u → r}
    (release + spill)[u, t − delay_u] − release[r,t] − spill[r,t]`, with
    reservoir bounds and an end-of-horizon floor.
  - **Generation**: `productivity_r · release[r,t]` MW enters the plant's zone
    balance (head-dependent productivity: non-±1 coefficients).
  - **Environmental flows**: some plants must pass a minimum flow
    `release + spill ≥ min_release`.
  - **Water value**: stored water left at the end is credited at a price per MWh
    times the productivity of every plant it can still pass, so the LP trades
    thermal cost now against hydro energy later.

Columns per period: thermal/renewable units + 2·ties + 3·plants (≈ one plant
per 12 per-period columns), matched to the target.

# Feasibility control

  - `feasible`: a planted release schedule (peaking with the load shape, kept
    inside the reservoir band, never below the environmental flow) is simulated
    down each cascade; the thermal witness tracks load net of hydro output and
    zonal demand is defined from the result. End-of-horizon floors sit at or
    below the simulated final volumes.
  - `infeasible`: `:drought` — one basin's reservoirs start near their floors
    (which may not be drawn below), inflows fall to 20–40 %, and the outlet's
    environmental minimum flow over the horizon exceeds all the water that can
    reach it by 8–20 %. Exposing it needs the basin's water-balance rows over
    every period, so no single row (and no short propagation chain) shows it.
    `HydroDroughtCertificate` records the water budget.
  - `unknown`: natural inflow conditions (wet to dry), natural end-of-horizon
    targets (80–105 % of the initial content) and natural load.

# Fields

  - `core::EnergyDispatchCore` (no `:hydro` units — hydro lives in the basins)
  - `plant_zone`, `plant_basin`, `downstream` (0 = basin outlet), `delay`
  - `productivity` (MW per m³/s), `max_release` (m³/s), `min_release`
    (environmental flow, 0 = none), `volume_min`, `volume_max`,
    `volume_initial`, `volume_target` (flow-hours)
  - `inflow::Matrix{Float64}` (plant × period), `prior_release` (pre-horizon
    release + spill per plant, m³/s), `water_value` (\$ per flow-hour stored)
  - `spill_cost::Float64`
  - `feasibility_status`, `feasible_witness`, `infeasibility_certificate`
"""
struct HydrothermalDispatchProblem <: ProblemGenerator
    core::EnergyDispatchCore
    plant_zone::Vector{Int}
    plant_basin::Vector{Int}
    downstream::Vector{Int}
    delay::Vector{Int}
    productivity::Vector{Float64}
    max_release::Vector{Float64}
    min_release::Vector{Float64}
    volume_min::Vector{Float64}
    volume_max::Vector{Float64}
    volume_initial::Vector{Float64}
    volume_target::Vector{Float64}
    inflow::Matrix{Float64}
    prior_release::Vector{Float64}
    water_value::Vector{Float64}
    spill_cost::Float64
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, HydrothermalWitness}
    infeasibility_certificate::Union{Nothing, HydroDroughtCertificate}
end

"""Upstream plants feeding each plant directly."""
function _hydro_upstream(downstream::Vector{Int})
    up = [Int[] for _ in eachindex(downstream)]
    for u in eachindex(downstream)
        downstream[u] > 0 && push!(up[downstream[u]], u)
    end
    return up
end

"""
    _hydro_arrivals(r, t, up, delay, release, spill, prior) -> Float64

Water reaching plant `r` in period `t` from its direct upstream plants: their
release + spill `delay` periods earlier, or their pre-horizon release when that
falls before the horizon.
"""
function _hydro_arrivals(r, t, up, delay, release, spill, prior)
    a = 0.0
    for u in up[r]
        s = t - delay[u]
        a += s >= 1 ? release[u, s] + spill[u, s] : prior[u]
    end
    return a
end

"""
    _hydro_available_water(prob, basin) -> Float64

Water that can leave `basin` through its outlet over the horizon: initial
storage above each plant's end-of-horizon floor, all natural inflows, and
pre-horizon releases arriving during the horizon.
"""
function _hydro_available_water(p, basin::Vector{Int})
    T = size(p.inflow, 2)
    w = 0.0
    for r in basin
        w += p.volume_initial[r] - max(p.volume_min[r], p.volume_target[r])
        w += sum(@view p.inflow[r, :])
        p.downstream[r] > 0 && (w += min(p.delay[r], T) * p.prior_release[r])
    end
    return w
end

"""
    HydrothermalDispatchProblem(target_variables, feasibility_status, seed)

Build a hydrothermal scheduling instance with about `target_variables`
columns. See the type docstring.
"""
function HydrothermalDispatchProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    T, Z, layout, per_period = _ed_dimensions(rng, target_variables)
    L = length(layout[3])
    P = max(2, round(Int, per_period / 12))
    techs, zones = _ed_sample_fleet(rng, Z, max(Z, per_period - 2L - 3P), _ -> 1; exclude=(:hydro,))
    margin = feasibility_status == unknown ? _e_unif(rng, (0.88, 1.20)) : _e_unif(rng, (1.10, 1.30))
    nt = _ed_sample_core(rng, layout, T, techs, zones; margin=margin)

    # --- River basins -------------------------------------------------------
    plant_basin = zeros(Int, P)
    downstream = zeros(Int, P)
    delay = zeros(Int, P)
    plant_zone = zeros(Int, P)
    r = 1
    nb = 0
    while r <= P
        k = min(P - r + 1, rand(rng, 2:5))
        P - (r + k - 1) == 1 && (k += 1)   # never leave a lone plant behind
        nb += 1
        z = rand(rng, 1:Z)
        for i in 0:(k - 1)
            plant_basin[r + i] = nb
            plant_zone[r + i] = rand(rng) < 0.8 ? z : rand(rng, 1:Z)
            if i < k - 1
                jump = (i < k - 2 && rand(rng) < 0.3) ? 2 : 1   # tributary joins further down
                downstream[r + i] = r + i + jump
                delay[r + i] = rand(rng, 0:3)
            end
        end
        r += k
    end
    up = _hydro_upstream(downstream)
    productivity = [_e_logunif(rng, (0.2, 1.5)) for _ in 1:P]
    max_release = [_e_logunif(rng, (50.0, 600.0)) for _ in 1:P]
    volume_max = [max_release[r] * _e_logunif(rng, (12.0, 400.0)) for r in 1:P]
    volume_min = 0.1 .* volume_max
    volume_initial = [volume_min[r] + _e_unif(rng, (0.3, 0.75)) * (volume_max[r] - volume_min[r]) for r in 1:P]
    headwater = [isempty(up[r]) for r in 1:P]
    base_inflow = [max_release[r] * (headwater[r] ? _e_unif(rng, (0.15, 0.5)) : _e_unif(rng, (0.02, 0.12))) for r in 1:P]
    wetness = feasibility_status == unknown ? _e_unif(rng, (0.5, 1.4)) : 1.0
    inflow = zeros(P, T)
    for b in 1:nb
        level = 1.0
        for t in 1:T
            (t - 1) % 24 == 0 && (level = clamp(level * exp(0.1 * randn(rng)), 0.5, 1.8))
            for r in 1:P
                plant_basin[r] == b || continue
                inflow[r, t] = wetness * base_inflow[r] * level * (1 + 0.03 * randn(rng))
            end
        end
    end
    min_release = [rand(rng) < 0.5 ? _e_unif(rng, (0.03, 0.10)) * max_release[r] : 0.0 for r in 1:P]
    prior_release = [base_inflow[r] + _e_unif(rng, (0.2, 0.5)) * max_release[r] for r in 1:P]
    for r in 1:P
        prior_release[r] = max(prior_release[r], min_release[r])
    end
    price = _e_unif(rng, (20.0, 45.0))
    water_value = zeros(P)
    for r in 1:P
        d = r
        while d > 0
            water_value[r] += price * productivity[d]
            d = downstream[d]
        end
    end
    spill_cost = 0.05

    witness = nothing
    certificate = nothing
    if feasibility_status == unknown
        demand, x0 = _ed_natural(rng, nt, techs, zones, Z, T)
        core = _ed_assemble(nt, techs, zones, Z, T, demand, x0)
        volume_target = [volume_initial[r] * _e_unif(rng, (0.80, 1.05)) for r in 1:P]
    else
        # Planted release schedule: peaking with the system load shape, kept in
        # the reservoir band, simulated down each cascade (index order is
        # upstream-first).
        sysload = vec(sum(nt.natural_demand; dims=1))
        shape = sysload ./ maximum(sysload)
        release = zeros(P, T)
        spill = zeros(P, T)
        volume = zeros(P, T)
        for r in 1:P
            u = _e_unif(rng, (0.2, 0.6))
            v = volume_initial[r]
            band_lo = volume_min[r] + 0.05 * (volume_max[r] - volume_min[r])
            band_hi = volume_max[r] - 0.05 * (volume_max[r] - volume_min[r])
            for t in 1:T
                water = inflow[r, t] + _hydro_arrivals(r, t, up, delay, release, spill, prior_release)
                q = max(min_release[r], u * (0.6 + 0.4 * shape[t]) * max_release[r])
                q = min(q, max_release[r])
                if v + water - q < band_lo
                    q = max(min_release[r], v + water - band_lo)
                    if v + water - q < band_lo
                        # Not enough water even at the environmental flow: the
                        # planted inflow is topped up (it is data).
                        inflow[r, t] += band_lo - (v + water - q)
                        water = inflow[r, t] + _hydro_arrivals(r, t, up, delay, release, spill, prior_release)
                    end
                end
                s = 0.0
                if v + water - q > band_hi
                    q = min(max_release[r], v + water - band_hi)
                    s = max(0.0, v + water - q - band_hi)
                end
                release[r, t] = q
                spill[r, t] = s
                v = v + water - q - s
                volume[r, t] = v
            end
        end
        volume_target = [min(volume[r, T], volume_initial[r] * _e_unif(rng, (0.85, 1.0))) for r in 1:P]
        hydro = zeros(Z, T)
        for r in 1:P, t in 1:T
            hydro[plant_zone[r], t] += productivity[r] * release[r, t]
        end
        demand, x0, dw = _ed_plant(rng, nt, techs, zones, Z, T; extra=hydro)
        core = _ed_assemble(nt, techs, zones, Z, T, demand, x0)
        witness = HydrothermalWitness(dw, release, spill, volume)

        if feasibility_status == infeasible
            witness = nothing
            # Drought in one basin: reservoirs near their floors and held there,
            # low inflows, and an outlet minimum flow above the water budget.
            b = rand(rng, 1:nb)
            basin = [r for r in 1:P if plant_basin[r] == b]
            outlet = basin[end]
            dry = _e_unif(rng, (0.2, 0.4))
            for r in basin
                volume_initial[r] = volume_min[r] + _e_unif(rng, (0.02, 0.08)) * (volume_max[r] - volume_min[r])
                volume_target[r] = volume_initial[r]
                inflow[r, :] .*= dry
            end
            tmp = (; volume_initial, volume_min, volume_target, inflow, downstream, delay, prior_release)
            avail = _hydro_available_water(tmp, basin)
            m = _e_unif(rng, (0.08, 0.20))
            min_release[outlet] = (1 + m) * avail / T
            certificate = HydroDroughtCertificate(basin, outlet, avail, T * min_release[outlet])
        end
    end

    return HydrothermalDispatchProblem(
        core,
        plant_zone,
        plant_basin,
        downstream,
        delay,
        productivity,
        max_release,
        min_release,
        volume_min,
        volume_max,
        volume_initial,
        volume_target,
        inflow,
        prior_release,
        water_value,
        spill_cost,
        feasibility_status,
        witness,
        certificate,
    )
end

"""
    build_model(prob::HydrothermalDispatchProblem)

Dispatch core + per-plant release, spill and volume columns, cascaded
water-balance rows with travel delays, environmental-flow rows and hydro
generation in the zonal balances; minimize thermal cost + wheeling + spill cost
− value of the water left at the end of the horizon.
"""
function build_model(prob::HydrothermalDispatchProblem)
    model = Model()
    c = prob.core
    T = c.n_periods
    P = length(prob.plant_zone)
    x, _, _, balance, objective = _ed_core_variables!(model, c)

    @variable(model, 0 <= release[r=1:P, t=1:T] <= prob.max_release[r])
    @variable(model, spill[r=1:P, t=1:T] >= 0)
    vol_lb = [t == T ? max(prob.volume_min[r], prob.volume_target[r]) : prob.volume_min[r] for r in 1:P, t in 1:T]
    @variable(model, vol_lb[r, t] <= volume[r=1:P, t=1:T] <= prob.volume_max[r])

    up = _hydro_upstream(prob.downstream)
    for r in 1:P, t in 1:T
        add_to_expression!(balance[prob.plant_zone[r], t], prob.productivity[r], release[r, t])
        add_to_expression!(objective, prob.spill_cost, spill[r, t])
    end
    for r in 1:P
        add_to_expression!(objective, -prob.water_value[r], volume[r, T])
    end
    _ed_core_rows!(model, c, x, balance)

    for r in 1:P, t in 1:T
        lhs = AffExpr(0.0)
        add_to_expression!(lhs, 1.0, volume[r, t])
        t > 1 && add_to_expression!(lhs, -1.0, volume[r, t - 1])
        add_to_expression!(lhs, 1.0, release[r, t])
        add_to_expression!(lhs, 1.0, spill[r, t])
        rhs = prob.inflow[r, t] + (t == 1 ? prob.volume_initial[r] : 0.0)
        for u in up[r]
            s = t - prob.delay[u]
            if s >= 1
                add_to_expression!(lhs, -1.0, release[u, s])
                add_to_expression!(lhs, -1.0, spill[u, s])
            else
                rhs += prob.prior_release[u]
            end
        end
        @constraint(model, lhs == rhs)
        if prob.min_release[r] > 0
            @constraint(model, release[r, t] + spill[r, t] >= prob.min_release[r])
        end
    end

    @objective(model, Min, objective)
    return model
end

register_variant(
    :energy,
    :hydrothermal,
    HydrothermalDispatchProblem,
    "Short-term hydrothermal scheduling: multi-area thermal dispatch with cascaded reservoirs (travel delays, water balance, environmental flows, end-of-horizon water value)";
    tags=[:energy, :network, :staircase],
)
