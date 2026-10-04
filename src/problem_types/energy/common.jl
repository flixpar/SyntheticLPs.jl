using JuMP
using Random
using Distributions
using LinearAlgebra
using SparseArrays

# =============================================================================
# Shared machinery for the energy category.
#
# Two families live here:
#
#   * the multi-area, multi-period ECONOMIC DISPATCH core used by `standard`,
#     `reserves` and `storage` (zones joined by lossy tie-lines, a technology-
#     grounded generation fleet with must-run floors and ramp limits, curtailable
#     renewables, and zonal load profiles); and
#   * the bus-level TRANSMISSION GRID helpers used by `dc_opf` and
#     `security_constrained_dc_opf` (a geometric meshed grid, susceptances and
#     ratings by voltage class, a reduced-Laplacian DC power-flow solve, bridge
#     detection, and load-pocket certificates).
#
# Every sampler takes the constructor-local `rng` first; nothing here touches the
# global RNG and nothing in a `build_model` path samples.
# =============================================================================

# -----------------------------------------------------------------------------
# Technology catalogue
# -----------------------------------------------------------------------------

"""
Generation technology catalogue shared by the energy variants. Ranges are
`(low, high)` and are sampled per unit, so units of one technology are
heterogeneous:

  - `capacity`: nameplate MW (sampled log-uniformly)
  - `min_stable`: must-run floor as a fraction of available capacity (the unit is
    committed for the horizon, as in a security-constrained economic dispatch run
    after unit commitment); peakers and variable renewables have no floor
  - `ramp`: hourly ramp limit as a fraction of capacity (`Inf` = no ramp row)
  - `cost`: marginal cost, \$/MWh (fuel × heat rate + variable O&M)
  - `emission`: tCO2/MWh
  - `weight`: base share of the unit count, before zonal resource multipliers
  - `variable`: availability follows a weather profile (wind, solar, hydro)
  - `spin10`/`nonspin30`: share of hourly ramp deliverable in 10 / 30 minutes
    (`0` = the technology does not offer that reserve product)
"""
const ENERGY_TECHNOLOGIES = (
    nuclear=(
        capacity=(700.0, 1400.0),
        min_stable=(0.80, 0.92),
        ramp=(0.02, 0.05),
        cost=(7.0, 13.0),
        emission=(0.0, 0.0),
        weight=0.04,
        variable=false,
        spin10=0.0,
        nonspin30=0.0,
    ),
    coal=(
        capacity=(200.0, 900.0),
        min_stable=(0.35, 0.50),
        ramp=(0.08, 0.18),
        cost=(22.0, 40.0),
        emission=(0.85, 1.05),
        weight=0.14,
        variable=false,
        spin10=1 / 6,
        nonspin30=1 / 2,
    ),
    ccgt=(
        capacity=(150.0, 600.0),
        min_stable=(0.30, 0.45),
        ramp=(0.30, 0.60),
        cost=(30.0, 55.0),
        emission=(0.33, 0.42),
        weight=0.22,
        variable=false,
        spin10=1 / 6,
        nonspin30=1 / 2,
    ),
    gas_ct=(
        capacity=(30.0, 200.0),
        min_stable=(0.0, 0.0),
        ramp=(Inf, Inf),
        cost=(65.0, 140.0),
        emission=(0.50, 0.65),
        weight=0.15,
        variable=false,
        spin10=0.5,
        nonspin30=1.0,
    ),
    oil=(
        capacity=(15.0, 80.0),
        min_stable=(0.0, 0.0),
        ramp=(Inf, Inf),
        cost=(150.0, 260.0),
        emission=(0.70, 0.80),
        weight=0.03,
        variable=false,
        spin10=0.4,
        nonspin30=1.0,
    ),
    biomass=(
        capacity=(20.0, 90.0),
        min_stable=(0.30, 0.50),
        ramp=(0.10, 0.25),
        cost=(35.0, 60.0),
        emission=(0.05, 0.10),
        weight=0.04,
        variable=false,
        spin10=1 / 6,
        nonspin30=1 / 2,
    ),
    hydro=(
        capacity=(40.0, 400.0),
        min_stable=(0.05, 0.15),
        ramp=(0.50, 0.90),
        cost=(2.0, 8.0),
        emission=(0.0, 0.0),
        weight=0.10,
        variable=true,
        spin10=1 / 3,
        nonspin30=1.0,
    ),
    wind=(
        capacity=(40.0, 300.0),
        min_stable=(0.0, 0.0),
        ramp=(Inf, Inf),
        cost=(0.0, 3.0),
        emission=(0.0, 0.0),
        weight=0.18,
        variable=true,
        spin10=0.0,
        nonspin30=0.0,
    ),
    solar=(
        capacity=(20.0, 250.0),
        min_stable=(0.0, 0.0),
        ramp=(Inf, Inf),
        cost=(0.0, 2.0),
        emission=(0.0, 0.0),
        weight=0.10,
        variable=true,
        spin10=0.0,
        nonspin30=0.0,
    ),
)

const ENERGY_TECH_NAMES = collect(keys(ENERGY_TECHNOLOGIES))

"""Firm-capacity credit of each technology at system peak (planning convention)."""
const ENERGY_FIRM_CREDIT = Dict(
    :nuclear => 0.95,
    :coal => 0.92,
    :ccgt => 0.95,
    :gas_ct => 0.95,
    :oil => 0.9,
    :biomass => 0.9,
    :hydro => 0.6,
    :wind => 0.15,
    :solar => 0.3,
)

"""Natural planning horizons (hours) the dispatch family snaps to."""
const ENERGY_HORIZONS = (4, 6, 8, 12, 16, 24, 36, 48, 72, 96, 120, 144, 168)

_e_unif(rng::AbstractRNG, r) = r[1] == r[2] ? float(r[1]) : r[1] + (r[2] - r[1]) * rand(rng)
_e_logunif(rng::AbstractRNG, r) = exp(_e_unif(rng, (log(r[1]), log(r[2]))))

# -----------------------------------------------------------------------------
# Load and weather profiles
# -----------------------------------------------------------------------------

# Normalized 24-hour load shapes (peak = 1).
const ENERGY_RESIDENTIAL_SHAPE = [
    0.55, 0.50, 0.48, 0.47, 0.49, 0.56, 0.68, 0.78, 0.80, 0.78, 0.76, 0.75,
    0.75, 0.74, 0.75, 0.78, 0.85, 0.95, 1.00, 0.98, 0.93, 0.83, 0.71, 0.61,
]
const ENERGY_COMMERCIAL_SHAPE = [
    0.45, 0.43, 0.42, 0.42, 0.43, 0.50, 0.62, 0.80, 0.92, 0.97, 1.00, 1.00,
    0.99, 1.00, 0.99, 0.96, 0.90, 0.80, 0.70, 0.62, 0.56, 0.52, 0.49, 0.47,
]
const ENERGY_INDUSTRIAL_SHAPE = [
    0.78, 0.77, 0.76, 0.76, 0.77, 0.82, 0.90, 0.96, 0.99, 1.00, 1.00, 0.99,
    0.98, 0.99, 1.00, 0.99, 0.97, 0.94, 0.90, 0.87, 0.85, 0.83, 0.81, 0.79,
]

"""
    _energy_load_profile(rng, T) -> Vector{Float64}

Hourly load shape (peak ≈ 1) for one zone: a residential/commercial/industrial
customer mix, a weekday/weekend effect, an AR(1) daily weather factor, and small
hourly noise.
"""
function _energy_load_profile(rng::AbstractRNG, T::Int)
    mix = rand(rng, Dirichlet([2.0, 1.5, 1.0]))
    base = mix[1] .* ENERGY_RESIDENTIAL_SHAPE .+ mix[2] .* ENERGY_COMMERCIAL_SHAPE .+
        mix[3] .* ENERGY_INDUSTRIAL_SHAPE
    base ./= maximum(base)
    first_day = rand(rng, 0:6)
    weather = 1.0
    profile = zeros(Float64, T)
    for t in 1:T
        h = (t - 1) % 24 + 1
        if h == 1 || t == 1
            weather = clamp(1.0 + 0.6 * (weather - 1.0) + 0.04 * randn(rng), 0.85, 1.15)
        end
        day = (first_day + (t - 1) ÷ 24) % 7
        weekend = day >= 5 ? 0.88 + 0.08 * mix[1] : 1.0
        profile[t] = base[h] * weather * weekend * (1.0 + 0.015 * randn(rng))
    end
    return profile
end

"""
    _energy_wind_profile(rng, T) -> Vector{Float64}

Zone-level wind capacity factor: a mean-reverting AR(1) process (hourly
persistence ≈ 0.9) clamped to [0.02, 0.95].
"""
function _energy_wind_profile(rng::AbstractRNG, T::Int)
    mu = _e_unif(rng, (0.25, 0.45))
    w = clamp(mu + 0.15 * randn(rng), 0.02, 0.95)
    out = zeros(Float64, T)
    for t in 1:T
        w = clamp(mu + 0.9 * (w - mu) + 0.06 * randn(rng), 0.02, 0.95)
        out[t] = w
    end
    return out
end

"""
    _energy_solar_profile(rng, T) -> Vector{Float64}

Zone-level solar capacity factor: a clear-sky bell between sunrise (~06:00) and
sunset (~19:00) scaled by a daily cloudiness factor.
"""
function _energy_solar_profile(rng::AbstractRNG, T::Int)
    out = zeros(Float64, T)
    cloud = 1.0
    peak_cf = _e_unif(rng, (0.65, 0.85))
    for t in 1:T
        h = (t - 1) % 24
        if h == 0 || t == 1
            cloud = rand(rng, Beta(5.0, 1.8))
        end
        s = h >= 6 && h <= 19 ? sin(π * (h - 5.5) / 14.0) : 0.0
        out[t] = max(0.0, peak_cf * cloud * max(s, 0.0)^1.3 * (1.0 + 0.05 * randn(rng)))
    end
    return clamp.(out, 0.0, 1.0)
end

"""Zone-level hydro availability: a seasonal level with mild daily variation."""
function _energy_hydro_profile(rng::AbstractRNG, T::Int)
    level = _e_unif(rng, (0.45, 0.85))
    out = zeros(Float64, T)
    for t in 1:T
        if (t - 1) % 24 == 0
            level = clamp(level * (1.0 + 0.04 * randn(rng)), 0.3, 0.95)
        end
        out[t] = level
    end
    return out
end

# -----------------------------------------------------------------------------
# Zonal network layout (dispatch family)
# -----------------------------------------------------------------------------

"""
    _energy_zone_layout(rng, Z) -> (x, y, from, to)

Place `Z` balancing zones on a 100 × 100 map and join them with tie-lines: a
Euclidean minimum spanning tree (Prim, O(Z²)) plus short meshing ties so that
most zones have two or more interconnections (≈ 1.4 ties per zone, as in real
multi-area systems).
"""
function _energy_zone_layout(rng::AbstractRNG, Z::Int)
    x = 100 .* rand(rng, Z)
    y = 100 .* rand(rng, Z)
    from = Int[]
    to = Int[]
    Z <= 1 && return x, y, from, to
    d(a, b) = hypot(x[a] - x[b], y[a] - y[b])
    in_tree = falses(Z)
    best = fill(Inf, Z)
    parent = zeros(Int, Z)
    in_tree[1] = true
    for v in 2:Z
        best[v] = d(1, v)
        parent[v] = 1
    end
    for _ in 2:Z
        v = 0
        bv = Inf
        for u in 1:Z
            if !in_tree[u] && best[u] < bv
                bv = best[u]
                v = u
            end
        end
        in_tree[v] = true
        push!(from, parent[v])
        push!(to, v)
        for u in 1:Z
            if !in_tree[u] && d(v, u) < best[u]
                best[u] = d(v, u)
                parent[u] = v
            end
        end
    end
    Z <= 2 && return x, y, from, to
    edges = Set{Tuple{Int, Int}}((min(a, b), max(a, b)) for (a, b) in zip(from, to))
    target_ties = round(Int, 1.4 * Z)
    degree = zeros(Int, Z)
    for (a, b) in zip(from, to)
        degree[a] += 1
        degree[b] += 1
    end
    # Leaves first: connect each to its nearest non-adjacent zone.
    for v in shuffle(rng, collect(1:Z))
        length(from) >= target_ties && break
        degree[v] == 1 || continue
        rand(rng) < 0.8 || continue
        cand = sort([u for u in 1:Z if u != v && !((min(u, v), max(u, v)) in edges)]; by=u -> d(u, v))
        isempty(cand) && continue
        u = cand[1]
        push!(edges, (min(u, v), max(u, v)))
        push!(from, v)
        push!(to, u)
        degree[u] += 1
        degree[v] += 1
    end
    attempts = 0
    while length(from) < target_ties && attempts < 50 * Z
        attempts += 1
        v = rand(rng, 1:Z)
        # A random short tie: one of the three nearest zones.
        cand = partialsort!([u for u in 1:Z if u != v], 1:min(3, Z - 1); by=u -> d(u, v))
        u = cand[rand(rng, eachindex(cand))]
        key = (min(u, v), max(u, v))
        key in edges && continue
        push!(edges, key)
        push!(from, v)
        push!(to, u)
    end
    return x, y, from, to
end

# -----------------------------------------------------------------------------
# Economic-dispatch core data
# -----------------------------------------------------------------------------

"""
    EnergyDispatchCore

Data shared by the multi-area economic-dispatch variants (`standard`,
`reserves`, `storage`). Index sets: zones `z = 1:n_zones`, tie-lines
`l = 1:length(tie_from)`, units `g = 1:length(unit_tech)`, periods
`t = 1:n_periods` (hours).

  - `zone_x`, `zone_y`: zone centroids on a 100 × 100 map
  - `tie_from`, `tie_to`, `tie_capacity` (MW, per direction), `tie_loss`
    (fraction lost in transit), `tie_cost` (\$/MWh wheeling charge)
  - `unit_tech`, `unit_zone`, `capacity` (MW), `min_stable` (fraction of
    available capacity that must run), `cost` (\$/MWh), `emission_rate`
    (tCO2/MWh), `ramp_up`/`ramp_down` (MW per period, `Inf` = unconstrained)
  - `availability[g, t]`: available fraction of nameplate (weather for wind,
    solar and hydro; forced outages for peakers; 1 otherwise)
  - `initial_output[g]`: dispatch in the hour before the horizon; the first
    period's ramp window around it is folded into the variable bounds
  - `demand[z, t]`: zonal load, MW
"""
struct EnergyDispatchCore
    n_zones::Int
    n_periods::Int
    zone_x::Vector{Float64}
    zone_y::Vector{Float64}
    tie_from::Vector{Int}
    tie_to::Vector{Int}
    tie_capacity::Vector{Float64}
    tie_loss::Vector{Float64}
    tie_cost::Vector{Float64}
    unit_tech::Vector{Symbol}
    unit_zone::Vector{Int}
    capacity::Vector{Float64}
    min_stable::Vector{Float64}
    cost::Vector{Float64}
    emission_rate::Vector{Float64}
    ramp_up::Vector{Float64}
    ramp_down::Vector{Float64}
    availability::Matrix{Float64}
    initial_output::Vector{Float64}
    demand::Matrix{Float64}
end

_ed_n_units(c::EnergyDispatchCore) = length(c.unit_tech)
_ed_n_ties(c::EnergyDispatchCore) = length(c.tie_from)

"""Whether unit `g` gets ramp rows (a finite ramp limit that can bind)."""
_ed_ramped(c::EnergyDispatchCore, g::Int) =
    isfinite(c.ramp_up[g]) && c.ramp_up[g] < c.capacity[g] * (1 - c.min_stable[g])

"""Lower bound of `x[g, t]` (must-run floor, first-period ramp window folded in)."""
function _ed_lower(c::EnergyDispatchCore, g::Int, t::Int)
    lb = c.min_stable[g] * c.capacity[g] * c.availability[g, t]
    if t == 1 && _ed_ramped(c, g)
        lb = max(lb, c.initial_output[g] - c.ramp_down[g])
    end
    return lb
end

"""Upper bound of `x[g, t]` (available capacity, first-period ramp window folded in)."""
function _ed_upper(c::EnergyDispatchCore, g::Int, t::Int)
    ub = c.capacity[g] * c.availability[g, t]
    if t == 1 && _ed_ramped(c, g)
        ub = min(ub, c.initial_output[g] + c.ramp_up[g])
    end
    return max(ub, _ed_lower(c, g, t))
end

"""System generation capacity available in period `t` (sum of upper bounds)."""
_ed_system_upper(c::EnergyDispatchCore, t::Int) = sum(_ed_upper(c, g, t) for g in 1:_ed_n_units(c))

"""Units located in each zone."""
function _ed_units_by_zone(c::EnergyDispatchCore)
    out = [Int[] for _ in 1:c.n_zones]
    for g in 1:_ed_n_units(c)
        push!(out[c.unit_zone[g]], g)
    end
    return out
end

"""
    _ed_horizon(target) -> Int

Snap the horizon to a natural planning length growing like `0.75·√target`
(24 h at 1k variables, 72 h at 10k, one week from ~50k up).
"""
function _ed_horizon(target::Int)
    raw = 0.75 * sqrt(max(target, 1))
    best = ENERGY_HORIZONS[1]
    for h in ENERGY_HORIZONS
        abs(h - raw) < abs(best - raw) && (best = h)
    end
    return best
end

"""
    _ed_sample_fleet(rng, Z, budget, unit_cols; min_per_zone=1)

Sample units into `Z` zones until the per-period column budget is spent.
`unit_cols(tech)` is how many columns (per period) one unit of `tech` adds in the
calling variant (1 for plain dispatch; more when it carries reserve products).
Zones get lognormal size weights and resource multipliers (wind-, solar-, hydro-
or coal-rich regions), so fleets are regionally distinct. Every zone receives at
least one dispatchable unit. Technologies in `exclude` are never sampled.
Returns `(tech, zone)` vectors.
"""
function _ed_sample_fleet(rng::AbstractRNG, Z::Int, budget::Int, unit_cols; exclude=())
    zone_weight = exp.(0.5 .* randn(rng, Z))
    zone_weight ./= sum(zone_weight)
    base = [k in exclude ? 0.0 : ENERGY_TECHNOLOGIES[k].weight for k in ENERGY_TECH_NAMES]
    zone_mix = Vector{Vector{Float64}}(undef, Z)
    for z in 1:Z
        m = base .* exp.(0.6 .* randn(rng, length(base)))
        zone_mix[z] = m ./ sum(m)
    end
    techs = Symbol[]
    zones = Int[]
    remaining = budget
    dispatchable = [t for t in (:ccgt, :coal, :gas_ct, :hydro) if !(t in exclude)]
    # Seed: one dispatchable unit per zone.
    for z in 1:Z
        tech = dispatchable[rand(rng, 1:length(dispatchable))]
        c = unit_cols(tech)
        if c > remaining
            tech = :gas_ct
            c = unit_cols(tech)
        end
        c > remaining && break
        push!(techs, tech)
        push!(zones, z)
        remaining -= c
    end
    cheapest = minimum(unit_cols(k) for k in ENERGY_TECH_NAMES if !(k in exclude))
    zone_cdf = cumsum(zone_weight)
    while remaining >= cheapest
        z = min(Z, searchsortedfirst(zone_cdf, rand(rng)))
        mix = zone_mix[z]
        k = min(length(mix), searchsortedfirst(cumsum(mix), rand(rng)))
        tech = ENERGY_TECH_NAMES[k]
        c = unit_cols(tech)
        if c > remaining
            # Fill the tail exactly with the cheapest technology that fits.
            fits = [t for t in ENERGY_TECH_NAMES if unit_cols(t) <= remaining && !(t in exclude)]
            tech = fits[rand(rng, 1:length(fits))]
            c = unit_cols(tech)
        end
        push!(techs, tech)
        push!(zones, z)
        remaining -= c
    end
    return techs, zones
end

"""
    _ed_sample_core(rng, layout, T, techs, zones; margin) -> NamedTuple

Sample everything for the dispatch core except the final demand/initial state:
tie parameters for the zone `layout` (from `_energy_zone_layout`), unit
parameters, availability profiles, and the NATURAL zonal demand (sized so system
firm capacity / coincident peak ≈ `margin`).
"""
function _ed_sample_core(
    rng::AbstractRNG, layout, T::Int, techs::Vector{Symbol}, zones::Vector{Int}; margin::Float64
)
    zx, zy, tie_from, tie_to = layout
    Z = length(zx)
    G = length(techs)
    capacity = zeros(G)
    min_stable = zeros(G)
    cost = zeros(G)
    emission = zeros(G)
    ramp_up = fill(Inf, G)
    ramp_down = fill(Inf, G)
    fuel_index = exp.(0.08 .* randn(rng, Z))   # regional fuel-price index
    for g in 1:G
        spec = ENERGY_TECHNOLOGIES[techs[g]]
        capacity[g] = _e_logunif(rng, spec.capacity)
        min_stable[g] = _e_unif(rng, spec.min_stable)
        cost[g] = _e_unif(rng, spec.cost) * (spec.emission[2] > 0 ? fuel_index[zones[g]] : 1.0)
        emission[g] = _e_unif(rng, spec.emission)
        if isfinite(spec.ramp[1])
            frac = _e_unif(rng, spec.ramp)
            ramp_up[g] = frac * capacity[g]
            ramp_down[g] = frac * capacity[g] * _e_unif(rng, (0.85, 1.15))
        end
    end

    # Weather profiles are zone-level (spatially correlated within a zone) with
    # unit-level noise on top.
    wind = [_energy_wind_profile(rng, T) for _ in 1:Z]
    solar = [_energy_solar_profile(rng, T) for _ in 1:Z]
    hydro = [_energy_hydro_profile(rng, T) for _ in 1:Z]
    availability = ones(Float64, G, T)
    for g in 1:G
        z = zones[g]
        tech = techs[g]
        if tech == :wind
            s = _e_unif(rng, (0.85, 1.1))
            for t in 1:T
                availability[g, t] = clamp(wind[z][t] * s * (1 + 0.05 * randn(rng)), 0.0, 1.0)
            end
        elseif tech == :solar
            s = _e_unif(rng, (0.9, 1.05))
            for t in 1:T
                availability[g, t] = clamp(solar[z][t] * s, 0.0, 1.0)
            end
        elseif tech == :hydro
            s = _e_unif(rng, (0.85, 1.1))
            for t in 1:T
                availability[g, t] = clamp(hydro[z][t] * s, 0.1, 1.0)
            end
        elseif tech in (:gas_ct, :oil)
            # Peakers carry occasional forced outages (they have no ramp rows, so
            # an outage never makes a ramp window empty).
            if rand(rng) < 0.08 && T >= 4
                len = rand(rng, 1:max(1, T ÷ 6))
                s0 = rand(rng, 1:(T - len + 1))
                availability[g, s0:(s0 + len - 1)] .= 0.0
            end
        end
    end

    # Natural zonal demand: zone peaks proportional to firm capacity with an
    # import/export imbalance factor, rescaled to the system planning margin.
    firm = zeros(Z)
    for g in 1:G
        firm[zones[g]] += capacity[g] * ENERGY_FIRM_CREDIT[techs[g]]
    end
    profiles = [_energy_load_profile(rng, T) for _ in 1:Z]
    zone_peak = [firm[z] * exp(0.3 * randn(rng)) + 1.0 for z in 1:Z]
    system_profile = zeros(T)
    for z in 1:Z, t in 1:T
        system_profile[t] += zone_peak[z] * profiles[z][t]
    end
    scale = sum(firm) / margin / maximum(system_profile)
    zone_peak .*= scale
    natural_demand = zeros(Z, T)
    for z in 1:Z, t in 1:T
        natural_demand[z, t] = zone_peak[z] * profiles[z][t]
    end

    # Commitment: the dispatch runs after unit commitment, so only part of the
    # thermal fleet is online with a must-run floor; the rest is offline-capable
    # (floor 0). A sane commitment never forces more than 75 % of a zone's
    # lightest-hour load, so units are decommitted until that holds.
    commit_prob = Dict(:nuclear => 1.0, :coal => 0.6, :ccgt => 0.5, :biomass => 0.7, :hydro => 0.4)
    for g in 1:G
        rand(rng) < get(commit_prob, techs[g], 0.0) || (min_stable[g] = 0.0)
    end
    for z in 1:Z
        floor_z(t) = sum((min_stable[g] * capacity[g] * availability[g, t] for g in 1:G if zones[g] == z); init=0.0)
        limit = 0.75 * minimum(natural_demand[z, t] for t in 1:T)
        candidates = shuffle(rng, [g for g in 1:G if zones[g] == z && min_stable[g] > 0])
        sort!(candidates; by=g -> techs[g] == :nuclear)   # keep nuclear committed longest
        for g in candidates
            maximum(floor_z(t) for t in 1:T) <= limit && break
            min_stable[g] = 0.0
        end
    end

    # Tie-lines: transfer capability a fraction of the smaller endpoint's peak,
    # distance-dependent losses, small wheeling charges.
    L = length(tie_from)
    tie_capacity = zeros(L)
    tie_loss = zeros(L)
    tie_cost = zeros(L)
    for l in 1:L
        a, b = tie_from[l], tie_to[l]
        tie_capacity[l] = _e_unif(rng, (0.15, 0.45)) * min(zone_peak[a], zone_peak[b]) + 10.0
        dist = hypot(zx[a] - zx[b], zy[a] - zy[b])
        tie_loss[l] = clamp(0.005 + 0.0003 * dist + 0.003 * rand(rng), 0.005, 0.06)
        tie_cost[l] = _e_unif(rng, (0.2, 1.5))
    end
    # Transmission is planned around import needs: a zone whose peak exceeds its
    # local available capacity gets interconnections able to cover the deficit
    # with a 25 % margin, and so does every pair of neighbouring zones (their
    # coincident joint deficit through the ties leaving the pair) — otherwise
    # two adjacent importers form a tiny load pocket that is trivially
    # infeasible.
    local_cap = zeros(Z, T)
    for g in 1:G, t in 1:T
        local_cap[zones[g], t] += capacity[g] * availability[g, t]
    end
    short = natural_demand .- local_cap
    function cover!(set, deficit)
        deficit > 0 || return
        out = [l for l in 1:L if (tie_from[l] in set) != (tie_to[l] in set)]
        isempty(out) && return
        capability = sum((1 - tie_loss[l]) * tie_capacity[l] for l in out)
        capability < 1.25 * deficit && (tie_capacity[out] .*= 1.25 * deficit / capability)
    end
    for z in 1:Z
        cover!((z,), maximum(short[z, :]))
    end
    for l in 1:L
        a, b = tie_from[l], tie_to[l]
        cover!((a, b), maximum(short[a, t] + short[b, t] for t in 1:T))
    end

    return (;
        zx, zy, tie_from, tie_to, tie_capacity, tie_loss, tie_cost, capacity, min_stable, cost,
        emission, ramp_up, ramp_down, availability, natural_demand,
    )
end

"""
    _ed_track_zone(lb, ub, lo, hi, target) -> Vector{Float64}

Dispatch a zone's controllable units to hit `target` MW: every unit moves to the
same fraction `β` of its `[lb, ub]` range, clamped to its ramp window `[lo, hi]`
(monotone in `β`, so bisection finds it). Out-of-range targets return the window
end.
"""
function _ed_track_zone(lb, ub, lo, hi, target::Float64)
    n = length(lb)
    n == 0 && return Float64[]
    f(β) = sum(clamp(lb[i] + β * (ub[i] - lb[i]), lo[i], hi[i]) for i in 1:n)
    if target <= f(0.0)
        return [clamp(lb[i], lo[i], hi[i]) for i in 1:n]
    elseif target >= f(1.0)
        return [clamp(ub[i], lo[i], hi[i]) for i in 1:n]
    end
    a, b = 0.0, 1.0
    for _ in 1:60
        m = (a + b) / 2
        f(m) < target ? (a = m) : (b = m)
    end
    β = (a + b) / 2
    return [clamp(lb[i] + β * (ub[i] - lb[i]), lo[i], hi[i]) for i in 1:n]
end

"""
    _ed_tracking_dispatch(rng, nt, techs, zones, T, demand, flow_fwd, flow_bwd; extra=nothing)

Ramp-feasible dispatch that tracks each zone's residual load (load − renewables −
net imports − `extra`, e.g. planted storage net discharge). Renewables run at `curtail` × available output (a per-unit factor
in [0.9, 1]); controllable units follow a common utilization fraction clamped to
their ramp windows. Period 1 is dispatched without a ramp window and becomes the
initial output, so the first-period window is always satisfied. Returns
`(x, initial_output)`.
"""
function _ed_tracking_dispatch(
    rng::AbstractRNG, nt, techs, zones, T, demand, flow_fwd, flow_bwd; extra=nothing
)
    G = length(techs)
    Z = size(demand, 1)
    by_zone = [Int[] for _ in 1:Z]
    for g in 1:G
        push!(by_zone[zones[g]], g)
    end
    net_import = zeros(Z, T)
    for l in eachindex(nt.tie_from), t in 1:T
        a, b = nt.tie_from[l], nt.tie_to[l]
        keep = 1 - nt.tie_loss[l]
        net_import[a, t] += keep * flow_bwd[l, t] - flow_fwd[l, t]
        net_import[b, t] += keep * flow_fwd[l, t] - flow_bwd[l, t]
    end
    curtail = [_e_unif(rng, (0.9, 1.0)) for _ in 1:G]
    x = zeros(G, T)
    for z in 1:Z
        units = by_zone[z]
        ren = [g for g in units if techs[g] in (:wind, :solar)]
        ctl = [g for g in units if !(techs[g] in (:wind, :solar))]
        for t in 1:T
            rsum = 0.0
            for g in ren
                x[g, t] = curtail[g] * nt.capacity[g] * nt.availability[g, t]
                rsum += x[g, t]
            end
            lb = [nt.min_stable[g] * nt.capacity[g] * nt.availability[g, t] for g in ctl]
            ub = [nt.capacity[g] * nt.availability[g, t] for g in ctl]
            if t == 1
                lo, hi = lb, ub
            else
                lo = [
                    isfinite(nt.ramp_down[g]) ? max(lb[i], x[g, t - 1] - nt.ramp_down[g]) : lb[i]
                    for (i, g) in enumerate(ctl)
                ]
                hi = [
                    isfinite(nt.ramp_up[g]) ? min(ub[i], x[g, t - 1] + nt.ramp_up[g]) : ub[i]
                    for (i, g) in enumerate(ctl)
                ]
                # Availability drops (hydro) can push the floor above the cap of
                # the window; keep the window non-empty by honouring the ramp.
                for i in eachindex(lo)
                    if lo[i] > hi[i]
                        hi[i] = lo[i]
                    end
                end
            end
            residual = demand[z, t] - rsum - net_import[z, t] - (extra === nothing ? 0.0 : extra[z, t])
            vals = _ed_track_zone(lb, ub, lo, hi, residual)
            for (i, g) in enumerate(ctl)
                x[g, t] = vals[i]
            end
        end
    end
    return x, x[:, 1], net_import
end

"""
    _ed_planted_flows(rng, nt, Z, T, demand) -> (flow_fwd, flow_bwd)

Planted tie-line schedule for the witness: each tie carries a moderate exchange
(10–45 % of its capability, following the system load shape) from the zone with
the cheaper average fleet toward the dearer one.
"""
function _ed_planted_flows(rng::AbstractRNG, nt, techs, zones, Z, T, demand)
    L = length(nt.tie_from)
    zone_cost = zeros(Z)
    zone_cap = zeros(Z)
    for g in eachindex(techs)
        zone_cost[zones[g]] += nt.cost[g] * nt.capacity[g]
        zone_cap[zones[g]] += nt.capacity[g]
    end
    zone_cost ./= max.(zone_cap, 1.0)
    sysload = vec(sum(demand; dims=1))
    shape = sysload ./ maximum(sysload)
    fwd = zeros(L, T)
    bwd = zeros(L, T)
    for l in 1:L
        a, b = nt.tie_from[l], nt.tie_to[l]
        rho = _e_unif(rng, (0.1, 0.45))
        forward = zone_cost[a] < zone_cost[b] ? rand(rng) < 0.85 : rand(rng) < 0.15
        for t in 1:T
            v = rho * nt.tie_capacity[l] * (0.7 + 0.3 * shape[t])
            forward ? (fwd[l, t] = v) : (bwd[l, t] = v)
        end
    end
    return fwd, bwd
end

"""
    EnergyDispatchWitness

Planted feasible point of the dispatch core: unit output `output[g, t]` and the
directional tie flows `flow_fwd[l, t]` (from → to) and `flow_bwd[l, t]`.
"""
struct EnergyDispatchWitness
    output::Matrix{Float64}
    flow_fwd::Matrix{Float64}
    flow_bwd::Matrix{Float64}
end

"""
    _ed_plant(rng, nt, techs, zones, Z, T; extra=nothing) -> (demand, initial_output, witness)

Construct the planted witness and the demand it serves. Tie flows are planted
first; each zone's controllable units then track its natural residual load
within their ramp windows, and the zonal demand is DEFINED as the resulting
supply plus net imports (plus `extra[z, t]`, a planted storage net discharge), so
the zonal balance holds exactly. The planted flows
are halved until every zone keeps at least 30 % of its natural load.
"""
function _ed_plant(rng::AbstractRNG, nt, techs, zones, Z, T; extra=nothing)
    fwd, bwd = _ed_planted_flows(rng, nt, techs, zones, Z, T, nt.natural_demand)
    x = zeros(length(techs), T)
    x0 = zeros(length(techs))
    demand = zeros(Z, T)
    for attempt in 1:8
        sub = MersenneTwister(rand(rng, UInt64))
        x, x0, net_import =
            _ed_tracking_dispatch(sub, nt, techs, zones, T, nt.natural_demand, fwd, bwd; extra=extra)
        demand .= 0.0
        for g in eachindex(techs), t in 1:T
            demand[zones[g], t] += x[g, t]
        end
        demand .+= net_import
        extra === nothing || (demand .+= extra)
        ok = all(demand[z, t] >= 0.3 * nt.natural_demand[z, t] for z in 1:Z, t in 1:T)
        (ok || attempt == 8) && break
        fwd .*= attempt < 7 ? 0.5 : 0.0
        bwd .*= attempt < 7 ? 0.5 : 0.0
    end
    return demand, x0, EnergyDispatchWitness(x, fwd, bwd)
end

"""
    _ed_assemble(nt, techs, zones, Z, T, demand, x0) -> EnergyDispatchCore
"""
_ed_assemble(nt, techs, zones, Z, T, demand, x0) = EnergyDispatchCore(
    Z,
    T,
    nt.zx,
    nt.zy,
    nt.tie_from,
    nt.tie_to,
    nt.tie_capacity,
    nt.tie_loss,
    nt.tie_cost,
    techs,
    zones,
    nt.capacity,
    nt.min_stable,
    nt.cost,
    nt.emission,
    nt.ramp_up,
    nt.ramp_down,
    nt.availability,
    x0,
    demand,
)

"""
    _ed_natural(rng, nt, techs, zones, Z, T) -> (demand, initial_output)

Natural (unplanted) instance: the natural zonal load and an initial state given
by the zero-exchange tracking dispatch of the first period.
"""
function _ed_natural(rng::AbstractRNG, nt, techs, zones, Z, T)
    L = length(nt.tie_from)
    x, _, _ = _ed_tracking_dispatch(rng, nt, techs, zones, min(T, 1), nt.natural_demand[:, 1:1], zeros(L, 1), zeros(L, 1))
    return copy(nt.natural_demand), x[:, 1]
end

"""
    _ed_dimensions(rng, target) -> (T, Z, layout, per_period)

Horizon, zone count, zone layout and the per-period column budget shared by the
dispatch variants: `T` snaps `0.75·√target` to a natural horizon, the
per-period budget is `round(target / T)`, and `Z ≈ budget / 16` (at least 2).
"""
function _ed_dimensions(rng::AbstractRNG, target::Int)
    target = max(target, 16)
    T = _ed_horizon(target)
    per_period = max(6, round(Int, target / T))
    Z = per_period < 32 ? 2 : clamp(round(Int, per_period / 16), 2, 100_000)
    layout = _energy_zone_layout(rng, Z)
    return T, Z, layout, per_period
end

"""Emissions of a dispatch matrix `x` (units × periods)."""
_ed_emissions(c::EnergyDispatchCore, x::AbstractMatrix) =
    sum(c.emission_rate[g] * x[g, t] for g in 1:_ed_n_units(c), t in 1:c.n_periods)

"""
    _ed_plant_system_shortage!(rng, c; margin) -> EnergyAggregateCertificate

Fallback contradiction: the whole system (all zones) as one pocket in its peak
hour — total demand above total available capacity, spread so that every zone's
own balance row and every proper subset of zones stays satisfiable through the
(strengthened) ties.
"""
function _ed_plant_system_shortage!(rng::AbstractRNG, c::EnergyDispatchCore; margin::Float64)
    S = collect(1:c.n_zones)
    t = argmax([_ed_system_demand(c, t) / _ed_system_upper(c, t) for t in 1:c.n_periods])
    need = (1 + margin) * _ed_pocket_supply(c, S, t)
    _ed_spread_demand!(c, S, t, need) || error("energy: could not plant a system shortage")
    return EnergyAggregateCertificate(:import_pocket, S, [t], _ed_pocket_supply(c, S, t), _ed_system_demand(c, t))
end

# -----------------------------------------------------------------------------
# Dispatch-core model pieces
# -----------------------------------------------------------------------------

"""
    _ed_core_variables!(model, core) -> (x, flow_fwd, flow_bwd, balance, objective)

Add the dispatch core's columns — unit output `x[g, t]` (bounded by the must-run
floor and available capacity, with the first-period ramp window folded in) and
directional tie flows — and return the zonal balance expressions
(supply + delivered imports − exports, still without the demand) and the base
objective (energy cost + wheeling). The caller adds variant terms, then calls
`_ed_core_rows!`.
"""
function _ed_core_variables!(model::Model, c::EnergyDispatchCore)
    G = _ed_n_units(c)
    T = c.n_periods
    L = _ed_n_ties(c)
    LB = [_ed_lower(c, g, t) for g in 1:G, t in 1:T]
    UB = [_ed_upper(c, g, t) for g in 1:G, t in 1:T]
    @variable(model, LB[g, t] <= x[g=1:G, t=1:T] <= UB[g, t])
    @variable(model, 0 <= flow_fwd[l=1:L, t=1:T] <= c.tie_capacity[l])
    @variable(model, 0 <= flow_bwd[l=1:L, t=1:T] <= c.tie_capacity[l])

    balance = [AffExpr(0.0) for _ in 1:c.n_zones, _ in 1:T]
    objective = AffExpr(0.0)
    for g in 1:G, t in 1:T
        add_to_expression!(balance[c.unit_zone[g], t], 1.0, x[g, t])
        add_to_expression!(objective, c.cost[g], x[g, t])
    end
    for l in 1:L, t in 1:T
        a, b = c.tie_from[l], c.tie_to[l]
        keep = 1.0 - c.tie_loss[l]
        add_to_expression!(balance[a, t], -1.0, flow_fwd[l, t])
        add_to_expression!(balance[b, t], keep, flow_fwd[l, t])
        add_to_expression!(balance[b, t], -1.0, flow_bwd[l, t])
        add_to_expression!(balance[a, t], keep, flow_bwd[l, t])
        add_to_expression!(objective, c.tie_cost[l], flow_fwd[l, t])
        add_to_expression!(objective, c.tie_cost[l], flow_bwd[l, t])
    end
    return x, flow_fwd, flow_bwd, balance, objective
end

"""
    _ed_core_rows!(model, core, x, balance; spin=nothing)

Add the zonal power-balance equalities and, for every ramp-limited unit, one
ranged ramp row `−ramp_down[g] ≤ x[g,t] − x[g,t−1] ≤ ramp_up[g]`. When `spin`
is supplied (reserves variant), spinning reserve must be deployable within the
up-ramp, so the pair becomes two rows: `x[g,t] + spin[g,t] − x[g,t−1] ≤
ramp_up[g]` and `x[g,t−1] − x[g,t] ≤ ramp_down[g]` (`spin[g]` is `nothing` for
units without spinning reserve).
"""
function _ed_core_rows!(model::Model, c::EnergyDispatchCore, x, balance; spin=nothing)
    T = c.n_periods
    @constraint(model, zonal_balance[z=1:c.n_zones, t=1:T], balance[z, t] == c.demand[z, t])
    for g in 1:_ed_n_units(c)
        _ed_ramped(c, g) || continue
        s = spin === nothing ? nothing : spin[g]
        for t in 2:T
            if s === nothing
                # One ranged row (presolve would merge the parallel pair anyway).
                @constraint(model, -c.ramp_down[g] <= x[g, t] - x[g, t - 1] <= c.ramp_up[g])
            else
                @constraint(model, x[g, t] + s[t] - x[g, t - 1] <= c.ramp_up[g])
                @constraint(model, x[g, t - 1] - x[g, t] <= c.ramp_down[g])
            end
        end
    end
    return nothing
end

"""Total system load in period `t`."""
_ed_system_demand(c::EnergyDispatchCore, t::Int) = sum(@view c.demand[:, t])

# -----------------------------------------------------------------------------
# Witness check (solver-free)
# -----------------------------------------------------------------------------

"""
    _ed_witness_violation(core, w; extra_supply=nothing, spin=nothing) -> Float64

Largest absolute violation of the dispatch core's rows and bounds by witness `w`
(0 means feasible). `extra_supply[z, t]` adds storage net discharge to the zonal
balance; `spin[g, t]` tightens the up-ramp rows as in the reserves variant.
"""
function _ed_witness_violation(c::EnergyDispatchCore, w::EnergyDispatchWitness; extra_supply=nothing, spin=nothing)
    G = _ed_n_units(c)
    T = c.n_periods
    viol = 0.0
    for g in 1:G, t in 1:T
        v = w.output[g, t]
        viol = max(viol, _ed_lower(c, g, t) - v, v - _ed_upper(c, g, t))
    end
    for l in 1:_ed_n_ties(c), t in 1:T
        viol = max(viol, -w.flow_fwd[l, t], -w.flow_bwd[l, t])
        viol = max(viol, w.flow_fwd[l, t] - c.tie_capacity[l], w.flow_bwd[l, t] - c.tie_capacity[l])
    end
    bal = zeros(c.n_zones, T)
    for g in 1:G, t in 1:T
        bal[c.unit_zone[g], t] += w.output[g, t]
    end
    for l in 1:_ed_n_ties(c), t in 1:T
        a, b = c.tie_from[l], c.tie_to[l]
        keep = 1 - c.tie_loss[l]
        bal[a, t] += keep * w.flow_bwd[l, t] - w.flow_fwd[l, t]
        bal[b, t] += keep * w.flow_fwd[l, t] - w.flow_bwd[l, t]
    end
    extra_supply === nothing || (bal .+= extra_supply)
    for z in 1:c.n_zones, t in 1:T
        viol = max(viol, abs(bal[z, t] - c.demand[z, t]) / max(1.0, c.demand[z, t]))
    end
    for g in 1:G
        _ed_ramped(c, g) || continue
        for t in 2:T
            up = w.output[g, t] - w.output[g, t - 1] + (spin === nothing ? 0.0 : spin[g, t])
            viol = max(viol, up - c.ramp_up[g], w.output[g, t - 1] - w.output[g, t] - c.ramp_down[g])
        end
    end
    return viol
end

# -----------------------------------------------------------------------------
# Aggregate infeasibility certificates (dispatch family)
# -----------------------------------------------------------------------------

"""
    EnergyAggregateCertificate

LP-row infeasibility certificate for the dispatch family. Each `kind` names a
nonnegative combination of model rows whose implied bound `supply_bound` falls
strictly below `requirement`:

  - `:import_pocket` — sum the zonal balance rows of the connected zone set
    `zones` in period `periods[1]`. Supply inside the pocket is at most its units'
    upper bounds plus the delivered capability of the tie-lines crossing into it
    (`Σ (1 − loss)·capacity`); exports and internal losses only add to the need.
    So `supply_bound = Σ_{g ∈ pocket} ub[g,t] + Σ_{cut ties} (1 − loss)·cap <
    requirement = Σ_{z ∈ pocket} demand[z,t]`.
  - `:emissions_budget` — every period's system balance gives `Σ_g x[g,t] ≥
    D_t` (losses are nonnegative), so emitting units must cover
    `D_t − Σ_{clean} ub − Σ_{emitting} lb`; at the lowest emission rate that is at
    least the bound recorded in `requirement` (summed over `periods`), which
    exceeds the budget `supply_bound` (the row's right-hand side).
  - `:reserve_scarcity` — system balance + headroom rows `x + spin + nonspin ≤
    ub` + the operating-reserve row give `D_t + R_t ≤ Σ_g ub[g,t]`;
    `supply_bound = Σ ub`, `requirement = D_t + R_t`.
  - `:energy_limited_peak` — over the consecutive window `periods`, system
    balance and the state-of-charge recursion give `Σ_W D_t ≤ Σ_W Σ_g ub[g,t] +
    Σ_s η_dis[s]·(soc_max[s] − soc_min[s])`; `supply_bound` is that right side.
"""
struct EnergyAggregateCertificate
    kind::Symbol
    zones::Vector{Int}
    periods::Vector{Int}
    supply_bound::Float64
    requirement::Float64
end

"""Ties with exactly one endpoint in the zone set `S` (the pocket's cut)."""
function _ed_cut_ties(c, S)
    inS = falses(c.n_zones)
    inS[S] .= true
    return [l for l in 1:length(c.tie_from) if inS[c.tie_from[l]] != inS[c.tie_to[l]]]
end

"""Recompute an `:import_pocket` certificate's supply bound from the core."""
function _ed_pocket_supply(c::EnergyDispatchCore, S::Vector{Int}, t::Int)
    inS = falses(c.n_zones)
    inS[S] .= true
    local_ub = sum((_ed_upper(c, g, t) for g in 1:_ed_n_units(c) if inS[c.unit_zone[g]]); init=0.0)
    imports = sum(((1 - c.tie_loss[l]) * c.tie_capacity[l] for l in _ed_cut_ties(c, S)); init=0.0)
    return local_ub + imports
end

"""
Single-row capability of zone `z` in period `t`: the largest value its balance
row can reach on its own (all local units at their upper bounds, every incident
tie importing at capability). Presolve sees exactly this, so planted
contradictions keep each zone's demand strictly below it.
"""
function _ed_zone_row_max(c::EnergyDispatchCore, z::Int, t::Int; extra::Float64=0.0)
    s = extra
    for g in 1:_ed_n_units(c)
        c.unit_zone[g] == z && (s += _ed_upper(c, g, t))
    end
    for l in 1:_ed_n_ties(c)
        if c.tie_from[l] == z || c.tie_to[l] == z
            s += (1 - c.tie_loss[l]) * c.tie_capacity[l]
        end
    end
    return s
end

"""
    _ed_spread_demand!(c, zones, t, total; extra=nothing) -> Bool

Set the period-`t` demand of the connected zone set `zones` to `total`, in
proportion to each zone's own supply weight — local available capacity, plus
`extra[z]` (e.g. local storage power), plus the delivered capability of its ties
leaving the set. If `total` exceeds the set's total weight (a planted pocket),
every zone carries the same relative deficit, and the ties inside the set are
raised to at least 1.2 × the set's total deficit: any proper subset can then
import its own share through one internal tie, so the only contradiction is the
whole set's (presolve's bound propagation finds small pockets, not large ones).
Always succeeds for a connected set.
"""
function _ed_spread_demand!(c::EnergyDispatchCore, zones::Vector{Int}, t::Int, total::Float64; extra=nothing)
    inS = falses(c.n_zones)
    inS[zones] .= true
    w = zeros(c.n_zones)
    for g in 1:_ed_n_units(c)
        inS[c.unit_zone[g]] && (w[c.unit_zone[g]] += _ed_upper(c, g, t))
    end
    extra === nothing || (w[zones] .+= extra[zones])
    for l in 1:_ed_n_ties(c)
        a, b = c.tie_from[l], c.tie_to[l]
        if inS[a] != inS[b]
            inS[a] && (w[a] += (1 - c.tie_loss[l]) * c.tie_capacity[l])
            inS[b] && (w[b] += (1 - c.tie_loss[l]) * c.tie_capacity[l])
        end
    end
    W = sum(w[zones])
    W > 0 || return false
    for z in zones
        c.demand[z, t] = total * w[z] / W
    end
    deficit = total - W
    if deficit > 0
        for l in 1:_ed_n_ties(c)
            if inS[c.tie_from[l]] && inS[c.tie_to[l]]
                c.tie_capacity[l] = max(c.tie_capacity[l], 1.2 * deficit)
            end
        end
    end
    return true
end

"""
    _ed_raise_system_demand!(c, t, total)

Raise period-`t` system demand to `total` (at most the system's available
capacity) by giving every zone the same fraction of its local headroom, so each
zone stays self-sufficient and no subset of zones is short on its own.
"""
function _ed_raise_system_demand!(c::EnergyDispatchCore, t::Int, total::Float64)
    ub = zeros(c.n_zones)
    for g in 1:_ed_n_units(c)
        ub[c.unit_zone[g]] += _ed_upper(c, g, t)
    end
    head = [max(0.0, ub[z] - c.demand[z, t]) for z in 1:c.n_zones]
    rise = total - _ed_system_demand(c, t)
    rise <= 0 && return true
    sum(head) >= rise || return false
    α = rise / sum(head)
    for z in 1:c.n_zones
        c.demand[z, t] += α * head[z]
    end
    return true
end

"""Connected zone set grown by BFS from `seed` over the tie graph."""
function _ed_bfs_zones(c::EnergyDispatchCore, seed::Int, size::Int)
    adj = [Int[] for _ in 1:c.n_zones]
    for l in 1:_ed_n_ties(c)
        push!(adj[c.tie_from[l]], c.tie_to[l])
        push!(adj[c.tie_to[l]], c.tie_from[l])
    end
    S = [seed]
    seen = falses(c.n_zones)
    seen[seed] = true
    head = 1
    while head <= length(S) && length(S) < size
        for u in adj[S[head]]
            if !seen[u] && length(S) < size
                seen[u] = true
                push!(S, u)
            end
        end
        head += 1
    end
    return S
end

"""
    _ed_plant_pocket!(rng, c; margin) -> Union{Nothing, EnergyAggregateCertificate}

Turn a connected set of ≥ 2 zones into an import-constrained load pocket in its
peak hour: the pocket's demand is raised to `(1 + margin)` × (local capacity +
delivered import capability), spread by `_ed_spread_demand!` so that no zone's
own row and no proper subset of the pocket is contradictory on its own.
"""
function _ed_plant_pocket!(rng::AbstractRNG, c::EnergyDispatchCore; margin::Float64)
    c.n_zones >= 3 || return nothing
    for _ in 1:10
        seed = rand(rng, 1:c.n_zones)
        size = clamp(round(Int, c.n_zones * _e_unif(rng, (0.15, 0.35))), min(4, c.n_zones - 1), c.n_zones - 1)
        S = _ed_bfs_zones(c, seed, size)
        length(S) >= 2 || continue
        isempty(_ed_cut_ties(c, S)) && continue
        t = argmax([sum(c.demand[z, t] for z in S) for t in 1:c.n_periods])
        need = (1 + margin) * _ed_pocket_supply(c, S, t)
        _ed_spread_demand!(c, S, t, need) || continue
        supply = _ed_pocket_supply(c, S, t)
        req = sum(c.demand[z, t] for z in S)
        req > supply || continue
        return EnergyAggregateCertificate(:import_pocket, sort(S), [t], supply, req)
    end
    return nothing
end

"""
    _ed_emission_lower_bound(c) -> (bound, row_min)

Lower bound on horizon emissions implied by balance rows and bounds (see
`:emissions_budget`), and the emission row's own minimum activity
`Σ e_g lb[g,t]` (what presolve can see from the row alone).
"""
function _ed_emission_lower_bound(c::EnergyDispatchCore)
    G = _ed_n_units(c)
    emitting = [g for g in 1:G if c.emission_rate[g] > 0]
    isempty(emitting) && return 0.0, 0.0
    emin = minimum(c.emission_rate[g] for g in emitting)
    bound = 0.0
    row_min = 0.0
    for t in 1:c.n_periods
        clean_ub = sum((_ed_upper(c, g, t) for g in 1:G if c.emission_rate[g] == 0); init=0.0)
        dirty_lb = sum(_ed_lower(c, g, t) for g in emitting)
        dirty_lb_emis = sum(c.emission_rate[g] * _ed_lower(c, g, t) for g in emitting)
        need = max(0.0, _ed_system_demand(c, t) - clean_ub - dirty_lb)
        bound += dirty_lb_emis + emin * need
        row_min += dirty_lb_emis
    end
    return bound, row_min
end

# =============================================================================
# Transmission-grid helpers (dc_opf, security_constrained_dc_opf)
# =============================================================================

"""
    _energy_grid(rng, B, L) -> (x, y, from, to, len, ehv)

Geometric meshed transmission grid on `B` buses with ≈ `L` lines. Buses cluster
around load centres (70 %) over a sparse rural background (30 %) on a map whose
side grows like √B (constant bus density). Lines come from 6-nearest-neighbour
candidates (spatial hashing, O(B)): a Kruskal minimum spanning forest, bridges
between forest components, an extra-high-voltage backbone (a spanning tree
over the load centres' hub buses plus some meshing, flagged in `ehv`), then
meshing — radial leaves are first given a second connection, then short
candidates are added until `L` lines exist. Real grids are like this: mostly
meshed, average degree ≈ 2.6–3.2, few radial spurs, and a low-impedance EHV
overlay that keeps voltage-angle spreads moderate as the grid grows.
"""
function _energy_grid(rng::AbstractRNG, B::Int, L::Int)
    side = 15.0 * sqrt(B)
    n_centres = max(1, round(Int, B / 50))
    cx = side .* rand(rng, n_centres)
    cy = side .* rand(rng, n_centres)
    spread = 0.25 * side / sqrt(n_centres)
    x = zeros(B)
    y = zeros(B)
    hub = zeros(Int, n_centres)
    hub_dist = fill(Inf, n_centres)
    for b in 1:B
        if rand(rng) < 0.7
            k = rand(rng, 1:n_centres)
            x[b] = clamp(cx[k] + spread * randn(rng), 0.0, side)
            y[b] = clamp(cy[k] + spread * randn(rng), 0.0, side)
            d = hypot(x[b] - cx[k], y[b] - cy[k])
            d < hub_dist[k] && ((hub_dist[k], hub[k]) = (d, b))
        else
            x[b] = side * rand(rng)
            y[b] = side * rand(rng)
        end
    end
    # Spatial hash.
    cell = 15.0
    ncell = max(1, ceil(Int, side / cell))
    buckets = Dict{Tuple{Int, Int}, Vector{Int}}()
    cellof(b) = (min(ncell, 1 + floor(Int, x[b] / cell)), min(ncell, 1 + floor(Int, y[b] / cell)))
    for b in 1:B
        push!(get!(buckets, cellof(b), Int[]), b)
    end
    k = min(6, B - 1)
    cand = Set{Tuple{Int, Int}}()
    for b in 1:B
        cxb, cyb = cellof(b)
        found = Int[]
        r = 0
        while true
            for i in (cxb - r):(cxb + r), j in (cyb - r):(cyb + r)
                (max(abs(i - cxb), abs(j - cyb)) == r) || continue
                v = get(buckets, (i, j), nothing)
                v === nothing && continue
                for u in v
                    u != b && push!(found, u)
                end
            end
            # Ring r covers every point within distance r·cell of b's cell.
            (length(found) >= k && r >= 1) && break
            r > ncell && break
            r += 1
        end
        dists = [hypot(x[u] - x[b], y[u] - y[b]) for u in found]
        order = sortperm(dists)
        for i in order[1:min(k, length(order))]
            u = found[i]
            push!(cand, (min(u, b), max(u, b)))
        end
    end
    candidates = collect(cand)
    clen = [hypot(x[a] - x[b], y[a] - y[b]) for (a, b) in candidates]
    order = sortperm(clen)
    parent = collect(1:B)
    function findroot(v)
        while parent[v] != v
            parent[v] = parent[parent[v]]
            v = parent[v]
        end
        return v
    end
    from = Int[]
    to = Int[]
    used = Set{Tuple{Int, Int}}()
    for i in order
        a, b = candidates[i]
        ra, rb = findroot(a), findroot(b)
        ra == rb && continue
        parent[ra] = rb
        push!(from, a)
        push!(to, b)
        push!(used, (a, b))
    end
    # Join the remaining forest components (chain them by nearest representatives).
    roots = unique(findroot(b) for b in 1:B)
    if length(roots) > 1
        members = Dict(r => Int[] for r in roots)
        for b in 1:B
            push!(members[findroot(b)], b)
        end
        comps = [members[r] for r in roots]
        sort!(comps; by=length, rev=true)
        giant = copy(comps[1])
        for comp in comps[2:end]
            rep = comp[rand(rng, eachindex(comp))]
            # Nearest bus already in the connected part (sampled if huge).
            pool = length(giant) > 4000 ? giant[rand(rng, 1:length(giant), 4000)] : giant
            u = pool[argmin([hypot(x[v] - x[rep], y[v] - y[rep]) for v in pool])]
            push!(from, rep)
            push!(to, u)
            push!(used, (min(rep, u), max(rep, u)))
            append!(giant, comp)
        end
    end
    # Extra-high-voltage backbone between load-centre hubs (Prim over hubs).
    ehv_keys = Set{Tuple{Int, Int}}()
    hubs = unique(filter(>(0), hub))
    H = length(hubs)
    if H >= 2
        hd(i, j) = hypot(x[hubs[i]] - x[hubs[j]], y[hubs[i]] - y[hubs[j]])
        intree = falses(H)
        best = fill(Inf, H)
        par = zeros(Int, H)
        intree[1] = true
        for j in 2:H
            best[j] = hd(1, j)
            par[j] = 1
        end
        hub_edges = Tuple{Int, Int}[]
        for _ in 2:H
            j = 0
            bj = Inf
            for k in 1:H
                if !intree[k] && best[k] < bj
                    bj = best[k]
                    j = k
                end
            end
            intree[j] = true
            push!(hub_edges, (par[j], j))
            for k in 1:H
                if !intree[k] && hd(j, k) < best[k]
                    best[k] = hd(j, k)
                    par[k] = j
                end
            end
        end
        # Mesh the backbone: some hubs also link to their nearest non-tree hub.
        for i in 1:H
            rand(rng) < 0.4 || continue
            others = [j for j in 1:H if j != i]
            j = others[argmin([hd(i, j) for j in others])]
            push!(hub_edges, (i, j))
        end
        for (i, j) in hub_edges
            a, b = hubs[i], hubs[j]
            key = (min(a, b), max(a, b))
            (a == b || key in ehv_keys) && continue
            push!(ehv_keys, key)
            if !(key in used)
                push!(used, key)
                push!(from, a)
                push!(to, b)
            end
        end
    end
    degree = zeros(Int, B)
    for (a, b) in zip(from, to)
        degree[a] += 1
        degree[b] += 1
    end
    nbr_cand = [Tuple{Float64, Int}[] for _ in 1:B]
    for (i, (a, b)) in enumerate(candidates)
        push!(nbr_cand[a], (clen[i], b))
        push!(nbr_cand[b], (clen[i], a))
    end
    for v in 1:B
        sort!(nbr_cand[v])
    end
    # Mesh radial leaves first.
    for v in shuffle(rng, collect(1:B))
        length(from) >= L && break
        degree[v] == 1 || continue
        rand(rng) < 0.85 || continue
        for (_, u) in nbr_cand[v]
            key = (min(u, v), max(u, v))
            key in used && continue
            push!(used, key)
            push!(from, v)
            push!(to, u)
            degree[u] += 1
            degree[v] += 1
            break
        end
    end
    # Then short candidates at random (rank-biased toward short lines).
    rest = [candidates[i] for i in order if !(candidates[i] in used)]
    i = 1
    while length(from) < L && i <= length(rest)
        a, b = rest[i]
        if rand(rng) < 0.7 || length(rest) - i < (L - length(from)) * 2
            push!(used, (a, b))
            push!(from, a)
            push!(to, b)
        end
        i += 1
    end
    len = [max(1.0, hypot(x[from[l]] - x[to[l]], y[from[l]] - y[to[l]])) for l in eachindex(from)]
    ehv = BitVector([(min(from[l], to[l]), max(from[l], to[l])) in ehv_keys for l in eachindex(from)])
    return x, y, from, to, len, ehv
end

"""
    _energy_line_parameters(rng, len, ehv) -> (susceptance, rating, voltage_class)

Per-line DC susceptance (1/x in p.u. on a 100 MVA base; with angles in
centiradians, `flow[MW] = susceptance·Δθ`) and thermal rating (MW) by voltage
class, from typical per-km reactances: backbone lines are 500 kV (`:ehv`, x ≈
0.00011 p.u./km, 2000–3200 MW), long lines are mostly 345 kV (`:hv`, x ≈
0.00033 p.u./km, 800–1500 MW), the rest 115–230 kV (`:lv`, x ≈ 0.001 p.u./km,
150–450 MW). Susceptances are clamped to [5, 1000].
"""
function _energy_line_parameters(rng::AbstractRNG, len::Vector{Float64}, ehv::AbstractVector{Bool})
    L = length(len)
    cutoff = L == 0 ? 0.0 : sort(len)[clamp(ceil(Int, 0.75 * L), 1, L)]
    sus = zeros(L)
    rating = zeros(L)
    class = fill(:lv, L)
    for l in 1:L
        class[l] = ehv[l] ? :ehv : (len[l] >= cutoff ? (rand(rng) < 0.7 ? :hv : :lv) : (rand(rng) < 0.1 ? :hv : :lv))
        per_km, lo, hi = class[l] == :ehv ? (9000.0, 2000.0, 3200.0) :
            class[l] == :hv ? (3000.0, 800.0, 1500.0) : (1000.0, 150.0, 450.0)
        sus[l] = clamp(per_km / len[l] * _e_unif(rng, (0.8, 1.2)), 5.0, 1000.0)
        rating[l] = _e_unif(rng, (lo, hi))
    end
    return sus, rating, class
end

"""
    _dc_flows(B, from, to, sus, injection, ref; outage=0) -> (θ, flow)

DC power flow: solve the reduced Laplacian (reference bus removed) for the bus
angles that realize `injection`, then line flows `sus[l]·(θ_from − θ_to)`. Line
`outage` (if nonzero) is removed from the network (its flow is reported as 0).
The network minus the outage must stay connected.
"""
function _dc_flows(B::Int, from, to, sus, injection::Vector{Float64}, ref::Int; outage::Int=0)
    L = length(from)
    rows = zeros(Int, 4L)
    cols = zeros(Int, 4L)
    vals = zeros(Float64, 4L)
    for l in 1:L
        a, b = from[l], to[l]
        s = l == outage ? 0.0 : sus[l]
        k = 4(l - 1)
        rows[k + 1], cols[k + 1], vals[k + 1] = a, a, s
        rows[k + 2], cols[k + 2], vals[k + 2] = b, b, s
        rows[k + 3], cols[k + 3], vals[k + 3] = a, b, -s
        rows[k + 4], cols[k + 4], vals[k + 4] = b, a, -s
    end
    Lap = sparse(rows, cols, vals, B, B)
    keep = [b for b in 1:B if b != ref]
    θ = zeros(B)
    θ[keep] = Lap[keep, keep] \ injection[keep]
    flow = [l == outage ? 0.0 : sus[l] * (θ[from[l]] - θ[to[l]]) for l in 1:L]
    return θ, flow
end

"""
    _graph_bridges(B, from, to) -> BitVector

Bridges of the (multi)graph — lines whose outage disconnects the grid (Tarjan
low-link, iterative DFS).
"""
function _graph_bridges(B::Int, from, to)
    L = length(from)
    adj = [Tuple{Int, Int}[] for _ in 1:B]
    for l in 1:L
        push!(adj[from[l]], (to[l], l))
        push!(adj[to[l]], (from[l], l))
    end
    disc = zeros(Int, B)
    low = zeros(Int, B)
    is_bridge = falses(L)
    timer = 0
    for root in 1:B
        disc[root] == 0 || continue
        timer += 1
        disc[root] = timer
        low[root] = timer
        stack = [(root, 0, 1)]   # (vertex, parent edge, next adjacency index)
        while !isempty(stack)
            v, pe, i = stack[end]
            if i <= length(adj[v])
                stack[end] = (v, pe, i + 1)
                u, l = adj[v][i]
                l == pe && continue
                if disc[u] == 0
                    timer += 1
                    disc[u] = timer
                    low[u] = timer
                    push!(stack, (u, l, 1))
                else
                    low[v] = min(low[v], disc[u])
                end
            else
                pop!(stack)
                if !isempty(stack)
                    p = stack[end][1]
                    low[p] = min(low[p], low[v])
                    low[v] > disc[p] && (is_bridge[pe] = true)
                end
            end
        end
    end
    return is_bridge
end

"""Connected bus set grown by BFS from `seed` (≤ `size` buses, never `exclude`)."""
function _grid_bfs(B::Int, from, to, seed::Int, size::Int; exclude::Int=0)
    adj = [Int[] for _ in 1:B]
    for l in eachindex(from)
        push!(adj[from[l]], to[l])
        push!(adj[to[l]], from[l])
    end
    S = [seed]
    seen = falses(B)
    seen[seed] = true
    exclude > 0 && (seen[exclude] = true)
    head = 1
    while head <= length(S) && length(S) < size
        for u in adj[S[head]]
            if !seen[u] && length(S) < size
                seen[u] = true
                push!(S, u)
            end
        end
        head += 1
    end
    return S
end

"""
    _energy_grid_fleet(rng, B, G; hour) -> NamedTuple

Generators for a bus-level snapshot: technology mix from the shared catalogue,
bus siting (peakers near load centres, renewables anywhere), per-unit cost
heterogeneity, and snapshot availability for the hour `hour` (solar follows
the sun, wind a system-wide level with local noise, hydro a seasonal level).
`pmin` is the must-run floor of committed units.
"""
function _energy_grid_fleet(rng::AbstractRNG, B::Int, G::Int; hour::Int)
    weights = [ENERGY_TECHNOLOGIES[k].weight for k in ENERGY_TECH_NAMES]
    cdf = cumsum(weights ./ sum(weights))
    tech = Symbol[]
    for g in 1:G
        k = min(length(cdf), searchsortedfirst(cdf, rand(rng)))
        push!(tech, ENERGY_TECH_NAMES[k])
    end
    # Guarantee some dispatchable capacity in small systems.
    tech[1] = :ccgt
    G >= 2 && (tech[2] = :coal)
    gen_bus = rand(rng, 1:B, G)
    wind_level = _e_unif(rng, (0.15, 0.7))
    solar_level = 6 <= hour <= 19 ? max(0.0, sin(π * (hour - 5.5) / 14.0))^1.3 * _e_unif(rng, (0.5, 0.85)) : 0.0
    hydro_level = _e_unif(rng, (0.45, 0.85))
    pmin = zeros(G)
    pmax = zeros(G)
    cost = zeros(G)
    for g in 1:G
        spec = ENERGY_TECHNOLOGIES[tech[g]]
        cap = _e_logunif(rng, spec.capacity)
        avail = if tech[g] == :wind
            clamp(wind_level * _e_unif(rng, (0.7, 1.3)), 0.0, 1.0)
        elseif tech[g] == :solar
            clamp(solar_level * _e_unif(rng, (0.85, 1.1)), 0.0, 1.0)
        elseif tech[g] == :hydro
            clamp(hydro_level * _e_unif(rng, (0.85, 1.1)), 0.1, 1.0)
        else
            1.0
        end
        pmax[g] = cap * avail
        pmin[g] = _e_unif(rng, spec.min_stable) * pmax[g]
        cost[g] = _e_unif(rng, spec.cost)
    end
    return (; tech, gen_bus, pmin, pmax, cost)
end

"""
    _energy_bus_loads(rng, B) -> Vector{Float64}

Load shares over buses (summing to 1): ≈ 20 % of buses are pure switching
stations with no load; the rest draw lognormal shares.
"""
function _energy_bus_loads(rng::AbstractRNG, B::Int)
    w = zeros(B)
    for b in 1:B
        rand(rng) < 0.2 && continue
        w[b] = exp(0.9 * randn(rng))
    end
    sum(w) == 0 && (w[1] = 1.0)
    return w ./ sum(w)
end

"""
    DCPocketCertificate

Load-pocket infeasibility certificate for the DC network variants. Summing the
nodal balance rows of the connected bus set `buses` (in the base case when
`contingency == 0`, otherwise in the post-contingency network of contingency
index `contingency`, whose outaged line is `outaged_line`) gives
`Σ_{g in pocket} p_g − Σ_{b in pocket} d_b = net flow out across the cut`. The
crossing lines `cut_lines` (minus the outaged one) carry at most their rating
each, so `local_capacity + import_capability < pocket_demand` is a
contradiction from LP rows and bounds alone.
"""
struct DCPocketCertificate
    buses::Vector{Int}
    cut_lines::Vector{Int}
    contingency::Int
    outaged_line::Int
    local_capacity::Float64
    import_capability::Float64
    pocket_demand::Float64
end

"""Lines with exactly one endpoint in the bus set `S`."""
function _grid_cut(B::Int, from, to, S)
    inS = falses(B)
    inS[S] .= true
    return [l for l in eachindex(from) if inS[from[l]] != inS[to[l]]]
end
