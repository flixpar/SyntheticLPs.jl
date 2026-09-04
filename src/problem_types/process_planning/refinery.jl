using JuMP
using Random
using Distributions

# Refinery configuration ladder, from a simple topping-plus-reform plant to a
# deep-conversion complex. Each level names the processing units present; the
# constructor derives the stream set, blend pairs, and products from them.
const _REFINERY_CONFIGS =
    (:topping_reform, :hydroskimming, :catalytic, :deep_conversion)

# Quality attribute indices into `quality[c, s, a]`.
const _RP_OCT = 1  # blending octane number, (R+M)/2 basis
const _RP_RVP = 2  # Reid vapour pressure, psi
const _RP_SUL = 3  # sulphur, wt %
const _RP_CET = 4  # blending cetane number
const _RP_DEN = 5  # density, kg/L at 15 C
const _RP_VIS = 6  # kinematic viscosity, cSt at 50 C
const _RP_N_ATTRS = 6

"""
Maximum accepted variable target for `process_planning/refinery`.

The blend block carries one variable per (crude, stream, product, period), and
the same tensor is mirrored in the witness arrays, the stream-balance rows, and
JuMP's variable store, so a target near one million requires a multi-gigabyte
working set. The dimension ladder tops out well below the `supply_chain` cap:
the largest honest instance (30 crudes, deep-conversion configuration, weekly
periods) reaches roughly 200,000 variables, so anything above that is rejected
rather than silently under-sized.
"""
const MAX_REFINERY_PLANNING_VARIABLES = 200_000

"""
A complete primal point of the refinery planning LP: crude purchases and tank
levels, per-crude CDU feed with swing-cut displacements, per-mode unit feed by
crude origin, assay-origin blend allocations, product sales, and product tank
levels for every period. Feasibility can be re-checked by pure arithmetic
against every row of the model (see `test/problem_types/process_planning.jl`,
which also hands the point to JuMP through `primal_feasibility_report`).
"""
struct RefineryPlanWitness
    purchase::Matrix{Float64}         # [crude, period]
    crude_inventory::Matrix{Float64}  # [crude, period]
    crude_feed::Matrix{Float64}       # [crude, period]
    swing::Array{Float64,3}           # [crude, boundary, period]
    mode_feed::Array{Float64,3}       # [crude, mode, period]
    blend::Array{Float64,3}           # [crude, blend pair, period]
    sales::Matrix{Float64}            # [product, period]
    product_inventory::Matrix{Float64}# [product, period]
end

"""
Cumulative supply-demand refutation for an infeasible instance. Product
`product` must ship at least `demand` barrels over periods `1:horizon`, while
every row of the model limits shipments to `initial_inventory +
yield_bound * crude_bound` barrels: crude runs cannot exceed the CDU capacity
or the sum of initial tank stock and purchase upper bounds (per-crude
inventory balance), and at most `yield_bound` barrels of blendstock can be
produced per barrel of crude `argmax`-crude even when every unit runs at its
best mode (`yield_bound` carries the swing-cut allowance, since swing
variables can also shift cut volume between neighbouring cuts). Both bounds
are aggregations of linear rows and variable bounds
only, so the certificate refutes the model as built - the variant is a pure
LP, and no integrality is involved.
"""
struct RefinerySupplyCertificate
    product::Int
    horizon::Int
    demand::Float64
    initial_inventory::Float64
    cdu_bound::Float64
    purchase_bound::Float64
    crude_bound::Float64
    yield_bound::Float64
    upper_bound::Float64
    margin::Float64
end

"""
The correlated market condition applied to an unknown-status instance: all
crude purchase ceilings and unit capacities are scaled by `supply_factor`
while term-contract sales floors move by `demand_factor`. The factors are
positioned on the horizon by the golden-ratio sequence of the seed, so any
block of seeds produces a genuine feasibility mix instead of a binomial
accident concentrated at one end of the band.
"""
struct RefineryMarketScenario
    supply_factor::Float64
    demand_factor::Float64
    position::Float64
end

"""
    RefineryPlanningProblem <: ProblemGenerator

Multi-period petroleum-refinery production-planning LP in the classical
"refinery LP" shape: buy crude under term/spot contracts, distill it on the
CDU with swing cuts, upgrade intermediates on conversion units that run in
several modes, blend finished products from assay-origin streams under linear
quality specifications, carry crude and product inventories, and sell against
seasonal term and spot demand. The objective maximises refinery profit.

Structural pieces, all drawn from published refinery planning models:

- crude assays: per-crude distillation-cut yields and cut qualities (octane,
  sulphur, cetane, density, viscosity) sampled from a lightness/sweetness
  slate, so stream qualities depend on crude origin exactly as blend-by-assay
  refinery LPs require;
- swing cuts: bounded volume transfer between adjacent CDU cuts;
- multi-mode units: severity-paired reformer modes with mode-specific
  reformate pools, and max-gasoline / max-distillate operation on the FCC and
  hydrocracker;
- linear blend quality rows: octane and cetane blend linearly, RVP blends
  linearly in the Chevron index `RVP^1.25`, and viscosity blends linearly in
  the Walter index `cSt^(1/3)`;
- industry-band product specifications selected as the tightest band the
  planted reference recipe meets with margin (octane 84-87 / 89-93 AKI,
  sulphur 15 ppm to 3.5 wt %, cetane 40-51, fuel-oil viscosity 180-700 cSt),
  so specifications are realistic without ever exceeding what the refinery
  can actually blend;
- seasonal demand and prices with product-specific phases (summer gasoline,
  winter heating oil, paving asphalt) absorbed by tankage.

`feasible_witness` is populated only for a requested-feasible instance,
`infeasibility_certificate` only for a requested-infeasible one, and
`market_scenario` only for an unknown-status sample. `build_model` is
deterministic and the variant is a pure LP, so `relax_integer` has nothing
to relax.

The exact variable count is
`n_periods * (3*n_crudes + n_crudes*n_swing + n_crudes*n_modes +
n_crudes*n_blend_pairs + 2*n_products)` and the row count is
`n_periods * (n_crudes + n_crudes*n_streams + n_units + n_products +
n_active_specs + 2)`. Targets through
`MAX_REFINERY_PLANNING_VARIABLES` are supported; larger targets raise
`ArgumentError` before any assay data is allocated.
"""
struct RefineryPlanningProblem <: ProblemGenerator
    configuration::Symbol
    n_periods::Int
    n_crudes::Int
    stream_names::Vector{Symbol}
    cut_yield::Matrix{Float64}          # [crude, CDU cut 1:6]
    quality::Array{Float64,3}           # [crude, stream, attribute]
    swing_pairs::Vector{Tuple{Int,Int}} # (lighter cut, heavier cut) stream indices
    swing_lo::Matrix{Float64}           # [crude, swing boundary]
    swing_hi::Matrix{Float64}           # [crude, swing boundary]
    blend_pairs::Vector{Tuple{Int,Int}} # (stream, product) blend eligibility,
                                        # the index order of the blend tensor
    unit_names::Vector{Symbol}
    unit_feed::Vector{Int}              # feed stream index per unit
    mode_unit::Vector{Int}              # owning unit per mode
    mode_yields::Vector{Vector{Tuple{Int,Float64}}}  # output stream, vol/feed
    mode_cost::Vector{Float64}          # $/bbl feed
    unit_capacity::Matrix{Float64}      # [unit, period], outage-adjusted
    product_names::Vector{Symbol}
    product_price::Matrix{Float64}      # [product, period]
    sales_floor::Matrix{Float64}        # [product, period]
    sales_ceiling::Matrix{Float64}      # [product, period]
    product_tank::Vector{Float64}
    initial_product_inventory::Vector{Float64}
    spec_direction::Matrix{Int}         # [product, attribute]: -1/0/+1
    spec_rhs::Array{Float64,3}          # [product, attribute, period]
    crude_price::Vector{Float64}
    purchase_floor::Matrix{Float64}     # [crude, period]
    purchase_ceiling::Matrix{Float64}   # [crude, period]
    initial_crude_inventory::Vector{Float64}
    crude_tank_capacity::Float64
    cdu_capacity::Float64
    crude_carrying_cost::Float64
    product_carrying_cost::Vector{Float64}
    feasible_witness::Union{Nothing,RefineryPlanWitness}
    infeasibility_certificate::Union{Nothing,RefinerySupplyCertificate}
    market_scenario::Union{Nothing,RefineryMarketScenario}
    feasibility_status::FeasibilityStatus
end

_rp_variable_count(n_crudes, n_swing, n_modes, n_blend_pairs, n_products,
                   n_periods) =
    n_periods * (3 * n_crudes + n_crudes * n_swing + n_crudes * n_modes +
                 n_crudes * n_blend_pairs + 2 * n_products)

# Static blend compatibility: which streams may enter which product pools.
const _RP_BLEND_TABLE = Dict{Symbol,Vector{Symbol}}(
    :PGAS => [:RGAS, :FGAS, :CKGAS],
    :RGL  => [:LN, :ISOM, :R91, :R95, :R98, :FCCG, :CNAPH, :HCN],
    :PGL  => [:LN, :ISOM, :R91, :R95, :R98, :FCCG, :CNAPH, :HCN],
    :LNAPH=> [:LN],
    :JET  => [:HKER, :HCK],
    :DIES => [:HLGO, :LGO, :LCO, :CGO, :HCD, :HCK, :KERO, :HKER, :HGO],
    :HFO  => [:RESID, :VGO, :VRES, :SLRY, :CGO, :LCO, :HGO],
    :ASPH => [:VRES, :SLRY],
    :COKE => [:COKE],
)

# Base blending affinity per (stream, product); the constructor multiplies by
# lognormal noise, so gamma splits stay varied while biased towards the
# dispositions a real planner would choose (reformate to premium, cutter stock
# to fuel oil, treated diesel to the clean diesel pool).
const _RP_AFFINITY = Dict{Tuple{Symbol,Symbol},Float64}(
    (:LN, :RGL) => 0.50, (:LN, :PGL) => 0.15, (:LN, :LNAPH) => 0.60,
    (:ISOM, :RGL) => 1.00, (:ISOM, :PGL) => 0.70,
    (:R91, :RGL) => 1.00, (:R91, :PGL) => 0.40,
    (:R95, :RGL) => 1.00, (:R95, :PGL) => 1.00,
    (:R98, :RGL) => 1.00, (:R98, :PGL) => 1.40,
    (:FCCG, :RGL) => 1.00, (:FCCG, :PGL) => 0.80,
    (:CNAPH, :RGL) => 0.25, (:CNAPH, :PGL) => 0.05,
    (:HCN, :RGL) => 0.80, (:HCN, :PGL) => 0.40,
    (:HKER, :JET) => 1.00,
    (:HCK, :JET) => 0.80, (:HCK, :DIES) => 0.50,
    (:HLGO, :DIES) => 1.00,
    (:LGO, :DIES) => 0.55,
    (:LCO, :DIES) => 0.70, (:LCO, :HFO) => 0.50,
    (:CGO, :DIES) => 0.30, (:CGO, :HFO) => 0.40,
    (:HCD, :DIES) => 1.10,
    (:KERO, :DIES) => 0.60,
    (:HKER, :DIES) => 0.30,
    (:HGO, :DIES) => 0.25,
    (:RESID, :HFO) => 1.00,
    (:VGO, :HFO) => 0.70,
    (:HGO, :HFO) => 0.60,
    (:VRES, :HFO) => 1.00, (:VRES, :ASPH) => 1.00,
    (:SLRY, :HFO) => 0.90, (:SLRY, :ASPH) => 0.50,
    (:COKE, :COKE) => 1.00,
)

# Product base prices ($/bbl), seasonal amplitude, and demand-seasonality
# amplitude; the constructor adds noise and phases the season per instance.
const _RP_PRICE_TABLE = Dict{Symbol,Float64}(
    :PGAS => 46.0, :RGL => 74.0, :PGL => 81.0, :LNAPH => 58.0, :JET => 78.0,
    :DIES => 82.0, :HFO => 50.0, :ASPH => 58.0, :COKE => 18.0,
)

# Industry specification bands per (product, attribute), ordered loose to
# tight. The constructor picks the tightest band the planted recipe clears
# with margin; if none clears, the specification is dropped for that product.
const _RP_SPEC_GE = Dict{Tuple{Symbol,Int},Vector{Float64}}(
    (:RGL, _RP_OCT) => [84.0, 85.0, 86.0, 87.0],
    (:PGL, _RP_OCT) => [89.0, 91.0, 92.0, 93.0],
    (:DIES, _RP_CET) => [40.0, 45.0, 51.0],
    (:ASPH, _RP_VIS) => [150.0, 300.0],
)
const _RP_SPEC_LE = Dict{Tuple{Symbol,Int},Vector{Float64}}(
    (:JET, _RP_SUL) => [0.3, 0.2, 0.1],
    (:JET, _RP_DEN) => [0.84, 0.83],
    (:DIES, _RP_SUL) => [0.5, 0.05, 0.005, 0.0015],
    (:HFO, _RP_SUL) => [3.5, 2.0, 1.0, 0.5],
    (:HFO, _RP_VIS) => [700.0, 380.0, 180.0],
)

# Fraction of the feed stream each unit processes in the reference plan.
_rp_reference_utilization(rng::AbstractRNG, unit::Symbol) = begin
    unit == :NHT && return 1.0
    unit == :ISOM && return 1.0
    unit == :VDU && return rand(rng, Uniform(0.85, 1.0))
    unit == :KHT && return rand(rng, Uniform(0.72, 1.0))
    unit == :DHT && return rand(rng, Uniform(0.88, 1.0))
    unit == :REF && return rand(rng, Uniform(0.88, 0.99))
    unit == :HCU && return rand(rng, Uniform(0.35, 0.85))
    unit == :FCC && return rand(rng, Uniform(0.60, 0.95))
    unit == :CKR && return rand(rng, Uniform(0.55, 0.95))
    return 1.0
end

"""
Assemble the unit list, mode lists, and stream list for a configuration.
Reformer severities come in adjacent pairs (or a triple) so mode-specific
reformate pools stay contiguous on the octane ladder. Returns the ordered
unit definitions in topological feed order (a unit always follows the units
producing its feed).
"""
function _rp_unit_layout(rng::AbstractRNG, configuration::Symbol,
                         n_ref_modes::Int=2)
    severity = rand(rng) < 0.5 ? (:r91, :r95) : (:r95, :r98)
    severity = rand(rng) < 0.25 && n_ref_modes >= 3 ?
               (:r91, :r95, :r98) : severity
    if n_ref_modes == 1
        severity = (rand(rng) < 0.6 ? first(severity) : last(severity),)
    end
    reformate = Dict(:r91 => :R91, :r95 => :R95, :r98 => :R98)
    yield91 = Dict(:r91 => 0.91, :r95 => 0.87, :r98 => 0.83)
    gas91 = Dict(:r91 => 0.065, :r95 => 0.085, :r98 => 0.105)
    cost91 = Dict(:r91 => 2.3, :r95 => 3.2, :r98 => 4.1)

    ref_modes = Vector{Tuple{Symbol,Vector{Pair{Symbol,Float64}},Float64}}()
    for sev in severity
        push!(ref_modes, (sev, [Pair(:RGAS, gas91[sev]),
                                Pair(reformate[sev], yield91[sev])],
                          cost91[sev]))
    end

    units = Vector{Tuple{Symbol,Symbol,Vector{Tuple{Symbol,Vector{Pair{Symbol,Float64}},Float64}}}}()
    ref_feed = configuration == :topping_reform ? :HN : :HTN
    push!(units, (:REF, ref_feed, ref_modes))

    configuration == :topping_reform && return units

    push!(units, (:NHT, :HN, [(:desul, [Pair(:HTN, 0.995)], 1.2)]))
    push!(units, (:ISOM, :LN, [(:once_through, [Pair(:ISOM, 0.975)], 1.4)]))
    push!(units, (:KHT, :KERO, [(:desul, [Pair(:HKER, 0.992)], 1.1)]))
    push!(units, (:DHT, :LGO, [(:desul, [Pair(:HLGO, 0.990)], 1.6)]))
    if configuration != :hydroskimming
        vdu_vgo = rand(rng, Uniform(0.38, 0.50))
        push!(units, (:VDU, :RESID,
                     [(:vac, [Pair(:VGO, vdu_vgo),
                              Pair(:VRES, 0.985 - vdu_vgo)], 0.9)]))
        push!(units, (:FCC, :VGO, [
            (:max_gasoline, [Pair(:FGAS, 0.13), Pair(:FCCG, 0.55),
                             Pair(:LCO, 0.19), Pair(:SLRY, 0.09)], 2.4),
            (:max_distillate, [Pair(:FGAS, 0.10), Pair(:FCCG, 0.42),
                               Pair(:LCO, 0.31), Pair(:SLRY, 0.11)], 2.7),
        ]))
        if configuration == :deep_conversion || rand(rng) < 0.4
            push!(units, (:HCU, :VGO, [
                (:max_distillate, [Pair(:HCN, 0.14), Pair(:HCK, 0.24),
                                   Pair(:HCD, 0.56)], 4.8),
                (:max_naphtha, [Pair(:HCN, 0.34), Pair(:HCK, 0.18),
                                Pair(:HCD, 0.40)], 5.4),
            ]))
        end
        configuration == :deep_conversion &&
            push!(units, (:CKR, :VRES, [
                (:delayed, [Pair(:CKGAS, 0.09), Pair(:CNAPH, 0.14),
                            Pair(:CGO, 0.42), Pair(:COKE, 0.31)], 4.2),
            ]))
    end
    return units
end

_rp_streams_and_pairs(unit_defs, products::Vector{Symbol}) = begin
    stream_names = Symbol[:LN, :HN, :KERO, :LGO, :HGO, :RESID]
    for (uname, _, modes) in unit_defs, (_, yields, _) in modes,
        (out, _) in yields
        out in stream_names || push!(stream_names, out)
    end
    stream_index = Dict(name => i for (i, name) in enumerate(stream_names))
    pairs = Tuple{Int,Int}[]
    for (p, product) in enumerate(products), name in _RP_BLEND_TABLE[product]
        s = get(stream_index, name, 0)
        s > 0 && push!(pairs, (s, p))
    end
    sort!(pairs)
    return stream_names, stream_index, pairs
end

"""
Choose configuration, dimensions, swing boundaries, and product slate from a
variable target. Candidate configurations are sampled from the rng (complexity
and crude-slate width scaling with the target), each contributes its exact
per-period variable coefficient, and the horizon length is solved in closed
form and scanned. The deterministic minimal configuration (single crude,
single reformer mode, no swing cuts, base product slate) is always offered as
a candidate so small targets stay reachable.
"""
function _rp_choose_dimensions(rng::AbstractRNG, target_variables::Int)
    target_variables <= MAX_REFINERY_PLANNING_VARIABLES ||
        throw(ArgumentError(
            "process_planning/refinery supports target_variables <= " *
            "$(MAX_REFINERY_PLANNING_VARIABLES); requested $target_variables. " *
            "The assay-origin blend block is mirrored across several parallel " *
            "structures, so larger models need a multi-gigabyte working set.",
        ))
    target = max(target_variables, 1)

    n_candidates = 7
    best = nothing
    best_score = (Inf, Inf, Inf)
    for candidate in 1:n_candidates
        minimal = candidate == n_candidates
        progress = candidate / n_candidates
        complexity = minimal ? 0 :
                     target < 150 ? rand(rng, 0:1) :
                     target < 1_200 ? rand(rng, 1:2) :
                     target < 8_000 ? rand(rng, 1:3) : rand(rng, 2:3)
        configuration = _REFINERY_CONFIGS[complexity + 1]
        n_crudes = minimal ? 1 :
                   clamp(round(Int,
                               1 + 3 * progress +
                               26 * (target / MAX_REFINERY_PLANNING_VARIABLES)^0.6 *
                               rand(rng, Uniform(0.5, 1.3))),
                        1, 30)
        n_swing = minimal || target < 90 ? 0 :
                  target < 350 ? (rand(rng) < 0.5 ? 0 : 2) :
                  target < 3_000 ? (rand(rng) < 0.3 ? 2 : 5) : 5
        n_ref_modes = minimal || target <= 62 ? 1 :
                      rand(rng) < 0.25 ? 3 : 2
        products = _rp_product_slate(rng, configuration, target, minimal)
        unit_defs = _rp_unit_layout(rng, configuration, n_ref_modes)
        best_layout = unit_defs
        n_modes = sum(length(modes) for (_, _, modes) in unit_defs)
        _, _, pairs = _rp_streams_and_pairs(unit_defs, products)
        per_period = 3 * n_crudes + n_crudes * n_swing + n_crudes * n_modes +
                     n_crudes * length(pairs) + 2 * length(products)
        t_star = clamp(round(Int, target / per_period), 3, 126)
        for n_periods in max(3, t_star - 2):min(126, t_star + 2)
            size = _rp_variable_count(n_crudes, n_swing, n_modes,
                                      length(pairs), length(products), n_periods)
            error = abs(size - target) / target
            shape = abs(log(n_periods / clamp(6 * sqrt(target / 300), 3, 60)))
            score = (error, shape, rand(rng))
            if score < best_score
                best_score = score
                best = (configuration, n_crudes, n_swing, products, n_periods,
                        n_ref_modes, best_layout)
            end
        end
    end
    return best
end

function _rp_product_slate(rng::AbstractRNG, configuration::Symbol, target::Int,
                           minimal::Bool=false)
    products = Symbol[:PGAS, :RGL, :DIES, :HFO]
    minimal && return products
    push!(products, :LNAPH)
    (target >= 250 || configuration != :topping_reform) &&
        rand(rng) < 0.85 && push!(products, :PGL)
    configuration != :topping_reform && rand(rng) < 0.75 &&
        push!(products, :JET)
    configuration in (:catalytic, :deep_conversion) &&
        rand(rng) < 0.6 && push!(products, :ASPH)
    configuration == :deep_conversion && push!(products, :COKE)
    return products
end

"""
Sample the crude slate: lightness and sweetness per crude, distillation-cut
yields interpolated between a heavy and a light assay with per-cut noise
(renormalised so the assay sums to one barrel), cut qualities that inherit
sulphur from the crude with cut-dependent concentration factors, and a crude
price carrying lightness and sweetness premia. Topping-reform slates are
restricted to sweet crudes because the configuration has no desulphurisation
to fall back on, exactly as simple hydroskimming refineries run.
"""
function _rp_crude_slate(rng::AbstractRNG, n_crudes::Int, configuration::Symbol)
    lightness = rand(rng, Uniform(0.05, 0.95), n_crudes)
    sweet_cap = configuration == :topping_reform ? 0.30 : 3.0
    sulfur = similar(lightness)
    for c in 1:n_crudes
        sulfur[c] = clamp(3.1 * (1 - lightness[c])^1.3 *
                          rand(rng, LogNormal(0, 0.35)), 0.05, sweet_cap)
    end
    heavy = [0.04, 0.06, 0.07, 0.14, 0.08, 0.61]
    light = [0.17, 0.15, 0.14, 0.20, 0.06, 0.28]
    cut_yield = zeros(Float64, n_crudes, 6)
    for c in 1:n_crudes
        ln_min = configuration == :topping_reform
        for s in 1:6
            base = heavy[s] + (light[s] - heavy[s]) * lightness[c]
            # A topping-reform plant has no isomerizer, so its gasoline pool
            # leans on straight-run light naphtha; keep the assay naphtha-lean
            # and heavy-naphtha-rich so the pool octane stays defensible.
            if ln_min && s == 1
                base = min(base, 0.05)
            elseif ln_min && s == 2
                base = max(base, 0.16)
            end
            cut_yield[c, s] = max(0.015, base + rand(rng, Uniform(-0.015, 0.015)))
        end
        cut_yield[c, :] ./= sum(cut_yield[c, :])
    end
    crude_price = [clamp(58.0 + 9.0 * lightness[c] - 1.8 * sulfur[c] +
                         rand(rng, Normal(0, 2.0)), 45.0, 82.0)
                   for c in 1:n_crudes]
    return lightness, sulfur, cut_yield, crude_price
end

"""
Derive the quality of every stream of every crude origin. CDU-cut qualities
come from the assay; conversion-unit output qualities follow the feed origin
through realistic transforms (treated streams drop to single-digit-ppm
sulphur and gain cetane, cycle-oil sulphur concentrates above its vacuum
gas-oil feed, reformate octane follows severity).
"""
function _rp_qualities(
    rng::AbstractRNG,
    lightness::Vector{Float64},
    sulfur::Vector{Float64},
    n_crudes::Int,
    stream_names::Vector{Symbol},
    stream_index::Dict{Symbol,Int},
    mode_defs,
)
    S = length(stream_names)
    quality = zeros(Float64, n_crudes, S, _RP_N_ATTRS)
    present(name) = haskey(stream_index, name)
    # Function-scoped row accessor: closures assigned inside a `for` body are
    # loop-local in Julia, so `q` must live outside the loops below.
    q(c::Int, name::Symbol) = quality[c, stream_index[name], :]
    for c in 1:n_crudes
        q(c, :LN)[_RP_OCT] = rand(rng, Uniform(62, 72))
        q(c, :LN)[_RP_RVP] = rand(rng, Uniform(10.5, 13.5))
        q(c, :LN)[_RP_SUL] = min(sulfur[c] * rand(rng, Uniform(0.005, 0.02)), 0.02)
        q(c, :LN)[_RP_CET] = rand(rng, Uniform(28, 35))
        q(c, :LN)[_RP_DEN] = rand(rng, Uniform(0.66, 0.70))
        q(c, :LN)[_RP_VIS] = rand(rng, Uniform(0.6, 0.8))

        q(c, :HN)[_RP_OCT] = rand(rng, Uniform(36, 48))
        q(c, :HN)[_RP_RVP] = rand(rng, Uniform(1.5, 3.0))
        q(c, :HN)[_RP_SUL] = sulfur[c] * rand(rng, Uniform(0.05, 0.15))
        q(c, :HN)[_RP_DEN] = rand(rng, Uniform(0.72, 0.76))
        q(c, :HN)[_RP_VIS] = rand(rng, Uniform(0.9, 1.3))

        q(c, :KERO)[_RP_SUL] = min(sulfur[c] * rand(rng, Uniform(0.35, 0.6)), 3.2)
        q(c, :KERO)[_RP_CET] = rand(rng, Uniform(42, 52))
        q(c, :KERO)[_RP_DEN] = rand(rng, Uniform(0.78, 0.82))
        q(c, :KERO)[_RP_VIS] = rand(rng, Uniform(1.4, 2.2))

        q(c, :LGO)[_RP_SUL] = sulfur[c] * rand(rng, Uniform(0.9, 1.3))
        q(c, :LGO)[_RP_CET] = rand(rng, Uniform(46, 58))
        q(c, :LGO)[_RP_DEN] = rand(rng, Uniform(0.83, 0.87))
        q(c, :LGO)[_RP_VIS] = rand(rng, Uniform(3, 8))

        q(c, :HGO)[_RP_SUL] = sulfur[c] * rand(rng, Uniform(1.0, 1.4))
        q(c, :HGO)[_RP_CET] = rand(rng, Uniform(38, 50))
        q(c, :HGO)[_RP_DEN] = rand(rng, Uniform(0.87, 0.92))
        q(c, :HGO)[_RP_VIS] = rand(rng, Uniform(9, 22))

        q(c, :RESID)[_RP_SUL] = min(sulfur[c] * rand(rng, Uniform(1.25, 1.6)), 3.2)
        q(c, :RESID)[_RP_DEN] = rand(rng, Uniform(0.94, 0.99))
        q(c, :RESID)[_RP_VIS] = rand(rng, Uniform(120, 380))

        present(:VGO) || continue
        q(c, :VGO)[_RP_SUL] = sulfur[c] * rand(rng, Uniform(1.15, 1.45))
        q(c, :VGO)[_RP_CET] = rand(rng, Uniform(32, 42))
        q(c, :VGO)[_RP_DEN] = rand(rng, Uniform(0.90, 0.93))
        q(c, :VGO)[_RP_VIS] = rand(rng, Uniform(28, 65))

        q(c, :VRES)[_RP_SUL] = min(sulfur[c] * rand(rng, Uniform(1.35, 1.65)), 3.5)
        q(c, :VRES)[_RP_DEN] = rand(rng, Uniform(0.96, 1.02))
        q(c, :VRES)[_RP_VIS] = rand(rng, Uniform(500, 2600))
    end

    # Per-instance (not per-crude) blending values for crack streams.
    fccg_oct = rand(rng, Uniform(87, 91))
    isom_oct = rand(rng, Uniform(83, 88))
    for c in 1:n_crudes
        present(:HTN) || continue
        q(c, :HTN)[_RP_OCT] = quality[c, stream_index[:HN], _RP_OCT] + 2
        q(c, :HTN)[_RP_SUL] = 0.0005
        q(c, :HTN)[_RP_DEN] = quality[c, stream_index[:HN], _RP_DEN]
        q(c, :HTN)[_RP_VIS] = quality[c, stream_index[:HN], _RP_VIS]
    end
    for c in 1:n_crudes
        present(:ISOM) || continue
        q(c, :ISOM)[_RP_OCT] = isom_oct + rand(rng, Normal(0, 0.5))
        q(c, :ISOM)[_RP_RVP] = rand(rng, Uniform(11, 14))
        q(c, :ISOM)[_RP_SUL] = 0.0001
        q(c, :ISOM)[_RP_DEN] = rand(rng, Uniform(0.67, 0.70))
        q(c, :ISOM)[_RP_VIS] = rand(rng, Uniform(0.6, 0.8))
    end
    octane91 = Dict(:r91 => 91.0, :r95 => 95.0, :r98 => 98.0)
    for (label, yields, _) in mode_defs, (out, _) in yields
        out == :RGAS && continue
        for c in 1:n_crudes
            q(c, out)[_RP_OCT] = octane91[label] + rand(rng, Normal(0, 0.5))
            q(c, out)[_RP_RVP] = rand(rng, Uniform(2.5, 5.0))
            q(c, out)[_RP_SUL] = 0.0002
            q(c, out)[_RP_DEN] = rand(rng, Uniform(0.78, 0.82))
            q(c, out)[_RP_VIS] = rand(rng, Uniform(0.7, 1.1))
        end
    end
    for c in 1:n_crudes
        present(:HKER) || continue
        q(c, :HKER)[_RP_SUL] = rand(rng, Uniform(0.0006, 0.002))
        q(c, :HKER)[_RP_CET] = quality[c, stream_index[:KERO], _RP_CET] + 1
        q(c, :HKER)[_RP_DEN] = quality[c, stream_index[:KERO], _RP_DEN] - 0.002
        q(c, :HKER)[_RP_VIS] = rand(rng, Uniform(1.5, 2.0))
    end
    for c in 1:n_crudes
        present(:HLGO) || continue
        q(c, :HLGO)[_RP_SUL] = rand(rng, Uniform(0.0006, 0.0012))
        q(c, :HLGO)[_RP_CET] = min(quality[c, stream_index[:LGO], _RP_CET] + 3, 62)
        q(c, :HLGO)[_RP_DEN] = quality[c, stream_index[:LGO], _RP_DEN]
        q(c, :HLGO)[_RP_VIS] = rand(rng, Uniform(2.5, 5.0))
    end
    for c in 1:n_crudes
        present(:FCCG) || continue
        q(c, :FCCG)[_RP_OCT] = clamp(fccg_oct + rand(rng, Normal(0, 0.7)), 85, 92)
        q(c, :FCCG)[_RP_RVP] = rand(rng, Uniform(4.5, 6.5))
        q(c, :FCCG)[_RP_SUL] = rand(rng, Uniform(0.02, 0.06))
        q(c, :FCCG)[_RP_DEN] = rand(rng, Uniform(0.74, 0.77))
        q(c, :FCCG)[_RP_VIS] = rand(rng, Uniform(0.5, 0.7))
        q(c, :LCO)[_RP_SUL] = min(
            quality[c, stream_index[:VGO], _RP_SUL] * rand(rng, Uniform(1.25, 1.55)), 3.2)
        q(c, :LCO)[_RP_CET] = rand(rng, Uniform(20, 28))
        q(c, :LCO)[_RP_DEN] = rand(rng, Uniform(0.90, 0.95))
        q(c, :LCO)[_RP_VIS] = rand(rng, Uniform(3, 6))
        q(c, :SLRY)[_RP_SUL] = min(
            quality[c, stream_index[:VGO], _RP_SUL] * rand(rng, Uniform(1.3, 1.6)), 3.45)
        q(c, :SLRY)[_RP_DEN] = rand(rng, Uniform(0.95, 1.0))
        q(c, :SLRY)[_RP_VIS] = rand(rng, Uniform(45, 120))
    end
    for c in 1:n_crudes
        present(:HCD) || continue
        q(c, :HCN)[_RP_OCT] = rand(rng, Uniform(78, 84))
        q(c, :HCN)[_RP_RVP] = rand(rng, Uniform(7, 10))
        q(c, :HCN)[_RP_SUL] = 0.0001
        q(c, :HCN)[_RP_DEN] = rand(rng, Uniform(0.70, 0.74))
        q(c, :HCK)[_RP_SUL] = 0.0001
        q(c, :HCK)[_RP_CET] = rand(rng, Uniform(48, 55))
        q(c, :HCK)[_RP_DEN] = rand(rng, Uniform(0.79, 0.82))
        q(c, :HCK)[_RP_VIS] = rand(rng, Uniform(2, 3.5))
        q(c, :HCD)[_RP_SUL] = 0.0001
        q(c, :HCD)[_RP_CET] = rand(rng, Uniform(55, 65))
        q(c, :HCD)[_RP_DEN] = rand(rng, Uniform(0.82, 0.85))
        q(c, :HCD)[_RP_VIS] = rand(rng, Uniform(4, 8))
    end
    for c in 1:n_crudes
        present(:CNAPH) || continue
        q(c, :CNAPH)[_RP_OCT] = rand(rng, Uniform(60, 66))
        q(c, :CNAPH)[_RP_RVP] = rand(rng, Uniform(8, 11))
        q(c, :CNAPH)[_RP_SUL] = rand(rng, Uniform(0.05, 0.3))
        q(c, :CNAPH)[_RP_DEN] = rand(rng, Uniform(0.72, 0.75))
        q(c, :CNAPH)[_RP_VIS] = rand(rng, Uniform(0.6, 0.9))
        q(c, :CGO)[_RP_SUL] = min(
            quality[c, stream_index[:VRES], _RP_SUL] * 0.9, 3.2)
        q(c, :CGO)[_RP_CET] = rand(rng, Uniform(25, 35))
        q(c, :CGO)[_RP_DEN] = rand(rng, Uniform(0.90, 0.94))
        q(c, :CGO)[_RP_VIS] = rand(rng, Uniform(8, 25))
        q(c, :COKE)[_RP_SUL] = min(
            quality[c, stream_index[:VRES], _RP_SUL] * rand(rng, Uniform(1.3, 1.7)), 6.0)
    end
    return quality
end

"""
Seasonal demand deviation per product: amplitude and phase from the product
class (summer gasoline and paving asphalt, winter heating oil, shoulder
diesel), normalised to mean one so the reference plan sells what it produces
over the horizon while tankage absorbs the within-horizon swings.
"""
function _rp_demand_deviation(rng::AbstractRNG, products::Vector{Symbol},
                              n_periods::Int, gasoline_peak::Int)
    amplitude = Dict{Symbol,Tuple{Float64,Float64}}(
        :PGAS => (0.03, 0.06), :RGL => (0.10, 0.18), :PGL => (0.08, 0.15),
        :LNAPH => (0.04, 0.08), :JET => (0.06, 0.12), :DIES => (0.08, 0.15),
        :HFO => (0.25, 0.50), :ASPH => (0.50, 1.60), :COKE => (0.02, 0.05),
    )
    phase = Dict{Symbol,Int}(
        :PGAS => gasoline_peak + n_periods ÷ 3, :RGL => gasoline_peak,
        :PGL => gasoline_peak, :LNAPH => gasoline_peak, :JET => gasoline_peak,
        :DIES => gasoline_peak - n_periods ÷ 4, :HFO => gasoline_peak + n_periods ÷ 2,
        :ASPH => gasoline_peak, :COKE => gasoline_peak + n_periods ÷ 3,
    )
    delta = ones(Float64, length(products), n_periods)
    for (p, product) in enumerate(products)
        amp = rand(rng, Uniform(amplitude[product]...))
        delta[p, :] .= _pp_seasonal_deviation(rng, amp, phase[product],
                                              n_periods)
    end
    return delta
end

"""
Maximum fraction of one barrel of crude `c` that can end up as blendstock for
product `p`, ignoring capacities (an upper bound, since capacities only
reduce production): a recursion over the stream network that takes the best
of direct blending and committing the stream to each unit mode. Used by the
infeasibility certificate.
"""
function _rp_yield_path(cut_yield::Matrix{Float64}, unit_feed::Vector{Int},
                        mode_unit::Vector{Int},
                        mode_yields::Vector{Vector{Tuple{Int,Float64}}},
                        blend_pairs::Vector{Tuple{Int,Int}}, n_products::Int)
    pair_set = Set(blend_pairs)
    modes_on = Dict{Int,Vector{Int}}()
    for m in eachindex(mode_unit)
        push!(get!(modes_on, unit_feed[mode_unit[m]], Int[]), m)
    end
    reach = Dict{Tuple{Int,Int},Float64}()
    function _reach(s::Int, p::Int)
        key = (s, p)
        haskey(reach, key) && return reach[key]
        reach[key] = 0.0  # guards against cycles (the network is a DAG)
        best = (s, p) in pair_set ? 1.0 : 0.0
        for m in get(modes_on, s, Int[])
            best = max(best,
                       sum(y * _reach(o, p) for (o, y) in mode_yields[m]))
        end
        reach[key] = best
        return best
    end
    ypath = zeros(Float64, size(cut_yield, 1), n_products)
    for c in 1:size(cut_yield, 1), p in 1:n_products
        ypath[c, p] = sum(cut_yield[c, s] * _reach(s, p) for s in 1:6)
    end
    return ypath
end

function RefineryPlanningProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)
    configuration, n_crudes, n_swing, products, n_periods, _,
    unit_defs = _rp_choose_dimensions(rng, target)
    C, T, P = n_crudes, n_periods, length(products)
    stream_names, stream_index, pairs = _rp_streams_and_pairs(unit_defs, products)
    S = length(stream_names)
    lightness, sulfur, cut_yield, crude_price =
        _rp_crude_slate(rng, C, configuration)
    quality = _rp_qualities(rng, lightness, sulfur, C, stream_names,
                            stream_index, unit_defs[1][3])

    unit_names = Symbol[]
    unit_feed = Int[]
    mode_unit = Int[]
    mode_yields = Vector{Tuple{Int,Float64}}[]
    mode_cost = Float64[]
    for (u, (uname, feed_sym, modes)) in enumerate(unit_defs)
        push!(unit_names, uname)
        push!(unit_feed, stream_index[feed_sym])
        for (_, yields, cost) in modes
            push!(mode_unit, u)
            push!(mode_yields, [(stream_index[o], y) for (o, y) in yields])
            push!(mode_cost, cost)
        end
    end
    U = length(unit_names)
    M = length(mode_unit)
    product_index = Dict(name => i for (i, name) in enumerate(products))

    swing_set = sort!(randperm(rng, 5)[1:n_swing])
    swing_pairs = [(j, j + 1) for j in swing_set]
    swing_band = rand(rng, Uniform(0.02, 0.045), C, max(n_swing, 1))
    swing_lo = zeros(Float64, C, n_swing)
    swing_hi = zeros(Float64, C, n_swing)

    # Crude run profile: refinery scale in thousand bbl per period (a
    # 60-400 kbpd refinery over a month), with mild seasonality and one or
    # two partial turnarounds. Working in thousand-barrels keeps variable
    # bounds near 1e2-1e4 instead of 1e7, which matters for solver scaling.
    bpd = 10^rand(rng, Uniform(log10(6.0e4), log10(4.0e5)))
    run_level = bpd * 30.0 / 1000.0
    profile = [1 + 0.04 * sin(2π * t / T + rand(rng, Uniform(0, 2π))) for t in 1:T]
    for _ in 1:(rand(rng) < 0.5 ? 0 : rand(rng) < 0.7 ? 1 : 2)
        start = rand(rng, 2:max(2, T - 2))
        len = rand(rng, 2:min(3, T))
        profile[start:min(start + len - 1, T)] .*= rand(rng, Uniform(0.65, 0.80))
    end
    run = run_level .* profile .* rand(rng, Uniform(0.97, 1.03), T)
    mix_weights = exp.(rand(rng, Normal(0, 0.5), C))
    mix_weights ./= sum(mix_weights)
    crude_feed = [mix_weights[c] * run[t] for c in 1:C, t in 1:T]

    # Swing bounds are volumes, not assay fractions: the band (2-4.5% of the
    # source cut) scaled by the crude's smallest per-period run, so the single
    # [crude, swing] bound is valid in every period. Sizing from the assay
    # fraction alone would pin the swing variables six orders of magnitude
    # below the cut volumes they transfer between.
    min_run = minimum(crude_feed, dims = 2)
    for k in 1:n_swing
        swing_lo[:, k] .= -swing_band[:, k] .*
                          cut_yield[:, swing_pairs[k][1]] .* min_run
        swing_hi[:, k] .= swing_band[:, k] .*
                          cut_yield[:, swing_pairs[k][2]] .* min_run
    end

    # Reference plan: propagate stream availabilities in feed topological
    # order, running each unit at a realistic fraction of its feed stream.
    topo_rank = Dict(:NHT => 1, :ISOM => 2, :VDU => 3, :KHT => 4, :DHT => 5,
                     :HCU => 6, :FCC => 7, :REF => 8, :CKR => 9)
    order = sort(1:U, by = u -> topo_rank[unit_names[u]])
    unit_of = Dict(name => u for (u, name) in enumerate(unit_names))
    # Units whose feed stream has no blend destination (the naphtha chain into
    # the reformer) cannot turn down without stranding production against the
    # equality stream balance: they run at full availability with no outage.
    blendable = Set(s for (s, _) in pairs)
    orphan_feed = [unit_feed[u] ∉ blendable for u in 1:U]
    outage = ones(Float64, U, T)
    flexible = [u for u in 1:U if !orphan_feed[u]]
    for _ in 1:(rand(rng) < 0.45 ? 0 : rand(rng) < 0.75 ? 1 : 2)
        isempty(flexible) && break
        u = rand(rng, flexible)
        start = rand(rng, 1:T)
        len = rand(rng, 2:min(4, T))
        outage[u, start:min(start + len - 1, T)] .= rand(rng, Uniform(0.5, 0.75))
    end

    fraction = [_rp_reference_utilization(rng, name) for name in unit_names]
    for u in 1:U
        orphan_feed[u] && (fraction[u] = 1.0)
    end
    if haskey(unit_of, :ISOM) && :LNAPH in products
        # Leave some light naphtha for the merchant naphtha market.
        fraction[unit_of[:ISOM]] = rand(rng, Uniform(0.70, 0.95))
    end
    if haskey(unit_of, :HCU) && haskey(unit_of, :FCC)
        fraction[unit_of[:FCC]] =
            min(fraction[unit_of[:FCC]], 0.97 - fraction[unit_of[:HCU]])
    end

    mode_weight = zeros(Float64, M)
    for u in 1:U
        ms = findall(==(u), mode_unit)
        dominant = unit_names[u] == :REF ? last(ms) : first(ms)
        if length(ms) == 1
            mode_weight[ms[1]] = 1.0
        else
            mode_weight[dominant] = rand(rng, Uniform(0.55, 0.85))
            rest = [m for m in ms if m != dominant]
            for m in rest
                mode_weight[m] = (1 - mode_weight[dominant]) / length(rest)
            end
        end
    end

    avail = zeros(Float64, C, S, T)
    for c in 1:C, s in 1:6, t in 1:T
        avail[c, s, t] = cut_yield[c, s] * crude_feed[c, t]
    end
    mode_feed_plan = zeros(Float64, C, M, T)
    unit_capacity = zeros(Float64, U, T)
    for u in order
        s_feed = unit_feed[u]
        avail_total = [sum(avail[:, s_feed, t]) for t in 1:T]
        base_cap = fraction[u] * maximum(avail_total) * rand(rng, Uniform(1.06, 1.25))
        for t in 1:T
            unit_capacity[u, t] = base_cap * outage[u, t]
        end
        ms = findall(==(u), mode_unit)
        for t in 1:T
            planned = orphan_feed[u] ? avail_total[t] :
                      min(fraction[u] * avail_total[t], unit_capacity[u, t])
            if unit_names[u] == :CKR
                # Cap coker naphtha at a tenth of the gasoline pool so the
                # low-octane crack stream cannot swamp regular gasoline.
                rgl = sum(avail[c, stream_index[name], t]
                          for c in 1:C
                          for name in (:LN, :ISOM, :R91, :R95, :R98, :FCCG)
                          if haskey(stream_index, name))
                naph = sum(y for (o, y) in mode_yields[ms[1]]
                           if o == stream_index[:CNAPH])
                planned = min(planned, 0.10 * rgl / naph)
            end
            for c in 1:C
                share = avail_total[t] > 0 ? avail[c, s_feed, t] / avail_total[t] : 0.0
                origin_feed = planned * share
                avail[c, s_feed, t] -= origin_feed
                for m in ms
                    mode_feed_plan[c, m, t] = mode_weight[m] * origin_feed
                    for (out, y) in mode_yields[m]
                        avail[c, out, t] += y * mode_weight[m] * origin_feed
                    end
                end
            end
        end
    end

    # Blend allocation: affinity-biased shares of every stream's net
    # availability across its eligible products.
    gamma = zeros(Float64, S, P)
    for (s, p) in pairs
        affinity = get(_RP_AFFINITY, (stream_names[s], products[p]), 0.5)
        gamma[s, p] = affinity * rand(rng, LogNormal(0, 0.35))
    end
    for s in 1:S
        total = sum(gamma[s, :])
        total > 0 && (gamma[s, :] ./= total)
    end
    blend_plan = zeros(Float64, C, length(pairs), T)
    volume = zeros(Float64, P, T)
    blended_quality = zeros(Float64, P, _RP_N_ATTRS, T)
    # RVP and viscosity blend through their indices, so the spec search must
    # average the transformed values, not the raw ones.
    blended_rvp_index = zeros(Float64, P, T)
    blended_vis_index = zeros(Float64, P, T)
    for (k, (s, p)) in enumerate(pairs), c in 1:C, t in 1:T
        b = gamma[s, p] * avail[c, s, t]
        blend_plan[c, k, t] = b
        volume[p, t] += b
        for a in 1:_RP_N_ATTRS
            blended_quality[p, a, t] += quality[c, s, a] * b
        end
        blended_rvp_index[p, t] += quality[c, s, _RP_RVP]^1.25 * b
        blended_vis_index[p, t] += cbrt(quality[c, s, _RP_VIS]) * b
    end
    for p in 1:P, a in 1:_RP_N_ATTRS, t in 1:T
        volume[p, t] > 0 &&
            (blended_quality[p, a, t] /= volume[p, t])
    end
    for p in 1:P, t in 1:T
        volume[p, t] > 0 || continue
        blended_rvp_index[p, t] /= volume[p, t]
        blended_vis_index[p, t] /= volume[p, t]
    end

    # Specifications: tightest industry band the planted recipe clears with
    # margin in every period; missing bands drop the specification.
    gasoline_peak = clamp(round(Int, T * rand(rng, Uniform(0.35, 0.65))), 1, T)
    spec_direction = zeros(Int, P, _RP_N_ATTRS)
    spec_rhs = zeros(Float64, P, _RP_N_ATTRS, T)
    margin = Dict(_RP_OCT => 0.35, _RP_CET => 0.4, _RP_SUL => 0.0,
                  _RP_DEN => 0.004, _RP_VIS => 0.0)
    for (p, product) in enumerate(products)
        for a in 1:_RP_N_ATTRS
            achieved = a == _RP_VIS ?
                       minimum(blended_vis_index[p, :] .^ 3) :
                       minimum(blended_quality[p, a, :] .+ eps())
            if haskey(_RP_SPEC_GE, (product, a)) &&
               (a != _RP_VIS || achieved > 0.0)
                if a == _RP_VIS
                    for tier in reverse(_RP_SPEC_GE[(product, a)])
                        if tier <= achieved * 0.88
                            spec_direction[p, a] = 1
                            spec_rhs[p, a, :] .= tier
                            break
                        end
                    end
                else
                    for tier in reverse(_RP_SPEC_GE[(product, a)])
                        if tier <= achieved - margin[a]
                            spec_direction[p, a] = 1
                            spec_rhs[p, a, :] .= tier
                            break
                        end
                    end
                end
            end
            achieved = a == _RP_VIS ?
                       maximum(blended_vis_index[p, :] .^ 3) :
                       maximum(blended_quality[p, a, :])
            if haskey(_RP_SPEC_LE, (product, a))
                rel = a == _RP_SUL ? 1.18 : a == _RP_VIS ? 1.15 : 1.0
                add = a == _RP_SUL ? 1.0e-4 : margin[a]
                for tier in _RP_SPEC_LE[(product, a)]
                    if tier >= achieved * rel + add
                        spec_direction[p, a] = -1
                        spec_rhs[p, a, :] .= tier
                        break
                    end
                end
            end
        end
        if product in (:RGL, :PGL)
            # Seasonal RVP window (chevron-index blending handled in
            # build_model): summer floor at 9 psi or the planted pool plus
            # margin, winter 3-4 psi looser; dropped if the pool is too
            # volatile for any credible summer cap.
            pool_rvp = maximum(blended_rvp_index[p, t]^(1 / 1.25)
                               for t in 1:T)
            summer = max(9.0, pool_rvp + 0.35)
            if summer <= 10.8
                winter = summer + rand(rng, Uniform(2.8, 4.2))
                spec_direction[p, _RP_RVP] = -1
                for t in 1:T
                    warm = clamp(cos(2π * (t - gasoline_peak) / T), 0.0, 1.0)
                    spec_rhs[p, _RP_RVP, t] = summer + (winter - summer) * (1 - warm)
                end
            end
        end
    end

    # Sales plan and tankage: seasonal demand around production, inventory
    # trajectory bounded by the planted initial stock and tank size.
    delta = _rp_demand_deviation(rng, products, T, gasoline_peak)
    sales_plan = volume .* delta
    initial_product_inventory = zeros(Float64, P)
    product_tank = zeros(Float64, P)
    for p in 1:P
        cum = cumsum(volume[p, :] .- sales_plan[p, :])
        mean_volume = sum(volume[p, :]) / T
        initial_product_inventory[p] =
            -min(0.0, minimum(cum)) + 0.08 * mean_volume
        product_tank[p] = max(
            initial_product_inventory[p] + max(0.0, maximum(cum)) +
            0.12 * mean_volume,
            1.1 * initial_product_inventory[p],
        )
    end

    term_fraction = Float64[p == product_index[:PGAS] ? 0.0 :
                           products[p] in (:LNAPH, :COKE) ?
                           rand(rng, Uniform(0.3, 0.8)) :
                           rand(rng, Uniform(0.5, 0.95)) for p in 1:P]
    sales_floor = term_fraction .* sales_plan
    spot_window = rand(rng, Uniform(0.10, 0.50), P)
    sales_ceiling = sales_plan .* (1.0 .+ spot_window)

    product_price = zeros(Float64, P, T)
    for (p, product) in enumerate(products), t in 1:T
        base = _RP_PRICE_TABLE[product] * rand(rng, LogNormal(0, 0.06))
        product_price[p, t] =
            base * (1 + 0.4 * (delta[p, t] - 1)) * rand(rng, Uniform(0.98, 1.02))
    end

    term_crude = [c == 1 ? true : rand(rng) < 0.6 for c in 1:C]
    purchase_floor = zeros(Float64, C, T)
    purchase_ceiling = zeros(Float64, C, T)
    for c in 1:C, t in 1:T
        purchase_floor[c, t] =
            term_crude[c] ? crude_feed[c, t] * rand(rng, Uniform(0.55, 0.80)) : 0.0
        purchase_ceiling[c, t] = crude_feed[c, t] * rand(rng, Uniform(1.15, 1.50))
    end
    initial_crude_inventory = [mix_weights[c] * run_level *
                               rand(rng, Uniform(0.10, 0.22)) for c in 1:C]
    crude_tank_capacity = sum(initial_crude_inventory) * rand(rng, Uniform(1.15, 1.35))
    cdu_capacity = maximum(run) * rand(rng, Uniform(1.05, 1.15))

    purchase_plan = copy(crude_feed)
    crude_inventory_plan = repeat(initial_crude_inventory, 1, T)
    product_inventory_plan = zeros(Float64, P, T)
    previous = copy(initial_product_inventory)
    for p in 1:P, t in 1:T
        product_inventory_plan[p, t] = previous[p] + volume[p, t] - sales_plan[p, t]
        previous[p] = product_inventory_plan[p, t]
    end

    feasible_witness = nothing
    infeasibility_certificate = nothing
    market_scenario = nothing

    if feasibility_status == feasible
        feasible_witness = RefineryPlanWitness(
            purchase_plan, crude_inventory_plan, crude_feed,
            zeros(Float64, C, n_swing, T), mode_feed_plan, blend_plan,
            sales_plan, product_inventory_plan,
        )
    elseif feasibility_status == infeasible
        ypath = _rp_yield_path(cut_yield, unit_feed, mode_unit, mode_yields,
                               pairs, P)
        horizon = clamp(round(Int, T * rand(rng, Uniform(0.55, 1.0))), 2, T)
        # Sum the per-period ceilings over the horizon window: they vary with
        # the crude-run profile, so `horizon * ceiling[c, 1]` is not a valid
        # implication of the purchase bounds when a later period runs harder
        # than the first.
        raw_purchase_bound = sum(initial_crude_inventory[c] +
                                 sum(purchase_ceiling[c, 1:horizon])
                                 for c in 1:C)
        raw_crude_bound = min(horizon * cdu_capacity, raw_purchase_bound)
        # Cut the product whose term demand is largest relative to the crude
        # its yield path can possibly convert - that is where the aggregated
        # supply bound is tight enough for the certificate to bite.
        tightness = [sum(sales_floor[p, 1:horizon]) /
                     max(maximum(ypath[:, p]) * raw_crude_bound, eps())
                     for p in 1:P]
        ranked = sortperm(tightness; rev = true)
        cut_product = ranked[rand(rng, 1:min(3, P))]
        initial_product_inventory[cut_product] *= rand(rng, Uniform(0.05, 0.20))
        demand_raise = rand(rng, Uniform(1.05, 1.25))
        sales_floor[cut_product, 1:horizon] .*= demand_raise
        sales_floor[cut_product, 1:horizon] .=
            min.(sales_floor[cut_product, 1:horizon],
                 0.98 .* sales_ceiling[cut_product, 1:horizon])
        demand_cum = sum(sales_floor[cut_product, 1:horizon])
        desired_upper = demand_cum * rand(rng, Uniform(0.60, 0.85))
        # Cap the planted initial stock at a fraction of the shrunken demand
        # target: if it exceeded `desired_upper` the scale numerator would go
        # negative, `scale` would clamp at its floor, and the margin below
        # could turn negative.
        initial_product_inventory[cut_product] =
            min(initial_product_inventory[cut_product], 0.4 * desired_upper)
        # Swing cuts can move at most sum_k band_k * yield(heavy_k) <= 0.045
        # barrels per barrel of crude into neighbouring cuts, so the yield
        # path bound is inflated by that provable swing allowance.
        yield_bound = maximum(ypath[:, cut_product]) + 0.05
        scale = clamp((desired_upper - initial_product_inventory[cut_product]) /
                      max(yield_bound * raw_crude_bound, eps()),
                      1.0e-3, 0.95)
        purchase_ceiling .*= scale
        purchase_floor .*= min(1.0, scale)
        purchase_ceiling .= max.(purchase_ceiling, purchase_floor)
        cdu_capacity *= scale

        purchase_bound = sum(initial_crude_inventory[c] +
                             sum(purchase_ceiling[c, 1:horizon]) for c in 1:C)
        crude_bound = min(horizon * cdu_capacity, purchase_bound)
        upper_bound = initial_product_inventory[cut_product] +
                      yield_bound * crude_bound
        certificate_margin = demand_cum - upper_bound
        @assert certificate_margin > 1e-6
        infeasibility_certificate = RefinerySupplyCertificate(
            cut_product, horizon, demand_cum,
            initial_product_inventory[cut_product], horizon * cdu_capacity,
            purchase_bound, crude_bound, yield_bound, upper_bound,
            certificate_margin,
        )
    else
        position = _pp_seed_position(seed)
        supply_factor = 0.42 + 0.53 * position
        demand_factor = 1 + 0.27 * (0.5 - position)
        # Scale only the crude supply side. Unit capacities stay untouched:
        # with equality stream balances, a starved processing chain plus term
        # purchase floors would confound the intended demand-versus-supply
        # question with a different, structural infeasibility.
        purchase_ceiling .*= supply_factor
        purchase_ceiling .= max.(purchase_ceiling, purchase_floor)
        purchase_floor .= min.(purchase_floor, purchase_ceiling)
        cdu_capacity *= supply_factor
        sales_floor .*= demand_factor
        sales_ceiling .= max.(sales_ceiling, sales_floor)
        market_scenario = RefineryMarketScenario(
            supply_factor, demand_factor, position)
    end

    return RefineryPlanningProblem(
        configuration, T, C, stream_names, cut_yield, quality, swing_pairs,
        swing_lo, swing_hi, pairs, unit_names, unit_feed, mode_unit, mode_yields,
        mode_cost, unit_capacity, products, product_price, sales_floor,
        sales_ceiling, product_tank, initial_product_inventory,
        spec_direction, spec_rhs, crude_price, purchase_floor,
        purchase_ceiling, initial_crude_inventory, crude_tank_capacity,
        cdu_capacity, rand(rng, Uniform(0.4, 0.9)),
        rand(rng, Uniform(0.5, 1.3), P), feasible_witness,
        infeasibility_certificate, market_scenario, feasibility_status,
    )
end

function build_model(prob::RefineryPlanningProblem)
    model = Model()
    C, T, P = prob.n_crudes, prob.n_periods, length(prob.product_names)
    S = length(prob.stream_names)
    M = length(prob.mode_unit)
    K = length(prob.swing_pairs)
    blend_pairs = prob.blend_pairs
    B = length(blend_pairs)

    @variable(model, prob.purchase_floor[c, t] <=
              purchase[c = 1:C, t = 1:T] <= prob.purchase_ceiling[c, t])
    @variable(model, crude_inventory[c = 1:C, t = 1:T] >= 0)
    @variable(model, crude_feed[c = 1:C, t = 1:T] >= 0)
    @variable(model, prob.swing_lo[c, k] <=
              swing[c = 1:C, k = 1:K, t = 1:T] <= prob.swing_hi[c, k])
    @variable(model, mode_feed[c = 1:C, m = 1:M, t = 1:T] >= 0)
    @variable(model, blend[c = 1:C, k = 1:B, t = 1:T] >= 0)
    @variable(model, prob.sales_floor[p, t] <=
              sales[p = 1:P, t = 1:T] <= prob.sales_ceiling[p, t])
    @variable(model, 0 <= product_inventory[p = 1:P, t = 1:T] <=
              prob.product_tank[p])

    @objective(model, Max,
        sum(prob.product_price[p, t] * sales[p, t] for p in 1:P, t in 1:T) -
        sum(prob.crude_price[c] * purchase[c, t] for c in 1:C, t in 1:T) -
        sum(prob.mode_cost[m] * mode_feed[c, m, t]
            for c in 1:C, m in 1:M, t in 1:T) -
        prob.crude_carrying_cost *
        sum(crude_inventory[c, t] for c in 1:C, t in 1:T) -
        sum(prob.product_carrying_cost[p] * product_inventory[p, t]
            for p in 1:P, t in 1:T))

    for c in 1:C, t in 1:T
        @constraint(model,
            crude_inventory[c, t] ==
            (t == 1 ? prob.initial_crude_inventory[c] : crude_inventory[c, t - 1]) +
            purchase[c, t] - crude_feed[c, t])
    end
    for t in 1:T
        @constraint(model, sum(crude_inventory[:, t]) <= prob.crude_tank_capacity)
        @constraint(model, sum(crude_feed[:, t]) <= prob.cdu_capacity)
    end

    # Stream producers and consumers, precomputed for the balance rows.
    blend_k_of_stream = Dict{Int,Vector{Int}}()
    for (k, (s, _)) in enumerate(blend_pairs)
        push!(get!(blend_k_of_stream, s, Int[]), k)
    end
    blend_ks_of_product = Dict{Int,Vector{Int}}()
    for (k, (_, p)) in enumerate(blend_pairs)
        push!(get!(blend_ks_of_product, p, Int[]), k)
    end
    mode_consumers_of_stream = Dict{Int,Vector{Int}}()
    for m in 1:M
        push!(get!(mode_consumers_of_stream, prob.unit_feed[prob.mode_unit[m]],
                   Int[]), m)
    end
    mode_producers_of_stream = Dict{Int,Vector{Tuple{Int,Float64}}}()
    for m in 1:M, (out, y) in prob.mode_yields[m]
        push!(get!(mode_producers_of_stream, out, Tuple{Int,Float64}[]), (m, y))
    end
    swing_into = Dict{Int,Int}()   # cut -> swing index gaining at its own boundary
    swing_from = Dict{Int,Int}()   # cut -> swing index losing at its upper boundary
    for (k, (light, heavy)) in enumerate(prob.swing_pairs)
        swing_into[light] = k
        swing_from[heavy] = k
    end

    for c in 1:C, s in 1:S, t in 1:T
        production = AffExpr(0.0)
        if s <= 6
            add_to_expression!(production, prob.cut_yield[c, s], crude_feed[c, t])
            haskey(swing_into, s) &&
                add_to_expression!(production, 1.0, swing[c, swing_into[s], t])
            haskey(swing_from, s) &&
                add_to_expression!(production, -1.0, swing[c, swing_from[s], t])
        end
        for (m, y) in get(mode_producers_of_stream, s, Tuple{Int,Float64}[])
            add_to_expression!(production, y, mode_feed[c, m, t])
        end
        consumption = AffExpr(0.0)
        for k in get(blend_k_of_stream, s, Int[])
            add_to_expression!(consumption, 1.0, blend[c, k, t])
        end
        for m in get(mode_consumers_of_stream, s, Int[])
            add_to_expression!(consumption, 1.0, mode_feed[c, m, t])
        end
        @constraint(model, production == consumption)
    end

    for u in 1:length(prob.unit_names), t in 1:T
        @constraint(model,
            sum(mode_feed[c, m, t]
                for c in 1:C, m in 1:M if prob.mode_unit[m] == u) <=
            prob.unit_capacity[u, t])
    end

    for p in 1:P, t in 1:T
        @constraint(model,
            product_inventory[p, t] ==
            (t == 1 ? prob.initial_product_inventory[p] :
             product_inventory[p, t - 1]) +
            sum(blend[c, k, t]
                for k in get(blend_ks_of_product, p, Int[]), c in 1:C) -
            sales[p, t])
    end

    # Quality specifications. RVP and viscosity enter through their linear
    # blending indices (Chevron RVP^1.25 and Walter cSt^(1/3)); the index
    # columns are transformed once, not per coefficient access.
    quality_index = copy(prob.quality)
    quality_index[:, :, _RP_RVP] .= prob.quality[:, :, _RP_RVP] .^ 1.25
    quality_index[:, :, _RP_VIS] .= cbrt.(prob.quality[:, :, _RP_VIS])
    for p in 1:P, a in 1:_RP_N_ATTRS
        direction = prob.spec_direction[p, a]
        direction == 0 && continue
        for t in 1:T
            raw_rhs = prob.spec_rhs[p, a, t]
            rhs = a == _RP_RVP ? raw_rhs^1.25 : a == _RP_VIS ? cbrt(raw_rhs) :
                  raw_rhs
            excess = AffExpr(0.0)
            for k in get(blend_ks_of_product, p, Int[])
                s = blend_pairs[k][1]
                for c in 1:C
                    add_to_expression!(excess,
                                       quality_index[c, s, a] - rhs,
                                       blend[c, k, t])
                end
            end
            if direction == 1
                @constraint(model, excess >= 0)
            else
                @constraint(model, excess <= 0)
            end
        end
    end
    return model
end

register_variant(
    :process_planning,
    :refinery,
    RefineryPlanningProblem,
    "Multi-period refinery planning LP with crude assays and contracts, CDU swing cuts, multi-mode conversion units, assay-origin product blending under industry quality specifications, seasonal demand, and tank inventories",
    default = true,
)
