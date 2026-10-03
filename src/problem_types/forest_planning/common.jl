using JuMP
using Random
using Distributions

# =============================================================================
# Shared machinery for the forest_planning category: species / region
# catalogs, Chapman-Richards yield model, landscape (watershed + stratum)
# sampling, the column store shared by Model I and Model II, the planted
# area-control witness, the Lagrangian (dynamic-programming) infeasibility
# certificate, requirement calibration, and the common `build_model`.
# =============================================================================

"""
Largest `target_variables` accepted by the forest-planning generators. The
column store keeps every prescription's harvest-volume entries, NPV, ending
inventory and lookup keys at once (roughly 150 bytes per column), and the
witness bisection plus the certificate's multiplier search each sweep the whole
store tens of times, so larger targets are rejected with an `ArgumentError`
instead of being silently undersized (same convention as
`supply_chain/network_planning`).
"""
const FOREST_PLANNING_MAX_VARIABLES = 1_000_000

"Product (timber assortment) catalog, in the order product indices refer to."
const FOREST_PRODUCTS = (:softwood_sawlog, :hardwood_sawlog, :pulpwood)

"Multipliers on (max volume, growth rate) for site classes 1 (best) to 3 (poorest)."
const FOREST_SITE_VOLUME = (1.25, 1.0, 0.75)
const FOREST_SITE_GROWTH = (1.10, 1.0, 0.90)

"Fraction of non-sawlog stem volume recovered as pulpwood (top and defect loss)."
const FOREST_PULP_UTILIZATION = 0.85

"""
    ForestTypeSpec

Catalog entry for one forest type (species group). Yield follows a
Chapman-Richards curve `V(a) = max_volume * (1 - exp(-growth_rate * a))^shape`
(m³/ha at stand age `a` years, site class 2); the sawlog share of harvested
volume rises logistically with age, `saw_max / (1 + exp(-(a - saw_age50) / saw_width))`.
`thin_window = (lo, hi)` is the commercial-thinning age window (`(0, 0)` when
the type is not thinned); `natural_lag` is the regeneration delay (years) of
natural regeneration; `price_premium` scales the regional stumpage prices.
"""
struct ForestTypeSpec
    name::Symbol
    softwood::Bool
    max_volume::Float64
    growth_rate::Float64
    shape::Float64
    min_harvest_age::Float64
    thin_window::Tuple{Float64, Float64}
    saw_max::Float64
    saw_age50::Float64
    saw_width::Float64
    natural_lag::Float64
    price_premium::Float64
end

const FOREST_TYPE_CATALOG = Dict{Symbol, ForestTypeSpec}(
    # Pacific Northwest
    :douglas_fir => ForestTypeSpec(:douglas_fir, true, 1100.0, 0.030, 2.8, 35.0, (20.0, 30.0), 0.85, 35.0, 6.0, 8.0, 1.15),
    :western_hemlock => ForestTypeSpec(:western_hemlock, true, 1000.0, 0.028, 2.6, 40.0, (20.0, 30.0), 0.80, 40.0, 7.0, 5.0, 0.85),
    :red_alder => ForestTypeSpec(:red_alder, false, 550.0, 0.060, 2.0, 25.0, (0.0, 0.0), 0.50, 30.0, 5.0, 3.0, 0.90),
    # Southeast
    :loblolly_pine => ForestTypeSpec(:loblolly_pine, true, 500.0, 0.070, 2.5, 15.0, (12.0, 18.0), 0.70, 25.0, 4.0, 4.0, 1.05),
    :slash_pine => ForestTypeSpec(:slash_pine, true, 430.0, 0.065, 2.4, 15.0, (12.0, 18.0), 0.65, 26.0, 4.0, 4.0, 0.95),
    :upland_hardwood => ForestTypeSpec(:upland_hardwood, false, 330.0, 0.030, 2.2, 40.0, (0.0, 0.0), 0.55, 50.0, 8.0, 3.0, 1.00),
    # Lake States
    :aspen => ForestTypeSpec(:aspen, false, 300.0, 0.060, 2.3, 30.0, (0.0, 0.0), 0.25, 40.0, 6.0, 1.0, 0.70),
    :northern_hardwood => ForestTypeSpec(:northern_hardwood, false, 350.0, 0.025, 2.0, 50.0, (0.0, 0.0), 0.60, 60.0, 9.0, 6.0, 1.20),
    :red_pine => ForestTypeSpec(:red_pine, true, 600.0, 0.035, 2.5, 35.0, (25.0, 40.0), 0.80, 40.0, 6.0, 8.0, 1.00),
    :spruce_fir => ForestTypeSpec(:spruce_fir, true, 380.0, 0.030, 2.6, 45.0, (0.0, 0.0), 0.55, 55.0, 8.0, 6.0, 0.90),
    # Interior West
    :ponderosa_pine => ForestTypeSpec(:ponderosa_pine, true, 420.0, 0.022, 2.3, 50.0, (30.0, 45.0), 0.85, 50.0, 8.0, 10.0, 1.10),
    :lodgepole_pine => ForestTypeSpec(:lodgepole_pine, true, 380.0, 0.030, 2.5, 50.0, (0.0, 0.0), 0.60, 55.0, 8.0, 5.0, 0.85),
    :mixed_conifer => ForestTypeSpec(:mixed_conifer, true, 650.0, 0.025, 2.5, 50.0, (30.0, 50.0), 0.75, 50.0, 8.0, 8.0, 1.00),
)

"""
    ForestRegionSpec

A regional forest-economy profile: period length (years), the forest types and
their landscape weights, the plantation type that hardwood / low-value types
may be converted to, and stumpage-price and regeneration-cost ranges
(US dollars per m³ and per ha).
"""
struct ForestRegionSpec
    name::Symbol
    period_length::Int
    types::Vector{Symbol}
    weights::Vector{Float64}
    plantation_type::Symbol
    convertible::Vector{Symbol}
    softwood_saw_price::Tuple{Float64, Float64}
    hardwood_saw_price::Tuple{Float64, Float64}
    pulp_price::Tuple{Float64, Float64}
    natural_regen_cost::Tuple{Float64, Float64}
    plant_cost::Tuple{Float64, Float64}
end

const FOREST_REGIONS = (
    ForestRegionSpec(:pacific_northwest, 10, [:douglas_fir, :western_hemlock, :red_alder], [0.55, 0.30, 0.15],
        :douglas_fir, [:red_alder], (60.0, 85.0), (45.0, 70.0), (8.0, 15.0), (150.0, 300.0), (700.0, 1200.0)),
    ForestRegionSpec(:southeast, 5, [:loblolly_pine, :slash_pine, :upland_hardwood], [0.50, 0.25, 0.25],
        :loblolly_pine, [:upland_hardwood], (25.0, 40.0), (25.0, 45.0), (8.0, 14.0), (100.0, 250.0), (450.0, 800.0)),
    ForestRegionSpec(:lake_states, 10, [:aspen, :northern_hardwood, :red_pine, :spruce_fir], [0.35, 0.30, 0.20, 0.15],
        :red_pine, [:aspen], (30.0, 50.0), (40.0, 80.0), (10.0, 20.0), (80.0, 200.0), (500.0, 900.0)),
    ForestRegionSpec(:interior_west, 10, [:ponderosa_pine, :lodgepole_pine, :mixed_conifer], [0.40, 0.30, 0.30],
        :ponderosa_pine, [:lodgepole_pine], (25.0, 45.0), (20.0, 35.0), (3.0, 10.0), (120.0, 250.0), (600.0, 1100.0)),
)

"""
    ForestStandModel

An instance-level growth model: a forest type on a site class under a
regeneration regime (`:existing` for the initial natural stands, `:natural`
for naturally regenerated stands, `:planted` for planted improved stock).
`max_volume` and `growth_rate` already include the site multipliers and, for
`:planted`, the genetic gain; `lag` is the regeneration delay in years (only
`:natural` has one). Yield at stand age `a` is
`max_volume * (1 - exp(-growth_rate * (a - lag)))^shape` for `a > lag`, else 0.
"""
struct ForestStandModel
    type_index::Int
    site_class::Int
    regime::Symbol
    max_volume::Float64
    growth_rate::Float64
    shape::Float64
    lag::Float64
end

"""
    ForestPlanningWitness

Planted feasible schedule (area-control heuristic): `areas[j]` is the hectares
assigned to column `j`; `harvest[t, k]` the product-`k` volume (m³) it cuts in
period `t`; every period cuts exactly `flat_volume` m³ in total (a perfectly
even flow); `ending_inventory` is its standing volume (m³) at the end of the
horizon. Every requirement of a `feasible` instance is derived from this
schedule with a margin, so it satisfies every row of the built model.
"""
struct ForestPlanningWitness
    areas::Vector{Float64}
    harvest::Matrix{Float64}
    flat_volume::Float64
    ending_inventory::Float64
end

"""
    ForestPlanningCertificate

Lagrangian (Farkas) infeasibility certificate built from LP rows alone. With
inventory weight `μ = inventory_weight >= 0`, `source_values` holds one value
`π` per area-accounting row (strata first, then Model II regeneration nodes)
satisfying, for every column `j` from source `src(j)` (into node `dst(j)` or
nowhere),

    π[src(j)] >= CH[j] + μ * EI[j] + π[dst(j)]   (π[nothing] = 0),

where `CH[j]` is the column's total harvest volume over the horizon (the sum of
its harvest-definition coefficients) and `EI[j]` its ending-inventory
coefficient. Multiplying the stratum-area equalities by `π`, the node-balance
equalities by `π`, summing the harvest-definition rows and the `T*K` supply
lower bounds, and adding `μ` times the ending-inventory row shows every
feasible point satisfies

    bound = Σ_s area[s] * π[s] >= T * Σ_k min_supply[k] + μ * min_ending_inventory = required,

but the instance has `bound < required`. The values are the exact
dynamic-programming (longest path) values, so `bound` is the tightest bound for
this `μ`. Even-flow, green-up and mill-capacity rows only shrink the feasible
set and are not used.
"""
struct ForestPlanningCertificate
    inventory_weight::Float64
    source_values::Vector{Float64}
    bound::Float64
    required::Float64
end

"""
    ForestPlanningProblem{F} <: ProblemGenerator

Forest harvest-scheduling LP over a multi-watershed landscape of analysis areas
(strata). `F` is the formulation: `:model_i` (Johnson & Scheurman Model I —
one column per whole-horizon prescription of a stratum) or `:model_ii` (Model
II — clearcut-and-regenerate area flows through regeneration nodes, a network
with side constraints). Concrete aliases: [`ForestModelIProblem`](@ref) and
[`ForestModelIIProblem`](@ref); see `docs/forest_planning.md`.

# Columns

Column `j` is an area variable (ha) leaving source `col_source[j]` (a stratum
`1..n_strata`, or regeneration node `n_strata + i`) and, in Model II, entering
node `col_dest[j]` (`0` = none). `col_thin[j]` is the commercial-thinning
period (0 = none), `col_cut1[j]` the first clearcut period (0 = none),
`col_regen1[j]` the regeneration option after it (index into
`regen_targets[model]`), and `col_cut2[j]` Model I's second clearcut (0 =
none). Harvest-volume coefficients are stored sparsely as
`(vol_period, vol_product, vol_amount)` triples in `vol_ptr[j]:vol_ptr[j+1]-1`
(m³/ha); `col_npv[j]` is the discounted net revenue (USD/ha) and
`col_ending_inventory[j]` the standing volume (m³/ha) at the horizon's end.

# Rows

stratum area accounting (equalities), Model II node balance (equalities),
harvest-volume definitions `harvest[t,k] = Σ_j v[j,t,k] area[j]`, two-sided
even flow on total volume with tolerance `even_flow_tolerance`, watershed
green-up limits (area clearcut in any `greenup_window` consecutive periods at
most `greenup_fraction[z] * zone_area[z]`), and an ending-inventory floor.
Mill-supply contracts are bounds `min_supply[k] <= harvest[t,k] <= max_supply[k]`.
"""
struct ForestPlanningProblem{F} <: ProblemGenerator
    region::Symbol
    period_length::Int
    n_periods::Int
    discount_rate::Float64
    products::Vector{Symbol}
    forest_types::Vector{ForestTypeSpec}
    stand_models::Vector{ForestStandModel}
    regen_targets::Vector{Vector{Int}}
    regen_costs::Vector{Vector{Float64}}
    saw_price::Vector{Float64}
    pulp_price::Vector{Float64}
    harvest_cost::Float64
    thinning_cost::Float64
    thinning_fraction::Float64
    thinning_recovery::Float64
    # landscape
    stratum_zone::Vector{Int}
    stratum_model::Vector{Int}
    stratum_age::Vector{Float64}
    stratum_area::Vector{Float64}
    zone_area::Vector{Float64}
    greenup_fraction::Vector{Float64}
    greenup_window::Int
    node_model::Vector{Int}
    node_zone::Vector{Int}
    node_period::Vector{Int}
    # columns
    col_source::Vector{Int32}
    col_dest::Vector{Int32}
    col_thin::Vector{Int16}
    col_cut1::Vector{Int16}
    col_regen1::Vector{Int8}
    col_cut2::Vector{Int16}
    vol_ptr::Vector{Int}
    vol_period::Vector{Int16}
    vol_product::Vector{Int8}
    vol_amount::Vector{Float64}
    col_npv::Vector{Float64}
    col_ending_inventory::Vector{Float64}
    # requirements
    even_flow_tolerance::Float64
    base_min_supply::Vector{Float64}
    base_min_ending_inventory::Float64
    supply_scale::Float64
    inventory_scale::Float64
    lagrangian_scale::Float64
    min_supply::Vector{Float64}
    max_supply::Vector{Float64}
    min_ending_inventory::Float64
    feasible_witness::Union{Nothing, ForestPlanningWitness}
    infeasibility_certificate::Union{Nothing, ForestPlanningCertificate}
    feasibility_status::FeasibilityStatus
end

# -----------------------------------------------------------------------------
# Yield model
# -----------------------------------------------------------------------------

"Standing volume (m³/ha) of stand model `sm` at stand age `age` years."
function _forest_volume(sm::ForestStandModel, age::Float64)
    a = age - sm.lag
    a <= 0.0 && return 0.0
    return sm.max_volume * (1.0 - exp(-sm.growth_rate * a))^sm.shape
end

"Sawlog share of merchantable volume for stand model `sm` (type `spec`) at age `age`."
function _forest_saw_fraction(spec::ForestTypeSpec, sm::ForestStandModel, age::Float64)
    a = age - sm.lag
    return spec.saw_max / (1.0 + exp(-(a - spec.saw_age50) / spec.saw_width))
end

"Effective (post-lag) age test for clearcut eligibility."
_forest_mature(spec::ForestTypeSpec, sm::ForestStandModel, age::Float64) =
    age - sm.lag >= spec.min_harvest_age - 1.0e-9

# -----------------------------------------------------------------------------
# Construction context and column store
# -----------------------------------------------------------------------------

"Mutable construction state shared by both formulations (internal)."
mutable struct _ForestBuilder
    region::ForestRegionSpec
    L::Int
    T::Int
    discount::Vector{Float64}          # per-period discount factor (mid-period)
    discount_rate::Float64
    product_index::Vector{Int}         # FOREST_PRODUCTS index -> instance product index (0 = absent)
    products::Vector{Symbol}
    types::Vector{ForestTypeSpec}
    type_lookup::Dict{Symbol, Int}
    models::Vector{ForestStandModel}
    model_lookup::Dict{Tuple{Int, Int, Symbol}, Int}
    regen_targets::Vector{Vector{Int}}
    regen_costs::Vector{Vector{Float64}}
    planted_gain::Float64
    natural_cost::Vector{Float64}      # per type
    plant_cost::Vector{Float64}        # per type
    conversion_cost::Float64
    saw_price::Vector{Float64}
    pulp_price::Vector{Float64}
    harvest_cost::Float64
    thinning_cost::Float64
    thinning_fraction::Float64
    thinning_recovery::Float64
    # landscape
    stratum_zone::Vector{Int}
    stratum_model::Vector{Int}
    stratum_age::Vector{Float64}
    stratum_area::Vector{Float64}
    zone_area::Vector{Float64}
    greenup_fraction::Vector{Float64}
    greenup_window::Int
    node_model::Vector{Int}
    node_zone::Vector{Int}
    node_period::Vector{Int}
    node_lookup::Dict{Tuple{Int, Int, Int}, Int}
    source_pref::Vector{Int}           # witness regeneration preference per stratum (then per source)
    node_pref::Vector{Int}             # witness regeneration preference per node
    # columns
    col_source::Vector{Int32}
    col_dest::Vector{Int32}
    col_thin::Vector{Int16}
    col_cut1::Vector{Int16}
    col_regen1::Vector{Int8}
    col_cut2::Vector{Int16}
    vol_ptr::Vector{Int}
    vol_period::Vector{Int16}
    vol_product::Vector{Int8}
    vol_amount::Vector{Float64}
    col_npv::Vector{Float64}
    col_ei::Vector{Float64}
    cutvol1::Vector{Float64}           # clearcut volume (m³/ha) at col_cut1
    cutvol2::Vector{Float64}           # clearcut volume (m³/ha) at col_cut2
    col_key::Dict{Int, Int}            # unthinned column lookup (see _forest_key)
    end_col::Vector{Int}               # per source: unthinned no-clearcut column
end

"Pack an unthinned column's `(source, cut1, regen1, cut2)` into a lookup key."
_forest_key(b::_ForestBuilder, src::Int, cut1::Int, r1::Int, cut2::Int) =
    ((src * (b.T + 1) + cut1) * 4 + r1) * (b.T + 1) + cut2

"Return (creating on first use) the stand-model index for `(type, site, regime)`."
function _forest_model!(b::_ForestBuilder, ti::Int, site::Int, regime::Symbol)
    key = (ti, site, regime)
    idx = get(b.model_lookup, key, 0)
    idx > 0 && return idx
    spec = b.types[ti]
    gain = regime == :planted ? b.planted_gain : 1.0
    lag = regime == :natural ? spec.natural_lag : 0.0
    sm = ForestStandModel(
        ti, site, regime, spec.max_volume * FOREST_SITE_VOLUME[site] * gain,
        spec.growth_rate * FOREST_SITE_GROWTH[site], spec.shape, lag,
    )
    push!(b.models, sm)
    idx = length(b.models)
    b.model_lookup[key] = idx
    push!(b.regen_targets, Int[])
    push!(b.regen_costs, Float64[])
    # Regeneration options after a clearcut of this model: natural
    # regeneration, planting improved stock of the same type, and (for
    # convertible types) conversion to the regional plantation type.
    targets = Int[]
    costs = Float64[]
    nat = _forest_model!(b, ti, site, :natural)
    push!(targets, nat)
    push!(costs, b.natural_cost[ti])
    pl = _forest_model!(b, ti, site, :planted)
    push!(targets, pl)
    push!(costs, b.plant_cost[ti])
    if spec.name in b.region.convertible
        pti = b.type_lookup[b.region.plantation_type]
        cv = _forest_model!(b, pti, site, :planted)
        push!(targets, cv)
        push!(costs, b.conversion_cost)
    end
    b.regen_targets[idx] = targets
    b.regen_costs[idx] = costs
    return idx
end

"Same-regime replanting option for Model I's second rotation (index into regen_targets)."
function _forest_same_regime_option(b::_ForestBuilder, m::Int)
    sm = b.models[m]
    for (r, tgt) in enumerate(b.regen_targets[m])
        tm = b.models[tgt]
        if tm.type_index == sm.type_index && tm.regime == (sm.regime == :planted ? :planted : :natural)
            return r
        end
    end
    return 1
end

"Periods of rotation needed before a stand regenerated as model `m` can be clearcut."
function _forest_min_rotation_periods(b::_ForestBuilder, m::Int)
    sm = b.models[m]
    spec = b.types[sm.type_index]
    return max(1, ceil(Int, (spec.min_harvest_age + sm.lag - 1.0e-9) / b.L))
end

"""
Event accumulator for one column: harvest-volume entries, NPV, and the
clearcut volumes used by the witness heuristic.
"""
mutable struct _ForestColumnDraft
    periods::Vector{Int16}
    prods::Vector{Int8}
    amounts::Vector{Float64}
    npv::Float64
    cutvol1::Float64
    cutvol2::Float64
end
_ForestColumnDraft() = _ForestColumnDraft(Int16[], Int8[], Float64[], 0.0, 0.0, 0.0)

function _forest_draft_reset!(d::_ForestColumnDraft)
    empty!(d.periods)
    empty!(d.prods)
    empty!(d.amounts)
    d.npv = 0.0
    d.cutvol1 = 0.0
    d.cutvol2 = 0.0
    return d
end

function _forest_draft_add!(d::_ForestColumnDraft, t::Int, k::Int, amount::Float64)
    amount <= 1.0e-9 && return 0.0
    push!(d.periods, Int16(t))
    push!(d.prods, Int8(k))
    push!(d.amounts, amount)
    return amount
end

"Record a commercial thinning of stand model `m` at period `t`, stand age `age`."
function _forest_thin_event!(d::_ForestColumnDraft, b::_ForestBuilder, m::Int, t::Int, age::Float64)
    sm = b.models[m]
    spec = b.types[sm.type_index]
    removed = b.thinning_fraction * _forest_volume(sm, age)
    saw = 0.3 * _forest_saw_fraction(spec, sm, age) * removed
    pulp = (removed - saw) * FOREST_PULP_UTILIZATION
    ks = spec.softwood ? b.product_index[1] : b.product_index[2]
    kp = b.product_index[3]
    _forest_draft_add!(d, t, ks, saw)
    _forest_draft_add!(d, t, kp, pulp)
    revenue = 0.75 * (saw * b.saw_price[sm.type_index] + pulp * b.pulp_price[sm.type_index]) - b.thinning_cost
    d.npv += b.discount[t] * revenue
    return nothing
end

"""
Standing volume of stand model `m` at age `age`, optionally after a thinning at
age `thin_age` (removal `θ V(thin_age)`, recovering at rate `ρ` per year).
"""
function _forest_standing(b::_ForestBuilder, m::Int, age::Float64, thin_age::Float64)
    sm = b.models[m]
    v = _forest_volume(sm, age)
    if thin_age >= 0.0
        v -= b.thinning_fraction * _forest_volume(sm, thin_age) * exp(-b.thinning_recovery * (age - thin_age))
    end
    return max(v, 0.0)
end

"Record a clearcut of stand model `m` at period `t`, age `age`; returns the total volume (m³/ha)."
function _forest_clearcut_event!(
    d::_ForestColumnDraft, b::_ForestBuilder, m::Int, t::Int, age::Float64, thin_age::Float64
)
    sm = b.models[m]
    spec = b.types[sm.type_index]
    v = _forest_standing(b, m, age, thin_age)
    sigma = _forest_saw_fraction(spec, sm, age)
    thin_age >= 0.0 && (sigma = min(0.95, sigma + 0.10))
    saw = sigma * v
    pulp = (1.0 - sigma) * v * FOREST_PULP_UTILIZATION
    ks = spec.softwood ? b.product_index[1] : b.product_index[2]
    kp = b.product_index[3]
    total = _forest_draft_add!(d, t, ks, saw) + _forest_draft_add!(d, t, kp, pulp)
    revenue = saw * b.saw_price[sm.type_index] + pulp * b.pulp_price[sm.type_index] - b.harvest_cost
    d.npv += b.discount[t] * revenue
    return total
end

"Charge regeneration option `r` of stand model `m` at period `t`; returns the new model."
function _forest_regen_event!(d::_ForestColumnDraft, b::_ForestBuilder, m::Int, r::Int, t::Int)
    d.npv -= b.discount[t] * b.regen_costs[m][r]
    return b.regen_targets[m][r]
end

"""
Model units: the yield model works in m³/ha and USD/ha, but the stored column
data (and hence every harvest, supply, inventory and objective quantity of the
model) is in thousands — 1000 m³ and 1000 USD — which keeps the harvest
accounting bounds and the inventory floor within a few orders of magnitude of
the unit area coefficients (HiGHS warns about 1e6-scale bounds otherwise).
"""
const FOREST_UNIT = 1000.0

"Append a finished draft to the column store (rescaled by `FOREST_UNIT`); returns the column index."
function _forest_push_column!(
    b::_ForestBuilder, d::_ForestColumnDraft, src::Int, dest::Int, thin::Int, cut1::Int, r1::Int,
    cut2::Int, ei::Float64,
)
    push!(b.col_source, Int32(src))
    push!(b.col_dest, Int32(dest))
    push!(b.col_thin, Int16(thin))
    push!(b.col_cut1, Int16(cut1))
    push!(b.col_regen1, Int8(r1))
    push!(b.col_cut2, Int16(cut2))
    append!(b.vol_period, d.periods)
    append!(b.vol_product, d.prods)
    for a in d.amounts
        push!(b.vol_amount, a / FOREST_UNIT)
    end
    push!(b.vol_ptr, length(b.vol_amount) + 1)
    push!(b.col_npv, d.npv / FOREST_UNIT)
    push!(b.col_ei, ei / FOREST_UNIT)
    push!(b.cutvol1, d.cutvol1 / FOREST_UNIT)
    push!(b.cutvol2, d.cutvol2 / FOREST_UNIT)
    return length(b.col_source)
end

"""
Internal source id offset for Model II regeneration nodes during construction.
Strata keep ids `1..n_strata`; node `i` is `FOREST_NODE_OFFSET + i` until
`_forest_remap_sources!` renumbers it to `n_strata + i` once the stratum count
is final (strata keep streaming in after the first nodes appear).
"""
const FOREST_NODE_OFFSET = 1 << 30

"Renumber node sources to `n_strata + i` and build the unthinned-column lookups."
function _forest_remap_sources!(b::_ForestBuilder)
    S = length(b.stratum_area)
    @inbounds for j in eachindex(b.col_source)
        src = Int(b.col_source[j])
        src > FOREST_NODE_OFFSET && (b.col_source[j] = Int32(S + src - FOREST_NODE_OFFSET))
    end
    b.source_pref = vcat(b.source_pref, b.node_pref)
    b.end_col = zeros(Int, S + length(b.node_period))
    empty!(b.col_key)
    sizehint!(b.col_key, length(b.col_source))
    @inbounds for j in eachindex(b.col_source)
        b.col_thin[j] == 0 || continue
        src = Int(b.col_source[j])
        cut1 = Int(b.col_cut1[j])
        b.col_key[_forest_key(b, src, cut1, Int(b.col_regen1[j]), Int(b.col_cut2[j]))] = j
        cut1 == 0 && (b.end_col[src] = j)
    end
    return b
end

# -----------------------------------------------------------------------------
# Instance-level sampling
# -----------------------------------------------------------------------------

"Sample the period count from the target size and the region's period length."
function _forest_n_periods(rng::AbstractRNG, target::Int, L::Int)
    if L == 5
        target <= 300 && return rand(rng, 6:8)
        target <= 3000 && return rand(rng, 8:12)
        return rand(rng, 12:18)
    else
        target <= 300 && return rand(rng, 4:5)
        target <= 3000 && return rand(rng, 6:8)
        return rand(rng, 8:12)
    end
end

_forest_uniform(rng::AbstractRNG, r::Tuple{Float64, Float64}) = r[1] + (r[2] - r[1]) * rand(rng)

"Draw the region, horizon, prices, costs and silvicultural parameters."
function _forest_builder(rng::AbstractRNG, target::Int)
    region = FOREST_REGIONS[rand(rng, 1:length(FOREST_REGIONS))]
    L = region.period_length
    T = _forest_n_periods(rng, target, L)
    r = 0.03 + 0.03 * rand(rng)
    discount = [(1.0 + r)^(-(t - 0.5) * L) for t in 1:T]

    types = [FOREST_TYPE_CATALOG[name] for name in region.types]
    type_lookup = Dict(spec.name => i for (i, spec) in enumerate(types))
    has_sw = any(s.softwood for s in types)
    has_hw = any(!s.softwood for s in types)
    products = Symbol[]
    product_index = zeros(Int, 3)
    for (pi, present) in enumerate((has_sw, has_hw, true))
        present || continue
        push!(products, FOREST_PRODUCTS[pi])
        product_index[pi] = length(products)
    end

    sw_saw = _forest_uniform(rng, region.softwood_saw_price)
    hw_saw = _forest_uniform(rng, region.hardwood_saw_price)
    pulp = _forest_uniform(rng, region.pulp_price)
    saw_price = [(s.softwood ? sw_saw : hw_saw) * s.price_premium * (0.92 + 0.16 * rand(rng)) for s in types]
    pulp_price = [pulp * (0.9 + 0.2 * rand(rng)) for _ in types]
    nat_cost = [_forest_uniform(rng, region.natural_regen_cost) for _ in types]
    plant_cost = [_forest_uniform(rng, region.plant_cost) for _ in types]
    conversion_cost = 1.25 * _forest_uniform(rng, region.plant_cost)

    return _ForestBuilder(
        region, L, T, discount, r, product_index, products, types, type_lookup,
        ForestStandModel[], Dict{Tuple{Int, Int, Symbol}, Int}(), Vector{Int}[], Vector{Float64}[],
        1.10 + 0.15 * rand(rng), nat_cost, plant_cost, conversion_cost, saw_price, pulp_price,
        80.0 + 120.0 * rand(rng),                 # clearcut fixed cost $/ha (roads, layout, admin)
        60.0 + 60.0 * rand(rng),                  # thinning fixed cost $/ha
        0.25 + 0.10 * rand(rng),                  # thinning removal fraction
        0.02 + 0.02 * rand(rng),                  # thinning recovery rate (1/yr)
        Int[], Int[], Float64[], Float64[], Float64[], Float64[],
        max(1, ceil(Int, 20 / L)),                # 20-year green-up / hydrologic recovery window
        Int[], Int[], Int[], Dict{Tuple{Int, Int, Int}, Int}(), Int[], Int[],
        Int32[], Int32[], Int16[], Int16[], Int8[], Int16[], [1], Int16[], Int8[], Float64[],
        Float64[], Float64[], Float64[], Float64[], Dict{Int, Int}(), Int[],
    )
end

"""
    _forest_sample_zone(rng, b, age_profile) -> (greenup_fraction, candidates)

Sample one watershed (management zone): its forest-type mix, site tendency,
age history, green-up limit, and 5-14 candidate strata
`(type index, site class, age class, area)` with unique
`(type, site class, age class)` combinations (duplicate draws merge their
areas, as analysis areas do in real inventories). The first candidate is
mature at period 1, so every watershed carries harvestable volume from the
start; every candidate can be clearcut at least once within the horizon
(otherwise it would carry a single, presolve-fixed column).
"""
function _forest_sample_zone(rng::AbstractRNG, b::_ForestBuilder, age_profile::NTuple{3, Float64})
    region = b.region
    greenup = 0.25 + 0.15 * rand(rng)
    w = region.weights .* exp.(0.8 .* randn(rng, length(region.weights)))
    w ./= sum(w)
    tilt = randn(rng)                        # valley (+) vs ridge (-) watershed
    site_w = [exp(0.7 * tilt), 1.6, exp(-0.7 * tilt)]
    site_w ./= sum(site_w)
    age_shift = 0.25 * randn(rng)            # watershed disturbance history
    n_strata = rand(rng, 5:14)

    cands = Tuple{Int, Int, Int, Float64}[]
    seen = Dict{Tuple{Int, Int, Int}, Int}()
    L = b.L
    T = b.T
    for i in 1:n_strata
        ti = rand(rng, Categorical(w))
        spec = b.types[ti]
        site = rand(rng, Categorical(site_w))
        rot = 1.3 * spec.min_harvest_age
        u = rand(rng)
        age = if u < age_profile[1]                  # mature legacy stands
            rot * (0.6 + 1.9 * rand(rng))
        elseif u < age_profile[1] + age_profile[2]   # plantation-era bulge
            rot * (0.6 + 0.25 * randn(rng))
        else                                         # balanced
            rot * 1.8 * rand(rng)
        end
        age *= 1.0 + age_shift
        if i == 1
            age = spec.min_harvest_age + L * (rand(rng) * 3.0)
        end
        cls = max(1, round(Int, age / L))
        if i == 1
            cls = max(cls, ceil(Int, (spec.min_harvest_age - 0.5 * L) / L))
        end
        cls = max(cls, ceil(Int, (spec.min_harvest_age - (T - 0.5) * L) / L))
        area = clamp(exp(log(45.0) + 0.7 * randn(rng)), 3.0, 600.0)
        key = (ti, site, cls)
        if haskey(seen, key)
            c = cands[seen[key]]
            cands[seen[key]] = (c[1], c[2], c[3], c[4] + area)
            continue
        end
        push!(cands, (ti, site, cls, area))
        seen[key] = length(cands)
    end
    return greenup, cands
end

"""
Commit a sampled stratum (opening watershed `zone` on first use with
green-up fraction `greenup`); returns the stratum index.
"""
function _forest_commit_stratum!(
    rng::AbstractRNG, b::_ForestBuilder, zone::Int, greenup::Float64, cand::Tuple{Int, Int, Int, Float64}
)
    if zone > length(b.zone_area)
        push!(b.zone_area, 0.0)
        push!(b.greenup_fraction, greenup)
    end
    ti, site, cls, area = cand
    m = _forest_model!(b, ti, site, :existing)
    push!(b.stratum_zone, zone)
    push!(b.stratum_model, m)
    push!(b.stratum_age, Float64(cls * b.L))
    push!(b.stratum_area, area)
    b.zone_area[zone] += area
    push!(b.source_pref, rand(rng, 1:length(b.regen_targets[m])))
    return length(b.stratum_area)
end

"Draw the instance's age-structure mixture (mature legacy, plantation bulge, balanced)."
function _forest_age_profile(rng::AbstractRNG)
    w = rand(rng, Dirichlet([1.5, 1.5, 1.5]))
    return (w[1], w[2], w[3])
end

# -----------------------------------------------------------------------------
# Lagrangian / dynamic-programming bound
# -----------------------------------------------------------------------------

"""
Column grouping by source plus a reverse-topological source order (Model II
nodes by decreasing period, then strata), used by the DP bound.
"""
struct _ForestSourceIndex
    ptr::Vector{Int}
    cols::Vector{Int}
    order::Vector{Int}
end

function _forest_source_index(col_source, n_strata::Int, node_period::Vector{Int})
    n_src = n_strata + length(node_period)
    counts = zeros(Int, n_src + 1)
    for s in col_source
        counts[s + 1] += 1
    end
    ptr = cumsum(vcat(1, counts[2:end]))
    fill_pos = copy(ptr)
    cols = Vector{Int}(undef, length(col_source))
    for (j, s) in enumerate(col_source)
        cols[fill_pos[s]] = j
        fill_pos[s] += 1
    end
    node_order = sortperm(node_period; rev=true)
    order = vcat(n_strata .+ node_order, collect(1:n_strata))
    return _ForestSourceIndex(ptr, cols, order)
end

"Total harvest volume (m³/ha) over the horizon of every column."
function _forest_column_harvest_totals(vol_ptr::Vector{Int}, vol_amount::Vector{Float64})
    n = length(vol_ptr) - 1
    ch = zeros(n)
    for j in 1:n
        acc = 0.0
        for e in vol_ptr[j]:(vol_ptr[j + 1] - 1)
            acc += vol_amount[e]
        end
        ch[j] = acc
    end
    return ch
end

"""
DP values `π` for inventory weight `μ`: `π[src] = max_j (CH[j] + μ EI[j] + π[dst(j)])`
over the columns of `src`. Returns `(π, bound = Σ_s area[s] π[s])`.
"""
function _forest_dp_values(
    idx::_ForestSourceIndex, col_dest, n_strata::Int, ch::Vector{Float64}, ei::Vector{Float64},
    areas::Vector{Float64}, mu::Float64,
)
    pi_vals = fill(-Inf, length(idx.ptr) - 1)
    for src in idx.order
        best = -Inf
        for p in idx.ptr[src]:(idx.ptr[src + 1] - 1)
            j = idx.cols[p]
            d = col_dest[j]
            v = ch[j] + mu * ei[j] + (d > 0 ? pi_vals[n_strata + d] : 0.0)
            v > best && (best = v)
        end
        pi_vals[src] = best
    end
    bound = 0.0
    for s in 1:n_strata
        bound += areas[s] * pi_vals[s]
    end
    return pi_vals, bound
end

"Minimize a function of `μ ∈ [0, ∞)` (quasi-convex) by a log grid plus golden-section refinement."
function _forest_minimize_mu(f::Function)
    grid = vcat(0.0, [10.0^e for e in range(-3.0, 2.0; length=26)])
    vals = [f(m) for m in grid]
    i = argmin(vals)
    lo = grid[max(1, i - 1)]
    hi = grid[min(length(grid), i + 1)]
    best_mu, best_val = grid[i], vals[i]
    g = (sqrt(5.0) - 1.0) / 2.0
    a, c = lo, hi
    x1 = c - g * (c - a)
    x2 = a + g * (c - a)
    f1, f2 = f(x1), f(x2)
    for _ in 1:40
        if f1 <= f2
            c, x2, f2 = x2, x1, f1
            x1 = c - g * (c - a)
            f1 = f(x1)
        else
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (c - a)
            f2 = f(x2)
        end
    end
    for (m, v) in ((x1, f1), (x2, f2))
        if v < best_val
            best_mu, best_val = m, v
        end
    end
    return best_mu, best_val
end

# -----------------------------------------------------------------------------
# Planted area-control witness
# -----------------------------------------------------------------------------

mutable struct _ForestParcel
    area::Float64
    holding::Int        # column currently holding this area
    zone::Int
    src::Int            # Model II source / Model I stratum
    gen::Int            # Model I rotation generation (0 = initial stand)
    cut1::Int           # Model I: first clearcut period of the holding prescription
    r1::Int             # Model I: regeneration option after cut1
end

"""
    _forest_greedy(b, F, V, caps) -> (ok, x)

Area-control heuristic: in every period cut exactly `V` m³ by clearcutting the
highest-volume eligible parcels first (fractional areas allowed), never
letting any watershed's clearcut area within a green-up window exceed
`caps[z]`. Harvested area is regenerated with the source's preferred option
and becomes a new parcel (Model I: a second rotation, if the prescription
exists; Model II: the destination regeneration node). Returns the column
areas `x` of the resulting schedule.
"""
function _forest_greedy(b::_ForestBuilder, F::Symbol, V::Float64, caps::Vector{Float64})
    T = b.T
    w = b.greenup_window
    S = length(b.stratum_area)
    x = zeros(length(b.col_source))
    parcels = _ForestParcel[]
    for s in 1:S
        e = b.end_col[s]
        x[e] += b.stratum_area[s]
        push!(parcels, _ForestParcel(b.stratum_area[s], e, b.stratum_zone[s], s, 0, 0, 0))
    end
    node_parcel = Dict{Int, Int}()
    cut_area = zeros(length(b.zone_area), T)
    cands = Tuple{Float64, Int, Int}[]
    for t in 1:T
        empty!(cands)
        for (pi, p) in enumerate(parcels)
            p.area <= 1.0e-12 && continue
            c, vol = _forest_harvest_option(b, F, p, t)
            c == 0 && continue
            vol <= 0.0 && continue
            push!(cands, (vol, pi, c))
        end
        sort!(cands; by=first, rev=true)
        need = V
        for (vol, pi, c) in cands
            need <= 1.0e-12 * V && break
            p = parcels[pi]
            z = p.zone
            used = 0.0
            for tau in max(1, t - w + 1):t
                used += cut_area[z, tau]
            end
            allow = caps[z] - used
            allow <= 1.0e-12 && continue
            a = min(p.area, need / vol, allow)
            a <= 0.0 && continue
            p.area -= a
            x[p.holding] -= a
            x[c] += a
            cut_area[z, t] += a
            need -= a * vol
            # successor parcel
            if F == :model_i
                if p.gen == 0
                    push!(parcels, _ForestParcel(a, c, z, p.src, 1, t, Int(b.col_regen1[c])))
                end
            else
                d = Int(b.col_dest[c])
                if d > 0
                    src = S + d
                    e = b.end_col[src]
                    x[e] += a
                    q = get(node_parcel, d, 0)
                    if q == 0
                        push!(parcels, _ForestParcel(a, e, z, src, 0, 0, 0))
                        node_parcel[d] = length(parcels)
                    else
                        parcels[q].area += a
                    end
                end
            end
        end
        need > 1.0e-9 * V && return false, x
    end
    @inbounds for j in eachindex(x)
        x[j] < 0.0 && (x[j] = 0.0)
    end
    return true, x
end

"Unthinned clearcut column (and its cut volume) harvesting parcel `p` at period `t`, or `(0, 0.0)`."
function _forest_harvest_option(b::_ForestBuilder, F::Symbol, p::_ForestParcel, t::Int)
    if F == :model_i
        if p.gen == 0
            pref = b.source_pref[p.src]
            nopt = length(b.regen_targets[b.stratum_model[p.src]])
            for k in 0:(nopt - 1)
                r = mod1(pref + k, nopt)
                c = get(b.col_key, _forest_key(b, p.src, t, r, 0), 0)
                c > 0 && return c, b.cutvol1[c]
            end
            return 0, 0.0
        elseif p.gen == 1
            c = get(b.col_key, _forest_key(b, p.src, p.cut1, p.r1, t), 0)
            c > 0 && return c, b.cutvol2[c]
            return 0, 0.0
        end
        return 0, 0.0
    else
        src = p.src
        m = src <= length(b.stratum_area) ? b.stratum_model[src] : b.node_model[src - length(b.stratum_area)]
        pref = b.source_pref[src]
        nopt = length(b.regen_targets[m])
        for k in 0:(nopt - 1)
            r = mod1(pref + k, nopt)
            c = get(b.col_key, _forest_key(b, src, t, r, 0), 0)
            c > 0 && return c, b.cutvol1[c]
        end
        return 0, 0.0
    end
end

"Product-by-period harvest and ending inventory of column areas `x`."
function _forest_schedule_totals(b::_ForestBuilder, x::Vector{Float64})
    K = length(b.products)
    H = zeros(b.T, K)
    ei = 0.0
    for j in eachindex(x)
        xj = x[j]
        xj == 0.0 && continue
        for e in b.vol_ptr[j]:(b.vol_ptr[j + 1] - 1)
            H[b.vol_period[e], b.vol_product[e]] += b.vol_amount[e] * xj
        end
        ei += b.col_ei[j] * xj
    end
    return H, ei
end

"""
Find the largest flat flow the area-control heuristic sustains (bisection on
`V`), then plant a schedule at `λ ∈ [0.70, 0.95]` of it.
"""
function _forest_plant_witness(rng::AbstractRNG, b::_ForestBuilder, F::Symbol, cumulative_max::Float64)
    caps = [0.9 * b.greenup_fraction[z] * b.zone_area[z] for z in eachindex(b.zone_area)]
    lo, hi = 0.0, cumulative_max / b.T
    for _ in 1:22
        mid = 0.5 * (lo + hi)
        ok, _ = _forest_greedy(b, F, mid, caps)
        ok ? (lo = mid) : (hi = mid)
    end
    lambda = 0.70 + 0.25 * rand(rng)
    V = lambda * lo
    for _ in 1:40
        ok, x = _forest_greedy(b, F, V, caps)
        ok && return V, x
        V *= 0.9
    end
    error("forest_planning: area-control witness failed (no sustainable flat flow found)")
end

# -----------------------------------------------------------------------------
# Shared constructor tail: witness, requirements, feasibility calibration
# -----------------------------------------------------------------------------

function _forest_finalize(
    ::Type{ForestPlanningProblem{F}}, rng::AbstractRNG, b::_ForestBuilder, status::FeasibilityStatus
) where {F}
    _forest_remap_sources!(b)
    S = length(b.stratum_area)
    T = b.T
    K = length(b.products)
    idx = _forest_source_index(b.col_source, S, b.node_period)
    ch = _forest_column_harvest_totals(b.vol_ptr, b.vol_amount)
    _, cum_max = _forest_dp_values(idx, b.col_dest, S, ch, b.col_ei, b.stratum_area, 0.0)

    V, x = _forest_plant_witness(rng, b, F, cum_max)
    Hw, eiw = _forest_schedule_totals(b, x)

    # Requirements derived from the planted schedule, with margins.
    delta = 0.05 + 0.15 * rand(rng)
    base_min_supply = [(0.75 + 0.17 * rand(rng)) * minimum(@view Hw[:, k]) for k in 1:K]
    base_ei = (0.85 + 0.10 * rand(rng)) * eiw
    max_supply = [(1.3 + 0.5 * rand(rng)) * maximum(@view Hw[:, k]) + 0.3 * V for k in 1:K]

    a = T * sum(base_min_supply)
    bb = base_ei
    _, ei_max = _forest_dp_values(idx, b.col_dest, S, zeros(length(ch)), b.col_ei, b.stratum_area, 1.0)
    ratio(mu) = _forest_dp_values(idx, b.col_dest, S, ch, b.col_ei, b.stratum_area, mu)[2] / (a + mu * bb)
    mu_star, theta_star = _forest_minimize_mu(ratio)
    theta_star >= 1.0 - 1.0e-9 ||
        error("forest_planning: Lagrangian bound below the planted schedule (internal error)")

    supply_scale = 1.0
    inventory_scale = 1.0
    witness = nothing
    certificate = nothing
    ei_cap = 0.93 * ei_max / bb             # keep the ending-inventory row satisfiable on its own
    if status == feasible
        witness = ForestPlanningWitness(x, Hw, V, eiw)
    elseif status == infeasible
        margin = 0.04 + 0.08 * rand(rng)
        s = theta_star * (1.0 + margin)
        mu_c = mu_star
        if a <= 0.0
            # Degenerate tiny instance whose planted schedule supports no
            # positive per-product contract: certify through the inventory
            # floor alone, with μ large enough that the volume terms cannot
            # close the gap (B(μ) <= cum_max + μ ei_max).
            inventory_scale = (1.0 + margin) * ei_max / bb
            mu_c = 2.2 * (1.0 + 0.5 * margin) * cum_max / (margin * ei_max) + 1.0
        elseif s <= ei_cap
            supply_scale = inventory_scale = s
        else
            # Joint scaling would make the ending-inventory row infeasible on
            # its own (a single-row contradiction presolve detects). Hold
            # inventory just below its maximum and raise supply instead.
            inventory_scale = max(1.0, ei_cap)
            g(mu) = (1.0 + margin) * _forest_dp_values(idx, b.col_dest, S, ch, b.col_ei, b.stratum_area, mu)[2] -
                mu * bb * inventory_scale
            mu_c, gval = _forest_minimize_mu(g)
            supply_scale = max(1.0, gval / a)
        end
        pi_vals, bound = _forest_dp_values(idx, b.col_dest, S, ch, b.col_ei, b.stratum_area, mu_c)
        required = a * supply_scale + mu_c * bb * inventory_scale
        bound * (1.0 + 0.5 * margin) <= required ||
            error(
                "forest_planning: infeasibility certificate lacks margin (internal error: " *
                "bound=$bound required=$required a=$a b=$bb mu=$mu_c theta=$theta_star " *
                "scales=($supply_scale, $inventory_scale) ei_cap=$ei_cap)",
            )
        certificate = ForestPlanningCertificate(mu_c, pi_vals, bound, required)
    else
        # Natural contract levels drawn on a continuum from the planted
        # (feasible) level to beyond the certified-infeasible level.
        s = 1.0 + (1.08 * theta_star - 1.0) * rand(rng)
        supply_scale = s
        inventory_scale = min(s, max(1.0, ei_cap))
    end

    min_supply = supply_scale .* base_min_supply
    max_supply = max.(max_supply, 1.25 .* min_supply)
    min_ei = inventory_scale * base_ei

    return ForestPlanningProblem{F}(
        b.region.name, b.L, T, b.discount_rate, b.products, b.types, b.models, b.regen_targets,
        b.regen_costs, b.saw_price, b.pulp_price, b.harvest_cost, b.thinning_cost,
        b.thinning_fraction, b.thinning_recovery, b.stratum_zone, b.stratum_model, b.stratum_age,
        b.stratum_area, b.zone_area, b.greenup_fraction, b.greenup_window, b.node_model,
        b.node_zone, b.node_period, b.col_source, b.col_dest, b.col_thin, b.col_cut1,
        b.col_regen1, b.col_cut2, b.vol_ptr, b.vol_period, b.vol_product, b.vol_amount,
        b.col_npv, b.col_ei, delta, base_min_supply, base_ei, supply_scale, inventory_scale,
        theta_star, min_supply, max_supply, min_ei, witness, certificate, status,
    )
end

"Validate the target and return `(rng, builder, column budget)` for a constructor."
function _forest_start(target_variables::Int, seed::Int)
    target_variables > FOREST_PLANNING_MAX_VARIABLES && throw(
        ArgumentError(
            "forest_planning supports at most $(FOREST_PLANNING_MAX_VARIABLES) variables " *
            "(requested $target_variables)",
        ),
    )
    rng = MersenneTwister(seed)
    b = _forest_builder(rng, max(target_variables, 1))
    # A floor of 30 area columns keeps tiny instances schedulable (clearcut
    # options in every period); smaller targets round up to it.
    budget = max(target_variables - b.T * length(b.products), 30)
    return rng, b, budget
end

"Total columns (area variables plus `harvest[t, k]` accounting variables)."
forest_num_variables(prob::ForestPlanningProblem) = length(prob.col_source) + prob.n_periods * length(prob.products)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

"""
    build_model(prob::ForestPlanningProblem)

Build the harvest-scheduling LP from stored data only (shared by Model I and
Model II). Variables: `area[j] >= 0` (ha per column) and accounting variables
`min_supply[k] <= harvest[t, k] <= max_supply[k]` (m³). Maximizes discounted
net revenue `Σ_j col_npv[j] * area[j]`. Rows: `stratum_area`, `node_balance`
(Model II only), `harvest_definition`, `even_flow_lower` / `even_flow_upper`,
`greenup` (watershed x period, non-empty rows only) and `ending_inventory`.
"""
function build_model(prob::ForestPlanningProblem)
    model = Model()
    T = prob.n_periods
    K = length(prob.products)
    n = length(prob.col_source)
    S = length(prob.stratum_area)
    N = length(prob.node_period)
    Z = length(prob.zone_area)
    w = prob.greenup_window

    @variable(model, area[1:n] >= 0)
    @variable(model, prob.min_supply[k] <= harvest[t=1:T, k=1:K] <= prob.max_supply[k])
    @objective(model, Max, sum(prob.col_npv[j] * area[j] for j in 1:n))

    src_expr = [AffExpr(0.0) for _ in 1:(S + N)]
    hexpr = [AffExpr(0.0) for _ in 1:T, _ in 1:K]
    gexpr = [AffExpr(0.0) for _ in 1:Z, _ in 1:T]
    ei_expr = AffExpr(0.0)
    @inbounds for j in 1:n
        v = area[j]
        src = Int(prob.col_source[j])
        add_to_expression!(src_expr[src], 1.0, v)
        d = Int(prob.col_dest[j])
        d > 0 && add_to_expression!(src_expr[S + d], -1.0, v)
        for e in prob.vol_ptr[j]:(prob.vol_ptr[j + 1] - 1)
            add_to_expression!(hexpr[prob.vol_period[e], prob.vol_product[e]], prob.vol_amount[e], v)
        end
        z = src <= S ? prob.stratum_zone[src] : prob.node_zone[src - S]
        for c in (Int(prob.col_cut1[j]), Int(prob.col_cut2[j]))
            c == 0 && continue
            for t in c:min(T, c + w - 1)
                add_to_expression!(gexpr[z, t], 1.0, v)
            end
        end
        ei = prob.col_ending_inventory[j]
        ei != 0.0 && add_to_expression!(ei_expr, ei, v)
    end

    @constraint(model, stratum_area[s=1:S], src_expr[s] == prob.stratum_area[s])
    if N > 0
        @constraint(model, node_balance[i=1:N], src_expr[S + i] == 0.0)
    end
    for t in 1:T, k in 1:K
        add_to_expression!(hexpr[t, k], -1.0, harvest[t, k])
    end
    @constraint(model, harvest_definition[t=1:T, k=1:K], hexpr[t, k] == 0.0)
    δ = prob.even_flow_tolerance
    @constraint(
        model, even_flow_lower[t=2:T],
        sum(harvest[t, k] for k in 1:K) - (1.0 - δ) * sum(harvest[t - 1, k] for k in 1:K) >= 0.0
    )
    @constraint(
        model, even_flow_upper[t=2:T],
        sum(harvest[t, k] for k in 1:K) - (1.0 + δ) * sum(harvest[t - 1, k] for k in 1:K) <= 0.0
    )
    for z in 1:Z, t in 1:T
        isempty(gexpr[z, t].terms) && continue
        @constraint(model, gexpr[z, t] <= prob.greenup_fraction[z] * prob.zone_area[z])
    end
    @constraint(model, ending_inventory, ei_expr >= prob.min_ending_inventory)
    return model
end
