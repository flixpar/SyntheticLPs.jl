using JuMP
using Random
using Distributions

# ---------------------------------------------------------------------------
# Calibration sources
# ---------------------------------------------------------------------------
# Crude archetypes, cut yields, and cut properties follow published assays
# (Hibernia, Bakken and WTI assay sheets reproduced in J. Jechura, "Refinery
# Feedstocks & Products", Colorado School of Mines, 2019) and the light-sweet /
# medium-sour / heavy-sour slates used in refinery planning studies.
#
# Conversion yields follow reported commercial ranges: FCC max-gasoline at
# 55-65 vol% gasoline, 25-30 vol% LPG, 15-20 vol% LCO and 4-10 vol% slurry, with
# max-distillate shifting roughly ten points from gasoline to LCO; catalytic
# reforming at 78-85 vol% reformate for 96-102 RON; hydrocracking at 110-115%
# volume swell; delayed coking at a coke yield near 1.6 x CCR (20-30 wt%).
#
# Product specifications and the price scale follow the multi-period refinery
# planning literature (Castillo Castillo & Mahalec; Li, Lin, Su & Xie,
# arXiv:2504.08642, Tables 1-2): gasoline RON 88.5/92.5 floors, 10 wt-ppm
# sulfur on road fuels, 0.3 wt% on jet, cetane floors near 40-51, and finished
# products priced $90-125/bbl against $55-85/bbl crude.
#
# Blending is written on component streams of fixed quality, which is what makes
# an industrial planning model an LP: a pool of unknown composition would make
# the quality rows bilinear. Weight-basis properties (sulfur) carry a density
# factor, so a specification stated on mass stays linear in volumetric flows.

# ---------------------------------------------------------------------------
# Quality vector layout
# ---------------------------------------------------------------------------

"""Quality properties carried by every process stream, in vector order."""
const PP_QUALITY_NAMES = (
    :density, :sulfur, :ron, :mon, :rvp, :aromatics, :cetane, :cold_flow, :viscosity_index
)
const PP_N_QUALITIES = length(PP_QUALITY_NAMES)

const PP_Q_DENSITY = 1      # specific gravity at 60F
const PP_Q_SULFUR = 2       # weight ppm
const PP_Q_RON = 3          # research octane number
const PP_Q_MON = 4          # motor octane number
const PP_Q_RVP = 5          # Reid vapour pressure, psi
const PP_Q_AROMATICS = 6    # vol%
const PP_Q_CETANE = 7       # cetane index
const PP_Q_COLD_FLOW = 8    # pour/freeze point, degrees C
const PP_Q_VISCOSITY = 9    # viscosity blending index

"""
Qualities that blend on mass rather than on volume. Their blend rows are
multiplied by stream density, which is how a wt% specification is written
against volumetric flows without leaving the linear world.
"""
const PP_WEIGHT_BASIS_QUALITIES = (PP_Q_SULFUR,)

_pp_is_weight_basis(q::Int) = q in PP_WEIGHT_BASIS_QUALITIES

# ---------------------------------------------------------------------------
# Stream property tables
# ---------------------------------------------------------------------------
# Entries are (density, sulfur, RON, MON, RVP, aromatics, cetane, cold flow,
# viscosity index). Sulfur is carried in weight ppm, the unit product
# specifications are written in; for a crude cut the entry is instead a
# multiplier on whole-crude sulfur (wt%), converted when the stream is built.

const _PP_CUT_QUALITY = (
    lpg_cut=(0.560, 0.010, 94.0, 90.0, 55.0, 0.0, 0.0, -110.0, -25.0),
    light_naphtha=(0.665, 0.030, 68.0, 66.0, 12.0, 2.0, 0.0, -100.0, -11.0),
    heavy_naphtha=(0.760, 0.080, 45.0, 43.0, 1.5, 16.0, 0.0, -90.0, -6.0),
    kerosene=(0.805, 0.200, 0.0, 0.0, 0.1, 18.0, 44.0, -47.0, 8.0),
    distillate=(0.850, 0.550, 0.0, 0.0, 0.05, 25.0, 53.0, -8.0, 16.0),
    gasoil=(0.875, 0.900, 0.0, 0.0, 0.02, 32.0, 48.0, 0.0, 25.0),
    vgo=(0.905, 1.400, 0.0, 0.0, 0.0, 40.0, 0.0, 28.0, 35.0),
    resid=(0.985, 2.600, 0.0, 0.0, 0.0, 55.0, 0.0, 45.0, 45.0),
)

const _PP_STREAM_QUALITY = (
    treated_naphtha=(0.760, 0.5, 45.0, 43.0, 1.5, 16.0, 0.0, -90.0, -6.0),
    isomerate=(0.655, 0.5, 84.0, 82.0, 14.0, 2.0, 0.0, -85.0, -12.0),
    reformate_mid=(0.780, 0.5, 96.0, 86.0, 4.0, 58.0, 0.0, -60.0, -6.0),
    reformate_high=(0.800, 0.5, 102.0, 90.0, 3.0, 70.0, 0.0, -55.0, -5.0),
    fcc_gasoline=(0.745, 9.0, 92.0, 80.0, 6.5, 30.0, 0.0, -60.0, -8.0),
    alkylate=(0.700, 0.5, 95.0, 92.0, 5.5, 0.5, 0.0, -70.0, -10.0),
    hc_naphtha=(0.730, 0.5, 68.0, 66.0, 8.0, 6.0, 0.0, -85.0, -9.0),
    coker_naphtha=(0.740, 5500.0, 60.0, 58.0, 7.0, 12.0, 0.0, -80.0, -8.0),
    jet_component=(0.800, 5.0, 0.0, 0.0, 0.1, 17.0, 45.0, -50.0, 8.0),
    hc_kero=(0.795, 3.0, 0.0, 0.0, 0.1, 12.0, 47.0, -52.0, 7.0),
    hc_diesel=(0.833, 3.0, 0.0, 0.0, 0.05, 15.0, 58.0, -14.0, 14.0),
    ulsd_component=(0.838, 8.0, 0.0, 0.0, 0.05, 22.0, 53.5, -11.0, 16.0),
    gasoil_treated=(0.860, 450.0, 0.0, 0.0, 0.03, 28.0, 47.0, -4.0, 20.0),
    lco=(0.945, 3000.0, 0.0, 0.0, 0.02, 70.0, 22.0, -25.0, 20.0),
    coker_gasoil=(0.905, 11000.0, 0.0, 0.0, 0.0, 45.0, 35.0, 20.0, 30.0),
    treated_vgo=(0.898, 1500.0, 0.0, 0.0, 0.0, 38.0, 0.0, 26.0, 33.0),
    treated_resid=(0.975, 5500.0, 0.0, 0.0, 0.0, 52.0, 0.0, 42.0, 43.0),
    slurry=(1.060, 14000.0, 0.0, 0.0, 0.0, 85.0, 0.0, 15.0, 42.0),
    coke=(1.100, 25000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    fuel_gas=(0.300, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    lpg=(0.560, 10.0, 94.0, 90.0, 55.0, 0.0, 0.0, -110.0, -25.0),
    butane=(0.585, 0.2, 93.0, 90.0, 52.0, 0.0, 0.0, -120.0, -26.0),
    ethanol=(0.794, 0.0, 108.0, 90.0, 18.0, 0.0, 0.0, -114.0, -14.0),
    mtbe=(0.745, 0.0, 118.0, 101.0, 8.0, 0.0, 0.0, -109.0, -13.0),
)

"""Crude cuts in boiling order; a reduced slate lumps cuts into a neighbour."""
const _PP_ALL_CUTS = (
    :lpg_cut, :light_naphtha, :heavy_naphtha, :kerosene, :distillate, :gasoil, :vgo, :resid
)

"""Cut slates, coarsest first. A coarse slate is a smaller assay, not a different crude."""
const _PP_CUT_SLATES = (
    [:light_naphtha, :distillate, :resid],
    [:light_naphtha, :kerosene, :distillate, :resid],
    [:light_naphtha, :heavy_naphtha, :kerosene, :distillate, :gasoil, :resid],
    [:lpg_cut, :light_naphtha, :heavy_naphtha, :kerosene, :distillate, :gasoil, :vgo, :resid],
)

"""Where a cut's yield goes when the slate does not carry that cut."""
const _PP_CUT_LUMP = (
    lpg_cut=:light_naphtha,
    light_naphtha=:light_naphtha,
    heavy_naphtha=:light_naphtha,
    kerosene=:distillate,
    distillate=:distillate,
    gasoil=:distillate,
    vgo=:resid,
    resid=:resid,
)

"""
Crude archetypes: `(name, API gravity, sulfur wt%, volumetric cut yields)`.
Yields are ordered as `_PP_ALL_CUTS` and sum to one.
"""
const _PP_CRUDE_ARCHETYPES = (
    (:condensate, 50.0, 0.05, (0.060, 0.230, 0.300, 0.180, 0.130, 0.050, 0.045, 0.005)),
    (:light_sweet, 39.0, 0.25, (0.030, 0.100, 0.170, 0.140, 0.150, 0.090, 0.230, 0.090)),
    (:medium_sour, 32.0, 1.60, (0.022, 0.070, 0.135, 0.125, 0.145, 0.093, 0.250, 0.160)),
    (:heavy_sour, 24.0, 2.90, (0.015, 0.045, 0.095, 0.095, 0.125, 0.085, 0.250, 0.290)),
    (:extra_heavy, 18.0, 3.60, (0.010, 0.030, 0.070, 0.080, 0.110, 0.080, 0.240, 0.380)),
)

# ---------------------------------------------------------------------------
# Process unit catalog
# ---------------------------------------------------------------------------

"""One operating mode of a process unit: an output slate, a capacity de-rate, and a cost factor."""
struct PPModeTemplate
    name::Symbol
    yields::Vector{Pair{Symbol, Float64}}
    capacity_factor::Float64
    cost_factor::Float64
end

"""
A process unit template.

`stage` orders the flowsheet: a unit only accepts feeds produced at a strictly
lower stage, so the instantiated network is always a DAG and the planted plan
can be propagated forward through it. `capacity_fraction` sizes the unit
relative to the crude charge.
"""
struct PPUnitTemplate
    key::Symbol
    stage::Int
    feed_classes::Vector{Symbol}
    modes::Vector{PPModeTemplate}
    capacity_fraction::Float64
    operating_cost::Float64
end

const _PP_UNIT_TEMPLATES = PPUnitTemplate[
    PPUnitTemplate(
        :naphtha_ht,
        2,
        [:heavy_naphtha, :coker_naphtha],
        [PPModeTemplate(:standard, [:treated_naphtha => 0.985, :fuel_gas => 0.015], 1.0, 1.0)],
        0.20,
        0.55,
    ),
    PPUnitTemplate(
        :isomerization,
        1,
        [:light_naphtha],
        [PPModeTemplate(:standard, [:isomerate => 0.970, :fuel_gas => 0.030], 1.0, 1.0)],
        0.08,
        1.10,
    ),
    PPUnitTemplate(
        :kero_ht,
        1,
        [:kerosene],
        [PPModeTemplate(:standard, [:jet_component => 0.995, :fuel_gas => 0.005], 1.0, 1.0)],
        0.14,
        0.90,
    ),
    PPUnitTemplate(
        :vgo_ht,
        1,
        [:vgo],
        [PPModeTemplate(:standard, [:treated_vgo => 0.990, :fuel_gas => 0.010], 1.0, 1.0)],
        0.22,
        1.40,
    ),
    PPUnitTemplate(
        :resid_ht,
        1,
        [:resid],
        [PPModeTemplate(:standard, [:treated_resid => 0.980, :fuel_gas => 0.020], 1.0, 1.0)],
        0.12,
        2.20,
    ),
    PPUnitTemplate(
        :coker,
        1,
        [:resid],
        [
            PPModeTemplate(
                :fuel_grade,
                [
                    :coker_naphtha => 0.140,
                    :coker_gasoil => 0.500,
                    :coke => 0.200,
                    :lpg => 0.050,
                    :fuel_gas => 0.060,
                ],
                1.0,
                1.0,
            ),
            PPModeTemplate(
                :anode_grade,
                [
                    :coker_naphtha => 0.120,
                    :coker_gasoil => 0.470,
                    :coke => 0.250,
                    :lpg => 0.040,
                    :fuel_gas => 0.060,
                ],
                0.92,
                1.08,
            ),
        ],
        0.16,
        3.90,
    ),
    PPUnitTemplate(
        :reformer,
        3,
        [:treated_naphtha],
        [
            PPModeTemplate(
                :mid_severity,
                [:reformate_mid => 0.845, :lpg => 0.085, :fuel_gas => 0.070],
                1.0,
                1.0,
            ),
            PPModeTemplate(
                :high_severity,
                [:reformate_high => 0.780, :lpg => 0.130, :fuel_gas => 0.090],
                0.95,
                1.15,
            ),
        ],
        0.18,
        2.00,
    ),
    PPUnitTemplate(
        :fcc,
        2,
        [:vgo, :treated_vgo, :coker_gasoil],
        [
            PPModeTemplate(
                :max_gasoline,
                [
                    :fcc_gasoline => 0.570,
                    :lco => 0.160,
                    :slurry => 0.070,
                    :lpg => 0.240,
                    :fuel_gas => 0.040,
                ],
                1.0,
                1.0,
            ),
            PPModeTemplate(
                :max_distillate,
                [
                    :fcc_gasoline => 0.440,
                    :lco => 0.300,
                    :slurry => 0.090,
                    :lpg => 0.170,
                    :fuel_gas => 0.035,
                ],
                0.97,
                0.98,
            ),
            PPModeTemplate(
                :max_olefins,
                [
                    :fcc_gasoline => 0.470,
                    :lco => 0.150,
                    :slurry => 0.070,
                    :lpg => 0.330,
                    :fuel_gas => 0.060,
                ],
                0.93,
                1.12,
            ),
        ],
        0.32,
        3.40,
    ),
    PPUnitTemplate(
        :hydrocracker,
        2,
        [:vgo, :treated_vgo, :coker_gasoil],
        [
            PPModeTemplate(
                :max_diesel,
                [:hc_naphtha => 0.150, :hc_kero => 0.200, :hc_diesel => 0.700, :lpg => 0.050],
                1.0,
                1.0,
            ),
            PPModeTemplate(
                :max_jet,
                [:hc_naphtha => 0.170, :hc_kero => 0.450, :hc_diesel => 0.420, :lpg => 0.060],
                0.98,
                1.05,
            ),
            PPModeTemplate(
                :max_naphtha,
                [:hc_naphtha => 0.550, :hc_kero => 0.250, :hc_diesel => 0.220, :lpg => 0.120],
                0.94,
                1.10,
            ),
        ],
        0.18,
        4.80,
    ),
    PPUnitTemplate(
        :diesel_ht,
        3,
        [:distillate, :gasoil, :lco, :coker_gasoil],
        [
            PPModeTemplate(:ulsd, [:ulsd_component => 0.980, :fuel_gas => 0.020], 1.0, 1.0),
            PPModeTemplate(:mild, [:gasoil_treated => 0.990, :fuel_gas => 0.010], 1.12, 0.72),
        ],
        0.30,
        1.60,
    ),
    PPUnitTemplate(
        :alkylation,
        4,
        [:lpg],
        [PPModeTemplate(:standard, [:alkylate => 0.620, :fuel_gas => 0.050], 1.0, 1.0)],
        0.07,
        5.50,
    ),
]

"""Units a level adds on top of a bare topping refinery."""
const _PP_LEVEL_UNITS = (
    Symbol[],
    [:naphtha_ht, :reformer, :kero_ht, :diesel_ht],
    [:naphtha_ht, :reformer, :kero_ht, :diesel_ht, :isomerization, :vgo_ht, :fcc, :alkylation],
    [
        :naphtha_ht,
        :reformer,
        :kero_ht,
        :diesel_ht,
        :isomerization,
        :vgo_ht,
        :fcc,
        :alkylation,
        :coker,
        :hydrocracker,
        :resid_ht,
    ],
)

"""
Units that run whenever the crude unit does, and so carry a minimum rate even in
a model with no on/off decision. A conversion unit can be idled for a period, so
stating a turndown for it needs a run indicator (see `mode_switching`).
"""
const _PP_CONTINUOUS_UNITS = (:naphtha_ht, :kero_ht, :vgo_ht, :diesel_ht, :resid_ht)

"""Units that may be dropped from a level without leaving an implausible refinery."""
const _PP_OPTIONAL_UNITS = (
    :isomerization, :vgo_ht, :alkylation, :resid_ht, :hydrocracker, :kero_ht
)

# ---------------------------------------------------------------------------
# Finished product catalog
# ---------------------------------------------------------------------------

"""
A finished-product template: admissible blend components, quality window,
reference price (\$/bbl) and the share of refinery output it typically takes.
"""
struct PPProductTemplate
    key::Symbol
    component_classes::Vector{Symbol}
    spec_min::Vector{Pair{Int, Float64}}
    spec_max::Vector{Pair{Int, Float64}}
    price::Float64
    demand_share::Float64
end

const _PP_PRODUCT_TEMPLATES = PPProductTemplate[
    PPProductTemplate(
        :regular_gasoline,
        [
            :light_naphtha,
            :isomerate,
            :reformate_mid,
            :reformate_high,
            :fcc_gasoline,
            :alkylate,
            :hc_naphtha,
            :butane,
            :ethanol,
            :mtbe,
        ],
        [PP_Q_RON => 91.0, PP_Q_MON => 82.5, PP_Q_DENSITY => 0.720],
        [PP_Q_RVP => 9.0, PP_Q_SULFUR => 10.0, PP_Q_AROMATICS => 35.0, PP_Q_DENSITY => 0.775],
        95.0,
        0.28,
    ),
    PPProductTemplate(
        :premium_gasoline,
        [
            :light_naphtha,
            :isomerate,
            :reformate_mid,
            :reformate_high,
            :fcc_gasoline,
            :alkylate,
            :hc_naphtha,
            :butane,
            :ethanol,
            :mtbe,
        ],
        [PP_Q_RON => 95.0, PP_Q_MON => 85.0, PP_Q_DENSITY => 0.720],
        [PP_Q_RVP => 9.0, PP_Q_SULFUR => 10.0, PP_Q_AROMATICS => 35.0, PP_Q_DENSITY => 0.775],
        105.0,
        0.09,
    ),
    PPProductTemplate(
        :jet_a1,
        [:kerosene, :jet_component, :hc_kero],
        [PP_Q_DENSITY => 0.775],
        [
            PP_Q_SULFUR => 3000.0,
            PP_Q_AROMATICS => 25.0,
            PP_Q_COLD_FLOW => -47.0,
            PP_Q_DENSITY => 0.840,
        ],
        100.0,
        0.11,
    ),
    PPProductTemplate(
        :ulsd,
        [:ulsd_component, :hc_diesel, :hc_kero, :jet_component],
        [PP_Q_CETANE => 51.0, PP_Q_DENSITY => 0.820],
        [PP_Q_SULFUR => 10.0, PP_Q_DENSITY => 0.845, PP_Q_COLD_FLOW => -5.0],
        102.0,
        0.27,
    ),
    PPProductTemplate(
        :heating_gasoil,
        [:distillate, :gasoil, :kerosene, :ulsd_component, :gasoil_treated, :hc_diesel, :lco],
        [PP_Q_CETANE => 40.0],
        [PP_Q_SULFUR => 1000.0, PP_Q_DENSITY => 0.900],
        90.0,
        0.08,
    ),
    PPProductTemplate(
        :fuel_oil,
        [:resid, :treated_resid, :slurry, :lco, :gasoil, :coker_gasoil, :distillate],
        Pair{Int, Float64}[],
        [PP_Q_SULFUR => 5000.0, PP_Q_DENSITY => 0.991, PP_Q_VISCOSITY => 36.9],
        58.0,
        0.12,
    ),
    PPProductTemplate(
        :lpg_product, [:lpg_cut, :lpg], Pair{Int, Float64}[], [PP_Q_SULFUR => 200.0], 45.0, 0.03
    ),
    PPProductTemplate(
        :petcoke, [:coke], Pair{Int, Float64}[], [PP_Q_SULFUR => 40000.0], 10.0, 0.02
    ),
]

const _PP_LEVEL_PRODUCTS = (
    [:regular_gasoline, :heating_gasoil, :fuel_oil],
    [
        :regular_gasoline,
        :premium_gasoline,
        :jet_a1,
        :ulsd,
        :heating_gasoil,
        :fuel_oil,
        :lpg_product,
    ],
    [
        :regular_gasoline,
        :premium_gasoline,
        :jet_a1,
        :ulsd,
        :heating_gasoil,
        :fuel_oil,
        :lpg_product,
    ],
    [
        :regular_gasoline,
        :premium_gasoline,
        :jet_a1,
        :ulsd,
        :heating_gasoil,
        :fuel_oil,
        :lpg_product,
        :petcoke,
    ],
)

"""Purchased blendstocks, with purchase cost in \$/bbl."""
const _PP_PURCHASE_CLASSES = (:butane, :ethanol, :mtbe)
const _PP_PURCHASE_COST = (52.0, 78.0, 118.0)

"""
Spot value in \$/bbl of the streams that trade well below finished-product value
(refinery fuel, LPG, petcoke, bunker cutter stock). Every other stream is priced
at a discount to the cheapest grade it could have gone into.
"""
const _PP_SPOT_PRICE = (fuel_gas=22.0, lpg=46.0, lpg_cut=46.0, coke=11.0, slurry=38.0)

"""Classes with no tankage in a planning model (gas and solids)."""
const _PP_NON_STORABLE = (:fuel_gas, :coke)

_pp_unit_template(key::Symbol) =
    _PP_UNIT_TEMPLATES[findfirst(t -> t.key == key, _PP_UNIT_TEMPLATES)]

_pp_product_template(key::Symbol) =
    _PP_PRODUCT_TEMPLATES[findfirst(t -> t.key == key, _PP_PRODUCT_TEMPLATES)]

_pp_is_cut_class(class::Symbol) = class in _PP_ALL_CUTS

function _pp_class_quality(class::Symbol)
    _pp_is_cut_class(class) && return getproperty(_PP_CUT_QUALITY, class)
    return getproperty(_PP_STREAM_QUALITY, class)
end

# ---------------------------------------------------------------------------
# Structure and skeleton: the parts of an instance that carry no random data
# ---------------------------------------------------------------------------

"""
    PPStructure

The structural choices of an instance — cut slate, instantiated units (a repeated
key is a parallel train), the modes each unit may run, instantiated product
grades, and purchased blendstocks. Drawn before sizing so that the realised
variable count is an exact, RNG-free function of the structure and the crude
count.
"""
struct PPStructure
    level::Int
    cut_classes::Vector{Symbol}
    unit_keys::Vector{Symbol}
    unit_modes::Vector{Vector{Symbol}}
    product_keys::Vector{Symbol}
    purchase_classes::Vector{Symbol}
end

"""
    PPSkeleton

The realised network for a given structure and crude count: stream list, unit
feed/output index sets, product component sets, and the storable / purchasable /
spot-saleable index sets. Pure function of `(structure, n_crudes)`, so the
variable count can be inverted for the target size before any data is drawn.
"""
struct PPSkeleton
    n_crudes::Int
    cut_classes::Vector{Symbol}
    cut_stream::Matrix{Int}
    stream_classes::Vector{Symbol}
    stream_crude::Vector{Int}
    stream_stage::Vector{Int}
    unit_keys::Vector{Symbol}
    unit_stage::Vector{Int}
    unit_modes::Vector{Vector{Symbol}}
    unit_feeds::Vector{Vector{Int}}
    unit_outputs::Vector{Vector{Int}}
    unit_mode_yields::Vector{Vector{Vector{Pair{Int, Float64}}}}
    product_keys::Vector{Symbol}
    product_components::Vector{Vector{Int}}
    storable::Vector{Int}
    purchasable::Vector{Int}
    spot::Vector{Int}
end

n_streams(sk::PPSkeleton) = length(sk.stream_classes)
n_units(sk::PPSkeleton) = length(sk.unit_keys)
n_products(sk::PPSkeleton) = length(sk.product_keys)

"""
    _pp_structure(rng, target, level; single_mode) -> PPStructure

Draw the structural shape of one refinery configuration at the given complexity
`level`: topping, hydroskimming, cracking, or full conversion. The cut slate, the
number of finished grades and the number of parallel trains scale with the
requested size; which level an instance ends up at is decided by
[`_pp_dimensions`](@ref), which keeps the configuration that fits the target.

With `single_mode=true` each unit is fixed to one randomly chosen operating mode,
which is what makes the `refinery` variant a pure LP: only that mode's product
streams are instantiated.
"""
function _pp_structure(rng::AbstractRNG, target::Int, level::Int; single_mode::Bool=false)
    scale = log10(max(target, 10))
    slate_low = target < 90 ? 1 : (target < 200 ? 2 : (target < 600 ? 3 : 4))
    slate_high = target < 200 ? 3 : 4
    cut_classes = copy(_PP_CUT_SLATES[rand(rng, slate_low:slate_high)])

    unit_keys = Symbol[]
    unit_modes = Vector{Symbol}[]
    trains = clamp(floor(Int, scale) - 3, 1, 3)
    for key in _PP_LEVEL_UNITS[level]
        if key in _PP_OPTIONAL_UNITS && rand(rng) < 0.25
            continue
        end
        template = _pp_unit_template(key)
        count = key in (:fcc, :diesel_ht, :naphtha_ht) ? trains : 1
        for _ in 1:count
            push!(unit_keys, key)
            push!(
                unit_modes,
                if single_mode
                    [template.modes[rand(rng, 1:length(template.modes))].name]
                else
                    [m.name for m in template.modes]
                end,
            )
        end
    end

    grade_extra = clamp(floor(Int, scale) - 3, 0, 6)
    product_keys = Symbol[]
    # A very small request cannot carry a full grade slate; keep the two largest
    # pools (a gasoline and a distillate) and drop the rest.
    slate = target < 90 ? _PP_LEVEL_PRODUCTS[level][1:min(2, end)] : _PP_LEVEL_PRODUCTS[level]
    for key in slate
        template = _pp_product_template(key)
        grades = 1 + (rand(rng) < 0.6 ? rand(rng, 0:grade_extra) : 0)
        # Gasoline and diesel carry seasonal and regional grades before the
        # niche products do.
        template.demand_share >= 0.10 && (grades += rand(rng, 0:grade_extra))
        for _ in 1:grades
            push!(product_keys, key)
        end
    end

    purchase_classes = Symbol[]
    if level >= 2
        for (i, class) in enumerate(_PP_PURCHASE_CLASSES)
            rand(rng) < (i == 1 ? 0.9 : 0.5) && push!(purchase_classes, class)
        end
    elseif rand(rng) < 0.5
        push!(purchase_classes, :butane)
    end

    return PPStructure(level, cut_classes, unit_keys, unit_modes, product_keys, purchase_classes)
end

"""
    _pp_realize(structure, n_crudes) -> PPSkeleton

Instantiate the network. Crude cuts are segregated per crude (each carries the
quality of that crude's assay, which is what keeps the blend rows linear);
conversion products are single streams shared by every train that makes them.
Units with no available feed and products with no available component are
dropped to a fixpoint, and any stream left without a sink becomes spot-saleable
(refinery fuel), so the network never contains a dead node.
"""
function _pp_realize(structure::PPStructure, n_crudes::Int)
    cut_classes = structure.cut_classes
    n_cuts = length(cut_classes)

    # Which classes actually exist: crude cuts and purchases to begin with, then
    # the outputs of every unit that finds a feed. Iterated to a fixpoint so a
    # unit dropped for want of feed cannot leave its products behind.
    template_of = [_pp_unit_template(k) for k in structure.unit_keys]
    unit_order = sortperm([(t.stage, i) for (i, t) in enumerate(template_of)])
    keep = trues(length(structure.unit_keys))
    changed = true
    while changed
        changed = false
        available = Set{Symbol}(cut_classes)
        union!(available, structure.purchase_classes)
        for i in unit_order
            keep[i] || continue
            template = template_of[i]
            if !any(cls in available for cls in template.feed_classes)
                keep[i] = false
                changed = true
                continue
            end
            for mode_name in structure.unit_modes[i]
                mode = template.modes[findfirst(m -> m.name == mode_name, template.modes)]
                for (cls, _) in mode.yields
                    push!(available, cls)
                end
            end
        end
    end

    unit_keys = Symbol[]
    unit_stage = Int[]
    unit_modes = Vector{Symbol}[]
    for i in unit_order
        keep[i] || continue
        push!(unit_keys, structure.unit_keys[i])
        push!(unit_stage, template_of[i].stage)
        push!(unit_modes, copy(structure.unit_modes[i]))
    end

    stream_classes = Symbol[]
    stream_crude = Int[]
    stream_stage = Int[]
    cut_stream = zeros(Int, n_crudes, n_cuts)
    for c in 1:n_crudes, k in 1:n_cuts
        push!(stream_classes, cut_classes[k])
        push!(stream_crude, c)
        push!(stream_stage, 0)
        cut_stream[c, k] = length(stream_classes)
    end

    class_index = Dict{Symbol, Int}()
    # A stream's stage is the *latest* stage that can make it, so the forward
    # plan only allocates it once every producer has run. The unit catalog keeps
    # every feed strictly below its consumer's stage under this rule, which is
    # what makes the flowsheet a DAG.
    function _stream_for_class(class::Symbol, stage::Int)
        if haskey(class_index, class)
            s = class_index[class]
            stream_stage[s] = max(stream_stage[s], stage)
            return s
        end
        push!(stream_classes, class)
        push!(stream_crude, 0)
        push!(stream_stage, stage)
        class_index[class] = length(stream_classes)
        return class_index[class]
    end

    purchasable = [_stream_for_class(cls, 0) for cls in structure.purchase_classes]
    unit_mode_yields = Vector{Vector{Pair{Int, Float64}}}[]
    unit_outputs = Vector{Int}[]
    for i in eachindex(unit_keys)
        template = _pp_unit_template(unit_keys[i])
        mode_yields = Vector{Pair{Int, Float64}}[]
        outputs = Int[]
        for mode_name in unit_modes[i]
            mode = template.modes[findfirst(x -> x.name == mode_name, template.modes)]
            pairs = Pair{Int, Float64}[]
            for (cls, y) in mode.yields
                s = _stream_for_class(cls, template.stage)
                push!(pairs, s => y)
                s in outputs || push!(outputs, s)
            end
            push!(mode_yields, pairs)
        end
        push!(unit_mode_yields, mode_yields)
        push!(unit_outputs, outputs)
    end

    # Feed sets: every instantiated stream of an admissible class made earlier.
    class_members = Dict{Symbol, Vector{Int}}()
    for (s, class) in enumerate(stream_classes)
        push!(get!(class_members, class, Int[]), s)
    end
    unit_feeds = Vector{Int}[]
    for i in eachindex(unit_keys)
        template = _pp_unit_template(unit_keys[i])
        feeds = Int[]
        for cls in template.feed_classes, s in get(class_members, cls, Int[])
            stream_stage[s] < template.stage && push!(feeds, s)
        end
        push!(unit_feeds, sort!(feeds))
    end

    product_keys = Symbol[]
    product_components = Vector{Int}[]
    for key in structure.product_keys
        template = _pp_product_template(key)
        comps = Int[]
        for cls in template.component_classes
            append!(comps, get(class_members, cls, Int[]))
        end
        isempty(comps) && continue
        push!(product_keys, key)
        push!(product_components, sort!(comps))
    end

    storable = [s for s in eachindex(stream_classes) if !(stream_classes[s] in _PP_NON_STORABLE)]

    # Outside markets exist for bulk cuts and refinery by-products, but not for
    # every upgraded blend component.  Giving reformate, alkylate, hydrotreated
    # diesel, and similar streams an automatic outlet makes integrated refinery
    # choices artificially easy.  A stream with no process or blend sink still
    # receives a disposal/merchant outlet so the generated graph has no dead
    # nodes; purchased blendstocks are excluded to prevent buy-and-resell loops.
    merchant_classes = Set((
        :fuel_gas,
        :lpg,
        :lpg_cut,
        :coke,
        :slurry,
        :light_naphtha,
        :heavy_naphtha,
        :kerosene,
        :distillate,
        :gasoil,
        :vgo,
        :resid,
        :coker_gasoil,
    ))
    downstream = falses(length(stream_classes))
    for feeds in unit_feeds, s in feeds
        downstream[s] = true
    end
    for components in product_components, s in components
        downstream[s] = true
    end
    spot = [
        s for s in eachindex(stream_classes) if
        !(s in purchasable) && (stream_classes[s] in merchant_classes || !downstream[s])
    ]

    return PPSkeleton(
        n_crudes,
        cut_classes,
        cut_stream,
        stream_classes,
        stream_crude,
        stream_stage,
        unit_keys,
        unit_stage,
        unit_modes,
        unit_feeds,
        unit_outputs,
        unit_mode_yields,
        product_keys,
        product_components,
        storable,
        purchasable,
        spot,
    )
end

"""
    _pp_variables_per_period(sk; mode_vars) -> Int

Exact number of decision variables the model creates per period.

With `mode_vars=false` (the `refinery` LP) a unit contributes one feed variable
per admissible feed stream plus its throughput. With `mode_vars=true` (the
`mode_switching` MILP) those are replicated per operating mode and each mode also
carries a run indicator and a start-up indicator.
"""
function _pp_variables_per_period(sk::PPSkeleton; mode_vars::Bool, extra_variables::Int=0)
    total = 3 * sk.n_crudes + extra_variables
    for i in 1:n_units(sk)
        feeds = length(sk.unit_feeds[i])
        if mode_vars
            total += length(sk.unit_modes[i]) * (feeds + 3)
        else
            total += feeds + 1
        end
    end
    for comps in sk.product_components
        total += length(comps)
    end
    total += length(sk.storable) + length(sk.purchasable) + length(sk.spot)
    total += 2 * n_products(sk)
    return total
end

"""
    _pp_level_floor(target) -> Int

The least refinery complexity level a request of `target` variables is served at:
topping below 200 variables, hydroskimming to 900, cracking above it.

The size score cannot be left to decide this. A topping refinery has by far the
smallest per-period block, so it can land a total exactly on the target where a
configuration with conversion units cannot, and it then wins on the score's first
element — relative size error — before the shape term that prefers the more
complete refinery is ever consulted. Left to the score, roughly half of all
requests came back as a bare crude-cut-and-blend LP at every size, including the
largest. Stating the complexity the scale deserves as a floor and letting the
search work under it costs size accuracy only where the block is coarse relative
to the target: measured over 1000 seeds per target, the worst relative error
across the three callers rises to 16% just above the 200 threshold and is under
5% from 400 variables up, against the 20% the category contracts for.
"""
_pp_level_floor(target::Int) =
    if target < 200
        1
    elseif target < 900
        2
    else
        3
    end

"""
    _pp_dimensions(rng, target; mode_vars, minimum_level=1,
                   extra_variables=0, compact_hydroprocessing=false)
        -> (structure, skeleton, n_periods)

Choose the refinery configuration, crude count and horizon that land the variable
count on `target`.

One configuration is drawn per complexity level; for each, the per-period count
is affine in the crude count, so two probes give its slope and intercept exactly
and the crude count needed for a given horizon follows in closed form. Every
(configuration, horizon, crude count) triple is then scored on relative size
error first and shape second, without either being forced into a horizon of one
period. Ties favour operationally ordinary shapes — a horizon of one to two dozen
periods, a crude menu of a handful of grades — and the more complete refinery.

Complexity itself is set by [`_pp_level_floor`](@ref) rather than left to that
score, which cannot carry it: see the note there.

`compact_hydroprocessing=true` offers a one-unit diesel-hydrotreating structure
for small mode/H2 requests. It preserves the defining physical feature without
forcing the much larger seven-product hydroskimming configuration.
"""
function _pp_dimensions(
    rng::AbstractRNG,
    target::Int;
    mode_vars::Bool,
    minimum_level::Int=1,
    extra_variables::Int=0,
    compact_hydroprocessing::Bool=false,
)
    horizon_pref = clamp(round(Int, 3.0 * log10(max(target, 10))), 4, 26)
    best_structure = nothing
    best_crudes, best_periods = 1, 2
    best_score = (Inf, Inf)
    1 <= minimum_level <= length(_PP_LEVEL_UNITS) ||
        throw(ArgumentError("minimum_level must name a refinery complexity level"))
    candidates = Tuple{PPStructure, Int}[]
    # The ordinary hydroskimming configuration has four conversion units and
    # seven products. At very small targets that alone is too large, but falling
    # back to a topping refinery would make the mode and hydrogen variants
    # vacuous. Offer a compact diesel-hydrotreating line instead: it still has a
    # physical hydroprocessing feed and, when requested, two operating modes,
    # while retaining enough room for a multi-period horizon.
    if compact_hydroprocessing && target <= 160
        modes = mode_vars ? [:ulsd, :mild] : [:ulsd]
        push!(
            candidates,
            (
                PPStructure(
                    2, copy(_PP_CUT_SLATES[1]), [:diesel_ht], [modes], [:heating_gasoil], Symbol[]
                ),
                2,
            ),
        )
    end
    for level in max(minimum_level, _pp_level_floor(target)):length(_PP_LEVEL_UNITS)
        push!(candidates, (_pp_structure(rng, target, level; single_mode=(!mode_vars)), level))
    end
    for (structure, level) in candidates
        # The per-period count is exactly affine in the crude count for a fixed
        # structure, so two probes pin it down and the crude count needed for a
        # given horizon follows in closed form.
        v2 = _pp_variables_per_period(
            _pp_realize(structure, 2); mode_vars=mode_vars, extra_variables=extra_variables
        )
        v3 = _pp_variables_per_period(
            _pp_realize(structure, 3); mode_vars=mode_vars, extra_variables=extra_variables
        )
        slope = max(v3 - v2, 1)
        intercept = v2 - 2 * slope
        # Keep the horizon genuinely multi-period: search from four periods
        # whenever the smallest crude menu leaves room for four, and from two
        # when it does not.
        longest = max(2, fld(target, max(slope + intercept, 1)))
        shortest = min(4, longest)
        for T in shortest:52
            c_star = (target / T - intercept) / slope
            for C in unique(clamp.(round.(Int, [c_star - 1, c_star, c_star + 1]), 1, 4000))
                total = (slope * C + intercept) * T
                err = abs(total - target) / target
                shape =
                    abs(T - horizon_pref) / 26 +
                    abs(log(C / 6)) / 6 +
                    0.15 * (length(_PP_LEVEL_UNITS) - level)
                score = (round(err; digits=3), shape)
                if score < best_score
                    best_score = score
                    best_structure = structure
                    best_crudes, best_periods = C, T
                end
            end
        end
    end
    return best_structure, _pp_realize(best_structure, best_crudes), best_periods
end

# ---------------------------------------------------------------------------
# Flowsheet: the skeleton with all sampled data attached
# ---------------------------------------------------------------------------

"""One operating mode of an instantiated unit: per-feed yields, capacity de-rate and cost."""
struct RefineryUnitMode
    name::Symbol
    yields::Matrix{Float64}     # [feed index, output index] volumetric yield
    capacity_factor::Float64
    operating_cost::Float64
end

"""An instantiated process unit."""
struct RefineryUnit
    name::Symbol
    key::Symbol
    stage::Int
    feeds::Vector{Int}
    outputs::Vector{Int}
    modes::Vector{RefineryUnitMode}
end

"""An instantiated finished product: admissible components and its quality window."""
struct RefineryProduct
    name::Symbol
    key::Symbol
    components::Vector{Int}
    spec_min::Vector{Float64}
    spec_max::Vector{Float64}
end

"""
    RefineryFlowsheet

The static refinery network: crude assays, the streams they cut into, the
conversion units, and the finished products with their blend windows. Every
stream carries a fixed quality vector, which is what keeps the blending rows
linear.
"""
struct RefineryFlowsheet
    n_crudes::Int
    crude_names::Vector{Symbol}
    crude_api::Vector{Float64}
    crude_sulfur::Vector{Float64}
    cut_classes::Vector{Symbol}
    cut_yields::Matrix{Float64}
    cut_stream::Matrix{Int}
    stream_names::Vector{Symbol}
    stream_classes::Vector{Symbol}
    stream_crude::Vector{Int}
    stream_stage::Vector{Int}
    qualities::Matrix{Float64}      # [stream, quality]
    units::Vector{RefineryUnit}
    products::Vector{RefineryProduct}
    storable::Vector{Int}
    purchasable::Vector{Int}
    spot::Vector{Int}
end

n_streams(fs::RefineryFlowsheet) = length(fs.stream_classes)
n_units(fs::RefineryFlowsheet) = length(fs.units)
n_products(fs::RefineryFlowsheet) = length(fs.products)

"""
    _pp_build_flowsheet(rng, skeleton) -> RefineryFlowsheet

Attach sampled data to a realised network: crude archetypes and their assay
yields, stream qualities (crude cuts inherit the sulfur of their crude and are
perturbed around the assay, conversion products sit at their unit's typical
severity), per-feed unit yields, and product blend windows.
"""
function _pp_build_flowsheet(rng::AbstractRNG, sk::PPSkeleton)
    C = sk.n_crudes
    cuts = sk.cut_classes
    n_cuts = length(cuts)

    archetype_index = Vector{Int}(undef, C)
    for c in 1:C
        archetype_index[c] =
            c <= length(_PP_CRUDE_ARCHETYPES) ? c : rand(rng, 1:length(_PP_CRUDE_ARCHETYPES))
    end
    C > 1 && shuffle!(rng, archetype_index)

    crude_names = Vector{Symbol}(undef, C)
    crude_api = Vector{Float64}(undef, C)
    crude_sulfur = Vector{Float64}(undef, C)
    cut_yields = zeros(Float64, C, n_cuts)
    for c in 1:C
        name, api, sulfur, yields = _PP_CRUDE_ARCHETYPES[archetype_index[c]]
        crude_names[c] = Symbol(name, :_, c)
        crude_api[c] = round(api * rand(rng, Uniform(0.94, 1.06)); digits=1)
        crude_sulfur[c] = round(max(0.01, sulfur * rand(rng, Uniform(0.75, 1.30))); digits=3)
        # Lump the full assay onto the instance's cut slate, then perturb. A cut
        # the slate carries keeps its own yield; one it does not is folded into
        # its neighbour (light ends into light naphtha, gas oil into distillate,
        # vacuum gas oil into residue), following the chain until a cut the slate
        # actually has is reached.
        lumped = zeros(Float64, n_cuts)
        for (k, class) in enumerate(_PP_ALL_CUTS)
            resolved = class
            for _ in 1:length(_PP_ALL_CUTS)
                resolved in cuts && break
                resolved = getproperty(_PP_CUT_LUMP, resolved)
            end
            idx = findfirst(==(resolved), cuts)
            idx === nothing && (idx = n_cuts)
            lumped[idx] += yields[k]
        end
        noisy = [max(1e-3, lumped[k] * rand(rng, Uniform(0.88, 1.12))) for k in 1:n_cuts]
        cut_yields[c, :] .= noisy ./ sum(noisy)
    end

    S = length(sk.stream_classes)
    qualities = zeros(Float64, S, PP_N_QUALITIES)
    stream_names = Vector{Symbol}(undef, S)
    for s in 1:S
        class = sk.stream_classes[s]
        base = _pp_class_quality(class)
        crude = sk.stream_crude[s]
        stream_names[s] = crude == 0 ? class : Symbol(class, :_, crude)
        for q in 1:PP_N_QUALITIES
            value = base[q]
            if q == PP_Q_SULFUR && crude != 0
                # Cut tables hold a multiplier on whole-crude sulfur (wt%);
                # stream qualities carry sulfur in weight ppm.
                value *= crude_sulfur[crude] * 10_000
            end
            if q == PP_Q_DENSITY
                value *= rand(rng, Uniform(0.985, 1.015))
            elseif q in (PP_Q_RON, PP_Q_MON, PP_Q_CETANE)
                value += value > 0 ? rand(rng, Uniform(-1.5, 1.5)) : 0.0
            elseif q == PP_Q_COLD_FLOW
                value += rand(rng, Uniform(-4.0, 4.0))
            elseif value != 0.0
                value *= rand(rng, Uniform(0.90, 1.10))
            end
            qualities[s, q] = value
        end
        # A crude cut inherits its crude's character: heavier crude, heavier and
        # more aromatic cuts.
        if crude != 0
            heaviness = clamp(1.0 + (34.0 - crude_api[crude]) / 240.0, 0.94, 1.08)
            qualities[s, PP_Q_DENSITY] *= heaviness
            qualities[s, PP_Q_AROMATICS] *= heaviness
            qualities[s, PP_Q_CETANE] /= heaviness
        end
        qualities[s, PP_Q_DENSITY] = clamp(qualities[s, PP_Q_DENSITY], 0.25, 1.15)
        qualities[s, PP_Q_SULFUR] = max(qualities[s, PP_Q_SULFUR], 0.0)
    end

    units = RefineryUnit[]
    train_count = Dict{Symbol, Int}()
    for i in eachindex(sk.unit_keys)
        key = sk.unit_keys[i]
        template = _pp_unit_template(key)
        train = get(train_count, key, 0) + 1
        train_count[key] = train
        name = train == 1 && count(==(key), sk.unit_keys) == 1 ? key : Symbol(key, :_, train)
        feeds = sk.unit_feeds[i]
        outputs = sk.unit_outputs[i]
        modes = RefineryUnitMode[]
        for (m, mode_name) in enumerate(sk.unit_modes[i])
            mode = template.modes[findfirst(x -> x.name == mode_name, template.modes)]
            base = zeros(Float64, length(outputs))
            for (s, y) in sk.unit_mode_yields[i][m]
                base[findfirst(==(s), outputs)] = y
            end
            total = sum(base)
            yields = zeros(Float64, length(feeds), length(outputs))
            for (f, stream) in enumerate(feeds)
                # Heavier, dirtier feed cracks to less light product; the mode's
                # total volumetric yield is preserved.
                density = qualities[stream, PP_Q_DENSITY]
                tilt = clamp(1.0 + (0.90 - density) * 0.35, 0.90, 1.10)
                row = [
                    base[o] *
                    (o <= length(outputs) ÷ 2 ? tilt : 2.0 - tilt) *
                    rand(rng, Uniform(0.96, 1.04)) for o in eachindex(outputs)
                ]
                scale = total / max(sum(row), 1e-9)
                yields[f, :] .= row .* scale
            end
            push!(
                modes,
                RefineryUnitMode(
                    mode_name,
                    yields,
                    mode.capacity_factor,
                    round(
                        template.operating_cost * mode.cost_factor * rand(rng, Uniform(0.85, 1.15));
                        digits=3,
                    ),
                ),
            )
        end
        push!(units, RefineryUnit(name, key, template.stage, feeds, outputs, modes))
    end

    products = RefineryProduct[]
    grade_count = Dict{Symbol, Int}()
    for i in eachindex(sk.product_keys)
        key = sk.product_keys[i]
        template = _pp_product_template(key)
        grade = get(grade_count, key, 0) + 1
        grade_count[key] = grade
        name = grade == 1 && count(==(key), sk.product_keys) == 1 ? key : Symbol(key, :_, grade)
        spec_min = fill(-Inf, PP_N_QUALITIES)
        spec_max = fill(Inf, PP_N_QUALITIES)
        for (q, bound) in template.spec_min
            spec_min[q] = bound * rand(rng, Uniform(0.98, 1.02))
        end
        for (q, bound) in template.spec_max
            spec_max[q] = bound * rand(rng, Uniform(0.98, 1.02))
        end
        # Later grades of the same product are the tighter (premium) ones.
        if grade > 1
            for q in 1:PP_N_QUALITIES
                isfinite(spec_min[q]) && (spec_min[q] *= 1.0 + 0.01 * (grade - 1))
                isfinite(spec_max[q]) && (spec_max[q] *= 1.0 - 0.01 * (grade - 1))
            end
        end
        push!(products, RefineryProduct(name, key, sk.product_components[i], spec_min, spec_max))
    end

    return RefineryFlowsheet(
        C,
        crude_names,
        crude_api,
        crude_sulfur,
        cuts,
        cut_yields,
        sk.cut_stream,
        stream_names,
        sk.stream_classes,
        sk.stream_crude,
        sk.stream_stage,
        qualities,
        units,
        products,
        sk.storable,
        sk.purchasable,
        sk.spot,
    )
end

# ---------------------------------------------------------------------------
# Multi-period data
# ---------------------------------------------------------------------------

"""
    ProcessPlanData

Time-varying data of a multi-period plan: crude economics and availability, unit
and tank capacities, purchase and spot limits, product prices and the demand
window. All volumes are thousands of barrels per period; prices are \$/bbl, so the
objective is in thousands of dollars.
"""
struct ProcessPlanData
    n_periods::Int
    period_days::Float64
    nameplate::Float64
    crude_price::Matrix{Float64}
    crude_availability::Matrix{Float64}
    crude_tank_capacity::Vector{Float64}
    crude_initial_inventory::Vector{Float64}
    cdu_capacity::Vector{Float64}
    cdu_min_throughput::Vector{Float64}
    cdu_sulfur_limit::Float64
    unit_capacity::Matrix{Float64}
    unit_min_throughput::Matrix{Float64}
    unit_switch_cost::Vector{Float64}
    stream_tank_capacity::Vector{Float64}
    stream_initial_inventory::Vector{Float64}
    stream_purchase_limit::Vector{Float64}
    stream_purchase_cost::Matrix{Float64}
    stream_spot_limit::Vector{Float64}
    stream_spot_price::Vector{Float64}
    stream_holding_cost::Vector{Float64}
    product_price::Matrix{Float64}
    demand_min::Matrix{Float64}
    demand_max::Matrix{Float64}
    product_tank_capacity::Vector{Float64}
    product_initial_inventory::Vector{Float64}
    product_holding_cost::Vector{Float64}
    renewable_min_fraction::Float64
    renewable_max_fraction::Float64
end

"""
    RefineryOperatingPlan

A complete primal point of the planning model: crude purchases and runs, unit
feeds (per unit, per feed, per period), the operating mode each unit runs,
blend recipes, inventories, purchases, spot sales and finished-product sales.
Stored on requested-feasible instances so the plan can be re-checked against
every row by arithmetic alone.
"""
struct RefineryOperatingPlan
    crude_buy::Matrix{Float64}
    crude_run::Matrix{Float64}
    crude_inventory::Matrix{Float64}
    unit_mode::Matrix{Int}
    unit_feed::Vector{Matrix{Float64}}
    blend::Vector{Matrix{Float64}}
    stream_inventory::Matrix{Float64}
    stream_purchase::Matrix{Float64}
    stream_spot::Matrix{Float64}
    product_sales::Matrix{Float64}
    product_inventory::Matrix{Float64}
end

"""Structural reason a requested-infeasible planning instance has no plan."""
@enum RefineryInfeasibilityKind begin
    refinery_contract_above_conversion_bound
    refinery_specification_outside_component_range
end

"""
    RefineryInfeasibilityCertificate

Solver-independent proof stored on a requested-infeasible instance.

`refinery_contract_above_conversion_bound` compares the contracted volume over
the horizon with an upper bound on everything the refinery can make: summing the
stream balances against the yield potential `M` (the largest volume of finished
product one barrel of a stream can ever become, computed backwards through the
flowsheet DAG) bounds total finished production by what the crude menu, the crude
unit and the purchased blendstocks can supply.

`refinery_specification_outside_component_range` names a product whose quality
window excludes every one of its components: with all blend coefficients of that
row strictly one-signed and blend volumes nonnegative, the row forces the whole
blend to zero, so the contract cannot be served out of production or the opening
tank.
"""
struct RefineryInfeasibilityCertificate
    kind::RefineryInfeasibilityKind
    product::Int
    quality::Int
    is_maximum_specification::Bool
    achievable::Float64
    required::Float64
end

# ---------------------------------------------------------------------------
# Yield potential: the dual multipliers behind the aggregate certificate
# ---------------------------------------------------------------------------

"""
    _pp_yield_potential(fs) -> (stream_potential, crude_potential)

`stream_potential[s]` is the largest volume of finished product a barrel of
stream `s` can ever become, and `crude_potential[c]` the same for a barrel of
crude `c`. Computed backwards through the flowsheet stages: a stream is worth at
least one if it may be blended into a product, and at least the yield-weighted
potential of the outputs of any unit it can feed (over every mode). Because each
stream's value dominates the value of all of its sinks, multiplying the stream
balances by these numbers and summing telescopes into an upper bound on total
finished production.
"""
function _pp_yield_potential(fs::RefineryFlowsheet)
    S = n_streams(fs)
    potential = zeros(Float64, S)
    for product in fs.products, s in product.components
        potential[s] = 1.0
    end
    stages = isempty(fs.units) ? Int[] : sort(unique(u.stage for u in fs.units))
    for stage in reverse(stages)
        for unit in fs.units
            unit.stage == stage || continue
            for (f, s) in enumerate(unit.feeds)
                value = 0.0
                for mode in unit.modes
                    mode_value = sum(
                        mode.yields[f, o] * potential[unit.outputs[o]] for
                        o in eachindex(unit.outputs)
                    )
                    value = max(value, mode_value)
                end
                potential[s] = max(potential[s], value)
            end
        end
    end
    crude_potential = [
        sum(
            fs.cut_yields[c, k] * potential[fs.cut_stream[c, k]] for k in eachindex(fs.cut_classes)
        ) for c in 1:fs.n_crudes
    ]
    return potential, crude_potential
end

"""
    _pp_production_bound(fs, data) -> Float64

Upper bound on the finished-product volume the refinery can put into the market
over the whole horizon: the yield potential of every barrel of crude it can buy
(capped by both the crude menu and the crude unit), of every purchased
blendstock, and of everything already in the tanks when the horizon opens.
"""
function _pp_production_bound(fs::RefineryFlowsheet, data::ProcessPlanData)
    potential, crude_potential = _pp_yield_potential(fs)
    supply = sum(
        crude_potential[c] *
        (sum(view(data.crude_availability, c, :)) + data.crude_initial_inventory[c]) for
        c in 1:fs.n_crudes
    )
    charge = maximum(crude_potential) * sum(data.cdu_capacity)
    bound = min(supply, charge)
    for s in fs.purchasable
        bound += potential[s] * data.stream_purchase_limit[s] * data.n_periods
    end
    for s in 1:n_streams(fs)
        bound += potential[s] * data.stream_initial_inventory[s]
    end
    bound += sum(data.product_initial_inventory)
    return bound
end

# ---------------------------------------------------------------------------
# The planted operating plan
# ---------------------------------------------------------------------------

"""
    _pp_operating_plan(rng, fs, n_periods, charge, mode_choice,
                       crude_opening, stream_opening, product_opening)

Simulate one complete refinery operation forward through the flowsheet.

Crude is charged in fixed proportions, each stream banks a small fraction of what
is available and splits the rest across the units, blends and spot sales that can
take it, and each unit converts its feed at the yields of the mode it runs. The
last sink of every stream absorbs the residual, so all balance rows hold exactly
rather than approximately. The result is a feasible point for any capacity, tank,
availability and demand window at least as wide as the plan's own usage, provided
the opening inventories are the ones passed in here.
"""
function _pp_operating_plan(
    rng::AbstractRNG,
    fs::RefineryFlowsheet,
    T::Int,
    charge::Vector{Float64},
    mode_choice::Matrix{Int},
    crude_opening::Vector{Float64},
    stream_opening::Vector{Float64},
    product_opening::Vector{Float64},
)
    C = fs.n_crudes
    S = n_streams(fs)
    U = n_units(fs)
    P = n_products(fs)

    share = rand(rng, Dirichlet(fill(2.5, C)))
    crude_hold = [rand(rng, Uniform(0.02, 0.22)) for _ in 1:C]
    stream_hold = [
        fs.stream_classes[s] in _PP_NON_STORABLE ? 0.0 : rand(rng, Uniform(0.0, 0.14)) for s in 1:S
    ]
    product_hold = [rand(rng, Uniform(0.0, 0.18)) for _ in 1:P]
    purchase_rate = Dict(s => rand(rng, Uniform(0.004, 0.030)) for s in fs.purchasable)

    # Sinks of each stream, in a fixed order: units, then products, then spot.
    unit_sinks = [Tuple{Int, Int}[] for _ in 1:S]
    for u in 1:U, (f, s) in enumerate(fs.units[u].feeds)
        push!(unit_sinks[s], (u, f))
    end
    product_sinks = [Tuple{Int, Int}[] for _ in 1:S]
    for p in 1:P, (b, s) in enumerate(fs.products[p].components)
        push!(product_sinks[s], (p, b))
    end
    weights = Vector{Vector{Float64}}(undef, S)
    for s in 1:S
        n_sink = length(unit_sinks[s]) + length(product_sinks[s]) + (s in fs.spot ? 1 : 0)
        if n_sink == 0
            weights[s] = Float64[]
            continue
        end
        alpha = vcat(
            fill(3.0, length(unit_sinks[s])),
            fill(1.6, length(product_sinks[s])),
            s in fs.spot ? [0.7] : Float64[],
        )
        weights[s] = rand(rng, Dirichlet(alpha))
    end

    crude_buy = zeros(Float64, C, T)
    crude_run = zeros(Float64, C, T)
    crude_inventory = zeros(Float64, C, T)
    unit_feed = [zeros(Float64, length(fs.units[u].feeds), T) for u in 1:U]
    blend = [zeros(Float64, length(fs.products[p].components), T) for p in 1:P]
    stream_inventory = zeros(Float64, S, T)
    stream_purchase = zeros(Float64, S, T)
    stream_spot = zeros(Float64, S, T)
    product_sales = zeros(Float64, P, T)
    product_inventory = zeros(Float64, P, T)

    # Walk every stage from the crude cuts up, so a stream is allocated only
    # after all of its producers have run and each unit sees its complete feed.
    max_stage = max(maximum(fs.stream_stage), U == 0 ? 0 : maximum(fs.units[u].stage for u in 1:U))
    stages = 0:max_stage
    streams_by_stage = Dict(
        stage => [s for s in 1:S if fs.stream_stage[s] == stage] for stage in stages
    )
    units_by_stage = Dict(
        stage => [u for u in 1:U if fs.units[u].stage == stage] for stage in stages
    )

    production = zeros(Float64, S)
    for t in 1:T
        fill!(production, 0.0)
        for c in 1:C
            run = share[c] * charge[t]
            previous = t == 1 ? crude_opening[c] : crude_inventory[c, t - 1]
            target = crude_hold[c] * run
            buy = run + target - previous
            if buy < 0.0
                target = max(previous - run, 0.0)
                buy = run + target - previous
            end
            crude_run[c, t] = run
            crude_buy[c, t] = buy
            crude_inventory[c, t] = target
            for k in eachindex(fs.cut_classes)
                production[fs.cut_stream[c, k]] += fs.cut_yields[c, k] * run
            end
        end
        for s in fs.purchasable
            stream_purchase[s, t] = purchase_rate[s] * charge[t]
        end

        for stage in stages
            for s in streams_by_stage[stage]
                previous = t == 1 ? stream_opening[s] : stream_inventory[s, t - 1]
                available = production[s] + stream_purchase[s, t] + previous
                held = stream_hold[s] * available
                stream_inventory[s, t] = held
                remaining = available - held
                sinks = length(unit_sinks[s]) + length(product_sinks[s]) + (s in fs.spot ? 1 : 0)
                if sinks == 0
                    # Nothing can take this stream: bank it instead of losing it.
                    stream_inventory[s, t] = available
                    continue
                end
                assigned = 0.0
                index = 0
                for (u, f) in unit_sinks[s]
                    index += 1
                    amount = index == sinks ? remaining - assigned : weights[s][index] * remaining
                    unit_feed[u][f, t] = amount
                    assigned += amount
                end
                for (p, b) in product_sinks[s]
                    index += 1
                    amount = index == sinks ? remaining - assigned : weights[s][index] * remaining
                    blend[p][b, t] = amount
                    assigned += amount
                end
                if s in fs.spot
                    stream_spot[s, t] = remaining - assigned
                end
            end
            for u in get(units_by_stage, stage + 1, Int[])
                unit = fs.units[u]
                mode = unit.modes[mode_choice[u, t]]
                for (f, _) in enumerate(unit.feeds)
                    flow = unit_feed[u][f, t]
                    flow == 0.0 && continue
                    for (o, out) in enumerate(unit.outputs)
                        production[out] += mode.yields[f, o] * flow
                    end
                end
            end
        end

        for p in 1:P
            made = sum(view(blend[p], :, t))
            previous = t == 1 ? product_opening[p] : product_inventory[p, t - 1]
            held = min(product_hold[p] * made, made + previous)
            product_inventory[p, t] = held
            product_sales[p, t] = made + previous - held
        end
    end

    return RefineryOperatingPlan(
        crude_buy,
        crude_run,
        crude_inventory,
        mode_choice,
        unit_feed,
        blend,
        stream_inventory,
        stream_purchase,
        stream_spot,
        product_sales,
        product_inventory,
    )
end

"""Throughput of every unit in every period under `plan`."""
function _pp_plan_throughput(fs::RefineryFlowsheet, plan::RefineryOperatingPlan)
    U = n_units(fs)
    T = size(plan.crude_run, 2)
    throughput = zeros(Float64, U, T)
    for u in 1:U, t in 1:T
        throughput[u, t] = sum(view(plan.unit_feed[u], :, t))
    end
    return throughput
end

"""
    _pp_blend_quality(fs, product, volumes) -> Vector{Float64}

Quality vector of one blend. Most volumetric properties are volume-weighted;
RVP uses the Chevron vapour-pressure blending index (`RVP^1.25`), and sulfur is
mass-weighted, matching the density factor the model puts on that row.
"""
function _pp_blend_quality(
    fs::RefineryFlowsheet, product::RefineryProduct, volumes::AbstractVector{<:Real}
)
    total = sum(volumes)
    quality = zeros(Float64, PP_N_QUALITIES)
    total <= 0.0 && return quality
    mass = sum(
        fs.qualities[s, PP_Q_DENSITY] * volumes[b] for (b, s) in enumerate(product.components)
    )
    for q in 1:PP_N_QUALITIES
        if _pp_is_weight_basis(q)
            mass <= 0.0 && continue
            quality[q] =
                sum(
                    fs.qualities[s, PP_Q_DENSITY] * fs.qualities[s, q] * volumes[b] for
                    (b, s) in enumerate(product.components)
                ) / mass
        elseif q == PP_Q_RVP
            quality[q] =
                (
                    sum(
                        fs.qualities[s, q]^1.25 * volumes[b] for
                        (b, s) in enumerate(product.components)
                    ) / total
                )^(1 / 1.25)
        else
            quality[q] =
                sum(fs.qualities[s, q] * volumes[b] for (b, s) in enumerate(product.components)) /
                total
        end
    end
    return quality
end

"""Ethanol volume and total gasoline blend volume in one period of a plan."""
function _pp_renewable_gasoline_volume(fs::RefineryFlowsheet, plan::RefineryOperatingPlan, t::Int)
    renewable = 0.0
    gasoline = 0.0
    for (p, product) in enumerate(fs.products)
        product.key in (:regular_gasoline, :premium_gasoline) || continue
        for (b, s) in enumerate(product.components)
            volume = plan.blend[p][b, t]
            gasoline += volume
            fs.stream_classes[s] == :ethanol && (renewable += volume)
        end
    end
    return renewable, gasoline
end

# ---------------------------------------------------------------------------
# Witness and certificate verification (no solver involved)
# ---------------------------------------------------------------------------

"""
    refinery_plan_satisfies(fs, data, plan; atol=1e-6) -> Bool

Re-check a planted operating plan against every row of the planning model:
crude balances and availability, crude-unit capacity and charge sulfur, stream
balances, tank capacities, unit capacities and turndown, purchase and spot
limits, product balances and tanks, the demand window, and every blend
specification.
"""
function refinery_plan_satisfies(
    fs::RefineryFlowsheet, data::ProcessPlanData, plan::RefineryOperatingPlan; atol::Float64=1e-6
)
    C = fs.n_crudes
    S = n_streams(fs)
    U = n_units(fs)
    P = n_products(fs)
    T = data.n_periods
    scale = max(1.0, data.nameplate)
    tol = atol * scale

    size(plan.crude_run) == (C, T) || return false
    size(plan.unit_mode) == (U, T) || return false
    all(>=(-tol), plan.crude_buy) || return false
    all(>=(-tol), plan.crude_run) || return false
    all(>=(-tol), plan.crude_inventory) || return false
    all(>=(-tol), plan.stream_inventory) || return false
    all(>=(-tol), plan.stream_spot) || return false
    all(>=(-tol), plan.stream_purchase) || return false
    all(>=(-tol), plan.product_sales) || return false
    all(>=(-tol), plan.product_inventory) || return false
    all(m -> all(>=(-tol), m), plan.unit_feed) || return false
    all(m -> all(>=(-tol), m), plan.blend) || return false

    # Stream incidence, so each balance row is assembled in time proportional to
    # the number of nonzeros it actually has.
    producers = [Tuple{Int, Int}[] for _ in 1:S]
    consumers = [Tuple{Int, Int}[] for _ in 1:S]
    blenders = [Tuple{Int, Int}[] for _ in 1:S]
    for u in 1:U
        for (o, s) in enumerate(fs.units[u].outputs)
            push!(producers[s], (u, o))
        end
        for (f, s) in enumerate(fs.units[u].feeds)
            push!(consumers[s], (u, f))
        end
    end
    for p in 1:P, (b, s) in enumerate(fs.products[p].components)
        push!(blenders[s], (p, b))
    end
    is_storable = falses(S)
    is_purchasable = falses(S)
    is_spot = falses(S)
    is_storable[fs.storable] .= true
    is_purchasable[fs.purchasable] .= true
    is_spot[fs.spot] .= true

    production = zeros(Float64, S)
    for t in 1:T
        fill!(production, 0.0)
        for c in 1:C
            previous = t == 1 ? data.crude_initial_inventory[c] : plan.crude_inventory[c, t - 1]
            abs(
                previous + plan.crude_buy[c, t] - plan.crude_run[c, t] - plan.crude_inventory[c, t]
            ) <= tol || return false
            plan.crude_buy[c, t] <= data.crude_availability[c, t] + tol || return false
            plan.crude_inventory[c, t] <= data.crude_tank_capacity[c] + tol || return false
            for k in eachindex(fs.cut_classes)
                production[fs.cut_stream[c, k]] += fs.cut_yields[c, k] * plan.crude_run[c, t]
            end
        end
        charge = sum(view(plan.crude_run, :, t))
        charge <= data.cdu_capacity[t] + tol || return false
        charge + tol >= data.cdu_min_throughput[t] || return false
        sum((fs.crude_sulfur[c] - data.cdu_sulfur_limit) * plan.crude_run[c, t] for c in 1:C) <=
        tol || return false

        for u in 1:U
            unit = fs.units[u]
            mode_index = plan.unit_mode[u, t]
            1 <= mode_index <= length(unit.modes) || return false
            mode = unit.modes[mode_index]
            throughput = sum(view(plan.unit_feed[u], :, t))
            throughput <= data.unit_capacity[u, t] * mode.capacity_factor + tol || return false
            throughput + tol >= data.unit_min_throughput[u, t] || return false
            for f in eachindex(unit.feeds)
                flow = plan.unit_feed[u][f, t]
                flow == 0.0 && continue
                for (o, out) in enumerate(unit.outputs)
                    production[out] += mode.yields[f, o] * flow
                end
            end
        end

        for s in 1:S
            consumed = 0.0
            for (u, f) in consumers[s]
                consumed += plan.unit_feed[u][f, t]
            end
            for (p, b) in blenders[s]
                consumed += plan.blend[p][b, t]
            end
            previous = t == 1 ? data.stream_initial_inventory[s] : plan.stream_inventory[s, t - 1]
            balance =
                production[s] + plan.stream_purchase[s, t] + previous - consumed -
                plan.stream_spot[s, t] - plan.stream_inventory[s, t]
            abs(balance) <= tol || return false
            plan.stream_inventory[s, t] <= data.stream_tank_capacity[s] + tol || return false
            plan.stream_purchase[s, t] <= data.stream_purchase_limit[s] + tol || return false
            plan.stream_spot[s, t] <= data.stream_spot_limit[s] + tol || return false
            is_purchasable[s] || plan.stream_purchase[s, t] <= tol || return false
            is_spot[s] || plan.stream_spot[s, t] <= tol || return false
            is_storable[s] || plan.stream_inventory[s, t] <= tol || return false
        end

        for p in 1:P
            product = fs.products[p]
            made = sum(view(plan.blend[p], :, t))
            previous = t == 1 ? data.product_initial_inventory[p] : plan.product_inventory[p, t - 1]
            abs(previous + made - plan.product_sales[p, t] - plan.product_inventory[p, t]) <= tol ||
                return false
            plan.product_sales[p, t] + tol >= data.demand_min[p, t] || return false
            plan.product_sales[p, t] <= data.demand_max[p, t] + tol || return false
            plan.product_inventory[p, t] <= data.product_tank_capacity[p] + tol || return false
            made <= tol && continue
            quality = _pp_blend_quality(fs, product, view(plan.blend[p], :, t))
            for q in 1:PP_N_QUALITIES
                row_tol = atol * max(1.0, abs(quality[q])) * max(1.0, made)
                if isfinite(product.spec_min[q])
                    (quality[q] - product.spec_min[q]) * made + row_tol >= 0 || return false
                end
                if isfinite(product.spec_max[q])
                    (quality[q] - product.spec_max[q]) * made <= row_tol || return false
                end
            end
        end
    end
    renewable = 0.0
    gasoline = 0.0
    for t in 1:T
        period_renewable, period_gasoline = _pp_renewable_gasoline_volume(fs, plan, t)
        renewable += period_renewable
        gasoline += period_gasoline
        period_renewable <= data.renewable_max_fraction * period_gasoline + tol || return false
    end
    renewable + tol >= data.renewable_min_fraction * gasoline || return false
    return true
end

"""
    refinery_certificate_holds(fs, data, certificate; atol=1e-6) -> Bool

Recompute a stored infeasibility certificate from the instance data and check
that it still refutes the instance. No optimization solver is used.
"""
function refinery_certificate_holds(
    fs::RefineryFlowsheet,
    data::ProcessPlanData,
    certificate::RefineryInfeasibilityCertificate;
    atol::Float64=1e-6,
)
    if certificate.kind == refinery_contract_above_conversion_bound
        certificate.product == 0 || return false
        achievable = _pp_production_bound(fs, data)
        required = sum(data.demand_min)
        scale = max(1.0, abs(achievable), abs(required))
        isapprox(certificate.achievable, achievable; rtol=1e-9, atol=atol * scale) || return false
        isapprox(certificate.required, required; rtol=1e-9, atol=atol * scale) || return false
        return achievable + atol * scale < required
    end

    p = certificate.product
    q = certificate.quality
    1 <= p <= n_products(fs) || return false
    1 <= q <= PP_N_QUALITIES || return false
    product = fs.products[p]
    bound = certificate.is_maximum_specification ? product.spec_max[q] : product.spec_min[q]
    isfinite(bound) || return false
    isapprox(certificate.required, bound; rtol=1e-9, atol=atol * max(1.0, abs(bound))) ||
        return false

    values = [fs.qualities[s, q] for s in product.components]
    achievable = certificate.is_maximum_specification ? minimum(values) : maximum(values)
    isapprox(
        certificate.achievable, achievable; rtol=1e-9, atol=atol * max(1.0, abs(achievable))
    ) || return false
    margin = atol * max(1.0, abs(achievable), abs(bound))
    if certificate.is_maximum_specification
        # Every component sits above the cap, so the row forces the blend to zero.
        achievable > bound + margin || return false
    else
        achievable + margin < bound || return false
    end
    # The blend is pinned at zero, so the contract must exceed the opening tank.
    contracted = sum(view(data.demand_min, p, :))
    contracted > data.product_initial_inventory[p] + margin || return false
    return true
end

# ---------------------------------------------------------------------------
# Instance calibration: market data, capacities and the feasibility contract
# ---------------------------------------------------------------------------

"""
    _pp_market_path(rng, T, base; volatility, seasonality, phase, period_days)

Multiplicative path around `base`: an AR(1) wander for market noise plus an
annual seasonal swing. Used for crude and product prices, product demand and
crude-unit utilisation, so a horizon has the shape planners actually optimize
against (a gasoline summer, a distillate winter, drifting crude).
"""
function _pp_market_path(
    rng::AbstractRNG,
    T::Int,
    base::Real;
    volatility::Float64=0.05,
    seasonality::Float64=0.0,
    phase::Float64=0.0,
    period_days::Float64=365.25 / max(T, 2),
)
    path = Vector{Float64}(undef, T)
    shock = 0.0
    for t in 1:T
        shock = 0.65 * shock + rand(rng, Normal(0.0, volatility))
        season = seasonality * sin(2pi * period_days * (t - 1) / 365.25 + phase)
        path[t] = max(base * (1.0 + shock + season), 0.05 * base)
    end
    return path
end

"""Per-stream production in every period implied by a plan (crude cuts plus unit yields)."""
function _pp_stream_production(fs::RefineryFlowsheet, plan::RefineryOperatingPlan)
    S = n_streams(fs)
    T = size(plan.crude_run, 2)
    production = zeros(Float64, S, T)
    for t in 1:T
        for c in 1:fs.n_crudes, k in eachindex(fs.cut_classes)
            production[fs.cut_stream[c, k], t] += fs.cut_yields[c, k] * plan.crude_run[c, t]
        end
        for u in 1:n_units(fs)
            unit = fs.units[u]
            mode = unit.modes[plan.unit_mode[u, t]]
            for f in eachindex(unit.feeds)
                flow = plan.unit_feed[u][f, t]
                flow == 0.0 && continue
                for (o, out) in enumerate(unit.outputs)
                    production[out, t] += mode.yields[f, o] * flow
                end
            end
        end
    end
    return production
end

"""Seasonal phase of a product family: gasoline peaks in summer, heating oil in winter."""
function _pp_demand_phase(key::Symbol)
    key in (:regular_gasoline, :premium_gasoline) && return 0.0
    key in (:heating_gasoil, :fuel_oil) && return Float64(pi)
    key == :jet_a1 && return 0.4
    return 0.0
end

"""
    _pp_impossible_specification!(rng, fs, data, nameplate) -> certificate or nothing

Tighten one existing product specification past every component that could go
into that blend, and contract the grade so the tightening bites. With all
coefficients of that row one-signed and blend volumes nonnegative, the row pins
the whole blend at zero for every period, so a positive contract cannot be met.
Returns the certificate, or `nothing` when no product carries a usable spec.
"""
function _pp_impossible_specification!(
    rng::AbstractRNG, fs::RefineryFlowsheet, data::ProcessPlanData, nameplate::Float64
)
    P = n_products(fs)
    P == 0 && return nothing
    for p in randperm(rng, P)
        product = fs.products[p]
        candidates = Tuple{Int, Bool}[]
        for q in 1:PP_N_QUALITIES
            values = [fs.qualities[s, q] for s in product.components]
            if isfinite(product.spec_max[q]) && minimum(values) > 1e-9
                push!(candidates, (q, true))
            end
            if isfinite(product.spec_min[q]) && maximum(values) > 1e-9
                push!(candidates, (q, false))
            end
        end
        isempty(candidates) && continue
        q, is_max = candidates[rand(rng, 1:length(candidates))]
        values = [fs.qualities[s, q] for s in product.components]
        # The certificate argues from the tightened bound alone. Withdraw an
        # opposing bound the tightening would leave on the wrong side of it: an
        # empty published window is not a quality specification, and it would add
        # a second, unrecorded reason for the infeasibility.
        if is_max
            achievable = minimum(values)
            bound = achievable * rand(rng, Uniform(0.80, 0.94))
            product.spec_max[q] = bound
            product.spec_min[q] > bound && (product.spec_min[q] = -Inf)
        else
            achievable = maximum(values)
            bound = achievable * rand(rng, Uniform(1.06, 1.25))
            product.spec_min[q] = bound
            product.spec_max[q] < bound && (product.spec_max[q] = Inf)
        end
        # The grade must actually be contracted, out of an empty opening tank.
        data.product_initial_inventory[p] = 0.0
        floor_demand = 0.01 * nameplate
        for t in 1:data.n_periods
            data.demand_min[p, t] = max(data.demand_min[p, t], floor_demand)
            data.demand_max[p, t] = max(data.demand_max[p, t], data.demand_min[p, t] * 1.05)
        end
        return RefineryInfeasibilityCertificate(
            refinery_specification_outside_component_range, p, q, is_max, achievable, bound
        )
    end
    return nothing
end

"""
    _pp_starve_contracts!(rng, fs, data) -> certificate

Raise the contracted volumes until they exceed everything the refinery could
possibly make over the horizon, and return the matching aggregate certificate.
"""
function _pp_starve_contracts!(rng::AbstractRNG, fs::RefineryFlowsheet, data::ProcessPlanData)
    bound = _pp_production_bound(fs, data)
    required = sum(data.demand_min)
    wanted = bound * rand(rng, Uniform(1.10, 1.45))
    P = n_products(fs)
    T = data.n_periods
    if required <= 0.0
        share = wanted / (P * T)
        data.demand_min .= share
    else
        data.demand_min .*= wanted / required
    end
    for p in 1:P, t in 1:T
        data.demand_max[p, t] = max(data.demand_max[p, t], data.demand_min[p, t] * 1.05)
    end
    return RefineryInfeasibilityCertificate(
        refinery_contract_above_conversion_bound, 0, 0, false, bound, sum(data.demand_min)
    )
end

"""
    _pp_plan_instance(rng, fs, n_periods, status, mode_choice)
        -> (data, plan, certificate)

Draw the market and asset data of one multi-period instance around a planted
operation of the flowsheet.

A nominal operation is simulated first (crude slate, routing, blend recipes and
inventory build), and the instance data is then placed around it:

- `feasible`: every capacity, tank, availability, purchase and spot limit is
  sized above the plan's own usage, the demand window brackets the plan's sales,
  and each blend specification is relaxed to the quality the plan's recipe
  actually achieves. The plan is therefore a feasible point, and is returned as
  the witness.
- `unknown`: assets are sized from engineering design rules (a unit's typical
  fraction of crude charge) rather than from the plan, contracts are drawn from a
  market view that straddles what the plan produced, and each quality window is
  stated at the edge of what the configuration supports — sometimes just inside
  it, sometimes just outside, never past the single best component. Whether the
  slate, the units and the specifications can serve all the contracts together is
  left genuinely open.
- `infeasible`: the `unknown` data is then broken in one of two auditable ways —
  contracts beyond the conversion bound, or a specification outside the range of
  every admissible component.
"""
function _pp_plan_instance(
    rng::AbstractRNG,
    fs::RefineryFlowsheet,
    T::Int,
    status::FeasibilityStatus,
    mode_choice::Matrix{Int};
    conditional_rates::Bool=false,
    unknown_position::Float64=0.5,
)
    C = fs.n_crudes
    S = n_streams(fs)
    U = n_units(fs)
    P = n_products(fs)
    0.0 <= unknown_position <= 1.0 || throw(ArgumentError("unknown_position must lie in [0, 1]"))
    market_position = status == unknown ? unknown_position : 0.5
    supply_factor = 0.78 + 0.52 * market_position
    demand_factor = 1.22 - 0.52 * market_position

    period_days = rand(rng, [7.0, 14.0, 30.0])
    calendar_phase = rand(rng, Uniform(0.0, 2pi))
    nameplate = rand(rng, Uniform(60.0, 380.0)) * period_days   # kbbl per period
    utilisation = _pp_market_path(
        rng,
        T,
        rand(rng, Uniform(0.84, 0.97));
        volatility=0.03,
        seasonality=0.04,
        phase=calendar_phase,
        period_days=period_days,
    )
    charge = [nameplate * clamp(utilisation[t], 0.45, 1.0) for t in 1:T]

    # Opening inventories are drawn from the scale of a zero-stock probe run, and
    # the plan is then replayed against them.
    zeros_c, zeros_s, zeros_p = zeros(C), zeros(S), zeros(P)
    probe = _pp_operating_plan(rng, fs, T, charge, mode_choice, zeros_c, zeros_s, zeros_p)
    probe_production = _pp_stream_production(fs, probe)
    is_storable = falses(S)
    is_storable[fs.storable] .= true
    crude_opening = [charge[1] / C * rand(rng, Uniform(0.0, 0.35)) for _ in 1:C]
    stream_opening = [
        if is_storable[s]
            maximum(view(probe_production, s, :)) * rand(rng, Uniform(0.0, 0.30))
        else
            0.0
        end for s in 1:S
    ]
    product_opening = [
        maximum(view(probe.product_sales, p, :)) * rand(rng, Uniform(0.0, 0.25)) for p in 1:P
    ]
    plan = _pp_operating_plan(
        rng, fs, T, charge, mode_choice, crude_opening, stream_opening, product_opening
    )

    throughput = _pp_plan_throughput(fs, plan)
    production = _pp_stream_production(fs, plan)
    planned = status == feasible

    # --- crude economics -------------------------------------------------
    crude_price = zeros(Float64, C, T)
    crude_availability = zeros(Float64, C, T)
    crude_tank_capacity = zeros(Float64, C)
    energy_factor = _pp_market_path(rng, T, 1.0; volatility=0.035, period_days=period_days)
    for c in 1:C
        # Crude is priced off a marker at 38 API and 0.4 wt% sulfur, with the
        # usual light/heavy and sweet/sour differentials: roughly \$0.35 a barrel
        # per API degree and \$3.5 per wt% of sulfur, which puts an extra-heavy
        # sour crude some \$20 below a condensate rather than \$45 below it.
        base = 80.0 + 0.35 * (fs.crude_api[c] - 38.0) - 3.5 * (fs.crude_sulfur[c] - 0.4)
        idiosyncratic = _pp_market_path(rng, T, 1.0; volatility=0.012, period_days=period_days)
        crude_price[c, :] .= base * rand(rng, Uniform(0.95, 1.06)) .* energy_factor .* idiosyncratic
        # Term cargoes are sized against the whole charge, not per crude, so a
        # long crude menu does not silently multiply the supply available.
        cargo = nameplate * rand(rng, Uniform(0.55, 1.70)) / C
        for t in 1:T
            offered = cargo * rand(rng, Uniform(0.85, 1.20))
            !planned && (offered *= supply_factor)
            crude_availability[c, t] = if planned
                max(offered, plan.crude_buy[c, t] * rand(rng, Uniform(1.05, 1.45)))
            else
                offered
            end
        end
        peak = maximum(view(plan.crude_inventory, c, :))
        design = nameplate * rand(rng, Uniform(0.08, 0.35))
        crude_tank_capacity[c] = if planned
            max(design, peak * rand(rng, Uniform(1.10, 1.60)), crude_opening[c] * 1.05)
        else
            max(design, crude_opening[c] * 1.05)
        end
    end

    # --- crude unit and conversion capacity ------------------------------
    cdu_capacity = Vector{Float64}(undef, T)
    cdu_min_throughput = Vector{Float64}(undef, T)
    design_cdu = nameplate * rand(rng, Uniform(0.98, 1.08))
    for t in 1:T
        cdu_capacity[t] = if planned
            max(design_cdu, charge[t] * rand(rng, Uniform(1.02, 1.20)))
        else
            design_cdu * rand(rng, Uniform(0.90, 1.05)) * (0.88 + 0.22 * market_position)
        end
        cdu_min_throughput[t] = min(charge[t] * rand(rng, Uniform(0.0, 0.75)), cdu_capacity[t])
    end

    unit_capacity = zeros(Float64, U, T)
    unit_min_throughput = zeros(Float64, U, T)
    unit_switch_cost = zeros(Float64, U)
    for u in 1:U
        template = _pp_unit_template(fs.units[u].key)
        design = nameplate * template.capacity_fraction * rand(rng, Uniform(0.85, 1.20))
        unit_switch_cost[u] = round(rand(rng, Uniform(40.0, 400.0)); digits=1)
        # A turndown row on a unit that can be idled needs an on/off decision to
        # gate it, which a pure LP cannot express. With run indicators available
        # (`conditional_rates`) every unit can therefore carry a minimum rate;
        # without them, only the primary treating units, which run whenever the
        # crude unit does.
        eligible = conditional_rates || fs.units[u].key in _PP_CONTINUOUS_UNITS
        turndown = if eligible && rand(rng) < (conditional_rates ? 0.75 : 0.6)
            rand(rng, Uniform(0.10, 0.45))
        else
            0.0
        end
        availability = ones(Float64, T)
        if T >= 4 && rand(rng) < 0.45
            outage_length = rand(rng, 1:min(3, T - 2))
            outage_start = rand(rng, 2:(T - outage_length))
            availability[outage_start:(outage_start + outage_length - 1)] .= rand(
                rng, Uniform(0.35, 0.70)
            )
        end
        for t in 1:T
            factor = fs.units[u].modes[mode_choice[u, t]].capacity_factor
            required = throughput[u, t] / factor
            available_design = design * availability[t]
            unit_capacity[u, t] = if planned
                max(available_design, required * rand(rng, Uniform(1.05, 1.40)))
            else
                available_design * rand(rng, Uniform(0.88, 1.15)) * (0.82 + 0.38 * market_position)
            end
            unit_min_throughput[u, t] = min(
                throughput[u, t] * turndown, unit_capacity[u, t] * factor * 0.9
            )
        end
    end

    # Whatever the turndown rows force through the plant has to have somewhere to
    # go. Bound the volume of each stream that a minimum rate can push out, so
    # the spot outlets below are always wide enough to take it.
    forced = zeros(Float64, S)
    period_forced = zeros(Float64, S)
    for t in 1:T
        for c in 1:C, k in eachindex(fs.cut_classes)
            s = fs.cut_stream[c, k]
            forced[s] = max(forced[s], fs.cut_yields[c, k] * cdu_min_throughput[t])
        end
        # Parallel trains hold their minimum rates in the same period, so what a
        # shared stream is forced to take is the sum over the units that make it,
        # not the largest single contribution.
        fill!(period_forced, 0.0)
        for u in 1:U
            unit_min_throughput[u, t] <= 0.0 && continue
            unit = fs.units[u]
            for (o, out) in enumerate(unit.outputs)
                best = maximum(mode.yields[f, o] for mode in unit.modes, f in eachindex(unit.feeds))
                period_forced[out] += best * unit_min_throughput[u, t]
            end
        end
        for s in 1:S
            forced[s] = max(forced[s], period_forced[s])
        end
    end

    charge_sulfur = maximum(
        sum(fs.crude_sulfur[c] * plan.crude_run[c, t] for c in 1:C) / max(charge[t], 1e-9) for
        t in 1:T
    )
    cdu_sulfur_limit =
        planned ? charge_sulfur * rand(rng, Uniform(1.02, 1.20)) : rand(rng, Uniform(1.4, 3.2))

    # --- intermediate tanks, purchases and spot sales --------------------
    stream_tank_capacity = zeros(Float64, S)
    stream_initial_inventory = copy(stream_opening)
    stream_purchase_limit = zeros(Float64, S)
    stream_purchase_cost = zeros(Float64, S, T)
    stream_spot_limit = zeros(Float64, S)
    stream_spot_price = zeros(Float64, S)
    stream_holding_cost = zeros(Float64, S)
    for s in 1:S
        typical = maximum(view(production, s, :))
        if is_storable[s]
            peak = maximum(view(plan.stream_inventory, s, :))
            design = max(typical, 0.002 * nameplate) * rand(rng, Uniform(0.25, 1.30))
            stream_tank_capacity[s] = if planned
                max(design, peak * rand(rng, Uniform(1.15, 1.70)), stream_opening[s] * 1.05)
            else
                max(design, stream_opening[s] * 1.05)
            end
            stream_holding_cost[s] = round(rand(rng, Uniform(0.10, 0.85)); digits=3)
        end
        if s in fs.purchasable
            class = fs.stream_classes[s]
            index = findfirst(==(class), _PP_PURCHASE_CLASSES)
            base = index === nothing ? 60.0 : _PP_PURCHASE_COST[index]
            stream_purchase_cost[s, :] .=
                base * rand(rng, Uniform(0.92, 1.10)) .* energy_factor .* _pp_market_path(
                    rng,
                    T,
                    1.0;
                    volatility=0.02,
                    seasonality=0.03,
                    phase=calendar_phase,
                    period_days=period_days,
                )
            offered = nameplate * rand(rng, Uniform(0.01, 0.06))
            !planned && (offered *= 0.80 + 0.45 * market_position)
            planned_purchase = maximum(view(plan.stream_purchase, s, :))
            stream_purchase_limit[s] =
                planned ? max(offered, planned_purchase * rand(rng, Uniform(1.10, 1.80))) : offered
        end
        if s in fs.spot
            offered = max(typical, 0.002 * nameplate) * rand(rng, Uniform(0.35, 1.20))
            # Refinery fuel gas is burned in the plant's own furnaces and coke,
            # LPG and slurry move on deep commodity markets, so those outlets are
            # not the binding ones.
            fs.stream_classes[s] in (:fuel_gas, :coke, :lpg, :lpg_cut, :slurry) &&
                (offered *= rand(rng, Uniform(1.6, 3.5)))
            planned_spot = maximum(view(plan.stream_spot, s, :))
            stream_spot_limit[s] = max(
                offered,
                1.05 * forced[s],
                planned ? planned_spot * rand(rng, Uniform(1.05, 1.60)) : 0.0,
            )
        end
    end

    # --- finished products ------------------------------------------------
    product_price = zeros(Float64, P, T)
    demand_min = zeros(Float64, P, T)
    demand_max = zeros(Float64, P, T)
    product_tank_capacity = zeros(Float64, P)
    product_initial_inventory = copy(product_opening)
    product_holding_cost = zeros(Float64, P)
    for p in 1:P
        template = _pp_product_template(fs.products[p].key)
        phase = calendar_phase + _pp_demand_phase(fs.products[p].key)
        product_price[p, :] .=
            template.price * rand(rng, Uniform(0.94, 1.08)) .* energy_factor .* _pp_market_path(
                rng,
                T,
                1.0;
                volatility=0.025,
                seasonality=0.05,
                phase=phase,
                period_days=period_days,
            )
        market = _pp_market_path(
            rng, T, 1.0; volatility=0.06, seasonality=0.12, phase=phase, period_days=period_days
        )
        product_holding_cost[p] = round(rand(rng, Uniform(0.15, 1.00)); digits=3)
        peak = maximum(view(plan.product_inventory, p, :))
        design = maximum(view(plan.product_sales, p, :)) * rand(rng, Uniform(0.15, 0.60))
        product_tank_capacity[p] = if planned
            max(design, peak * rand(rng, Uniform(1.15, 1.80)), product_opening[p] * 1.05)
        else
            max(design, product_opening[p] * 1.05)
        end
        contract = rand(rng, Uniform(0.45, 0.90))
        for t in 1:T
            sales = plan.product_sales[p, t]
            if planned
                demand_min[p, t] = sales * contract * min(market[t], 1.0)
                demand_max[p, t] = max(
                    sales * rand(rng, Uniform(1.05, 1.60)), demand_min[p, t] * 1.05
                )
            else
                demand_min[p, t] =
                    sales * market[t] * rand(rng, Uniform(0.70, 1.40)) * demand_factor
                demand_max[p, t] = max(
                    sales * market[t] * rand(rng, Uniform(1.00, 1.70)), demand_min[p, t] * 1.05
                )
            end
        end
    end

    # An intermediate stream trades at a discount to the cheapest grade it could
    # have gone into; refinery fuel, LPG, coke and slurry trade on their own
    # (much lower) markets.
    blend_value = zeros(Float64, S)
    has_outlet = falses(S)
    for (p, product) in enumerate(fs.products)
        average = sum(view(product_price, p, :)) / T
        for s in product.components
            blend_value[s] = has_outlet[s] ? min(blend_value[s], average) : average
            has_outlet[s] = true
        end
    end
    for s in fs.spot
        class = fs.stream_classes[s]
        base = if hasproperty(_PP_SPOT_PRICE, class)
            getproperty(_PP_SPOT_PRICE, class)
        elseif has_outlet[s]
            blend_value[s] * rand(rng, Uniform(0.55, 0.85))
        else
            45.0
        end
        stream_spot_price[s] = round(base * rand(rng, Uniform(0.90, 1.10)); digits=2)
    end

    renewable_total = 0.0
    gasoline_total = 0.0
    maximum_period_fraction = 0.0
    for t in 1:T
        renewable, gasoline = _pp_renewable_gasoline_volume(fs, plan, t)
        renewable_total += renewable
        gasoline_total += gasoline
        gasoline > 0 &&
            (maximum_period_fraction = max(maximum_period_fraction, renewable / gasoline))
    end
    has_ethanol = any(==(:ethanol), fs.stream_classes)
    observed_fraction = gasoline_total > 0 ? renewable_total / gasoline_total : 0.0
    renewable_min_fraction = if !has_ethanol || gasoline_total <= 0
        0.0
    elseif planned
        observed_fraction * rand(rng, Uniform(0.45, 0.85))
    else
        rand(rng, Uniform(0.04, 0.10)) * demand_factor
    end
    renewable_max_fraction = if !has_ethanol
        1.0
    elseif planned
        max(0.15, 1.05 * maximum_period_fraction)
    else
        rand(rng, Uniform(0.12, 0.18)) * (0.90 + 0.40 * market_position)
    end
    renewable_min_fraction = min(renewable_min_fraction, 0.95 * renewable_max_fraction)

    data = ProcessPlanData(
        T,
        period_days,
        nameplate,
        crude_price,
        crude_availability,
        crude_tank_capacity,
        crude_opening,
        cdu_capacity,
        cdu_min_throughput,
        cdu_sulfur_limit,
        unit_capacity,
        unit_min_throughput,
        unit_switch_cost,
        stream_tank_capacity,
        stream_initial_inventory,
        stream_purchase_limit,
        stream_purchase_cost,
        stream_spot_limit,
        stream_spot_price,
        stream_holding_cost,
        product_price,
        demand_min,
        demand_max,
        product_tank_capacity,
        product_initial_inventory,
        product_holding_cost,
        renewable_min_fraction,
        renewable_max_fraction,
    )

    certificate = nothing
    if status == feasible
        _pp_settle_specifications!(rng, fs, plan; slack_low=0.005, slack_high=0.05)
    else
        slack_low = status == unknown ? -0.04 + 0.06 * market_position : -0.03
        _pp_settle_specifications!(rng, fs, plan; slack_low=slack_low, slack_high=0.06)
    end
    if status == infeasible
        certificate =
            rand(rng) < 0.5 ? _pp_impossible_specification!(rng, fs, data, nameplate) : nothing
        certificate === nothing && (certificate = _pp_starve_contracts!(rng, fs, data))
    end
    return data, plan, certificate
end

"""
    _pp_settle_specifications!(rng, fs, plan; slack_low, slack_high)

Reconcile the published quality windows with what the planted recipe achieves.

A published bound the recipe already meets is left alone — that is the ordinary
case, a grade a refinery is configured to make. A bound the recipe misses is
re-stated relative to the quality the recipe reaches, offset by a slack drawn
from `[slack_low, slack_high]` (as a fraction of that quality's magnitude, so
signed properties such as cold flow move in the right direction).

Positive slack states the window the plan can honour, which is what makes a
requested-feasible instance feasible. A range straddling zero states a window at
the edge of what the configuration supports, sometimes just inside it and
sometimes just outside — and the solver can blend more sharply than the plan's
fixed routing does, so which way it falls is genuinely open.
"""
function _pp_settle_specifications!(
    rng::AbstractRNG,
    fs::RefineryFlowsheet,
    plan::RefineryOperatingPlan;
    slack_low::Float64,
    slack_high::Float64,
)
    T = size(plan.crude_run, 2)
    for (p, product) in enumerate(fs.products)
        achieved_min = fill(Inf, PP_N_QUALITIES)
        achieved_max = fill(-Inf, PP_N_QUALITIES)
        for t in 1:T
            volumes = view(plan.blend[p], :, t)
            sum(volumes) <= 0.0 && continue
            quality = _pp_blend_quality(fs, product, volumes)
            for q in 1:PP_N_QUALITIES
                achieved_min[q] = min(achieved_min[q], quality[q])
                achieved_max[q] = max(achieved_max[q], quality[q])
            end
        end
        for q in 1:PP_N_QUALITIES
            isfinite(achieved_min[q]) || continue
            # The single best component defines what the grade can reach at all.
            # A window is never closed past it here: a grade that cannot be made
            # in isolation is the requested-infeasible branch's business, not a
            # side effect of stating a specification.
            values = [fs.qualities[s, q] for s in product.components]
            reachable_low = minimum(values)
            reachable_high = maximum(values)
            if isfinite(product.spec_min[q]) && product.spec_min[q] > achieved_min[q]
                slack = rand(rng, Uniform(slack_low, slack_high))
                candidate = achieved_min[q] - slack * max(abs(achieved_min[q]), 1.0)
                product.spec_min[q] = min(
                    candidate, reachable_high - 0.005 * max(abs(reachable_high), 1.0)
                )
            end
            if isfinite(product.spec_max[q]) && product.spec_max[q] < achieved_max[q]
                slack = rand(rng, Uniform(slack_low, slack_high))
                candidate = achieved_max[q] + slack * max(abs(achieved_max[q]), 1.0)
                product.spec_max[q] = max(
                    candidate, reachable_low + 0.005 * max(abs(reachable_low), 1.0)
                )
            end
            # A negative slack applied to a narrow published window (ULSD density
            # is the tight one) can push the two ends past each other, leaving a
            # grade that cannot be blended at all. That is the
            # requested-infeasible branch's business, not a side effect of
            # stating a specification, so reopen the window around the quality
            # the recipe actually reaches.
            if isfinite(product.spec_min[q]) &&
                isfinite(product.spec_max[q]) &&
                product.spec_min[q] > product.spec_max[q]
                product.spec_min[q] = min(product.spec_min[q], achieved_min[q])
                product.spec_max[q] = max(product.spec_max[q], achieved_max[q])
            end
        end
    end
    return nothing
end
