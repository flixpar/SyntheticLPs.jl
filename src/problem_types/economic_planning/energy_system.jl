using JuMP
using Random
using Distributions

"""
Largest `target_variables` accepted by `EnergySystemProblem`; larger requests
raise an `ArgumentError` rather than being silently undersized (same convention
as `telecom_network_design/standard`).
"""
const ENERGY_SYSTEM_MAX_VARIABLES = 1_000_000

"""PJ produced per GW of capacity running a full year (8760 h × 3.6 TJ/MWh)."""
const ES_PJ_PER_GW_YEAR = 31.536

"""Combustion emission factors (Mt CO2 per PJ of fuel input)."""
const ES_EMISSION_FACTOR = Dict(:COA => 0.0946, :GAS => 0.0561, :PET => 0.0733)

const ES_PRIMARY = (:COA, :GAS, :OIL, :URN, :BIO)
const ES_DEMANDS = (:RH, :RA, :IP, :TP, :TF)
const ES_PROFILE_NAMES = (:flat, :heat, :appliance, :ev, :industry)

# Commodity -> module that introduces it.
const ES_COMMODITY_MODULE = Dict(
    :GAS => 1,
    :COA => 1,
    :RA => 1,
    :RH => 1,
    :URN => 2,
    :IP => 2,
    :OIL => 3,
    :PET => 3,
    :TP => 3,
    :TF => 3,
    :BIO => 4,
    :H2 => 5,
    :HET => 6,
)
const ES_COMMODITY_ORDER = (:COA, :GAS, :OIL, :URN, :BIO, :PET, :H2, :HET, :RH, :RA, :IP, :TP, :TF)

"""
Technology template: one row of the technology database. `eff` is output per
unit of input (a COP for heat pumps, service per PJ for end-use devices); `avail`
is the dispatch availability (or annual capacity factor for variable
renewables); costs are M\$ per unit of capacity (`inv`), per unit of capacity
and year (`fom`) and per unit of activity (`vom`).
"""
struct ESTemplate
    name::Symbol
    mod::Int
    kind::Symbol
    input::Symbol
    output::Symbol
    eff::NTuple{2, Float64}
    output2::Symbol
    ratio2::NTuple{2, Float64}
    avail::NTuple{2, Float64}
    profile::Symbol
    life::Int
    inv::NTuple{2, Float64}
    fom::NTuple{2, Float64}
    vom::NTuple{2, Float64}
    credit::NTuple{2, Float64}
    growth::Float64
    potential::Bool
    classes::Bool
    incumbent::Bool
    share::NTuple{2, Float64}
end

function _es_t(
    name,
    mod,
    kind,
    input,
    output,
    eff,
    avail,
    profile,
    life,
    inv,
    fom,
    vom;
    output2=:none,
    ratio2=(0.0, 0.0),
    credit=(0.0, 0.0),
    growth=0.1,
    potential=false,
    classes=false,
    incumbent=true,
    share=(0.0, 0.0),
)
    return ESTemplate(
        name,
        mod,
        kind,
        input,
        output,
        eff,
        output2,
        ratio2,
        avail,
        profile,
        life,
        inv,
        fom,
        vom,
        credit,
        growth,
        potential,
        classes,
        incumbent,
        share,
    )
end

# `share` = reference share (first period, last period) within the template's
# production group: end-use devices per demand, H2 producers, heat producers,
# dispatchable generators (of the residual load), variable renewables (of the
# renewable target). Shares are renormalised over the templates present.
const ES_TEMPLATES = ESTemplate[
    # ---- electricity generation (capacity in GW, activity per timeslice in PJ)
    _es_t(
        :E_COAL,
        1,
        :gen,
        :COA,
        :ELC,
        (0.33, 0.45),
        (0.85, 0.92),
        :flat,
        40,
        (1800.0, 2600.0),
        (40.0, 60.0),
        (1.0, 2.0);
        credit=(0.9, 0.95),
        growth=0.05,
        share=(0.40, 0.15),
    ),
    _es_t(
        :E_GASCC,
        1,
        :gen,
        :GAS,
        :ELC,
        (0.50, 0.60),
        (0.88, 0.93),
        :flat,
        30,
        (800.0, 1100.0),
        (20.0, 30.0),
        (0.8, 1.5);
        credit=(0.9, 0.95),
        growth=0.08,
        share=(0.35, 0.40),
    ),
    _es_t(
        :E_WON,
        1,
        :gen,
        :none,
        :ELC,
        (1.0, 1.0),
        (0.25, 0.38),
        :wind,
        25,
        (1200.0, 1700.0),
        (30.0, 45.0),
        (0.0, 0.0);
        credit=(0.05, 0.15),
        growth=0.15,
        potential=true,
        classes=true,
        incumbent=false,
        share=(0.5, 0.4),
    ),
    _es_t(
        :E_SPV,
        2,
        :gen,
        :none,
        :ELC,
        (1.0, 1.0),
        (0.12, 0.22),
        :solar,
        25,
        (600.0, 1100.0),
        (10.0, 20.0),
        (0.0, 0.0);
        credit=(0.0, 0.05),
        growth=0.20,
        potential=true,
        classes=true,
        incumbent=false,
        share=(0.4, 0.45),
    ),
    _es_t(
        :E_GASGT,
        2,
        :gen,
        :GAS,
        :ELC,
        (0.30, 0.38),
        (0.90, 0.95),
        :flat,
        25,
        (450.0, 700.0),
        (10.0, 15.0),
        (2.0, 4.0);
        credit=(0.95, 1.0),
        growth=0.10,
        share=(0.05, 0.05),
    ),
    _es_t(
        :E_NUC,
        2,
        :gen,
        :URN,
        :ELC,
        (0.33, 0.36),
        (0.85, 0.92),
        :flat,
        60,
        (5000.0, 8000.0),
        (100.0, 150.0),
        (0.5, 1.0);
        credit=(0.9, 0.95),
        growth=0.05,
        potential=true,
        share=(0.15, 0.15),
    ),
    _es_t(
        :E_BIO,
        4,
        :gen,
        :BIO,
        :ELC,
        (0.30, 0.38),
        (0.80, 0.90),
        :flat,
        30,
        (2500.0, 3500.0),
        (60.0, 100.0),
        (1.0, 3.0);
        credit=(0.85, 0.9),
        growth=0.08,
        share=(0.05, 0.10),
    ),
    _es_t(
        :E_HYD,
        4,
        :gen,
        :none,
        :ELC,
        (1.0, 1.0),
        (0.35, 0.50),
        :hydro,
        80,
        (2500.0, 4000.0),
        (20.0, 40.0),
        (0.0, 0.0);
        credit=(0.5, 0.7),
        growth=0.03,
        potential=true,
        share=(0.0, 0.0),
    ),
    _es_t(
        :E_H2,
        5,
        :gen,
        :H2,
        :ELC,
        (0.50, 0.60),
        (0.90, 0.95),
        :flat,
        25,
        (700.0, 1000.0),
        (15.0, 25.0),
        (1.0, 2.0);
        credit=(0.9, 0.95),
        growth=0.15,
        incumbent=false,
        share=(0.0, 0.0),
    ),
    _es_t(
        :E_CHP,
        6,
        :gen,
        :GAS,
        :ELC,
        (0.33, 0.40),
        (0.80, 0.90),
        :flat,
        30,
        (1100.0, 1500.0),
        (30.0, 45.0),
        (1.0, 2.0);
        output2=:HET,
        ratio2=(1.0, 1.4),
        credit=(0.85, 0.9),
        growth=0.08,
        potential=true,
        share=(0.4, 0.4),
    ),
    _es_t(
        :E_WOFF,
        6,
        :gen,
        :none,
        :ELC,
        (1.0, 1.0),
        (0.38, 0.50),
        :windoff,
        25,
        (2500.0, 3500.0),
        (60.0, 90.0),
        (0.0, 0.0);
        credit=(0.1, 0.2),
        growth=0.15,
        potential=true,
        classes=true,
        incumbent=false,
        share=(0.1, 0.15),
    ),
    # ---- conversion (capacity in PJ/yr of output)
    _es_t(
        :REF,
        3,
        :annual,
        :OIL,
        :PET,
        (0.90, 0.95),
        (0.85, 0.92),
        :flat,
        40,
        (15.0, 25.0),
        (0.5, 1.0),
        (0.3, 0.6);
        growth=0.05,
        share=(1.0, 1.0),
    ),
    _es_t(
        :H2_ELY,
        5,
        :annual,
        :ELC,
        :H2,
        (0.62, 0.72),
        (0.5, 0.7),
        :flat,
        20,
        (30.0, 55.0),
        (1.0, 2.0),
        (0.1, 0.3);
        growth=0.25,
        incumbent=false,
        share=(0.2, 0.6),
    ),
    _es_t(
        :H2_SMR,
        5,
        :annual,
        :GAS,
        :H2,
        (0.68, 0.76),
        (0.85, 0.92),
        :flat,
        25,
        (15.0, 25.0),
        (0.5, 1.0),
        (0.3, 0.5);
        growth=0.08,
        share=(0.8, 0.4),
    ),
    _es_t(
        :H_BOIL_GAS,
        6,
        :annual,
        :GAS,
        :HET,
        (0.88, 0.93),
        (0.5, 0.7),
        :flat,
        25,
        (3.0, 6.0),
        (0.1, 0.2),
        (0.1, 0.3);
        growth=0.10,
        share=(0.5, 0.3),
    ),
    _es_t(
        :H_HP,
        6,
        :annual,
        :ELC,
        :HET,
        (2.5, 3.5),
        (0.5, 0.7),
        :heat,
        20,
        (20.0, 35.0),
        (0.4, 0.8),
        (0.1, 0.2);
        growth=0.15,
        incumbent=false,
        share=(0.25, 0.45),
    ),
    _es_t(
        :H_BOIL_BIO,
        6,
        :annual,
        :BIO,
        :HET,
        (0.80, 0.88),
        (0.5, 0.7),
        :flat,
        25,
        (6.0, 10.0),
        (0.2, 0.4),
        (0.2, 0.4);
        growth=0.10,
        share=(0.25, 0.25),
    ),
    # ---- end-use devices (capacity in service units per year)
    _es_t(
        :RA_STD,
        1,
        :annual,
        :ELC,
        :RA,
        (1.0, 1.0),
        (0.85, 0.95),
        :appliance,
        12,
        (20.0, 40.0),
        (0.5, 1.0),
        (0.0, 0.0);
        growth=0.10,
        classes=true,
        share=(1.0, 1.0),
    ),
    _es_t(
        :RH_GAS,
        1,
        :annual,
        :GAS,
        :RH,
        (0.85, 0.95),
        (0.30, 0.40),
        :flat,
        20,
        (30.0, 50.0),
        (1.0, 2.0),
        (0.0, 0.0);
        growth=0.08,
        classes=true,
        share=(0.55, 0.35),
    ),
    _es_t(
        :RH_HP,
        1,
        :annual,
        :ELC,
        :RH,
        (2.5, 4.0),
        (0.30, 0.40),
        :heat,
        18,
        (100.0, 160.0),
        (2.0, 3.0),
        (0.0, 0.0);
        growth=0.15,
        classes=true,
        incumbent=false,
        share=(0.10, 0.35),
    ),
    _es_t(
        :IP_GAS,
        2,
        :annual,
        :GAS,
        :IP,
        (0.85, 0.90),
        (0.70, 0.85),
        :flat,
        25,
        (10.0, 20.0),
        (0.3, 0.6),
        (0.1, 0.2);
        growth=0.08,
        share=(0.45, 0.35),
    ),
    _es_t(
        :IP_COAL,
        2,
        :annual,
        :COA,
        :IP,
        (0.78, 0.85),
        (0.70, 0.85),
        :flat,
        30,
        (12.0, 22.0),
        (0.4, 0.7),
        (0.1, 0.3);
        growth=0.05,
        share=(0.25, 0.10),
    ),
    _es_t(
        :IP_ELC,
        2,
        :annual,
        :ELC,
        :IP,
        (0.95, 0.99),
        (0.70, 0.85),
        :industry,
        25,
        (15.0, 30.0),
        (0.4, 0.8),
        (0.1, 0.2);
        growth=0.12,
        classes=true,
        share=(0.20, 0.35),
    ),
    _es_t(
        :TP_ICE,
        3,
        :annual,
        :PET,
        :TP,
        (0.55, 0.75),
        (0.85, 0.95),
        :flat,
        15,
        (900.0, 1400.0),
        (20.0, 40.0),
        (0.0, 0.0);
        growth=0.08,
        classes=true,
        share=(0.90, 0.50),
    ),
    _es_t(
        :TP_BEV,
        3,
        :annual,
        :ELC,
        :TP,
        (1.8, 2.4),
        (0.85, 0.95),
        :ev,
        15,
        (1100.0, 1700.0),
        (15.0, 30.0),
        (0.0, 0.0);
        growth=0.20,
        classes=true,
        incumbent=false,
        share=(0.05, 0.35),
    ),
    _es_t(
        :TF_DSL,
        3,
        :annual,
        :PET,
        :TF,
        (0.9, 1.3),
        (0.85, 0.95),
        :flat,
        12,
        (150.0, 250.0),
        (5.0, 10.0),
        (0.0, 0.0);
        growth=0.08,
        classes=true,
        share=(0.92, 0.60),
    ),
    _es_t(
        :TF_BEV,
        3,
        :annual,
        :ELC,
        :TF,
        (2.0, 3.0),
        (0.85, 0.95),
        :ev,
        12,
        (220.0, 350.0),
        (4.0, 8.0),
        (0.0, 0.0);
        growth=0.20,
        incumbent=false,
        share=(0.02, 0.20),
    ),
    _es_t(
        :RH_OIL,
        3,
        :annual,
        :PET,
        :RH,
        (0.80, 0.88),
        (0.30, 0.40),
        :flat,
        20,
        (30.0, 45.0),
        (1.0, 2.0),
        (0.0, 0.0);
        growth=0.05,
        share=(0.15, 0.05),
    ),
    _es_t(
        :RH_BIO,
        4,
        :annual,
        :BIO,
        :RH,
        (0.70, 0.82),
        (0.30, 0.40),
        :flat,
        20,
        (50.0, 80.0),
        (1.5, 2.5),
        (0.0, 0.0);
        growth=0.10,
        share=(0.08, 0.10),
    ),
    _es_t(
        :IP_BIO,
        4,
        :annual,
        :BIO,
        :IP,
        (0.72, 0.80),
        (0.70, 0.85),
        :flat,
        25,
        (15.0, 25.0),
        (0.4, 0.8),
        (0.1, 0.3);
        growth=0.10,
        share=(0.05, 0.10),
    ),
    _es_t(
        :IP_H2,
        5,
        :annual,
        :H2,
        :IP,
        (0.85, 0.90),
        (0.70, 0.85),
        :flat,
        25,
        (20.0, 35.0),
        (0.5, 0.9),
        (0.1, 0.2);
        growth=0.20,
        incumbent=false,
        share=(0.0, 0.10),
    ),
    _es_t(
        :TP_FCEV,
        5,
        :annual,
        :H2,
        :TP,
        (0.9, 1.2),
        (0.85, 0.95),
        :flat,
        15,
        (1300.0, 2000.0),
        (20.0, 40.0),
        (0.0, 0.0);
        growth=0.20,
        incumbent=false,
        share=(0.0, 0.05),
    ),
    _es_t(
        :TF_H2,
        5,
        :annual,
        :H2,
        :TF,
        (1.2, 1.6),
        (0.85, 0.95),
        :flat,
        12,
        (250.0, 400.0),
        (5.0, 10.0),
        (0.0, 0.0);
        growth=0.20,
        incumbent=false,
        share=(0.0, 0.10),
    ),
    _es_t(
        :RH_DH,
        6,
        :annual,
        :HET,
        :RH,
        (0.93, 0.97),
        (0.30, 0.40),
        :flat,
        30,
        (40.0, 70.0),
        (1.0, 2.0),
        (0.0, 0.0);
        growth=0.08,
        share=(0.08, 0.12),
    ),
    _es_t(
        :RH_RES,
        6,
        :annual,
        :ELC,
        :RH,
        (0.99, 1.0),
        (0.30, 0.40),
        :heat,
        20,
        (10.0, 20.0),
        (0.3, 0.6),
        (0.0, 0.0);
        growth=0.10,
        share=(0.04, 0.03),
    ),
    _es_t(
        :TP_RAIL,
        6,
        :annual,
        :ELC,
        :TP,
        (3.0, 5.0),
        (0.80, 0.90),
        :flat,
        40,
        (2000.0, 3000.0),
        (40.0, 80.0),
        (0.0, 0.0);
        growth=0.05,
        potential=true,
        share=(0.05, 0.10),
    ),
    _es_t(
        :TF_RAIL,
        6,
        :annual,
        :ELC,
        :TF,
        (4.0, 6.0),
        (0.80, 0.90),
        :flat,
        40,
        (300.0, 500.0),
        (8.0, 15.0),
        (0.0, 0.0);
        growth=0.05,
        potential=true,
        share=(0.08, 0.10),
    ),
]

# Class multipliers: efficiency classes for devices, resource-quality classes
# for variable renewables (best sites first).
const ES_CLASS_SHARE = (0.55, 0.25, 0.12, 0.08)
const ES_DEVICE_EFF = (1.0, 1.12, 1.24, 1.36)
const ES_DEVICE_COST = (1.0, 1.25, 1.55, 1.9)
const ES_SITE_QUALITY = (1.0, 0.88, 0.77, 0.67)
const ES_SITE_COST = (1.0, 1.04, 1.08, 1.12)
const ES_MAX_CLASSES = 4
const ES_MAX_DOMESTIC_STEPS = 4

"""
    EnergySystemWitness

Planted feasible plan in the model's flat column layout: `act` (activity),
`cap` (installed capacity), `ncap` (new capacity) and interconnector `flow`
(zero in the reference plan, which serves every region from its own resources).
"""
struct EnergySystemWitness
    act::Vector{Float64}
    cap::Vector{Float64}
    ncap::Vector{Float64}
    flow::Vector{Float64}
end

"""
    EnergySystemCertificate

Farkas certificate from LP rows and bounds only. `commodity_value[r, c]` and
`electricity_value[r]` are dual "values" `π ≥ 0` of the commodity balances of
every period in `periods` (each scaled by `period_weight`), with
`emission_weight` `θ` on the emission row of the period (`:emission_cap`), on
the cumulative budget row (`:carbon_budget`), or zero (`:supply_shortfall`).
They are the greatest fixed point of `π_out ≤ (θ e_p + Σ π_in · in_p) / out_p`
over every process without a capacity potential (fossil and conversion plants,
end-use devices, interconnectors), with fossil fuels valued 0 under an emission
argument and primary energy valued 1 under a supply argument. Every unbounded
column therefore has nonpositive aggregated coefficient. Processes whose value
exceeds their emission charge — supply steps (activity bounds), and potential-
limited plants — are paid for through the multipliers listed in
`capact_multipliers` (capacity–activity rows), `transfer_multipliers`
(capacity-transfer rows) and `growth_multipliers` (market-growth rows), which
bound their capacity by its potential or by the build-out that the growth rows
allow, whichever is smaller.

Weighted service demand `demand_value = Σ π_d D_d` then exceeds
`available_value + emission_weight · emission_limit` (the most value the
bounded resources and the emission allowance can supply) by a planted margin.
"""
struct EnergySystemCertificate
    mode::Symbol
    periods::Vector{Int}
    period_weight::Vector{Float64}
    commodity_value::Matrix{Float64}
    electricity_value::Vector{Float64}
    emission_weight::Float64
    capact_multipliers::Vector{Tuple{Int, Float64}}
    transfer_multipliers::Vector{Tuple{Int, Float64}}
    growth_multipliers::Vector{Tuple{Int, Float64}}
    demand_value::Float64
    available_value::Float64
    emission_limit::Float64
end

"""
    EnergySystemProblem <: ProblemGenerator

Technology-rich energy-system capacity-expansion LP in the style of
TIMES/MARKAL/MESSAGE: a multi-region, multi-period process network from primary
resources through conversion to end-use energy services.

# Structure

  - **Commodities** (per region): primary fuels (coal, gas, crude, uranium,
    biomass), secondary fuels (refined products, hydrogen, district heat),
    electricity (balanced per timeslice), and end-use service demands
    (residential heat, appliances, industrial process heat, passenger and freight
    transport, in PJ / Gpkm / Gtkm).
  - **Processes**: cost-stepped supply (domestic steps with cumulative
    reserves, capacity-limited imports); power plants (activity per timeslice,
    capacity in GW, 31.536 PJ/GW-yr, availability profiles for wind/solar/hydro,
    CHP with a heat co-product); refineries, electrolysers, SMR, boilers and
    heat pumps; end-use devices in efficiency classes; renewables in
    resource-quality classes with potentials.
  - **Rows**: commodity balances (equalities for fuels, ≥ for heat, electricity
    and service demands); capacity–activity rows; vintaged capacity transfer
    `cap_t = residual_t + Σ_{v > t-life} ncap_v`; market-growth rows
    `ncap_t ≤ g ncap_{t-1} + seed`; a peak-reserve row per region and period
    with endogenous peak load; per-period emission caps plus a cumulative
    carbon budget (linking every region and period); cumulative reserves of
    domestic resource steps; interconnector flows with losses between regions.
  - **Objective**: discounted investment (with salvage for life beyond the
    horizon), fixed and variable O&M, fuel supply and transmission costs, M\$.

Coefficients span the ranges of real models: efficiencies 0.3–0.95, COPs 2.5–4,
capacity factors 0.12–0.95, 31.536 PJ/GW-yr conversion, investment costs from
~3 to ~8000 M\$ per unit, emission factors ~0.05–0.1 Mt/PJ.

# Feasibility control

  - `feasible`: a reference technology mix is planted (`EnergySystemWitness`):
    devices meet demands by reference shares that shift toward electrification
    over time; hydrogen, heat and electricity are produced by reference shares
    (renewables at a target share, dispatchable plants filling the residual load
    slice by slice); capacities, residual stocks, growth seeds, potentials,
    supply steps, import capacities, reserves, peak reserve, and emission
    limits are all set with margins around it.
  - `infeasible`: one of `:emission_cap` (a period's cap 15–40% below the
    certified minimum emissions), `:carbon_budget` (the cumulative budget
    15–40% below the sum of those minima, a multi-period certificate), or
    `:supply_shortfall` (a permitting freeze holds clean capacity at the existing
    stock and, from some period on, domestic production and import capacity are
    cut so the certified minimum primary-energy requirement exceeds availability
    by 15–40%), with an `EnergySystemCertificate`. All modes aggregate the whole
    process network (and, through the growth rows, several periods); none is
    visible in a single row.
  - `unknown`: one ambition level `u ~ U(0.25, 1.05)` per instance places every
    period's emission cap at fraction `u` of the way from the certified emission
    floor to the reference plan's emissions; the true minimum lies in between,
    so instances fall on both sides.

# Sizing

Columns are exactly `T (R c + 2 S L)` for `R` regions with `c` columns per
region-period and `L` interconnectors with `S` timeslices; modules of the
technology database, renewable/device classes and supply steps are added until
`c` matches the per-region budget, so instances land within a fraction of a
percent of the target (the minimum is 44 columns). Targets above
`ENERGY_SYSTEM_MAX_VARIABLES` raise an `ArgumentError`.
"""
struct EnergySystemProblem <: ProblemGenerator
    n_regions::Int
    n_periods::Int
    period_length::Int
    n_slices::Int
    slice_duration::Vector{Float64}
    commodities::Vector{Symbol}
    profiles::Matrix{Float64}
    # technology instances
    tech_name::Vector{Symbol}
    tech_class::Vector{Int}
    tech_region::Vector{Int}
    tech_kind::Vector{Symbol}
    tech_input::Vector{Int}
    tech_input_coef::Vector{Float64}
    tech_output::Vector{Int}
    tech_output2::Vector{Int}
    tech_output2_coef::Vector{Float64}
    tech_emission::Vector{Float64}
    tech_profile::Vector{Int}
    tech_capfac::Vector{Float64}
    tech_avail::Matrix{Float64}
    tech_life::Vector{Int}
    tech_varcost::Vector{Float64}
    tech_invcost::Vector{Float64}
    tech_fixcost::Vector{Float64}
    tech_credit::Vector{Float64}
    tech_growth::Vector{Float64}
    tech_seed::Vector{Float64}
    tech_first_build::Vector{Float64}
    tech_potential::Vector{Float64}
    tech_residual::Matrix{Float64}
    supply_bound::Matrix{Float64}
    supply_reserve::Vector{Float64}
    demand::Array{Float64, 3}
    peak_slice::Vector{Int}
    reserve_margin::Float64
    lines::Vector{Tuple{Int, Int}}
    line_capacity::Vector{Float64}
    line_efficiency::Vector{Float64}
    line_cost::Vector{Float64}
    emission_cap::Vector{Float64}
    emission_budget::Float64
    discount_factor::Vector{Float64}
    feasible_witness::Union{Nothing, EnergySystemWitness}
    infeasibility_certificate::Union{Nothing, EnergySystemCertificate}
    feasibility_status::FeasibilityStatus
end

# ---------------------------------------------------------------------------
# Column / row layout (shared by the constructor, build_model and tests)
# ---------------------------------------------------------------------------

"""
    ESLayout

Flat column indexing derived deterministically from an instance:
activity column of tech `k`, period `t`, slot `ℓ` (timeslice for generators,
1 otherwise) is `act_offset[k] + (t-1) * slots[k] + ℓ`; capacity tech `k` has
`cap_index[k] > 0` and its capacity / new-capacity column in period `t` is
`(cap_index[k]-1) * T + t`; flow of line `l`, direction `dir ∈ (1, 2)`, period
`t`, slice `ℓ` is `((2(l-1) + dir - 1) T + t - 1) S + ℓ`.
"""
struct ESLayout
    slots::Vector{Int}
    act_offset::Vector{Int}
    n_act::Int
    cap_index::Vector{Int}
    n_capt::Int
    n_flow::Int
end

function _es_layout(tech_kind::Vector{Symbol}, n_periods::Int, n_slices::Int, n_lines::Int)
    n = length(tech_kind)
    slots = [k == :gen ? n_slices : 1 for k in tech_kind]
    act_offset = zeros(Int, n)
    off = 0
    for k in 1:n
        act_offset[k] = off
        off += n_periods * slots[k]
    end
    cap_index = zeros(Int, n)
    c = 0
    for k in 1:n
        if tech_kind[k] != :supply
            c += 1
            cap_index[k] = c
        end
    end
    return ESLayout(
        slots, act_offset, off, cap_index, c * n_periods, 2 * n_lines * n_periods * n_slices
    )
end

_es_layout(p::EnergySystemProblem) =
    _es_layout(p.tech_kind, p.n_periods, p.n_slices, length(p.lines))

es_act(L::ESLayout, k::Int, t::Int, s::Int=1) = L.act_offset[k] + (t - 1) * L.slots[k] + s
es_cap(L::ESLayout, k::Int, t::Int, T::Int) = (L.cap_index[k] - 1) * T + t
es_flow(l::Int, dir::Int, t::Int, s::Int, T::Int, S::Int) =
    ((2 * (l - 1) + dir - 1) * T + t - 1) * S + s

# ---------------------------------------------------------------------------
# Timeslices and profiles
# ---------------------------------------------------------------------------

const ES_SEASON_FACTORS = Dict(
    # winter, spring, summer, autumn
    :heat => (2.0, 0.9, 0.15, 0.95),
    :appliance => (1.15, 0.95, 0.95, 0.95),
    :ev => (1.05, 1.0, 0.95, 1.0),
    :industry => (1.0, 1.0, 0.95, 1.0),
    :flat => (1.0, 1.0, 1.0, 1.0),
    :solar => (0.55, 1.1, 1.45, 0.9),
    :wind => (1.3, 1.0, 0.7, 1.0),
    :windoff => (1.25, 1.0, 0.75, 1.05),
    :hydro => (0.8, 1.35, 1.05, 0.8),
)

function _es_hour_factor(kind::Symbol, h::Float64)
    if kind == :solar
        return (6 < h < 18) ? sin(pi * (h - 6) / 12) : 0.0
    elseif kind == :wind || kind == :windoff
        return 1 + 0.1 * cos(2pi * (h - 3) / 24)
    elseif kind == :heat
        return 0.7 + 0.5 * exp(-((h - 7) / 2.5)^2) + 0.6 * exp(-((h - 19) / 3)^2)
    elseif kind == :appliance
        return 0.5 +
               1.0 * exp(-((h - 19) / 3)^2) +
               0.4 * exp(-((h - 8) / 2)^2) +
               (8 < h < 18 ? 0.3 : 0.0)
    elseif kind == :ev
        return (h < 6 || h >= 22) ? 1.4 : (0.7 + 0.3 * exp(-((h - 18) / 1.5)^2))
    elseif kind == :industry
        return (7 <= h < 19) ? 1.25 : 0.75
    end
    return 1.0
end

"""
    _es_shape(kind, n_seasons, n_dayparts) -> Vector{Float64}

Raw (season × daypart) shape of a profile kind over the timeslices, slice
`ℓ = (s-1) n_dayparts + d`, averaging the hourly factor over each daypart's hours
and the four-season factors over merged seasons.
"""
function _es_shape(kind::Symbol, n_seasons::Int, n_dayparts::Int)
    sf = ES_SEASON_FACTORS[kind]
    season = if n_seasons == 4
        collect(sf)
    elseif n_seasons == 2
        [(sf[1] + sf[4]) / 2, (sf[2] + sf[3]) / 2]
    else
        [sum(sf) / 4]
    end
    day = Float64[]
    for d in 1:n_dayparts
        lo = 24 * (d - 1) / n_dayparts
        hi = 24 * d / n_dayparts
        hours = range(lo + 0.25, hi - 0.25; length=max(2, round(Int, 2 * (hi - lo))))
        push!(day, sum(_es_hour_factor(kind, h) for h in hours) / length(hours))
    end
    return [season[s] * day[d] for s in 1:n_seasons for d in 1:n_dayparts]
end

# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

function _es_region_columns(
    templates::Vector{ESTemplate}, n_classes::Dict{Symbol, Int}, steps::Dict{Symbol, Int}, S::Int
)
    c = 0
    for tp in templates
        k = n_classes[tp.name]
        c += k * (tp.kind == :gen ? S + 2 : 3)
    end
    for (f, n) in steps
        c += n + (f == :BIO ? 0 : 1)
    end
    return c
end

_es_n_lines(R::Int) = R <= 1 ? 0 : (R - 1) + (R >= 3 ? round(Int, 0.5 * (R - 1)) : 0)

"""
    _es_dimensions(rng, target)

Choose periods, period length, timeslices, regions, technology modules,
classes and supply steps so that `T (R c + 2 S L(R))` matches `target`.
"""
function _es_dimensions(rng::AbstractRNG, target::Int)
    Δ = rand(rng, (5, 5, 10))
    T = target < 120 ? 2 : (target < 600 ? rand(rng, 3:4) : rand(rng, 4:9))
    seasons_days = if target < 150
        (1, 1)
    elseif target < 1500
        rand(rng, ((1, 2), (2, 1), (2, 2), (1, 4)))
    elseif target < 20_000
        rand(rng, ((2, 2), (2, 3), (4, 2), (2, 4), (4, 3)))
    else
        rand(rng, ((4, 2), (4, 3), (4, 4), (4, 6)))
    end
    S = seasons_days[1] * seasons_days[2]

    function config(m::Int)
        tps = [tp for tp in ES_TEMPLATES if tp.mod <= m]
        fuels = [f for f in ES_PRIMARY if ES_COMMODITY_MODULE[f] <= m]
        return tps, fuels
    end
    base_count(m) = begin
        local tps, fuels
        tps, fuels = config(m)
        _es_region_columns(tps, Dict(tp.name => 1 for tp in tps), Dict(f => 1 for f in fuels), S)
    end
    m = 1
    while m < 6 && T * base_count(m + 1) <= target
        m += 1
    end
    templates, fuels = config(m)
    n_classes = Dict(tp.name => 1 for tp in templates)
    steps = Dict(f => 1 for f in fuels)
    c0 = base_count(m)
    c_max = _es_region_columns(
        templates,
        Dict(tp.name => (tp.classes ? ES_MAX_CLASSES : 1) for tp in templates),
        Dict(f => ES_MAX_DOMESTIC_STEPS for f in fuels),
        S,
    )
    richness = rand(rng, Uniform(1.0, 2.0))
    c_des = min(c_max, richness * c0)
    # Regions for a horizon: the count whose per-region budget fits [c0, c_max];
    # when going from R to R+1 regions jumps over that window (interconnector
    # columns are lumpy), nearby horizons are tried and the best fit is kept.
    budget(T, R) = (target / T - 2S * _es_n_lines(R)) / R
    function regions_for(T)
        local R, c
        R = max(1, round(Int, target / (T * (c_des + 3S))))
        while R > 1 && budget(T, R) < c0
            R -= 1
        end
        while budget(T, R) > c_max
            R += 1
        end
        c = clamp(budget(T, R), c0, c_max)
        return R, abs(T * (R * c + 2S * _es_n_lines(R)) - target)
    end
    T0 = T
    R, err = regions_for(T)
    for Tc in max(2, T0 - 2):min(10, T0 + 2)
        Rc, ec = regions_for(Tc)
        if ec < err - 0.002 * target
            T, R, err = Tc, Rc, ec
        end
    end
    b = budget(T, R)

    # Greedy fill toward the per-region budget: renewable site classes, device
    # efficiency classes, then domestic supply steps (1 column each).
    c = c0
    ren = shuffle(rng, [tp.name for tp in templates if tp.classes && tp.kind == :gen])
    dev = shuffle(rng, [tp.name for tp in templates if tp.classes && tp.kind == :annual])
    progress = true
    while progress
        progress = false
        for name in ren
            if n_classes[name] < ES_MAX_CLASSES && c + S + 2 <= b
                n_classes[name] += 1
                c += S + 2
                progress = true
            end
        end
        for name in dev
            if n_classes[name] < ES_MAX_CLASSES && c + 3 <= b
                n_classes[name] += 1
                c += 3
                progress = true
            end
        end
    end
    fl = shuffle(rng, collect(fuels))
    progress = true
    while progress
        progress = false
        for f in fl
            if steps[f] < ES_MAX_DOMESTIC_STEPS && c + 1 <= b + 0.5
                steps[f] += 1
                c += 1
                progress = true
            end
        end
    end
    return (;
        Δ,
        T,
        S,
        n_seasons=seasons_days[1],
        n_dayparts=seasons_days[2],
        R,
        m,
        templates,
        fuels,
        n_classes,
        steps,
    )
end

# ---------------------------------------------------------------------------
# Dual-value fixed point (certificates)
# ---------------------------------------------------------------------------

"""
    _es_values(n_comm, procs, seeds, θ) -> π

Greatest `π ≥ 0` (entries `Inf` until reached) with
`π_out · out_p ≤ θ e_p + Σ π_in in_p` for every unbounded single-output process
`p` in `procs` (a vector of `(inputs, outputs, emission, unbounded)` with
inputs/outputs as `(commodity, coefficient)` lists), starting from the fixed
`seeds`. Bellman–Ford style relaxation; every cycle of unbounded processes has
gain ≤ 1 (efficiency products below one, no heat-to-power), so it terminates.
Commodities no unbounded process produces are then valued from their unbounded
consumers: the least value that keeps every consumer's inequality.
"""
function _es_values(
    n_comm::Int,
    procs::Vector{Tuple{Vector{Tuple{Int, Float64}}, Vector{Tuple{Int, Float64}}, Float64, Bool}},
    seeds::Dict{Int, Float64},
    θ::Float64,
)
    π = fill(Inf, n_comm)
    for (c, v) in seeds
        π[c] = v
    end
    seeded = Set(keys(seeds))
    for _ in 1:(10 * n_comm + 10)
        changed = false
        for (ins, outs, e, unb) in procs
            (unb && length(outs) == 1) || continue
            c, o = outs[1]
            c in seeded && continue
            val = θ * e
            for (ci, a) in ins
                val += π[ci] * a
            end
            val /= o
            if val < π[c] * (1 - 1e-13) - 1e-300
                π[c] = val
                changed = true
            end
        end
        changed || break
    end
    for c in 1:n_comm
        isinf(π[c]) || continue
        need = 0.0
        for (ins, outs, e, unb) in procs
            unb || continue
            a_c = 0.0
            rest = -θ * e
            for (ci, a) in ins
                ci == c ? (a_c += a) : (rest -= π[ci] * a)
            end
            a_c > 0 || continue
            for (co, o) in outs
                rest += isinf(π[co]) ? 0.0 : π[co] * o
            end
            need = max(need, rest / a_c)
        end
        π[c] = need
    end
    return π
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""Uniform draw from a `(lo, hi)` range that may be degenerate (`lo == hi`)."""
_es_draw(rng::AbstractRNG, r::NTuple{2, Float64}) =
    r[1] == r[2] ? r[1] : rand(rng, Uniform(r[1], r[2]))

"""
    EnergySystemProblem(target_variables, feasibility_status, seed)

Construct an energy-system planning instance with about `target_variables`
columns (exact formula in the type docstring).
"""
function EnergySystemProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= ENERGY_SYSTEM_MAX_VARIABLES || throw(
        ArgumentError(
            "economic_planning/energy_system supports at most $ENERGY_SYSTEM_MAX_VARIABLES " *
            "variables; requested $target_variables.",
        ),
    )
    rng = MersenneTwister(seed)
    dims = _es_dimensions(rng, target_variables)
    Δ, T, S, R = dims.Δ, dims.T, dims.S, dims.R
    templates = dims.templates

    # ---- timeslices and profiles ---------------------------------------------
    dur = fill(1.0 / S, S)
    shapes = Dict(
        k => _es_shape(k, dims.n_seasons, dims.n_dayparts) for k in keys(ES_SEASON_FACTORS)
    )
    profiles = zeros(length(ES_PROFILE_NAMES), S)
    for (i, k) in enumerate(ES_PROFILE_NAMES)
        w = dur .* shapes[k]
        profiles[i, :] .= w ./ sum(w)
    end
    avail_shape(k) = (sh=shapes[k]; sh ./ sum(dur .* sh))

    # ---- commodities -----------------------------------------------------------
    commodities = [c for c in ES_COMMODITY_ORDER if ES_COMMODITY_MODULE[c] <= dims.m]
    cidx = Dict(c => i for (i, c) in enumerate(commodities))
    NC = length(commodities)
    ELC = -1
    cid(c::Symbol) = c == :ELC ? ELC : (c == :none ? 0 : cidx[c])
    demands = [d for d in ES_DEMANDS if haskey(cidx, d)]

    # ---- region geography, demands --------------------------------------------
    pos = [(100 * rand(rng), 100 * rand(rng)) for _ in 1:R]
    scale = [exp(rand(rng, Uniform(log(0.2), log(5.0)))) for _ in 1:R]
    base_demand = Dict(
        :RH => (150.0, 300.0),
        :RA => (80.0, 150.0),
        :IP => (200.0, 400.0),
        :TP => (100.0, 150.0),
        :TF => (50.0, 200.0),
    )
    demand_growth = Dict(
        :RH => (0.0, 0.005),
        :RA => (0.01, 0.025),
        :IP => (0.005, 0.02),
        :TP => (0.005, 0.015),
        :TF => (0.01, 0.025),
    )
    D = zeros(R, NC, T)
    for r in 1:R, d in demands
        b = scale[r] * rand(rng, Uniform(base_demand[d]...))
        gr = rand(rng, Uniform(demand_growth[d]...))
        for t in 1:T
            D[r, cidx[d], t] = b * (1 + gr)^(Δ * (t - 1))
        end
    end

    # ---- technology instances --------------------------------------------------
    tech_name = Symbol[]
    tech_template = Int[]
    tech_class = Int[]
    tech_region = Int[]
    tech_kind = Symbol[]
    tech_input = Int[]
    tech_input_coef = Float64[]
    tech_output = Int[]
    tech_output2 = Int[]
    tech_output2_coef = Float64[]
    tech_emission = Float64[]
    tech_profile = Int[]
    tech_capfac = Float64[]
    avail_rows = Vector{Vector{Float64}}()
    tech_life = Int[]
    tech_varcost = Float64[]
    tech_invcost = Float64[]
    tech_fixcost = Float64[]
    tech_credit = Float64[]
    tech_growth = Float64[]
    supply_cost_rank = Int[]
    template_index = Dict(tp.name => i for (i, tp) in enumerate(ES_TEMPLATES))
    supply_price = Dict(
        :COA => ((1.5, 3.0), (2.5, 4.0)),
        :GAS => ((3.0, 7.0), (6.0, 12.0)),
        :OIL => ((5.0, 9.0), (8.0, 14.0)),
        :URN => ((0.5, 1.0), (0.8, 1.5)),
        :BIO => ((3.0, 6.0), (6.0, 10.0)),
    )
    for r in 1:R
        # supply steps: domestic steps with rising cost, then an import step
        for f in dims.fuels
            n_dom = dims.steps[f]
            lo, hi = supply_price[f][1]
            costs = sort([rand(rng, Uniform(lo, hi)) for _ in 1:n_dom])
            for (i, cst) in enumerate(costs)
                push!(tech_name, f)
                push!(tech_template, 0)
                push!(tech_class, i)
                push!(tech_region, r)
                push!(tech_kind, :supply)
                push!(tech_input, 0)
                push!(tech_input_coef, 0.0)
                push!(tech_output, cidx[f])
                push!(tech_output2, 0)
                push!(tech_output2_coef, 0.0)
                push!(tech_emission, 0.0)
                push!(tech_profile, 0)
                push!(tech_capfac, 0.0)
                push!(avail_rows, zeros(S))
                push!(tech_life, 0)
                push!(tech_varcost, cst)
                push!(tech_invcost, 0.0)
                push!(tech_fixcost, 0.0)
                push!(tech_credit, 0.0)
                push!(tech_growth, 0.0)
                push!(supply_cost_rank, i)
            end
            if f != :BIO
                ilo, ihi = supply_price[f][2]
                push!(tech_name, f)
                push!(tech_template, 0)
                push!(tech_class, 0)
                push!(tech_region, r)
                push!(tech_kind, :supply)
                push!(tech_input, 0)
                push!(tech_input_coef, 0.0)
                push!(tech_output, cidx[f])
                push!(tech_output2, 0)
                push!(tech_output2_coef, 0.0)
                push!(tech_emission, 0.0)
                push!(tech_profile, 0)
                push!(tech_capfac, 0.0)
                push!(avail_rows, zeros(S))
                push!(tech_life, 0)
                push!(tech_varcost, max(rand(rng, Uniform(ilo, ihi)), costs[end] * 1.05))
                push!(tech_invcost, 0.0)
                push!(tech_fixcost, 0.0)
                push!(tech_credit, 0.0)
                push!(tech_growth, 0.0)
                push!(supply_cost_rank, n_dom + 1)
            end
        end
        for tp in templates
            for c in 1:dims.n_classes[tp.name]
                push!(tech_name, tp.name)
                push!(tech_template, template_index[tp.name])
                push!(tech_class, c)
                push!(tech_region, r)
                push!(tech_kind, tp.kind)
                eff = _es_draw(rng, tp.eff)
                inv = _es_draw(rng, tp.inv)
                av = _es_draw(rng, tp.avail)
                if tp.kind == :annual && tp.classes
                    eff *= ES_DEVICE_EFF[c]
                    inv *= ES_DEVICE_COST[c]
                elseif tp.kind == :gen && tp.classes
                    av *= ES_SITE_QUALITY[c]
                    inv *= ES_SITE_COST[c]
                end
                push!(tech_input, cid(tp.input))
                push!(tech_input_coef, tp.input == :none ? 0.0 : 1.0 / eff)
                push!(tech_output, cid(tp.output))
                push!(tech_output2, tp.output2 == :none ? 0 : cid(tp.output2))
                push!(tech_output2_coef, tp.output2 == :none ? 0.0 : _es_draw(rng, tp.ratio2))
                push!(
                    tech_emission,
                    get(ES_EMISSION_FACTOR, tp.input, 0.0) * (tp.input == :none ? 0.0 : 1.0 / eff),
                )
                push!(
                    tech_profile,
                    if tp.kind == :annual && tp.input == :ELC
                        findfirst(==(tp.profile), ES_PROFILE_NAMES)
                    else
                        0
                    end,
                )
                push!(tech_capfac, tp.kind == :gen ? ES_PJ_PER_GW_YEAR : 1.0)
                if tp.kind == :gen
                    if tp.profile == :flat
                        push!(avail_rows, fill(av, S))
                    else
                        push!(avail_rows, min.(0.98, av .* avail_shape(tp.profile)))
                    end
                else
                    push!(avail_rows, fill(av, S))
                end
                push!(tech_life, max(1, round(Int, tp.life / Δ)))
                push!(tech_varcost, _es_draw(rng, tp.vom))
                push!(tech_invcost, inv)
                push!(tech_fixcost, _es_draw(rng, tp.fom))
                push!(tech_credit, _es_draw(rng, tp.credit))
                push!(tech_growth, (1 + tp.growth)^Δ)
                push!(supply_cost_rank, 0)
            end
        end
    end
    ntech = length(tech_name)
    tech_avail = zeros(ntech, S)
    for k in 1:ntech
        tech_avail[k, :] .= avail_rows[k]
    end

    # ---- interconnectors -------------------------------------------------------
    lines = Tuple{Int, Int}[]
    if R >= 2
        dist(a, b) = hypot(pos[a][1] - pos[b][1], pos[a][2] - pos[b][2])
        in_tree = falses(R)
        in_tree[1] = true
        best = [dist(1, j) for j in 1:R]
        parent = ones(Int, R)
        for _ in 2:R
            j = 0
            bd = Inf
            for v in 1:R
                if !in_tree[v] && best[v] < bd
                    bd = best[v]
                    j = v
                end
            end
            in_tree[j] = true
            push!(lines, minmax(parent[j], j))
            for v in 1:R
                if !in_tree[v] && dist(j, v) < best[v]
                    best[v] = dist(j, v)
                    parent[v] = j
                end
            end
        end
        extra = _es_n_lines(R) - (R - 1)
        if extra > 0
            existing = Set(lines)
            cands = Tuple{Float64, Int, Int}[]
            for a in 1:R
                nbrs = sort([(dist(a, b), b) for b in 1:R if b != a])[1:min(R - 1, 6)]
                for (d, b) in nbrs
                    e = minmax(a, b)
                    e in existing || push!(cands, (d, e[1], e[2]))
                end
            end
            sort!(cands)
            for (_, a, b) in cands
                length(lines) >= R - 1 + extra && break
                e = (a, b)
                e in existing && continue
                push!(lines, e)
                push!(existing, e)
            end
            # Degenerate geometry: fall back to any unused pair.
            a = 1
            while length(lines) < R - 1 + extra
                for b in (a + 1):R
                    length(lines) >= R - 1 + extra && break
                    (a, b) in existing && continue
                    push!(lines, (a, b))
                    push!(existing, (a, b))
                end
                a += 1
            end
        end
    end
    nL = length(lines)
    line_capacity = [rand(rng, Uniform(0.5, 5.0)) * sqrt(scale[a] * scale[b]) for (a, b) in lines]
    line_efficiency = [
        clamp(
            1 - 0.0005 * hypot(pos[a][1] - pos[b][1], pos[a][2] - pos[b][2]) -
            rand(rng, Uniform(0.005, 0.02)),
            0.9,
            0.995,
        ) for (a, b) in lines
    ]
    line_cost = [rand(rng, Uniform(0.5, 2.0)) for _ in lines]

    layout = _es_layout(tech_kind, T, S, nL)
    techs_of_region = [Int[] for _ in 1:R]
    for k in 1:ntech
        push!(techs_of_region[tech_region[k]], k)
    end

    # ---- reference plan (witness) ---------------------------------------------
    act = zeros(layout.n_act)
    req = zeros(ntech, T)                    # required capacity
    elec_use = zeros(R, T, S)                # consumption by annual ELC users
    tau(t) = T == 1 ? 1.0 : (t - 1) / (T - 1)
    # region-specific reference share noise per template
    share_noise = [Dict(tp.name => rand(rng, LogNormal(0.0, 0.3)) for tp in templates) for _ in 1:R]
    ren_share = [
        (s1=rand(rng, Uniform(0.05, 0.3)); (s1, min(0.85, s1 + rand(rng, Uniform(0.2, 0.5))))) for
        _ in 1:R
    ]
    hydro_share = [rand(rng) < 0.6 ? rand(rng, Uniform(0.02, 0.15)) : 0.01 for _ in 1:R]
    chp_heat_share = [rand(rng, Uniform(0.25, 0.5)) for _ in 1:R]
    use = zeros(R, NC, T)                    # annual commodity use (inputs)

    function allocate!(r, t, producers, amount)
        local w, tot, a, ci, tp, s
        # Split `amount` of output across producer instances by reference share
        # (template share interpolated over time x region noise x class share).
        w = Float64[]
        for k in producers
            tp = ES_TEMPLATES[tech_template[k]]
            s = (tp.share[1] + (tp.share[2] - tp.share[1]) * tau(t)) * share_noise[r][tp.name]
            push!(
                w,
                s * ES_CLASS_SHARE[tech_class[k]] / sum(ES_CLASS_SHARE[1:dims.n_classes[tp.name]]),
            )
        end
        tot = sum(w)
        tot > 0 || (w.=1.0; tot=length(w))
        for (i, k) in enumerate(producers)
            a = amount * w[i] / tot
            act[es_act(layout, k, t)] += a
            ci = tech_input[k]
            if ci == ELC
                for s in 1:S
                    elec_use[r, t, s] += a * tech_input_coef[k] * profiles[tech_profile[k], s]
                end
            elseif ci > 0
                use[r, ci, t] += a * tech_input_coef[k]
            end
            req[k, t] = max(req[k, t], a / (tech_avail[k, 1] * tech_capfac[k]))
        end
    end
    annual_producers(r, c) =
        [k for k in techs_of_region[r] if tech_kind[k] == :annual && tech_output[k] == c]

    for r in 1:R, t in 1:T
        for d in demands
            allocate!(r, t, annual_producers(r, cidx[d]), D[r, cidx[d], t])
        end
        if haskey(cidx, :H2)
            allocate!(r, t, annual_producers(r, cidx[:H2]), use[r, cidx[:H2], t])
        end
        chp = [k for k in techs_of_region[r] if tech_output2[k] > 0]
        if haskey(cidx, :HET)
            heat = use[r, cidx[:HET], t]
            chp_heat = isempty(chp) ? 0.0 : chp_heat_share[r] * heat
            allocate!(r, t, annual_producers(r, cidx[:HET]), heat - chp_heat)
            for k in chp
                # CHP runs flat; its electricity output covers the heat share.
                elec = chp_heat / length(chp) / tech_output2_coef[k]
                for s in 1:S
                    a = elec * dur[s]
                    act[es_act(layout, k, t, s)] += a
                    req[k, t] = max(req[k, t], a / (tech_avail[k, s] * ES_PJ_PER_GW_YEAR * dur[s]))
                end
                use[r, tech_input[k], t] += elec * tech_input_coef[k]
            end
        end
    end
    # Electricity: renewables at their target share (full availability), then
    # dispatchable plants fill the residual load slice by slice.
    for r in 1:R, t in 1:T
        gens = [k for k in techs_of_region[r] if tech_kind[k] == :gen && tech_output2[k] == 0]
        vre = [k for k in gens if ES_TEMPLATES[tech_template[k]].potential && tech_input[k] == 0]
        disp = [
            k for k in gens if !(k in vre) &&
                ES_TEMPLATES[tech_template[k]].share[1] + ES_TEMPLATES[tech_template[k]].share[2] >
                0
        ]
        load = elec_use[r, t, :]
        annual_load = sum(load)
        supply = zeros(S)
        for k in techs_of_region[r]
            tech_output2[k] > 0 || continue
            for s in 1:S
                supply[s] += act[es_act(layout, k, t, s)]
            end
        end
        target_share = ren_share[r][1] + (ren_share[r][2] - ren_share[r][1]) * tau(t)
        w = Float64[]
        for k in vre
            tp = ES_TEMPLATES[tech_template[k]]
            share = if tp.name == :E_HYD
                0.0
            else
                (tp.share[1] + (tp.share[2] - tp.share[1]) * tau(t)) * share_noise[r][tp.name]
            end
            push!(w, share * ES_CLASS_SHARE[tech_class[k]])
        end
        wsum = sum(w; init=0.0)
        for (i, k) in enumerate(vre)
            tp = ES_TEMPLATES[tech_template[k]]
            energy = if tp.name == :E_HYD
                hydro_share[r] * annual_load
            else
                (wsum > 0 ? target_share * annual_load * w[i] / wsum : 0.0)
            end
            per_gw = sum(tech_avail[k, s] * ES_PJ_PER_GW_YEAR * dur[s] for s in 1:S)
            capk = energy / per_gw
            req[k, t] = max(req[k, t], capk)
            for s in 1:S
                a = tech_avail[k, s] * ES_PJ_PER_GW_YEAR * dur[s] * capk
                act[es_act(layout, k, t, s)] += a
                supply[s] += a
            end
        end
        residual = max.(load .- supply, 0.0)
        ws = Float64[]
        for k in disp
            tp = ES_TEMPLATES[tech_template[k]]
            push!(
                ws, (tp.share[1] + (tp.share[2] - tp.share[1]) * tau(t)) * share_noise[r][tp.name]
            )
        end
        wtot = sum(ws; init=0.0)
        for (i, k) in enumerate(disp)
            for s in 1:S
                a = residual[s] * ws[i] / wtot
                act[es_act(layout, k, t, s)] += a
                req[k, t] = max(req[k, t], a / (tech_avail[k, s] * ES_PJ_PER_GW_YEAR * dur[s]))
                use[r, tech_input[k], t] += a * tech_input_coef[k]
            end
        end
    end
    if haskey(cidx, :PET)
        for r in 1:R, t in 1:T
            allocate!(r, t, annual_producers(r, cidx[:PET]), use[r, cidx[:PET], t])
        end
    end

    # Peak slice per region (highest consumption per hour in period 1) and peak
    # reserve: top up gas turbines (or CCGT) so firm capacity covers the peak.
    peak_slice = [argmax([elec_use[r, 1, s] / dur[s] for s in 1:S]) for r in 1:R]
    reserve_margin = rand(rng, Uniform(0.10, 0.20))

    # Margins and residual stocks, then a greedy vintage build plan.
    cap_margin = [rand(rng, Uniform(1.02, 1.15)) for _ in 1:ntech]
    for k in 1:ntech
        tech_kind[k] == :supply && continue
        req[k, :] .*= cap_margin[k]
    end
    for r in 1:R, t in 1:T
        firm = sum(
            tech_credit[k] * req[k, t] for k in techs_of_region[r] if tech_kind[k] == :gen; init=0.0
        )
        need =
            (1 + reserve_margin) * elec_use[r, t, peak_slice[r]] /
            (ES_PJ_PER_GW_YEAR * dur[peak_slice[r]])
        if firm < 1.03 * need
            gt = [k for k in techs_of_region[r] if tech_name[k] == :E_GASGT]
            isempty(gt) && (gt = [k for k in techs_of_region[r] if tech_name[k] == :E_GASCC])
            k = gt[1]
            req[k, t] += (1.03 * need - firm) / tech_credit[k]
        end
    end
    residual = zeros(ntech, T)
    cap_w = zeros(layout.n_capt)
    ncap_w = zeros(layout.n_capt)
    first_build = fill(Inf, ntech)
    seedv = zeros(ntech)
    potential = fill(Inf, ntech)
    for k in 1:ntech
        tech_kind[k] == :supply && continue
        tp = ES_TEMPLATES[tech_template[k]]
        L = tech_life[k]
        res1 = if tp.incumbent
            rand(rng, Uniform(0.6, 1.1)) * req[k, 1]
        else
            rand(rng, Uniform(0.0, 0.15)) * req[k, 1]
        end
        res_life = rand(rng, 2:max(2, L))
        for t in 1:T
            residual[k, t] = res1 * max(0.0, 1 - (t - 1) / res_life)
        end
        built = zeros(T)
        for t in 1:T
            vint = sum(built[v] for v in max(1, t - L + 1):(t - 1); init=0.0)
            built[t] = max(0.0, req[k, t] - residual[k, t] - vint)
            cap_w[es_cap(layout, k, t, T)] = residual[k, t] + vint + built[t]
            ncap_w[es_cap(layout, k, t, T)] = built[t]
        end
        g = tech_growth[k]
        base = 0.02 * maximum(req[k, :]) + 1e-3
        jump = maximum((built[t] - g * built[t - 1] for t in 2:T); init=0.0)
        seedv[k] = (max(jump, 0.0) + base) * rand(rng, Uniform(1.2, 2.0))
        first_build[k] = (built[1] + base) * rand(rng, Uniform(1.2, 2.0))
        if tp.potential
            maxcap = maximum(cap_w[es_cap(layout, k, t, T)] for t in 1:T)
            potential[k] = max(maxcap, 1e-3) * (
                if tp.kind == :gen && tp.input == :none
                    rand(rng, Uniform(1.3, 2.5))
                else
                    rand(rng, Uniform(1.1, 1.6))
                end
            )
        end
    end

    # Supply: domestic steps sized against period-1 use, imports cover the rest.
    supply_bound = fill(Inf, ntech, T)
    supply_reserve = fill(Inf, ntech)
    for r in 1:R, f in dims.fuels
        ks = sort(
            [k for k in techs_of_region[r] if tech_kind[k] == :supply && tech_output[k] == cidx[f]];
            by=k -> supply_cost_rank[k],
        )
        dom = [k for k in ks if supply_cost_rank[k] <= dims.steps[f]]
        imp = [k for k in ks if supply_cost_rank[k] > dims.steps[f]]
        base_use = max(maximum(use[r, cidx[f], :]), 1e-3)
        richness = f == :BIO ? rand(rng, Uniform(1.1, 1.5)) : rand(rng, Uniform(0.1, 1.3))
        fr = length(dom) == 1 ? [1.0] : rand(rng, Dirichlet(length(dom), 1.0))
        sizes = richness * base_use .* fr
        for t in 1:T
            remaining = use[r, cidx[f], t]
            for (i, k) in enumerate(dom)
                a = min(remaining, sizes[i])
                act[es_act(layout, k, t)] = a
                remaining -= a
            end
            for k in imp
                act[es_act(layout, k, t)] = remaining
                remaining = 0.0
            end
            @assert remaining <= 1e-9 * base_use
        end
        for (i, k) in enumerate(dom)
            supply_bound[k, :] .= sizes[i]
            cum = Δ * sum(act[es_act(layout, k, t)] for t in 1:T)
            supply_reserve[k] = max(
                cum * rand(rng, Uniform(1.1, 1.6)), sizes[i] * Δ * T * rand(rng, Uniform(0.4, 1.0))
            )
        end
        for k in imp
            m_imp = maximum(act[es_act(layout, k, t)] for t in 1:T)
            supply_bound[k, :] .= m_imp * rand(rng, Uniform(1.2, 2.0)) + 0.05 * base_use
        end
    end

    # Emissions of the reference plan.
    function plan_emissions(t)
        local e
        e = 0.0
        for k in 1:ntech
            tech_emission[k] == 0 && continue
            for s in 1:layout.slots[k]
                e += tech_emission[k] * act[es_act(layout, k, t, s)]
            end
        end
        return e
    end
    Ew = [plan_emissions(t) for t in 1:T]
    emission_cap = Ew .* rand(rng, Uniform(1.05, 1.3), T) .+ 1e-6
    emission_budget = Δ * sum(Ew) * rand(rng, Uniform(1.03, 1.2)) + 1e-6

    # ---- certificates / lower bounds --------------------------------------------
    # Global commodity ids: annual (r, c) -> (r-1) NC + c; electricity of r -> NC R + r.
    gid(r, c) = c == ELC ? NC * R + r : (r - 1) * NC + c
    n_comm = (NC + 1) * R
    unbounded(k) = tech_kind[k] != :supply && !isfinite(potential[k])
    function process_list(θ)
        local procs, ins, outs, r
        procs = Tuple{Vector{Tuple{Int, Float64}}, Vector{Tuple{Int, Float64}}, Float64, Bool}[]
        for k in 1:ntech
            r = tech_region[k]
            ins = if tech_input[k] == 0
                Tuple{Int, Float64}[]
            else
                [(gid(r, tech_input[k]), tech_input_coef[k])]
            end
            outs = [(gid(r, tech_output[k]), 1.0)]
            tech_output2[k] > 0 && push!(outs, (gid(r, tech_output2[k]), tech_output2_coef[k]))
            push!(procs, (ins, outs, tech_emission[k], unbounded(k)))
        end
        for (l, (a, b)) in enumerate(lines)
            push!(procs, ([(gid(a, ELC), 1.0)], [(gid(b, ELC), line_efficiency[l])], 0.0, true))
            push!(procs, ([(gid(b, ELC), 1.0)], [(gid(a, ELC), line_efficiency[l])], 0.0, true))
        end
        return procs
    end
    function dual_values(mode)
        local θ, seeds
        θ = mode == :supply_shortfall ? 0.0 : 1.0
        seeds = Dict{Int, Float64}()
        for r in 1:R, f in dims.fuels
            if mode == :supply_shortfall
                seeds[gid(r, cidx[f])] = 1.0
            elseif f != :BIO
                seeds[gid(r, cidx[f])] = 0.0
            end
        end
        return θ, _es_values(n_comm, process_list(θ), seeds, θ)
    end
    # Excess value per unit activity of tech k (in slice s) given π, θ.
    function excess(k, π, θ)
        local v, r
        r = tech_region[k]
        v = π[gid(r, tech_output[k])]
        tech_output2[k] > 0 && (v += π[gid(r, tech_output2[k])] * tech_output2_coef[k])
        tech_input[k] != 0 && (v -= π[gid(r, tech_input[k])] * tech_input_coef[k])
        return v - θ * tech_emission[k]
    end
    # Upper bound on cap[k, t] through the growth rows (Inf if no growth rows).
    function growth_route(k, t)
        local L, g, win, w, r1, bound
        L = tech_life[k]
        g = tech_growth[k]
        win = max(1, t - L + 1):t
        w = zeros(T + 1)
        for v in t:-1:2
            w[v] = (v in win ? 1.0 : 0.0) + g * w[v + 1]
        end
        r1 = (1 in win ? 1.0 : 0.0) + g * w[2]
        bound = residual[k, t] + sum(w[v] * seedv[k] for v in 2:t; init=0.0) + r1 * first_build[k]
        return bound, w
    end
    # Certificate pieces for one period: returns (demand value, available value,
    # capact, transfer, growth multiplier lists) for weight `wt`.
    function period_pieces(π, θ, t, wt; supply_scale=1.0)
        local dem, avail, capact, transfer, growth, perunit, coef, gb, w, c, v
        dem = 0.0
        for r in 1:R, d in demands
            dem += wt * π[gid(r, cidx[d])] * D[r, cidx[d], t]
        end
        avail = 0.0
        capact = Tuple{Int, Float64}[]
        transfer = Tuple{Int, Float64}[]
        growth = Tuple{Int, Float64}[]
        for k in 1:ntech
            v = excess(k, π, θ)
            v > 1e-12 || continue
            if tech_kind[k] == :supply
                avail += wt * v * supply_bound[k, t] * supply_scale
                continue
            end
            unbounded(k) && continue  # cannot happen at the fixed point (v ≤ 0)
            perunit = 0.0
            for s in 1:layout.slots[k]
                coef = tech_avail[k, s] * tech_capfac[k] * (tech_kind[k] == :gen ? dur[s] : 1.0)
                push!(capact, (es_act(layout, k, t, s), -wt * v))
                perunit += wt * v * coef
            end
            gb, w = growth_route(k, t)
            if gb < potential[k]
                avail += perunit * gb
                c = es_cap(layout, k, t, T)
                push!(transfer, (c, -perunit))
                for vv in 2:t
                    w[vv] > 0 && push!(growth, (es_cap(layout, k, vv, T), -perunit * w[vv]))
                end
            else
                avail += perunit * potential[k]
            end
        end
        return dem, avail, capact, transfer, growth
    end

    certificate = nothing
    if feasibility_status != feasible
        θe, πe = dual_values(:emission_cap)
        lb = zeros(T)
        for t in 1:T
            dem, avail, _, _, _ = period_pieces(πe, θe, t, 1.0)
            lb[t] = dem - avail      # certified lower bound on period-t emissions
        end
        if feasibility_status == unknown
            # One ambition level per instance (with a little per-period jitter):
            # the cap path sits at fraction u of the way from the certified
            # emission floor to the reference plan's emissions. The true minimum
            # lies in between, so instances fall on both sides.
            u = rand(rng, Uniform(0.25, 1.05))
            for t in 1:T
                lo = max(lb[t], 0.0)
                hi = max(Ew[t], lo)
                emission_cap[t] = max(1e-6, lo + (u + rand(rng, Uniform(-0.03, 0.03))) * (hi - lo))
            end
        else
            mode = if rand(rng) < 0.4
                :emission_cap
            else
                (rand(rng) < 0.5 ? :carbon_budget : :supply_shortfall)
            end
            margin = rand(rng, Uniform(1.15, 1.4))
            # An emission argument needs a clearly positive certified minimum.
            if mode == :carbon_budget && sum(lb) <= 0.05 * sum(Ew)
                mode = :emission_cap
            end
            if mode == :emission_cap && maximum(lb[t] / Ew[t] for t in 1:T) <= 0.05
                mode = :supply_shortfall
            end
            if mode == :emission_cap
                cands = [t for t in cld(T, 2):T if lb[t] > 0.05 * Ew[t]]
                isempty(cands) && (cands = [argmax(lb)])
                tstar = rand(rng, cands)
                emission_cap[tstar] = lb[tstar] / margin
                dem, avail, ca, tr, gr = period_pieces(πe, θe, tstar, 1.0)
                certificate = EnergySystemCertificate(
                    mode,
                    [tstar],
                    [1.0],
                    permutedims(reshape(πe[1:(NC * R)], NC, R)),
                    πe[(NC * R + 1):end],
                    1.0,
                    ca,
                    tr,
                    gr,
                    dem,
                    avail,
                    emission_cap[tstar],
                )
            elseif mode == :carbon_budget
                dem = avail = 0.0
                ca = Tuple{Int, Float64}[]
                tr = Tuple{Int, Float64}[]
                gr = Tuple{Int, Float64}[]
                for t in 1:T
                    d1, a1, c1, t1, g1 = period_pieces(πe, θe, t, Float64(Δ))
                    dem += d1
                    avail += a1
                    append!(ca, c1)
                    append!(tr, t1)
                    append!(gr, g1)
                end
                emission_budget = (dem - avail) / margin
                certificate = EnergySystemCertificate(
                    mode,
                    collect(1:T),
                    fill(Float64(Δ), T),
                    permutedims(reshape(πe[1:(NC * R)], NC, R)),
                    πe[(NC * R + 1):end],
                    1.0,
                    ca,
                    tr,
                    gr,
                    dem,
                    avail,
                    emission_budget,
                )
            end
            if mode == :supply_shortfall
                θs, πs = dual_values(:supply_shortfall)
                tstar = rand(rng, cld(T, 2):T)
                # Scenario: a permitting freeze (no clean capacity beyond today's
                # stock) plus a supply disruption from t* on (domestic production
                # and import capacity cut). π values service demand at its
                # minimum primary-energy content over the most efficient chains.
                for k in 1:ntech
                    isfinite(potential[k]) || continue
                    potential[k] = max(maximum(residual[k, :]), 1e-3)
                end
                dem, avail_all, _, _, _ = period_pieces(πs, θs, tstar, 1.0)
                _, avail_nos, _, _, _ = period_pieces(πs, θs, tstar, 1.0; supply_scale=0.0)
                if avail_nos > 0.7 * dem
                    # Existing clean stock alone nearly covers the (efficiency-
                    # minimal) need: a demand surge keeps the shortfall substantial.
                    D[:, :, tstar:T] .*= avail_nos / (0.7 * dem)
                    dem, avail_all, _, _, _ = period_pieces(πs, θs, tstar, 1.0)
                    _, avail_nos, _, _, _ = period_pieces(πs, θs, tstar, 1.0; supply_scale=0.0)
                end
                supply_val = avail_all - avail_nos
                need = dem - avail_nos
                α = min(1.0, need / (supply_val * margin))
                for k in 1:ntech
                    tech_kind[k] == :supply || continue
                    for t in tstar:T
                        supply_bound[k, t] *= α
                    end
                end
                dem, avail, ca, tr, gr = period_pieces(πs, θs, tstar, 1.0)
                certificate = EnergySystemCertificate(
                    mode,
                    [tstar],
                    [1.0],
                    permutedims(reshape(πs[1:(NC * R)], NC, R)),
                    πs[(NC * R + 1):end],
                    0.0,
                    ca,
                    tr,
                    gr,
                    dem,
                    avail,
                    0.0,
                )
            end
        end
    end

    witness = if feasibility_status == feasible
        EnergySystemWitness(act, cap_w, ncap_w, zeros(layout.n_flow))
    else
        nothing
    end
    discount = rand(rng, Uniform(0.03, 0.08))
    discount_factor = [(1 + discount)^(-Δ * (t - 1)) for t in 1:T]

    return EnergySystemProblem(
        R,
        T,
        Δ,
        S,
        dur,
        commodities,
        profiles,
        tech_name,
        tech_class,
        tech_region,
        tech_kind,
        tech_input,
        tech_input_coef,
        tech_output,
        tech_output2,
        tech_output2_coef,
        tech_emission,
        tech_profile,
        tech_capfac,
        tech_avail,
        tech_life,
        tech_varcost,
        tech_invcost,
        tech_fixcost,
        tech_credit,
        tech_growth,
        seedv,
        first_build,
        potential,
        residual,
        supply_bound,
        supply_reserve,
        D,
        peak_slice,
        reserve_margin,
        lines,
        line_capacity,
        line_efficiency,
        line_cost,
        emission_cap,
        emission_budget,
        discount_factor,
        witness,
        certificate,
        feasibility_status,
    )
end

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

"""
    build_model(prob::EnergySystemProblem)

Build the energy-system LP. Deterministic. Registered containers: variables
`act`, `cap`, `ncap`, `flow` (flat, see `ESLayout`); constraints `bal`
(annual commodity balances, index `((r-1) NC + c - 1) T + t`), `bal_elc`
(index `((r-1) T + t - 1) S + s`), `capact[act column]`, `transfer[cap
column]`, `growth[ncap column]`, `peak[r, t]`, `emission[t]`, `budget`,
`reserve[supply tech]`.
"""
function build_model(prob::EnergySystemProblem)
    model = Model()
    L = _es_layout(prob)
    R, T, S, Δ = prob.n_regions, prob.n_periods, prob.n_slices, prob.period_length
    NC = length(prob.commodities)
    ntech = length(prob.tech_name)
    dur = prob.slice_duration
    nL = length(prob.lines)

    @variable(model, act[1:L.n_act] >= 0)
    @variable(model, cap[1:L.n_capt] >= 0)
    @variable(model, ncap[1:L.n_capt] >= 0)
    for k in 1:ntech
        if prob.tech_kind[k] == :supply
            for t in 1:T
                ub = prob.supply_bound[k, t]
                isfinite(ub) && set_upper_bound(act[es_act(L, k, t)], ub)
            end
            continue
        end
        for t in 1:T
            isfinite(prob.tech_potential[k]) &&
                set_upper_bound(cap[es_cap(L, k, t, T)], prob.tech_potential[k])
        end
        isfinite(prob.tech_first_build[k]) &&
            set_upper_bound(ncap[es_cap(L, k, 1, T)], prob.tech_first_build[k])
    end
    flow_ub = zeros(L.n_flow)
    for l in 1:nL, dir in 1:2, t in 1:T, s in 1:S
        flow_ub[es_flow(l, dir, t, s, T, S)] = prob.line_capacity[l] * ES_PJ_PER_GW_YEAR * dur[s]
    end
    @variable(model, 0 <= flow[i = 1:L.n_flow] <= flow_ub[i])

    bal = [AffExpr(0.0) for _ in 1:(R * NC * T)]
    bal_elc = [AffExpr(0.0) for _ in 1:(R * T * S)]
    emis = [AffExpr(0.0) for _ in 1:T]
    peak = [AffExpr(0.0) for _ in 1:R, _ in 1:T]
    bal_id(r, c, t) = ((r - 1) * NC + c - 1) * T + t
    elc_id(r, t, s) = ((r - 1) * T + t - 1) * S + s
    obj = AffExpr(0.0)
    for k in 1:ntech
        r = prob.tech_region[k]
        ci, co, co2 = prob.tech_input[k], prob.tech_output[k], prob.tech_output2[k]
        a_in, a2 = prob.tech_input_coef[k], prob.tech_output2_coef[k]
        e = prob.tech_emission[k]
        prof = prob.tech_profile[k]
        for t in 1:T
            w = prob.discount_factor[t] * Δ
            for s in 1:L.slots[k]
                v = act[es_act(L, k, t, s)]
                if prob.tech_kind[k] == :gen
                    add_to_expression!(bal_elc[elc_id(r, t, s)], 1.0, v)
                    co2 > 0 && add_to_expression!(bal[bal_id(r, co2, t)], a2, v)
                else
                    add_to_expression!(bal[bal_id(r, co, t)], 1.0, v)
                end
                if ci == -1
                    for s2 in 1:S
                        add_to_expression!(
                            bal_elc[elc_id(r, t, s2)], -a_in * prob.profiles[prof, s2], v
                        )
                        s2 == prob.peak_slice[r] && add_to_expression!(
                            peak[r, t],
                            -(1 + prob.reserve_margin) * a_in * prob.profiles[prof, s2] /
                            (ES_PJ_PER_GW_YEAR * dur[s2]),
                            v,
                        )
                    end
                elseif ci > 0
                    add_to_expression!(bal[bal_id(r, ci, t)], -a_in, v)
                end
                e != 0 && add_to_expression!(emis[t], e, v)
                add_to_expression!(obj, w * prob.tech_varcost[k], v)
            end
        end
    end
    for l in 1:nL
        a, b = prob.lines[l]
        for dir in 1:2, t in 1:T, s in 1:S
            from, to = dir == 1 ? (a, b) : (b, a)
            v = flow[es_flow(l, dir, t, s, T, S)]
            add_to_expression!(bal_elc[elc_id(from, t, s)], -1.0, v)
            add_to_expression!(bal_elc[elc_id(to, t, s)], prob.line_efficiency[l], v)
            add_to_expression!(obj, prob.discount_factor[t] * Δ * prob.line_cost[l], v)
        end
    end

    # Balances: equality for fuels, >= for heat, electricity and demands.
    rhs = zeros(R * NC * T)
    sense_eq = falses(R * NC * T)
    for r in 1:R, c in 1:NC, t in 1:T
        i = bal_id(r, c, t)
        sym = prob.commodities[c]
        rhs[i] = prob.demand[r, c, t]
        sense_eq[i] = sym in (:COA, :GAS, :OIL, :URN, :BIO, :PET, :H2)
    end
    @constraint(
        model,
        bal_c[i = 1:(R * NC * T)],
        bal[i] in (sense_eq[i] ? MOI.EqualTo(rhs[i]) : MOI.GreaterThan(rhs[i]))
    )
    model[:bal] = bal_c
    @constraint(model, bal_elc_c[i = 1:(R * T * S)], bal_elc[i] >= 0)
    model[:bal_elc] = bal_elc_c

    # Capacity-activity, transfer, growth, peak.
    capact_cols = Int[]
    capact_cap = Int[]
    capact_coef = Float64[]
    for k in 1:ntech
        L.cap_index[k] == 0 && continue
        for t in 1:T, s in 1:L.slots[k]
            push!(capact_cols, es_act(L, k, t, s))
            push!(capact_cap, es_cap(L, k, t, T))
            push!(
                capact_coef,
                prob.tech_avail[k, s] *
                prob.tech_capfac[k] *
                (prob.tech_kind[k] == :gen ? dur[s] : 1.0),
            )
        end
    end
    @constraint(
        model,
        capact_c[i = 1:length(capact_cols)],
        act[capact_cols[i]] - capact_coef[i] * cap[capact_cap[i]] <= 0
    )
    model[:capact] = Containers.DenseAxisArray(collect(capact_c), capact_cols)

    transfer_ids = Int[]
    growth_ids = Int[]
    for k in 1:ntech
        L.cap_index[k] == 0 && continue
        life = prob.tech_life[k]
        for t in 1:T
            c = es_cap(L, k, t, T)
            push!(transfer_ids, c)
            t >= 2 && prob.tech_growth[k] > 0 && push!(growth_ids, c)
        end
    end
    tech_of_cap = zeros(Int, L.n_capt)
    t_of_cap = zeros(Int, L.n_capt)
    for k in 1:ntech
        L.cap_index[k] == 0 && continue
        for t in 1:T
            tech_of_cap[es_cap(L, k, t, T)] = k
            t_of_cap[es_cap(L, k, t, T)] = t
        end
    end
    @constraint(
        model,
        transfer[c in transfer_ids],
        cap[c] - sum(
            ncap[c - t_of_cap[c] + v] for
            v in max(1, t_of_cap[c] - prob.tech_life[tech_of_cap[c]] + 1):t_of_cap[c]
        ) == prob.tech_residual[tech_of_cap[c], t_of_cap[c]]
    )
    @constraint(
        model,
        growth[c in growth_ids],
        ncap[c] - prob.tech_growth[tech_of_cap[c]] * ncap[c - 1] <= prob.tech_seed[tech_of_cap[c]]
    )
    for k in 1:ntech
        prob.tech_kind[k] == :gen || continue
        r = prob.tech_region[k]
        for t in 1:T
            add_to_expression!(peak[r, t], prob.tech_credit[k], cap[es_cap(L, k, t, T)])
        end
    end
    @constraint(model, peak_c[r = 1:R, t = 1:T], peak[r, t] >= 0)
    model[:peak] = peak_c

    @constraint(model, emission[t = 1:T], emis[t] <= prob.emission_cap[t])
    @constraint(model, budget, sum(Δ * emis[t] for t in 1:T) <= prob.emission_budget)
    reserve_ids = [
        k for k in 1:ntech if prob.tech_kind[k] == :supply && isfinite(prob.supply_reserve[k])
    ]
    @constraint(
        model,
        reserve[k in reserve_ids],
        sum(Δ * act[es_act(L, k, t)] for t in 1:T) <= prob.supply_reserve[k]
    )

    # Costs: investment with salvage, fixed O&M on installed capacity.
    for k in 1:ntech
        L.cap_index[k] == 0 && continue
        life = prob.tech_life[k]
        for t in 1:T
            c = es_cap(L, k, t, T)
            salvage = min(1.0, (T - t + 1) / life)
            add_to_expression!(
                obj, prob.discount_factor[t] * salvage * prob.tech_invcost[k], ncap[c]
            )
            add_to_expression!(obj, prob.discount_factor[t] * Δ * prob.tech_fixcost[k], cap[c])
        end
    end
    @objective(model, Min, obj)
    return model
end

register_variant(
    :economic_planning,
    :energy_system,
    EnergySystemProblem,
    "TIMES/MESSAGE-style multi-region energy-system capacity-expansion LP: supply steps, power plants by timeslice, conversion and end-use device classes, vintaged capacity, growth limits, peak reserve, emission caps and carbon budget; planted reference-mix witness and dual-value (emission/supply) Farkas certificates";
    tags=[:energy, :staircase],
    max_target_variables=1_000_000,
)
