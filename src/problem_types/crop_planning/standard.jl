using JuMP
using Random
using Distributions
using StatsBase

# Crop catalog: family, whether it needs irrigation, irrigated yield (t/ha),
# rainfed yield factor, price (USD/t), variable cost (USD/ha), nitrogen need
# and residual credit to next year's crop (kg N/ha), irrigation water (m³/ha)
# and labour (h/ha) in spring, summer, autumn and winter.
const _CROP_CATALOG = (
    (name=:winter_wheat, family=:cereal, irrigated_only=false, yield=7.5, rainfed=0.75, price=230.0, cost=650.0,
        n_need=160.0, n_credit=0.0, water=3500.0, labor=(3.0, 6.0, 5.0, 1.0)),
    (name=:maize, family=:cereal, irrigated_only=false, yield=11.0, rainfed=0.60, price=200.0, cost=900.0,
        n_need=200.0, n_credit=0.0, water=6000.0, labor=(6.0, 4.0, 6.0, 0.0)),
    (name=:barley, family=:cereal, irrigated_only=false, yield=6.0, rainfed=0.80, price=210.0, cost=550.0,
        n_need=120.0, n_credit=0.0, water=2500.0, labor=(5.0, 5.0, 1.0, 0.0)),
    (name=:sorghum, family=:cereal, irrigated_only=false, yield=6.5, rainfed=0.80, price=190.0, cost=500.0,
        n_need=110.0, n_credit=0.0, water=3500.0, labor=(5.0, 3.0, 4.0, 0.0)),
    (name=:rice, family=:cereal, irrigated_only=true, yield=8.0, rainfed=0.0, price=330.0, cost=1200.0,
        n_need=140.0, n_credit=0.0, water=12000.0, labor=(12.0, 8.0, 8.0, 0.0)),
    (name=:oats, family=:cereal, irrigated_only=false, yield=4.5, rainfed=0.85, price=200.0, cost=450.0,
        n_need=90.0, n_credit=0.0, water=2500.0, labor=(5.0, 4.0, 1.0, 0.0)),
    (name=:soybean, family=:legume, irrigated_only=false, yield=3.6, rainfed=0.70, price=450.0, cost=600.0,
        n_need=20.0, n_credit=50.0, water=4500.0, labor=(5.0, 3.0, 5.0, 0.0)),
    (name=:field_pea, family=:legume, irrigated_only=false, yield=3.5, rainfed=0.80, price=330.0, cost=500.0,
        n_need=15.0, n_credit=45.0, water=2500.0, labor=(5.0, 4.0, 1.0, 0.0)),
    (name=:chickpea, family=:legume, irrigated_only=false, yield=2.2, rainfed=0.80, price=700.0, cost=550.0,
        n_need=15.0, n_credit=35.0, water=2500.0, labor=(5.0, 4.0, 1.0, 0.0)),
    (name=:lentil, family=:legume, irrigated_only=false, yield=1.8, rainfed=0.80, price=750.0, cost=500.0,
        n_need=15.0, n_credit=35.0, water=2000.0, labor=(5.0, 4.0, 1.0, 0.0)),
    (name=:canola, family=:oilseed, irrigated_only=false, yield=3.6, rainfed=0.75, price=500.0, cost=700.0,
        n_need=170.0, n_credit=10.0, water=3000.0, labor=(2.0, 6.0, 5.0, 0.0)),
    (name=:sunflower, family=:oilseed, irrigated_only=false, yield=2.8, rainfed=0.80, price=480.0, cost=550.0,
        n_need=90.0, n_credit=5.0, water=3500.0, labor=(5.0, 3.0, 5.0, 0.0)),
    (name=:cotton, family=:fiber, irrigated_only=true, yield=1.8, rainfed=0.0, price=1600.0, cost=1400.0,
        n_need=150.0, n_credit=0.0, water=8000.0, labor=(8.0, 12.0, 10.0, 0.0)),
    (name=:sugar_beet, family=:root, irrigated_only=false, yield=70.0, rainfed=0.70, price=45.0, cost=1600.0,
        n_need=140.0, n_credit=0.0, water=6000.0, labor=(10.0, 6.0, 14.0, 0.0)),
    (name=:potato, family=:root, irrigated_only=true, yield=45.0, rainfed=0.0, price=180.0, cost=4500.0,
        n_need=180.0, n_credit=0.0, water=5000.0, labor=(25.0, 20.0, 30.0, 0.0)),
    (name=:processing_tomato, family=:vegetable, irrigated_only=true, yield=90.0, rainfed=0.0, price=85.0,
        cost=5500.0, n_need=160.0, n_credit=0.0, water=7000.0, labor=(30.0, 60.0, 10.0, 0.0)),
    (name=:onion, family=:vegetable, irrigated_only=true, yield=50.0, rainfed=0.0, price=200.0, cost=5500.0,
        n_need=130.0, n_credit=0.0, water=6000.0, labor=(40.0, 50.0, 20.0, 0.0)),
    (name=:alfalfa, family=:forage, irrigated_only=false, yield=12.0, rainfed=0.60, price=170.0, cost=900.0,
        n_need=0.0, n_credit=80.0, water=9000.0, labor=(10.0, 15.0, 8.0, 0.0)),
)

"""Crop families that must not follow themselves on a field (disease/pest breaks)."""
const CROP_ROTATION_FAMILIES = (:legume, :oilseed, :root, :vegetable)
const CROP_SEASONS = (:spring, :summer, :autumn, :winter)
const CROP_SOILS = (:loam, :clay, :sandy, :silt)
# Soil suitability multipliers by crop family (0 = not grown on that soil).
const _CROP_SOIL_FACTOR = Dict(
    :loam => Dict(:cereal => 1.0, :legume => 1.0, :oilseed => 1.0, :fiber => 1.0, :root => 1.0, :vegetable => 1.0, :forage => 1.0),
    :clay => Dict(:cereal => 1.05, :legume => 0.9, :oilseed => 1.0, :fiber => 1.0, :root => 0.75, :vegetable => 0.85, :forage => 1.0),
    :sandy => Dict(:cereal => 0.85, :legume => 0.9, :oilseed => 0.9, :fiber => 0.9, :root => 1.1, :vegetable => 1.0, :forage => 0.85),
    :silt => Dict(:cereal => 1.0, :legume => 1.05, :oilseed => 1.0, :fiber => 1.0, :root => 0.95, :vegetable => 1.05, :forage => 1.0),
)

"""
Reason a requested-infeasible crop plan cannot exist.

  - `crop_land_shortage`: the contracted production of every crop in a region
    and year needs more land (at the best yield any field there achieves) than
    the region has.
  - `crop_water_shortage`: the contracted production of the region's
    irrigation-only crops needs more irrigation water (at the most
    water-efficient field) than the district's whole-year allocation.
"""
@enum CropInfeasibilityKind begin
    crop_land_shortage
    crop_water_shortage
end

"""
    CropInfeasibilityCertificate

LP-row proof for `CropPlanningProblem`: region `region`, year `year`; for a land
shortage `achievable` is the region's land and `required` the minimum land the
contracts need, for a water shortage they are the water allocation and the
minimum water the contracted irrigation-only crops need. `achievable <
required` by at least 7%.
"""
struct CropInfeasibilityCertificate
    kind::CropInfeasibilityKind
    region::Int
    year::Int
    achievable::Float64
    required::Float64
end

"""
    CropPlan

A complete crop plan: `area[k]` hectares on area variable `k` (`area_vars`),
`fertilizer[j, t]` kg N per field-year, `hired[farm, season, t]` hours, and
`sales[c, r, t, tier]` tonnes.
"""
struct CropPlan
    area::Vector{Float64}
    fertilizer::Matrix{Float64}
    hired::Array{Float64, 3}
    sales::Array{Float64, 4}
end

"""
    CropPlanningProblem <: ProblemGenerator

Regional multi-year crop-rotation planning: farms in several irrigation
districts / market regions decide the crop mix of every field for a horizon of
years, linked by rotation rules, nitrogen carry-over, seasonal family and hired
labour, district water allocations and tiered regional markets with processing
contracts.

# Formulation

Indices: crops `c`, regions `r`, farms, fields `j` (area `A_j`, soil, irrigable
or rainfed, in one farm and region), years `t = 1..T`, seasons `m`, price tiers
`k`. Variables (all ≥ 0):

  - `a[j,c,t]` ha of crop `c` on field `j` in year `t` (allowed pairs only:
    soil suitability, irrigation-only crops on irrigable fields, the farm's
    crop list);
  - `fert[j,t]` purchased nitrogen (kg);
  - `hire[farm,m,t] ≤ hire_cap` seasonal hired labour (h);
  - `sell[c,r,t,k] ≤ tier_width` sales in price tier `k` (falling prices).

Maximize revenue − crop costs − fertilizer − hired labour. Rows:

  - land `Σ_c a[j,c,t] ≤ A_j`;
  - rotation `Σ_{c∈F} (a[j,c,t] + a[j,c,t+1]) ≤ A_j` for each break family
    `F` (legumes, oilseeds, roots, vegetables) grown on the field;
  - nitrogen `fert[j,t] + Σ_c credit_c a[j,c,t−1] − Σ_c need_c a[j,c,t] ≥ −N0_j[t=1]`
    (legumes and alfalfa leave nitrogen for the next crop);
  - farm fertilizer quota `Σ_{j∈farm} fert[j,t] ≤ quota`;
  - seasonal labour `Σ labour_cm a − hire[farm,m,t] ≤ family_labour` and a farm
    seasonal-hire budget `Σ_m hire[farm,m,t] ≤ hire_budget`;
  - district water per irrigation season `Σ water_c share_m a ≤ allocation`;
  - market `Σ_k sell[c,r,t,k] ≤ Σ_j yield_jc a[j,c,t]` and processing/food
    security contracts `Σ_j yield_jc a[j,c,t] ≥ contract[c,r,t]`.

# Sizing

Crops `≈ 2 + target^0.25` (3–18), years `≈ 1.5 + log10(target)` (2–8),
regions `≈ target / 12000` (1–12). Farms of 3–15 fields are added until the
variable count (area pairs + fertilizer + hire + sales tiers) reaches the
target; the last field's crop list is trimmed so the count lands within
about one year-block (`T`) of it.

# Feasibility

  - `feasible`: a planted rotation (each field's crop sequence avoids repeating
    a break family) with exact fertilizer and hired labour; capacities,
    quotas and allocations 1.05–1.30× its use and contracts 50–90% of its
    production. Stored as a `CropPlan` witness.
  - `infeasible`: contracts raised past a regional land (default when a region
    has no irrigation-only crop) or district water limit, with a typed
    `CropInfeasibilityCertificate` — an aggregation of land/water, production
    and contract rows, not a single-row contradiction.
  - `unknown`: nominal allocations and contracts; may go either way.
"""
struct CropPlanningProblem <: ProblemGenerator
    crops::Vector{Int}
    n_regions::Int
    n_years::Int
    n_tiers::Int
    field_area::Vector{Float64}
    field_farm::Vector{Int}
    field_region::Vector{Int}
    field_soil::Vector{Symbol}
    field_irrigable::Vector{Bool}
    farm_region::Vector{Int}
    area_vars::Vector{Tuple{Int, Int, Int}}
    yields::Dict{Tuple{Int, Int}, Float64}
    crop_cost::Vector{Float64}
    price::Matrix{Float64}
    tier_width::Array{Float64, 3}
    tier_factor::Vector{Float64}
    initial_nitrogen::Vector{Float64}
    fert_price::Float64
    fert_quota::Matrix{Float64}
    family_labor::Array{Float64, 3}
    hire_cap::Array{Float64, 3}
    hire_budget::Matrix{Float64}
    hire_cost::Float64
    water_allocation::Array{Float64, 3}
    contract::Array{Float64, 3}
    feasible_witness::Union{Nothing, CropPlan}
    infeasibility_certificate::Union{Nothing, CropInfeasibilityCertificate}
    feasibility_status::FeasibilityStatus
end

const _CROP_WATER_SEASONS = ((1, 0.35), (2, 0.65))  # spring and summer shares

_crop_spec(prob::CropPlanningProblem, c::Int) = _CROP_CATALOG[prob.crops[c]]

function _crop_field_allowed(rng::AbstractRNG, crops::Vector{Int}, soil::Symbol, irrigable::Bool)
    allowed = Int[]
    for (c, ci) in enumerate(crops)
        spec = _CROP_CATALOG[ci]
        spec.irrigated_only && !irrigable && continue
        _CROP_SOIL_FACTOR[soil][spec.family] > 0 || continue
        rand(rng) < 0.85 && push!(allowed, c)
    end
    isempty(allowed) && push!(allowed, findfirst(c -> !_CROP_CATALOG[c].irrigated_only, crops))
    return allowed
end

"""
    crop_plan_satisfies(prob, plan=prob.feasible_witness; atol=1e-6)

Check a `CropPlan` against every bound and row of the model, without a solver.
"""
function crop_plan_satisfies(
    prob::CropPlanningProblem, plan::Union{Nothing, CropPlan}=prob.feasible_witness; atol::Float64=1e-6
)
    plan === nothing && return false
    a, fert, hire, sell = plan.area, plan.fertilizer, plan.hired, plan.sales
    all(>=(-atol), a) && all(>=(-atol), fert) && all(>=(-atol), hire) && all(>=(-atol), sell) || return false
    tol(v) = atol * max(1.0, abs(v))
    J, T, R, C = length(prob.field_area), prob.n_years, prob.n_regions, length(prob.crops)
    F = length(prob.farm_region)
    land = zeros(J, T)
    family = Dict{Tuple{Int, Symbol, Int}, Float64}()
    nitrogen = zeros(J, T)
    credit = zeros(J, T)
    labor = zeros(F, length(CROP_SEASONS), T)
    water = zeros(R, length(CROP_SEASONS), T)
    production = zeros(C, R, T)
    for (k, (j, c, t)) in enumerate(prob.area_vars)
        spec = _crop_spec(prob, c)
        land[j, t] += a[k]
        if spec.family in CROP_ROTATION_FAMILIES
            family[(j, spec.family, t)] = get(family, (j, spec.family, t), 0.0) + a[k]
        end
        nitrogen[j, t] += spec.n_need * a[k]
        t < T && (credit[j, t + 1] += spec.n_credit * a[k])
        for m in eachindex(CROP_SEASONS)
            labor[prob.field_farm[j], m, t] += spec.labor[m] * a[k]
        end
        if prob.field_irrigable[j]
            for (m, share) in _CROP_WATER_SEASONS
                water[prob.field_region[j], m, t] += spec.water * share * a[k]
            end
        end
        production[c, prob.field_region[j], t] += prob.yields[(j, c)] * a[k]
    end
    for j in 1:J, t in 1:T
        land[j, t] <= prob.field_area[j] + tol(prob.field_area[j]) || return false
        n0 = t == 1 ? prob.initial_nitrogen[j] : 0.0
        fert[j, t] + credit[j, t] + n0 + tol(nitrogen[j, t]) >= nitrogen[j, t] || return false
    end
    for ((j, fam, t), area) in family
        t < T || continue
        area + get(family, (j, fam, t + 1), 0.0) <= prob.field_area[j] + tol(prob.field_area[j]) || return false
    end
    for f in 1:F, t in 1:T
        used = sum(fert[j, t] for j in eachindex(prob.field_farm) if prob.field_farm[j] == f; init=0.0)
        used <= prob.fert_quota[f, t] + tol(used) || return false
        sum(hire[f, :, t]) <= prob.hire_budget[f, t] + tol(prob.hire_budget[f, t]) || return false
        for m in eachindex(CROP_SEASONS)
            hire[f, m, t] <= prob.hire_cap[f, m, t] + tol(prob.hire_cap[f, m, t]) || return false
            labor[f, m, t] - hire[f, m, t] <= prob.family_labor[f, m, t] + tol(labor[f, m, t]) || return false
        end
    end
    for r in 1:R, t in 1:T, (m, _) in _CROP_WATER_SEASONS
        water[r, m, t] <= prob.water_allocation[r, m, t] + tol(water[r, m, t]) || return false
    end
    for c in 1:C, r in 1:R, t in 1:T
        sold = sum(sell[c, r, t, :])
        sold <= production[c, r, t] + tol(production[c, r, t]) || return false
        production[c, r, t] + tol(production[c, r, t]) >= prob.contract[c, r, t] || return false
        for k in 1:prob.n_tiers
            sell[c, r, t, k] <= prob.tier_width[c, r, k] + tol(prob.tier_width[c, r, k]) || return false
        end
    end
    return true
end

function _crop_best_yield(prob::CropPlanningProblem, c::Int, r::Int)
    best = 0.0
    for (j, cc, t) in prob.area_vars
        (cc == c && t == 1 && prob.field_region[j] == r) || continue
        best = max(best, prob.yields[(j, c)])
    end
    return best
end

function _crop_certificate_value(prob::CropPlanningProblem, kind::CropInfeasibilityKind, r::Int, t::Int)
    C = length(prob.crops)
    if kind == crop_land_shortage
        achievable = sum(prob.field_area[j] for j in eachindex(prob.field_area) if prob.field_region[j] == r)
        required = 0.0
        for c in 1:C
            prob.contract[c, r, t] > 0 || continue
            y = _crop_best_yield(prob, c, r)
            required += y > 0 ? prob.contract[c, r, t] / y : Inf
        end
    else
        achievable = sum(prob.water_allocation[r, m, t] for (m, _) in _CROP_WATER_SEASONS)
        required = 0.0
        for c in 1:C
            spec = _crop_spec(prob, c)
            (spec.irrigated_only && prob.contract[c, r, t] > 0) || continue
            y = _crop_best_yield(prob, c, r)
            required += y > 0 ? prob.contract[c, r, t] * spec.water / y : Inf
        end
    end
    return achievable, required
end

"""
    crop_certificate_holds(prob::CropPlanningProblem)

Recompute the stored certificate from the data and check `achievable < required`.
"""
function crop_certificate_holds(prob::CropPlanningProblem)
    cert = prob.infeasibility_certificate
    cert === nothing && return false
    1 <= cert.region <= prob.n_regions && 1 <= cert.year <= prob.n_years || return false
    achievable, required = _crop_certificate_value(prob, cert.kind, cert.region, cert.year)
    isapprox(achievable, cert.achievable; rtol=1e-9) && isapprox(required, cert.required; rtol=1e-9) || return false
    return achievable < required * (1 - 1e-9)
end

"""
    CropPlanningProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a regional crop-rotation planning instance (see the type).
"""
function CropPlanningProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)
    n_crops = clamp(round(Int, 2 + target^0.25), 3, length(_CROP_CATALOG))
    T = clamp(round(Int, 1.5 + log10(target)), 2, 8)
    R = clamp(round(Int, target / 12000), 1, 12)
    K = target < 1000 ? 1 : 3
    S = length(CROP_SEASONS)
    # Always one cereal and one legume; the rest at random.
    cereals = [i for i in eachindex(_CROP_CATALOG) if _CROP_CATALOG[i].family == :cereal && !_CROP_CATALOG[i].irrigated_only]
    legumes = [i for i in eachindex(_CROP_CATALOG) if _CROP_CATALOG[i].family == :legume]
    core = [rand(rng, cereals), rand(rng, legumes)]
    others = shuffle(rng, setdiff(collect(eachindex(_CROP_CATALOG)), core))
    crops = sort(vcat(core, others[1:(n_crops - 2)]))
    C = length(crops)
    irrigable_share = rand(rng, Uniform(0.2, 0.8), R)

    field_area = Float64[]
    field_farm = Int[]
    field_region = Int[]
    field_soil = Symbol[]
    field_irrigable = Bool[]
    farm_region = Int[]
    area_vars = Tuple{Int, Int, Int}[]
    allowed_of = Vector{Vector{Int}}()
    count = C * R * T * K
    fields_left = 0
    while count < target || isempty(field_area)
        if fields_left == 0
            push!(farm_region, (length(farm_region) % R) + 1)
            fields_left = rand(rng, 3:15)
            count += S * T
        end
        f = length(farm_region)
        r = farm_region[f]
        j = length(field_area) + 1
        soil = rand(rng, CROP_SOILS)
        irrigable = rand(rng) < irrigable_share[r]
        push!(field_area, clamp(rand(rng, LogNormal(log(25.0), 0.7)), 2.0, 300.0))
        push!(field_farm, f)
        push!(field_region, r)
        push!(field_soil, soil)
        push!(field_irrigable, irrigable)
        allowed = _crop_field_allowed(rng, crops, soil, irrigable)
        # The last field's crop list is trimmed so the count lands on target.
        if count + (length(allowed) + 1) * T > target && !isempty(field_area[1:(end - 1)])
            keep = clamp(round(Int, (target - count) / T) - 1, 1, length(allowed))
            allowed = allowed[1:keep]
        end
        push!(allowed_of, allowed)
        for t in 1:T, c in allowed
            push!(area_vars, (j, c, t))
        end
        count += length(allowed) * T + T
        fields_left -= 1
    end
    J = length(field_area)
    Fm = length(farm_region)

    yields = Dict{Tuple{Int, Int}, Float64}()
    for j in 1:J, c in allowed_of[j]
        spec = _CROP_CATALOG[crops[c]]
        factor = _CROP_SOIL_FACTOR[field_soil[j]][spec.family] * (field_irrigable[j] ? 1.0 : spec.rainfed)
        yields[(j, c)] = spec.yield * factor * rand(rng, LogNormal(0.0, 0.10))
    end
    crop_cost = [_CROP_CATALOG[ci].cost * rand(rng, LogNormal(0.0, 0.08)) for ci in crops]
    price = [_CROP_CATALOG[crops[c]].price * rand(rng, LogNormal(0.0, 0.10)) for c in 1:C, t in 1:T]
    tier_factor = K == 1 ? [1.0] : [1.0, 0.85, 0.65]
    initial_nitrogen = [rand(rng, Uniform(0.0, 60.0)) * field_area[j] for j in 1:J]
    fert_price = rand(rng, Uniform(0.9, 1.4))
    hire_cost = rand(rng, Uniform(15.0, 25.0))

    # Planted rotation: each field's crop sequence never repeats a break
    # family in consecutive years; 75-100% of the field is cropped.
    index = Dict(v => n for (n, v) in enumerate(area_vars))
    a = zeros(Float64, length(area_vars))
    for j in 1:J
        previous_family = :none
        previous_area = Dict{Symbol, Float64}()
        for t in 1:T
            options = [c for c in allowed_of[j] if !(_CROP_CATALOG[crops[c]].family == previous_family &&
                                                     previous_family in CROP_ROTATION_FAMILIES)]
            isempty(options) && (options = allowed_of[j])
            c1 = rand(rng, options)
            share = rand(rng, Uniform(0.75, 1.0))
            if length(options) >= 2 && rand(rng) < 0.4
                c2 = rand(rng, setdiff(options, [c1]))
                split = rand(rng, Uniform(0.4, 0.8))
                a[index[(j, c1, t)]] += share * split * field_area[j]
                a[index[(j, c2, t)]] += share * (1 - split) * field_area[j]
            else
                a[index[(j, c1, t)]] += share * field_area[j]
            end
            # Rotation rows: a break family's area this year plus last year's
            # must fit the field; trim (leave fallow) where it would not.
            current_area = Dict{Symbol, Float64}()
            for c in allowed_of[j]
                fam = _CROP_CATALOG[crops[c]].family
                fam in CROP_ROTATION_FAMILIES || continue
                current_area[fam] = get(current_area, fam, 0.0) + a[index[(j, c, t)]]
            end
            for (fam, area) in current_area
                room = 0.999 * field_area[j] - get(previous_area, fam, 0.0)
                area > room || continue
                for c in allowed_of[j]
                    _CROP_CATALOG[crops[c]].family == fam || continue
                    a[index[(j, c, t)]] *= max(room, 0.0) / area
                end
                current_area[fam] = max(room, 0.0)
            end
            previous_area = current_area
            fams = unique([_CROP_CATALOG[crops[c]].family for c in allowed_of[j] if a[index[(j, c, t)]] > 0])
            previous_family = length(fams) == 1 ? only(fams) : :mixed
        end
    end
    # Usage of the planted plan.
    need = zeros(J, T)
    credit = zeros(J, T)
    labor = zeros(Fm, S, T)
    water = zeros(R, S, T)
    production = zeros(C, R, T)
    for (k, (j, c, t)) in enumerate(area_vars)
        spec = _CROP_CATALOG[crops[c]]
        need[j, t] += spec.n_need * a[k]
        t < T && (credit[j, t + 1] += spec.n_credit * a[k])
        for m in 1:S
            labor[field_farm[j], m, t] += spec.labor[m] * a[k]
        end
        if field_irrigable[j]
            for (m, share) in _CROP_WATER_SEASONS
                water[field_region[j], m, t] += spec.water * share * a[k]
            end
        end
        production[c, field_region[j], t] += yields[(j, c)] * a[k]
    end
    fert = [max(0.0, need[j, t] - credit[j, t] - (t == 1 ? initial_nitrogen[j] : 0.0)) for j in 1:J, t in 1:T]
    # Market tiers sized on the planted production (unknown: on the same
    # nominal plan): the full-price tier takes 40-80% of it.
    tier_width = fill(Inf, C, R, K)
    for c in 1:C, r in 1:R
        nominal = max(sum(production[c, r, :]) / T, 1.0)
        if K == 3
            tier_width[c, r, 1] = nominal * rand(rng, Uniform(0.4, 0.8))
            tier_width[c, r, 2] = nominal * rand(rng, Uniform(0.2, 0.5))
        end
    end

    family_labor = zeros(Fm, S, T)
    hire_cap = zeros(Fm, S, T)
    hire_budget = zeros(Fm, T)
    fert_quota = zeros(Fm, T)
    water_allocation = zeros(R, S, T)
    contract = zeros(C, R, T)
    witness = nothing
    if feasibility_status == unknown
        for f in 1:Fm, t in 1:T
            for m in 1:S
                family_labor[f, m, t] = labor[f, m, t] * rand(rng, Uniform(0.5, 1.2))
                hire_cap[f, m, t] = max(labor[f, m, t], 10.0) * rand(rng, Uniform(0.2, 0.8))
            end
            hire_budget[f, t] = sum(hire_cap[f, :, t]) * rand(rng, Uniform(0.5, 1.0))
            farm_fert = sum(fert[j, t] for j in 1:J if field_farm[j] == f)
            fert_quota[f, t] = max(farm_fert, 100.0) * rand(rng, Uniform(0.7, 1.5))
        end
        tightness = rand(rng, Uniform(0.6, 1.4))
        for r in 1:R, t in 1:T, (m, _) in _CROP_WATER_SEASONS
            water_allocation[r, m, t] = max(water[r, m, t], 1000.0) * tightness * rand(rng, Uniform(0.8, 1.25))
        end
        # Food-security / processing mandates: one instance-wide ambition
        # level applied to every crop's nominal production (with crop noise),
        # so the region may or may not be able to grow all of it.
        ambition = rand(rng, Uniform(0.5, 1.3))
        for c in 1:C, r in 1:R, t in 1:T
            production[c, r, t] > 0 && rand(rng) < 0.8 || continue
            contract[c, r, t] = production[c, r, t] * ambition * rand(rng, Uniform(0.85, 1.15))
        end
    else
        hired = zeros(Fm, S, T)
        for f in 1:Fm, t in 1:T
            for m in 1:S
                family_labor[f, m, t] = labor[f, m, t] * rand(rng, Uniform(0.6, 1.05))
                hired[f, m, t] = max(0.0, labor[f, m, t] - family_labor[f, m, t])
                hire_cap[f, m, t] = hired[f, m, t] * rand(rng, Uniform(1.05, 1.5)) + 0.1 * labor[f, m, t] + 1.0
            end
            hire_budget[f, t] = sum(hired[f, :, t]) * rand(rng, Uniform(1.05, 1.3)) + 1.0
            farm_fert = sum(fert[j, t] for j in 1:J if field_farm[j] == f)
            fert_quota[f, t] = farm_fert * rand(rng, Uniform(1.05, 1.3)) + 10.0
        end
        for r in 1:R, t in 1:T, (m, _) in _CROP_WATER_SEASONS
            water_allocation[r, m, t] = water[r, m, t] * rand(rng, Uniform(1.05, 1.30)) + 100.0
        end
        for c in 1:C, r in 1:R, t in 1:T
            production[c, r, t] > 0 && rand(rng) < 0.5 || continue
            contract[c, r, t] = production[c, r, t] * rand(rng, Uniform(0.5, 0.9))
        end
        sell = zeros(Float64, C, R, T, K)
        for c in 1:C, r in 1:R, t in 1:T
            left = production[c, r, t]
            for k in 1:K
                q = min(left, tier_width[c, r, k])
                sell[c, r, t, k] = q
                left -= q
            end
        end
        witness = CropPlan(a, fert, hired, sell)
    end

    prob = CropPlanningProblem(
        crops, R, T, K, field_area, field_farm, field_region, field_soil, field_irrigable, farm_region, area_vars,
        yields, crop_cost, price, tier_width, tier_factor, initial_nitrogen, fert_price, fert_quota, family_labor,
        hire_cap, hire_budget, hire_cost, water_allocation, contract, witness, nothing, feasibility_status,
    )
    if feasibility_status == infeasible
        r = rand(rng, 1:R)
        t = rand(rng, 1:T)
        # Largest production of each crop that no single row can rule out:
        # every field at its implied bound (its area, and on irrigable fields
        # the season allocation over the crop's seasonal water need). Keeping
        # each contract at most 80% of it leaves the contradiction to the
        # aggregate of several crops' land or water — invisible to presolve.
        solo = zeros(Float64, C)
        for j in 1:J, c in allowed_of[j]
            field_region[j] == r || continue
            spec = _CROP_CATALOG[crops[c]]
            bound = field_area[j]
            if field_irrigable[j]
                bound = min(bound, minimum(water_allocation[r, m, t] / (spec.water * share) for (m, share) in _CROP_WATER_SEASONS))
            end
            solo[c] += yields[(j, c)] * bound
        end
        irrigated = [c for c in 1:C if _CROP_CATALOG[crops[c]].irrigated_only && solo[c] > 0]
        kind = length(irrigated) >= 2 && rand(rng) < 0.5 ? crop_water_shortage : crop_land_shortage
        local achievable, required
        for attempt in 1:2
            targets = kind == crop_water_shortage ? irrigated : [c for c in 1:C if solo[c] > 0]
            base = [max(contract[c, r, t], production[c, r, t], 0.05 * solo[c]) for c in targets]
            caps = [0.8 * solo[c] for c in targets]
            for (n, c) in enumerate(targets)
                contract[c, r, t] = min(base[n], caps[n])
            end
            achievable, _ = _crop_certificate_value(prob, kind, r, t)
            goal = achievable * rand(rng, Uniform(1.08, 1.25))
            for (n, c) in enumerate(targets)
                contract[c, r, t] = caps[n]
            end
            _, ceiling = _crop_certificate_value(prob, kind, r, t)
            if ceiling < goal && attempt == 1
                kind = kind == crop_water_shortage ? crop_land_shortage : crop_water_shortage
                (kind == crop_water_shortage && length(irrigated) < 2) && (kind = crop_land_shortage)
                continue
            end
            # Bisection on a common scale of the base contracts, each capped.
            lo_s, hi_s = 0.0, 1.0
            while true
                for (n, c) in enumerate(targets)
                    contract[c, r, t] = min(hi_s * base[n], caps[n])
                end
                (_crop_certificate_value(prob, kind, r, t)[2] >= goal || hi_s > 1e6) && break
                hi_s *= 2
            end
            for _ in 1:60
                mid = (lo_s + hi_s) / 2
                for (n, c) in enumerate(targets)
                    contract[c, r, t] = min(mid * base[n], caps[n])
                end
                _crop_certificate_value(prob, kind, r, t)[2] >= goal ? (hi_s = mid) : (lo_s = mid)
            end
            for (n, c) in enumerate(targets)
                contract[c, r, t] = min(hi_s * base[n], caps[n])
            end
            break
        end
        achievable, required = _crop_certificate_value(prob, kind, r, t)
        prob = CropPlanningProblem(
            crops, R, T, K, field_area, field_farm, field_region, field_soil, field_irrigable, farm_region,
            area_vars, yields, crop_cost, price, tier_width, tier_factor, initial_nitrogen, fert_price, fert_quota,
            family_labor, hire_cap, hire_budget, hire_cost, water_allocation, contract, nothing,
            CropInfeasibilityCertificate(kind, r, t, achievable, required), feasibility_status,
        )
        @assert crop_certificate_holds(prob)
    elseif feasibility_status == feasible
        @assert crop_plan_satisfies(prob)
    end
    return prob
end

"""
    build_model(prob::CropPlanningProblem)

Build the regional crop-rotation planning LP (deterministic; see the type).
"""
function build_model(prob::CropPlanningProblem)
    model = Model()
    J, T, R, C, K = length(prob.field_area), prob.n_years, prob.n_regions, length(prob.crops), prob.n_tiers
    Fm = length(prob.farm_region)
    S = length(CROP_SEASONS)
    V = length(prob.area_vars)
    @variable(model, a[1:V] >= 0)
    @variable(model, fert[1:J, 1:T] >= 0)
    @variable(model, 0 <= hire[f=1:Fm, m=1:S, t=1:T] <= prob.hire_cap[f, m, t])
    @variable(model, 0 <= sell[c=1:C, r=1:R, t=1:T, k=1:K] <= prob.tier_width[c, r, k])
    @objective(
        model,
        Max,
        sum(prob.tier_factor[k] * prob.price[c, t] * sell[c, r, t, k] for c in 1:C, r in 1:R, t in 1:T, k in 1:K) -
        sum(prob.crop_cost[c] * a[v] for (v, (_, c, _)) in enumerate(prob.area_vars)) -
        prob.fert_price * sum(fert) - prob.hire_cost * sum(hire)
    )
    by_field_year = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (v, (j, _, t)) in enumerate(prob.area_vars)
        push!(get!(by_field_year, (j, t), Int[]), v)
    end
    spec(v) = _crop_spec(prob, prob.area_vars[v][2])
    for j in 1:J, t in 1:T
        vs = by_field_year[(j, t)]
        @constraint(model, sum(a[v] for v in vs) <= prob.field_area[j])
        if t < T
            next = by_field_year[(j, t + 1)]
            for fam in CROP_ROTATION_FAMILIES
                now_f = [v for v in vs if spec(v).family == fam]
                next_f = [v for v in next if spec(v).family == fam]
                isempty(now_f) && continue
                @constraint(model, sum(a[v] for v in now_f) + sum(a[v] for v in next_f) <= prob.field_area[j])
            end
        end
        previous = t == 1 ? Int[] : [v for v in by_field_year[(j, t - 1)] if spec(v).n_credit > 0]
        @constraint(
            model,
            fert[j, t] + sum(spec(v).n_credit * a[v] for v in previous; init=0.0) -
            sum(spec(v).n_need * a[v] for v in vs if spec(v).n_need > 0; init=0.0) >=
            (t == 1 ? -prob.initial_nitrogen[j] : 0.0)
        )
    end
    fields_of = [findall(==(f), prob.field_farm) for f in 1:Fm]
    for f in 1:Fm, t in 1:T
        @constraint(model, sum(fert[j, t] for j in fields_of[f]) <= prob.fert_quota[f, t])
        @constraint(model, sum(hire[f, m, t] for m in 1:S) <= prob.hire_budget[f, t])
        vs = [v for j in fields_of[f] for v in by_field_year[(j, t)]]
        for m in 1:S
            users = [v for v in vs if spec(v).labor[m] > 0]
            isempty(users) && continue
            @constraint(model, sum(spec(v).labor[m] * a[v] for v in users) - hire[f, m, t] <= prob.family_labor[f, m, t])
        end
    end
    for r in 1:R, t in 1:T
        vs = [v for j in 1:J if prob.field_region[j] == r && prob.field_irrigable[j] for v in by_field_year[(j, t)]]
        isempty(vs) && continue
        for (m, share) in _CROP_WATER_SEASONS
            @constraint(model, sum(spec(v).water * share * a[v] for v in vs) <= prob.water_allocation[r, m, t])
        end
    end
    producers = Dict{Tuple{Int, Int, Int}, Vector{Int}}()
    for (v, (j, c, t)) in enumerate(prob.area_vars)
        push!(get!(producers, (c, prob.field_region[j], t), Int[]), v)
    end
    for c in 1:C, r in 1:R, t in 1:T
        vs = get(producers, (c, r, t), Int[])
        produced = @expression(model, sum(prob.yields[(prob.area_vars[v][1], c)] * a[v] for v in vs; init=0.0))
        @constraint(model, sum(sell[c, r, t, k] for k in 1:K) - produced <= 0)
        prob.contract[c, r, t] > 0 && @constraint(model, produced >= prob.contract[c, r, t])
    end
    return model
end

register_variant(
    :crop_planning,
    :standard,
    CropPlanningProblem,
    "Regional multi-year crop-rotation planning: fields across farms and irrigation districts with " *
    "rotation breaks, nitrogen carry-over, seasonal labour, water allocations, tiered markets and contracts";
    tags=[:agriculture, :packing],
)
