using Random
using Distributions
using StatsBase

# Shared data for the blending variants: secondary-aluminium alloy production,
# the classic large-scale blending LP (casthouses charge furnaces with scrap,
# primary metal and master alloys so every melt lands inside its alloy's
# chemical-composition window). Compositions are weight percent and follow the
# Aluminum Association registrations and typical ISRI scrap-class chemistries.

"""Alloying and tramp elements tracked in every composition (wt%)."""
const BLEND_ELEMENTS = (:Si, :Fe, :Cu, :Mn, :Mg, :Zn, :Cr, :Ti)

"""Alloy families; scrap compatibility and hardener needs are by family."""
const BLEND_FAMILIES = (:f1xxx, :f2xxx, :f3xxx, :f5xxx, :f6xxx, :f7xxx, :cast)

# Alloy grades: family, composition window (min, max per element, wt%),
# selling price (USD/t) and relative order frequency.
const _BLEND_GRADES = (
    (name=:AA1050, family=:f1xxx, lo=(0, 0, 0, 0, 0, 0, 0, 0),
        hi=(0.25, 0.40, 0.05, 0.05, 0.05, 0.07, 0.03, 0.05), price=2650.0, weight=0.6),
    (name=:AA3003, family=:f3xxx, lo=(0, 0, 0.05, 1.0, 0, 0, 0, 0),
        hi=(0.6, 0.7, 0.20, 1.5, 0.05, 0.10, 0.05, 0.10), price=2800.0, weight=1.0),
    (name=:AA3104, family=:f3xxx, lo=(0, 0, 0.05, 0.8, 0.8, 0, 0, 0),
        hi=(0.6, 0.8, 0.25, 1.4, 1.3, 0.25, 0.05, 0.10), price=2900.0, weight=1.2),
    (name=:AA5052, family=:f5xxx, lo=(0, 0, 0, 0, 2.2, 0, 0.15, 0),
        hi=(0.25, 0.40, 0.10, 0.10, 2.8, 0.10, 0.35, 0.10), price=3000.0, weight=0.8),
    (name=:AA5182, family=:f5xxx, lo=(0, 0, 0, 0.20, 4.0, 0, 0, 0),
        hi=(0.20, 0.35, 0.15, 0.50, 5.0, 0.25, 0.10, 0.10), price=3100.0, weight=0.8),
    (name=:AA6061, family=:f6xxx, lo=(0.40, 0, 0.15, 0, 0.8, 0, 0.04, 0),
        hi=(0.8, 0.7, 0.40, 0.15, 1.2, 0.25, 0.35, 0.15), price=3200.0, weight=1.0),
    (name=:AA6063, family=:f6xxx, lo=(0.20, 0, 0, 0, 0.45, 0, 0, 0),
        hi=(0.6, 0.35, 0.10, 0.10, 0.9, 0.10, 0.10, 0.10), price=2950.0, weight=1.4),
    (name=:AA6082, family=:f6xxx, lo=(0.7, 0, 0, 0.40, 0.6, 0, 0, 0),
        hi=(1.3, 0.5, 0.10, 1.0, 1.2, 0.20, 0.25, 0.10), price=3100.0, weight=0.8),
    (name=:AA2024, family=:f2xxx, lo=(0, 0, 3.8, 0.3, 1.2, 0, 0, 0),
        hi=(0.5, 0.5, 4.9, 0.9, 1.8, 0.25, 0.10, 0.15), price=4200.0, weight=0.3),
    (name=:AA7075, family=:f7xxx, lo=(0, 0, 1.2, 0, 2.1, 5.1, 0.18, 0),
        hi=(0.4, 0.5, 2.0, 0.3, 2.9, 6.1, 0.28, 0.20), price=4600.0, weight=0.3),
    (name=:A356, family=:cast, lo=(6.5, 0, 0, 0, 0.25, 0, 0, 0),
        hi=(7.5, 0.20, 0.20, 0.10, 0.45, 0.10, 0.05, 0.20), price=2900.0, weight=0.7),
    (name=:A380, family=:cast, lo=(7.5, 0, 3.0, 0, 0, 0, 0, 0),
        hi=(9.5, 1.3, 4.0, 0.5, 0.10, 3.0, 0.10, 0.20), price=2500.0, weight=0.9),
)

# Primary metal: composition, cost (USD/t), melt yield.
const _BLEND_PRIMARIES = (
    (name=:P1020, comp=(0.05, 0.10, 0.002, 0.002, 0.002, 0.01, 0.001, 0.004), cost=2600.0, yield=0.995),
    (name=:P0406, comp=(0.03, 0.04, 0.001, 0.001, 0.001, 0.005, 0.001, 0.002), cost=2850.0, yield=0.995),
)

# Master alloys / hardeners: the element they carry (index into BLEND_ELEMENTS),
# composition, cost and melt yield (magnesium burns off).
const _BLEND_HARDENERS = (
    (name=:AlSi50, element=1, comp=(50.0, 0.3, 0.02, 0.01, 0.0, 0.01, 0.0, 0.02), cost=2600.0, yield=0.98),
    (name=:Al50Cu, element=3, comp=(0.1, 0.15, 50.0, 0.01, 0.01, 0.02, 0.0, 0.0), cost=5500.0, yield=0.99),
    (name=:AlMn20, element=4, comp=(0.15, 0.25, 0.02, 20.0, 0.0, 0.0, 0.0, 0.0), cost=2900.0, yield=0.98),
    (name=:magnesium_ingot, element=5, comp=(0.01, 0.01, 0.005, 0.01, 99.8, 0.005, 0.0, 0.0), cost=3600.0, yield=0.93),
    (name=:zinc_ingot, element=6, comp=(0.0, 0.002, 0.002, 0.0, 0.0, 99.99, 0.0, 0.0), cost=3100.0, yield=0.98),
    (name=:AlCr10, element=7, comp=(0.15, 0.25, 0.02, 0.01, 0.0, 0.0, 10.0, 0.0), cost=4500.0, yield=0.98),
    (name=:Al10Ti, element=8, comp=(0.1, 0.2, 0.01, 0.0, 0.0, 0.0, 0.0, 10.0), cost=4200.0, yield=0.99),
)

# Scrap classes: mean composition, cost, melt yield and the alloy families a
# casthouse lets them into (segregation rules).
const _BLEND_SCRAP_CLASSES = (
    (name=:foil_1xxx, comp=(0.08, 0.35, 0.01, 0.01, 0.01, 0.02, 0.005, 0.01), cost=2350.0, yield=0.95,
        families=(:f1xxx, :f3xxx, :f6xxx, :cast)),
    (name=:clean_sheet_clips, comp=(0.15, 0.45, 0.08, 0.70, 0.20, 0.05, 0.02, 0.02), cost=2300.0, yield=0.97,
        families=(:f3xxx, :f5xxx, :cast)),
    (name=:used_beverage_cans, comp=(0.25, 0.50, 0.15, 0.90, 1.60, 0.10, 0.03, 0.02), cost=1800.0, yield=0.88,
        families=(:f3xxx, :f5xxx, :cast)),
    (name=:taint_tabor, comp=(0.40, 0.55, 0.15, 0.50, 0.80, 0.20, 0.05, 0.03), cost=1900.0, yield=0.93,
        families=(:f3xxx, :f6xxx, :cast)),
    (name=:extrusion_6063, comp=(0.45, 0.25, 0.03, 0.04, 0.60, 0.03, 0.02, 0.02), cost=2200.0, yield=0.96,
        families=(:f6xxx, :f3xxx, :cast)),
    (name=:clips_5xxx, comp=(0.15, 0.30, 0.07, 0.35, 3.50, 0.10, 0.10, 0.02), cost=2250.0, yield=0.96,
        families=(:f5xxx, :f3xxx)),
    (name=:turnings_2xxx, comp=(0.20, 0.30, 4.30, 0.60, 1.50, 0.10, 0.03, 0.03), cost=1900.0, yield=0.85,
        families=(:f2xxx, :cast)),
    (name=:turnings_7xxx, comp=(0.10, 0.20, 1.60, 0.05, 2.50, 5.60, 0.20, 0.03), cost=2000.0, yield=0.85,
        families=(:f7xxx, :cast)),
    (name=:painted_siding, comp=(0.30, 0.55, 0.20, 0.50, 0.50, 0.20, 0.05, 0.03), cost=1750.0, yield=0.90,
        families=(:f3xxx, :cast)),
    (name=:twitch, comp=(1.00, 0.80, 1.00, 0.30, 0.60, 0.80, 0.05, 0.05), cost=1600.0, yield=0.90,
        families=(:cast,)),
    (name=:zorba, comp=(2.50, 1.00, 2.00, 0.30, 0.50, 1.50, 0.05, 0.05), cost=1400.0, yield=0.85,
        families=(:cast,)),
    (name=:mixed_cast, comp=(7.50, 0.80, 2.00, 0.30, 0.30, 1.00, 0.05, 0.10), cost=1500.0, yield=0.88,
        families=(:cast,)),
    (name=:wheels_a356, comp=(7.00, 0.15, 0.05, 0.03, 0.35, 0.05, 0.01, 0.12), cost=2100.0, yield=0.92,
        families=(:cast,)),
)

"""
    BlendMaterials

A sampled material list: `kind[i]` is `:primary`, `:hardener` or `:scrap`;
`source[i]` indexes the matching catalog (`_BLEND_PRIMARIES`,
`_BLEND_HARDENERS` or `_BLEND_SCRAP_CLASSES`); `comp[e, i]` is wt% of element
`e`; `cost` USD/t; `yield` the fraction of charged metal recovered; `site[i]`
the plant whose yard holds the material (`0` = available to every plant).
"""
struct BlendMaterials
    kind::Vector{Symbol}
    source::Vector{Int}
    comp::Matrix{Float64}
    cost::Vector{Float64}
    yield::Vector{Float64}
    site::Vector{Int}
end

_blend_family(grade::Int) = _BLEND_GRADES[grade].family

"""Is material `i` allowed in an alloy of `grade` (scrap segregation, hardener need)?"""
function blend_compatible(mats::BlendMaterials, i::Int, grade::Int)
    kind = mats.kind[i]
    kind == :primary && return true
    if kind == :hardener
        return _BLEND_GRADES[grade].lo[_BLEND_HARDENERS[mats.source[i]].element] > 0
    end
    return _blend_family(grade) in _BLEND_SCRAP_CLASSES[mats.source[i]].families
end

function _blend_sample_grade(rng::AbstractRNG)
    w = [g.weight for g in _BLEND_GRADES]
    return rand(rng, Categorical(w ./ sum(w)))
end

"""
    _blend_lot(rng, class) -> (comp, cost)

One scrap lot of a class: the class chemistry with lot-to-lot scatter, plus a
lot-wide contamination factor on the tramp elements (Fe, Cu, Zn) that also
discounts the price, so dirtier lots are cheaper — the trade-off that makes
scrap blending a real LP.
"""
function _blend_lot(rng::AbstractRNG, class::Int)
    spec = _BLEND_SCRAP_CLASSES[class]
    contamination = rand(rng, LogNormal(0.0, 0.2))
    comp = zeros(Float64, length(BLEND_ELEMENTS))
    for e in eachindex(BLEND_ELEMENTS)
        tramp = e in (2, 3, 6) ? contamination : 1.0
        comp[e] = spec.comp[e] * tramp * rand(rng, LogNormal(0.0, 0.18))
    end
    cost = spec.cost * rand(rng, LogNormal(0.0, 0.05)) * (1.0 - 0.08 * (contamination - 1.0))
    return _blend_assay.(comp), cost
end

"""
    _blend_assay(value)

A reported spark-spectrometer assay: three significant digits, and values
below the 0.005 wt% detection limit reported (and planned) as zero.
"""
_blend_assay(value::Real) = value < 0.005 ? 0.0 : round(Float64(value); sigdigits=3)

"""Primary and hardener materials (one of each catalog entry), shared by all plants."""
function _blend_base_materials()
    kind = Symbol[]
    source = Int[]
    comp = Vector{Float64}[]
    cost = Float64[]
    yield = Float64[]
    for (p, spec) in enumerate(_BLEND_PRIMARIES)
        push!(kind, :primary); push!(source, p); push!(comp, _blend_assay.(collect(spec.comp)))
        push!(cost, spec.cost); push!(yield, spec.yield)
    end
    for (h, spec) in enumerate(_BLEND_HARDENERS)
        push!(kind, :hardener); push!(source, h); push!(comp, _blend_assay.(collect(spec.comp)))
        push!(cost, spec.cost); push!(yield, spec.yield)
    end
    return kind, source, comp, cost, yield
end

"""
    _blend_recipe(rng, grade, candidates, comp, yields) -> (fractions, composition)

A plausible charge recipe for one melt of `grade` over the `candidates`
material indices: 1–3 compatible scrap lots (Dirichlet weights) at a scrap share
capped so no tramp element exceeds 85% of its limit, the rest primary metal,
then master alloys added (four fixed-point passes) to bring every element with
a minimum to a target inside its window. Returns mass fractions aligned with
`candidates` and the resulting composition. The recipe is not cost-optimal; it
is what a melt-shop planner would write by hand, which makes it a good witness.
"""
function _blend_recipe(
    rng::AbstractRNG,
    grade::Int,
    candidates::Vector{Int},
    kind::Vector{Symbol},
    source::Vector{Int},
    comp::AbstractMatrix{Float64},
)
    spec = _BLEND_GRADES[grade]
    n_el = length(BLEND_ELEMENTS)
    lo = collect(Float64, spec.lo)
    hi = collect(Float64, spec.hi)
    target = [lo[e] > 0 ? lo[e] + rand(rng, Uniform(0.35, 0.65)) * (hi[e] - lo[e]) : 0.0 for e in 1:n_el]
    mass = zeros(Float64, length(candidates))
    scrap = [k for k in eachindex(candidates) if kind[candidates[k]] == :scrap]
    primary = [k for k in eachindex(candidates) if kind[candidates[k]] == :primary]
    sigma = 0.0
    if !isempty(scrap)
        chosen = length(scrap) <= 3 ? copy(scrap) : sample(rng, scrap, rand(rng, 1:3); replace=false)
        weights = length(chosen) == 1 ? [1.0] : rand(rng, Dirichlet(fill(2.0, length(chosen))))
        mix = sum(weights[j] .* view(comp, :, candidates[chosen[j]]) for j in eachindex(chosen))
        base = view(comp, :, candidates[first(primary)])
        sigma = rand(rng, Uniform(0.15, 0.75))
        for e in 1:n_el
            if mix[e] > base[e]
                sigma = min(sigma, max(0.0, (0.85 * hi[e] - base[e]) / (mix[e] - base[e])))
            end
        end
        for (j, k) in enumerate(chosen)
            mass[k] += sigma * weights[j]
        end
    end
    pw = length(primary) == 1 ? [1.0] : rand(rng, Dirichlet(fill(1.5, length(primary))))
    for (j, k) in enumerate(primary)
        mass[k] += (1.0 - sigma) * pw[j]
    end
    for _ in 1:4
        for e in 1:n_el
            lo[e] > 0 || continue
            h = findfirst(k -> kind[candidates[k]] == :hardener && _BLEND_HARDENERS[source[candidates[k]]].element == e, eachindex(candidates))
            h === nothing && continue
            total = sum(mass)
            current = sum(mass[k] * comp[e, candidates[k]] for k in eachindex(candidates)) / total
            current >= target[e] && continue
            content = comp[e, candidates[h]]
            mass[h] += (target[e] - current) * total / (content - target[e])
        end
    end
    fractions = mass ./ sum(mass)
    composition = [sum(fractions[k] * comp[e, candidates[k]] for k in eachindex(candidates)) for e in 1:n_el]
    return fractions, composition
end

"""
    _blend_window(grade, composition) -> (lo, hi)

The grade's registered window, widened (by 2%) only where a planted recipe's
composition falls outside it.
"""
function _blend_window(grade::Int, composition::AbstractVector{<:Real})
    spec = _BLEND_GRADES[grade]
    lo = [min(Float64(spec.lo[e]), spec.lo[e] > 0 ? 0.98 * composition[e] : 0.0) for e in eachindex(BLEND_ELEMENTS)]
    hi = [max(Float64(spec.hi[e]), 1.02 * composition[e]) for e in eachindex(BLEND_ELEMENTS)]
    return lo, hi
end

"""
    _blend_max_excess(a, yields, upper, demand)

Exact optimum of `max Σ a_i x_i  s.t.  Σ yields_i x_i ≥ demand, 0 ≤ x ≤ upper`
(continuous covering knapsack): take every item with `a_i > 0` in full, then
cover any remaining demand with the items losing least per tonne of output.
`-Inf` if even all material cannot cover `demand`.
"""
function _blend_max_excess(a::AbstractVector, yields::AbstractVector, upper::AbstractVector, demand::Real)
    value = 0.0
    produced = 0.0
    for i in eachindex(a)
        if a[i] > 0
            value += a[i] * upper[i]
            produced += yields[i] * upper[i]
        end
    end
    remaining = demand - produced
    remaining <= 0 && return value
    rest = [i for i in eachindex(a) if a[i] <= 0]
    sort!(rest; by=i -> -a[i] / yields[i])
    for i in rest
        remaining <= 0 && break
        amount = min(Float64(upper[i]), remaining / yields[i])
        value += a[i] * amount
        remaining -= yields[i] * amount
    end
    return remaining <= 1e-9 * max(1.0, demand) ? value : -Inf
end

"""Element rows that can bind: a minimum row only if some material is below it,
a maximum row only if some material is above it (others are implied by x ≥ 0)."""
function _blend_active_rows(comp::AbstractMatrix, materials::AbstractVector{Int}, lo, hi)
    mins = [e for e in eachindex(BLEND_ELEMENTS) if lo[e] > 0 && any(comp[e, i] < lo[e] for i in materials)]
    maxs = [e for e in eachindex(BLEND_ELEMENTS) if any(comp[e, i] > hi[e] for i in materials)]
    return mins, maxs
end
