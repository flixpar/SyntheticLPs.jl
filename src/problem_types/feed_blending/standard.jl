using JuMP
using Random
using Distributions
using StatsBase

"""Nutrients of the feed generator (as-fed basis), in content-matrix row order."""
const FEED_NUTRIENTS = (
    :metabolizable_energy,  # Mcal/kg
    :crude_protein,         # %
    :crude_fat,             # %
    :crude_fiber,           # %
    :calcium,               # %
    :available_phosphorus,  # %
    :sodium,                # %
    :digestible_lysine,     # %
    :digestible_methionine, # %
    :digestible_met_cys,    # %
    :digestible_threonine,  # %
    :ndf,                   # % neutral detergent fiber
)
const _FEED_CA = 5
const _FEED_AVP = 6

# Ingredient catalog: class, nutrients (FEED_NUTRIENTS order) and price USD/t.
# Values are rounded NRC/feed-table means.
const _FEED_INGREDIENTS = (
    (name=:corn, class=:grain, n=(3.35, 8.0, 3.7, 2.2, 0.02, 0.08, 0.02, 0.21, 0.16, 0.33, 0.25, 9.5), price=220.0),
    (name=:wheat, class=:grain, n=(3.10, 12.5, 1.8, 2.6, 0.05, 0.13, 0.02, 0.30, 0.19, 0.44, 0.33, 12.0), price=240.0),
    (name=:barley, class=:grain, n=(2.70, 11.0, 2.0, 5.0, 0.06, 0.12, 0.03, 0.33, 0.16, 0.36, 0.30, 19.0), price=210.0),
    (name=:sorghum, class=:grain, n=(3.25, 9.5, 3.0, 2.5, 0.03, 0.08, 0.01, 0.18, 0.14, 0.29, 0.26, 10.0), price=200.0),
    (name=:soybean_meal_48, class=:protein, n=(2.45, 47.5, 1.5, 3.5, 0.30, 0.20, 0.02, 2.65, 0.60, 1.25, 1.65, 9.0), price=430.0),
    (name=:soybean_meal_44, class=:protein, n=(2.25, 44.0, 1.5, 6.0, 0.30, 0.18, 0.02, 2.45, 0.56, 1.17, 1.55, 13.0), price=400.0),
    (name=:canola_meal, class=:protein, n=(2.00, 36.0, 3.5, 11.5, 0.65, 0.40, 0.05, 1.65, 0.62, 1.40, 1.25, 25.0), price=320.0),
    (name=:sunflower_meal, class=:protein, n=(1.90, 34.0, 1.5, 21.0, 0.40, 0.30, 0.10, 1.00, 0.70, 1.20, 1.05, 38.0), price=280.0),
    (name=:peas, class=:protein, n=(2.60, 22.0, 1.2, 6.0, 0.10, 0.20, 0.02, 1.40, 0.18, 0.45, 0.70, 12.0), price=300.0),
    (name=:cottonseed_meal, class=:protein, n=(2.00, 41.0, 1.5, 12.0, 0.20, 0.30, 0.05, 1.30, 0.50, 1.10, 1.05, 28.0), price=300.0),
    (name=:corn_gluten_meal, class=:protein, n=(3.70, 60.0, 2.5, 1.5, 0.05, 0.15, 0.05, 0.85, 1.35, 2.30, 1.80, 9.0), price=650.0),
    (name=:fish_meal, class=:animal, n=(2.95, 64.0, 9.0, 1.0, 4.00, 2.60, 0.80, 4.60, 1.70, 2.20, 2.40, 0.0), price=1500.0),
    (name=:meat_bone_meal, class=:animal, n=(2.30, 50.0, 10.0, 2.5, 10.0, 4.50, 0.75, 2.20, 0.60, 0.95, 1.40, 0.0), price=450.0),
    (name=:ddgs, class=:byproduct, n=(2.80, 27.0, 9.0, 7.5, 0.05, 0.40, 0.20, 0.55, 0.45, 0.85, 0.80, 33.0), price=230.0),
    (name=:wheat_middlings, class=:byproduct, n=(2.20, 16.0, 4.0, 8.0, 0.12, 0.35, 0.03, 0.55, 0.20, 0.45, 0.42, 36.0), price=170.0),
    (name=:wheat_bran, class=:byproduct, n=(1.60, 15.5, 4.0, 11.0, 0.13, 0.40, 0.04, 0.48, 0.18, 0.43, 0.40, 45.0), price=160.0),
    (name=:rice_bran, class=:byproduct, n=(2.60, 13.0, 14.0, 12.0, 0.08, 0.25, 0.03, 0.45, 0.20, 0.40, 0.35, 25.0), price=190.0),
    (name=:palm_kernel_meal, class=:byproduct, n=(1.80, 16.0, 8.0, 16.0, 0.25, 0.30, 0.03, 0.40, 0.25, 0.45, 0.45, 65.0), price=140.0),
    (name=:alfalfa_meal, class=:forage, n=(1.20, 17.0, 2.5, 25.0, 1.40, 0.25, 0.10, 0.60, 0.20, 0.40, 0.55, 45.0), price=250.0),
    (name=:molasses, class=:energy, n=(2.00, 4.0, 0.0, 0.0, 0.80, 0.02, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0), price=180.0),
    (name=:soybean_oil, class=:fat, n=(8.50, 0.0, 99.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), price=1100.0),
    (name=:tallow, class=:fat, n=(7.80, 0.0, 99.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), price=900.0),
    (name=:limestone, class=:mineral, n=(0.0, 0.0, 0.0, 0.0, 38.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0), price=60.0),
    (name=:dicalcium_phosphate, class=:mineral, n=(0.0, 0.0, 0.0, 0.0, 22.0, 18.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0), price=700.0),
    (name=:monocalcium_phosphate, class=:mineral, n=(0.0, 0.0, 0.0, 0.0, 16.0, 21.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0), price=800.0),
    (name=:salt, class=:mineral, n=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 39.0, 0.0, 0.0, 0.0, 0.0, 0.0), price=120.0),
    (name=:sodium_bicarbonate, class=:mineral, n=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 27.0, 0.0, 0.0, 0.0, 0.0, 0.0), price=400.0),
    (name=:lysine_hcl, class=:amino, n=(4.00, 95.0, 0.0, 0.0, 0.0, 0.0, 0.0, 78.0, 0.0, 0.0, 0.0, 0.0), price=1700.0),
    (name=:dl_methionine, class=:amino, n=(5.00, 58.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.0, 99.0, 0.0, 0.0), price=3500.0),
    (name=:l_threonine, class=:amino, n=(3.60, 72.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 98.0, 0.0), price=2200.0),
    (name=:urea, class=:npn, n=(0.0, 281.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), price=450.0),
    (name=:premix, class=:premix, n=(0.0, 0.0, 0.0, 0.0, 12.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0), price=2500.0),
)

# Species groups: maximum inclusion fraction per ingredient class (overridden
# per ingredient below); 0 excludes the ingredient.
const _FEED_GROUPS = (:poultry, :swine, :ruminant, :aqua)
const _FEED_CLASS_LIMIT = Dict(
    :poultry => Dict(:grain => 0.70, :protein => 0.40, :animal => 0.05, :byproduct => 0.10, :forage => 0.03,
        :energy => 0.02, :fat => 0.06, :mineral => 0.10, :amino => 0.006, :npn => 0.0, :premix => 0.008),
    :swine => Dict(:grain => 0.80, :protein => 0.35, :animal => 0.05, :byproduct => 0.25, :forage => 0.05,
        :energy => 0.05, :fat => 0.05, :mineral => 0.03, :amino => 0.006, :npn => 0.0, :premix => 0.008),
    :ruminant => Dict(:grain => 0.60, :protein => 0.30, :animal => 0.0, :byproduct => 0.35, :forage => 0.40,
        :energy => 0.08, :fat => 0.04, :mineral => 0.03, :amino => 0.0, :npn => 0.012, :premix => 0.015),
    :aqua => Dict(:grain => 0.40, :protein => 0.45, :animal => 0.25, :byproduct => 0.20, :forage => 0.02,
        :energy => 0.03, :fat => 0.10, :mineral => 0.04, :amino => 0.008, :npn => 0.0, :premix => 0.015),
)
# Ingredient-specific inclusion caps (anti-nutritional factors, palatability).
const _FEED_ITEM_LIMIT = Dict(
    (:poultry, :barley) => 0.15, (:poultry, :canola_meal) => 0.10, (:poultry, :sunflower_meal) => 0.08,
    (:poultry, :cottonseed_meal) => 0.05, (:poultry, :peas) => 0.15, (:poultry, :corn_gluten_meal) => 0.08,
    (:poultry, :limestone) => 0.10, (:poultry, :salt) => 0.006, (:poultry, :sodium_bicarbonate) => 0.005,
    (:poultry, :dicalcium_phosphate) => 0.03, (:poultry, :monocalcium_phosphate) => 0.03,
    (:swine, :cottonseed_meal) => 0.05, (:swine, :canola_meal) => 0.12, (:swine, :salt) => 0.008,
    (:swine, :sodium_bicarbonate) => 0.005, (:swine, :limestone) => 0.02,
    (:ruminant, :cottonseed_meal) => 0.20, (:ruminant, :salt) => 0.012, (:ruminant, :sodium_bicarbonate) => 0.012,
    (:ruminant, :limestone) => 0.025, (:aqua, :salt) => 0.005, (:aqua, :limestone) => 0.02,
)

# Formula types: species group and (min, max) spec per nutrient (Inf = none).
const _FEED_FORMULAS = (
    (name=:broiler_starter, group=:poultry,
        spec=((2.95, 3.10), (21.5, 24.0), (0.0, 7.0), (0.0, 4.0), (0.90, 1.05), (0.45, 0.55), (0.16, 0.23),
            (1.25, Inf), (0.50, Inf), (0.92, Inf), (0.83, Inf), (0.0, 15.0)), ca_p=(1.8, 2.3)),
    (name=:broiler_grower, group=:poultry,
        spec=((3.05, 3.20), (19.5, 22.0), (0.0, 8.0), (0.0, 4.5), (0.80, 0.95), (0.40, 0.50), (0.16, 0.23),
            (1.12, Inf), (0.46, Inf), (0.86, Inf), (0.75, Inf), (0.0, 16.0)), ca_p=(1.8, 2.3)),
    (name=:broiler_finisher, group=:poultry,
        spec=((3.10, 3.25), (18.0, 20.5), (0.0, 9.0), (0.0, 5.0), (0.72, 0.88), (0.36, 0.46), (0.15, 0.23),
            (1.00, Inf), (0.42, Inf), (0.78, Inf), (0.68, Inf), (0.0, 17.0)), ca_p=(1.8, 2.4)),
    (name=:layer, group=:poultry,
        spec=((2.70, 2.90), (16.0, 18.5), (0.0, 7.0), (0.0, 6.0), (3.6, 4.3), (0.38, 0.48), (0.15, 0.22),
            (0.78, Inf), (0.38, Inf), (0.68, Inf), (0.55, Inf), (0.0, 18.0)), ca_p=(8.0, 11.0)),
    (name=:swine_starter, group=:swine,
        spec=((3.10, 3.35), (20.0, 23.0), (0.0, 8.0), (0.0, 3.5), (0.70, 0.90), (0.40, 0.50), (0.20, 0.35),
            (1.35, Inf), (0.39, Inf), (0.74, Inf), (0.79, Inf), (0.0, 14.0)), ca_p=(1.5, 2.0)),
    (name=:swine_grower, group=:swine,
        spec=((3.05, 3.30), (16.0, 19.0), (0.0, 8.0), (0.0, 5.0), (0.60, 0.75), (0.26, 0.36), (0.10, 0.25),
            (0.98, Inf), (0.28, Inf), (0.56, Inf), (0.60, Inf), (0.0, 20.0)), ca_p=(1.6, 2.3)),
    (name=:swine_finisher, group=:swine,
        spec=((3.00, 3.30), (13.5, 16.5), (0.0, 8.0), (0.0, 6.0), (0.50, 0.70), (0.20, 0.30), (0.10, 0.25),
            (0.73, Inf), (0.21, Inf), (0.44, Inf), (0.48, Inf), (0.0, 22.0)), ca_p=(1.7, 2.6)),
    (name=:dairy_concentrate, group=:ruminant,
        spec=((2.50, 2.90), (18.0, 24.0), (0.0, 6.0), (6.0, 14.0), (0.80, 1.40), (0.40, 0.70), (0.25, 0.60),
            (0.0, Inf), (0.0, Inf), (0.0, Inf), (0.0, Inf), (20.0, 40.0)), ca_p=(1.4, 2.6)),
    (name=:beef_finisher, group=:ruminant,
        spec=((2.70, 3.10), (12.0, 15.0), (0.0, 6.0), (3.0, 10.0), (0.50, 0.90), (0.25, 0.50), (0.10, 0.40),
            (0.0, Inf), (0.0, Inf), (0.0, Inf), (0.0, Inf), (12.0, 30.0)), ca_p=(1.5, 2.8)),
    (name=:aqua_grower, group=:aqua,
        spec=((2.80, 3.30), (30.0, 36.0), (4.0, 10.0), (0.0, 6.0), (0.60, 1.50), (0.60, 1.00), (0.10, 0.50),
            (1.50, Inf), (0.60, Inf), (0.90, Inf), (1.10, Inf), (0.0, 20.0)), ca_p=(0.8, 2.2)),
)

# Reference recipe per species group: (ingredient class => fraction), mapped
# onto the stocked ingredients of each class.
const _FEED_TEMPLATE = Dict(
    :poultry => (:grain => 0.58, :protein => 0.31, :byproduct => 0.03, :fat => 0.035, :mineral => 0.035,
        :amino => 0.005, :premix => 0.005),
    :swine => (:grain => 0.68, :protein => 0.20, :byproduct => 0.08, :fat => 0.01, :mineral => 0.022,
        :amino => 0.003, :premix => 0.005),
    :ruminant => (:grain => 0.42, :byproduct => 0.25, :protein => 0.15, :forage => 0.10, :energy => 0.04,
        :mineral => 0.025, :npn => 0.005, :premix => 0.01),
    :aqua => (:protein => 0.38, :animal => 0.12, :grain => 0.25, :byproduct => 0.12, :fat => 0.07,
        :mineral => 0.03, :amino => 0.004, :premix => 0.01),
)

"""Ingredients every mill stocks (they alone can fill any species' batch)."""
const _FEED_STAPLES = (:corn, :soybean_meal_48, :limestone, :dicalcium_phosphate, :salt, :premix, :soybean_oil)

"""
Reason a requested-infeasible feed instance has no feasible formulation.

  - `feed_nutrient_unreachable`: one formula's minimum for one nutrient exceeds
    the most any recipe filling its batch can contain within the inclusion
    limits, mill stock and supplier contracts (an exact fractional knapsack over
    the batch equality).
  - `feed_mill_short`: one mill's batches need more tonnage than all the
    ingredients it may use can supply.
"""
@enum FeedInfeasibilityKind begin
    feed_nutrient_unreachable
    feed_mill_short
end

"""
    FeedInfeasibilityCertificate

LP-row proof for `FeedBlendingProblem`. `formula` and `nutrient` locate a
`feed_nutrient_unreachable` proof, `mill` a `feed_mill_short` one (unused
fields 0). `achievable < required` by at least 5%.
"""
struct FeedInfeasibilityCertificate
    kind::FeedInfeasibilityKind
    formula::Int
    nutrient::Int
    mill::Int
    achievable::Float64
    required::Float64
end

"""
    FeedBlendingProblem <: ProblemGenerator

Least-cost feed formulation for a network of feed mills: every mill produces a
book of formulas (species and growth phase: broiler, layer, swine, dairy, beef,
aquaculture) in fixed batch tonnages from the ingredients it stocks, and mills
draw on shared supplier contracts.

# Formulation

Variables `x[k] ∈ [0, inclusion_cap[k]]` tonnes of ingredient lot `i` in formula
`f` for every allowed pair `k = (i, f)` (species exclusions — no animal protein
for ruminants, urea only for ruminants — and the mill's stock list decide which
pairs exist; anti-nutritional and palatability limits are the inclusion caps,
and premix has a minimum inclusion as a lower bound). Minimize ingredient cost.
Rows per formula, with batch tonnage `D_f`:

  - batch `Σ_i x[i,f] = D_f`;
  - nutrient specifications in absolute form, `Σ_i a[j,i] x[i,f] ≥ lo[j,f] D_f`
    and `≤ hi[j,f] D_f` (sparse: only ingredients carrying `j`; a maximum is
    written only if some ingredient exceeds it);
  - the calcium : available-phosphorus ratio band, homogeneous with mixed
    signs: `Σ_i (Ca_i − r_hi P_i) x ≤ 0`, `Σ_i (Ca_i − r_lo P_i) x ≥ 0`.

Coupling rows: mill stock `Σ_{f at m} x[i,f] ≤ stock[i]` for every lot at the
mill, and supplier contracts `Σ_{lots of s} … ≤ contract[s]` shared by the
mills buying from supplier `s`. Ingredient lots are catalog ingredients from a
supplier, with lot-to-lot quality scatter.

# Sizing

Mills `≈ target / 2500` (1–40). Formulas are added round-robin over the mills
until the number of allowed pairs reaches the target; the last formula drops
optional (non-staple) lots so the count lands within a few pairs of it.

# Feasibility

  - `feasible`: each formula gets its species' reference recipe mapped onto
    the stocked ingredients and clipped to the inclusion caps; specifications
    are the reference ones, relaxed only where that recipe falls outside; stock
    and contracts are 1.02–1.30× its use. Stored as `feasible_witness`.
  - `infeasible`: one mutation with a `FeedInfeasibilityCertificate` — a
    customer specification above a formula's reachable maximum (default) or a
    mill's stock below its batch book. Neither is visible to a single row.
  - `unknown`: reference specifications with noise, stock and contracts drawn
    around the reference recipes' use.
"""
struct FeedBlendingProblem <: ProblemGenerator
    n_mills::Int
    lot_ingredient::Vector{Int}
    lot_supplier::Vector{Int}
    lot_mill::Vector{Int}
    content::Matrix{Float64}
    cost::Vector{Float64}
    formula_type::Vector{Int}
    formula_mill::Vector{Int}
    batch::Vector{Float64}
    pairs::Vector{Tuple{Int, Int}}
    formula_pairs::Vector{UnitRange{Int}}
    lower::Vector{Float64}
    upper::Vector{Float64}
    spec_lo::Matrix{Float64}
    spec_hi::Matrix{Float64}
    ratio_band::Matrix{Float64}
    stock::Vector{Float64}
    contract::Vector{Float64}
    feasible_witness::Union{Nothing, Vector{Float64}}
    infeasibility_certificate::Union{Nothing, FeedInfeasibilityCertificate}
    feasibility_status::FeasibilityStatus
end

function _feed_inclusion_cap(group::Symbol, ingredient::Int)
    spec = _FEED_INGREDIENTS[ingredient]
    cap = _FEED_CLASS_LIMIT[group][spec.class]
    return min(cap, get(_FEED_ITEM_LIMIT, (group, spec.name), cap))
end

"""
    _feed_sample_lots(rng, n_mills, n_suppliers)

Ingredient lots: each catalog ingredient is offered by 1–3 suppliers with
their own quality scatter and price; each mill stocks most ingredients from one
of those suppliers. Returns per-lot ingredient, supplier, mill, content, cost.
"""
function _feed_sample_lots(rng::AbstractRNG, n_mills::Int)
    n_items = length(_FEED_INGREDIENTS)
    offers = [rand(rng, 1:3) for _ in 1:n_items]
    supplier_quality = [
        [rand(rng, LogNormal(0.0, 0.04), length(FEED_NUTRIENTS)) for _ in 1:offers[c]] for c in 1:n_items
    ]
    supplier_price = [[rand(rng, LogNormal(0.0, 0.06)) for _ in 1:offers[c]] for c in 1:n_items]
    supplier_offset = cumsum(vcat(0, offers[1:(end - 1)]))
    ingredient = Int[]
    supplier = Int[]
    mill = Int[]
    content = Vector{Float64}[]
    cost = Float64[]
    for m in 1:n_mills, c in 1:n_items
        # Core ingredients are always stocked; the rest with probability 0.75.
        core = _FEED_INGREDIENTS[c].name in _FEED_STAPLES
        core || rand(rng) < 0.75 || continue
        s = rand(rng, 1:offers[c])
        push!(ingredient, c)
        push!(supplier, supplier_offset[c] + s)
        push!(mill, m)
        base = collect(_FEED_INGREDIENTS[c].n)
        push!(content, round.(base .* supplier_quality[c][s] .* rand(rng, LogNormal(0.0, 0.02), length(base)); sigdigits=3))
        push!(cost, _FEED_INGREDIENTS[c].price * supplier_price[c][s] * rand(rng, LogNormal(0.0, 0.03)))
    end
    return ingredient, supplier, mill, reduce(hcat, content), cost, sum(offers)
end

"""
    _feed_reference_recipe(rng, group, lots, ingredient, caps)

The species' reference recipe mapped onto the allowed lots of one formula:
each template class share is split over 1–2 random lots of that class, clipped
to the inclusion caps, and the clipped mass is redistributed over lots with
remaining headroom (grains first). Returns fractions summing to 1.
"""
function _feed_reference_recipe(rng::AbstractRNG, group::Symbol, lots::Vector{Int}, ingredient, caps::Vector{Float64})
    frac = zeros(Float64, length(lots))
    by_class = Dict{Symbol, Vector{Int}}()
    for (j, l) in enumerate(lots)
        push!(get!(by_class, _FEED_INGREDIENTS[ingredient[l]].class, Int[]), j)
    end
    for (class, share) in _FEED_TEMPLATE[group]
        members = get(by_class, class, Int[])
        isempty(members) && continue
        if class == :mineral
            # Limestone, phosphate and salt each get a part of the mineral share.
            w = rand(rng, Dirichlet(fill(2.0, length(members))))
            frac[members] .+= share .* w
            continue
        end
        k = min(length(members), rand(rng, 1:2))
        chosen = sample(rng, members, k; replace=false)
        w = k == 1 ? [1.0] : rand(rng, Dirichlet([2.0, 2.0]))
        frac[chosen] .+= share .* w .* rand(rng, Uniform(0.85, 1.15))
    end
    for _ in 1:50
        frac .= min.(frac, 0.98 .* caps)
        missing_mass = 1.0 - sum(frac)
        abs(missing_mass) < 1e-12 && break
        if missing_mass > 0
            room = [max(0.0, 0.98 * caps[j] - frac[j]) for j in eachindex(frac)]
            grains = [j for j in eachindex(frac) if _FEED_INGREDIENTS[ingredient[lots[j]]].class in (:grain, :byproduct, :protein)]
            pool = sum(room[grains]; init=0.0) > 0 ? grains : collect(eachindex(frac))
            total_room = sum(room[pool])
            total_room <= 0 && break
            frac[pool] .+= room[pool] .* min(1.0, missing_mass / total_room)
        else
            frac .*= 1.0 / sum(frac)
        end
    end
    abs(sum(frac) - 1.0) <= 1e-9 || throw(ArgumentError("inclusion caps cannot fill a $(group) batch"))
    return frac ./ sum(frac)
end

function _feed_profile(content::Matrix{Float64}, lots::AbstractVector{Int}, frac::AbstractVector{<:Real})
    return [sum(content[n, lots[j]] * frac[j] for j in eachindex(lots)) for n in eachindex(FEED_NUTRIENTS)]
end

"""
    _feed_knapsack_max(values, caps, batch)

Exact `max Σ v_i x_i s.t. Σ x_i = batch, 0 ≤ x_i ≤ caps_i` (fill the richest
first); `-Inf` if the caps cannot fill the batch.
"""
function _feed_knapsack_max(values::AbstractVector, caps::AbstractVector, batch::Real)
    remaining = Float64(batch)
    total = 0.0
    for j in sortperm(values; rev=true)
        remaining <= 0 && break
        amount = min(Float64(caps[j]), remaining)
        total += values[j] * amount
        remaining -= amount
    end
    return remaining <= 1e-9 * max(1.0, batch) ? total : -Inf
end

# Effective per-pair upper bound for certificate arithmetic: inclusion cap,
# the lot's mill stock and its supplier contract.
function _feed_pair_caps(prob_upper, stock, contract, lot_supplier, pairs, ks)
    return [min(prob_upper[k], stock[pairs[k][1]], contract[lot_supplier[pairs[k][1]]]) for k in ks]
end

"""
    feed_formulation_satisfies(prob, x=prob.feasible_witness; atol=1e-7)

Check a formulation (tonnes per pair) against every bound and row.
"""
function feed_formulation_satisfies(
    prob::FeedBlendingProblem, x::Union{Nothing, AbstractVector{<:Real}}=prob.feasible_witness; atol::Float64=1e-7
)
    x === nothing && return false
    length(x) == length(prob.pairs) || return false
    tol(v) = atol * max(1.0, abs(v))
    for k in eachindex(x)
        prob.lower[k] - tol(prob.lower[k]) <= x[k] <= prob.upper[k] + tol(prob.upper[k]) || return false
    end
    for (f, ks) in enumerate(prob.formula_pairs)
        D = prob.batch[f]
        abs(sum(x[k] for k in ks) - D) <= tol(D) || return false
        lots = [prob.pairs[k][1] for k in ks]
        amounts = [x[k] for k in ks]
        for n in eachindex(FEED_NUTRIENTS)
            total = sum(prob.content[n, lots[j]] * amounts[j] for j in eachindex(lots))
            total + tol(D) >= prob.spec_lo[n, f] * D || return false
            isfinite(prob.spec_hi[n, f]) && (total <= prob.spec_hi[n, f] * D + tol(D) || return false)
        end
        ca = sum(prob.content[_FEED_CA, lots[j]] * amounts[j] for j in eachindex(lots))
        p = sum(prob.content[_FEED_AVP, lots[j]] * amounts[j] for j in eachindex(lots))
        ca + tol(D) >= prob.ratio_band[1, f] * p || return false
        ca <= prob.ratio_band[2, f] * p + tol(D) || return false
    end
    used = zeros(Float64, length(prob.stock))
    for (k, (l, _)) in enumerate(prob.pairs)
        used[l] += x[k]
    end
    all(l -> used[l] <= prob.stock[l] + tol(used[l]), eachindex(used)) || return false
    by_supplier = zeros(Float64, length(prob.contract))
    for l in eachindex(used)
        by_supplier[prob.lot_supplier[l]] += used[l]
    end
    all(s -> by_supplier[s] <= prob.contract[s] + tol(by_supplier[s]), eachindex(by_supplier)) || return false
    return true
end

# Per-lot tonnage a mill's formulas can draw: the smaller of its stock, its
# supplier contract and the summed inclusion caps of the formulas using it.
function _feed_mill_lot_caps(stock, contract, lot_supplier, upper, pairs, formula_pairs, formulas)
    caps = Dict{Int, Float64}()
    for f in formulas, k in formula_pairs[f]
        caps[pairs[k][1]] = get(caps, pairs[k][1], 0.0) + upper[k]
    end
    lots = sort!(collect(keys(caps)))
    return lots, [min(stock[l], contract[lot_supplier[l]], caps[l]) for l in lots]
end

function _feed_mill_capacity(stock, contract, lot_supplier, upper, pairs, formula_pairs, formulas)
    _, effective = _feed_mill_lot_caps(stock, contract, lot_supplier, upper, pairs, formula_pairs, formulas)
    return sum(effective)
end

"""
    feed_certificate_holds(prob::FeedBlendingProblem)

Recompute the stored certificate from the data and check `achievable < required`.
"""
function feed_certificate_holds(prob::FeedBlendingProblem)
    cert = prob.infeasibility_certificate
    cert === nothing && return false
    if cert.kind == feed_nutrient_unreachable
        1 <= cert.formula <= length(prob.batch) || return false
        ks = prob.formula_pairs[cert.formula]
        caps = _feed_pair_caps(prob.upper, prob.stock, prob.contract, prob.lot_supplier, prob.pairs, ks)
        values = [prob.content[cert.nutrient, prob.pairs[k][1]] for k in ks]
        achievable = _feed_knapsack_max(values, caps, prob.batch[cert.formula])
        required = prob.spec_lo[cert.nutrient, cert.formula] * prob.batch[cert.formula]
    else
        1 <= cert.mill <= prob.n_mills || return false
        formulas = findall(==(cert.mill), prob.formula_mill)
        achievable = _feed_mill_capacity(
            prob.stock, prob.contract, prob.lot_supplier, prob.upper, prob.pairs, prob.formula_pairs, formulas
        )
        required = sum(prob.batch[f] for f in formulas)
    end
    isapprox(achievable, cert.achievable; rtol=1e-8, atol=1e-9) || return false
    isapprox(required, cert.required; rtol=1e-8, atol=1e-9) || return false
    return achievable < required * (1 - 1e-9)
end

"""
    FeedBlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a feed-mill network formulation instance (see `FeedBlendingProblem`).
"""
function FeedBlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)
    n_mills = clamp(round(Int, target / 2500 * rand(rng, Uniform(0.8, 1.25))), 1, 40)
    lot_ingredient, lot_supplier, lot_mill, content, cost, n_suppliers = _feed_sample_lots(rng, n_mills)
    n_lots = length(lot_ingredient)
    lots_at = [findall(==(m), lot_mill) for m in 1:n_mills]
    # Each mill's formula book favours a few species (integrators specialise).
    mill_weights = [rand(rng, Dirichlet(fill(0.6, length(_FEED_FORMULAS)))) for _ in 1:n_mills]

    formula_type = Int[]
    formula_mill = Int[]
    batch = Float64[]
    pairs = Tuple{Int, Int}[]
    formula_pairs = UnitRange{Int}[]
    lower = Float64[]
    upper = Float64[]
    while length(pairs) < target || isempty(formula_type)
        f = length(formula_type) + 1
        m = (f - 1) % n_mills + 1
        t = rand(rng, Categorical(mill_weights[m]))
        group = _FEED_FORMULAS[t].group
        D = clamp(rand(rng, LogNormal(log(25.0), 0.8)), 2.0, 400.0)
        push!(formula_type, t)
        push!(formula_mill, m)
        push!(batch, D)
        first_pair = length(pairs) + 1
        usable = [l for l in lots_at[m] if _feed_inclusion_cap(group, lot_ingredient[l]) > 0]
        # The last formula drops optional lots so the count lands on target;
        # the staple lots (always stocked) keep its batch makeable.
        if f > 1 && length(pairs) + length(usable) > target
            staple = [l for l in usable if _FEED_INGREDIENTS[lot_ingredient[l]].name in _FEED_STAPLES]
            optional = [l for l in usable if !(l in staple)]
            keep = clamp(target - length(pairs) - length(staple), 0, length(optional))
            usable = sort(vcat(staple, optional[1:keep]))
        end
        for l in usable
            cap = _feed_inclusion_cap(group, lot_ingredient[l])
            push!(pairs, (l, f))
            push!(upper, cap * D)
            push!(lower, _FEED_INGREDIENTS[lot_ingredient[l]].class == :premix ? 0.002 * D : 0.0)
        end
        push!(formula_pairs, first_pair:length(pairs))
    end
    n_formulas = length(formula_type)

    spec_lo = zeros(Float64, length(FEED_NUTRIENTS), n_formulas)
    spec_hi = fill(Inf, length(FEED_NUTRIENTS), n_formulas)
    ratio_band = zeros(Float64, 2, n_formulas)
    for f in 1:n_formulas
        ref = _FEED_FORMULAS[formula_type[f]]
        for n in eachindex(FEED_NUTRIENTS)
            spec_lo[n, f], spec_hi[n, f] = ref.spec[n]
        end
        ratio_band[:, f] .= ref.ca_p
    end

    # Reference recipes: the planted witness (feasible/infeasible) or the
    # nominal use around which unknown instances draw stock and contracts.
    x = zeros(Float64, length(pairs))
    for f in 1:n_formulas
        ks = formula_pairs[f]
        lots = [pairs[k][1] for k in ks]
        caps = [upper[k] / batch[f] for k in ks]
        frac = _feed_reference_recipe(rng, _FEED_FORMULAS[formula_type[f]].group, lots, lot_ingredient, caps)
        x[ks] .= batch[f] .* frac
    end
    used = zeros(Float64, n_lots)
    for (k, (l, _)) in enumerate(pairs)
        used[l] += x[k]
    end
    by_supplier = zeros(Float64, n_suppliers)
    for l in 1:n_lots
        by_supplier[lot_supplier[l]] += used[l]
    end

    stock = zeros(Float64, n_lots)
    contract = fill(Inf, n_suppliers)
    witness = nothing
    # Specifications: the reference ones, relaxed only where the formula's
    # reference recipe falls outside them, so every formula on its own can be
    # made (a nutritionist would not issue an unmakeable formula).
    for f in 1:n_formulas
        ks = formula_pairs[f]
        lots = [pairs[k][1] for k in ks]
        profile = _feed_profile(content, lots, x[ks] ./ batch[f])
        for n in eachindex(FEED_NUTRIENTS)
            spec_lo[n, f] = min(spec_lo[n, f], profile[n] * rand(rng, Uniform(0.95, 0.99)))
            isfinite(spec_hi[n, f]) && (spec_hi[n, f] = max(spec_hi[n, f], profile[n] * rand(rng, Uniform(1.01, 1.05))))
        end
        ratio = profile[_FEED_CA] / max(profile[_FEED_AVP], 1e-9)
        ratio_band[1, f] = min(ratio_band[1, f], 0.97 * ratio)
        ratio_band[2, f] = max(ratio_band[2, f], 1.03 * ratio)
    end
    if feasibility_status == unknown
        # Supply is this season's draw around the reference recipes' use: one
        # instance-wide tightness with per-lot and per-contract scatter, so the
        # mills may or may not be able to make their whole book.
        tightness = rand(rng, Uniform(0.85, 1.4))
        for l in 1:n_lots
            stock[l] = max(used[l], 0.5) * tightness * rand(rng, Uniform(0.85, 1.2))
        end
        for s in 1:n_suppliers
            by_supplier[s] > 0 && (contract[s] = by_supplier[s] * rand(rng, Uniform(0.85, 1.5)))
        end
    else
        for l in 1:n_lots
            stock[l] = used[l] > 0 ? used[l] * rand(rng, Uniform(1.02, 1.30)) : rand(rng, Uniform(1.0, 20.0))
        end
        for s in 1:n_suppliers
            by_supplier[s] > 0 && (contract[s] = by_supplier[s] * rand(rng, Uniform(1.02, 1.25)))
        end
        witness = x
    end

    # A contract covering a single lot is folded into that lot's stock, so
    # every contract left is a genuine multi-mill coupling row.
    for s in 1:n_suppliers
        isfinite(contract[s]) || continue
        lots = findall(==(s), lot_supplier)
        if length(lots) == 1
            stock[only(lots)] = min(stock[only(lots)], contract[s])
            contract[s] = Inf
        end
    end

    certificate = nothing
    if feasibility_status == infeasible
        witness = nothing
        # A customer specification above a formula's reachable maximum, on a
        # nutrient where the batch equality (not the caps of the few carriers)
        # is what limits it — otherwise the single nutrient row would already
        # be contradicted by the column bounds and presolve would see it.
        nutrient_mode = nothing
        if rand(rng) < 0.6
            for f in shuffle(rng, collect(1:n_formulas))
                ks = formula_pairs[f]
                caps = _feed_pair_caps(upper, stock, contract, lot_supplier, pairs, ks)
                options = Tuple{Int, Float64}[]
                for n in eachindex(FEED_NUTRIENTS)
                    spec_lo[n, f] > 0 || continue
                    values = [content[n, pairs[k][1]] for k in ks]
                    achievable = _feed_knapsack_max(values, caps, batch[f])
                    row_bound = sum(values[j] * caps[j] for j in eachindex(ks))
                    row_bound >= 1.3 * achievable && push!(options, (n, achievable))
                end
                isempty(options) && continue
                nutrient_mode = (f, rand(rng, options)...)
                break
            end
        end
        if nutrient_mode !== nothing
            f, n, achievable = nutrient_mode
            spec_lo[n, f] = achievable * rand(rng, Uniform(1.05, 1.12)) / batch[f]
            spec_hi[n, f] = max(spec_hi[n, f], 1.1 * spec_lo[n, f])
            certificate = FeedInfeasibilityCertificate(
                feed_nutrient_unreachable, f, n, 0, achievable, spec_lo[n, f] * batch[f]
            )
        else
            # One mill's stock falls short of its whole batch book.
            m = rand(rng, 1:n_mills)
            formulas = findall(==(m), formula_mill)
            required = sum(batch[f] for f in formulas)
            lots, effective = _feed_mill_lot_caps(stock, contract, lot_supplier, upper, pairs, formula_pairs, formulas)
            theta = required / rand(rng, Uniform(1.08, 1.25)) / sum(effective)
            for (j, l) in enumerate(lots)
                stock[l] = theta * effective[j]
            end
            achievable = _feed_mill_capacity(stock, contract, lot_supplier, upper, pairs, formula_pairs, formulas)
            certificate = FeedInfeasibilityCertificate(feed_mill_short, 0, 0, m, achievable, required)
        end
    end

    prob = FeedBlendingProblem(
        n_mills, lot_ingredient, lot_supplier, lot_mill, content, cost, formula_type, formula_mill, batch, pairs,
        formula_pairs, lower, upper, spec_lo, spec_hi, ratio_band, stock, contract, witness, certificate,
        feasibility_status,
    )
    feasibility_status == feasible && @assert feed_formulation_satisfies(prob)
    feasibility_status == infeasible && @assert feed_certificate_holds(prob)
    return prob
end

"""
    build_model(prob::FeedBlendingProblem)

Build the feed-mill network formulation LP (deterministic; see the type).
"""
function build_model(prob::FeedBlendingProblem)
    model = Model()
    K = length(prob.pairs)
    @variable(model, prob.lower[k] <= x[k=1:K] <= prob.upper[k])
    @objective(model, Min, sum(prob.cost[l] * x[k] for (k, (l, _)) in enumerate(prob.pairs)))
    C = prob.content
    for (f, ks) in enumerate(prob.formula_pairs)
        D = prob.batch[f]
        @constraint(model, sum(x[k] for k in ks) == D)
        for n in eachindex(FEED_NUTRIENTS)
            carriers = [k for k in ks if C[n, prob.pairs[k][1]] > 0]
            if prob.spec_lo[n, f] > 0
                @constraint(model, sum(C[n, prob.pairs[k][1]] * x[k] for k in carriers) >= prob.spec_lo[n, f] * D)
            end
            if isfinite(prob.spec_hi[n, f]) && any(C[n, prob.pairs[k][1]] > prob.spec_hi[n, f] for k in ks)
                @constraint(model, sum(C[n, prob.pairs[k][1]] * x[k] for k in carriers) <= prob.spec_hi[n, f] * D)
            end
        end
        lo, hi = prob.ratio_band[1, f], prob.ratio_band[2, f]
        @constraint(model, sum((C[_FEED_CA, prob.pairs[k][1]] - lo * C[_FEED_AVP, prob.pairs[k][1]]) * x[k] for k in ks) >= 0)
        @constraint(model, sum((C[_FEED_CA, prob.pairs[k][1]] - hi * C[_FEED_AVP, prob.pairs[k][1]]) * x[k] for k in ks) <= 0)
    end
    at_lot = [Int[] for _ in eachindex(prob.stock)]
    for (k, (l, _)) in enumerate(prob.pairs)
        push!(at_lot[l], k)
    end
    for l in eachindex(at_lot)
        isempty(at_lot[l]) && continue
        @constraint(model, sum(x[k] for k in at_lot[l]) <= prob.stock[l])
    end
    for s in eachindex(prob.contract)
        isfinite(prob.contract[s]) || continue
        lots = [l for l in eachindex(prob.stock) if prob.lot_supplier[l] == s && !isempty(at_lot[l])]
        isempty(lots) && continue
        @constraint(model, sum(x[k] for l in lots for k in at_lot[l]) <= prob.contract[s])
    end
    return model
end

register_variant(
    :feed_blending,
    :standard,
    FeedBlendingProblem,
    "Least-cost feed formulation for a network of feed mills: species/phase formula books in fixed " *
    "batches, NRC-style nutrient and Ca:P specs, inclusion limits, mill stock and shared supplier contracts";
    tags=[:agriculture, :blending, :block_angular],
)
