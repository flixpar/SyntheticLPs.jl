using Random
using Distributions
using StatsBase

# Shared food-composition and dietary-reference catalog for the diet_problem
# variants. All values are per serving (foods) or per person per day
# (requirements) and are rounded from USDA FoodData Central category medians and
# the US Dietary Reference Intakes, so magnitudes, sparsity and correlations look
# like a real food-composition table rather than iid noise.

"""
Nutrients tracked by the diet generators, in the row order of every content
matrix. Energy (kcal) is not sampled: it is computed from the macronutrients with
Atwater factors (4 protein + 9 fat + 4 carbohydrate), so energy, fat and
saturated-fat columns are correlated exactly as in real composition data.
"""
const DIET_NUTRIENTS = (
    :energy,        # kcal
    :protein,       # g
    :fat,           # g
    :saturated_fat, # g
    :carbohydrate,  # g
    :fiber,         # g
    :added_sugar,   # g
    :sodium,        # mg
    :calcium,       # mg
    :iron,          # mg
    :potassium,     # mg
    :magnesium,     # mg
    :zinc,          # mg
    :vitamin_a,     # µg RAE
    :vitamin_c,     # mg
    :vitamin_d,     # µg
    :vitamin_b12,   # µg
    :folate,        # µg DFE
)

const DIET_ENERGY = 1
const DIET_PROTEIN = 2
const DIET_FAT = 3
const DIET_SATFAT = 4
const DIET_CARB = 5
const DIET_FIBER = 6
const DIET_SUGAR = 7
const DIET_SODIUM = 8
# Micronutrients with a minimum requirement (calcium .. folate).
const DIET_MICRONUTRIENTS = 9:18

"""
Food categories. Every food belongs to exactly one; the category fixes its
nutrient profile, cost level, portion limit and the food-group it counts toward.
"""
const DIET_FOOD_CATEGORIES = (
    :grains,
    :vegetables,
    :fruits,
    :dairy,
    :meat_poultry,
    :fish_seafood,
    :eggs,
    :legumes,
    :nuts_seeds,
    :fats_oils,
    :sweets_snacks,
    :beverages,
    :mixed_dishes,
)

# Columns: protein g, fat g, sat-fat fraction of fat, carbohydrate g, fiber g,
# added-sugar fraction of carbohydrate, sodium mg, calcium mg, iron mg,
# potassium mg, magnesium mg, zinc mg, vitamin A µg, vitamin C mg, vitamin D µg,
# vitamin B12 µg, folate µg — per serving, conditional on being present.
const _DIET_PROFILE_MEDIAN = [
    5.0 2.0 0.20 30.0 2.5 0.06 200.0 30.0 1.5 90.0 30.0 0.8 150.0 2.0 1.0 1.0 60.0
    2.0 0.3 0.15 7.0 2.5 0.00 30.0 40.0 0.8 300.0 25.0 0.3 150.0 25.0 0.0 0.0 50.0
    0.8 0.3 0.10 18.0 2.5 0.00 2.0 15.0 0.3 250.0 15.0 0.1 40.0 30.0 0.0 0.0 20.0
    8.0 5.0 0.62 12.0 0.5 0.25 120.0 300.0 0.1 350.0 27.0 1.0 100.0 0.5 2.5 1.0 12.0
    25.0 10.0 0.35 0.5 0.0 0.00 300.0 15.0 2.0 300.0 25.0 4.0 10.0 0.5 0.3 2.0 8.0
    22.0 6.0 0.20 0.5 0.0 0.00 250.0 30.0 0.8 350.0 35.0 0.8 20.0 0.5 8.0 3.0 15.0
    6.5 5.0 0.32 0.5 0.0 0.00 70.0 28.0 0.9 70.0 6.0 0.6 80.0 0.0 1.1 0.5 24.0
    8.0 0.7 0.15 21.0 7.0 0.00 150.0 40.0 2.2 350.0 45.0 1.0 5.0 1.0 0.0 0.0 130.0
    6.0 15.0 0.12 6.0 3.0 0.05 5.0 50.0 1.2 200.0 75.0 1.2 1.0 0.3 0.0 0.0 15.0
    0.3 13.0 0.25 0.3 0.0 0.00 60.0 2.0 0.05 5.0 1.0 0.05 60.0 0.0 0.5 0.0 0.0
    2.0 8.0 0.45 28.0 1.0 0.55 150.0 25.0 1.0 100.0 15.0 0.4 15.0 0.5 0.1 0.1 15.0
    1.5 0.5 0.30 20.0 0.5 0.45 20.0 40.0 0.3 250.0 15.0 0.2 30.0 40.0 1.5 0.4 25.0
    15.0 12.0 0.38 35.0 3.5 0.08 700.0 120.0 2.5 400.0 45.0 2.5 80.0 8.0 0.5 0.8 60.0
]

# Probability that the nutrient is present at all (same column order).
const _DIET_PROFILE_PRESENCE = [
    1.0 0.95 1.0 1.0 0.95 0.6 0.9 0.9 1.0 1.0 1.0 1.0 0.15 0.1 0.1 0.15 0.9
    1.0 0.8 1.0 1.0 1.0 0.0 0.9 1.0 1.0 1.0 1.0 0.9 0.75 0.9 0.0 0.0 1.0
    1.0 0.8 1.0 1.0 1.0 0.0 0.5 1.0 0.9 1.0 1.0 0.8 0.5 1.0 0.0 0.0 0.9
    1.0 0.9 1.0 1.0 0.05 0.5 1.0 1.0 0.6 1.0 1.0 1.0 0.9 0.4 0.8 1.0 0.9
    1.0 1.0 1.0 0.3 0.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 0.4 0.2 0.5 1.0 0.8
    1.0 1.0 1.0 0.2 0.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 0.6 0.2 1.0 1.0 0.8
    1.0 1.0 1.0 1.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 1.0 1.0 1.0
    1.0 1.0 1.0 1.0 1.0 0.0 0.7 1.0 1.0 1.0 1.0 1.0 0.3 0.5 0.0 0.0 1.0
    1.0 1.0 1.0 1.0 1.0 0.3 0.5 1.0 1.0 1.0 1.0 1.0 0.2 0.3 0.0 0.0 0.9
    0.1 1.0 1.0 0.1 0.0 0.0 0.4 0.3 0.1 0.3 0.1 0.1 0.4 0.0 0.25 0.0 0.0
    1.0 1.0 1.0 1.0 0.7 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.4 0.2 0.2 0.2 0.7
    0.7 0.4 1.0 1.0 0.2 0.6 1.0 0.9 0.6 1.0 1.0 0.5 0.4 0.6 0.3 0.25 0.5
    1.0 1.0 1.0 1.0 1.0 0.8 1.0 1.0 1.0 1.0 1.0 1.0 0.9 0.7 0.6 0.8 1.0
]

const _DIET_COST_MEDIAN = (
    0.25, 0.45, 0.55, 0.40, 1.40, 1.90, 0.30, 0.30, 0.55, 0.12, 0.60, 0.65, 2.20
)
const _DIET_MAX_SERVINGS = (4.0, 3.5, 3.0, 3.0, 2.0, 1.5, 2.0, 2.0, 1.5, 3.0, 1.5, 2.0, 1.5)
# Servings per day of each category in a 2000-kcal guideline eating pattern.
const _DIET_PATTERN_SERVINGS = (4.5, 3.0, 2.0, 2.5, 1.0, 0.4, 0.5, 0.6, 0.4, 1.5, 0.4, 0.8, 0.4)

"""
USDA dietary food groups a category counts toward (`food_groups` variant):
1 grains, 2 vegetables, 3 fruits, 4 dairy, 5 protein foods; 0 = none
(fats, sweets, beverages, mixed dishes).
"""
const DIET_FOOD_GROUPS = (:grains, :vegetables, :fruits, :dairy, :protein_foods)
const _DIET_CATEGORY_GROUP = (1, 2, 3, 4, 5, 5, 5, 5, 5, 0, 0, 0, 0)

"""
Demographic groups with their estimated energy requirement (EER, kcal), the
minimum requirements (RDA/AI) for protein, fiber and the ten micronutrients
(calcium .. folate, in `DIET_NUTRIENTS` order) and the sodium limit (CDRR, mg).
"""
const DIET_DEMOGRAPHICS = (
    (
        name=:child_4_8,
        eer=1500.0,
        protein=19.0,
        fiber=25.0,
        sodium=1900.0,
        micro=(1000.0, 10.0, 2300.0, 130.0, 5.0, 400.0, 25.0, 15.0, 1.2, 200.0),
    ),
    (
        name=:boy_9_13,
        eer=2000.0,
        protein=34.0,
        fiber=31.0,
        sodium=2200.0,
        micro=(1300.0, 8.0, 2500.0, 240.0, 8.0, 600.0, 45.0, 15.0, 1.8, 300.0),
    ),
    (
        name=:girl_9_13,
        eer=1800.0,
        protein=34.0,
        fiber=26.0,
        sodium=2200.0,
        micro=(1300.0, 8.0, 2300.0, 240.0, 8.0, 600.0, 45.0, 15.0, 1.8, 300.0),
    ),
    (
        name=:male_14_18,
        eer=2600.0,
        protein=52.0,
        fiber=38.0,
        sodium=2300.0,
        micro=(1300.0, 11.0, 3000.0, 410.0, 11.0, 900.0, 75.0, 15.0, 2.4, 400.0),
    ),
    (
        name=:female_14_18,
        eer=2000.0,
        protein=46.0,
        fiber=26.0,
        sodium=2300.0,
        micro=(1300.0, 15.0, 2300.0, 360.0, 9.0, 700.0, 65.0, 15.0, 2.4, 400.0),
    ),
    (
        name=:man_19_50,
        eer=2600.0,
        protein=56.0,
        fiber=38.0,
        sodium=2300.0,
        micro=(1000.0, 8.0, 3400.0, 400.0, 11.0, 900.0, 90.0, 15.0, 2.4, 400.0),
    ),
    (
        name=:woman_19_50,
        eer=2000.0,
        protein=46.0,
        fiber=25.0,
        sodium=2300.0,
        micro=(1000.0, 18.0, 2600.0, 310.0, 8.0, 700.0, 75.0, 15.0, 2.4, 400.0),
    ),
    (
        name=:man_51_plus,
        eer=2300.0,
        protein=56.0,
        fiber=30.0,
        sodium=2300.0,
        micro=(1000.0, 8.0, 3400.0, 420.0, 11.0, 900.0, 90.0, 20.0, 2.4, 400.0),
    ),
    (
        name=:woman_51_plus,
        eer=1800.0,
        protein=46.0,
        fiber=21.0,
        sodium=2300.0,
        micro=(1200.0, 8.0, 2600.0, 320.0, 8.0, 700.0, 75.0, 20.0, 2.4, 400.0),
    ),
    (
        name=:pregnant,
        eer=2300.0,
        protein=71.0,
        fiber=28.0,
        sodium=2300.0,
        micro=(1000.0, 27.0, 2900.0, 350.0, 11.0, 770.0, 85.0, 15.0, 2.6, 600.0),
    ),
    (
        name=:lactating,
        eer=2400.0,
        protein=71.0,
        fiber=29.0,
        sodium=2300.0,
        micro=(1000.0, 9.0, 2800.0, 310.0, 12.0, 1300.0, 120.0, 15.0, 2.8, 500.0),
    ),
)

"""
    diet_reference_minimum(demographic, nutrient)

Reference minimum daily intake of `nutrient` (an index into `DIET_NUTRIENTS`)
for a `DIET_DEMOGRAPHICS` entry; energy returns the EER.
"""
function diet_reference_minimum(demographic, nutrient::Int)
    nutrient == DIET_ENERGY && return demographic.eer
    nutrient == DIET_PROTEIN && return demographic.protein
    nutrient == DIET_FIBER && return demographic.fiber
    nutrient in DIET_MICRONUTRIENTS && return demographic.micro[nutrient - 8]
    throw(ArgumentError("nutrient $(DIET_NUTRIENTS[nutrient]) has no minimum requirement"))
end

"""
    DietFoodTable

A sampled food-composition table: `category[f]` indexes
`DIET_FOOD_CATEGORIES`, `content[k, f]` is nutrient `k` per serving of food `f`
(rows in `DIET_NUTRIENTS` order), `cost[f]` is the price per serving and
`max_servings[f]` the per-person daily portion limit for a 2000-kcal appetite.
"""
struct DietFoodTable
    category::Vector{Int}
    content::Matrix{Float64}
    cost::Vector{Float64}
    max_servings::Vector{Float64}
end

"""
    _diet_sample_categories(rng, n_foods)

Category of each food: every category appears once while there is room, the
rest follow the share of each category in a food-composition table (plant foods
and mixed dishes are the most numerous). Shuffled so indices carry no structure.
"""
function _diet_sample_categories(rng::AbstractRNG, n_foods::Int)
    weights = [1.6, 1.8, 1.3, 1.0, 1.3, 0.9, 0.3, 0.8, 0.7, 0.5, 1.0, 0.8, 1.4]
    categories = Int[]
    for c in shuffle(rng, collect(eachindex(DIET_FOOD_CATEGORIES)))
        length(categories) < n_foods && push!(categories, c)
    end
    probabilities = weights ./ sum(weights)
    while length(categories) < n_foods
        push!(categories, rand(rng, Categorical(probabilities)))
    end
    return shuffle!(rng, categories)
end

"""
    _diet_sample_food_table(rng, n_foods)

Sample a role-correlated food-composition table. Every food draws a portion
factor shared by all of its nutrients (large servings are rich in everything)
and a price that rises with it, so nutrient columns are correlated within a food
and nutrient profiles cluster by category.
"""
function _diet_sample_food_table(rng::AbstractRNG, n_foods::Int)
    category = _diet_sample_categories(rng, n_foods)
    content = zeros(Float64, length(DIET_NUTRIENTS), n_foods)
    cost = zeros(Float64, n_foods)
    max_servings = zeros(Float64, n_foods)
    for f in 1:n_foods
        c = category[f]
        portion = rand(rng, LogNormal(0.0, 0.22))
        values = zeros(Float64, size(_DIET_PROFILE_MEDIAN, 2))
        for k in eachindex(values)
            p = _DIET_PROFILE_PRESENCE[c, k]
            median = _DIET_PROFILE_MEDIAN[c, k]
            (median > 0.0 && rand(rng) < p) || continue
            if k == 3 || k == 6
                # Fractions: saturated share of fat, added-sugar share of carbs.
                values[k] = clamp(median * rand(rng, LogNormal(0.0, 0.25)), 0.0, 0.9)
            else
                sigma = k >= 13 ? 0.6 : 0.4
                values[k] = portion * median * rand(rng, LogNormal(0.0, sigma))
            end
        end
        protein, fat, carb = values[1], values[2], values[4]
        content[DIET_PROTEIN, f] = protein
        content[DIET_FAT, f] = fat
        content[DIET_SATFAT, f] = fat * values[3]
        content[DIET_CARB, f] = carb
        content[DIET_FIBER, f] = min(values[5], 0.6 * carb + 0.5)
        content[DIET_SUGAR, f] = carb * values[6]
        for k in 7:17
            content[k + 1, f] = values[k]
        end
        energy = 4.0 * protein + 9.0 * fat + 4.0 * carb
        content[DIET_ENERGY, f] = max(energy * rand(rng, Uniform(0.96, 1.04)), 5.0)
        cost[f] = _DIET_COST_MEDIAN[c] * portion^0.8 * rand(rng, LogNormal(0.0, 0.3))
        max_servings[f] = _DIET_MAX_SERVINGS[c] * rand(rng, Uniform(0.7, 1.3))
    end
    return DietFoodTable(category, content, cost, max_servings)
end

"""
    _diet_pattern_diet(rng, table, appetite, foods_by_category; choices=1:3)

A guideline-pattern daily diet: each category's pattern servings (scaled by the
person's `appetite` = EER / 2000) are split over 1–3 random foods of that
category with Dirichlet weights and clipped at the portion limits. Returns a
servings vector over all foods. This is a plausible menu, not a cost-optimal
one, which is what makes it a good planted witness.
"""
function _diet_pattern_diet(
    rng::AbstractRNG,
    table::DietFoodTable,
    appetite::Float64,
    foods_by_category::Vector{Vector{Int}};
    choices::UnitRange{Int}=1:3,
)
    servings = zeros(Float64, length(table.cost))
    for c in eachindex(DIET_FOOD_CATEGORIES)
        members = foods_by_category[c]
        isempty(members) && continue
        total = _DIET_PATTERN_SERVINGS[c] * appetite * rand(rng, Uniform(0.75, 1.25))
        k = min(length(members), rand(rng, choices))
        chosen = k == length(members) ? members : sample(rng, members, k; replace=false)
        weights = k == 1 ? [1.0] : rand(rng, Dirichlet(fill(2.0, k)))
        for (j, f) in enumerate(chosen)
            servings[f] += min(total * weights[j], 0.9 * table.max_servings[f] * appetite)
        end
    end
    return servings
end

"""
    _diet_max_under_energy_cap(content, upper, nutrient, energy_cap)

Exact optimum of the one-row continuous knapsack
`max Σ_f content[nutrient,f] x_f  s.t.  Σ_f content[energy,f] x_f ≤ energy_cap,
0 ≤ x ≤ upper`: fill foods in decreasing nutrient-per-kcal order. Any diet that
respects the energy ceiling and the portion limits delivers at most this much of
`nutrient`, so a requirement above it is infeasible from LP rows alone.
"""
function _diet_max_under_energy_cap(
    content::AbstractMatrix{<:Real}, upper::AbstractVector{<:Real}, nutrient::Int, energy_cap::Real
)
    n = length(upper)
    order = sort(1:n; by=f -> -content[nutrient, f] / content[DIET_ENERGY, f])
    remaining = Float64(energy_cap)
    total = 0.0
    for f in order
        content[nutrient, f] > 0.0 || break
        remaining <= 0.0 && break
        amount = min(Float64(upper[f]), remaining / content[DIET_ENERGY, f])
        total += content[nutrient, f] * amount
        remaining -= content[DIET_ENERGY, f] * amount
    end
    return total
end

function _diet_foods_by_category(category::Vector{Int})
    groups = [Int[] for _ in DIET_FOOD_CATEGORIES]
    for (f, c) in enumerate(category)
        push!(groups[c], f)
    end
    return groups
end

"""
Daily dietary-guideline limits shared by the diet variants: energy band as a
fraction of EER, saturated fat and added sugar as maximum shares of energy, and
the acceptable total-fat share band.
"""
const DIET_ENERGY_BAND = (0.90, 1.10)
const DIET_SATFAT_SHARE = 0.10
const DIET_SUGAR_SHARE = 0.10
const DIET_FAT_SHARE_BAND = (0.20, 0.35)
