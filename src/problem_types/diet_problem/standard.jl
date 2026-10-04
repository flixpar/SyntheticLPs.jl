using JuMP
using Random
using Distributions
using StatsBase

"""
Reason a requested-infeasible diet instance has no feasible plan.

  - `diet_supply_shortage`: population-wide. Summing nutrient `k`'s minimum row
    over every cohort (weighted by headcount) needs more of `k` than the foods
    carrying it can deliver, even if every one of them is used up to the smaller
    of its supply limit and the cohorts' combined portion limits.
  - `diet_energy_squeeze`: one cohort. Its minimum for nutrient `k` exceeds the
    most of `k` any diet can contain without breaking the cohort's energy
    ceiling and portion limits (an exact one-row fractional knapsack).
"""
@enum DietInfeasibilityKind begin
    diet_supply_shortage
    diet_energy_squeeze
end

"""
    DietInfeasibilityCertificate

Solver-free proof of infeasibility built from LP rows alone (it survives any
relaxation). `nutrient` indexes `DIET_NUTRIENTS`; `cohort` is the squeezed cohort
(`0` for a population-wide supply shortage). `achievable < required` with a
margin of at least 5%; `diet_certificate_holds` recomputes both from the data.
"""
struct DietInfeasibilityCertificate
    kind::DietInfeasibilityKind
    nutrient::Int
    cohort::Int
    achievable::Float64
    required::Float64
end

"""
    DietProblem <: ProblemGenerator

Population diet planning: the least-cost daily diet for many population cohorts
(age/sex/life-stage groups at institutions — schools, hospitals, care homes,
barracks) who share the food supply of one procurement region.

# Formulation

Decision `x[f, g] ∈ [0, upper[f, g]]` is the daily servings of food `f` per
person of cohort `g`; `upper` is the food's portion limit scaled by the cohort's
appetite. The objective minimizes `Σ_g headcount[g] Σ_f cost[f] x[f, g]`. Per
cohort (every row is a full diet row over the food list):

  - energy band `energy_band[1,g] ≤ Σ_f kcal_f x ≤ energy_band[2,g]` (ranged row);
  - nutrient minimums for protein, fiber and the tracked micronutrients;
  - a sodium ceiling;
  - dietary-guideline share rows, homogeneous with mixed signs:
    `Σ_f (9·satfat_f − s·kcal_f) x ≤ 0` (saturated fat ≤ s of energy), the same
    for added sugar (`4·sugar_f`), and a total-fat share band (two rows).

Shared supply couples the cohorts: `Σ_g headcount[g] x[f, g] ≤ supply[f]` for the
foods with limited regional availability. The food table is role-correlated
(`_diet_sample_food_table`): nutrient profiles cluster by food category, energy
follows the macronutrients, and requirements come from the Dietary Reference
Intakes of each cohort's demographic, so many rows bind at the optimum.

# Sizing

`n_foods * n_cohorts` variables. The food list has `≈ 4√target` items (8–200);
`n_cohorts = round(target / n_foods_nominal)`, then `n_foods = round(target /
n_cohorts)`, so the count is within `n_cohorts / 2` of the target. Rows:
`n_cohorts * (3 + |min_nutrients| + has_sugar_limit + 2 has_fat_band) +
count(isfinite, supply)`.

# Feasibility

  - `feasible`: each cohort gets a guideline-pattern diet (a plausible menu, not
    the optimum). Requirements are the DRIs, lowered only where that diet falls
    short; limits are the guideline limits, raised only where it exceeds them;
    supplies of the foods it uses are 1.02–1.30× its consumption, so they bind
    near the optimum. The plan is stored as `feasible_witness`.
  - `infeasible`: the feasible construction plus one mutation with a typed
    `DietInfeasibilityCertificate` (supply shortage by default, energy squeeze
    otherwise). Neither is a single-row contradiction a presolve can see.
  - `unknown`: DRI requirements, guideline limits and supplies drawn around a
    nominal consumption estimate with no planted point.
"""
struct DietProblem <: ProblemGenerator
    n_foods::Int
    n_cohorts::Int
    food_category::Vector{Int}
    content::Matrix{Float64}
    cost::Vector{Float64}
    upper::Matrix{Float64}
    demographic::Vector{Symbol}
    headcount::Vector{Float64}
    min_nutrients::Vector{Int}
    min_requirement::Matrix{Float64}
    energy_band::Matrix{Float64}
    sodium_limit::Vector{Float64}
    satfat_share::Vector{Float64}
    sugar_share::Vector{Float64}
    fat_share_band::Matrix{Float64}
    has_sugar_limit::Bool
    has_fat_band::Bool
    supply::Vector{Float64}
    feasible_witness::Union{Nothing, Matrix{Float64}}
    infeasibility_certificate::Union{Nothing, DietInfeasibilityCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    diet_dimensions(target_variables, rng) -> (n_foods, n_cohorts)

Food-list length and cohort count for a target (see `DietProblem`).
"""
function _diet_standard_dimensions(rng::AbstractRNG, target_variables::Int)
    target = max(target_variables, 1)
    nominal = clamp(round(Int, 4.0 * sqrt(target) * rand(rng, Uniform(0.8, 1.25))), 8, 200)
    n_cohorts = max(1, round(Int, target / nominal))
    n_foods = max(8, round(Int, target / n_cohorts))
    return n_foods, n_cohorts
end

"""
    _diet_intake(content, servings)

Daily nutrient intake vector of a servings vector.
"""
_diet_intake(content::AbstractMatrix, servings::AbstractVector) = content * servings

"""
    _diet_repair_pattern!(servings, table, upper, nutrients, appetite_targets)

Top up a pattern diet that is very poor in a tracked nutrient with the food
richest in it (within portion limits), so no planted requirement collapses to a
token value.
"""
function _diet_repair_pattern!(
    servings::Vector{Float64},
    content::Matrix{Float64},
    upper::AbstractVector{Float64},
    nutrients::Vector{Int},
    references::Vector{Float64},
)
    for (r, k) in enumerate(nutrients)
        intake = sum(content[k, f] * servings[f] for f in eachindex(servings))
        intake >= 0.5 * references[r] && continue
        best = argmax(view(content, k, :))
        content[k, best] > 0.0 || continue
        headroom = max(0.0, 0.9 * upper[best] - servings[best])
        servings[best] += min(headroom, (0.5 * references[r] - intake) / content[k, best])
    end
    return servings
end

"""
    diet_plan_satisfies(prob::DietProblem, servings=prob.feasible_witness; atol=1e-7)

Check a per-person servings matrix (`n_foods × n_cohorts`) against every row and
bound of the model, without a solver.
"""
function diet_plan_satisfies(
    prob::DietProblem,
    servings::Union{Nothing, AbstractMatrix{<:Real}}=prob.feasible_witness;
    atol::Float64=1e-7,
)
    servings === nothing && return false
    size(servings) == (prob.n_foods, prob.n_cohorts) || return false
    tol(v) = atol * max(1.0, abs(v))
    for g in 1:prob.n_cohorts, f in 1:prob.n_foods
        -atol <= servings[f, g] <= prob.upper[f, g] + tol(prob.upper[f, g]) || return false
    end
    for g in 1:prob.n_cohorts
        intake = prob.content * view(servings, :, g)
        energy = intake[DIET_ENERGY]
        prob.energy_band[1, g] - tol(energy) <= energy <= prob.energy_band[2, g] + tol(energy) ||
            return false
        for (r, k) in enumerate(prob.min_nutrients)
            intake[k] + tol(intake[k]) >= prob.min_requirement[r, g] || return false
        end
        intake[DIET_SODIUM] <= prob.sodium_limit[g] + tol(intake[DIET_SODIUM]) || return false
        9.0 * intake[DIET_SATFAT] <= prob.satfat_share[g] * energy + tol(energy) || return false
        if prob.has_sugar_limit
            4.0 * intake[DIET_SUGAR] <= prob.sugar_share[g] * energy + tol(energy) || return false
        end
        if prob.has_fat_band
            9.0 * intake[DIET_FAT] + tol(energy) >= prob.fat_share_band[1, g] * energy ||
                return false
            9.0 * intake[DIET_FAT] <= prob.fat_share_band[2, g] * energy + tol(energy) ||
                return false
        end
    end
    for f in 1:prob.n_foods
        isfinite(prob.supply[f]) || continue
        used = sum(prob.headcount[g] * servings[f, g] for g in 1:prob.n_cohorts)
        used <= prob.supply[f] + tol(prob.supply[f]) || return false
    end
    return true
end

function _diet_population_caps(prob_supply, headcount, upper)
    n_foods = size(upper, 1)
    return [
        min(prob_supply[f], sum(headcount[g] * upper[f, g] for g in eachindex(headcount))) for
        f in 1:n_foods
    ]
end

"""
    diet_certificate_holds(prob::DietProblem; rtol=1e-9)

Recompute the stored infeasibility certificate from the instance data and check
that it proves infeasibility (`achievable < required`).
"""
function diet_certificate_holds(prob::DietProblem; rtol::Float64=1e-9)
    cert = prob.infeasibility_certificate
    cert === nothing && return false
    r = findfirst(==(cert.nutrient), prob.min_nutrients)
    r === nothing && return false
    if cert.kind == diet_supply_shortage
        cert.cohort == 0 || return false
        caps = _diet_population_caps(prob.supply, prob.headcount, prob.upper)
        achievable = sum(prob.content[cert.nutrient, f] * caps[f] for f in 1:prob.n_foods)
        required = sum(
            prob.headcount[g] * prob.min_requirement[r, g] for g in 1:prob.n_cohorts
        )
    else
        1 <= cert.cohort <= prob.n_cohorts || return false
        g = cert.cohort
        achievable = _diet_max_under_energy_cap(
            prob.content, view(prob.upper, :, g), cert.nutrient, prob.energy_band[2, g]
        )
        required = prob.min_requirement[r, g]
    end
    isapprox(achievable, cert.achievable; rtol=1e-8, atol=1e-9) || return false
    isapprox(required, cert.required; rtol=1e-8, atol=1e-9) || return false
    return achievable < required * (1 - rtol)
end

"""
    DietProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a population diet-planning instance (see `DietProblem`).
"""
function DietProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    n_foods, n_cohorts = _diet_standard_dimensions(rng, target_variables)

    table = _diet_sample_food_table(rng, n_foods)
    content = table.content
    by_category = _diet_foods_by_category(table.category)

    # Cohorts: a demographic (life-stage weights favour adults and children),
    # a headcount (institution sizes are heavy-tailed) and an activity factor.
    demo_weights = [1.2, 1.0, 1.0, 0.9, 0.9, 1.6, 1.6, 1.1, 1.1, 0.3, 0.3]
    demo_dist = Categorical(demo_weights ./ sum(demo_weights))
    demographic_index = [rand(rng, demo_dist) for _ in 1:n_cohorts]
    demographic = [DIET_DEMOGRAPHICS[d].name for d in demographic_index]
    headcount = [Float64(max(10, round(Int, rand(rng, LogNormal(log(120.0), 0.8))))) for
                 _ in 1:n_cohorts]
    activity = rand(rng, Uniform(0.92, 1.10), n_cohorts)
    appetite = [DIET_DEMOGRAPHICS[demographic_index[g]].eer * activity[g] / 2000.0 for
                g in 1:n_cohorts]
    upper = [table.max_servings[f] * appetite[g] for f in 1:n_foods, g in 1:n_cohorts]

    # Tracked minimum nutrients: protein, fiber and 6-10 micronutrients.
    n_micro = rand(rng, 6:length(DIET_MICRONUTRIENTS))
    micros = sort(sample(rng, collect(DIET_MICRONUTRIENTS), n_micro; replace=false))
    min_nutrients = vcat([DIET_PROTEIN, DIET_FIBER], micros)
    has_sugar_limit = rand(rng) < 0.85
    has_fat_band = rand(rng) < 0.8

    reference = [
        diet_reference_minimum(DIET_DEMOGRAPHICS[demographic_index[g]], k) *
        (k == DIET_PROTEIN ? 1.0 : activity[g]^0.5) for k in min_nutrients, g in 1:n_cohorts
    ]
    eer = [DIET_DEMOGRAPHICS[demographic_index[g]].eer * activity[g] for g in 1:n_cohorts]
    cdrr = [DIET_DEMOGRAPHICS[demographic_index[g]].sodium for g in 1:n_cohorts]

    n_min = length(min_nutrients)
    min_requirement = zeros(Float64, n_min, n_cohorts)
    energy_band = zeros(Float64, 2, n_cohorts)
    sodium_limit = zeros(Float64, n_cohorts)
    satfat_share = fill(DIET_SATFAT_SHARE, n_cohorts)
    sugar_share = fill(DIET_SUGAR_SHARE, n_cohorts)
    fat_share_band = repeat([DIET_FAT_SHARE_BAND[1], DIET_FAT_SHARE_BAND[2]], 1, n_cohorts)
    supply = fill(Inf, n_foods)
    witness = nothing

    if feasibility_status == unknown
        for g in 1:n_cohorts
            for r in 1:n_min
                min_requirement[r, g] = reference[r, g] * rand(rng, Uniform(0.97, 1.03))
            end
            energy_band[1, g] = DIET_ENERGY_BAND[1] * eer[g]
            energy_band[2, g] = DIET_ENERGY_BAND[2] * eer[g]
            sodium_limit[g] = cdrr[g]
        end
        # Supplies around a nominal regional consumption estimate (each food's
        # equal share of its category's pattern servings), scaled by an
        # instance-wide market tightness and a per-category supply shock (a
        # poor fishing season, a dairy shortage). A shock to the few categories
        # carrying a nutrient can leave the whole region short of it, so the
        # instance may be feasible or not.
        limited_fraction = rand(rng, Uniform(0.8, 1.0))
        tightness = rand(rng, Uniform(0.5, 1.4))
        shock = rand(rng, LogNormal(0.0, 0.7), length(DIET_FOOD_CATEGORIES))
        population = sum(headcount[g] * appetite[g] for g in 1:n_cohorts)
        for f in 1:n_foods
            rand(rng) < limited_fraction || continue
            c = table.category[f]
            share = _DIET_PATTERN_SERVINGS[c] / length(by_category[c])
            supply[f] = population * share * tightness * shock[c] * rand(rng, LogNormal(0.0, 0.3))
        end
    else
        servings = zeros(Float64, n_foods, n_cohorts)
        for g in 1:n_cohorts
            plan = _diet_pattern_diet(rng, table, appetite[g], by_category)
            plan .= min.(plan, 0.95 .* view(upper, :, g))
            _diet_repair_pattern!(plan, content, view(upper, :, g), min_nutrients, reference[:, g])
            servings[:, g] .= plan
            intake = _diet_intake(content, plan)
            for (r, k) in enumerate(min_nutrients)
                min_requirement[r, g] = min(reference[r, g], intake[k] * rand(rng, Uniform(0.90, 0.98)))
            end
            energy = intake[DIET_ENERGY]
            energy_band[1, g] = min(DIET_ENERGY_BAND[1] * eer[g], 0.97 * energy)
            energy_band[2, g] = max(DIET_ENERGY_BAND[2] * eer[g], 1.03 * energy)
            sodium_limit[g] = max(cdrr[g], intake[DIET_SODIUM] * rand(rng, Uniform(1.02, 1.08)))
            satfat_share[g] = max(
                DIET_SATFAT_SHARE, 9.0 * intake[DIET_SATFAT] / energy + rand(rng, Uniform(0.005, 0.02))
            )
            sugar_share[g] = max(
                DIET_SUGAR_SHARE, 4.0 * intake[DIET_SUGAR] / energy + rand(rng, Uniform(0.005, 0.02))
            )
            fat_share = 9.0 * intake[DIET_FAT] / energy
            fat_share_band[1, g] = min(DIET_FAT_SHARE_BAND[1], fat_share - rand(rng, Uniform(0.005, 0.02)))
            fat_share_band[2, g] = max(DIET_FAT_SHARE_BAND[2], fat_share + rand(rng, Uniform(0.005, 0.02)))
        end
        usage = servings * headcount
        for f in 1:n_foods
            usage[f] > 0.0 && rand(rng) < 0.6 || continue
            supply[f] = usage[f] * rand(rng, Uniform(1.02, 1.30))
        end
        witness = servings
    end

    certificate = nothing
    if feasibility_status == infeasible
        if rand(rng) < 0.6
            # Population-wide shortage of the sources of a scarce nutrient: pick
            # one of the three tracked minimum nutrients with fewest carriers and
            # cut every carrier's supply proportionally.
            candidates = collect(3:n_min)
            carriers = [count(>(0.0), view(content, min_nutrients[r], :)) for r in candidates]
            pool = candidates[sortperm(carriers)][1:min(3, length(candidates))]
            r = rand(rng, pool)
            k = min_nutrients[r]
            required = sum(headcount[g] * min_requirement[r, g] for g in 1:n_cohorts)
            caps = _diet_population_caps(supply, headcount, upper)
            available = sum(content[k, f] * caps[f] for f in 1:n_foods)
            theta = required / rand(rng, Uniform(1.08, 1.25)) / available
            for f in 1:n_foods
                content[k, f] > 0.0 || continue
                supply[f] = theta * caps[f]
            end
            caps = _diet_population_caps(supply, headcount, upper)
            achievable = sum(content[k, f] * caps[f] for f in 1:n_foods)
            certificate = DietInfeasibilityCertificate(diet_supply_shortage, k, 0, achievable, required)
        else
            # One cohort's requirement for a nutrient is raised above the most a
            # diet within its energy ceiling can provide.
            g = rand(rng, 1:n_cohorts)
            r = rand(rng, 1:n_min)
            k = min_nutrients[r]
            achievable = _diet_max_under_energy_cap(content, view(upper, :, g), k, energy_band[2, g])
            min_requirement[r, g] = achievable * rand(rng, Uniform(1.06, 1.15))
            certificate = DietInfeasibilityCertificate(
                diet_energy_squeeze, k, g, achievable, min_requirement[r, g]
            )
        end
        witness = nothing
    end

    prob = DietProblem(
        n_foods,
        n_cohorts,
        table.category,
        content,
        table.cost,
        upper,
        demographic,
        headcount,
        min_nutrients,
        min_requirement,
        energy_band,
        sodium_limit,
        satfat_share,
        sugar_share,
        fat_share_band,
        has_sugar_limit,
        has_fat_band,
        supply,
        witness,
        certificate,
        feasibility_status,
    )
    feasibility_status == feasible && @assert diet_plan_satisfies(prob)
    feasibility_status == infeasible && @assert diet_certificate_holds(prob)
    return prob
end

"""
    build_model(prob::DietProblem)

Build the population diet LP (deterministic; see `DietProblem`).
"""
function build_model(prob::DietProblem)
    model = Model()
    F, G = prob.n_foods, prob.n_cohorts
    C = prob.content
    @variable(model, 0 <= x[f=1:F, g=1:G] <= prob.upper[f, g])
    @objective(model, Min, sum(prob.headcount[g] * prob.cost[f] * x[f, g] for f in 1:F, g in 1:G))

    carriers = [findall(>(0.0), view(C, k, :)) for k in eachindex(DIET_NUTRIENTS)]
    satfat_coef(g) = [9.0 * C[DIET_SATFAT, f] - prob.satfat_share[g] * C[DIET_ENERGY, f] for f in 1:F]
    sugar_coef(g) = [4.0 * C[DIET_SUGAR, f] - prob.sugar_share[g] * C[DIET_ENERGY, f] for f in 1:F]
    fat_coef(g, s) = [9.0 * C[DIET_FAT, f] - s * C[DIET_ENERGY, f] for f in 1:F]

    for g in 1:G
        @constraint(
            model,
            prob.energy_band[1, g] <= sum(C[DIET_ENERGY, f] * x[f, g] for f in 1:F) <= prob.energy_band[2, g]
        )
        for (r, k) in enumerate(prob.min_nutrients)
            @constraint(model, sum(C[k, f] * x[f, g] for f in carriers[k]) >= prob.min_requirement[r, g])
        end
        @constraint(model, sum(C[DIET_SODIUM, f] * x[f, g] for f in carriers[DIET_SODIUM]) <= prob.sodium_limit[g])
        a = satfat_coef(g)
        @constraint(model, sum(a[f] * x[f, g] for f in 1:F) <= 0)
        if prob.has_sugar_limit
            a = sugar_coef(g)
            @constraint(model, sum(a[f] * x[f, g] for f in 1:F) <= 0)
        end
        if prob.has_fat_band
            a = fat_coef(g, prob.fat_share_band[1, g])
            @constraint(model, sum(a[f] * x[f, g] for f in 1:F) >= 0)
            a = fat_coef(g, prob.fat_share_band[2, g])
            @constraint(model, sum(a[f] * x[f, g] for f in 1:F) <= 0)
        end
    end
    for f in 1:F
        isfinite(prob.supply[f]) || continue
        @constraint(model, sum(prob.headcount[g] * x[f, g] for g in 1:G) <= prob.supply[f])
    end
    return model
end

register_variant(
    :diet_problem,
    :standard,
    DietProblem,
    "Least-cost population diet: DRI-based nutrient rows, energy band and guideline share " *
    "limits for many cohorts sharing limited regional food supplies";
    default=true,
)
