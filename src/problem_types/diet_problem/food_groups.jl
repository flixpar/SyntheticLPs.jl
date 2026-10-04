using JuMP
using Random
using Distributions
using StatsBase

# Storage class of each food category: 1 dry store, 2 refrigerated, 3 frozen.
const _MENU_STORAGE_CLASS = (1, 2, 2, 2, 3, 3, 2, 1, 1, 1, 1, 1, 2)
const MENU_STORAGE_CLASSES = (:dry, :refrigerated, :frozen)
# Storage volume per serving (litres) and daily spoilage rate by category.
const _MENU_VOLUME = (0.15, 0.30, 0.30, 0.25, 0.20, 0.20, 0.08, 0.10, 0.06, 0.03, 0.10, 0.30, 0.35)
const _MENU_DECAY = (0.002, 0.06, 0.05, 0.03, 0.004, 0.004, 0.01, 0.001, 0.002, 0.001, 0.003, 0.004, 0.15)
# Seasonal price amplitude by category (fresh produce swings most).
const _MENU_SEASONALITY = (0.03, 0.25, 0.30, 0.05, 0.08, 0.10, 0.06, 0.03, 0.04, 0.03, 0.03, 0.05, 0.05)
# Daily food-group servings band per 2000-kcal appetite (grains, vegetables,
# fruits, dairy, protein foods), after the US dietary-guideline patterns.
const _MENU_GROUP_BAND = ((3.0, 7.0), (2.0, 5.0), (1.5, 3.5), (2.0, 3.5), (1.5, 4.0))

"""
Reason a requested-infeasible menu-planning instance has no feasible plan.

  - `menu_variety_shortage`: in week `week`, the daily minimum servings of food
    group `group` summed over the week exceed what the group's foods can supply
    under their weekly variety caps and daily portion limits.
  - `menu_energy_squeeze`: on day `day` the protein minimum exceeds the most
    protein any menu can contain within that day's energy ceiling and portion
    limits (an exact one-row fractional knapsack).
"""
@enum MenuInfeasibilityKind begin
    menu_variety_shortage
    menu_energy_squeeze
end

"""
    MenuInfeasibilityCertificate

LP-row infeasibility proof for `FoodGroupsDietProblem` (`group`/`week` for a
variety shortage, `day` for an energy squeeze; unused indices are 0).
`achievable < required` by at least 5%; `menu_certificate_holds` recomputes it.
"""
struct MenuInfeasibilityCertificate
    kind::MenuInfeasibilityKind
    group::Int
    week::Int
    day::Int
    achievable::Float64
    required::Float64
end

"""
    MenuPlan

A complete menu-planning solution, all per person: servings `servings[f, d]`,
weekly purchases `purchases[f, w]` (delivered at the start of week `w`),
same-day retail top-ups `spot[f, d]` and end-of-day inventories
`inventory[f, d]`.
"""
struct MenuPlan
    servings::Matrix{Float64}
    purchases::Matrix{Float64}
    spot::Matrix{Float64}
    inventory::Matrix{Float64}
end

"""
    FoodGroupsDietProblem <: ProblemGenerator

Multi-week institutional menu planning (a school, hospital or care-home
kitchen) with dietary-guideline food-group servings, weekly nutrient targets,
menu-variety rules and a perishable food inventory.

# Formulation

For foods `f`, days `d = 1..D` and weeks `w = 1..W` (`W = ⌈D/7⌉`, deliveries
arrive at the start of each week):

  - `s[f, d] ∈ [0, upper[f]]`: servings per person of food `f` on day `d`;
  - `b[f, w] ≥ 0`: servings per person bought for delivery at the start of
    week `w`;
  - `spot[f, d] ≥ 0`: same-day retail top-up purchases at a markup, on days
    without a delivery (on a delivery day the weekly order dominates them);
  - `I[f, d] ∈ [0, shelf_limit[f]]`: end-of-day stock, in servings per person
    (each item has its own shelf or bin space).

Quantities are per person (the kitchen scales them by `headcount`), which keeps
the balance rows unit-coefficient. Minimize
`headcount · (Σ price[f, w] b[f, w] + Σ markup[f] price[f, w(d)] spot[f, d] +
Σ holding[f] I[f, d])`, where prices follow a
seasonal cycle (fresh produce swings most). Rows:

  - daily: energy band (ranged), protein minimum, sodium ceiling, saturated-fat
    share of energy (homogeneous, mixed signs), and a servings band for every
    dietary food group present (grains, vegetables, fruits, dairy, protein
    foods) — ranged rows over the group's foods;
  - weekly: fiber and micronutrient minimums on the week's total intake (menu
    standards are weekly averages), and a variety cap per food
    `Σ_{d∈w} s[f, d] ≤ variety_cap[f]`;
  - inventory balance per food and day
    `I[f,d] = (1 − decay[f]) I[f,d−1] + [d starts w] b[f,w] + spot[f,d] − s[f,d]`
    with `I[f,0] = 0` (perishables lose `decay` of their stock every day);
  - storage capacity (litres per person) per storage class (dry,
    refrigerated, frozen) and day.

# Sizing

`3 * n_foods * n_days` variables. The food list has `≈ 3√target` items
(13–250); `n_days = round(target / (3 n_foods))` (1–364) and then
`n_foods = round(target / (3 n_days))`.

# Feasibility

  - `feasible`: a guideline-pattern menu is drawn day by day (rotating 1–2
    foods per category), purchased weekly to exactly cover consumption after
    spoilage, and every requirement, band, variety cap and storage capacity is
    set so that plan satisfies it — DRI and guideline values, relaxed only where
    the plan falls short. Stored as a `MenuPlan` witness.
  - `infeasible`: one mutation with a typed `MenuInfeasibilityCertificate` —
    tightened variety caps for one food group (default) or a protein minimum
    above the energy-capped maximum on one day. Both need aggregation over
    several rows; neither is a single-row contradiction.
  - `unknown`: DRI/guideline requirements, rule-of-thumb variety caps and
    nominal storage capacities with no planted point; tight storage or variety
    rules can make it infeasible.
"""
struct FoodGroupsDietProblem <: ProblemGenerator
    n_foods::Int
    n_days::Int
    n_weeks::Int
    headcount::Float64
    food_category::Vector{Int}
    content::Matrix{Float64}
    upper::Vector{Float64}
    price::Matrix{Float64}
    spot_markup::Vector{Float64}
    holding_cost::Vector{Float64}
    decay::Vector{Float64}
    volume::Vector{Float64}
    storage_class::Vector{Int}
    storage_capacity::Matrix{Float64}
    shelf_limit::Vector{Float64}
    variety_cap::Vector{Float64}
    group_band::Array{Float64, 3}
    energy_band::Matrix{Float64}
    protein_min::Vector{Float64}
    sodium_limit::Vector{Float64}
    satfat_share::Vector{Float64}
    weekly_nutrients::Vector{Int}
    weekly_min::Matrix{Float64}
    feasible_witness::Union{Nothing, MenuPlan}
    infeasibility_certificate::Union{Nothing, MenuInfeasibilityCertificate}
    feasibility_status::FeasibilityStatus
end

menu_week_days(n_days::Int, w::Int) = (7 * (w - 1) + 1):min(7 * w, n_days)

function _menu_dimensions(rng::AbstractRNG, target_variables::Int)
    target = max(target_variables, 1)
    nominal = clamp(round(Int, 3.0 * sqrt(target) * rand(rng, Uniform(0.8, 1.25))), 13, 250)
    n_days = clamp(round(Int, target / (3 * nominal)), 1, 364)
    n_weeks = cld(n_days, 7)
    n_foods = max(13, round(Int, target / (3 * n_days)))
    return n_foods, n_days, n_weeks
end

function _menu_group_members(category::Vector{Int})
    members = [Int[] for _ in DIET_FOOD_GROUPS]
    for (f, c) in enumerate(category)
        g = _DIET_CATEGORY_GROUP[c]
        g > 0 && push!(members[g], f)
    end
    return members
end

"""
    menu_plan_satisfies(prob::FoodGroupsDietProblem, plan=prob.feasible_witness; atol=1e-6)

Check a `MenuPlan` against every bound and row of the model without a solver.
"""
function menu_plan_satisfies(
    prob::FoodGroupsDietProblem,
    plan::Union{Nothing, MenuPlan}=prob.feasible_witness;
    atol::Float64=1e-6,
)
    plan === nothing && return false
    F, D, W = prob.n_foods, prob.n_days, prob.n_weeks
    s, b, sp, I = plan.servings, plan.purchases, plan.spot, plan.inventory
    size(s) == (F, D) && size(b) == (F, W) && size(sp) == (F, D) && size(I) == (F, D) || return false
    all(>=(-atol), sp) || return false
    all(d -> (d - 1) % 7 != 0 || all(iszero, view(sp, :, d)), 1:D) || return false
    tol(v) = atol * max(1.0, abs(v))
    all(>=(-atol), s) && all(>=(-atol), b) || return false
    all(>=(-atol), I) || return false
    all(d -> all(f -> I[f, d] <= prob.shelf_limit[f] + tol(prob.shelf_limit[f]), 1:F), 1:D) || return false
    for d in 1:D, f in 1:F
        s[f, d] <= prob.upper[f] + tol(prob.upper[f]) || return false
    end
    members = _menu_group_members(prob.food_category)
    for d in 1:D
        intake = prob.content * view(s, :, d)
        e = intake[DIET_ENERGY]
        prob.energy_band[1, d] - tol(e) <= e <= prob.energy_band[2, d] + tol(e) || return false
        intake[DIET_PROTEIN] + tol(e) >= prob.protein_min[d] || return false
        intake[DIET_SODIUM] <= prob.sodium_limit[d] + tol(intake[DIET_SODIUM]) || return false
        9.0 * intake[DIET_SATFAT] <= prob.satfat_share[d] * e + tol(e) || return false
        for g in eachindex(members)
            isempty(members[g]) && continue
            total = sum(s[f, d] for f in members[g])
            prob.group_band[1, g, d] - tol(total) <= total <= prob.group_band[2, g, d] + tol(total) ||
                return false
        end
    end
    for w in 1:W
        days = menu_week_days(D, w)
        weekly = prob.content * vec(sum(view(s, :, days); dims=2))
        for (r, k) in enumerate(prob.weekly_nutrients)
            weekly[k] + tol(weekly[k]) >= prob.weekly_min[r, w] || return false
        end
        for f in 1:F
            sum(s[f, d] for d in days) <= prob.variety_cap[f] + tol(prob.variety_cap[f]) ||
                return false
        end
    end
    for f in 1:F
        previous = 0.0
        for d in 1:D
            delivery = (d - 1) % 7 == 0 ? b[f, (d - 1) ÷ 7 + 1] : 0.0
            expected = (1 - prob.decay[f]) * previous + delivery + sp[f, d] - s[f, d]
            abs(I[f, d] - expected) <= tol(max(abs(expected), delivery)) || return false
            previous = I[f, d]
        end
    end
    for d in 1:D, c in eachindex(MENU_STORAGE_CLASSES)
        used = sum(
            prob.volume[f] * I[f, d] for f in 1:F if prob.storage_class[f] == c; init=0.0
        )
        used <= prob.storage_capacity[c, d] + tol(prob.storage_capacity[c, d]) || return false
    end
    return true
end

function _menu_group_week_capacity(prob::FoodGroupsDietProblem, group::Int, week::Int)
    days = menu_week_days(prob.n_days, week)
    members = _menu_group_members(prob.food_category)[group]
    return sum(min(prob.variety_cap[f], length(days) * prob.upper[f]) for f in members; init=0.0)
end

"""
    menu_certificate_holds(prob::FoodGroupsDietProblem; rtol=1e-9)

Recompute the stored certificate from the data and check `achievable < required`.
"""
function menu_certificate_holds(prob::FoodGroupsDietProblem; rtol::Float64=1e-9)
    cert = prob.infeasibility_certificate
    cert === nothing && return false
    if cert.kind == menu_variety_shortage
        1 <= cert.group <= length(DIET_FOOD_GROUPS) && 1 <= cert.week <= prob.n_weeks || return false
        achievable = _menu_group_week_capacity(prob, cert.group, cert.week)
        required = sum(prob.group_band[1, cert.group, d] for d in menu_week_days(prob.n_days, cert.week))
    else
        1 <= cert.day <= prob.n_days || return false
        achievable = _diet_max_under_energy_cap(
            prob.content, prob.upper, DIET_PROTEIN, prob.energy_band[2, cert.day]
        )
        required = prob.protein_min[cert.day]
    end
    isapprox(achievable, cert.achievable; rtol=1e-8, atol=1e-9) || return false
    isapprox(required, cert.required; rtol=1e-8, atol=1e-9) || return false
    return achievable < required * (1 - rtol)
end

"""
    FoodGroupsDietProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a menu-planning instance (see `FoodGroupsDietProblem`).
"""
function FoodGroupsDietProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    F, D, W = _menu_dimensions(rng, target_variables)

    table = _diet_sample_food_table(rng, F)
    content = table.content
    category = table.category
    by_category = _diet_foods_by_category(category)
    members = _menu_group_members(category)

    demo = DIET_DEMOGRAPHICS[rand(rng, eachindex(DIET_DEMOGRAPHICS))]
    appetite = demo.eer / 2000.0 * rand(rng, Uniform(0.95, 1.08))
    eer = 2000.0 * appetite
    headcount = Float64(max(20, round(Int, rand(rng, LogNormal(log(300.0), 0.7)))))
    upper = table.max_servings .* appetite

    storage_class = [_MENU_STORAGE_CLASS[c] for c in category]
    volume = [_MENU_VOLUME[c] * rand(rng, LogNormal(0.0, 0.25)) for c in category]
    decay = [_MENU_DECAY[c] * rand(rng, Uniform(0.7, 1.3)) for c in category]
    holding_cost = [table.cost[f] * (0.002 + 0.5 * decay[f]) for f in 1:F]
    phase = rand(rng, Uniform(0.0, 2pi), length(DIET_FOOD_CATEGORIES))
    start_week = rand(rng, 0:51)
    price = [
        table.cost[f] *
        (1 + _MENU_SEASONALITY[category[f]] * sin(2pi * (start_week + w) / 52 + phase[category[f]])) *
        rand(rng, LogNormal(0.0, 0.05)) for f in 1:F, w in 1:W
    ]
    # Same-day retail top-ups cost 30-80% more than the weekly delivery.
    spot_markup = rand(rng, Uniform(1.3, 1.8), F)

    n_micro = rand(rng, 5:length(DIET_MICRONUTRIENTS))
    micros = sort(sample(rng, collect(DIET_MICRONUTRIENTS), n_micro; replace=false))
    weekly_nutrients = vcat([DIET_FIBER], micros)
    weekly_reference = [diet_reference_minimum(demo, k) for k in weekly_nutrients]

    energy_band = zeros(Float64, 2, D)
    protein_min = zeros(Float64, D)
    sodium_limit = fill(demo.sodium, D)
    satfat_share = fill(DIET_SATFAT_SHARE, D)
    group_band = zeros(Float64, 2, length(DIET_FOOD_GROUPS), D)
    for d in 1:D, g in eachindex(DIET_FOOD_GROUPS)
        group_band[1, g, d] = _MENU_GROUP_BAND[g][1] * appetite
        group_band[2, g, d] = _MENU_GROUP_BAND[g][2] * appetite
    end
    weekly_min = zeros(Float64, length(weekly_nutrients), W)
    storage_capacity = zeros(Float64, length(MENU_STORAGE_CLASSES), D)
    shelf_limit = zeros(Float64, F)
    # Menu rule: no food more than 2-4 typical servings a week.
    variety_cap = [rand(rng, Uniform(2.0, 4.0)) * min(1.0, upper[f]) for f in 1:F]
    witness = nothing

    if feasibility_status == unknown
        for d in 1:D
            energy_band[1, d] = DIET_ENERGY_BAND[1] * eer
            energy_band[2, d] = DIET_ENERGY_BAND[2] * eer
            protein_min[d] = demo.protein
        end
        for w in 1:W
            n = length(menu_week_days(D, w))
            weekly_min[:, w] .= n .* weekly_reference .* rand(rng, Uniform(0.85, 1.0))
        end
        # Planners calibrate the variety rule to the size of each food group's
        # list: a group's foods share its weekly minimum servings with a
        # 0.75-2.0x allowance, so a short list under a strict rule may not cover it.
        for g in eachindex(members), f in members[g]
            share = 7.0 * group_band[1, g, 1] / length(members[g])
            variety_cap[f] = share * rand(rng, Uniform(0.75, 2.0))
        end
        # Storage sized against the average stock of a weekly delivery cycle.
        for c in eachindex(MENU_STORAGE_CLASSES)
            nominal = 0.0
            for cat in eachindex(DIET_FOOD_CATEGORIES)
                _MENU_STORAGE_CLASS[cat] == c || continue
                nominal += _DIET_PATTERN_SERVINGS[cat] * appetite * _MENU_VOLUME[cat]
            end
            storage_capacity[c, :] .= nominal * 6.0 * rand(rng, Uniform(0.3, 1.2))
        end
        # Bin space per item: around a week of its category's pattern share.
        for f in 1:F
            c = category[f]
            weekly = 7.0 * _DIET_PATTERN_SERVINGS[c] * appetite / length(by_category[c])
            shelf_limit[f] = weekly * rand(rng, Uniform(0.8, 2.0))
        end
    else
        s = zeros(Float64, F, D)
        for d in 1:D
            s[:, d] .= _diet_pattern_diet(rng, table, appetite, by_category; choices=1:2)
            s[:, d] .= min.(view(s, :, d), 0.95 .* upper)
        end
        # Variety caps admit the plan; weekly purchases cover each week exactly
        # after spoilage (I = 0 at the end of a week, up to rounding).
        b = zeros(Float64, F, W)
        I = zeros(Float64, F, D)
        for w in 1:W
            days = menu_week_days(D, w)
            for f in 1:F
                weekly = sum(s[f, d] for d in days)
                variety_cap[f] = max(variety_cap[f], weekly * rand(rng, Uniform(1.02, 1.15)))
                need = sum(
                    s[f, d] / (1 - decay[f])^(d - first(days)) for d in days
                )
                b[f, w] = need * (1 + 1e-7)
            end
        end
        for f in 1:F
            previous = 0.0
            for d in 1:D
                delivery = (d - 1) % 7 == 0 ? b[f, (d - 1) ÷ 7 + 1] : 0.0
                I[f, d] = max(0.0, (1 - decay[f]) * previous + delivery - s[f, d])
                previous = I[f, d]
            end
        end
        for d in 1:D
            intake = content * view(s, :, d)
            e = intake[DIET_ENERGY]
            energy_band[1, d] = min(DIET_ENERGY_BAND[1] * eer, 0.97 * e)
            energy_band[2, d] = max(DIET_ENERGY_BAND[2] * eer, 1.03 * e)
            protein_min[d] = min(demo.protein, intake[DIET_PROTEIN] * rand(rng, Uniform(0.92, 0.99)))
            sodium_limit[d] = max(demo.sodium, intake[DIET_SODIUM] * rand(rng, Uniform(1.01, 1.06)))
            satfat_share[d] = max(DIET_SATFAT_SHARE, 9.0 * intake[DIET_SATFAT] / e + 0.01)
            for g in eachindex(members)
                total = sum(s[f, d] for f in members[g]; init=0.0)
                group_band[1, g, d] = min(group_band[1, g, d], 0.97 * total)
                group_band[2, g, d] = max(group_band[2, g, d], 1.03 * total)
            end
            for c in eachindex(MENU_STORAGE_CLASSES)
                used = sum(volume[f] * I[f, d] for f in 1:F if storage_class[f] == c; init=0.0)
                storage_capacity[c, d] = used
            end
        end
        for c in eachindex(MENU_STORAGE_CLASSES)
            peak = maximum(view(storage_capacity, c, :))
            storage_capacity[c, :] .= max(peak, 1.0) * rand(rng, Uniform(1.05, 1.3))
        end
        for f in 1:F
            shelf_limit[f] = maximum(view(I, f, :)) * rand(rng, Uniform(1.1, 1.6)) + 0.1 * upper[f]
        end
        for w in 1:W
            days = menu_week_days(D, w)
            weekly = content * vec(sum(view(s, :, days); dims=2))
            for (r, k) in enumerate(weekly_nutrients)
                weekly_min[r, w] = min(
                    length(days) * weekly_reference[r], weekly[k] * rand(rng, Uniform(0.9, 0.98))
                )
            end
        end
        witness = MenuPlan(s, b, zeros(Float64, F, D), I)
    end

    certificate = nothing
    if feasibility_status == infeasible
        present = [g for g in eachindex(members) if !isempty(members[g])]
        if rand(rng) < 0.6 && !isempty(present)
            # The menu committee's variety rule is tightened for one food group
            # until a week's group minimum can no longer be served.
            g = rand(rng, present)
            w = rand(rng, 1:W)
            days = menu_week_days(D, w)
            required = sum(group_band[1, g, d] for d in days)
            capacity = [min(variety_cap[f], length(days) * upper[f]) for f in members[g]]
            theta = required / rand(rng, Uniform(1.08, 1.25)) / sum(capacity)
            for (j, f) in enumerate(members[g])
                variety_cap[f] = theta * capacity[j]
            end
            tmp = FoodGroupsDietProblem(
                F, D, W, headcount, category, content, upper, price, spot_markup, holding_cost, decay,
                volume, storage_class, storage_capacity, shelf_limit, variety_cap, group_band, energy_band,
                protein_min, sodium_limit, satfat_share, weekly_nutrients, weekly_min, nothing,
                nothing, feasibility_status,
            )
            achievable = _menu_group_week_capacity(tmp, g, w)
            certificate = MenuInfeasibilityCertificate(
                menu_variety_shortage, g, w, 0, achievable, required
            )
        else
            d = rand(rng, 1:D)
            achievable = _diet_max_under_energy_cap(content, upper, DIET_PROTEIN, energy_band[2, d])
            protein_min[d] = achievable * rand(rng, Uniform(1.06, 1.15))
            certificate = MenuInfeasibilityCertificate(
                menu_energy_squeeze, 0, 0, d, achievable, protein_min[d]
            )
        end
        witness = nothing
    end

    prob = FoodGroupsDietProblem(
        F, D, W, headcount, category, content, upper, price, spot_markup, holding_cost, decay, volume,
        storage_class, storage_capacity, shelf_limit, variety_cap, group_band, energy_band, protein_min,
        sodium_limit, satfat_share, weekly_nutrients, weekly_min, witness, certificate,
        feasibility_status,
    )
    feasibility_status == feasible && @assert menu_plan_satisfies(prob)
    feasibility_status == infeasible && @assert menu_certificate_holds(prob)
    return prob
end

"""
    build_model(prob::FoodGroupsDietProblem)

Build the menu-planning LP (deterministic; see `FoodGroupsDietProblem`).
"""
function build_model(prob::FoodGroupsDietProblem)
    model = Model()
    F, D, W = prob.n_foods, prob.n_days, prob.n_weeks
    C = prob.content
    @variable(model, 0 <= s[f=1:F, d=1:D] <= prob.upper[f])
    @variable(model, b[1:F, 1:W] >= 0)
    topup_days = [d for d in 1:D if (d - 1) % 7 != 0]
    @variable(model, spot[1:F, topup_days] >= 0)
    @variable(model, 0 <= I[f=1:F, d=1:D] <= prob.shelf_limit[f])
    @objective(
        model,
        Min,
        prob.headcount * (
            sum(prob.price[f, w] * b[f, w] for f in 1:F, w in 1:W) +
            sum(prob.spot_markup[f] * prob.price[f, (d - 1) ÷ 7 + 1] * spot[f, d] for f in 1:F, d in topup_days; init=0.0) +
            sum(prob.holding_cost[f] * I[f, d] for f in 1:F, d in 1:D)
        )
    )

    carriers = [findall(>(0.0), view(C, k, :)) for k in eachindex(DIET_NUTRIENTS)]
    members = _menu_group_members(prob.food_category)
    for d in 1:D
        @constraint(
            model,
            prob.energy_band[1, d] <= sum(C[DIET_ENERGY, f] * s[f, d] for f in 1:F) <= prob.energy_band[2, d]
        )
        @constraint(model, sum(C[DIET_PROTEIN, f] * s[f, d] for f in carriers[DIET_PROTEIN]) >= prob.protein_min[d])
        @constraint(model, sum(C[DIET_SODIUM, f] * s[f, d] for f in carriers[DIET_SODIUM]) <= prob.sodium_limit[d])
        @constraint(
            model,
            sum((9.0 * C[DIET_SATFAT, f] - prob.satfat_share[d] * C[DIET_ENERGY, f]) * s[f, d] for f in 1:F) <= 0
        )
        for g in eachindex(members)
            isempty(members[g]) && continue
            @constraint(
                model,
                prob.group_band[1, g, d] <= sum(s[f, d] for f in members[g]) <= prob.group_band[2, g, d]
            )
        end
    end
    for w in 1:W
        days = menu_week_days(D, w)
        for (r, k) in enumerate(prob.weekly_nutrients)
            @constraint(model, sum(C[k, f] * s[f, d] for f in carriers[k], d in days) >= prob.weekly_min[r, w])
        end
        for f in 1:F
            @constraint(model, sum(s[f, d] for d in days) <= prob.variety_cap[f])
        end
    end
    for f in 1:F, d in 1:D
        delivery = (d - 1) % 7 == 0 ? b[f, (d - 1) ÷ 7 + 1] : spot[f, d]
        if d == 1
            @constraint(model, I[f, d] == delivery - s[f, d])
        else
            @constraint(model, I[f, d] == (1 - prob.decay[f]) * I[f, d - 1] + delivery - s[f, d])
        end
    end
    for d in 1:D, c in eachindex(MENU_STORAGE_CLASSES)
        foods = [f for f in 1:F if prob.storage_class[f] == c]
        isempty(foods) && continue
        @constraint(model, sum(prob.volume[f] * I[f, d] for f in foods) <= prob.storage_capacity[c, d])
    end
    return model
end

register_variant(
    :diet_problem,
    :food_groups,
    FoodGroupsDietProblem,
    "Multi-week menu planning with daily food-group servings bands, weekly nutrient " *
    "targets, variety caps, and a perishable weekly-delivery inventory with storage limits",
)
