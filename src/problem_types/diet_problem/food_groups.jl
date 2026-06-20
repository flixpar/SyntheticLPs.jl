using JuMP
using Random
using Distributions

"""
    FoodGroupsDietProblem <: ProblemGenerator

Generator for diet problems with per-food-group minimum total quantity requirements.

# Overview
Models a realistic diet planning problem in which each food is assigned to a food
group (e.g. grains, proteins, vegetables) and the diet must include at least a
minimum total quantity from each group, in addition to meeting nutrient minimum
requirements within a cost budget and per-food supply limits. The single set of
decision variables is the consumed amount of each food. The objective minimizes
total food cost. Constraints enforce nutrient minimums, per-food supply limits, an
overall cost budget, and a minimum total served quantity for every food group.

# Fields
- `n_foods::Int`: Number of different foods (equals the decision-variable count)
- `n_nutrients::Int`: Number of different nutrients
- `costs::Vector{Float64}`: Cost per unit of each food
- `nutrient_content::Matrix{Float64}`: Nutrient content per unit of food (`n_foods × n_nutrients`)
- `requirements::Vector{Float64}`: Minimum required intake for each nutrient
- `food_supply_limits::Vector{Float64}`: Maximum available amount of each food
- `cost_budget::Float64`: Maximum total food cost
- `n_food_groups::Int`: Number of food groups
- `food_group_assignments::Vector{Int}`: Food-group index for each food
- `min_servings_per_group::Vector{Float64}`: Minimum total served quantity per food group
"""
struct FoodGroupsDietProblem <: ProblemGenerator
    n_foods::Int
    n_nutrients::Int
    costs::Vector{Float64}
    nutrient_content::Matrix{Float64}
    requirements::Vector{Float64}
    food_supply_limits::Vector{Float64}
    cost_budget::Float64
    n_food_groups::Int
    food_group_assignments::Vector{Int}
    min_servings_per_group::Vector{Float64}
end

"""
    FoodGroupsDietProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a food-groups diet problem instance.

The model creates exactly one decision variable per food (`x[1:n_foods]`), so
`n_foods` is set equal to `target_variables`; the variable count therefore matches
the target exactly.

Feasibility handling:
- `feasible`: a concrete baseline diet is constructed first; every constraint
  (nutrient minimums, supply limits, cost budget, and group floors) is then set so
  that this baseline diet satisfies it with margin, guaranteeing a feasible point.
- `infeasible`: the feasible instance is built, then one food group's minimum
  servings is forced above the group's total available supply, a deterministic
  contradiction with a clear margin.
- `unknown`: a natural instance is generated with no forced infeasibility (it is
  built exactly like the feasible case).

# Arguments
- `target_variables`: Target number of variables (equals `n_foods`)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function FoodGroupsDietProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Variable count = n_foods (one consumption variable per food), so n_foods = target.
    n_foods = max(4, target_variables)

    # Scale nutrient count and parameter ranges with problem size.
    if n_foods <= 100
        n_nutrients = rand(5:min(25, max(5, n_foods ÷ 4)))
        cost_range = (rand(0.5:0.1:2.0), rand(3.0:0.5:8.0))
        nutrient_range = (rand(0.05:0.01:0.15), rand(1.5:0.1:3.0))
    elseif n_foods <= 1000
        n_nutrients = rand(15:min(75, max(15, n_foods ÷ 8)))
        cost_range = (rand(0.1:0.05:1.0), rand(2.0:0.5:10.0))
        nutrient_range = (rand(0.01:0.005:0.1), rand(1.0:0.2:4.0))
    else
        n_nutrients = rand(25:min(150, max(25, n_foods ÷ 15)))
        cost_range = (rand(0.05:0.01:0.5), rand(1.0:0.2:15.0))
        nutrient_range = (rand(0.005:0.001:0.05), rand(0.5:0.1:5.0))
    end

    # Basic food data.
    min_cost, max_cost = cost_range
    costs = rand(min_cost:0.1:max_cost, n_foods)

    min_nutrient, max_nutrient = nutrient_range
    # Ensure strictly positive nutrient content so every nutrient is achievable.
    nutrient_content = rand(min_nutrient:0.01:max_nutrient, n_foods, n_nutrients)

    # Food-group assignments: ensure each group has at least one food.
    n_food_groups = rand(4:min(8, max(4, n_foods ÷ 5)))
    food_group_assignments = rand(1:n_food_groups, n_foods)
    # Guarantee every group is non-empty by seeding one food per group.
    perm = randperm(n_foods)
    for g in 1:n_food_groups
        food_group_assignments[perm[g]] = g
    end

    # Resolve the actual status: unknown is treated as a natural (feasible-style)
    # instance with no forced infeasibility.
    actual_status = feasibility_status
    if feasibility_status == unknown
        actual_status = feasible
    end

    # --- Build a concrete baseline diet that everything is sized around ---
    # Rank foods by nutrition per unit cost and allocate a realistic diet.
    nutrition_scores = [sum(nutrient_content[i, :]) / n_nutrients for i in 1:n_foods]
    cost_effectiveness = nutrition_scores ./ costs
    order = sortperm(cost_effectiveness, rev=true)

    baseline_diet = zeros(n_foods)
    primary_count = max(3, round(Int, n_foods * 0.6))
    primary_foods = order[1:primary_count]
    base_consumption = 100.0
    primary_total = base_consumption * 0.75
    primary_eff_sum = sum(cost_effectiveness[primary_foods])
    for i in primary_foods
        w = cost_effectiveness[i] / primary_eff_sum
        baseline_diet[i] = primary_total * w * (0.7 + rand() * 0.6)
    end
    secondary_foods = order[(primary_count + 1):end]
    if !isempty(secondary_foods)
        secondary_total = base_consumption * 0.25
        for i in secondary_foods
            baseline_diet[i] = secondary_total / length(secondary_foods) * (0.5 + rand())
        end
    end
    # Normalize total baseline consumption.
    baseline_diet .*= base_consumption / sum(baseline_diet)
    # Ensure every food has a small strictly-positive baseline so it can absorb
    # group floors and supply headroom is well defined.
    for i in 1:n_foods
        baseline_diet[i] = max(baseline_diet[i], 1e-3)
    end

    # --- Supply limits: generous headroom above the baseline diet ---
    food_supply_limits = [baseline_diet[i] * (2.0 + rand() * 2.0) for i in 1:n_foods]

    # --- Nutrient requirements: below baseline achievement (with tolerance) ---
    requirements = zeros(n_nutrients)
    for j in 1:n_nutrients
        achieved = sum(nutrient_content[i, j] * baseline_diet[i] for i in 1:n_foods)
        # Require 70-90% of what the baseline achieves -> baseline satisfies it.
        requirements[j] = achieved * (0.7 + rand() * 0.2)
    end

    # --- Cost budget: above baseline cost ---
    baseline_cost = sum(costs[i] * baseline_diet[i] for i in 1:n_foods)
    cost_budget = baseline_cost * (1.15 + rand() * 0.35)

    # --- Group floors: below the baseline served quantity per group ---
    # group_baseline[g] = total baseline consumption of foods in group g.
    # FIX: floors are sized off the actual baseline consumption (and capped below
    # group availability) so the baseline diet provably satisfies every floor.
    group_baseline = zeros(n_food_groups)
    group_availability = zeros(n_food_groups)
    for i in 1:n_foods
        g = food_group_assignments[i]
        group_baseline[g] += baseline_diet[i]
        group_availability[g] += food_supply_limits[i]
    end

    min_servings_per_group = zeros(n_food_groups)
    for g in 1:n_food_groups
        # Floor is a fraction of what the baseline already serves from the group;
        # this is automatically <= group availability since baseline <= supply.
        min_servings_per_group[g] = group_baseline[g] * (0.3 + rand() * 0.4)
    end

    # --- Force a deterministic infeasibility if requested ---
    if actual_status == infeasible
        # Require more from one group than the group can ever supply (clear margin).
        target_group = rand(1:n_food_groups)
        min_servings_per_group[target_group] = group_availability[target_group] * (1.5 + rand() * 0.5)
    end

    return FoodGroupsDietProblem(
        n_foods,
        n_nutrients,
        costs,
        nutrient_content,
        requirements,
        food_supply_limits,
        cost_budget,
        n_food_groups,
        food_group_assignments,
        min_servings_per_group,
    )
end

"""
    build_model(prob::FoodGroupsDietProblem)

Build a JuMP model for the food-groups diet problem. Deterministic — uses only the
data stored in the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::FoodGroupsDietProblem)
    model = Model()

    # Decision variables: amount consumed of each food. Count = n_foods.
    @variable(model, x[1:prob.n_foods] >= 0)

    # Objective: minimize total food cost.
    @objective(model, Min, sum(prob.costs[i] * x[i] for i in 1:prob.n_foods))

    # Nutrient minimum requirements.
    for j in 1:prob.n_nutrients
        @constraint(model, sum(prob.nutrient_content[i, j] * x[i] for i in 1:prob.n_foods) >= prob.requirements[j])
    end

    # Per-food supply limits.
    for i in 1:prob.n_foods
        if prob.food_supply_limits[i] < Inf
            @constraint(model, x[i] <= prob.food_supply_limits[i])
        end
    end

    # Overall cost budget.
    if prob.cost_budget < Inf
        @constraint(model, sum(prob.costs[i] * x[i] for i in 1:prob.n_foods) <= prob.cost_budget)
    end

    # Minimum total served quantity per food group.
    for g in 1:prob.n_food_groups
        foods_in_group = [i for i in 1:prob.n_foods if prob.food_group_assignments[i] == g]
        if !isempty(foods_in_group)
            @constraint(model, sum(x[i] for i in foods_in_group) >= prob.min_servings_per_group[g])
        end
    end

    return model
end

# Register the variant
register_variant(
    :diet_problem,
    :food_groups,
    FoodGroupsDietProblem,
    "Diet problem requiring a minimum total quantity from each food group while meeting nutrient needs at minimum cost",
)
