using JuMP
using Random
using Distributions

"""
Diet problem variants.

# Variants
- `diet_standard`: Basic diet - minimize cost meeting nutrient requirements
- `diet_nutrient_bounds`: Both minimum and maximum nutrient limits (sodium, fat caps)
- `diet_food_groups`: Minimum servings from each food group required
- `diet_variety`: Minimum number of different foods must be used
- `diet_macro_ratios`: Specific protein/carb/fat ratio requirements
- `diet_calorie_range`: Both minimum and maximum calorie constraints
- `diet_allergen_free`: Certain foods completely excluded
- `diet_meal_plan`: Structured breakfast/lunch/dinner composition
"""
@enum DietVariant begin
    diet_standard
    diet_nutrient_bounds
    diet_food_groups
    diet_variety
    diet_macro_ratios
    diet_calorie_range
    diet_allergen_free
    diet_meal_plan
end

"""
    DietProblem <: ProblemGenerator

Generator for diet problems that minimize the cost of food while meeting nutritional requirements.

This problem models realistic diet optimization with:
- Multiple foods with varying costs and nutrient content
- Nutrient minimum requirements
- Food supply availability limits
- Total cost budget constraint
- Minimum and maximum consumption constraints for specific foods

# Fields
All data generated in constructor based on target_variables and feasibility_status:
- `n_foods::Int`: Number of different foods
- `n_nutrients::Int`: Number of different nutrients
- `costs::Vector{Float64}`: Cost per unit of each food
- `nutrient_content::Matrix{Float64}`: Nutrient content per unit of food
- `requirements::Vector{Float64}`: Minimum nutrient requirements
- `food_supply_limits::Vector{Float64}`: Supply limit for each food
- `cost_budget::Float64`: Total cost budget
- `min_food_amounts::Dict{Int, Float64}`: Minimum consumption requirements
- `max_food_amounts::Dict{Int, Float64}`: Maximum consumption limits
"""
struct DietProblem <: ProblemGenerator
    n_foods::Int
    n_nutrients::Int
    costs::Vector{Float64}
    nutrient_content::Matrix{Float64}
    requirements::Vector{Float64}
    food_supply_limits::Vector{Float64}
    cost_budget::Float64
    min_food_amounts::Dict{Int, Float64}
    max_food_amounts::Dict{Int, Float64}
    variant::DietVariant
    # Nutrient bounds variant
    nutrient_upper_bounds::Union{Vector{Float64}, Nothing}
    # Food groups variant
    n_food_groups::Int
    food_group_assignments::Union{Vector{Int}, Nothing}
    min_servings_per_group::Union{Vector{Float64}, Nothing}
    # Variety variant
    min_foods_used::Int
    use_threshold::Float64
    # Macro ratios variant
    protein_idx::Int
    carb_idx::Int
    fat_idx::Int
    min_protein_ratio::Float64
    max_protein_ratio::Float64
    min_carb_ratio::Float64
    max_carb_ratio::Float64
    min_fat_ratio::Float64
    max_fat_ratio::Float64
    # Calorie range variant
    calorie_idx::Int
    min_calories::Float64
    max_calories::Float64
    # Allergen free variant
    excluded_foods::Union{Set{Int}, Nothing}
    # Meal plan variant
    n_meals::Int
    food_meal_compat::Union{Matrix{Bool}, Nothing}
    meal_calorie_targets::Union{Vector{Float64}, Nothing}
    meal_calorie_tolerance::Float64
end

# Backwards compatibility constructor
function DietProblem(n_foods::Int, n_nutrients::Int, costs::Vector{Float64},
                     nutrient_content::Matrix{Float64}, requirements::Vector{Float64},
                     food_supply_limits::Vector{Float64}, cost_budget::Float64,
                     min_food_amounts::Dict{Int, Float64}, max_food_amounts::Dict{Int, Float64})
    DietProblem(
        n_foods, n_nutrients, costs, nutrient_content, requirements,
        food_supply_limits, cost_budget, min_food_amounts, max_food_amounts,
        diet_standard,
        nothing, 0, nothing, nothing,
        0, 0.0,
        0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0, 0.0, 0.0,
        nothing,
        0, nothing, nothing, 0.0
    )
end

"""
    DietProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                variant::DietVariant=diet_standard)

Construct a diet problem instance with sophisticated verified impossibility scenarios.

# Sophisticated Feasibility Logic Preserved:
For FEASIBLE instances:
- **Baseline diet construction**: Uses cost-effectiveness optimization to find realistic diet
- **Nutrition-cost optimization**: Ranks foods by nutrition per unit cost
- **Challenging constraints**: Sets tight tolerances (2-12%) around baseline achievement
- **Supply pressure**: Creates realistic market constraints (seasonal, market, normal)
- **Budget pressure**: Tight (105-115%), moderate (110-125%), or generous budgets

For INFEASIBLE instances (4 verified impossibility scenarios):
1. **Verified nutrient impossibility**: Calculates true maximum achievable nutrient, sets requirement above it
2. **Verified budget impossibility**: Calculates proven minimum cost needed, sets budget below it
3. **Verified supply shortage**: Strategically reduces supply until target nutrient becomes impossible
4. **Verified over-constrained system**: Multiple individually-reasonable constraints that together are impossible

FINAL VERIFICATION: For all infeasible instances, calculates absolute maximum achievable and forces
requirement to 200-300% of maximum with large margin to avoid numerical issues.

# Arguments
- `target_variables`: Target number of variables (n_foods)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
- `variant`: Diet problem variant (default: diet_standard)
"""
function DietProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                     variant::DietVariant=diet_standard)
    Random.seed!(seed)

    n_foods = target_variables

    # Scale nutrients based on problem size
    if target_variables <= 100
        n_nutrients = rand(5:min(25, max(5, Int(target_variables ÷ 4))))
        cost_range = (rand(0.5:0.1:2.0), rand(3.0:0.5:8.0))
        nutrient_range = (rand(0.05:0.01:0.15), rand(1.5:0.1:3.0))
    elseif target_variables <= 1000
        n_nutrients = rand(15:min(75, max(15, Int(target_variables ÷ 8))))
        cost_range = (rand(0.1:0.05:1.0), rand(2.0:0.5:10.0))
        nutrient_range = (rand(0.01:0.005:0.1), rand(1.0:0.2:4.0))
    else
        n_nutrients = rand(25:min(150, max(25, Int(target_variables ÷ 15))))
        cost_range = (rand(0.05:0.01:0.5), rand(1.0:0.2:15.0))
        nutrient_range = (rand(0.005:0.001:0.05), rand(0.5:0.1:5.0))
    end

    # Generate basic food data
    min_cost, max_cost = cost_range
    c = rand(min_cost:0.1:max_cost, n_foods)

    min_nutrient, max_nutrient = nutrient_range
    a = rand(min_nutrient:0.1:max_nutrient, n_foods, n_nutrients)

    # Initialize constraint variables
    b = zeros(n_nutrients)
    food_supply_limits = fill(Inf, n_foods)
    cost_budget = Inf
    min_food_amounts = Dict{Int, Float64}()
    max_food_amounts = Dict{Int, Float64}()

    # Determine actual feasibility status
    actual_status = feasibility_status
    if feasibility_status == unknown
        actual_status = rand() < 0.75 ? feasible : infeasible
    end

    if actual_status == feasible
        # SOPHISTICATED FEASIBLE APPROACH: Create challenging but feasible constraints

        # Step 1: Calculate nutrition efficiency
        nutrition_scores = zeros(n_foods)
        for i in 1:n_foods
            nutrition_scores[i] = sum(a[i, :]) / n_nutrients
        end

        cost_effectiveness = nutrition_scores ./ c
        effectiveness_order = sortperm(cost_effectiveness, rev=true)

        # Step 2: Generate realistic baseline diet
        baseline_diet = zeros(n_foods)

        primary_count = max(3, round(Int, n_foods * 0.6))
        primary_foods = effectiveness_order[1:primary_count]

        base_consumption = 100.0
        primary_total = base_consumption * 0.75

        for i in primary_foods
            effectiveness_weight = cost_effectiveness[i] / sum(cost_effectiveness[primary_foods])
            baseline_amount = primary_total * effectiveness_weight
            baseline_diet[i] = baseline_amount * (0.7 + rand() * 0.6)
        end

        secondary_foods = effectiveness_order[(primary_count + 1):end]
        if !isempty(secondary_foods)
            secondary_total = base_consumption * 0.25
            for i in secondary_foods
                baseline_diet[i] = secondary_total / length(secondary_foods) * (0.5 + rand())
            end
        end

        total_baseline = sum(baseline_diet)
        baseline_diet .*= base_consumption / total_baseline

        # Step 3: Calculate achieved nutrient levels
        achieved_nutrients = zeros(n_nutrients)
        for j in 1:n_nutrients
            achieved_nutrients[j] = sum(a[i, j] * baseline_diet[i] for i in 1:n_foods)
        end

        # Step 4: Set challenging nutrient requirements
        tolerance_scenario = rand(1:3)
        tolerance_level = if tolerance_scenario == 1
            0.02 + rand() * 0.03  # 2-5% tolerance
        elseif tolerance_scenario == 2
            0.05 + rand() * 0.05  # 5-10% tolerance
        else
            0.08 + rand() * 0.04  # 8-12% tolerance
        end

        for j in 1:n_nutrients
            tolerance = tolerance_level
            position_in_band = 0.7 + rand() * 0.15

            total_range = 2 * tolerance * achieved_nutrients[j] / (1 - 2 * tolerance + 2 * tolerance * position_in_band)
            lower_bound = achieved_nutrients[j] - total_range * position_in_band

            b[j] = max(0.0, lower_bound)
        end

        # Step 5: Add realistic supply constraints
        supply_scenario = rand(1:3)
        if supply_scenario == 1
            # Seasonal availability
            critical_foods = primary_foods[1:max(2, div(length(primary_foods), 3))]
            for i in 1:n_foods
                if i in critical_foods
                    food_supply_limits[i] = baseline_diet[i] * (1.1 + rand() * 0.3)
                else
                    food_supply_limits[i] = baseline_diet[i] * (1.5 + rand())
                end
            end
        elseif supply_scenario == 2
            # Market supply
            expensive_foods = sortperm(c, rev=true)[1:max(2, div(n_foods, 4))]
            for i in 1:n_foods
                if i in expensive_foods
                    food_supply_limits[i] = baseline_diet[i] * (1.2 + rand() * 0.4)
                else
                    food_supply_limits[i] = baseline_diet[i] * (2.0 + rand() * 2.0)
                end
            end
        else
            # Normal supply
            for i in 1:n_foods
                food_supply_limits[i] = baseline_diet[i] * (3.0 + rand() * 2.0)
            end
        end

        # Step 6: Set challenging cost budget
        baseline_cost = sum(c[i] * baseline_diet[i] for i in 1:n_foods)
        cost_pressure = rand(1:3)
        if cost_pressure == 1
            cost_budget = baseline_cost * (1.05 + rand() * 0.10)
        elseif cost_pressure == 2
            cost_budget = baseline_cost * (1.10 + rand() * 0.15)
        else
            cost_budget = baseline_cost * (1.5 + rand() * 0.5)
        end

        # Step 7: Add realistic consumption preferences
        if rand() < 0.7
            preferred_foods = randperm(n_foods)[1:max(1, div(n_foods, 6))]
            for i in preferred_foods
                min_food_amounts[i] = baseline_diet[i] * (0.6 + rand() * 0.3)
            end

            limited_foods = randperm(n_foods)[1:max(1, div(n_foods, 5))]
            for i in limited_foods
                max_food_amounts[i] = baseline_diet[i] * (1.3 + rand() * 0.4)
            end
        end

    else  # :infeasible - Create verified mathematical impossibilities

        scenario = rand(1:4)

        if scenario == 1
            # SCENARIO 1: Verified nutrient impossibility conflict
            base_supply = 100.0
            for i in 1:n_foods
                food_supply_limits[i] = base_supply * (0.5 + rand() * 1.5)
            end

            max_achievable_nutrients = zeros(n_nutrients)
            for j in 1:n_nutrients
                max_achievable_nutrients[j] = sum(a[i, j] * food_supply_limits[i] for i in 1:n_foods)
            end

            target_nutrient = rand(1:n_nutrients)
            b[target_nutrient] = max_achievable_nutrients[target_nutrient] * (1.2 + rand() * 0.3)

            for j in 1:n_nutrients
                if j != target_nutrient
                    b[j] = max_achievable_nutrients[j] * (0.3 + rand() * 0.4)
                end
            end

            final_max_achievable = sum(a[i, target_nutrient] * food_supply_limits[i] for i in 1:n_foods)
            if final_max_achievable >= b[target_nutrient]
                b[target_nutrient] = final_max_achievable * 1.15
            end

            cost_budget = sum(c[i] * food_supply_limits[i] for i in 1:n_foods)

        elseif scenario == 2
            # SCENARIO 2: Verified budget impossibility conflict
            for i in 1:n_foods
                food_supply_limits[i] = 500.0
            end

            for j in 1:n_nutrients
                best_content = maximum(a[:, j])
                target_units = 20.0 + rand() * 30.0
                b[j] = best_content * target_units
            end

            proven_min_cost = 0.0
            for j in 1:n_nutrients
                best_cost_efficiency = Inf
                for i in 1:n_foods
                    if a[i, j] > 0
                        cost_per_nutrient_unit = c[i] / a[i, j]
                        best_cost_efficiency = min(best_cost_efficiency, cost_per_nutrient_unit)
                    end
                end

                if best_cost_efficiency < Inf
                    proven_min_cost += b[j] * best_cost_efficiency
                end
            end

            if proven_min_cost > 0
                cost_budget = proven_min_cost * (0.7 + rand() * 0.2)
            else
                avg_cost = sum(c) / n_foods
                cost_budget = avg_cost * 5.0
            end

            verification_min_cost = 0.0
            for j in 1:n_nutrients
                cheapest_cost_for_nutrient = Inf
                for i in 1:n_foods
                    if a[i, j] > 0
                        cost_for_requirement = (b[j] / a[i, j]) * c[i]
                        cheapest_cost_for_nutrient = min(cheapest_cost_for_nutrient, cost_for_requirement)
                    end
                end
                if cheapest_cost_for_nutrient < Inf
                    verification_min_cost += cheapest_cost_for_nutrient
                end
            end

            if verification_min_cost > 0 && cost_budget >= verification_min_cost * 0.95
                cost_budget = verification_min_cost * 0.8
            end

        elseif scenario == 3
            # SCENARIO 3: Verified supply shortage conflict
            for j in 1:n_nutrients
                best_content = maximum(a[:, j])
                target_units = 30.0 + rand() * 30.0
                b[j] = best_content * target_units
            end

            base_supply = 200.0
            for i in 1:n_foods
                food_supply_limits[i] = base_supply * (0.8 + rand() * 0.4)
            end

            target_nutrient = rand(1:n_nutrients)
            current_max = sum(a[i, target_nutrient] * food_supply_limits[i] for i in 1:n_foods)

            nutrient_contributions = [(a[i, target_nutrient] * food_supply_limits[i], i) for i in 1:n_foods]
            sort!(nutrient_contributions, rev=true)

            reduction_needed = current_max - b[target_nutrient] * 0.95
            remaining_reduction = reduction_needed

            for (contribution, food_idx) in nutrient_contributions
                if remaining_reduction > 0
                    max_reduction = food_supply_limits[food_idx] * 0.9
                    actual_reduction = min(remaining_reduction / a[food_idx, target_nutrient], max_reduction)

                    new_supply = max(10.0, food_supply_limits[food_idx] - actual_reduction)
                    reduction_achieved = (food_supply_limits[food_idx] - new_supply) * a[food_idx, target_nutrient]

                    food_supply_limits[food_idx] = new_supply
                    remaining_reduction -= reduction_achieved

                    if remaining_reduction <= 0
                        break
                    end
                end
            end

            final_max = sum(a[i, target_nutrient] * food_supply_limits[i] for i in 1:n_foods)
            if final_max >= b[target_nutrient] * 0.99
                b[target_nutrient] = final_max * (1.1 + rand() * 0.1)
            end

            cost_budget = sum(c[i] * food_supply_limits[i] for i in 1:n_foods) * 1.5

        else  # scenario == 4
            # SCENARIO 4: Verified over-constrained system
            baseline_consumption = 100.0
            baseline_diet = zeros(n_foods)

            nutrition_scores = [sum(a[i, :]) for i in 1:n_foods]
            cost_effectiveness = nutrition_scores ./ c
            total_effectiveness = sum(cost_effectiveness)

            for i in 1:n_foods
                base_share = cost_effectiveness[i] / total_effectiveness
                baseline_diet[i] = baseline_consumption * base_share * (0.5 + rand())
            end

            total_baseline = sum(baseline_diet)
            baseline_diet .*= baseline_consumption / total_baseline

            for j in 1:n_nutrients
                baseline_achievement = sum(a[i, j] * baseline_diet[i] for i in 1:n_foods)
                b[j] = baseline_achievement * (1.1 + rand() * 0.2)
            end

            for i in 1:n_foods
                food_supply_limits[i] = baseline_diet[i] * (1.2 + rand() * 0.6)
            end

            baseline_cost = sum(c[i] * baseline_diet[i] for i in 1:n_foods)
            cost_budget = baseline_cost * (1.1 + rand() * 0.2)

            expensive_foods = sortperm(c, rev=true)[1:max(2, div(n_foods, 5))]
            num_required = max(1, div(length(expensive_foods), 2))
            required_foods = expensive_foods[1:num_required]

            for i in required_foods
                min_food_amounts[i] = baseline_diet[i] * (1.3 + rand() * 0.4)
            end

            nutritious_foods = []
            avg_nutrition = sum(nutrition_scores) / n_foods
            for i in 1:n_foods
                if nutrition_scores[i] > avg_nutrition * 1.2
                    push!(nutritious_foods, i)
                end
            end

            if !isempty(nutritious_foods)
                num_restricted = max(1, div(length(nutritious_foods), 3))
                restricted_foods = nutritious_foods[1:min(num_restricted, length(nutritious_foods))]

                for i in restricted_foods
                    max_food_amounts[i] = baseline_diet[i] * (0.8 + rand() * 0.3)
                end
            end

            # Force mathematical impossibility
            target_nutrient = rand(1:n_nutrients)

            max_achievable_target = 0.0
            for i in 1:n_foods
                min_usage = get(min_food_amounts, i, 0.0)
                max_usage = food_supply_limits[i]

                if haskey(max_food_amounts, i)
                    max_usage = min(max_usage, max_food_amounts[i])
                end

                feasible_max = max(0.0, min(max_usage, max(min_usage, max_usage)))
                max_achievable_target += a[i, target_nutrient] * feasible_max
            end

            b[target_nutrient] = max_achievable_target * (1.2 + rand() * 0.2)

            for j in 1:n_nutrients
                if j != target_nutrient
                    max_achievable_j = 0.0
                    for i in 1:n_foods
                        min_usage = get(min_food_amounts, i, 0.0)
                        max_usage = min(food_supply_limits[i], get(max_food_amounts, i, food_supply_limits[i]))
                        feasible_max = max(0.0, min(max_usage, max(min_usage, max_usage)))
                        max_achievable_j += a[i, j] * feasible_max
                    end
                    b[j] = max_achievable_j * (0.7 + rand() * 0.2)
                end
            end
        end
    end

    # FINAL VERIFICATION: Guarantee infeasibility for infeasible instances
    if actual_status == infeasible
        verified_max_achievable = zeros(n_nutrients)

        for j in 1:n_nutrients
            max_possible_j = 0.0

            for i in 1:n_foods
                min_usage_i = get(min_food_amounts, i, 0.0)
                max_usage_i = food_supply_limits[i]

                if haskey(max_food_amounts, i)
                    max_usage_i = min(max_usage_i, max_food_amounts[i])
                end

                if min_usage_i > max_usage_i
                    feasible_usage = 0.0
                else
                    feasible_usage = max_usage_i
                end

                max_possible_j += a[i, j] * feasible_usage
            end

            verified_max_achievable[j] = max_possible_j
        end

        worst_violation_ratio = 0.0
        target_nutrient_final = 1

        for j in 1:n_nutrients
            if verified_max_achievable[j] > 0
                violation_ratio = b[j] / verified_max_achievable[j]
                if violation_ratio > worst_violation_ratio
                    worst_violation_ratio = violation_ratio
                    target_nutrient_final = j
                end
            end
        end

        if verified_max_achievable[target_nutrient_final] > 0
            b[target_nutrient_final] = verified_max_achievable[target_nutrient_final] * (2.0 + rand())
        else
            b[target_nutrient_final] = 100.0 + rand() * 100.0
        end
    end

    # Initialize variant-specific fields
    nutrient_upper_bounds = nothing
    n_food_groups = 0
    food_group_assignments = nothing
    min_servings_per_group = nothing
    min_foods_used = 0
    use_threshold = 0.0
    protein_idx = 0
    carb_idx = 0
    fat_idx = 0
    min_protein_ratio = 0.0
    max_protein_ratio = 0.0
    min_carb_ratio = 0.0
    max_carb_ratio = 0.0
    min_fat_ratio = 0.0
    max_fat_ratio = 0.0
    calorie_idx = 0
    min_calories = 0.0
    max_calories = 0.0
    excluded_foods = nothing
    n_meals = 0
    food_meal_compat = nothing
    meal_calorie_targets = nothing
    meal_calorie_tolerance = 0.0

    # Generate variant-specific data
    if variant == diet_nutrient_bounds
        # Both min and max nutrient constraints (like sodium and saturated fat limits)
        nutrient_upper_bounds = zeros(n_nutrients)
        for j in 1:n_nutrients
            # Upper bound is 1.5-3x the minimum requirement
            nutrient_upper_bounds[j] = b[j] * rand(Uniform(1.5, 3.0))
        end
        if feasibility_status == infeasible
            # Make some upper bounds below minimum requirements
            violated_nutrient = rand(1:n_nutrients)
            nutrient_upper_bounds[violated_nutrient] = b[violated_nutrient] * 0.8
        end

    elseif variant == diet_food_groups
        # Foods assigned to groups, minimum servings from each group
        n_food_groups = rand(4:min(8, max(4, n_foods ÷ 5)))
        food_group_assignments = rand(1:n_food_groups, n_foods)

        # Calculate how much is available per group
        group_availability = zeros(n_food_groups)
        for i in 1:n_foods
            g = food_group_assignments[i]
            group_availability[g] += food_supply_limits[i]
        end

        min_servings_per_group = zeros(n_food_groups)
        for g in 1:n_food_groups
            if group_availability[g] > 0
                min_servings_per_group[g] = group_availability[g] * rand(Uniform(0.1, 0.3))
            end
        end

        if feasibility_status == infeasible
            # Make one group requirement impossible
            target_group = rand(1:n_food_groups)
            min_servings_per_group[target_group] = group_availability[target_group] * 1.5
        end

    elseif variant == diet_variety
        # Minimum number of different foods that must be used
        min_foods_used = max(3, min(n_foods - 2, rand(n_foods ÷ 3:n_foods ÷ 2)))
        use_threshold = 0.01  # Food counts as "used" if x[i] >= threshold

        if feasibility_status == infeasible
            # Require more foods than available considering supply limits
            available_count = count(food_supply_limits .>= use_threshold)
            min_foods_used = available_count + rand(1:3)
        end

    elseif variant == diet_macro_ratios
        # Assign first 3 nutrients as protein, carbs, fat for ratio constraints
        protein_idx = min(1, n_nutrients)
        carb_idx = min(2, n_nutrients)
        fat_idx = min(3, n_nutrients)

        # Typical macro ratios (percentages of total macros)
        # Protein: 10-35%, Carbs: 45-65%, Fat: 20-35%
        min_protein_ratio = rand(Uniform(0.10, 0.15))
        max_protein_ratio = rand(Uniform(0.30, 0.35))
        min_carb_ratio = rand(Uniform(0.45, 0.50))
        max_carb_ratio = rand(Uniform(0.60, 0.65))
        min_fat_ratio = rand(Uniform(0.20, 0.25))
        max_fat_ratio = rand(Uniform(0.30, 0.35))

        if feasibility_status == infeasible
            # Make ratios impossible (they don't sum to 1)
            min_protein_ratio = 0.40
            min_carb_ratio = 0.50
            min_fat_ratio = 0.30
        end

    elseif variant == diet_calorie_range
        # First nutrient is calories with both min and max
        calorie_idx = 1
        # Typical daily calorie ranges
        target_calories = rand(Uniform(1800, 2500))
        min_calories = target_calories * rand(Uniform(0.90, 0.95))
        max_calories = target_calories * rand(Uniform(1.05, 1.10))

        if feasibility_status == infeasible
            # Make max less than min
            min_calories = target_calories
            max_calories = target_calories * 0.8
        end

    elseif variant == diet_allergen_free
        # Some foods are excluded (allergens)
        n_excluded = max(1, rand(n_foods ÷ 10:n_foods ÷ 5))
        excluded_foods = Set(randperm(n_foods)[1:n_excluded])

        # Increase supply limits for non-excluded foods to maintain feasibility
        if feasibility_status == feasible
            for i in 1:n_foods
                if !(i in excluded_foods)
                    food_supply_limits[i] *= rand(Uniform(1.2, 1.5))
                end
            end
        elseif feasibility_status == infeasible
            # Exclude all foods that are good sources of a critical nutrient
            critical_nutrient = argmax(b)
            contributions = [(a[i, critical_nutrient], i) for i in 1:n_foods]
            sort!(contributions, rev=true)
            # Exclude top contributors
            for k in 1:min(n_foods ÷ 2, length(contributions))
                push!(excluded_foods, contributions[k][2])
            end
        end

    elseif variant == diet_meal_plan
        # Foods assigned to meals (breakfast/lunch/dinner)
        n_meals = 3
        food_meal_compat = zeros(Bool, n_foods, n_meals)

        # Each food is compatible with 1-2 meals
        for i in 1:n_foods
            n_compat = rand(1:2)
            compat_meals = randperm(n_meals)[1:n_compat]
            for m in compat_meals
                food_meal_compat[i, m] = true
            end
        end

        # Calorie targets per meal
        total_daily_calories = rand(Uniform(1800, 2500))
        meal_calorie_targets = zeros(n_meals)
        meal_calorie_targets[1] = total_daily_calories * rand(Uniform(0.20, 0.30))  # Breakfast
        meal_calorie_targets[2] = total_daily_calories * rand(Uniform(0.35, 0.45))  # Lunch
        meal_calorie_targets[3] = total_daily_calories - meal_calorie_targets[1] - meal_calorie_targets[2]  # Dinner
        meal_calorie_tolerance = rand(Uniform(0.10, 0.20))

        if feasibility_status == infeasible
            # Make one meal impossible by limiting compatible foods
            target_meal = rand(1:n_meals)
            for i in 1:n_foods
                if food_meal_compat[i, target_meal]
                    food_supply_limits[i] *= 0.1
                end
            end
        end
    end

    return DietProblem(
        n_foods,
        n_nutrients,
        c,
        a,
        b,
        food_supply_limits,
        cost_budget,
        min_food_amounts,
        max_food_amounts,
        variant,
        nutrient_upper_bounds,
        n_food_groups,
        food_group_assignments,
        min_servings_per_group,
        min_foods_used,
        use_threshold,
        protein_idx,
        carb_idx,
        fat_idx,
        min_protein_ratio,
        max_protein_ratio,
        min_carb_ratio,
        max_carb_ratio,
        min_fat_ratio,
        max_fat_ratio,
        calorie_idx,
        min_calories,
        max_calories,
        excluded_foods,
        n_meals,
        food_meal_compat,
        meal_calorie_targets,
        meal_calorie_tolerance
    )
end

"""
    build_model(prob::DietProblem)

Build a JuMP model for the diet problem based on its variant.

# Arguments
- `prob`: DietProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::DietProblem)
    model = Model()

    if prob.variant == diet_standard || prob.variant == diet_nutrient_bounds ||
       prob.variant == diet_food_groups || prob.variant == diet_variety ||
       prob.variant == diet_macro_ratios || prob.variant == diet_calorie_range ||
       prob.variant == diet_allergen_free
        # Standard food consumption variables
        @variable(model, x[1:prob.n_foods] >= 0)

        @objective(model, Min, sum(prob.costs[i] * x[i] for i in 1:prob.n_foods))

        # Nutrient requirements
        for j in 1:prob.n_nutrients
            @constraint(model, sum(prob.nutrient_content[i, j] * x[i] for i in 1:prob.n_foods) >= prob.requirements[j])
        end

        # Food supply limits
        for i in 1:prob.n_foods
            if prob.food_supply_limits[i] < Inf
                @constraint(model, x[i] <= prob.food_supply_limits[i])
            end
        end

        # Cost budget constraint
        if prob.cost_budget < Inf
            @constraint(model, sum(prob.costs[i] * x[i] for i in 1:prob.n_foods) <= prob.cost_budget)
        end

        # Minimum consumption requirements
        for (i, min_amount) in prob.min_food_amounts
            @constraint(model, x[i] >= min_amount)
        end

        # Maximum consumption limits
        for (i, max_amount) in prob.max_food_amounts
            @constraint(model, x[i] <= max_amount)
        end

        # Variant-specific constraints
        if prob.variant == diet_nutrient_bounds && prob.nutrient_upper_bounds !== nothing
            # Upper bounds on nutrients (sodium, saturated fat limits)
            for j in 1:prob.n_nutrients
                @constraint(model, sum(prob.nutrient_content[i, j] * x[i] for i in 1:prob.n_foods) <= prob.nutrient_upper_bounds[j])
            end

        elseif prob.variant == diet_food_groups && prob.food_group_assignments !== nothing
            # Minimum servings from each food group
            for g in 1:prob.n_food_groups
                foods_in_group = [i for i in 1:prob.n_foods if prob.food_group_assignments[i] == g]
                if !isempty(foods_in_group)
                    @constraint(model, sum(x[i] for i in foods_in_group) >= prob.min_servings_per_group[g])
                end
            end

        elseif prob.variant == diet_variety && prob.min_foods_used > 0
            # Minimum number of different foods used (linearized with binary variables)
            @variable(model, y[1:prob.n_foods], Bin)  # 1 if food is used
            M = maximum(prob.food_supply_limits[isfinite.(prob.food_supply_limits)])

            for i in 1:prob.n_foods
                # x[i] >= threshold => y[i] = 1 (approximated)
                @constraint(model, x[i] <= M * y[i])
                @constraint(model, x[i] >= prob.use_threshold * y[i] - prob.use_threshold * (1 - y[i]))
            end
            @constraint(model, sum(y) >= prob.min_foods_used)

        elseif prob.variant == diet_macro_ratios && prob.protein_idx > 0
            # Macronutrient ratio constraints
            total_macro = @expression(model,
                sum(prob.nutrient_content[i, prob.protein_idx] * x[i] for i in 1:prob.n_foods) +
                sum(prob.nutrient_content[i, prob.carb_idx] * x[i] for i in 1:prob.n_foods) +
                sum(prob.nutrient_content[i, prob.fat_idx] * x[i] for i in 1:prob.n_foods))

            protein_intake = @expression(model, sum(prob.nutrient_content[i, prob.protein_idx] * x[i] for i in 1:prob.n_foods))
            carb_intake = @expression(model, sum(prob.nutrient_content[i, prob.carb_idx] * x[i] for i in 1:prob.n_foods))
            fat_intake = @expression(model, sum(prob.nutrient_content[i, prob.fat_idx] * x[i] for i in 1:prob.n_foods))

            # protein >= min_protein_ratio * total
            @constraint(model, protein_intake >= prob.min_protein_ratio * total_macro)
            @constraint(model, protein_intake <= prob.max_protein_ratio * total_macro)
            @constraint(model, carb_intake >= prob.min_carb_ratio * total_macro)
            @constraint(model, carb_intake <= prob.max_carb_ratio * total_macro)
            @constraint(model, fat_intake >= prob.min_fat_ratio * total_macro)
            @constraint(model, fat_intake <= prob.max_fat_ratio * total_macro)

        elseif prob.variant == diet_calorie_range && prob.calorie_idx > 0
            # Calorie range constraints
            total_calories = @expression(model, sum(prob.nutrient_content[i, prob.calorie_idx] * x[i] for i in 1:prob.n_foods))
            @constraint(model, total_calories >= prob.min_calories)
            @constraint(model, total_calories <= prob.max_calories)

        elseif prob.variant == diet_allergen_free && prob.excluded_foods !== nothing
            # Excluded foods cannot be used
            for i in prob.excluded_foods
                @constraint(model, x[i] == 0)
            end
        end

    elseif prob.variant == diet_meal_plan
        # Food consumption per meal
        @variable(model, x[1:prob.n_foods, 1:prob.n_meals] >= 0)

        @objective(model, Min, sum(prob.costs[i] * x[i, m] for i in 1:prob.n_foods, m in 1:prob.n_meals))

        # Total nutrient requirements across all meals
        for j in 1:prob.n_nutrients
            @constraint(model, sum(prob.nutrient_content[i, j] * x[i, m]
                for i in 1:prob.n_foods, m in 1:prob.n_meals) >= prob.requirements[j])
        end

        # Food supply limits (total across meals)
        for i in 1:prob.n_foods
            if prob.food_supply_limits[i] < Inf
                @constraint(model, sum(x[i, m] for m in 1:prob.n_meals) <= prob.food_supply_limits[i])
            end
        end

        # Cost budget
        if prob.cost_budget < Inf
            @constraint(model, sum(prob.costs[i] * x[i, m] for i in 1:prob.n_foods, m in 1:prob.n_meals) <= prob.cost_budget)
        end

        # Food-meal compatibility (only compatible foods can be used in each meal)
        if prob.food_meal_compat !== nothing
            for i in 1:prob.n_foods, m in 1:prob.n_meals
                if !prob.food_meal_compat[i, m]
                    @constraint(model, x[i, m] == 0)
                end
            end
        end

        # Calorie targets per meal (with tolerance)
        if prob.meal_calorie_targets !== nothing && prob.n_nutrients >= 1
            calorie_idx = 1
            for m in 1:prob.n_meals
                meal_calories = @expression(model, sum(prob.nutrient_content[i, calorie_idx] * x[i, m] for i in 1:prob.n_foods))
                target = prob.meal_calorie_targets[m]
                tolerance = prob.meal_calorie_tolerance
                @constraint(model, meal_calories >= target * (1 - tolerance))
                @constraint(model, meal_calories <= target * (1 + tolerance))
            end
        end
    end

    return model
end

# Register the problem type
register_problem(
    :diet_problem,
    DietProblem,
    "Diet problem with variants including standard, nutrient bounds, food groups, variety, macro ratios, calorie range, allergen free, and meal planning"
)
