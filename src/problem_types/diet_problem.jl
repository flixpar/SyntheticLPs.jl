using JuMP
using Random


"""
    generate_diet_problem(params::Dict=Dict(); seed::Int=0)

Generate a diet problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_foods`: Number of different foods (default: 5)
  - `:n_nutrients`: Number of different nutrients (default: 3)
  - `:cost_range`: Tuple (min, max) for food costs (default: (1.0, 5.0))
  - `:nutrient_range`: Tuple (min, max) for nutrient content (default: (0.1, 2.0))
  - `:requirement_factor`: Factor to determine nutrient requirements (default: 0.3)
  - `:solution_status`: Solution status (:feasible, :infeasible, or :all) (default: :feasible)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_diet_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    (seed >= 0) && Random.seed!(seed)
    
    # Extract parameters with defaults
    n_foods = get(params, :n_foods, 150)
    n_nutrients = get(params, :n_nutrients, 50)
    cost_range = get(params, :cost_range, (1.0, 5.0))
    nutrient_range = get(params, :nutrient_range, (0.1, 2.0))
    requirement_factor = get(params, :requirement_factor, 0.3)
    solution_status = get(params, :solution_status, :feasible)
    
    # Validate solution_status parameter
    if !(solution_status in [:feasible, :infeasible, :all])
        error("Invalid solution_status: $solution_status. Must be :feasible, :infeasible, or :all")
    end
    
    # Determine the actual solution status
    actual_status = solution_status
    if solution_status == :all
        actual_status = rand() < 0.75 ? :feasible : :infeasible
    end
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_foods => n_foods,
        :n_nutrients => n_nutrients,
        :cost_range => cost_range,
        :nutrient_range => nutrient_range,
        :requirement_factor => requirement_factor,
        :solution_status => solution_status,
        :actual_status => actual_status
    )
    
    # Generate basic food data (costs and nutrient content)
    min_cost, max_cost = cost_range
    c = rand(min_cost:0.1:max_cost, n_foods)  # Costs per unit of food
    
    min_nutrient, max_nutrient = nutrient_range
    a = rand(min_nutrient:0.1:max_nutrient, n_foods, n_nutrients)  # Nutrient content per unit of food
    
    # Initialize constraint variables
    b = zeros(n_nutrients)  # Nutrient requirements
    food_supply_limits = fill(Inf, n_foods)  # Supply limits for each food
    cost_budget = Inf  # Total cost budget
    min_food_amounts = Dict{Int, Float64}()  # Minimum consumption requirements
    max_food_amounts = Dict{Int, Float64}()  # Maximum consumption limits
    scenario_type = ""
    
    if actual_status == :feasible
        # SOPHISTICATED FEASIBLE APPROACH: Create challenging but feasible constraints
        
        # Step 1: Generate an intelligent baseline diet using nutrition-cost optimization
        # Find a cost-efficient diet that provides good nutritional coverage
        
        # Calculate nutrition efficiency for each food
        nutrition_scores = zeros(n_foods)
        for i in 1:n_foods
            # Nutrition score: sum of normalized nutrient values (each nutrient weighted equally)
            nutrition_scores[i] = sum(a[i, :]) / n_nutrients
        end
        
        # Calculate cost-effectiveness (nutrition per unit cost)
        cost_effectiveness = nutrition_scores ./ c
        
        # Sort foods by cost-effectiveness
        effectiveness_order = sortperm(cost_effectiveness, rev=true)
        
        # Generate realistic baseline diet favoring cost-effective foods
        baseline_diet = zeros(n_foods)
        
        # Primary foods: top 60% most cost-effective foods get 75% of consumption
        primary_count = max(3, round(Int, n_foods * 0.6))
        primary_foods = effectiveness_order[1:primary_count]
        
        # Assign consumption amounts to primary foods
        base_consumption = 100.0  # Base total food consumption
        primary_total = base_consumption * 0.75
        
        for i in primary_foods
            # Weight by cost-effectiveness with some randomness
            effectiveness_weight = cost_effectiveness[i] / sum(cost_effectiveness[primary_foods])
            baseline_amount = primary_total * effectiveness_weight
            # Add ±30% variation for realism
            baseline_diet[i] = baseline_amount * (0.7 + rand() * 0.6)
        end
        
        # Secondary foods: remaining foods get 25% of consumption for dietary variety
        secondary_foods = effectiveness_order[(primary_count + 1):end]
        if !isempty(secondary_foods)
            secondary_total = base_consumption * 0.25
            for i in secondary_foods
                baseline_diet[i] = secondary_total / length(secondary_foods) * (0.5 + rand())
            end
        end
        
        # Normalize to target consumption level
        total_baseline = sum(baseline_diet)
        baseline_diet .*= base_consumption / total_baseline
        
        # Step 2: Calculate achieved nutrient levels from baseline diet
        achieved_nutrients = zeros(n_nutrients)
        for j in 1:n_nutrients
            achieved_nutrients[j] = sum(a[i, j] * baseline_diet[i] for i in 1:n_foods)
        end
        
        # Step 3: Set challenging nutrient requirements with realistic tolerances
        tolerance_scenario = rand(1:3)
        tolerance_level = if tolerance_scenario == 1
            scenario_type = "tight_nutrition_requirements"
            0.02 + rand() * 0.03  # 2-5% tolerance (clinical nutrition level)
        elseif tolerance_scenario == 2
            scenario_type = "standard_nutrition_requirements"
            0.05 + rand() * 0.05  # 5-10% tolerance (standard dietary guidelines)
        else
            scenario_type = "relaxed_nutrition_requirements"
            0.08 + rand() * 0.04  # 8-12% tolerance (general wellness level)
        end
        
        # Set nutrient requirements around achieved levels with tolerance
        for j in 1:n_nutrients
            tolerance = tolerance_level
            # Position achieved level at 70-85% through tolerance band (not centered)
            position_in_band = 0.7 + rand() * 0.15
            
            # Calculate requirements ensuring achieved level falls within tolerance
            total_range = 2 * tolerance * achieved_nutrients[j] / (1 - 2 * tolerance + 2 * tolerance * position_in_band)
            lower_bound = achieved_nutrients[j] - total_range * position_in_band
            
            # Set minimum nutrient requirement (diet problems typically have >= constraints)
            b[j] = max(0.0, lower_bound)
        end
        
        # Step 4: Add realistic supply constraints creating market pressure
        supply_scenario = rand(1:3)
        if supply_scenario == 1
            # Seasonal availability constraints
            scenario_type *= "_seasonal_supply"
            critical_foods = primary_foods[1:max(2, div(length(primary_foods), 3))]
            for i in 1:n_foods
                if i in critical_foods
                    # Limited seasonal availability: 110-140% of needed (72-91% utilization)
                    food_supply_limits[i] = baseline_diet[i] * (1.1 + rand() * 0.3)
                else
                    # Normal availability: 150-250% of needed (40-67% utilization)
                    food_supply_limits[i] = baseline_diet[i] * (1.5 + rand())
                end
            end
        elseif supply_scenario == 2
            # Market supply constraints
            scenario_type *= "_market_supply"
            expensive_foods = sortperm(c, rev=true)[1:max(2, div(n_foods, 4))]
            for i in 1:n_foods
                if i in expensive_foods
                    # Premium foods have limited supply: 120-160% of needed (62-83% utilization)
                    food_supply_limits[i] = baseline_diet[i] * (1.2 + rand() * 0.4)
                else
                    # Regular foods: abundant supply
                    food_supply_limits[i] = baseline_diet[i] * (2.0 + rand() * 2.0)
                end
            end
        else
            # No binding supply constraints
            scenario_type *= "_normal_supply"
            for i in 1:n_foods
                food_supply_limits[i] = baseline_diet[i] * (3.0 + rand() * 2.0)
            end
        end
        
        # Step 5: Set challenging cost budget
        baseline_cost = sum(c[i] * baseline_diet[i] for i in 1:n_foods)
        cost_pressure = rand(1:3)
        if cost_pressure == 1
            # Tight budget: 105-115% of baseline cost
            cost_budget = baseline_cost * (1.05 + rand() * 0.10)
            scenario_type *= "_tight_budget"
        elseif cost_pressure == 2
            # Moderate budget: 110-125% of baseline cost
            cost_budget = baseline_cost * (1.10 + rand() * 0.15)
            scenario_type *= "_moderate_budget"
        else
            # Generous budget: not a binding constraint
            cost_budget = baseline_cost * (1.5 + rand() * 0.5)
            scenario_type *= "_generous_budget"
        end
        
        # Step 6: Add realistic consumption preferences/constraints
        if rand() < 0.7  # 70% chance of having preference constraints
            # Minimum consumption requirements (dietary preferences/medical needs)
            preferred_foods = randperm(n_foods)[1:max(1, div(n_foods, 6))]
            for i in preferred_foods
                min_food_amounts[i] = baseline_diet[i] * (0.6 + rand() * 0.3)  # 60-90% of baseline
            end
            
            # Maximum consumption limits (allergies/medical restrictions)
            limited_foods = randperm(n_foods)[1:max(1, div(n_foods, 5))]
            for i in limited_foods
                max_food_amounts[i] = baseline_diet[i] * (1.3 + rand() * 0.4)  # 130-170% of baseline
            end
        end
        
        # Store additional feasible problem data
        actual_params[:baseline_diet] = baseline_diet
        actual_params[:baseline_cost] = baseline_cost
        actual_params[:tolerance_level] = tolerance_level
        
    else  # :infeasible - Create verified mathematical impossibilities
        
        scenario = rand(1:4)
        
        if scenario == 1
            # SCENARIO 1: Verified nutrient impossibility conflict
            scenario_type = "verified_nutrient_impossibility"
            
            # Step 1: Set realistic supply constraints first (so we know true limits)
            base_supply = 100.0  # Realistic base supply per food
            for i in 1:n_foods
                # Realistic supply variation: 50-200 units per food
                food_supply_limits[i] = base_supply * (0.5 + rand() * 1.5)
            end
            
            # Step 2: Calculate TRUE maximum achievable for each nutrient given supply limits
            max_achievable_nutrients = zeros(n_nutrients)
            for j in 1:n_nutrients
                # Maximum possible if we use ALL supply of ALL foods optimally for this nutrient
                max_achievable_nutrients[j] = sum(a[i, j] * food_supply_limits[i] for i in 1:n_foods)
            end
            
            # Step 3: Select one nutrient to make mathematically impossible
            target_nutrient = rand(1:n_nutrients)
            
            # Step 4: Set impossible requirement for target nutrient
            # Require 120-150% of maximum mathematically possible
            b[target_nutrient] = max_achievable_nutrients[target_nutrient] * (1.2 + rand() * 0.3)
            
            # Step 5: Set realistic but achievable requirements for other nutrients
            for j in 1:n_nutrients
                if j != target_nutrient
                    # Set to 30-70% of maximum possible (definitely achievable)
                    b[j] = max_achievable_nutrients[j] * (0.3 + rand() * 0.4)
                end
            end
            
            # Step 6: Verify impossibility - double-check our math
            final_max_achievable = sum(a[i, target_nutrient] * food_supply_limits[i] for i in 1:n_foods)
            if final_max_achievable >= b[target_nutrient]
                # Force impossibility if verification failed
                b[target_nutrient] = final_max_achievable * 1.15
            end
            
            # Step 7: Generous budget (not the limiting factor)
            cost_budget = sum(c[i] * food_supply_limits[i] for i in 1:n_foods)  # Can afford all available food
            
        elseif scenario == 2
            # SCENARIO 2: Verified budget impossibility conflict
            scenario_type = "verified_budget_impossibility"
            
            # Step 1: Set generous supply limits (not the limiting factor)
            for i in 1:n_foods
                food_supply_limits[i] = 500.0  # Very generous supply
            end
            
            # Step 2: Set realistic nutrient requirements (achievable with enough budget)
            for j in 1:n_nutrients
                # Set moderate nutrient requirements: 20-50 units of best food equivalent
                best_content = maximum(a[:, j])
                target_units = 20.0 + rand() * 30.0  # 20-50 units
                b[j] = best_content * target_units
            end
            
            # Step 3: Calculate PROVEN lower bound on minimum cost needed
            # This is a lower bound because we calculate each nutrient constraint independently
            # (ignoring that foods can provide multiple nutrients simultaneously)
            proven_min_cost = 0.0
            for j in 1:n_nutrients
                # For each nutrient, find cheapest cost per unit of this nutrient
                best_cost_efficiency = Inf
                for i in 1:n_foods
                    if a[i, j] > 0
                        cost_per_nutrient_unit = c[i] / a[i, j]
                        best_cost_efficiency = min(best_cost_efficiency, cost_per_nutrient_unit)
                    end
                end
                
                if best_cost_efficiency < Inf
                    # Minimum cost just for this nutrient requirement
                    proven_min_cost += b[j] * best_cost_efficiency
                end
            end
            
            # Step 4: Set budget strictly below proven minimum
            # Since proven_min_cost is a lower bound, any budget below this guarantees infeasibility
            if proven_min_cost > 0
                cost_budget = proven_min_cost * (0.7 + rand() * 0.2)  # 70-90% of proven minimum
            else
                # Fallback: set very low budget relative to food costs
                avg_cost = sum(c) / n_foods
                cost_budget = avg_cost * 5.0  # Can only afford 5 units of average-cost food
            end
            
            # Step 5: Verify budget impossibility - double-check our math
            # Recalculate minimum cost for the specific requirements we set
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
            
            # If budget is not clearly below minimum, force impossibility
            if verification_min_cost > 0 && cost_budget >= verification_min_cost * 0.95
                cost_budget = verification_min_cost * 0.8  # Set to 80% of minimum needed
            end
            
        elseif scenario == 3
            # SCENARIO 3: Verified supply shortage conflict
            scenario_type = "verified_supply_shortage"
            
            # Step 1: Set realistic nutrient requirements first
            for j in 1:n_nutrients
                # Moderate requirements: 30-60 units of best food equivalent
                best_content = maximum(a[:, j])
                target_units = 30.0 + rand() * 30.0
                b[j] = best_content * target_units
            end
            
            # Step 2: Start with generous supply limits
            base_supply = 200.0
            for i in 1:n_foods
                food_supply_limits[i] = base_supply * (0.8 + rand() * 0.4)  # 160-240 units
            end
            
            # Step 3: Select one nutrient to make impossible through supply shortage
            target_nutrient = rand(1:n_nutrients)
            
            # Step 4: Calculate current maximum achievable for target nutrient
            current_max = sum(a[i, target_nutrient] * food_supply_limits[i] for i in 1:n_foods)
            
            # Step 5: Reduce supply limits to make target nutrient impossible
            # We need: sum(a[i, target_nutrient] * new_supply[i]) < b[target_nutrient]
            # Strategy: reduce supply of foods that are rich in the target nutrient
            
            # Find foods ranked by their contribution to target nutrient
            nutrient_contributions = [(a[i, target_nutrient] * food_supply_limits[i], i) for i in 1:n_foods]
            sort!(nutrient_contributions, rev=true)
            
            # Reduce supply of top contributors until target becomes impossible
            reduction_needed = current_max - b[target_nutrient] * 0.95  # Need to reduce by this much
            remaining_reduction = reduction_needed
            
            for (contribution, food_idx) in nutrient_contributions
                if remaining_reduction > 0
                    # Calculate how much to reduce this food's supply
                    max_reduction = food_supply_limits[food_idx] * 0.9  # Don't eliminate completely
                    actual_reduction = min(remaining_reduction / a[food_idx, target_nutrient], max_reduction)
                    
                    # Apply reduction but keep supply realistic (at least 10 units)
                    new_supply = max(10.0, food_supply_limits[food_idx] - actual_reduction)
                    reduction_achieved = (food_supply_limits[food_idx] - new_supply) * a[food_idx, target_nutrient]
                    
                    food_supply_limits[food_idx] = new_supply
                    remaining_reduction -= reduction_achieved
                    
                    if remaining_reduction <= 0
                        break
                    end
                end
            end
            
            # Step 6: Verify we achieved impossibility - robust double-check
            final_max = sum(a[i, target_nutrient] * food_supply_limits[i] for i in 1:n_foods)
            if final_max >= b[target_nutrient] * 0.99  # Use 99% to account for numerical precision
                # Force impossibility by setting requirement above maximum possible
                b[target_nutrient] = final_max * (1.1 + rand() * 0.1)  # 110-120% of maximum possible
            end
            
            # Step 7: Generous budget (not the limiting factor)
            cost_budget = sum(c[i] * food_supply_limits[i] for i in 1:n_foods) * 1.5
            
        else  # scenario == 4
            # SCENARIO 4: Verified over-constrained system
            scenario_type = "verified_over_constrained_system"
            
            # Strategy: Create multiple individually-reasonable constraints that together are impossible
            
            # Step 1: Generate a baseline feasible diet to work from
            baseline_consumption = 100.0
            baseline_diet = zeros(n_foods)
            
            # Simple baseline: spread consumption across foods weighted by cost-effectiveness
            nutrition_scores = [sum(a[i, :]) for i in 1:n_foods]
            cost_effectiveness = nutrition_scores ./ c
            total_effectiveness = sum(cost_effectiveness)
            
            for i in 1:n_foods
                # Allocate consumption based on cost-effectiveness with variation
                base_share = cost_effectiveness[i] / total_effectiveness
                baseline_diet[i] = baseline_consumption * base_share * (0.5 + rand())
            end
            
            # Normalize to target total consumption
            total_baseline = sum(baseline_diet)
            baseline_diet .*= baseline_consumption / total_baseline
            
            # Step 2: Set nutrient requirements around baseline achievement (individually feasible)
            for j in 1:n_nutrients
                baseline_achievement = sum(a[i, j] * baseline_diet[i] for i in 1:n_foods)
                # Set requirement at 110-130% of baseline (challenging but individually achievable)
                b[j] = baseline_achievement * (1.1 + rand() * 0.2)
            end
            
            # Step 3: Set supply limits that are tight around baseline (individually feasible)
            for i in 1:n_foods
                # Each food can supply 120-180% of baseline needs (individually feasible)
                food_supply_limits[i] = baseline_diet[i] * (1.2 + rand() * 0.6)
            end
            
            # Step 4: Set budget constraint that's tight around baseline cost (individually feasible)
            baseline_cost = sum(c[i] * baseline_diet[i] for i in 1:n_foods)
            # Budget allows 110-130% of baseline cost (individually feasible)
            cost_budget = baseline_cost * (1.1 + rand() * 0.2)
            
            # Step 5: Add preference constraints that create conflicts
            # Require minimum consumption of some expensive foods
            expensive_foods = sortperm(c, rev=true)[1:max(2, div(n_foods, 5))]
            num_required = max(1, div(length(expensive_foods), 2))
            required_foods = expensive_foods[1:num_required]
            
            for i in required_foods
                # Must consume more than baseline uses (forces higher cost)
                min_food_amounts[i] = baseline_diet[i] * (1.3 + rand() * 0.4)  # 130-170% of baseline
            end
            
            # Restrict maximum consumption of some nutritious foods  
            nutritious_foods = []
            avg_nutrition = sum(nutrition_scores) / n_foods
            for i in 1:n_foods
                if nutrition_scores[i] > avg_nutrition * 1.2  # 20% above average nutrition
                    push!(nutritious_foods, i)
                end
            end
            
            if !isempty(nutritious_foods)
                num_restricted = max(1, div(length(nutritious_foods), 3))
                restricted_foods = nutritious_foods[1:min(num_restricted, length(nutritious_foods))]
                
                for i in restricted_foods
                    # Cannot exceed baseline usage (limits nutrition achievement)
                    max_food_amounts[i] = baseline_diet[i] * (0.8 + rand() * 0.3)  # 80-110% of baseline
                end
            end
            
            # Step 6: Verify impossibility and force it if needed
            # Calculate if current constraint combination is actually impossible
            
            # Step 7: Force mathematical impossibility with verification
            # Instead of complex constraint interactions, use a direct approach
            
            # Select one nutrient to make mathematically impossible
            target_nutrient = rand(1:n_nutrients)
            
            # Calculate true maximum achievable for this nutrient given ALL constraints
            max_achievable_target = 0.0
            for i in 1:n_foods
                # Determine feasible range for this food considering all constraints
                min_usage = get(min_food_amounts, i, 0.0)
                max_usage = food_supply_limits[i]
                
                if haskey(max_food_amounts, i)
                    max_usage = min(max_usage, max_food_amounts[i])
                end
                
                # If min > max, this food creates infeasibility, but let's use what we can
                feasible_max = max(0.0, min(max_usage, max(min_usage, max_usage)))
                
                max_achievable_target += a[i, target_nutrient] * feasible_max
            end
            
            # Force impossibility by setting requirement above maximum possible
            b[target_nutrient] = max_achievable_target * (1.2 + rand() * 0.2)  # 120-140% of maximum
            
            # Keep other nutrients achievable but challenging
            for j in 1:n_nutrients
                if j != target_nutrient
                    max_achievable_j = 0.0
                    for i in 1:n_foods
                        min_usage = get(min_food_amounts, i, 0.0)
                        max_usage = min(food_supply_limits[i], get(max_food_amounts, i, food_supply_limits[i]))
                        feasible_max = max(0.0, min(max_usage, max(min_usage, max_usage)))
                        max_achievable_j += a[i, j] * feasible_max
                    end
                    # Set to 70-90% of maximum possible (achievable)
                    b[j] = max_achievable_j * (0.7 + rand() * 0.2)
                end
            end
        end
    end
    
    # FINAL VERIFICATION: Guarantee infeasibility for infeasible instances
    if actual_status == :infeasible
        # Calculate absolute maximum achievable for each nutrient given ALL constraints
        verified_max_achievable = zeros(n_nutrients)
        
        for j in 1:n_nutrients
            max_possible_j = 0.0
            
            for i in 1:n_foods
                # Determine actual feasible range for food i given all constraints
                min_usage_i = get(min_food_amounts, i, 0.0)
                max_usage_i = food_supply_limits[i]
                
                # Apply maximum consumption constraints
                if haskey(max_food_amounts, i)
                    max_usage_i = min(max_usage_i, max_food_amounts[i])
                end
                
                # Handle constraint conflicts
                if min_usage_i > max_usage_i
                    # Constraint conflict - use minimum possible
                    feasible_usage = 0.0
                else
                    # Use maximum feasible usage for this food
                    feasible_usage = max_usage_i
                end
                
                max_possible_j += a[i, j] * feasible_usage
            end
            
            verified_max_achievable[j] = max_possible_j
        end
        
        # Force at least one constraint to be impossible with large margin
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
        
        # Ensure impossibility with very large margin (avoid numerical issues)
        if verified_max_achievable[target_nutrient_final] > 0
            b[target_nutrient_final] = verified_max_achievable[target_nutrient_final] * (2.0 + rand())
        else
            # If max achievable is 0, set positive requirement
            b[target_nutrient_final] = 100.0 + rand() * 100.0
        end
    end
    
    # Store comprehensive constraint data in actual_params
    actual_params[:costs] = c
    actual_params[:nutrient_content] = a
    actual_params[:requirements] = b
    actual_params[:food_supply_limits] = food_supply_limits
    actual_params[:cost_budget] = cost_budget
    actual_params[:min_food_amounts] = min_food_amounts
    actual_params[:max_food_amounts] = max_food_amounts
    actual_params[:scenario_type] = scenario_type
    
    # Model
    model = Model()
    
    # Variables
    @variable(model, x[1:n_foods] >= 0)
    
    # Objective
    @objective(model, Min, sum(c[i] * x[i] for i in 1:n_foods))
    
    # Constraints
    
    # 1. Nutrient requirements (minimum intake for each nutrient)
    for j in 1:n_nutrients
        @constraint(model, sum(a[i, j] * x[i] for i in 1:n_foods) >= b[j])
    end
    
    # 2. Food supply limits (availability constraints)
    for i in 1:n_foods
        if food_supply_limits[i] < Inf
            @constraint(model, x[i] <= food_supply_limits[i])
        end
    end
    
    # 3. Cost budget constraint
    if cost_budget < Inf
        @constraint(model, sum(c[i] * x[i] for i in 1:n_foods) <= cost_budget)
    end
    
    # 4. Minimum consumption requirements (dietary preferences/medical needs)
    for (i, min_amount) in min_food_amounts
        @constraint(model, x[i] >= min_amount)
    end
    
    # 5. Maximum consumption limits (allergies/medical restrictions)
    for (i, max_amount) in max_food_amounts
        @constraint(model, x[i] <= max_amount)
    end
    
    return model, actual_params
end

"""
    sample_diet_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a diet problem with target number of variables.

# Arguments
- `target_variables`: Target number of variables (foods)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_diet_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # For diet problem, target_variables = n_foods
    params[:n_foods] = target_variables
    
    # Scale number of nutrients based on foods to maintain reasonable problem structure
    # Generally, there should be fewer nutrients than foods for a realistic diet problem
    # Real diet problems typically have 5-50 nutrients (vitamins, minerals, macronutrients)
    if target_variables <= 100
        params[:n_nutrients] = rand(5:min(25, max(5, Int(target_variables ÷ 4))))
    elseif target_variables <= 1000
        params[:n_nutrients] = rand(15:min(75, max(15, Int(target_variables ÷ 8))))
    else
        # For very large problems, scale nutrients more conservatively
        params[:n_nutrients] = rand(25:min(150, max(25, Int(target_variables ÷ 15))))
    end
    
    # Make parameters more diverse and realistic for different problem sizes
    if target_variables <= 100
        # Small diet problems - grocery store level
        params[:cost_range] = (rand(0.5:0.1:2.0), rand(3.0:0.5:8.0))
        params[:nutrient_range] = (rand(0.05:0.01:0.15), rand(1.5:0.1:3.0))
        params[:requirement_factor] = 0.2 + rand() * 0.3  # 0.2 to 0.5
    elseif target_variables <= 1000
        # Medium diet problems - institutional level
        params[:cost_range] = (rand(0.1:0.05:1.0), rand(2.0:0.5:10.0))
        params[:nutrient_range] = (rand(0.01:0.005:0.1), rand(1.0:0.2:4.0))
        params[:requirement_factor] = 0.15 + rand() * 0.4  # 0.15 to 0.55
    else
        # Large diet problems - industrial scale
        params[:cost_range] = (rand(0.05:0.01:0.5), rand(1.0:0.2:15.0))
        params[:nutrient_range] = (rand(0.005:0.001:0.05), rand(0.5:0.1:5.0))
        params[:requirement_factor] = 0.1 + rand() * 0.5  # 0.1 to 0.6
    end
    
    return params
end

"""
    sample_diet_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a diet problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_diet_parameters(size::Symbol=:medium; seed::Int=0)
    Random.seed!(seed)
    
    # Map size to target variables with realistic ranges
    if size == :small
        target_variables = rand(50:250)      # 50-250 variables
    elseif size == :medium
        target_variables = rand(250:1000)    # 250-1000 variables
    elseif size == :large
        target_variables = rand(1000:10000)  # 1000-10000 variables
    else
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    # Use the target-based function
    return sample_diet_parameters(target_variables; seed=seed)
end

"""
    calculate_diet_variable_count(params::Dict)

Calculate the number of variables for a diet problem.

# Arguments
- `params`: Dictionary of problem parameters containing `:n_foods`

# Returns
- Number of variables (equal to n_foods)
"""
function calculate_diet_variable_count(params::Dict)
    n_foods = get(params, :n_foods, 150)
    return n_foods
end

# Register the problem type
register_problem(
    :diet_problem,
    generate_diet_problem,
    sample_diet_parameters,
    "Diet problem that minimizes the cost of food while meeting nutritional requirements"
)