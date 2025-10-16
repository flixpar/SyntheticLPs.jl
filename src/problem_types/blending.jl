using JuMP
using Random


"""
    generate_blending_problem(params::Dict=Dict(); seed::Int=0)

Generate a blending problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_ingredients`: Number of ingredients (default: 5)
  - `:n_attributes`: Number of quality attributes (default: 3)
  - `:cost_range`: Tuple (min, max) for ingredient costs (default: (10, 100))
  - `:attribute_range`: Tuple (min, max) for attribute values (default: (0.1, 0.9))
  - `:min_blend_amount`: Minimum amount to produce (default: 100)
  - `:solution_status`: Solution status (:feasible, :infeasible, or :all) (default: :feasible)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_blending_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_ingredients = get(params, :n_ingredients, 5)
    n_attributes = get(params, :n_attributes, 3)
    cost_range = get(params, :cost_range, (10, 100))
    attribute_range = get(params, :attribute_range, (0.1, 0.9))
    min_blend_amount = get(params, :min_blend_amount, 100)
    solution_status = get(params, :solution_status, :feasible)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_ingredients => n_ingredients,
        :n_attributes => n_attributes,
        :cost_range => cost_range,
        :attribute_range => attribute_range,
        :min_blend_amount => min_blend_amount,
        :solution_status => solution_status
    )
    
    # Random data generation
    min_cost, max_cost = cost_range
    costs = rand(min_cost:max_cost, n_ingredients)  # Cost per unit of each ingredient
    
    min_attr, max_attr = attribute_range
    attributes = rand(min_attr:0.01:max_attr, n_ingredients, n_attributes)  # Attribute values for each ingredient
    
    # Determine the actual solution status
    actual_status = solution_status
    if solution_status == :all
        actual_status = rand() < 0.5 ? :feasible : :infeasible
    end
    
    # Initialize constraints
    lower_bounds = zeros(n_attributes)
    upper_bounds = zeros(n_attributes)
    supply_limits = fill(Inf, n_ingredients)
    cost_budget = Inf
    scenario_type = ""
    
    if actual_status == :feasible
        # CHALLENGING FEASIBLE APPROACH: Create tight but feasible constraints
        
        # Step 1: Generate an intelligent baseline solution using cost-quality optimization
        # Start with cost-efficient ingredients and adjust for quality requirements
        
        # Create a smart initial solution - balance low cost with quality needs
        cost_efficiency = zeros(n_ingredients)
        for i in 1:n_ingredients
            # Quality score: average of normalized attributes
            quality_score = sum(attributes[i, :]) / n_attributes
            # Efficiency: quality per unit cost
            cost_efficiency[i] = quality_score / costs[i]
        end
        
        # Sort ingredients by cost efficiency
        efficiency_order = sortperm(cost_efficiency, rev=true)
        
        # Generate solution favoring efficient ingredients but with realistic diversity
        blend_amounts = zeros(n_ingredients)
        remaining_amount = min_blend_amount
        
        # Use top 60% most efficient ingredients as primary components
        primary_count = max(3, round(Int, n_ingredients * 0.6))
        primary_ingredients = efficiency_order[1:primary_count]
        
        # Allocate 80% of volume to primary ingredients
        primary_total = min_blend_amount * 0.8
        for i in primary_ingredients
            # Weight allocation by efficiency but add randomness
            efficiency_weight = cost_efficiency[i] / sum(cost_efficiency[primary_ingredients])
            base_amount = primary_total * efficiency_weight
            # Add ±20% variation
            blend_amounts[i] = base_amount * (0.8 + rand() * 0.4)
        end
        
        # Allocate remaining 20% to other ingredients for diversity/constraints
        secondary_total = min_blend_amount * 0.2
        secondary_ingredients = efficiency_order[(primary_count + 1):end]
        for i in secondary_ingredients
            if !isempty(secondary_ingredients)
                blend_amounts[i] = secondary_total / length(secondary_ingredients) * (0.5 + rand())
            end
        end
        
        # Normalize to exact minimum blend amount
        total_amount = sum(blend_amounts)
        blend_amounts .*= min_blend_amount / total_amount
        
        # Step 2: Calculate achieved quality from this optimized solution
        achieved_qualities = zeros(n_attributes)
        for j in 1:n_attributes
            achieved_qualities[j] = sum(attributes[i, j] * blend_amounts[i] for i in 1:n_ingredients) / sum(blend_amounts)
        end
        
        # Step 3: Set TIGHT quality bounds with realistic tolerances
        scenario = rand(1:3)
        tolerance_level = if scenario == 1
            "tight_spec"
            0.025 + rand() * 0.025  # 2.5-5% tolerance (pharmaceutical/aerospace grade)
        elseif scenario == 2
            "standard_spec"
            0.04 + rand() * 0.03   # 4-7% tolerance (industrial standard)
        else
            "relaxed_spec"
            0.06 + rand() * 0.02   # 6-8% tolerance (commodity grade)
        end
        
        for j in 1:n_attributes
            tolerance = tolerance_level
            # Position achieved quality at 60-80% through the tolerance band (not centered)
            position_in_band = 0.6 + rand() * 0.2
            
            # Calculate bounds ensuring achieved quality is at position_in_band
            total_range = 2 * tolerance * achieved_qualities[j] / (1 - 2 * tolerance + 2 * tolerance * position_in_band)
            lower_bound = achieved_qualities[j] - total_range * position_in_band
            upper_bound = lower_bound + total_range
            
            lower_bounds[j] = max(min_attr, lower_bound)
            upper_bounds[j] = min(max_attr, upper_bound)
        end
        
        # Step 4: Set binding supply constraints - create realistic supply chain pressure
        critical_ingredients = primary_ingredients[1:max(2, div(length(primary_ingredients), 2))]
        
        for i in 1:n_ingredients
            if i in critical_ingredients
                # Critical ingredients: tight supply (110-130% of used, meaning 77-91% utilization)
                supply_limits[i] = blend_amounts[i] * (1.1 + rand() * 0.2)
            else
                # Non-critical: moderate constraints (130-170% of used, meaning 59-77% utilization)
                supply_limits[i] = blend_amounts[i] * (1.3 + rand() * 0.4)
            end
        end
        
        # Step 5: Set tight cost budget - force efficiency
        actual_cost = sum(costs[i] * blend_amounts[i] for i in 1:n_ingredients)
        cost_pressure = rand(1:3)
        if cost_pressure == 1
            # High cost pressure (106-112% of baseline cost)
            cost_budget = actual_cost * (1.06 + rand() * 0.06)
            scenario_type = "cost_pressured_feasible"
        elseif cost_pressure == 2
            # Moderate cost pressure (110-116% of baseline cost)
            cost_budget = actual_cost * (1.10 + rand() * 0.06)
            scenario_type = "standard_cost_feasible"
        else
            # Investment budget available (115-125% of baseline cost)
            cost_budget = actual_cost * (1.15 + rand() * 0.10)
            scenario_type = "investment_grade_feasible"
        end
        
        # Step 6: Add realistic additional constraints for complexity
        
        # Minimum usage requirements (regulatory/contractual)
        min_usage_required = Dict{Int, Float64}()
        required_ingredients = randperm(n_ingredients)[1:max(1, div(n_ingredients, 4))]
        for i in required_ingredients
            min_required = blend_amounts[i] * (0.7 + rand() * 0.2)  # 70-90% of current usage
            min_usage_required[i] = min_required
        end
        
        # Maximum percentage limits (safety/quality)
        max_usage_limits = Dict{Int, Float64}()
        limited_ingredients = randperm(n_ingredients)[1:max(1, div(n_ingredients, 3))]
        for i in limited_ingredients
            max_limit = blend_amounts[i] * (1.2 + rand() * 0.3)  # 120-150% of current usage
            max_usage_limits[i] = max_limit
        end
        
        # Store additional constraints
        actual_params[:min_usage_required] = min_usage_required
        actual_params[:max_usage_limits] = max_usage_limits
        actual_params[:baseline_solution] = blend_amounts
        actual_params[:baseline_cost] = actual_cost
        actual_params[:tolerance_level] = tolerance_level
        
    else  # :infeasible - Create systematic mathematical conflicts
        
        scenario = rand(1:4)
        
        if scenario == 1
            # Strategy 1: Supply shortage conflict
            scenario_type = "supply_shortage_conflict"
            
            # Step 1: Identify high-quality ingredients for each attribute
            quality_leaders = Vector{Int}[]
            for j in 1:n_attributes
                # Find ingredients with top 25% quality for this attribute
                quality_values = [(attributes[i, j], i) for i in 1:n_ingredients]
                sort!(quality_values, rev=true)
                top_count = max(1, div(n_ingredients, 4))
                leaders = [pair[2] for pair in quality_values[1:top_count]]
                push!(quality_leaders, leaders)
            end
            
            # Step 2: Set demanding quality requirements
            for j in 1:n_attributes
                # Require quality that only top ingredients can provide
                min_quality_needed = maximum(attributes[i, j] for i in quality_leaders[j]) * 0.95
                lower_bounds[j] = max(min_attr, min_quality_needed)
                upper_bounds[j] = min(max_attr, min_quality_needed * 1.05)
            end
            
            # Step 3: Severely limit supply of ALL quality leader ingredients
            critical_ingredients = unique(vcat(quality_leaders...))
            total_needed = min_blend_amount
            available_from_critical = 0.0
            
            for i in critical_ingredients
                # Limit each critical ingredient to much less than needed
                supply_limits[i] = total_needed * 0.15 / length(critical_ingredients)  # Each gets 15% of total need
                available_from_critical += supply_limits[i]
            end
            
            # Non-critical ingredients get normal supply but can't meet quality
            for i in 1:n_ingredients
                if !(i in critical_ingredients)
                    supply_limits[i] = min_blend_amount * 2.0
                end
            end
            
            # Generous cost budget (not the limiting factor)
            cost_budget = maximum(costs) * min_blend_amount * 2.0
            
        elseif scenario == 2
            # Strategy 2: Budget impossibility conflict
            scenario_type = "budget_impossibility_conflict"
            
            # Step 1: Require premium quality that forces expensive ingredients
            for j in 1:n_attributes
                # Find ingredients with highest quality for this attribute
                best_quality = maximum(attributes[:, j])
                lower_bounds[j] = max(min_attr, best_quality * 0.95)
                upper_bounds[j] = min(max_attr, best_quality * 1.05)
            end
            
            # Step 2: Identify which ingredients can meet these quality requirements
            qualifying_ingredients = Int[]
            for i in 1:n_ingredients
                can_qualify = true
                for j in 1:n_attributes
                    if attributes[i, j] < lower_bounds[j]
                        can_qualify = false
                        break
                    end
                end
                if can_qualify
                    push!(qualifying_ingredients, i)
                end
            end
            
            # Step 3: Calculate minimum cost to meet requirements
            if !isempty(qualifying_ingredients)
                min_cost_per_unit = minimum(costs[i] for i in qualifying_ingredients)
                min_total_cost = min_cost_per_unit * min_blend_amount
            else
                # If no single ingredient qualifies, use most expensive combination
                min_total_cost = maximum(costs) * min_blend_amount
            end
            
            # Step 4: Set budget below minimum needed cost
            cost_budget = min_total_cost * (0.6 + rand() * 0.3)  # 60-90% of minimum needed
            
            # Generous supply (not the limiting factor)
            for i in 1:n_ingredients
                supply_limits[i] = min_blend_amount * 3.0
            end
            
        elseif scenario == 3
            # Strategy 3: Impossible quality conflict
            scenario_type = "impossible_quality_conflict"
            
            # Step 1: Find attributes with natural trade-offs
            if n_attributes >= 2
                # Require very high values for multiple attributes simultaneously
                for j in 1:min(n_attributes, 3)  # Limit to first 3 attributes
                    max_possible = maximum(attributes[:, j])
                    # Require 98-100% of the maximum possible value
                    lower_bounds[j] = max(min_attr, max_possible * (0.98 + rand() * 0.02))
                    upper_bounds[j] = min(max_attr, lower_bounds[j] * 1.01)
                end
                
                # Other attributes get reasonable bounds
                for j in (min(n_attributes, 3) + 1):n_attributes
                    avg_val = sum(attributes[:, j]) / n_ingredients
                    lower_bounds[j] = max(min_attr, avg_val * 0.8)
                    upper_bounds[j] = min(max_attr, avg_val * 1.2)
                end
            end
            
            # Normal supply and cost (not limiting factors)
            for i in 1:n_ingredients
                supply_limits[i] = min_blend_amount * 2.0
            end
            cost_budget = maximum(costs) * min_blend_amount * 2.0
            
        else  # scenario == 4
            # Strategy 4: Over-constrained system
            scenario_type = "over_constrained_system"
            
            # Create multiple conflicting constraints
            
            # Quality constraints requiring different ingredient profiles
            for j in 1:n_attributes
                if j <= 2
                    # First two attributes: require near-maximum values
                    max_val = maximum(attributes[:, j])
                    lower_bounds[j] = max(min_attr, max_val * 0.97)
                    upper_bounds[j] = min(max_attr, max_val * 1.01)
                else
                    # Other attributes: require mid-range values
                    mid_val = (minimum(attributes[:, j]) + maximum(attributes[:, j])) / 2
                    lower_bounds[j] = max(min_attr, mid_val * 0.95)
                    upper_bounds[j] = min(max_attr, mid_val * 1.05)
                end
            end
            
            # Supply constraints that limit high-quality ingredients
            expensive_ingredients = sortperm(costs, rev=true)[1:max(2, div(n_ingredients, 3))]
            for i in expensive_ingredients
                supply_limits[i] = min_blend_amount * 0.2  # Very limited
            end
            
            # Others get normal supply
            for i in 1:n_ingredients
                if !(i in expensive_ingredients)
                    supply_limits[i] = min_blend_amount * 1.5
                end
            end
            
            # Tight budget constraint
            avg_cost = sum(costs) / n_ingredients
            cost_budget = avg_cost * min_blend_amount * 0.9  # Below average cost
        end
    end
    
    # Store generated data in params
    actual_params[:costs] = costs
    actual_params[:attributes] = attributes
    actual_params[:lower_bounds] = lower_bounds
    actual_params[:upper_bounds] = upper_bounds
    actual_params[:supply_limits] = supply_limits
    actual_params[:cost_budget] = cost_budget
    actual_params[:scenario_type] = scenario_type
    actual_params[:actual_solution_status] = actual_status
    
    # Model
    model = Model()
    
    # Variables
    @variable(model, x[1:n_ingredients] >= 0)  # Amount of each ingredient to use
    
    # Objective: Minimize total cost
    @objective(model, Min, sum(costs[i] * x[i] for i in 1:n_ingredients))
    
    # Constraints
    
    # Minimum blend amount
    @constraint(model, sum(x[i] for i in 1:n_ingredients) >= min_blend_amount)
    
    # Supply limits for each ingredient
    for i in 1:n_ingredients
        if supply_limits[i] < Inf
            @constraint(model, x[i] <= supply_limits[i])
        end
    end
    
    # Cost budget constraint
    if cost_budget < Inf
        @constraint(model, sum(costs[i] * x[i] for i in 1:n_ingredients) <= cost_budget)
    end
    
    # Additional realistic constraints (if in feasible mode)
    if haskey(actual_params, :min_usage_required)
        min_usage_required = actual_params[:min_usage_required]
        for (i, min_amount) in min_usage_required
            @constraint(model, x[i] >= min_amount)
        end
    end
    
    if haskey(actual_params, :max_usage_limits)
        max_usage_limits = actual_params[:max_usage_limits]
        for (i, max_amount) in max_usage_limits
            @constraint(model, x[i] <= max_amount)
        end
    end
    
    # Quality attribute bounds
    for j in 1:n_attributes
        # Lower bound on attribute j in the final blend
        @constraint(model, 
            sum(attributes[i, j] * x[i] for i in 1:n_ingredients) >= 
            lower_bounds[j] * sum(x[i] for i in 1:n_ingredients)
        )
        
        # Upper bound on attribute j in the final blend
        @constraint(model, 
            sum(attributes[i, j] * x[i] for i in 1:n_ingredients) <= 
            upper_bounds[j] * sum(x[i] for i in 1:n_ingredients)
        )
    end
    
    return model, actual_params
end

"""
    sample_blending_parameters(target_variables::Int; seed::Int=0, solution_status::Symbol=:feasible)

Sample realistic parameters for a blending problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)
- `solution_status`: Solution status (:feasible, :infeasible, or :all) (default: :feasible)

# Returns
- Dictionary of sampled parameters
"""
function sample_blending_parameters(target_variables::Int; seed::Int=0, solution_status::Symbol=:feasible)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Start with defaults
    params[:n_ingredients] = max(3, min(500, target_variables))  # Direct mapping since variables = n_ingredients
    params[:n_attributes] = rand(2:15)  # Doesn't affect variable count but scale with complexity
    params[:min_blend_amount] = rand(100:20000)  # Scale with problem size
    
    # Iteratively adjust if needed (though for blending, it's direct)
    for iteration in 1:5
        current_vars = calculate_blending_variable_count(params)
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust n_ingredients directly since it's the only parameter affecting variable count
        if current_vars < target_variables
            params[:n_ingredients] = min(500, params[:n_ingredients] + 1)
        elseif current_vars > target_variables
            params[:n_ingredients] = max(3, params[:n_ingredients] - 1)
        end
    end
    
    # These parameters are not strongly size-dependent
    params[:cost_range] = (10, 100)
    params[:attribute_range] = (0.1, 0.9)
    
    # Include solution status
    params[:solution_status] = solution_status
    
    return params
end

"""
    sample_blending_parameters(size::Symbol=:medium; seed::Int=0, solution_status::Symbol=:feasible)

Sample realistic parameters for a blending problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)
- `solution_status`: Solution status (:feasible, :infeasible, or :all) (default: :feasible)

# Returns
- Dictionary of sampled parameters
"""
function sample_blending_parameters(size::Symbol=:medium; seed::Int=0, solution_status::Symbol=:feasible)
    # Map size categories to approximate target variable counts
    target_map = Dict(
        :small => rand(50:250),
        :medium => rand(250:1000),
        :large => rand(1000:10000)
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_blending_parameters(target_map[size]; seed=seed, solution_status=solution_status)
end

"""
    calculate_blending_variable_count(params::Dict)

Calculate the number of variables in a blending problem.

# Arguments
- `params`: Dictionary of problem parameters

# Returns
- Number of variables in the problem
"""
function calculate_blending_variable_count(params::Dict)
    # Extract the number of ingredients parameter
    n_ingredients = get(params, :n_ingredients, 5)
    
    # The problem has one variable for each ingredient: x[1:n_ingredients]
    return n_ingredients
end

# Register the problem type
register_problem(
    :blending,
    generate_blending_problem,
    sample_blending_parameters,
    "Blending problem that minimizes cost while meeting quality requirements for a mixture"
)