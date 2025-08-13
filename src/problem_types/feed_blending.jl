using JuMP
using Random
using Distributions
using Statistics

"""
    generate_feed_blending_problem(params::Dict=Dict(); seed::Int=0)

Generate a feed blending (diet) problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:num_ingredients`: Number of ingredients available for the blend (default: 10)
  - `:num_nutrients`: Number of nutrients to consider in constraints (default: 8)
  - `:batch_size`: Required total batch size (default: 1000.0)
  - `:min_requirement_factor`: Factor to control nutrient minimum requirements (default: 0.4)
  - `:max_limit_factor`: Factor to control nutrient maximum limits (default: 1.5)
  - `:availability_prob`: Probability of an ingredient having an availability constraint (default: 0.3)
  - `:ratio_constraint_prob`: Probability of adding ratio constraints (default: 0.2)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_feed_blending_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    rng = Random.MersenneTwister(seed)
    
    # Extract parameters with defaults
    num_ingredients = get(params, :num_ingredients, 10)
    num_nutrients = get(params, :num_nutrients, 8)
    batch_size = get(params, :batch_size, 1000.0)
    min_requirement_factor = get(params, :min_requirement_factor, 0.4)
    max_limit_factor = get(params, :max_limit_factor, 1.5)
    availability_prob = get(params, :availability_prob, 0.3)
    ratio_constraint_prob = get(params, :ratio_constraint_prob, 0.2)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :num_ingredients => num_ingredients,
        :num_nutrients => num_nutrients,
        :batch_size => batch_size,
        :min_requirement_factor => min_requirement_factor,
        :max_limit_factor => max_limit_factor,
        :availability_prob => availability_prob,
        :ratio_constraint_prob => ratio_constraint_prob
    )
    
    # ---- Generate parameters ----
    
    # 1. Ingredient costs: lognormal distribution to represent market variability
    # Scale costs based on problem size to reflect different market contexts
    if num_ingredients <= 250
        # Small scale: Local/regional ingredients, higher variability
        cost_mu = log(4.0)   # median ≈ $4 per unit (higher local costs)
        cost_sigma = 0.8     # higher variability for local markets
    elseif num_ingredients <= 1000
        # Medium scale: Commercial sourcing, moderate variability
        cost_mu = log(2.5)   # median ≈ $2.5 per unit (bulk pricing)
        cost_sigma = 0.6     # moderate variability
    else
        # Large scale: Industrial sourcing, commodity pricing
        cost_mu = log(1.8)   # median ≈ $1.8 per unit (commodity prices)
        cost_sigma = 0.4     # lower variability for commodity markets
    end
    
    costs = exp.(rand(rng, Normal(cost_mu, cost_sigma), num_ingredients))
    
    # 2. Nutrient content matrix: amount of each nutrient j in each ingredient i
    # Values often represent percentages or concentrations and vary by nutrient type
    
    # Create a matrix of nutrient contents with some structure and sparsity
    nutrient_content = zeros(num_nutrients, num_ingredients)
    
    # Define different nutrient types with different scales and distributions
    nutrient_types = rand(rng, 1:4, num_nutrients)
    
    for j in 1:num_nutrients
        if nutrient_types[j] == 1
            # Type 1: Major nutrients (e.g., protein, energy) - most ingredients contain these
            # Values might be percentages like 10-30%
            for i in 1:num_ingredients
                nutrient_content[j, i] = max(0, rand(rng, Normal(20.0, 7.0)))
                
                # Some ingredients might be especially rich or poor in this nutrient
                if rand(rng) < 0.15
                    nutrient_content[j, i] *= rand(rng, Uniform(1.5, 2.5))
                elseif rand(rng) < 0.15
                    nutrient_content[j, i] *= rand(rng, Uniform(0.2, 0.6))
                end
            end
            
        elseif nutrient_types[j] == 2
            # Type 2: Minor nutrients (e.g., specific minerals) - moderate sparsity
            # Some ingredients might not contain these at all
            for i in 1:num_ingredients
                if rand(rng) < 0.7  # 70% chance of containing this nutrient
                    nutrient_content[j, i] = max(0, rand(rng, Normal(2.0, 1.0)))
                    
                    # Some ingredients might be especially rich in this nutrient
                    if rand(rng) < 0.2
                        nutrient_content[j, i] *= rand(rng, Uniform(2.0, 5.0))
                    end
                end
            end
            
        elseif nutrient_types[j] == 3
            # Type 3: Trace nutrients (e.g., vitamins) - higher sparsity
            # Only some ingredients contain significant amounts
            for i in 1:num_ingredients
                if rand(rng) < 0.3  # 30% chance of containing this nutrient
                    # Small values, possibly measured in mg/kg or similar
                    nutrient_content[j, i] = max(0, rand(rng, Normal(0.5, 0.3)))
                    
                    # Some ingredients might be especially rich sources
                    if rand(rng) < 0.25
                        nutrient_content[j, i] *= rand(rng, Uniform(3.0, 10.0))
                    end
                end
            end
            
        else  # type 4
            # Type 4: Anti-nutrients or upper-limited compounds (e.g., fiber, toxins)
            # These often need maximum constraints
            for i in 1:num_ingredients
                if rand(rng) < 0.6  # 60% chance
                    nutrient_content[j, i] = max(0, rand(rng, Normal(5.0, 3.0)))
                    
                    # Some ingredients might be especially high in these
                    if rand(rng) < 0.2
                        nutrient_content[j, i] *= rand(rng, Uniform(1.5, 3.0))
                    end
                end
            end
        end
    end
    
    # Make sure every nutrient exists in at least one ingredient and every ingredient contains at least one nutrient
    for j in 1:num_nutrients
        if all(nutrient_content[j, :] .== 0)
            # If no ingredient contains this nutrient, add it to some random ingredients
            for _ in 1:max(1, ceil(Int, 0.2 * num_ingredients))
                i = rand(rng, 1:num_ingredients)
                nutrient_content[j, i] = max(0, rand(rng, Normal(2.0, 1.0)))
            end
        end
    end
    
    for i in 1:num_ingredients
        if all(nutrient_content[:, i] .== 0)
            # If this ingredient contains no nutrients, add some random nutrients
            for _ in 1:max(1, ceil(Int, 0.2 * num_nutrients))
                j = rand(rng, 1:num_nutrients)
                nutrient_content[j, i] = max(0, rand(rng, Normal(2.0, 1.0)))
            end
        end
    end
    
    # 3. Nutrient requirements
    # Determine achievable ranges based on the generated nutrient content
    
    # First, find minimum and maximum possible for each nutrient if we used 100% of a single ingredient
    max_possible_nutrients = zeros(num_nutrients)
    for j in 1:num_nutrients
        # Maximum amount of nutrient j possible in batch_size
        max_per_ingredient = [nutrient_content[j, i] * batch_size for i in 1:num_ingredients]
        max_possible_nutrients[j] = maximum(max_per_ingredient)
    end
    
    # Now generate minimum requirements and maximum limits
    min_requirements = zeros(num_nutrients)
    max_limits = fill(Inf, num_nutrients)
    
    for j in 1:num_nutrients
        # Based on nutrient type, decide whether to have min, max, or both constraints
        if nutrient_types[j] == 1  # Major nutrients - typically have minimum requirements
            # Set minimum requirement between 20-60% of maximum possible
            min_requirements[j] = rand(rng, Uniform(0.2, 0.6)) * max_possible_nutrients[j] * min_requirement_factor
            
            # Some may also have maximum limits (80-120% of minimum)
            if rand(rng) < 0.3
                max_limits[j] = min_requirements[j] * rand(rng, Uniform(1.2, 2.0)) * max_limit_factor
            end
            
        elseif nutrient_types[j] == 2  # Minor nutrients - often have minimum requirements
            # Set minimum requirement between 10-50% of maximum possible
            min_requirements[j] = rand(rng, Uniform(0.1, 0.5)) * max_possible_nutrients[j] * min_requirement_factor
            
            # Rarely have maximum limits
            if rand(rng) < 0.2
                max_limits[j] = min_requirements[j] * rand(rng, Uniform(1.5, 3.0)) * max_limit_factor
            end
            
        elseif nutrient_types[j] == 3  # Trace nutrients - some have minimum requirements
            if rand(rng) < 0.7  # 70% chance of minimum requirement
                # Set minimum requirement between 5-40% of maximum possible
                min_requirements[j] = rand(rng, Uniform(0.05, 0.4)) * max_possible_nutrients[j] * min_requirement_factor
            end
            
            # Very rarely have maximum limits
            if rand(rng) < 0.1
                max_limits[j] = min_requirements[j] > 0 ? 
                               min_requirements[j] * rand(rng, Uniform(2.0, 5.0)) * max_limit_factor : 
                               rand(rng, Uniform(0.1, 0.3)) * max_possible_nutrients[j] * max_limit_factor
            end
            
        else  # Type 4 - anti-nutrients or upper-limited compounds
            # Usually have maximum limits but no minimum requirements
            if rand(rng) < 0.8  # 80% chance of maximum limit
                # Set maximum limit between 20-70% of maximum possible
                max_limits[j] = rand(rng, Uniform(0.2, 0.7)) * max_possible_nutrients[j] * max_limit_factor
            end
            
            # Very rarely have minimum requirements
            if rand(rng) < 0.1
                min_requirements[j] = rand(rng, Uniform(0.05, 0.2)) * 
                                     (max_limits[j] < Inf ? max_limits[j] : max_possible_nutrients[j]) * min_requirement_factor
            end
        end
    end
    
    # 4. Ingredient availabilities (optional)
    # For some ingredients, there might be availability constraints
    availabilities = fill(Inf, num_ingredients)
    
    # Add availability constraints for some ingredients (based on availability_prob)
    # Scale availability patterns based on problem size
    for i in 1:num_ingredients
        if rand(rng) < availability_prob
            if num_ingredients <= 250
                # Small scale: Tight local supply constraints
                availabilities[i] = rand(rng, truncated(Normal(0.4, 0.15), 0.1, 0.8)) * batch_size
            elseif num_ingredients <= 1000
                # Medium scale: Moderate supply constraints
                availabilities[i] = rand(rng, truncated(Normal(0.6, 0.2), 0.2, 1.2)) * batch_size
            else
                # Large scale: Diverse supply sources, some very constrained, some abundant
                if rand(rng) < 0.3
                    # 30% chance of very constrained specialty ingredients
                    availabilities[i] = rand(rng, truncated(Normal(0.2, 0.1), 0.05, 0.5)) * batch_size
                else
                    # 70% chance of more abundant commodity ingredients
                    availabilities[i] = rand(rng, truncated(Normal(0.8, 0.3), 0.3, 2.0)) * batch_size
                end
            end
        end
    end
    
    # Store generated data in params
    actual_params[:costs] = costs
    actual_params[:nutrient_content] = nutrient_content
    actual_params[:nutrient_types] = nutrient_types
    actual_params[:min_requirements] = min_requirements
    actual_params[:max_limits] = max_limits
    actual_params[:availabilities] = availabilities
    actual_params[:max_possible_nutrients] = max_possible_nutrients
    
    # 5. Create JuMP model
    model = Model()
    
    # Add decision variables: amount of each ingredient to use
    @variable(model, x[1:num_ingredients] >= 0)
    
    # Set objective: minimize total cost
    @objective(model, Min, sum(costs[i] * x[i] for i in 1:num_ingredients))
    
    # Add batch size constraint
    @constraint(model, sum(x[i] for i in 1:num_ingredients) == batch_size)
    
    # Add nutritional requirement constraints
    for j in 1:num_nutrients
        if min_requirements[j] > 0
            @constraint(model, sum(nutrient_content[j, i] * x[i] for i in 1:num_ingredients) >= min_requirements[j])
        end
        
        if max_limits[j] < Inf
            @constraint(model, sum(nutrient_content[j, i] * x[i] for i in 1:num_ingredients) <= max_limits[j])
        end
    end
    
    # Add availability constraints
    for i in 1:num_ingredients
        if availabilities[i] < Inf
            @constraint(model, x[i] <= availabilities[i])
        end
    end
    
    # Add ratio constraints (linearized) for selected nutrients based on ratio_constraint_prob
    if rand(rng) < ratio_constraint_prob
        # Determine number of ratio constraints to add (up to 30% of nutrients)
        num_ratio_constraints = rand(rng, 1:ceil(Int, 0.3 * num_nutrients))
        ratio_constraints = []
        
        # Select random nutrients for ratio constraints
        nutrient_indices = sample(rng, 1:num_nutrients, min(num_ratio_constraints, num_nutrients), replace=false)
        
        for j in nutrient_indices
            # Only add ratio constraint if this nutrient exists in ingredients
            if any(nutrient_content[j, :] .> 0)
                # Determine if it's a minimum or maximum percentage
                is_min = rand(rng) < 0.7  # 70% chance for minimum ratio
                
                # Calculate achievable range
                positive_values = filter(x -> x > 0, nutrient_content[j, :])
                if !isempty(positive_values)
                    max_percentage = maximum(nutrient_content[j, :])
                    min_percentage = minimum(positive_values)
                    
                    # Set target percentage within achievable range
                    if is_min
                        # Target percentage between 20-80% of maximum
                        target_pct = rand(rng, Uniform(0.2, 0.8)) * max_percentage
                        
                        # Linearized constraint: Σ (a_ji - target_pct) * x_i >= 0
                        @constraint(model, sum((nutrient_content[j, i] - target_pct) * x[i] for i in 1:num_ingredients) >= 0)
                        push!(ratio_constraints, (j, target_pct, "min"))
                    else
                        # Target percentage between 120-180% of minimum
                        target_pct = rand(rng, Uniform(1.2, 1.8)) * min_percentage
                        
                        # Linearized constraint: Σ (a_ji - target_pct) * x_i <= 0
                        @constraint(model, sum((nutrient_content[j, i] - target_pct) * x[i] for i in 1:num_ingredients) <= 0)
                        push!(ratio_constraints, (j, target_pct, "max"))
                    end
                end
            end
        end
        
        # Store ratio constraints
        actual_params[:ratio_constraints] = ratio_constraints
    else
        actual_params[:ratio_constraints] = []
    end
    
    return model, actual_params
end

"""
    sample_feed_blending_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a feed blending problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_feed_blending_parameters(target_variables::Int; seed::Int=0)
    rng = Random.MersenneTwister(seed)
    
    params = Dict{Symbol, Any}()
    
    # Start with target_variables = num_ingredients
    params[:num_ingredients] = max(3, target_variables)
    
    # Scale parameters based on problem size to reflect realistic feed blending scenarios
    if target_variables <= 250
        # Small scale: Small farm or local feed mill
        params[:num_nutrients] = rand(rng, 4:8)  # Basic nutritional requirements
        params[:batch_size] = rand(rng, truncated(Normal(500.0, 200.0), 100.0, 2000.0))
        params[:min_requirement_factor] = rand(rng, truncated(Normal(0.4, 0.1), 0.2, 0.6))
        params[:max_limit_factor] = rand(rng, truncated(Normal(1.4, 0.2), 1.1, 1.8))
        params[:availability_prob] = rand(rng, truncated(Normal(0.25, 0.1), 0.1, 0.4))
        params[:ratio_constraint_prob] = rand(rng, truncated(Normal(0.15, 0.05), 0.05, 0.25))
        
    elseif target_variables <= 1000
        # Medium scale: Commercial feed mill
        params[:num_nutrients] = rand(rng, 6:12)  # More complex nutritional profiles
        params[:batch_size] = rand(rng, truncated(Normal(2000.0, 800.0), 500.0, 10000.0))
        params[:min_requirement_factor] = rand(rng, truncated(Normal(0.35, 0.1), 0.2, 0.5))
        params[:max_limit_factor] = rand(rng, truncated(Normal(1.5, 0.2), 1.2, 2.0))
        params[:availability_prob] = rand(rng, truncated(Normal(0.3, 0.1), 0.15, 0.45))
        params[:ratio_constraint_prob] = rand(rng, truncated(Normal(0.2, 0.05), 0.1, 0.3))
        
    else
        # Large scale: Industrial feed production
        params[:num_nutrients] = rand(rng, 8:20)  # Complex nutritional requirements
        params[:batch_size] = rand(rng, truncated(Normal(10000.0, 5000.0), 2000.0, 50000.0))
        params[:min_requirement_factor] = rand(rng, truncated(Normal(0.3, 0.1), 0.15, 0.45))
        params[:max_limit_factor] = rand(rng, truncated(Normal(1.6, 0.25), 1.3, 2.2))
        params[:availability_prob] = rand(rng, truncated(Normal(0.35, 0.1), 0.2, 0.5))
        params[:ratio_constraint_prob] = rand(rng, truncated(Normal(0.25, 0.05), 0.15, 0.35))
    end
    
    # Iteratively adjust if needed (though for feed blending, it's direct)
    for iteration in 1:5
        current_vars = calculate_feed_blending_variable_count(params)
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust num_ingredients directly since it's the only parameter affecting variable count
        if current_vars < target_variables
            params[:num_ingredients] = params[:num_ingredients] + 1
        elseif current_vars > target_variables
            params[:num_ingredients] = max(3, params[:num_ingredients] - 1)
        end
    end
    
    return params
end

"""
    sample_feed_blending_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a feed blending problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_feed_blending_parameters(size::Symbol=:medium; seed::Int=0)
    rng = Random.MersenneTwister(seed)
    
    # Map size categories to realistic target variable counts
    target_map = Dict(
        :small => rand(rng, 50:250),      # Small farm or local feed mill
        :medium => rand(rng, 250:1000),   # Commercial feed mill
        :large => rand(rng, 1000:10000)   # Industrial feed production
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_feed_blending_parameters(target_map[size]; seed=seed)
end

"""
    calculate_feed_blending_variable_count(params::Dict)

Calculate the number of variables in a feed blending problem.

# Arguments
- `params`: Dictionary of problem parameters

# Returns
- Number of variables in the problem
"""
function calculate_feed_blending_variable_count(params::Dict)
    # Extract the number of ingredients parameter
    num_ingredients = get(params, :num_ingredients, 10)
    
    # The problem has one variable for each ingredient: x[1:num_ingredients]
    return num_ingredients
end

# Register the problem type
register_problem(
    :feed_blending,
    generate_feed_blending_problem,
    sample_feed_blending_parameters,
    "Feed blending (diet) problem that finds the least-cost mixture of ingredients while satisfying nutritional requirements"
)