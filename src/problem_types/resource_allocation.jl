using JuMP
using Random

"""
    generate_resource_allocation_problem(params::Dict=Dict(); seed::Int=0)

Generate a resource allocation optimization problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_activities`: Number of activities to allocate resources to (default: 10)
  - `:n_resources`: Number of resource types (default: 5)
  - `:min_resource`: Minimum amount of each resource available (default: 50)
  - `:max_resource`: Maximum amount of each resource available (default: 200)
  - `:min_profit`: Minimum profit per unit of activity (default: 0.5)
  - `:max_profit`: Maximum profit per unit of activity (default: 20.0)
  - `:min_usage`: Minimum resource usage per unit of activity (default: 0.1)
  - `:max_usage`: Maximum resource usage per unit of activity (default: 5.0)
  - `:correlation_strength`: Correlation between resource usage and profit (default: 0.7)
  - `:add_min_constraints`: Whether to add minimum activity level constraints (default: true)
  - `:min_level_prob`: Probability of having a minimum level constraint (default: 0.3)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_resource_allocation_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_activities = get(params, :n_activities, 10)
    n_resources = get(params, :n_resources, 5)
    min_resource = get(params, :min_resource, 50)
    max_resource = get(params, :max_resource, 200)
    min_profit = get(params, :min_profit, 0.5)
    max_profit = get(params, :max_profit, 20.0)
    min_usage = get(params, :min_usage, 0.1)
    max_usage = get(params, :max_usage, 5.0)
    correlation_strength = get(params, :correlation_strength, 0.7)
    add_min_constraints = get(params, :add_min_constraints, true)
    min_level_prob = get(params, :min_level_prob, 0.3)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_activities => n_activities,
        :n_resources => n_resources,
        :min_resource => min_resource,
        :max_resource => max_resource,
        :min_profit => min_profit,
        :max_profit => max_profit,
        :min_usage => min_usage,
        :max_usage => max_usage,
        :correlation_strength => correlation_strength,
        :add_min_constraints => add_min_constraints,
        :min_level_prob => min_level_prob
    )
    
    # Generate data with correlation between profit and resource usage
    
    # Generate "quality factors" for each activity
    quality_factors = rand(n_activities)
    
    # Generate profits correlated with quality factors
    base_profit = rand(n_activities) .* (max_profit - min_profit) .+ min_profit
    quality_profit = quality_factors .* (max_profit - min_profit)
    profits = base_profit + correlation_strength * quality_profit
    
    # Generate resource usage with correlation to quality
    usage = zeros(n_activities, n_resources)
    
    for i in 1:n_activities
        for j in 1:n_resources
            # Base usage level
            base_usage = rand() * (max_usage - min_usage) + min_usage
            
            # Add correlation with quality
            quality_usage = quality_factors[i] * (max_usage - min_usage)
            
            # Combined usage
            usage[i, j] = base_usage + correlation_strength * quality_usage
        end
    end
    
    # Generate resource availability levels
    # Calculate expected usage for each resource if activities are balanced
    expected_usage = sum(usage, dims=1) / n_activities
    
    # Set resources above expected usage to ensure feasibility
    resources = vec(expected_usage) .* rand(n_resources) .* n_activities ./ 2
    
    # Ensure resources are within bounds
    resources = max.(resources, min_resource)
    resources = min.(resources, max_resource)
    
    # Calculate minimum activity levels (for some activities)
    min_levels = zeros(n_activities)
    
    if add_min_constraints
        for i in 1:n_activities
            if rand() < min_level_prob
                # Set minimum level at 10-30% of maximum possible
                max_possible = minimum([resources[j] / usage[i, j] for j in 1:n_resources])
                min_levels[i] = rand(0.1:0.05:0.3) * max_possible
            end
        end
    end
    
    # Store generated data in params
    actual_params[:profits] = profits
    actual_params[:usage] = usage
    actual_params[:resources] = resources
    actual_params[:min_levels] = min_levels
    actual_params[:quality_factors] = quality_factors
    
    # Create model
    model = Model()
    
    # Decision variables: activity levels
    @variable(model, x[1:n_activities] >= 0)
    
    # Resource constraints
    for j in 1:n_resources
        @constraint(model, sum(usage[i,j] * x[i] for i in 1:n_activities) <= resources[j])
    end
    
    # Minimum level constraints
    for i in 1:n_activities
        if min_levels[i] > 0
            @constraint(model, x[i] >= min_levels[i])
        end
    end
    
    # Objective: maximize profit
    @objective(model, Max, sum(profits[i] * x[i] for i in 1:n_activities))
    
    return model, actual_params
end

"""
    sample_resource_allocation_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a resource allocation problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_resource_allocation_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Start with defaults - target_variables = n_activities
    params[:n_activities] = max(3, min(2000, target_variables))
    params[:n_resources] = rand(2:50)  # Scale with problem complexity
    params[:min_resource] = rand(50:1000)
    params[:max_resource] = rand(200:10000)
    
    # Iteratively adjust if needed (though for resource allocation, it's direct)
    for iteration in 1:5
        current_vars = calculate_resource_allocation_variable_count(params)
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust n_activities directly since it's the only parameter affecting variable count
        if current_vars < target_variables
            params[:n_activities] = min(2000, params[:n_activities] + 1)
        elseif current_vars > target_variables
            params[:n_activities] = max(3, params[:n_activities] - 1)
        end
    end
    
    # Parameters that scale with problem complexity
    params[:min_profit] = rand(0.1:0.1:2.0)
    params[:max_profit] = rand(5.0:5.0:100.0)
    params[:min_usage] = rand(0.01:0.01:0.5)
    params[:max_usage] = rand(1.0:1.0:20.0)
    params[:correlation_strength] = rand(0.5:0.1:0.9)
    params[:add_min_constraints] = rand() < 0.7  # 70% chance
    params[:min_level_prob] = rand(0.2:0.1:0.5)
    
    return params
end

"""
    sample_resource_allocation_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a resource allocation problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_resource_allocation_parameters(size::Symbol=:medium; seed::Int=0)
    # Map size categories to approximate target variable counts
    target_map = Dict(
        :small => rand(50:250),
        :medium => rand(250:1000),
        :large => rand(1000:10000)
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_resource_allocation_parameters(target_map[size]; seed=seed)
end

"""
    calculate_resource_allocation_variable_count(params::Dict)

Calculate the number of variables in a resource allocation problem.

# Arguments
- `params`: Dictionary of problem parameters containing :n_activities

# Returns
- Number of variables (equal to number of activities)
"""
function calculate_resource_allocation_variable_count(params::Dict)
    n_activities = get(params, :n_activities, 10)
    return n_activities
end

# Register the problem type
register_problem(
    :resource_allocation,
    generate_resource_allocation_problem,
    sample_resource_allocation_parameters,
    "Resource allocation problem that maximizes profit by allocating limited resources to competing activities"
)