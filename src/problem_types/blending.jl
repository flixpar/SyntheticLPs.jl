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
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_ingredients => n_ingredients,
        :n_attributes => n_attributes,
        :cost_range => cost_range,
        :attribute_range => attribute_range,
        :min_blend_amount => min_blend_amount
    )
    
    # Random data generation
    min_cost, max_cost = cost_range
    costs = rand(min_cost:max_cost, n_ingredients)  # Cost per unit of each ingredient
    
    min_attr, max_attr = attribute_range
    attributes = rand(min_attr:0.01:max_attr, n_ingredients, n_attributes)  # Attribute values for each ingredient
    
    # Generate target attribute ranges
    # These are weighted averages of the ingredient attributes to ensure feasibility
    lower_bounds = zeros(n_attributes)
    upper_bounds = zeros(n_attributes)
    
    for j in 1:n_attributes
        # Weighted average of attribute values with random weights
        weights = rand(n_ingredients)
        weights ./= sum(weights)
        target = sum(attributes[i, j] * weights[i] for i in 1:n_ingredients)
        
        # Set bounds around the target value
        lower_bounds[j] = max(min_attr, target * 0.8)  # Lower bound 20% below target
        upper_bounds[j] = min(max_attr, target * 1.2)  # Upper bound 20% above target
    end
    
    # Store generated data in params
    actual_params[:costs] = costs
    actual_params[:attributes] = attributes
    actual_params[:lower_bounds] = lower_bounds
    actual_params[:upper_bounds] = upper_bounds
    
    # Model
    model = Model()
    
    # Variables
    @variable(model, x[1:n_ingredients] >= 0)  # Amount of each ingredient to use
    
    # Objective: Minimize total cost
    @objective(model, Min, sum(costs[i] * x[i] for i in 1:n_ingredients))
    
    # Constraints
    
    # Minimum blend amount
    @constraint(model, sum(x[i] for i in 1:n_ingredients) >= min_blend_amount)
    
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
    sample_blending_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a blending problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_blending_parameters(target_variables::Int; seed::Int=0)
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
    
    return params
end

"""
    sample_blending_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a blending problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_blending_parameters(size::Symbol=:medium; seed::Int=0)
    # Map size categories to approximate target variable counts
    target_map = Dict(
        :small => rand(50:250),
        :medium => rand(250:1000),
        :large => rand(1000:10000)
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_blending_parameters(target_map[size]; seed=seed)
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