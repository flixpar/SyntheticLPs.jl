using JuMP
using Random


"""
    generate_production_planning_problem(params::Dict=Dict(); seed::Int=0)

Generate a production planning problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_products`: Number of products (default: 4)
  - `:n_resources`: Number of resources (default: 3)
  - `:profit_range`: Tuple (min, max) for profits per unit (default: (20, 100))
  - `:usage_range`: Tuple (min, max) for resource usage per unit (default: (1, 10))
  - `:resource_factor`: Factor to determine resource availability (default: 0.6)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_production_planning_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_products = get(params, :n_products, 4)
    n_resources = get(params, :n_resources, 3)
    profit_range = get(params, :profit_range, (20, 100))
    usage_range = get(params, :usage_range, (1, 10))
    resource_factor = get(params, :resource_factor, 0.6)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_products => n_products,
        :n_resources => n_resources,
        :profit_range => profit_range,
        :usage_range => usage_range,
        :resource_factor => resource_factor
    )
    
    # Random data generation
    min_profit, max_profit = profit_range
    profits = rand(min_profit:max_profit, n_products)  # Profits per unit of product
    
    min_usage, max_usage = usage_range
    usage = rand(min_usage:max_usage, n_products, n_resources)  # Resource usage per unit of product
    
    # Calculate resource availability to ensure feasibility
    resources = sum(usage, dims=1)[:] * resource_factor
    
    # Store generated data in params
    actual_params[:profits] = profits
    actual_params[:usage] = usage
    actual_params[:resources] = resources
    
    # Model
    model = Model()
    
    # Variables
    @variable(model, x[1:n_products] >= 0)
    
    # Objective
    @objective(model, Max, sum(profits[i] * x[i] for i in 1:n_products))
    
    # Constraints
    for j in 1:n_resources
        @constraint(model, sum(usage[i, j] * x[i] for i in 1:n_products) <= resources[j])
    end
    
    return model, actual_params
end

"""
    sample_production_planning_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a production planning problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_production_planning_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Start with defaults - target_variables = n_products
    params[:n_products] = max(2, min(2000, target_variables))
    params[:n_resources] = rand(1:50)  # Scale with problem complexity
    
    # Iteratively adjust if needed (though for production planning, it's direct)
    for iteration in 1:5
        current_vars = calculate_production_planning_variable_count(params)
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust n_products directly since it's the only parameter affecting variable count
        if current_vars < target_variables
            params[:n_products] = min(2000, params[:n_products] + 1)
        elseif current_vars > target_variables
            params[:n_products] = max(2, params[:n_products] - 1)
        end
    end
    
    # These parameters scale with problem complexity
    params[:profit_range] = (10, 500)
    params[:usage_range] = (0.1, 50.0)
    params[:resource_factor] = rand(0.4:0.1:0.8)
    
    return params
end

"""
    sample_production_planning_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a production planning problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_production_planning_parameters(size::Symbol=:medium; seed::Int=0)
    # Map size categories to approximate target variable counts
    target_map = Dict(
        :small => rand(50:250),
        :medium => rand(250:1000),
        :large => rand(1000:10000)
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_production_planning_parameters(target_map[size]; seed=seed)
end

"""
    calculate_production_planning_variable_count(params::Dict)

Calculate the total number of variables for a production planning problem.

# Arguments
- `params`: Dictionary of problem parameters

# Returns
- Integer: Total number of variables in the problem
"""
function calculate_production_planning_variable_count(params::Dict)
    # Extract parameters with defaults
    n_products = get(params, :n_products, 4)
    
    # Variables: x[1:n_products] >= 0
    return n_products
end

# Register the problem type
register_problem(
    :production_planning,
    generate_production_planning_problem,
    sample_production_planning_parameters,
    "Production planning problem that maximizes profit subject to resource constraints"
)