using JuMP
using Random


"""
    generate_knapsack_problem(params::Dict=Dict(); seed::Int=0)

Generate a knapsack problem instance (fractional).

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_items`: Number of items (default: 6)
  - `:capacity`: Knapsack capacity (default: 50)
  - `:value_range`: Tuple (min, max) for item values (default: (10, 100))
  - `:weight_range`: Tuple (min, max) for item weights (default: (5, 20))
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_knapsack_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_items = get(params, :n_items, 150)
    capacity = get(params, :capacity, 500)
    value_range = get(params, :value_range, (10, 100))
    weight_range = get(params, :weight_range, (5, 20))
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_items => n_items,
        :capacity => capacity,
        :value_range => value_range,
        :weight_range => weight_range
    )
    
    # Random data generation
    min_value, max_value = value_range
    values = rand(min_value:max_value, n_items)
    
    min_weight, max_weight = weight_range
    weights = rand(min_weight:max_weight, n_items)
    
    # Store generated data in params
    actual_params[:values] = values
    actual_params[:weights] = weights
    
    # Model
    model = Model()
    
    # Variables - using fractional knapsack version
    @variable(model, 0 <= x[1:n_items] <= 1)
    
    # Objective
    @objective(model, Max, sum(values[i] * x[i] for i in 1:n_items))
    
    # Constraint
    @constraint(model, sum(weights[i] * x[i] for i in 1:n_items) <= capacity)
    
    return model, actual_params
end

"""
    sample_knapsack_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a knapsack problem with target number of variables.

# Arguments
- `target_variables`: Target number of variables (items)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_knapsack_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # For knapsack, target_variables = n_items
    params[:n_items] = target_variables
    
    # Scale capacity based on number of items to maintain reasonable problem structure
    # Capacity should be tight enough to make the problem challenging
    # Average weight per item is ~12.5 (midpoint of 5-20)
    total_avg_weight = target_variables * 12.5
    
    # Set capacity to be 30-70% of total average weight for interesting problems
    capacity_ratio = 0.3 + rand() * 0.4  # Random between 0.3 and 0.7
    params[:capacity] = round(Int, total_avg_weight * capacity_ratio)
    
    # Add some variability to ensure capacity is not too predictable
    params[:capacity] = max(1, params[:capacity] + rand(-50:50))
    
    # Make value and weight ranges more diverse for larger problems
    if target_variables <= 100
        params[:value_range] = (rand(5:20), rand(80:150))
        params[:weight_range] = (rand(3:8), rand(15:25))
    elseif target_variables <= 1000
        params[:value_range] = (rand(10:30), rand(100:300))
        params[:weight_range] = (rand(5:15), rand(20:40))
    else
        # Large problems - more extreme ranges for diversity
        params[:value_range] = (rand(20:50), rand(200:500))
        params[:weight_range] = (rand(10:25), rand(30:60))
    end
    
    return params
end

"""
    sample_knapsack_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a knapsack problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_knapsack_parameters(size::Symbol=:medium; seed::Int=0)
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
    return sample_knapsack_parameters(target_variables; seed=seed)
end

"""
    calculate_knapsack_variable_count(params::Dict)

Calculate the number of variables for a knapsack problem.

# Arguments
- `params`: Dictionary of problem parameters containing `:n_items`

# Returns
- Number of variables (equal to n_items)
"""
function calculate_knapsack_variable_count(params::Dict)
    n_items = get(params, :n_items, 150)
    return n_items
end

# Register the problem type
register_problem(
    :knapsack,
    generate_knapsack_problem,
    sample_knapsack_parameters,
    "Knapsack problem that maximizes the value of items selected under a weight constraint"
)