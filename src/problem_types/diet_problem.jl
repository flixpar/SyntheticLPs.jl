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
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_foods => n_foods,
        :n_nutrients => n_nutrients,
        :cost_range => cost_range,
        :nutrient_range => nutrient_range,
        :requirement_factor => requirement_factor
    )
    
    # Random data generation
    min_cost, max_cost = cost_range
    c = rand(min_cost:0.1:max_cost, n_foods)  # Costs per unit of food
    
    min_nutrient, max_nutrient = nutrient_range
    a = rand(min_nutrient:0.1:max_nutrient, n_foods, n_nutrients)  # Nutrient content per unit of food
    b = sum(a, dims=1)[:] * requirement_factor  # Minimum nutrient requirements (ensure feasibility)
    
    # Store generated data in params
    actual_params[:costs] = c
    actual_params[:nutrient_content] = a
    actual_params[:requirements] = b
    
    # Model
    model = Model()
    
    # Variables
    @variable(model, x[1:n_foods] >= 0)
    
    # Objective
    @objective(model, Min, sum(c[i] * x[i] for i in 1:n_foods))
    
    # Constraints
    for j in 1:n_nutrients
        @constraint(model, sum(a[i, j] * x[i] for i in 1:n_foods) >= b[j])
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