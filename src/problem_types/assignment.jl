using JuMP
using Random


"""
    generate_assignment_problem(params::Dict=Dict(); seed::Int=0)

Generate an assignment problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_workers`: Number of workers (default: 5)
  - `:n_tasks`: Number of tasks (default: 5)
  - `:cost_range`: Tuple (min, max) for assignment costs (default: (5, 20))
  - `:balanced`: Whether the problem is balanced (default: true)
  - `:cost_variation`: How much cost varies - :low, :medium, :high (default: :medium)
  - `:specialization`: Whether some workers are specialized for certain tasks (default: false)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_assignment_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_workers = get(params, :n_workers, 5)
    n_tasks = get(params, :n_tasks, 5)
    cost_range = get(params, :cost_range, (5, 20))
    balanced = get(params, :balanced, true)
    cost_variation = get(params, :cost_variation, :medium)
    specialization = get(params, :specialization, false)
    
    # Force balanced problem if requested
    if balanced
        n_tasks = n_workers
    end
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_workers => n_workers,
        :n_tasks => n_tasks,
        :cost_range => cost_range,
        :balanced => balanced,
        :cost_variation => cost_variation,
        :specialization => specialization
    )
    
    # Generate assignment costs based on variation and specialization
    min_cost, max_cost = cost_range
    costs = zeros(Int, n_workers, n_tasks)
    
    if specialization
        # Create worker specializations - some workers are much better at certain tasks
        for i in 1:n_workers
            # Each worker has 1-3 tasks they're specialized for
            n_specializations = rand(1:min(3, n_tasks))
            specialized_tasks = randperm(n_tasks)[1:n_specializations]
            
            for j in 1:n_tasks
                if j in specialized_tasks
                    # Lower cost for specialized tasks
                    costs[i, j] = rand(min_cost:round(Int, min_cost + 0.3 * (max_cost - min_cost)))
                else
                    # Higher cost for non-specialized tasks
                    costs[i, j] = rand(round(Int, min_cost + 0.5 * (max_cost - min_cost)):max_cost)
                end
            end
        end
    else
        # Standard cost generation based on variation level
        if cost_variation == :low
            # Low variation - costs are more uniform
            mean_cost = (min_cost + max_cost) / 2
            range_factor = 0.3
            for i in 1:n_workers, j in 1:n_tasks
                low = max(min_cost, round(Int, mean_cost - range_factor * (max_cost - min_cost)))
                high = min(max_cost, round(Int, mean_cost + range_factor * (max_cost - min_cost)))
                costs[i, j] = rand(low:high)
            end
        elseif cost_variation == :high
            # High variation - costs vary widely, some extreme values
            for i in 1:n_workers, j in 1:n_tasks
                if rand() < 0.1  # 10% chance of extreme costs
                    costs[i, j] = rand() < 0.5 ? min_cost : max_cost
                else
                    costs[i, j] = rand(min_cost:max_cost)
                end
            end
        else  # :medium
            # Medium variation - standard uniform distribution
            for i in 1:n_workers, j in 1:n_tasks
                costs[i, j] = rand(min_cost:max_cost)
            end
        end
    end
    
    # Store generated data in params
    actual_params[:costs] = costs
    
    # Model
    model = Model()
    
    # Variables (1 if worker i is assigned to task j, 0 otherwise)
    @variable(model, x[1:n_workers, 1:n_tasks], Bin)
    
    # Objective: Minimize total assignment cost
    @objective(model, Min, sum(costs[i, j] * x[i, j] for i in 1:n_workers, j in 1:n_tasks))
    
    # Constraints
    
    # Each worker can be assigned to at most one task
    for i in 1:n_workers
        @constraint(model, sum(x[i, j] for j in 1:n_tasks) <= 1)
    end
    
    # Each task must be assigned to exactly one worker
    for j in 1:n_tasks
        if j <= n_workers || !balanced
            @constraint(model, sum(x[i, j] for i in 1:n_workers) == 1)
        end
    end
    
    return model, actual_params
end

"""
    sample_assignment_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for an assignment problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables (n_workers × n_tasks)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_assignment_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Determine problem characteristics based on size
    if target_variables <= 250
        # Small problems: often balanced, simple cost structure
        balanced_prob = 0.8
        cost_base = (5, 30)
        specialization_prob = 0.2
        cost_variation_dist = [0.6, 0.3, 0.1]  # [low, medium, high]
    elseif target_variables <= 1000
        # Medium problems: mix of balanced/unbalanced, moderate complexity
        balanced_prob = 0.6
        cost_base = (10, 100)
        specialization_prob = 0.4
        cost_variation_dist = [0.3, 0.5, 0.2]  # [low, medium, high]
    else
        # Large problems: more likely unbalanced, complex cost structures
        balanced_prob = 0.4
        cost_base = (50, 500)
        specialization_prob = 0.6
        cost_variation_dist = [0.2, 0.3, 0.5]  # [low, medium, high]
    end
    
    # Determine if this should be a balanced problem
    balanced = rand() < balanced_prob
    params[:balanced] = balanced
    
    if balanced
        # For balanced problems, n_workers = n_tasks, so target_variables = n_workers²
        n_workers = max(5, round(Int, sqrt(target_variables)))
        # Adjust to get closer to target
        actual_vars = n_workers * n_workers
        if actual_vars < target_variables * 0.9
            n_workers = max(5, round(Int, sqrt(target_variables * 1.1)))
        elseif actual_vars > target_variables * 1.1
            n_workers = max(5, round(Int, sqrt(target_variables * 0.9)))
        end
        params[:n_workers] = n_workers
        # n_tasks will be set equal to n_workers by the generator
    else
        # For unbalanced problems, calculate optimal dimensions
        sqrt_target = sqrt(target_variables)
        
        # Create realistic worker/task ratios
        # In real assignment problems, ratios typically range from 0.5 to 2.0
        ratio = 0.5 + rand() * 1.5  # ratio between 0.5 and 2.0
        
        n_workers = max(5, round(Int, sqrt_target * sqrt(ratio)))
        n_tasks = max(5, round(Int, target_variables / n_workers))
        
        # Fine-tune to get closer to target
        for _ in 1:3  # Multiple iterations to converge
            current_vars = n_workers * n_tasks
            if current_vars < target_variables * 0.9
                # Too few variables, increase the dimension that's further from sqrt
                if abs(n_workers - sqrt_target) > abs(n_tasks - sqrt_target)
                    n_workers = max(5, round(Int, n_workers * 1.1))
                else
                    n_tasks = max(5, round(Int, n_tasks * 1.1))
                end
            elseif current_vars > target_variables * 1.1
                # Too many variables, decrease the dimension that's further from sqrt
                if abs(n_workers - sqrt_target) > abs(n_tasks - sqrt_target)
                    n_workers = max(5, round(Int, n_workers * 0.9))
                else
                    n_tasks = max(5, round(Int, n_tasks * 0.9))
                end
            else
                break
            end
        end
        
        params[:n_workers] = n_workers
        params[:n_tasks] = n_tasks
    end
    
    # Set cost range based on problem size
    min_cost, max_cost = cost_base
    # Add some randomization to cost range
    range_multiplier = 0.8 + rand() * 0.4  # 0.8 to 1.2
    adjusted_max = max(min_cost + 5, round(Int, max_cost * range_multiplier))
    params[:cost_range] = (min_cost, adjusted_max)
    
    # Set cost variation based on problem size
    variation_choice = rand()
    if variation_choice < cost_variation_dist[1]
        params[:cost_variation] = :low
    elseif variation_choice < cost_variation_dist[1] + cost_variation_dist[2]
        params[:cost_variation] = :medium
    else
        params[:cost_variation] = :high
    end
    
    # Set specialization based on problem size
    params[:specialization] = rand() < specialization_prob
    
    return params
end

"""
    sample_assignment_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for an assignment problem using size categories.
This is a legacy function that calls the target-based version.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_assignment_parameters(size::Symbol=:medium; seed::Int=0)
    # Map size categories to target variable counts
    # Updated to match the required ranges: small=50-250, medium=250-1000, large=1000-10000
    target_map = Dict(
        :small => 150,     # Target 150 variables (range: 50-250)
        :medium => 500,    # Target 500 variables (range: 250-1000)  
        :large => 2000     # Target 2000 variables (range: 1000-10000)
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_assignment_parameters(target_map[size]; seed=seed)
end

"""
    calculate_assignment_variable_count(params::Dict)

Calculate the number of variables in an assignment problem.

# Arguments
- `params`: Dictionary of problem parameters containing :n_workers and :n_tasks

# Returns
- Integer representing the total number of variables (n_workers × n_tasks)
"""
function calculate_assignment_variable_count(params::Dict)
    n_workers = get(params, :n_workers, 5)
    n_tasks = get(params, :n_tasks, 5)
    
    # Handle balanced problems where n_tasks is set to n_workers
    balanced = get(params, :balanced, true)
    if balanced
        n_tasks = n_workers
    end
    
    return n_workers * n_tasks
end

# Register the problem type
register_problem(
    :assignment,
    generate_assignment_problem,
    sample_assignment_parameters,
    "Assignment problem that assigns workers to tasks at minimum cost with realistic scaling and cost structures"
)