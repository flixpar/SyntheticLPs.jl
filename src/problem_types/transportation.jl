using JuMP
using Random

"""
    generate_transportation_problem(params::Dict=Dict(); seed::Int=0)

Generate a transportation problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_sources`: Number of supply sources (default: 4)
  - `:n_destinations`: Number of demand destinations (default: 5)
  - `:supply_range`: Tuple (min, max) for supply values (default: (50, 100))
  - `:demand_range`: Tuple (min, max) for demand values (default: (30, 80))
  - `:cost_range`: Tuple (min, max) for transportation costs (default: (5, 20))
  - `:solution_status`: Desired feasibility status (:feasible [default], :infeasible, :all)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_transportation_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_sources = get(params, :n_sources, 4)
    n_destinations = get(params, :n_destinations, 5)
    supply_range = get(params, :supply_range, (50, 100))
    demand_range = get(params, :demand_range, (30, 80))
    cost_range = get(params, :cost_range, (5, 20))
    solution_status = get(params, :solution_status, :feasible)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_sources => n_sources,
        :n_destinations => n_destinations,
        :supply_range => supply_range,
        :demand_range => demand_range,
        :cost_range => cost_range,
        :solution_status => solution_status
    )
    
    # Random data generation
    min_supply, max_supply = supply_range
    s = rand(min_supply:max_supply, n_sources)  # Supply at each source
    
    min_demand, max_demand = demand_range
    d = rand(min_demand:max_demand, n_destinations)  # Demand at each destination
    
    # Robust feasibility control without simplifying the LP structure
    total_supply = sum(s)
    total_demand = sum(d)

    # Helper to distribute a given integer amount across entries of a vector in a randomized, spread-out way
    function distribute_additions!(vec::Vector{Int}, amount::Int)
        if amount <= 0
            return
        end
        # Base proportional split using random weights
        w = rand(length(vec))
        w_sum = sum(w)
        base = floor.(Int, (w ./ w_sum) .* amount)
        remainder = amount - sum(base)
        if remainder > 0
            for idx in randperm(length(vec))[1:remainder]
                base[idx] += 1
            end
        end
        vec .+= base
    end

    if solution_status == :feasible
        # Guarantee feasibility: ensure total_supply >= total_demand exactly, avoiding rounding pitfalls
        if total_supply < total_demand
            shortage = total_demand - total_supply
            distribute_additions!(s, shortage)
            total_supply = sum(s)
        end
        # No need to inflate demands; keeping slack maintains realism while ensuring feasibility
    elseif solution_status == :infeasible
        # Guarantee infeasibility: ensure total_demand >= total_supply + margin
        # Target a small relative shortage to keep realism (2%–10% of total supply), minimum 1 unit
        target_margin = max(1, round(Int, (0.02 + 0.08 * rand()) * max(total_supply, 1)))
        # Compute how much we must increase demands to exceed supply by the margin
        missing = (total_supply + target_margin) - total_demand
        if missing > 0
            distribute_additions!(d, missing)
        end
        total_demand = sum(d)
    else
        # :all -> do not force either direction; keep the natural draws to preserve variety
    end
    
    min_cost, max_cost = cost_range
    c = rand(min_cost:max_cost, n_sources, n_destinations)  # Transportation costs
    
    # Store generated data in params
    actual_params[:supplies] = s
    actual_params[:demands] = d
    actual_params[:costs] = c
    
    # Model
    model = Model()
    
    # Variables
    @variable(model, x[1:n_sources, 1:n_destinations] >= 0)
    
    # Objective
    @objective(model, Min, sum(c[i, j] * x[i, j] for i in 1:n_sources, j in 1:n_destinations))
    
    # Constraints
    for i in 1:n_sources
        @constraint(model, sum(x[i, j] for j in 1:n_destinations) <= s[i])
    end
    for j in 1:n_destinations
        @constraint(model, sum(x[i, j] for i in 1:n_sources) >= d[j])
    end
    
    return model, actual_params
end

"""
    sample_transportation_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a transportation problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables (n_sources × n_destinations)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_transportation_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Calculate optimal dimensions for the target number of variables
    # For transportation: variables = n_sources × n_destinations
    # Start with square root as base, then add randomization
    sqrt_target = sqrt(target_variables)
    
    # Add some randomization to create variety while staying close to target
    # Use a ratio between sources and destinations that's realistic
    ratio = 0.5 + rand() * 1.0  # ratio between 0.5 and 1.5
    
    n_sources = max(2, round(Int, sqrt_target * ratio))
    n_destinations = max(2, round(Int, target_variables / n_sources))
    
    # Fine-tune to get closer to target
    current_vars = n_sources * n_destinations
    if current_vars < target_variables * 0.9
        # Too few variables, increase the larger dimension
        if n_sources >= n_destinations
            n_sources = max(n_sources, round(Int, target_variables / n_destinations))
        else
            n_destinations = max(n_destinations, round(Int, target_variables / n_sources))
        end
    elseif current_vars > target_variables * 1.1
        # Too many variables, decrease the larger dimension
        if n_sources >= n_destinations
            n_sources = max(2, round(Int, target_variables / n_destinations))
        else
            n_destinations = max(2, round(Int, target_variables / n_sources))
        end
    end
    
    params[:n_sources] = n_sources
    params[:n_destinations] = n_destinations
    
    # Set realistic parameter ranges based on problem size
    total_vars = n_sources * n_destinations
    if total_vars <= 250
        # Small problems - local/regional transportation
        params[:supply_range] = (rand(50:100), rand(200:500))
        params[:demand_range] = (rand(30:80), rand(150:300))
        params[:cost_range] = (rand(5:15), rand(25:60))
    elseif total_vars <= 1000
        # Medium problems - national/regional networks
        params[:supply_range] = (rand(100:500), rand(1000:5000))
        params[:demand_range] = (rand(80:300), rand(800:3000))
        params[:cost_range] = (rand(10:30), rand(50:150))
    else
        # Large problems - international/global networks
        params[:supply_range] = (rand(500:2000), rand(5000:50000))
        params[:demand_range] = (rand(300:1500), rand(3000:30000))
        params[:cost_range] = (rand(20:100), rand(100:500))
    end
    
    return params
end

"""
    sample_transportation_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a transportation problem using size categories.
This is a legacy function that calls the target-based version.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_transportation_parameters(size::Symbol=:medium; seed::Int=0)
    # Map size categories to target variable counts
    target_map = Dict(
        :small => rand(50:250),
        :medium => rand(250:1000), 
        :large => rand(1000:10000)
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_transportation_parameters(target_map[size]; seed=seed)
end

"""
    calculate_transportation_variable_count(params::Dict)

Calculate the number of variables in a transportation problem.

# Arguments
- `params`: Dictionary of problem parameters containing :n_sources and :n_destinations

# Returns
- Integer representing the total number of variables (n_sources × n_destinations)
"""
function calculate_transportation_variable_count(params::Dict)
    n_sources = get(params, :n_sources, 4)
    n_destinations = get(params, :n_destinations, 5)
    return n_sources * n_destinations
end

# Register the problem type
register_problem(
    :transportation,
    generate_transportation_problem,
    sample_transportation_parameters,
    "Transportation problem that optimizes shipping goods from sources to destinations at minimum cost"
)