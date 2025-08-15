using JuMP
using Random
using Distributions

"""
    generate_facility_location_problem(params::Dict=Dict(); seed::Int=0)

Generate a facility location problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_facilities`: Number of potential facility locations (default: 10)
  - `:n_customers`: Number of customers (default: 50)
  - `:min_demand`: Minimum customer demand (default: 10.0)
  - `:max_demand`: Maximum customer demand (default: 100.0)
  - `:fixed_cost_min`: Minimum fixed cost for opening a facility (default: 100000.0)
  - `:fixed_cost_max`: Maximum fixed cost for opening a facility (default: 500000.0)
  - `:capacity_factor`: Facility capacity as multiple of average total demand/n_facilities (default: 1.5)
  - `:transport_cost_per_km`: Base transport cost per unit per km (default: 1.0)
  - `:budget_factor`: Total budget as fraction of cost of opening all facilities (default: 0.7)
  - `:grid_width`: Width of the geographic area (default: 1000.0)
  - `:grid_height`: Height of the geographic area (default: 1000.0)
  - `:solution_status`: Desired feasibility status for the generated instance. One of `:feasible`,
    `:infeasible`, or `:all`. Default: `:feasible`. When `:feasible`, minimally adjust the budget to
    guarantee enough capacity can be opened. When `:infeasible`, set the budget strictly below the
    fractional-knapsack threshold so that even with fractional openings the capacity cannot meet total demand.
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_facility_location_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_facilities = get(params, :n_facilities, 10)
    n_customers = get(params, :n_customers, 50)
    min_demand = get(params, :min_demand, 10.0)
    max_demand = get(params, :max_demand, 100.0)
    fixed_cost_min = get(params, :fixed_cost_min, 100000.0)
    fixed_cost_max = get(params, :fixed_cost_max, 500000.0)
    fixed_cost_range = (fixed_cost_min, fixed_cost_max)
    capacity_factor = get(params, :capacity_factor, 1.5)
    transport_cost_per_km = get(params, :transport_cost_per_km, 1.0)
    budget_factor = get(params, :budget_factor, 0.7)
    grid_width = get(params, :grid_width, 1000.0)
    grid_height = get(params, :grid_height, 1000.0)
    grid_size = (grid_width, grid_height)
    solution_status = get(params, :solution_status, :feasible)
    if solution_status isa String
        solution_status = Symbol(lowercase(solution_status))
    end
    if !(solution_status in (:feasible, :infeasible, :all))
        error("Unknown solution_status=$(solution_status). Use :feasible, :infeasible, or :all")
    end
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_facilities => n_facilities,
        :n_customers => n_customers,
        :min_demand => min_demand,
        :max_demand => max_demand,
        :fixed_cost_min => fixed_cost_min,
        :fixed_cost_max => fixed_cost_max,
        :capacity_factor => capacity_factor,
        :transport_cost_per_km => transport_cost_per_km,
        :budget_factor => budget_factor,
        :grid_width => grid_width,
        :grid_height => grid_height,
        :solution_status => solution_status
    )
    
    # Generate locations using clustering to create realistic geographic distribution
    width, height = grid_size
    
    # Generate facility locations (more spread out)
    facility_locs = [(width * rand(), height * rand()) for _ in 1:n_facilities]
    
    # Generate customer locations (clustered around population centers)
    n_clusters = max(2, div(n_customers, 20))
    cluster_centers = [(width * rand(), height * rand()) for _ in 1:n_clusters]
    customer_locs = Tuple{Float64,Float64}[]
    for _ in 1:n_customers
        center = rand(cluster_centers)
        # Add normally distributed offset from cluster center
        x = clamp(center[1] + randn() * (width/10), 0, width)
        y = clamp(center[2] + randn() * (height/10), 0, height)
        push!(customer_locs, (x, y))
    end
    
    # Generate customer demands (log-normal distribution)
    demands = Dict{Int,Float64}()
    for c in 1:n_customers
        demands[c] = exp(rand(Normal(log((min_demand + max_demand)/2), 0.5)))
    end
    
    # Calculate total demand for capacity planning
    total_demand = sum(values(demands))
    avg_facility_capacity = (total_demand / n_facilities) * capacity_factor
    
    # Generate facility fixed costs and capacities
    fixed_costs = Dict{Int,Float64}()
    capacities = Dict{Int,Float64}()
    min_fixed, max_fixed = fixed_cost_range
    
    for w in 1:n_facilities
        # Costs somewhat correlated with capacity
        capacity = avg_facility_capacity * (0.8 + 0.4 * rand())
        capacities[w] = capacity
        
        # Fixed costs correlated with capacity and location desirability
        location_factor = 1.0 + 0.2 * (facility_locs[w][1] / width +
                                      facility_locs[w][2] / height)
        fixed_costs[w] = clamp(
            location_factor * (min_fixed + (capacity/avg_facility_capacity) * (max_fixed - min_fixed)),
            min_fixed,
            max_fixed
        )
    end
    
    # Calculate shipping costs based on distances
    shipping_costs = Dict{Tuple{Int,Int},Float64}()
    for w in 1:n_facilities
        for c in 1:n_customers
            distance = sqrt(
                (facility_locs[w][1] - customer_locs[c][1])^2 +
                (facility_locs[w][2] - customer_locs[c][2])^2
            )
            # Add some random variation to transport costs
            shipping_costs[(w,c)] = distance * transport_cost_per_km * (0.9 + 0.2 * rand())
        end
    end
    
    # Set budget as fraction of cost of opening all facilities
    budget = sum(values(fixed_costs)) * budget_factor
    original_budget = budget
    
    # Store generated data in params
    actual_params[:facility_locs] = facility_locs
    actual_params[:customer_locs] = customer_locs
    actual_params[:demands] = demands
    actual_params[:fixed_costs] = fixed_costs
    actual_params[:capacities] = capacities
    actual_params[:shipping_costs] = shipping_costs
    
    # Enforce requested feasibility/infeasibility while preserving realism and diversity.
    # Feasibility in this model depends on whether sufficient capacity can be opened under the budget.
    # With relaxed integrality, the limiting factor is a fractional knapsack on (capacity, fixed_cost).
    caps_vec = [capacities[w] for w in 1:n_facilities]
    costs_vec = [fixed_costs[w] for w in 1:n_facilities]
    ratios = [caps_vec[i] / max(costs_vec[i], eps()) for i in 1:n_facilities]
    order_desc = sortperm(ratios, rev=true)
    total_capacity = sum(caps_vec)
    
    # Helper: minimal budget (under fractional openings) to reach total_demand
    function fractional_budget_to_reach(cap_target::Float64)
        cum_cap = 0.0
        cum_cost = 0.0
        for idx in order_desc
            cap_i = caps_vec[idx]
            cost_i = costs_vec[idx]
            if cum_cap + cap_i >= cap_target
                rem = cap_target - cum_cap
                # fractional part of this facility
                frac_cost = cost_i * (rem / cap_i)
                return cum_cost + frac_cost
            else
                cum_cap += cap_i
                cum_cost += cost_i
            end
        end
        # If even all capacity is insufficient, return +Inf to indicate infeasibility regardless of budget
        return Inf
    end
    
    # Helper: construct a concrete feasible integer subset (greedy by best capacity-per-cost)
    function greedy_integer_subset_for(cap_target::Float64)
        selected = Int[]
        cum_cap = 0.0
        cum_cost = 0.0
        for idx in order_desc
            push!(selected, idx)
            cum_cap += caps_vec[idx]
            cum_cost += costs_vec[idx]
            if cum_cap + 1e-9 >= cap_target
                return selected, cum_cost
            end
        end
        return selected, cum_cost  # may still be below target if total capacity < target
    end
    
    if solution_status == :feasible
        # If total capacity < demand, scale capacities minimally to ensure realism and feasibility
        if total_capacity < total_demand
            scale = (1.05 * total_demand) / max(total_capacity, eps())
            for w in 1:n_facilities
                capacities[w] *= scale
            end
            total_capacity = sum(values(capacities))
            caps_vec = [capacities[w] for w in 1:n_facilities]
            ratios = [caps_vec[i] / max(costs_vec[i], eps()) for i in 1:n_facilities]
            order_desc = sortperm(ratios, rev=true)
        end
        # Build an explicit feasible subset and ensure budget covers it with a small slack
        selected_idxs, min_int_budget = greedy_integer_subset_for(total_demand)
        slack_factor = 1.02 + 0.23 * rand()
        budget = max(original_budget, min_int_budget * slack_factor)
        actual_params[:feasible_subset] = selected_idxs
        actual_params[:feasible_subset_cost] = min_int_budget
        actual_params[:budget_original] = original_budget
    elseif solution_status == :infeasible
        # Compute the fractional-knapsack budget threshold to reach total_demand
        b_thresh = fractional_budget_to_reach(total_demand)
        actual_params[:relaxed_budget_threshold] = b_thresh
        actual_params[:budget_original] = original_budget
        if isfinite(b_thresh)
            # Set budget strictly below threshold to guarantee infeasibility under LP relaxation
            tighten = rand(0.75:0.01:0.95)
            budget = min(original_budget, b_thresh * tighten)
        else
            # Even opening all facilities cannot meet demand; any budget is infeasible.
            # Keep budget moderate for realism.
            budget = min(original_budget, sum(values(fixed_costs)) * rand(0.4:0.01:0.8))
        end
    else
        # :all → keep stochastic construction
        budget = original_budget
    end
    
    actual_params[:budget] = budget
    
    # Model
    model = Model()
    
    # Decision variables
    @variable(model, y[1:n_facilities], Bin)  # 1 if facility is opened
    @variable(model, x[1:n_facilities, 1:n_customers] >= 0)  # shipping quantities
    
    # Objective: Minimize total cost
    @objective(model, Min,
        sum(fixed_costs[w] * y[w] for w in 1:n_facilities) +
        sum(shipping_costs[(w,c)] * x[w,c]
            for w in 1:n_facilities, c in 1:n_customers)
    )
    
    # Customer demand satisfaction
    for c in 1:n_customers
        @constraint(model,
            sum(x[w,c] for w in 1:n_facilities) >= demands[c]
        )
    end
    
    # Facility capacity
    for w in 1:n_facilities
        @constraint(model,
            sum(x[w,c] for c in 1:n_customers) <= capacities[w] * y[w]
        )
    end
    
    # Budget constraint
    @constraint(model,
        sum(fixed_costs[w] * y[w] for w in 1:n_facilities) <= budget
    )
    
    return model, actual_params
end

"""
    sample_facility_location_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a facility location problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters

# Details
For facility location: target_variables = n_facilities × (n_customers + 1)
We optimize for realistic n_facilities and n_customers values that yield the target.
"""
function sample_facility_location_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Target: n_facilities × (n_customers + 1) = target_variables
    # We need to find realistic values of n_facilities and n_customers
    
    # Set realistic ranges based on problem scale - expand ranges for larger problems
    if target_variables <= 100
        # Small problems
        min_facilities = 2
        max_facilities = 20
        min_customers = 1
        max_customers = 40
    elseif target_variables <= 1000
        # Medium problems
        min_facilities = 3
        max_facilities = 100
        min_customers = 5
        max_customers = 200
    else
        # Large problems
        min_facilities = 5
        max_facilities = 500
        min_customers = 10
        max_customers = 2000
    end
    
    best_n_facilities = min_facilities
    best_n_customers = min_customers
    best_error = Inf
    
    # Search for optimal n_facilities and n_customers
    for n_facilities in min_facilities:max_facilities
        # Given n_facilities, solve for n_customers
        # target_variables = n_facilities × (n_customers + 1)
        # n_customers = (target_variables / n_facilities) - 1
        n_customers_exact = (target_variables / n_facilities) - 1
        
        # Check if this gives a reasonable n_customers
        if n_customers_exact >= min_customers && n_customers_exact <= max_customers
            n_customers = round(Int, n_customers_exact)
            
            # Calculate actual variables with this combination
            actual_vars = n_facilities * (n_customers + 1)
            error = abs(actual_vars - target_variables) / target_variables
            
            if error < best_error
                best_error = error
                best_n_facilities = n_facilities
                best_n_customers = n_customers
            end
        end
    end
    
    # If we couldn't find a good solution within 10% error, use a heuristic approach
    if best_error > 0.1
        # Use square root heuristic as fallback
        # Assume n_customers ≈ 4 × n_facilities (typical ratio)
        # target_variables ≈ n_facilities × (4 × n_facilities + 1) ≈ 4 × n_facilities²
        n_facilities_approx = max(min_facilities, min(max_facilities, round(Int, sqrt(target_variables / 4))))
        n_customers_approx = max(min_customers, min(max_customers, round(Int, (target_variables / n_facilities_approx) - 1)))
        
        best_n_facilities = n_facilities_approx
        best_n_customers = n_customers_approx
    end
    
    params[:n_facilities] = best_n_facilities
    params[:n_customers] = best_n_customers
    
    # Set realistic parameters based on problem size
    total_entities = best_n_facilities + best_n_customers
    
    # Scale geographic area and transport costs with problem size
    if target_variables <= 100
        # Small regional problems (city/county level)
        params[:grid_width] = rand(200.0:50.0:800.0)
        params[:grid_height] = rand(200.0:50.0:800.0)
        params[:transport_cost_per_km] = rand(0.5:0.1:1.2)
        params[:min_demand] = rand(5.0:1.0:20.0)
        params[:max_demand] = rand(50.0:10.0:150.0)
        params[:fixed_cost_min] = rand(20000.0:5000.0:80000.0)
        params[:fixed_cost_max] = rand(100000.0:20000.0:300000.0)
    elseif target_variables <= 1000
        # Medium regional problems (state/province level)
        params[:grid_width] = rand(500.0:100.0:2000.0)
        params[:grid_height] = rand(500.0:100.0:2000.0)
        params[:transport_cost_per_km] = rand(0.8:0.1:1.8)
        params[:min_demand] = rand(10.0:2.0:30.0)
        params[:max_demand] = rand(80.0:20.0:200.0)
        params[:fixed_cost_min] = rand(50000.0:10000.0:150000.0)
        params[:fixed_cost_max] = rand(250000.0:50000.0:600000.0)
    else
        # Large national/international problems
        params[:grid_width] = rand(1000.0:200.0:5000.0)
        params[:grid_height] = rand(1000.0:200.0:5000.0)
        params[:transport_cost_per_km] = rand(1.0:0.2:3.0)
        params[:min_demand] = rand(20.0:5.0:60.0)
        params[:max_demand] = rand(150.0:50.0:500.0)
        params[:fixed_cost_min] = rand(100000.0:20000.0:300000.0)
        params[:fixed_cost_max] = rand(500000.0:100000.0:1500000.0)
    end
    
    # Scale capacity factor and budget factor with problem complexity
    if target_variables <= 100
        params[:capacity_factor] = rand(1.1:0.05:1.6)
        params[:budget_factor] = rand(0.4:0.05:0.8)
    elseif target_variables <= 1000
        params[:capacity_factor] = rand(1.2:0.05:1.8)
        params[:budget_factor] = rand(0.5:0.05:0.9)
    else
        params[:capacity_factor] = rand(1.3:0.1:2.0)
        params[:budget_factor] = rand(0.6:0.05:0.95)
    end
    
    return params
end

"""
    sample_facility_location_parameters(size::Symbol; seed::Int=0)

Legacy function for backward compatibility. Sample realistic parameters for a facility location problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_facility_location_parameters(size::Symbol; seed::Int=0)
    # Map size categories to target variable count ranges
    target_map = Dict(
        :small => rand(50:250),
        :medium => rand(250:1000),
        :large => rand(1000:10000)
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_facility_location_parameters(target_map[size]; seed=seed)
end

"""
    calculate_facility_location_variable_count(params::Dict)

Calculate the total number of variables for a facility location problem.

# Arguments
- `params`: Dictionary of problem parameters containing :n_facilities and :n_customers

# Returns
- Total number of variables in the problem
"""
function calculate_facility_location_variable_count(params::Dict)
    # Extract parameters with defaults
    n_facilities = get(params, :n_facilities, 10)
    n_customers = get(params, :n_customers, 50)
    
    # Variables:
    # - Binary variables y[1:n_facilities]: n_facilities variables
    # - Continuous variables x[1:n_facilities, 1:n_customers]: n_facilities × n_customers variables
    # Total: n_facilities + (n_facilities × n_customers) = n_facilities × (n_customers + 1)
    
    return n_facilities * (n_customers + 1)
end

# Register the problem type
register_problem(
    :facility_location,
    generate_facility_location_problem,
    sample_facility_location_parameters,
    "Facility location problem that minimizes the cost of opening facilities and shipping to customers while meeting demand"
)