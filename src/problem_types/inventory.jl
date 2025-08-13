using JuMP
using Random
using Distributions

"""
    generate_inventory_problem(params::Dict=Dict(); seed::Int=0)

Generate an inventory control problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_periods`: Number of periods in the planning horizon (default: 6)
  - `:prod_capacity`: Production capacity per period (default: 100)
  - `:initial_inventory`: Initial inventory (default: 20)
  - `:demand_min`: Minimum demand per period (default: 50)
  - `:demand_max`: Maximum demand per period (default: 100)
  - `:prod_cost_min`: Minimum production cost per unit (default: 10)
  - `:prod_cost_max`: Maximum production cost per unit (default: 20)
  - `:holding_cost_min`: Minimum holding cost per unit (default: 1)
  - `:holding_cost_max`: Maximum holding cost per unit (default: 5)
  - `:backlog_allowed`: Whether backlogging is allowed (default: false)
  - `:backlog_cost_factor`: Backlog cost as a factor of production cost (default: 2.0)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_inventory_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_periods = get(params, :n_periods, 6)
    prod_capacity = get(params, :prod_capacity, 100)
    initial_inventory = get(params, :initial_inventory, 20)
    demand_min = get(params, :demand_min, 50)
    demand_max = get(params, :demand_max, 100)
    prod_cost_min = get(params, :prod_cost_min, 10)
    prod_cost_max = get(params, :prod_cost_max, 20)
    holding_cost_min = get(params, :holding_cost_min, 1)
    holding_cost_max = get(params, :holding_cost_max, 5)
    backlog_allowed = get(params, :backlog_allowed, false)
    backlog_cost_factor = get(params, :backlog_cost_factor, 2.0)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_periods => n_periods,
        :prod_capacity => prod_capacity,
        :initial_inventory => initial_inventory,
        :demand_min => demand_min,
        :demand_max => demand_max,
        :prod_cost_min => prod_cost_min,
        :prod_cost_max => prod_cost_max,
        :holding_cost_min => holding_cost_min,
        :holding_cost_max => holding_cost_max,
        :backlog_allowed => backlog_allowed,
        :backlog_cost_factor => backlog_cost_factor
    )
    
    # Generate realistic random data using proper distributions
    demand_mean = (demand_min + demand_max) / 2
    demand_std = (demand_max - demand_min) / 4  # 95% of values within range
    
    # Generate base demands using normal distribution, then clamp to range
    base_demands = rand(Normal(demand_mean, demand_std), n_periods)
    demands = round.(Int, clamp.(base_demands, demand_min, demand_max))
    
    # Generate production costs with some variation
    prod_cost_mean = (prod_cost_min + prod_cost_max) / 2
    prod_cost_std = (prod_cost_max - prod_cost_min) / 4
    production_costs = clamp.(rand(Normal(prod_cost_mean, prod_cost_std), n_periods), prod_cost_min, prod_cost_max)
    
    # Generate holding costs with some variation
    holding_cost_mean = (holding_cost_min + holding_cost_max) / 2
    holding_cost_std = (holding_cost_max - holding_cost_min) / 4
    holding_costs = clamp.(rand(Normal(holding_cost_mean, holding_cost_std), n_periods), holding_cost_min, holding_cost_max)
    
    # Add realistic seasonal and trend patterns
    if rand() < 0.6  # 60% chance of seasonal demand
        # Multiple seasonal patterns based on period length
        if n_periods >= 12
            # Annual seasonality (12-period cycle)
            annual_seasonality = 1.0 .+ 0.2 * sin.(2π * (1:n_periods) / 12)
            demands = round.(Int, demands .* annual_seasonality)
        end
        
        if n_periods >= 52
            # Weekly seasonality (7-period cycle) for very long horizons
            weekly_seasonality = 1.0 .+ 0.1 * sin.(2π * (1:n_periods) / 7)
            demands = round.(Int, demands .* weekly_seasonality)
        end
        
        if n_periods >= 24
            # Quarterly seasonality (4-period cycle)
            quarterly_seasonality = 1.0 .+ 0.15 * sin.(2π * (1:n_periods) / (n_periods/4))
            demands = round.(Int, demands .* quarterly_seasonality)
        end
    end
    
    if rand() < 0.4  # 40% chance of cost trends
        # Add trend to production costs (inflation/deflation)
        trend_direction = rand() < 0.7 ? 1 : -1  # 70% chance of increasing costs
        trend_strength = rand(Uniform(0.001, 0.01))  # 0.1% to 1% per period
        trend_values = [exp(trend_direction * trend_strength * t) for t in 1:n_periods]
        production_costs = production_costs .* trend_values
    end
    
    if rand() < 0.3  # 30% chance of holding cost trends
        # Holding costs may change due to warehouse costs, interest rates, etc.
        holding_trend_direction = rand() < 0.6 ? 1 : -1
        holding_trend_strength = rand(Uniform(0.0005, 0.005))
        holding_trend_values = [exp(holding_trend_direction * holding_trend_strength * t) for t in 1:n_periods]
        holding_costs = holding_costs .* holding_trend_values
    end
    
    # Add occasional demand spikes/drops (supply chain disruptions, promotions, etc.)
    if rand() < 0.2  # 20% chance of demand disruptions
        n_disruptions = rand(Poisson(max(1, n_periods / 20)))  # ~1 disruption per 20 periods
        for _ in 1:n_disruptions
            disruption_period = rand(1:n_periods)
            disruption_factor = rand() < 0.5 ? rand(Uniform(0.3, 0.7)) : rand(Uniform(1.4, 2.0))  # Drop or spike
            demands[disruption_period] = round(Int, demands[disruption_period] * disruption_factor)
        end
    end
    
    # Ensure demands stay within reasonable bounds
    demands = max.(demands, max(1, demand_min ÷ 2))
    demands = min.(demands, demand_max * 2)
    
    # Calculate backlog costs if allowed
    backlog_costs = production_costs .* backlog_cost_factor
    
    # Store generated data in params
    actual_params[:demands] = demands
    actual_params[:production_costs] = production_costs
    actual_params[:holding_costs] = holding_costs
    actual_params[:backlog_costs] = backlog_costs
    
    # Create model
    model = Model()
    
    if backlog_allowed
        # Variables with backlogging
        @variable(model, x[1:n_periods] >= 0)  # Production quantity
        @variable(model, I_plus[0:n_periods] >= 0)  # Inventory level (on hand)
        @variable(model, I_minus[0:n_periods] >= 0)  # Backlog level
        
        # Objective with backlogging costs
        @objective(model, Min, 
            sum(production_costs[t] * x[t] + 
                holding_costs[t] * I_plus[t] + 
                backlog_costs[t] * I_minus[t] for t in 1:n_periods))
        
        # Initial conditions
        @constraint(model, I_plus[0] == initial_inventory)
        @constraint(model, I_minus[0] == 0)
        
        # Inventory balance constraints with backlogging
        for t in 1:n_periods
            @constraint(model, I_plus[t-1] - I_minus[t-1] + x[t] - demands[t] == I_plus[t] - I_minus[t])
            @constraint(model, x[t] <= prod_capacity)  # Production capacity
        end
    else
        # Variables without backlogging
        @variable(model, x[1:n_periods] >= 0)  # Production quantity
        @variable(model, I[0:n_periods] >= 0)  # Inventory level
        
        # Objective
        @objective(model, Min, 
            sum(production_costs[t] * x[t] + holding_costs[t] * I[t] for t in 1:n_periods))
        
        # Initial inventory
        @constraint(model, I[0] == initial_inventory)
        
        # Inventory balance and capacity constraints
        for t in 1:n_periods
            @constraint(model, I[t-1] + x[t] - demands[t] == I[t])  # Inventory balance
            @constraint(model, x[t] <= prod_capacity)  # Production capacity
        end
    end
    
    return model, actual_params
end

"""
    sample_inventory_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for an inventory control problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_inventory_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Determine business scale based on target variables
    # Small: 50-250 vars, Medium: 250-1000 vars, Large: 1000-10000 vars
    if target_variables <= 250
        scale = :small
    elseif target_variables <= 1000
        scale = :medium
    else
        scale = :large
    end
    
    # Backlog probability increases with scale (larger businesses more likely to allow backlog)
    backlog_prob = scale == :small ? 0.2 : (scale == :medium ? 0.4 : 0.6)
    params[:backlog_allowed] = rand(Bernoulli(backlog_prob))
    
    # Calculate approximate n_periods based on target variables
    # Variables: x[1:n_periods], I[0:n_periods] (and I_minus[0:n_periods] if backlog allowed)
    if params[:backlog_allowed]
        # target_variables = 3*n_periods + 2
        params[:n_periods] = max(2, min(5000, round(Int, (target_variables - 2) / 3)))
    else
        # target_variables = 2*n_periods + 1
        params[:n_periods] = max(2, min(5000, round(Int, (target_variables - 1) / 2)))
    end
    
    # Iteratively adjust to get closer to target
    for iteration in 1:10
        current_vars = calculate_inventory_variable_count(params)
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust n_periods
        if current_vars < target_variables
            params[:n_periods] = min(5000, params[:n_periods] + 1)
        elseif current_vars > target_variables
            params[:n_periods] = max(2, params[:n_periods] - 1)
        end
    end
    
    # Scale-dependent parameters using realistic distributions
    if scale == :small
        # Small business parameters
        params[:prod_capacity] = round(Int, rand(Uniform(50, 500)))
        demand_base = round(Int, rand(Uniform(10, 100)))
        demand_volatility = rand(Uniform(0.2, 0.5))
        params[:demand_min] = max(1, round(Int, demand_base * (1 - demand_volatility)))
        params[:demand_max] = round(Int, demand_base * (1 + demand_volatility))
        
        # Initial inventory as percentage of average demand
        avg_demand = (params[:demand_min] + params[:demand_max]) / 2
        params[:initial_inventory] = round(Int, avg_demand * rand(Uniform(0.1, 0.5)))
        
        # Production costs
        prod_cost_base = rand(Uniform(10, 100))
        prod_cost_spread = rand(Uniform(0.1, 0.3))
        params[:prod_cost_min] = round(Int, prod_cost_base * (1 - prod_cost_spread))
        params[:prod_cost_max] = round(Int, prod_cost_base * (1 + prod_cost_spread))
        
        # Holding costs (typically 5-25% of production cost annually)
        holding_cost_rate = rand(Uniform(0.05, 0.25)) / 12  # Monthly rate
        params[:holding_cost_min] = max(0.01, round(prod_cost_base * holding_cost_rate * 0.8, digits=2))
        params[:holding_cost_max] = round(prod_cost_base * holding_cost_rate * 1.2, digits=2)
        
    elseif scale == :medium
        # Medium enterprise parameters
        params[:prod_capacity] = round(Int, rand(Uniform(200, 2000)))
        demand_base = round(Int, rand(Uniform(50, 1000)))
        demand_volatility = rand(Uniform(0.15, 0.4))
        params[:demand_min] = max(1, round(Int, demand_base * (1 - demand_volatility)))
        params[:demand_max] = round(Int, demand_base * (1 + demand_volatility))
        
        # Initial inventory
        avg_demand = (params[:demand_min] + params[:demand_max]) / 2
        params[:initial_inventory] = round(Int, avg_demand * rand(Uniform(0.05, 0.4)))
        
        # Production costs
        prod_cost_base = rand(Uniform(5, 200))
        prod_cost_spread = rand(Uniform(0.05, 0.25))
        params[:prod_cost_min] = round(Int, prod_cost_base * (1 - prod_cost_spread))
        params[:prod_cost_max] = round(Int, prod_cost_base * (1 + prod_cost_spread))
        
        # Holding costs
        holding_cost_rate = rand(Uniform(0.03, 0.20)) / 12  # Monthly rate
        params[:holding_cost_min] = max(0.01, round(prod_cost_base * holding_cost_rate * 0.8, digits=2))
        params[:holding_cost_max] = round(prod_cost_base * holding_cost_rate * 1.2, digits=2)
        
    else  # large scale
        # Large enterprise parameters
        params[:prod_capacity] = round(Int, rand(Uniform(1000, 50000)))
        demand_base = round(Int, rand(Uniform(100, 10000)))
        demand_volatility = rand(Uniform(0.1, 0.3))
        params[:demand_min] = max(1, round(Int, demand_base * (1 - demand_volatility)))
        params[:demand_max] = round(Int, demand_base * (1 + demand_volatility))
        
        # Initial inventory
        avg_demand = (params[:demand_min] + params[:demand_max]) / 2
        params[:initial_inventory] = round(Int, avg_demand * rand(Uniform(0.02, 0.3)))
        
        # Production costs
        prod_cost_base = rand(Uniform(1, 500))
        prod_cost_spread = rand(Uniform(0.02, 0.20))
        params[:prod_cost_min] = round(Int, prod_cost_base * (1 - prod_cost_spread))
        params[:prod_cost_max] = round(Int, prod_cost_base * (1 + prod_cost_spread))
        
        # Holding costs
        holding_cost_rate = rand(Uniform(0.01, 0.15)) / 12  # Monthly rate
        params[:holding_cost_min] = max(0.01, round(prod_cost_base * holding_cost_rate * 0.8, digits=2))
        params[:holding_cost_max] = round(prod_cost_base * holding_cost_rate * 1.2, digits=2)
    end
    
    # Backlog cost factor (typically 1.5-5x production cost)
    params[:backlog_cost_factor] = rand(Uniform(1.5, 5.0))
    
    return params
end

"""
    sample_inventory_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for an inventory control problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_inventory_parameters(size::Symbol=:medium; seed::Int=0)
    # Map size categories to realistic target variable counts
    target_map = Dict(
        :small => rand(50:250),      # Small business: 25-125 periods
        :medium => rand(250:1000),   # Medium enterprise: 80-500 periods  
        :large => rand(1000:10000)   # Large enterprise: 300-5000 periods
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_inventory_parameters(target_map[size]; seed=seed)
end

"""
    calculate_inventory_variable_count(params::Dict)

Calculate the total number of variables for an inventory control problem.

# Arguments
- `params`: Dictionary of problem parameters

# Returns
- Integer: Total number of variables in the problem
"""
function calculate_inventory_variable_count(params::Dict)
    # Extract parameters with defaults
    n_periods = get(params, :n_periods, 6)
    backlog_allowed = get(params, :backlog_allowed, false)
    
    if backlog_allowed
        # Variables: x[1:n_periods], I_plus[0:n_periods], I_minus[0:n_periods]
        # Total: n_periods + (n_periods + 1) + (n_periods + 1) = 3*n_periods + 2
        return 3 * n_periods + 2
    else
        # Variables: x[1:n_periods], I[0:n_periods]
        # Total: n_periods + (n_periods + 1) = 2*n_periods + 1
        return 2 * n_periods + 1
    end
end

# Register the problem type
register_problem(
    :inventory,
    generate_inventory_problem,
    sample_inventory_parameters,
    "Inventory control problem that minimizes production and holding costs while meeting demand over multiple periods"
)