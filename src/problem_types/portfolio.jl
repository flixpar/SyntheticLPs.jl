using JuMP
using Random


"""
    generate_portfolio_problem(params::Dict=Dict(); seed::Int=0)

Generate an investment portfolio optimization problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_options`: Number of investment options (default: 5)
  - `:total_investment`: Total investment amount (default: 100000)
  - `:max_risk`: Maximum acceptable risk (default: 5000)
  - `:return_range`: Tuple (min, max) for expected returns (default: (0.05, 0.15))
  - `:risk_range`: Tuple (min, max) for investment risks (default: (0.05, 0.15))
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_portfolio_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_options = get(params, :n_options, 150)
    total_investment = get(params, :total_investment, 1000000)
    max_risk = get(params, :max_risk, 50000)
    return_range = get(params, :return_range, (0.05, 0.15))
    risk_range = get(params, :risk_range, (0.05, 0.15))
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_options => n_options,
        :total_investment => total_investment,
        :max_risk => max_risk,
        :return_range => return_range,
        :risk_range => risk_range
    )
    
    # Random data generation
    min_return, max_return = return_range
    r = rand(min_return:0.01:max_return, n_options)  # Expected returns
    
    min_risk, max_risk = risk_range
    q = rand(min_risk:0.01:max_risk, n_options)  # Risks
    
    # Store generated data in params
    actual_params[:returns] = r
    actual_params[:risks] = q
    
    # Model
    model = Model()
    
    # Variables
    @variable(model, x[1:n_options] >= 0)
    
    # Objective
    @objective(model, Max, sum(r[i] * x[i] for i in 1:n_options))
    
    # Constraints
    @constraint(model, sum(q[i] * x[i] for i in 1:n_options) <= max_risk)
    @constraint(model, sum(x[i] for i in 1:n_options) == total_investment)
    
    return model, actual_params
end

"""
    sample_portfolio_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a portfolio optimization problem with target number of variables.

# Arguments
- `target_variables`: Target number of variables (investment options)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_portfolio_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # For portfolio problem, target_variables = n_options
    params[:n_options] = target_variables
    
    # Scale investment amounts based on number of options to maintain reasonable problem structure
    # More options typically means larger portfolios and institutional-level investing
    if target_variables <= 100
        # Small portfolio - individual investor level
        params[:total_investment] = rand(10000:5000:100000)
        risk_tolerance = rand(0.05:0.01:0.15)
        params[:max_risk] = round(Int, params[:total_investment] * risk_tolerance)
    elseif target_variables <= 1000
        # Medium portfolio - institutional level
        params[:total_investment] = rand(100000:50000:10000000)
        risk_tolerance = rand(0.03:0.005:0.12)
        params[:max_risk] = round(Int, params[:total_investment] * risk_tolerance)
    else
        # Large portfolio - mega fund level
        params[:total_investment] = rand(1000000:500000:1000000000)
        risk_tolerance = rand(0.02:0.002:0.08)
        params[:max_risk] = round(Int, params[:total_investment] * risk_tolerance)
    end
    
    # Make return and risk ranges more diverse and realistic for different market conditions
    if target_variables <= 100
        # Individual investor - broader risk tolerance, domestic focus
        params[:return_range] = (rand(0.02:0.005:0.08), rand(0.12:0.01:0.25))
        params[:risk_range] = (rand(0.03:0.005:0.08), rand(0.15:0.01:0.30))
    elseif target_variables <= 1000
        # Institutional investor - more conservative, global diversification
        params[:return_range] = (rand(0.01:0.002:0.05), rand(0.08:0.005:0.18))
        params[:risk_range] = (rand(0.02:0.003:0.06), rand(0.12:0.005:0.22))
    else
        # Mega fund - very conservative, complex instruments
        params[:return_range] = (rand(0.005:0.001:0.03), rand(0.05:0.002:0.15))
        params[:risk_range] = (rand(0.01:0.001:0.04), rand(0.08:0.002:0.18))
    end
    
    return params
end

"""
    sample_portfolio_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a portfolio optimization problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_portfolio_parameters(size::Symbol=:medium; seed::Int=0)
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
    return sample_portfolio_parameters(target_variables; seed=seed)
end

"""
    calculate_portfolio_variable_count(params::Dict)

Calculate the number of variables for a portfolio optimization problem.

# Arguments
- `params`: Dictionary of problem parameters containing `:n_options`

# Returns
- Number of variables (equal to n_options)
"""
function calculate_portfolio_variable_count(params::Dict)
    n_options = get(params, :n_options, 150)
    return n_options
end

# Register the problem type
register_problem(
    :portfolio,
    generate_portfolio_problem,
    sample_portfolio_parameters,
    "Portfolio optimization problem that maximizes returns while limiting risk"
)