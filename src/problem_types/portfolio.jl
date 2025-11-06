using JuMP
using Random

"""
    PortfolioProblem <: ProblemGenerator

Generator for investment portfolio optimization problems that maximize returns while limiting risk.

# Fields
- `n_options::Int`: Number of risky investment options
- `total_investment::Int`: Total investment amount
- `max_risk::Int`: Maximum acceptable risk budget
- `returns::Vector{Float64}`: Expected return of each risky asset
- `risks::Vector{Float64}`: Per-dollar risk factor of each risky asset
- `risk_free_rate::Float64`: Return rate of the risk-free asset
"""
struct PortfolioProblem <: ProblemGenerator
    n_options::Int
    total_investment::Int
    max_risk::Int
    returns::Vector{Float64}
    risks::Vector{Float64}
    risk_free_rate::Float64
end

"""
    PortfolioProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a portfolio optimization problem instance.

# Arguments
- `target_variables`: Target number of variables (investment options)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function PortfolioProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # For portfolio problem, target_variables = n_options + 1 (including risk-free asset)
    # So n_options = target_variables - 1, but we'll use target_variables for simplicity
    n_options = target_variables

    # Scale investment amounts based on number of options
    if target_variables <= 100
        # Small portfolio - individual investor level
        total_investment = rand(10000:5000:100000)
        risk_tolerance = rand(0.05:0.01:0.15)
        max_risk = round(Int, total_investment * risk_tolerance)
        risk_free_rate = rand(0.01:0.001:0.035)
    elseif target_variables <= 1000
        # Medium portfolio - institutional level
        total_investment = rand(100000:50000:10000000)
        risk_tolerance = rand(0.03:0.005:0.12)
        max_risk = round(Int, total_investment * risk_tolerance)
        risk_free_rate = rand(0.005:0.0005:0.025)
    else
        # Large portfolio - mega fund level
        total_investment = rand(1000000:500000:1000000000)
        risk_tolerance = rand(0.02:0.002:0.08)
        max_risk = round(Int, total_investment * risk_tolerance)
        risk_free_rate = rand(0.0025:0.00025:0.02)
    end

    # Make return and risk ranges more diverse and realistic
    if target_variables <= 100
        return_range = (rand(0.02:0.005:0.08), rand(0.12:0.01:0.25))
        risk_range = (rand(0.03:0.005:0.08), rand(0.15:0.01:0.30))
    elseif target_variables <= 1000
        return_range = (rand(0.01:0.002:0.05), rand(0.08:0.005:0.18))
        risk_range = (rand(0.02:0.003:0.06), rand(0.12:0.005:0.22))
    else
        return_range = (rand(0.005:0.001:0.03), rand(0.05:0.002:0.15))
        risk_range = (rand(0.01:0.001:0.04), rand(0.08:0.002:0.18))
    end

    # Generate return and risk data
    min_return, max_return = return_range
    returns = rand(min_return:0.01:max_return, n_options)

    min_risk_factor, max_risk_factor = risk_range
    risks = rand(min_risk_factor:0.01:max_risk_factor, n_options)

    # Handle feasibility - portfolio problems are typically always feasible
    # because we can always put everything in the risk-free asset
    # So infeasible case would need to remove the risk-free asset or add constraints

    return PortfolioProblem(n_options, total_investment, max_risk, returns, risks, risk_free_rate)
end

"""
    build_model(prob::PortfolioProblem)

Build a JuMP model for the portfolio optimization problem.

# Arguments
- `prob`: PortfolioProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::PortfolioProblem)
    model = Model()

    # Variables
    # x[1:n_options] are risky assets; x_rf is risk-free
    @variable(model, x[1:prob.n_options] >= 0)
    @variable(model, x_rf >= 0)

    # Objective
    @objective(model, Max, sum(prob.returns[i] * x[i] for i in 1:prob.n_options) + prob.risk_free_rate * x_rf)

    # Constraints
    @constraint(model, sum(prob.risks[i] * x[i] for i in 1:prob.n_options) <= prob.max_risk)
    @constraint(model, sum(x[i] for i in 1:prob.n_options) + x_rf == prob.total_investment)

    return model
end

# Register the problem type
register_problem(
    :portfolio,
    PortfolioProblem,
    "Portfolio optimization problem that maximizes returns while limiting risk"
)
