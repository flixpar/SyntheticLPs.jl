using JuMP
using Random
using Distributions
using LinearAlgebra

"""
Portfolio optimization problem variants.

# Variants
- `portfolio_standard`: Classic portfolio with risk budget constraint
- `portfolio_mean_variance`: Mean-variance optimization (Markowitz style)
- `portfolio_min_risk`: Minimize risk for target return
- `portfolio_max_sharpe`: Maximize Sharpe ratio approximation
- `portfolio_cvar`: Minimize CVaR (Conditional Value at Risk)
- `portfolio_tracking`: Minimize tracking error to benchmark
- `portfolio_sector_constrained`: Sector allocation limits
- `portfolio_cardinality`: Limit number of assets held
- `portfolio_turnover`: Limit portfolio turnover from previous holdings
- `portfolio_esg`: ESG score requirements
"""
@enum PortfolioVariant begin
    portfolio_standard
    portfolio_mean_variance
    portfolio_min_risk
    portfolio_max_sharpe
    portfolio_cvar
    portfolio_tracking
    portfolio_sector_constrained
    portfolio_cardinality
    portfolio_turnover
    portfolio_esg
end

"""
    PortfolioProblem <: ProblemGenerator

Generator for investment portfolio optimization problems with multiple variants.

# Fields
- `n_options::Int`: Number of risky investment options
- `total_investment::Float64`: Total investment amount
- `max_risk::Float64`: Maximum acceptable risk budget
- `returns::Vector{Float64}`: Expected return of each risky asset
- `risks::Vector{Float64}`: Per-dollar risk factor of each risky asset
- `risk_free_rate::Float64`: Return rate of the risk-free asset
- `variant::PortfolioVariant`: Problem variant type
- Plus variant-specific fields for each problem type
"""
struct PortfolioProblem <: ProblemGenerator
    n_options::Int
    total_investment::Float64
    max_risk::Float64
    returns::Vector{Float64}
    risks::Vector{Float64}
    risk_free_rate::Float64
    variant::PortfolioVariant
    # Mean-variance variant
    covariance::Union{Matrix{Float64}, Nothing}
    risk_aversion::Float64
    # Min risk variant
    target_return::Float64
    # CVaR variant
    scenarios::Union{Matrix{Float64}, Nothing}
    scenario_probs::Union{Vector{Float64}, Nothing}
    cvar_alpha::Float64
    # Tracking variant
    benchmark_weights::Union{Vector{Float64}, Nothing}
    tracking_limit::Float64
    # Sector constrained variant
    sectors::Union{Vector{Int}, Nothing}
    n_sectors::Int
    sector_limits::Union{Vector{Float64}, Nothing}
    # Cardinality variant
    max_assets::Int
    # Turnover variant
    previous_weights::Union{Vector{Float64}, Nothing}
    max_turnover::Float64
    # ESG variant
    esg_scores::Union{Vector{Float64}, Nothing}
    min_esg_score::Float64
end

# Backwards compatibility constructor
function PortfolioProblem(n_options::Int, total_investment::Int, max_risk::Int,
                          returns::Vector{Float64}, risks::Vector{Float64}, risk_free_rate::Float64)
    PortfolioProblem(
        n_options, Float64(total_investment), Float64(max_risk), returns, risks, risk_free_rate,
        portfolio_standard, nothing, 0.0, 0.0, nothing, nothing, 0.95,
        nothing, 0.0, nothing, 0, nothing, n_options, nothing, 1.0, nothing, 0.0
    )
end

"""
    PortfolioProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                     variant::PortfolioVariant=portfolio_standard)

Construct a portfolio optimization problem instance with the specified variant.
"""
function PortfolioProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                          variant::PortfolioVariant=portfolio_standard)
    Random.seed!(seed)

    n_options = target_variables

    # Scale investment amounts based on number of options
    if target_variables <= 100
        total_investment = Float64(rand(10000:5000:100000))
        risk_tolerance = rand(0.05:0.01:0.15)
        max_risk = total_investment * risk_tolerance
        risk_free_rate = rand(0.01:0.001:0.035)
    elseif target_variables <= 1000
        total_investment = Float64(rand(100000:50000:10000000))
        risk_tolerance = rand(0.03:0.005:0.12)
        max_risk = total_investment * risk_tolerance
        risk_free_rate = rand(0.005:0.0005:0.025)
    else
        total_investment = Float64(rand(1000000:500000:1000000000))
        risk_tolerance = rand(0.02:0.002:0.08)
        max_risk = total_investment * risk_tolerance
        risk_free_rate = rand(0.0025:0.00025:0.02)
    end

    # Generate return and risk data
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

    min_return, max_return = return_range
    returns = [rand(Uniform(min_return, max_return)) for _ in 1:n_options]

    min_risk_factor, max_risk_factor = risk_range
    risks = [rand(Uniform(min_risk_factor, max_risk_factor)) for _ in 1:n_options]

    # Initialize variant-specific fields
    covariance = nothing
    risk_aversion = 0.0
    target_return = 0.0
    scenarios = nothing
    scenario_probs = nothing
    cvar_alpha = 0.95
    benchmark_weights = nothing
    tracking_limit = 0.0
    sectors = nothing
    n_sectors = 0
    sector_limits = nothing
    max_assets = n_options
    previous_weights = nothing
    max_turnover = 1.0
    esg_scores = nothing
    min_esg_score = 0.0

    # Generate variant-specific data
    if variant == portfolio_mean_variance || variant == portfolio_min_risk || variant == portfolio_max_sharpe
        # Generate realistic covariance matrix using factor model
        n_factors = min(5, max(2, n_options ÷ 10))
        factor_loadings = randn(n_options, n_factors) .* 0.3
        factor_cov = diagm(rand(Uniform(0.01, 0.05), n_factors))
        idiosyncratic = diagm(rand(Uniform(0.001, 0.02), n_options))
        covariance = factor_loadings * factor_cov * factor_loadings' + idiosyncratic

        # Ensure positive semi-definite
        covariance = (covariance + covariance') / 2
        min_eigenvalue = minimum(eigvals(covariance))
        if min_eigenvalue < 0
            covariance += diagm(fill(-min_eigenvalue + 0.001, n_options))
        end

        if variant == portfolio_mean_variance
            risk_aversion = rand(Uniform(1.0, 10.0))
        elseif variant == portfolio_min_risk
            target_return = mean(returns) * rand(Uniform(0.8, 1.2))
        end

    elseif variant == portfolio_cvar
        # Generate return scenarios
        n_scenarios = max(50, min(500, n_options * 5))
        scenarios = zeros(n_scenarios, n_options)

        for i in 1:n_options
            mu = returns[i]
            sigma = risks[i]
            # Generate scenarios with fat tails (t-distribution)
            scenarios[:, i] = mu .+ sigma .* rand(TDist(5), n_scenarios)
        end

        scenario_probs = fill(1.0 / n_scenarios, n_scenarios)
        cvar_alpha = rand([0.90, 0.95, 0.99])

    elseif variant == portfolio_tracking
        # Generate benchmark portfolio (e.g., market cap weighted)
        raw_weights = rand(Pareto(1.5), n_options)
        benchmark_weights = raw_weights ./ sum(raw_weights)
        tracking_limit = rand(Uniform(0.02, 0.10))  # 2-10% tracking error limit

    elseif variant == portfolio_sector_constrained
        # Assign assets to sectors
        n_sectors = min(11, max(3, n_options ÷ 10))  # GICS has 11 sectors
        sectors = rand(1:n_sectors, n_options)

        # Sector limits (typically 20-40% max per sector)
        sector_limits = fill(rand(Uniform(0.20, 0.40)), n_sectors)

    elseif variant == portfolio_cardinality
        # Maximum number of assets to hold (typically 20-50 for active management)
        max_assets = max(5, min(50, round(Int, n_options * rand(Uniform(0.1, 0.3)))))

    elseif variant == portfolio_turnover
        # Generate previous portfolio holdings
        prev_raw = rand(n_options)
        prev_raw[rand(1:n_options, round(Int, n_options * 0.3))] .= 0  # Some zeros
        previous_weights = prev_raw ./ sum(prev_raw)
        max_turnover = rand(Uniform(0.1, 0.5))  # 10-50% turnover limit

    elseif variant == portfolio_esg
        # Generate ESG scores (0-100 scale, normalized to 0-1)
        esg_scores = rand(Beta(5, 2), n_options)  # Skewed towards higher scores
        min_esg_score = rand(Uniform(0.5, 0.8))  # Minimum portfolio ESG
    end

    # Handle feasibility
    if feasibility_status == infeasible
        if variant == portfolio_min_risk
            # Set target return higher than maximum possible
            target_return = maximum(returns) * 1.5
        elseif variant == portfolio_tracking
            tracking_limit = 0.001  # Impossible to achieve
        elseif variant == portfolio_sector_constrained
            # Make sector limits sum to less than 1
            sector_limits = fill(0.05, n_sectors)
        elseif variant == portfolio_cardinality
            max_assets = 0  # Cannot hold any assets
        elseif variant == portfolio_turnover
            max_turnover = 0.0  # No turnover allowed with different previous holdings
        elseif variant == portfolio_esg
            min_esg_score = 1.1  # Impossible ESG score
        else
            # Standard: make risk constraint too tight
            max_risk = 0.0
        end
    elseif feasibility_status == feasible
        # Ensure feasibility for each variant
        if variant == portfolio_min_risk
            target_return = minimum(returns) * 0.9  # Achievable return
        elseif variant == portfolio_sector_constrained
            # Ensure at least one sector can hold 100%
            sector_limits[1] = 1.0
        elseif variant == portfolio_cardinality
            max_assets = max(1, max_assets)
        elseif variant == portfolio_turnover
            max_turnover = max(0.5, max_turnover)
        elseif variant == portfolio_esg
            min_esg_score = min(minimum(esg_scores), min_esg_score)
        end
    end

    return PortfolioProblem(
        n_options, total_investment, max_risk, returns, risks, risk_free_rate, variant,
        covariance, risk_aversion, target_return,
        scenarios, scenario_probs, cvar_alpha,
        benchmark_weights, tracking_limit,
        sectors, n_sectors, sector_limits,
        max_assets, previous_weights, max_turnover,
        esg_scores, min_esg_score
    )
end

"""
    build_model(prob::PortfolioProblem)

Build a JuMP model for the portfolio optimization problem based on its variant.
"""
function build_model(prob::PortfolioProblem)
    model = Model()

    if prob.variant == portfolio_standard
        # Standard portfolio: maximize returns with risk budget
        @variable(model, x[1:prob.n_options] >= 0)
        @variable(model, x_rf >= 0)

        @objective(model, Max, sum(prob.returns[i] * x[i] for i in 1:prob.n_options) + prob.risk_free_rate * x_rf)

        @constraint(model, sum(prob.risks[i] * x[i] for i in 1:prob.n_options) <= prob.max_risk)
        @constraint(model, sum(x[i] for i in 1:prob.n_options) + x_rf == prob.total_investment)

    elseif prob.variant == portfolio_mean_variance
        # Mean-variance: maximize return - λ * variance (linearized approximation)
        @variable(model, x[1:prob.n_options] >= 0)
        @variable(model, x_rf >= 0)
        @variable(model, z[1:prob.n_options] >= 0)

        # Approximate: maximize returns - λ * sum of individual variances
        @objective(model, Max,
            sum(prob.returns[i] * x[i] for i in 1:prob.n_options) + prob.risk_free_rate * x_rf -
            prob.risk_aversion * sum(z[i] for i in 1:prob.n_options))

        @constraint(model, sum(x[i] for i in 1:prob.n_options) + x_rf == prob.total_investment)

        # Linearized variance approximation
        for i in 1:prob.n_options
            @constraint(model, z[i] >= prob.covariance[i,i] * x[i] / prob.total_investment)
        end

    elseif prob.variant == portfolio_min_risk
        # Minimize variance for target return (linearized)
        @variable(model, x[1:prob.n_options] >= 0)
        @variable(model, x_rf >= 0)
        @variable(model, z[1:prob.n_options] >= 0)

        @objective(model, Min, sum(z[i] for i in 1:prob.n_options))

        @constraint(model, sum(x[i] for i in 1:prob.n_options) + x_rf == prob.total_investment)
        @constraint(model, sum(prob.returns[i] * x[i] for i in 1:prob.n_options) + prob.risk_free_rate * x_rf >=
                          prob.target_return * prob.total_investment)

        for i in 1:prob.n_options
            @constraint(model, z[i] >= prob.covariance[i,i] * x[i] / prob.total_investment)
        end

    elseif prob.variant == portfolio_max_sharpe
        # Maximize Sharpe ratio (linearized formulation using substitution)
        @variable(model, y[1:prob.n_options] >= 0)
        @variable(model, k >= 0.001)  # Scaling variable with lower bound

        # Maximize excess return (normalized)
        @objective(model, Max, sum((prob.returns[i] - prob.risk_free_rate) * y[i] for i in 1:prob.n_options))

        @constraint(model, sum(y[i] for i in 1:prob.n_options) == 1.0)

        # Risk budget constraint
        @constraint(model, sum(sqrt(prob.covariance[i,i]) * y[i] for i in 1:prob.n_options) <= 0.5)

    elseif prob.variant == portfolio_cvar
        # Minimize CVaR (Conditional Value at Risk)
        n_scenarios = size(prob.scenarios, 1)

        @variable(model, x[1:prob.n_options] >= 0)
        @variable(model, x_rf >= 0)
        @variable(model, VaR)
        @variable(model, excess_loss[1:n_scenarios] >= 0)

        @objective(model, Min, VaR + 1/(1 - prob.cvar_alpha) *
                          sum(prob.scenario_probs[s] * excess_loss[s] for s in 1:n_scenarios))

        @constraint(model, sum(x[i] for i in 1:prob.n_options) + x_rf == prob.total_investment)

        min_return = prob.risk_free_rate * 0.5
        @constraint(model, sum(prob.returns[i] * x[i] for i in 1:prob.n_options) +
                          prob.risk_free_rate * x_rf >= min_return * prob.total_investment)

        for s in 1:n_scenarios
            scenario_return = sum(prob.scenarios[s, i] * x[i] for i in 1:prob.n_options) +
                             prob.risk_free_rate * x_rf
            @constraint(model, excess_loss[s] >= -scenario_return - VaR)
        end

    elseif prob.variant == portfolio_tracking
        # Minimize tracking error to benchmark
        @variable(model, x[1:prob.n_options] >= 0)
        @variable(model, deviation_plus[1:prob.n_options] >= 0)
        @variable(model, deviation_minus[1:prob.n_options] >= 0)

        @objective(model, Max, sum(prob.returns[i] * x[i] for i in 1:prob.n_options))

        @constraint(model, sum(x[i] for i in 1:prob.n_options) == prob.total_investment)

        for i in 1:prob.n_options
            @constraint(model, x[i] - prob.benchmark_weights[i] * prob.total_investment ==
                              deviation_plus[i] - deviation_minus[i])
        end

        @constraint(model, sum(deviation_plus[i] + deviation_minus[i]
                               for i in 1:prob.n_options) <= prob.tracking_limit * prob.total_investment)

    elseif prob.variant == portfolio_sector_constrained
        @variable(model, x[1:prob.n_options] >= 0)
        @variable(model, x_rf >= 0)

        @objective(model, Max, sum(prob.returns[i] * x[i] for i in 1:prob.n_options) +
                          prob.risk_free_rate * x_rf)

        @constraint(model, sum(prob.risks[i] * x[i] for i in 1:prob.n_options) <= prob.max_risk)
        @constraint(model, sum(x[i] for i in 1:prob.n_options) + x_rf == prob.total_investment)

        for s in 1:prob.n_sectors
            sector_assets = [i for i in 1:prob.n_options if prob.sectors[i] == s]
            if !isempty(sector_assets)
                @constraint(model, sum(x[i] for i in sector_assets) <=
                                  prob.sector_limits[s] * prob.total_investment)
            end
        end

    elseif prob.variant == portfolio_cardinality
        @variable(model, x[1:prob.n_options] >= 0)
        @variable(model, x_rf >= 0)
        @variable(model, y[1:prob.n_options], Bin)

        @objective(model, Max, sum(prob.returns[i] * x[i] for i in 1:prob.n_options) +
                          prob.risk_free_rate * x_rf)

        @constraint(model, sum(prob.risks[i] * x[i] for i in 1:prob.n_options) <= prob.max_risk)
        @constraint(model, sum(x[i] for i in 1:prob.n_options) + x_rf == prob.total_investment)
        @constraint(model, sum(y[i] for i in 1:prob.n_options) <= prob.max_assets)

        for i in 1:prob.n_options
            @constraint(model, x[i] <= prob.total_investment * y[i])
        end

    elseif prob.variant == portfolio_turnover
        @variable(model, x[1:prob.n_options] >= 0)
        @variable(model, x_rf >= 0)
        @variable(model, buy[1:prob.n_options] >= 0)
        @variable(model, sell[1:prob.n_options] >= 0)

        @objective(model, Max, sum(prob.returns[i] * x[i] for i in 1:prob.n_options) +
                          prob.risk_free_rate * x_rf)

        @constraint(model, sum(prob.risks[i] * x[i] for i in 1:prob.n_options) <= prob.max_risk)
        @constraint(model, sum(x[i] for i in 1:prob.n_options) + x_rf == prob.total_investment)

        for i in 1:prob.n_options
            @constraint(model, x[i] - prob.previous_weights[i] * prob.total_investment ==
                              buy[i] - sell[i])
        end

        @constraint(model, sum(buy[i] + sell[i] for i in 1:prob.n_options) <=
                          prob.max_turnover * prob.total_investment)

    elseif prob.variant == portfolio_esg
        @variable(model, x[1:prob.n_options] >= 0)
        @variable(model, x_rf >= 0)

        @objective(model, Max, sum(prob.returns[i] * x[i] for i in 1:prob.n_options) +
                          prob.risk_free_rate * x_rf)

        @constraint(model, sum(prob.risks[i] * x[i] for i in 1:prob.n_options) <= prob.max_risk)
        @constraint(model, sum(x[i] for i in 1:prob.n_options) + x_rf == prob.total_investment)

        @constraint(model, sum(prob.esg_scores[i] * x[i] for i in 1:prob.n_options) >=
                          prob.min_esg_score * sum(x[i] for i in 1:prob.n_options))
    end

    return model
end

# Register the problem type
register_problem(
    :portfolio,
    PortfolioProblem,
    "Portfolio optimization with variants including standard, mean-variance, min-risk, max-Sharpe, CVaR, tracking error, sector-constrained, cardinality, turnover, and ESG formulations"
)
