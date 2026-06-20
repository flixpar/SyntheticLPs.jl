using JuMP
using Random

"""
    QuantileRegressionProblem <: ProblemGenerator

Generator for quantile regression as a linear program.

# Overview
Fits coefficients `beta` minimizing the asymmetric "pinball" loss
`Σ_i ρ_τ(y_i − X_i · beta)`, where `ρ_τ(r) = τ·max(r, 0) + (1−τ)·max(−r, 0)`,
subject to box bounds on the coefficients and a coefficient side constraint. The
loss is linearized by splitting each residual into nonnegative positive/negative
parts (`u_i − v_i = y_i − X_i · beta`). The constraint matrix is **dense**. With
`τ = 0.5` this reduces to (scaled) LAD; other quantiles model conditional
quantile estimation.

# Fields
- `n_samples::Int`: Number of observations
- `n_features::Int`: Number of regression coefficients
- `tau::Float64`: Target quantile level in (0, 1)
- `X::Matrix{Float64}`: Design matrix (n_samples × n_features)
- `y::Vector{Float64}`: Response vector
- `beta_lower::Vector{Float64}`: Lower bound on each coefficient
- `beta_upper::Vector{Float64}`: Upper bound on each coefficient
- `side_coef::Vector{Float64}`: Coefficients of the side constraint on `beta`
- `side_rhs::Float64`: Right-hand side of the side constraint
"""
struct QuantileRegressionProblem <: ProblemGenerator
    n_samples::Int
    n_features::Int
    tau::Float64
    X::Matrix{Float64}
    y::Vector{Float64}
    beta_lower::Vector{Float64}
    beta_upper::Vector{Float64}
    side_coef::Vector{Float64}
    side_rhs::Float64
end

"""
    QuantileRegressionProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a quantile regression instance. Variables: `beta` (n_features) plus
positive/negative residual parts `u`, `v` per sample, for a total of
`n_features + 2 * n_samples`.
"""
function QuantileRegressionProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Variables = n + 2m, with an overdetermined system (m > n).
    ratio = rand(Uniform(2.0, 6.0))                       # samples per feature
    n_features = max(2, round(Int, target_variables / (1 + 2 * ratio)))
    n_samples = max(n_features + 1, round(Int, ratio * n_features))

    tau = rand(Uniform(0.1, 0.9))

    data = generate_regression_data(n_features, n_samples, feasibility_status)

    return QuantileRegressionProblem(n_samples, n_features, tau, data.X, data.y,
                                     data.beta_lower, data.beta_upper,
                                     data.side_coef, data.side_rhs)
end

"""
    build_model(prob::QuantileRegressionProblem)

Build a JuMP model for quantile regression. Deterministic — uses only struct data.
"""
function build_model(prob::QuantileRegressionProblem)
    model = Model()

    m = prob.n_samples
    n = prob.n_features
    τ = prob.tau

    @variable(model, prob.beta_lower[j] <= beta[j in 1:n] <= prob.beta_upper[j])
    @variable(model, u[1:m] >= 0)
    @variable(model, v[1:m] >= 0)

    @objective(model, Min, sum(τ * u[i] + (1 - τ) * v[i] for i in 1:m))

    for i in 1:m
        @constraint(model, u[i] - v[i] == prob.y[i] - sum(prob.X[i, j] * beta[j] for j in 1:n))
    end

    @constraint(model, sum(prob.side_coef[j] * beta[j] for j in 1:n) <= prob.side_rhs)

    return model
end

# Register the variant.
register_variant(
    :regression,
    :quantile,
    QuantileRegressionProblem,
    "Quantile (pinball-loss) regression as a dense LP with coefficient box and side constraints",
)
