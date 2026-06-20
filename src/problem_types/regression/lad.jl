using JuMP
using Random

"""
    LADRegressionProblem <: ProblemGenerator

Generator for least-absolute-deviations (L1 / LAD) regression as a linear
program.

# Overview
Fits coefficients `beta` minimizing the sum of absolute residuals
`Σ_i |y_i − X_i · beta|`, subject to box bounds on the coefficients and a
coefficient side constraint. The L1 loss is linearized with one residual
variable per sample (`e_i ≥ ±(y_i − X_i · beta)`). The constraint matrix is
**dense** (every residual row involves all coefficients), giving a numerical
profile distinct from the sparse network/allocation generators.

# Fields
- `n_samples::Int`: Number of observations
- `n_features::Int`: Number of regression coefficients
- `X::Matrix{Float64}`: Design matrix (n_samples × n_features)
- `y::Vector{Float64}`: Response vector
- `beta_lower::Vector{Float64}`: Lower bound on each coefficient
- `beta_upper::Vector{Float64}`: Upper bound on each coefficient
- `side_coef::Vector{Float64}`: Coefficients of the side constraint on `beta`
- `side_rhs::Float64`: Right-hand side of the side constraint
"""
struct LADRegressionProblem <: ProblemGenerator
    n_samples::Int
    n_features::Int
    X::Matrix{Float64}
    y::Vector{Float64}
    beta_lower::Vector{Float64}
    beta_upper::Vector{Float64}
    side_coef::Vector{Float64}
    side_rhs::Float64
end

"""
    LADRegressionProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a LAD regression instance. Variables: `beta` (n_features) plus one
residual `e` per sample, for a total of `n_features + n_samples`.
"""
function LADRegressionProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Variables = n + m, with an overdetermined system (m > n).
    ratio = rand(Uniform(3.0, 8.0))                       # samples per feature
    n_features = max(2, round(Int, target_variables / (1 + ratio)))
    n_samples = max(n_features + 1, round(Int, ratio * n_features))

    data = generate_regression_data(n_features, n_samples, feasibility_status)

    return LADRegressionProblem(n_samples, n_features, data.X, data.y,
                                data.beta_lower, data.beta_upper,
                                data.side_coef, data.side_rhs)
end

"""
    build_model(prob::LADRegressionProblem)

Build a JuMP model for LAD regression. Deterministic — uses only struct data.
"""
function build_model(prob::LADRegressionProblem)
    model = Model()

    m = prob.n_samples
    n = prob.n_features

    @variable(model, prob.beta_lower[j] <= beta[j in 1:n] <= prob.beta_upper[j])
    @variable(model, e[1:m] >= 0)

    @objective(model, Min, sum(e[i] for i in 1:m))

    for i in 1:m
        resid = prob.y[i] - sum(prob.X[i, j] * beta[j] for j in 1:n)
        @constraint(model, e[i] >= resid)
        @constraint(model, e[i] >= -resid)
    end

    @constraint(model, sum(prob.side_coef[j] * beta[j] for j in 1:n) <= prob.side_rhs)

    return model
end

# Register the variant (category default).
register_variant(
    :regression,
    :lad,
    LADRegressionProblem,
    "Least-absolute-deviations (L1) regression as a dense LP with coefficient box and side constraints",
    default = true,
)
