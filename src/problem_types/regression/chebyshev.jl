using JuMP
using Random

"""
    ChebyshevRegressionProblem <: ProblemGenerator

Generator for Chebyshev (L∞ / minimax) regression as a linear program.

# Overview
Fits coefficients `beta` minimizing the maximum absolute residual
`max_i |y_i − X_i · beta|`, subject to box bounds on the coefficients and a
coefficient side constraint. A single scalar `t` bounds every residual
(`−t ≤ y_i − X_i · beta ≤ t`), so the model has very few variables but many
**dense** rows — an overdetermined minimax fit with a structure distinct from
the L1/quantile variants.

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
struct ChebyshevRegressionProblem <: ProblemGenerator
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
    ChebyshevRegressionProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a Chebyshev regression instance. Variables: `beta` (n_features) plus a
single bound `t`, for a total of `n_features + 1`.
"""
function ChebyshevRegressionProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Variables = n + 1; the system is heavily overdetermined in the rows.
    n_features = max(2, target_variables - 1)
    n_samples = max(n_features + 1, round(Int, n_features * rand(Uniform(3.0, 8.0))))

    data = generate_regression_data(n_features, n_samples, feasibility_status)

    return ChebyshevRegressionProblem(n_samples, n_features, data.X, data.y,
                                      data.beta_lower, data.beta_upper,
                                      data.side_coef, data.side_rhs)
end

"""
    build_model(prob::ChebyshevRegressionProblem)

Build a JuMP model for Chebyshev regression. Deterministic — uses only struct data.
"""
function build_model(prob::ChebyshevRegressionProblem)
    model = Model()

    m = prob.n_samples
    n = prob.n_features

    @variable(model, prob.beta_lower[j] <= beta[j in 1:n] <= prob.beta_upper[j])
    @variable(model, t >= 0)

    @objective(model, Min, t)

    for i in 1:m
        resid = prob.y[i] - sum(prob.X[i, j] * beta[j] for j in 1:n)
        @constraint(model, resid <= t)
        @constraint(model, -resid <= t)
    end

    @constraint(model, sum(prob.side_coef[j] * beta[j] for j in 1:n) <= prob.side_rhs)

    return model
end

# Register the variant.
register_variant(
    :regression,
    :chebyshev,
    ChebyshevRegressionProblem,
    "Chebyshev (L∞ / minimax) regression as a dense LP with coefficient box and side constraints",
)
