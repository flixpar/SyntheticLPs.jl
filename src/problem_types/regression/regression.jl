# regression category
#
# Entry point for the `regression` problem category: linear programs arising from
# robust / constrained statistical regression. Unlike the sparse, combinatorial
# network and allocation generators, these produce *dense* data-matrix LPs, which
# diversifies the numerical and structural profile of the test set.
#
# The three variants (least-absolute-deviations, quantile, and Chebyshev/minimax
# regression) share the same underlying data — a design matrix, a response, box
# bounds on the coefficients, and a coefficient side constraint — so the common
# data generation lives here and each variant file only differs in its objective
# (the loss being minimized).

using Random
using Distributions

"""
    generate_regression_data(n_features, n_samples, feasibility_status)

Generate the shared data for a constrained regression LP: a dense design matrix
`X` (n_samples × n_features), a response `y`, per-coefficient box bounds
`[beta_lower, beta_upper]`, and a single coefficient side constraint
`dot(side_coef, beta) <= side_rhs`.

The regression loss itself is always feasible and bounded; feasibility is
controlled through the side constraint relative to the coefficient box:
- `feasible`: `side_rhs` lies within the achievable range, so the box (and hence
  the feasible region) is non-empty.
- `infeasible`: `side_rhs` is set strictly below `sum(beta_lower)` (the minimum
  achievable value of `dot(ones, beta)`), making the region empty.
- `unknown`: resolved at random to either `feasible` or `infeasible`.

Returns a `NamedTuple` `(X, y, beta_lower, beta_upper, side_coef, side_rhs)`.
"""
function generate_regression_data(n_features::Int, n_samples::Int,
                                  feasibility_status::FeasibilityStatus)
    n = n_features
    m = n_samples

    # Ground-truth coefficients and a dense Gaussian design matrix.
    true_beta = rand(Uniform(-2.0, 2.0), n)
    X = rand(Normal(0.0, 1.0), m, n)
    noise = rand(Normal(0.0, rand(Uniform(0.1, 0.5))), m)
    y = X * true_beta .+ noise

    # Coefficient box bounds, generous around the truth so the unconstrained
    # optimum is interior (keeps the feasible/unknown cases realistic).
    half_width = rand(Uniform(3.0, 6.0), n)
    beta_lower = true_beta .- half_width
    beta_upper = true_beta .+ half_width

    # Side constraint on the coefficients: dot(side_coef, beta) <= side_rhs.
    side_coef = ones(Float64, n)               # constraint on the sum of coefficients
    lo = sum(beta_lower)                        # min achievable value of the sum
    hi = sum(beta_upper)                        # max achievable value of the sum
    span = hi - lo

    actual_status = feasibility_status
    if feasibility_status == unknown
        actual_status = rand() < 0.7 ? feasible : infeasible
    end

    if actual_status == feasible
        side_rhs = lo + rand(Uniform(0.4, 0.9)) * span
    else
        side_rhs = lo - rand(Uniform(1.0, 3.0))     # strictly below the minimum sum
    end

    return (X = X, y = y, beta_lower = beta_lower, beta_upper = beta_upper,
            side_coef = side_coef, side_rhs = side_rhs)
end

include("lad.jl")
include("quantile.jl")
include("chebyshev.jl")
