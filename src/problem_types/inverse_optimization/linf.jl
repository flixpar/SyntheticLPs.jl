using JuMP
using Random
using Distributions
using SparseArrays

"""
    InverseLPMaxErrorProblem <: ProblemGenerator

Generator for the min–max (weighted L∞) *inverse linear program*: the same
cost-inference problem as `inverse_optimization/standard` under the uniform
deviation norm — the other polynomially-solvable member of the p-norm family of
inverse LPs studied by Ahuja–Orlin (*Operations Research* 49(5), 2001).

# Overview
The forward LP, the observed decision `x̂`, the prior `ĉ` and the admissible box
`[ℓ, u]` are exactly as in `standard`; only the distance being minimized
changes. Instead of the sum of absolute deviations the model minimizes the
largest weighted deviation, linearized with one epigraph variable `t`:

    minimize    t
    subject to  A'y <= c                       (dual feasibility, n rows)
                b'y == x̂'c                     (strong duality, 1 row)
                w_j (c_j - ĉ_j) <= t            (upper deviation, n rows)
                w_j (ĉ_j - c_j) <= t            (lower deviation, n rows)
                0 <= y <= ȳ,  ℓ <= c <= u,  t >= 0

with `ȳ` the implied dual upper bounds of `standard` (valid, cutting nothing,
and needed for the solver to certify infeasibility at scale).

Min–max deviation is the natural objective when the costs must be recovered
*uniformly* well — e.g. tariffs or prices that may not move by more than a
common relative amount — and it produces a different dual structure and optimal
face than the L1 model on the same data.

# Planted ground truth and feasibility profiles
Identical to `standard`: the feasible profile plants shadow prices and derives
the true cost as `A'y*` (an `InverseCostWitness`); the infeasible profile makes
the observation strictly positive and strictly interior, refuted by a
`StrictInteriorCertificate` that uses LP rows alone; the unknown profile is the
same unguaranteed coin flip. The epigraph variable and rows are the only
structural difference, so the instance data is shared with `standard` and both
variants exercise the same ground truth under two norms.

# Fields
Same as `InverseLPProblem`.
"""
struct InverseLPMaxErrorProblem <: ProblemGenerator
    num_rows::Int
    num_cols::Int
    forward_matrix::SparseMatrixCSC{Float64,Int}
    forward_rhs::Vector{Float64}
    reference_point::Vector{Float64}
    prior_cost::Vector{Float64}
    cost_lower::Vector{Float64}
    cost_upper::Vector{Float64}
    deviation_weights::Vector{Float64}
    feasible_witness::Union{Nothing,InverseCostWitness}
    infeasibility_certificate::Union{Nothing,StrictInteriorCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    InverseLPMaxErrorProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a min–max inverse LP instance. The direct epigraph formulation
carries `n + m + 1` variables: inferred costs, forward duals, and one maximum
deviation. It avoids the redundant split variables used by an L1 model.
"""
function InverseLPMaxErrorProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    _check_inverse_target(target_variables)
    rng = MersenneTwister(seed)

    row_ratio = rand(rng, Uniform(0.5, 1.6))
    num_cols = clamp(round(Int, (target_variables - 1) / (1.0 + row_ratio)), 4,
                     max(4, target_variables - 3))
    num_rows = max(2, (target_variables - 1) - num_cols)

    data = _sample_cost_inference_data(rng, num_cols, num_rows, feasibility_status)

    return InverseLPMaxErrorProblem(num_rows, num_cols,
                                    data.forward_matrix, data.forward_rhs,
                                    data.reference_point, data.prior_cost,
                                    data.cost_lower, data.cost_upper,
                                    data.deviation_weights,
                                    data.feasible_witness, data.infeasibility_certificate,
                                    feasibility_status)
end

"""
    build_model(prob::InverseLPMaxErrorProblem)

Build the min–max inverse LP. Deterministic — uses only struct data.
"""
function build_model(prob::InverseLPMaxErrorProblem)
    model = Model()
    m, n = prob.num_rows, prob.num_cols
    A = prob.forward_matrix

    @variable(model, 0 <= y[i=1:m] <= _implied_dual_upper(A, prob.cost_upper)[i])
    @variable(model, prob.cost_lower[j] <= c[j=1:n] <= prob.cost_upper[j])
    @variable(model, max_dev >= 0)

    for j in 1:n
        @constraint(model,
                    sum(A.nzval[k] * y[A.rowval[k]] for k in nzrange(A, j)) <= c[j])
    end

    @constraint(model,
                sum(prob.forward_rhs[i] * y[i] for i in 1:m) ==
                sum(prob.reference_point[j] * c[j] for j in 1:n))

    for j in 1:n
        @constraint(model,
                    prob.deviation_weights[j] * (c[j] - prob.prior_cost[j]) <= max_dev)
        @constraint(model,
                    prob.deviation_weights[j] * (prob.prior_cost[j] - c[j]) <= max_dev)
    end

    @objective(model, Min, max_dev)

    return model
end

register_variant(
    :inverse_optimization,
    :linf,
    InverseLPMaxErrorProblem,
    "Min-max (weighted L-infinity) inverse linear program: minimize the largest weighted cost deviation that makes an observed plan optimal for a forward LP",
)
