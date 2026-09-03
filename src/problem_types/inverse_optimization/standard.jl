using JuMP
using Random
using Distributions
using SparseArrays

"""
    InverseLPProblem <: ProblemGenerator

Generator for the classical weighted-L1 *inverse linear program* — the
Ahuja–Orlin inverse LP, the reference formulation of inverse optimization
(Ahuja & Orlin, *Operations Research* 49(5), 2001; earlier LP-duality
treatments in Zhang & Liu, *J. Comput. Appl. Math.* 72, 1996).

# Overview

A forward cost-minimization LP is given as **data**:

    minimize    c'x
    subject to  A x >= b,   x >= 0

together with an observed decision `x̂` (a production plan that was actually
run) and an analyst's prior `ĉ` on the costs. The decision variables of the
*built* model are the forward parameters to be inferred: a cost vector `c`
inside an admissible box `[ℓ, u]` around the prior, and the dual vector `y >= 0`
certifying optimality of the observation. LP duality turns "make `x̂` optimal"
into linear rows — dual feasibility `A'y <= c` column by column, plus the
strong-duality row `b'y == x̂'c` — and the objective recovers the costs closest
to the prior under weighted absolute deviation:

    minimize    Σ_j w_j (p_j + q_j)
    subject to  A'y <= c                    (dual feasibility, n rows)
                b'y == x̂'c                  (strong duality, 1 row)
                c - p + q == ĉ              (deviation split, n rows)
                0 <= y <= ȳ,  ℓ <= c <= u,  p, q >= 0

where `ȳ` collects the dual upper bounds implied by the dual-feasibility rows
and the box (`y_i <= u_j / A[i,j]` for every column `j` the row prices); they
are stated explicitly because the solver needs finite dual bounds to certify
infeasibility reliably at scale, and they cut nothing.

Because the forward model carries `x >= 0` as a variable bound, dual
feasibility is the *inequality* `A'y <= c` (the equality form `A'y == c` is
equivalent only for a forward model whose bounds are all explicit rows).
Whenever the rows hold, `x̂` is optimal for the forward LP under `c`: for any
`x` feasible in the forward model, `c'x >= (A'y)'x >= (Ax)'y >= b'y == c'x̂`.

The admissible box plays the role normalization constraints play elsewhere in
the inverse literature (the generic `‖c‖ = 1` of Chan–Lee–Terekhov, or the
`sum(c) == 1` specialization of Babier et al.): a linear forward model is
invariant to `c -> αc`, so without bounding `c` away from zero the trivial
rescaling `c -> 0, y -> 0` is always feasible. Interval coefficients are
themselves a studied inverse-LP setting (Mostafaee–Hladík–Černý, *J. Comput.
Appl. Math.* 292, 2016), and here they also encode realistic prior knowledge:
costs known up to a confidence factor.

# Planted ground truth

The instance is planted in the dual→primal direction: shadow prices `y* > 0`
are sampled on a set of active rows that covers every column, the true cost is
defined as `c* = A'y*` (dual feasibility holds with equality and complementary
slackness is automatic), the right-hand sides are derived from what the
observed plan consumes (`A x̂ = b` on active rows, slack elsewhere), and the
prior is the true cost corrupted by truncated multiplicative (lognormal) noise
— the "noisy optimum" observation model used throughout the inverse-optimization
literature. The observed plan is therefore *exactly* optimal for the true
cost, which lies inside the box, so a `feasible` request is feasible by
construction and its optimal objective value is at most the planted deviation
`Σ w_j |ĉ_j - c*_j|`.

# Feasibility profiles

  - `feasible`: stores an `InverseCostWitness` (the true cost and duals).
  - `infeasible`: the observation is strictly positive and *strictly interior*
    (`A x̂ > b` in every row) — no cost vector bounded away from zero can explain
    an interior decision, and a `StrictInteriorCertificate` proves it from the
    LP rows alone.
  - `unknown`: a coin flip between the planted mechanism with a wider noise /
    narrower box combination (the true cost may leave the box) and an unplanted
    prior, so instances land on both sides with no guarantee either way.

# Fields

  - `num_rows::Int`: Forward constraint count `m`
  - `num_cols::Int`: Forward variable count `n`
  - `forward_matrix::SparseMatrixCSC{Float64,Int}`: Technology matrix `A` (m × n, nonnegative)
  - `forward_rhs::Vector{Float64}`: Forward right-hand sides `b`
  - `reference_point::Vector{Float64}`: Observed decision `x̂`
  - `prior_cost::Vector{Float64}`: Analyst's prior cost `ĉ`
  - `cost_lower::Vector{Float64}`: Admissible cost lower bounds `ℓ`
  - `cost_upper::Vector{Float64}`: Admissible cost upper bounds `u`
  - `deviation_weights::Vector{Float64}`: Weighted-deviation weights `w`
  - `feasible_witness::Union{Nothing,InverseCostWitness}`: set for `feasible`
  - `infeasibility_certificate::Union{Nothing,StrictInteriorCertificate}`: set for `infeasible`
  - `feasibility_status::FeasibilityStatus`: Requested profile
"""
struct InverseLPProblem <: ProblemGenerator
    num_rows::Int
    num_cols::Int
    forward_matrix::SparseMatrixCSC{Float64, Int}
    forward_rhs::Vector{Float64}
    reference_point::Vector{Float64}
    prior_cost::Vector{Float64}
    cost_lower::Vector{Float64}
    cost_upper::Vector{Float64}
    deviation_weights::Vector{Float64}
    feasible_witness::Union{Nothing, InverseCostWitness}
    infeasibility_certificate::Union{Nothing, StrictInteriorCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    InverseLPProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a weighted-L1 inverse LP instance. The built model carries
`3n + m` variables — inferred costs `c` (n), deviation split `p, q` (2n), and
forward duals `y` (m) — with `n` and `m` solved from the target for a sampled
row-to-column ratio, so the target is hit exactly whenever the row count is
not clamped at its floor.
"""
function InverseLPProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    _check_inverse_target(target_variables)
    rng = MersenneTwister(seed)

    row_ratio = rand(rng, Uniform(0.5, 1.6))
    num_cols = clamp(
        round(Int, target_variables / (3.0 + row_ratio)), 4, max(4, target_variables - 6)
    )
    num_rows = max(2, target_variables - 3 * num_cols)

    data = _sample_cost_inference_data(rng, num_cols, num_rows, feasibility_status)

    return InverseLPProblem(
        num_rows,
        num_cols,
        data.forward_matrix,
        data.forward_rhs,
        data.reference_point,
        data.prior_cost,
        data.cost_lower,
        data.cost_upper,
        data.deviation_weights,
        data.feasible_witness,
        data.infeasibility_certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::InverseLPProblem)

Build the weighted-L1 inverse LP. Deterministic — uses only struct data.
"""
function build_model(prob::InverseLPProblem)
    model = Model()
    m, n = prob.num_rows, prob.num_cols
    A = prob.forward_matrix

    @variable(model, 0 <= y[i = 1:m] <= _implied_dual_upper(A, prob.cost_upper)[i])
    @variable(model, prob.cost_lower[j] <= c[j = 1:n] <= prob.cost_upper[j])
    @variable(model, dev_plus[1:n] >= 0)
    @variable(model, dev_minus[1:n] >= 0)

    # Dual feasibility: the reduced cost of every forward column is nonnegative.
    for j in 1:n
        @constraint(model, sum(A.nzval[k] * y[A.rowval[k]] for k in nzrange(A, j)) <= c[j])
    end

    # Strong duality: the dual objective attains the observed primal cost, which
    # together with dual feasibility certifies optimality of the observation.
    @constraint(
        model,
        sum(prob.forward_rhs[i] * y[i] for i in 1:m) ==
            sum(prob.reference_point[j] * c[j] for j in 1:n)
    )

    for j in 1:n
        @constraint(model, c[j] - dev_plus[j] + dev_minus[j] == prob.prior_cost[j])
    end

    @objective(
        model, Min, sum(prob.deviation_weights[j] * (dev_plus[j] + dev_minus[j]) for j in 1:n)
    )

    return model
end

register_variant(
    :inverse_optimization,
    :standard,
    InverseLPProblem,
    "Weighted-L1 inverse linear program (Ahuja-Orlin): recover the box-bounded cost vector closest to a prior that makes an observed plan optimal for a forward LP";
    default=true,
)
