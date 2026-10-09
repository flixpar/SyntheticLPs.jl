using JuMP
using Random
using Distributions
using SparseArrays
using LinearAlgebra

"""
Planted ground truth for a `feasible` `inverse_optimization/restricted_optimal_value`
instance: the cost vector `cost` and its dual certificate `duals`. The plan that
must stay optimal is `reference_point` (a struct field); under `cost` the plan
is optimal *and* its optimal value equals the requested target, with
`duals' * forward_rhs == cost' * reference_point == target_value`.
"""
struct InverseValueWitness
    cost::Vector{Float64}
    duals::Vector{Float64}
end

"""
Structured infeasibility certificate for
`inverse_optimization/restricted_optimal_value`: a *cheaper alternative plan*.
`alternative_plan` is a nonnegative forward plan with
`forward_matrix * alternative_plan >= forward_rhs`, and even when every cost
sits at its upper bound it costs `alternative_value = cost_upper' *
alternative_plan`, strictly below the requested `target_value`.

Any point satisfying the built rows gives the chain (with `x̄` the alternative,
`y >= 0`, `x̄ >= 0`)

    τ = b'y <= (A x̄)'y = x̄'(A'y) <= x̄'c <= x̄'u = alternative_value < τ,

using the strong-duality row, forward feasibility of `x̄`, the dual-feasibility
rows `A'y <= c`, and the box `c <= u` — a contradiction from model rows alone.
In words: weak duality caps every certifiable optimal value at the cost of any
feasible plan, and the target asks for more than the cheaper plan costs even at
the most expensive admissible prices. Unlike an out-of-range target on the
single pricing row, refuting it needs the aggregation of every
dual-feasibility row with the plan's weights, so presolve cannot detect it and
simplex has to do the work. `margin = target_value - alternative_value > 0`.
"""
struct CheaperPlanCertificate
    alternative_plan::Vector{Float64}
    alternative_value::Float64
    target_value::Float64
    margin::Float64
end

"""
    InverseOptimalValueProblem <: ProblemGenerator

Generator for the *inverse optimal value problem* in its LP-representable
(restricted) form of Jia–Guan–Qian–Pardalos (2023): instead of explaining an
observed decision, move the optimal value of the forward model onto a target.

# Overview

The forward cost-minimization LP `min c'x s.t. Ax >= b, x >= 0` is given as
data, together with an observed plan `x⁰` that currently solves it, a prior `ĉ`
on the costs, and a management target `τ` for the optimal value (a budget the
plan's cost must hit). The decision variables of the *built* model are the
inferred costs `c` inside an admissible box and the duals `y` certifying the
plan's optimality; the value target enters as one extra equality:

    minimize    Σ_j w_j (p_j + q_j)
    subject to  A'y <= c                       (dual feasibility, n rows)
                b'y == τ                       (strong duality at the target, 1 row)
                c'x⁰ == τ                      (plan prices at the target, 1 row)
                c - p + q == ĉ                 (deviation split, n rows)
                0 <= y <= ȳ,  ℓ <= c <= u,  p, q >= 0

where `ȳ` collects the dual upper bounds implied by dual feasibility and the
admissible box (`y_i <= u_j / A[i,j]`); valid for every solution, cutting
nothing.

The strong-duality row pins the dual objective to the target while the pricing
row pins the plan's cost to it, so whenever the rows hold the plan is optimal
for the forward LP under `c` *and* the optimal value equals `τ` exactly.

Why this restricted form: the general inverse optimal value problem of Ahmed &
Guan (*Mathematical Programming* 102(1), 2005) is NP-hard — the optimal value
`z(·)` is concave piecewise-linear in the adjusted data, so its *super*level
sets are unions of half-spaces, and encoding attainment from below is
disjunctive. (Concretely, writing the attainment requirement through LP duality
needs a row `(b + Δb)'y >= τ`, bilinear in the adjustment and the duals.)
Fixing the plan that must remain optimal — the restriction of Jia et al. —
keeps every row linear, which is what makes this generator an LP.

# Planted ground truth

As in `standard`, shadow prices `ỹ > 0` are sampled on active rows covering
every column, the true cost is `c* = A'ỹ`, and the right-hand sides are derived
from what the plan consumes. The target is then planted by *rescaling the
ground truth*: `c = α c*`, `y = α ỹ` with `α = 1 ± ε` keeps every optimality
row satisfied while moving the value to `τ = α · c*'x⁰`, and the prior is
`α c*` corrupted by truncated multiplicative noise — so the planted pair
satisfies every row of the built model, the target differs from the current
value `c*'x⁰` by a genuine margin, and the optimal objective is at most the
planted deviation.

# Feasibility profiles

  - `feasible`: stores an `InverseValueWitness` (the rescaled cost and duals).
  - `infeasible`: the forward rows are relaxed just enough that a perturbed
    *alternative plan* `x̄` becomes feasible, priced at `u'x̄ ≈ (0.74–0.86) u'x⁰`
    even at the upper cost bounds, and the target is set 4–10% above `u'x̄`.
    The target stays well inside the pricing range `[ℓ'x⁰, u'x⁰]` of the
    observed plan, so neither value row is out of range on its own; the
    contradiction is the weak-duality bound `b'y <= u'x̄`, refuted by a
    `CheaperPlanCertificate` that aggregates every dual-feasibility row
    (invisible to presolve, found by simplex).
  - `unknown`: the rescaling factor and the box radius are sampled independently
    (the planted cost may leave the box), so the target may or may not be
    attainable, with no guarantee either way.

# Fields

  - `num_rows::Int`: Forward constraint count `m`
  - `num_cols::Int`: Forward variable count `n`
  - `forward_matrix::SparseMatrixCSC{Float64,Int}`: Technology matrix `A` (m × n, nonnegative)
  - `forward_rhs::Vector{Float64}`: Forward right-hand sides `b`
  - `reference_point::Vector{Float64}`: Observed plan `x⁰`, which must stay optimal
  - `prior_cost::Vector{Float64}`: Analyst's prior cost `ĉ`
  - `cost_lower::Vector{Float64}`, `cost_upper::Vector{Float64}`: Admissible box
  - `target_value::Float64`: Requested optimal value `τ`
  - `deviation_weights::Vector{Float64}`: Weighted-deviation weights `w`
  - `feasible_witness::Union{Nothing,InverseValueWitness}`: set for `feasible`
  - `infeasibility_certificate::Union{Nothing,CheaperPlanCertificate}`: set for `infeasible`
  - `feasibility_status::FeasibilityStatus`: Requested profile
"""
struct InverseOptimalValueProblem <: ProblemGenerator
    num_rows::Int
    num_cols::Int
    forward_matrix::SparseMatrixCSC{Float64, Int}
    forward_rhs::Vector{Float64}
    reference_point::Vector{Float64}
    prior_cost::Vector{Float64}
    cost_lower::Vector{Float64}
    cost_upper::Vector{Float64}
    target_value::Float64
    deviation_weights::Vector{Float64}
    feasible_witness::Union{Nothing, InverseValueWitness}
    infeasibility_certificate::Union{Nothing, CheaperPlanCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    InverseOptimalValueProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an inverse optimal value instance. The built model carries `3n + m`
variables — costs `c` (n), deviation split `p, q` (2n), and forward duals `y`
(m) — with a row-to-column ratio sampled in a narrower band than `standard` so
the two variants produce differently shaped dual blocks.
"""
function InverseOptimalValueProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    _check_inverse_target(target_variables)
    rng = MersenneTwister(seed)

    row_ratio = rand(rng, Uniform(0.3, 0.9))
    num_cols = clamp(
        round(Int, target_variables / (3.0 + row_ratio)), 4, max(4, target_variables - 6)
    )
    num_rows = max(2, target_variables - 3 * num_cols)

    # The infeasible profile starts from a planted (feasible-profile) instance
    # and removes attainability through the target alone; see below.
    sample_status = feasibility_status == infeasible ? feasible : feasibility_status
    data = _sample_cost_inference_data(rng, num_cols, num_rows, sample_status)

    n = num_cols
    prior = data.prior_cost
    lower = data.cost_lower
    upper = data.cost_upper
    plan = data.reference_point

    # The target is planted by rescaling the ground truth: α c* with duals α ỹ
    # satisfies every optimality row while moving the plan's value to
    # α · c*'x⁰. The prior and its box are rescaled with it, so the box keeps
    # containing the planted cost (they are relative intervals).
    feasible_witness = nothing
    certificate = nothing
    target = 0.0
    forward_rhs = data.forward_rhs
    if feasibility_status == infeasible
        # Cheaper alternative plan: perturb the observed plan multiplicatively
        # (substituting between activities) and rescale it so that, even at
        # the upper cost bounds, it costs 74–86% of the observed plan's
        # ceiling price. Lowering the right-hand sides to what both plans
        # consume keeps the observed plan feasible and makes the alternative
        # feasible too; the target is then set 4–10% above the alternative's
        # ceiling cost, which weak duality forbids.
        A = data.forward_matrix
        alternative = plan .* exp.(rand(rng, Normal(0.0, 0.35), n))
        ceiling_ratio = rand(rng, Uniform(0.74, 0.86))
        alternative .*= ceiling_ratio * dot(upper, plan) / dot(upper, alternative)
        alternative_use = A * alternative
        forward_rhs = [
            min(forward_rhs[i], floor(alternative_use[i] * 100.0) / 100.0) for
            i in eachindex(forward_rhs)
        ]
        alternative_value = dot(upper, alternative)
        target = alternative_value * (1.0 + rand(rng, Uniform(0.04, 0.10)))
        certificate = CheaperPlanCertificate(
            alternative, alternative_value, target, target - alternative_value
        )
    elseif data.true_cost !== nothing
        rescale = 1.0 + (rand(rng) < 0.5 ? -1.0 : 1.0) * (
            if feasibility_status == feasible
                rand(rng, Uniform(0.03, 0.18))
            else
                rand(rng, Uniform(0.05, 0.40))
            end
        )
        prior .*= rescale
        lower .*= rescale
        upper .*= rescale
        target = sum(rescale * data.true_cost[j] * plan[j] for j in 1:n)
        if feasibility_status == feasible
            feasible_witness = InverseValueWitness(
                rescale .* data.true_cost, rescale .* data.true_duals
            )
        end
    else
        # Unplanted prior (unknown profile): price the plan at a drifted
        # multiple of the prior's own pricing, with no guarantee of
        # attainability either way.
        drift = 1.0 + (rand(rng) < 0.5 ? -1.0 : 1.0) * rand(rng, Uniform(0.05, 0.35))
        target = sum(prior[j] * plan[j] for j in 1:n) * drift
    end

    return InverseOptimalValueProblem(
        num_rows,
        num_cols,
        data.forward_matrix,
        forward_rhs,
        plan,
        prior,
        lower,
        upper,
        target,
        data.deviation_weights,
        feasible_witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::InverseOptimalValueProblem)

Build the inverse optimal value LP. Deterministic — uses only struct data.
"""
function build_model(prob::InverseOptimalValueProblem)
    model = Model()
    m, n = prob.num_rows, prob.num_cols
    A = prob.forward_matrix

    # Computed once: evaluating it inside the macro would redo the O(nnz)
    # transpose for every dual and make the build quadratic.
    dual_upper = _implied_dual_upper(A, prob.cost_upper)
    @variable(model, 0 <= y[i = 1:m] <= dual_upper[i])
    @variable(model, prob.cost_lower[j] <= c[j = 1:n] <= prob.cost_upper[j])
    @variable(model, dev_plus[1:n] >= 0)
    @variable(model, dev_minus[1:n] >= 0)

    for j in 1:n
        @constraint(model, sum(A.nzval[k] * y[A.rowval[k]] for k in nzrange(A, j)) <= c[j])
    end

    # Strong duality and the pricing row, both pinned to the target.
    @constraint(model, sum(prob.forward_rhs[i] * y[i] for i in 1:m) == prob.target_value)
    @constraint(model, sum(prob.reference_point[j] * c[j] for j in 1:n) == prob.target_value)

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
    :restricted_optimal_value,
    InverseOptimalValueProblem,
    "Inverse optimal value problem (Ahmed-Guan / Jia-Guan-Qian-Pardalos restricted LP form): adjust box-bounded costs minimally so an observed plan stays optimal while its optimal value hits a target";
    tags=[:production],
    min_target_variables=2,
    max_target_variables=250_000,
)
