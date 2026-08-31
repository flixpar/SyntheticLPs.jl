# Shared types and sampling machinery for the `inverse_optimization` category.
#
# Every variant in this category is an *inverse* problem: a forward LP is given
# as data together with an observation of its solution (or a target for its
# optimal value), and the decision variables of the built model are the forward
# parameters to be inferred. The generators plant a ground truth first — a cost
# vector plus a dual certificate that makes the observation optimal — and derive
# the observable data from it, mirroring how inverse-optimization benchmarks are
# built in the literature: the observation comes from a decision maker who
# optimizes against a *true* parameter that differs from the analyst's prior.

using SparseArrays
using Random
using Distributions
using StatsBase
using LinearAlgebra
using Statistics

"""
    INVERSE_MAX_VARIABLES

Documented ceiling on `target_variables` for every `inverse_optimization`
variant. Each inverse formulation expands its forward LP by a constant factor
(three to five times as many columns, plus a full dual block), so 250,000
inverse variables already encode a forward model with tens of thousands of
columns and an equal number of dual-feasibility rows. Larger requests raise
`ArgumentError` instead of silently undersizing, mirroring the caps documented
on `supply_chain/network_planning` and `telecom_network_design/standard`.
"""
const INVERSE_MAX_VARIABLES = 250_000

"""
    _check_inverse_target(target_variables)

Validate a `target_variables` request for an `inverse_optimization` variant:
at least 2 variables and at most [`INVERSE_MAX_VARIABLES`](@ref).
"""
function _check_inverse_target(target_variables::Int)
    target_variables >= 2 || throw(ArgumentError(
        "target_variables must be >= 2 (got $target_variables)."))
    target_variables <= INVERSE_MAX_VARIABLES || throw(ArgumentError(
        "inverse_optimization variants are capped at $INVERSE_MAX_VARIABLES " *
        "variables: each inverse model expands its forward LP by a constant " *
        "factor in both variables and rows (got request for " *
        "$target_variables)."))
end

"""
Planted ground truth for a `feasible` cost-inference instance
(`inverse_optimization/standard` and `/linf`): the true cost vector `cost` —
the parameter the inverse problem is asked to recover — and the dual
certificate `duals` proving `reference_point` optimal for the forward LP under
that cost.

By construction `cost = forward_matrix' * duals` with `duals >= 0` supported on
the rows that are active at the reference point, so dual feasibility holds with
equality everywhere and complementary slackness is automatic. The stored pair
satisfies every row of the built inverse model whenever `cost` lies inside the
admissible box.
"""
struct InverseCostWitness
    cost::Vector{Float64}
    duals::Vector{Float64}
end

"""
Structured infeasibility certificate for cost-inference instances
(`inverse_optimization/standard` and `/linf`): the observed decision is strictly
positive and *strictly interior* to the forward polyhedron — `slacks` stores
`A*x̂ - b`, all entries strictly positive — while every admissible cost vector
is bounded below by `cost_lower > 0` componentwise.

Any (cost, dual) pair satisfying the built rows must obey
`x̂'c >= (A'y)'x̂ = (A*x̂)'y = (b + slacks)'y >= b'y + min(slacks)'y`, so the
strong-duality row `b'y == x̂'c` forces `y = 0` (all slacks positive, `y >= 0`),
which in turn forces `x̂'c = 0 >= cost_lower'x̂ > 0` — a contradiction. An
interior observation simply cannot be explained by any positive cost vector,
which is exactly the "decision maker is not cost-minimizing" situation the
inverse-optimization literature uses for unexplainable observations. The
refutation uses LP rows alone, so it survives `relax_integer` and
`bounds_to_constraints` (the model is an LP to begin with).
"""
struct StrictInteriorCertificate
    slacks::Vector{Float64}
end

"""
    _sample_technology_matrix(rng, m, n)

Sparse nonnegative `m×n` resource-consumption matrix: each column consumes
between 3 and 10 resources with lognormal coefficients clipped to `[0.1, 6.0]`
and rounded to two decimals (measured process data). Every column is nonempty by
construction; empty rows are repaired by planting one consumption entry in a
random column, so downstream right-hand sides stay positive.
"""
function _sample_technology_matrix(rng::AbstractRNG, m::Int, n::Int)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    row_touched = falses(m)
    max_nnz = max(1, min(m, 10))
    min_nnz = min(3, max_nnz)
    for j in 1:n
        k = rand(rng, min_nnz:max_nnz)
        for i in sample(rng, 1:m, k; replace=false)
            v = round(rand(rng, LogNormal(log(1.1), 0.5)); digits=2)
            v = clamp(v, 0.1, 6.0)
            push!(rows, i)
            push!(cols, j)
            push!(vals, v)
            row_touched[i] = true
        end
    end
    # No empty rows: plant one consumption entry so no right-hand side is
    # forced to zero (or below) by a vacuous row.
    for i in 1:m
        row_touched[i] && continue
        j = rand(rng, 1:n)
        push!(rows, i)
        push!(cols, j)
        push!(vals, round(rand(rng, Uniform(0.05, 4.0)); digits=2))
    end
    return sparse(rows, cols, vals, m, n)
end

"""
    _sample_cost_inference_data(rng, n, m, status)

Sample the shared pieces of a cost-inference instance (`standard`, `linf`, and
`optimal_value` variants): the forward rows `A`, `b`; the observed
`reference_point`; the analyst's `prior_cost` and the admissible box
`cost_lower`/`cost_upper` around it; heteroscedastic `deviation_weights`; the
status-specific metadata — a planted [`InverseCostWitness`](@ref) for
`feasible` and a [`StrictInteriorCertificate`](@ref) for `infeasible` — and the
planted ground truth `true_cost`/`true_duals` themselves (`nothing` whenever no
dual was planted), so other variants can build value targets on top of it.

The feasible profile is planted in the direction dual→primal: duals are sampled
first on a set of active rows, the true cost is defined as `A'y*` (dual
feasibility with equality, complementary slackness automatic), the
right-hand sides are derived from what the observed plan consumes, and the
prior is the true cost corrupted by truncated multiplicative noise. The
observed plan is therefore *exactly* optimal for the true cost, and the inverse
problem always has the planted solution available inside its box.

The `unknown` profile flips a fair coin between the planted mechanism with a
wider noise / narrower box combination (the true cost may fall outside the box,
so the instance is not guaranteed feasible) and an unplanted mechanism whose
prior is sampled independently of any dual structure — the observation may or
may not be explainable, with no guarantee either way.
"""
function _sample_cost_inference_data(rng::AbstractRNG, n::Int, m::Int,
                                     status::FeasibilityStatus)
    A = _sample_technology_matrix(rng, m, n)

    # --- Observed plan ----------------------------------------------------
    support_prob = status == infeasible ? 1.0 :
                   status == feasible ? rand(rng, Uniform(0.70, 0.95)) :
                                       rand(rng, Uniform(0.50, 0.95))
    volume_center = rand(rng, LogNormal(log(rand(rng, Uniform(6.0, 60.0))), 0.3))
    reference_point = [rand(rng) < support_prob ?
                       round(rand(rng, LogNormal(log(volume_center), 0.5)); digits=2) :
                       0.0 for _ in 1:n]
    if all(iszero, reference_point)
        reference_point[rand(rng, 1:n)] = round(rand(rng, LogNormal(log(volume_center), 0.5)); digits=2)
    end
    # Unit normalization: a plan is measured in whatever units make its average
    # committed quantity a two-digit number. This keeps the strong-duality row's
    # coefficient range (b on one side, x̂ on the other) uniformly scaled
    # regardless of the sampled volume regime, which the LP solver needs to
    # decide large instances reliably.
    unit_scale = 20.0 / mean(reference_point[findall(!iszero, reference_point)])
    reference_point .= round.(reference_point .* unit_scale; digits=2)

    consumption = A * reference_point
    # Repair rows whose support misses the plan's support: consumption must be
    # strictly positive so every derived right-hand side is positive.
    support = findall(!iszero, reference_point)
    for i in 1:m
        consumption[i] > 0 && continue
        j = rand(rng, support)
        planted = round(rand(rng, Uniform(0.05, 4.0)); digits=2)
        A[i, j] = planted
        consumption[i] = planted * reference_point[j]
    end

    witness = nothing
    certificate = nothing
    true_cost = nothing
    true_duals = nothing

    if status == infeasible
        # Strictly interior observation: every row slack, every coordinate
        # positive. No positive cost vector can make it optimal. The margins
        # are deliberately loud (deep slack, tight cost box) so the refutation
        # is numerically easy for an LP solver to certify at scale.
        slacks = consumption .* rand(rng, Uniform(0.25, 0.60), m)
        forward_rhs = consumption .- slacks
        prior_cost = round.(rand(rng, LogNormal(log(rand(rng, Uniform(1.0, 15.0))), 0.45), n);
                            sigdigits=4)
        kappa = rand(rng, Uniform(0.20, 0.45))
        certificate = StrictInteriorCertificate(copy(slacks))
    else
        planted = status == feasible || rand(rng) < 0.55
        if planted
            # Duals first: y* > 0 on a set of active rows covering every
            # column, so c* = A'y* is strictly positive and every column is
            # complementary by construction.
            active_prob = status == feasible ? rand(rng, Uniform(0.25, 0.60)) :
                                              rand(rng, Uniform(0.40, 0.80))
            active = [i for i in 1:m if rand(rng) < active_prob]
            isempty(active) && push!(active, rand(rng, 1:m))
            covered = _columns_reached(A, active, n)
            for j in 1:n
                covered[j] && continue
                # Anchor the column with one more active row so the true cost
                # stays strictly positive in every coordinate.
                nzrows = view(A.rowval, nzrange(A, j))
                push!(active, rand(rng, nzrows))
            end
            sort!(unique!(active))

            true_duals = zeros(m)
            price_center = rand(rng, Uniform(0.5, 5.0))
            for i in active
                true_duals[i] = round(rand(rng, LogNormal(log(price_center), 0.45)); digits=2)
            end
            true_cost = A' * true_duals

            if status == feasible
                noise_sigma = rand(rng, Uniform(0.06, 0.25))
                # The box must contain the true cost: keep its radius safely
                # above the truncation bound of the prior noise.
                kappa = max(rand(rng, Uniform(0.35, 0.80)), 2.7 * noise_sigma)
            else
                noise_sigma = rand(rng, Uniform(0.12, 0.32))
                kappa = rand(rng, Uniform(0.35, 0.90))
            end
            eta = rand(rng, truncated(Normal(0.0, noise_sigma),
                                      -2.5 * noise_sigma, 2.5 * noise_sigma), n)
            prior_cost = round.(true_cost .* exp.(eta); sigdigits=4)

            headroom = rand(rng, Uniform(0.08, 0.45), m)
            active_set = Set(active)
            forward_rhs = [i in active_set ? consumption[i] :
                           round(consumption[i] * (1.0 - headroom[i]); digits=2)
                           for i in 1:m]
            if status == feasible
                witness = InverseCostWitness(true_cost, true_duals)
            end
        else
            # Unplanted: active rows exist (so a dual might too), but the
            # prior carries no information about any consistent cost.
            active_prob = rand(rng, Uniform(0.40, 0.80))
            active = [i for i in 1:m if rand(rng) < active_prob]
            isempty(active) && push!(active, rand(rng, 1:m))
            sort!(unique!(active))
            active_set = Set(active)
            headroom = rand(rng, Uniform(0.08, 0.45), m)
            forward_rhs = [i in active_set ? consumption[i] :
                           round(consumption[i] * (1.0 - headroom[i]); digits=2)
                           for i in 1:m]
            prior_cost = round.(rand(rng, LogNormal(log(rand(rng, Uniform(1.0, 12.0))), 0.5), n);
                                sigdigits=4)
            kappa = rand(rng, Uniform(0.40, 0.90))
        end
    end

    cost_lower = prior_cost .* exp.(-kappa)
    cost_upper = prior_cost .* exp.(kappa)

    # Heteroscedastic deviation weights: the prior on expensive coefficients is
    # measured in relative terms, so its absolute precision is lower. Rescaled
    # to unit mean so objective magnitudes stay comparable across instances.
    precision = rand(rng, Uniform(0.05, 0.30), n)
    deviation_weights = 1.0 ./ (precision .* max.(prior_cost, 1e-8))
    deviation_weights .*= n / sum(deviation_weights)

    return (forward_matrix = A,
            forward_rhs = forward_rhs,
            reference_point = reference_point,
            prior_cost = prior_cost,
            cost_lower = cost_lower,
            cost_upper = cost_upper,
            deviation_weights = deviation_weights,
            feasible_witness = witness,
            infeasibility_certificate = certificate,
            true_cost = true_cost,
            true_duals = true_duals)
end

"""
    _columns_reached(A, rows, n)

Columns of `A` having a nonzero in one of `rows`.
"""
function _columns_reached(A::SparseMatrixCSC, rows::Vector{Int}, n::Int)
    reached = falses(n)
    rowset = Set(rows)
    for j in 1:n
        for k in nzrange(A, j)
            if A.rowval[k] in rowset
                reached[j] = true
                break
            end
        end
    end
    return reached
end

"""
    _implied_dual_upper(A, cost_upper) -> Vector{Float64}

Valid implied upper bounds on the forward duals: for any row `i` and any column
`j` with `A[i, j] > 0`, dual feasibility `A'y <= c` together with `y >= 0` and
`c <= cost_upper` forces `y_i <= cost_upper[j] / A[i, j]`. (For a fixed cost
vector pass it as `cost_upper`.)

Stating these bounds explicitly does not change any variant's feasible set —
they are implied by the model's own rows — but it gives the solver finite,
propagatable bounds on the dual block. Without them, certifying the
*infeasibility* of a large interior-observation instance requires a Farkas
argument spanning every dual-feasibility row at once, and simplex phase one
stalls numerically above a few thousand variables; with them the models stay
decidable at any supported size.
"""
function _implied_dual_upper(A::SparseMatrixCSC, cost_upper::Vector{Float64})
    bounds = fill(Inf, size(A, 1))
    AT = SparseMatrixCSC(transpose(A))
    for i in 1:size(A, 1)
        tightest = Inf
        for k in nzrange(AT, i)
            j = AT.rowval[k]
            tightest = min(tightest, cost_upper[j] / AT.nzval[k])
        end
        isfinite(tightest) && (bounds[i] = tightest * (1.0 + 1e-9))
    end
    return bounds
end

"""
    _around(x, lo)

Search order around a starting value, floored at `lo`: `x, x-1, x+1, x-2, ...`.
Used by the sizing searches that must land a shape on an exact variable count.
"""
function _around(x::Int, lo::Int)
    out = Int[x]
    for k in 1:(2 * max(x - lo, 8))
        push!(out, iseven(k) ? x + k ÷ 2 : x - (k + 1) ÷ 2)
    end
    return [i for i in out if i >= lo]
end
