using JuMP
using LinearAlgebra
using Random
using SparseArrays

const BASIS_PURSUIT_PROFILES = (:gaussian, :correlated_columns, :sparse_measurements)

"""
Nonzero budget for the measurement matrix: each column keeps at most
`max(8, BASIS_PURSUIT_NNZ_BUDGET ÷ n_features)` entries, so the split LP has at
most ≈ `2 · BASIS_PURSUIT_NNZ_BUDGET` nonzeros (3M) at any size. Small instances
(up to roughly 1.5k features) stay fully dense.
"""
const BASIS_PURSUIT_NNZ_BUDGET = 1_500_000

"""
    BasisPursuitCertificate

Algebraic proof that a basis-pursuit instance is infeasible:
`Σ_k multipliers[k] * A[rows[k], :] == 0` (up to roundoff) while
`Σ_k multipliers[k] * b[rows[k]] == rhs_gap` with `|rhs_gap|` bounded away from
zero. One measurement row is a linear combination of 2–4 others whose
right-hand side disagrees with the same combination, so the contradiction needs
at least three rows — it is not a parallel-row pair that presolve can spot.
"""
struct BasisPursuitCertificate
    rows::Vector{Int}
    multipliers::Vector{Float64}
    rhs_gap::Float64
end

"""
    BasisPursuitProblem <: ProblemGenerator

Weighted basis pursuit:

```math
\\min_x \\sum_j w_j |x_j| \\quad \\text{s.t.} \\quad A x = b.
```

The model uses nonnegative positive/negative splits `x = x_pos - x_neg`.
Every instance stores its matrix profile, source sparse signal, resolved
feasibility status, and (exactly when infeasible) a multi-row dependency
certificate. `source_signal` generated the RHS before certificate injection; it
is a feasible witness only when `resolved_status == feasible`.

The three matrix profiles have materially different structure:

  - `gaussian`: Gaussian columns; when the instance is small enough to be dense
    the matrix is whitened to orthonormal measurement rows, otherwise each
    column has a random support of the per-column budget (sparse Gaussian
    sensing).
  - `correlated_columns`: groups of highly coherent columns generated from
    shared latent prototypes (on a shared support) plus small orthogonal
    perturbations;
  - `sparse_measurements`: sparse signed measurements (≈12% of rows per
    column, never above the budget) with randomized supports.

Columns are capped at the [`BASIS_PURSUIT_NNZ_BUDGET`](@ref) nonzero budget, so
the model has at most ≈3M nonzeros (≈ 30 nonzeros per column at 100k
variables) and builds in seconds.

The split formulation always has an even number of variables. An even target of
at least two is met exactly; an odd target is rounded up by one; targets below
two produce the minimum two-variable formulation.

`unknown` requests resolve to a planted feasible instance with probability 0.8
and a certified infeasible one otherwise (stored in `resolved_status`): an
underdetermined full-row-rank system is always consistent, so there is no
natural borderline to sample.
"""
struct BasisPursuitProblem <: ProblemGenerator
    n_features::Int
    n_measurements::Int
    profile::Symbol
    resolved_status::FeasibilityStatus
    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    weights::Vector{Float64}
    source_signal::Vector{Float64}
    support::Vector{Int}
    certificate::Union{Nothing, BasisPursuitCertificate}
end

"""Per-column nonzero count under the budget."""
_basis_pursuit_column_nnz(n_measurements::Int, n_features::Int) =
    min(n_measurements, max(8, BASIS_PURSUIT_NNZ_BUDGET ÷ max(1, n_features)))

"""
    _basis_pursuit_append_column!(I, J, V, rows, vals, j)

Append column `j` to COO arrays, dropping entries below `1e-3` of the column's
largest magnitude: near-zero Gaussian draws add nothing to the measurement model
but widen the matrix coefficient range (and the simplex's numerical trouble) by
orders of magnitude.
"""
function _basis_pursuit_append_column!(I, J, V, rows, vals, j)
    cutoff = 1e-3 * maximum(abs, vals)
    for k in eachindex(rows)
        abs(vals[k]) < cutoff && continue
        push!(I, rows[k])
        push!(J, j)
        push!(V, vals[k])
    end
    return nothing
end

function _basis_pursuit_gaussian_matrix(
    rng::AbstractRNG, n_measurements::Int, n_features::Int, width::Int
)
    if width >= n_measurements
        A = randn(rng, n_measurements, n_features)
        if n_measurements <= n_features
            # Whitening gives A*A' = I up to roundoff while preserving dense,
            # Gaussian-derived row spaces.
            L = cholesky(Symmetric(A * transpose(A))).L
            A = L \ A
        else
            # Only reached by the one-feature minimum, where two rows are needed so
            # an infeasible request can still carry a certificate.
            A ./= norm(A)
        end
        return sparse(A)
    end
    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, width * n_features)
    for j in 1:n_features
        rows = _regression_distinct(rng, n_measurements, width)
        vals = randn(rng, width)
        vals ./= norm(vals)
        _basis_pursuit_append_column!(I, J, V, rows, vals, j)
    end
    return sparse(I, J, V, n_measurements, n_features)
end

function _basis_pursuit_correlated_matrix(
    rng::AbstractRNG, n_measurements::Int, n_features::Int, width::Int
)
    n_groups = min(n_features, max(1, round(Int, sqrt(n_features))))
    supports = [sort(_regression_distinct(rng, n_measurements, width)) for _ in 1:n_groups]
    prototypes = [normalize(randn(rng, width)) for _ in 1:n_groups]

    # Every group is populated before shuffling, avoiding blocks of correlated
    # columns in the stored ordering.
    assignments = [mod1(j, n_groups) for j in 1:n_features]
    shuffle!(rng, assignments)
    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, width * n_features)
    for j in 1:n_features
        g = assignments[j]
        prototype = prototypes[g]
        perturbation = randn(rng, width)
        # A fixed relative perturbation norm keeps column coherence independent
        # of the support size.
        perturbation -= dot(perturbation, prototype) .* prototype
        perturbation ./= max(norm(perturbation), eps())
        column = prototype + 0.08 * perturbation
        column .*= (0.75 + 0.5 * rand(rng)) / norm(column)
        _basis_pursuit_append_column!(I, J, V, supports[g], column, j)
    end
    return sparse(I, J, V, n_measurements, n_features)
end

function _basis_pursuit_sparse_matrix(
    rng::AbstractRNG, n_measurements::Int, n_features::Int, width_cap::Int
)
    width = clamp(round(Int, 0.12 * n_measurements), 1, width_cap)
    I = Int[]
    J = Int[]
    V = Float64[]
    for j in 1:n_features
        rows = _regression_distinct(rng, n_measurements, width)
        vals = [(rand(rng, Bool) ? 1.0 : -1.0) * (0.5 + rand(rng)) for _ in 1:width]
        vals .*= (0.75 + 0.5 * rand(rng)) / norm(vals)
        append!(I, rows)
        append!(J, fill(j, width))
        append!(V, vals)
    end
    return sparse(I, J, V, n_measurements, n_features)
end

"""Give every empty measurement row one signed entry in a random column."""
function _basis_pursuit_fill_empty_rows!(rng::AbstractRNG, A::SparseMatrixCSC{Float64, Int})
    used = falses(size(A, 1))
    used[rowvals(A)] .= true
    all(used) && return A
    I, J, V = findnz(A)
    for i in findall(!, used)
        push!(I, i)
        push!(J, rand(rng, 1:size(A, 2)))
        push!(V, (rand(rng, Bool) ? 1.0 : -1.0) * (0.5 + 0.5 * rand(rng)))
    end
    return sparse(I, J, V, size(A)...)
end

function _basis_pursuit_matrix(
    rng::AbstractRNG, n_measurements::Int, n_features::Int, profile::Symbol
)
    width = _basis_pursuit_column_nnz(n_measurements, n_features)
    A = if profile == :gaussian
        _basis_pursuit_gaussian_matrix(rng, n_measurements, n_features, width)
    elseif profile == :correlated_columns
        _basis_pursuit_correlated_matrix(rng, n_measurements, n_features, width)
    elseif profile == :sparse_measurements
        _basis_pursuit_sparse_matrix(rng, n_measurements, n_features, width)
    else
        error("Unknown basis-pursuit matrix profile: $profile")
    end
    A = _basis_pursuit_fill_empty_rows!(rng, A)

    # Store a random column permutation so neither profile structure nor the
    # planted support is encoded by low column indices.
    return A[:, randperm(rng, n_features)]
end

"""
    _inject_basis_pursuit_certificate!(A, b, rng)

Overwrite one measurement row (and its right-hand side) with a linear
combination of 2–4 other rows plus a nonzero RHS gap, returning the new matrix
and the certificate. Columns whose only nonzero sat in the overwritten row are
first re-measured on a source row so no split variable becomes unmeasured.
"""
function _inject_basis_pursuit_certificate!(
    A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}, rng::AbstractRNG
)
    n_measurements, n_features = size(A)
    n_sources = min(n_measurements - 1, rand(rng, 2:4))
    picked = _regression_distinct(rng, n_measurements, n_sources + 1)
    sources = picked[1:n_sources]
    target = picked[end]
    coefficients = [(rand(rng, Bool) ? 1.0 : -1.0) * (0.5 + 1.5 * rand(rng)) for _ in 1:n_sources]

    I, J, V = findnz(A)
    keep = I .!= target
    I, J, V = I[keep], J[keep], V[keep]
    measured = falses(n_features)
    measured[J] .= true
    for j in findall(!, measured)
        push!(I, sources[1])
        push!(J, j)
        push!(V, (rand(rng, Bool) ? 1.0 : -1.0) * (0.5 + rand(rng)))
    end
    A = sparse(I, J, V, n_measurements, n_features)

    combined = zeros(Float64, n_features)
    for (s, c) in zip(sources, coefficients)
        row = A[s, :]
        for (j, v) in zip(findnz(row)...)
            combined[j] += c * v
        end
    end
    cols = findall(!iszero, combined)
    A = A + sparse(fill(target, length(cols)), cols, combined[cols], n_measurements, n_features)

    scale = max(1.0, sqrt(sum(abs2, b) / length(b)))
    gap = (rand(rng, Bool) ? 1.0 : -1.0) * (0.5 + 2.5 * rand(rng)) * scale
    b[target] = sum(coefficients[k] * b[sources[k]] for k in 1:n_sources) + gap

    rows = vcat(sources, target)
    multipliers = vcat(coefficients, -1.0)
    rhs_gap = sum(multipliers[k] * b[rows[k]] for k in eachindex(rows))
    return A, BasisPursuitCertificate(rows, multipliers, rhs_gap)
end

"""
    BasisPursuitProblem(target_variables, feasibility_status, seed)

Construct a reproducible weighted basis-pursuit instance with a local RNG.
The stored `source_signal` first generates `b = A * source_signal`. For feasible
instances it remains an exact witness. Infeasible instances then overwrite one
measurement row by a combination of 2–4 others with an inconsistent right-hand
side (see [`BasisPursuitCertificate`](@ref)).
"""
function BasisPursuitProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)

    n_features = max(1, cld(max(target_variables, 1), 2))
    n_measurements = if n_features <= 3
        n_features <= 2 ? 2 : 3
    else
        clamp(round(Int, (0.30 + 0.20 * rand(rng)) * n_features), 3, n_features - 1)
    end
    profile = BASIS_PURSUIT_PROFILES[rand(rng, eachindex(BASIS_PURSUIT_PROFILES))]
    A = _basis_pursuit_matrix(rng, n_measurements, n_features, profile)

    max_support = max(1, min(n_features, n_measurements ÷ 3))
    support_size = clamp(round(Int, (0.05 + 0.10 * rand(rng)) * n_features), 1, max_support)
    support = sort(_regression_distinct(rng, n_features, support_size))
    source_signal = zeros(Float64, n_features)
    for j in support
        source_signal[j] = (rand(rng, Bool) ? 1.0 : -1.0) * (0.75 + 2.25 * rand(rng))
    end
    b = A * source_signal

    # Numerical cancellation is extraordinarily unlikely, but preserving a
    # nonzero RHS guarantees that every feasible optimum has positive cost.
    if norm(b) <= 1.0e-10
        support = [first(support)]
        fill!(source_signal, 0.0)
        source_signal[first(support)] = 1.0 + rand(rng)
        b = A * source_signal
    end

    weights = 0.5 .+ 1.5 .* rand(rng, n_features)
    resolved_status = if feasibility_status == unknown
        (rand(rng) < 0.8 ? feasible : infeasible)
    else
        feasibility_status
    end

    certificate = nothing
    if resolved_status == infeasible
        A, certificate = _inject_basis_pursuit_certificate!(A, b, rng)
    end

    return BasisPursuitProblem(
        n_features,
        n_measurements,
        profile,
        resolved_status,
        A,
        b,
        weights,
        source_signal,
        support,
        certificate,
    )
end

"""
    build_model(prob::BasisPursuitProblem)

Build the canonical positive/negative-split weighted basis-pursuit LP using only
stored data, row by row from the sparse matrix (linear in its nonzeros). The
objective is bounded below by zero because all weights are strictly positive and
both variable blocks are nonnegative.
"""
function build_model(prob::BasisPursuitProblem)
    model = Model()
    n = prob.n_features

    @variable(model, x_pos[1:n] >= 0)
    @variable(model, x_neg[1:n] >= 0)
    objective = AffExpr(0.0)
    sizehint!(objective.terms, 2n)
    for j in 1:n
        add_to_expression!(objective, prob.weights[j], x_pos[j])
        add_to_expression!(objective, prob.weights[j], x_neg[j])
    end
    @objective(model, Min, objective)

    At = sparse(transpose(prob.A))                       # columns of At are rows of A
    cols = rowvals(At)
    vals = nonzeros(At)
    measurements = Vector{ConstraintRef}(undef, prob.n_measurements)
    for i in 1:prob.n_measurements
        range = nzrange(At, i)
        expr = AffExpr(0.0)
        sizehint!(expr.terms, 2 * length(range))
        for ptr in range
            add_to_expression!(expr, vals[ptr], x_pos[cols[ptr]])
            add_to_expression!(expr, -vals[ptr], x_neg[cols[ptr]])
        end
        measurements[i] = @constraint(model, expr == prob.b[i])
    end
    model[:measurements] = measurements

    return model
end

register_variant(
    :regression,
    :basis_pursuit,
    BasisPursuitProblem,
    "Weighted basis-pursuit sparse recovery with Gaussian, coherent-column, and sparse measurement profiles (column-capped nonzeros)",
)
