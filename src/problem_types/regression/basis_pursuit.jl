using JuMP
using LinearAlgebra
using Random

const BASIS_PURSUIT_PROFILES = (
    :gaussian_well_conditioned,
    :correlated_columns,
    :sparse_measurements,
)

"""
    BasisPursuitProblem <: ProblemGenerator

Weighted basis pursuit:

```math
\\min_x \\sum_j w_j |x_j| \\quad \\text{s.t.} \\quad A x = b.
```

The model uses nonnegative positive/negative splits `x = x_pos - x_neg`.
Every instance stores its matrix profile, planted sparse signal, resolved
feasibility status, and (when infeasible) a two-row contradiction certificate.

The three matrix profiles have materially different structure:
- `gaussian_well_conditioned`: a dense Gaussian matrix whitened to have
  orthonormal measurement rows (except at the unavoidable one-feature minimum);
- `correlated_columns`: dense groups of highly coherent columns generated from
  shared latent directions plus small independent perturbations;
- `sparse_measurements`: sparse signed measurements with randomized supports.

The split formulation always has an even number of variables. An even target of
at least two is met exactly; an odd target is rounded up by one; targets below
two produce the minimum two-variable formulation.
"""
struct BasisPursuitProblem <: ProblemGenerator
    n_features::Int
    n_measurements::Int
    profile::Symbol
    resolved_status::FeasibilityStatus
    A::Matrix{Float64}
    b::Vector{Float64}
    weights::Vector{Float64}
    planted_signal::Vector{Float64}
    support::Vector{Int}
    certificate_rows::Tuple{Int,Int}
    certificate_multiplier::Float64
    certificate_rhs_gap::Float64
end

function _basis_pursuit_gaussian_matrix(
    rng::AbstractRNG,
    n_measurements::Int,
    n_features::Int,
)
    A = randn(rng, n_measurements, n_features)
    if n_measurements <= n_features
        # Whitening gives A*A' = I up to roundoff while preserving dense,
        # Gaussian-derived row spaces.
        L = cholesky(Symmetric(A * transpose(A))).L
        A = L \ A
    else
        # Only reached by the one-feature minimum, where two rows are needed so
        # an infeasible request can still carry a two-row certificate.
        A ./= norm(A)
    end
    return A
end

function _basis_pursuit_correlated_matrix(
    rng::AbstractRNG,
    n_measurements::Int,
    n_features::Int,
)
    n_groups = min(n_features, max(1, round(Int, sqrt(n_features))))
    prototypes = randn(rng, n_measurements, n_groups)
    for g in 1:n_groups
        prototypes[:, g] ./= norm(prototypes[:, g])
    end

    # Every group is populated before shuffling, avoiding blocks of correlated
    # columns in the stored ordering.
    assignments = [mod1(j, n_groups) for j in 1:n_features]
    shuffle!(rng, assignments)
    A = Matrix{Float64}(undef, n_measurements, n_features)
    for j in 1:n_features
        column = prototypes[:, assignments[j]] + 0.03 * randn(rng, n_measurements)
        column ./= norm(column)
        A[:, j] = (0.75 + 0.5 * rand(rng)) * column
    end
    return A
end

function _basis_pursuit_sparse_matrix(
    rng::AbstractRNG,
    n_measurements::Int,
    n_features::Int,
)
    A = zeros(Float64, n_measurements, n_features)
    width = clamp(round(Int, 0.12 * n_measurements), 1, n_measurements)
    for j in 1:n_features
        rows = randperm(rng, n_measurements)[1:width]
        for i in rows
            A[i, j] = (rand(rng, Bool) ? 1.0 : -1.0) * (0.5 + rand(rng))
        end
    end

    # Preserve sparsity while ensuring every measurement and feature is active.
    for i in 1:n_measurements
        if all(iszero, @view A[i, :])
            j = rand(rng, 1:n_features)
            A[i, j] = rand(rng, Bool) ? 1.0 : -1.0
        end
    end
    for j in 1:n_features
        A[:, j] .*= (0.75 + 0.5 * rand(rng)) / norm(@view A[:, j])
    end
    return A
end

function _basis_pursuit_matrix(
    rng::AbstractRNG,
    n_measurements::Int,
    n_features::Int,
    profile::Symbol,
)
    A = if profile == :gaussian_well_conditioned
        _basis_pursuit_gaussian_matrix(rng, n_measurements, n_features)
    elseif profile == :correlated_columns
        _basis_pursuit_correlated_matrix(rng, n_measurements, n_features)
    elseif profile == :sparse_measurements
        _basis_pursuit_sparse_matrix(rng, n_measurements, n_features)
    else
        error("Unknown basis-pursuit matrix profile: $profile")
    end

    # Matrix generation may have internal grouping or traversal order. Store a
    # random column permutation so neither profile structure nor planted support
    # is encoded by low column indices.
    return A[:, randperm(rng, n_features)]
end

"""
    BasisPursuitProblem(target_variables, feasibility_status, seed)

Construct a reproducible weighted basis-pursuit instance with a local RNG.
Feasible instances use `b = A * planted_signal`. Infeasible instances replace
one measurement row by a proportional copy of another while shifting its right
hand side, yielding the explicit contradiction
`A[r₂, :] = λA[r₁, :]` but `b[r₂] != λb[r₁]`.

An `unknown` request naturally resolves to a planted feasible instance with
probability 0.8 and a certified infeasible instance otherwise; the result is
stored in `resolved_status`.
"""
function BasisPursuitProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = MersenneTwister(seed)

    n_features = max(1, cld(max(target_variables, 1), 2))
    n_measurements = if n_features <= 2
        2
    else
        clamp(round(Int, (0.30 + 0.20 * rand(rng)) * n_features), 2, n_features - 1)
    end
    profile = BASIS_PURSUIT_PROFILES[rand(rng, eachindex(BASIS_PURSUIT_PROFILES))]
    A = _basis_pursuit_matrix(rng, n_measurements, n_features, profile)

    max_support = max(1, min(n_features, n_measurements ÷ 3))
    support_size = clamp(
        round(Int, (0.05 + 0.10 * rand(rng)) * n_features),
        1,
        max_support,
    )
    support = sort(randperm(rng, n_features)[1:support_size])
    planted_signal = zeros(Float64, n_features)
    for j in support
        planted_signal[j] =
            (rand(rng, Bool) ? 1.0 : -1.0) * (0.75 + 2.25 * rand(rng))
    end
    b = A * planted_signal

    # Numerical cancellation is extraordinarily unlikely, but preserving a
    # nonzero RHS guarantees that every feasible optimum has positive cost.
    if norm(b) <= 1.0e-10
        support = [first(support)]
        fill!(planted_signal, 0.0)
        planted_signal[first(support)] = 1.0 + rand(rng)
        b = A * planted_signal
    end

    weights = 0.5 .+ 1.5 .* rand(rng, n_features)
    resolved_status = feasibility_status == unknown ?
        (rand(rng) < 0.8 ? feasible : infeasible) : feasibility_status

    certificate_rows = (0, 0)
    certificate_multiplier = 0.0
    certificate_rhs_gap = 0.0
    if resolved_status == infeasible
        row_order = randperm(rng, n_measurements)
        r1, r2 = row_order[1], row_order[2]
        multipliers = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
        certificate_multiplier = multipliers[rand(rng, eachindex(multipliers))]
        certificate_rhs_gap = (rand(rng, Bool) ? 1.0 : -1.0) * (0.5 + 2.5 * rand(rng))
        A[r2, :] = certificate_multiplier .* A[r1, :]
        b[r2] = certificate_multiplier * b[r1] + certificate_rhs_gap
        certificate_rows = (r1, r2)
    end

    return BasisPursuitProblem(
        n_features,
        n_measurements,
        profile,
        resolved_status,
        A,
        b,
        weights,
        planted_signal,
        support,
        certificate_rows,
        certificate_multiplier,
        certificate_rhs_gap,
    )
end

"""
    build_model(prob::BasisPursuitProblem)

Build the canonical positive/negative-split weighted basis-pursuit LP using only
stored data. The objective is bounded below by zero because all weights are
strictly positive and both variable blocks are nonnegative.
"""
function build_model(prob::BasisPursuitProblem)
    model = Model()

    @variable(model, x_pos[1:prob.n_features] >= 0)
    @variable(model, x_neg[1:prob.n_features] >= 0)
    @objective(
        model,
        Min,
        sum(prob.weights[j] * (x_pos[j] + x_neg[j]) for j in 1:prob.n_features),
    )
    @constraint(
        model,
        measurements[i in 1:prob.n_measurements],
        sum(prob.A[i, j] * (x_pos[j] - x_neg[j]) for j in 1:prob.n_features) ==
            prob.b[i],
    )

    return model
end

register_variant(
    :regression,
    :basis_pursuit,
    BasisPursuitProblem,
    "Weighted basis-pursuit sparse recovery with Gaussian, coherent-column, and sparse measurement profiles",
)
