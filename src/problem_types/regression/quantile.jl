using JuMP
using Random
using Distributions

"""
    QuantileWitness

Planted feasible point of a [`QuantileRegressionProblem`](@ref): the
data-generating coefficients (`intercept`, dense `demographic` coefficients,
sparse `code` coefficients). Every reference band contains the witness
prediction with slack at least half the band's half-width; the residual split
`u - v = y - prediction` follows directly.
"""
struct QuantileWitness
    intercept::Float64
    demographic::Vector{Float64}
    code::Vector{Float64}
end

"""
    QuantileCertificate

Infeasibility proof for a [`QuantileRegressionProblem`](@ref). Reference
`cohort` is the exact average of the `members` profiles, so for any
coefficients its prediction equals the members' mean prediction. The band rows
give `pred(member) ≤ band_upper[member]` and `pred(cohort) ≥ band_lower[cohort]`;
weighting the member rows by `1/|members|` and subtracting the cohort row leaves
`0 ≥ gap` with `gap = band_lower[cohort] − mean(band_upper[members]) > 0`.
"""
struct QuantileCertificate
    cohort::Int
    members::Vector{Int}
    gap::Float64
end

"""
    QuantileRegressionProblem <: ProblemGenerator

L1-penalized quantile regression of a skewed, heteroscedastic cost on sparse
indicator codes, with plausibility bands on predictions at reference profiles.

# Data profile

Claims-style data: each sample (patient) carries `1 + Poisson(λ)` distinct
indicator codes drawn from a Zipf-popularity vocabulary (every code appears in at
least `min_count` samples, the usual minimum-frequency pruning), plus a few
dense standardized demographic covariates. A sparse subset of codes drives the
cost; noise is right-skewed and its scale grows with the expected cost
(heteroscedastic) — the setting where quantile regression is used.

# Formulation

```math
\\min \\sum_i \\big(τ u_i + (1-τ) v_i\\big) + λ \\sum_j (β^+_j + β^-_j)
\\quad\\text{s.t.}\\quad
u_i - v_i + b_0 + z_i·δ + x_i·(β^+ - β^-) = y_i,\\quad
\\ell_r \\le b_0 + z_r·δ + x_r·(β^+ - β^-) \\le h_r,
```

with `u, v, β^± ≥ 0` and `b_0`, `δ` free. The reference profiles `r` are sampled
patients plus *cohort means* (exact averages of 3–6 member profiles), each with a
ranged plausibility band on its predicted quantile.

# Feasibility

Feasibility is decided entirely by the band system (the residual split absorbs
any fit):

  - `feasible`: every band contains the planted prediction with slack ≥ half
    its half-width (`feasible_witness`).
  - `infeasible`: as feasible, then one cohort's lower band is raised above its
    members' mean upper band (`infeasibility_certificate`; a multi-row Farkas
    combination, not a single-row bound conflict).
  - `unknown`: band centres are the planted predictions plus independent
    estimation noise, with the noise scale calibrated so that roughly half of
    the instances have mutually consistent cohort bands. Nothing is planted.

# Sizing

Variables = `1 + n_demographic + 2·n_codes + 2·n_samples`, exact for targets
≥ 30. Rows = `n_samples + n_references`; nonzeros ≈ `n_samples·(2·codes per
sample + n_demographic + 3)` (≈ 1–2M at 100k variables).
"""
struct QuantileRegressionProblem <: ProblemGenerator
    n_samples::Int
    n_codes::Int
    n_demographic::Int
    tau::Float64
    penalty::Float64
    codes::Vector{Vector{Int}}
    Z::Matrix{Float64}
    y::Vector{Float64}
    reference_codes::Vector{Vector{Int}}
    reference_values::Vector{Vector{Float64}}
    reference_Z::Matrix{Float64}
    band_lower::Vector{Float64}
    band_upper::Vector{Float64}
    cohorts::Vector{Vector{Int}}
    cohort_rows::Vector{Int}
    feasible_witness::Union{Nothing, QuantileWitness}
    infeasibility_certificate::Union{Nothing, QuantileCertificate}
end

"""Prediction of a profile (sparse codes with values, dense covariates) under given coefficients."""
function _quantile_predict(intercept, demographic, code_coef, cols, vals, z)
    value = intercept + dot(z, demographic)
    for k in eachindex(cols)
        value += vals[k] * code_coef[cols[k]]
    end
    return value
end

"""
    _quantile_unknown_noise_scale(half_widths, cohorts, cohort_rows; target=0.5)

Bisection for the band-noise multiplier `s` (noise sd `s·hw_r`) under which the
probability that every cohort band overlaps its members' mean band is `target`.
"""
function _quantile_unknown_noise_scale(
    half_widths::Vector{Float64},
    cohorts::Vector{Vector{Int}},
    cohort_rows::Vector{Int};
    target::Float64=0.5,
)
    reach = Float64[]
    spread = Float64[]
    for (c, members) in enumerate(cohorts)
        k = length(members)
        row = cohort_rows[c]
        push!(reach, half_widths[row] + sum(half_widths[members]) / k)
        push!(spread, sqrt(half_widths[row]^2 + sum(abs2, half_widths[members]) / k^2))
    end
    logp(s) = sum(log(max(2 * cdf(Normal(), reach[c] / (s * spread[c])) - 1, 1e-300)) for c in eachindex(reach))
    lo, hi = 1e-3, 1e3
    for _ in 1:80
        mid = sqrt(lo * hi)
        logp(mid) > log(target) ? (lo = mid) : (hi = mid)
    end
    return sqrt(lo * hi)
end

"""
    QuantileRegressionProblem(target_variables, feasibility_status, seed)

Construct an L1-penalized quantile regression instance with a constructor-local
RNG. See the type docstring for the data profile, sizing, and contracts.
"""
function QuantileRegressionProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    V = max(target_variables, 30)

    # --- Dimensions: V = 1 + d + 2p + 2m exactly. ---
    n_demographic = rand(rng, 3:6)
    isodd(V - 1 - n_demographic) && (n_demographic -= 1)
    half = (V - 1 - n_demographic) ÷ 2                     # p + m
    ratio = rand(rng, Uniform(0.8, 2.0))                   # samples per code
    n_codes = clamp(round(Int, half / (1 + ratio)), 2, half - 8)
    n_samples = half - n_codes

    tau = rand(rng, (0.1, 0.25, 0.5, 0.75, 0.9))
    mean_codes = rand(rng, Uniform(4.0, 12.0))
    lengths = [min(n_codes, 1 + rand(rng, Poisson(mean_codes))) for _ in 1:n_samples]
    min_count = min(5, n_samples)
    codes = _regression_sparse_incidence(
        rng, n_samples, n_codes, lengths, rand(rng, Uniform(0.7, 1.0)), min_count
    )
    Z = randn(rng, n_samples, n_demographic)

    # --- Sparse cost drivers and a heteroscedastic, right-skewed response. ---
    code_coef = zeros(Float64, n_codes)
    n_active = max(1, round(Int, n_codes * rand(rng, Uniform(0.05, 0.15))))
    for j in randperm(rng, n_codes)[1:n_active]
        magnitude = exp(rand(rng, Normal(log(1.5), 0.8)))
        code_coef[j] = rand(rng) < 0.85 ? magnitude : -0.5 * magnitude
    end
    demographic = rand(rng, Normal(0.0, 0.5), n_demographic)
    intercept = rand(rng, Uniform(1.0, 4.0))
    sigma0 = rand(rng, Uniform(0.3, 1.0))
    ones_cache = Dict{Int, Vector{Float64}}()
    unit_vals(k) = get!(() -> ones(Float64, k), ones_cache, k)
    y = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        mean_cost = _quantile_predict(
            intercept, demographic, code_coef, codes[i], unit_vals(length(codes[i])), view(Z, i, :)
        )
        scale = sigma0 * (0.5 + 0.25 * abs(mean_cost))
        y[i] = mean_cost + scale * (rand(rng, Gamma(2.0, 1.0)) - 2.0)
    end
    max_tau = max(tau, 1 - tau)
    penalty = rand(rng, Uniform(0.1, 0.6)) * max_tau * min_count

    # --- Reference profiles: sampled patients, cohort members, cohort means. ---
    n_cohorts = clamp(round(Int, sqrt(n_samples) / 5), 1, 25)
    cohort_size = [rand(rng, 3:6) for _ in 1:n_cohorts]
    while sum(cohort_size) > n_samples - 1 && n_cohorts > 1
        n_cohorts -= 1
        pop!(cohort_size)
    end
    cohort_size[1] = min(cohort_size[1], n_samples - 1)
    n_single = clamp(round(Int, 0.01 * n_samples), 1, 200)
    n_single = min(n_single, n_samples - sum(cohort_size))
    picks = randperm(rng, n_samples)[1:(sum(cohort_size) + n_single)]

    reference_codes = Vector{Vector{Int}}()
    reference_values = Vector{Vector{Float64}}()
    reference_Z_rows = Vector{Vector{Float64}}()
    cohorts = Vector{Vector{Int}}()
    cohort_rows = Int[]
    cursor = 0
    for c in 1:n_cohorts
        members = Int[]
        for _ in 1:cohort_size[c]
            cursor += 1
            i = picks[cursor]
            push!(reference_codes, codes[i])
            push!(reference_values, unit_vals(length(codes[i])))
            push!(reference_Z_rows, Z[i, :])
            push!(members, length(reference_codes))
        end
        # Cohort mean profile: average of the member rows (fractional prevalence).
        weight = 1.0 / length(members)
        acc = Dict{Int, Float64}()
        for r in members, j in reference_codes[r]
            acc[j] = get(acc, j, 0.0) + weight
        end
        cols = sort!(collect(keys(acc)))
        push!(reference_codes, cols)
        push!(reference_values, [acc[j] for j in cols])
        push!(reference_Z_rows, sum(reference_Z_rows[r] for r in members) .* weight)
        push!(cohorts, members)
        push!(cohort_rows, length(reference_codes))
    end
    for _ in 1:n_single
        cursor += 1
        i = picks[cursor]
        push!(reference_codes, codes[i])
        push!(reference_values, unit_vals(length(codes[i])))
        push!(reference_Z_rows, Z[i, :])
    end
    n_ref = length(reference_codes)
    reference_Z = Matrix{Float64}(undef, n_ref, n_demographic)
    for r in 1:n_ref
        reference_Z[r, :] .= reference_Z_rows[r]
    end
    predictions = [
        _quantile_predict(
            intercept,
            demographic,
            code_coef,
            reference_codes[r],
            reference_values[r],
            view(reference_Z, r, :),
        ) for r in 1:n_ref
    ]
    # Band half-widths scale with the local noise level.
    half_widths = [sigma0 * (0.5 + 0.25 * abs(p)) * rand(rng, Uniform(0.6, 1.5)) for p in predictions]

    witness = nothing
    certificate = nothing
    if feasibility_status == unknown
        s = _quantile_unknown_noise_scale(half_widths, cohorts, cohort_rows)
        centres = predictions .+ s .* half_widths .* randn(rng, n_ref)
    else
        centres = predictions .+ half_widths .* rand(rng, Uniform(-0.5, 0.5), n_ref)
    end
    band_lower = centres .- half_widths
    band_upper = centres .+ half_widths

    if feasibility_status == feasible
        witness = QuantileWitness(intercept, demographic, code_coef)
    elseif feasibility_status == infeasible
        c = rand(rng, 1:n_cohorts)
        members = cohorts[c]
        row = cohort_rows[c]
        mean_upper = sum(band_upper[members]) / length(members)
        gap = rand(rng, Uniform(0.3, 0.8)) * half_widths[row]
        band_lower[row] = mean_upper + gap
        band_upper[row] = band_lower[row] + 2 * half_widths[row]
        certificate = QuantileCertificate(row, copy(members), gap)
    end

    return QuantileRegressionProblem(
        n_samples,
        n_codes,
        n_demographic,
        tau,
        penalty,
        codes,
        Z,
        y,
        reference_codes,
        reference_values,
        reference_Z,
        band_lower,
        band_upper,
        cohorts,
        cohort_rows,
        witness,
        certificate,
    )
end

"""
    build_model(prob::QuantileRegressionProblem)

Build the penalized quantile-regression LP. Deterministic and linear in the
number of nonzeros.
"""
function build_model(prob::QuantileRegressionProblem)
    model = Model()
    m = prob.n_samples
    p = prob.n_codes
    d = prob.n_demographic
    τ = prob.tau

    @variable(model, intercept)
    @variable(model, demographic[1:d])
    @variable(model, code_pos[1:p] >= 0)
    @variable(model, code_neg[1:p] >= 0)
    @variable(model, u[1:m] >= 0)
    @variable(model, v[1:m] >= 0)

    objective = AffExpr(0.0)
    sizehint!(objective.terms, 2m + 2p)
    for i in 1:m
        add_to_expression!(objective, τ, u[i])
        add_to_expression!(objective, 1 - τ, v[i])
    end
    for j in 1:p
        add_to_expression!(objective, prob.penalty, code_pos[j])
        add_to_expression!(objective, prob.penalty, code_neg[j])
    end
    @objective(model, Min, objective)

    function prediction(cols, vals, z)
        expr = AffExpr(0.0)
        sizehint!(expr.terms, 2 * length(cols) + d + 3)
        add_to_expression!(expr, 1.0, intercept)
        for k in 1:d
            add_to_expression!(expr, z[k], demographic[k])
        end
        for k in eachindex(cols)
            add_to_expression!(expr, vals[k], code_pos[cols[k]])
            add_to_expression!(expr, -vals[k], code_neg[cols[k]])
        end
        return expr
    end

    for i in 1:m
        cols = prob.codes[i]
        fit = prediction(cols, ones(length(cols)), view(prob.Z, i, :))
        add_to_expression!(fit, 1.0, u[i])
        add_to_expression!(fit, -1.0, v[i])
        @constraint(model, fit == prob.y[i])
    end
    for r in eachindex(prob.reference_codes)
        band = prediction(prob.reference_codes[r], prob.reference_values[r], view(prob.reference_Z, r, :))
        @constraint(model, prob.band_lower[r] <= band <= prob.band_upper[r])
    end

    return model
end

register_variant(
    :regression,
    :quantile,
    QuantileRegressionProblem,
    "L1-penalized quantile regression of a heteroscedastic cost on sparse indicator codes with cohort prediction bands";
    tags=[:statistics],
)
