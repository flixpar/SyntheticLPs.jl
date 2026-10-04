using JuMP
using Random
using Distributions

"""
    LADWitness

Planted feasible point of a [`LADRegressionProblem`](@ref): the data-generating
coefficients. `intercept`, `beta` (continuous covariates), and `gamma`
(reference-coded fixed effects, concatenated factor by factor) give fitted
values whose unweighted absolute residuals sum to `loss`, and
`loss < loss_budget` with a planted margin. The residual variables are
`e_i = |y_i - fitted_i|`.
"""
struct LADWitness
    intercept::Float64
    beta::Vector{Float64}
    gamma::Vector{Float64}
    loss::Float64
end

"""
    LADCertificate

Infeasibility proof for a [`LADRegressionProblem`](@ref). Every
`(a, b) ∈ pairs` is a replicate pair — identical design rows — with
`y[a] - y[b] > 0`. Adding the residual rows `e_a ≥ y_a - fit` and
`e_b ≥ fit - y_b` cancels the (shared) fitted value, so every feasible point
satisfies `e_a + e_b ≥ y[a] - y[b]`. Summing over the disjoint pairs gives
`Σ_i e_i ≥ lower_bound`, which contradicts the loss-budget row
`Σ_i e_i ≤ budget < lower_bound`. The proof uses only LP rows and the bounds
`e ≥ 0`, so it needs `2·|pairs| + 1` rows combined — not a single-row bound
contradiction that presolve can see.
"""
struct LADCertificate
    pairs::Vector{Tuple{Int, Int}}
    lower_bound::Float64
    budget::Float64
end

"""
    LADRegressionProblem <: ProblemGenerator

Robust fixed-effects least-absolute-deviations (L1) regression with a total
absolute-error budget.

# Data profile

Tabular data in the style of an applied econometrics panel:

  - `n_continuous` dense covariates driven by a few latent factors (correlated
    columns), some log-normally skewed, each with its own unit scale;
  - one to three categorical factors (e.g. store, region, product line) with
    Zipf-distributed level frequencies, reference-coded as fixed effects (each
    sample row touches exactly one level column per factor);
  - heavy-tailed Student-t noise plus a contaminated subset of gross vertical
    outliers, half of which are also bad leverage points;
  - replicate measurements: a fraction of samples repeat an earlier design row
    exactly (repeated measurements at the same settings) with fresh noise;
  - positive survey/sampling weights in the objective.

# Formulation

```math
\\min \\sum_i w_i e_i \\quad\\text{s.t.}\\quad
e_i \\ge y_i - \\hat y_i,\\; e_i \\ge \\hat y_i - y_i,\\;
\\sum_i e_i \\le L,\\; e \\ge 0,
```

where `ŷ_i = β₀ + x_i·β + Σ_f γ_{f, level_f(i)}`; `β₀`, `β`, `γ` are free. The
classic two-row LAD form (inequality pairs) distinguishes it structurally from
the equality-split `quantile` variant.

# Feasibility

  - `feasible`: `L` exceeds the planted coefficients' loss by 5–25%
    (`feasible_witness`).
  - `infeasible`: `L` is 70–90% of the replicate-pair lower bound
    `Σ |y_a − y_b|` (`infeasibility_certificate`).
  - `unknown`: `L` is 86–100% of the planted loss (the LAD optimum is typically
    92–97% of it); whether the LAD optimum fits
    under it depends on the data, so neither proof is stored.

# Sizing

Variables = `1 + n_continuous + Σ_f (levels_f − 1) + n_samples`, matching the
target exactly for targets ≥ 12. Rows = `2·n_samples + 1`; nonzeros
≈ `2·n_samples·(n_continuous + n_factors + 2)` (≈ 4–5M at 100k variables).

# Fields

  - `n_samples`, `n_continuous`: sample and dense-covariate counts
  - `levels::Vector{Int}`: level count per categorical factor (level 1 is the
    reference level and has no column)
  - `X::Matrix{Float64}`: dense covariates, stored `n_continuous × n_samples`
    (one column per sample, for row-wise model building)
  - `level_codes::Matrix{Int}`: `n_factors × n_samples` level of each sample
  - `y`, `weights`: response and positive objective weights
  - `loss_budget::Float64`: `L`
  - `replicate_pairs::Vector{Tuple{Int,Int}}`: `(original, replicate)` sample pairs
  - `outliers::Vector{Int}`: contaminated samples
  - `feasible_witness`, `infeasibility_certificate`: see above
"""
struct LADRegressionProblem <: ProblemGenerator
    n_samples::Int
    n_continuous::Int
    levels::Vector{Int}
    X::Matrix{Float64}
    level_codes::Matrix{Int}
    y::Vector{Float64}
    weights::Vector{Float64}
    loss_budget::Float64
    replicate_pairs::Vector{Tuple{Int, Int}}
    outliers::Vector{Int}
    feasible_witness::Union{Nothing, LADWitness}
    infeasibility_certificate::Union{Nothing, LADCertificate}
end

"""Column offset of each factor's first non-reference level within `gamma`."""
_lad_gamma_offsets(levels::Vector{Int}) = cumsum(vcat(0, [l - 1 for l in levels[1:(end - 1)]]))

"""
    _lad_fitted(prob_parts..., intercept, beta, gamma, i)

Fitted value of sample `i` under the given coefficients.
"""
function _lad_fitted(
    X::Matrix{Float64},
    level_codes::Matrix{Int},
    offsets::Vector{Int},
    intercept::Float64,
    beta::Vector{Float64},
    gamma::Vector{Float64},
    i::Int,
)
    value = intercept
    for j in axes(X, 1)
        value += X[j, i] * beta[j]
    end
    for f in axes(level_codes, 1)
        level = level_codes[f, i]
        level > 1 && (value += gamma[offsets[f] + level - 1])
    end
    return value
end

"""
    LADRegressionProblem(target_variables, feasibility_status, seed)

Construct a robust fixed-effects LAD instance with a constructor-local RNG. See
the type docstring for the data profile, sizing, and feasibility contracts.
"""
function LADRegressionProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    V = max(target_variables, 12)

    # --- Dimensions: samples dominate, the rest are coefficients. ---
    n_samples = max(8, round(Int, V * rand(rng, Uniform(0.80, 0.90))))
    n_coef = max(3, V - n_samples)                         # 1 + continuous + fixed effects
    n_continuous = clamp(rand(rng, 6:20), 1, max(1, (n_coef - 1) ÷ 2))
    n_fixed = n_coef - 1 - n_continuous
    n_factors = n_fixed == 0 ? 0 : min(rand(rng, 1:3), n_fixed)
    levels = Int[]
    if n_factors > 0
        shares = rand(rng, Uniform(0.5, 1.5), n_factors)
        extra = n_fixed - n_factors
        alloc = ones(Int, n_factors)                       # each factor ≥ 1 non-reference level
        raw = extra .* shares ./ sum(shares)
        alloc .+= floor.(Int, raw)
        leftover = n_fixed - sum(alloc)
        for k in 1:leftover
            alloc[mod1(k, n_factors)] += 1
        end
        levels = alloc .+ 1
    end

    # Replicates: the last `n_replicates` samples repeat earlier design rows.
    n_replicates = clamp(
        round(Int, n_samples * rand(rng, Uniform(0.10, 0.25)) / 2), 1, n_samples ÷ 3
    )
    n_original = n_samples - n_replicates

    # --- Correlated dense covariates from a latent factor model. ---
    n_latent = min(n_continuous, rand(rng, 2:4))
    loadings = rand(rng, Normal(0.0, 0.8), n_continuous, n_latent)
    latent = randn(rng, n_latent, n_original)
    X = loadings * latent .+ 0.6 .* randn(rng, n_continuous, n_original)
    skewed = rand(rng, n_continuous) .< 0.3
    scales = exp.(rand(rng, Normal(0.0, 1.0), n_continuous))
    for j in 1:n_continuous
        if skewed[j]
            X[j, :] .= exp.(0.5 .* X[j, :])                 # log-normal style covariate
        end
        X[j, :] .*= scales[j]
    end

    # --- Categorical factors with Zipf level frequencies (every level used). ---
    level_codes = ones(Int, n_factors, n_original)
    for f in 1:n_factors
        L = levels[f]
        cdf = _regression_zipf_cdf(L, rand(rng, Uniform(0.5, 1.1)))
        codes = [_regression_sample_rank(rng, cdf) for _ in 1:n_original]
        # Guarantee every level is observed so no fixed-effect column is empty.
        slots = randperm(rng, n_original)
        for l in 1:min(L, n_original)
            codes[slots[l]] = l
        end
        level_codes[f, :] .= codes
    end

    # --- Gross outliers: vertical, half also bad leverage points. ---
    n_outliers = round(Int, n_original * rand(rng, Uniform(0.02, 0.10)))
    outliers = sort(randperm(rng, n_original)[1:n_outliers])
    for (k, i) in enumerate(outliers)
        if isodd(k)
            j = rand(rng, 1:n_continuous)
            X[j, i] *= rand(rng, Uniform(3.0, 8.0))
        end
    end

    # --- Replicate rows copy original designs exactly. ---
    originals = randperm(rng, n_original)[1:n_replicates]
    X = hcat(X, X[:, originals])
    level_codes = hcat(level_codes, level_codes[:, originals])
    replicate_pairs = [(originals[k], n_original + k) for k in 1:n_replicates]

    # --- Truth and response. ---
    offsets = _lad_gamma_offsets(levels)
    beta = rand(rng, Normal(0.0, 1.0), n_continuous) ./ scales
    gamma = Float64[]
    for f in 1:n_factors
        append!(gamma, rand(rng, Normal(0.0, rand(rng, Uniform(0.5, 2.0))), levels[f] - 1))
    end
    intercept = rand(rng, Uniform(-5.0, 5.0))
    sigma = rand(rng, Uniform(0.5, 2.0))
    nu = rand(rng, Uniform(1.5, 5.0))
    noise = clamp.(sigma .* rand(rng, TDist(nu), n_samples), -200 * sigma, 200 * sigma)
    for i in outliers
        noise[i] += (rand(rng, Bool) ? 1.0 : -1.0) * rand(rng, Uniform(15.0, 60.0)) * sigma
    end
    fitted = [_lad_fitted(X, level_codes, offsets, intercept, beta, gamma, i) for i in 1:n_samples]
    y = fitted .+ noise
    weights = exp.(rand(rng, Normal(0.0, 0.4), n_samples))

    witness_loss = sum(abs, noise)
    pair_bound = sum(abs(y[a] - y[b]) for (a, b) in replicate_pairs)

    # --- Feasibility through the absolute-error budget. ---
    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        loss_budget = witness_loss * rand(rng, Uniform(1.05, 1.25))
        witness = LADWitness(intercept, beta, gamma, witness_loss)
    elseif feasibility_status == infeasible
        if pair_bound <= 1e-6
            # Degenerate replicate noise (practically impossible): separate one pair.
            a, b = replicate_pairs[1]
            y[a] += sigma
            pair_bound = sum(abs(y[a] - y[b]) for (a, b) in replicate_pairs)
        end
        loss_budget = pair_bound * rand(rng, Uniform(0.70, 0.90))
        oriented = [(y[a] >= y[b] ? (a, b) : (b, a)) for (a, b) in replicate_pairs]
        certificate = LADCertificate(oriented, pair_bound, loss_budget)
    else
        loss_budget = witness_loss * rand(rng, Uniform(0.86, 1.0))
    end

    return LADRegressionProblem(
        n_samples,
        n_continuous,
        levels,
        X,
        level_codes,
        y,
        weights,
        loss_budget,
        replicate_pairs,
        outliers,
        witness,
        certificate,
    )
end

"""
    build_model(prob::LADRegressionProblem)

Build the two-row LAD LP. Deterministic and linear in the number of nonzeros.
"""
function build_model(prob::LADRegressionProblem)
    model = Model()
    m = prob.n_samples
    d = prob.n_continuous
    offsets = _lad_gamma_offsets(prob.levels)
    n_gamma = sum(l - 1 for l in prob.levels; init=0)

    @variable(model, intercept)
    @variable(model, beta[1:d])
    @variable(model, gamma[1:n_gamma])
    @variable(model, e[1:m] >= 0)

    objective = AffExpr(0.0)
    sizehint!(objective.terms, m)
    for i in 1:m
        add_to_expression!(objective, prob.weights[i], e[i])
    end
    @objective(model, Min, objective)

    for i in 1:m
        fit = AffExpr(0.0)
        sizehint!(fit.terms, d + length(prob.levels) + 2)
        add_to_expression!(fit, 1.0, intercept)
        for j in 1:d
            add_to_expression!(fit, prob.X[j, i], beta[j])
        end
        for f in eachindex(prob.levels)
            level = prob.level_codes[f, i]
            level > 1 && add_to_expression!(fit, 1.0, gamma[offsets[f] + level - 1])
        end
        @constraint(model, e[i] + fit >= prob.y[i])
        @constraint(model, e[i] - fit >= -prob.y[i])
    end

    budget = AffExpr(0.0)
    sizehint!(budget.terms, m)
    for i in 1:m
        add_to_expression!(budget, 1.0, e[i])
    end
    @constraint(model, loss_budget, budget <= prob.loss_budget)

    return model
end

register_variant(
    :regression,
    :lad,
    LADRegressionProblem,
    "Robust fixed-effects LAD regression with correlated covariates, categorical effects, heavy-tailed outliers, replicates, and an absolute-error budget";
    default=true,
    tags=[:statistics, :dense],
)
