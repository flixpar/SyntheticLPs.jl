# regression category
#
# Entry point for the `regression` problem category: linear programs arising from
# robust, sparse, and minimax statistical estimation. Each variant has its own
# data profile and its own feasibility mechanism:
#
#   - `lad`           — robust fixed-effects LAD regression (dense correlated
#                       covariates, one-hot categorical effects, heavy tails,
#                       outliers, replicate measurements; replicate-pair loss
#                       budget certificate)
#   - `quantile`      — L1-penalized quantile regression of a skewed,
#                       heteroscedastic cost on sparse indicator codes, with
#                       prediction bands at reference profiles (cohort-mean band
#                       certificate)
#   - `chebyshev`     — weighted minimax (L∞) tensor B-spline surface fit to
#                       scattered measurements (local spline-cell certificate)
#   - `basis_pursuit` — weighted L1 sparse recovery (multi-row dependency
#                       certificate)
#   - `l1_svm`        — 1-norm linear SVM text classification with a hinge-loss
#                       budget (conflicting-duplicate certificate)
#
# All variants keep the model nonzeros linear in the target: design matrices are
# either column-capped dense blocks, sparse, or locally supported, so a
# 100k-variable request builds a model with a few million nonzeros in seconds.

using Random
using Distributions
using LinearAlgebra
using SparseArrays

"""
    _regression_zipf_cdf(n, exponent)

Cumulative (unnormalized) Zipf weights `cumsum(1 ./ (1:n) .^ exponent)` for
O(log n) inverse-CDF sampling of ranked vocabularies (codes, words).
"""
_regression_zipf_cdf(n::Int, exponent::Float64) = cumsum(1.0 ./ (1:n) .^ exponent)

"""
    _regression_sample_rank(rng, cdf)

Draw one rank from the cumulative weights `cdf` by inverse-CDF search.
"""
function _regression_sample_rank(rng::AbstractRNG, cdf::Vector{Float64})
    return min(length(cdf), searchsortedfirst(cdf, rand(rng) * cdf[end]))
end

"""
    _regression_distinct(rng, n, k)

`k` distinct integers from `1:n` (Floyd's algorithm, `O(k)` expected time), in
random order. Avoids the `O(n)` cost of `randperm(n)[1:k]` when `k ≪ n`.
"""
function _regression_distinct(rng::AbstractRNG, n::Int, k::Int)
    k >= n && return randperm(rng, n)
    chosen = Set{Int}()
    out = Vector{Int}(undef, k)
    q = 0
    for j in (n - k + 1):n
        t = rand(rng, 1:j)
        pick = t in chosen ? j : t
        push!(chosen, pick)
        out[q += 1] = pick
    end
    return shuffle!(rng, out)
end

"""
    _regression_sparse_incidence(rng, n_rows, n_items, lengths, exponent, min_count)

Sample a sparse row-by-item incidence structure: row `i` receives
`lengths[i]` distinct items drawn from a Zipf(`exponent`) popularity ranking
over a randomly permuted vocabulary, and every item is then topped up to appear
in at least `min_count` rows (the usual minimum-document-frequency vocabulary
pruning, applied in reverse). Returns a vector of sorted item lists, one per row.

Runs in `O(total nonzeros · log n_items)`; nothing dense is materialized.
"""
function _regression_sparse_incidence(
    rng::AbstractRNG,
    n_rows::Int,
    n_items::Int,
    lengths::Vector{Int},
    exponent::Float64,
    min_count::Int,
)
    cdf = _regression_zipf_cdf(n_items, exponent)
    vocabulary = randperm(rng, n_items)   # rank -> item id (popularity not tied to index)
    rows = Vector{Vector{Int}}(undef, n_rows)
    counts = zeros(Int, n_items)
    for i in 1:n_rows
        want = min(lengths[i], n_items)
        items = Int[]
        sizehint!(items, want)
        attempts = 0
        while length(items) < want && attempts < 20 * want
            attempts += 1
            item = vocabulary[_regression_sample_rank(rng, cdf)]
            item in items || push!(items, item)
        end
        rows[i] = items
        for item in items
            counts[item] += 1
        end
    end
    floor_count = min(min_count, n_rows)
    for item in 1:n_items
        while counts[item] < floor_count
            i = rand(rng, 1:n_rows)
            if !(item in rows[i])
                push!(rows[i], item)
                counts[item] += 1
            end
        end
    end
    foreach(sort!, rows)
    return rows
end

"""
    _regression_affine_expr(vars, cols, vals)

Build the affine expression `Σ_k vals[k] * vars[cols[k]]` with a pre-sized
term dictionary — the row-construction primitive every regression `build_model`
uses so that model building stays linear in the number of nonzeros.
"""
function _regression_affine_expr(vars, cols::AbstractVector{Int}, vals::AbstractVector{Float64})
    expr = AffExpr(0.0)
    sizehint!(expr.terms, length(cols) + 4)
    for k in eachindex(cols)
        iszero(vals[k]) && continue
        add_to_expression!(expr, vals[k], vars[cols[k]])
    end
    return expr
end

include("lad.jl")
include("quantile.jl")
include("chebyshev.jl")
include("basis_pursuit.jl")
include("l1_svm.jl")
