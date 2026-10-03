# portfolio category
#
# Entry point for the `portfolio` problem category: scenario-based portfolio
# construction LPs (CVaR-constrained allocation and benchmark tracking).
#
# Both variants share a factor-structured scenario market generated here.
# Asset returns in scenario `s` are
#
#     r[s, i] = Σ_k F[s, k] · B[i, k]  +  E[i, s],
#
# where `F` holds the returns of a handful of common factors (market, styles,
# and one industry factor per sector), `B` the asset exposures (dense for market
# and styles, a 0/1 sector indicator for industries), and `E` sparse
# idiosyncratic shocks: each scenario draws shocks for a random subset of
# `J ≪ n` assets, scaled by `√(n/J)` so every asset's idiosyncratic variance is
# preserved in expectation. The LPs then use *factor exposure variables*
# `f = Bᵀx`, so each scenario row has `K + J + O(1)` nonzeros instead of `n` — a
# 100k-variable instance has a few million nonzeros rather than the
# `n_scenarios × n_assets` dense block (6+ GB) of a naive formulation.

using Random
using Distributions
using LinearAlgebra
using SparseArrays

"""
    PortfolioMarket

Factor-structured scenario market shared by the portfolio variants.

  - `n_assets`, `n_scenarios`, `n_styles`, `n_sectors`
  - `style_loadings::Matrix{Float64}`: `n_assets × (1 + n_styles)` exposures to
    the market (column 1) and style factors
  - `sector::Vector{Int}`: industry sector of each asset (industry exposure is 1)
  - `factor_returns::Matrix{Float64}`: `n_scenarios × n_factors` with columns
    ordered market, styles, sectors (`n_factors = 1 + n_styles + n_sectors`)
  - `idiosyncratic::SparseMatrixCSC{Float64,Int}`: `n_assets × n_scenarios`
    sparse shocks (one column per scenario)
  - `expected_returns::Vector{Float64}`: forecast (factor premia + alpha)
  - `benchmark::Vector{Float64}`: cap-weighted benchmark weights (sum to one)
"""
struct PortfolioMarket
    n_assets::Int
    n_scenarios::Int
    n_styles::Int
    n_sectors::Int
    style_loadings::Matrix{Float64}
    sector::Vector{Int}
    factor_returns::Matrix{Float64}
    idiosyncratic::SparseMatrixCSC{Float64, Int}
    expected_returns::Vector{Float64}
    benchmark::Vector{Float64}
end

"""Number of factors `1 + n_styles + n_sectors`."""
_portfolio_n_factors(market::PortfolioMarket) = 1 + market.n_styles + market.n_sectors

"""
    _portfolio_distinct(rng, n, k)

`k` distinct integers from `1:n` by Floyd's algorithm (`O(k)` expected time).
"""
function _portfolio_distinct(rng::AbstractRNG, n::Int, k::Int)
    k >= n && return randperm(rng, n)
    chosen = Set{Int}()
    out = Vector{Int}(undef, k)
    for (q, j) in enumerate((n - k + 1):n)
        t = rand(rng, 1:j)
        pick = t in chosen ? j : t
        push!(chosen, pick)
        out[q] = pick
    end
    return out
end

"""
    _portfolio_balanced_groups(rng, n, g)

Assign `n` items to `g` groups so every group is non-empty, with random
(Dirichlet-like) group sizes.
"""
function _portfolio_balanced_groups(rng::AbstractRNG, n::Int, g::Int)
    g = clamp(g, 1, n)
    assignment = Vector{Int}(undef, n)
    perm = randperm(rng, n)
    shares = rand(rng, Uniform(0.5, 1.5), g)
    cdf = cumsum(shares ./ sum(shares))
    for (q, i) in enumerate(perm)
        assignment[i] = q <= g ? q : min(g, searchsortedfirst(cdf, rand(rng)))
    end
    return assignment
end

"""
    _portfolio_exposures(market, x)

Factor exposures `f = Bᵀx` of a weight vector over all assets (market, styles,
then sector weights).
"""
function _portfolio_exposures(market::PortfolioMarket, x::AbstractVector{Float64})
    f = zeros(Float64, _portfolio_n_factors(market))
    f[1:(1 + market.n_styles)] .= transpose(market.style_loadings) * x
    for i in 1:market.n_assets
        f[1 + market.n_styles + market.sector[i]] += x[i]
    end
    return f
end

"""
    _portfolio_scenario_returns(market, x)

Portfolio return in every scenario, `F·(Bᵀx) + Eᵀx`, for weights `x` over all
assets.
"""
function _portfolio_scenario_returns(market::PortfolioMarket, x::AbstractVector{Float64})
    return market.factor_returns * _portfolio_exposures(market, x) .+ transpose(market.idiosyncratic) * x
end

"""
    _portfolio_cvar(losses, beta)

Rockafellar–Uryasev CVaR of the equally likely `losses` at level `beta`:
`min_α α + Σ_s (L_s − α)⁺ / ((1 − β) S)`, returned with its minimizing `α`.
The objective is piecewise linear in `α` with breakpoints at the losses, so the
minimum is found exactly by scanning the sorted losses around the VaR index.
"""
function _portfolio_cvar(losses::Vector{Float64}, beta::Float64)
    S = length(losses)
    c = 1.0 / ((1.0 - beta) * S)
    sorted = sort(losses; rev=true)
    k = clamp(ceil(Int, (1.0 - beta) * S), 1, S)
    best = Inf
    best_alpha = sorted[k]
    for j in max(1, k - 2):min(S, k + 2)
        alpha = sorted[j]
        value = alpha + c * sum(max(l - alpha, 0.0) for l in losses)
        if value < best
            best = value
            best_alpha = alpha
        end
    end
    return best, best_alpha
end

"""
    _portfolio_waterfill(target, caps)

Closest-in-shape capped allocation: `x_i = min(c · target_i, caps_i)` with the
scalar `c ≥ 0` chosen by bisection so that `Σ x = 1`. Requires
`Σ caps > 1` and positive `target` on enough assets; every entry respects its cap
exactly (unlike clip-then-renormalize, which can push weights back above their
caps).
"""
function _portfolio_waterfill(target::Vector{Float64}, caps::Vector{Float64})
    sum(caps) > 1.0 || throw(ArgumentError("caps must sum above one"))
    total(c) = sum(min(c * target[i], caps[i]) for i in eachindex(target))
    lo, hi = 0.0, 1.0
    while total(hi) < 1.0
        hi *= 2
        hi > 1e12 && throw(ArgumentError("target support too small for the caps"))
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        total(mid) < 1.0 ? (lo = mid) : (hi = mid)
    end
    x = [min(hi * target[i], caps[i]) for i in eachindex(target)]
    # Remove the bisection residual from uncapped entries (keeps caps exact).
    slack = 1.0 - sum(x)
    free = findall(i -> x[i] < caps[i] - 1e-12 && x[i] > 0, eachindex(x))
    if !isempty(free) && abs(slack) > 0
        room = sum(x[free])
        for i in free
            x[i] += slack * x[i] / room
        end
    end
    return x
end

"""
    _portfolio_extreme_on_capped_simplex(values, caps, sense)

Exact `min` (`sense = :min`) or `max` of `values·x` over
`{Σx = 1, 0 ≤ x ≤ caps}` (a fractional knapsack: fill the best entries to their
caps). Used to bound portfolio quantities for infeasibility certificates.
"""
function _portfolio_extreme_on_capped_simplex(
    values::Vector{Float64}, caps::Vector{Float64}, sense::Symbol
)
    order = sortperm(values; rev=(sense == :max))
    remaining = 1.0
    total = 0.0
    for i in order
        take = min(caps[i], remaining)
        total += take * values[i]
        remaining -= take
        remaining <= 0 && break
    end
    remaining > 1e-12 && throw(ArgumentError("caps sum below one"))
    return total
end

"""
    _portfolio_floor_bound(costs, caps, groups, floors)

Valid lower bound on `min costs·x` over `{Σx = 1, 0 ≤ x ≤ caps, x(group g) ≥ floors[g]}`
via the Lagrangian dual

    λ + Σ_g μ_g floors[g] + Σ_i min(0, costs[i] − λ − μ_{groups[i]}) · caps[i],   μ ≥ 0,

which bounds the minimum for *any* `λ` and `μ ≥ 0` (weak duality). The
multipliers come from the greedy primal (each floor met by its group's cheapest
assets, the remaining budget by the globally cheapest capacity), which makes the
bound tight for this laminar structure. Returns `(bound, λ, μ)`.
"""
function _portfolio_floor_bound(
    costs::Vector{Float64}, caps::Vector{Float64}, groups::Vector{Int}, floors::Vector{Float64}
)
    n_groups = length(floors)
    remaining = copy(caps)
    marginal_floor = fill(-Inf, n_groups)
    for g in 1:n_groups
        need = floors[g]
        need <= 0 && continue
        members = findall(==(g), groups)
        for i in members[sortperm(costs[members])]
            take = min(remaining[i], need)
            take <= 0 && continue
            remaining[i] -= take
            need -= take
            marginal_floor[g] = costs[i]
            need <= 1e-15 && break
        end
    end
    budget = 1.0 - sum(max.(floors, 0.0))
    λ = minimum(costs[i] for i in eachindex(costs) if remaining[i] > 1e-15; init=maximum(costs))
    for i in sortperm(costs)
        budget <= 1e-15 && break
        take = min(remaining[i], budget)
        take <= 0 && continue
        budget -= take
        λ = costs[i]
    end
    μ = [max(0.0, marginal_floor[g] - λ) for g in 1:n_groups]
    bound = λ + sum(μ[g] * floors[g] for g in 1:n_groups)
    for i in eachindex(costs)
        bound += min(0.0, costs[i] - λ - μ[groups[i]]) * caps[i]
    end
    return bound, λ, μ
end

"""
    _portfolio_market(rng, n_assets, n_scenarios, n_styles, n_sectors; kwargs...)

Generate a factor-structured scenario market (see the file header).

Keyword arguments:

  - `market_beta`: per-asset market betas (default `Normal(1, 0.25)`)
  - `idio_vol`: per-asset idiosyncratic volatility (default `U(0.04, 0.10)`)
  - `crash_probability`: probability of a market-crash regime per scenario
  - `shocks_per_scenario`: idiosyncratic shocks per scenario `J`
  - `style_scale`: per-asset multiplier on style loadings (default ones)
"""
function _portfolio_market(
    rng::AbstractRNG,
    n_assets::Int,
    n_scenarios::Int,
    n_styles::Int,
    n_sectors::Int;
    market_beta::Union{Nothing, Vector{Float64}}=nothing,
    idio_vol::Union{Nothing, Vector{Float64}}=nothing,
    crash_probability::Float64=0.05,
    shocks_per_scenario::Int=24,
    style_scale::Union{Nothing, Vector{Float64}}=nothing,
    sector::Union{Nothing, Vector{Int}}=nothing,
)
    n = n_assets
    S = n_scenarios
    sector = sector === nothing ? _portfolio_balanced_groups(rng, n, n_sectors) : sector
    n_sectors = maximum(sector)
    beta = market_beta === nothing ? rand(rng, Normal(1.0, 0.25), n) : market_beta
    vols = idio_vol === nothing ? rand(rng, Uniform(0.04, 0.10), n) : idio_vol
    scale = style_scale === nothing ? ones(n) : style_scale

    style_loadings = Matrix{Float64}(undef, n, 1 + n_styles)
    style_loadings[:, 1] .= beta
    for k in 1:n_styles
        style_loadings[:, 1 + k] .= rand(rng, Normal(0.0, 0.6), n) .* scale
    end

    # Factor returns (monthly): a fat-tailed market with a crash regime,
    # style and industry factors whose volatility rises in a crash.
    K = 1 + n_styles + n_sectors
    F = Matrix{Float64}(undef, S, K)
    for s in 1:S
        crash = rand(rng) < crash_probability
        F[s, 1] = crash ? rand(rng, Normal(-0.12, 0.04)) : 0.007 + 0.035 * rand(rng, TDist(5)) / sqrt(5 / 3)
        stress = crash ? 1.8 : 1.0
        for k in 1:n_styles
            F[s, 1 + k] = stress * rand(rng, Normal(0.0, 0.015))
        end
        for g in 1:n_sectors
            F[s, 1 + n_styles + g] = stress * rand(rng, Normal(0.0, 0.025))
        end
    end

    # Sparse idiosyncratic shocks with variance-preserving scaling.
    J = min(n, shocks_per_scenario)
    inflate = sqrt(n / J)
    I = Vector{Int}(undef, J * S)
    Jc = Vector{Int}(undef, J * S)
    V = Vector{Float64}(undef, J * S)
    q = 0
    for s in 1:S
        for i in _portfolio_distinct(rng, n, J)
            q += 1
            I[q] = i
            Jc[q] = s
            V[q] = inflate * vols[i] * rand(rng, TDist(4)) / sqrt(2.0)
        end
    end
    E = sparse(I, Jc, V, n, S)

    # Forecasts: factor premia plus a small cross-sectional alpha signal.
    premia = vcat(0.006, rand(rng, Normal(0.0, 0.002), n_styles), rand(rng, Normal(0.0, 0.002), n_sectors))
    mu = style_loadings * premia[1:(1 + n_styles)]
    for i in 1:n
        mu[i] += premia[1 + n_styles + sector[i]] + rand(rng, Normal(0.0, 0.002))
    end

    # Cap-weighted benchmark (log-normal market caps).
    caps = rand(rng, LogNormal(0.0, 1.0), n)
    benchmark = caps ./ sum(caps)

    return PortfolioMarket(n, S, n_styles, n_sectors, style_loadings, sector, F, E, mu, benchmark)
end

"""
    _portfolio_scenario_expressions(market, x, f)

Return a vector of affine expressions, one per scenario, for the portfolio
return `Σ_k F[s,k] f_k + Σ_{i ∈ J_s} E[i,s] x_i`, where `x` is indexed by asset
(entries may be `nothing` for assets without a weight variable).
"""
function _portfolio_scenario_expressions(market::PortfolioMarket, x, f)
    E = market.idiosyncratic
    rows = rowvals(E)
    vals = nonzeros(E)
    K = _portfolio_n_factors(market)
    exprs = Vector{AffExpr}(undef, market.n_scenarios)
    for s in 1:market.n_scenarios
        expr = AffExpr(0.0)
        range = nzrange(E, s)
        sizehint!(expr.terms, K + length(range) + 2)
        for k in 1:K
            add_to_expression!(expr, market.factor_returns[s, k], f[k])
        end
        for ptr in range
            xi = x[rows[ptr]]
            xi === nothing && continue
            add_to_expression!(expr, vals[ptr], xi)
        end
        exprs[s] = expr
    end
    return exprs
end

include("cvar.jl")
include("tracking_error.jl")
