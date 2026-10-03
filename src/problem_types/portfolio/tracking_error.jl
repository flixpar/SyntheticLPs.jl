using JuMP
using Random
using Distributions

"""
    TrackingErrorWitness

Planted feasible point of a [`TrackingErrorPortfolioProblem`](@ref): weights
over the investable assets (the benchmark restricted to the investable universe
and water-filled under 90% of the position caps), their factor `exposures`, and
their mean absolute tracking error `tracking_error < te_budget`.
"""
struct TrackingErrorWitness
    weights::Vector{Float64}
    exposures::Vector{Float64}
    tracking_error::Float64
end

"""
    TrackingErrorCertificate

Exclusion-versus-neutrality proof. The commodity style factor (column
`factor` of the exposure block) must stay within `band_lower` of the benchmark
exposure, but every investable asset has commodity loading at most
`max_investable_loading < band_lower`. The factor-definition row
`f = Σ_i B_i x_i` together with the budget row `Σ x_i = 1` and `x ≥ 0` gives
`f ≤ max_investable_loading`, contradicting the bound `f ≥ band_lower`.
"""
struct TrackingErrorCertificate
    factor::Int
    max_investable_loading::Float64
    band_lower::Float64
end

"""
    TrackingErrorPortfolioProblem <: ProblemGenerator

Enhanced-index (tracking) equity portfolio under an ESG exclusion list.

# Data

An equity universe on the factor-structured scenario market of `portfolio.jl`.
One sector is *energy*, and the last style factor is a *commodity* factor on
which energy names load heavily (1.5–3) and everything else lightly (−0.01 to
0.1). An exclusion list (1–4% of names) removes some names from the investable universe; the
benchmark still holds them, so the excluded weight is an unavoidable active
bet.

# Formulation

Variables: weights `x_i` for the investable names, factor exposures `f`
(market, styles, sector weights), and per-scenario absolute active returns `u_s`:

```math
\\max \\sum_i α_i x_i \\quad\\text{s.t.}\\quad
u_s \\ge \\pm\\Big(F_s·f + \\sum_{i \\in J_s} E_{is} x_i - r_s·b\\Big),\\;
\\tfrac1S \\sum_s u_s \\le \\mathrm{TE},\\; \\sum_i x_i = 1,
```

with `f = Bᵀx` (definition rows), active factor bands
`|f_k − f^b_k| ≤ δ_k` and active sector bands as bounds on `f`, and position
caps as bounds on `x`. The benchmark scenario return `r_s·b` (including
excluded names) is a constant.

# Feasibility

  - `feasible`: a small random exclusion list; the investable benchmark is
    water-filled under 90% of the caps, and the bands and TE budget are widened
    around it (`feasible_witness`).
  - `infeasible`: the whole energy sector is excluded while the commodity band
    demands more commodity exposure than any investable name carries
    (`infeasibility_certificate`; factor row + budget row + bounds — not a
    single-row bound conflict).
  - `unknown`: a small random exclusion list and a natural mandate (bands drawn
    relative to the benchmark, TE budget 0.4–1.1× the naive
    exclusion-renormalized portfolio's TE, floored at 5% of the benchmark's
    scenario MAD), with no repair.

# Sizing

Variables = `n_investable + n_factors + n_scenarios`, exact for targets ≥ 60,
with `n_assets ≈ 15–30%` of the target. Rows
= `2·n_scenarios + n_factors + 2`; nonzeros ≈ `2·S·(K + J + 1) + 7n` (≈ 3–5M at
100k variables).
"""
struct TrackingErrorPortfolioProblem <: ProblemGenerator
    market::PortfolioMarket
    investable::Vector{Int}
    excluded::Vector{Int}
    max_position::Vector{Float64}
    exposure_lower::Vector{Float64}
    exposure_upper::Vector{Float64}
    benchmark_returns::Vector{Float64}
    te_budget::Float64
    energy_sector::Int
    feasible_witness::Union{Nothing, TrackingErrorWitness}
    infeasibility_certificate::Union{Nothing, TrackingErrorCertificate}
end

"""Mean absolute active return of full-universe weights `x` against benchmark returns."""
function _tracking_error(market::PortfolioMarket, x::Vector{Float64}, benchmark_returns::Vector{Float64})
    active = _portfolio_scenario_returns(market, x) .- benchmark_returns
    return sum(abs, active) / length(active)
end

"""
    TrackingErrorPortfolioProblem(target_variables, feasibility_status, seed)

Construct an enhanced-index tracking instance with a constructor-local RNG. See
the type docstring for sizing and contracts.
"""
function TrackingErrorPortfolioProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    V = max(target_variables, 40)

    # --- Dimensions: V ≈ n_investable + K + S. ---
    share = rand(rng, Uniform(0.15, 0.30))
    n = max(8, round(Int, share * V))
    n_sectors = clamp(rand(rng, 8:11), 2, max(2, n ÷ 4))
    n_styles = clamp(rand(rng, 3:6), 2, max(2, n ÷ 4))     # last style = commodity
    K = 1 + n_styles + n_sectors

    # Sectors: sector 1 is energy (5–10% of names).
    sector = Vector{Int}(undef, n)
    n_energy = clamp(round(Int, n * rand(rng, Uniform(0.05, 0.10))), 1, n ÷ 4)
    perm = randperm(rng, n)
    energy_names = sort(perm[1:n_energy])
    sector[energy_names] .= 1
    rest = perm[(n_energy + 1):end]
    sector[rest] .= 1 .+ _portfolio_balanced_groups(rng, length(rest), n_sectors - 1)

    # Exclusions: a small random ESG list (never energy-only here).
    infeasible_request = feasibility_status == infeasible
    n_random_excl = clamp(round(Int, n * rand(rng, Uniform(0.01, 0.04))), 1, n ÷ 4)
    excluded = sort(unique(vcat(
        _portfolio_distinct(rng, n, n_random_excl),
        infeasible_request ? energy_names : Int[],
    )))
    investable = setdiff(1:n, excluded)
    n_inv = length(investable)
    S = max(10, V - n_inv - K)

    market = _portfolio_market(
        rng,
        n,
        S,
        n_styles,
        n_sectors;
        sector=sector,
        crash_probability=rand(rng, Uniform(0.02, 0.05)),
        shocks_per_scenario=rand(rng, 16:32),
    )
    commodity = 1 + n_styles                               # column in style_loadings
    for i in 1:n
        market.style_loadings[i, commodity] = sector[i] == 1 ? rand(rng, Uniform(1.5, 3.0)) : rand(rng, Uniform(-0.01, 0.1))
    end
    # Energy names are a meaningful share of the cap-weighted benchmark.
    b = market.benchmark
    energy_weight = sum(b[energy_names])
    target_energy = rand(rng, Uniform(0.08, 0.12))
    b[energy_names] .*= target_energy / energy_weight
    others = setdiff(1:n, energy_names)
    b[others] .*= (1 - target_energy) / sum(b[others])
    # Re-derive scenario-dependent data after editing loadings and weights.
    benchmark_returns = _portfolio_scenario_returns(market, b)
    bench_exposure = _portfolio_exposures(market, b)
    n_style_cols = 1 + n_styles

    # --- Natural mandate. ---
    max_position = zeros(Float64, n)
    for i in investable
        max_position[i] = max(b[i] * rand(rng, Uniform(1.5, 3.0)), rand(rng, Uniform(2.0, 5.0)) / n)
    end
    inv_caps = max_position[investable]
    sum(inv_caps) < 1.5 && (max_position[investable] .*= 1.5 / sum(inv_caps))
    exposure_lower = Vector{Float64}(undef, K)
    exposure_upper = Vector{Float64}(undef, K)
    for k in 1:K
        width = k <= n_style_cols ? rand(rng, Uniform(0.05, 0.2)) : rand(rng, Uniform(0.02, 0.06))
        exposure_lower[k] = bench_exposure[k] - width
        exposure_upper[k] = bench_exposure[k] + width
    end
    for k in (n_style_cols + 1):K
        exposure_lower[k] = max(0.0, exposure_lower[k])
    end
    # Sector bands apply only to sectors with investable names left: an
    # excluded sector's active weight is the exclusion itself.
    for g in 1:n_sectors
        if !any(market.sector[i] == g for i in investable)
            exposure_lower[n_style_cols + g] = 0.0
        end
    end
    naive = zeros(Float64, n)
    naive[investable] .= b[investable] ./ sum(b[investable])
    naive_te = _tracking_error(market, naive, benchmark_returns)
    bench_mad = sum(abs, benchmark_returns .- sum(benchmark_returns) / S) / S
    te_budget = max(naive_te, 0.05 * bench_mad) * rand(rng, Uniform(0.4, 1.1))

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        x_ref = zeros(Float64, n)
        x_ref[investable] .= _portfolio_waterfill(b[investable], 0.9 .* max_position[investable])
        f_ref = _portfolio_exposures(market, x_ref)
        for k in 1:K
            exposure_lower[k] = min(exposure_lower[k], f_ref[k] - 0.01)
            exposure_upper[k] = max(exposure_upper[k], f_ref[k] + 0.01)
        end
        ref_te = _tracking_error(market, x_ref, benchmark_returns)
        te_budget = max(te_budget, ref_te * rand(rng, Uniform(1.05, 1.3)))
        witness = TrackingErrorWitness(x_ref[investable], f_ref, ref_te)
    elseif infeasible_request
        loadings = market.style_loadings[investable, commodity]
        max_loading = maximum(loadings)
        # Keep the factor row satisfiable on its own under the position caps
        # (Σ_{B>0} B·cap above the band), so only the budget row exposes the
        # contradiction — presolve's single-row activity check cannot.
        row_max() = sum(max(loadings[j], 0.0) * max_position[investable[j]] for j in eachindex(investable))
        if 0.9 * row_max() < 1.5 * max_loading
            factor = 1.5 * max_loading / (0.9 * row_max())
            for i in investable
                max_position[i] = min(1.0, max_position[i] * factor)
            end
        end
        ceiling = min(bench_exposure[commodity], 0.9 * row_max())
        gap = ceiling - max_loading
        gap > 1e-3 || error("internal: commodity exposure gap not positive")
        exposure_lower[commodity] = max_loading + gap * rand(rng, Uniform(0.3, 0.6))
        exposure_upper[commodity] = bench_exposure[commodity] + (bench_exposure[commodity] - exposure_lower[commodity])
        certificate = TrackingErrorCertificate(commodity, max_loading, exposure_lower[commodity])
    end

    return TrackingErrorPortfolioProblem(
        market,
        investable,
        excluded,
        max_position,
        exposure_lower,
        exposure_upper,
        benchmark_returns,
        te_budget,
        1,
        witness,
        certificate,
    )
end

"""
    build_model(prob::TrackingErrorPortfolioProblem)

Build the tracking-error LP. Deterministic and linear in the number of nonzeros.
"""
function build_model(prob::TrackingErrorPortfolioProblem)
    model = Model()
    market = prob.market
    n = market.n_assets
    S = market.n_scenarios
    K = _portfolio_n_factors(market)
    n_style_cols = 1 + market.n_styles

    @variable(model, 0 <= x[j=1:length(prob.investable)] <= prob.max_position[prob.investable[j]])
    @variable(model, prob.exposure_lower[k] <= exposure[k=1:K] <= prob.exposure_upper[k])
    @variable(model, u[1:S] >= 0)

    asset_var = Vector{Union{Nothing, VariableRef}}(nothing, n)
    for (j, i) in enumerate(prob.investable)
        asset_var[i] = x[j]
    end

    objective = AffExpr(0.0)
    for (j, i) in enumerate(prob.investable)
        add_to_expression!(objective, market.expected_returns[i], x[j])
    end
    @objective(model, Max, objective)

    for k in 1:K
        expr = AffExpr(0.0)
        add_to_expression!(expr, 1.0, exposure[k])
        for (j, i) in enumerate(prob.investable)
            loading = k <= n_style_cols ? market.style_loadings[i, k] : (market.sector[i] == k - n_style_cols ? 1.0 : 0.0)
            iszero(loading) || add_to_expression!(expr, -loading, x[j])
        end
        @constraint(model, expr == 0)
    end

    active = _portfolio_scenario_expressions(market, asset_var, exposure)
    for s in 1:S
        r = prob.benchmark_returns[s]
        @constraint(model, u[s] - active[s] >= -r)
        @constraint(model, u[s] + active[s] >= r)
    end
    te = AffExpr(0.0)
    sizehint!(te.terms, S)
    for s in 1:S
        add_to_expression!(te, 1.0 / S, u[s])
    end
    @constraint(model, te_budget, te <= prob.te_budget)
    @constraint(model, budget, sum(x) == 1.0)

    return model
end

register_variant(
    :portfolio,
    :tracking_error,
    TrackingErrorPortfolioProblem,
    "Enhanced-index tracking under an ESG exclusion list: max alpha under a MAD tracking-error budget, factor and sector bands, and position caps",
)
