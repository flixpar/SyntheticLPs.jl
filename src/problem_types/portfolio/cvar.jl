using JuMP
using Random
using Distributions

"""Asset-class profiles: (name, beta mean, beta sd, idio vol range, style scale, cost range)."""
const CVAR_ASSET_CLASSES = (
    (:equity_developed, 1.0, 0.22, (0.05, 0.09), 1.0, (0.0005, 0.0015)),
    (:government_bonds, -0.05, 0.05, (0.008, 0.02), 0.2, (0.0002, 0.0006)),
    (:equity_emerging, 1.2, 0.30, (0.07, 0.12), 1.0, (0.0015, 0.004)),
    (:credit, 0.3, 0.10, (0.015, 0.03), 0.4, (0.0006, 0.0015)),
    (:alternatives, 0.6, 0.25, (0.04, 0.08), 0.6, (0.002, 0.005)),
)

"""
    CVaRWitness

Planted feasible point of a [`PortfolioProblem`](@ref): weights `x` (the
benchmark water-filled under 90% of the position caps — every cap respected
exactly), their factor `exposures` (market, styles, then sector weights), the
minimizing VaR level `alpha`, the portfolio `cvar`, and its `turnover`
`Σ|x − benchmark|`. Shortfalls are `z_s = max(L_s − alpha, 0)`.
"""
struct CVaRWitness
    weights::Vector{Float64}
    exposures::Vector{Float64}
    alpha::Float64
    cvar::Float64
    turnover::Float64
end

"""
    CVaRTailCertificate

Crash-tail proof that the CVaR limit is unattainable. `tail` holds
`⌈(1−β)S⌉` scenarios (the worst market-factor draws). Uniform multipliers
`q_s = 1/|tail| ≤ 1/((1−β)S)` on their shortfall rows, combined with the CVaR
row, give `cvar_limit ≥ α + Σ_s q_s z_s ≥ Σ_s q_s L_s(x) = ℓ·x` with `ℓ`
(`asset_tail_loss`) the mean tail loss per asset. The budget row (multiplier
`λ = budget_multiplier`), the asset-class floor rows (multipliers
`μ = class_multipliers ≥ 0`), and the position bounds then give
`ℓ·x ≥ loss_bound = λ + Σ_c μ_c floor_c + Σ_i min(0, ℓ_i − λ − μ_{c(i)}) cap_i`
(Lagrangian weak duality — valid for any such multipliers; see
`_portfolio_floor_bound`). The instance is infeasible because
`cvar_limit < loss_bound` (by 15–40%). The class floors matter: government
bonds rally in the crash, so without the equity floors the bound would be
negative.
"""
struct CVaRTailCertificate
    tail::Vector{Int}
    asset_tail_loss::Vector{Float64}
    budget_multiplier::Float64
    class_multipliers::Vector{Float64}
    loss_bound::Float64
    cvar_limit::Float64
end

"""
    ClassFloorCertificate

The asset-class floors sum to `floor_sum > 1`, but the classes partition the
assets and the budget row forces `Σx = 1` (sum of the class rows' lower sides
versus the budget row).
"""
struct ClassFloorCertificate
    floor_sum::Float64
end

"""
    TurnoverSectorCertificate

Sector caps (bounds on the sector-exposure variables) sit below the benchmark
weight of `sectors`, forcing sales of at least `deficit = Σ (b_g − cap_g)`;
because the budget row keeps total weight at one, purchases equal sales, so
`Σ|x − b| ≥ 2·deficit > turnover_limit`.
"""
struct TurnoverSectorCertificate
    sectors::Vector{Int}
    deficit::Float64
    turnover_limit::Float64
end

"""
    PortfolioProblem <: ProblemGenerator

CVaR-constrained multi-asset portfolio construction (institutional mandate).

# Data

A multi-asset universe (developed/emerging equity, government bonds, credit,
alternatives — class-specific betas, idiosyncratic volatilities, and trading
costs) on the factor-structured scenario market of `portfolio.jl`: a fat-tailed
market factor with a crash regime, style factors, one industry factor per
sector, and sparse idiosyncratic jump events. The benchmark is
cap-weighted.

# Formulation

```math
\\max \\sum_i μ_i x_i - \\sum_i c_i (d^+_i + d^-_i)
```

subject to factor-exposure definitions `f = Bᵀx` (market and style exposures
banded by variable bounds; sector exposures capped by variable bounds), the
Rockafellar–Uryasev CVaR rows `z_s + F_s·f + E_s·x + α ≥ 0`, the CVaR limit
`α + Σ z_s/((1−β)S) ≤ limit`, full investment, region caps, asset-class ranges,
position caps (variable bounds), and turnover `d⁺ − d⁻ = x − b`,
`Σ(d⁺ + d⁻) ≤ T`.

# Feasibility

  - `feasible`: the benchmark is water-filled under 90% of the position caps
    (`_portfolio_waterfill`, exact caps — fixing the old clip-then-renormalize
    bug), and every constraint is widened around it (`feasible_witness`).
  - `infeasible`: the feasible construction, then one contradiction
    (`infeasibility_mode`):
      * `:crash_tail` (≈50%, default) — the CVaR limit is set below the exact
        minimum crash-tail loss over the capped simplex (needs the tail
        shortfall rows, the CVaR row, the budget row, and bounds);
      * `:class_floor` (≈25%) — asset-class floors sum above one;
      * `:turnover_sector` (≈25%) — sector caps below benchmark sector weights
        demand more turnover than allowed.
    (`infeasibility_certificate` holds the matching typed proof.)
  - `unknown`: the natural mandate (limits drawn relative to the benchmark,
    CVaR limit 0.55–0.95× the benchmark's CVaR; the attainable minimum is
    typically 0.55–0.9× it) with no repair, so outcomes are two-sided.

# Sizing

Variables = `3·n_assets + n_factors + n_scenarios + 1`, exact for targets ≥ 40,
with `n_assets ≈ 10–18%` of the target. Scenario rows have
`n_factors + J + 2` nonzeros (`J ≤ 24` idiosyncratic shocks), so nonzeros grow
linearly (≈ 3M at 100k variables).
"""
struct PortfolioProblem <: ProblemGenerator
    market::PortfolioMarket
    asset_class::Vector{Int}
    region::Vector{Int}
    class_names::Vector{Symbol}
    cvar_level::Float64
    cvar_limit::Float64
    max_position::Vector{Float64}
    exposure_lower::Vector{Float64}
    exposure_upper::Vector{Float64}
    sector_upper::Vector{Float64}
    region_upper::Vector{Float64}
    class_lower::Vector{Float64}
    class_upper::Vector{Float64}
    transaction_cost::Vector{Float64}
    turnover_limit::Float64
    infeasibility_mode::Symbol
    feasible_witness::Union{Nothing, CVaRWitness}
    infeasibility_certificate::Union{
        Nothing, CVaRTailCertificate, ClassFloorCertificate, TurnoverSectorCertificate
    }
end

"""Group sums of `x` under an assignment vector."""
function _portfolio_group_sums(x::AbstractVector{Float64}, groups::Vector{Int}, n_groups::Int)
    sums = zeros(Float64, n_groups)
    for i in eachindex(x)
        sums[groups[i]] += x[i]
    end
    return sums
end

"""
    PortfolioProblem(target_variables, feasibility_status, seed)

Construct a CVaR portfolio instance with a constructor-local RNG. See the type
docstring for sizing and contracts.
"""
function PortfolioProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    V = max(target_variables, 40)

    # --- Dimensions: V = 3n + K + S + 1 exactly. ---
    share = rand(rng, Uniform(0.10, 0.18))
    sector_draw = rand(rng, 8:12)
    style_draw = rand(rng, 3:6)
    n = max(4, round(Int, share * V))
    n_sectors = n_styles = S = 0
    while true
        n_sectors = clamp(sector_draw, 2, max(2, n ÷ 3))
        n_styles = clamp(style_draw, 1, max(1, n ÷ 4))
        S = V - 3n - (1 + n_styles + n_sectors) - 1
        (S >= max(10, n ÷ 2) || n <= 4) && break
        n -= 1
    end
    S = max(S, 10)

    # --- Asset classes, regions, and class-specific characteristics. ---
    n_classes = min(rand(rng, 3:5), n)
    asset_class = _portfolio_balanced_groups(rng, n, n_classes)
    n_regions = clamp(rand(rng, 3:6), 1, max(1, n ÷ 2))
    region = _portfolio_balanced_groups(rng, n, n_regions)
    betas = Vector{Float64}(undef, n)
    vols = Vector{Float64}(undef, n)
    scale = Vector{Float64}(undef, n)
    costs = Vector{Float64}(undef, n)
    for i in 1:n
        _, bmean, bsd, vrange, sscale, crange = CVAR_ASSET_CLASSES[asset_class[i]]
        betas[i] = rand(rng, Normal(bmean, bsd))
        vols[i] = rand(rng, Uniform(vrange...))
        scale[i] = sscale
        costs[i] = rand(rng, Uniform(crange...))
    end
    market = _portfolio_market(
        rng,
        n,
        S,
        n_styles,
        n_sectors;
        market_beta=betas,
        idio_vol=vols,
        style_scale=scale,
        crash_probability=rand(rng, Uniform(0.03, 0.08)),
        shocks_per_scenario=rand(rng, 12:24),
    )
    n_sectors = market.n_sectors
    b = market.benchmark
    K = _portfolio_n_factors(market)
    n_style_cols = 1 + n_styles
    cvar_level = rand(rng, (0.90, 0.95, 0.975, 0.99))

    # --- Natural mandate, drawn relative to the benchmark. ---
    max_position = [max(b[i] * rand(rng, Uniform(1.5, 3.0)), rand(rng, Uniform(2.0, 5.0)) / n) for i in 1:n]
    sum(max_position) < 1.5 && (max_position .*= 1.5 / sum(max_position))
    bench_exposure = _portfolio_exposures(market, b)
    exposure_lower = [bench_exposure[k] - rand(rng, Uniform(0.05, 0.25)) for k in 1:n_style_cols]
    exposure_upper = [bench_exposure[k] + rand(rng, Uniform(0.05, 0.25)) for k in 1:n_style_cols]
    bench_sector = bench_exposure[(n_style_cols + 1):end]
    sector_upper = [min(1.0, w * rand(rng, Uniform(1.2, 2.0)) + 0.01) for w in bench_sector]
    bench_region = _portfolio_group_sums(b, region, n_regions)
    region_upper = [min(1.0, w * rand(rng, Uniform(1.2, 2.0)) + 0.01) for w in bench_region]
    bench_class = _portfolio_group_sums(b, asset_class, n_classes)
    class_lower = [w * rand(rng, Uniform(0.5, 0.85)) for w in bench_class]
    class_upper = [min(1.0, w * rand(rng, Uniform(1.15, 1.6)) + 0.02) for w in bench_class]
    turnover_limit = rand(rng, Uniform(0.1, 0.6))
    bench_cvar, _ = _portfolio_cvar(-_portfolio_scenario_returns(market, b), cvar_level)
    cvar_limit = bench_cvar * rand(rng, Uniform(0.55, 0.95))

    witness = nothing
    certificate = nothing
    mode = :none
    if feasibility_status != unknown
        # Reference portfolio: benchmark water-filled under 90% of the caps.
        x_ref = _portfolio_waterfill(b, 0.9 .* max_position)
        f_ref = _portfolio_exposures(market, x_ref)
        for k in 1:n_style_cols
            exposure_lower[k] = min(exposure_lower[k], f_ref[k] - 0.02)
            exposure_upper[k] = max(exposure_upper[k], f_ref[k] + 0.02)
        end
        for g in 1:n_sectors
            sector_upper[g] = max(sector_upper[g], min(1.0, f_ref[n_style_cols + g] * 1.1 + 0.005))
        end
        ref_region = _portfolio_group_sums(x_ref, region, n_regions)
        region_upper .= max.(region_upper, min.(1.0, ref_region .* 1.1 .+ 0.005))
        ref_class = _portfolio_group_sums(x_ref, asset_class, n_classes)
        class_lower .= min.(class_lower, ref_class .* 0.9)
        class_upper .= max.(class_upper, min.(1.0, ref_class .* 1.1 .+ 0.005))
        ref_turnover = sum(abs.(x_ref .- b))
        turnover_limit = max(turnover_limit, ref_turnover * 1.1 + 0.01)
        ref_cvar, ref_alpha = _portfolio_cvar(-_portfolio_scenario_returns(market, x_ref), cvar_level)
        cvar_limit = max(cvar_limit, ref_cvar + abs(ref_cvar) * rand(rng, Uniform(0.05, 0.2)) + 1e-4)

        if feasibility_status == feasible
            witness = CVaRWitness(x_ref, f_ref, ref_alpha, ref_cvar, ref_turnover)
        else
            draw = rand(rng)
            mode = draw < 0.5 ? :crash_tail : (draw < 0.75 ? :class_floor : :turnover_sector)
            if mode == :crash_tail
                n_tail = clamp(ceil(Int, (1 - cvar_level) * S), 1, S)
                tail = sortperm(view(market.factor_returns, :, 1))[1:n_tail]
                mean_factor = vec(sum(market.factor_returns[tail, :]; dims=1)) ./ n_tail
                tail_loss = Vector{Float64}(undef, n)
                for i in 1:n
                    r = dot(view(market.style_loadings, i, :), view(mean_factor, 1:n_style_cols))
                    r += mean_factor[n_style_cols + market.sector[i]]
                    tail_loss[i] = -r
                end
                idio_tail = market.idiosyncratic[:, tail]
                tail_loss .-= vec(sum(idio_tail; dims=2)) ./ n_tail
                bound, λ, μ = _portfolio_floor_bound(tail_loss, max_position, asset_class, class_lower)
                if bound > 1e-3
                    cvar_limit = bound * rand(rng, Uniform(0.6, 0.85))
                    certificate = CVaRTailCertificate(sort(tail), tail_loss, λ, μ, bound, cvar_limit)
                else
                    mode = :class_floor                   # defensive assets gain in the crash
                end
            end
            if mode == :turnover_sector
                n_hit = min(n_sectors, rand(rng, 1:3))
                hit = sortperm(bench_sector; rev=true)[1:n_hit]
                for g in hit
                    sector_upper[g] = bench_sector[g] * rand(rng, Uniform(0.3, 0.6))
                end
                deficit = sum(bench_sector[g] - sector_upper[g] for g in hit)
                turnover_limit = 2 * deficit * rand(rng, Uniform(0.5, 0.85))
                certificate = TurnoverSectorCertificate(sort(hit), deficit, turnover_limit)
            end
            if mode == :class_floor
                target_sum = rand(rng, Uniform(1.05, 1.2))
                class_lower .*= target_sum / sum(class_lower)
                class_upper .= max.(class_upper, min.(1.0, class_lower .+ 0.02))
                certificate = ClassFloorCertificate(sum(class_lower))
            end
        end
    end

    return PortfolioProblem(
        market,
        asset_class,
        region,
        [CVAR_ASSET_CLASSES[c][1] for c in 1:n_classes],
        cvar_level,
        cvar_limit,
        max_position,
        exposure_lower,
        exposure_upper,
        sector_upper,
        region_upper,
        class_lower,
        class_upper,
        costs,
        turnover_limit,
        mode,
        witness,
        certificate,
    )
end

"""
    build_model(prob::PortfolioProblem)

Build the CVaR portfolio LP. Deterministic and linear in the number of nonzeros.
"""
function build_model(prob::PortfolioProblem)
    model = Model()
    market = prob.market
    n = market.n_assets
    S = market.n_scenarios
    n_style_cols = 1 + market.n_styles
    G = market.n_sectors

    @variable(model, 0 <= x[i=1:n] <= prob.max_position[i])
    @variable(model, prob.exposure_lower[k] <= style_exposure[k=1:n_style_cols] <= prob.exposure_upper[k])
    @variable(model, 0 <= sector_exposure[g=1:G] <= prob.sector_upper[g])
    @variable(model, z[1:S] >= 0)
    @variable(model, -1.0 <= alpha <= 1.0)          # VaR level: a monthly loss fraction
    @variable(model, buy[1:n] >= 0)
    @variable(model, sell[1:n] >= 0)

    objective = AffExpr(0.0)
    sizehint!(objective.terms, 3n)
    for i in 1:n
        add_to_expression!(objective, market.expected_returns[i], x[i])
        add_to_expression!(objective, -prob.transaction_cost[i], buy[i])
        add_to_expression!(objective, -prob.transaction_cost[i], sell[i])
    end
    @objective(model, Max, objective)

    # Factor exposure definitions f = Bᵀx.
    for k in 1:n_style_cols
        expr = AffExpr(0.0)
        sizehint!(expr.terms, n + 1)
        add_to_expression!(expr, 1.0, style_exposure[k])
        for i in 1:n
            add_to_expression!(expr, -market.style_loadings[i, k], x[i])
        end
        @constraint(model, expr == 0)
    end
    sector_members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(sector_members[market.sector[i]], i)
    end
    for g in 1:G
        @constraint(model, sector_exposure[g] - sum(x[i] for i in sector_members[g]) == 0)
    end

    # CVaR shortfall rows: z_s ≥ L_s − α with L_s = −(scenario return).
    f = vcat(style_exposure, sector_exposure)
    returns = _portfolio_scenario_expressions(market, x, f)
    for s in 1:S
        add_to_expression!(returns[s], 1.0, z[s])
        add_to_expression!(returns[s], 1.0, alpha)
        @constraint(model, returns[s] >= 0)
    end
    tail_weight = 1.0 / ((1.0 - prob.cvar_level) * S)
    cvar = AffExpr(0.0)
    sizehint!(cvar.terms, S + 1)
    add_to_expression!(cvar, 1.0, alpha)
    for s in 1:S
        add_to_expression!(cvar, tail_weight, z[s])
    end
    @constraint(model, cvar_limit, cvar <= prob.cvar_limit)

    @constraint(model, budget, sum(x) == 1.0)

    for (groups, n_groups, kind) in ((prob.region, length(prob.region_upper), :region), (prob.asset_class, length(prob.class_lower), :class))
        members = [Int[] for _ in 1:n_groups]
        for i in 1:n
            push!(members[groups[i]], i)
        end
        for g in 1:n_groups
            expr = sum(x[i] for i in members[g])
            if kind == :region
                @constraint(model, expr <= prob.region_upper[g])
            else
                @constraint(model, prob.class_lower[g] <= expr <= prob.class_upper[g])
            end
        end
    end

    for i in 1:n
        @constraint(model, buy[i] - sell[i] - x[i] == -market.benchmark[i])
    end
    turnover = AffExpr(0.0)
    sizehint!(turnover.terms, 2n)
    for i in 1:n
        add_to_expression!(turnover, 1.0, buy[i])
        add_to_expression!(turnover, 1.0, sell[i])
    end
    @constraint(model, turnover_limit, turnover <= prob.turnover_limit)

    return model
end

register_variant(
    :portfolio,
    :cvar,
    PortfolioProblem,
    "Multi-asset CVaR portfolio on a factor-structured crash-regime scenario market with exposure, sector, region, class, position, and turnover limits";
    default=true,
    tags=[:finance, :dense],
)
