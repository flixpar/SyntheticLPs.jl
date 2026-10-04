using JuMP
using Random
using Distributions

"""
Asset classes available to the pension fund, in the order they are added as
the universe grows with the target. Fields: kind (`:cash`, `:bond`, `:equity`,
`:real`), interest-rate duration (years), annual risk premium over cash,
market beta, idiosyncratic annual volatility, proportional transaction cost,
and the maximum portfolio weight.
"""
const _ALM_ASSET_CLASSES = (
    (name=:cash, kind=:cash, duration=0.0, premium=0.0, beta=0.0, vol=0.0, tc=0.0, max_weight=1.0),
    (
        name=:gov_bond,
        kind=:bond,
        duration=7.0,
        premium=0.006,
        beta=0.0,
        vol=0.01,
        tc=0.002,
        max_weight=0.8,
    ),
    (
        name=:equity_dom,
        kind=:equity,
        duration=0.0,
        premium=0.045,
        beta=1.0,
        vol=0.06,
        tc=0.006,
        max_weight=0.45,
    ),
    (
        name=:corp_bond,
        kind=:bond,
        duration=5.0,
        premium=0.014,
        beta=0.15,
        vol=0.02,
        tc=0.004,
        max_weight=0.4,
    ),
    (
        name=:equity_intl,
        kind=:equity,
        duration=0.0,
        premium=0.05,
        beta=0.9,
        vol=0.09,
        tc=0.008,
        max_weight=0.35,
    ),
    (
        name=:real_estate,
        kind=:real,
        duration=0.0,
        premium=0.035,
        beta=0.45,
        vol=0.08,
        tc=0.024,
        max_weight=0.2,
    ),
    (
        name=:index_linked,
        kind=:bond,
        duration=10.0,
        premium=0.004,
        beta=0.0,
        vol=0.015,
        tc=0.003,
        max_weight=0.4,
    ),
    (
        name=:equity_em,
        kind=:equity,
        duration=0.0,
        premium=0.065,
        beta=1.25,
        vol=0.14,
        tc=0.016,
        max_weight=0.15,
    ),
    (
        name=:high_yield,
        kind=:bond,
        duration=4.0,
        premium=0.03,
        beta=0.45,
        vol=0.05,
        tc=0.01,
        max_weight=0.15,
    ),
    (
        name=:infrastructure,
        kind=:real,
        duration=0.0,
        premium=0.04,
        beta=0.35,
        vol=0.07,
        tc=0.03,
        max_weight=0.15,
    ),
)

"""
Planted fixed-mix policy of a requested-feasible `MultistageALMProblem`: at every
node the fund rebalances to the same `weights`, paying proportional transaction
costs. `holdings[a, n]`, `buys[a, n]`, and `sells[a, n]` (row 1, cash, has no
trades) satisfy every asset and cash balance row exactly (node wealth is the
column sum); the weights respect every per-asset and equity cap and
`exposure_scale` is drawn above the policy's best funding ratio, so every
exposure limit holds; and the funding floors hold because `min_funding_ratio`
(the policy's lowest wealth-to-liability ratio over all non-root nodes) is at
least the required ratio. `shortfall[k]` is the shortfall of the `k`-th leaf.
"""
struct MultistageALMWitness
    weights::Vector{Float64}
    holdings::Matrix{Float64}
    buys::Matrix{Float64}
    sells::Matrix{Float64}
    shortfall::Vector{Float64}
    min_funding_ratio::Float64
end

"""
Funding-floor infeasibility certificate along one root-to-node path. Summing a
node's asset and cash balance rows gives
`W_n = Σ_a R[a,n] h[a,parent] + inflow_n - outflow_n - transaction costs`. With
`0 ≤ h[a,parent] ≤ cap[a,parent]` (the parent's exposure-limit rows) and
`Σ_a h[a,parent] = W_parent`, the return term is at most the fractional-knapsack
value `growth_bound * W_parent` (best returns filled first; that value is
nondecreasing in `W_parent`). Chaining from the known root wealth along `path`
yields `wealth_bound[k]` for each path node; at the last node the funding floor
demands `required_wealth = funding_ratio * liability_value`, which exceeds
`wealth_bound[end]` by `margin`. The proof combines balance, wealth, and
exposure rows along a whole path with return-weighted multipliers — a
combination presolve does not find.
"""
struct MultistageALMCertificate
    path::Vector{Int}
    growth_bound::Vector{Float64}
    wealth_bound::Vector{Float64}
    required_wealth::Float64
    margin::Float64
end

"""
    MultistageALMProblem <: ProblemGenerator

Multistage stochastic asset–liability management (ALM) LP of a defined-benefit
pension fund on a scenario tree (Cariño–Ziemba / Consigli–Dempster style).

# Overview

The fund starts with known holdings and rebalances at every node of a scenario
tree. At node `n` (stage `t ≥ 1`) asset values are multiplied by the node's
gross returns, contributions flow in, and benefit payments flow out. Decisions
at each node are holdings `h[a, n] ≥ 0`, purchases `buy[a, n]`, and sales
`sell[a, n]` of every non-cash asset, plus the node's wealth; cash is the
settlement account (no borrowing). Constraints per node:

  - asset balance: `h[a,n] = R[a,n] h[a,parent] + buy[a,n] - sell[a,n]`;
  - cash balance: `h[cash,n] = R[cash,n] h[cash,parent] + Σ (1 - tc_a) sell[a,n]
    - Σ (1 + tc_a) buy[a,n] + inflow_n - outflow_n`;
  - wealth definition: `wealth[n] = Σ_a h[a,n]`;
  - exposure limits in liability units: the bound
    `h[a,n] ≤ max_weight_a * exposure_scale * L_n` for every risky asset, and a
    joint equity-limit row, where `L_n` is the node's liability value (a risk
    budget scaled to the strategic funding level);
  - regulatory funding floor (non-root nodes), as the bound
    `wealth[n] ≥ funding_ratio * L_n`;
  - leaves: `shortfall_n + wealth[n] ≥ target_ratio * L_n`.

The objective maximizes expected terminal wealth (relative to the initial
liability value) minus a convex penalty on terminal underfunding.
Nonanticipativity is implicit in the node formulation: siblings share their
parent's decision. The constraint matrix is a staircase along every path and a
tree overall — structurally different from the two-stage dual block-angular
`standard` variant (recourse here is nested through many stages, and the
coupling is through state carried node to node).

# Data grounding

Each tree edge draws a market shock, a short-rate change, and inflation.
Cash earns the parent's short rate; bonds earn it plus a premium minus
`duration × Δrate`; equities and real assets load on the market shock with
asset-specific betas and idiosyncratic noise; index-linked bonds earn
inflation. Benefit payments grow with inflation; the liability value is the
annuity value of payments at the node's short rate, so rising rates lower
liabilities and bond prices together (the classic ALM hedge).

# Feasibility control

Everything except the funding ratio is drawn identically for all statuses.

  - `feasible`: `funding_ratio = U(0.88, 0.97) ×` the minimum funding ratio
    reached by a planted fixed-mix policy (stored as
    [`MultistageALMWitness`](@ref)).
  - `infeasible`: along the path with the weakest best-case wealth bound,
    `funding_ratio = U(1.04, 1.10) ×` that bound's ratio — no policy, even a
    clairvoyant one within the exposure limits, can meet the floor there ([`MultistageALMCertificate`](@ref)).
  - `unknown`: `funding_ratio` drawn in the upper part (log scale) of the
    interval between the two thresholds, so the best nonanticipative policy
    decides; no witness or certificate is stored.

# Size

Every node has `n_assets` holdings, `2 (n_assets - 1)` trades, and a wealth
column; leaves add a shortfall column. With a uniform branching factor `b` over
`n_stages` stages and the final stage trimmed to `n_leaves` nodes (each
pre-leaf node keeps at least one child) the variable count is
`n_internal (3A - 1) + n_leaves 3A`, within about `3A / 2` of the target. Rows
per node: `A` balances, one wealth definition, and one equity limit, plus a
target row at each leaf.
"""
struct MultistageALMProblem <: ProblemGenerator
    n_assets::Int
    n_stages::Int
    asset_names::Vector{Symbol}
    asset_kind::Vector{Symbol}
    transaction_cost::Vector{Float64}
    max_weight::Vector{Float64}
    equity_cap::Float64
    exposure_scale::Float64
    parent::Vector{Int}
    stage::Vector{Int}
    probability::Vector{Float64}
    returns::Matrix{Float64}
    inflow::Vector{Float64}
    outflow::Vector{Float64}
    liability_value::Vector{Float64}
    initial_holdings::Vector{Float64}
    funding_ratio::Float64
    target_ratio::Float64
    shortfall_penalty::Float64
    stage_years::Float64
    feasible_witness::Union{Nothing, MultistageALMWitness}
    infeasibility_certificate::Union{Nothing, MultistageALMCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _alm_dimensions(target) -> (A, T, b, n_leaves)

Asset count, stage count, branching factor, and final-stage node count. Asset
universes grow with the target (3 classes for tiny trees, all 10 near 100k);
the search minimizes the size error, then prefers 3–5 stages with a branching
factor of at least 3.
"""
function _alm_dimensions(target::Int)
    t = max(target, 1)
    ideal_assets = clamp(round(Int, 1.0 + 1.9 * log10(t)), 3, length(_ALM_ASSET_CLASSES))
    best = (3, 2, 2, 4)
    best_score = (typemax(Int), Inf)
    for A in max(3, ideal_assets - 1):min(length(_ALM_ASSET_CLASSES), ideal_assets + 1),
        T in 2:6,
        b in 2:40

        internal = sum(b^k for k in 0:(T - 1))
        internal * (3A - 1) > 2t && continue
        leaves_hi = b^T
        leaves_lo = b^(T - 1)
        leaves = clamp(round(Int, (t - internal * (3A - 1)) / 3A), leaves_lo, leaves_hi)
        total = internal * (3A - 1) + leaves * 3A
        size_error = abs(total - t)
        shape =
            abs(A - ideal_assets) +
            (T < 3 ? 1.5 : 0.0) +
            (T > 5 ? 1.0 : 0.0) +
            (b < 3 ? 1.0 : 0.0) +
            0.3 * (1 - leaves / leaves_hi)
        score = (size_error <= max(1, 3A) ? 0 : size_error, shape)
        if score < best_score
            best_score = score
            best = (A, T, b, leaves)
        end
    end
    return best
end

"""
    _alm_tree(T, b, n_leaves) -> (parent, stage)

Breadth-first scenario tree: stages `0:T-1` are complete `b`-ary, and the
`n_leaves` final-stage nodes are dealt round-robin to the stage `T-1` nodes so
every pre-leaf node keeps at least one child.
"""
function _alm_tree(T::Int, b::Int, n_leaves::Int)
    parent = [0]
    stage = [0]
    frontier = [1]
    for t in 1:(T - 1)
        next = Int[]
        for p in frontier, _ in 1:b
            push!(parent, p)
            push!(stage, t)
            push!(next, length(parent))
        end
        frontier = next
    end
    counts = zeros(Int, length(frontier))
    for k in 1:n_leaves
        counts[mod1(k, length(frontier))] += 1
    end
    for (idx, p) in enumerate(frontier), _ in 1:counts[idx]
        push!(parent, p)
        push!(stage, T)
    end
    return parent, stage
end

"""
    _alm_fixed_mix(returns, inflow, outflow, parent, initial_holdings, weights, tc)

Simulate the fixed-mix policy: at each node, rebalance pre-trade values `v` to
`weights * W` where the post-trade wealth `W` solves
`W = Σ v + inflow - outflow - Σ_a tc_a |weights_a W - v_a|` (a contraction for
small costs, iterated to machine precision). Returns holdings, buys, sells.
"""
function _alm_fixed_mix(returns, inflow, outflow, parent, initial_holdings, weights, tc)
    A, N = size(returns)
    holdings = zeros(Float64, A, N)
    buys = zeros(Float64, A, N)
    sells = zeros(Float64, A, N)
    v = zeros(Float64, A)
    for n in 1:N
        if parent[n] == 0
            v .= initial_holdings
        else
            for a in 1:A
                v[a] = returns[a, n] * holdings[a, parent[n]]
            end
        end
        pre = sum(v) + inflow[n] - outflow[n]
        W = pre
        for _ in 1:200
            W_new = pre - sum(tc[a] * abs(weights[a] * W - v[a]) for a in 2:A)
            converged = abs(W_new - W) <= 1e-13 * max(1.0, abs(W))
            W = W_new
            converged && break
        end
        for a in 2:A
            holdings[a, n] = weights[a] * W
            buys[a, n] = max(0.0, holdings[a, n] - v[a])
            sells[a, n] = max(0.0, v[a] - holdings[a, n])
        end
        # Cash closes the balance exactly (absorbs the fixed-point residual).
        holdings[1, n] =
            v[1] + inflow[n] - outflow[n] +
            sum((1 - tc[a]) * sells[a, n] - (1 + tc[a]) * buys[a, n] for a in 2:A)
    end
    return holdings, buys, sells
end

"""
    _alm_best_growth(returns, caps, wealth) -> Float64

Largest gross growth `Σ_a returns[a] h[a] / wealth` over portfolios with
`Σ h = wealth` and `0 ≤ h ≤ caps` (fill the best returns first — a fractional
knapsack). Concave and nondecreasing in `wealth` once multiplied by it, so
applying it to an upper bound on the parent's wealth bounds the child's.
"""
function _alm_best_growth(returns, caps, wealth::Float64)
    wealth <= 0 && return maximum(returns)
    remaining = wealth
    value = 0.0
    for a in sortperm(returns; rev=true)
        take = min(caps[a], remaining)
        value += returns[a] * take
        remaining -= take
        remaining <= 0 && break
    end
    return value / wealth
end

function MultistageALMProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    A, T, b, n_leaves = _alm_dimensions(target_variables)
    classes = _ALM_ASSET_CLASSES[1:A]
    parent, stage = _alm_tree(T, b, n_leaves)
    N = length(parent)

    # --- Economic scenario generator on the tree ---
    stage_years = rand(rng, (1.0, 2.0, 3.0))
    Δ = stage_years
    rate0 = rand(rng, Uniform(0.01, 0.045))
    rate_vol = rand(rng, Uniform(0.006, 0.012))
    market_premium_scale = rand(rng, Uniform(0.8, 1.2))
    market_vol = rand(rng, Uniform(0.14, 0.20))
    inflation_mean = rand(rng, Uniform(0.015, 0.03))
    rate = zeros(Float64, N)
    cpi = ones(Float64, N)
    returns = ones(Float64, A, N)
    rate[1] = rate0
    for n in 2:N
        p = parent[n]
        z_market = randn(rng)
        z_rate = randn(rng)
        drate = rate_vol * sqrt(Δ) * z_rate
        rate[n] = clamp(rate[p] + drate, -0.005, 0.10)
        drate = rate[n] - rate[p]
        inflation = inflation_mean * Δ + 0.012 * sqrt(Δ) * randn(rng) + 0.4 * drate
        cpi[n] = cpi[p] * exp(inflation)
        market = market_vol * sqrt(Δ) * z_market
        for (a, c) in enumerate(classes)
            carry = rate[p] * Δ
            logret = if c.kind == :cash
                carry
            elseif c.name == :index_linked
                inflation + 0.006 * Δ - 0.5 * c.duration * drate + c.vol * sqrt(Δ) * randn(rng)
            elseif c.kind == :bond
                carry + c.premium * Δ - c.duration * drate +
                c.beta * 0.5 * market +
                c.vol * sqrt(Δ) * randn(rng)
            else
                total_vol2 = (c.beta * market_vol)^2 + c.vol^2
                carry +
                (market_premium_scale * c.premium - total_vol2 / 2) * Δ +
                c.beta * market +
                (c.kind == :real ? 0.3 * inflation : 0.0) +
                c.vol * sqrt(Δ) * randn(rng)
            end
            returns[a, n] = exp(logret)
        end
    end

    # Conditional probabilities: uniform over each node's children.
    n_children = zeros(Int, N)
    for n in 2:N
        n_children[parent[n]] += 1
    end
    probability = ones(Float64, N)
    for n in 2:N
        probability[n] = probability[parent[n]] / n_children[parent[n]]
    end

    # --- Liabilities: inflation-indexed benefits, annuity-valued at the node rate ---
    horizon_years = rand(rng, Uniform(12.0, 25.0))
    annual_benefit = rand(rng, Uniform(0.045, 0.075))  # relative to initial liability value
    annuity(r) = abs(r) < 1e-6 ? horizon_years : (1 - exp(-r * horizon_years)) / r
    benefit_scale = 1.0 / (annual_benefit * annuity(rate0))   # makes L_root == 1
    liability_value = [annual_benefit * benefit_scale * cpi[n] * annuity(rate[n]) for n in 1:N]
    outflow = [n == 1 ? 0.0 : annual_benefit * cpi[n] * Δ for n in 1:N]
    contribution_ratio = rand(rng, Uniform(0.3, 0.8))
    inflow = [n == 1 ? 0.0 : contribution_ratio * annual_benefit * cpi[n] * Δ for n in 1:N]

    # --- Initial portfolio and policy parameters ---
    tc = [c.tc for c in classes]
    max_weight = [c.max_weight for c in classes]
    is_equity = [c.kind == :equity for c in classes]
    equity_cap = rand(rng, Uniform(0.45, 0.65))
    initial_funding = rand(rng, Uniform(1.0, 1.25))
    raw = [rand(rng, Uniform(0.5, 1.5)) * min(c.max_weight, 0.3) for c in classes]
    initial_holdings = initial_funding .* raw ./ sum(raw)

    # Conservative fixed mix within every cap: bond-heavy, some equity.
    mix = zeros(Float64, A)
    for (a, c) in enumerate(classes)
        mix[a] = if c.kind == :cash
            rand(rng, Uniform(0.03, 0.08))
        elseif c.kind == :bond
            rand(rng, Uniform(0.5, 1.0)) * c.max_weight
        else
            rand(rng, Uniform(0.2, 0.6)) * c.max_weight
        end
    end
    mix ./= sum(mix)
    # Renormalizing can push a weight over its cap; move any excess to cash.
    excess = sum(max.(mix .- max_weight, 0.0))
    mix .= min.(mix, max_weight)
    mix[1] += excess
    equity_share = sum(mix[is_equity])
    if equity_share > 0.8 * equity_cap
        freed = equity_share - 0.8 * equity_cap
        mix[is_equity] .*= 0.8 * equity_cap / equity_share
        mix[1] += freed
    end

    holdings, buys, sells = _alm_fixed_mix(
        returns, inflow, outflow, parent, initial_holdings, mix, tc
    )
    wealth = vec(sum(holdings; dims=1))
    @assert all(>(0), wealth)
    @assert all(>=(-1e-12), holdings)
    witness_ratio = minimum(wealth[n] / liability_value[n] for n in 2:N)

    # Exposure limits are set in liability units (a risk budget scaled to the
    # fund's strategic funding level), loose enough for the planted policy in
    # its best states. Wealth-relative caps `h <= w * Σh` are homogeneous rows
    # whose cone made HiGHS's dual simplex return an unknown status on some
    # infeasible instances instead of proving infeasibility.
    best_ratio = maximum(wealth[n] / liability_value[n] for n in 1:N)
    exposure_scale = max(rand(rng, Uniform(1.2, 1.5)), 1.02 * best_ratio)

    # Best-case wealth bound along every path: a clairvoyant fund that, at the
    # parent, holds the assets with the best realized returns up to their
    # exposure limits (cash is unlimited).
    growth = ones(Float64, N)
    bound = zeros(Float64, N)
    bound[1] = sum(initial_holdings)
    for n in 2:N
        p = parent[n]
        caps = [
            max_weight[a] < 1.0 ? max_weight[a] * exposure_scale * liability_value[p] : Inf for
            a in 1:A
        ]
        growth[n] = _alm_best_growth(view(returns, :, n), caps, bound[p])
        bound[n] = growth[n] * bound[p] + inflow[n] - outflow[n]
    end
    weakest = 1 + argmin([bound[n] / liability_value[n] for n in 2:N])
    bound_ratio = bound[weakest] / liability_value[weakest]
    @assert bound_ratio >= witness_ratio - 1e-9

    target_ratio = rand(rng, Uniform(1.1, 1.3))
    shortfall_penalty = rand(rng, Uniform(2.0, 6.0))

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        funding_ratio = witness_ratio * rand(rng, Uniform(0.88, 0.97))
        leaves = [n for n in 1:N if stage[n] == T]
        shortfall = [max(0.0, target_ratio * liability_value[n] - wealth[n]) for n in leaves]
        witness = MultistageALMWitness(mix, holdings, buys, sells, shortfall, witness_ratio)
    elseif feasibility_status == infeasible
        funding_ratio = bound_ratio * rand(rng, Uniform(1.04, 1.10))
        path = Int[]
        n = weakest
        while n != 0
            pushfirst!(path, n)
            n = parent[n]
        end
        required = funding_ratio * liability_value[weakest]
        certificate = MultistageALMCertificate(
            path, growth[path], bound[path], required, required - bound[weakest]
        )
    else
        # The best nonanticipative policy sits well above the fixed mix, so
        # draw from the upper part of the [fixed-mix, clairvoyant] interval
        # (log scale) to keep both outcomes common.
        lo, hi = log(witness_ratio), log(1.03 * bound_ratio)
        funding_ratio = exp(lo + rand(rng, Uniform(0.45, 1.0)) * (hi - lo))
    end

    return MultistageALMProblem(
        A,
        T,
        [c.name for c in classes],
        [c.kind for c in classes],
        tc,
        max_weight,
        equity_cap,
        exposure_scale,
        parent,
        stage,
        probability,
        returns,
        inflow,
        outflow,
        liability_value,
        initial_holdings,
        funding_ratio,
        target_ratio,
        shortfall_penalty,
        stage_years,
        witness,
        certificate,
        feasibility_status,
    )
end

function build_model(prob::MultistageALMProblem)
    model = Model()
    A = prob.n_assets
    N = length(prob.parent)
    leaves = [n for n in 1:N if prob.stage[n] == prob.n_stages]
    equities = [a for a in 1:A if prob.asset_kind[a] == :equity]
    L = prob.liability_value

    # Exposure limits in liability units are plain upper bounds on holdings.
    exposure(a, n) =
        prob.max_weight[a] < 1.0 ? prob.max_weight[a] * prob.exposure_scale * L[n] : Inf
    @variable(model, 0 <= h[a = 1:A, n = 1:N] <= exposure(a, n))
    @variable(model, buy[2:A, 1:N] >= 0)
    @variable(model, sell[2:A, 1:N] >= 0)
    # Wealth; the regulatory funding floor is its lower bound at non-root nodes.
    @variable(model, wealth[n = 1:N] >= (n == 1 ? 0.0 : prob.funding_ratio * L[n]))
    @variable(model, shortfall[leaves] >= 0)

    @objective(
        model,
        Max,
        sum(
            prob.probability[n] * (wealth[n] - prob.shortfall_penalty * shortfall[n]) for
            n in leaves
        )
    )

    previous(a, n) =
        prob.parent[n] == 0 ? prob.initial_holdings[a] : prob.returns[a, n] * h[a, prob.parent[n]]

    @constraint(
        model, asset_balance[a = 2:A, n = 1:N], h[a, n] == previous(a, n) + buy[a, n] - sell[a, n]
    )
    @constraint(
        model,
        cash_balance[n = 1:N],
        h[1, n] ==
            previous(1, n) +
        sum(
            (1 - prob.transaction_cost[a]) * sell[a, n] -
            (1 + prob.transaction_cost[a]) * buy[a, n] for a in 2:A
        ) +
        prob.inflow[n] - prob.outflow[n]
    )
    @constraint(model, wealth_definition[n = 1:N], wealth[n] == sum(h[a, n] for a in 1:A))
    @constraint(
        model,
        equity_limit[n = 1:N],
        sum(h[a, n] for a in equities) <= prob.equity_cap * prob.exposure_scale * L[n]
    )
    @constraint(
        model, terminal_target[n in leaves], shortfall[n] + wealth[n] >= prob.target_ratio * L[n]
    )
    return model
end

register_variant(
    :stochastic_program,
    :multistage_alm,
    MultistageALMProblem,
    "Multistage stochastic asset-liability management LP of a pension fund on a scenario tree: nested rebalancing with transaction costs, allocation caps, inflation-indexed liabilities, and a regulatory funding floor at every node";
    tags=[:finance, :staircase],
)
