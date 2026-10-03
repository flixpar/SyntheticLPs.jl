using JuMP
using Random
using Distributions

const INVENTORY_MDP_WEEKLY_CYCLES = (1, 2, 4, 13, 26, 52)
const INVENTORY_MDP_DAILY_CYCLES = (7, 14, 28, 91, 182, 364)
const INVENTORY_MDP_STREAMS = [:shortage, :on_hand, :orders]

"""
    InventoryControlMDP <: AbstractMDPProblem

Seasonal joint pricing-and-replenishment MDP for a single item with
backlogging and a replenishment lead time (Federgruen & Heching 1999; Song &
Zipkin 1993 for the seasonal / Markov-modulated demand), written as the
occupation-measure LP.

# State and actions

State `(i, o, e)`: net inventory `i ∈ -max_backlog:max_inventory` (negative =
backlog, in lots), the pipeline `o = (o_1, ..., o_ℓ)` of orders placed in the
last `lead_time = ℓ ∈ {0, 1, 2}` periods (`o_1` arrives next), and the seasonal
phase `e ∈ 1:n_phases` of a weekly (`n_phases ∈ {1,2,4,13,26,52}`) or daily
(`{7,...,364}`, with a weekday pattern) review cycle. The inventory position
`i + Σo` never exceeds `max_inventory` (the storage / position cap), so the
state set is `{(i, o) : i >= -max_backlog, 0 <= o_t <= max_order,
i + Σo <= max_inventory}` per phase; `state_i`, `state_pipeline`, and
`state_phase` decode a state index.

Action `(q, j)`: order `q ∈ 0:max_order` lots subject to the position cap
`i + Σo + q <= max_inventory`, and charge price level `j`
(`price_multipliers[j] * base_price`, at most three levels). Pairs of a state
are ordered `q`-major, so pair `(q, j)` of state `s` is
`state_ptr[s] + q * P + (j - 1)`; `action_label = 100q + j`.

# Dynamics and costs

Demand in phase `e` at price level `j` is negative binomial with mean
`phase_demand_mean[e] * price_multipliers[j]^(-elasticity)` and dispersion
`dispersion` (upper tail below 1e-4 lumped). The stock available against
demand is `a = i + q` when `ℓ = 0` (immediate delivery) and `a = i` otherwise;
then `i' = max(a - D, -max_backlog) (+ o_1 if ℓ >= 1)` — backlog beyond the
limit is lost — the pipeline shifts (`o' = (o_2, ..., q)`), and the phase
advances cyclically. Per-period cost: `fixed_order_cost * [q > 0] +
unit_cost * q + holding_cost * E[(a-D)^+] + backorder_cost * E[backlog'] +
lost_sale_penalty * E[lost] - price_j * (E[D] - E[lost])` (negative profit).

Secondary streams: `:shortage = E[(D - a^+)^+]` (units not served from stock),
`:on_hand = E[(a - D)^+]`, `:orders = [q > 0]`. The position cap is
deliberately tight against peak lead-time demand (`max_inventory ≈ 1.1-1.6 ×
(ℓ+1) ×` peak mean), so some shortage is unavoidable under every policy — which
is what makes a shortage budget refutable with a real margin.

# Feasibility

The unconstrained LP is always feasible and bounded. With probability 1/2
(always for `infeasible`) one service-level row `Σ shortage·x <= B` is added
(`budget_streams == [1]`): `feasible` puts `B` above the shortage of the
service-oriented reference policy (a base-stock policy on the inventory
position; the stored witness), `infeasible` 8-25% below the certified minimum
shortage over all policies (the `MDPDualCertificate`), `unknown` on either side
of that minimum. See `_mdp_plant_budgets`.
"""
struct InventoryControlMDP <: AbstractMDPProblem
    review::Symbol
    n_phases::Int
    lead_time::Int
    max_inventory::Int
    max_backlog::Int
    max_order::Int
    state_i::Vector{Int}
    state_pipeline::Vector{Vector{Int}}
    state_phase::Vector{Int}
    price_multipliers::Vector{Float64}
    phase_demand_mean::Vector{Float64}
    elasticity::Float64
    dispersion::Float64
    unit_cost::Float64
    base_price::Float64
    holding_cost::Float64
    backorder_cost::Float64
    lost_sale_penalty::Float64
    fixed_order_cost::Float64
    mdp::MDPData
    criterion::Symbol
    discount::Float64
    rhs::Vector{Float64}
    normalization::Float64
    budget_streams::Vector{Int}
    budgets::Vector{Float64}
    feasible_witness::Union{Nothing, MDPOccupationWitness}
    infeasibility_certificate::Union{Nothing, MDPDualCertificate}
    feasibility_status::FeasibilityStatus
end

"""Per-phase states `(i, pipeline)` with `i + Σ pipeline <= I`, in enumeration order."""
function _inventory_mdp_phase_states(ℓ::Int, I::Int, B::Int, Q::Int)
    states = Tuple{Int, Vector{Int}}[]
    if ℓ == 0
        for i in (-B):I
            push!(states, (i, Int[]))
        end
    elseif ℓ == 1
        for o1 in 0:Q, i in (-B):(I - o1)
            push!(states, (i, [o1]))
        end
    else
        for o1 in 0:Q, o2 in 0:Q, i in (-B):(I - o1 - o2)
            push!(states, (i, [o1, o2]))
        end
    end
    return states
end

"""Exact state-action pair count of the inventory MDP (without building it)."""
function _inventory_mdp_pairs(ℓ::Int, I::Int, B::Int, Q::Int, P::Int, E::Int)
    total = 0
    if ℓ == 0
        for i in (-B):I
            total += min(Q, I - i) + 1
        end
    elseif ℓ == 1
        for o1 in 0:Q, i in (-B):(I - o1)
            total += min(Q, I - i - o1) + 1
        end
    else
        for o1 in 0:Q, o2 in 0:Q, i in (-B):(I - o1 - o2)
            total += min(Q, I - i - o1 - o2) + 1
        end
    end
    return E * P * total
end

"""Integer dimensions implied by a peak mean demand `μ` (lots) and the shape ratios."""
function _inventory_mdp_dims(μ::Float64, ℓ::Int, tightness::Float64, q_ratio::Float64, b_ratio::Float64)
    I = max(2, round(Int, tightness * (ℓ + 1) * μ))
    Q = max(1, round(Int, q_ratio * μ))
    B = max(1, round(Int, b_ratio * μ))
    return I, B, Q
end

"""
    _inventory_mdp_size(target, prefs...) -> (ℓ, E, P, μ)

Grid search over lead time, cycle length, price-menu size, and peak demand for
the exact pair count closest to `target`, mildly preferring the sampled lead
time, review cadence, price-menu size, and demand scale (so the structure
varies across seeds while the count stays within a few percent).
"""
function _inventory_mdp_size(
    target::Int,
    review::Symbol,
    ℓ_pref::Int,
    P_pref::Int,
    μ_pref::Float64,
    tightness::Float64,
    q_ratio::Float64,
    b_ratio::Float64,
)
    best = (Inf, 0, 1, 1, 1.0)
    for ℓ in 0:2, (cycles, pen) in (
        (INVENTORY_MDP_WEEKLY_CYCLES, review == :weekly ? 0.0 : 0.04),
        (INVENTORY_MDP_DAILY_CYCLES, review == :daily ? 0.0 : 0.04),
    )
        for μ in 0.5:0.25:8.0
            I, B, Q = _inventory_mdp_dims(μ, ℓ, tightness, q_ratio, b_ratio)
            base = _inventory_mdp_pairs(ℓ, I, B, Q, 1, 1)
            nst = length(_inventory_mdp_phase_states(ℓ, I, B, Q))
            for E in cycles, P in 1:3
                n = base * E * P
                # Keep the LP from turning wide-and-thin: penalize more than
                # ~16 actions (columns) per state (balance row).
                width = max(0.0, base * P / nst - 16.0) / 16.0
                score =
                    abs(n - target) / target +
                    pen +
                    0.1 * width +
                    0.04 * abs(ℓ - ℓ_pref) +
                    0.03 * abs(P - P_pref) +
                    0.03 * abs(log(μ / μ_pref))
                if score < best[1]
                    best = (score, ℓ, E, P, μ)
                end
            end
        end
    end
    return best[2], best[3], best[4], best[5]
end

"""
    InventoryControlMDP(target_variables, feasibility_status, seed)

Sample a seasonal pricing-and-replenishment MDP with about `target_variables`
state-action pairs (exact count `_inventory_mdp_pairs`, typically within a few
percent of the target). Targets above `MDP_MAX_PAIRS` raise an
`ArgumentError`.
"""
function InventoryControlMDP(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
    service_row::Union{Nothing, Bool}=nothing,
)
    _mdp_check_target(target_variables, "inventory_control")
    rng = MersenneTwister(seed)

    review = rand(rng) < 0.4 ? :daily : :weekly
    ℓ_pref = let u = rand(rng)
        u < 0.25 ? 0 : (u < 0.7 ? 1 : 2)
    end
    P_pref = rand(rng, 1:3)
    μ_pref = clamp(target_variables^0.15, 1.0, 6.0) * (0.8 + 0.4 * rand(rng))
    tightness = 1.1 + 0.5 * rand(rng)          # position cap / peak lead-time demand
    q_ratio = 1.2 + 1.0 * rand(rng)            # largest order / peak mean demand
    b_ratio = 0.8 + 1.2 * rand(rng)            # backlog limit / peak mean demand
    ℓ, E, P, peak_mean = _inventory_mdp_size(
        target_variables, review, ℓ_pref, P_pref, μ_pref, tightness, q_ratio, b_ratio
    )
    I, B, Q = _inventory_mdp_dims(peak_mean, ℓ, tightness, q_ratio, b_ratio)
    review = E in INVENTORY_MDP_DAILY_CYCLES ? :daily : :weekly
    periods_per_year = review == :daily ? 364 : 52

    # Demand: seasonal profile (plus a weekday pattern under daily review)
    # peaking at `peak_mean` lots per period at the base price.
    amp = 0.15 + 0.35 * rand(rng)
    shift = rand(rng)
    weekday = [0.85, 0.9, 0.95, 1.0, 1.1, 1.3, 0.9] .* (0.9 .+ 0.2 .* rand(rng, 7))
    weekday ./= sum(weekday) / 7
    profile = map(1:E) do e
        f = E == 1 ? 1.0 : 1.0 + amp * sin(2π * ((e - 1) / E - shift))
        review == :daily ? f * weekday[mod1(e, 7)] : f
    end
    phase_mean = peak_mean .* profile ./ maximum(profile)
    elasticity = 1.2 + 1.0 * rand(rng)
    dispersion = 3.0 + 9.0 * rand(rng)
    rho = P == 1 ? [1.0] : collect(range(0.88, 1.12; length=P))

    unit_cost = 5.0 + 75.0 * rand(rng)
    base_price = unit_cost * (1.3 + 0.7 * rand(rng))
    holding = unit_cost * (0.15 + 0.2 * rand(rng)) / periods_per_year + 0.002 * unit_cost
    backorder = base_price * (0.05 + 0.2 * rand(rng))
    lost_pen = base_price * (0.3 + 0.7 * rand(rng))
    fixed = unit_cost * peak_mean * (0.2 + 1.0 * rand(rng))
    service_level = 0.9 + 0.08 * rand(rng)
    criterion, discount = _mdp_criterion(rng, (20.0, 400.0))
    has_service_row = rand(rng) < 0.5 || feasibility_status == infeasible
    service_row === nothing || (has_service_row = service_row)

    # State table and index map.
    pstates = _inventory_mdp_phase_states(ℓ, I, B, Q)
    Sp = length(pstates)
    L = I + B + 1
    local_idx = zeros(Int, L, ℓ >= 1 ? Q + 1 : 1, ℓ >= 2 ? Q + 1 : 1)
    for (r, (i, o)) in enumerate(pstates)
        local_idx[i + B + 1, (ℓ >= 1 ? o[1] + 1 : 1), (ℓ >= 2 ? o[2] + 1 : 1)] = r
    end
    state_index(e, i, o1, o2) = (e - 1) * Sp + local_idx[i + B + 1, o1 + 1, o2 + 1]

    # Demand outcome per (phase, price, available stock a): post-demand level
    # distribution (before any pipeline arrival) and expectations. Cached
    # because many pairs share (e, j, a).
    pmfs = [
        _mdp_truncated_pmf(_mdp_negbin(phase_mean[e] * rho[j]^(-elasticity), dispersion), 1e-4) for
        e in 1:E, j in 1:P
    ]
    mid_lv = Array{Vector{Int}}(undef, E, P, L)
    mid_pr = Array{Vector{Float64}}(undef, E, P, L)
    expct = Array{NTuple{5, Float64}}(undef, E, P, L)   # on_hand, backlog', lost, shortage, sold
    for e in 1:E, j in 1:P, ai in 1:L
        a = ai - B - 1
        lv = Int[]
        pr = Float64[]
        oh = bl = lo = sh = sold = 0.0
        for (di, p) in enumerate(pmfs[e, j])
            d = di - 1
            i2 = max(a - d, -B)
            if !isempty(lv) && lv[end] == i2
                pr[end] += p
            else
                push!(lv, i2)
                push!(pr, p)
            end
            lost = max(d - a - B, 0)
            oh += p * max(a - d, 0)
            bl += p * min(max(d - a, 0), B)
            lo += p * lost
            sh += p * max(d - max(a, 0), 0)
            sold += p * (d - lost)
        end
        mid_lv[e, j, ai] = lv
        mid_pr[e, j, ai] = pr ./ sum(pr)
        expct[e, j, ai] = (oh, bl, lo, sh, sold)
    end

    # Service-oriented reference policy at the price level nearest the base
    # price: base-stock on the inventory position, ordering up to S_e (the
    # service-level quantile of (ℓ+1)-period demand, capped by the position
    # cap) whenever the position is below it.
    jref = argmin(abs.(rho .- 1.0))
    S_level = [
        clamp(
            round(Int, quantile(_mdp_negbin((ℓ + 1) * phase_mean[e], (ℓ + 1) * dispersion), service_level)),
            1,
            I,
        ) for e in 1:E
    ]

    n_est = _inventory_mdp_pairs(ℓ, I, B, Q, P, E)
    b = MDPBuilder(3; sizehint=n_est, nnzhint=n_est * 8)
    ref = Vector{Int}(undef, E * Sp)
    state_i = Vector{Int}(undef, E * Sp)
    state_pipeline = Vector{Vector{Int}}(undef, E * Sp)
    state_phase = Vector{Int}(undef, E * Sp)
    for e in 1:E, (r, (i, o)) in enumerate(pstates)
        s = (e - 1) * Sp + r
        state_i[s], state_pipeline[s], state_phase[s] = i, o, e
        _mdp_begin_state!(b)
        e2 = mod1(e + 1, E)
        position = i + sum(o; init=0)
        qmax = min(Q, I - position)
        qref = position < S_level[e] ? min(S_level[e] - position, qmax) : 0
        for q in 0:qmax, j in 1:P
            a = ℓ == 0 ? i + q : i
            ai = a + B + 1
            arrival = ℓ >= 1 ? o[1] : 0
            o1n, o2n = ℓ == 0 ? (0, 0) : (ℓ == 1 ? (q, 0) : (o[2], q))
            for (lv, p) in zip(mid_lv[e, j, ai], mid_pr[e, j, ai])
                _mdp_add_succ!(b, state_index(e2, lv + arrival, o1n, o2n), p)
            end
            oh, bl, lo, sh, sold = expct[e, j, ai]
            c =
                fixed * (q > 0) + unit_cost * q + holding * oh + backorder * bl + lost_pen * lo -
                base_price * rho[j] * sold
            k = _mdp_end_pair!(b, c, (sh, oh, q > 0 ? 1.0 : 0.0), 100q + j)
            if q == qref && j == jref
                ref[s] = k
            end
        end
    end
    mdp = _mdp_finish(b, copy(INVENTORY_MDP_STREAMS), ref)

    # Typical start: beginning of the cycle, empty pipeline, modest stock.
    start = zeros(E * Sp)
    for (r, (i, o)) in enumerate(pstates)
        if all(==(0), o) && 0 <= i <= S_level[1]
            start[r] = 1.0
        end
    end
    rhs, N = _mdp_rhs(criterion, discount, start)
    budget_streams = has_service_row ? [1] : Int[]
    budgets, witness, certificate = _mdp_plant_budgets(
        rng, mdp, criterion, discount, rhs, N, feasibility_status, budget_streams
    )

    return InventoryControlMDP(
        review,
        E,
        ℓ,
        I,
        B,
        Q,
        state_i,
        state_pipeline,
        state_phase,
        rho,
        phase_mean,
        elasticity,
        dispersion,
        unit_cost,
        base_price,
        holding,
        backorder,
        lost_pen,
        fixed,
        mdp,
        criterion,
        discount,
        rhs,
        N,
        budget_streams,
        budgets,
        witness,
        certificate,
        feasibility_status,
    )
end

register_variant(
    :markov_decision_process,
    :inventory_control,
    InventoryControlMDP,
    "Occupation-measure LP of a seasonal joint pricing-and-replenishment MDP (lead-time pipeline, backlogging, position cap, negative-binomial demand, fixed ordering cost) under discounted or average cost, with an optional shortage service-level row refuted by a value-function Farkas certificate",
    default=true,
)
