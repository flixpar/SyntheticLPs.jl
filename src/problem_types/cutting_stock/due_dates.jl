using JuMP
using Random

"""
    DueDatePlanWitness

Just-in-time single-item plan for a `feasible` instance: item `i` is cut with
the maximal single-item pattern `pattern[i]` exactly `usage[i, t]` times in
period `t`, producing only the shortfall left after the carried inventory.
Inventories stay nonnegative and every period's per-stock-type usage is within
that period's availability.
"""
struct DueDatePlanWitness
    pattern::Vector{Int}
    usage::Matrix{Int}
end

"""
    CumulativeShortageCertificate

Relaxation-valid infeasibility proof: summing the balance rows of every item
over periods `1..period` (inventory is nonnegative and starts at zero) shows
that the material due by `period`, `demand_length = sum_i len_i * sum_{s <= period} d_is`,
must be cut from the stock delivered by then, at most `supply_length =
sum_{s <= period} sum_k L_k * S_ks` millimetres — yet `demand_length >= 1.04 *
supply_length`. Multipliers `len_i` on those balance rows and `L_k` on those
availability rows; the proof spans `period * (n_items + n_stock)` rows.
"""
struct CumulativeShortageCertificate
    period::Int
    demand_length::Float64
    supply_length::Float64
end

"""
    DueDatesCuttingStockProblem <: ProblemGenerator

Multi-period cutting stock with due-date order buckets, inventory carryover,
and per-period stock deliveries.

# Overview

Order lengths, stock types and the pattern pool come from the shared
cutting-stock generators (`cs_piece_lengths`, `cs_stock_types`,
`cs_generate_patterns`). Every item has an order due in every period
(lumpy lognormal bucket sizes on a seasonal profile; a zero-demand balance row
would let presolve aggregate it away through the implied-free inventory). Any pattern may run in
any period (`x[j, t] >= 0`); surplus pieces are carried as inventory at a
holding cost proportional to their length; stock deliveries `S_kt` cap the bars
of each type cut per period.

```text
min  sum_{j,t} cost[stock[j]] x_jt + sum_{i,t} h_i inv_it
s.t. inv_{i,t-1} + sum_j a_ij x_jt - inv_it = d_it     for every item i, period t (inv_{i,0} = 0)
     sum_{j on stock k} x_jt <= S_kt                   for every stock type k, period t
```

Sizing: `T = clamp(round(2 log10 n), 4, 12)` periods (fewer for tiny targets),
`n_types = round(n / (16 T))`, `n_patterns = n ÷ T - n_types`; columns
`T * (n_patterns + n_types)`, within `T - 1` of the target (at tiny targets
whose one or two items admit too few distinct patterns, pattern columns are
traded one-for-one for extra item types, keeping that total). Rows
`T * (n_types + n_stock)` ≈ 6% of the columns.

# Feasibility

  - `feasible`: just-in-time single-item plan (`DueDatePlanWitness`);
    deliveries are its per-period usage times `U(1.05, 1.35)`, rounded up.
  - `infeasible`: the opening delivery is short (a delayed supplier
    shipment): period-1 stock is scaled so the material due in period 1
    exceeds it by `U(8%, 20%)` (`CumulativeShortageCertificate` with
    `period = 1`; nothing carries into period 1, so the proof combines every
    item's period-1 balance row with the period-1 availability rows). Later
    cut-off periods are equally valid certificates, but at 100k columns
    HiGHS's dual simplex intermittently failed to verify the resulting
    multi-period dual ray (status UNKNOWN, while IPM proves infeasibility), so
    the generator uses the robust first-period form.
  - `unknown`: deliveries proportional to the plan's usage, scaled per period
    to `U(0.95, 1.12) * U(0.85, 1.15)` times that period's due material; with
    carryover the outcome depends on the cumulative profile, decided by the LP.
"""
struct DueDatesCuttingStockProblem <: ProblemGenerator
    n_periods::Int
    stock_lengths::Vector{Int}
    stock_costs::Vector{Float64}
    piece_lengths::Vector{Int}
    demands::Matrix{Int}            # n_types x T
    holding_costs::Vector{Float64}
    patterns::CSPatterns
    availability::Matrix{Int}       # n_stock x T
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, DueDatePlanWitness}
    infeasibility_certificate::Union{Nothing, CumulativeShortageCertificate}
end

"""
    cs_due_dates_dimensions(n) -> (T, n_types, n_patterns, n_stock)
"""
function cs_due_dates_dimensions(n::Int)
    T = clamp(round(Int, 2 * log10(max(n, 1))), 4, 12)
    T = clamp(T, 1, max(1, n ÷ 2))
    per_period = max(2, n ÷ T)
    n_types = max(1, round(Int, n / (16 * T)), min(6, per_period ÷ 4))
    n_types = min(n_types, per_period - 1)
    n_patterns = max(1, per_period - n_types)
    n_stock = clamp(round(Int, log10(max(n, 1))) - 1, 1, 4)
    n_stock = clamp(n_stock, 1, max(1, n_patterns ÷ n_types))
    return T, n_types, n_patterns, n_stock
end

"""
    _cs_due_dates_pattern_space(stock_lengths, piece_lengths, cap) -> Int

Number of distinct single-item and two-item patterns (the space
`cs_generate_patterns`'s deterministic fallback enumerates), counted up to `cap`.
"""
function _cs_due_dates_pattern_space(
    stock_lengths::Vector{Int}, piece_lengths::Vector{Int}, cap::Int
)
    count = 0
    n = length(piece_lengths)
    for L in stock_lengths, i in 1:n
        count += L ÷ piece_lengths[i]
        for j in (i + 1):n, ci in 1:(L ÷ piece_lengths[i])
            count += (L - ci * piece_lengths[i]) ÷ piece_lengths[j]
            count >= cap && return count
        end
        count >= cap && return count
    end
    return count
end

function DueDatesCuttingStockProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))
    rng = MersenneTwister(seed)
    T, n_types, n_patterns, n_stock = cs_due_dates_dimensions(target_variables)

    stock_lengths, stock_costs = cs_stock_types(rng, n_stock)
    max_piece = floor(Int, 0.45 * stock_lengths[end])
    piece_lengths = cs_piece_lengths(rng, n_types, max_piece)
    # Tiny targets: one or two items on one stock can have fewer distinct
    # single/two-item patterns than requested (an item cut at most 4 times per
    # bar has only 4 single-item patterns). Trade pattern columns for item
    # types — each swap keeps `T * (n_patterns + n_types)` unchanged — until
    # the deterministic fallback enumeration is guaranteed to fill the pool.
    while _cs_due_dates_pattern_space(stock_lengths, piece_lengths, n_patterns) < n_patterns &&
          n_patterns - 1 >= (n_types + 1) * n_stock
        append!(piece_lengths, cs_piece_lengths(rng, 1, max_piece))
        n_types += 1
        n_patterns -= 1
    end
    # Fewer singles than patterns is guaranteed by the dimension rule only for
    # stocks every item fits; drop to singles-only generation if needed.
    patterns = cs_generate_patterns(rng, stock_lengths, piece_lengths, n_patterns)
    single = cs_single_index(patterns, stock_lengths, piece_lengths)

    # Order book: every item is due in every period (a zero-demand balance
    # row would make its inventory column implied-free and let presolve
    # aggregate the row away), with a seasonal profile and lumpy lognormal
    # bucket sizes.
    season = [1.0 + 0.3 * sin(2pi * (t + 3 * rand(rng)) / max(T, 2)) for t in 1:T]
    base = cs_demands(rng, n_types)
    demands = zeros(Int, n_types, T)
    for i in 1:n_types, t in 1:T
        demands[i, t] = max(1, round(Int, base[i] * season[t] * exp(0.6 * randn(rng) - 0.18) / 2))
    end
    # Holding cost per piece-period: ~1.5% of the material value per period.
    unit_mm_cost = minimum(stock_costs[k] / stock_lengths[k] for k in 1:n_stock)
    holding_costs = [
        0.015 * unit_mm_cost * piece_lengths[i] * (0.7 + 0.6 * rand(rng)) for i in 1:n_types
    ]

    # Just-in-time single-item plan.
    pattern = zeros(Int, n_types)
    stock_of = zeros(Int, n_types)
    for i in 1:n_types
        fits = [k for k in 1:n_stock if single[i, k] > 0]
        stock_of[i] = fits[rand(rng, 1:length(fits))]
        pattern[i] = single[i, stock_of[i]]
    end
    usage = zeros(Int, n_types, T)
    per_stock = zeros(Int, n_stock, T)
    for i in 1:n_types
        y = stock_lengths[stock_of[i]] ÷ piece_lengths[i]
        inv = 0
        for t in 1:T
            short = demands[i, t] - inv
            u = short > 0 ? cld(short, y) : 0
            usage[i, t] = u
            inv += u * y - demands[i, t]
            per_stock[stock_of[i], t] += u
        end
    end
    material = [sum(Float64(piece_lengths[i]) * demands[i, t] for i in 1:n_types) for t in 1:T]

    feasible_witness = nothing
    infeasibility_certificate = nothing
    availability = zeros(Int, n_stock, T)
    if feasibility_status == feasible
        for k in 1:n_stock, t in 1:T
            availability[k, t] = ceil(Int, per_stock[k, t] * (1.05 + 0.30 * rand(rng)))
        end
        feasible_witness = DueDatePlanWitness(pattern, usage)
    elseif feasibility_status == infeasible
        for k in 1:n_stock, t in 1:T
            availability[k, t] = ceil(Int, per_stock[k, t] * (1.05 + 0.30 * rand(rng)))
        end
        tstar = 1   # see the docstring: robust ray verification at scale
        due = sum(material[1:tstar])
        supply = sum(
            Float64(stock_lengths[k]) * availability[k, t] for k in 1:n_stock, t in 1:tstar
        )
        scale = due / ((1.08 + 0.12 * rand(rng)) * supply)
        for k in 1:n_stock, t in 1:tstar
            availability[k, t] = floor(Int, availability[k, t] * scale)
        end
        supply = sum(
            Float64(stock_lengths[k]) * availability[k, t] for k in 1:n_stock, t in 1:tstar
        )
        due >= 1.04 * supply || error("due_dates: cumulative certificate lost its margin")
        infeasibility_certificate = CumulativeShortageCertificate(tstar, due, supply)
    else
        rho = 0.95 + 0.17 * rand(rng)
        for t in 1:T
            weights = [per_stock[k, t] + 0.5 for k in 1:n_stock]
            wl = sum(stock_lengths[k] * weights[k] for k in 1:n_stock)
            target_len = rho * (0.85 + 0.30 * rand(rng)) * material[t]
            for k in 1:n_stock
                availability[k, t] = round(Int, weights[k] * target_len / wl)
            end
        end
    end

    return DueDatesCuttingStockProblem(
        T,
        stock_lengths,
        stock_costs,
        piece_lengths,
        demands,
        holding_costs,
        patterns,
        availability,
        feasibility_status,
        feasible_witness,
        infeasibility_certificate,
    )
end

function build_model(prob::DueDatesCuttingStockProblem)
    model = Model()
    pats = prob.patterns
    P = length(pats)
    T = prob.n_periods
    m = length(prob.piece_lengths)
    S = length(prob.stock_lengths)

    @variable(model, x[1:P, 1:T] >= 0)
    @variable(model, inventory[1:m, 1:T] >= 0)
    @objective(
        model,
        Min,
        sum(prob.stock_costs[pats.stock[j]] * x[j, t] for j in 1:P, t in 1:T) +
            sum(prob.holding_costs[i] * inventory[i, t] for i in 1:m, t in 1:T)
    )

    for t in 1:T
        produced = [AffExpr() for _ in 1:m]
        used = [AffExpr() for _ in 1:S]
        for j in 1:P
            for (i, c) in zip(pats.items[j], pats.counts[j])
                add_to_expression!(produced[i], c, x[j, t])
            end
            add_to_expression!(used[pats.stock[j]], 1.0, x[j, t])
        end
        for i in 1:m
            add_to_expression!(produced[i], -1.0, inventory[i, t])
            t > 1 && add_to_expression!(produced[i], 1.0, inventory[i, t - 1])
            @constraint(model, produced[i] == prob.demands[i, t])
        end
        for k in 1:S
            isempty(used[k].terms) && continue
            @constraint(model, used[k] <= prob.availability[k, t])
        end
    end
    return model
end

register_variant(
    :cutting_stock,
    :due_dates,
    DueDatesCuttingStockProblem,
    "Multi-period cutting stock with due-date order buckets, inventory carryover at a holding cost, and per-period stock deliveries";
    tags=[:production, :staircase],
)
