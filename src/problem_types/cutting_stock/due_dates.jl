using JuMP
using Random
using Distributions
using StatsBase

"""
    DueDatesCuttingStockProblem <: ProblemGenerator

Generator for multi-period (time-phased) cutting stock / lot-sizing problems with
per-period due demand, inventory carryover, and a per-period stock cap.

# Overview
Models one-dimensional cutting stock spread over a planning horizon. In each
period the mill may run cutting patterns (limited by a per-period stock cap), the
cut pieces satisfy that period's due demand, and any surplus is carried forward as
inventory at a per-piece holding cost. The decisions are the continuous pattern
usage counts `x[j, t]` per period and the on-hand inventory `inventory[i, t]` of
each piece type at the end of each period (with an initial inventory level
`inventory[i, 0]`). The objective minimizes total stock pieces used plus the cost
of holding inventory between periods, so the temporal structure is non-degenerate:
producing early to cover a later due date incurs a real holding penalty.

Inventory balance for each piece `i` and period `t`:
`inventory[i, t-1] + (pieces cut in period t) == period_demands[i, t] + inventory[i, t]`.

# Fields
- `piece_lengths::Vector{Float64}`: Length of each piece type required
- `patterns::Vector{Vector{Int}}`: Cutting patterns (how many of each piece per stock)
- `stock_length::Float64`: Length of stock material
- `n_periods::Int`: Number of planning periods
- `period_demands::Matrix{Int}`: Due demand per piece type (rows) and period (cols)
- `holding_costs::Vector{Float64}`: Per-period holding cost per unit of each piece type
- `period_stock_cap::Int`: Maximum number of stock pieces that can be cut in any single period
"""
struct DueDatesCuttingStockProblem <: ProblemGenerator
    piece_lengths::Vector{Float64}
    patterns::Vector{Vector{Int}}
    stock_length::Float64
    n_periods::Int
    period_demands::Matrix{Int}
    holding_costs::Vector{Float64}
    period_stock_cap::Int
end

"""
Helper: generate feasible single-piece and mixed cutting patterns for a stock length.
(Self-contained name to avoid clashing with the standard variant's helper.)
"""
function generate_due_dates_patterns(standard_length, piece_lengths, max_patterns, waste_factor=0.1)
    patterns = Vector{Vector{Int}}()

    # Single-piece patterns (guarantee every piece type is producible)
    for (i, piece_length) in enumerate(piece_lengths)
        pattern = zeros(Int, length(piece_lengths))
        pattern[i] = max(1, floor(Int, standard_length / piece_length))
        push!(patterns, pattern)
    end

    # Complex mixed patterns (genuine, distinct combinations). Use a generous
    # attempt budget so the genuine pattern space is explored thoroughly before
    # any fallback padding is needed.
    attempts = 0
    max_attempts = max_patterns * 40

    while length(patterns) < max_patterns && attempts < max_attempts
        attempts += 1

        new_pattern = zeros(Int, length(piece_lengths))
        remaining_length = standard_length
        indices = collect(1:length(piece_lengths))

        num_types_to_use = min(length(piece_lengths),
                               max(1, round(Int, rand(Exponential(2.0)))))

        selected_indices = sample(indices, num_types_to_use, replace=false)

        while !isempty(selected_indices)
            weights = [standard_length / piece_lengths[i] for i in selected_indices]
            idx = sample(selected_indices, Weights(weights))

            if piece_lengths[idx] <= remaining_length
                new_pattern[idx] += 1
                remaining_length -= piece_lengths[idx]

                if remaining_length / standard_length <= waste_factor
                    break
                end
            else
                filter!(i -> piece_lengths[i] <= remaining_length, selected_indices)
            end
        end

        if sum(new_pattern) > 0 && !(new_pattern in patterns)
            push!(patterns, new_pattern)
        end
    end

    # If the genuine mixed-pattern space is exhausted before reaching the requested
    # count (only happens when there are few piece types), top up with DISTINCT
    # reduced-yield single-piece patterns — a piece cut at fewer than the maximum
    # per stock. These are valid, non-duplicate columns (a real mill does make
    # sub-maximal cuts); they are economically dominated but add genuine cutting
    # options rather than exact-duplicate filler. If even the distinct sub-maximal
    # space is exhausted, fewer patterns are returned and the caller adapts.
    if length(patterns) < max_patterns
        for i in 1:length(piece_lengths)
            max_fit = max(1, floor(Int, standard_length / piece_lengths[i]))
            for count in (max_fit - 1):-1:1
                length(patterns) >= max_patterns && break
                pat = zeros(Int, length(piece_lengths))
                pat[i] = count
                pat in patterns || push!(patterns, pat)
            end
            length(patterns) >= max_patterns && break
        end
    end

    return patterns
end

"""
    DueDatesCuttingStockProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-period cutting stock problem instance with due dates.

# Variable count
The JuMP model creates `x[1:n_patterns, 1:n_periods]` and
`inventory[1:n_pieces, 0:n_periods]`, so:
`var_count = n_patterns * n_periods + n_pieces * (n_periods + 1)`.
Dimensions are sized in the constructor to hit `target_variables`.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function DueDatesCuttingStockProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Scale-dependent parameters ---
    if target_variables <= 250
        n_periods = rand(3:5)
        stock_length = rand(Uniform(3.0, 8.0))
        demand_min = rand(5:20)
        demand_max = rand(40:120)
        common_lengths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        common_length_prob = rand(Uniform(0.3, 0.6))
        waste_factor = rand(Uniform(0.05, 0.15))
    elseif target_variables <= 1000
        n_periods = rand(4:7)
        stock_length = rand(Uniform(6.0, 12.0))
        demand_min = rand(20:80)
        demand_max = rand(150:600)
        common_lengths = [1.0, 1.2, 1.5, 2.0, 2.4, 3.0, 4.0, 6.0]
        common_length_prob = rand(Uniform(0.4, 0.7))
        waste_factor = rand(Uniform(0.03, 0.10))
    else
        n_periods = rand(6:10)
        stock_length = rand(Uniform(8.0, 20.0))
        demand_min = rand(100:400)
        demand_max = rand(800:5000)
        common_lengths = [1.0, 1.2, 1.5, 2.0, 2.4, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        common_length_prob = rand(Uniform(0.5, 0.8))
        waste_factor = rand(Uniform(0.02, 0.08))
    end

    # --- Dimension sizing to hit target ---
    # var_count = n_patterns * n_periods + n_pieces * (n_periods + 1)
    # Choose n_pieces, then set n_patterns so the total lands near target.
    # n_patterns >= n_pieces (single-piece patterns are always created).
    n_pieces = clamp(round(Int, sqrt(target_variables) / 1.5), 3, 60)

    # Reserve the inventory variables, then split the rest into x[j,t].
    inv_vars = n_pieces * (n_periods + 1)
    remaining = target_variables - inv_vars
    n_patterns = max(n_pieces, round(Int, remaining / n_periods))

    # --- Generate piece lengths ---
    piece_lengths = Float64[]
    effective_max_length = min(stock_length * 0.95, stock_length - 0.1)

    for _ in 1:n_pieces
        if rand() < common_length_prob && !isempty(common_lengths)
            base_length = rand(common_lengths)
            if base_length > effective_max_length
                valid_lengths = filter(x -> x <= effective_max_length, common_lengths)
                base_length = isempty(valid_lengths) ? effective_max_length * 0.8 : rand(valid_lengths)
            end
            variation = rand(Normal(0, 0.02))
            len = clamp(base_length + variation, 0.1, effective_max_length)
            push!(piece_lengths, round(len, digits=2))
        else
            normalized = rand(Beta(2.0, 3.0))
            len = 0.1 + (effective_max_length - 0.1) * normalized
            precision = stock_length > 10 ? 0.1 : 0.05
            len = round(len / precision) * precision
            push!(piece_lengths, len)
        end
    end

    unique!(piece_lengths)
    n_pieces = length(piece_lengths)

    # Recompute n_patterns after de-duplication so the var count stays on target.
    inv_vars = n_pieces * (n_periods + 1)
    remaining = target_variables - inv_vars
    n_patterns = max(n_pieces, round(Int, remaining / n_periods))

    # --- Generate cutting patterns ---
    patterns = generate_due_dates_patterns(stock_length, piece_lengths, n_patterns, waste_factor)
    n_patterns = length(patterns)

    # --- Holding costs (small relative to a stock piece; keeps temporal structure
    #     active but not dominant) ---
    holding_costs = [rand(Uniform(0.01, 0.1)) for _ in 1:n_pieces]

    # --- Best single-period throughput per piece (pieces cut per stock used) ---
    best_eff = zeros(Int, n_pieces)
    for i in 1:n_pieces
        e = 0
        for p in patterns
            e = max(e, p[i])
        end
        best_eff[i] = max(e, max(1, floor(Int, stock_length / piece_lengths[i])))
    end

    # --- Base per-period due demand ---
    period_demands = zeros(Int, n_pieces, n_periods)
    for i in 1:n_pieces
        for t in 1:n_periods
            period_demands[i, t] = rand(demand_min:demand_max)
        end
    end

    # --- Feasibility handling ---
    actual_status = feasibility_status
    if feasibility_status == unknown
        actual_status = rand() < 0.6 ? feasible : infeasible
    end

    # A loose, safe per-period stock cap: enough to cut everything due in any one
    # period for every piece, with slack. Used as the baseline for both feasible
    # and infeasible constructions.
    function max_stocks_for_period(t)
        total = 0
        for i in 1:n_pieces
            total += ceil(Int, period_demands[i, t] / best_eff[i])
        end
        return total
    end
    baseline_cap = maximum(max_stocks_for_period(t) for t in 1:n_periods)

    if actual_status == feasible
        # Generous cap so demand is always producible within its due period.
        # This also guards against a late-period demand spike vs the per-period cap:
        # the cap covers the worst single period plus a comfortable margin.
        period_stock_cap = ceil(Int, baseline_cap * rand(Uniform(1.5, 2.5))) + n_pieces

    elseif actual_status == infeasible
        # REAL infeasibility: an early-period due demand that cannot be met even by
        # cutting at the per-period cap, with no prior inventory to draw on.
        # Initial inventory is 0, so period-1 due demand must be produced in period 1.
        # We pick a target piece, set its period-1 due demand to exceed the most it
        # could be produced under the cap (with margin), and make the cap binding.

        # Cap allows producing at most `cap` stocks in period 1. Best per-stock yield
        # of the target piece is best_eff[target]. So max producible in period 1 is
        # cap * best_eff[target]. Force period_demands[target, 1] above that.
        target_piece = argmin(piece_lengths)  # smallest piece -> highest yield, but we beat it

        # Set a modest, binding per-period cap.
        period_stock_cap = max(1, ceil(Int, baseline_cap * rand(Uniform(0.4, 0.7))))

        max_producible_p1 = period_stock_cap * best_eff[target_piece]
        margin = max(1, round(Int, max_producible_p1 * rand(Uniform(0.3, 0.8))))
        period_demands[target_piece, 1] = max_producible_p1 + margin
    else
        # Should not reach here, but keep a sane default.
        period_stock_cap = ceil(Int, baseline_cap * 2.0) + n_pieces
    end

    return DueDatesCuttingStockProblem(
        piece_lengths, patterns, stock_length, n_periods,
        period_demands, holding_costs, period_stock_cap,
    )
end

"""
    build_model(prob::DueDatesCuttingStockProblem)

Build a JuMP model for the multi-period cutting stock problem with due dates.
Deterministic — uses only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::DueDatesCuttingStockProblem)
    model = Model()

    n_patterns = length(prob.patterns)
    n_pieces = length(prob.piece_lengths)
    n_periods = prob.n_periods

    # Variables
    # var_count = n_patterns * n_periods + n_pieces * (n_periods + 1)
    @variable(model, x[1:n_patterns, 1:n_periods] >= 0)          # pattern usage per period
    @variable(model, inventory[1:n_pieces, 0:n_periods] >= 0)    # end-of-period inventory

    # Objective: total stock pieces used + inventory holding cost
    @objective(model, Min,
        sum(x[j, t] for j in 1:n_patterns, t in 1:n_periods) +
        sum(prob.holding_costs[i] * inventory[i, t] for i in 1:n_pieces, t in 1:n_periods)
    )

    # Initial inventory is zero
    for i in 1:n_pieces
        @constraint(model, inventory[i, 0] == 0)
    end

    # Inventory balance + due-demand satisfaction per piece and period
    for i in 1:n_pieces
        for t in 1:n_periods
            production = sum(prob.patterns[j][i] * x[j, t] for j in 1:n_patterns)
            @constraint(model,
                inventory[i, t-1] + production == prob.period_demands[i, t] + inventory[i, t])
        end
    end

    # Per-period stock cap
    for t in 1:n_periods
        @constraint(model, sum(x[j, t] for j in 1:n_patterns) <= prob.period_stock_cap)
    end

    return model
end

# Register the variant
register_variant(
    :cutting_stock,
    :due_dates,
    DueDatesCuttingStockProblem,
    "Multi-period cutting stock with per-period due demand, inventory carryover at a holding cost, and a per-period stock cap",
)
