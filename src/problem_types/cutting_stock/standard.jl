using JuMP
using Random

"""
    StockPlanWitness

Integer operating plan for a `feasible` instance: item `i` is cut with the
maximal single-item pattern `pattern[i]` (on a stock type it fits) exactly
`usage[i] = cld(demand[i], yield)` times. Production covers every demand in
exact integer arithmetic and the plan's per-stock-type usage is within the
availabilities, so the point is feasible even for the integer problem.
"""
struct StockPlanWitness
    pattern::Vector{Int}
    usage::Vector{Int}
end

"""
    CuttingStockProblem <: ProblemGenerator

Multi-stock one-dimensional cutting stock: the Gilmore-Gomory master LP over a
large pool of generated patterns, with many order lengths and several stock-bar
lengths in limited supply.

# Overview

Items are integer-millimetre order lengths (`cs_piece_lengths`, up to 45% of
the longest bar) with lognormal order quantities; stock types are 1-5 bar
lengths from a 6-13.5 m catalogue priced per millimetre with a long-bar
discount. Patterns come from `cs_generate_patterns`: the maximal single-item
pattern of every (item, stock) pair that fits, then knapsack-like greedy fills
of 2-6 random items with a longest-first top-up (low trim loss) — the kind of
columns column generation produces. One continuous variable per pattern.

```text
min  sum_j cost[stock[j]] * x_j
s.t. sum_j a_ij x_j >= d_i                 for every item type i
     sum_{j on stock k} x_j <= S_k         for every stock type k
     x >= 0
```

Sizing: exactly `target_variables` patterns; `n_stock = clamp(round(log10 n),
1, 5)` (at most `n / 4`), `n_types = clamp(round(n / 20), 1, n / n_stock)` (a few more for tiny
targets), so
rows `n_types + n_stock` grow linearly (~5% of the columns) and every column
has 2-7 nonzeros. Generation is near-linear (hash-set deduplication).

# Feasibility

  - `feasible`: a single-item plan (each item on a random stock type it fits)
    is stored as `StockPlanWitness`; availabilities are its per-type usage
    times `U(1.05, 1.35)` (rounded up). Mixed patterns are cheaper per piece,
    so the LP optimum uses fewer bars and the cheap stock types bind.
  - `infeasible`: the same availabilities scaled down so the ordered material
    exceeds the total stock length by `U(8%, 20%)`
    (`MaterialShortageCertificate`, a Farkas combination of every row).
  - `unknown`: availabilities proportional to the single-item plan, scaled so
    total stock length is `U(0.97, 1.10)` times the ordered material — around
    the trim-loss threshold the best patterns can reach, so the LP decides.
"""
struct CuttingStockProblem <: ProblemGenerator
    stock_lengths::Vector{Int}
    stock_costs::Vector{Float64}
    piece_lengths::Vector{Int}
    demands::Vector{Int}
    patterns::CSPatterns
    availability::Vector{Int}
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, StockPlanWitness}
    infeasibility_certificate::Union{Nothing, MaterialShortageCertificate}
end

"""
    cs_standard_dimensions(n) -> (n_stock, n_types)

Stock types and item types for `n` patterns.
"""
function cs_standard_dimensions(n::Int)
    n_stock = clamp(round(Int, log10(max(n, 1))), 1, 5)
    n_stock = min(n_stock, max(1, n ÷ 4))
    # Small targets need a few extra item types or the pattern space is too
    # small to hold `n` distinct patterns.
    n_types = max(round(Int, n / 20), min(8, n ÷ (2 * n_stock)), min(3, n ÷ n_stock))
    n_types = clamp(n_types, 1, max(1, n ÷ n_stock))
    return n_stock, n_types
end

# Single-item plan: item i on a random stock type it fits.
function _cs_single_plan(rng, stock_lengths, piece_lengths, demands, single)
    n_types = length(piece_lengths)
    pattern = zeros(Int, n_types)
    usage = zeros(Int, n_types)
    per_stock = zeros(Int, length(stock_lengths))
    for i in 1:n_types
        fits = [k for k in eachindex(stock_lengths) if single[i, k] > 0]
        k = fits[rand(rng, 1:length(fits))]
        pattern[i] = single[i, k]
        yield = stock_lengths[k] ÷ piece_lengths[i]
        usage[i] = cld(demands[i], yield)
        per_stock[k] += usage[i]
    end
    return pattern, usage, per_stock
end

function CuttingStockProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))
    rng = MersenneTwister(seed)
    n = target_variables
    n_stock, n_types = cs_standard_dimensions(n)

    stock_lengths, stock_costs = cs_stock_types(rng, n_stock)
    piece_lengths = cs_piece_lengths(rng, n_types, floor(Int, 0.45 * stock_lengths[end]))
    demands = cs_demands(rng, n_types)
    patterns = cs_generate_patterns(rng, stock_lengths, piece_lengths, n)
    single = cs_single_index(patterns, stock_lengths, piece_lengths)

    pattern, usage, per_stock = _cs_single_plan(rng, stock_lengths, piece_lengths, demands, single)
    material = sum(Float64(piece_lengths[i]) * demands[i] for i in 1:n_types)

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        availability = [ceil(Int, per_stock[k] * (1.05 + 0.30 * rand(rng))) for k in 1:n_stock]
        feasible_witness = StockPlanWitness(pattern, usage)
    else
        base = [per_stock[k] * (1.05 + 0.30 * rand(rng)) + 1.0 for k in 1:n_stock]
        supply = sum(stock_lengths[k] * base[k] for k in 1:n_stock)
        ratio = if feasibility_status == infeasible
            1.0 / (1.08 + 0.12 * rand(rng))
        else
            0.97 + 0.13 * rand(rng)
        end
        availability = [floor(Int, base[k] * ratio * material / supply) for k in 1:n_stock]
        if feasibility_status == infeasible
            supply = sum(Float64(stock_lengths[k]) * availability[k] for k in 1:n_stock)
            infeasibility_certificate = MaterialShortageCertificate(material, supply)
            material >= 1.04 * supply ||
                error("cutting_stock: shortage certificate lost its margin")
        end
    end

    return CuttingStockProblem(
        stock_lengths,
        stock_costs,
        piece_lengths,
        demands,
        patterns,
        availability,
        feasibility_status,
        feasible_witness,
        infeasibility_certificate,
    )
end

function build_model(prob::CuttingStockProblem)
    model = Model()
    pats = prob.patterns
    n = length(pats)
    @variable(model, x[1:n] >= 0)
    @objective(model, Min, sum(prob.stock_costs[pats.stock[j]] * x[j] for j in 1:n))

    produced = [AffExpr() for _ in eachindex(prob.piece_lengths)]
    used = [AffExpr() for _ in eachindex(prob.stock_lengths)]
    for j in 1:n
        for (i, c) in zip(pats.items[j], pats.counts[j])
            add_to_expression!(produced[i], c, x[j])
        end
        add_to_expression!(used[pats.stock[j]], 1.0, x[j])
    end
    @constraint(model, demand[i in eachindex(produced)], produced[i] >= prob.demands[i])
    for k in eachindex(used)
        isempty(used[k].terms) && continue
        @constraint(model, used[k] <= prob.availability[k])
    end
    return model
end

register_variant(
    :cutting_stock,
    :standard,
    CuttingStockProblem,
    "Multi-stock Gilmore-Gomory cutting stock LP: many order lengths, several bar lengths in limited supply, and a large generated pattern pool";
    default=true,
    tags=[:production, :covering],
)
