using JuMP
using Random
using Distributions
using StatsBase

"""
    StockPlanWitness

Integer operating plan for a `feasible` instance: run the single-piece pattern
of piece type `i` exactly `usage[i] = cld(demands[i], s_i)` times, where
`s_i = patterns[i][i] = floor(stock_length / piece_lengths[i])` is the yield of
that direct pattern. The constructor emits every piece type's single-piece
pattern first and in piece order, so `usage[i]` rides pattern index `i`.

The plan is verified in exact integer arithmetic:

  - production of piece `i` is `s_i * usage[i] >= demands[i]` (ceil division),
  - total stock consumed is `total_stock = sum(usage) <= stock_limit`.

No rounding tolerance is needed anywhere, and because the plan is integral it
would remain a witness even if the pattern variables were required to be
integer.
"""
struct StockPlanWitness
    usage::Vector{Int}
    total_stock::Int
end

"""
    StockShortageCertificate

Structured infeasibility certificate: of all generated patterns, the best
yields `max_yield_per_stock = e_i = max_j patterns[j][i]` pieces of type
`piece_index` per stock roll (at least `floor(L / ℓ_i) >= 1`, since the
single-piece pattern is always present). Any nonnegative usage vector `x`
therefore produces at most

`e_i * sum(x) <= e_i * stock_limit < demand`

units of that piece, while its demand row requires at least `demand`. The
demand row and the stock row are jointly unsatisfiable — a two-row Farkas
argument that needs no integer reasoning and survives `relax_integer = true`.
The invariant `demand > stock_limit * max_yield_per_stock` is checked in the
constructor before the struct is returned.
"""
struct StockShortageCertificate
    piece_index::Int
    max_yield_per_stock::Int
    stock_limit::Int
    demand::Int
end

"""
    CuttingStockProblem <: ProblemGenerator

Generator for one-dimensional cutting stock problems with certificate-backed
feasibility control.

# Overview

Models one-dimensional cutting stock: stock material of one standard length is
cut into demanded piece lengths using a fixed list of cutting patterns. The
decisions are continuous usage counts per pattern, the objective minimizes the
total number of stock pieces used, demand rows require enough pieces of every
requested length, and a stock-limit row caps total pattern usage. This is the
LP relaxation of the pattern-count problem; the sibling `integer_patterns`
variant covers the integral formulation.

# Provable feasibility mechanism

Let `s_i = floor(L / ℓ_i)` be the yield of piece `i`'s single-piece pattern and
`e_i = max_j patterns[j][i] >= s_i` its best yield over *all* final patterns.
Two elementary consequences of `x >= 0` steer all three profiles:

  - **Upper bound per piece**: production of piece `i` never exceeds
    `e_i * sum(x)`, so a stock limit `S` with `d_i > S * e_i` for any single
    `i` certifies infeasibility.
  - **Trivial plan**: running each single-piece pattern `cld(d_i, s_i)` times
    meets every demand using `U = sum_i cld(d_i, s_i)` stock pieces, so a
    limit `S >= U` certifies feasibility.

The profiles place the stock limit relative to the trivial plan's usage `U`:

  - `feasible`: `S = U * U(1.3, 1.8)`, and the trivial plan itself is stored as
    a `StockPlanWitness`. The budget is generous but real: any plan restricted
    to single-piece patterns must respect it (the stored plan uses 55-77% of
    it), while better mixed patterns may leave slack at the optimum.
  - `infeasible`: scenario-flavored demands are scaled up (rush order, seasonal
    spike, ...), then `S = U * U(0.30, 0.55)` — the mill holds only a fraction
    of what the scaled order book needs. The piece type with the largest
    demand-per-yield ratio is the natural bottleneck; its demand is raised (if
    the draw did not already do so) to `ceil(margin * S * e_i)` with
    `margin ~ U(1.2, 1.5)`, and the resulting `StockShortageCertificate` is
    stored. The margin makes the contradiction robust to solver tolerances,
    not merely to exact arithmetic.
  - `unknown`: `S = U * exp(N(-0.15, 0.45))`, log-centered slightly below the
    trivial plan's need. The outcome hinges on how well the sampled mixed
    patterns pack against the drawn demands, and lands near an even
    OPTIMAL/INFEASIBLE split at every scale. Neither a witness nor a
    certificate is stored.

# Fields

  - `stock_length::Float64`: Length of one stock piece
  - `piece_lengths::Vector{Float64}`: Length of each piece type required
  - `demands::Vector{Int}`: Demand for each piece type (all `>= 1`)
  - `patterns::Vector{Vector{Int}}`: Cutting patterns, `patterns[j][i]` = count
    of piece type `i` cut by pattern `j`; entries `1:length(piece_lengths)` are
    the single-piece patterns in piece order
  - `stock_limit::Int`: Cap on total pattern usage (always `>= 1`; there is no
    unlimited mode)
  - `scenario::Symbol`: Demand regime the instance narrates
    (`:steady_demand`, `:rush_order`, `:seasonal_spike`, `:backlog_clearing`,
    `:mixed`, or `:capacity_review`)
  - `feasible_witness::Union{Nothing,StockPlanWitness}`: set for `feasible`
  - `infeasibility_certificate::Union{Nothing,StockShortageCertificate}`: set for `infeasible`
  - `feasibility_status::FeasibilityStatus`: Requested profile
"""
struct CuttingStockProblem <: ProblemGenerator
    stock_length::Float64
    piece_lengths::Vector{Float64}
    demands::Vector{Int}
    patterns::Vector{Vector{Int}}
    stock_limit::Int
    scenario::Symbol
    feasible_witness::Union{Nothing, StockPlanWitness}
    infeasibility_certificate::Union{Nothing, StockShortageCertificate}
    feasibility_status::FeasibilityStatus
end

# Add a pattern unless it is empty or already present. `seen` mirrors
# `patterns` as a Set so the distinctness check stays O(1) at large pattern
# counts instead of scanning the whole list per attempt.
function cs_add_pattern!(
    patterns::Vector{Vector{Int}}, seen::Set{Vector{Int}}, pattern::Vector{Int}
)
    sum(pattern) > 0 || return false
    pattern in seen && return false
    push!(patterns, pattern)
    push!(seen, pattern)
    return true
end

"""
    cs_generate_patterns(rng, stock_length, piece_lengths, n_patterns, waste_factor)

Generate exactly `n_patterns` distinct feasible cutting patterns. Three stages:

1. One single-piece pattern per piece type, first and in piece order. These
   anchor the yield bounds (`s_i >= 1` keeps every demand row coverable) and
   the `StockPlanWitness` mapping (`patterns[i]` is piece `i`'s direct
   pattern).
2. Greedy residual fill: pick a handful of piece types, then repeatedly cut
   the length that maximizes pieces per remaining stock (weighted sampling),
   stopping early once the residual drops below `waste_factor`. Exits early
   with a non-pattern only when nothing fits.
3. Deterministic top-up enumerating sub-maximal single-type and two-type
   patterns, guaranteeing the count even if the greedy stage stalls on a
   small piece-type pool.

Errors if the exact count is still unreachable (only possible for
pathologically tiny targets).
"""
function cs_generate_patterns(
    rng::AbstractRNG,
    stock_length::Float64,
    piece_lengths::Vector{Float64},
    n_patterns::Int,
    waste_factor::Float64,
)
    n_types = length(piece_lengths)
    patterns = Vector{Vector{Int}}()
    seen = Set{Vector{Int}}()

    # Stage 1: single-piece patterns (floor of the exact ratio can only round
    # down, so every pattern provably fits within the stock length).
    for i in 1:n_types
        pattern = zeros(Int, n_types)
        pattern[i] = floor(Int, stock_length / piece_lengths[i])
        cs_add_pattern!(patterns, seen, pattern)
    end

    # Stage 2: greedy residual fill with weighted piece selection.
    attempts = 0
    while length(patterns) < n_patterns && attempts < 5 * n_patterns
        attempts += 1
        new_pattern = zeros(Int, n_types)
        remaining_length = stock_length

        num_types_to_use = min(n_types, max(1, round(Int, rand(rng, Exponential(2.0)))))
        selected_indices = sample(rng, collect(1:n_types), num_types_to_use; replace=false)

        while !isempty(selected_indices)
            # Shorter pieces fit more per stock, so weight the choice by
            # stock_length / piece_length.
            weights = [stock_length / piece_lengths[i] for i in selected_indices]
            idx = sample(rng, selected_indices, Weights(weights))

            if piece_lengths[idx] <= remaining_length
                new_pattern[idx] += 1
                remaining_length -= piece_lengths[idx]
                if remaining_length / stock_length <= waste_factor
                    break
                end
            else
                filter!(i -> piece_lengths[i] <= remaining_length, selected_indices)
            end
        end

        cs_add_pattern!(patterns, seen, new_pattern)
    end

    # Stage 3: deterministic top-up. Sub-maximal runs of one type, then
    # two-type combinations, give a large duplicate-free pool.
    for i in 1:n_types
        max_count = floor(Int, stock_length / piece_lengths[i])
        for count in 1:(max_count - 1)
            length(patterns) >= n_patterns && break
            pattern = zeros(Int, n_types)
            pattern[i] = count
            cs_add_pattern!(patterns, seen, pattern)
        end
        length(patterns) >= n_patterns && break
    end
    for i in 1:n_types, j in (i + 1):n_types
        for i_count in 1:floor(Int, stock_length / piece_lengths[i])
            remaining = stock_length - i_count * piece_lengths[i]
            for j_count in 1:floor(Int, remaining / piece_lengths[j])
                length(patterns) >= n_patterns && break
                pattern = zeros(Int, n_types)
                pattern[i] = i_count
                pattern[j] = j_count
                cs_add_pattern!(patterns, seen, pattern)
            end
            length(patterns) >= n_patterns && break
        end
        length(patterns) >= n_patterns && break
    end

    length(patterns) == n_patterns ||
        error("Could generate only $(length(patterns)) of $n_patterns cutting patterns")
    return patterns
end

"""
    cs_demand_scaling_factors(rng, is_common, scenario)

Scenario-flavored multipliers applied to the booked demands before the stock
limit is drawn, so the eventual shortage reads as a business event rather than
an arbitrary capacity cut. Rush orders concentrate on catalog (common) lengths
while bespoke sizes stay near nominal; the other regimes move the whole book.

Keyed on the tracked `is_common` flag rather than float membership in the
common-length list: generated common lengths carry `Normal(0, 0.02)` jitter,
so they almost never equal a catalog value exactly and the old membership test
never fired.
"""
function cs_demand_scaling_factors(rng::AbstractRNG, is_common::Vector{Bool}, scenario::Symbol)
    n_pieces = length(is_common)
    scaling_factors = ones(Float64, n_pieces)

    if scenario == :rush_order
        for i in 1:n_pieces
            scaling_factors[i] =
                is_common[i] ? rand(rng, Uniform(2.0, 4.0)) : rand(rng, Uniform(0.8, 1.5))
        end
    elseif scenario == :seasonal_spike
        base_spike = rand(rng, Uniform(1.8, 2.5))
        scaling_factors .= base_spike .* rand(rng, Uniform(0.8, 1.2), n_pieces)
    elseif scenario == :backlog_clearing
        scaling_factors .= rand(rng, Uniform(2.2, 3.5), n_pieces)
    else  # :mixed
        scaling_factors .= rand(rng, Uniform(1.5, 2.8), n_pieces)
    end

    return scaling_factors
end

"""
    CuttingStockProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a cutting stock instance with exactly `target_variables` pattern
variables and the requested feasibility profile. All randomness lives here,
drawn from a constructor-local `MersenneTwister(seed)`.

# Arguments

  - `target_variables`: Exact number of variables (cutting patterns)
  - `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
  - `seed`: Random seed for reproducibility
"""
function CuttingStockProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))

    rng = MersenneTwister(seed)
    n_patterns = target_variables

    # Scale parameters based on target variable count
    if target_variables <= 250
        n_piece_types = rand(rng, 3:min(15, max(3, target_variables ÷ 10)))
        stock_length = rand(rng, Uniform(3.0, 8.0))
        demand_min = rand(rng, 5:20)
        demand_max = rand(rng, 50:200)
        common_lengths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        common_length_prob = rand(rng, Uniform(0.3, 0.6))
        waste_factor = rand(rng, Uniform(0.05, 0.15))
    elseif target_variables <= 1000
        n_piece_types = rand(rng, 8:min(50, max(8, target_variables ÷ 20)))
        stock_length = rand(rng, Uniform(6.0, 12.0))
        demand_min = rand(rng, 20:100)
        demand_max = rand(rng, 200:1000)
        common_lengths = [1.0, 1.2, 1.5, 2.0, 2.4, 3.0, 4.0, 6.0]
        common_length_prob = rand(rng, Uniform(0.4, 0.7))
        waste_factor = rand(rng, Uniform(0.03, 0.10))
    else
        n_piece_types = rand(rng, 20:min(200, max(20, target_variables ÷ 50)))
        stock_length = rand(rng, Uniform(8.0, 20.0))
        demand_min = rand(rng, 100:500)
        demand_max = rand(rng, 1000:10000)
        common_lengths = [1.0, 1.2, 1.5, 2.0, 2.4, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        common_length_prob = rand(rng, Uniform(0.5, 0.8))
        waste_factor = rand(rng, Uniform(0.02, 0.08))
    end

    # Enough piece-type diversity that the distinct-pattern pool reaches
    # n_patterns; with only a handful of types the pool stalls far below
    # target and the realized variable count undershoots the request. But
    # never more piece types than patterns: every type needs its own
    # single-piece pattern or its demand row would be structurally uncoverable
    # (the degenerate "no-pattern" flaw this generator used to have).
    n_piece_types = max(n_piece_types, clamp(round(Int, n_patterns / 4), 20, 200))
    n_piece_types = min(n_piece_types, n_patterns)

    # Generate realistic piece lengths (all must fit in stock), tracking which
    # ones are catalog sizes. Commonness is recorded as a flag because the
    # jittered, rounded lengths almost never equal a catalog value exactly.
    effective_max_length = min(stock_length * 0.95, stock_length - 0.1)
    piece_lengths = Float64[]
    is_common = Bool[]
    for _ in 1:n_piece_types
        if rand(rng) < common_length_prob
            valid_lengths = filter(<=(effective_max_length), common_lengths)
            base_length =
                isempty(valid_lengths) ? effective_max_length * 0.8 : rand(rng, valid_lengths)
            variation = rand(rng, Normal(0, 0.02))
            length = clamp(base_length + variation, 0.1, effective_max_length)
            length = round(length; digits=2)
            common = true
        else
            normalized = rand(rng, Beta(2.0, 3.0))
            length = 0.1 + (effective_max_length - 0.1) * normalized
            precision = stock_length > 10 ? 0.1 : 0.05
            length = round(length / precision) * precision
            common = false
        end
        # Dedupe on the fly so the lengths and the commonness flag stay in
        # lockstep (a trailing unique! on the lengths alone would desync them).
        length in piece_lengths || (push!(piece_lengths, length); push!(is_common, common))
    end
    n_types = length(piece_lengths)

    # Base demands from a lognormal whose parameters are quantile-matched to
    # the drawn range: [demand_min, demand_max] is a ~2-sigma band, so the
    # clamp below only ever trims the outer ~2% of draws. (The old median at
    # (min+max)/2 left up to a third of draws below demand_min, piling a point
    # mass on the minimum.) Catalog sizes are ordered in higher volume with
    # less spread than bespoke ones.
    spread = log(demand_max / demand_min)
    base_center = log(sqrt(demand_min * demand_max))
    demands = zeros(Int, n_types)
    for i in 1:n_types
        sigma = (is_common[i] ? 0.75 : 1.0) * spread / 4
        center = base_center + (is_common[i] ? log(1.25) : 0.0)
        raw = rand(rng, LogNormal(center, sigma))
        if raw < 50
            raw = round(raw / 5) * 5
        elseif raw < 200
            raw = round(raw / 10) * 10
        else
            raw = round(raw / 25) * 25
        end
        demands[i] = clamp(round(Int, raw), demand_min, demand_max)
    end

    # Pattern list is shared by all three profiles; feasibility is decided
    # afterwards by the stock limit alone (plus the certificate's demand bump).
    patterns = cs_generate_patterns(rng, stock_length, piece_lengths, n_patterns, waste_factor)

    # Yields computed from the FINAL pattern list: s_i is the single-piece
    # yield (used by the trivial plan), e_i the best over all patterns (used
    # by the certificate). e_i >= s_i >= 1 always.
    single_yield = [patterns[i][i] for i in 1:n_types]
    max_yield = [maximum(patterns[j][i] for j in 1:n_patterns) for i in 1:n_types]

    feasible_witness = nothing
    infeasibility_certificate = nothing

    if feasibility_status == feasible
        # Steady market: mild independent jitter around the booked orders.
        scenario = :steady_demand
        demands = [max(1, round(Int, d * rand(rng, Uniform(0.8, 1.2)))) for d in demands]

        # The trivial plan: cld keeps everything in exact integer arithmetic.
        usage = [cld(demands[i], single_yield[i]) for i in 1:n_types]
        plan_stock = sum(usage)

        # Generous-but-binding budget: 1.3-1.8x what the direct plan needs, so
        # the stock row genuinely constrains single-piece-only plans while the
        # stored witness proves the instance feasible with room to spare.
        stock_limit = max(plan_stock, round(Int, plan_stock * rand(rng, Uniform(1.3, 1.8))))
        feasible_witness = StockPlanWitness(usage, plan_stock)
    elseif feasibility_status == infeasible
        # Scenario-flavored demand spike, then a stock budget that covers only
        # a fraction of what the scaled order book needs.
        scenario = rand(rng, (:rush_order, :seasonal_spike, :backlog_clearing, :mixed))
        scaling_factors = cs_demand_scaling_factors(rng, is_common, scenario)
        demands = [max(1, round(Int, d * f)) for (d, f) in zip(demands, scaling_factors)]
        plan_stock = sum(cld(demands[i], single_yield[i]) for i in 1:n_types)
        stock_limit = max(1, round(Int, plan_stock * rand(rng, Uniform(0.30, 0.55))))

        # Certify on the FINAL patterns and demands: the bottleneck piece is
        # the one whose demand is hardest to serve per stock roll. Raise its
        # demand to margin * S * e_i if the scenario draw did not already push
        # it past the provable bound -- raising, never lowering, so the other
        # rows keep their drawn structure.
        cert_idx = argmax(i -> demands[i] / max_yield[i], 1:n_types)
        margin = rand(rng, Uniform(1.2, 1.5))
        demands[cert_idx] = max(
            demands[cert_idx], ceil(Int, margin * stock_limit * max_yield[cert_idx])
        )
        infeasibility_certificate = StockShortageCertificate(
            cert_idx, max_yield[cert_idx], stock_limit, demands[cert_idx]
        )
    else
        # Capacity review: the budget is log-centered slightly *below* the
        # direct plan's need (median ~0.86x), assuming the mixed patterns will
        # close the gap. Sometimes they do, sometimes they do not -- which is
        # exactly what `unknown` should mean, at every scale (empirically a
        # near-even OPTIMAL/INFEASIBLE split).
        scenario = :capacity_review
        demands = [max(1, round(Int, d * rand(rng, Uniform(0.9, 1.1)))) for d in demands]
        plan_stock = sum(cld(demands[i], single_yield[i]) for i in 1:n_types)
        stock_limit = max(1, round(Int, plan_stock * exp(rand(rng, Normal(-0.15, 0.45)))))
    end

    # Guard the certificate invariant against future edits: it must hold on
    # the returned fields themselves, never on intermediate values.
    if infeasibility_certificate !== nothing
        c = infeasibility_certificate
        c.demand > c.stock_limit * c.max_yield_per_stock || error(
            "stock-shortage certificate invariant violated: " *
            "$(c.demand) <= $(c.stock_limit) * $(c.max_yield_per_stock)",
        )
    end
    if feasible_witness !== nothing
        w = feasible_witness
        w.total_stock <= stock_limit ||
            error("plan witness uses $(w.total_stock) stock but the limit is $stock_limit")
    end

    return CuttingStockProblem(
        stock_length,
        piece_lengths,
        demands,
        patterns,
        stock_limit,
        scenario,
        feasible_witness,
        infeasibility_certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::CuttingStockProblem)

Build the JuMP model: one continuous usage variable per pattern, minimizing
total stock used, subject to the demand rows and the stock-limit row. The
stock limit is always finite (`>= 1`), so the model always carries the budget
row the certificate reasons about. Deterministic: built from `prob`'s fields
only.
"""
function build_model(prob::CuttingStockProblem)
    model = Model()

    n_patterns = length(prob.patterns)
    n_pieces = length(prob.piece_lengths)

    @variable(model, x[1:n_patterns] >= 0)

    @objective(model, Min, sum(x))

    # Meet demand for each piece size
    for i in 1:n_pieces
        @constraint(
            model,
            sum(prob.patterns[j][i] * x[j] for j in 1:n_patterns if prob.patterns[j][i] > 0) >=
                prob.demands[i]
        )
    end

    # Stock limit on total pattern usage
    @constraint(model, sum(x) <= prob.stock_limit)

    return model
end

# Register the variant
register_variant(
    :cutting_stock,
    :standard,
    CuttingStockProblem,
    "Cutting stock optimization problem that minimizes waste by determining optimal cutting patterns for stock material to satisfy demand for pieces of various lengths",
)
