using JuMP
using Random

"""
    IntegerPatternCuttingStockProblem <: ProblemGenerator

An integer pattern-usage formulation of one-dimensional cutting stock. Unlike
the category's continuous standard variant, every pattern count is a
nonnegative general-integer variable. The generated pattern matrix includes a
direct pattern for each piece type plus diverse mixed patterns.

# Fields
- `stock_length`: Length of one stock roll.
- `piece_lengths`: Requested piece lengths.
- `demands`: Minimum production by piece type.
- `patterns`: Piece-type by pattern matrix.
- `stock_limit`: Maximum total pattern usage.
- `planted_usage`: Integer witness used to generate feasible demands.
"""
struct IntegerPatternCuttingStockProblem <: ProblemGenerator
    stock_length::Int
    piece_lengths::Vector{Int}
    demands::Vector{Int}
    patterns::Matrix{Int}
    stock_limit::Float64
    planted_usage::Vector{Int}
end

# Add a feasible pattern unless it is empty or already present.
function ics_add_pattern!(patterns::Vector{Vector{Int}}, pattern::Vector{Int})
    sum(pattern) > 0 || return false
    pattern in patterns && return false
    push!(patterns, pattern)
    return true
end

# Generate exactly `n_patterns` distinct feasible patterns. Direct patterns are
# inserted first so every demand row is coverable; random mixed patterns are
# followed by deterministic single/pair enumeration as a guaranteed fallback.
function ics_generate_patterns(
    rng::AbstractRNG,
    stock_length::Int,
    piece_lengths::Vector{Int},
    n_patterns::Int,
)
    n_types = length(piece_lengths)
    patterns = Vector{Vector{Int}}()

    for piece in 1:n_types
        length(patterns) >= n_patterns && break
        pattern = zeros(Int, n_types)
        pattern[piece] = div(stock_length, piece_lengths[piece])
        ics_add_pattern!(patterns, pattern)
    end

    attempts = 0
    while length(patterns) < n_patterns && attempts < 100 * n_patterns
        attempts += 1
        pattern = zeros(Int, n_types)
        remaining = stock_length
        for piece in randperm(rng, n_types)
            max_count = div(remaining, piece_lengths[piece])
            max_count == 0 && continue
            if rand(rng) < 0.45
                count = rand(rng, 0:max_count)
                pattern[piece] = count
                remaining -= count * piece_lengths[piece]
            end
        end
        if sum(pattern) == 0
            piece = rand(rng, 1:n_types)
            pattern[piece] = 1
        end
        ics_add_pattern!(patterns, pattern)
    end

    # Deterministic fallback also diversifies non-maximal and two-type patterns.
    for first in 1:n_types
        for first_count in 1:div(stock_length, piece_lengths[first])
            length(patterns) >= n_patterns && break
            pattern = zeros(Int, n_types)
            pattern[first] = first_count
            ics_add_pattern!(patterns, pattern)
        end
        length(patterns) >= n_patterns && break
    end
    for first in 1:n_types, second in (first + 1):n_types
        for first_count in 1:div(stock_length, piece_lengths[first])
            remaining = stock_length - first_count * piece_lengths[first]
            for second_count in 1:div(remaining, piece_lengths[second])
                length(patterns) >= n_patterns && break
                pattern = zeros(Int, n_types)
                pattern[first] = first_count
                pattern[second] = second_count
                ics_add_pattern!(patterns, pattern)
            end
            length(patterns) >= n_patterns && break
        end
        length(patterns) >= n_patterns && break
    end

    length(patterns) == n_patterns ||
        error("Could generate only $(length(patterns)) of $n_patterns cutting patterns")
    return hcat(patterns...)
end

"""
    IntegerPatternCuttingStockProblem(target_variables, feasibility_status, seed)

Construct an integer cutting-stock instance with exactly `target_variables`
pattern variables. `feasible` instances admit `planted_usage`. `infeasible`
instances impose a stock limit strictly below

`sum(piece_length[i] * demand[i]) / stock_length`,

a necessary lower bound on total pattern usage that certifies infeasibility even
when general-integer variables are relaxed. `unknown` places the limit near that
lower bound without asserting a result.
"""
function IntegerPatternCuttingStockProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))

    rng = MersenneTwister(seed)
    n_patterns = target_variables
    n_piece_types = min(n_patterns, clamp(round(Int, sqrt(n_patterns)), 4, 30))
    stock_length = 120

    length_pool = collect(8:55)
    selected = randperm(rng, length(length_pool))[1:n_piece_types]
    piece_lengths = sort(length_pool[selected])
    patterns = ics_generate_patterns(rng, stock_length, piece_lengths, n_patterns)

    planted_usage = zeros(Int, n_patterns)
    # The first n_piece_types columns are direct patterns, so giving each one
    # positive usage guarantees positive production in every demand row.
    for pattern in 1:min(n_piece_types, n_patterns)
        planted_usage[pattern] = rand(rng, 1:3)
    end
    for pattern in (n_piece_types + 1):n_patterns
        rand(rng) < 0.12 && (planted_usage[pattern] = rand(rng, 1:2))
    end

    planted_production = patterns * planted_usage
    demands = [max(1, floor(Int, produced * (0.65 + 0.25 * rand(rng))))
               for produced in planted_production]
    planted_stock = sum(planted_usage)
    volume_lower_bound = sum(piece_lengths .* demands) / stock_length

    stock_limit = if feasibility_status == feasible
        Float64(planted_stock)
    elseif feasibility_status == infeasible
        volume_lower_bound * (0.65 + 0.20 * rand(rng))
    else
        volume_lower_bound * (0.90 + 0.35 * rand(rng))
    end

    return IntegerPatternCuttingStockProblem(
        stock_length,
        piece_lengths,
        demands,
        patterns,
        stock_limit,
        planted_usage,
    )
end

"""
    build_model(prob::IntegerPatternCuttingStockProblem)

Build the general-integer pattern-usage model. The model minimizes stock rolls
while meeting every demand and respecting the generated stock limit.
"""
function build_model(prob::IntegerPatternCuttingStockProblem)
    model = Model()
    n_piece_types, n_patterns = size(prob.patterns)

    @variable(model, pattern_usage[1:n_patterns] >= 0, Int)
    @objective(model, Min, sum(pattern_usage))

    for piece in 1:n_piece_types
        @constraint(
            model,
            sum(prob.patterns[piece, pattern] * pattern_usage[pattern]
                for pattern in 1:n_patterns) >= prob.demands[piece],
        )
    end
    @constraint(model, sum(pattern_usage) <= prob.stock_limit)

    return model
end

register_variant(
    :cutting_stock,
    :integer_patterns,
    IntegerPatternCuttingStockProblem,
    "Integer cutting stock with general-integer pattern counts and a relaxation-safe stock-volume certificate",
)

