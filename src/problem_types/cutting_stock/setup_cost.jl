using JuMP
using Random
using Distributions
using StatsBase

"""
    SetupCostCuttingStockProblem <: ProblemGenerator

Generator for cutting stock optimization problems with a fixed setup cost per pattern.

# Overview
Models one-dimensional cutting stock where, in addition to consuming stock pieces,
activating (setting up) a cutting pattern incurs a fixed cost. The decisions are a
continuous usage count `x[j]` for each pattern and a binary activation `y[j]`
indicating whether the pattern is used at all. The objective minimizes the total
number of stock pieces consumed plus the total setup cost of the activated
patterns. Demand constraints require enough pieces of every requested length, a
linking constraint `x[j] <= M * y[j]` forces a setup whenever a pattern is used,
and an optional stock limit caps total pattern usage.

The setup costs are scaled relative to the per-roll material value (each roll
contributes `1.0` to the `sum(x)` term of the objective). They are sized so that a
setup is worth a small number of rolls, which makes the consolidation tradeoff
(use fewer distinct patterns vs. accept some extra waste) non-degenerate rather
than setup-dominated.

# Fields
- `piece_lengths::Vector{Float64}`: Length of each piece type required
- `demands::Vector{Int}`: Demand for each piece type
- `patterns::Vector{Vector{Int}}`: Cutting patterns (how many of each piece per stock)
- `stock_length::Float64`: Length of stock material
- `stock_limit::Int`: Maximum number of stock pieces available (0 = unlimited)
- `setup_costs::Vector{Float64}`: Fixed setup cost incurred when a pattern is activated
- `big_m::Vector{Float64}`: Per-pattern big-M coefficient linking `x[j]` to its activation `y[j]`
"""
struct SetupCostCuttingStockProblem <: ProblemGenerator
    piece_lengths::Vector{Float64}
    demands::Vector{Int}
    patterns::Vector{Vector{Int}}
    stock_length::Float64
    stock_limit::Int
    setup_costs::Vector{Float64}
    big_m::Vector{Float64}
end

"""
    SetupCostCuttingStockProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a cutting-stock-with-setup-cost problem instance.

The model has two variable sets of equal size: continuous usage `x[1:n_patterns]`
and binary activation `y[1:n_patterns]`. Total variables = 2 * n_patterns, so the
number of generated patterns is sized to roughly `target_variables / 2`.

# Arguments
- `target_variables`: Target number of variables (= 2 * number of cutting patterns)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function SetupCostCuttingStockProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Variable count = n_patterns (x) + n_patterns (y) = 2 * n_patterns.
    # Size the number of patterns to hit the target variable count.
    max_patterns = max(2, round(Int, target_variables / 2))

    # Scale parameters based on target variable count.
    #
    # The number of DISTINCT cutting patterns is what sets the variable count, and
    # it grows combinatorially with both the number of piece types and how many
    # pieces fit per roll. With too few piece types and a short stock, the pattern
    # generator quickly exhausts distinct patterns and undershoots the target. To
    # reliably reach ~max_patterns distinct patterns we keep the number of piece
    # types comfortably large relative to the target and use a generous stock
    # length so many pieces fit per roll.
    base_piece_types = max(6, ceil(Int, 1.5 * sqrt(max_patterns)))

    if target_variables <= 250
        n_piece_types = clamp(base_piece_types, 6, 40)
        stock_length = rand(Uniform(6.0, 10.0))
        demand_min = rand(5:20)
        demand_max = rand(50:200)
        common_lengths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        common_length_prob = rand(Uniform(0.3, 0.6))
        waste_factor = rand(Uniform(0.05, 0.15))
    elseif target_variables <= 1000
        n_piece_types = clamp(base_piece_types, 12, 80)
        stock_length = rand(Uniform(8.0, 14.0))
        demand_min = rand(20:100)
        demand_max = rand(200:1000)
        common_lengths = [1.0, 1.2, 1.5, 2.0, 2.4, 3.0, 4.0, 6.0]
        common_length_prob = rand(Uniform(0.4, 0.7))
        waste_factor = rand(Uniform(0.03, 0.10))
    else
        n_piece_types = clamp(base_piece_types, 25, 200)
        stock_length = rand(Uniform(10.0, 20.0))
        demand_min = rand(100:500)
        demand_max = rand(1000:10000)
        common_lengths = [1.0, 1.2, 1.5, 2.0, 2.4, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        common_length_prob = rand(Uniform(0.5, 0.8))
        waste_factor = rand(Uniform(0.02, 0.08))
    end

    # Generate realistic piece lengths (all must fit in stock)
    piece_lengths = Float64[]
    effective_max_length = min(stock_length * 0.95, stock_length - 0.1)

    for i in 1:n_piece_types
        if rand() < common_length_prob && !isempty(common_lengths)
            base_length = rand(common_lengths)
            if base_length > effective_max_length
                valid_lengths = filter(x -> x <= effective_max_length, common_lengths)
                base_length = isempty(valid_lengths) ? effective_max_length * 0.8 : rand(valid_lengths)
            end
            variation = rand(Normal(0, 0.02))
            length = clamp(base_length + variation, 0.1, effective_max_length)
            push!(piece_lengths, round(length, digits=2))
        else
            α, β = 2.0, 3.0
            normalized = rand(Beta(α, β))
            length = 0.1 + (effective_max_length - 0.1) * normalized
            precision = stock_length > 10 ? 0.1 : 0.05
            length = round(length / precision) * precision
            push!(piece_lengths, length)
        end
    end

    unique!(piece_lengths)
    n_piece_types = length(piece_lengths)

    # Generate initial demands using realistic distributions
    base_demands = Int[]
    for length in piece_lengths
        if length in common_lengths
            μ = log((demand_min + demand_max) / 1.3)
            σ = 0.5
        else
            μ = log((demand_min + demand_max) / 2.0)
            σ = 0.7
        end

        base_demand = rand(LogNormal(μ, σ))
        if base_demand < 50
            base_demand = round(base_demand / 5) * 5
        elseif base_demand < 200
            base_demand = round(base_demand / 10) * 10
        else
            base_demand = round(base_demand / 25) * 25
        end

        push!(base_demands, clamp(round(Int, base_demand), demand_min, demand_max))
    end

    # Always generate a full set of feasible patterns (single-piece + mixed) so
    # that demand is satisfiable in principle; feasibility is then controlled
    # purely through the stock limit.
    patterns = generate_setup_cost_patterns(stock_length, piece_lengths, max_patterns, waste_factor)
    n_patterns = length(patterns)

    # Apply demand variation (realistic manufacturing scenario).
    demands = copy(base_demands)
    for i in 1:length(demands)
        variation = rand(Uniform(0.8, 1.2))
        demands[i] = max(1, round(Int, demands[i] * variation))
    end

    # Compute the best per-pattern efficiency for each piece (max units of piece i
    # producible from a single stock roll across all patterns). Used both for a
    # tight big-M and for the infeasibility construction.
    best_eff = ones(Int, n_piece_types)
    for i in 1:n_piece_types
        e = 0
        for pattern in patterns
            e = max(e, pattern[i])
        end
        if e == 0
            e = max(1, floor(Int, stock_length / piece_lengths[i]))
        end
        best_eff[i] = e
    end

    # Minimum number of rolls needed for each piece if cut with its best pattern.
    min_rolls_per_piece = [ceil(Int, demands[i] / best_eff[i]) for i in 1:n_piece_types]
    # A safe (but reasonably tight) upper bound on rolls needed for total demand.
    max_rolls_needed = max(1, sum(min_rolls_per_piece))

    # Determine target feasibility (unknown -> natural instance, no forced infeasibility).
    target_feasible = feasibility_status == feasible ? true :
                      feasibility_status == infeasible ? false : true

    if feasibility_status == infeasible
        target_feasible = false
    end

    if target_feasible
        # No binding stock limit: every pattern usage is allowed up to big-M.
        stock_limit = 0
    else
        # INFEASIBLE: impose a stock limit strictly below the minimum number of
        # rolls required (with margin), so total demand cannot be met regardless
        # of which patterns are activated. The largest single-piece requirement
        # is the dominant lower bound.
        hardest = maximum(min_rolls_per_piece)
        stock_limit = max(1, floor(Int, hardest * rand(Uniform(0.5, 0.7))))
        # Guarantee the contradiction with a safety margin.
        stock_limit = min(stock_limit, hardest - 1)
        stock_limit = max(1, stock_limit)
    end

    # Setup costs scaled RELATIVE to per-roll material value (1.0 per roll in the
    # objective). A setup is worth a few rolls so the consolidation tradeoff is
    # non-degenerate (neither free nor setup-dominated). The cost is tied to each
    # pattern's trim waste: low-waste (efficient) patterns are cheaper to set up,
    # which is realistic and breaks the symmetry between near-equivalent patterns
    # (keeping the MILP's LP relaxation tight and fast to solve to optimality).
    setup_costs = Vector{Float64}(undef, n_patterns)
    for j in 1:n_patterns
        used_length = sum(patterns[j][i] * piece_lengths[i] for i in 1:n_piece_types)
        trim_fraction = clamp(1.0 - used_length / stock_length, 0.0, 1.0)
        # Base cost ~1 roll, plus up to ~2 rolls penalty for waste, with a small
        # per-pattern jitter so ties are broken deterministically. Kept modest
        # (sub-handful of rolls) so the setup term does not dominate the objective
        # and the MILP's LP relaxation stays tight and fast to prove optimal.
        setup_costs[j] = 1.0 + 2.0 * trim_fraction + rand(Uniform(0.0, 0.3))
    end

    # Valid PER-PATTERN big-M. Overproduction is allowed (the demand
    # constraints are >=), so cutting pattern j to satisfy its HIGHEST-demand
    # piece may legitimately overproduce a co-produced low-demand byproduct.
    # The bound must therefore be the *largest* per-piece requirement the
    # pattern can be cut to meet, x[j] <= max_i ceil(demand[i] / pattern[j][i]);
    # using the minimum (as a tighter bound) would wrongly forbid valid and
    # sometimes optimal uses of the pattern. Capped by the global
    # max_rolls_needed and >= 1 for degenerate/empty patterns.
    big_m = Vector{Float64}(undef, n_patterns)
    for j in 1:n_patterns
        cap = 0
        for i in 1:n_piece_types
            if patterns[j][i] > 0
                cap = max(cap, ceil(Int, demands[i] / patterns[j][i]))
            end
        end
        big_m[j] = float(clamp(cap, 1, max_rolls_needed))
    end

    return SetupCostCuttingStockProblem(
        piece_lengths, demands, patterns, stock_length, stock_limit, setup_costs, big_m
    )
end

"""
Helper: generate feasible one-dimensional cutting patterns for the setup-cost variant.

Produces direct single-piece patterns (guaranteeing demand is satisfiable in
principle) plus sampled mixed patterns up to `max_patterns`. Self-contained so it
does not collide with helpers defined in sibling variant files.
"""
function generate_setup_cost_patterns(standard_length, piece_lengths, max_patterns, waste_factor=0.1)
    patterns = Vector{Vector{Int}}()

    # Single-piece patterns
    for (i, piece_length) in enumerate(piece_lengths)
        pattern = zeros(Int, length(piece_lengths))
        pattern[i] = max(1, floor(Int, standard_length / piece_length))
        push!(patterns, pattern)
    end

    # Mixed patterns
    attempts = 0
    max_attempts = max_patterns * 10

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

    return patterns
end

"""
    build_model(prob::SetupCostCuttingStockProblem)

Build a JuMP model for the cutting-stock-with-setup-cost problem. Deterministic —
uses only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::SetupCostCuttingStockProblem)
    model = Model()

    n_patterns = length(prob.patterns)
    n_pieces = length(prob.piece_lengths)

    # Variables: continuous pattern usage x and binary pattern activation y.
    # Total variables = n_patterns (x) + n_patterns (y) = 2 * n_patterns.
    @variable(model, x[1:n_patterns] >= 0)
    @variable(model, y[1:n_patterns], Bin)

    # Objective: minimize stock pieces used + total setup cost of active patterns.
    @objective(model, Min, sum(x) + sum(prob.setup_costs[j] * y[j] for j in 1:n_patterns))

    # Meet demand for each piece size.
    for i in 1:n_pieces
        @constraint(model, sum(prob.patterns[j][i] * x[j] for j in 1:n_patterns) >= prob.demands[i])
    end

    # Link usage to activation: a pattern can only be used if it is set up.
    for j in 1:n_patterns
        @constraint(model, x[j] <= prob.big_m[j] * y[j])
    end

    # Optional stock limit (drives the infeasibility path).
    if prob.stock_limit > 0
        @constraint(model, sum(x) <= prob.stock_limit)
    end

    return model
end

# Register the variant
register_variant(
    :cutting_stock,
    :setup_cost,
    SetupCostCuttingStockProblem,
    "Cutting stock with a fixed setup cost per activated pattern (binary activation linked to usage), minimizing stock pieces plus setup costs",
)
