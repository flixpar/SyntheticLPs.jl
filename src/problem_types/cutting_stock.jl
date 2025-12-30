using JuMP
using Random
using Distributions
using StatsBase

"""
Cutting stock problem variants.

# Variants
- `cut_standard`: Basic cutting stock - minimize number of stock pieces used
- `cut_multi_stock`: Multiple stock sizes available
- `cut_setup_cost`: Include fixed setup cost per pattern used
- `cut_trim_limit`: Maximum acceptable trim loss percentage
- `cut_due_dates`: Time-phased demand with due dates
- `cut_min_runs`: Minimum production runs per pattern
"""
@enum CuttingStockVariant begin
    cut_standard
    cut_multi_stock
    cut_setup_cost
    cut_trim_limit
    cut_due_dates
    cut_min_runs
end

"""
    CuttingStockProblem <: ProblemGenerator

Generator for cutting stock optimization problems with mathematically guaranteed feasibility control.

# Fields
- `piece_lengths::Vector{Float64}`: Length of each piece type required
- `demands::Vector{Int}`: Demand for each piece type
- `patterns::Vector{Vector{Int}}`: Cutting patterns (how many of each piece per stock)
- `stock_length::Float64`: Length of stock material
- `stock_limit::Int`: Maximum number of stock pieces available (0 = unlimited)
"""
struct CuttingStockProblem <: ProblemGenerator
    piece_lengths::Vector{Float64}
    demands::Vector{Int}
    patterns::Vector{Vector{Int}}
    stock_length::Float64
    stock_limit::Int
    variant::CuttingStockVariant
    # Multi-stock variant
    n_stock_types::Int
    stock_lengths::Union{Vector{Float64}, Nothing}
    stock_costs::Union{Vector{Float64}, Nothing}
    patterns_by_stock::Union{Vector{Vector{Vector{Int}}}, Nothing}
    # Setup cost variant
    setup_costs::Union{Vector{Float64}, Nothing}
    # Trim limit variant
    max_trim_fraction::Float64
    # Due dates variant
    n_periods::Int
    period_demands::Union{Matrix{Int}, Nothing}
    # Min runs variant
    min_runs::Int
end

# Backwards compatibility
function CuttingStockProblem(piece_lengths::Vector{Float64}, demands::Vector{Int},
                             patterns::Vector{Vector{Int}}, stock_length::Float64, stock_limit::Int)
    CuttingStockProblem(
        piece_lengths, demands, patterns, stock_length, stock_limit,
        cut_standard,
        0, nothing, nothing, nothing,
        nothing,
        0.0,
        0, nothing,
        0
    )
end

"""
    CuttingStockProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                        variant::CuttingStockVariant=cut_standard)

Construct a cutting stock problem instance with guaranteed feasibility properties.

# Arguments
- `target_variables`: Target number of variables (cutting patterns)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
- `variant`: Cutting stock problem variant (default: cut_standard)
"""
function CuttingStockProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                             variant::CuttingStockVariant=cut_standard)
    Random.seed!(seed)

    max_patterns = target_variables

    # Scale parameters based on target variable count
    if target_variables <= 250
        n_piece_types = rand(3:min(15, max(3, target_variables ÷ 10)))
        stock_length = rand(Uniform(3.0, 8.0))
        demand_min = rand(5:20)
        demand_max = rand(50:200)
        common_lengths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        common_length_prob = rand(Uniform(0.3, 0.6))
        waste_factor = rand(Uniform(0.05, 0.15))
    elseif target_variables <= 1000
        n_piece_types = rand(8:min(50, max(8, target_variables ÷ 20)))
        stock_length = rand(Uniform(6.0, 12.0))
        demand_min = rand(20:100)
        demand_max = rand(200:1000)
        common_lengths = [1.0, 1.2, 1.5, 2.0, 2.4, 3.0, 4.0, 6.0]
        common_length_prob = rand(Uniform(0.4, 0.7))
        waste_factor = rand(Uniform(0.03, 0.10))
    else
        n_piece_types = rand(20:min(200, max(20, target_variables ÷ 50)))
        stock_length = rand(Uniform(8.0, 20.0))
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

    # Determine target feasibility
    target_feasible = feasibility_status == feasible ? true :
                     feasibility_status == infeasible ? false :
                     rand() < 0.5

    # Generate patterns and adjust for feasibility
    demands, stock_limit, patterns = adjust_for_feasibility(
        base_demands, piece_lengths, stock_length, common_lengths,
        target_feasible, waste_factor, max_patterns
    )

    # Initialize variant-specific fields
    n_stock_types = 0
    stock_lengths_arr = nothing
    stock_costs = nothing
    patterns_by_stock = nothing
    setup_costs = nothing
    max_trim_fraction = 0.0
    n_periods = 0
    period_demands = nothing
    min_runs = 0

    n_piece_types = length(piece_lengths)

    if variant == cut_multi_stock
        # Multiple stock sizes available
        n_stock_types = rand(2:4)
        stock_lengths_arr = [stock_length * (0.5 + 0.5 * i / n_stock_types) for i in 1:n_stock_types]
        stock_costs = [l * rand(Uniform(0.8, 1.2)) for l in stock_lengths_arr]

        # Generate patterns for each stock type
        patterns_by_stock = Vector{Vector{Vector{Int}}}()
        for s in 1:n_stock_types
            s_patterns = generate_cutting_patterns(stock_lengths_arr[s], piece_lengths, max_patterns ÷ n_stock_types, waste_factor)
            push!(patterns_by_stock, s_patterns)
        end

    elseif variant == cut_setup_cost
        # Fixed setup cost per pattern used
        setup_costs = [rand(Uniform(50.0, 200.0)) for _ in 1:length(patterns)]

    elseif variant == cut_trim_limit
        # Maximum trim loss
        max_trim_fraction = rand(Uniform(0.05, 0.15))

        if feasibility_status == infeasible
            max_trim_fraction = 0.001  # Impossibly tight
        end

    elseif variant == cut_due_dates
        # Multi-period with due dates
        n_periods = rand(3:6)
        period_demands = zeros(Int, n_piece_types, n_periods)

        for i in 1:n_piece_types
            remaining = demands[i]
            for t in 1:n_periods
                if t == n_periods
                    period_demands[i, t] = remaining
                else
                    period_demands[i, t] = rand(0:max(0, remaining ÷ (n_periods - t + 1)))
                    remaining -= period_demands[i, t]
                end
            end
        end

    elseif variant == cut_min_runs
        # Minimum production runs per pattern
        min_runs = rand(5:20)

        if feasibility_status == infeasible
            min_runs = sum(demands) + 100  # Impossibly high
        end
    end

    return CuttingStockProblem(
        piece_lengths, demands, patterns, stock_length, stock_limit,
        variant,
        n_stock_types, stock_lengths_arr, stock_costs, patterns_by_stock,
        setup_costs,
        max_trim_fraction,
        n_periods, period_demands,
        min_runs
    )
end

"""
Adjust demands and constraints to achieve desired feasibility with mathematical guarantees
"""
function adjust_for_feasibility(base_demands, piece_lengths, stock_length,
                               common_lengths, target_feasible, waste_factor, max_patterns)

    if target_feasible
        # FEASIBLE: Create realistic, solvable manufacturing scenarios
        patterns = generate_cutting_patterns(stock_length, piece_lengths, max_patterns, waste_factor)
        demands = copy(base_demands)

        for i in 1:length(demands)
            variation = rand(Uniform(0.8, 1.2))
            demands[i] = max(1, round(Int, demands[i] * variation))
        end

        stock_limit = 0  # No limit

        return demands, stock_limit, patterns

    else
        # INFEASIBLE: Create mathematically guaranteed infeasible scenarios
        scenarios = [:rush_order, :seasonal_spike, :backlog_clearing, :mixed]
        scenario = rand(scenarios)

        infeasibility_method = rand(1:3)

        if infeasibility_method == 1
            # Method 1: Single piece impossibility
            patterns = generate_cutting_patterns(stock_length, piece_lengths, max_patterns, waste_factor)
            target_piece_idx = argmin(piece_lengths)

            reasonable_stock_limit = round(Int, sum(base_demands) * rand(Uniform(0.8, 1.2)))

            best_efficiency = 0
            for pattern in patterns
                if pattern[target_piece_idx] > 0
                    best_efficiency = max(best_efficiency, pattern[target_piece_idx])
                end
            end
            if best_efficiency == 0
                best_efficiency = floor(Int, stock_length / piece_lengths[target_piece_idx])
            end

            max_possible_target = best_efficiency * reasonable_stock_limit

            scaling_factors = calculate_demand_scaling_factors(piece_lengths, common_lengths, scenario)
            demands = [max(1, round(Int, d * f)) for (d, f) in zip(base_demands, scaling_factors)]
            demands[target_piece_idx] = max_possible_target + round(Int, max_possible_target * rand(Uniform(0.3, 0.8)))

            stock_limit = reasonable_stock_limit

        elseif infeasibility_method == 2
            # Method 2: No-pattern scenario
            target_piece_idx = rand(1:length(piece_lengths))

            patterns = []
            for (i, piece_length) in enumerate(piece_lengths)
                if i != target_piece_idx
                    pattern = zeros(Int, length(piece_lengths))
                    pattern[i] = max(1, floor(Int, stock_length / piece_length))
                    push!(patterns, pattern)
                end
            end

            while length(patterns) < max(5, length(piece_lengths) - 1)
                pattern = zeros(Int, length(piece_lengths))
                available_indices = [i for i in 1:length(piece_lengths) if i != target_piece_idx]
                for idx in available_indices
                    if rand() < 0.3
                        max_fit = floor(Int, stock_length / piece_lengths[idx])
                        pattern[idx] = rand(1:max(1, max_fit))
                    end
                end
                if sum(pattern) > 0
                    push!(patterns, pattern)
                end
            end

            scaling_factors = calculate_demand_scaling_factors(piece_lengths, common_lengths, scenario)
            demands = [max(1, round(Int, d * f)) for (d, f) in zip(base_demands, scaling_factors)]
            demands[target_piece_idx] = max(1, round(Int, base_demands[target_piece_idx] * scaling_factors[target_piece_idx]))

            stock_limit = 0

        else
            # Method 3: Combined stock+demand contradiction
            patterns = generate_cutting_patterns(stock_length, piece_lengths, max_patterns, waste_factor)

            min_total_stock_needed = 0
            for i in 1:length(piece_lengths)
                best_efficiency = 0
                for pattern in patterns
                    if pattern[i] > 0
                        best_efficiency = max(best_efficiency, pattern[i])
                    end
                end
                if best_efficiency == 0
                    best_efficiency = floor(Int, stock_length / piece_lengths[i])
                end
                min_stock_for_piece_i = ceil(base_demands[i] / best_efficiency)
                min_total_stock_needed = max(min_total_stock_needed, min_stock_for_piece_i)
            end

            stock_limit = max(1, round(Int, min_total_stock_needed * rand(Uniform(0.4, 0.7))))

            scaling_factors = calculate_demand_scaling_factors(piece_lengths, common_lengths, scenario)
            enhanced_factors = [f * rand(Uniform(1.2, 1.8)) for f in scaling_factors]
            demands = [max(1, round(Int, d * f)) for (d, f) in zip(base_demands, enhanced_factors)]
        end

        return demands, stock_limit, patterns
    end
end

"""
Calculate realistic demand scaling based on business scenarios
"""
function calculate_demand_scaling_factors(piece_lengths, common_lengths, scenario::Symbol)
    n_pieces = length(piece_lengths)
    scaling_factors = ones(Float64, n_pieces)

    if scenario == :rush_order
        for i in 1:n_pieces
            if piece_lengths[i] in common_lengths
                scaling_factors[i] = rand(Uniform(2.0, 4.0))
            else
                scaling_factors[i] = rand(Uniform(0.8, 1.5))
            end
        end
    elseif scenario == :seasonal_spike
        base_spike = rand(Uniform(1.8, 2.5))
        for i in 1:n_pieces
            scaling_factors[i] = base_spike * rand(Uniform(0.8, 1.2))
        end
    elseif scenario == :backlog_clearing
        for i in 1:n_pieces
            scaling_factors[i] = rand(Uniform(2.2, 3.5))
        end
    else  # :mixed
        for i in 1:n_pieces
            scaling_factors[i] = rand(Uniform(1.5, 2.8))
        end
    end

    return scaling_factors
end

"""
Helper function to generate feasible cutting patterns
"""
function generate_cutting_patterns(standard_length, piece_lengths, max_patterns, waste_factor=0.1)
    patterns = Vector{Vector{Int}}()

    # Single-piece patterns
    for (i, piece_length) in enumerate(piece_lengths)
        pattern = zeros(Int, length(piece_lengths))
        pattern[i] = floor(Int, standard_length / piece_length)
        push!(patterns, pattern)
    end

    # Complex patterns
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
    build_model(prob::CuttingStockProblem)

Build a JuMP model for the cutting stock problem based on its variant.
"""
function build_model(prob::CuttingStockProblem)
    model = Model()

    n_patterns = length(prob.patterns)
    n_pieces = length(prob.piece_lengths)

    if prob.variant == cut_standard || prob.variant == cut_trim_limit || prob.variant == cut_min_runs
        @variable(model, x[1:n_patterns] >= 0)

        @objective(model, Min, sum(x))

        # Meet demand for each piece size
        for i in 1:n_pieces
            @constraint(model, sum(prob.patterns[j][i] * x[j] for j in 1:n_patterns) >= prob.demands[i])
        end

        # Stock limit constraint if specified
        if prob.stock_limit > 0
            @constraint(model, sum(x) <= prob.stock_limit)
        end

        # Variant-specific constraints
        if prob.variant == cut_trim_limit && prob.max_trim_fraction > 0
            # Total stock used
            total_stock_used = @expression(model, sum(prob.stock_length * x[j] for j in 1:n_patterns))

            # Total material in pieces
            total_pieces_length = @expression(model,
                sum(prob.patterns[j][i] * prob.piece_lengths[i] * x[j]
                    for j in 1:n_patterns, i in 1:n_pieces))

            # Trim loss constraint
            @constraint(model, total_stock_used - total_pieces_length <= prob.max_trim_fraction * total_stock_used)

        elseif prob.variant == cut_min_runs && prob.min_runs > 0
            # Binary variable for pattern use
            @variable(model, y[1:n_patterns], Bin)
            M = sum(prob.demands) * 10  # Big-M

            for j in 1:n_patterns
                @constraint(model, x[j] <= M * y[j])
                @constraint(model, x[j] >= prob.min_runs * y[j])
            end
        end

    elseif prob.variant == cut_multi_stock
        # Multiple stock sizes
        n_stock = prob.n_stock_types
        @variable(model, x[1:n_stock, 1:maximum(length.(prob.patterns_by_stock))] >= 0)

        # Minimize cost
        @objective(model, Min, sum(
            prob.stock_costs[s] * x[s, j]
            for s in 1:n_stock for j in 1:length(prob.patterns_by_stock[s])
        ))

        # Meet demand
        for i in 1:n_pieces
            @constraint(model, sum(
                prob.patterns_by_stock[s][j][i] * x[s, j]
                for s in 1:n_stock for j in 1:length(prob.patterns_by_stock[s])
            ) >= prob.demands[i])
        end

        # Stock limit
        if prob.stock_limit > 0
            @constraint(model, sum(
                x[s, j] for s in 1:n_stock for j in 1:length(prob.patterns_by_stock[s])
            ) <= prob.stock_limit)
        end

    elseif prob.variant == cut_setup_cost
        @variable(model, x[1:n_patterns] >= 0)
        @variable(model, y[1:n_patterns], Bin)  # Pattern used

        # Minimize stock + setup costs
        @objective(model, Min, sum(x) + sum(prob.setup_costs[j] * y[j] for j in 1:n_patterns))

        # Meet demand
        for i in 1:n_pieces
            @constraint(model, sum(prob.patterns[j][i] * x[j] for j in 1:n_patterns) >= prob.demands[i])
        end

        # Link x and y
        M = sum(prob.demands) * 10
        for j in 1:n_patterns
            @constraint(model, x[j] <= M * y[j])
        end

        # Stock limit
        if prob.stock_limit > 0
            @constraint(model, sum(x) <= prob.stock_limit)
        end

    elseif prob.variant == cut_due_dates
        # Multi-period cutting
        n_periods = prob.n_periods
        @variable(model, x[1:n_patterns, 1:n_periods] >= 0)
        @variable(model, inventory[1:n_pieces, 0:n_periods] >= 0)

        # Minimize total stock
        @objective(model, Min, sum(x))

        # Initial inventory is 0
        for i in 1:n_pieces
            @constraint(model, inventory[i, 0] == 0)
        end

        # Inventory balance and demand satisfaction
        for i in 1:n_pieces
            for t in 1:n_periods
                production = sum(prob.patterns[j][i] * x[j, t] for j in 1:n_patterns)
                @constraint(model, inventory[i, t-1] + production == prob.period_demands[i, t] + inventory[i, t])
            end
        end

        # Stock limit per period
        if prob.stock_limit > 0
            for t in 1:n_periods
                @constraint(model, sum(x[j, t] for j in 1:n_patterns) <= prob.stock_limit / n_periods)
            end
        end
    end

    return model
end

# Register the problem type
register_problem(
    :cutting_stock,
    CuttingStockProblem,
    "Cutting stock problem with variants including standard, multi-stock, setup cost, trim limit, due dates, and minimum runs"
)
