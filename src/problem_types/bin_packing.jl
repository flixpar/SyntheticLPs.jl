using JuMP
using Random
using Distributions
using StatsBase

"""
    BinPackingProblem <: ProblemGenerator

Generator for bin packing problems that minimize the number of bins needed to pack items.

# Fields
- `n_items::Int`: Number of items to pack
- `n_bins::Int`: Maximum number of bins available (upper bound U)
- `item_sizes::Vector{Float64}`: Size/weight of each item
- `bin_capacity::Float64`: Capacity of each bin
- `item_categories::Vector{Int}`: Category for each item (for incompatibility)
- `incompatible_pairs::Vector{Tuple{Int,Int}}`: Pairs of incompatible categories
"""
struct BinPackingProblem <: ProblemGenerator
    n_items::Int
    n_bins::Int
    item_sizes::Vector{Float64}
    bin_capacity::Float64
    item_categories::Vector{Int}
    incompatible_pairs::Vector{Tuple{Int,Int}}
end

"""
    BinPackingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a bin packing problem instance with realistic item size distributions.

# Arguments
- `target_variables`: Target number of variables (n_items × n_bins + n_bins)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function BinPackingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem scale and parameters
    if target_variables <= 100
        min_items, max_items = 5, 20
        min_bins, max_bins = 3, 10
        capacity_base = rand(Uniform(50.0, 100.0))
        size_alpha, size_beta = 2.0, 5.0  # Beta distribution parameters (skewed toward small)
        common_sizes_prob = 0.4
        incompatibility_prob = 0.2
    elseif target_variables <= 1000
        min_items, max_items = 20, 100
        min_bins, max_bins = 10, 40
        capacity_base = rand(Uniform(100.0, 500.0))
        size_alpha, size_beta = 2.0, 6.0
        common_sizes_prob = 0.5
        incompatibility_prob = 0.3
    else
        min_items, max_items = 100, 500
        min_bins, max_bins = 40, 200
        capacity_base = rand(Uniform(500.0, 2000.0))
        size_alpha, size_beta = 2.0, 7.0
        common_sizes_prob = 0.6
        incompatibility_prob = 0.4
    end

    # Find optimal n_items and n_bins to match target_variables
    # target_variables ≈ n_items * n_bins + n_bins = n_bins * (n_items + 1)
    best_n_items = min_items
    best_n_bins = min_bins
    best_error = Inf

    for n_bins in min_bins:max_bins
        n_items_exact = (target_variables / n_bins) - 1

        if n_items_exact >= min_items && n_items_exact <= max_items
            n_items = round(Int, n_items_exact)
            actual_vars = n_items * n_bins + n_bins
            error = abs(actual_vars - target_variables) / target_variables

            if error < best_error
                best_error = error
                best_n_items = n_items
                best_n_bins = n_bins
            end
        end
    end

    # If we can't get within 10%, try alternative approach
    if best_error > 0.1
        # Use approximation: n_bins ≈ sqrt(target_variables)
        n_bins_approx = max(min_bins, min(max_bins, round(Int, sqrt(target_variables / 2))))
        n_items_approx = max(min_items, min(max_items, round(Int, (target_variables / n_bins_approx) - 1)))

        best_n_items = n_items_approx
        best_n_bins = n_bins_approx
    end

    n_items = best_n_items
    n_bins_upper_bound = best_n_bins

    # Generate realistic item sizes using Beta distribution (skewed toward smaller items)
    item_sizes = Float64[]

    # Common standardized sizes (boxes, packages)
    common_sizes_base = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    common_sizes = [s * capacity_base for s in common_sizes_base]

    for i in 1:n_items
        if rand() < common_sizes_prob && !isempty(common_sizes)
            # Use a common standardized size with small variation
            base_size = rand(common_sizes)
            variation = rand(Normal(0, 0.02 * base_size))
            size = clamp(base_size + variation, 0.05 * capacity_base, 0.9 * capacity_base)
        else
            # Use Beta distribution for varied sizes
            beta_val = rand(Beta(size_alpha, size_beta))
            # Scale to reasonable range (5% to 90% of capacity)
            size = 0.05 * capacity_base + beta_val * (0.85 * capacity_base)
        end
        push!(item_sizes, round(size, digits=2))
    end

    # Generate item categories for incompatibility constraints
    n_categories = max(2, min(5, round(Int, sqrt(n_items))))
    item_categories = [rand(1:n_categories) for _ in 1:n_items]

    # Generate incompatible category pairs
    incompatible_pairs = Tuple{Int,Int}[]
    if rand() < incompatibility_prob && n_categories >= 2
        n_incompatible = rand(1:min(3, div(n_categories * (n_categories - 1), 2)))
        all_pairs = [(i, j) for i in 1:n_categories for j in (i+1):n_categories]
        if !isempty(all_pairs)
            selected_pairs = sample(all_pairs, min(n_incompatible, length(all_pairs)), replace=false)
            incompatible_pairs = selected_pairs
        end
    end

    # Calculate total item volume
    total_item_volume = sum(item_sizes)

    # Determine feasibility and adjust parameters
    solution_status = feasibility_status == feasible ? :feasible :
                     feasibility_status == infeasible ? :infeasible :
                     :unknown

    if solution_status == :feasible
        # Ensure feasibility: total capacity must exceed total volume
        # Use first-fit decreasing theoretical bound: at most ⌈1.7 * OPT⌉
        min_bins_needed = ceil(Int, total_item_volume / capacity_base)

        # Add buffer for safety (10-20%)
        safety_factor = 1.1 + rand() * 0.1
        n_bins_upper_bound = max(n_bins_upper_bound, ceil(Int, min_bins_needed * safety_factor))

        # Verify by greedy simulation
        bins_needed = simulate_first_fit_decreasing(item_sizes, capacity_base)
        if bins_needed > n_bins_upper_bound
            n_bins_upper_bound = bins_needed + rand(1:3)
        end

    elseif solution_status == :infeasible
        # Create guaranteed infeasibility
        infeasibility_method = rand(1:2)

        if infeasibility_method == 1
            # Method 1: Insufficient total capacity
            min_bins_needed = ceil(Int, total_item_volume / capacity_base)
            # Set upper bound below minimum needed
            reduction_factor = rand(Uniform(0.5, 0.9))
            n_bins_upper_bound = max(1, floor(Int, min_bins_needed * reduction_factor))

        else
            # Method 2: Add oversized items that make packing impossible
            # Add a few items that together exceed capacity with incompatibility
            n_oversized = rand(2:min(4, max(2, div(n_items, 5))))
            oversized_category = n_categories + 1

            for _ in 1:n_oversized
                # Items that are 60-80% of capacity
                oversized_item = rand(Uniform(0.6, 0.8)) * capacity_base
                push!(item_sizes, round(oversized_item, digits=2))
                push!(item_categories, oversized_category)
                n_items += 1
            end

            # Make all oversized items mutually incompatible
            for i in 1:(n_oversized-1)
                for j in (i+1):n_oversized
                    push!(incompatible_pairs, (oversized_category, oversized_category))
                end
            end

            # Recalculate bins needed
            total_item_volume = sum(item_sizes)
            min_bins_needed = n_oversized + ceil(Int, (total_item_volume - n_oversized * 0.7 * capacity_base) / capacity_base)
            n_bins_upper_bound = max(1, min_bins_needed - rand(1:max(1, div(n_oversized, 2))))
        end

    else  # :unknown
        # Random capacity ratio
        min_bins_needed = ceil(Int, total_item_volume / capacity_base)
        ratio = 0.9 + rand() * 0.3  # 0.9 to 1.2
        n_bins_upper_bound = max(1, round(Int, min_bins_needed * ratio))
    end

    return BinPackingProblem(n_items, n_bins_upper_bound, item_sizes, capacity_base,
                            item_categories, incompatible_pairs)
end

"""
    simulate_first_fit_decreasing(item_sizes::Vector{Float64}, capacity::Float64)

Simulate first-fit decreasing heuristic to estimate bins needed.
"""
function simulate_first_fit_decreasing(item_sizes::Vector{Float64}, capacity::Float64)
    sorted_items = sort(item_sizes, rev=true)
    bins = Float64[]

    for item in sorted_items
        placed = false
        for (i, bin_space) in enumerate(bins)
            if bin_space + item <= capacity
                bins[i] += item
                placed = true
                break
            end
        end
        if !placed
            push!(bins, item)
        end
    end

    return length(bins)
end

"""
    build_model(prob::BinPackingProblem)

Build a JuMP model for the bin packing problem.

# Arguments
- `prob`: BinPackingProblem instance

# Returns
- `model`: The JuMP model

# Formulation
- Variables: X[i,j] (item i in bin j), Y[j] (bin j used)
- Objective: Minimize number of bins used
- Constraints: Each item assigned once, capacity limits, incompatibility
"""
function build_model(prob::BinPackingProblem)
    model = Model()

    n_items = prob.n_items
    n_bins = prob.n_bins

    # Variables
    @variable(model, x[1:n_items, 1:n_bins], Bin)
    @variable(model, y[1:n_bins], Bin)

    # Objective: minimize number of bins used
    @objective(model, Min, sum(y[j] for j in 1:n_bins))

    # Each item must be assigned to exactly one bin
    for i in 1:n_items
        @constraint(model, sum(x[i, j] for j in 1:n_bins) == 1)
    end

    # Bin capacity constraints
    for j in 1:n_bins
        @constraint(model, sum(prob.item_sizes[i] * x[i, j] for i in 1:n_items) <= prob.bin_capacity * y[j])
    end

    # Linking constraints: items can only go in used bins
    for i in 1:n_items
        for j in 1:n_bins
            @constraint(model, x[i, j] <= y[j])
        end
    end

    # Incompatibility constraints
    if !isempty(prob.incompatible_pairs)
        for j in 1:n_bins
            for (cat1, cat2) in prob.incompatible_pairs
                # Items from incompatible categories cannot be in same bin
                items_cat1 = [i for i in 1:n_items if prob.item_categories[i] == cat1]
                items_cat2 = [i for i in 1:n_items if prob.item_categories[i] == cat2]

                if !isempty(items_cat1) && !isempty(items_cat2)
                    @constraint(model,
                        sum(x[i, j] for i in items_cat1) +
                        sum(x[i, j] for i in items_cat2) <= 1
                    )
                end
            end
        end
    end

    return model
end

# Register the problem type
register_problem(
    :bin_packing,
    BinPackingProblem,
    "Bin packing problem that minimizes the number of bins needed to pack items with realistic size distributions and optional incompatibility constraints"
)
