using JuMP
using Random
using Distributions
using StatsBase

"""
    BinPackingProblem <: ProblemGenerator

Generator for bin packing problems that minimize the number of bins needed to pack
items, with realistic item-size distributions and category-conflict constraints.

# Overview
Models the classic bin packing problem. The decisions are binary assignments of
items to bins, binary bin-usage indicators, and per-bin per-category presence
indicators. The objective minimizes the number of bins used. Each item is assigned
to exactly one bin, each bin's packed size cannot exceed its capacity, items may
only occupy used bins, and pairs of conflicting categories may not co-occur in the
same bin (enforced through the presence indicators, NOT by summing raw assignment
variables — so two items of the *same* category may freely share a bin).

# Fields
- `n_items::Int`: Number of items to pack
- `n_bins::Int`: Maximum number of bins available (upper bound)
- `n_categories::Int`: Number of item categories
- `item_sizes::Vector{Float64}`: Size/weight of each item
- `bin_capacity::Float64`: Capacity of each bin
- `item_categories::Vector{Int}`: Category index (1..n_categories) for each item
- `incompatible_pairs::Vector{Tuple{Int,Int}}`: Pairs of conflicting categories
"""
struct BinPackingProblem <: ProblemGenerator
    n_items::Int
    n_bins::Int
    n_categories::Int
    item_sizes::Vector{Float64}
    bin_capacity::Float64
    item_categories::Vector{Int}
    incompatible_pairs::Vector{Tuple{Int,Int}}
end

"""
    simulate_first_fit_decreasing(item_sizes, capacity, item_categories, incompatible_pairs)

Greedy first-fit-decreasing heuristic that respects category conflicts. Returns the
number of bins used by the heuristic, giving a constructive upper bound on the bins
required for a conflict-respecting packing.
"""
function simulate_first_fit_decreasing(
    item_sizes::Vector{Float64},
    capacity::Float64,
    item_categories::Vector{Int},
    incompatible_pairs::Vector{Tuple{Int,Int}},
)
    # Build conflict lookup over categories
    conflicts = Set{Tuple{Int,Int}}()
    for (a, b) in incompatible_pairs
        push!(conflicts, (min(a, b), max(a, b)))
    end
    function cat_conflicts(c1::Int, c2::Int)
        return (min(c1, c2), max(c1, c2)) in conflicts
    end

    order = sortperm(item_sizes, rev=true)
    bin_loads = Float64[]
    bin_cats = Vector{Set{Int}}()

    for idx in order
        item = item_sizes[idx]
        cat = item_categories[idx]
        placed = false
        for b in eachindex(bin_loads)
            # Capacity check
            bin_loads[b] + item > capacity && continue
            # Conflict check against categories already in the bin
            if any(cat_conflicts(cat, c) for c in bin_cats[b])
                continue
            end
            bin_loads[b] += item
            push!(bin_cats[b], cat)
            placed = true
            break
        end
        if !placed
            push!(bin_loads, item)
            push!(bin_cats, Set{Int}([cat]))
        end
    end

    return length(bin_loads)
end

"""
    BinPackingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a bin packing problem instance with realistic item size distributions.

Variable count = n_bins * (n_items + 1 + n_categories), summing the assignment
binaries x[i,j], the bin-usage binaries y[j], and the per-bin per-category presence
binaries p[c,j]. Dimensions are sized so this total lands near `target_variables`.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function BinPackingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem scale and parameters
    if target_variables <= 100
        min_items, max_items = 5, 20
        min_bins, max_bins = 3, 12
        capacity_base = rand(Uniform(50.0, 100.0))
        size_alpha, size_beta = 2.0, 5.0  # Beta distribution params (skewed small)
        common_sizes_prob = 0.4
        incompatibility_prob = 0.3
    elseif target_variables <= 1000
        min_items, max_items = 20, 100
        min_bins, max_bins = 8, 40
        capacity_base = rand(Uniform(100.0, 500.0))
        size_alpha, size_beta = 2.0, 6.0
        common_sizes_prob = 0.5
        incompatibility_prob = 0.4
    else
        min_items, max_items = 100, 500
        min_bins, max_bins = 30, 200
        capacity_base = rand(Uniform(500.0, 2000.0))
        size_alpha, size_beta = 2.0, 7.0
        common_sizes_prob = 0.6
        incompatibility_prob = 0.5
    end

    # Var count = n_bins * (n_items + 1 + n_categories). Crucially, n_bins is NOT a
    # free dimension: the feasibility/unknown logic below sets it to the actual
    # packing requirement (~ density * n_items, where density is the mean item fill
    # fraction). Sizing n_items against a freely-chosen n_bins therefore overshoots
    # the target badly. Instead we estimate the packing density analytically and
    # size n_items so that (density-implied n_bins) * (n_items + 1 + cats) ≈ target.
    function est_categories(n_items)
        return max(2, min(5, round(Int, sqrt(n_items))))
    end

    # Expected item fill fraction (mean over the common-size and Beta paths).
    mean_common = 0.27  # mean of common_sizes_base
    mean_beta = size_alpha / (size_alpha + size_beta)
    mean_fraction = common_sizes_prob * mean_common +
                    (1 - common_sizes_prob) * (0.05 + mean_beta * 0.85)
    # First-fit-decreasing packs at ~88% fill, and the natural (unknown/feasible)
    # bin count is ~1.05x the ideal ceil(volume/capacity). Bins per item:
    bins_per_item = mean_fraction / 0.88 * 1.05
    # Only an upper sanity cap here; the lower bin count follows naturally from the
    # item count so a target just above a band threshold is not forced to overshoot.
    est_bins(n) = clamp(round(Int, bins_per_item * n), 1, max_bins)

    # Search a wide item range (down to a small floor) so the closed-form optimum
    # n_items ≈ sqrt(target / bins_per_item) is reachable regardless of band floors.
    lo = max(3, min(min_items, round(Int, sqrt(target_variables / bins_per_item) / 2)))
    best_n_items = lo
    best_error = Inf
    for n in lo:max_items
        cats = est_categories(n)
        actual_vars = est_bins(n) * (n + 1 + cats)
        err = abs(actual_vars - target_variables) / target_variables
        if err < best_error
            best_error = err
            best_n_items = n
        end
    end

    n_items = best_n_items
    n_bins_upper_bound = est_bins(n_items)

    # Generate realistic item sizes using Beta distribution (skewed toward small)
    item_sizes = Float64[]
    common_sizes_base = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    common_sizes = [s * capacity_base for s in common_sizes_base]

    for i in 1:n_items
        if rand() < common_sizes_prob && !isempty(common_sizes)
            base_size = rand(common_sizes)
            variation = rand(Normal(0, 0.02 * base_size))
            sz = clamp(base_size + variation, 0.05 * capacity_base, 0.9 * capacity_base)
        else
            beta_val = rand(Beta(size_alpha, size_beta))
            sz = 0.05 * capacity_base + beta_val * (0.85 * capacity_base)
        end
        push!(item_sizes, round(sz, digits=2))
    end

    # Item categories for conflict constraints
    n_categories = max(2, min(5, round(Int, sqrt(n_items))))
    item_categories = [rand(1:n_categories) for _ in 1:n_items]

    # Conflicting (distinct) category pairs
    incompatible_pairs = Tuple{Int,Int}[]
    if rand() < incompatibility_prob && n_categories >= 2
        all_pairs = [(i, j) for i in 1:n_categories for j in (i + 1):n_categories]
        if !isempty(all_pairs)
            n_incompatible = rand(1:min(3, length(all_pairs)))
            incompatible_pairs = collect(sample(all_pairs, n_incompatible, replace=false))
        end
    end

    # Resolve feasibility status (unknown -> natural instance, no forced contradiction)
    actual_status = feasibility_status

    total_item_volume = sum(item_sizes)

    if actual_status == feasible
        # Constructive feasibility: use a conflict-respecting FFD packing to find how
        # many bins are actually needed, then provide at least that many (plus buffer).
        bins_needed = simulate_first_fit_decreasing(
            item_sizes, capacity_base, item_categories, incompatible_pairs
        )
        # Small buffer so a feasible packing exists without materially inflating the
        # variable count above the target.
        safety = rand(0:1)
        n_bins_upper_bound = max(n_bins_upper_bound, bins_needed + safety)

    elseif actual_status == infeasible
        # Preserve the target-sized dimensions and force a deterministic
        # contradiction that survives LP relaxation: total item volume exceeds
        # the aggregate capacity of all bins. Summing the bin-capacity rows gives
        # sum(item_sizes) <= n_bins * capacity, while the assignment equalities
        # force every item to be packed exactly once.
        capacity_base = total_item_volume / (n_bins_upper_bound * rand(Uniform(1.10, 1.30)))

    else  # unknown: natural instance, capacity near the requirement
        min_bins_needed = ceil(Int, total_item_volume / capacity_base)
        ratio = 0.9 + rand() * 0.4  # 0.9 to 1.3
        n_bins_upper_bound = max(1, round(Int, min_bins_needed * ratio))
    end

    return BinPackingProblem(
        n_items, n_bins_upper_bound, n_categories, item_sizes, capacity_base,
        item_categories, incompatible_pairs
    )
end

"""
    build_model(prob::BinPackingProblem)

Build a JuMP model for the bin packing problem. Completely deterministic — uses only
the struct's fields.

# Formulation
- Variables: x[i,j] (item i in bin j), y[j] (bin j used), p[c,j] (category c present
  in bin j).
- Objective: minimize the number of bins used.
- Constraints: each item assigned exactly once; per-bin capacity (linked to y);
  items only in used bins; category presence linking x[i,j] <= p[cat(i),j]; and
  conflict bans p[c1,j] + p[c2,j] <= 1 for each conflicting category pair.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::BinPackingProblem)
    model = Model()

    n_items = prob.n_items
    n_bins = prob.n_bins
    n_categories = prob.n_categories

    # Variables: total = n_bins * (n_items + 1 + n_categories)
    @variable(model, x[1:n_items, 1:n_bins], Bin)
    @variable(model, y[1:n_bins], Bin)
    @variable(model, p[1:n_categories, 1:n_bins], Bin)

    # Objective: minimize number of bins used
    @objective(model, Min, sum(y[j] for j in 1:n_bins))

    # Each item assigned to exactly one bin
    for i in 1:n_items
        @constraint(model, sum(x[i, j] for j in 1:n_bins) == 1)
    end

    # Bin capacity (and bin must be open to hold anything)
    for j in 1:n_bins
        @constraint(model, sum(prob.item_sizes[i] * x[i, j] for i in 1:n_items) <= prob.bin_capacity * y[j])
    end

    # Items can only go in used bins
    for i in 1:n_items
        for j in 1:n_bins
            @constraint(model, x[i, j] <= y[j])
        end
    end

    # Category presence linking: if item i is in bin j, its category is present in j
    for i in 1:n_items
        c = prob.item_categories[i]
        for j in 1:n_bins
            @constraint(model, x[i, j] <= p[c, j])
        end
    end

    # Conflict bans: two conflicting categories cannot both be present in a bin
    for (cat1, cat2) in prob.incompatible_pairs
        for j in 1:n_bins
            @constraint(model, p[cat1, j] + p[cat2, j] <= 1)
        end
    end

    return model
end

# Register the variant
register_variant(
    :bin_packing,
    :standard,
    BinPackingProblem,
    "Bin packing that minimizes bins used, with realistic item sizes and true category-conflict constraints",
)
