using JuMP
using Random
using Distributions

"""
Knapsack problem variants.

# Variants
- `knap_standard`: Basic fractional knapsack maximizing value under weight constraint
- `knap_multi_dim`: Multi-dimensional knapsack with multiple resource constraints
- `knap_bounded`: Bounded knapsack with upper limits on item quantities
- `knap_multiple`: Multiple knapsacks with item assignment
- `knap_conflict`: Items with conflict constraints (cannot select both)
- `knap_dependency`: Item dependencies (selecting i requires selecting j)
- `knap_group`: Group constraints (min/max items from each group)
- `knap_min_fill`: Minimum fill requirement (knapsack must be X% full)
"""
@enum KnapsackVariant begin
    knap_standard
    knap_multi_dim
    knap_bounded
    knap_multiple
    knap_conflict
    knap_dependency
    knap_group
    knap_min_fill
end

"""
    KnapsackProblem <: ProblemGenerator

Generator for knapsack problems with multiple variants.

# Fields
- `n_items::Int`: Number of items
- `capacity::Int`: Knapsack capacity (primary resource)
- `values::Vector{Int}`: Value of each item
- `weights::Vector{Int}`: Weight of each item
- `variant::KnapsackVariant`: Problem variant
# Multi-dimensional variant
- `n_resources::Int`: Number of resource dimensions
- `resource_usage::Matrix{Float64}`: Usage of each resource by each item
- `resource_capacities::Vector{Float64}`: Capacity for each resource
# Bounded variant
- `upper_bounds::Vector{Int}`: Maximum quantity of each item
# Multiple knapsacks variant
- `n_knapsacks::Int`: Number of knapsacks
- `knapsack_capacities::Vector{Int}`: Capacity of each knapsack
# Conflict variant
- `conflicts::Vector{Tuple{Int,Int}}`: Pairs of conflicting items
# Dependency variant
- `dependencies::Vector{Tuple{Int,Int}}`: (i,j) means item i requires item j
# Group variant
- `n_groups::Int`: Number of item groups
- `item_groups::Vector{Int}`: Group assignment for each item
- `group_min::Vector{Int}`: Minimum items from each group
- `group_max::Vector{Int}`: Maximum items from each group
# Min fill variant
- `min_fill_fraction::Float64`: Minimum fraction of capacity that must be used
"""
struct KnapsackProblem <: ProblemGenerator
    n_items::Int
    capacity::Int
    values::Vector{Int}
    weights::Vector{Int}
    variant::KnapsackVariant
    # Multi-dimensional
    n_resources::Int
    resource_usage::Union{Matrix{Float64}, Nothing}
    resource_capacities::Union{Vector{Float64}, Nothing}
    # Bounded
    upper_bounds::Union{Vector{Int}, Nothing}
    # Multiple knapsacks
    n_knapsacks::Int
    knapsack_capacities::Union{Vector{Int}, Nothing}
    # Conflict
    conflicts::Union{Vector{Tuple{Int,Int}}, Nothing}
    # Dependency
    dependencies::Union{Vector{Tuple{Int,Int}}, Nothing}
    # Group
    n_groups::Int
    item_groups::Union{Vector{Int}, Nothing}
    group_min::Union{Vector{Int}, Nothing}
    group_max::Union{Vector{Int}, Nothing}
    # Min fill
    min_fill_fraction::Float64
end

# Backwards compatibility constructor
function KnapsackProblem(n_items::Int, capacity::Int, values::Vector{Int}, weights::Vector{Int})
    KnapsackProblem(
        n_items, capacity, values, weights, knap_standard,
        0, nothing, nothing,
        nothing,
        0, nothing,
        nothing,
        nothing,
        0, nothing, nothing, nothing,
        0.0
    )
end

"""
    KnapsackProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                    variant::KnapsackVariant=knap_standard)

Construct a knapsack problem instance with the specified variant.
"""
function KnapsackProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                         variant::KnapsackVariant=knap_standard)
    Random.seed!(seed)

    n_items = target_variables

    # Scale value and weight ranges based on problem size
    if target_variables <= 100
        value_range = (rand(5:20), rand(80:150))
        weight_range = (rand(3:8), rand(15:25))
    elseif target_variables <= 1000
        value_range = (rand(10:30), rand(100:300))
        weight_range = (rand(5:15), rand(20:40))
    else
        value_range = (rand(20:50), rand(200:500))
        weight_range = (rand(10:25), rand(30:60))
    end

    min_value, max_value = value_range
    values = rand(min_value:max_value, n_items)

    min_weight, max_weight = weight_range
    weights = rand(min_weight:max_weight, n_items)

    total_weight = sum(weights)

    # Base capacity
    capacity_ratio = rand(Uniform(0.3, 0.7))
    capacity = round(Int, total_weight * capacity_ratio)
    capacity = max(1, capacity)

    # Initialize variant fields
    n_resources = 0
    resource_usage = nothing
    resource_capacities = nothing
    upper_bounds = nothing
    n_knapsacks = 0
    knapsack_capacities = nothing
    conflicts = nothing
    dependencies = nothing
    n_groups = 0
    item_groups = nothing
    group_min = nothing
    group_max = nothing
    min_fill_fraction = 0.0

    if variant == knap_multi_dim
        # Multiple resource dimensions (weight, volume, etc.)
        n_resources = rand(2:min(5, max(2, n_items ÷ 10)))
        resource_usage = zeros(n_items, n_resources)

        for i in 1:n_items
            resource_usage[i, 1] = Float64(weights[i])  # First resource is weight
            for r in 2:n_resources
                # Other resources with varying usage
                resource_usage[i, r] = rand(Uniform(1.0, 20.0))
            end
        end

        resource_capacities = zeros(n_resources)
        resource_capacities[1] = Float64(capacity)
        for r in 2:n_resources
            total_r = sum(resource_usage[:, r])
            resource_capacities[r] = total_r * rand(Uniform(0.4, 0.7))
        end

        if feasibility_status == infeasible
            # Make one resource very tight
            tight_r = rand(1:n_resources)
            resource_capacities[tight_r] *= 0.05
        end

    elseif variant == knap_bounded
        # Upper bounds on item quantities
        upper_bounds = [rand(1:5) for _ in 1:n_items]

        if feasibility_status == infeasible
            # Require total quantity that exceeds capacity
            capacity = round(Int, sum(weights .* upper_bounds) * 1.5)
            # But add a constraint requiring at least this much weight
            min_fill_fraction = 1.1
        end

    elseif variant == knap_multiple
        # Multiple knapsacks
        n_knapsacks = rand(2:min(5, max(2, n_items ÷ 5)))
        knapsack_capacities = [round(Int, total_weight * rand(Uniform(0.15, 0.35))) for _ in 1:n_knapsacks]

        if feasibility_status == infeasible
            # Reduce all capacities
            knapsack_capacities = [round(Int, c * 0.1) for c in knapsack_capacities]
            # But we need some min assignment - add min requirements later
        end

    elseif variant == knap_conflict
        # Pairs of conflicting items
        n_conflicts = rand(n_items ÷ 5:n_items ÷ 2)
        conflicts = Tuple{Int,Int}[]

        for _ in 1:n_conflicts
            i = rand(1:n_items)
            j = rand(1:n_items)
            if i != j && !((i, j) in conflicts) && !((j, i) in conflicts)
                push!(conflicts, (i, j))
            end
        end

        if feasibility_status == infeasible
            # Create a conflict cycle that's required
            # E.g., items 1,2,3 all conflict, but we need at least 2 of them
        end

    elseif variant == knap_dependency
        # Item dependencies: (i,j) means selecting item i requires selecting item j
        n_deps = rand(n_items ÷ 5:n_items ÷ 3)
        dependencies = Tuple{Int,Int}[]

        for _ in 1:n_deps
            i = rand(1:n_items)
            j = rand(1:n_items)
            if i != j && !((i, j) in dependencies)
                push!(dependencies, (i, j))
            end
        end

        if feasibility_status == infeasible
            # Create circular dependency
            push!(dependencies, (1, 2))
            push!(dependencies, (2, 1))
        end

    elseif variant == knap_group
        # Group constraints
        n_groups = rand(3:min(8, max(3, n_items ÷ 5)))
        item_groups = rand(1:n_groups, n_items)

        # Count items per group
        group_counts = zeros(Int, n_groups)
        for g in item_groups
            group_counts[g] += 1
        end

        group_min = zeros(Int, n_groups)
        group_max = zeros(Int, n_groups)

        for g in 1:n_groups
            if group_counts[g] > 0
                group_min[g] = rand(0:max(0, group_counts[g] ÷ 3))
                group_max[g] = rand(max(1, group_counts[g] ÷ 2):group_counts[g])
            end
        end

        if feasibility_status == infeasible
            # Make one group min > max
            target_group = rand(1:n_groups)
            group_min[target_group] = group_counts[target_group] + 1
        end

    elseif variant == knap_min_fill
        # Minimum fill requirement
        min_fill_fraction = rand(Uniform(0.3, 0.6))

        if feasibility_status == infeasible
            # Require more than 100% fill
            min_fill_fraction = 1.2
        end
    end

    return KnapsackProblem(
        n_items, capacity, values, weights, variant,
        n_resources, resource_usage, resource_capacities,
        upper_bounds,
        n_knapsacks, knapsack_capacities,
        conflicts,
        dependencies,
        n_groups, item_groups, group_min, group_max,
        min_fill_fraction
    )
end

"""
    build_model(prob::KnapsackProblem)

Build a JuMP model for the knapsack problem based on its variant.
"""
function build_model(prob::KnapsackProblem)
    model = Model()

    if prob.variant == knap_standard
        @variable(model, 0 <= x[1:prob.n_items] <= 1)
        @objective(model, Max, sum(prob.values[i] * x[i] for i in 1:prob.n_items))
        @constraint(model, sum(prob.weights[i] * x[i] for i in 1:prob.n_items) <= prob.capacity)

    elseif prob.variant == knap_multi_dim
        @variable(model, 0 <= x[1:prob.n_items] <= 1)
        @objective(model, Max, sum(prob.values[i] * x[i] for i in 1:prob.n_items))

        # Multi-dimensional resource constraints
        for r in 1:prob.n_resources
            @constraint(model, sum(prob.resource_usage[i, r] * x[i] for i in 1:prob.n_items) <= prob.resource_capacities[r])
        end

    elseif prob.variant == knap_bounded
        # Bounded knapsack with integer quantities
        @variable(model, 0 <= x[i=1:prob.n_items] <= prob.upper_bounds[i], Int)
        @objective(model, Max, sum(prob.values[i] * x[i] for i in 1:prob.n_items))
        @constraint(model, sum(prob.weights[i] * x[i] for i in 1:prob.n_items) <= prob.capacity)

        if prob.min_fill_fraction > 0
            @constraint(model, sum(prob.weights[i] * x[i] for i in 1:prob.n_items) >= prob.capacity * prob.min_fill_fraction)
        end

    elseif prob.variant == knap_multiple
        # Assign items to multiple knapsacks
        @variable(model, 0 <= x[1:prob.n_items, 1:prob.n_knapsacks] <= 1)

        @objective(model, Max, sum(prob.values[i] * sum(x[i, k] for k in 1:prob.n_knapsacks) for i in 1:prob.n_items))

        # Each item can only be assigned to one knapsack (total fraction <= 1)
        for i in 1:prob.n_items
            @constraint(model, sum(x[i, k] for k in 1:prob.n_knapsacks) <= 1)
        end

        # Capacity constraints for each knapsack
        for k in 1:prob.n_knapsacks
            @constraint(model, sum(prob.weights[i] * x[i, k] for i in 1:prob.n_items) <= prob.knapsack_capacities[k])
        end

    elseif prob.variant == knap_conflict
        # Binary knapsack with conflict constraints
        @variable(model, x[1:prob.n_items], Bin)
        @objective(model, Max, sum(prob.values[i] * x[i] for i in 1:prob.n_items))
        @constraint(model, sum(prob.weights[i] * x[i] for i in 1:prob.n_items) <= prob.capacity)

        # Conflict constraints
        if prob.conflicts !== nothing
            for (i, j) in prob.conflicts
                @constraint(model, x[i] + x[j] <= 1)
            end
        end

    elseif prob.variant == knap_dependency
        # Binary knapsack with dependency constraints
        @variable(model, x[1:prob.n_items], Bin)
        @objective(model, Max, sum(prob.values[i] * x[i] for i in 1:prob.n_items))
        @constraint(model, sum(prob.weights[i] * x[i] for i in 1:prob.n_items) <= prob.capacity)

        # Dependency constraints: if x[i] = 1, then x[j] must = 1
        if prob.dependencies !== nothing
            for (i, j) in prob.dependencies
                @constraint(model, x[i] <= x[j])
            end
        end

    elseif prob.variant == knap_group
        # Binary knapsack with group constraints
        @variable(model, x[1:prob.n_items], Bin)
        @objective(model, Max, sum(prob.values[i] * x[i] for i in 1:prob.n_items))
        @constraint(model, sum(prob.weights[i] * x[i] for i in 1:prob.n_items) <= prob.capacity)

        # Group constraints
        if prob.item_groups !== nothing
            for g in 1:prob.n_groups
                items_in_group = [i for i in 1:prob.n_items if prob.item_groups[i] == g]
                if !isempty(items_in_group)
                    @constraint(model, sum(x[i] for i in items_in_group) >= prob.group_min[g])
                    @constraint(model, sum(x[i] for i in items_in_group) <= prob.group_max[g])
                end
            end
        end

    elseif prob.variant == knap_min_fill
        # Fractional knapsack with minimum fill requirement
        @variable(model, 0 <= x[1:prob.n_items] <= 1)
        @objective(model, Max, sum(prob.values[i] * x[i] for i in 1:prob.n_items))
        @constraint(model, sum(prob.weights[i] * x[i] for i in 1:prob.n_items) <= prob.capacity)
        @constraint(model, sum(prob.weights[i] * x[i] for i in 1:prob.n_items) >= prob.capacity * prob.min_fill_fraction)
    end

    return model
end

# Register the problem type
register_problem(
    :knapsack,
    KnapsackProblem,
    "Knapsack problem with variants including standard, multi-dimensional, bounded, multiple knapsacks, conflict, dependency, group constraints, and minimum fill"
)
