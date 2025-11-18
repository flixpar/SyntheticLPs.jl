using JuMP
using Random
using Distributions

"""
    CargoLoading <: ProblemGenerator

Generator for cargo loading problems (container/truck loading optimization).

This problem optimizes loading of items into containers/trucks with weight
and volume constraints, maximizing value or minimizing cost while respecting
loading constraints.

# Fields
- `n_items::Int`: Number of items to potentially load
- `n_containers::Int`: Number of available containers/trucks
- `item_weights::Vector{Float64}`: Weight of each item
- `item_volumes::Vector{Float64}`: Volume of each item
- `item_values::Vector{Float64}`: Value/priority of each item
- `item_fragilities::Vector{Int}`: Fragility class (1=robust, 2=normal, 3=fragile)
- `container_weight_capacities::Vector{Float64}`: Weight capacity of each container
- `container_volume_capacities::Vector{Float64}`: Volume capacity of each container
- `container_costs::Vector{Float64}`: Cost to use each container
- `loading_costs::Matrix{Float64}`: Cost to load item into container
- `min_items_required::Int`: Minimum number of items that must be loaded
"""
struct CargoLoading <: ProblemGenerator
    n_items::Int
    n_containers::Int
    item_weights::Vector{Float64}
    item_volumes::Vector{Float64}
    item_values::Vector{Float64}
    item_fragilities::Vector{Int}
    container_weight_capacities::Vector{Float64}
    container_volume_capacities::Vector{Float64}
    container_costs::Vector{Float64}
    loading_costs::Matrix{Float64}
    min_items_required::Int
end

"""
    CargoLoading(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a cargo loading problem instance.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Details
Variables: n_items × n_containers (assignment) + n_containers (container usage)
Target: n_items × n_containers + n_containers
"""
function CargoLoading(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine scale
    if target_variables <= 150
        min_items, max_items = 5, 30
        min_containers, max_containers = 2, 8
        weight_range = (10.0, 200.0)
        volume_range = (5.0, 100.0)
        value_range = (50.0, 1000.0)
        container_weight_range = (500.0, 2000.0)
        container_volume_range = (300.0, 1500.0)
    elseif target_variables <= 800
        min_items, max_items = 20, 80
        min_containers, max_containers = 5, 15
        weight_range = (20.0, 500.0)
        volume_range = (10.0, 300.0)
        value_range = (100.0, 3000.0)
        container_weight_range = (1000.0, 5000.0)
        container_volume_range = (600.0, 3000.0)
    else
        min_items, max_items = 50, 200
        min_containers, max_containers = 8, 30
        weight_range = (50.0, 1000.0)
        volume_range = (20.0, 800.0)
        value_range = (200.0, 10000.0)
        container_weight_range = (2000.0, 15000.0)
        container_volume_range = (1000.0, 10000.0)
    end

    # Solve for dimensions
    # Variables: n_items × n_containers + n_containers
    # target = n_items × n_containers + n_containers = n_containers × (n_items + 1)
    best_config = (min_items, min_containers)
    best_error = Inf

    for n_containers in min_containers:max_containers
        # Solve for n_items
        # target = n_containers × (n_items + 1)
        n_items_exact = (target_variables / n_containers) - 1

        if n_items_exact >= min_items && n_items_exact <= max_items
            n_items = round(Int, n_items_exact)
            actual_vars = n_items * n_containers + n_containers
            error = abs(actual_vars - target_variables) / target_variables

            if error < best_error
                best_error = error
                best_config = (n_items, n_containers)
            end
        end
    end

    if best_error > 0.1
        # Heuristic fallback
        n_containers = max(min_containers, min(max_containers, round(Int, sqrt(target_variables / 10))))
        n_items = max(min_items, min(max_items, round(Int, (target_variables / n_containers) - 1)))
        best_config = (n_items, n_containers)
    end

    n_items, n_containers = best_config

    # Generate item properties using log-normal distributions for realism

    # Weights
    min_weight, max_weight = weight_range
    log_mean_w = log(sqrt(min_weight * max_weight))
    log_std_w = log(max_weight / min_weight) / 4
    item_weights = [clamp(exp(rand(Normal(log_mean_w, log_std_w))), min_weight, max_weight)
                    for _ in 1:n_items]
    item_weights = round.(item_weights, digits=2)

    # Volumes (correlated with weight but with variation)
    min_volume, max_volume = volume_range
    item_volumes = Float64[]
    for i in 1:n_items
        # Volume roughly proportional to weight with variation
        weight_ratio = (item_weights[i] - min_weight) / (max_weight - min_weight)
        base_volume = min_volume + weight_ratio * (max_volume - min_volume)
        # Add variation (density differences)
        volume = base_volume * (0.7 + 0.6 * rand())
        push!(item_volumes, round(clamp(volume, min_volume, max_volume), digits=2))
    end

    # Values (higher for heavier/larger items on average, but with variation)
    min_value, max_value = value_range
    log_mean_v = log(sqrt(min_value * max_value))
    log_std_v = log(max_value / min_value) / 3
    item_values = [clamp(exp(rand(Normal(log_mean_v, log_std_v))), min_value, max_value)
                   for _ in 1:n_items]
    item_values = round.(item_values, digits=2)

    # Fragilities (1=robust, 2=normal, 3=fragile)
    fragility_probs = [0.3, 0.5, 0.2]  # 30% robust, 50% normal, 20% fragile
    item_fragilities = [rand(Categorical(fragility_probs)) for _ in 1:n_items]

    # Container capacities
    min_weight_cap, max_weight_cap = container_weight_range
    min_volume_cap, max_volume_cap = container_volume_range

    container_weight_capacities = [round(rand(Uniform(min_weight_cap, max_weight_cap)), digits=2)
                                   for _ in 1:n_containers]

    container_volume_capacities = [round(rand(Uniform(min_volume_cap, max_volume_cap)), digits=2)
                                   for _ in 1:n_containers]

    # Container costs (larger containers cost more)
    avg_weight_cap = mean(container_weight_capacities)
    container_costs = [round(1000.0 * (container_weight_capacities[c] / avg_weight_cap)^0.8 * (0.9 + 0.2 * rand()), digits=2)
                       for c in 1:n_containers]

    # Loading costs (depend on item and container characteristics)
    loading_costs = zeros(n_items, n_containers)
    for i in 1:n_items
        for c in 1:n_containers
            # Base cost proportional to item weight
            base_cost = item_weights[i] * rand(0.5:0.1:2.0)
            # Fragile items cost more to load
            fragility_mult = item_fragilities[i] == 3 ? 1.5 : (item_fragilities[i] == 2 ? 1.0 : 0.8)
            loading_costs[i, c] = round(base_cost * fragility_mult, digits=2)
        end
    end

    # Minimum items required (delivery commitment)
    min_items_required = round(Int, n_items * rand(0.5:0.05:0.8))

    # Adjust for feasibility
    if feasibility_status == feasible
        # Ensure total capacity can handle all items
        total_weight = sum(item_weights)
        total_volume = sum(item_volumes)
        total_weight_cap = sum(container_weight_capacities)
        total_volume_cap = sum(container_volume_capacities)

        # Adjust capacities if needed
        if total_weight_cap < total_weight
            scale = (total_weight * 1.2) / total_weight_cap
            container_weight_capacities .*= scale
            container_weight_capacities = round.(container_weight_capacities, digits=2)
        end

        if total_volume_cap < total_volume
            scale = (total_volume * 1.2) / total_volume_cap
            container_volume_capacities .*= scale
            container_volume_capacities = round.(container_volume_capacities, digits=2)
        end

    elseif feasibility_status == infeasible
        # Create infeasibility
        choice = rand()
        if choice < 0.4
            # Insufficient weight capacity
            scale = rand(0.4:0.05:0.75)
            container_weight_capacities .*= scale
            container_weight_capacities = round.(container_weight_capacities, digits=2)
        elseif choice < 0.7
            # Insufficient volume capacity
            scale = rand(0.4:0.05:0.75)
            container_volume_capacities .*= scale
            container_volume_capacities = round.(container_volume_capacities, digits=2)
        else
            # Impossible minimum requirement
            # Ensure even optimal packing can't meet requirement
            total_weight_cap = sum(container_weight_capacities)
            total_volume_cap = sum(container_volume_capacities)

            # Count how many items can theoretically fit
            sorted_items = sortperm(item_weights)
            feasible_count = 0
            cumulative_weight = 0.0
            cumulative_volume = 0.0

            for idx in sorted_items
                if cumulative_weight + item_weights[idx] <= total_weight_cap &&
                   cumulative_volume + item_volumes[idx] <= total_volume_cap
                    feasible_count += 1
                    cumulative_weight += item_weights[idx]
                    cumulative_volume += item_volumes[idx]
                else
                    break
                end
            end

            # Set requirement higher than feasible
            min_items_required = feasible_count + rand(1:max(1, n_items - feasible_count))
        end
    end

    return CargoLoading(
        n_items,
        n_containers,
        item_weights,
        item_volumes,
        item_values,
        item_fragilities,
        container_weight_capacities,
        container_volume_capacities,
        container_costs,
        loading_costs,
        min_items_required
    )
end

"""
    build_model(prob::CargoLoading)

Build a JuMP model for the cargo loading problem.

# Arguments
- `prob`: CargoLoading instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::CargoLoading)
    model = Model()

    I = prob.n_items
    C = prob.n_containers

    # Decision variables

    # x[i,c] = 1 if item i is loaded into container c
    @variable(model, 0 <= x[1:I, 1:C] <= 1)

    # y[c] = 1 if container c is used
    @variable(model, y[1:C], Bin)

    # z[i] = 1 if item i is loaded (into any container)
    @variable(model, 0 <= z[1:I] <= 1)

    # Objective: maximize value - costs
    @objective(model, Max,
        # Value from loaded items
        sum(prob.item_values[i] * z[i] for i in 1:I) -
        # Container costs
        sum(prob.container_costs[c] * y[c] for c in 1:C) -
        # Loading costs
        sum(prob.loading_costs[i, c] * x[i, c] for i in 1:I, c in 1:C)
    )

    # Constraints

    # Each item loaded at most once (across all containers)
    for i in 1:I
        @constraint(model, sum(x[i, c] for c in 1:C) <= 1)
    end

    # Link z to x: item is loaded iff assigned to some container
    for i in 1:I
        @constraint(model, z[i] == sum(x[i, c] for c in 1:C))
    end

    # Weight capacity constraints
    for c in 1:C
        @constraint(model,
            sum(prob.item_weights[i] * x[i, c] for i in 1:I) <=
            prob.container_weight_capacities[c] * y[c]
        )
    end

    # Volume capacity constraints
    for c in 1:C
        @constraint(model,
            sum(prob.item_volumes[i] * x[i, c] for i in 1:I) <=
            prob.container_volume_capacities[c] * y[c]
        )
    end

    # Minimum items requirement
    @constraint(model, sum(z[i] for i in 1:I) >= prob.min_items_required)

    # Can only load into containers that are used
    for c in 1:C
        for i in 1:I
            @constraint(model, x[i, c] <= y[c])
        end
    end

    # Fragility constraints: fragile items shouldn't be overloaded in same container
    # Limit number of fragile items per container
    for c in 1:C
        fragile_items = [i for i in 1:I if prob.item_fragilities[i] == 3]
        if !isempty(fragile_items)
            max_fragile = max(1, round(Int, length(fragile_items) * 0.4))
            @constraint(model,
                sum(x[i, c] for i in fragile_items) <= max_fragile
            )
        end
    end

    return model
end

# Register the problem type
register_problem(
    :cargo_loading,
    CargoLoading,
    "Cargo loading problem optimizing container/truck loading with weight and volume constraints"
)
