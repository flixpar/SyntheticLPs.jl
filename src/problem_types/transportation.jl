using JuMP
using Random
using StatsBase

"""
    TransportationProblem <: ProblemGenerator

Generator for transportation problems that optimize shipping goods from sources to destinations at minimum cost.

# Fields
- `n_sources::Int`: Number of supply sources
- `n_destinations::Int`: Number of demand destinations
- `supplies::Vector{Int}`: Supply at each source
- `demands::Vector{Int}`: Demand at each destination
- `costs::Matrix{Int}`: Transportation cost from each source to each destination
"""
struct TransportationProblem <: ProblemGenerator
    n_sources::Int
    n_destinations::Int
    supplies::Vector{Int}
    demands::Vector{Int}
    costs::Matrix{Int}
end

"""Determine near-square dimensions to approach a target variable count."""
function determine_dimensions(target_variables::Int)
    sqrt_target = sqrt(target_variables)
    ratio = 0.5 + rand() * 1.0  # ratio between 0.5 and 1.5

    n_sources = max(2, round(Int, sqrt_target * ratio))
    n_destinations = max(2, round(Int, target_variables / max(n_sources, 1)))

    current_vars = n_sources * n_destinations
    if current_vars < target_variables * 0.9
        if n_sources >= n_destinations
            n_sources = max(n_sources, round(Int, target_variables / max(n_destinations, 1)))
        else
            n_destinations = max(n_destinations, round(Int, target_variables / max(n_sources, 1)))
        end
    elseif current_vars > target_variables * 1.1
        if n_sources >= n_destinations
            n_sources = max(2, round(Int, target_variables / max(n_destinations, 1)))
        else
            n_destinations = max(2, round(Int, target_variables / max(n_sources, 1)))
        end
    end

    return n_sources, n_destinations
end

"""Return realistic supply, demand, and cost ranges based on problem size."""
function base_parameter_ranges(total_vars::Int)
    if total_vars <= 250
        supply_range = (rand(50:100), rand(200:500))
        demand_range = (rand(30:80), rand(150:300))
        cost_range = (rand(5:15), rand(25:60))
    elseif total_vars <= 1000
        supply_range = (rand(100:500), rand(1000:5000))
        demand_range = (rand(80:300), rand(800:3000))
        cost_range = (rand(10:30), rand(50:150))
    else
        supply_range = (rand(500:2000), rand(5000:50000))
        demand_range = (rand(300:1500), rand(3000:30000))
        cost_range = (rand(20:100), rand(100:500))
    end

    return supply_range, demand_range, cost_range
end

function distribute_positive!(vec::Vector{Int}, amount::Int)
    if amount <= 0 || isempty(vec)
        return
    end
    weights = rand(length(vec))
    w_sum = sum(weights)
    base = w_sum == 0 ? fill(0, length(vec)) : floor.(Int, (weights ./ w_sum) .* amount)
    remainder = amount - sum(base)
    if remainder > 0
        idxs = randperm(length(vec))[1:min(remainder, length(vec))]
        for idx in idxs
            base[idx] += 1
        end
    end
    vec .+= base
end

function distribute_negative!(vec::Vector{Int}, amount::Int)
    remaining = amount
    if remaining <= 0 || isempty(vec)
        return
    end
    while remaining > 0
        changed = false
        for idx in randperm(length(vec))
            if vec[idx] > 1
                vec[idx] -= 1
                remaining -= 1
                changed = true
                if remaining == 0
                    break
                end
            end
        end
        if !changed
            break
        end
    end
end

function adjust_vector_total!(vec::Vector{Int}, delta::Int)
    if delta > 0
        distribute_positive!(vec, delta)
    elseif delta < 0
        distribute_negative!(vec, -delta)
    end
end

function rescale_vector_total!(vec::Vector{Int}, target_total::Int)
    current_total = sum(vec)
    if length(vec) == 0
        return vec
    elseif current_total <= 0 || target_total <= 0
        fill!(vec, max(1, round(Int, target_total / max(length(vec), 1))))
        adjust_vector_total!(vec, target_total - sum(vec))
        return vec
    end

    scale = target_total / current_total
    vec .= max.(1, round.(Int, vec .* scale))
    adjust_vector_total!(vec, target_total - sum(vec))
    return vec
end

function apply_subset_scaling!(vec::Vector{Int}, fraction::Float64, multiplier_range::Tuple{Float64, Float64})
    if isempty(vec)
        return vec
    end
    count = max(1, round(Int, length(vec) * fraction))
    selected = randperm(length(vec))[1:min(count, length(vec))]
    low, high = multiplier_range
    for idx in selected
        multiplier = low + rand() * (high - low)
        vec[idx] = max(1, round(Int, vec[idx] * multiplier))
    end
    return vec
end

function generate_cost_matrix(cost_range::Tuple{Int, Int}, source_bias::Vector{Float64}, destination_bias::Vector{Float64}; variability::Float64 = 0.2)
    n_sources = length(source_bias)
    n_destinations = length(destination_bias)
    min_cost, max_cost = cost_range
    base_costs = rand(min_cost:max_cost, n_sources, n_destinations)
    source_scale = reshape(source_bias, n_sources, 1)
    destination_scale = reshape(destination_bias, 1, n_destinations)
    noise = 1 .+ variability .* (rand(n_sources, n_destinations) .- 0.5)
    scaled = base_costs .* source_scale .* destination_scale .* noise
    return max.(1, round.(Int, scaled))
end

function generate_distance_based_costs(n_sources::Int, n_destinations::Int, cost_range::Tuple{Int, Int}; distance_scale::Float64 = 90.0, noise_factor::Float64 = 0.2)
    min_cost, max_cost = cost_range
    base_costs = rand(min_cost:max_cost, n_sources, n_destinations)
    source_coords = rand(n_sources, 2)
    destination_coords = rand(n_destinations, 2)
    costs = similar(base_costs)
    for i in 1:n_sources
        for j in 1:n_destinations
            dx = source_coords[i, 1] - destination_coords[j, 1]
            dy = source_coords[i, 2] - destination_coords[j, 2]
            dist = sqrt(dx^2 + dy^2)
            surcharge = distance_scale * dist
            noise = 1 .+ noise_factor * (rand() - 0.5)
            costs[i, j] = max(1, round(Int, (base_costs[i, j] + surcharge) * noise))
        end
    end
    return costs
end

function generate_balanced_manufacturing_distribution(n_sources::Int, n_destinations::Int, total_vars::Int)
    supply_range, demand_range, cost_range = base_parameter_ranges(total_vars)
    supplies = rand(supply_range[1]:supply_range[2], n_sources)
    demands = rand(demand_range[1]:demand_range[2], n_destinations)

    target_supply_total = max(1, round(Int, sum(demands) * (0.95 + 0.1 * rand())))
    rescale_vector_total!(supplies, target_supply_total)

    source_bias = 0.9 .+ 0.2 .* rand(n_sources)
    destination_bias = 0.9 .+ 0.2 .* rand(n_destinations)
    costs = generate_cost_matrix(cost_range, source_bias, destination_bias; variability = 0.15)

    return supplies, demands, costs
end

function generate_retail_demand_spikes_distribution(n_sources::Int, n_destinations::Int, total_vars::Int)
    supply_range, demand_range, cost_range = base_parameter_ranges(total_vars)
    demands = rand(demand_range[1]:demand_range[2], n_destinations)
    apply_subset_scaling!(demands, 0.3, (1.7, 2.8))

    supplies = rand(supply_range[1]:supply_range[2], n_sources)
    target_supply_total = max(1, round(Int, sum(demands) * (1.05 + 0.15 * rand())))
    rescale_vector_total!(supplies, target_supply_total)

    source_bias = 0.85 .+ 0.25 .* rand(n_sources)
    destination_bias = 0.8 .+ 0.5 .* rand(n_destinations)
    costs = generate_cost_matrix(cost_range, source_bias, destination_bias; variability = 0.2)

    return supplies, demands, costs
end

function generate_regional_cross_docking_distribution(n_sources::Int, n_destinations::Int, total_vars::Int)
    supply_range, demand_range, cost_range = base_parameter_ranges(total_vars)
    supplies = rand(supply_range[1]:supply_range[2], n_sources)
    apply_subset_scaling!(supplies, 0.2, (1.4, 2.2))

    demands = rand(demand_range[1]:demand_range[2], n_destinations)
    apply_subset_scaling!(demands, 0.15, (1.2, 1.8))
    rescale_vector_total!(supplies, max(1, round(Int, sum(demands) * (1.0 + 0.05 * rand()))))

    source_bias = 0.8 .+ 0.4 .* rand(n_sources)
    destination_bias = 0.85 .+ 0.35 .* rand(n_destinations)
    costs = generate_cost_matrix(cost_range, source_bias, destination_bias; variability = 0.25)

    return supplies, demands, costs
end

function generate_seasonal_agriculture_distribution(n_sources::Int, n_destinations::Int, total_vars::Int)
    supply_range, demand_range, cost_range = base_parameter_ranges(total_vars)
    supplies = rand(supply_range[1]:supply_range[2], n_sources)
    apply_subset_scaling!(supplies, 0.15, (2.0, 3.5))

    demands = rand(demand_range[1]:demand_range[2], n_destinations)
    apply_subset_scaling!(demands, 0.25, (1.2, 1.6))

    rescale_vector_total!(demands, max(1, round(Int, sum(supplies) * (0.9 + 0.1 * rand()))))

    source_bias = 0.7 .+ 0.5 .* rand(n_sources)
    destination_bias = 0.9 .+ 0.3 .* rand(n_destinations)
    costs = generate_cost_matrix(cost_range, source_bias, destination_bias; variability = 0.35)

    return supplies, demands, costs
end

function generate_emergency_relief_distribution(n_sources::Int, n_destinations::Int, total_vars::Int)
    supply_range, demand_range, cost_range = base_parameter_ranges(total_vars)
    supplies = rand(supply_range[1]:supply_range[2], n_sources)
    demands = rand(demand_range[1]:demand_range[2], n_destinations)
    apply_subset_scaling!(demands, 0.4, (1.5, 2.5))

    rescale_vector_total!(supplies, max(1, round(Int, sum(demands) * (0.85 + 0.1 * rand()))))

    costs = generate_distance_based_costs(n_sources, n_destinations, cost_range; distance_scale = 120.0, noise_factor = 0.25)

    return supplies, demands, costs
end

const TRANSPORT_VARIANT_SPECS = [
    (name = :balanced_manufacturing, weight = 0.35, generator = generate_balanced_manufacturing_distribution),
    (name = :retail_demand_spikes, weight = 0.25, generator = generate_retail_demand_spikes_distribution),
    (name = :regional_cross_docking, weight = 0.2, generator = generate_regional_cross_docking_distribution),
    (name = :seasonal_agriculture, weight = 0.12, generator = generate_seasonal_agriculture_distribution),
    (name = :emergency_relief, weight = 0.08, generator = generate_emergency_relief_distribution),
]

function sample_transportation_variant()
    weights = StatsBase.Weights(Float64[spec.weight for spec in TRANSPORT_VARIANT_SPECS])
    return StatsBase.sample(TRANSPORT_VARIANT_SPECS, weights)
end

function enforce_feasibility!(supplies::Vector{Int}, demands::Vector{Int}, feasibility_status::FeasibilityStatus)
    total_supply = sum(supplies)
    total_demand = sum(demands)

    if feasibility_status == feasible
        if total_supply < total_demand
            distribute_positive!(supplies, total_demand - total_supply)
        end
    elseif feasibility_status == infeasible
        target_margin = max(1, round(Int, (0.02 + 0.08 * rand()) * max(total_supply, 1)))
        required = (total_supply + target_margin) - total_demand
        if required > 0
            distribute_positive!(demands, required)
        end
    end
end

"""
    TransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a transportation problem instance.

# Arguments
- `target_variables`: Target number of variables (n_sources × n_destinations)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function TransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    n_sources, n_destinations = determine_dimensions(target_variables)
    total_vars = n_sources * n_destinations

    variant = sample_transportation_variant()
    supplies, demands, costs = variant.generator(n_sources, n_destinations, total_vars)

    enforce_feasibility!(supplies, demands, feasibility_status)

    return TransportationProblem(n_sources, n_destinations, supplies, demands, costs)
end

"""
    build_model(prob::TransportationProblem)

Build a JuMP model for the transportation problem.

# Arguments
- `prob`: TransportationProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::TransportationProblem)
    model = Model()

    # Variables
    @variable(model, x[1:prob.n_sources, 1:prob.n_destinations] >= 0)

    # Objective
    @objective(model, Min, sum(prob.costs[i, j] * x[i, j] for i in 1:prob.n_sources, j in 1:prob.n_destinations))

    # Constraints
    for i in 1:prob.n_sources
        @constraint(model, sum(x[i, j] for j in 1:prob.n_destinations) <= prob.supplies[i])
    end
    for j in 1:prob.n_destinations
        @constraint(model, sum(x[i, j] for i in 1:prob.n_sources) >= prob.demands[j])
    end

    return model
end

# Register the problem type
register_problem(
    :transportation,
    TransportationProblem,
    "Transportation problem that optimizes shipping goods from sources to destinations at minimum cost"
)
