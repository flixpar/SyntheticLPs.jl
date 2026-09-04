using JuMP
using Random

"""
    TransportationProblem <: ProblemGenerator

Generator for transportation problems that optimize shipping goods from sources to destinations at minimum cost.

# Overview

Models the classic transportation planning problem. The decisions are shipment
amounts on every source-destination lane. The objective minimizes total shipping
cost. Source constraints limit outbound shipments by available supply, and
destination constraints require inbound shipments to meet demand.

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

"""
    TransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a transportation problem instance.

# Arguments

  - `target_variables`: Target number of variables (n_sources × n_destinations)
  - `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
  - `seed`: Random seed for reproducibility
"""
function TransportationProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)

    # Calculate dimensions to achieve target number of variables
    sqrt_target = sqrt(target_variables)
    ratio = 0.5 + rand(rng) * 1.0  # ratio between 0.5 and 1.5

    n_sources = max(2, round(Int, sqrt_target * ratio))
    n_destinations = max(2, round(Int, target_variables / n_sources))

    # Fine-tune to get closer to target
    current_vars = n_sources * n_destinations
    if current_vars < target_variables * 0.9
        if n_sources >= n_destinations
            n_sources = max(n_sources, round(Int, target_variables / n_destinations))
        else
            n_destinations = max(n_destinations, round(Int, target_variables / n_sources))
        end
    elseif current_vars > target_variables * 1.1
        if n_sources >= n_destinations
            n_sources = max(2, round(Int, target_variables / n_destinations))
        else
            n_destinations = max(2, round(Int, target_variables / n_sources))
        end
    end

    # Set realistic parameter ranges based on problem size
    total_vars = n_sources * n_destinations
    if total_vars <= 250
        supply_range = (rand(rng, 50:100), rand(rng, 200:500))
        demand_range = (rand(rng, 30:80), rand(rng, 150:300))
        cost_range = (rand(rng, 5:15), rand(rng, 25:60))
    elseif total_vars <= 1000
        supply_range = (rand(rng, 100:500), rand(rng, 1000:5000))
        demand_range = (rand(rng, 80:300), rand(rng, 800:3000))
        cost_range = (rand(rng, 10:30), rand(rng, 50:150))
    else
        supply_range = (rand(rng, 500:2000), rand(rng, 5000:50000))
        demand_range = (rand(rng, 300:1500), rand(rng, 3000:30000))
        cost_range = (rand(rng, 20:100), rand(rng, 100:500))
    end

    # Generate random data
    min_supply, max_supply = supply_range
    supplies = rand(rng, min_supply:max_supply, n_sources)

    min_demand, max_demand = demand_range
    demands = rand(rng, min_demand:max_demand, n_destinations)

    min_cost, max_cost = cost_range
    costs = rand(rng, min_cost:max_cost, n_sources, n_destinations)

    # Helper function to distribute additions across a vector
    function distribute_additions!(vec::Vector{Int}, amount::Int)
        if amount <= 0
            return nothing
        end
        w = rand(rng, length(vec))
        w_sum = sum(w)
        base = floor.(Int, (w ./ w_sum) .* amount)
        remainder = amount - sum(base)
        if remainder > 0
            for idx in randperm(rng, length(vec))[1:remainder]
                base[idx] += 1
            end
        end
        vec .+= base
    end

    # Adjust for feasibility
    total_supply = sum(supplies)
    total_demand = sum(demands)

    if feasibility_status == feasible
        # Guarantee feasibility: ensure total_supply >= total_demand
        if total_supply < total_demand
            shortage = total_demand - total_supply
            distribute_additions!(supplies, shortage)
        end
    elseif feasibility_status == infeasible
        # Guarantee infeasibility: ensure total_demand > total_supply with margin
        target_margin = max(1, round(Int, (0.02 + 0.08 * rand(rng)) * max(total_supply, 1)))
        missing = (total_supply + target_margin) - total_demand
        if missing > 0
            distribute_additions!(demands, missing)
        end
    end
    # For unknown, leave as-is

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
    @objective(
        model,
        Min,
        sum(prob.costs[i, j] * x[i, j] for i in 1:prob.n_sources, j in 1:prob.n_destinations)
    )

    # Constraints
    for i in 1:prob.n_sources
        @constraint(model, sum(x[i, j] for j in 1:prob.n_destinations) <= prob.supplies[i])
    end
    for j in 1:prob.n_destinations
        @constraint(model, sum(x[i, j] for i in 1:prob.n_sources) >= prob.demands[j])
    end

    return model
end

# Register the variant
register_variant(
    :transportation,
    :standard,
    TransportationProblem,
    "Transportation problem that optimizes shipping goods from sources to destinations at minimum cost",
)
