using JuMP
using Random

"""
    BasicTransportation <: ProblemGenerator

Generator for basic transportation problems that optimize shipping goods from sources to destinations at minimum cost.

# Fields
- `n_sources::Int`: Number of supply sources
- `n_destinations::Int`: Number of demand destinations
- `supplies::Vector{Int}`: Supply at each source
- `demands::Vector{Int}`: Demand at each destination
- `costs::Matrix{Int}`: Transportation cost from each source to each destination
"""
struct BasicTransportation <: ProblemGenerator
    n_sources::Int
    n_destinations::Int
    supplies::Vector{Int}
    demands::Vector{Int}
    costs::Matrix{Int}
end

"""
    BasicTransportation(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a basic transportation problem instance.

# Arguments
- `target_variables`: Target number of variables (n_sources × n_destinations)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function BasicTransportation(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Calculate dimensions to achieve target number of variables
    sqrt_target = sqrt(target_variables)
    ratio = 0.5 + rand() * 1.0  # ratio between 0.5 and 1.5

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

    # Generate random data
    min_supply, max_supply = supply_range
    supplies = rand(min_supply:max_supply, n_sources)

    min_demand, max_demand = demand_range
    demands = rand(min_demand:max_demand, n_destinations)

    min_cost, max_cost = cost_range
    costs = rand(min_cost:max_cost, n_sources, n_destinations)

    # Helper function to distribute additions across a vector
    function distribute_additions!(vec::Vector{Int}, amount::Int)
        if amount <= 0
            return
        end
        w = rand(length(vec))
        w_sum = sum(w)
        base = floor.(Int, (w ./ w_sum) .* amount)
        remainder = amount - sum(base)
        if remainder > 0
            for idx in randperm(length(vec))[1:remainder]
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
        target_margin = max(1, round(Int, (0.02 + 0.08 * rand()) * max(total_supply, 1)))
        missing = (total_supply + target_margin) - total_demand
        if missing > 0
            distribute_additions!(demands, missing)
        end
    end
    # For unknown, leave as-is

    return BasicTransportation(n_sources, n_destinations, supplies, demands, costs)
end

"""
    build_model(prob::BasicTransportation)

Build a JuMP model for the basic transportation problem.

# Arguments
- `prob`: BasicTransportation instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::BasicTransportation)
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
    :basic_transportation,
    BasicTransportation,
    "Basic transportation problem that optimizes shipping goods from sources to destinations at minimum cost"
)
