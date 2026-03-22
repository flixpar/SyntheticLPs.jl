using JuMP
using Random

"""
    KnapsackProblem <: ProblemGenerator

Generator for knapsack problems (fractional) that maximize the value of items selected under a weight constraint.

# Fields
- `n_items::Int`: Number of items
- `capacity::Int`: Knapsack capacity
- `values::Vector{Int}`: Value of each item
- `weights::Vector{Int}`: Weight of each item
"""
struct KnapsackProblem <: ProblemGenerator
    n_items::Int
    capacity::Int
    values::Vector{Int}
    weights::Vector{Int}
    min_value::Float64
end

"""
    KnapsackProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a knapsack problem instance.

# Arguments
- `target_variables`: Target number of variables (items)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function KnapsackProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # For knapsack, target_variables = n_items
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

    # Generate item values and weights
    min_value, max_value = value_range
    values = rand(min_value:max_value, n_items)

    min_weight, max_weight = weight_range
    weights = rand(min_weight:max_weight, n_items)

    # Calculate total average weight
    total_avg_weight = sum(weights)

    # Determine capacity based on feasibility status
    # Set capacity to 30-70% of total weight for interesting problems
    capacity_ratio = 0.3 + rand() * 0.4
    capacity = round(Int, total_avg_weight * capacity_ratio)
    capacity = max(1, capacity + rand(-50:50))

    # Handle feasibility
    min_value = 0.0

    actual_status = feasibility_status
    if feasibility_status == unknown
        actual_status = rand() < 0.7 ? feasible : infeasible
    end

    if actual_status == infeasible
        # Calculate max achievable value under capacity constraint
        # For fractional knapsack: sort by value/weight ratio, pack greedily
        ratios = values ./ max.(weights, 1)
        sorted_idx = sortperm(ratios, rev=true)
        remaining_cap = Float64(capacity)
        max_achievable = 0.0
        for i in sorted_idx
            take = min(1.0, remaining_cap / weights[i])
            max_achievable += values[i] * take
            remaining_cap -= weights[i] * take
            if remaining_cap <= 0
                break
            end
        end
        # Require more value than achievable
        min_value = max_achievable * (1.1 + 0.3 * rand())
    end

    return KnapsackProblem(n_items, capacity, values, weights, min_value)
end

"""
    build_model(prob::KnapsackProblem)

Build a JuMP model for the knapsack problem.

# Arguments
- `prob`: KnapsackProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::KnapsackProblem)
    model = Model()

    # Variables - using fractional knapsack version
    @variable(model, 0 <= x[1:prob.n_items] <= 1)

    # Objective
    @objective(model, Max, sum(prob.values[i] * x[i] for i in 1:prob.n_items))

    # Constraint
    @constraint(model, sum(prob.weights[i] * x[i] for i in 1:prob.n_items) <= prob.capacity)

    # Minimum value constraint (for infeasibility)
    if prob.min_value > 0
        @constraint(model, sum(prob.values[i] * x[i] for i in 1:prob.n_items) >= prob.min_value)
    end

    return model
end

# Register the problem type
register_problem(
    :knapsack,
    KnapsackProblem,
    "Knapsack problem that maximizes the value of items selected under a weight constraint"
)
