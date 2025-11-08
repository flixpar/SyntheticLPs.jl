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
    if feasibility_status == feasible || feasibility_status == unknown
        # Set capacity to 30-70% of total weight for interesting problems
        capacity_ratio = 0.3 + rand() * 0.4
        capacity = round(Int, total_avg_weight * capacity_ratio)
        capacity = max(1, capacity + rand(-50:50))  # Add variability
    else  # infeasible
        # Knapsack problems are always feasible (can select no items)
        # So we create an "interesting" infeasible case by requiring all items
        # which exceeds capacity (not a standard knapsack formulation)
        # For now, treat as feasible
        capacity_ratio = 0.3 + rand() * 0.4
        capacity = round(Int, total_avg_weight * capacity_ratio)
        capacity = max(1, capacity + rand(-50:50))
    end

    return KnapsackProblem(n_items, capacity, values, weights)
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

    return model
end

# Register the problem type
register_problem(
    :knapsack,
    KnapsackProblem,
    "Knapsack problem that maximizes the value of items selected under a weight constraint"
)
