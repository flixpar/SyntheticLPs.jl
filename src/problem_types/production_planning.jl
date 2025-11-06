using JuMP
using Random

"""
    ProductionPlanningProblem <: ProblemGenerator

Generator for production planning problems.

# Fields
- `n_products::Int`: Number of products
- `n_resources::Int`: Number of resources
- `profits::Vector{Int}`: Profit per unit of each product
- `usage::Matrix{Int}`: Resource usage per unit of each product
- `resources::Vector{Float64}`: Available resources
"""
struct ProductionPlanningProblem <: ProblemGenerator
    n_products::Int
    n_resources::Int
    profits::Vector{Int}
    usage::Matrix{Int}
    resources::Vector{Float64}
end

"""
    ProductionPlanningProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a production planning problem instance.
"""
function ProductionPlanningProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Direct mapping: variables = products
    n_products = max(2, min(2000, target_variables))
    n_resources = rand(1:50)

    # Determine profit and usage ranges based on scale
    profit_range = (10, 500)
    usage_range = (0.1, 50.0)
    resource_factor = rand(0.4:0.1:0.8)

    # Generate random data
    min_profit, max_profit = profit_range
    profits = rand(min_profit:max_profit, n_products)

    min_usage, max_usage = usage_range
    usage = rand(min_usage:max_usage, n_products, n_resources)

    # Calculate resource availability to ensure feasibility
    resources = sum(usage, dims=1)[:] * resource_factor

    return ProductionPlanningProblem(n_products, n_resources, profits, usage, resources)
end

"""
    build_model(prob::ProductionPlanningProblem)

Build a JuMP model for the production planning problem.
"""
function build_model(prob::ProductionPlanningProblem)
    model = Model()

    @variable(model, x[1:prob.n_products] >= 0)
    @objective(model, Max, sum(prob.profits[i] * x[i] for i in 1:prob.n_products))

    for j in 1:prob.n_resources
        @constraint(model, sum(prob.usage[i, j] * x[i] for i in 1:prob.n_products) <= prob.resources[j])
    end

    return model
end

# Register the problem type
register_problem(
    :production_planning,
    ProductionPlanningProblem,
    "Production planning problem that maximizes profit subject to resource constraints"
)
