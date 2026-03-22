using JuMP
using Random

"""
    ProductionPlanningProblem <: ProblemGenerator

Generator for production planning problems.

# Fields
- `n_products::Int`: Number of products
- `n_resources::Int`: Number of resources
- `profits::Vector{Int}`: Profit per unit of each product
- `usage::Matrix{Float64}`: Resource usage per unit of each product
- `resources::Vector{Float64}`: Available resources
"""
struct ProductionPlanningProblem <: ProblemGenerator
    n_products::Int
    n_resources::Int
    profits::Vector{Int}
    usage::Matrix{Float64}
    resources::Vector{Float64}
    min_production::Vector{Float64}
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

    # Calculate resource availability
    resources = sum(usage, dims=1)[:] * resource_factor

    # Handle feasibility
    min_production = zeros(n_products)

    actual_status = feasibility_status
    if feasibility_status == unknown
        actual_status = rand() < 0.7 ? feasible : infeasible
    end

    if actual_status == infeasible
        # Calculate max possible production per product
        max_possible = [minimum(resources[j] / usage[i, j] for j in 1:n_resources) for i in 1:n_products]

        # Set minimum production for a subset of products
        n_constrained = max(2, rand(max(1, n_products ÷ 4):max(2, n_products ÷ 2)))
        selected = randperm(n_products)[1:n_constrained]
        for i in selected
            min_production[i] = max_possible[i] * (0.3 + 0.3 * rand())
        end

        # Reduce the most stressed resource to create infeasibility
        required = [sum(usage[i, j] * min_production[i] for i in 1:n_products) for j in 1:n_resources]
        ratios = [required[j] / max(resources[j], eps()) for j in 1:n_resources]
        critical_j = argmax(ratios)
        resources[critical_j] = required[critical_j] * (0.7 + 0.2 * rand())
    end

    return ProductionPlanningProblem(n_products, n_resources, profits, usage, resources, min_production)
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

    for i in 1:prob.n_products
        if prob.min_production[i] > 0
            @constraint(model, x[i] >= prob.min_production[i])
        end
    end

    return model
end

# Register the problem type
register_problem(
    :production_planning,
    ProductionPlanningProblem,
    "Production planning problem that maximizes profit subject to resource constraints"
)
