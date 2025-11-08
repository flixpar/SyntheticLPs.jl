using JuMP
using Random
using Distributions
using LinearAlgebra
using StatsBase

"""
    ProductMixProblem <: ProblemGenerator

Generator for product mix optimization problems.

# Fields
- `num_products::Int`: Number of products
- `num_resources::Int`: Number of resources
- `profits::Vector{Float64}`: Profit per unit of each product
- `usage_matrix::Matrix{Float64}`: Resource usage per unit (num_resources × num_products)
- `availabilities::Vector{Float64}`: Available amount of each resource
- `lower_bounds::Vector{Float64}`: Minimum production level for each product
- `upper_bounds::Vector{Float64}`: Maximum production level for each product
"""
struct ProductMixProblem <: ProblemGenerator
    num_products::Int
    num_resources::Int
    profits::Vector{Float64}
    usage_matrix::Matrix{Float64}
    availabilities::Vector{Float64}
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
end

"""
    ProductMixProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a product mix problem instance.

# Arguments
- `target_variables`: Target number of variables (products)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function ProductMixProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # For product mix, variables = num_products
    num_products = max(2, min(10000, target_variables))

    # Scale parameters based on problem size
    if target_variables <= 250
        # Small operations
        num_resources = rand(DiscreteUniform(3, 8))
        sparsity = rand(Beta(2, 6))
        profit_min = rand(LogNormal(log(15), 0.4))
        profit_max = rand(LogNormal(log(120), 0.3))
        resource_usage_min = rand(LogNormal(log(1.0), 0.3))
        resource_usage_max = rand(LogNormal(log(5), 0.3))
        market_constraint_prob = rand(Beta(4, 6))
        correlation_strength = rand(Beta(4, 3))
    elseif target_variables <= 1000
        # Medium operations
        resource_range = 5:15
        beta_sample = rand(Beta(2, 3))
        num_resources = resource_range[max(1, min(length(resource_range), round(Int, beta_sample * length(resource_range)) + 1))]
        sparsity = rand(Beta(3, 4))
        profit_min = rand(LogNormal(log(8), 0.5))
        profit_max = rand(LogNormal(log(75), 0.4))
        resource_usage_min = rand(LogNormal(log(0.6), 0.4))
        resource_usage_max = rand(LogNormal(log(4.5), 0.4))
        market_constraint_prob = rand(Beta(5, 5))
        correlation_strength = rand(Beta(6, 4))
    else
        # Large operations
        log_mean = log(18)
        log_std = 0.4
        sample_val = rand(LogNormal(log_mean, log_std))
        num_resources = max(8, min(30, round(Int, sample_val)))
        sparsity = rand(Beta(2, 3))
        profit_min = rand(LogNormal(log(3), 0.6))
        profit_max = rand(LogNormal(log(45), 0.5))
        resource_usage_min = rand(LogNormal(log(0.3), 0.5))
        resource_usage_max = rand(LogNormal(log(4), 0.5))
        market_constraint_prob = rand(Beta(6, 4))
        correlation_strength = rand(Beta(8, 3))
    end

    # Randomly select industry type
    industry_types = ["manufacturing", "food_processing", "electronics", "furniture", "chemical", "automotive"]
    industry_weights = if target_variables <= 250
        [0.25, 0.35, 0.15, 0.20, 0.03, 0.02]
    elseif target_variables <= 1000
        [0.40, 0.15, 0.25, 0.10, 0.08, 0.02]
    else
        [0.35, 0.08, 0.20, 0.05, 0.17, 0.15]
    end
    industry_type = sample(industry_types, Weights(industry_weights))

    # Apply industry-specific adjustments
    if industry_type == "food_processing"
        profit_min *= 0.6
        profit_max *= 0.7
        resource_usage_max *= 0.8
        market_constraint_prob *= 1.5
    elseif industry_type == "electronics"
        profit_min *= 1.5
        profit_max *= 2.2
        resource_usage_max *= 1.4
        sparsity *= 1.3
    elseif industry_type == "furniture"
        profit_min *= 0.8
        profit_max *= 0.9
        resource_usage_min *= 1.2
        market_constraint_prob *= 1.2
    elseif industry_type == "chemical"
        profit_min *= 1.1
        profit_max *= 1.3
        resource_usage_min *= 1.8
        resource_usage_max *= 2.0
        correlation_strength *= 1.2
    elseif industry_type == "automotive"
        profit_min *= 3.0
        profit_max *= 4.0
        resource_usage_min *= 2.0
        resource_usage_max *= 2.5
        sparsity *= 0.8
        market_constraint_prob *= 0.8
    end

    # Generate quality factors
    quality_factors = rand(Beta(2, 2), num_products)

    # Generate profits
    base_profits = rand(LogNormal(log((profit_min + profit_max) / 2), 0.3), num_products)
    base_profits = clamp.(base_profits, profit_min, profit_max)
    quality_component = quality_factors .* (profit_max - profit_min) * 0.5
    profits = base_profits + correlation_strength * quality_component

    # Generate usage matrix
    usage_matrix = zeros(num_resources, num_products)

    for i in 1:num_resources
        base_usage = rand(LogNormal(log((resource_usage_min + resource_usage_max) / 2), 0.4))
        base_usage = clamp(base_usage, resource_usage_min, resource_usage_max)

        for j in 1:num_products
            if rand() < sparsity
                usage_matrix[i, j] = 0.0
                continue
            end

            random_component = rand(Gamma(2, resource_usage_max / 6))
            random_component = min(random_component, resource_usage_max / 2)

            quality_multiplier = 0.5 + correlation_strength * quality_factors[j]
            usage = base_usage * quality_multiplier + random_component * (1 - correlation_strength)
            usage_matrix[i, j] = max(0.0, usage)
        end
    end

    # Ensure each product uses at least one resource
    for j in 1:num_products
        if all(usage_matrix[:, j] .== 0)
            resource_idx = rand(1:num_resources)
            usage_value = rand(LogNormal(log((resource_usage_min + resource_usage_max) / 2), 0.3))
            usage_matrix[resource_idx, j] = clamp(usage_value, resource_usage_min, resource_usage_max)
        end
    end

    # Ensure each resource is used by at least one product
    for i in 1:num_resources
        if all(usage_matrix[i, :] .== 0)
            product_idx = rand(1:num_products)
            usage_value = rand(LogNormal(log((resource_usage_min + resource_usage_max) / 2), 0.3))
            usage_matrix[i, product_idx] = clamp(usage_value, resource_usage_min, resource_usage_max)
        end
    end

    # Generate resource availabilities
    avg_usage_per_resource = mean(usage_matrix, dims=2) .* num_products / 2

    variability = rand(LogNormal(log(1.0), 0.3), num_resources)
    variability = clamp.(variability, 0.5, 2.0)

    constraint_factors = rand(Beta(3, 2), num_resources)
    constraint_factors = 0.6 .+ 0.6 * constraint_factors

    availabilities = avg_usage_per_resource[:] .* variability .* constraint_factors

    # Calculate max possible production for each product
    max_possible = Float64[]
    for j in 1:num_products
        resource_limits = Float64[]
        for i in 1:num_resources
            if usage_matrix[i, j] > 0
                push!(resource_limits, availabilities[i] / usage_matrix[i, j])
            else
                push!(resource_limits, Inf)
            end
        end
        push!(max_possible, minimum(resource_limits))
    end

    # Initialize market bounds
    lower_bounds = zeros(num_products)
    upper_bounds = fill(Inf, num_products)

    # Add market constraints
    for j in 1:num_products
        if rand() < market_constraint_prob/2
            lower_factor = rand(Beta(2, 6))
            lower_bounds[j] = lower_factor * 0.35 * max_possible[j]
        end

        if rand() < market_constraint_prob/2
            upper_factor = rand(Beta(4, 2))
            upper_bounds[j] = (0.4 + upper_factor * 0.55) * max_possible[j]
        end
    end

    # Adjust for feasibility status
    solution_status = feasibility_status == feasible ? :feasible :
                     feasibility_status == infeasible ? :infeasible : :all

    if solution_status == :feasible
        # Ensure per-product bounds are consistent
        for j in 1:num_products
            if isfinite(upper_bounds[j]) && lower_bounds[j] > upper_bounds[j]
                lower_bounds[j] = max(0.0, 0.98 * upper_bounds[j])
            end
        end

        # Scale lower bounds if aggregate demand exceeds capacity
        required = [sum(usage_matrix[i, j] * lower_bounds[j] for j in 1:num_products) for i in 1:num_resources]
        scales = Float64[]
        for i in 1:num_resources
            req_i = required[i]
            if req_i > 0
                push!(scales, availabilities[i] / req_i)
            end
        end
        if !isempty(scales)
            lb_scale = min(1.0, minimum(scales))
            if lb_scale < 1.0
                lower_bounds .*= lb_scale * 0.98
            end
        end

        # Final guard for LB vs UB after scaling
        for j in 1:num_products
            if isfinite(upper_bounds[j]) && lower_bounds[j] > upper_bounds[j]
                lower_bounds[j] = max(0.0, 0.98 * upper_bounds[j])
            end
        end

    elseif solution_status == :infeasible
        # Ensure at least some positive lower bounds
        if all(lower_bounds .== 0.0)
            critical_res = argmax([sum(usage_matrix[i, :]) for i in 1:num_resources])
            candidates = [j for j in 1:num_products if usage_matrix[critical_res, j] > 0.0]
            if isempty(candidates)
                candidates = collect(1:num_products)
            end
            num_assign = max(1, round(Int, 0.2 * length(candidates)))
            selected = sample(candidates, min(num_assign, length(candidates)); replace=false)
            for j in selected
                lb_factor = 0.15 + 0.25 * rand()
                lower_bounds[j] = max(lower_bounds[j], lb_factor * max_possible[j])
            end
        end

        # Recompute required usage
        required = [sum(usage_matrix[i, j] * lower_bounds[j] for j in 1:num_products) for i in 1:num_resources]

        # Pick most stressed resource
        ratios = [required[i] > 0 ? required[i] / max(availabilities[i], eps()) : 0.0 for i in 1:num_resources]
        critical_i = argmax(ratios)
        if required[critical_i] == 0.0
            critical_i = argmax([sum(usage_matrix[i, :]) for i in 1:num_resources])
            if sum(usage_matrix[critical_i, :]) == 0.0
                critical_i = 1
            end
            if all(usage_matrix[critical_i, :] .== 0.0)
                pj = rand(1:num_products)
                usage_matrix[critical_i, pj] = max(usage_matrix[critical_i, pj], resource_usage_min)
                required[critical_i] = usage_matrix[critical_i, pj] * max(lower_bounds[pj], resource_usage_min)
            end
        end

        # Reduce availability to create infeasibility
        shortage_margin = 0.10 + 0.25 * rand()
        new_avail = max(0.0, required[critical_i] * (1.0 - shortage_margin))
        availabilities[critical_i] = new_avail

        # Optionally reduce another resource
        if num_resources >= 2 && rand() < 0.3
            other_i = critical_i % num_resources + 1
            req_other = required[other_i]
            if req_other > 0.0
                availabilities[other_i] = max(0.0, req_other * (1.0 - (0.05 + 0.15 * rand())))
            end
        end
    end

    return ProductMixProblem(num_products, num_resources, profits, usage_matrix, availabilities, lower_bounds, upper_bounds)
end

"""
    build_model(prob::ProductMixProblem)

Build a JuMP model for the product mix problem.

# Arguments
- `prob`: ProductMixProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::ProductMixProblem)
    model = Model()

    # Decision variables
    @variable(model, x[1:prob.num_products] >= 0)

    # Objective
    @objective(model, Max, sum(prob.profits[j] * x[j] for j in 1:prob.num_products))

    # Resource constraints
    for i in 1:prob.num_resources
        @constraint(model, sum(prob.usage_matrix[i, j] * x[j] for j in 1:prob.num_products) <= prob.availabilities[i])
    end

    # Market constraints
    for j in 1:prob.num_products
        if prob.lower_bounds[j] > 0
            @constraint(model, x[j] >= prob.lower_bounds[j])
        end

        if prob.upper_bounds[j] < Inf
            @constraint(model, x[j] <= prob.upper_bounds[j])
        end
    end

    return model
end

# Register the problem type
register_problem(
    :product_mix,
    ProductMixProblem,
    "Product mix optimization problem that maximizes profit by determining optimal production quantities subject to resource constraints"
)
