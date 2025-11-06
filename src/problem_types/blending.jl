using JuMP
using Random

"""
    BlendingProblem <: ProblemGenerator

Generator for blending problems that minimize cost while meeting quality requirements.

# Fields
- `n_ingredients::Int`: Number of ingredients
- `n_attributes::Int`: Number of quality attributes
- `costs::Vector{Int}`: Cost per unit of each ingredient
- `attributes::Matrix{Float64}`: Attribute values (n_ingredients × n_attributes)
- `lower_bounds::Vector{Float64}`: Minimum required value for each attribute
- `upper_bounds::Vector{Float64}`: Maximum allowed value for each attribute
- `supply_limits::Vector{Float64}`: Maximum available amount of each ingredient
- `cost_budget::Float64`: Maximum total cost allowed
- `min_blend_amount::Float64`: Minimum amount to produce
- `min_usage_required::Dict{Int,Float64}`: Minimum usage requirements (optional)
- `max_usage_limits::Dict{Int,Float64}`: Maximum usage limits (optional)
"""
struct BlendingProblem <: ProblemGenerator
    n_ingredients::Int
    n_attributes::Int
    costs::Vector{Int}
    attributes::Matrix{Float64}
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    supply_limits::Vector{Float64}
    cost_budget::Float64
    min_blend_amount::Float64
    min_usage_required::Dict{Int,Float64}
    max_usage_limits::Dict{Int,Float64}
end

"""
    BlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a blending problem instance.

# Arguments
- `target_variables`: Target number of variables (ingredients)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function BlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # For blending, variables = n_ingredients
    n_ingredients = max(3, min(500, target_variables))
    n_attributes = rand(2:15)
    min_blend_amount = Float64(rand(100:20000))

    cost_range = (10, 100)
    attribute_range = (0.1, 0.9)

    min_cost, max_cost = cost_range
    min_attr, max_attr = attribute_range

    # Generate data
    costs = rand(min_cost:max_cost, n_ingredients)
    attributes = rand(min_attr:0.01:max_attr, n_ingredients, n_attributes)

    # Determine actual status
    solution_status = feasibility_status == feasible ? :feasible :
                     feasibility_status == infeasible ? :infeasible : :all
    actual_status = solution_status
    if solution_status == :all
        actual_status = rand() < 0.5 ? :feasible : :infeasible
    end

    # Initialize
    lower_bounds = zeros(n_attributes)
    upper_bounds = zeros(n_attributes)
    supply_limits = fill(Inf, n_ingredients)
    cost_budget = Inf
    min_usage_required = Dict{Int,Float64}()
    max_usage_limits = Dict{Int,Float64}()

    if actual_status == :feasible
        # Generate intelligent baseline solution
        cost_efficiency = zeros(n_ingredients)
        for i in 1:n_ingredients
            quality_score = sum(attributes[i, :]) / n_attributes
            cost_efficiency[i] = quality_score / costs[i]
        end

        efficiency_order = sortperm(cost_efficiency, rev=true)

        blend_amounts = zeros(n_ingredients)
        primary_count = max(3, round(Int, n_ingredients * 0.6))
        primary_ingredients = efficiency_order[1:primary_count]

        primary_total = min_blend_amount * 0.8
        for i in primary_ingredients
            efficiency_weight = cost_efficiency[i] / sum(cost_efficiency[primary_ingredients])
            base_amount = primary_total * efficiency_weight
            blend_amounts[i] = base_amount * (0.8 + rand() * 0.4)
        end

        secondary_total = min_blend_amount * 0.2
        secondary_ingredients = efficiency_order[(primary_count + 1):end]
        for i in secondary_ingredients
            if !isempty(secondary_ingredients)
                blend_amounts[i] = secondary_total / length(secondary_ingredients) * (0.5 + rand())
            end
        end

        total_amount = sum(blend_amounts)
        blend_amounts .*= min_blend_amount / total_amount

        # Calculate achieved qualities
        achieved_qualities = zeros(n_attributes)
        for j in 1:n_attributes
            achieved_qualities[j] = sum(attributes[i, j] * blend_amounts[i] for i in 1:n_ingredients) / sum(blend_amounts)
        end

        # Set tight quality bounds
        scenario = rand(1:3)
        tolerance_level = if scenario == 1
            0.025 + rand() * 0.025
        elseif scenario == 2
            0.04 + rand() * 0.03
        else
            0.06 + rand() * 0.02
        end

        for j in 1:n_attributes
            tolerance = tolerance_level
            position_in_band = 0.6 + rand() * 0.2

            total_range = 2 * tolerance * achieved_qualities[j] / (1 - 2 * tolerance + 2 * tolerance * position_in_band)
            lower_bound = achieved_qualities[j] - total_range * position_in_band
            upper_bound = lower_bound + total_range

            lower_bounds[j] = max(min_attr, lower_bound)
            upper_bounds[j] = min(max_attr, upper_bound)
        end

        # Supply constraints
        critical_ingredients = primary_ingredients[1:max(2, div(length(primary_ingredients), 2))]

        for i in 1:n_ingredients
            if i in critical_ingredients
                supply_limits[i] = blend_amounts[i] * (1.1 + rand() * 0.2)
            else
                supply_limits[i] = blend_amounts[i] * (1.3 + rand() * 0.4)
            end
        end

        # Cost budget
        actual_cost = sum(costs[i] * blend_amounts[i] for i in 1:n_ingredients)
        cost_pressure = rand(1:3)
        if cost_pressure == 1
            cost_budget = actual_cost * (1.06 + rand() * 0.06)
        elseif cost_pressure == 2
            cost_budget = actual_cost * (1.10 + rand() * 0.06)
        else
            cost_budget = actual_cost * (1.15 + rand() * 0.10)
        end

        # Additional constraints
        required_ingredients = randperm(n_ingredients)[1:max(1, div(n_ingredients, 4))]
        for i in required_ingredients
            min_required = blend_amounts[i] * (0.7 + rand() * 0.2)
            min_usage_required[i] = min_required
        end

        limited_ingredients = randperm(n_ingredients)[1:max(1, div(n_ingredients, 3))]
        for i in limited_ingredients
            max_limit = blend_amounts[i] * (1.2 + rand() * 0.3)
            max_usage_limits[i] = max_limit
        end

    else  # :infeasible
        scenario = rand(1:4)

        if scenario == 1
            # Supply shortage conflict
            quality_leaders = Vector{Int}[]
            for j in 1:n_attributes
                quality_values = [(attributes[i, j], i) for i in 1:n_ingredients]
                sort!(quality_values, rev=true)
                top_count = max(1, div(n_ingredients, 4))
                leaders = [pair[2] for pair in quality_values[1:top_count]]
                push!(quality_leaders, leaders)
            end

            for j in 1:n_attributes
                min_quality_needed = maximum(attributes[i, j] for i in quality_leaders[j]) * 0.95
                lower_bounds[j] = max(min_attr, min_quality_needed)
                upper_bounds[j] = min(max_attr, min_quality_needed * 1.05)
            end

            critical_ingredients = unique(vcat(quality_leaders...))
            for i in critical_ingredients
                supply_limits[i] = min_blend_amount * 0.15 / length(critical_ingredients)
            end

            for i in 1:n_ingredients
                if !(i in critical_ingredients)
                    supply_limits[i] = min_blend_amount * 2.0
                end
            end

            cost_budget = maximum(costs) * min_blend_amount * 2.0

        elseif scenario == 2
            # Budget impossibility
            for j in 1:n_attributes
                best_quality = maximum(attributes[:, j])
                lower_bounds[j] = max(min_attr, best_quality * 0.95)
                upper_bounds[j] = min(max_attr, best_quality * 1.05)
            end

            qualifying_ingredients = Int[]
            for i in 1:n_ingredients
                can_qualify = true
                for j in 1:n_attributes
                    if attributes[i, j] < lower_bounds[j]
                        can_qualify = false
                        break
                    end
                end
                if can_qualify
                    push!(qualifying_ingredients, i)
                end
            end

            if !isempty(qualifying_ingredients)
                min_cost_per_unit = minimum(costs[i] for i in qualifying_ingredients)
                min_total_cost = min_cost_per_unit * min_blend_amount
            else
                min_total_cost = maximum(costs) * min_blend_amount
            end

            cost_budget = min_total_cost * (0.6 + rand() * 0.3)

            for i in 1:n_ingredients
                supply_limits[i] = min_blend_amount * 3.0
            end

        elseif scenario == 3
            # Impossible quality conflict
            if n_attributes >= 2
                for j in 1:min(n_attributes, 3)
                    max_possible = maximum(attributes[:, j])
                    lower_bounds[j] = max(min_attr, max_possible * (0.98 + rand() * 0.02))
                    upper_bounds[j] = min(max_attr, lower_bounds[j] * 1.01)
                end

                for j in (min(n_attributes, 3) + 1):n_attributes
                    avg_val = sum(attributes[:, j]) / n_ingredients
                    lower_bounds[j] = max(min_attr, avg_val * 0.8)
                    upper_bounds[j] = min(max_attr, avg_val * 1.2)
                end
            end

            for i in 1:n_ingredients
                supply_limits[i] = min_blend_amount * 2.0
            end
            cost_budget = maximum(costs) * min_blend_amount * 2.0

        else  # scenario == 4
            # Over-constrained system
            for j in 1:n_attributes
                if j <= 2
                    max_val = maximum(attributes[:, j])
                    lower_bounds[j] = max(min_attr, max_val * 0.97)
                    upper_bounds[j] = min(max_attr, max_val * 1.01)
                else
                    mid_val = (minimum(attributes[:, j]) + maximum(attributes[:, j])) / 2
                    lower_bounds[j] = max(min_attr, mid_val * 0.95)
                    upper_bounds[j] = min(max_attr, mid_val * 1.05)
                end
            end

            expensive_ingredients = sortperm(costs, rev=true)[1:max(2, div(n_ingredients, 3))]
            for i in expensive_ingredients
                supply_limits[i] = min_blend_amount * 0.2
            end

            for i in 1:n_ingredients
                if !(i in expensive_ingredients)
                    supply_limits[i] = min_blend_amount * 1.5
                end
            end

            avg_cost = sum(costs) / n_ingredients
            cost_budget = avg_cost * min_blend_amount * 0.9
        end
    end

    return BlendingProblem(n_ingredients, n_attributes, costs, attributes, lower_bounds, upper_bounds,
                          supply_limits, cost_budget, min_blend_amount, min_usage_required, max_usage_limits)
end

"""
    build_model(prob::BlendingProblem)

Build a JuMP model for the blending problem.

# Arguments
- `prob`: BlendingProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::BlendingProblem)
    model = Model()

    # Variables
    @variable(model, x[1:prob.n_ingredients] >= 0)

    # Objective
    @objective(model, Min, sum(prob.costs[i] * x[i] for i in 1:prob.n_ingredients))

    # Minimum blend amount
    @constraint(model, sum(x[i] for i in 1:prob.n_ingredients) >= prob.min_blend_amount)

    # Supply limits
    for i in 1:prob.n_ingredients
        if prob.supply_limits[i] < Inf
            @constraint(model, x[i] <= prob.supply_limits[i])
        end
    end

    # Cost budget
    if prob.cost_budget < Inf
        @constraint(model, sum(prob.costs[i] * x[i] for i in 1:prob.n_ingredients) <= prob.cost_budget)
    end

    # Additional constraints
    for (i, min_amount) in prob.min_usage_required
        @constraint(model, x[i] >= min_amount)
    end

    for (i, max_amount) in prob.max_usage_limits
        @constraint(model, x[i] <= max_amount)
    end

    # Quality attribute bounds
    for j in 1:prob.n_attributes
        @constraint(model,
            sum(prob.attributes[i, j] * x[i] for i in 1:prob.n_ingredients) >=
            prob.lower_bounds[j] * sum(x[i] for i in 1:prob.n_ingredients)
        )

        @constraint(model,
            sum(prob.attributes[i, j] * x[i] for i in 1:prob.n_ingredients) <=
            prob.upper_bounds[j] * sum(x[i] for i in 1:prob.n_ingredients)
        )
    end

    return model
end

# Register the problem type
register_problem(
    :blending,
    BlendingProblem,
    "Blending problem that minimizes cost while meeting quality requirements for a mixture"
)
