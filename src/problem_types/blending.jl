using JuMP
using Random
using Distributions

"""
Blending problem variants.

# Variants
- `blend_standard`: Basic blending - minimize cost meeting quality requirements
- `blend_stability`: Stability constraints (shelf-life, temperature stability)
- `blend_safety`: Maximum toxin/contaminant level constraints
- `blend_equipment`: Equipment capacity limits (mixer size, batch constraints)
- `blend_ratio`: Specific ingredient ratio requirements
- `blend_multi_product`: Multiple products from shared ingredient pool
- `blend_max_quality`: Maximize quality score subject to budget
- `blend_target_match`: Match a target quality profile
"""
@enum BlendingVariant begin
    blend_standard
    blend_stability
    blend_safety
    blend_equipment
    blend_ratio
    blend_multi_product
    blend_max_quality
    blend_target_match
end

"""
    BlendingProblem <: ProblemGenerator

Generator for blending problems with multiple variants.
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
    variant::BlendingVariant
    # Stability variant
    stability_coeffs::Union{Vector{Float64}, Nothing}
    min_stability::Float64
    # Safety variant
    contaminant_levels::Union{Vector{Float64}, Nothing}
    max_contaminant::Float64
    # Equipment variant
    mixer_capacity::Float64
    n_batches::Int
    # Ratio variant
    ratio_pairs::Union{Vector{Tuple{Int,Int,Float64,Float64}}, Nothing}  # (i, j, min_ratio, max_ratio) meaning min <= x[i]/x[j] <= max
    # Multi-product variant
    n_products::Int
    product_amounts::Union{Vector{Float64}, Nothing}
    product_quality_lower::Union{Matrix{Float64}, Nothing}
    product_quality_upper::Union{Matrix{Float64}, Nothing}
    # Target match variant
    target_profile::Union{Vector{Float64}, Nothing}
    match_tolerance::Float64
end

# Backwards compatibility
function BlendingProblem(n_ingredients::Int, n_attributes::Int, costs::Vector{Int},
                         attributes::Matrix{Float64}, lower_bounds::Vector{Float64},
                         upper_bounds::Vector{Float64}, supply_limits::Vector{Float64},
                         cost_budget::Float64, min_blend_amount::Float64,
                         min_usage_required::Dict{Int,Float64}, max_usage_limits::Dict{Int,Float64})
    BlendingProblem(
        n_ingredients, n_attributes, costs, attributes, lower_bounds, upper_bounds,
        supply_limits, cost_budget, min_blend_amount, min_usage_required, max_usage_limits,
        blend_standard,
        nothing, 0.0,
        nothing, 0.0,
        0.0, 0,
        nothing,
        0, nothing, nothing, nothing,
        nothing, 0.0
    )
end

"""
    BlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                    variant::BlendingVariant=blend_standard)

Construct a blending problem instance with the specified variant.
"""
function BlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                         variant::BlendingVariant=blend_standard)
    Random.seed!(seed)

    n_ingredients = max(3, min(500, target_variables))
    n_attributes = rand(2:min(15, max(2, n_ingredients ÷ 3)))
    min_blend_amount = Float64(rand(100:20000))

    cost_range = (10, 100)
    attribute_range = (0.1, 0.9)

    min_cost, max_cost = cost_range
    min_attr, max_attr = attribute_range

    costs = rand(min_cost:max_cost, n_ingredients)
    attributes = rand(min_attr:0.01:max_attr, n_ingredients, n_attributes)

    # Base quality bounds
    lower_bounds = zeros(n_attributes)
    upper_bounds = zeros(n_attributes)
    for j in 1:n_attributes
        avg_attr = sum(attributes[:, j]) / n_ingredients
        lower_bounds[j] = avg_attr * rand(Uniform(0.7, 0.9))
        upper_bounds[j] = avg_attr * rand(Uniform(1.1, 1.3))
    end

    # Supply limits
    supply_limits = [min_blend_amount * rand(Uniform(0.3, 1.5)) for _ in 1:n_ingredients]

    # Cost budget
    avg_cost = sum(costs) / n_ingredients
    cost_budget = avg_cost * min_blend_amount * rand(Uniform(1.1, 1.5))

    min_usage_required = Dict{Int,Float64}()
    max_usage_limits = Dict{Int,Float64}()

    # Initialize variant fields
    stability_coeffs = nothing
    min_stability = 0.0
    contaminant_levels = nothing
    max_contaminant = 0.0
    mixer_capacity = 0.0
    n_batches = 0
    ratio_pairs = nothing
    n_products = 0
    product_amounts = nothing
    product_quality_lower = nothing
    product_quality_upper = nothing
    target_profile = nothing
    match_tolerance = 0.0

    if variant == blend_stability
        # Stability coefficients for each ingredient
        stability_coeffs = rand(Uniform(0.5, 1.0), n_ingredients)
        min_stability = sum(stability_coeffs) / n_ingredients * rand(Uniform(0.7, 0.85))

        if feasibility_status == infeasible
            min_stability = maximum(stability_coeffs) * 1.1
        end

    elseif variant == blend_safety
        # Contaminant levels per ingredient
        contaminant_levels = rand(Uniform(0.0, 0.1), n_ingredients)
        max_contaminant = 0.05 * min_blend_amount

        if feasibility_status == infeasible
            max_contaminant = minimum(contaminant_levels) * min_blend_amount * 0.5
        end

    elseif variant == blend_equipment
        # Mixer capacity limits
        mixer_capacity = min_blend_amount * rand(Uniform(0.3, 0.6))
        n_batches = ceil(Int, min_blend_amount / mixer_capacity) + rand(0:2)

        if feasibility_status == infeasible
            n_batches = max(1, floor(Int, min_blend_amount / mixer_capacity) - 1)
        end

    elseif variant == blend_ratio
        # Ratio constraints between ingredient pairs
        n_ratio_constraints = rand(2:min(5, n_ingredients ÷ 2))
        ratio_pairs = Tuple{Int,Int,Float64,Float64}[]

        for _ in 1:n_ratio_constraints
            i = rand(1:n_ingredients)
            j = rand(1:n_ingredients)
            while j == i
                j = rand(1:n_ingredients)
            end
            min_ratio = rand(Uniform(0.5, 1.5))
            max_ratio = min_ratio * rand(Uniform(1.2, 2.0))
            push!(ratio_pairs, (i, j, min_ratio, max_ratio))
        end

        if feasibility_status == infeasible
            # Contradicting ratios
            push!(ratio_pairs, (1, 2, 2.0, 3.0))
            push!(ratio_pairs, (2, 1, 2.0, 3.0))  # Impossible: x1/x2 >= 2 AND x2/x1 >= 2
        end

    elseif variant == blend_multi_product
        # Multiple products from shared ingredients
        n_products = rand(2:min(4, max(2, n_ingredients ÷ 3)))
        product_amounts = [min_blend_amount / n_products * rand(Uniform(0.8, 1.2)) for _ in 1:n_products]

        product_quality_lower = zeros(n_products, n_attributes)
        product_quality_upper = zeros(n_products, n_attributes)

        for p in 1:n_products
            for j in 1:n_attributes
                product_quality_lower[p, j] = lower_bounds[j] * rand(Uniform(0.9, 1.1))
                product_quality_upper[p, j] = upper_bounds[j] * rand(Uniform(0.9, 1.1))
            end
        end

        # Increase supply to accommodate multiple products
        total_needed = sum(product_amounts)
        supply_limits .*= (total_needed / min_blend_amount) * 1.2

        if feasibility_status == infeasible
            # Not enough supply for all products
            supply_limits .*= 0.3
        end

    elseif variant == blend_max_quality
        # Maximize quality instead of minimize cost
        # Quality is a weighted sum of attributes
        # The objective changes but we use existing structures

    elseif variant == blend_target_match
        # Match a target quality profile
        target_profile = rand(Uniform(min_attr, max_attr), n_attributes)
        match_tolerance = rand(Uniform(0.02, 0.08))

        # Update bounds to be around target
        for j in 1:n_attributes
            lower_bounds[j] = target_profile[j] * (1 - match_tolerance)
            upper_bounds[j] = target_profile[j] * (1 + match_tolerance)
        end

        if feasibility_status == infeasible
            # Target impossible to achieve
            target_profile = [maximum(attributes[:, j]) * 1.1 for j in 1:n_attributes]
            for j in 1:n_attributes
                lower_bounds[j] = target_profile[j] * 0.99
                upper_bounds[j] = target_profile[j] * 1.01
            end
        end
    end

    # Feasibility adjustments for standard variant
    if feasibility_status == infeasible && variant == blend_standard
        # Make quality requirements impossible
        for j in 1:min(2, n_attributes)
            lower_bounds[j] = maximum(attributes[:, j]) * 1.1
        end
    elseif feasibility_status == feasible
        # Ensure achievable
        supply_limits .*= 1.5
        cost_budget *= 1.3
    end

    return BlendingProblem(
        n_ingredients, n_attributes, costs, attributes, lower_bounds, upper_bounds,
        supply_limits, cost_budget, min_blend_amount, min_usage_required, max_usage_limits,
        variant,
        stability_coeffs, min_stability,
        contaminant_levels, max_contaminant,
        mixer_capacity, n_batches,
        ratio_pairs,
        n_products, product_amounts, product_quality_lower, product_quality_upper,
        target_profile, match_tolerance
    )
end

"""
    build_model(prob::BlendingProblem)

Build a JuMP model for the blending problem based on its variant.
"""
function build_model(prob::BlendingProblem)
    model = Model()

    if prob.variant == blend_standard || prob.variant == blend_stability ||
       prob.variant == blend_safety || prob.variant == blend_ratio ||
       prob.variant == blend_max_quality || prob.variant == blend_target_match

        @variable(model, x[1:prob.n_ingredients] >= 0)

        if prob.variant == blend_max_quality
            # Maximize weighted quality
            quality_weights = [1.0 / prob.n_attributes for _ in 1:prob.n_attributes]
            @objective(model, Max, sum(quality_weights[j] *
                sum(prob.attributes[i, j] * x[i] for i in 1:prob.n_ingredients) / sum(x)
                for j in 1:prob.n_attributes))
        else
            @objective(model, Min, sum(prob.costs[i] * x[i] for i in 1:prob.n_ingredients))
        end

        # Minimum blend amount
        @constraint(model, sum(x[i] for i in 1:prob.n_ingredients) >= prob.min_blend_amount)

        # Supply limits
        for i in 1:prob.n_ingredients
            if prob.supply_limits[i] < Inf
                @constraint(model, x[i] <= prob.supply_limits[i])
            end
        end

        # Cost budget (except for max quality)
        if prob.cost_budget < Inf && prob.variant != blend_max_quality
            @constraint(model, sum(prob.costs[i] * x[i] for i in 1:prob.n_ingredients) <= prob.cost_budget)
        elseif prob.variant == blend_max_quality
            # Cost becomes a constraint
            @constraint(model, sum(prob.costs[i] * x[i] for i in 1:prob.n_ingredients) <= prob.cost_budget)
        end

        # Quality bounds
        for j in 1:prob.n_attributes
            @constraint(model,
                sum(prob.attributes[i, j] * x[i] for i in 1:prob.n_ingredients) >=
                prob.lower_bounds[j] * sum(x[i] for i in 1:prob.n_ingredients))
            @constraint(model,
                sum(prob.attributes[i, j] * x[i] for i in 1:prob.n_ingredients) <=
                prob.upper_bounds[j] * sum(x[i] for i in 1:prob.n_ingredients))
        end

        # Variant-specific constraints
        if prob.variant == blend_stability && prob.stability_coeffs !== nothing
            @constraint(model,
                sum(prob.stability_coeffs[i] * x[i] for i in 1:prob.n_ingredients) >=
                prob.min_stability * sum(x[i] for i in 1:prob.n_ingredients))

        elseif prob.variant == blend_safety && prob.contaminant_levels !== nothing
            @constraint(model,
                sum(prob.contaminant_levels[i] * x[i] for i in 1:prob.n_ingredients) <= prob.max_contaminant)

        elseif prob.variant == blend_ratio && prob.ratio_pairs !== nothing
            for (i, j, min_ratio, max_ratio) in prob.ratio_pairs
                @constraint(model, x[i] >= min_ratio * x[j])
                @constraint(model, x[i] <= max_ratio * x[j])
            end
        end

    elseif prob.variant == blend_equipment
        # Batch-based blending
        @variable(model, x[1:prob.n_ingredients, 1:prob.n_batches] >= 0)

        @objective(model, Min, sum(prob.costs[i] * sum(x[i, b] for b in 1:prob.n_batches)
            for i in 1:prob.n_ingredients))

        # Total production meets minimum
        @constraint(model, sum(x[i, b] for i in 1:prob.n_ingredients, b in 1:prob.n_batches) >= prob.min_blend_amount)

        # Mixer capacity per batch
        for b in 1:prob.n_batches
            @constraint(model, sum(x[i, b] for i in 1:prob.n_ingredients) <= prob.mixer_capacity)
        end

        # Supply limits
        for i in 1:prob.n_ingredients
            if prob.supply_limits[i] < Inf
                @constraint(model, sum(x[i, b] for b in 1:prob.n_batches) <= prob.supply_limits[i])
            end
        end

        # Quality bounds (per batch)
        for b in 1:prob.n_batches
            for j in 1:prob.n_attributes
                @constraint(model,
                    sum(prob.attributes[i, j] * x[i, b] for i in 1:prob.n_ingredients) >=
                    prob.lower_bounds[j] * sum(x[i, b] for i in 1:prob.n_ingredients))
                @constraint(model,
                    sum(prob.attributes[i, j] * x[i, b] for i in 1:prob.n_ingredients) <=
                    prob.upper_bounds[j] * sum(x[i, b] for i in 1:prob.n_ingredients))
            end
        end

    elseif prob.variant == blend_multi_product
        # Ingredient allocation to multiple products
        @variable(model, x[1:prob.n_ingredients, 1:prob.n_products] >= 0)

        @objective(model, Min, sum(prob.costs[i] * sum(x[i, p] for p in 1:prob.n_products)
            for i in 1:prob.n_ingredients))

        # Product amounts
        for p in 1:prob.n_products
            @constraint(model, sum(x[i, p] for i in 1:prob.n_ingredients) >= prob.product_amounts[p])
        end

        # Supply limits
        for i in 1:prob.n_ingredients
            if prob.supply_limits[i] < Inf
                @constraint(model, sum(x[i, p] for p in 1:prob.n_products) <= prob.supply_limits[i])
            end
        end

        # Cost budget
        if prob.cost_budget < Inf
            @constraint(model, sum(prob.costs[i] * sum(x[i, p] for p in 1:prob.n_products)
                for i in 1:prob.n_ingredients) <= prob.cost_budget)
        end

        # Quality bounds per product
        for p in 1:prob.n_products
            for j in 1:prob.n_attributes
                @constraint(model,
                    sum(prob.attributes[i, j] * x[i, p] for i in 1:prob.n_ingredients) >=
                    prob.product_quality_lower[p, j] * sum(x[i, p] for i in 1:prob.n_ingredients))
                @constraint(model,
                    sum(prob.attributes[i, j] * x[i, p] for i in 1:prob.n_ingredients) <=
                    prob.product_quality_upper[p, j] * sum(x[i, p] for i in 1:prob.n_ingredients))
            end
        end
    end

    return model
end

# Register the problem type
register_problem(
    :blending,
    BlendingProblem,
    "Blending problem with variants including standard, stability, safety, equipment, ratio, multi-product, max quality, and target matching"
)
