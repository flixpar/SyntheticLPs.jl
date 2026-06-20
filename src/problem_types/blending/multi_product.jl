using JuMP
using Random
using Distributions

"""
    MultiProductBlendingProblem <: ProblemGenerator

Generator for multi-product blending problems that allocate a shared pool of
ingredients across several products at minimum cost.

# Overview
A single set of raw ingredients is blended into multiple distinct products. The
decisions are `x[i, p]`, the amount of ingredient `i` allocated to product `p`.
The objective minimizes total ingredient cost. Each product must reach a minimum
production amount and satisfy its own per-attribute quality band (a min/max on the
weighted average of each attribute). Each ingredient has a shared supply limit
spanning all products, and a global cost budget caps total spend.

# Fields
- `n_ingredients::Int`: Number of shared raw ingredients
- `n_products::Int`: Number of products produced from the shared pool
- `n_attributes::Int`: Number of quality attributes per product
- `costs::Vector{Int}`: Cost per unit of each ingredient
- `attributes::Matrix{Float64}`: Per-unit attribute values (`n_ingredients` × `n_attributes`)
- `supply_limits::Vector{Float64}`: Shared supply available for each ingredient (across all products)
- `cost_budget::Float64`: Maximum total blending cost
- `product_amounts::Vector{Float64}`: Minimum amount required for each product
- `product_quality_lower::Matrix{Float64}`: Lower quality band per product/attribute (`n_products` × `n_attributes`)
- `product_quality_upper::Matrix{Float64}`: Upper quality band per product/attribute (`n_products` × `n_attributes`)
"""
struct MultiProductBlendingProblem <: ProblemGenerator
    n_ingredients::Int
    n_products::Int
    n_attributes::Int
    costs::Vector{Int}
    attributes::Matrix{Float64}
    supply_limits::Vector{Float64}
    cost_budget::Float64
    product_amounts::Vector{Float64}
    product_quality_lower::Matrix{Float64}
    product_quality_upper::Matrix{Float64}
end

"""
    MultiProductBlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-product blending problem instance.

Variables: `x[i, p]` for ingredient `i` and product `p`.
Total = `n_ingredients * n_products`. The constructor first samples `n_products`,
then sets `n_ingredients = round(target_variables / n_products)` so that the
product lands near `target_variables`.

# Arguments
- `target_variables`: Target number of variables (`n_ingredients * n_products`)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function MultiProductBlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # Var-count formula: n_ingredients * n_products.
    # Choose n_products first, then size n_ingredients = round(target / n_products)
    # so the product of dimensions is close to target_variables.
    n_products = rand(2:4)
    n_ingredients = max(3, round(Int, target_variables / n_products))
    n_attributes = rand(2:min(10, max(2, n_ingredients ÷ 3)))

    min_blend_amount = Float64(rand(100:20000))

    # Ingredient costs and per-unit attribute values
    costs = rand(10:100, n_ingredients)
    min_attr, max_attr = 0.1, 0.9
    attributes = rand(min_attr:0.01:max_attr, n_ingredients, n_attributes)

    # Column (attribute) averages over ingredients -> the equal-weight blend
    # achieves exactly these averages, giving us a known feasible quality point.
    attr_avg = [sum(attributes[:, j]) / n_ingredients for j in 1:n_attributes]

    # Per-product required amounts
    product_amounts = [min_blend_amount / n_products * rand(Uniform(0.8, 1.2)) for _ in 1:n_products]
    total_needed = sum(product_amounts)

    # Per-product quality bands around the attribute averages. Because the band
    # straddles attr_avg[j], an (approximately) uniform blend satisfies it.
    product_quality_lower = zeros(n_products, n_attributes)
    product_quality_upper = zeros(n_products, n_attributes)
    for p in 1:n_products
        for j in 1:n_attributes
            product_quality_lower[p, j] = attr_avg[j] * rand(Uniform(0.6, 0.85))
            product_quality_upper[p, j] = attr_avg[j] * rand(Uniform(1.15, 1.4))
        end
    end

    # Shared per-ingredient supply. Sum of supplies must comfortably exceed total
    # product demand for a feasible allocation to exist.
    supply_limits = [rand(Uniform(0.3, 1.5)) for _ in 1:n_ingredients]
    supply_scale = (total_needed / sum(supply_limits)) * 1.5
    supply_limits .*= supply_scale

    # Cost budget sized off the average cost of the total production.
    avg_cost = sum(costs) / n_ingredients
    cost_budget = avg_cost * total_needed * rand(Uniform(1.3, 1.8))

    # --- Feasibility handling ---
    actual_status = feasibility_status
    if feasibility_status == unknown
        actual_status = rand() < 0.7 ? feasible : infeasible
    end

    if actual_status == feasible
        # The equal-weight allocation (each ingredient contributes
        # product_amounts[p]/n_ingredients to product p) hits attr_avg exactly,
        # satisfies the (straddling) quality bands, and uses
        # sum_p product_amounts[p]/n_ingredients per ingredient. Guarantee supply
        # and budget cover this reference allocation with margin.
        per_ingredient_use = total_needed / n_ingredients
        for i in 1:n_ingredients
            supply_limits[i] = max(supply_limits[i], per_ingredient_use * 1.5)
        end
        ref_cost = sum(costs[i] * per_ingredient_use for i in 1:n_ingredients)
        cost_budget = max(cost_budget, ref_cost * 1.5)

    elseif actual_status == infeasible
        # Deterministic contradiction: total available supply is strictly less
        # than total per-product demand, so no allocation can meet all products.
        # Set each ingredient's supply so the sum is well below total_needed.
        per_ingredient_cap = (total_needed * 0.5) / n_ingredients
        for i in 1:n_ingredients
            supply_limits[i] = per_ingredient_cap
        end
        # sum(supply_limits) = 0.5 * total_needed < total_needed (clear margin),
        # while sum_p sum_i x[i,p] must be >= total_needed -> infeasible.
        # Keep the budget generous so infeasibility is driven purely by supply.
        cost_budget = avg_cost * total_needed * 5.0
    end

    return MultiProductBlendingProblem(
        n_ingredients, n_products, n_attributes,
        costs, attributes, supply_limits, cost_budget,
        product_amounts, product_quality_lower, product_quality_upper,
    )
end

"""
    build_model(prob::MultiProductBlendingProblem)

Build a JuMP model for the multi-product blending problem. Deterministic — uses
only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::MultiProductBlendingProblem)
    model = Model()

    n = prob.n_ingredients
    P = prob.n_products

    # Variables: amount of ingredient i allocated to product p (n * P total)
    @variable(model, x[1:n, 1:P] >= 0)

    # Objective: minimize total ingredient cost across all products
    @objective(model, Min, sum(prob.costs[i] * sum(x[i, p] for p in 1:P) for i in 1:n))

    # Each product must reach its minimum production amount
    for p in 1:P
        @constraint(model, sum(x[i, p] for i in 1:n) >= prob.product_amounts[p])
    end

    # Shared supply limits (across all products) per ingredient
    for i in 1:n
        @constraint(model, sum(x[i, p] for p in 1:P) <= prob.supply_limits[i])
    end

    # Global cost budget
    @constraint(model, sum(prob.costs[i] * sum(x[i, p] for p in 1:P) for i in 1:n) <= prob.cost_budget)

    # Per-product quality bands on the weighted-average of each attribute
    for p in 1:P
        for j in 1:prob.n_attributes
            @constraint(model,
                sum(prob.attributes[i, j] * x[i, p] for i in 1:n) >=
                prob.product_quality_lower[p, j] * sum(x[i, p] for i in 1:n))
            @constraint(model,
                sum(prob.attributes[i, j] * x[i, p] for i in 1:n) <=
                prob.product_quality_upper[p, j] * sum(x[i, p] for i in 1:n))
        end
    end

    return model
end

# Register the variant
register_variant(
    :blending,
    :multi_product,
    MultiProductBlendingProblem,
    "Multi-product blending that allocates a shared ingredient pool across products at minimum cost",
)
