using JuMP
using Random
using Distributions

"""
    EquipmentBatchBlendingProblem <: ProblemGenerator

Generator for equipment-constrained batch blending problems.

# Overview
Models blending a product across several discrete production batches on a single
mixer. The decisions are the amount of each ingredient assigned to each batch,
`x[i, b]`. The objective minimizes total ingredient cost. Each batch is limited by
the mixer capacity, each ingredient is limited by its total supply across batches,
total production must meet a minimum blend amount, and every individual batch must
satisfy per-batch quality bounds on a set of blended attributes. Per-batch quality
targets are varied across batches to reduce symmetry and degeneracy.

# Fields
- `n_ingredients::Int`: Number of available ingredients
- `n_batches::Int`: Number of production batches (mixer runs)
- `n_attributes::Int`: Number of blended quality attributes
- `costs::Vector{Int}`: Unit cost of each ingredient
- `attributes::Matrix{Float64}`: Attribute value of each ingredient (n_ingredients × n_attributes)
- `lower_bounds::Matrix{Float64}`: Per-batch lower bound on each attribute (n_batches × n_attributes)
- `upper_bounds::Matrix{Float64}`: Per-batch upper bound on each attribute (n_batches × n_attributes)
- `supply_limits::Vector{Float64}`: Total supply available for each ingredient across all batches
- `min_blend_amount::Float64`: Minimum total amount of product to produce
- `mixer_capacity::Float64`: Maximum total amount that can be mixed in a single batch
"""
struct EquipmentBatchBlendingProblem <: ProblemGenerator
    n_ingredients::Int
    n_batches::Int
    n_attributes::Int
    costs::Vector{Int}
    attributes::Matrix{Float64}
    lower_bounds::Matrix{Float64}
    upper_bounds::Matrix{Float64}
    supply_limits::Vector{Float64}
    min_blend_amount::Float64
    mixer_capacity::Float64
end

"""
    EquipmentBatchBlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an equipment-constrained batch blending problem instance.

Variables: `x[i, b]` for each ingredient i and batch b.
Total = n_ingredients * n_batches. The number of batches is chosen first and the
number of ingredients is sized as round(target / n_batches) to hit the target.

# Arguments
- `target_variables`: Target number of variables (n_ingredients × n_batches)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function EquipmentBatchBlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # Var count formula: vars = n_ingredients * n_batches.
    # Choose n_batches first, then set n_ingredients = round(target / n_batches).
    n_batches = max(2, rand(3:6))
    n_ingredients = max(3, round(Int, target_variables / n_batches))
    n_attributes = rand(2:min(8, max(2, n_ingredients ÷ 3)))

    # --- Base ingredient data ---
    cost_range = (10, 100)
    attribute_range = (0.1, 0.9)
    min_cost, max_cost = cost_range
    min_attr, max_attr = attribute_range

    costs = rand(min_cost:max_cost, n_ingredients)
    attributes = rand(min_attr:0.01:max_attr, n_ingredients, n_attributes)

    min_blend_amount = Float64(rand(100:20000))

    # --- Base (average) quality bounds per attribute ---
    base_lower = zeros(n_attributes)
    base_upper = zeros(n_attributes)
    for j in 1:n_attributes
        avg_attr = sum(attributes[:, j]) / n_ingredients
        base_lower[j] = avg_attr * rand(Uniform(0.6, 0.85))
        base_upper[j] = avg_attr * rand(Uniform(1.15, 1.4))
    end

    # --- Per-batch quality bounds (vary targets across batches to reduce symmetry) ---
    # Each batch perturbs the base bounds, but always brackets the average attribute
    # value so an equal-ingredient mix remains feasible per batch.
    lower_bounds = zeros(n_batches, n_attributes)
    upper_bounds = zeros(n_batches, n_attributes)
    for b in 1:n_batches
        for j in 1:n_attributes
            avg_attr = sum(attributes[:, j]) / n_ingredients
            lb = base_lower[j] * rand(Uniform(0.92, 1.0))
            ub = base_upper[j] * rand(Uniform(1.0, 1.08))
            # Keep the average strictly inside [lb, ub] so a uniform mix is feasible.
            lb = min(lb, avg_attr * 0.95)
            ub = max(ub, avg_attr * 1.05)
            lower_bounds[b, j] = lb
            upper_bounds[b, j] = ub
        end
    end

    # --- Equipment (mixer) capacity and supply ---
    # Mixer capacity is sized so a single batch holds a meaningful fraction of demand.
    mixer_capacity = (min_blend_amount / n_batches) * rand(Uniform(1.1, 1.6))

    # Per-ingredient supply across all batches.
    supply_limits = [min_blend_amount * rand(Uniform(0.4, 1.2)) for _ in 1:n_ingredients]

    # --- Feasibility handling ---
    actual_status = feasibility_status
    if feasibility_status == unknown
        actual_status = rand() < 0.7 ? feasible : infeasible
    end

    if actual_status == feasible
        # Total mixer capacity comfortably exceeds demand.
        total_capacity = n_batches * mixer_capacity
        if total_capacity < min_blend_amount * 1.3
            mixer_capacity = (min_blend_amount * 1.3) / n_batches
        end
        # Total supply comfortably exceeds demand (any subset can carry the load).
        if sum(supply_limits) < min_blend_amount * 1.5
            scale = (min_blend_amount * 1.5) / sum(supply_limits)
            supply_limits .*= scale
        end
        # Ensure no single supply is so tiny it blocks the uniform mix; give every
        # ingredient enough room to contribute a uniform share if needed.
        min_share = min_blend_amount / n_ingredients
        for i in 1:n_ingredients
            supply_limits[i] = max(supply_limits[i], min_share * 1.5)
        end
    elseif actual_status == infeasible
        # Force total mixer capacity strictly below demand with a clear margin:
        # n_batches * mixer_capacity < min_blend_amount, so the minimum-production
        # constraint can never be met regardless of supply or quality.
        mixer_capacity = (min_blend_amount * 0.8) / n_batches
    end

    return EquipmentBatchBlendingProblem(
        n_ingredients, n_batches, n_attributes, costs, attributes,
        lower_bounds, upper_bounds, supply_limits, min_blend_amount, mixer_capacity,
    )
end

"""
    build_model(prob::EquipmentBatchBlendingProblem)

Build a JuMP model for the equipment-constrained batch blending problem.
Deterministic — uses only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::EquipmentBatchBlendingProblem)
    model = Model()

    n = prob.n_ingredients
    B = prob.n_batches

    # Variables: amount of ingredient i in batch b. Total vars = n_ingredients * n_batches.
    @variable(model, x[1:n, 1:B] >= 0)

    # Objective: minimize total ingredient cost across all batches.
    @objective(model, Min, sum(prob.costs[i] * sum(x[i, b] for b in 1:B) for i in 1:n))

    # Total production must meet the minimum blend amount.
    @constraint(model, sum(x[i, b] for i in 1:n, b in 1:B) >= prob.min_blend_amount)

    # Mixer capacity limit per batch.
    for b in 1:B
        @constraint(model, sum(x[i, b] for i in 1:n) <= prob.mixer_capacity)
    end

    # Supply limit per ingredient across all batches.
    for i in 1:n
        @constraint(model, sum(x[i, b] for b in 1:B) <= prob.supply_limits[i])
    end

    # Per-batch quality bounds (homogeneous in the batch total amount).
    for b in 1:B
        for j in 1:prob.n_attributes
            @constraint(model,
                sum(prob.attributes[i, j] * x[i, b] for i in 1:n) >=
                prob.lower_bounds[b, j] * sum(x[i, b] for i in 1:n))
            @constraint(model,
                sum(prob.attributes[i, j] * x[i, b] for i in 1:n) <=
                prob.upper_bounds[b, j] * sum(x[i, b] for i in 1:n))
        end
    end

    return model
end

# Register the variant
register_variant(
    :blending,
    :equipment_batches,
    EquipmentBatchBlendingProblem,
    "Batch blending limited by mixer capacity, ingredient supply, and per-batch quality bounds",
)
