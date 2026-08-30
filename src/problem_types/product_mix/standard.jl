using JuMP
using Random
using Distributions
using StatsBase

"""
Planted operating plan: the nominal production quantities the instance was
designed around, the resource consumption they induce, and the capacity slack
left over. For the `feasible` profile this point satisfies every row of the
built model (`slack .>= 0`, floors below `plan`, ceilings above it).
"""
struct ProductMixPlanWitness
    plan::Vector{Float64}
    consumption::Vector{Float64}
    slack::Vector{Float64}
end

"""
Structured infeasibility certificate: the minimum-production floors of
`products` all consume resource `resource`, and together they require
`required_usage` units of it while only `availability` are on hand. Because
every `x_j >= lower_bounds[j]` and every usage coefficient is nonnegative, the
capacity row for `resource` cannot be satisfied — an over-committed resource
rather than a bound clash, so the refutation needs the aggregate capacity row
and survives into the LP relaxation.
"""
struct ResourceOvercommitCertificate
    resource::Int
    products::Vector{Int}
    required_usage::Float64
    availability::Float64
end

"""
    ProductMixProblem <: ProblemGenerator

Generator for product mix optimization problems.

# Overview
Models profit-maximizing production mix decisions. The decisions are continuous
production quantities for each product. The objective maximizes total product
profit. Resource-capacity constraints limit aggregate consumption across
products, and market constraints impose product-level minimum commitments and
sales ceilings.

# Planted operating plan
Capacities and market floors are *not* sampled independently — that makes the
two accumulate against each other and drives large instances to certain
infeasibility. Instead a nominal production plan is sampled first; resource
capacities are derived from what that plan consumes (plus per-resource
headroom) and market floors/ceilings are derived as fractions/multiples of the
plan's own output. Capacity and commitments are therefore mutually consistent
by construction at every scale.

Because usage coefficients are nonnegative, the pointwise-smallest candidate
point is `x = lower_bounds`, so the instance is feasible **iff**
`floor_utilization = max_i (sum_j usage[i,j] * lower_bounds[j]) / availabilities[i]`
is at most 1 (the constructor always keeps `lower_bounds .<= upper_bounds`).
The three profiles simply place that single scalar:
- `feasible`: floors stay below the planted plan and capacities above its
  consumption, so `floor_utilization < 1` with room to spare.
- `unknown`: floors are raised and capacities tightened toward a target
  utilization drawn as `1 ± U(0.04, 0.35)`, i.e. a coin flip that lands on
  either side of feasibility at any problem size.
- `infeasible`: the same perturbation aims at a target in `[1.15, 1.60]`,
  backed by a `ResourceOvercommitCertificate`.

# Fields
- `num_products::Int`: Number of products
- `num_resources::Int`: Number of resources
- `profits::Vector{Float64}`: Profit per unit of each product
- `usage_matrix::Matrix{Float64}`: Resource usage per unit (num_resources × num_products)
- `availabilities::Vector{Float64}`: Available amount of each resource
- `lower_bounds::Vector{Float64}`: Minimum production level for each product
- `upper_bounds::Vector{Float64}`: Maximum production level for each product
- `nominal_plan::Vector{Float64}`: Planted operating plan the data is built around
- `floor_utilization::Float64`: Tightest resource's floor-induced utilization
- `industry::Symbol`: Sampled industry regime
- `feasible_witness::Union{Nothing,ProductMixPlanWitness}`: set for `feasible`
- `infeasibility_certificate::Union{Nothing,ResourceOvercommitCertificate}`: set for `infeasible`
- `feasibility_status::FeasibilityStatus`: Requested profile
"""
struct ProductMixProblem <: ProblemGenerator
    num_products::Int
    num_resources::Int
    profits::Vector{Float64}
    usage_matrix::Matrix{Float64}
    availabilities::Vector{Float64}
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    nominal_plan::Vector{Float64}
    floor_utilization::Float64
    industry::Symbol
    feasible_witness::Union{Nothing,ProductMixPlanWitness}
    infeasibility_certificate::Union{Nothing,ResourceOvercommitCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _product_mix_floor_usage(usage_matrix, lower_bounds) -> Vector{Float64}

Resource consumption implied by the market floors alone (`usage_matrix *
lower_bounds`). This is the smallest consumption any admissible plan can have,
since every usage coefficient is nonnegative.
"""
function _product_mix_floor_usage(usage_matrix::Matrix{Float64},
                                  lower_bounds::Vector{Float64})
    num_resources, num_products = size(usage_matrix)
    required = zeros(num_resources)
    for j in 1:num_products
        lb = lower_bounds[j]
        lb > 0.0 || continue
        for i in 1:num_resources
            required[i] += usage_matrix[i, j] * lb
        end
    end
    return required
end

"""
    _product_mix_utilization(usage_matrix, lower_bounds, availabilities) -> Float64

The tightest resource's floor-induced utilization. The instance is feasible
iff this is `<= 1` (given `lower_bounds .<= upper_bounds`).
"""
function _product_mix_utilization(usage_matrix::Matrix{Float64},
                                  lower_bounds::Vector{Float64},
                                  availabilities::Vector{Float64})
    required = _product_mix_floor_usage(usage_matrix, lower_bounds)
    return maximum(required[i] / max(availabilities[i], eps())
                   for i in 1:length(availabilities))
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
    rng = MersenneTwister(seed)

    # For product mix, variables = num_products
    num_products = max(2, min(10000, target_variables))

    # Scale parameters based on problem size
    if target_variables <= 250
        # Small operations
        num_resources = rand(rng, DiscreteUniform(3, 8))
        sparsity = rand(rng, Beta(2, 6))
        profit_min = rand(rng, LogNormal(log(15), 0.4))
        profit_max = rand(rng, LogNormal(log(120), 0.3))
        resource_usage_min = rand(rng, LogNormal(log(1.0), 0.3))
        resource_usage_max = rand(rng, LogNormal(log(5), 0.3))
        market_constraint_prob = rand(rng, Beta(4, 6))
        correlation_strength = rand(rng, Beta(4, 3))
        volume_center = rand(rng, LogNormal(log(140), 0.4))
    elseif target_variables <= 1000
        # Medium operations
        resource_range = 5:15
        beta_sample = rand(rng, Beta(2, 3))
        num_resources = resource_range[max(1, min(length(resource_range), round(Int, beta_sample * length(resource_range)) + 1))]
        sparsity = rand(rng, Beta(3, 4))
        profit_min = rand(rng, LogNormal(log(8), 0.5))
        profit_max = rand(rng, LogNormal(log(75), 0.4))
        resource_usage_min = rand(rng, LogNormal(log(0.6), 0.4))
        resource_usage_max = rand(rng, LogNormal(log(4.5), 0.4))
        market_constraint_prob = rand(rng, Beta(5, 5))
        correlation_strength = rand(rng, Beta(6, 4))
        volume_center = rand(rng, LogNormal(log(90), 0.45))
    else
        # Large operations
        log_mean = log(18)
        log_std = 0.4
        sample_val = rand(rng, LogNormal(log_mean, log_std))
        num_resources = max(8, min(30, round(Int, sample_val)))
        sparsity = rand(rng, Beta(2, 3))
        profit_min = rand(rng, LogNormal(log(3), 0.6))
        profit_max = rand(rng, LogNormal(log(45), 0.5))
        resource_usage_min = rand(rng, LogNormal(log(0.3), 0.5))
        resource_usage_max = rand(rng, LogNormal(log(4), 0.5))
        market_constraint_prob = rand(rng, Beta(6, 4))
        correlation_strength = rand(rng, Beta(8, 3))
        volume_center = rand(rng, LogNormal(log(50), 0.5))
    end

    # Randomly select industry type
    industry_types = [:manufacturing, :food_processing, :electronics, :furniture, :chemical, :automotive]
    industry_weights = if target_variables <= 250
        [0.25, 0.35, 0.15, 0.20, 0.03, 0.02]
    elseif target_variables <= 1000
        [0.40, 0.15, 0.25, 0.10, 0.08, 0.02]
    else
        [0.35, 0.08, 0.20, 0.05, 0.17, 0.15]
    end
    industry_type = sample(rng, industry_types, Weights(industry_weights))

    # Apply industry-specific adjustments
    if industry_type == :food_processing
        profit_min *= 0.6
        profit_max *= 0.7
        resource_usage_max *= 0.8
        market_constraint_prob *= 1.5
        volume_center *= 2.5
    elseif industry_type == :electronics
        profit_min *= 1.5
        profit_max *= 2.2
        resource_usage_max *= 1.4
        sparsity *= 1.3
        volume_center *= 1.2
    elseif industry_type == :furniture
        profit_min *= 0.8
        profit_max *= 0.9
        resource_usage_min *= 1.2
        market_constraint_prob *= 1.2
    elseif industry_type == :chemical
        profit_min *= 1.1
        profit_max *= 1.3
        resource_usage_min *= 1.8
        resource_usage_max *= 2.0
        correlation_strength *= 1.2
        volume_center *= 1.5
    elseif industry_type == :automotive
        profit_min *= 3.0
        profit_max *= 4.0
        resource_usage_min *= 2.0
        resource_usage_max *= 2.5
        sparsity *= 0.8
        market_constraint_prob *= 0.8
        volume_center *= 0.4
    end

    # Generate quality factors
    quality_factors = rand(rng, Beta(2, 2), num_products)

    # Generate profits
    base_profits = rand(rng, LogNormal(log((profit_min + profit_max) / 2), 0.3), num_products)
    base_profits = clamp.(base_profits, profit_min, profit_max)
    quality_component = quality_factors .* (profit_max - profit_min) * 0.5
    profits = base_profits + correlation_strength * quality_component

    # Generate usage matrix
    usage_matrix = zeros(num_resources, num_products)

    for i in 1:num_resources
        base_usage = rand(rng, LogNormal(log((resource_usage_min + resource_usage_max) / 2), 0.4))
        base_usage = clamp(base_usage, resource_usage_min, resource_usage_max)

        for j in 1:num_products
            if rand(rng) < sparsity
                usage_matrix[i, j] = 0.0
                continue
            end

            random_component = rand(rng, Gamma(2, resource_usage_max / 6))
            random_component = min(random_component, resource_usage_max / 2)

            quality_multiplier = 0.5 + correlation_strength * quality_factors[j]
            usage = base_usage * quality_multiplier + random_component * (1 - correlation_strength)
            usage_matrix[i, j] = max(0.0, usage)
        end
    end

    # Ensure each product uses at least one resource (keeps the LP bounded)
    for j in 1:num_products
        if all(usage_matrix[:, j] .== 0)
            resource_idx = rand(rng, 1:num_resources)
            usage_value = rand(rng, LogNormal(log((resource_usage_min + resource_usage_max) / 2), 0.3))
            usage_matrix[resource_idx, j] = clamp(usage_value, resource_usage_min, resource_usage_max)
        end
    end

    # Ensure each resource is used by at least one product
    for i in 1:num_resources
        if all(usage_matrix[i, :] .== 0)
            product_idx = rand(rng, 1:num_products)
            usage_value = rand(rng, LogNormal(log((resource_usage_min + resource_usage_max) / 2), 0.3))
            usage_matrix[i, product_idx] = clamp(usage_value, resource_usage_min, resource_usage_max)
        end
    end

    # --- Planted operating plan -------------------------------------------
    # A nominal production quantity for every product. Everything downstream
    # (capacities, floors, ceilings) is derived from this plan, so the data is
    # mutually consistent no matter how many products there are.
    nominal_plan = clamp.(rand(rng, LogNormal(log(volume_center), 0.55), num_products),
                          0.1 * volume_center, 10.0 * volume_center)

    consumption = usage_matrix * nominal_plan

    # Resource capacities = what the plan consumes plus per-resource headroom.
    # A handful of resources are deliberately tight so the LP has binding rows.
    headroom = clamp.(rand(rng, LogNormal(log(0.18), 0.8), num_resources), 0.02, 2.0)
    availabilities = consumption .* (1.0 .+ headroom)

    # --- Market floors and ceilings, both anchored on the plan -------------
    floor_prob = clamp(0.25 + 0.7 * market_constraint_prob, 0.25, 0.95)
    cap_prob = clamp(0.5 * market_constraint_prob, 0.05, 0.8)

    lower_bounds = zeros(num_products)
    upper_bounds = fill(Inf, num_products)
    for j in 1:num_products
        if rand(rng) < floor_prob
            # Committed minimum production: a strict fraction of the plan.
            lower_bounds[j] = (0.2 + 0.7 * rand(rng, Beta(2, 2))) * nominal_plan[j]
        end
        if rand(rng) < cap_prob
            # Market saturation ceiling: always above the plan.
            upper_bounds[j] = (1.05 + 1.4 * rand(rng, Beta(2, 2))) * nominal_plan[j]
        end
    end

    # At least one genuine commitment, so the utilization scalar below is
    # well defined and the perturbation has something to act on.
    if all(iszero, lower_bounds)
        j = rand(rng, 1:num_products)
        lower_bounds[j] = (0.2 + 0.7 * rand(rng, Beta(2, 2))) * nominal_plan[j]
        if isfinite(upper_bounds[j]) && upper_bounds[j] < 1.05 * lower_bounds[j]
            upper_bounds[j] = 1.05 * lower_bounds[j]
        end
    end

    # --- Feasibility profile ----------------------------------------------
    feasible_witness = nothing
    infeasibility_certificate = nothing

    if feasibility_status == feasible
        # Nothing to do: floors <= 0.9 * plan, ceilings >= 1.05 * plan and
        # capacities >= (1 + headroom) * consumption, so the planted plan is a
        # feasible point with strictly positive slack on every capacity row.
        feasible_witness = ProductMixPlanWitness(copy(nominal_plan), copy(consumption),
                                                 availabilities .- consumption)
    else
        # Push the instance toward a target floor utilization. The target is
        # the only thing that decides feasibility, so it stays a genuine coin
        # flip for `unknown` at every problem size.
        target = if feasibility_status == infeasible
            1.15 + 0.45 * rand(rng)
        else
            margin = 0.04 + 0.31 * rand(rng)
            rand(rng) < 0.5 ? 1.0 - margin : 1.0 + margin
        end

        current = _product_mix_utilization(usage_matrix, lower_bounds, availabilities)
        gap = target / current

        # Split the adjustment between raising commitments and tightening
        # capacity (a plant disruption), so neither side is pushed to an
        # unrealistic extreme; the capacity cut is capped and the remainder
        # lands on the floors.
        theta = 0.35 + 0.3 * rand(rng)
        capacity_scale = clamp(gap^(theta - 1.0), 0.35, 3.0)
        floor_scale = gap * capacity_scale

        availabilities .*= capacity_scale
        lower_bounds .*= floor_scale

        # Keep every ceiling strictly above its floor: the contradiction must
        # come from the aggregate capacity row, never from a bound clash.
        for j in 1:num_products
            if isfinite(upper_bounds[j]) && upper_bounds[j] < 1.05 * lower_bounds[j]
                upper_bounds[j] = 1.05 * lower_bounds[j]
            end
        end

        if feasibility_status == infeasible
            required = _product_mix_floor_usage(usage_matrix, lower_bounds)
            critical = argmax([required[i] / max(availabilities[i], eps())
                               for i in 1:num_resources])
            products = [j for j in 1:num_products
                        if lower_bounds[j] > 0.0 && usage_matrix[critical, j] > 0.0]
            infeasibility_certificate = ResourceOvercommitCertificate(
                critical, products, required[critical], availabilities[critical])
        end
    end

    floor_utilization = _product_mix_utilization(usage_matrix, lower_bounds, availabilities)

    return ProductMixProblem(num_products, num_resources, profits, usage_matrix,
                             availabilities, lower_bounds, upper_bounds,
                             nominal_plan, floor_utilization, industry_type,
                             feasible_witness, infeasibility_certificate,
                             feasibility_status)
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

    # Resource constraints (only the products that actually consume the resource)
    for i in 1:prob.num_resources
        @constraint(model, sum(prob.usage_matrix[i, j] * x[j]
                               for j in 1:prob.num_products
                               if prob.usage_matrix[i, j] > 0) <= prob.availabilities[i])
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

# Register the variant
register_variant(
    :product_mix,
    :standard,
    ProductMixProblem,
    "Product mix optimization problem that maximizes profit by determining optimal production quantities subject to resource constraints",
)
