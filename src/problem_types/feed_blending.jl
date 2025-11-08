using JuMP
using Random
using Distributions
using Statistics
using StatsBase

"""
    FeedBlendingProblem <: ProblemGenerator

Generator for feed blending (diet) problems that find the least-cost mixture of ingredients while satisfying nutritional requirements.

# Fields
- `num_ingredients::Int`: Number of ingredients available for the blend
- `num_nutrients::Int`: Number of nutrients to consider in constraints
- `batch_size::Float64`: Required total batch size
- `costs::Vector{Float64}`: Cost per unit for each ingredient
- `nutrient_content::Matrix{Float64}`: Amount of each nutrient j in each ingredient i
- `nutrient_types::Vector{Int}`: Type classification for each nutrient (1-4)
- `min_requirements::Vector{Float64}`: Minimum nutrient requirements
- `max_limits::Vector{Float64}`: Maximum nutrient limits
- `availabilities::Vector{Float64}`: Availability limits for each ingredient
- `ratio_constraints::Vector{Tuple}`: List of ratio constraints (nutrient_idx, target_pct, type)
"""
struct FeedBlendingProblem <: ProblemGenerator
    num_ingredients::Int
    num_nutrients::Int
    batch_size::Float64
    costs::Vector{Float64}
    nutrient_content::Matrix{Float64}
    nutrient_types::Vector{Int}
    min_requirements::Vector{Float64}
    max_limits::Vector{Float64}
    availabilities::Vector{Float64}
    ratio_constraints::Vector{Tuple}
end

"""
    FeedBlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a feed blending problem instance.

# Arguments
- `target_variables`: Target number of variables (num_ingredients)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function FeedBlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = Random.MersenneTwister(seed)

    # Set num_ingredients = target_variables
    num_ingredients = max(3, target_variables)

    # Scale parameters based on problem size
    if target_variables <= 250
        num_nutrients = rand(rng, 4:8)
        batch_size = rand(rng, truncated(Normal(500.0, 200.0), 100.0, 2000.0))
    elseif target_variables <= 1000
        num_nutrients = rand(rng, 6:12)
        batch_size = rand(rng, truncated(Normal(2000.0, 800.0), 500.0, 10000.0))
    else
        num_nutrients = rand(rng, 8:20)
        batch_size = rand(rng, truncated(Normal(10000.0, 5000.0), 2000.0, 50000.0))
    end

    # Generate costs with realistic distributions
    if num_ingredients <= 250
        cost_mu = log(4.0)
        cost_sigma = 0.8
    elseif num_ingredients <= 1000
        cost_mu = log(2.5)
        cost_sigma = 0.6
    else
        cost_mu = log(1.8)
        cost_sigma = 0.4
    end
    costs = exp.(rand(rng, Normal(cost_mu, cost_sigma), num_ingredients))

    # Generate nutrient content matrix
    nutrient_content = zeros(num_nutrients, num_ingredients)
    nutrient_types = rand(rng, 1:4, num_nutrients)

    for j in 1:num_nutrients
        if nutrient_types[j] == 1
            # Type 1: Major nutrients
            for i in 1:num_ingredients
                nutrient_content[j, i] = max(0, rand(rng, Normal(20.0, 7.0)))
                if rand(rng) < 0.15
                    nutrient_content[j, i] *= rand(rng, Uniform(1.5, 2.5))
                elseif rand(rng) < 0.15
                    nutrient_content[j, i] *= rand(rng, Uniform(0.2, 0.6))
                end
            end
        elseif nutrient_types[j] == 2
            # Type 2: Minor nutrients
            for i in 1:num_ingredients
                if rand(rng) < 0.7
                    nutrient_content[j, i] = max(0, rand(rng, Normal(2.0, 1.0)))
                    if rand(rng) < 0.2
                        nutrient_content[j, i] *= rand(rng, Uniform(2.0, 5.0))
                    end
                end
            end
        elseif nutrient_types[j] == 3
            # Type 3: Trace nutrients
            for i in 1:num_ingredients
                if rand(rng) < 0.3
                    nutrient_content[j, i] = max(0, rand(rng, Normal(0.5, 0.3)))
                    if rand(rng) < 0.25
                        nutrient_content[j, i] *= rand(rng, Uniform(3.0, 10.0))
                    end
                end
            end
        else
            # Type 4: Anti-nutrients or upper-limited compounds
            for i in 1:num_ingredients
                if rand(rng) < 0.6
                    nutrient_content[j, i] = max(0, rand(rng, Normal(5.0, 3.0)))
                    if rand(rng) < 0.2
                        nutrient_content[j, i] *= rand(rng, Uniform(1.5, 3.0))
                    end
                end
            end
        end
    end

    # Ensure every nutrient exists in at least one ingredient
    for j in 1:num_nutrients
        if all(nutrient_content[j, :] .== 0)
            for _ in 1:max(1, ceil(Int, 0.2 * num_ingredients))
                i = rand(rng, 1:num_ingredients)
                nutrient_content[j, i] = max(0, rand(rng, Normal(2.0, 1.0)))
            end
        end
    end

    # Ensure every ingredient contains at least one nutrient
    for i in 1:num_ingredients
        if all(nutrient_content[:, i] .== 0)
            for _ in 1:max(1, ceil(Int, 0.2 * num_nutrients))
                j = rand(rng, 1:num_nutrients)
                nutrient_content[j, i] = max(0, rand(rng, Normal(2.0, 1.0)))
            end
        end
    end

    # Generate availabilities
    availabilities = fill(Inf, num_ingredients)
    availability_prob = if num_ingredients <= 250
        rand(rng, truncated(Normal(0.25, 0.1), 0.1, 0.4))
    elseif num_ingredients <= 1000
        rand(rng, truncated(Normal(0.3, 0.1), 0.15, 0.45))
    else
        rand(rng, truncated(Normal(0.35, 0.1), 0.2, 0.5))
    end

    for i in 1:num_ingredients
        if rand(rng) < availability_prob
            if num_ingredients <= 250
                availabilities[i] = rand(rng, truncated(Normal(0.4, 0.15), 0.1, 0.8)) * batch_size
            elseif num_ingredients <= 1000
                availabilities[i] = rand(rng, truncated(Normal(0.6, 0.2), 0.2, 1.2)) * batch_size
            else
                if rand(rng) < 0.3
                    availabilities[i] = rand(rng, truncated(Normal(0.2, 0.1), 0.05, 0.5)) * batch_size
                else
                    availabilities[i] = rand(rng, truncated(Normal(0.8, 0.3), 0.3, 2.0)) * batch_size
                end
            end
        end
    end

    # Helper functions for achievable bounds
    function achievable_max_avg(j::Int)
        pairs = [(nutrient_content[j, i], i) for i in 1:num_ingredients]
        sort!(pairs, by = x -> -x[1])
        remaining = batch_size
        total = 0.0
        for (a, i) in pairs
            if remaining <= 1e-12
                break
            end
            cap = isfinite(availabilities[i]) ? max(availabilities[i], 0.0) : remaining
            take = min(cap, remaining)
            total += a * take
            remaining -= take
        end
        return total / batch_size
    end

    function achievable_min_avg(j::Int)
        pairs = [(nutrient_content[j, i], i) for i in 1:num_ingredients]
        sort!(pairs, by = x -> x[1])
        remaining = batch_size
        total = 0.0
        for (a, i) in pairs
            if remaining <= 1e-12
                break
            end
            cap = isfinite(availabilities[i]) ? max(availabilities[i], 0.0) : remaining
            take = min(cap, remaining)
            total += a * take
            remaining -= take
        end
        return total / batch_size
    end

    # Build base recipe helper
    function build_base_recipe()
        # Ensure enough capacity
        if feasibility_status == feasible
            total_cap = 0.0
            for i in 1:num_ingredients
                total_cap += isfinite(availabilities[i]) ? min(availabilities[i], batch_size) : batch_size
            end
            if total_cap + 1e-8 < batch_size
                cheap_idx = argmin(costs)
                availabilities[cheap_idx] = batch_size
            end
        end

        α = fill(1.0, num_ingredients)
        w = rand(rng, Dirichlet(α))
        x0 = batch_size .* w
        for i in 1:num_ingredients
            if isfinite(availabilities[i])
                x0[i] = min(x0[i], availabilities[i])
            end
        end
        remaining = batch_size - sum(x0)
        if remaining > 1e-8
            order = sortperm(collect(1:num_ingredients), by=i -> costs[i] * (0.8 + 0.4 * rand(rng)))
            for i in order
                cap = isfinite(availabilities[i]) ? max(availabilities[i] - x0[i], 0.0) : remaining
                add = min(cap, remaining)
                x0[i] += add
                remaining -= add
                if remaining <= 1e-8
                    break
                end
            end
        end
        return x0
    end

    x0 = build_base_recipe()
    nutrient_totals = [sum(nutrient_content[j, i] * x0[i] for i in 1:num_ingredients) for j in 1:num_nutrients]
    nutrient_avgs = nutrient_totals ./ batch_size

    # Generate nutrient requirements based on feasibility status
    min_requirements = zeros(num_nutrients)
    max_limits = fill(Inf, num_nutrients)

    if feasibility_status == feasible
        # Generate feasible requirements within achievable intervals
        for j in 1:num_nutrients
            amin = achievable_min_avg(j)
            amax = achievable_max_avg(j)
            if nutrient_types[j] == 1
                if rand(rng) < 0.95
                    q = rand(rng, Uniform(0.40, 0.75))
                    min_requirements[j] = (amin + q * (amax - amin)) * batch_size
                end
                if rand(rng) < 0.25
                    q = rand(rng, Uniform(0.80, 0.95))
                    max_limits[j] = (amin + q * (amax - amin)) * batch_size
                end
            elseif nutrient_types[j] == 2
                if rand(rng) < 0.7
                    q = rand(rng, Uniform(0.25, 0.60))
                    min_requirements[j] = (amin + q * (amax - amin)) * batch_size
                end
                if rand(rng) < 0.15
                    q = rand(rng, Uniform(0.75, 0.95))
                    max_limits[j] = (amin + q * (amax - amin)) * batch_size
                end
            elseif nutrient_types[j] == 3
                if rand(rng) < 0.5
                    q = rand(rng, Uniform(0.10, 0.45))
                    min_requirements[j] = (amin + q * (amax - amin)) * batch_size
                end
                if rand(rng) < 0.1
                    q = rand(rng, Uniform(0.70, 0.95))
                    max_limits[j] = (amin + q * (amax - amin)) * batch_size
                end
            else
                if rand(rng) < 0.85
                    q = rand(rng, Uniform(0.20, 0.60))
                    max_limits[j] = (amin + q * (amax - amin)) * batch_size
                end
                if rand(rng) < 0.1
                    q = rand(rng, Uniform(0.05, 0.20))
                    min_requirements[j] = (amin + q * (amax - amin)) * batch_size
                end
            end
            if isfinite(max_limits[j]) && min_requirements[j] > 0 && min_requirements[j] > max_limits[j]
                max_limits[j] = max(min_requirements[j] * rand(rng, Uniform(1.02, 1.15)), (amin + 0.98 * (amax - amin)) * batch_size)
            end
        end
        # Ensure x0 satisfies all constraints
        for j in 1:num_nutrients
            if min_requirements[j] > 0 && min_requirements[j] > nutrient_totals[j]
                min_requirements[j] = nutrient_totals[j] * rand(rng, Uniform(0.85, 0.98))
            end
            if isfinite(max_limits[j]) && max_limits[j] < nutrient_totals[j]
                max_limits[j] = nutrient_totals[j] * rand(rng, Uniform(1.02, 1.25))
            end
            if isfinite(max_limits[j]) && min_requirements[j] > max_limits[j]
                max_limits[j] = max(min_requirements[j] * rand(rng, Uniform(1.02, 1.15)), nutrient_totals[j] * 1.01)
            end
        end
    elseif feasibility_status == unknown
        # Random generation without guarantees
        max_possible_nutrients = [maximum([nutrient_content[j, i] * batch_size for i in 1:num_ingredients]) for j in 1:num_nutrients]
        min_requirement_factor = rand(rng, truncated(Normal(0.4, 0.1), 0.2, 0.6))
        max_limit_factor = rand(rng, truncated(Normal(1.5, 0.2), 1.1, 2.0))

        for j in 1:num_nutrients
            if nutrient_types[j] == 1
                min_requirements[j] = rand(rng, Uniform(0.2, 0.6)) * max_possible_nutrients[j] * min_requirement_factor
                if rand(rng) < 0.3
                    max_limits[j] = min_requirements[j] * rand(rng, Uniform(1.2, 2.0)) * max_limit_factor
                end
            elseif nutrient_types[j] == 2
                min_requirements[j] = rand(rng, Uniform(0.1, 0.5)) * max_possible_nutrients[j] * min_requirement_factor
                if rand(rng) < 0.2
                    max_limits[j] = min_requirements[j] * rand(rng, Uniform(1.5, 3.0)) * max_limit_factor
                end
            elseif nutrient_types[j] == 3
                if rand(rng) < 0.7
                    min_requirements[j] = rand(rng, Uniform(0.05, 0.4)) * max_possible_nutrients[j] * min_requirement_factor
                end
                if rand(rng) < 0.1
                    max_limits[j] = min_requirements[j] > 0 ?
                                   min_requirements[j] * rand(rng, Uniform(2.0, 5.0)) * max_limit_factor :
                                   rand(rng, Uniform(0.1, 0.3)) * max_possible_nutrients[j] * max_limit_factor
                end
            else
                if rand(rng) < 0.8
                    max_limits[j] = rand(rng, Uniform(0.2, 0.7)) * max_possible_nutrients[j] * max_limit_factor
                end
                if rand(rng) < 0.1
                    min_requirements[j] = rand(rng, Uniform(0.05, 0.2)) *
                                         (max_limits[j] < Inf ? max_limits[j] : max_possible_nutrients[j]) * min_requirement_factor
                end
            end
        end
    end

    # Generate ratio constraints
    ratio_constraints = Tuple[]
    ratio_constraint_prob = if num_ingredients <= 250
        rand(rng, truncated(Normal(0.15, 0.05), 0.05, 0.25))
    elseif num_ingredients <= 1000
        rand(rng, truncated(Normal(0.2, 0.05), 0.1, 0.3))
    else
        rand(rng, truncated(Normal(0.25, 0.05), 0.15, 0.35))
    end

    if rand(rng) < ratio_constraint_prob
        num_ratio_constraints = rand(rng, 1:ceil(Int, 0.3 * num_nutrients))
        nutrient_indices = StatsBase.sample(rng, 1:num_nutrients, min(num_ratio_constraints, num_nutrients), replace=false)

        for j in nutrient_indices
            if any(nutrient_content[j, :] .> 0)
                is_min = rand(rng) < 0.7
                positive_values = filter(v -> v > 0, nutrient_content[j, :])
                if !isempty(positive_values)
                    max_percentage = maximum(nutrient_content[j, :])
                    min_percentage = minimum(positive_values)
                    if feasibility_status == feasible
                        amin = achievable_min_avg(j)
                        amax = achievable_max_avg(j)
                        if is_min
                            bias_range = nutrient_types[j] == 1 ? (0.4, 0.7) : nutrient_types[j] == 2 ? (0.3, 0.6) : (0.2, 0.5)
                            q = rand(rng, Uniform(bias_range...))
                            target_pct = amin + q * (amax - amin)
                            target_pct = min(target_pct, nutrient_avgs[j] * rand(rng, Uniform(0.92, 0.98)))
                            push!(ratio_constraints, (j, target_pct, "min"))
                        else
                            bias_range = nutrient_types[j] == 4 ? (0.25, 0.55) : (0.65, 0.9)
                            q = rand(rng, Uniform(bias_range...))
                            target_pct = amin + q * (amax - amin)
                            target_pct = max(target_pct, nutrient_avgs[j] * rand(rng, Uniform(1.02, 1.15)))
                            push!(ratio_constraints, (j, target_pct, "max"))
                        end
                    else
                        if is_min
                            target_pct = rand(rng, Uniform(0.2, 0.8)) * max_percentage
                            push!(ratio_constraints, (j, target_pct, "min"))
                        else
                            target_pct = rand(rng, Uniform(1.2, 1.8)) * min_percentage
                            push!(ratio_constraints, (j, target_pct, "max"))
                        end
                    end
                end
            end
        end
    end

    # Infeasibility injection
    if feasibility_status == infeasible
        r = rand(rng)
        if r < 0.35
            # Ratio MIN above any ingredient content
            candidates = [j for j in 1:num_nutrients if any(nutrient_content[j, :] .> 0)]
            if !isempty(candidates)
                j = rand(rng, candidates)
                target_pct = maximum(nutrient_content[j, :]) * rand(rng, Uniform(1.02, 1.25))
                push!(ratio_constraints, (j, target_pct, "min_forced_infeasible"))
            else
                j = rand(rng, 1:num_nutrients)
                req = nutrient_totals[j] * rand(rng, Uniform(1.5, 2.0)) + 1.0
                min_requirements[j] = req
            end
        elseif r < 0.70
            # MIN above achievable max
            majors = [j for j in 1:num_nutrients if nutrient_types[j] == 1]
            j = isempty(majors) ? rand(rng, 1:num_nutrients) : rand(rng, majors)
            max_avg = achievable_max_avg(j)
            target_pct = max_avg * rand(rng, Uniform(1.02, 1.15))
            push!(ratio_constraints, (j, target_pct, "min_above_achievable"))
        elseif r < 0.85
            # MAX below achievable min
            j = rand(rng, 1:num_nutrients)
            min_avg = achievable_min_avg(j)
            if min_avg > 1e-9
                target_pct = min_avg * rand(rng, Uniform(0.6, 0.95))
                push!(ratio_constraints, (j, target_pct, "max_below_achievable_min"))
            else
                majors = [j for j in 1:num_nutrients if nutrient_types[j] == 1]
                j2 = isempty(majors) ? rand(rng, 1:num_nutrients) : rand(rng, majors)
                max_avg = achievable_max_avg(j2)
                target_pct = max_avg * rand(rng, Uniform(1.02, 1.15))
                push!(ratio_constraints, (j2, target_pct, "min_above_achievable_fallback"))
            end
        else
            # Availability shortage
            total_cap = 0.0
            caps = zeros(num_ingredients)
            for i in 1:num_ingredients
                cap_i = isfinite(availabilities[i]) ? min(availabilities[i], batch_size) : rand(rng, Uniform(0.01, 0.3)) * batch_size
                caps[i] = cap_i
                total_cap += cap_i
            end
            if total_cap >= batch_size
                scale = rand(rng, Uniform(0.6, 0.95)) * batch_size / total_cap
                total_cap = 0.0
                for i in 1:num_ingredients
                    caps[i] *= scale
                    total_cap += caps[i]
                end
            end
            if total_cap >= batch_size
                idxs = shuffle(rng, collect(1:num_ingredients))
                for i in idxs
                    if caps[i] > 0
                        total_cap -= caps[i]
                        caps[i] = 0.0
                        break
                    end
                end
            end
            for i in 1:num_ingredients
                availabilities[i] = caps[i]
            end
        end
    end

    return FeedBlendingProblem(
        num_ingredients,
        num_nutrients,
        batch_size,
        costs,
        nutrient_content,
        nutrient_types,
        min_requirements,
        max_limits,
        availabilities,
        ratio_constraints
    )
end

"""
    build_model(prob::FeedBlendingProblem)

Build a JuMP model for the feed blending problem.

# Arguments
- `prob`: FeedBlendingProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::FeedBlendingProblem)
    model = Model()

    @variable(model, x[1:prob.num_ingredients] >= 0)

    @objective(model, Min, sum(prob.costs[i] * x[i] for i in 1:prob.num_ingredients))

    @constraint(model, sum(x[i] for i in 1:prob.num_ingredients) == prob.batch_size)

    for j in 1:prob.num_nutrients
        if prob.min_requirements[j] > 0
            @constraint(model, sum(prob.nutrient_content[j, i] * x[i] for i in 1:prob.num_ingredients) >= prob.min_requirements[j])
        end

        if prob.max_limits[j] < Inf
            @constraint(model, sum(prob.nutrient_content[j, i] * x[i] for i in 1:prob.num_ingredients) <= prob.max_limits[j])
        end
    end

    for i in 1:prob.num_ingredients
        if prob.availabilities[i] < Inf
            @constraint(model, x[i] <= prob.availabilities[i])
        end
    end

    for (j, target_pct, constraint_type) in prob.ratio_constraints
        if contains(constraint_type, "min")
            @constraint(model, sum((prob.nutrient_content[j, i] - target_pct) * x[i] for i in 1:prob.num_ingredients) >= 0)
        else
            @constraint(model, sum((prob.nutrient_content[j, i] - target_pct) * x[i] for i in 1:prob.num_ingredients) <= 0)
        end
    end

    return model
end

# Register the problem type
register_problem(
    :feed_blending,
    FeedBlendingProblem,
    "Feed blending (diet) problem that finds the least-cost mixture of ingredients while satisfying nutritional requirements"
)
