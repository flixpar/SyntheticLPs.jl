using JuMP
using Random
using Distributions
using Statistics

"""
    generate_feed_blending_problem(params::Dict=Dict(); seed::Int=0)

Generate a feed blending (diet) problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:num_ingredients`: Number of ingredients available for the blend (default: 10)
  - `:num_nutrients`: Number of nutrients to consider in constraints (default: 8)
  - `:batch_size`: Required total batch size (default: 1000.0)
  - `:min_requirement_factor`: Factor to control nutrient minimum requirements (default: 0.4)
  - `:max_limit_factor`: Factor to control nutrient maximum limits (default: 1.5)
  - `:availability_prob`: Probability of an ingredient having an availability constraint (default: 0.3)
  - `:ratio_constraint_prob`: Probability of adding ratio constraints (default: 0.2)
  - `:solution_status`: Desired feasibility: `:feasible` (default), `:infeasible`, or `:all`
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_feed_blending_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    rng = Random.MersenneTwister(seed)
    
    # Extract parameters with defaults
    num_ingredients = get(params, :num_ingredients, 10)
    num_nutrients = get(params, :num_nutrients, 8)
    batch_size = get(params, :batch_size, 1000.0)
    min_requirement_factor = get(params, :min_requirement_factor, 0.4)
    max_limit_factor = get(params, :max_limit_factor, 1.5)
    availability_prob = get(params, :availability_prob, 0.3)
    ratio_constraint_prob = get(params, :ratio_constraint_prob, 0.2)
    solution_status = get(params, :solution_status, :feasible)
    if !(solution_status in (:feasible, :infeasible, :all))
        error("solution_status must be :feasible, :infeasible, or :all")
    end
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :num_ingredients => num_ingredients,
        :num_nutrients => num_nutrients,
        :batch_size => batch_size,
        :min_requirement_factor => min_requirement_factor,
        :max_limit_factor => max_limit_factor,
        :availability_prob => availability_prob,
        :ratio_constraint_prob => ratio_constraint_prob,
        :solution_status => solution_status
    )
    
    # ---- Generate parameters ----
    
    # 1. Ingredient costs: lognormal distribution to represent market variability
    # Scale costs based on problem size to reflect different market contexts
    if num_ingredients <= 250
        # Small scale: Local/regional ingredients, higher variability
        cost_mu = log(4.0)   # median ≈ $4 per unit (higher local costs)
        cost_sigma = 0.8     # higher variability for local markets
    elseif num_ingredients <= 1000
        # Medium scale: Commercial sourcing, moderate variability
        cost_mu = log(2.5)   # median ≈ $2.5 per unit (bulk pricing)
        cost_sigma = 0.6     # moderate variability
    else
        # Large scale: Industrial sourcing, commodity pricing
        cost_mu = log(1.8)   # median ≈ $1.8 per unit (commodity prices)
        cost_sigma = 0.4     # lower variability for commodity markets
    end
    
    costs = exp.(rand(rng, Normal(cost_mu, cost_sigma), num_ingredients))
    
    # 2. Nutrient content matrix: amount of each nutrient j in each ingredient i
    # Values often represent percentages or concentrations and vary by nutrient type
    
    # Create a matrix of nutrient contents with some structure and sparsity
    nutrient_content = zeros(num_nutrients, num_ingredients)
    
    # Define different nutrient types with different scales and distributions
    nutrient_types = rand(rng, 1:4, num_nutrients)
    
    for j in 1:num_nutrients
        if nutrient_types[j] == 1
            # Type 1: Major nutrients (e.g., protein, energy) - most ingredients contain these
            # Values might be percentages like 10-30%
            for i in 1:num_ingredients
                nutrient_content[j, i] = max(0, rand(rng, Normal(20.0, 7.0)))
                
                # Some ingredients might be especially rich or poor in this nutrient
                if rand(rng) < 0.15
                    nutrient_content[j, i] *= rand(rng, Uniform(1.5, 2.5))
                elseif rand(rng) < 0.15
                    nutrient_content[j, i] *= rand(rng, Uniform(0.2, 0.6))
                end
            end
            
        elseif nutrient_types[j] == 2
            # Type 2: Minor nutrients (e.g., specific minerals) - moderate sparsity
            # Some ingredients might not contain these at all
            for i in 1:num_ingredients
                if rand(rng) < 0.7  # 70% chance of containing this nutrient
                    nutrient_content[j, i] = max(0, rand(rng, Normal(2.0, 1.0)))
                    
                    # Some ingredients might be especially rich in this nutrient
                    if rand(rng) < 0.2
                        nutrient_content[j, i] *= rand(rng, Uniform(2.0, 5.0))
                    end
                end
            end
            
        elseif nutrient_types[j] == 3
            # Type 3: Trace nutrients (e.g., vitamins) - higher sparsity
            # Only some ingredients contain significant amounts
            for i in 1:num_ingredients
                if rand(rng) < 0.3  # 30% chance of containing this nutrient
                    # Small values, possibly measured in mg/kg or similar
                    nutrient_content[j, i] = max(0, rand(rng, Normal(0.5, 0.3)))
                    
                    # Some ingredients might be especially rich sources
                    if rand(rng) < 0.25
                        nutrient_content[j, i] *= rand(rng, Uniform(3.0, 10.0))
                    end
                end
            end
            
        else  # type 4
            # Type 4: Anti-nutrients or upper-limited compounds (e.g., fiber, toxins)
            # These often need maximum constraints
            for i in 1:num_ingredients
                if rand(rng) < 0.6  # 60% chance
                    nutrient_content[j, i] = max(0, rand(rng, Normal(5.0, 3.0)))
                    
                    # Some ingredients might be especially high in these
                    if rand(rng) < 0.2
                        nutrient_content[j, i] *= rand(rng, Uniform(1.5, 3.0))
                    end
                end
            end
        end
    end
    
    # Make sure every nutrient exists in at least one ingredient and every ingredient contains at least one nutrient
    for j in 1:num_nutrients
        if all(nutrient_content[j, :] .== 0)
            # If no ingredient contains this nutrient, add it to some random ingredients
            for _ in 1:max(1, ceil(Int, 0.2 * num_ingredients))
                i = rand(rng, 1:num_ingredients)
                nutrient_content[j, i] = max(0, rand(rng, Normal(2.0, 1.0)))
            end
        end
    end
    
    for i in 1:num_ingredients
        if all(nutrient_content[:, i] .== 0)
            # If this ingredient contains no nutrients, add some random nutrients
            for _ in 1:max(1, ceil(Int, 0.2 * num_nutrients))
                j = rand(rng, 1:num_nutrients)
                nutrient_content[j, i] = max(0, rand(rng, Normal(2.0, 1.0)))
            end
        end
    end
    
    # 3. Nutrient requirements
    # Determine achievable ranges based on the generated nutrient content
    
    # First, find minimum and maximum possible for each nutrient if we used 100% of a single ingredient
    max_possible_nutrients = zeros(num_nutrients)
    for j in 1:num_nutrients
        # Maximum amount of nutrient j possible in batch_size
        max_per_ingredient = [nutrient_content[j, i] * batch_size for i in 1:num_ingredients]
        max_possible_nutrients[j] = maximum(max_per_ingredient)
    end
    
    # Now generate minimum requirements and maximum limits
    min_requirements = zeros(num_nutrients)
    max_limits = fill(Inf, num_nutrients)
    
    for j in 1:num_nutrients
        # Based on nutrient type, decide whether to have min, max, or both constraints
        if nutrient_types[j] == 1  # Major nutrients - typically have minimum requirements
            # Set minimum requirement between 20-60% of maximum possible
            min_requirements[j] = rand(rng, Uniform(0.2, 0.6)) * max_possible_nutrients[j] * min_requirement_factor
            
            # Some may also have maximum limits (80-120% of minimum)
            if rand(rng) < 0.3
                max_limits[j] = min_requirements[j] * rand(rng, Uniform(1.2, 2.0)) * max_limit_factor
            end
            
        elseif nutrient_types[j] == 2  # Minor nutrients - often have minimum requirements
            # Set minimum requirement between 10-50% of maximum possible
            min_requirements[j] = rand(rng, Uniform(0.1, 0.5)) * max_possible_nutrients[j] * min_requirement_factor
            
            # Rarely have maximum limits
            if rand(rng) < 0.2
                max_limits[j] = min_requirements[j] * rand(rng, Uniform(1.5, 3.0)) * max_limit_factor
            end
            
        elseif nutrient_types[j] == 3  # Trace nutrients - some have minimum requirements
            if rand(rng) < 0.7  # 70% chance of minimum requirement
                # Set minimum requirement between 5-40% of maximum possible
                min_requirements[j] = rand(rng, Uniform(0.05, 0.4)) * max_possible_nutrients[j] * min_requirement_factor
            end
            
            # Very rarely have maximum limits
            if rand(rng) < 0.1
                max_limits[j] = min_requirements[j] > 0 ? 
                               min_requirements[j] * rand(rng, Uniform(2.0, 5.0)) * max_limit_factor : 
                               rand(rng, Uniform(0.1, 0.3)) * max_possible_nutrients[j] * max_limit_factor
            end
            
        else  # Type 4 - anti-nutrients or upper-limited compounds
            # Usually have maximum limits but no minimum requirements
            if rand(rng) < 0.8  # 80% chance of maximum limit
                # Set maximum limit between 20-70% of maximum possible
                max_limits[j] = rand(rng, Uniform(0.2, 0.7)) * max_possible_nutrients[j] * max_limit_factor
            end
            
            # Very rarely have minimum requirements
            if rand(rng) < 0.1
                min_requirements[j] = rand(rng, Uniform(0.05, 0.2)) * 
                                     (max_limits[j] < Inf ? max_limits[j] : max_possible_nutrients[j]) * min_requirement_factor
            end
        end
    end
    
    # 4. Ingredient availabilities (optional)
    # For some ingredients, there might be availability constraints
    availabilities = fill(Inf, num_ingredients)
    
    # Add availability constraints for some ingredients (based on availability_prob)
    # Scale availability patterns based on problem size
    for i in 1:num_ingredients
        if rand(rng) < availability_prob
            if num_ingredients <= 250
                # Small scale: Tight local supply constraints
                availabilities[i] = rand(rng, truncated(Normal(0.4, 0.15), 0.1, 0.8)) * batch_size
            elseif num_ingredients <= 1000
                # Medium scale: Moderate supply constraints
                availabilities[i] = rand(rng, truncated(Normal(0.6, 0.2), 0.2, 1.2)) * batch_size
            else
                # Large scale: Diverse supply sources, some very constrained, some abundant
                if rand(rng) < 0.3
                    # 30% chance of very constrained specialty ingredients
                    availabilities[i] = rand(rng, truncated(Normal(0.2, 0.1), 0.05, 0.5)) * batch_size
                else
                    # 70% chance of more abundant commodity ingredients
                    availabilities[i] = rand(rng, truncated(Normal(0.8, 0.3), 0.3, 2.0)) * batch_size
                end
            end
        end
    end
    
    # Store generated data in params
    actual_params[:costs] = costs
    actual_params[:nutrient_content] = nutrient_content
    actual_params[:nutrient_types] = nutrient_types
    actual_params[:min_requirements] = min_requirements
    actual_params[:max_limits] = max_limits
    actual_params[:availabilities] = availabilities
    actual_params[:max_possible_nutrients] = max_possible_nutrients
    
    # Helper to compute a feasible base recipe x0 that respects availability and batch
    function build_base_recipe()
        # Ensure there is enough available capacity to fill the batch (for :feasible)
        if solution_status == :feasible
            total_cap = 0.0
            for i in 1:num_ingredients
                total_cap += isfinite(availabilities[i]) ? min(availabilities[i], batch_size) : batch_size
            end
            if total_cap + 1e-8 < batch_size
                # Increase availability of a cheap ingredient to cover deficit
                cheap_idx = argmin(costs)
                availabilities[cheap_idx] = batch_size
            end
        end

        # Start with a random mixture (Dirichlet) to promote diversity
        α = fill(1.0, num_ingredients)
        w = rand(rng, Dirichlet(α))
        x0 = batch_size .* w
        # Clip by availability
        for i in 1:num_ingredients
            if isfinite(availabilities[i])
                x0[i] = min(x0[i], availabilities[i])
            end
        end
        # Top up to batch by allocating to remaining capacity, biased to cheaper ingredients
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
    # Note: x0 is used to promote diversity, but feasible constraints are not anchored to x0.

    # Nutrient totals/averages at x0 (for reference only)
    nutrient_totals = [sum(nutrient_content[j, i] * x0[i] for i in 1:num_ingredients) for j in 1:num_nutrients]
    nutrient_avgs = nutrient_totals ./ batch_size

    # Helper: compute achievable max average for nutrient j under availabilities
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

    # Helper: compute achievable min average for nutrient j under availabilities
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

    # For :feasible, resample min/max within achievable intervals to preserve realism without anchoring to x0
    if solution_status == :feasible
        new_min = zeros(num_nutrients)
        new_max = fill(Inf, num_nutrients)
        for j in 1:num_nutrients
            amin = achievable_min_avg(j)
            amax = achievable_max_avg(j)
            # If amin == amax, skip adding both bounds; retain possibility of only one consistent bound
            if nutrient_types[j] == 1
                # Majors: typically have a minimum; sometimes a max
                if rand(rng) < 0.95
                    q = rand(rng, Uniform(0.40, 0.75))
                    new_min[j] = (amin + q * (amax - amin)) * batch_size
                end
                if rand(rng) < 0.25
                    q = rand(rng, Uniform(0.80, 0.95))
                    new_max[j] = (amin + q * (amax - amin)) * batch_size
                end
            elseif nutrient_types[j] == 2
                # Minors: often have a min, rare max
                if rand(rng) < 0.7
                    q = rand(rng, Uniform(0.25, 0.60))
                    new_min[j] = (amin + q * (amax - amin)) * batch_size
                end
                if rand(rng) < 0.15
                    q = rand(rng, Uniform(0.75, 0.95))
                    new_max[j] = (amin + q * (amax - amin)) * batch_size
                end
            elseif nutrient_types[j] == 3
                # Trace: sometimes min, very rare max
                if rand(rng) < 0.5
                    q = rand(rng, Uniform(0.10, 0.45))
                    new_min[j] = (amin + q * (amax - amin)) * batch_size
                end
                if rand(rng) < 0.1
                    q = rand(rng, Uniform(0.70, 0.95))
                    new_max[j] = (amin + q * (amax - amin)) * batch_size
                end
            else
                # Type 4: often only a max
                if rand(rng) < 0.85
                    q = rand(rng, Uniform(0.20, 0.60))
                    new_max[j] = (amin + q * (amax - amin)) * batch_size
                end
                if rand(rng) < 0.1
                    q = rand(rng, Uniform(0.05, 0.20))
                    new_min[j] = (amin + q * (amax - amin)) * batch_size
                end
            end
            # Ensure consistency if both present
            if isfinite(new_max[j]) && new_min[j] > 0 && new_min[j] > new_max[j]
                # Nudge max above min slightly within achievable range
                new_max[j] = max(new_min[j] * rand(rng, Uniform(1.02, 1.15)), (amin + 0.98 * (amax - amin)) * batch_size)
            end
        end
        # Ensure the sampled bounds are satisfied by x0 to guarantee feasibility,
        # while keeping the samples anchored to achievable intervals.
        for j in 1:num_nutrients
            if new_min[j] > 0 && new_min[j] > nutrient_totals[j]
                new_min[j] = nutrient_totals[j] * rand(rng, Uniform(0.85, 0.98))
            end
            if isfinite(new_max[j]) && new_max[j] < nutrient_totals[j]
                new_max[j] = nutrient_totals[j] * rand(rng, Uniform(1.02, 1.25))
            end
            if isfinite(new_max[j]) && new_min[j] > new_max[j]
                new_max[j] = max(new_min[j] * rand(rng, Uniform(1.02, 1.15)), nutrient_totals[j] * 1.01)
            end
        end
        min_requirements = new_min
        max_limits = new_max
    end

    # 5. Create JuMP model
    model = Model()
    
    # Add decision variables: amount of each ingredient to use
    @variable(model, x[1:num_ingredients] >= 0)
    
    # Set objective: minimize total cost
    @objective(model, Min, sum(costs[i] * x[i] for i in 1:num_ingredients))
    
    # Add batch size constraint
    @constraint(model, sum(x[i] for i in 1:num_ingredients) == batch_size)
    
    # Add nutritional requirement constraints (possibly adjusted)
    for j in 1:num_nutrients
        if min_requirements[j] > 0
            @constraint(model, sum(nutrient_content[j, i] * x[i] for i in 1:num_ingredients) >= min_requirements[j])
        end
        
        if max_limits[j] < Inf
            @constraint(model, sum(nutrient_content[j, i] * x[i] for i in 1:num_ingredients) <= max_limits[j])
        end
    end
    
    # Add availability constraints
    for i in 1:num_ingredients
        if availabilities[i] < Inf
            @constraint(model, x[i] <= availabilities[i])
        end
    end
    
    ratio_constraints = []
    # Add ratio constraints (linearized) with feasibility-aware targets
    if rand(rng) < ratio_constraint_prob
        # Determine number of ratio constraints to add (up to 30% of nutrients)
        num_ratio_constraints = rand(rng, 1:ceil(Int, 0.3 * num_nutrients))
        nutrient_indices = sample(rng, 1:num_nutrients, min(num_ratio_constraints, num_nutrients), replace=false)

        for j in nutrient_indices
            if any(nutrient_content[j, :] .> 0)
                is_min = rand(rng) < 0.7  # 70% chance for minimum ratio
                positive_values = filter(v -> v > 0, nutrient_content[j, :])
                if !isempty(positive_values)
                    max_percentage = maximum(nutrient_content[j, :])
                    min_percentage = minimum(positive_values)
                    if solution_status == :feasible
                        # Choose target inside achievable interval [amin, amax] to ensure feasibility,
                        # with type-aware bias.
                        amin = achievable_min_avg(j)
                        amax = achievable_max_avg(j)
                        if is_min
                            bias_range = nutrient_types[j] == 1 ? (0.4, 0.7) : nutrient_types[j] == 2 ? (0.3, 0.6) : (0.2, 0.5)
                            q = rand(rng, Uniform(bias_range...))
                            target_pct = amin + q * (amax - amin)
                            # Ensure x0 satisfies the ratio min
                            target_pct = min(target_pct, nutrient_avgs[j] * rand(rng, Uniform(0.92, 0.98)))
                            @constraint(model, sum((nutrient_content[j, i] - target_pct) * x[i] for i in 1:num_ingredients) >= 0)
                            push!(ratio_constraints, (j, target_pct, "min"))
                        else
                            bias_range = nutrient_types[j] == 4 ? (0.25, 0.55) : (0.65, 0.9)
                            q = rand(rng, Uniform(bias_range...))
                            target_pct = amin + q * (amax - amin)
                            # Ensure x0 satisfies the ratio max
                            target_pct = max(target_pct, nutrient_avgs[j] * rand(rng, Uniform(1.02, 1.15)))
                            @constraint(model, sum((nutrient_content[j, i] - target_pct) * x[i] for i in 1:num_ingredients) <= 0)
                            push!(ratio_constraints, (j, target_pct, "max"))
                        end
                    else
                        # Original stochastic targets for :all or base before infeasibilization
                        if is_min
                            target_pct = rand(rng, Uniform(0.2, 0.8)) * max_percentage
                            @constraint(model, sum((nutrient_content[j, i] - target_pct) * x[i] for i in 1:num_ingredients) >= 0)
                            push!(ratio_constraints, (j, target_pct, "min"))
                        else
                            target_pct = rand(rng, Uniform(1.2, 1.8)) * min_percentage
                            @constraint(model, sum((nutrient_content[j, i] - target_pct) * x[i] for i in 1:num_ingredients) <= 0)
                            push!(ratio_constraints, (j, target_pct, "max"))
                        end
                    end
                end
            end
        end
    end

    actual_params[:ratio_constraints] = ratio_constraints

    # If infeasible requested, inject a guaranteed contradiction while preserving realism
    # Patterns (tuned split):
    #  - 35%: impossible ratio MIN above per-ingredient maximum content
    #  - 35%: ratio MIN above achievable max average under availability
    #  - 15%: ratio MAX below achievable min average under availability
    #  - 15%: availability shortage (sum of x_i caps < batch_size)
    if solution_status == :infeasible
        r = rand(rng)
        if r < 0.35
            # Force a ratio MIN above any ingredient content for some nutrient with nonzero presence
            candidates = [j for j in 1:num_nutrients if any(nutrient_content[j, :] .> 0)]
            if !isempty(candidates)
                j = rand(rng, candidates)
                target_pct = maximum(nutrient_content[j, :]) * rand(rng, Uniform(1.02, 1.25))
                @constraint(model, sum((nutrient_content[j, i] - target_pct) * x[i] for i in 1:num_ingredients) >= 0)
                push!(ratio_constraints, (j, target_pct, "min_forced_infeasible"))
            else
                # Fallback: enforce contradictory min/max on a random nutrient
                j = rand(rng, 1:num_nutrients)
                req = nutrient_totals[j] * rand(rng, Uniform(1.5, 2.0)) + 1.0
                @constraint(model, sum(nutrient_content[j, i] * x[i] for i in 1:num_ingredients) >= req)
            end
        elseif r < 0.70
            # Choose a nutrient (prefer major nutrient types) and set MIN above achievable max
            majors = [j for j in 1:num_nutrients if nutrient_types[j] == 1]
            j = isempty(majors) ? rand(rng, 1:num_nutrients) : rand(rng, majors)
            max_avg = achievable_max_avg(j)
            target_pct = max_avg * rand(rng, Uniform(1.02, 1.15))
            @constraint(model, sum((nutrient_content[j, i] - target_pct) * x[i] for i in 1:num_ingredients) >= 0)
            push!(ratio_constraints, (j, target_pct, "min_above_achievable"))
        elseif r < 0.85
            # Ratio MAX below achievable min average (generalized)
            j = rand(rng, 1:num_nutrients)
            min_avg = achievable_min_avg(j)
            if min_avg > 1e-9
                target_pct = min_avg * rand(rng, Uniform(0.6, 0.95))
                @constraint(model, sum((nutrient_content[j, i] - target_pct) * x[i] for i in 1:num_ingredients) <= 0)
                push!(ratio_constraints, (j, target_pct, "max_below_achievable_min"))
            else
                # Fallback to achievable-max violation pattern
                majors = [j for j in 1:num_nutrients if nutrient_types[j] == 1]
                j2 = isempty(majors) ? rand(rng, 1:num_nutrients) : rand(rng, majors)
                max_avg = achievable_max_avg(j2)
                target_pct = max_avg * rand(rng, Uniform(1.02, 1.15))
                @constraint(model, sum((nutrient_content[j2, i] - target_pct) * x[i] for i in 1:num_ingredients) >= 0)
                push!(ratio_constraints, (j2, target_pct, "min_above_achievable_fallback"))
            end
        else
            # Availability shortage: make sum of caps strictly less than batch_size
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
                # zero out a random ingredient to push under
                idxs = shuffle(rng, collect(1:num_ingredients))
                for i in idxs
                    if caps[i] > 0
                        total_cap -= caps[i]
                        caps[i] = 0.0
                        break
                    end
                end
            end
            # Add additional tighter capacity constraints to enforce shortage
            for i in 1:num_ingredients
                @constraint(model, x[i] <= caps[i])
            end
        end
        actual_params[:ratio_constraints] = ratio_constraints
    end
    
    return model, actual_params
end

"""
    sample_feed_blending_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a feed blending problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_feed_blending_parameters(target_variables::Int; seed::Int=0)
    rng = Random.MersenneTwister(seed)
    
    params = Dict{Symbol, Any}()
    
    # Start with target_variables = num_ingredients
    params[:num_ingredients] = max(3, target_variables)
    
    # Scale parameters based on problem size to reflect realistic feed blending scenarios
    if target_variables <= 250
        # Small scale: Small farm or local feed mill
        params[:num_nutrients] = rand(rng, 4:8)  # Basic nutritional requirements
        params[:batch_size] = rand(rng, truncated(Normal(500.0, 200.0), 100.0, 2000.0))
        params[:min_requirement_factor] = rand(rng, truncated(Normal(0.4, 0.1), 0.2, 0.6))
        params[:max_limit_factor] = rand(rng, truncated(Normal(1.4, 0.2), 1.1, 1.8))
        params[:availability_prob] = rand(rng, truncated(Normal(0.25, 0.1), 0.1, 0.4))
        params[:ratio_constraint_prob] = rand(rng, truncated(Normal(0.15, 0.05), 0.05, 0.25))
        
    elseif target_variables <= 1000
        # Medium scale: Commercial feed mill
        params[:num_nutrients] = rand(rng, 6:12)  # More complex nutritional profiles
        params[:batch_size] = rand(rng, truncated(Normal(2000.0, 800.0), 500.0, 10000.0))
        params[:min_requirement_factor] = rand(rng, truncated(Normal(0.35, 0.1), 0.2, 0.5))
        params[:max_limit_factor] = rand(rng, truncated(Normal(1.5, 0.2), 1.2, 2.0))
        params[:availability_prob] = rand(rng, truncated(Normal(0.3, 0.1), 0.15, 0.45))
        params[:ratio_constraint_prob] = rand(rng, truncated(Normal(0.2, 0.05), 0.1, 0.3))
        
    else
        # Large scale: Industrial feed production
        params[:num_nutrients] = rand(rng, 8:20)  # Complex nutritional requirements
        params[:batch_size] = rand(rng, truncated(Normal(10000.0, 5000.0), 2000.0, 50000.0))
        params[:min_requirement_factor] = rand(rng, truncated(Normal(0.3, 0.1), 0.15, 0.45))
        params[:max_limit_factor] = rand(rng, truncated(Normal(1.6, 0.25), 1.3, 2.2))
        params[:availability_prob] = rand(rng, truncated(Normal(0.35, 0.1), 0.2, 0.5))
        params[:ratio_constraint_prob] = rand(rng, truncated(Normal(0.25, 0.05), 0.15, 0.35))
    end
    
    # Iteratively adjust if needed (though for feed blending, it's direct)
    for iteration in 1:5
        current_vars = calculate_feed_blending_variable_count(params)
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust num_ingredients directly since it's the only parameter affecting variable count
        if current_vars < target_variables
            params[:num_ingredients] = params[:num_ingredients] + 1
        elseif current_vars > target_variables
            params[:num_ingredients] = max(3, params[:num_ingredients] - 1)
        end
    end
    
    return params
end

"""
    sample_feed_blending_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a feed blending problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_feed_blending_parameters(size::Symbol=:medium; seed::Int=0)
    rng = Random.MersenneTwister(seed)
    
    # Map size categories to realistic target variable counts
    target_map = Dict(
        :small => rand(rng, 50:250),      # Small farm or local feed mill
        :medium => rand(rng, 250:1000),   # Commercial feed mill
        :large => rand(rng, 1000:10000)   # Industrial feed production
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_feed_blending_parameters(target_map[size]; seed=seed)
end

"""
    calculate_feed_blending_variable_count(params::Dict)

Calculate the number of variables in a feed blending problem.

# Arguments
- `params`: Dictionary of problem parameters

# Returns
- Number of variables in the problem
"""
function calculate_feed_blending_variable_count(params::Dict)
    # Extract the number of ingredients parameter
    num_ingredients = get(params, :num_ingredients, 10)
    
    # The problem has one variable for each ingredient: x[1:num_ingredients]
    return num_ingredients
end

# Register the problem type
register_problem(
    :feed_blending,
    generate_feed_blending_problem,
    sample_feed_blending_parameters,
    "Feed blending (diet) problem that finds the least-cost mixture of ingredients while satisfying nutritional requirements"
)
