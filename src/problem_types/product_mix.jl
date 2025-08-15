using JuMP
using Random
using Distributions
using LinearAlgebra
using StatsBase

"""
    generate_product_mix_problem(params::Dict=Dict(); seed::Int=0)

Generate a product mix optimization problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:num_products`: Number of products to produce (default: 10)
  - `:num_resources`: Number of limited resources (default: 5)
  - `:sparsity`: Probability of a product not using a resource (default: 0.2)
  - `:profit_min`: Minimum profit contribution per product (default: 5.0)
  - `:profit_max`: Maximum profit contribution per product (default: 100.0)
  - `:resource_usage_min`: Minimum resource usage per product (default: 0.5)
  - `:resource_usage_max`: Maximum resource usage per product (default: 5.0)
  - `:market_constraint_prob`: Probability of adding market constraints for each product (default: 0.3)
  - `:correlation_strength`: Strength of correlation between profit and resource usage (default: 0.7)
  - `:industry_type`: Type of industry for more realistic parameters (default: "manufacturing")
  - `:solution_status`: Desired feasibility status (:feasible, :infeasible, or :all). If :feasible, guarantees feasibility; if :infeasible, guarantees infeasibility. (default: :feasible)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_product_mix_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    num_products = get(params, :num_products, 10)
    num_resources = get(params, :num_resources, 5)
    sparsity = get(params, :sparsity, 0.2)
    profit_min = get(params, :profit_min, 5.0)
    profit_max = get(params, :profit_max, 100.0)
    resource_usage_min = get(params, :resource_usage_min, 0.5)
    resource_usage_max = get(params, :resource_usage_max, 5.0)
    market_constraint_prob = get(params, :market_constraint_prob, 0.3)
    correlation_strength = get(params, :correlation_strength, 0.7)
    industry_type = get(params, :industry_type, "manufacturing")
    solution_status = get(params, :solution_status, :feasible)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :num_products => num_products,
        :num_resources => num_resources,
        :sparsity => sparsity,
        :profit_min => profit_min,
        :profit_max => profit_max,
        :resource_usage_min => resource_usage_min,
        :resource_usage_max => resource_usage_max,
        :market_constraint_prob => market_constraint_prob,
        :correlation_strength => correlation_strength,
        :industry_type => industry_type,
        :solution_status => solution_status
    )
    
    # Apply industry-specific adjustments for realistic scenarios
    if industry_type == "manufacturing"
        # Standard manufacturing - balanced profits and resource usage
        # No changes needed - baseline
    elseif industry_type == "food_processing"
        # Lower margins, perishable goods, consistent resource usage
        profit_min *= 0.6
        profit_max *= 0.7
        resource_usage_max *= 0.8
        market_constraint_prob *= 1.5  # More demand constraints due to perishability
    elseif industry_type == "electronics"
        # Higher margins, rapid obsolescence, variable resource usage
        profit_min *= 1.5
        profit_max *= 2.2
        resource_usage_max *= 1.4
        sparsity *= 1.3  # More specialized components
    elseif industry_type == "furniture"
        # Moderate margins, bulky products, seasonal demand
        profit_min *= 0.8
        profit_max *= 0.9
        resource_usage_min *= 1.2
        market_constraint_prob *= 1.2  # Seasonal/style constraints
    elseif industry_type == "chemical"
        # Process industry - high resource requirements, economies of scale
        profit_min *= 1.1
        profit_max *= 1.3
        resource_usage_min *= 1.8
        resource_usage_max *= 2.0
        correlation_strength *= 1.2  # Strong economies of scale
    elseif industry_type == "automotive"
        # High value products, complex supply chains, high resource requirements
        profit_min *= 3.0
        profit_max *= 4.0
        resource_usage_min *= 2.0
        resource_usage_max *= 2.5
        sparsity *= 0.8  # More integrated supply chains
        market_constraint_prob *= 0.8  # More predictable demand
    end
    
    # Generate correlated profit contributions and resource usage
    # Higher quality products use more resources but generate more profit
    
    # Generate a base "quality" factor for each product using Beta distribution
    # This creates more realistic quality distributions (some low, some high, most medium)
    quality_factors = rand(Beta(2, 2), num_products)  # Bell-shaped distribution around 0.5
    
    # Generate profit contributions correlated with quality using LogNormal
    base_profits = rand(LogNormal(log((profit_min + profit_max) / 2), 0.3), num_products)
    # Clamp to reasonable range
    base_profits = clamp.(base_profits, profit_min, profit_max)
    
    # Add quality correlation component
    quality_component = quality_factors .* (profit_max - profit_min) * 0.5
    profits = base_profits + correlation_strength * quality_component
    
    # Generate resource usage matrix with correlation to quality
    usage_matrix = zeros(num_resources, num_products)
    
    for i in 1:num_resources
        # Resource-specific base usage using LogNormal distribution
        base_usage = rand(LogNormal(log((resource_usage_min + resource_usage_max) / 2), 0.4))
        base_usage = clamp(base_usage, resource_usage_min, resource_usage_max)
        
        for j in 1:num_products
            # Apply sparsity by setting some coefficients to zero
            if rand() < sparsity
                usage_matrix[i, j] = 0.0
                continue
            end
            
            # Product-specific random component using Gamma distribution
            # Gamma provides right-skewed distribution typical of resource usage
            random_component = rand(Gamma(2, resource_usage_max / 6))
            random_component = min(random_component, resource_usage_max / 2)
            
            # Combined usage with quality correlation
            quality_multiplier = 0.5 + correlation_strength * quality_factors[j]
            usage = base_usage * quality_multiplier + random_component * (1 - correlation_strength)
            usage_matrix[i, j] = max(0.0, usage)
        end
    end
    
    # Ensure each product uses at least one resource and each resource is used by at least one product
    # This is to avoid trivial or degenerate problems
    
    # Ensure each product uses at least one resource
    for j in 1:num_products
        if all(usage_matrix[:, j] .== 0)
            # Assign a random resource to this product using LogNormal distribution
            resource_idx = rand(1:num_resources)
            usage_value = rand(LogNormal(log((resource_usage_min + resource_usage_max) / 2), 0.3))
            usage_matrix[resource_idx, j] = clamp(usage_value, resource_usage_min, resource_usage_max)
        end
    end
    
    # Ensure each resource is used by at least one product
    for i in 1:num_resources
        if all(usage_matrix[i, :] .== 0)
            # Assign this resource to a random product using LogNormal distribution
            product_idx = rand(1:num_products)
            usage_value = rand(LogNormal(log((resource_usage_min + resource_usage_max) / 2), 0.3))
            usage_matrix[i, product_idx] = clamp(usage_value, resource_usage_min, resource_usage_max)
        end
    end
    
    # Generate resource availabilities
    # Make sure they allow for a feasible solution
    avg_usage_per_resource = mean(usage_matrix, dims=2) .* num_products / 2
    
    # Add realistic variability to resource availabilities using LogNormal
    # This creates more realistic scenarios where some resources are more/less abundant
    variability = rand(LogNormal(log(1.0), 0.3), num_resources)  # Mean ~1.0, realistic spread
    variability = clamp.(variability, 0.5, 2.0)  # Keep within reasonable bounds
    
    # Some resources might be more constrained in realistic scenarios
    constraint_factors = rand(Beta(3, 2), num_resources)  # Slightly skewed toward higher values
    constraint_factors = 0.6 .+ 0.6 * constraint_factors  # Range: 0.6 to 1.2
    
    availabilities = avg_usage_per_resource[:] .* variability .* constraint_factors
    
    # Generate market constraints (min/max production for some products)
    
    # Estimate maximum possible production of each product if all resources were allocated to it
    max_possible = Float64[]
    for j in 1:num_products
        # Calculate how much of each resource would be needed for one unit
        resource_limits = Float64[]
        for i in 1:num_resources
            if usage_matrix[i, j] > 0
                push!(resource_limits, availabilities[i] / usage_matrix[i, j])
            else
                push!(resource_limits, Inf)
            end
        end
        # The limiting factor is the minimum ratio
        push!(max_possible, minimum(resource_limits))
    end
    
    # Initialize market bounds
    lower_bounds = zeros(num_products)
    upper_bounds = fill(Inf, num_products)
    
    # Add lower and upper bounds with the given probability
    for j in 1:num_products
        # Add a lower bound with probability/2
        if rand() < market_constraint_prob/2
            # Lower bound using Beta distribution for more realistic minimum demands
            lower_factor = rand(Beta(2, 6))  # Skewed toward lower values (5-25% typically)
            lower_bounds[j] = lower_factor * 0.35 * max_possible[j]
        end
        
        # Add an upper bound with probability/2
        if rand() < market_constraint_prob/2
            # Upper bound using Beta distribution for realistic market caps
            upper_factor = rand(Beta(4, 2))  # Skewed toward higher values (60-95% typically)
            upper_bounds[j] = (0.4 + upper_factor * 0.55) * max_possible[j]
        end
    end
    
    # Store generated data in params
    actual_params[:profits] = profits
    actual_params[:usage_matrix] = usage_matrix
    actual_params[:availabilities] = availabilities
    actual_params[:lower_bounds] = lower_bounds
    actual_params[:upper_bounds] = upper_bounds
    actual_params[:max_possible] = max_possible

    # Adjust constraints to meet desired solution status while preserving realism
    if solution_status == :feasible
        # Ensure per-product bounds are consistent
        for j in 1:num_products
            if isfinite(upper_bounds[j]) && lower_bounds[j] > upper_bounds[j]
                lower_bounds[j] = max(0.0, 0.98 * upper_bounds[j])
            end
        end

        # Scale lower bounds uniformly if their aggregate resource demand exceeds capacity
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
                # Apply a small safety margin to avoid numerical edge cases
                lower_bounds .*= lb_scale * 0.98
            end
        end

        # Final guard for LB vs UB after scaling
        for j in 1:num_products
            if isfinite(upper_bounds[j]) && lower_bounds[j] > upper_bounds[j]
                lower_bounds[j] = max(0.0, 0.98 * upper_bounds[j])
            end
        end

        # Update params with possibly adjusted bounds
        actual_params[:lower_bounds] = lower_bounds
        actual_params[:upper_bounds] = upper_bounds

    elseif solution_status == :infeasible
        # Construct a realistic infeasibility by creating a capacity shortfall relative to minimum commitments
        # Ensure at least some positive lower bounds exist
        if all(lower_bounds .== 0.0)
            # Assign lower bounds to a subset of products with nonzero resource usage on a random critical resource
            critical_res = argmax([sum(usage_matrix[i, :]) for i in 1:num_resources])
            candidates = [j for j in 1:num_products if usage_matrix[critical_res, j] > 0.0]
            if isempty(candidates)
                candidates = collect(1:num_products)
            end
            num_assign = max(1, round(Int, 0.2 * length(candidates)))
            selected = sample(candidates, min(num_assign, length(candidates)); replace=false)
            for j in selected
                # Set LB to a realistic fraction of standalone capacity
                lb_factor = 0.15 + 0.25 * rand()  # 15%-40%
                lower_bounds[j] = max(lower_bounds[j], lb_factor * max_possible[j])
            end
        end

        # Recompute required usage at lower bounds
        required = [sum(usage_matrix[i, j] * lower_bounds[j] for j in 1:num_products) for i in 1:num_resources]

        # Pick the most stressed resource (with positive requirement)
        ratios = [required[i] > 0 ? required[i] / max(availabilities[i], eps()) : 0.0 for i in 1:num_resources]
        critical_i = argmax(ratios)
        if required[critical_i] == 0.0
            # If still zero, pick a resource with highest total usage and enforce
            critical_i = argmax([sum(usage_matrix[i, :]) for i in 1:num_resources])
            # If still degenerate, just pick 1
            if sum(usage_matrix[critical_i, :]) == 0.0
                critical_i = 1
            end
            # Ensure at least one product contributes on the critical resource
            if all(usage_matrix[critical_i, :] .== 0.0)
                # Assign a small usage to a random product to avoid degenerate case
                pj = rand(1:num_products)
                usage_matrix[critical_i, pj] = max(usage_matrix[critical_i, pj], resource_usage_min)
                required[critical_i] = usage_matrix[critical_i, pj] * max(lower_bounds[pj], resource_usage_min)
            end
        end

        # Reduce availability on the critical resource below the required lower-bound demand
        shortage_margin = 0.10 + 0.25 * rand()  # 10%-35% shortfall
        new_avail = max(0.0, required[critical_i] * (1.0 - shortage_margin))
        availabilities[critical_i] = new_avail

        # Optionally reduce one additional resource to mimic broader supply shock
        if num_resources >= 2 && rand() < 0.3
            other_i = critical_i % num_resources + 1
            req_other = required[other_i]
            if req_other > 0.0
                availabilities[other_i] = max(0.0, req_other * (1.0 - (0.05 + 0.15 * rand())))
            end
        end

        actual_params[:availabilities] = availabilities
        actual_params[:lower_bounds] = lower_bounds
    end
    
    # Create model
    model = Model()
    
    # Add decision variables for each product
    @variable(model, x[1:num_products] >= 0)
    
    # Set objective function: maximize profit
    @objective(model, Max, sum(profits[j] * x[j] for j in 1:num_products))
    
    # Add resource constraints
    for i in 1:num_resources
        @constraint(model, sum(usage_matrix[i, j] * x[j] for j in 1:num_products) <= availabilities[i])
    end
    
    # Add market constraints
    for j in 1:num_products
        # Add lower bound if it's greater than zero
        if lower_bounds[j] > 0
            @constraint(model, x[j] >= lower_bounds[j])
        end
        
        # Add upper bound if it's less than infinity
        if upper_bounds[j] < Inf
            @constraint(model, x[j] <= upper_bounds[j])
        end
    end
    
    return model, actual_params
end

"""
    sample_product_mix_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a product mix optimization problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_product_mix_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Start with defaults - target_variables = num_products
    params[:num_products] = max(2, min(10000, target_variables))
    
    # Scale number of resources based on problem size with realistic distributions
    if target_variables <= 250
        # Small operations: 3-8 resources (labor, raw materials, equipment, space)
        # More likely to have fewer resources due to simpler operations
        params[:num_resources] = rand(DiscreteUniform(3, 8))
    elseif target_variables <= 1000
        # Medium operations: 5-15 resources (multiple production lines, specialized equipment)
        # Beta distribution skewed toward middle-lower range
        resource_range = 5:15
        beta_sample = rand(Beta(2, 3))  # Skewed toward lower values
        params[:num_resources] = resource_range[max(1, min(length(resource_range), 
                                                  round(Int, beta_sample * length(resource_range)) + 1))]
    else
        # Large operations: 8-30 resources (complex supply chains, multiple facilities)
        # LogNormal distribution for realistic resource complexity scaling
        log_mean = log(18)  # Geometric mean around 18 resources
        log_std = 0.4
        sample_val = rand(LogNormal(log_mean, log_std))
        params[:num_resources] = max(8, min(30, round(Int, sample_val)))
    end
    
    # Iteratively adjust if needed (though for product mix, it's direct)
    for iteration in 1:5
        current_vars = calculate_product_mix_variable_count(params)
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust num_products directly since it's the only parameter affecting variable count
        if current_vars < target_variables
            params[:num_products] = min(10000, params[:num_products] + 1)
        elseif current_vars > target_variables
            params[:num_products] = max(2, params[:num_products] - 1)
        end
    end
    
    # Size-dependent parameter scaling based on realistic business scales
    if target_variables <= 250
        # Small-scale operations (artisanal, small manufacturing, startup, local businesses)
        # - Higher per-unit profits due to specialization/premium positioning
        # - More manual processes, less resource specialization
        # - More volatile demand, higher market constraint probability
        params[:sparsity] = rand(Beta(2, 6))  # 0.05-0.5, mean ~0.25 (less specialization)
        params[:profit_min] = rand(LogNormal(log(15), 0.4))  # $8-30, geometric mean ~$15
        params[:profit_max] = rand(LogNormal(log(120), 0.3))  # $75-200, geometric mean ~$120
        params[:resource_usage_min] = rand(LogNormal(log(1.0), 0.3))  # 0.5-2.0, mean ~1.0
        params[:resource_usage_max] = rand(LogNormal(log(5), 0.3))  # 3-8, mean ~5
        params[:market_constraint_prob] = rand(Beta(4, 6))  # 0.2-0.7, mean ~0.4 (volatile demand)
        params[:correlation_strength] = rand(Beta(4, 3))  # 0.3-0.9, mean ~0.57 (less systematic)
    elseif target_variables <= 1000
        # Medium-scale operations (established manufacturing, regional chains, mid-size companies)
        # - Balanced efficiency and specialization
        # - Moderate economies of scale
        # - More predictable demand patterns
        params[:sparsity] = rand(Beta(3, 4))  # 0.15-0.75, mean ~0.43 (moderate specialization)
        params[:profit_min] = rand(LogNormal(log(8), 0.5))  # $3-20, geometric mean ~$8
        params[:profit_max] = rand(LogNormal(log(75), 0.4))  # $35-160, geometric mean ~$75
        params[:resource_usage_min] = rand(LogNormal(log(0.6), 0.4))  # 0.25-1.5, mean ~0.6
        params[:resource_usage_max] = rand(LogNormal(log(4.5), 0.4))  # 2-10, mean ~4.5
        params[:market_constraint_prob] = rand(Beta(5, 5))  # 0.25-0.75, mean ~0.5 (balanced)
        params[:correlation_strength] = rand(Beta(6, 4))  # 0.35-0.9, mean ~0.6 (systematic)
    else
        # Large-scale operations (multinational corporations, high-volume manufacturing)
        # - Lower per-unit profits due to economies of scale and competition
        # - High specialization, many products don't use many resources
        # - More stable demand, sophisticated supply chain management
        params[:sparsity] = rand(Beta(2, 3))  # 0.2-0.8, mean ~0.4 (high specialization)
        params[:profit_min] = rand(LogNormal(log(3), 0.6))  # $0.8-12, geometric mean ~$3
        params[:profit_max] = rand(LogNormal(log(45), 0.5))  # $18-120, geometric mean ~$45
        params[:resource_usage_min] = rand(LogNormal(log(0.3), 0.5))  # 0.1-0.8, mean ~0.3
        params[:resource_usage_max] = rand(LogNormal(log(4), 0.5))  # 1.5-10, mean ~4
        params[:market_constraint_prob] = rand(Beta(6, 4))  # 0.35-0.85, mean ~0.6 (managed demand)
        params[:correlation_strength] = rand(Beta(8, 3))  # 0.5-0.95, mean ~0.73 (highly systematic)
    end
    
    # Randomly select an industry type with scale-appropriate probabilities
    industry_types = ["manufacturing", "food_processing", "electronics", "furniture", "chemical", "automotive"]
    
    # Scale-dependent industry distribution (different industries dominate at different scales)
    industry_weights = if target_variables <= 250
        # Small scale: artisanal, local manufacturing, specialty food, custom furniture
        [0.25, 0.35, 0.15, 0.20, 0.03, 0.02]  # Food processing and furniture more common
    elseif target_variables <= 1000
        # Medium scale: established manufacturing, regional electronics, mid-size operations
        [0.40, 0.15, 0.25, 0.10, 0.08, 0.02]  # Manufacturing and electronics prevalent
    else
        # Large scale: multinational manufacturing, automotive, chemicals, high-tech
        [0.35, 0.08, 0.20, 0.05, 0.17, 0.15]  # Chemical and automotive more prominent
    end
    
    params[:industry_type] = sample(industry_types, Weights(industry_weights))
    
    return params
end

"""
    sample_product_mix_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a product mix optimization problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_product_mix_parameters(size::Symbol=:medium; seed::Int=0)
    Random.seed!(seed)
    
    # Map size categories to variable count ranges with random sampling
    target_variables = if size == :small
        rand(50:250)    # Small manufacturer: 50-250 products
    elseif size == :medium
        rand(250:1000)  # Medium enterprise: 250-1000 products
    elseif size == :large
        rand(1000:10000) # Large corporation: 1000-10000 products
    else
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_product_mix_parameters(target_variables; seed=seed)
end

"""
    calculate_product_mix_variable_count(params::Dict)

Calculate the total number of variables for a product mix problem.

# Arguments
- `params`: Dictionary of problem parameters

# Returns
- Integer: Total number of variables in the problem
"""
function calculate_product_mix_variable_count(params::Dict)
    # Extract parameters with defaults
    num_products = get(params, :num_products, 10)
    
    # Variables: x[1:num_products] >= 0
    return num_products
end

# Register the problem type
register_problem(
    :product_mix,
    generate_product_mix_problem,
    sample_product_mix_parameters,
    "Product mix optimization problem that maximizes profit by determining the optimal production quantities subject to resource constraints"
)