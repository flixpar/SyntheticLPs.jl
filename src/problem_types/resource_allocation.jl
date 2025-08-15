using JuMP
using Random
using Statistics

"""
    generate_resource_allocation_problem(params::Dict=Dict(); seed::Int=0)

Generate a resource allocation optimization problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_activities`: Number of activities to allocate resources to (default: 10)
  - `:n_resources`: Number of resource types (default: 5)
  - `:min_resource`: Minimum amount of each resource available (default: 50)
  - `:max_resource`: Maximum amount of each resource available (default: 200)
  - `:min_profit`: Minimum profit per unit of activity (default: 0.5)
  - `:max_profit`: Maximum profit per unit of activity (default: 20.0)
  - `:min_usage`: Minimum resource usage per unit of activity (default: 0.1)
  - `:max_usage`: Maximum resource usage per unit of activity (default: 5.0)
  - `:correlation_strength`: Correlation between resource usage and profit (default: 0.7)
  - `:add_min_constraints`: Whether to add minimum activity level constraints (default: true)
  - `:min_level_prob`: Probability of having a minimum level constraint (default: 0.3)
  - `:solution_status`: Desired status of generated instance: `:feasible` (default), `:infeasible`, or `:all`
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_resource_allocation_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_activities = get(params, :n_activities, 10)
    n_resources = get(params, :n_resources, 5)
    min_resource = get(params, :min_resource, 50)
    max_resource = get(params, :max_resource, 200)
    min_profit = get(params, :min_profit, 0.5)
    max_profit = get(params, :max_profit, 20.0)
    min_usage = get(params, :min_usage, 0.1)
    max_usage = get(params, :max_usage, 5.0)
    correlation_strength = get(params, :correlation_strength, 0.7)
    add_min_constraints = get(params, :add_min_constraints, true)
    min_level_prob = get(params, :min_level_prob, 0.3)
    # Target solution status (default to feasible). Accept strings or symbols.
    solution_status = get(params, :solution_status, :feasible)
    if solution_status isa String
        solution_status = Symbol(lowercase(solution_status))
    end
    if !(solution_status in (:feasible, :infeasible, :all))
        error("Unknown solution_status=$(solution_status). Use :feasible, :infeasible, or :all")
    end
    # When targeting infeasible instances, minimum level constraints are required for this model family.
    # Respect user's choice otherwise.
    effective_add_min_constraints = add_min_constraints || (solution_status == :infeasible)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_activities => n_activities,
        :n_resources => n_resources,
        :min_resource => min_resource,
        :max_resource => max_resource,
        :min_profit => min_profit,
        :max_profit => max_profit,
        :min_usage => min_usage,
        :max_usage => max_usage,
        :correlation_strength => correlation_strength,
        :add_min_constraints => add_min_constraints,
        :min_level_prob => min_level_prob
    )
    
    # Generate data with correlation between profit and resource usage
    
    # Generate "quality factors" for each activity
    quality_factors = rand(n_activities)
    
    # Generate profits correlated with quality factors
    base_profit = rand(n_activities) .* (max_profit - min_profit) .+ min_profit
    quality_profit = quality_factors .* (max_profit - min_profit)
    profits = base_profit + correlation_strength * quality_profit
    
    # Generate resource usage with correlation to quality
    usage = zeros(n_activities, n_resources)
    
    for i in 1:n_activities
        for j in 1:n_resources
            # Base usage level
            base_usage = rand() * (max_usage - min_usage) + min_usage
            
            # Add correlation with quality
            quality_usage = quality_factors[i] * (max_usage - min_usage)
            
            # Combined usage
            usage[i, j] = base_usage + correlation_strength * quality_usage
        end
    end
    
    # Helper: profit-aware demand plan (no capacities). Scales activity demand by
    # profit per average usage, with mild quality-based heterogeneity.
    function compute_demand_plan(profits_vec::Vector{Float64}, usage_mat::Array{Float64,2}, quality::Vector{Float64})
        demand = zeros(n_activities)
        for i in 1:n_activities
            avg_u = mean(view(usage_mat, i, :)) + 1e-9
            hetero = 0.6 + 0.6 * quality[i]            # 0.6 .. 1.2
            demand[i] = hetero * profits_vec[i] / avg_u
        end
        # Normalize to a reasonable magnitude to avoid extremes
        s = 1.0 / (mean(demand) + 1e-9)
        return demand .* s
    end
    
    # Helper: compute per-activity single-variable caps given capacities
    function compute_single_activity_caps(usage_mat::Array{Float64,2}, resource_caps::Vector{Float64})
        caps = zeros(n_activities)
        for i in 1:n_activities
            local_caps = Float64[]
            for j in 1:n_resources
                push!(local_caps, resource_caps[j] / usage_mat[i, j])
            end
            caps[i] = minimum(local_caps)
        end
        return caps
    end

    # Initialize outputs common to all modes
    resources = zeros(n_resources)
    min_levels = zeros(n_activities)
    baseline_x = zeros(n_activities)
    violated_resource = 0
    violated_resources = Int[]
    capacity_anchor_theta = zeros(n_resources)
    demand_x = zeros(n_activities)

    if solution_status == :all
        # Original stochastic construction (may be feasible or infeasible)
        expected_usage = sum(usage, dims=1) / n_activities
        resources = vec(expected_usage) .* rand(n_resources) .* n_activities ./ 2
        resources = max.(resources, min_resource)
        resources = min.(resources, max_resource)

        if add_min_constraints
            for i in 1:n_activities
                if rand() < min_level_prob
                    max_possible = minimum([resources[j] / usage[i, j] for j in 1:n_resources])
                    min_levels[i] = rand(0.1:0.05:0.3) * max_possible
                end
            end
        end
    else
        # Constructive generation with plan-driven floors and capacity anchoring
        # 1) Compute an economically sensible demand plan
        demand_x = compute_demand_plan(profits, usage, quality_factors)
        # 2) Anchor capacities around demand consumption with random slack/tightness
        cons = vec(sum(usage .* demand_x, dims=1))
        for j in 1:n_resources
            if solution_status == :feasible
                capacity_anchor_theta[j] = 1.10 + 0.40 * rand()  # 1.10 .. 1.50
            else
                # Keep close to plan; floors will push over
                capacity_anchor_theta[j] = 0.95 + 0.15 * rand()  # 0.95 .. 1.10
            end
            # Anchor, then clamp to bounds
            resources[j] = cons[j] * capacity_anchor_theta[j]
            resources[j] = max(resources[j], min_resource)
            resources[j] = min(resources[j], max_resource)
        end
        # 3) Build a baseline plan by scaling demand to sit safely within capacities
        #    This ensures feasibility baseline even after clamping
        cons_post = vec(sum(usage .* demand_x, dims=1))
        scale_to_fit = minimum([resources[j] / max(cons_post[j], 1e-12) for j in 1:n_resources])
        fill_fraction = solution_status == :feasible ? (0.75 + 0.2 * rand()) : (0.7 + 0.2 * rand())
        baseline_x = demand_x .* max(scale_to_fit * fill_fraction, 0.0)

        # 4) Minimum level constraints derived from baseline to keep realism
        if effective_add_min_constraints
            selected = falses(n_activities)
            for i in 1:n_activities
                if rand() < min_level_prob && baseline_x[i] > 0
                    selected[i] = true
                    min_levels[i] = rand(0.1:0.05:0.3) * baseline_x[i]
                end
            end
            # Ensure we have at least 1-2 min constraints to keep structure rich
            if count(selected) == 0
                idxs = findall(x -> x > 0, baseline_x)
                if !isempty(idxs)
                    pick = rand(idxs)
                    selected[pick] = true
                    min_levels[pick] = rand(0.1:0.05:0.3) * baseline_x[pick]
                end
            end
            if solution_status == :infeasible && count(selected) < 2
                # Add another activity to enable combined resource stress
                idxs = setdiff(collect(1:n_activities), findall(selected))
                idxs = [i for i in idxs if baseline_x[i] > 0]
                if !isempty(idxs)
                    pick2 = rand(idxs)
                    selected[pick2] = true
                    min_levels[pick2] = rand(0.1:0.05:0.3) * baseline_x[pick2]
                end
            end
        end

        if solution_status == :feasible
            # Feasibility is guaranteed because baseline_x satisfies A x <= resources,
            # and min_levels are set below baseline_x component-wise.
            # Nothing else to do.
        else
            # Infeasible: Create structured violations (single or multi-resource)
            xi_cap = compute_single_activity_caps(usage, resources)

            # Cap current min levels at 80% of individual caps for realism
            for i in 1:n_activities
                if min_levels[i] > 0
                    min_levels[i] = min(min_levels[i], 0.8 * xi_cap[i])
                end
            end

            # Choose number of resources to violate (1, 2, or 3)
            k_viol = n_resources == 1 ? 1 : (rand() < 0.5 ? 1 : (rand() < 0.6 ? min(2, n_resources) : min(3, n_resources)))
            # Seed with the most stressed resource by current floors
            min_cons = [sum(usage[i, j] * min_levels[i] for i in 1:n_activities) for j in 1:n_resources]
            ratios = [min_cons[j] / max(resources[j], eps()) for j in 1:n_resources]
            order_r = sortperm(ratios; rev=true)
            violated_resources = order_r[1:k_viol]
            deltas = [0.05 + 0.20 * rand() for _ in 1:k_viol]  # 5%..25%
            targets = [resources[r] * (1.0 + deltas[idx]) for (idx, r) in enumerate(violated_resources)]

            # Candidate set: existing min-level activities, plus top users of violated resources
            S = findall(i -> min_levels[i] > 0, 1:n_activities)
            if length(S) < 3
                scores = [sum(usage[i, r] for r in violated_resources) for i in 1:n_activities]
                add_order = sortperm(scores; rev=true)
                for t in 1:min(5, n_activities)
                    if !(add_order[t] in S)
                        push!(S, add_order[t])
                    end
                end
            end
            S = unique(S)
            if isempty(S)
                # Fallback: pick 2 best-capacity activities
                order = sortperm(xi_cap; rev=true)
                take = min(2, length(order))
                for k in 1:take
                    i = order[k]
                    min_levels[i] = 0.2 * xi_cap[i]
                    push!(S, i)
                end
            end

            # Iteratively raise floors to violate all chosen resources
            residual = [targets[idx] - sum(usage[i, r] * min_levels[i] for i in 1:n_activities) for (idx, r) in enumerate(violated_resources)]
            iter_guard = 0
            while any(residual .> 1e-9) && iter_guard < 3 * n_activities
                # Pick the activity that best reduces weighted residuals
                best_i = 0
                best_gain = 0.0
                best_inc = 0.0
                for i in S
                    cap_inc = max(0.0, 0.95 * xi_cap[i] - min_levels[i])
                    if cap_inc <= 0
                        continue
                    end
                    # Compute max increment that reduces all positive residuals
                    inc_bound = Inf
                    weighted_gain = 0.0
                    for (idx, r) in enumerate(violated_resources)
                        if residual[idx] > 0 && usage[i, r] > 0
                            inc_bound = min(inc_bound, residual[idx] / usage[i, r])
                            weighted_gain += usage[i, r] * residual[idx]
                        end
                    end
                    inc = min(cap_inc, isfinite(inc_bound) ? inc_bound : 0.0)
                    if inc > 0 && weighted_gain * inc > best_gain
                        best_gain = weighted_gain * inc
                        best_i = i
                        best_inc = inc
                    end
                end
                if best_i == 0 || best_inc <= 0
                    # Expand candidate set with another high-capacity activity
                    candidates = setdiff(collect(1:n_activities), S)
                    candidates = [i for i in candidates if 0.95 * xi_cap[i] > min_levels[i]]
                    if isempty(candidates)
                        break
                    end
                    i_new = candidates[argmax(xi_cap[candidates])]
                    push!(S, i_new)
                    # Prime with a small floor to enable contributions
                    min_levels[i_new] = max(min_levels[i_new], 0.1 * xi_cap[i_new])
                else
                    min_levels[best_i] += best_inc
                    # Update residuals
                    for (idx, r) in enumerate(violated_resources)
                        residual[idx] -= best_inc * usage[best_i, r]
                    end
                end
                iter_guard += 1
            end
            # Final safety: ensure violation across all selected resources
            final_ok = true
            for (idx, r) in enumerate(violated_resources)
                final_total = sum(usage[i, r] * min_levels[i] for i in 1:n_activities)
                if !(final_total > resources[r])
                    final_ok = false
                    # As a last resort, slightly reduce capacity or bump floors within caps
                    new_cap = max(min_resource, final_total * 0.95)
                    if new_cap < resources[r]
                        resources[r] = new_cap
                    else
                        # Increase floors up to cap
                        bump = 0.05
                        for i in 1:n_activities
                            if 0.95 * xi_cap[i] > min_levels[i]
                                min_levels[i] = min(min_levels[i] * (1.0 + bump), 0.95 * xi_cap[i])
                            end
                        end
                    end
                end
            end
            if !final_ok
                # Recheck and force if needed by small reductions where possible
                for (idx, r) in enumerate(violated_resources)
                    final_total = sum(usage[i, r] * min_levels[i] for i in 1:n_activities)
                    if !(final_total > resources[r])
                        resources[r] = max(min_resource, min(resources[r], final_total * 0.98))
                    end
                end
            end
        end
    end
    
    # Store generated data in params
    actual_params[:profits] = profits
    actual_params[:usage] = usage
    actual_params[:resources] = resources
    actual_params[:min_levels] = min_levels
    actual_params[:quality_factors] = quality_factors
    actual_params[:solution_status] = solution_status
    actual_params[:baseline_x] = baseline_x
    actual_params[:capacity_anchor_theta] = capacity_anchor_theta
    actual_params[:demand_x] = solution_status == :all ? nothing : demand_x
    if violated_resource != 0
        actual_params[:violated_resource] = violated_resource
    end
    if !isempty(violated_resources)
        actual_params[:violated_resources] = violated_resources
    end
    
    # Create model
    model = Model()
    
    # Decision variables: activity levels
    @variable(model, x[1:n_activities] >= 0)
    
    # Resource constraints
    for j in 1:n_resources
        @constraint(model, sum(usage[i,j] * x[i] for i in 1:n_activities) <= resources[j])
    end
    
    # Minimum level constraints
    for i in 1:n_activities
        if min_levels[i] > 0
            @constraint(model, x[i] >= min_levels[i])
        end
    end
    
    # Objective: maximize profit
    @objective(model, Max, sum(profits[i] * x[i] for i in 1:n_activities))
    
    return model, actual_params
end

"""
    sample_resource_allocation_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for a resource allocation problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_resource_allocation_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Start with defaults - target_variables = n_activities
    params[:n_activities] = max(3, min(2000, target_variables))
    params[:n_resources] = rand(2:50)  # Scale with problem complexity
    params[:min_resource] = rand(50:1000)
    params[:max_resource] = rand(200:10000)
    
    # Iteratively adjust if needed (though for resource allocation, it's direct)
    for iteration in 1:5
        current_vars = calculate_resource_allocation_variable_count(params)
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust n_activities directly since it's the only parameter affecting variable count
        if current_vars < target_variables
            params[:n_activities] = min(2000, params[:n_activities] + 1)
        elseif current_vars > target_variables
            params[:n_activities] = max(3, params[:n_activities] - 1)
        end
    end
    
    # Parameters that scale with problem complexity
    params[:min_profit] = rand(0.1:0.1:2.0)
    params[:max_profit] = rand(5.0:5.0:100.0)
    params[:min_usage] = rand(0.01:0.01:0.5)
    params[:max_usage] = rand(1.0:1.0:20.0)
    params[:correlation_strength] = rand(0.5:0.1:0.9)
    params[:add_min_constraints] = rand() < 0.7  # 70% chance
    params[:min_level_prob] = rand(0.2:0.1:0.5)
    
    return params
end

"""
    sample_resource_allocation_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for a resource allocation problem using size categories.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_resource_allocation_parameters(size::Symbol=:medium; seed::Int=0)
    # Map size categories to approximate target variable counts
    target_map = Dict(
        :small => rand(50:250),
        :medium => rand(250:1000),
        :large => rand(1000:10000)
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_resource_allocation_parameters(target_map[size]; seed=seed)
end

"""
    calculate_resource_allocation_variable_count(params::Dict)

Calculate the number of variables in a resource allocation problem.

# Arguments
- `params`: Dictionary of problem parameters containing :n_activities

# Returns
- Number of variables (equal to number of activities)
"""
function calculate_resource_allocation_variable_count(params::Dict)
    n_activities = get(params, :n_activities, 10)
    return n_activities
end

# Register the problem type
register_problem(
    :resource_allocation,
    generate_resource_allocation_problem,
    sample_resource_allocation_parameters,
    "Resource allocation problem that maximizes profit by allocating limited resources to competing activities"
)