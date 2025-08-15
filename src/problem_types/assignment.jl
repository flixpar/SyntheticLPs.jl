using JuMP
using Random


"""
    generate_assignment_problem(params::Dict=Dict(); seed::Int=0)

Generate an assignment problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:n_workers`: Number of workers (default: 5)
  - `:n_tasks`: Number of tasks (default: 5)
  - `:cost_range`: Tuple (min, max) for assignment costs (default: (5, 20))
  - `:balanced`: Whether the problem is balanced (default: true)
  - `:cost_variation`: How much cost varies - :low, :medium, :high (default: :medium)
  - `:specialization`: Whether some workers are specialized for certain tasks (default: false)
  - `:solution_status`: Desired feasibility status for the generated problem. One of
    `:feasible` (default), `:infeasible`, or `:all`.
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_assignment_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    n_workers_in = get(params, :n_workers, 5)
    n_tasks_in = get(params, :n_tasks, 5)
    cost_range = get(params, :cost_range, (5, 20))
    balanced_in = get(params, :balanced, true)
    cost_variation = get(params, :cost_variation, :medium)
    specialization = get(params, :specialization, false)
    solution_status = get(params, :solution_status, :feasible)

    # Variety/realism knobs (optional)
    compat_density_opt = get(params, :compatibility_density, nothing)
    n_skill_groups_opt = get(params, :n_skill_groups, nothing)
    infeas_hall_prob = get(params, :infeasible_hall_prob, 0.4)  # P(choose Hall-type for infeasible)
    cap_gap_rng = get(params, :infeasible_capacity_gap_ratio_range, (0.05, 0.25))  # gap as fraction of workers
    feas_slack_prob = get(params, :feasible_slack_prob, 0.3)  # P(add slack workers beyond equality)
    feas_slack_max = get(params, :feasible_slack_max, 3)  # Max extra workers when adding slack

    # Normalize solution_status symbol (accept strings too)
    if solution_status isa String
        solution_status = Symbol(lowercase(String(solution_status)))
    end
    if !(solution_status in (:feasible, :infeasible, :all))
        error("assignment: invalid :solution_status=$(solution_status). Use :feasible | :infeasible | :all")
    end

    # Decide final dimensions based on desired feasibility while preserving realism
    # Keep original intent (balanced/unbalanced), but enforce feasibility/infeasibility.
    n_workers = Int(n_workers_in)
    n_tasks = Int(n_tasks_in)
    balanced = Bool(balanced_in)

    # If balanced is requested and user did not specify contradictory dims, respect it first.
    if balanced
        n_tasks = n_workers
    end

    # Adjust counts to guarantee requested solution status
    if solution_status == :feasible
        # Ensure enough worker capacity to satisfy all tasks under <=1 per worker and ==1 per task.
        if n_workers < n_tasks
            # Add enough workers to meet/exceed tasks; sometimes add slack for variety
            add = n_tasks - n_workers
            if rand() < feas_slack_prob
                add += rand(1:max(1, Int(feas_slack_max)))
            end
            n_workers = n_workers + add
        end
        # Keep balanced flag as provided by user; feasibility is ensured regardless via extra workers.
    elseif solution_status == :infeasible
        # Force unbalanced and ensure fewer workers than tasks.
        balanced = false
        if n_workers >= n_tasks
            # Increase tasks by a randomized gap ratio in the provided range
            cap_low, cap_high = cap_gap_rng
            gap_ratio = clamp(rand() * (cap_high - cap_low) + cap_low, 0.01, 0.9)
            extra = max(1, ceil(Int, gap_ratio * n_workers))
            n_tasks = n_workers + extra
        end
        # If n_workers < n_tasks already, leave as-is.
    else
        # :all -> do not intervene beyond balanced preference already applied
        nothing
    end
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :n_workers => n_workers,
        :n_tasks => n_tasks,
        :cost_range => cost_range,
        :balanced => balanced,
        :cost_variation => cost_variation,
        :specialization => specialization,
        :solution_status => solution_status,
        :original_n_workers => n_workers_in,
        :original_n_tasks => n_tasks_in,
        :original_balanced => balanced_in,
        :compatibility_density => compat_density_opt,
        :n_skill_groups => n_skill_groups_opt,
        :infeasible_hall_prob => infeas_hall_prob,
        :infeasible_capacity_gap_ratio_range => cap_gap_rng,
        :feasible_slack_prob => feas_slack_prob,
        :feasible_slack_max => feas_slack_max
    )
    
    # Generate assignment costs based on variation and specialization
    min_cost, max_cost = cost_range
    costs = zeros(Int, n_workers, n_tasks)

    # Determine how many dummy workers (if any) we added to ensure feasibility
    n_dummy_workers = 0
    if solution_status == :feasible && !balanced_in && n_workers > n_workers_in && n_workers == n_tasks
        # We expanded workers to match tasks
        n_dummy_workers = n_workers - n_workers_in
    end
    actual_params[:dummy_workers] = n_dummy_workers
    
    # --- Compatibility structure (skills/availability) for realism and variety ---
    # We'll build an allowed-mask of feasible (i,j) pairs. This enables modeling
    # of worker-task compatibility and can be used to diversify infeasibility causes.
    allowed = trues(n_workers, n_tasks)

    # Heuristic densities based on scale for realism (smaller instances are denser),
    # optionally overridden by :compatibility_density
    total_vars_est = n_workers * n_tasks
    base_density = total_vars_est <= 250 ? 0.85 : (total_vars_est <= 1000 ? 0.70 : 0.50)
    if compat_density_opt !== nothing
        base_density = clamp(Float64(compat_density_opt), 0.02, 0.98)
    end

    # Skill-group structure
    gmax = min(6, max(2, round(Int, sqrt(min(n_workers, n_tasks)))))
    n_groups = n_skill_groups_opt === nothing ? rand(2:gmax) : clamp(Int(n_skill_groups_opt), 2, gmax)
    # Assign groups roughly evenly
    worker_groups = [rand(1:n_groups) for _ in 1:n_workers]
    task_groups = [rand(1:n_groups) for _ in 1:n_tasks]

    p_in = min(0.98, base_density)
    p_out = max(0.02, 0.3 * base_density)

    # Introduce compatibility structure for :feasible/:infeasible modes, or when user
    # explicitly provides compatibility knobs, to preserve historical :all behavior otherwise.
    apply_compat = (solution_status != :all) || (compat_density_opt !== nothing) || (n_skill_groups_opt !== nothing)
    if apply_compat
        for i in 1:n_workers, j in 1:n_tasks
            pij = task_groups[j] == worker_groups[i] ? p_in : p_out
            allowed[i, j] = rand() < pij
        end
    end

    # --- Infeasibility variety for :infeasible ---
    if solution_status == :infeasible
        # With probability, choose capacity shortfall (classic), else Hall-type violation
        use_capacity_shortfall = rand() >= infeas_hall_prob
        if use_capacity_shortfall
            # Already ensured n_tasks > n_workers above
            # No extra action needed; compatibility sparsity stays as sampled
        else
            # Construct a Hall-violation subset: choose K tasks whose neighborhood is a small
            # set of M workers with M < |K|. This creates infeasibility even if workers >= tasks.
            k = max(2, min(n_tasks, round(Int, 0.3 * n_tasks)))
            k = rand( max(2, round(Int, 0.2*n_tasks)) : max(2, min(n_tasks, round(Int, 0.5*n_tasks))) )
            hall_tasks = sort(randperm(n_tasks)[1:k])
            m = max(1, min(n_workers-1, rand( max(1, round(Int, 0.2*k)) : max(1, k-1) )))
            hall_workers = sort(randperm(n_workers)[1:m])
            # Restrict K tasks to only M workers; forbid all other edges for those tasks
            for j in hall_tasks
                for i in 1:n_workers
                    allowed[i, j] = (i in hall_workers)
                end
            end
        end
    end

    # --- Feasibility guarantees for :feasible ---
    if solution_status == :feasible
        # Ensure at least one injective assignment exists among allowed edges
        # by planting a matching task->distinct worker, possibly relaxing allowance if needed.
        if apply_compat
            used = falses(n_workers)
            task_order = randperm(n_tasks)
            for jj in task_order
                # Prefer within-group compatible workers not yet used
                cands = [i for i in 1:n_workers if allowed[i, jj] && !used[i]]
                if isempty(cands)
                    # Un-forbid a worker (prefer same group, then any) to guarantee feasibility
                    pref = [i for i in 1:n_workers if worker_groups[i] == task_groups[jj] && !used[i]]
                    if isempty(pref)
                        pref = [i for i in 1:n_workers if !used[i]]
                    end
                    if isempty(pref)
                        # Should not happen if n_workers >= n_tasks; as a fallback, allow someone even if used
                        pref = collect(1:n_workers)
                    end
                    chosen = rand(pref)
                    allowed[chosen, jj] = true
                    used[chosen] = true
                else
                    chosen = rand(cands)
                    used[chosen] = true
                end
            end
        end
    end

    # --- Cost generation ---
    if specialization
        # Create worker specializations - some workers are much better at certain tasks
        for i in 1:n_workers
            # Each worker has 1-3 tasks they're specialized for
            n_specializations = rand(1:min(3, n_tasks))
            specialized_tasks = randperm(n_tasks)[1:n_specializations]
            
            for j in 1:n_tasks
                if j in specialized_tasks
                    # Lower cost for specialized tasks
                    costs[i, j] = rand(min_cost:round(Int, min_cost + 0.3 * (max_cost - min_cost)))
                else
                    # Higher cost for non-specialized tasks
                    costs[i, j] = rand(round(Int, min_cost + 0.5 * (max_cost - min_cost)):max_cost)
                end
            end
        end
    else
        # Standard cost generation based on variation level
        if cost_variation == :low
            # Low variation - costs are more uniform
            mean_cost = (min_cost + max_cost) / 2
            range_factor = 0.3
            for i in 1:n_workers, j in 1:n_tasks
                low = max(min_cost, round(Int, mean_cost - range_factor * (max_cost - min_cost)))
                high = min(max_cost, round(Int, mean_cost + range_factor * (max_cost - min_cost)))
                # Adjust by group proximity (within-group slightly cheaper)
                bias = (worker_groups[i] == task_groups[j]) ? -0.1 : 0.1
                low_adj = clamp(round(Int, low + bias * (high - low)), min_cost, high)
                costs[i, j] = rand(low_adj:high)
            end
        elseif cost_variation == :high
            # High variation - costs vary widely, some extreme values
            for i in 1:n_workers, j in 1:n_tasks
                if rand() < 0.1  # 10% chance of extreme costs
                    costs[i, j] = rand() < 0.5 ? min_cost : max_cost
                else
                    # Within-group tends cheaper, cross-group tends higher
                    if worker_groups[i] == task_groups[j]
                        costs[i, j] = rand(min_cost:round(Int, min_cost + 0.6 * (max_cost - min_cost)))
                    else
                        costs[i, j] = rand(round(Int, min_cost + 0.3 * (max_cost - min_cost)):max_cost)
                    end
                end
            end
        else  # :medium
            # Medium variation - standard uniform distribution
            for i in 1:n_workers, j in 1:n_tasks
                if worker_groups[i] == task_groups[j]
                    costs[i, j] = rand(min_cost:round(Int, min_cost + 0.7 * (max_cost - min_cost)))
                else
                    costs[i, j] = rand(round(Int, min_cost + 0.2 * (max_cost - min_cost)):max_cost)
                end
            end
        end
    end
    
    # If we introduced dummy workers to enforce feasibility, overwrite their costs to be high
    if n_dummy_workers > 0
        high_base = max(max_cost, round(Int, max_cost + 0.2 * (max_cost - min_cost) + 5))
        high_top = high_base + max(5, round(Int, 0.3 * (max_cost - min_cost) + 5))
        for i in (n_workers - n_dummy_workers + 1):n_workers, j in 1:n_tasks
            costs[i, j] = rand(high_base:high_top)
        end
    end

    # Store generated data in params
    actual_params[:costs] = costs
    actual_params[:allowed] = allowed
    
    # Model
    model = Model()
    
    # Variables (1 if worker i is assigned to task j, 0 otherwise)
    @variable(model, x[1:n_workers, 1:n_tasks], Bin)
    
    # Objective: Minimize total assignment cost
    @objective(model, Min, sum(costs[i, j] * x[i, j] for i in 1:n_workers, j in 1:n_tasks))
    
    # Constraints
    
    # Each worker can be assigned to at most one task
    for i in 1:n_workers
        @constraint(model, sum(x[i, j] for j in 1:n_tasks) <= 1)
    end
    
    # Forbid incompatible assignments via zero upper bounds
    if apply_compat
        for i in 1:n_workers, j in 1:n_tasks
            if !allowed[i, j]
                @constraint(model, x[i, j] == 0)
            end
        end
    end

    # Each task must be assigned to exactly one worker
    for j in 1:n_tasks
        if j <= n_workers || !balanced
            @constraint(model, sum(x[i, j] for i in 1:n_workers) == 1)
        end
    end
    
    return model, actual_params
end

"""
    sample_assignment_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for an assignment problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables (n_workers × n_tasks)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_assignment_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Determine problem characteristics based on size
    if target_variables <= 250
        # Small problems: often balanced, simple cost structure
        balanced_prob = 0.8
        cost_base = (5, 30)
        specialization_prob = 0.2
        cost_variation_dist = [0.6, 0.3, 0.1]  # [low, medium, high]
    elseif target_variables <= 1000
        # Medium problems: mix of balanced/unbalanced, moderate complexity
        balanced_prob = 0.6
        cost_base = (10, 100)
        specialization_prob = 0.4
        cost_variation_dist = [0.3, 0.5, 0.2]  # [low, medium, high]
    else
        # Large problems: more likely unbalanced, complex cost structures
        balanced_prob = 0.4
        cost_base = (50, 500)
        specialization_prob = 0.6
        cost_variation_dist = [0.2, 0.3, 0.5]  # [low, medium, high]
    end
    
    # Determine if this should be a balanced problem
    balanced = rand() < balanced_prob
    params[:balanced] = balanced
    
    if balanced
        # For balanced problems, n_workers = n_tasks, so target_variables = n_workers²
        n_workers = max(5, round(Int, sqrt(target_variables)))
        # Adjust to get closer to target
        actual_vars = n_workers * n_workers
        if actual_vars < target_variables * 0.9
            n_workers = max(5, round(Int, sqrt(target_variables * 1.1)))
        elseif actual_vars > target_variables * 1.1
            n_workers = max(5, round(Int, sqrt(target_variables * 0.9)))
        end
        params[:n_workers] = n_workers
        # n_tasks will be set equal to n_workers by the generator
    else
        # For unbalanced problems, calculate optimal dimensions
        sqrt_target = sqrt(target_variables)
        
        # Create realistic worker/task ratios
        # In real assignment problems, ratios typically range from 0.5 to 2.0
        ratio = 0.5 + rand() * 1.5  # ratio between 0.5 and 2.0
        
        n_workers = max(5, round(Int, sqrt_target * sqrt(ratio)))
        n_tasks = max(5, round(Int, target_variables / n_workers))
        
        # Fine-tune to get closer to target
        for _ in 1:3  # Multiple iterations to converge
            current_vars = n_workers * n_tasks
            if current_vars < target_variables * 0.9
                # Too few variables, increase the dimension that's further from sqrt
                if abs(n_workers - sqrt_target) > abs(n_tasks - sqrt_target)
                    n_workers = max(5, round(Int, n_workers * 1.1))
                else
                    n_tasks = max(5, round(Int, n_tasks * 1.1))
                end
            elseif current_vars > target_variables * 1.1
                # Too many variables, decrease the dimension that's further from sqrt
                if abs(n_workers - sqrt_target) > abs(n_tasks - sqrt_target)
                    n_workers = max(5, round(Int, n_workers * 0.9))
                else
                    n_tasks = max(5, round(Int, n_tasks * 0.9))
                end
            else
                break
            end
        end
        
        params[:n_workers] = n_workers
        params[:n_tasks] = n_tasks
    end
    
    # Set cost range based on problem size
    min_cost, max_cost = cost_base
    # Add some randomization to cost range
    range_multiplier = 0.8 + rand() * 0.4  # 0.8 to 1.2
    adjusted_max = max(min_cost + 5, round(Int, max_cost * range_multiplier))
    params[:cost_range] = (min_cost, adjusted_max)
    
    # Set cost variation based on problem size
    variation_choice = rand()
    if variation_choice < cost_variation_dist[1]
        params[:cost_variation] = :low
    elseif variation_choice < cost_variation_dist[1] + cost_variation_dist[2]
        params[:cost_variation] = :medium
    else
        params[:cost_variation] = :high
    end
    
    # Set specialization based on problem size
    params[:specialization] = rand() < specialization_prob
    
    return params
end

"""
    sample_assignment_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for an assignment problem using size categories.
This is a legacy function that calls the target-based version.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_assignment_parameters(size::Symbol=:medium; seed::Int=0)
    # Map size categories to target variable counts
    # Updated to match the required ranges: small=50-250, medium=250-1000, large=1000-10000
    target_map = Dict(
        :small => 150,     # Target 150 variables (range: 50-250)
        :medium => 500,    # Target 500 variables (range: 250-1000)  
        :large => 2000     # Target 2000 variables (range: 1000-10000)
    )
    
    if !haskey(target_map, size)
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return sample_assignment_parameters(target_map[size]; seed=seed)
end

"""
    calculate_assignment_variable_count(params::Dict)

Calculate the number of variables in an assignment problem.

# Arguments
- `params`: Dictionary of problem parameters containing :n_workers and :n_tasks

# Returns
- Integer representing the total number of variables (n_workers × n_tasks)
"""
function calculate_assignment_variable_count(params::Dict)
    n_workers = get(params, :n_workers, 5)
    n_tasks = get(params, :n_tasks, 5)
    
    # Handle balanced problems where n_tasks is set to n_workers
    balanced = get(params, :balanced, true)
    if balanced
        n_tasks = n_workers
    end
    
    return n_workers * n_tasks
end

# Register the problem type
register_problem(
    :assignment,
    generate_assignment_problem,
    sample_assignment_parameters,
    "Assignment problem that assigns workers to tasks at minimum cost with realistic scaling and cost structures"
)
