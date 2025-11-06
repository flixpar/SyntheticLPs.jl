using JuMP
using Random

"""
    AssignmentProblem <: ProblemGenerator

Generator for assignment problems that assign workers to tasks at minimum cost.

# Fields
- `n_workers::Int`: Number of workers
- `n_tasks::Int`: Number of tasks
- `costs::Matrix{Int}`: Cost matrix (n_workers × n_tasks)
- `allowed::Matrix{Bool}`: Compatibility matrix indicating valid assignments
"""
struct AssignmentProblem <: ProblemGenerator
    n_workers::Int
    n_tasks::Int
    costs::Matrix{Int}
    allowed::Matrix{Bool}
end

"""
    AssignmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an assignment problem instance.

# Arguments
- `target_variables`: Target number of variables (n_workers × n_tasks)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function AssignmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem characteristics based on size
    if target_variables <= 250
        balanced_prob = 0.8
        cost_base = (5, 30)
        specialization_prob = 0.2
        cost_variation_weights = [0.6, 0.3, 0.1]
    elseif target_variables <= 1000
        balanced_prob = 0.6
        cost_base = (10, 100)
        specialization_prob = 0.4
        cost_variation_weights = [0.3, 0.5, 0.2]
    else
        balanced_prob = 0.4
        cost_base = (50, 500)
        specialization_prob = 0.6
        cost_variation_weights = [0.2, 0.3, 0.5]
    end

    # Determine if balanced
    balanced = rand() < balanced_prob

    # Calculate dimensions
    if balanced
        n_workers = max(5, round(Int, sqrt(target_variables)))
        actual_vars = n_workers * n_workers
        if actual_vars < target_variables * 0.9
            n_workers = max(5, round(Int, sqrt(target_variables * 1.1)))
        elseif actual_vars > target_variables * 1.1
            n_workers = max(5, round(Int, sqrt(target_variables * 0.9)))
        end
        n_tasks = n_workers
    else
        sqrt_target = sqrt(target_variables)
        ratio = 0.5 + rand() * 1.5

        n_workers = max(5, round(Int, sqrt_target * sqrt(ratio)))
        n_tasks = max(5, round(Int, target_variables / n_workers))

        for _ in 1:3
            current_vars = n_workers * n_tasks
            if current_vars < target_variables * 0.9
                if abs(n_workers - sqrt_target) > abs(n_tasks - sqrt_target)
                    n_workers = max(5, round(Int, n_workers * 1.1))
                else
                    n_tasks = max(5, round(Int, n_tasks * 1.1))
                end
            elseif current_vars > target_variables * 1.1
                if abs(n_workers - sqrt_target) > abs(n_tasks - sqrt_target)
                    n_workers = max(5, round(Int, n_workers * 0.9))
                else
                    n_tasks = max(5, round(Int, n_tasks * 0.9))
                end
            else
                break
            end
        end
    end

    # Adjust for feasibility
    solution_status = feasibility_status == feasible ? :feasible :
                     feasibility_status == infeasible ? :infeasible : :all
    feas_slack_prob = 0.3
    feas_slack_max = 3
    cap_gap_rng = (0.05, 0.25)

    if solution_status == :feasible
        if n_workers < n_tasks
            add = n_tasks - n_workers
            if rand() < feas_slack_prob
                add += rand(1:max(1, Int(feas_slack_max)))
            end
            n_workers = n_workers + add
        end
    elseif solution_status == :infeasible
        balanced = false
        if n_workers >= n_tasks
            cap_low, cap_high = cap_gap_rng
            gap_ratio = clamp(rand() * (cap_high - cap_low) + cap_low, 0.01, 0.9)
            extra = max(1, ceil(Int, gap_ratio * n_workers))
            n_tasks = n_workers + extra
        end
    end

    # Generate costs
    min_cost, max_cost = cost_base
    range_multiplier = 0.8 + rand() * 0.4
    adjusted_max = max(min_cost + 5, round(Int, max_cost * range_multiplier))

    costs = zeros(Int, n_workers, n_tasks)

    # Compatibility structure
    total_vars_est = n_workers * n_tasks
    base_density = total_vars_est <= 250 ? 0.85 : (total_vars_est <= 1000 ? 0.70 : 0.50)

    allowed = trues(n_workers, n_tasks)

    # Skill groups
    gmax = min(6, max(2, round(Int, sqrt(min(n_workers, n_tasks)))))
    n_groups = rand(2:gmax)
    worker_groups = [rand(1:n_groups) for _ in 1:n_workers]
    task_groups = [rand(1:n_groups) for _ in 1:n_tasks]

    p_in = min(0.98, base_density)
    p_out = max(0.02, 0.3 * base_density)

    apply_compat = solution_status != :all
    if apply_compat
        for i in 1:n_workers, j in 1:n_tasks
            pij = task_groups[j] == worker_groups[i] ? p_in : p_out
            allowed[i, j] = rand() < pij
        end
    end

    # Infeasibility logic
    infeas_hall_prob = 0.4
    if solution_status == :infeasible
        use_capacity_shortfall = rand() >= infeas_hall_prob
        if !use_capacity_shortfall
            # Hall violation
            k = max(2, min(n_tasks, round(Int, 0.3 * n_tasks)))
            k = rand(max(2, round(Int, 0.2*n_tasks)) : max(2, min(n_tasks, round(Int, 0.5*n_tasks))))
            hall_tasks = sort(randperm(n_tasks)[1:k])
            m = max(1, min(n_workers-1, rand(max(1, round(Int, 0.2*k)) : max(1, k-1))))
            hall_workers = sort(randperm(n_workers)[1:m])
            for j in hall_tasks
                for i in 1:n_workers
                    allowed[i, j] = (i in hall_workers)
                end
            end
        end
    end

    # Feasibility guarantees
    if solution_status == :feasible
        if apply_compat
            used = falses(n_workers)
            task_order = randperm(n_tasks)
            for jj in task_order
                cands = [i for i in 1:n_workers if allowed[i, jj] && !used[i]]
                if isempty(cands)
                    pref = [i for i in 1:n_workers if worker_groups[i] == task_groups[jj] && !used[i]]
                    if isempty(pref)
                        pref = [i for i in 1:n_workers if !used[i]]
                    end
                    if isempty(pref)
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

    # Cost generation
    specialization = rand() < specialization_prob
    variation_choice = rand()
    cost_variation = if variation_choice < cost_variation_weights[1]
        :low
    elseif variation_choice < cost_variation_weights[1] + cost_variation_weights[2]
        :medium
    else
        :high
    end

    if specialization
        for i in 1:n_workers
            n_specializations = rand(1:min(3, n_tasks))
            specialized_tasks = randperm(n_tasks)[1:n_specializations]

            for j in 1:n_tasks
                if j in specialized_tasks
                    costs[i, j] = rand(min_cost:round(Int, min_cost + 0.3 * (adjusted_max - min_cost)))
                else
                    costs[i, j] = rand(round(Int, min_cost + 0.5 * (adjusted_max - min_cost)):adjusted_max)
                end
            end
        end
    else
        if cost_variation == :low
            mean_cost = (min_cost + adjusted_max) / 2
            range_factor = 0.3
            for i in 1:n_workers, j in 1:n_tasks
                low = max(min_cost, round(Int, mean_cost - range_factor * (adjusted_max - min_cost)))
                high = min(adjusted_max, round(Int, mean_cost + range_factor * (adjusted_max - min_cost)))
                bias = (worker_groups[i] == task_groups[j]) ? -0.1 : 0.1
                low_adj = clamp(round(Int, low + bias * (high - low)), min_cost, high)
                costs[i, j] = rand(low_adj:high)
            end
        elseif cost_variation == :high
            for i in 1:n_workers, j in 1:n_tasks
                if rand() < 0.1
                    costs[i, j] = rand() < 0.5 ? min_cost : adjusted_max
                else
                    if worker_groups[i] == task_groups[j]
                        costs[i, j] = rand(min_cost:round(Int, min_cost + 0.6 * (adjusted_max - min_cost)))
                    else
                        costs[i, j] = rand(round(Int, min_cost + 0.3 * (adjusted_max - min_cost)):adjusted_max)
                    end
                end
            end
        else  # :medium
            for i in 1:n_workers, j in 1:n_tasks
                if worker_groups[i] == task_groups[j]
                    costs[i, j] = rand(min_cost:round(Int, min_cost + 0.7 * (adjusted_max - min_cost)))
                else
                    costs[i, j] = rand(round(Int, min_cost + 0.2 * (adjusted_max - min_cost)):adjusted_max)
                end
            end
        end
    end

    return AssignmentProblem(n_workers, n_tasks, costs, allowed)
end

"""
    build_model(prob::AssignmentProblem)

Build a JuMP model for the assignment problem.

# Arguments
- `prob`: AssignmentProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::AssignmentProblem)
    model = Model()

    # Variables
    @variable(model, x[1:prob.n_workers, 1:prob.n_tasks], Bin)

    # Objective
    @objective(model, Min, sum(prob.costs[i, j] * x[i, j] for i in 1:prob.n_workers, j in 1:prob.n_tasks))

    # Each worker assigned to at most one task
    for i in 1:prob.n_workers
        @constraint(model, sum(x[i, j] for j in 1:prob.n_tasks) <= 1)
    end

    # Forbid incompatible assignments
    for i in 1:prob.n_workers, j in 1:prob.n_tasks
        if !prob.allowed[i, j]
            @constraint(model, x[i, j] == 0)
        end
    end

    # Each task assigned to exactly one worker
    for j in 1:prob.n_tasks
        @constraint(model, sum(x[i, j] for i in 1:prob.n_workers) == 1)
    end

    return model
end

# Register the problem type
register_problem(
    :assignment,
    AssignmentProblem,
    "Assignment problem that assigns workers to tasks at minimum cost with realistic scaling and cost structures"
)
