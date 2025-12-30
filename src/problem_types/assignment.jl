using JuMP
using Random
using Distributions

"""
Assignment problem variants.

# Variants
- `assign_standard`: Basic assignment - minimize cost with one task per worker
- `assign_multi_assign`: Workers can handle multiple tasks (with capacity)
- `assign_skill_match`: Tasks require minimum skill levels from workers
- `assign_workload_balance`: Minimize maximum workload across workers
- `assign_preference`: Workers have preferences (some assignments vetoed)
- `assign_team`: Assign teams of workers to projects
- `assign_shift`: Multi-shift assignment with coverage requirements
- `assign_geographic`: Assignment with travel time/distance constraints
"""
@enum AssignmentVariant begin
    assign_standard
    assign_multi_assign
    assign_skill_match
    assign_workload_balance
    assign_preference
    assign_team
    assign_shift
    assign_geographic
end

"""
    AssignmentProblem <: ProblemGenerator

Generator for assignment problems with multiple variants.
"""
struct AssignmentProblem <: ProblemGenerator
    n_workers::Int
    n_tasks::Int
    costs::Matrix{Int}
    allowed::Matrix{Bool}
    variant::AssignmentVariant
    # Multi-assign variant
    worker_capacities::Union{Vector{Int}, Nothing}
    task_requirements::Union{Vector{Int}, Nothing}
    # Skill match variant
    n_skills::Int
    worker_skills::Union{Matrix{Float64}, Nothing}
    task_skill_reqs::Union{Matrix{Float64}, Nothing}
    # Workload balance variant
    task_workloads::Union{Vector{Float64}, Nothing}
    # Preference variant
    worker_preferences::Union{Matrix{Int}, Nothing}  # 1=preferred, 0=neutral, -1=vetoed
    preference_bonus::Float64
    # Team variant
    n_teams::Int
    team_members::Union{Vector{Vector{Int}}, Nothing}
    project_team_reqs::Union{Vector{Int}, Nothing}
    # Shift variant
    n_shifts::Int
    shift_tasks::Union{Vector{Vector{Int}}, Nothing}
    shift_requirements::Union{Vector{Int}, Nothing}
    worker_shift_avail::Union{Matrix{Bool}, Nothing}
    # Geographic variant
    worker_locs::Union{Vector{Tuple{Float64,Float64}}, Nothing}
    task_locs::Union{Vector{Tuple{Float64,Float64}}, Nothing}
    max_travel_distance::Float64
end

# Backwards compatibility
function AssignmentProblem(n_workers::Int, n_tasks::Int, costs::Matrix{Int}, allowed::Matrix{Bool})
    AssignmentProblem(
        n_workers, n_tasks, costs, allowed, assign_standard,
        nothing, nothing,
        0, nothing, nothing,
        nothing,
        nothing, 0.0,
        0, nothing, nothing,
        0, nothing, nothing, nothing,
        nothing, nothing, 0.0
    )
end

"""
    AssignmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                      variant::AssignmentVariant=assign_standard)

Construct an assignment problem instance with the specified variant.
"""
function AssignmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                           variant::AssignmentVariant=assign_standard)
    Random.seed!(seed)

    # Calculate dimensions
    n = max(5, round(Int, sqrt(target_variables)))
    n_workers = n + rand(-2:2)
    n_tasks = n + rand(-2:2)
    n_workers = max(5, n_workers)
    n_tasks = max(5, n_tasks)

    # Generate costs
    costs = rand(5:50, n_workers, n_tasks)

    # Compatibility matrix
    allowed = trues(n_workers, n_tasks)
    for i in 1:n_workers, j in 1:n_tasks
        if rand() < 0.1
            allowed[i, j] = false
        end
    end

    # Ensure feasibility for feasible instances (each task can be covered)
    if feasibility_status == feasible
        for j in 1:n_tasks
            if !any(allowed[:, j])
                allowed[rand(1:n_workers), j] = true
            end
        end
    elseif feasibility_status == infeasible
        # Make some tasks uncoverable
        for j in rand(1:n_tasks, max(1, n_tasks ÷ 4))
            allowed[:, j] .= false
        end
    end

    # Initialize variant fields
    worker_capacities = nothing
    task_requirements = nothing
    n_skills = 0
    worker_skills = nothing
    task_skill_reqs = nothing
    task_workloads = nothing
    worker_preferences = nothing
    preference_bonus = 0.0
    n_teams = 0
    team_members = nothing
    project_team_reqs = nothing
    n_shifts = 0
    shift_tasks = nothing
    shift_requirements = nothing
    worker_shift_avail = nothing
    worker_locs = nothing
    task_locs = nothing
    max_travel_distance = 0.0

    if variant == assign_multi_assign
        # Workers can handle multiple tasks
        worker_capacities = [rand(2:5) for _ in 1:n_workers]
        task_requirements = ones(Int, n_tasks)  # Each task needs 1 worker

        if feasibility_status == infeasible
            # More tasks than total capacity
            total_cap = sum(worker_capacities)
            task_requirements = [rand(1:2) for _ in 1:n_tasks]
            while sum(task_requirements) <= total_cap
                task_requirements[rand(1:n_tasks)] += 1
            end
        end

    elseif variant == assign_skill_match
        # Tasks require minimum skill levels
        n_skills = rand(2:min(5, max(2, n_workers ÷ 3)))
        worker_skills = rand(Uniform(0.0, 1.0), n_workers, n_skills)
        task_skill_reqs = rand(Uniform(0.2, 0.6), n_tasks, n_skills)

        # Update compatibility based on skills
        for i in 1:n_workers, j in 1:n_tasks
            if all(worker_skills[i, :] .>= task_skill_reqs[j, :])
                allowed[i, j] = true
            else
                allowed[i, j] = false
            end
        end

        if feasibility_status == feasible
            # Ensure each task has at least one capable worker
            for j in 1:n_tasks
                if !any(allowed[:, j])
                    best_worker = argmax([sum(worker_skills[i, :]) for i in 1:n_workers])
                    task_skill_reqs[j, :] = worker_skills[best_worker, :] .* 0.9
                    allowed[best_worker, j] = true
                end
            end
        elseif feasibility_status == infeasible
            # Make some tasks require impossible skill combinations
            for j in rand(1:n_tasks, max(1, n_tasks ÷ 4))
                task_skill_reqs[j, :] .= 1.1  # Impossible to meet
                allowed[:, j] .= false
            end
        end

    elseif variant == assign_workload_balance
        # Minimize maximum workload
        task_workloads = rand(Uniform(1.0, 10.0), n_tasks)

        # Standard compatibility
        if feasibility_status == infeasible
            # Make workload impossible to balance
            allowed[:, 1] .= false
            allowed[1, 1] = true
            task_workloads[1] = 1000.0  # Extreme workload on one task
        end

    elseif variant == assign_preference
        # Worker preferences
        worker_preferences = zeros(Int, n_workers, n_tasks)
        for i in 1:n_workers
            for j in 1:n_tasks
                r = rand()
                if r < 0.2
                    worker_preferences[i, j] = 1  # Preferred
                elseif r < 0.3
                    worker_preferences[i, j] = -1  # Vetoed
                    allowed[i, j] = false
                end
            end
        end
        preference_bonus = rand(Uniform(5.0, 15.0))

        if feasibility_status == feasible
            for j in 1:n_tasks
                if !any(allowed[:, j])
                    i = rand(1:n_workers)
                    allowed[i, j] = true
                    worker_preferences[i, j] = 0
                end
            end
        end

    elseif variant == assign_team
        # Assign teams to projects
        n_teams = rand(2:min(5, n_workers ÷ 2))
        team_members = Vector{Vector{Int}}()

        remaining = collect(1:n_workers)
        for t in 1:n_teams
            team_size = rand(2:min(4, length(remaining)))
            team = remaining[1:team_size]
            remaining = remaining[team_size+1:end]
            push!(team_members, team)
        end
        # Add remaining workers to existing teams
        for w in remaining
            push!(team_members[rand(1:n_teams)], w)
        end

        project_team_reqs = [rand(1:max(1, n_teams ÷ 2)) for _ in 1:n_tasks]

        if feasibility_status == infeasible
            # Require more teams than exist
            project_team_reqs = [n_teams + 1 for _ in 1:n_tasks]
        end

    elseif variant == assign_shift
        # Multi-shift scheduling
        n_shifts = rand(2:4)
        shift_tasks = [Int[] for _ in 1:n_shifts]
        for j in 1:n_tasks
            s = rand(1:n_shifts)
            push!(shift_tasks[s], j)
        end

        shift_requirements = [length(shift_tasks[s]) for s in 1:n_shifts]

        # Worker availability per shift
        worker_shift_avail = rand(Bool, n_workers, n_shifts)
        # Ensure some availability
        for i in 1:n_workers
            if !any(worker_shift_avail[i, :])
                worker_shift_avail[i, rand(1:n_shifts)] = true
            end
        end

        if feasibility_status == feasible
            for s in 1:n_shifts
                avail_workers = count(worker_shift_avail[:, s])
                shift_requirements[s] = min(shift_requirements[s], avail_workers)
            end
        elseif feasibility_status == infeasible
            # No available workers for some shift
            target_shift = rand(1:n_shifts)
            worker_shift_avail[:, target_shift] .= false
            shift_requirements[target_shift] = length(shift_tasks[target_shift])
        end

    elseif variant == assign_geographic
        # Geographic constraints
        worker_locs = [(rand(Uniform(0, 100)), rand(Uniform(0, 100))) for _ in 1:n_workers]
        task_locs = [(rand(Uniform(0, 100)), rand(Uniform(0, 100))) for _ in 1:n_tasks]

        # Maximum travel distance
        max_travel_distance = rand(Uniform(30, 70))

        # Update compatibility based on distance
        for i in 1:n_workers, j in 1:n_tasks
            dist = sqrt((worker_locs[i][1] - task_locs[j][1])^2 +
                       (worker_locs[i][2] - task_locs[j][2])^2)
            if dist > max_travel_distance
                allowed[i, j] = false
            end
        end

        if feasibility_status == feasible
            # Ensure coverage
            for j in 1:n_tasks
                if !any(allowed[:, j])
                    # Move a random worker closer
                    i = rand(1:n_workers)
                    worker_locs[i] = task_locs[j]
                    allowed[i, j] = true
                end
            end
        elseif feasibility_status == infeasible
            # Make some tasks unreachable
            for j in rand(1:n_tasks, max(1, n_tasks ÷ 4))
                task_locs[j] = (200.0, 200.0)  # Far away
                allowed[:, j] .= false
            end
        end
    end

    return AssignmentProblem(
        n_workers, n_tasks, costs, allowed, variant,
        worker_capacities, task_requirements,
        n_skills, worker_skills, task_skill_reqs,
        task_workloads,
        worker_preferences, preference_bonus,
        n_teams, team_members, project_team_reqs,
        n_shifts, shift_tasks, shift_requirements, worker_shift_avail,
        worker_locs, task_locs, max_travel_distance
    )
end

"""
    build_model(prob::AssignmentProblem)

Build a JuMP model for the assignment problem based on its variant.
"""
function build_model(prob::AssignmentProblem)
    model = Model()

    if prob.variant == assign_standard
        @variable(model, x[1:prob.n_workers, 1:prob.n_tasks], Bin)
        @objective(model, Min, sum(prob.costs[i, j] * x[i, j] for i in 1:prob.n_workers, j in 1:prob.n_tasks))

        for i in 1:prob.n_workers
            @constraint(model, sum(x[i, j] for j in 1:prob.n_tasks) <= 1)
        end

        for i in 1:prob.n_workers, j in 1:prob.n_tasks
            if !prob.allowed[i, j]
                @constraint(model, x[i, j] == 0)
            end
        end

        for j in 1:prob.n_tasks
            @constraint(model, sum(x[i, j] for i in 1:prob.n_workers) == 1)
        end

    elseif prob.variant == assign_multi_assign
        @variable(model, x[1:prob.n_workers, 1:prob.n_tasks], Bin)
        @objective(model, Min, sum(prob.costs[i, j] * x[i, j] for i in 1:prob.n_workers, j in 1:prob.n_tasks))

        # Worker capacity limits
        for i in 1:prob.n_workers
            @constraint(model, sum(x[i, j] for j in 1:prob.n_tasks) <= prob.worker_capacities[i])
        end

        # Task requirements
        for j in 1:prob.n_tasks
            @constraint(model, sum(x[i, j] for i in 1:prob.n_workers) >= prob.task_requirements[j])
        end

        # Compatibility
        for i in 1:prob.n_workers, j in 1:prob.n_tasks
            if !prob.allowed[i, j]
                @constraint(model, x[i, j] == 0)
            end
        end

    elseif prob.variant == assign_skill_match
        @variable(model, x[1:prob.n_workers, 1:prob.n_tasks], Bin)
        @objective(model, Min, sum(prob.costs[i, j] * x[i, j] for i in 1:prob.n_workers, j in 1:prob.n_tasks))

        for i in 1:prob.n_workers
            @constraint(model, sum(x[i, j] for j in 1:prob.n_tasks) <= 1)
        end

        for j in 1:prob.n_tasks
            @constraint(model, sum(x[i, j] for i in 1:prob.n_workers) == 1)
        end

        # Skill-based compatibility (already encoded in allowed matrix)
        for i in 1:prob.n_workers, j in 1:prob.n_tasks
            if !prob.allowed[i, j]
                @constraint(model, x[i, j] == 0)
            end
        end

    elseif prob.variant == assign_workload_balance
        @variable(model, x[1:prob.n_workers, 1:prob.n_tasks], Bin)
        @variable(model, max_workload >= 0)

        # Minimize maximum workload (minimax objective)
        @objective(model, Min, max_workload)

        for i in 1:prob.n_workers
            @constraint(model, sum(prob.task_workloads[j] * x[i, j] for j in 1:prob.n_tasks) <= max_workload)
        end

        for j in 1:prob.n_tasks
            @constraint(model, sum(x[i, j] for i in 1:prob.n_workers) == 1)
        end

        for i in 1:prob.n_workers, j in 1:prob.n_tasks
            if !prob.allowed[i, j]
                @constraint(model, x[i, j] == 0)
            end
        end

    elseif prob.variant == assign_preference
        @variable(model, x[1:prob.n_workers, 1:prob.n_tasks], Bin)

        # Minimize cost minus preference bonus
        @objective(model, Min,
            sum(prob.costs[i, j] * x[i, j] for i in 1:prob.n_workers, j in 1:prob.n_tasks) -
            prob.preference_bonus * sum(prob.worker_preferences[i, j] * x[i, j]
                for i in 1:prob.n_workers, j in 1:prob.n_tasks if prob.worker_preferences[i, j] > 0))

        for i in 1:prob.n_workers
            @constraint(model, sum(x[i, j] for j in 1:prob.n_tasks) <= 1)
        end

        for j in 1:prob.n_tasks
            @constraint(model, sum(x[i, j] for i in 1:prob.n_workers) == 1)
        end

        for i in 1:prob.n_workers, j in 1:prob.n_tasks
            if !prob.allowed[i, j]
                @constraint(model, x[i, j] == 0)
            end
        end

    elseif prob.variant == assign_team
        # Team assignment to projects
        @variable(model, y[1:prob.n_teams, 1:prob.n_tasks], Bin)  # Team-task assignment

        @objective(model, Min, sum(
            sum(prob.costs[w, j] for w in prob.team_members[t]) * y[t, j]
            for t in 1:prob.n_teams, j in 1:prob.n_tasks))

        # Each team at most one task
        for t in 1:prob.n_teams
            @constraint(model, sum(y[t, j] for j in 1:prob.n_tasks) <= 1)
        end

        # Task requirements (number of teams needed)
        for j in 1:prob.n_tasks
            @constraint(model, sum(y[t, j] for t in 1:prob.n_teams) >= prob.project_team_reqs[j])
        end

    elseif prob.variant == assign_shift
        # Multi-shift assignment
        @variable(model, x[1:prob.n_workers, 1:prob.n_tasks], Bin)
        @objective(model, Min, sum(prob.costs[i, j] * x[i, j] for i in 1:prob.n_workers, j in 1:prob.n_tasks))

        # Each task covered
        for j in 1:prob.n_tasks
            @constraint(model, sum(x[i, j] for i in 1:prob.n_workers) == 1)
        end

        # Worker can only work in shifts they're available
        for i in 1:prob.n_workers
            for s in 1:prob.n_shifts
                if !prob.worker_shift_avail[i, s]
                    for j in prob.shift_tasks[s]
                        @constraint(model, x[i, j] == 0)
                    end
                end
            end
        end

        # Shift requirements (workers per shift)
        for s in 1:prob.n_shifts
            if !isempty(prob.shift_tasks[s])
                available_workers = [i for i in 1:prob.n_workers if prob.worker_shift_avail[i, s]]
                if !isempty(available_workers)
                    @constraint(model, sum(sum(x[i, j] for j in prob.shift_tasks[s]) for i in available_workers) >= prob.shift_requirements[s])
                end
            end
        end

    elseif prob.variant == assign_geographic
        @variable(model, x[1:prob.n_workers, 1:prob.n_tasks], Bin)
        @objective(model, Min, sum(prob.costs[i, j] * x[i, j] for i in 1:prob.n_workers, j in 1:prob.n_tasks))

        for i in 1:prob.n_workers
            @constraint(model, sum(x[i, j] for j in 1:prob.n_tasks) <= 1)
        end

        for j in 1:prob.n_tasks
            @constraint(model, sum(x[i, j] for i in 1:prob.n_workers) == 1)
        end

        # Distance-based compatibility (already in allowed matrix)
        for i in 1:prob.n_workers, j in 1:prob.n_tasks
            if !prob.allowed[i, j]
                @constraint(model, x[i, j] == 0)
            end
        end
    end

    return model
end

# Register the problem type
register_problem(
    :assignment,
    AssignmentProblem,
    "Assignment problem with variants including standard, multi-assign, skill matching, workload balance, preferences, teams, shifts, and geographic constraints"
)
