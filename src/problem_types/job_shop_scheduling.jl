using JuMP
using Random
using Distributions
using Statistics

"""
    JobShopSchedulingProblem <: ProblemGenerator

Generator for realistic job shop scheduling problems with sequential routing, machine conflicts,
release/due dates, and feasibility control through deadline slack.
"""
struct JobShopSchedulingProblem <: ProblemGenerator
    n_jobs::Int
    n_machines::Int
    operations_per_job::Vector{Int}
    machine_sequences::Vector{Vector{Int}}
    processing_times::Vector{Vector{Float64}}
    release_times::Vector{Float64}
    due_dates::Vector{Float64}
    weights::Vector{Float64}
    job_operation_indices::Vector{Vector{Int}}
    machine_operation_indices::Vector{Vector{Int}}
    operation_job::Vector{Int}
    operation_machine::Vector{Int}
    operation_duration::Vector{Float64}
    machine_pairs::Vector{Tuple{Int, Int, Int}}
    horizon::Float64
    feasibility_status::FeasibilityStatus
    variable_count::Int
end

const JOB_SHOP_TOLERANCE = 0.10

"""
    JobShopSchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a job shop scheduling instance targeting a variable count within ±10% of the request.
"""
function JobShopSchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    desired_status = feasibility_status == unknown ? (rand() < 0.5 ? feasible : infeasible) : feasibility_status
    target = max(target_variables, 50)
    tol_low = floor(Int, (1 - JOB_SHOP_TOLERANCE) * target)
    tol_high = ceil(Int, (1 + JOB_SHOP_TOLERANCE) * target)

    scenario = if target <= 400
        (mach_range = 5:8, job_range = 12:40, mean_ops = (3.0, 4.5), max_ops = 7)
    elseif target <= 2000
        (mach_range = 6:12, job_range = 25:120, mean_ops = (4.0, 6.5), max_ops = 10)
    else
        (mach_range = 10:20, job_range = 60:250, mean_ops = (5.5, 8.5), max_ops = 14)
    end

    best_instance = nothing
    best_gap = Inf

    n_machines = rand(scenario.mach_range)
    mean_ops = rand(Uniform(scenario.mean_ops...))
    n_jobs = clamp(round(Int, sqrt(2 * n_machines * target) / mean_ops), first(scenario.job_range), last(scenario.job_range))

    for attempt in 1:60
        n_machines = rand(scenario.mach_range)
        mean_ops = clamp(mean_ops + randn() * 0.2, scenario.mean_ops[1], scenario.mean_ops[2])
        n_jobs = clamp(n_jobs, first(scenario.job_range), last(scenario.job_range))

        ops_per_job = sample_operations_per_job(n_jobs, mean_ops, scenario.max_ops, n_machines)
        machine_sequences, processing_times = build_routings(ops_per_job, n_machines)

        instance = finalize_job_shop(machine_sequences, processing_times, n_machines, desired_status)
        var_count = instance.variable_count

        gap = abs(var_count - target) / target
        if gap < best_gap
            best_gap = gap
            best_instance = instance
        end

        if tol_low <= var_count <= tol_high
            best_instance = instance
            break
        else
            scale = sqrt(target / max(var_count, 1))
            n_jobs = clamp(round(Int, n_jobs * scale), first(scenario.job_range), last(scenario.job_range))
            mean_ops = clamp(mean_ops * sqrt(scale), scenario.mean_ops[1], scenario.mean_ops[2])
        end
    end

    if best_instance === nothing
        error("Failed to generate job shop instance")
    end

    if !(tol_low <= best_instance.variable_count <= tol_high)
        # Adjust by duplicating or trimming final jobs until within tolerance
        adjusted = adjust_instance(best_instance, target, tol_low, tol_high, desired_status)
        best_instance = adjusted
    end

    return best_instance
end

function sample_operations_per_job(n_jobs::Int, mean_ops::Float64, max_ops::Int, n_machines::Int)
    min_ops = 2
    ops = Vector{Int}(undef, n_jobs)
    for j in 1:n_jobs
        candidate = clamp(round(Int, rand(Normal(mean_ops, max(1.0, 0.35 * mean_ops)))), min_ops, max_ops)
        candidate = min(candidate, n_machines + 2)  # allow small repeats only
        ops[j] = max(min_ops, candidate)
    end
    return ops
end

function build_routings(operations_per_job::Vector{Int}, n_machines::Int)
    machine_sequences = Vector{Vector{Int}}(undef, length(operations_per_job))
    processing_times = Vector{Vector{Float64}}(undef, length(operations_per_job))
    base_scales = collect(range(1.3, 0.6, length=n_machines))

    for (j, job_ops) in enumerate(operations_per_job)
        machines = Vector{Int}(undef, job_ops)
        perm = randperm(n_machines)
        idx = 1
        for op in 1:job_ops
            if idx > length(perm)
                perm = randperm(n_machines)
                idx = 1
            end
            machines[op] = perm[idx]
            if op > 1 && machines[op] == machines[op - 1]
                machines[op] = machines[op] % n_machines + 1
            end
            idx += 1
        end
        machine_sequences[j] = machines

        times = Float64[]
        for mach in machines
            scale = base_scales[mach] * rand(Uniform(0.8, 1.2))
            push!(times, rand(Gamma(2.5, 0.8 * scale)) + rand(Uniform(0.1, 0.6)))
        end
        processing_times[j] = times
    end

    return machine_sequences, processing_times
end

function finalize_job_shop(machine_sequences, processing_times, n_machines::Int, desired_status::FeasibilityStatus)
    n_jobs = length(machine_sequences)

    job_operation_indices = Vector{Vector{Int}}(undef, n_jobs)
    operation_job = Int[]
    operation_machine = Int[]
    operation_duration = Float64[]

    op_counter = 0
    for j in 1:n_jobs
        job_indices = Int[]
        for (mach, duration) in zip(machine_sequences[j], processing_times[j])
            op_counter += 1
            push!(job_indices, op_counter)
            push!(operation_job, j)
            push!(operation_machine, mach)
            push!(operation_duration, duration)
        end
        job_operation_indices[j] = job_indices
    end

    machine_operation_indices = [Int[] for _ in 1:n_machines]
    for (idx, mach) in enumerate(operation_machine)
        push!(machine_operation_indices[mach], idx)
    end

    machine_pairs = Tuple{Int, Int, Int}[]
    pair_count = 0
    for m in 1:n_machines
        ops = machine_operation_indices[m]
        for i in 1:length(ops)-1
            for j in i+1:length(ops)
                pair_count += 1
                push!(machine_pairs, (m, ops[i], ops[j]))
            end
        end
    end

    release_times = zeros(Float64, n_jobs)
    due_dates = zeros(Float64, n_jobs)
    weights = zeros(Float64, n_jobs)

    total_processing = sum(operation_duration)
    release_span = max(total_processing * 0.15, 5.0)

    horizon = total_processing * 1.6

    for j in 1:n_jobs
        release_times[j] = rand(Uniform(0, release_span))
        job_total = sum(processing_times[j])
        tight_factor = desired_status == feasible ? rand(Uniform(1.25, 1.7)) : rand(Uniform(0.4, 0.8))
        due_dates[j] = release_times[j] + job_total * tight_factor
        weights[j] = rand(Uniform(1.0, 5.0)) * (1.0 + job_total / max(job_total, 6.0))
        horizon = max(horizon, due_dates[j] + job_total)
    end

    makespan_var = 1
    total_variables = op_counter + n_jobs + makespan_var + pair_count

    return JobShopSchedulingProblem(
        n_jobs,
        n_machines,
        [length(seq) for seq in machine_sequences],
        machine_sequences,
        processing_times,
        release_times,
        due_dates,
        weights,
        job_operation_indices,
        machine_operation_indices,
        operation_job,
        operation_machine,
        operation_duration,
        machine_pairs,
        horizon,
        desired_status,
        total_variables,
    )
end

function adjust_instance(problem::JobShopSchedulingProblem, target::Int, tol_low::Int, tol_high::Int, status::FeasibilityStatus)
    instance = problem
    attempts = 0
    while (instance.variable_count < tol_low || instance.variable_count > tol_high) && attempts < 20
        attempts += 1
        scale = instance.variable_count < target ? 1.15 : 0.85
        extra_jobs = clamp(round(Int, instance.n_jobs * scale) - instance.n_jobs, -instance.n_jobs + 2, instance.n_jobs * 2)
        new_job_count = max(2, instance.n_jobs + extra_jobs)
        ops_per_job = sample_operations_per_job(new_job_count, mean(instance.operations_per_job), maximum(instance.operations_per_job), instance.n_machines)
        machine_sequences, processing_times = build_routings(ops_per_job, instance.n_machines)
        instance = finalize_job_shop(machine_sequences, processing_times, instance.n_machines, status)
    end
    return instance
end

function build_model(problem::JobShopSchedulingProblem)
    model = Model()

    n_ops = length(problem.operation_duration)
    n_jobs = problem.n_jobs

    @variable(model, start_time[1:n_ops] >= 0)
    @variable(model, completion[1:n_jobs] >= 0)
    @variable(model, makespan >= 0)
    @variable(model, order_var[1:length(problem.machine_pairs)], Bin)

    # Job sequencing constraints
    for (j, op_indices) in enumerate(problem.job_operation_indices)
        for idx in 1:length(op_indices)-1
            current = op_indices[idx]
            nxt = op_indices[idx + 1]
            @constraint(model, start_time[nxt] >= start_time[current] + problem.operation_duration[current])
        end
        first_idx = op_indices[1]
        @constraint(model, start_time[first_idx] >= problem.release_times[j])
        for op_idx in op_indices
            @constraint(model, completion[j] >= start_time[op_idx] + problem.operation_duration[op_idx])
        end
        @constraint(model, completion[j] <= problem.due_dates[j])
    end

    # Machine disjunctive constraints
    for (pair_idx, (machine, op_a, op_b)) in enumerate(problem.machine_pairs)
        bigM = problem.horizon
        @constraint(model, start_time[op_b] >= start_time[op_a] + problem.operation_duration[op_a] - bigM * (1 - order_var[pair_idx]))
        @constraint(model, start_time[op_a] >= start_time[op_b] + problem.operation_duration[op_b] - bigM * order_var[pair_idx])
    end

    for j in 1:n_jobs
        @constraint(model, makespan >= completion[j])
    end

    @objective(model, Min, sum(problem.weights[j] * completion[j] for j in 1:n_jobs) + 0.05 * makespan)

    return model
end

register_problem(:job_shop_scheduling, JobShopSchedulingProblem, "Job shop scheduling with machine conflicts and due date feasibility control")
