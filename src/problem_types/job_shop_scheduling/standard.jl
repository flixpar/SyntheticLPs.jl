using JuMP
using Random
using Distributions
using Statistics

"""
    JobShopSchedulingProblem <: ProblemGenerator

Generator for realistic job shop scheduling problems with sequential job routings,
machine no-overlap (disjunctive) constraints, release dates, and soft due dates.

# Overview
Each job consists of an ordered chain of operations, each requiring a specific
machine for a given processing time. Operations of the same job must run in
sequence; operations sharing a machine must not overlap (modeled with big-M
disjunctive constraints and a binary ordering variable per machine-operation
pair). Each job has a release date (its first operation cannot start earlier)
and a soft due date: lateness is penalized rather than forbidden through a
non-negative tardiness variable `T[j]` with `completion[j] - T[j] <= due_date[j]`.
The objective minimizes weighted tardiness `sum(weights[j] * T[j])` plus a small
makespan term (the textbook weighted-tardiness objective). Because tardiness is
unbounded above, a sequential schedule always exists, so `feasible`/`unknown`
instances are genuinely solvable. `infeasible` instances instead impose a HARD
deadline tighter than the unavoidable `release + total job processing time`,
creating a deterministic contradiction.

# Fields
- `n_jobs::Int`: Number of jobs
- `n_machines::Int`: Number of machines
- `n_ops::Int`: Total number of operations across all jobs
- `job_operation_indices::Vector{Vector{Int}}`: Global operation indices, in order, per job
- `operation_duration::Vector{Float64}`: Processing time of each operation
- `operation_machine::Vector{Int}`: Machine assigned to each operation
- `release_times::Vector{Float64}`: Earliest start time for each job's first operation
- `due_dates::Vector{Float64}`: Soft (or, for infeasible instances, hard) due date per job
- `weights::Vector{Float64}`: Tardiness penalty weight per job
- `job_total_processing::Vector{Float64}`: Sum of operation durations per job
- `machine_pairs::Vector{Tuple{Int,Int,Int}}`: `(machine, op_a, op_b)` pairs sharing a machine
- `horizon::Float64`: Big-M value (scheduling horizon) for disjunctive constraints
- `feasibility_status::FeasibilityStatus`: Resolved feasibility status of the instance
"""
struct JobShopSchedulingProblem <: ProblemGenerator
    n_jobs::Int
    n_machines::Int
    n_ops::Int
    job_operation_indices::Vector{Vector{Int}}
    operation_duration::Vector{Float64}
    operation_machine::Vector{Int}
    release_times::Vector{Float64}
    due_dates::Vector{Float64}
    weights::Vector{Float64}
    job_total_processing::Vector{Float64}
    machine_pairs::Vector{Tuple{Int,Int,Int}}
    horizon::Float64
    feasibility_status::FeasibilityStatus
end

"""
    sample_operations_per_job(n_jobs, mean_ops, max_ops, n_machines, rng_ok)

Sample the number of operations for each of `n_jobs` jobs from a Normal centered at
`mean_ops`, clamped to `[2, min(max_ops, n_machines + 2)]`. Returns a `Vector{Int}`.
"""
function sample_operations_per_job(n_jobs::Int, mean_ops::Float64, max_ops::Int, n_machines::Int)
    min_ops = 2
    cap = min(max_ops, n_machines + 2)
    ops = Vector{Int}(undef, n_jobs)
    for j in 1:n_jobs
        candidate = round(Int, rand(Normal(mean_ops, max(1.0, 0.35 * mean_ops))))
        ops[j] = clamp(candidate, min_ops, cap)
    end
    return ops
end

"""
    build_routings(operations_per_job, n_machines)

Build a random machine routing and processing-time vector for each job. Consecutive
operations within a job are forced onto distinct machines. Returns
`(machine_sequences, processing_times)`.
"""
function build_routings(operations_per_job::Vector{Int}, n_machines::Int)
    n_jobs = length(operations_per_job)
    machine_sequences = Vector{Vector{Int}}(undef, n_jobs)
    processing_times = Vector{Vector{Float64}}(undef, n_jobs)
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

"""
    count_machine_pairs(machine_sequences, n_machines)

Count the number of unordered machine-operation pairs (= number of binary ordering
variables in the model): the sum over machines of C(ops_on_machine, 2).
"""
function count_machine_pairs(machine_sequences::Vector{Vector{Int}}, n_machines::Int)
    counts = zeros(Int, n_machines)
    for seq in machine_sequences
        for m in seq
            counts[m] += 1
        end
    end
    return sum(c * (c - 1) ÷ 2 for c in counts)
end

"""
    JobShopSchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a job shop scheduling instance whose decision-variable count lands near
`target_variables`.

# Variable-count formula
`build_model` creates: `n_ops` start-time vars + `n_jobs` completion vars +
`n_jobs` tardiness vars + 1 makespan var + `pair_count` binary ordering vars,
i.e. total = `n_ops + 2*n_jobs + 1 + pair_count`, where
`pair_count = sum_m C(ops_on_machine_m, 2)`. The constructor iterates over the
number of jobs, computing this exact total for each candidate routing, and keeps
the candidate closest to the target.

# Arguments
- `target_variables`: Target number of decision variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function JobShopSchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Resolve status: 'unknown' produces a natural (non-forced) instance, so we
    # treat it like 'feasible' for data generation (no forced infeasibility).
    desired_status = feasibility_status
    target = max(target_variables, 20)

    # Scale-dependent structural ranges.
    if target <= 200
        mach_range = 4:7
        mean_ops_range = (3.0, 4.5)
        max_ops = 7
    elseif target <= 1000
        mach_range = 6:10
        mean_ops_range = (4.0, 6.0)
        max_ops = 10
    else
        mach_range = 9:16
        mean_ops_range = (5.0, 8.0)
        max_ops = 14
    end

    n_machines = rand(mach_range)
    mean_ops = rand(Uniform(mean_ops_range...))

    # Each job contributes ~mean_ops start vars + 2 (completion + tardiness), and the
    # machine pairs grow ~quadratically. Start with a rough estimate from the
    # approximation total ≈ n_jobs*(mean_ops + 2) + (n_jobs*mean_ops)^2 / (2*n_machines).
    function total_vars_for(machine_sequences::Vector{Vector{Int}})
        n_jobs = length(machine_sequences)
        n_ops = sum(length, machine_sequences)
        pair_count = count_machine_pairs(machine_sequences, n_machines)
        return n_ops + 2 * n_jobs + 1 + pair_count
    end

    # Initial guess for n_jobs solving the quadratic approximation for total = target.
    a = (mean_ops^2) / (2 * n_machines)
    b = mean_ops + 2.0
    # a*n_jobs^2 + b*n_jobs + 1 ≈ target
    disc = b^2 + 4 * a * (target - 1)
    n_jobs_guess = disc > 0 ? (-b + sqrt(disc)) / (2 * a) : sqrt(target)
    n_jobs = max(2, round(Int, n_jobs_guess))

    best_seqs = nothing
    best_times = nothing
    best_gap = Inf

    for _ in 1:80
        n_jobs = max(2, n_jobs)
        ops_per_job = sample_operations_per_job(n_jobs, mean_ops, max_ops, n_machines)
        machine_sequences, processing_times = build_routings(ops_per_job, n_machines)
        vc = total_vars_for(machine_sequences)
        gap = abs(vc - target) / target
        if gap < best_gap
            best_gap = gap
            best_seqs = machine_sequences
            best_times = processing_times
        end
        if gap <= 0.10
            break
        end
        # Adjust n_jobs toward the target (vc scales super-linearly in n_jobs).
        scale = sqrt(target / max(vc, 1))
        new_n_jobs = clamp(round(Int, n_jobs * scale), 2, max(2, n_jobs * 3))
        if new_n_jobs == n_jobs
            new_n_jobs = vc < target ? n_jobs + 1 : max(2, n_jobs - 1)
        end
        n_jobs = new_n_jobs
    end

    machine_sequences = best_seqs
    processing_times = best_times
    n_jobs = length(machine_sequences)

    # Flatten operations into a global indexing.
    job_operation_indices = Vector{Vector{Int}}(undef, n_jobs)
    operation_machine = Int[]
    operation_duration = Float64[]
    op_counter = 0
    for j in 1:n_jobs
        job_indices = Int[]
        for (mach, dur) in zip(machine_sequences[j], processing_times[j])
            op_counter += 1
            push!(job_indices, op_counter)
            push!(operation_machine, mach)
            push!(operation_duration, dur)
        end
        job_operation_indices[j] = job_indices
    end
    n_ops = op_counter

    # Per-machine operation lists -> disjunctive pairs.
    machine_operation_indices = [Int[] for _ in 1:n_machines]
    for (idx, mach) in enumerate(operation_machine)
        push!(machine_operation_indices[mach], idx)
    end
    machine_pairs = Tuple{Int,Int,Int}[]
    for m in 1:n_machines
        ops = machine_operation_indices[m]
        for i in 1:length(ops) - 1
            for k in i + 1:length(ops)
                push!(machine_pairs, (m, ops[i], ops[k]))
            end
        end
    end

    # Per-job totals and time data.
    job_total_processing = [sum(processing_times[j]) for j in 1:n_jobs]
    total_processing = sum(job_total_processing)
    release_span = max(total_processing * 0.15, 5.0)

    release_times = zeros(Float64, n_jobs)
    due_dates = zeros(Float64, n_jobs)
    weights = zeros(Float64, n_jobs)

    # Horizon (big-M) must dominate any feasible schedule: total processing plus
    # all release times is a safe, loose upper bound on every completion time.
    horizon = total_processing + release_span + 1.0

    for j in 1:n_jobs
        release_times[j] = rand(Uniform(0, release_span))
        jtot = job_total_processing[j]
        # Soft due date: a natural (possibly tight) target. Lateness is penalized,
        # not forbidden, so this never makes the model infeasible.
        tight_factor = rand(Uniform(0.9, 1.6))
        due_dates[j] = release_times[j] + jtot * tight_factor
        # Tardiness weight in [1, 5], lightly scaled by job size (longer jobs
        # matter slightly more). Clear, simple multiplier.
        weights[j] = rand(Uniform(1.0, 5.0)) * (1.0 + jtot / total_processing)
        horizon = max(horizon, due_dates[j] + jtot)
    end

    if desired_status == infeasible
        # Force a deterministic contradiction: pick one job and make its due date a
        # HARD deadline strictly tighter than its unavoidable minimum completion
        # (release + total processing on its own operations). The build_model
        # infeasible branch enforces completion[j] <= due_dates[j] WITHOUT a
        # tardiness escape, so the model is provably infeasible with clear margin.
        j = 1
        # Minimum possible completion of job j ignoring machine contention.
        min_completion = release_times[j] + job_total_processing[j]
        due_dates[j] = release_times[j] + job_total_processing[j] * 0.5  # well below min_completion
        @assert due_dates[j] < min_completion
    end

    return JobShopSchedulingProblem(
        n_jobs,
        n_machines,
        n_ops,
        job_operation_indices,
        operation_duration,
        operation_machine,
        release_times,
        due_dates,
        weights,
        job_total_processing,
        machine_pairs,
        horizon,
        desired_status,
    )
end

"""
    build_model(prob::JobShopSchedulingProblem)

Build a JuMP model for the job shop scheduling problem. Deterministic — uses only
data from the struct fields (no RNG).

For `feasible`/`unknown` instances, due dates are soft: each job has a tardiness
variable `T[j] >= 0` with `completion[j] - T[j] <= due_dates[j]`, and the objective
minimizes weighted tardiness. For `infeasible` instances, job 1's due date is a hard
deadline (`completion[1] <= due_dates[1]`) tighter than its minimum completion time,
producing a deterministic infeasibility.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::JobShopSchedulingProblem)
    model = Model()

    n_ops = prob.n_ops
    n_jobs = prob.n_jobs

    @variable(model, start_time[1:n_ops] >= 0)
    @variable(model, completion[1:n_jobs] >= 0)
    @variable(model, tardiness[1:n_jobs] >= 0)
    @variable(model, makespan >= 0)
    @variable(model, order_var[1:length(prob.machine_pairs)], Bin)

    infeas = prob.feasibility_status == infeasible

    # Job sequencing, release, completion and (soft/hard) due-date constraints.
    for (j, op_indices) in enumerate(prob.job_operation_indices)
        for idx in 1:length(op_indices) - 1
            cur = op_indices[idx]
            nxt = op_indices[idx + 1]
            @constraint(model, start_time[nxt] >= start_time[cur] + prob.operation_duration[cur])
        end
        first_idx = op_indices[1]
        @constraint(model, start_time[first_idx] >= prob.release_times[j])
        for op_idx in op_indices
            @constraint(model, completion[j] >= start_time[op_idx] + prob.operation_duration[op_idx])
        end
        if infeas && j == 1
            # HARD deadline (no tardiness escape): deterministic infeasibility.
            @constraint(model, completion[j] <= prob.due_dates[j])
        else
            # SOFT deadline: lateness absorbed by tardiness var.
            @constraint(model, completion[j] - tardiness[j] <= prob.due_dates[j])
        end
    end

    # Machine no-overlap via big-M disjunctions.
    for (pair_idx, (_machine, op_a, op_b)) in enumerate(prob.machine_pairs)
        bigM = prob.horizon
        @constraint(model, start_time[op_b] >= start_time[op_a] + prob.operation_duration[op_a] - bigM * (1 - order_var[pair_idx]))
        @constraint(model, start_time[op_a] >= start_time[op_b] + prob.operation_duration[op_b] - bigM * order_var[pair_idx])
    end

    for j in 1:n_jobs
        @constraint(model, makespan >= completion[j])
    end

    # Weighted-tardiness objective plus a small makespan regularizer.
    @objective(model, Min, sum(prob.weights[j] * tardiness[j] for j in 1:n_jobs) + 0.05 * makespan)

    return model
end

# Register the variant
register_variant(
    :job_shop_scheduling,
    :standard,
    JobShopSchedulingProblem,
    "Job shop scheduling with disjunctive machine no-overlap and weighted tardiness (soft due dates)",
)
