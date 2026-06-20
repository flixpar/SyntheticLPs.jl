using JuMP
using Random

"""
    WorkloadBalanceAssignmentProblem <: ProblemGenerator

Generator for workload-balanced (min-makespan) task assignment problems.

# Overview
Assigns each task to exactly one eligible worker so as to minimize the *makespan*
— the maximum total workload carried by any single worker — subject to per-worker
capacity limits. This is the classic minimax / load-balancing flavour of the
assignment problem and is structurally distinct from the standard min-cost
one-to-one assignment (no cost objective, no one-task-per-worker restriction): here
a worker may take on many tasks, and what matters is keeping the busiest worker as
lightly loaded as possible.

Each task `j` carries a processing load `load_j > 0`. Each worker `w` has a
capacity `cap_w` (a shift-length style upper bound on total workload). Eligibility
is governed by a skill mask: a worker is eligible for a task whose skill group it
covers, plus a sprinkling of cross-trained pairs. A single continuous auxiliary
variable `L` (the makespan) upper-bounds every worker's load and is the sole term
in the objective.

Formulation:

    minimize    L
    subject to  sum_w x[w,j] = 1                       for every task j   (assign once)
                sum_j load_j * x[w,j] <= L             for every worker w (makespan)
                sum_j load_j * x[w,j] <= cap_w         for every worker w (capacity)
                x[w,j] = 0                             for ineligible (w,j)
                x[w,j] in {0,1},  L >= 0

In the LP relaxation `x[w,j] in [0,1]`.

# Fields
- `n_workers::Int`: Number of workers
- `n_tasks::Int`: Number of tasks
- `loads::Vector{Float64}`: Processing load of each task (length `n_tasks`)
- `capacities::Vector{Float64}`: Workload capacity of each worker (length `n_workers`)
- `eligible::Matrix{Bool}`: Eligibility mask (`n_workers` × `n_tasks`); `true` if worker may do task
"""
struct WorkloadBalanceAssignmentProblem <: ProblemGenerator
    n_workers::Int
    n_tasks::Int
    loads::Vector{Float64}
    capacities::Vector{Float64}
    eligible::Matrix{Bool}
end

"""
    WorkloadBalanceAssignmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a workload-balanced assignment instance.

Variable-count formula (decision variables created by `build_model`):

    total = n_workers * n_tasks   (the assignment matrix x)
          + 1                     (the makespan auxiliary L)

The constructor picks `n_workers` and `n_tasks` so `n_workers*n_tasks + 1` lands
near `target_variables`. Tasks comfortably outnumber workers (T > W) so that
balancing is non-trivial.

# Arguments
- `target_variables`: Target number of decision variables (`n_workers*n_tasks + 1`)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Feasibility
- `feasible`: eligibility is generous, every task fits on every worker
  (`max load <= min cap`), and total capacity comfortably exceeds total load. A
  greedy LPT assignment is constructed; capacities are raised if needed so a
  concrete integer assignment exists, which makes the LP relaxation feasible.
- `infeasible`: capacities are scaled so `sum_w cap_w < total_load`. Summing the
  capacity constraints gives `sum_w sum_j load_j x[w,j] <= sum_w cap_w < total_load`,
  while the assignment constraints force the left side to equal `total_load`. The
  contradiction survives the LP relaxation (it is independent of integrality,
  eligibility, and the makespan variable).
- `unknown`: natural capacities, biased toward feasibility but not forced.
"""
function WorkloadBalanceAssignmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing: total = W*T + 1, with T > W ---
    # Pick a worker:task ratio r = T/W in (1.5, 3.0). Then W*T = r*W^2 ≈ target-1
    # => W ≈ sqrt((target-1)/r), T ≈ r*W.
    eff = max(4, target_variables - 1)
    ratio = 1.5 + 1.5 * rand()  # tasks are 1.5x .. 3.0x the worker count
    n_workers = max(2, round(Int, sqrt(eff / ratio)))
    n_tasks = max(n_workers + 1, round(Int, eff / n_workers))

    # --- Scale-tiered parameter ranges (processing times / shift lengths) ---
    total_vars = n_workers * n_tasks + 1
    if total_vars <= 250
        load_lo, load_hi = 1.0, 8.0
    elseif total_vars <= 1000
        load_lo, load_hi = 2.0, 20.0
    else
        load_lo, load_hi = 5.0, 60.0
    end

    # --- Task loads (log-normal-ish spread within the tier for realism) ---
    loads = Vector{Float64}(undef, n_tasks)
    for j in 1:n_tasks
        base = load_lo + (load_hi - load_lo) * rand()
        # Occasional heavy task to make balancing meaningful.
        if rand() < 0.15
            base *= 1.3 + 0.7 * rand()
        end
        loads[j] = round(base, digits=2)
    end
    total_load = sum(loads)
    max_load = maximum(loads)

    # --- Skill groups drive eligibility ---
    n_groups = max(1, min(n_workers, rand(2:max(2, round(Int, sqrt(n_workers) + 1)))))
    worker_group = [rand(1:n_groups) for _ in 1:n_workers]
    task_group = [rand(1:n_groups) for _ in 1:n_tasks]

    # In-group eligibility is high; cross-group eligibility is a smaller chance
    # (cross-training). Generous in the feasible case so coverage is easy.
    if feasibility_status == feasible
        p_in, p_out = 0.97, 0.45
    else
        p_in, p_out = 0.90, 0.25
    end

    eligible = falses(n_workers, n_tasks)
    for w in 1:n_workers, j in 1:n_tasks
        p = (worker_group[w] == task_group[j]) ? p_in : p_out
        eligible[w, j] = rand() < p
    end

    # Every task needs at least one eligible worker (always — even infeasible
    # instances should fail via the capacity contradiction, not via empty columns,
    # so the contradiction is the clean, intended reason).
    for j in 1:n_tasks
        if !any(@view eligible[:, j])
            eligible[rand(1:n_workers), j] = true
        end
    end

    # --- Capacities by feasibility intent ---
    capacities = Vector{Float64}(undef, n_workers)

    if feasibility_status == infeasible
        # Pigeonhole: total capacity strictly below total load.
        # total_cap = frac * total_load with frac in [0.70, 0.90].
        frac = 0.70 + 0.20 * rand()
        total_cap = frac * total_load
        # Distribute total_cap across workers with mild heterogeneity, then rescale
        # exactly so the sum equals total_cap (keeps the strict inequality intact).
        raw = [0.7 + 0.6 * rand() for _ in 1:n_workers]
        s = sum(raw)
        for w in 1:n_workers
            capacities[w] = total_cap * raw[w] / s
        end
        # sum(capacities) == total_cap == frac * total_load < total_load by
        # construction, and the floor() rounding below only widens the shortfall,
        # so the strict pigeonhole infeasibility holds with no extra rescaling.

    else
        # feasible / unknown: build generous, heterogeneous capacities and then,
        # for the `feasible` case, GUARANTEE a concrete assignment fits via LPT.

        # Average load per worker if perfectly balanced.
        avg = total_load / n_workers
        # Capacity headroom factor; feasible gets more slack than unknown.
        head_lo, head_hi = feasibility_status == feasible ? (1.6, 2.4) : (1.1, 1.8)
        for w in 1:n_workers
            head = head_lo + (head_hi - head_lo) * rand()
            capacities[w] = max(max_load, avg * head)
        end

        if feasibility_status == feasible
            # Every task must fit on every worker: cap_w >= max_load.
            for w in 1:n_workers
                capacities[w] = max(capacities[w], max_load)
            end

            # Greedy LPT: assign heaviest tasks first to the least-loaded eligible
            # worker. Compute resulting per-worker loads; if any exceeds its cap,
            # raise that cap so the concrete integer assignment is admissible.
            order = sortperm(loads; rev=true)
            assigned_load = zeros(Float64, n_workers)
            for j in order
                cands = [w for w in 1:n_workers if eligible[w, j]]
                # (Guaranteed non-empty by the coverage fix above.)
                best = argmin(w -> assigned_load[w], cands)
                assigned_load[best] += loads[j]
            end
            for w in 1:n_workers
                if assigned_load[w] > capacities[w]
                    # Raise with a small safety margin.
                    capacities[w] = assigned_load[w] * 1.05
                end
            end
        end
    end

    # Round capacities for tidy data without breaking the feasibility math:
    # round feasible/unknown caps UP, infeasible caps DOWN (preserves shortfall).
    if feasibility_status == infeasible
        capacities = floor.(capacities, digits=2)
    else
        capacities = ceil.(capacities, digits=2)
    end

    return WorkloadBalanceAssignmentProblem(n_workers, n_tasks, loads, capacities, eligible)
end

"""
    build_model(prob::WorkloadBalanceAssignmentProblem)

Build a JuMP model for the workload-balanced assignment problem. Deterministic —
uses only the struct fields.

Decision variables:
- `x[w, j]`: binary, worker `w` performs task `j`
- `L`: continuous makespan (maximum worker workload)

# Returns
- `model`: The JuMP model
"""
function build_model(prob::WorkloadBalanceAssignmentProblem)
    model = Model()

    W = prob.n_workers
    T = prob.n_tasks

    # Variables (total = W*T + 1)
    @variable(model, x[1:W, 1:T], Bin)
    @variable(model, L >= 0)

    # Objective: minimize the makespan.
    @objective(model, Min, L)

    # Each task assigned to exactly one worker.
    for j in 1:T
        @constraint(model, sum(x[w, j] for w in 1:W) == 1)
    end

    # Makespan: each worker's total load is bounded by L.
    for w in 1:W
        @constraint(model, sum(prob.loads[j] * x[w, j] for j in 1:T) <= L)
    end

    # Per-worker capacity (the hard limit enabling the pigeonhole infeasibility).
    for w in 1:W
        @constraint(model, sum(prob.loads[j] * x[w, j] for j in 1:T) <= prob.capacities[w])
    end

    # Eligibility: forbid ineligible (w, j) pairs.
    for w in 1:W, j in 1:T
        if !prob.eligible[w, j]
            @constraint(model, x[w, j] == 0)
        end
    end

    return model
end

# Register the variant
register_variant(
    :assignment,
    :workload_balance,
    WorkloadBalanceAssignmentProblem,
    "Workload-balanced task assignment minimizing the makespan (maximum worker workload) over eligible task-worker pairs with per-worker capacities",
)
