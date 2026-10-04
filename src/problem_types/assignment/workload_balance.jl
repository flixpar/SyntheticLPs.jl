using JuMP
using Random
using Distributions
using StatsBase

"""
Planted balanced assignment: `edge_of_task[t]` serves task `t` (an integer
point), built greedily (longest tasks first, each to the eligible worker whose
availability-scaled load grows least); `makespan` is its availability-scaled
makespan `max_w load[w] / availability[w]`, at most `max_makespan`.
"""
struct WorkloadBalanceWitness
    edge_of_task::Vector{Int}
    makespan::Float64
end

"""
Skill-group pigeonhole certificate (relaxation-proof). The `tasks` (one skill
group) can only be done by the `workers` (every eligible edge of a listed task
goes to a listed worker). Every listed task needs at least its fastest eligible
processing time, so

    required = sum_{t in tasks} min_{e eligible for t} processing_time[e]
            <= sum_{w in workers} availability[w] * max_makespan = available

The certificate stores both sides with `available < required` (by 5%-15%):
the group's workforce cannot absorb its workload within the overtime cap. It
combines many task rows with many worker rows, so presolve cannot see it.
"""
struct WorkloadBalanceCertificate
    tasks::Vector{Int}
    workers::Vector{Int}
    required::Float64
    available::Float64
end

"""
    WorkloadBalanceAssignmentProblem <: ProblemGenerator

Workload-balanced assignment on unrelated workers (R||Cmax-style LP) over a
sparse eligibility graph.

# Overview

Tasks (field jobs, tickets, deliveries) are assigned to eligible workers whose
processing times differ by speed and skill fit; workers have different
availabilities (part-time 0.5, 0.75, full-time 1.0). The model minimises a
weighted makespan plus assignment cost:

    minimize    makespan_weight * L + sum_e cost[e] x[e]
    subject to  sum_{e serving t} x[e] = 1                         every task t
                sum_{e of w} processing_time[e] x[e] <= availability[w] * L
                                                                   every worker w
                0 <= L <= max_makespan,  x binary

Processing times are worker-specific (`base[t] / speed[w]`, 25% slower for
cross-skill work, small noise), so the relaxation is a genuine
unrelated-machines LP — not the trivially fractional identical-machines one
(`L = total / n_workers`) — and the cost term breaks the massive degeneracy of
a pure makespan objective. Only eligible pairs are variables.

# Data grounding

Geography and skills as in `assignment/standard` (`_asg_world`); each task is
eligible for a lognormal number (mean 3-7) of its nearest workers holding its
skill (per-skill grid kNN; only a skill nobody holds falls back to the nearest
workers, 40% slower). Cross-trained holders work 15% slower than workers whose
primary trade it is. Task base duration is
lognormal (median 4 h, clipped to 0.5-10 h: one shift), worker speed lognormal
(sd 0.2); cost = wage x time +
travel.

# Feasibility control

  - `feasible`: `max_makespan` (the overtime cap) is 1.05-1.3x the planted
    greedy makespan; the planted assignment is the witness.
  - `infeasible`: a skill group with at least 4 tasks (and at least 3% of
    them), all served by holders only, has its task durations scaled up (a
    demand surge) until its fastest-possible workload exceeds the group
    workforce's capacity under the cap by 5%-15% (certificate above); the
    group is chosen among those where this surge keeps every single task
    doable within 90% of some eligible worker's cap, so no single row
    refutes the model. Without such a group, all tasks and workers are used.
  - `unknown`: `max_makespan` is 0.75-1.15x the planted greedy makespan — the
    fractional optimum can sit below the greedy one, so it may or may not fit.

# Sizing

Variables = eligible edges + 1, exactly `max(target_variables, 3)`. Rows =
`n_tasks + n_workers`.
"""
struct WorkloadBalanceAssignmentProblem <: ProblemGenerator
    n_workers::Int
    n_tasks::Int
    edges::Vector{Tuple{Int, Int}}
    processing_time::Vector{Float64}
    costs::Vector{Float64}
    availability::Vector{Float64}
    max_makespan::Float64
    makespan_weight::Float64
    worker_skills::Vector{Vector{Int}}
    task_skill::Vector{Int}
    worker_positions::Vector{Tuple{Float64, Float64}}
    task_positions::Vector{Tuple{Float64, Float64}}
    geography::Symbol
    feasible_witness::Union{Nothing, WorkloadBalanceWitness}
    infeasibility_certificate::Union{Nothing, WorkloadBalanceCertificate}
    feasibility_status::FeasibilityStatus
end

function WorkloadBalanceAssignmentProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    _asg_check_target(target_variables, "workload_balance")
    rng = MersenneTwister(seed)
    n_edges = max(target_variables, 3) - 1
    mean_k = 3.0 + 4.0 * rand(rng)
    T = max(2, round(Int, n_edges / mean_k))
    W = max(2, round(Int, T / (3.0 + 5.0 * rand(rng))))
    while W * T < n_edges
        W += 1
    end
    G = clamp(round(Int, sqrt(W) / 2), 1, 20)
    worker_pos, task_pos, worker_skills, task_skill, geography, primary = _asg_world(rng, W, T, G)

    # Every task is eligible for its nearest holders of the skill it needs
    # (per-skill grid kNN); only a skill nobody holds falls back to the
    # nearest workers at a cross-skill slowdown.
    K = min(W, max(20, round(Int, 4 * mean_k)))
    skilled = _asg_skill_candidates(worker_pos, worker_skills, task_pos, task_skill, G, K)
    unskilled = any(isempty, skilled) ? _geo_knn_query(worker_pos, task_pos, K) : Vector{Vector{Int}}()
    candidates(t) = isempty(skilled[t]) ? copy(unskilled[t]) : copy(skilled[t])
    cand = [candidates(t) for t in 1:T]
    cap = [length(c) for c in cand]
    k = _asg_edge_counts(rng, T, min(n_edges, sum(cap)), mean_k, cap)
    edges = Tuple{Int, Int}[]
    for t in 1:T, w in cand[t][1:k[t]]
        push!(edges, (w, t))
    end
    sort!(edges)
    E = length(edges)

    speed = [rand(rng, LogNormal(0.0, 0.2)) for _ in 1:W]
    availability = [rand(rng) < 0.6 ? 1.0 : (rand(rng) < 0.6 ? 0.75 : 0.5) for _ in 1:W]
    wage = [30.0 * rand(rng, LogNormal(0.0, 0.2)) for _ in 1:W]
    base = [clamp(4.0 * rand(rng, LogNormal(0.0, 0.6)), 0.5, 10.0) for _ in 1:T]
    # Primary trade at full speed, cross-trained trades 15% slower, unskilled
    # fallback 40% slower.
    fit = [
        task_skill[t] == primary[w] ? 1.0 : (task_skill[t] in worker_skills[w] ? 1.15 : 1.4) for
        (w, t) in edges
    ]
    noise = [rand(rng, LogNormal(0.0, 0.08)) for _ in 1:E]
    proc() = [round(base[t] / speed[w] * fit[e] * noise[e]; digits=3) for (e, (w, t)) in enumerate(edges)]
    processing_time = proc()
    travel = [0.5 * hypot(worker_pos[w][1] - task_pos[t][1], worker_pos[w][2] - task_pos[t][2]) for (w, t) in edges]
    of_task = [Int[] for _ in 1:T]
    for (e, (_, t)) in enumerate(edges)
        push!(of_task[t], e)
    end

    # Planted greedy (longest base first, least availability-scaled growth).
    load = zeros(W)
    edge_of_task = zeros(Int, T)
    for t in sortperm(base; rev=true)
        e = argmin(e -> ((load[edges[e][1]] + processing_time[e]) / availability[edges[e][1]], e), of_task[t])
        edge_of_task[t] = e
        load[edges[e][1]] += processing_time[e]
    end
    planted_makespan = maximum(load ./ availability)

    feasible_witness = nothing
    infeasibility_certificate = nothing
    max_makespan = if feasibility_status == unknown
        round(planted_makespan * (0.75 + 0.4 * rand(rng)); digits=2)
    else
        ceil(planted_makespan * (1.05 + 0.25 * rand(rng)); digits=2)
    end
    if feasibility_status == feasible
        feasible_witness = WorkloadBalanceWitness(edge_of_task, planted_makespan)
    elseif feasibility_status == infeasible
        # Pick a skill group (all its tasks served by holders only) whose
        # durations can surge enough to overload its workforce without any
        # single task becoming impossible on its own (which a single row
        # would reveal to presolve); fall back to all tasks.
        margin = 1.05 + 0.1 * rand(rng)
        fastest(ts) = sum(minimum(processing_time[e] for e in of_task[t]) for t in ts)
        workers_of(ts) = sort!(unique([edges[e][1] for t in ts for e in of_task[t]]))
        function surge_plan(ts)
            ws = workers_of(ts)
            available = sum(availability[w] for w in ws) * max_makespan
            need = margin * available / fastest(ts)
            # Largest surge keeping every task doable by some eligible worker
            # within 90% of that worker's cap.
            room = minimum(maximum(0.9 * availability[edges[e][1]] * max_makespan / processing_time[e] for e in of_task[t]) for t in ts)
            return need, room
        end
        counts = [count(==(g), task_skill) for g in 1:G]
        options = Vector{Int}[]
        for g in shuffle(rng, collect(1:G))
            counts[g] >= max(4, ceil(Int, 0.03T)) || continue
            ts = [t for t in 1:T if task_skill[t] == g]
            any(t -> isempty(skilled[t]), ts) && continue
            need, room = surge_plan(ts)
            need <= room && push!(options, ts)
        end
        tasks = isempty(options) ? collect(1:T) : options[1]
        workers = workers_of(tasks)
        available = sum(availability[w] for w in workers) * max_makespan
        if fastest(tasks) < margin * available
            surge = margin * available / fastest(tasks)
            for t in tasks
                base[t] *= surge
            end
            processing_time = proc()
            # Rounding to 3 digits can shave a hair: top up until strict.
            while fastest(tasks) <= available
                for t in tasks
                    base[t] *= 1.001
                end
                processing_time = proc()
            end
        end
        infeasibility_certificate = WorkloadBalanceCertificate(tasks, workers, fastest(tasks), available)
    end

    costs = [round(wage[w] * processing_time[e] + travel[e]; digits=2) for (e, (w, _)) in enumerate(edges)]
    typical_cost = sum(minimum(costs[e] for e in of_task[t]) for t in 1:T)
    makespan_weight = round(typical_cost / planted_makespan * (0.5 + 1.5 * rand(rng)); sigdigits=4)

    return WorkloadBalanceAssignmentProblem(
        W,
        T,
        edges,
        processing_time,
        costs,
        availability,
        max_makespan,
        makespan_weight,
        worker_skills,
        task_skill,
        worker_pos,
        task_pos,
        geography,
        feasible_witness,
        infeasibility_certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::WorkloadBalanceAssignmentProblem)

Build the sparse unrelated-workers balancing model. Deterministic — uses only
the struct fields.
"""
function build_model(prob::WorkloadBalanceAssignmentProblem)
    model = Model()
    E = length(prob.edges)
    @variable(model, x[1:E], Bin)
    @variable(model, 0 <= L <= prob.max_makespan)
    @objective(model, Min, prob.makespan_weight * L + sum(prob.costs[e] * x[e] for e in 1:E))
    of_task = [Int[] for _ in 1:(prob.n_tasks)]
    of_worker = [Int[] for _ in 1:(prob.n_workers)]
    for (e, (w, t)) in enumerate(prob.edges)
        push!(of_task[t], e)
        push!(of_worker[w], e)
    end
    for t in 1:(prob.n_tasks)
        @constraint(model, sum(x[e] for e in of_task[t]) == 1)
    end
    for w in 1:(prob.n_workers)
        @constraint(
            model,
            sum(prob.processing_time[e] * x[e] for e in of_worker[w]; init=AffExpr(0.0)) -
            prob.availability[w] * L <= 0
        )
    end
    return model
end

register_variant(
    :assignment,
    :workload_balance,
    WorkloadBalanceAssignmentProblem,
    "Workload-balanced assignment on unrelated workers over a sparse eligibility graph: worker-specific processing times, availability-scaled makespan rows, an overtime cap, and a cost term; planted greedy witness and a skill-group pigeonhole certificate",
)
