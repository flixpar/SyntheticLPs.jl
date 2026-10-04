using JuMP
using Random
using Distributions
using StatsBase

"""
Planted assignment: `edge_of_job[j]` is the edge (index into `edges`) that
serves job `j`; the workers of these edges are pairwise distinct, so it is a
matching covering every job (an integer point of the model).
"""
struct AssignmentWitness
    edge_of_job::Vector{Int}
end

"""
Hall-violator certificate: the `jobs` can only be served by the `workers`
(every edge of every listed job goes to a listed worker), and there are fewer
workers than jobs. Summing the listed jobs' rows (`= 1` each) and the listed
workers' rows (`<= 1` each) over the same edges gives
`length(jobs) <= length(workers)` — a contradiction from rows alone, valid in
the LP relaxation. The set is a whole skill group (a rare trade), not one job,
so presolve does not see it.
"""
struct AssignmentHallCertificate
    jobs::Vector{Int}
    workers::Vector{Int}
end

"""
    AssignmentProblem <: ProblemGenerator

Sparse linear assignment of field-service jobs to workers (technicians, crews,
drivers).

# Overview

Only compatible (worker, job) pairs are variables: each job is reachable by a
lognormal number (mean 5-12) of its nearest workers, qualified ones first
(cross-skill workers at a premium when too few qualified are near).

    minimize    sum_e cost[e] * x[e]
    subject to  sum_{e serving j} x[e]  = 1        every job j
                sum_{e of w}      x[e] <= 1        every worker w with an edge
                x binary

The constraint matrix is a bipartite incidence matrix, so the LP relaxation is
totally unimodular and integral — the classic assignment LP, highly
degenerate (`n_jobs` basic ones among `n_jobs + n_workers` rows). Its value for
LP research is that degeneracy at scale, on realistic sparse graphs.

# Data grounding

Workers and jobs come from one geographic population (`_asg_world`); job skill
needs follow a Zipf-like popularity, workers hold a primary skill plus up to
two cross-trained ones. Cost = worker wage (lognormal, median 30/h, seniority)
x job duration (lognormal, median 2 h) + travel (0.8 per distance unit) + a
40 premium on unqualified pairs.

# Feasibility control

  - `feasible`: 3%-25% more workers than jobs; a planted matching (jobs in
    random order to the nearest free candidate, qualified first) is forced into
    the edge set and stored as the witness.
  - `infeasible`: a skill group with at least 4 jobs (and at least 3% of them)
    is served only by workers holding that skill, and that workforce is cut to
    70%-90% of the group's job count (Hall violation). Tiny instances without
    such a group fall back to a worker shortfall over all jobs.
  - `unknown`: 97%-120% as many workers as jobs, every job restricted to
    qualified workers (nearest first), no planting: scarce skills may or may
    not be coverable — natural, either side.

# Fields

  - `n_workers`, `n_jobs`, `edges::Vector{Tuple{Int,Int}}` (sorted
    `(worker, job)`), `costs`, `qualified::Vector{Bool}` (per edge)
  - `worker_skills`, `job_skill`, `worker_positions`, `job_positions`,
    `geography`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct AssignmentProblem <: ProblemGenerator
    n_workers::Int
    n_jobs::Int
    edges::Vector{Tuple{Int, Int}}
    costs::Vector{Float64}
    qualified::Vector{Bool}
    worker_skills::Vector{Vector{Int}}
    job_skill::Vector{Int}
    worker_positions::Vector{Tuple{Float64, Float64}}
    job_positions::Vector{Tuple{Float64, Float64}}
    geography::Symbol
    feasible_witness::Union{Nothing, AssignmentWitness}
    infeasibility_certificate::Union{Nothing, AssignmentHallCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    AssignmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Variables = edges, exactly `max(target_variables, 2)` except when a tiny
instance cannot offer that many compatible pairs. Rows = `n_jobs` + workers
with at least one edge.
"""
function AssignmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    _asg_check_target(target_variables, "standard")
    rng = MersenneTwister(seed)
    n_edges = max(target_variables, 2)
    mean_k = 5.0 + 7.0 * rand(rng)
    T = max(2, round(Int, n_edges / mean_k))
    ratio = feasibility_status == unknown ? 0.97 + 0.23 * rand(rng) : 1.03 + 0.22 * rand(rng)
    W = max(2, ceil(Int, T * ratio))
    while W * T < n_edges
        W += 1
    end
    G = clamp(round(Int, sqrt(T) / 3), 2, 40)
    worker_pos, job_pos, worker_skills, job_skill, geography, _ = _asg_world(rng, W, T, G)

    # Infeasible: pick a Hall group and cut its workforce.
    hall_skill = 0
    if feasibility_status == infeasible
        counts = [count(==(g), job_skill) for g in 1:G]
        groups = [g for g in 1:G if counts[g] >= max(4, ceil(Int, 0.03T))]
        if !isempty(groups)
            hall_skill = rand(rng, groups)
            holders = [w for w in 1:W if hall_skill in worker_skills[w]]
            keep = min(counts[hall_skill] - 1, max(2, floor(Int, counts[hall_skill] * (0.7 + 0.2 * rand(rng)))))
            if length(holders) > keep
                for w in shuffle(rng, holders)[1:(length(holders) - keep)]
                    filter!(!=(hall_skill), worker_skills[w])
                    isempty(worker_skills[w]) && push!(worker_skills[w], hall_skill == 1 ? 2 : 1)
                end
            end
        else
            # Tiny instance: worker shortfall over all jobs.
            W = max(1, T - max(1, round(Int, 0.1T)))
            worker_pos = worker_pos[1:W]
            worker_skills = worker_skills[1:W]
        end
    end

    # Candidate workers per job, nearest first: the nearest holders of its
    # skill (per-skill grid kNN), then — unless the job must stay
    # qualified-only (Hall group, unknown profile) — the nearest other
    # workers at a cross-skill premium.
    K = min(W, max(30, round(Int, 4 * mean_k)))
    near = _geo_knn_query(worker_pos, job_pos, K)
    skilled = _asg_skill_candidates(worker_pos, worker_skills, job_pos, job_skill, G, K)
    qualified_only(j) = feasibility_status == unknown || job_skill[j] == hall_skill
    function candidates(j)
        q = skilled[j]
        if qualified_only(j)
            return isempty(q) ? copy(near[j]) : copy(q)
        end
        inq = Set(q)
        return vcat(q, [w for w in near[j] if !(w in inq)])
    end
    cand = [candidates(j) for j in 1:T]

    # Planted matching (feasible): forced into the edge set.
    planted = zeros(Int, T)
    if feasibility_status == feasible
        free = trues(W)
        for j in randperm(rng, T)
            w = findfirst(w -> free[w], cand[j])
            if w === nothing
                w = argmin(v -> (free[v] ? hypot(worker_pos[v][1] - job_pos[j][1], worker_pos[v][2] - job_pos[j][2]) : Inf, v), 1:W)
                pushfirst!(cand[j], w)
            else
                w = cand[j][w]
            end
            planted[j] = w
            free[w] = false
        end
        for j in 1:T
            filter!(!=(planted[j]), cand[j])
            pushfirst!(cand[j], planted[j])
        end
    end

    cap = [length(c) for c in cand]
    k = _asg_edge_counts(rng, T, min(n_edges, sum(cap)), mean_k, cap)
    edges = Tuple{Int, Int}[]
    for j in 1:T, w in cand[j][1:k[j]]
        push!(edges, (w, j))
    end
    sort!(edges)

    wage = [30.0 * rand(rng, LogNormal(0.0, 0.25)) for _ in 1:W]
    duration = [2.0 * rand(rng, LogNormal(0.0, 0.5)) for _ in 1:T]
    qualified = [job_skill[j] in worker_skills[w] for (w, j) in edges]
    costs = [
        round(
            wage[w] * duration[j] +
            0.8 * hypot(worker_pos[w][1] - job_pos[j][1], worker_pos[w][2] - job_pos[j][2]) +
            (qualified[e] ? 0.0 : 40.0);
            digits=2,
        ) for (e, (w, j)) in enumerate(edges)
    ]

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        index = Dict(e => i for (i, e) in enumerate(edges))
        feasible_witness = AssignmentWitness([index[(planted[j], j)] for j in 1:T])
    elseif feasibility_status == infeasible
        jobs = hall_skill == 0 ? collect(1:T) : [j for j in 1:T if job_skill[j] == hall_skill]
        in_set = falses(T)
        in_set[jobs] .= true
        workers = sort!(unique([w for (w, j) in edges if in_set[j]]))
        length(workers) < length(jobs) ||
            error("assignment/standard: Hall certificate failed to separate (seed $seed)")
        infeasibility_certificate = AssignmentHallCertificate(jobs, workers)
    end

    return AssignmentProblem(
        W,
        T,
        edges,
        costs,
        qualified,
        worker_skills,
        job_skill,
        worker_pos,
        job_pos,
        geography,
        feasible_witness,
        infeasibility_certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::AssignmentProblem)

Build the sparse assignment model. Deterministic — uses only the struct fields.
"""
function build_model(prob::AssignmentProblem)
    model = Model()
    E = length(prob.edges)
    @variable(model, x[1:E], Bin)
    @objective(model, Min, sum(prob.costs[e] * x[e] for e in 1:E))
    of_job = [Int[] for _ in 1:(prob.n_jobs)]
    of_worker = [Int[] for _ in 1:(prob.n_workers)]
    for (e, (w, j)) in enumerate(prob.edges)
        push!(of_job[j], e)
        push!(of_worker[w], e)
    end
    for j in 1:(prob.n_jobs)
        @constraint(model, sum(x[e] for e in of_job[j]) == 1)
    end
    for w in 1:(prob.n_workers)
        isempty(of_worker[w]) && continue
        @constraint(model, sum(x[e] for e in of_worker[w]) <= 1)
    end
    return model
end

register_variant(
    :assignment,
    :standard,
    AssignmentProblem,
    "Sparse linear assignment of field-service jobs to their nearest qualified workers (skill groups, cross-training premiums, wage x duration + travel costs); planted matching witness and a skill-group Hall-violator certificate";
    default=true,
    tags=[:scheduling, :bipartite, :unimodular, :degenerate],
    max_target_variables=1_000_000,
)
