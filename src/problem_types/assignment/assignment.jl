# assignment category
#
# Entry point for the `assignment` problem category: field-service style
# assignment of jobs to workers over SPARSE compatibility graphs (each job
# reachable by its nearest qualified workers), built on the shared geographic
# helpers in `network_flow/geo_network.jl`. This file holds the shared
# geography/skill/eligibility machinery; each included file is one variant.

using Random
using Distributions
using StatsBase

"""
Largest `target_variables` accepted by the assignment variants (the
constructors hold per-job candidate lists and edge lists).
"""
const ASSIGNMENT_MAX_VARIABLES = 1_000_000

function _asg_check_target(target_variables::Int, name::AbstractString)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= ASSIGNMENT_MAX_VARIABLES || throw(
        ArgumentError(
            "assignment/$name supports at most $ASSIGNMENT_MAX_VARIABLES variables; " *
            "requested $target_variables.",
        ),
    )
    return nothing
end

"""
    _asg_world(rng, n_workers, n_jobs, n_skills; cross_training=0.3)
        -> (worker_pos, job_pos, worker_skills, job_skill, geography, primary)

Workers and jobs drawn from one `_geo_positions` population (metro clusters,
uniform coverage or a corridor; region side `12 sqrt(n)`), split at random.
Skill popularity follows a Zipf-like law; each job needs one skill, each
worker has a primary skill drawn from the same law plus up to two
cross-trained skills (probability `cross_training` each).
"""
function _asg_world(
    rng::AbstractRNG, n_workers::Int, n_jobs::Int, n_skills::Int; cross_training::Float64=0.3
)
    n = n_workers + n_jobs
    geography = let r = rand(rng)
        r < 0.45 ? :clustered : (r < 0.8 ? :uniform : :corridor)
    end
    positions, _ = _geo_positions(rng, n, geography; span=12.0 * sqrt(n))
    perm = randperm(rng, n)
    worker_pos = positions[perm[1:n_workers]]
    job_pos = positions[perm[(n_workers + 1):end]]
    popularity = Weights(shuffle!(rng, [1.0 / g^0.8 for g in 1:n_skills]))
    job_skill = [sample(rng, 1:n_skills, popularity) for _ in 1:n_jobs]
    worker_skills = Vector{Vector{Int}}(undef, n_workers)
    primary = Vector{Int}(undef, n_workers)
    for w in 1:n_workers
        skills = [sample(rng, 1:n_skills, popularity)]
        primary[w] = skills[1]
        for _ in 1:2
            rand(rng) < cross_training && push!(skills, sample(rng, 1:n_skills, popularity))
        end
        worker_skills[w] = sort!(unique(skills))
    end
    return worker_pos, job_pos, worker_skills, job_skill, geography, primary
end

"""
    _asg_skill_candidates(worker_pos, worker_skills, job_pos, job_skill, n_skills, k)
        -> Vector{Vector{Int}}

For every job, the (up to) `k` nearest workers holding its skill, nearest
first — one batched grid kNN query per skill, so the cost stays near-linear
even when skills are rare. Jobs whose skill nobody holds get an empty list.
"""
function _asg_skill_candidates(
    worker_pos::Vector{Tuple{Float64, Float64}},
    worker_skills::Vector{Vector{Int}},
    job_pos::Vector{Tuple{Float64, Float64}},
    job_skill::Vector{Int},
    n_skills::Int,
    k::Int,
)
    holders = [Int[] for _ in 1:n_skills]
    for (w, skills) in enumerate(worker_skills), g in skills
        push!(holders[g], w)
    end
    jobs_of = [Int[] for _ in 1:n_skills]
    for (j, g) in enumerate(job_skill)
        push!(jobs_of[g], j)
    end
    result = [Int[] for _ in job_pos]
    for g in 1:n_skills
        (isempty(holders[g]) || isempty(jobs_of[g])) && continue
        near = _geo_knn_query(
            worker_pos[holders[g]], job_pos[jobs_of[g]], min(k, length(holders[g]))
        )
        for (i, j) in enumerate(jobs_of[g])
            result[j] = holders[g][near[i]]
        end
    end
    return result
end

"""
    _asg_edge_counts(rng, n_jobs, n_edges, mean_k, cap) -> Vector{Int}

Per-job eligible-edge counts: lognormal around `mean_k`, at least 2 (1 only
when the budget is below two per job) and at most `cap[j]`, adjusted by +-1
steps so they sum to exactly `n_edges` (requires
`n_jobs <= n_edges <= sum(cap)`).
"""
function _asg_edge_counts(
    rng::AbstractRNG, n_jobs::Int, n_edges::Int, mean_k::Float64, cap::Vector{Int}
)
    kmin = n_edges >= 2n_jobs ? 2 : 1
    k = [
        clamp(round(Int, mean_k * rand(rng, LogNormal(0.0, 0.3))), min(kmin, cap[j]), cap[j]) for
        j in 1:n_jobs
    ]
    diff = n_edges - sum(k)
    while diff != 0
        j = rand(rng, 1:n_jobs)
        if diff > 0 && k[j] < cap[j]
            k[j] += 1
            diff -= 1
        elseif diff < 0 && k[j] > min(kmin, cap[j])
            k[j] -= 1
            diff += 1
        end
    end
    return k
end

include("standard.jl")
include("workload_balance.jl")
