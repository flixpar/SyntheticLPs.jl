using JuMP
using Random
using Distributions

"""
Planted feasible r-allocation: the open hub set and, per node, the `r` hubs it
is allocated to, plus the reach window that keeps all of them admissible.
"""
struct HubBackupWitness
    hubs::Vector{Int}
    assignments::Vector{Vector{Int}}
    reach::Float64
end

"""
    RAllocationHubProblem <: ProblemGenerator

Generator for the **uncapacitated r-allocation p-hub median problem**
(UrApHMP; Peiro, Corberan & Marti 2014) with allocation reach windows.

# Overview

Relaxes `p_hub_median`'s single allocation: exactly `p` hubs are opened, and
every node is allocated to `r` of them (2 <= r <= p). Each origin-destination
pair still travels a single path `i -> k -> m -> j`, but may choose its entry
hub among the origin's `r` hubs and its exit hub among the destination's.
Primary/backup hub pairs are how airlines and parcel carriers protect node
service against hub disruptions, at a lower discount than full multiple
allocation.

The model keeps the four-index path-flow structure, with the allocation
linking relaxed from equalities to inequalities `sum_m x_ikmj <= w_ij * z_ik`
(a pair uses at most one of its origin's hubs). Flows and costs are
symmetrised, so each unordered pair is one commodity.

# Fields
As `PHubMedianProblem`, plus:
- `r::Int`: number of hubs every node is allocated to (exactly)
"""
struct RAllocationHubProblem <: ProblemGenerator
    n_nodes::Int
    p::Int
    r::Int
    chi::Float64
    alpha::Float64
    delta::Float64
    locations::Vector{Tuple{Float64,Float64}}
    dist::Matrix{Float64}
    cost::Matrix{Float64}
    flow::Matrix{Float64}
    reach::Float64
    admissible::Vector{Vector{Int}}
    feasible_witness::Union{Nothing,HubBackupWitness}
    infeasibility_certificate::Union{Nothing,DisjointRegionCertificate}
    feasibility_status::FeasibilityStatus
end

function _build_r_allocation(n_nodes::Int,
                             feasibility_status::FeasibilityStatus,
                             rng::AbstractRNG)
    n = n_nodes
    p = clamp(round(Int, n / 3) + rand(rng, 0:1), 3, min(8, n - 1))
    # Keep r below the node count so windows stay small at tiny sizes.
    r = max(2, min(p, n - 2, 2 + (rand(rng) < 0.25 ? 1 : 0)))

    if feasibility_status == infeasible
        q = p + 1
        centers = _hub_ring_centers(rng, q)
        node_group = vcat(collect(1:q), rand(rng, 1:q, max(0, n - q)))
        min_sep = minimum(hypot(centers[a][1] - centers[b][1],
                                centers[a][2] - centers[b][2])
                          for a in 1:q for b in (a + 1):q)
        spread = 0.15 * min_sep
        locations = [g <= q ? centers[g] :
                     (clamp(centers[node_group[g]][1] + rand(rng, Uniform(-spread, spread)),
                            0.0, 100.0),
                      clamp(centers[node_group[g]][2] + rand(rng, Uniform(-spread, spread)),
                            0.0, 100.0))
                     for g in 1:n]
        dist = _hub_distance_matrix(locations)
        groups = [Int[] for _ in 1:q]
        for g in 1:n
            push!(groups[node_group[g]], g)
        end
        # Nodes only reach candidates inside their own group, and each group
        # needs at least one hub of its own - more than the p allowed, which
        # refutes feasibility regardless of r.
        reach = 0.40 * min_sep
        certificate = DisjointRegionCertificate(groups, p)
        hubs = Int[]
        assignments = [Int[] for _ in 1:n]
    else
        shape = rand(rng, (:clustered, :corridor, :archipelago))
        locations = _hub_city_locations(rng, n, shape)
        dist = _hub_distance_matrix(locations)
        hubs, cover = _hub_cover_hubs(dist, p, r)
        # The r nearest planted hubs of every node, in order.
        assignments = [sort(hubs; by=k -> dist[i, k]) for i in 1:n]
        assignments = [a[1:min(r, length(a))] for a in assignments]
        reach = feasibility_status == feasible ?
                cover * rand(rng, Uniform(1.005, 1.1)) :
                cover * rand(rng, Uniform(0.8, 1.25))
        certificate = nothing
    end

    cost = _hub_detour_cost_matrix(rng, dist, 1.0, 1.35)
    populations = _hub_populations(rng, n)
    decay = rand(rng, Uniform(0.4, 1.0))
    noise = rand(rng, Uniform(0.6, 1.1))
    flow = _hub_gravity_flows(rng, n, populations, dist, decay, noise;
                              symmetric=true, scale=rand(rng, Uniform(20.0, 90.0)))
    admissible = _hub_reach_admissible(dist, reach)

    # Feasible requests must give every node r admissible candidates.
    if feasibility_status == feasible
        for i in 1:n
            while length(admissible[i]) < r
                push!(admissible[i],
                      sort(1:n; by=k -> dist[i, k])[length(admissible[i]) + 1])
            end
            sort!(admissible[i])
        end
        reach = maximum(maximum(dist[i, k] for k in admissible[i]) for i in 1:n)
    end

    witness = feasibility_status == feasible ?
              HubBackupWitness(hubs, assignments, reach) : nothing
    return RAllocationHubProblem(n, p, r, 1.0, rand(rng, Uniform(0.2, 0.8)), 1.0,
                                 locations, dist, cost, flow, reach, admissible,
                                 witness, certificate, feasibility_status)
end

"""
    RAllocationHubProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an r-allocation p-hub median instance. The variable count matches
`PHubMedianProblem`:

    vars = sum_{i<j} |A_i| * |A_j| + sum_i |A_i| + |union_i A_i|

Feasibility handling mirrors `p_hub_median` (cover witness / disjoint-region
certificate), except that feasible requests also guarantee `|A_i| >= r` for
every node.
"""
function RAllocationHubProblem(target_variables::Int,
                               feasibility_status::FeasibilityStatus, seed::Int)
    target = max(target_variables, 1)
    hint = clamp(round(Int, 2.0 * target^0.25), 3, 70)
    best = nothing
    best_score = (1, Inf)
    for attempt in 1:20
        rng = MersenneTwister(seed + 104729 * attempt)
        candidate = _build_r_allocation(hint, feasibility_status, rng)
        total = _number_of_variables(candidate.admissible)
        gap = abs(total - target) / target
        # Prefer candidates the corpus sizing tolerance accepts (within 25% of
        # the target, or at most 50 variables for tiny targets), then the
        # smallest relative gap.
        score = (gap <= 0.25 || total <= 50 ? 0 : 1, gap)
        if score < best_score
            best_score = score
            best = candidate
        end
        gap <= 0.05 && break
        ratio = clamp((target / max(total, 1))^0.25, 0.6, 1.6)
        next_hint = round(Int, hint * ratio)
        next_hint == hint && (next_hint += total < target ? 1 : -1)
        hint = clamp(next_hint, 3, 70)
    end
    return best::RAllocationHubProblem
end

"""
    build_model(prob::RAllocationHubProblem)

Build the four-index path-flow model with r-allocation linking. Deterministic.

Differences from `build_model(::PHubMedianProblem)`:
- allocation rows: `sum_{k in A_i} z_ik == r`
- every open hub allocates its own node to itself: `z_kk == y_k`
- path-to-allocation linking uses inequalities (a pair uses at most one of the
  origin's / destination's r hubs):
  `sum_m x_ikmj <= w_ij * z_ik` and `sum_k x_ikmj <= w_ij * z_jm`
"""
function build_model(prob::RAllocationHubProblem)
    model = Model()
    n = prob.n_nodes
    A = prob.admissible

    paths = NTuple{4,Int}[]
    for i in 1:n, j in (i + 1):n, k in A[i], m in A[j]
        push!(paths, (i, j, k, m))
    end
    allocations = NTuple{2,Int}[]
    for i in 1:n, k in A[i]
        push!(allocations, (i, k))
    end
    hub_candidates = sort!(collect(union(A...)))

    @variable(model, x[paths] >= 0)
    @variable(model, z[allocations], Bin)
    @variable(model, y[hub_candidates], Bin)

    by_pair = Dict{NTuple{2,Int},Vector{NTuple{4,Int}}}()
    by_first_hub = Dict{NTuple{3,Int},Vector{NTuple{4,Int}}}()
    by_last_hub = Dict{NTuple{3,Int},Vector{NTuple{4,Int}}}()
    for path in paths
        i, j, k, m = path
        push!(get!(by_pair, (i, j), NTuple{4,Int}[]), path)
        push!(get!(by_first_hub, (i, j, k), NTuple{4,Int}[]), path)
        push!(get!(by_last_hub, (i, j, m), NTuple{4,Int}[]), path)
    end
    empty_set = NTuple{4,Int}[]

    path_cost(path::NTuple{4,Int}) =
        prob.chi * prob.cost[path[1], path[3]] +
        prob.alpha * prob.cost[path[3], path[4]] +
        prob.delta * prob.cost[path[4], path[2]]

    @objective(model, Min, sum(path_cost(path) * x[path] for path in paths))

    for i in 1:n, j in (i + 1):n
        w = prob.flow[i, j]
        @constraint(model, sum(x[path] for path in get(by_pair, (i, j), empty_set)) == w)
        for k in A[i]
            @constraint(model,
                sum(x[path] for path in get(by_first_hub, (i, j, k), empty_set)) <=
                w * z[(i, k)])
        end
        for m in A[j]
            @constraint(model,
                sum(x[path] for path in get(by_last_hub, (i, j, m), empty_set)) <=
                w * z[(j, m)])
        end
    end

    for i in 1:n
        @constraint(model, sum(z[(i, k)] for k in A[i]) == prob.r)
    end
    for (i, k) in allocations
        @constraint(model, z[(i, k)] <= y[k])
    end
    for k in hub_candidates
        @constraint(model, z[(k, k)] == y[k])
    end
    @constraint(model, sum(y[k] for k in hub_candidates) == prob.p)

    return model
end

register_variant(
    :hub_location,
    :r_allocation,
    RAllocationHubProblem,
    "Uncapacitated r-allocation p-hub median with reach windows: every node keeps r primary/backup hubs (four-index path flows)",
)
