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
Relaxation-proof infeasibility certificate for r-allocation: node `groups`
whose admissible hub windows are pairwise disjoint (each window lies inside its
own group). For any node `i` of group `g`, the allocation row
`sum_{k in A_i} z_ik = r` and the linking rows `z_ik <= y_k` give
`sum_{k in g} y_k >= r`, so every group needs `r` open hubs and
`sum_k y_k >= r * length(groups) > p` - contradicting the exact-`p` row, in the
LP relaxation as well. Unlike the one-hub-per-group argument of
`DisjointRegionCertificate`, the deficit comes from the backup requirement
itself (`length(groups) = floor(p / r) + 1`).
"""
struct BackupRegionCertificate
    groups::Vector{Vector{Int}}
    p::Int
    r::Int
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
    locations::Vector{Tuple{Float64, Float64}}
    dist::Matrix{Float64}
    cost::Matrix{Float64}
    flow::Matrix{Float64}
    reach::Float64
    admissible::Vector{Vector{Int}}
    feasible_witness::Union{Nothing, HubBackupWitness}
    infeasibility_certificate::Union{Nothing, BackupRegionCertificate}
    feasibility_status::FeasibilityStatus
end

function _build_r_allocation(n_nodes::Int, feasibility_status::FeasibilityStatus, rng::AbstractRNG)
    n = n_nodes
    p = clamp(round(Int, n / 3) + rand(rng, 0:1), 3, min(8, n - 1))
    # Infeasible instances draw p anywhere up to that level: fewer hubs mean
    # fewer, larger island regions, whose windows (and hence sizes) vary.
    feasibility_status == infeasible && (p = rand(rng, min(3, p):p))
    # Keep r below the node count so windows stay small at tiny sizes.
    r = max(2, min(p, n - 2, 2 + (rand(rng) < 0.25 ? 1 : 0)))

    if feasibility_status == infeasible
        # q = floor(p/r) + 1 mutually unreachable island regions. Every node
        # must keep r hubs inside its own region, so each region needs r open
        # hubs: q*r > p (see BackupRegionCertificate). Regions get at least
        # r + 2 cities so no node's window is exactly r wide (which would let
        # presolve force hubs open by bounds alone and refute the instance
        # without simplex work).
        q = min(fld(p, r) + 1, n)
        locations, groups, min_sep = _hub_island_geography(rng, n, q; min_members=r + 2)
        dist = _hub_distance_matrix(locations)
        # In-region distances stay below 0.34 min_sep and cross-region ones
        # above 0.66 min_sep, so any reach in between keeps windows inside
        # their region. A reach drawn inside the region diameter (windows are
        # then padded to r + 2 in-region candidates below) and a random subset
        # of hub-capable cities per region make the window sizes vary between
        # draws - the region sizes alone barely change with the seed - so the
        # sizing loop can land close to the target.
        reach = 0.40 * min_sep * rand(rng, Uniform(0.55, 1.0))
        candidates = sort!(
            reduce(vcat, [shuffle(rng, g)[1:rand(rng, min(r + 2, length(g)):length(g))] for g in groups])
        )
        certificate = BackupRegionCertificate(groups, p, r)
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
        reach = if feasibility_status == feasible
            cover * rand(rng, Uniform(1.005, 1.1))
        else
            cover * rand(rng, Uniform(0.8, 1.25))
        end
        # Every city keeps at least r + 1 candidates (itself and its r nearest
        # neighbours): a window of at most r cities forces its hubs open, which
        # lets presolve settle an unknown instance without any simplex work.
        floor_reach = maximum(sort(dist[i, :])[min(r + 1, n)] for i in 1:n)
        reach = max(reach, floor_reach * (1 + 1e-9))
        candidates = collect(1:n)
        certificate = nothing
    end

    cost = _hub_detour_cost_matrix(rng, dist, 1.0, 1.35)
    populations = _hub_populations(rng, n)
    decay = rand(rng, Uniform(0.4, 1.0))
    noise = rand(rng, Uniform(0.6, 1.1))
    flow = _hub_gravity_flows(
        rng,
        n,
        populations,
        dist,
        decay,
        noise;
        symmetric=true,
        scale=rand(rng, Uniform(20.0, 90.0)),
    )
    admissible = _hub_reach_admissible(dist, reach; candidates=candidates)
    if feasibility_status == infeasible
        # Pad every window with the nearest hub-capable cities of its own
        # region up to r + 2, so no hub is forced open by bounds alone.
        for g in certificate.groups
            own = [k for k in candidates if k in g]
            for i in g
                for k in sort(own; by=k -> dist[i, k])
                    length(admissible[i]) >= min(r + 2, length(own)) && break
                    k in admissible[i] || push!(admissible[i], k)
                end
                sort!(admissible[i])
            end
        end
    end

    # Feasible requests must give every node r admissible candidates.
    if feasibility_status == feasible
        for i in 1:n
            while length(admissible[i]) < r
                push!(admissible[i], sort(1:n; by=k -> dist[i, k])[length(admissible[i]) + 1])
            end
            sort!(admissible[i])
        end
        reach = maximum(maximum(dist[i, k] for k in admissible[i]) for i in 1:n)
    end

    witness = feasibility_status == feasible ? HubBackupWitness(hubs, assignments, reach) : nothing
    return RAllocationHubProblem(
        n,
        p,
        r,
        1.0,
        rand(rng, Uniform(0.2, 0.8)),
        1.0,
        locations,
        dist,
        cost,
        flow,
        reach,
        admissible,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    RAllocationHubProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an r-allocation p-hub median instance. The variable count matches
`PHubMedianProblem`:

    vars = sum_{i<j} |A_i| * |A_j| + sum_i |A_i| + |union_i A_i|

Feasibility:

  - `feasible`: a `p`-hub cover whose `r` nearest hubs lie within reach of every
    node (`HubBackupWitness`); feasible requests also guarantee `|A_i| >= r`.
  - `infeasible`: `floor(p/r) + 1` disjoint island regions, each needing `r`
    hubs of its own (`BackupRegionCertificate`). Every region has at least
    `r + 2` hub-capable cities (a random subset of its cities) and every
    window is padded to at least `r + 2` of them inside its own region (so for
    this status `reach` is a lower bound on the window, not its radius); the
    deficit is an aggregate over many allocation and linking rows rather than
    a bound that presolve propagates.

Sizing: up to 120 draws walk the node-count hint and keep the one closest to
the target (stopping within 2.5%); one node is a ~20% step near 1k variables,
so the per-draw variation of reach and windows is what fills the gaps.
  - `unknown`: reach sampled at 0.8-1.25x the cover radius.

For `feasible` and `unknown` the reach is floored so every node has at least
`r + 1` admissible hubs (itself and its `r` nearest neighbours); a window of at
most `r` cities would force its hubs open by bounds alone.
"""
function RAllocationHubProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target = max(target_variables, 1)
    hint = clamp(round(Int, 2.0 * target^0.25), 3, 70)
    best = nothing
    best_score = (1, Inf)
    # Each node adds about 2/n of the path count, so near 1k variables one
    # node is a ~20% step; the fresh draw per attempt (reach, geography) is
    # what fills the gaps, hence a generous attempt budget and a tight stop.
    for attempt in 1:120
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
        gap <= 0.025 && break
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

    paths = NTuple{4, Int}[]
    for i in 1:n, j in (i + 1):n, k in A[i], m in A[j]
        push!(paths, (i, j, k, m))
    end
    allocations = NTuple{2, Int}[]
    for i in 1:n, k in A[i]
        push!(allocations, (i, k))
    end
    hub_candidates = sort!(collect(union(A...)))

    @variable(model, x[paths] >= 0)
    @variable(model, z[allocations], Bin)
    @variable(model, y[hub_candidates], Bin)

    by_pair = Dict{NTuple{2, Int}, Vector{NTuple{4, Int}}}()
    by_first_hub = Dict{NTuple{3, Int}, Vector{NTuple{4, Int}}}()
    by_last_hub = Dict{NTuple{3, Int}, Vector{NTuple{4, Int}}}()
    for path in paths
        i, j, k, m = path
        push!(get!(by_pair, (i, j), NTuple{4, Int}[]), path)
        push!(get!(by_first_hub, (i, j, k), NTuple{4, Int}[]), path)
        push!(get!(by_last_hub, (i, j, m), NTuple{4, Int}[]), path)
    end
    empty_set = NTuple{4, Int}[]

    path_cost(path::NTuple{4, Int}) =
        prob.chi * prob.cost[path[1], path[3]] +
        prob.alpha * prob.cost[path[3], path[4]] +
        prob.delta * prob.cost[path[4], path[2]]

    @objective(model, Min, sum(path_cost(path) * x[path] for path in paths))

    for i in 1:n, j in (i + 1):n
        w = prob.flow[i, j]
        @constraint(model, sum(x[path] for path in get(by_pair, (i, j), empty_set)) == w)
        for k in A[i]
            @constraint(
                model,
                sum(x[path] for path in get(by_first_hub, (i, j, k), empty_set)) <= w * z[(i, k)]
            )
        end
        for m in A[j]
            @constraint(
                model,
                sum(x[path] for path in get(by_last_hub, (i, j, m), empty_set)) <= w * z[(j, m)]
            )
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
    "Uncapacitated r-allocation p-hub median with reach windows: every node keeps r primary/backup hubs (four-index path flows)";
    tags=[:location, :multicommodity, :dual_block_angular],
)
