using JuMP
using Random
using Distributions

"""
Planted feasible allocation: the open hub set, each node's hub, and the reach
window that makes the assignment admissible.
"""
struct HubAssignmentWitness
    hubs::Vector{Int}
    assignment::Vector{Int}
    reach::Float64
end

"""
Relaxation-proof infeasibility certificate for the `p`-constrained variants:
`length(groups) == p + 1` node groups whose admissible hub sets are pairwise
disjoint. Every group's nodes must be served by hubs inside the group's own
window, so any feasible solution needs at least one open hub per group - more
than the `p` allowed. The argument uses only the disaggregated linking rows and
the exact-`p` row, so it refutes the LP relaxation as well.
"""
struct DisjointRegionCertificate
    groups::Vector{Vector{Int}}
    p::Int
end

"""
    PHubMedianProblem <: ProblemGenerator

Generator for the classical **uncapacitated single-allocation p-hub median
problem** (USA*pHMP; O'Kelly 1987, Campbell 1994) with allocation reach
windows.

# Overview

`p` of `n` cities are opened as hubs and every other city is allocated to a
single hub. Flow from `i` to `j` travels `i -> k -> m -> j`, paying `chi * c_ik`
to collect, `alpha * c_km` on the (discounted) inter-hub leg and `delta * c_mj`
to distribute. Reach windows (`reach`) forbid allocating a node to a hub
farther than `reach` away, modelling feeder-range / catchment restrictions.

The model is the tight four-index path-flow linearisation of Skorin-Kapov,
Skorin-Kapov & O'Kelly (1996): path variables `x_ikmj` tied to the allocation
variables by disaggregated equalities. Flows and costs are symmetrised (CAB
airline-passenger convention), so each *unordered* pair is one commodity. With
`relax_integer=true` (the package default) this yields the famously tight SKO
LP relaxation of the p-hub median problem.

# Fields
- `n_nodes::Int`: number of cities (nodes and hub candidates)
- `p::Int`: number of hubs to open (exactly)
- `chi::Float64`: collection cost multiplier (1.0 on the CAB grid)
- `alpha::Float64`: inter-hub discount factor, sampled in `[0.2, 0.8]` (CAB grid)
- `delta::Float64`: distribution cost multiplier (1.0 on the CAB grid)
- `locations::Vector{Tuple{Float64,Float64}}`: city coordinates
- `dist::Matrix{Float64}`: Euclidean distances (define the reach windows)
- `cost::Matrix{Float64}`: symmetric CAB-style network costs (detour-perturbed)
- `flow::Matrix{Float64}`: symmetric origin-destination volumes, zero diagonal
- `reach::Float64`: allocation reach window
- `admissible::Vector{Vector{Int}}`: `A_i`, hub candidates within reach of `i`
- `feasible_witness::Union{Nothing,HubAssignmentWitness}`: planted solution
- `infeasibility_certificate::Union{Nothing,DisjointRegionCertificate}`
- `feasibility_status::FeasibilityStatus`
"""
struct PHubMedianProblem <: ProblemGenerator
    n_nodes::Int
    p::Int
    chi::Float64
    alpha::Float64
    delta::Float64
    locations::Vector{Tuple{Float64,Float64}}
    dist::Matrix{Float64}
    cost::Matrix{Float64}
    flow::Matrix{Float64}
    reach::Float64
    admissible::Vector{Vector{Int}}
    feasible_witness::Union{Nothing,HubAssignmentWitness}
    infeasibility_certificate::Union{Nothing,DisjointRegionCertificate}
    feasibility_status::FeasibilityStatus
end

_number_of_variables(admissible::Vector{Vector{Int}}) =
    sum(length(admissible[i]) * length(admissible[j])
        for i in 1:length(admissible) for j in (i + 1):length(admissible);
        init=0) +
    sum(length(a) for a in admissible; init=0) +
    length(union(admissible...))

"""
    _hub_cover_hubs(dist, p, r) -> (hubs, radius)

Choose `p` hub candidates minimising the maximum distance from any node to its
`r`-th nearest chosen hub (`r = 1` for the p-hub median). Exhaustive over
subsets when `binomial(n, p)` is small; farthest-first traversal otherwise
(the classic 2-approximation for p-center).
"""
function _hub_cover_hubs(dist::Matrix{Float64}, p::Int, r::Int)
    n = size(dist, 1)
    r = clamp(r, 1, p)
    radius_of(hubs::Vector{Int}) =
        maximum(sort([dist[i, h] for h in hubs])[r] for i in 1:n)

    binom = 1.0
    for t in 0:(p - 1)
        binom *= (n - t) / (t + 1)
    end
    if binom <= 30_000.0 && n <= 26
        best_hubs = collect(1:p)
        best_radius = Inf
        stack = [Int[]]
        while !isempty(stack)
            current = pop!(stack)
            if length(current) == p
                radius = radius_of(current)
                if radius < best_radius
                    best_radius = radius
                    best_hubs = copy(current)
                end
                continue
            end
            need = p - length(current)
            for k in (isempty(current) ? 1 : last(current) + 1):(n - need + 1)
                push!(stack, vcat(current, [k]))
            end
        end
        return best_hubs, best_radius
    end

    # Farthest-first: repeatedly add the node whose r-th nearest chosen hub
    # is farthest away.
    hubs = Int[]
    nth = fill(Inf, n)
    while length(hubs) < p
        k = argmax(nth)
        push!(hubs, k)
        for i in 1:n
            ranks = sort([dist[i, h] for h in hubs])
            nth[i] = ranks[min(r, length(ranks))]
        end
    end
    return sort!(hubs), radius_of(hubs)
end

"""
    _hub_ring_centers(rng, q) -> Vector{Tuple{Float64,Float64}}

`q` well-separated group centers: jittered points on a circle of radius 33
inside the 100x100 region, guaranteeing pairwise separation of at least
`66 * sin(pi / q) - jitter` (>= 15 for the q <= 8 used here).
"""
function _hub_ring_centers(rng::AbstractRNG, q::Int)
    angle0 = rand(rng, Uniform(0.0, 2.0 * pi))
    centers = Tuple{Float64,Float64}[]
    for l in 1:q
        theta = angle0 + 2.0 * pi * (l - 1) / q + rand(rng, Uniform(-0.04, 0.04))
        push!(centers, (50.0 + 33.0 * cos(theta), 50.0 + 33.0 * sin(theta)))
    end
    return centers
end

"""
Build a complete PHubMedianProblem for a node-count hint. All randomness lives
here and in the helpers it calls; `build_model` is deterministic.
"""
function _build_p_hub_median(n_nodes::Int, feasibility_status::FeasibilityStatus,
                             rng::AbstractRNG)
    n = n_nodes
    p = clamp(round(Int, n / 4) + rand(rng, 0:1), 2, min(8, n - 1))

    if feasibility_status == infeasible
        # p + 1 mutually unreachable island groups: every group needs its own
        # hub, but only p may open. Disjoint admissible sets refute the LP
        # relaxation (see DisjointRegionCertificate).
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
        reach = 0.40 * min_sep
        certificate = DisjointRegionCertificate(groups, p)
        hubs = Int[]
        assignment = Int[]
    else
        shape = rand(rng, (:clustered, :corridor, :archipelago))
        locations = _hub_city_locations(rng, n, shape)
        dist = _hub_distance_matrix(locations)
        hubs, cover = _hub_cover_hubs(dist, p, 1)
        assignment = _hub_nearest_assignment(dist, hubs)
        reach = feasibility_status == feasible ?
                cover * rand(rng, Uniform(1.01, 1.08)) :
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

    witness = feasibility_status == feasible ?
              HubAssignmentWitness(hubs, assignment, reach) : nothing
    return PHubMedianProblem(n, p, 1.0, rand(rng, Uniform(0.2, 0.8)), 1.0,
                             locations, dist, cost, flow, reach, admissible,
                             witness, certificate, feasibility_status)
end

"""
    PHubMedianProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a p-hub median instance targeting `target_variables` variables.

# Variable count

With `A_i` the admissible hub list of node `i`, the model has one `x_ikmj` per
unordered pair and admissible hub pair, one `z_ik` per admissible allocation
and one `y_k` per reachable candidate:

    vars = sum_{i<j} |A_i| * |A_j| + sum_i |A_i| + |union_i A_i|

An iterative re-sizing loop adjusts the node-count hint so this lands close to
the target (within a few percent for most requests).

# Feasibility (relaxation-aware)
- `feasible`: hubs are placed to minimise the covering radius and the reach
  window is set just above it, so the planted assignment is admissible for the
  MIP and its LP relaxation (`feasible_witness`).
- `infeasible`: `p + 1` island groups with pairwise disjoint admissible sets
  (`DisjointRegionCertificate`); the exact-`p` row conflicts with the
  disaggregated linking rows already in the relaxation.
- `unknown`: the reach window is sampled around the covering radius (from 0.8x
  to 1.25x), leaving whether `p` hubs can serve every node genuinely
  undecided.
"""
function PHubMedianProblem(target_variables::Int,
                           feasibility_status::FeasibilityStatus, seed::Int)
    target = max(target_variables, 1)
    hint = clamp(round(Int, 2.0 * target^0.25), 3, 70)
    best = nothing
    best_score = (1, Inf)
    for attempt in 1:20
        rng = MersenneTwister(seed + 7919 * attempt)
        candidate = _build_p_hub_median(hint, feasibility_status, rng)
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
    return best::PHubMedianProblem
end

"""
    build_model(prob::PHubMedianProblem)

Build the four-index path-flow model. Deterministic - uses only struct fields.

# Model
Variables:
- `x[(i,j,k,m)] >= 0`: volume of pair `(i,j)` (unordered, `i < j`) routed
  `i -> k -> m -> j` with `k in A_i`, `m in A_j`
- `z[(i,k)] in {0,1}`: node `i` allocated to hub `k` (`k in A_i`)
- `y[k] in {0,1}`: candidate `k` opened as a hub (`k` reachable by someone)

Objective: `sum (chi*c_ik + alpha*c_km + delta*c_mj) * x_ikmj`
(`c_kk = 0`, so a single-hub path `i -> k -> k -> j` pays no transfer leg).

Constraints:
- demand: `sum_{k,m} x_ikmj == w_ij` for each unordered pair
- allocation equalities on both disaggregated sides:
  `sum_m x_ikmj == w_ij * z_ik` and `sum_k x_ikmj == w_ij * z_jm`
- opening: `z_ik <= y_k`
- hub self-allocation: `z_kk == y_k`
- exact p: `sum_k y_k == p`
"""
function build_model(prob::PHubMedianProblem)
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
                sum(x[path] for path in get(by_first_hub, (i, j, k), empty_set)) ==
                w * z[(i, k)])
        end
        for m in A[j]
            @constraint(model,
                sum(x[path] for path in get(by_last_hub, (i, j, m), empty_set)) ==
                w * z[(j, m)])
        end
    end

    for (i, k) in allocations
        @constraint(model, z[(i, k)] <= y[k])
    end
    # In the classical single-allocation formulation, opening candidate k and
    # assigning node k to itself are the same decision.  Keeping a separate y
    # variable is convenient for the sparse reach-window representation, but
    # the diagonal equality is needed to preserve that semantics and the SKO
    # relaxation.
    for k in hub_candidates
        @constraint(model, z[(k, k)] == y[k])
    end

    @constraint(model, sum(y[k] for k in hub_candidates) == prob.p)

    return model
end

register_variant(
    :hub_location,
    :p_hub_median,
    PHubMedianProblem,
    "Uncapacitated single-allocation p-hub median with reach windows (tight four-index path-flow formulation, CAB airline conventions)",
    default=true,
)
