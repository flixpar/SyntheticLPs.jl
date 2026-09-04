using JuMP
using Random
using Distributions

"""
Planted feasible solution for the multiple-allocation variants: an open hub
subset whose reach windows cover every node, and the reach window used.
"""
struct HubCoverWitness
    open_hubs::Vector{Int}
    reach::Float64
end

"""
Relaxation-proof infeasibility certificate for budget-constrained covering:
`length(groups)` node groups whose admissible hub sets are pairwise disjoint,
the minimum cost `groups * min_k f_k` of serving them (each group forces at
least one open hub, so `sum_k f_k y_k >= groups * min_k f_k`), and the budget
that was actually set below that minimum. The supply rows plus the
disaggregated linking rows force `sum_{k in A_i} y_k >= 1` for every node,
hence `sum_k y_k >= groups` - already in the LP relaxation, where the budget
row contradicts it.
"""
struct BudgetCoverCertificate
    groups::Vector{Vector{Int}}
    minimum_fixed_cost::Float64
    budget::Float64
end

"""
    MultipleAllocationHubProblem <: ProblemGenerator

Generator for the **fixed-charge multiple-allocation hub location problem**
with reach windows (Campbell 1994; the multiple-allocation counterpart of
O'Kelly 1992), in the per-destination multicommodity flow form of the
efficient flow models studied by Brimberg et al. (2021).

# Overview

Any city may use *any* open hub for each of its origin-destination pairs
(multiple allocation, the operating mode of less-than-truckload carriers and
parcel networks: freight is hauled to whichever nearby consolidation terminal
fits the lane). Opening hubs costs `f_k`, and a budget caps opening spending.
Reach windows restrict collection to hubs within `reach` of the origin and
distribution to hubs within `reach` of the destination, modelling feeder legs.

Flow destined for `j` is routed through a hub layer: collection arcs `i -> k`,
discounted transfer arcs `k -> m`, and a delivery arc `k -> j`. Costs are
metric and the discount is uniform, so an optimal path visits at most two hubs
(routing through extra hubs can never pay less) and the flow model is an exact
formulation.

# Data conventions (Australia Post)

Asymmetric flows with heavy right skew, and the published AP cost parameters
`chi = 3` (collection), `alpha = 0.75` (transfer), `delta = 2` (distribution),
each with small instance-level jitter.

# Fields

  - `n_nodes::Int`: number of cities
  - `hubs::Vector{Int}`: hub candidate sites (biased toward the largest cities)
  - `chi`, `alpha`, `delta`: leg cost multipliers
  - `locations`, `dist::Matrix{Float64}`: Euclidean, metric
  - `flow::Matrix{Float64}`: asymmetric O-D volumes, zero diagonal
  - `reach::Float64`: feeder reach window
  - `admissible::Vector{Vector{Int}}`: `A_i`, candidates within reach of `i`
  - `fixed_cost::Vector{Float64}`: `f_k` aligned with `hubs`
  - `budget::Float64`: opening budget
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct MultipleAllocationHubProblem <: ProblemGenerator
    n_nodes::Int
    hubs::Vector{Int}
    chi::Float64
    alpha::Float64
    delta::Float64
    locations::Vector{Tuple{Float64, Float64}}
    dist::Matrix{Float64}
    flow::Matrix{Float64}
    reach::Float64
    admissible::Vector{Vector{Int}}
    fixed_cost::Vector{Float64}
    budget::Float64
    feasible_witness::Union{Nothing, HubCoverWitness}
    infeasibility_certificate::Union{Nothing, BudgetCoverCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _hub_candidate_sites(rng, n, populations, h) -> Vector{Int}

Pick `h` hub candidate cities, biased toward the largest populations (major
cities host consolidation terminals) with enough randomness that mid-size
cities occasionally qualify.
"""
function _hub_candidate_sites(rng::AbstractRNG, n::Int, populations::Vector{Float64}, h::Int)
    score = populations .* exp.(rand(rng, Normal(0.0, 0.8), n))
    order = sortperm(score; rev=true)
    return sort(order[1:h])
end

"""
    _hub_greedy_cover(admissible, n) -> Vector{Int}

Greedy set cover of the nodes by hub reach sets: repeatedly open the hub whose
window covers the most still-uncovered nodes. Returns chosen node indices.
"""
function _hub_greedy_cover(admissible::Vector{Vector{Int}}, n::Int)
    covered = falses(n)
    chosen = Int[]
    while !all(covered)
        best_k, best_gain = 0, 0
        for k in 1:n
            k in chosen && continue
            gain = sum(1 for i in 1:n if !covered[i] && (k in admissible[i]); init=0)
            if gain > best_gain
                best_gain, best_k = gain, k
            end
        end
        best_k == 0 && break
        push!(chosen, best_k)
        for i in 1:n
            k = best_k
            k in admissible[i] && (covered[i] = true)
        end
    end
    return chosen
end

function _build_multiple_allocation(
    n_nodes::Int, n_hubs::Int, feasibility_status::FeasibilityStatus, rng::AbstractRNG
)
    n = n_nodes
    h = clamp(n_hubs, 2, n)

    populations = _hub_populations(rng, n)
    hubs = _hub_candidate_sites(rng, n, populations, h)

    if feasibility_status == infeasible
        # Disjoint island groups over the candidate set: covering every group
        # needs at least one hub each, and the budget is set below that cost.
        q = clamp(round(Int, h / 3), 2, 4)
        q = min(q, h, n)
        centers = _hub_ring_centers(rng, q)
        node_group = vcat(collect(1:q), rand(rng, 1:q, max(0, n - q)))
        min_sep = minimum(
            hypot(centers[a][1] - centers[b][1], centers[a][2] - centers[b][2]) for a in 1:q for
            b in (a + 1):q
        )
        spread = 0.15 * min_sep
        locations = [
            if g <= q
                centers[g]
            else
                (
                    clamp(
                        centers[node_group[g]][1] + rand(rng, Uniform(-spread, spread)), 0.0, 100.0
                    ),
                    clamp(
                        centers[node_group[g]][2] + rand(rng, Uniform(-spread, spread)), 0.0, 100.0
                    ),
                )
            end for g in 1:n
        ]
        dist = _hub_distance_matrix(locations)
        # Nodes 1..q sit exactly on their group centers; keep them (plus other
        # drawn candidates) as the hub set so each group owns at least one
        # reachable candidate.
        hub_set = sort(unique(vcat(collect(1:q), rand(rng, hubs, max(0, h - q)))))
        reach = 0.40 * min_sep
        admissible = _hub_reach_admissible(dist, reach; candidates=hub_set)
        groups = [Int[] for _ in 1:q]
        for g in 1:n
            push!(groups[node_group[g]], g)
        end
    else
        shape = rand(rng, (:clustered, :corridor, :archipelago))
        locations = _hub_city_locations(rng, n, shape)
        dist = _hub_distance_matrix(locations)
        # Smallest reach at which every node still sees its nearest candidate
        # (no empty windows), then sampled just above it.
        cover_reach = maximum(minimum(dist[i, k] for k in hubs) for i in 1:n)
        reach =
            cover_reach *
            rand(rng, feasibility_status == feasible ? Uniform(1.05, 1.2) : Uniform(0.99, 1.1))
        admissible = _hub_reach_admissible(dist, reach; candidates=hubs)
        hub_set = copy(hubs)
        groups = Vector{Int}[]
    end

    decay = rand(rng, Uniform(0.5, 1.1))
    noise = rand(rng, Uniform(0.7, 1.2))
    flow = _hub_gravity_flows(
        rng, n, populations, dist, decay, noise; symmetric=false, scale=rand(rng, Uniform(0.5, 2.0))
    )
    chi = rand(rng, Uniform(2.7, 3.3))
    delta = rand(rng, Uniform(1.8, 2.2))
    alpha = rand(rng, Uniform(0.7, 0.8))

    # Fixed costs calibrated so hub opening trades off against transport.
    flow_cost_scale = sum(flow[i, j] * dist[i, j] for i in 1:n, j in 1:n if i != j)
    base_fixed = flow_cost_scale / max(length(hub_set), 1) * rand(rng, Uniform(0.3, 0.9))
    fixed_cost = [base_fixed * exp(rand(rng, Uniform(log(0.8), log(1.25)))) for _ in hub_set]

    witness = nothing
    certificate = nothing
    budget = 0.0
    if feasibility_status == infeasible
        minimum_cost = q * minimum(fixed_cost)
        budget = minimum_cost * rand(rng, Uniform(0.75, 0.95))
        certificate = BudgetCoverCertificate(groups, minimum_cost, budget)
    else
        cover = _hub_greedy_cover(admissible, n)
        position = Dict(k => t for (t, k) in enumerate(hub_set))
        cover_cost = sum(fixed_cost[position[k]] for k in cover; init=0.0)
        total_cost = sum(fixed_cost)
        factor = feasibility_status == feasible ? Uniform(1.05, 1.35) : Uniform(0.8, 1.15)
        budget = min(total_cost, cover_cost * rand(rng, factor))
        witness = feasibility_status == feasible ? HubCoverWitness(cover, reach) : nothing
    end

    return MultipleAllocationHubProblem(
        n,
        hub_set,
        chi,
        alpha,
        delta,
        locations,
        dist,
        flow,
        reach,
        admissible,
        fixed_cost,
        budget,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    MultipleAllocationHubProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a fixed-charge multiple-allocation hub location instance.

# Variable count

Per destination `j`: one collection variable per (origin, admissible hub)
pair, one transfer variable per ordered hub pair (the inter-hub network is
complete among open hubs), and one delivery variable per admissible hub; plus
one opening variable per candidate:

    vars = sum_j [ sum_{i != j} |A_i| + h * (h - 1) + |A_j| ] + h

An iterative re-sizing loop adjusts the node-count hint (the candidate count
follows it) to land near the target.

# Feasibility (relaxation-aware)

  - `feasible`: the reach window admits a greedy candidate cover and the budget
    covers its cost (`HubCoverWitness`); multiple allocation can then route
    every pair through any single open hub.
  - `infeasible`: disjoint island groups with a budget below
    `groups * min_k f_k` (`BudgetCoverCertificate`) - the budget row conflicts
    with the covering forced by the supply and linking rows in the relaxation.
  - `unknown`: the budget is sampled around the greedy cover cost, which may or
    may not be enough once cheaper (including fractional) covers exist, leaving
    feasibility undecided.
"""
function MultipleAllocationHubProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target = max(target_variables, 1)
    hint_n = clamp(round(Int, 1.4 * target^(1 / 3)), 4, 90)
    hint_h = clamp(round(Int, hint_n / 3), 2, hint_n)
    best = nothing
    best_score = (1, Inf)
    for attempt in 1:18
        rng = MersenneTwister(seed + 15485863 * attempt)
        candidate = _build_multiple_allocation(hint_n, hint_h, feasibility_status, rng)
        n, h = candidate.n_nodes, length(candidate.hubs)
        A = candidate.admissible
        total =
            sum(
                sum(length(A[i]) for i in 1:n if i != j) + h * (h - 1) + length(A[j]) for j in 1:n
            ) + h
        gap = abs(total - target) / target
        score = (gap <= 0.25 || total <= 50 ? 0 : 1, gap)
        if score < best_score
            best_score = score
            best = candidate
        end
        gap <= 0.05 && break
        ratio = clamp((target / max(total, 1))^(1 / 3), 0.6, 1.6)
        next_n = round(Int, hint_n * ratio)
        next_h = round(Int, hint_h * ratio)
        if next_n == hint_n && next_h == hint_h
            step = total < target ? 1 : -1
            next_n += step
            next_h += step
        end
        hint_n = clamp(next_n, 4, 90)
        hint_h = clamp(next_h, 2, hint_n)
    end
    return best::MultipleAllocationHubProblem
end

"""
    build_model(prob::MultipleAllocationHubProblem)

Build the per-destination flow model. Deterministic - uses only struct fields.

Variables (for each destination `j`; `H` = candidates, `h = |H|`):

  - `u[(j,i,k)] >= 0`, `k in A_i`: collection arc `i -> k` (`i != j`)
  - `v[(j,k,m)] >= 0`, `k != m in H`: discounted transfer arc
  - `d[(j,k)] >= 0`, `k in A_j`: delivery arc `k -> j`
  - `y[k] in {0,1}`, `k in H`: open hub `k`

Objective: `sum_k f_k y_k + sum_j [ chi*d_ik*u + alpha*d_km*v + delta*d_kj*d ]`.

Constraints:

  - supply: `sum_{k in A_i} u == w_ij` for each origin `i != j`
  - delivery: `sum_{k in A_j} d == W^j` (the total volume destined for `j`)
  - hub conservation: inflow (collections plus transfers in) equals outflow
    (transfers out plus, for `k in A_j`, deliveries)
  - disaggregated linking to open hubs on both arc endpoints:
    `u <= w_ij y_k`, `v <= W^j y_k`, `v <= W^j y_m`, `d <= W^j y_k`
  - budget: `sum_k f_k y_k <= budget`
"""
function build_model(prob::MultipleAllocationHubProblem)
    model = Model()
    n = prob.n_nodes
    H = prob.hubs
    h = length(H)
    A = prob.admissible
    in_A = [Set(a) for a in A]

    collections = NTuple{3, Int}[]      # (j, i, k)
    transfers = NTuple{3, Int}[]        # (j, k, m)
    deliveries = NTuple{2, Int}[]       # (j, k)
    for j in 1:n
        for i in 1:n, k in A[i]
            i == j && continue
            push!(collections, (j, i, k))
        end
        for k in H, m in H
            k == m && continue
            push!(transfers, (j, k, m))
        end
        for k in A[j]
            push!(deliveries, (j, k))
        end
    end

    @variable(model, u[collections] >= 0)
    @variable(model, v[transfers] >= 0)
    @variable(model, d[deliveries] >= 0)
    @variable(model, y[H], Bin)

    position = Dict(k => t for (t, k) in enumerate(H))
    fixed_of(k) = prob.fixed_cost[position[k]]

    @objective(
        model,
        Min,
        sum(fixed_of(k) * y[k] for k in H) +
            sum(prob.chi * prob.dist[i, k] * u[(j, i, k)] for (j, i, k) in collections) +
            sum(prob.alpha * prob.dist[k, m] * v[(j, k, m)] for (j, k, m) in transfers) +
            sum(prob.delta * prob.dist[k, j] * d[(j, k)] for (j, k) in deliveries)
    )

    for j in 1:n
        w_j = sum(prob.flow[i, j] for i in 1:n if i != j)

        for i in 1:n
            i == j && continue
            @constraint(model, sum(u[(j, i, k)] for k in A[i]) == prob.flow[i, j])
        end
        @constraint(model, sum(d[(j, k)] for k in A[j]) == w_j)

        for k in H
            inflow = sum(u[(j, i, k)] for i in 1:n if i != j && k in in_A[i]; init=0.0)
            out_transfer = sum(v[(j, k, m)] for m in H if m != k; init=0.0)
            in_transfer = sum(v[(j, m, k)] for m in H if m != k; init=0.0)
            delivered = k in in_A[j] ? d[(j, k)] : 0.0
            @constraint(model, inflow + in_transfer == out_transfer + delivered)
        end

        for i in 1:n, k in A[i]
            i == j && continue
            @constraint(model, u[(j, i, k)] <= prob.flow[i, j] * y[k])
        end
        for k in H, m in H
            k == m && continue
            @constraint(model, v[(j, k, m)] <= w_j * y[k])
            @constraint(model, v[(j, k, m)] <= w_j * y[m])
        end
        for k in A[j]
            @constraint(model, d[(j, k)] <= w_j * y[k])
        end
    end

    @constraint(model, sum(fixed_of(k) * y[k] for k in H) <= prob.budget)

    return model
end

register_variant(
    :hub_location,
    :multiple_allocation,
    MultipleAllocationHubProblem,
    "Fixed-charge multiple-allocation hub location with feeder reach windows and an opening budget (per-destination flow formulation, AP postal cost conventions)",
)
