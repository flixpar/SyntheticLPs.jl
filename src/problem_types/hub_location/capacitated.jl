using JuMP
using Random
using Distributions

"""
Planted feasible solution for capacitated single allocation: the open hubs and
each node's hub, chosen so hub collection inflow respects the capacities.
"""
struct HubCapacitatedWitness
    open_hubs::Vector{Int}
    assignment::Vector{Int}
end

"""
Relaxation-proof infeasibility certificate: total hub capacity strictly below
total flow. Summing the capacity rows `sum_i O_i z_ik <= Gamma_k y_k` over all
candidates gives `W = sum_i O_i <= sum_k Gamma_k y_k <= sum_k Gamma_k < W`
(the first equality is the single-allocation rows), a contradiction that holds
in the LP relaxation.
"""
struct CapacityShortfallCertificate
    total_flow::Float64
    total_capacity::Float64
end

"""
    CapacitatedHubLocationProblem <: ProblemGenerator

Generator for the **capacitated single-allocation hub location problem**
(CSAHLP; Ernst & Krishnamoorthy 1999, with the capacity rows as corrected by
Correia, Nickel & Saldanha-da-Gama 2010).

# Overview

Single allocation with fixed hub opening costs and a capacity `Gamma_k` on the
total flow *collected* at each open hub (the Australia Post convention: the
capacity files bound incoming commodities, including a node's own volume,
entering the sorting centre). Every node is allocated to exactly one open hub;
its whole outbound volume `O_i` enters the network there and its whole inbound
volume `D_i` is delivered from there. Capacity is the binding resource: the
trade-off is between many expensive hubs (balanced loads, short detours) and
few cheap ones.

Per-destination flow variables route each commodity through the hub layer
(collection, discounted transfer, delivery); allocation binaries couple a
node's whole outbound/inbound volume to its single hub. Costs are metric, so
optimal paths visit at most two hubs and the flow formulation is exact.

# Data conventions (Australia Post)
Asymmetric lognormal-skewed flows and the AP cost parameters `chi = 3`,
`alpha = 0.75`, `delta = 2` with instance-level jitter. The `profile` field
mirrors the AP `CapL` (loose) / `CapT` (tight) capacity files:
- `:loose` - total capacity well above total flow (hub count is economic);
- `:tight` - total capacity close to total flow (capacity binds).

# Fields
- `n_nodes`, `hubs::Vector{Int}` (candidates), `profile::Symbol`
- `chi`, `alpha`, `delta`: leg cost multipliers
- `locations`, `dist::Matrix{Float64}`, `flow::Matrix{Float64}` (asymmetric)
- `outvolume::Vector{Float64}`, `involume::Vector{Float64}`: `O_i`, `D_i`
- `fixed_cost::Vector{Float64}`, `capacity::Vector{Float64}` (aligned with `hubs`)
- `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct CapacitatedHubLocationProblem <: ProblemGenerator
    n_nodes::Int
    hubs::Vector{Int}
    profile::Symbol
    chi::Float64
    alpha::Float64
    delta::Float64
    locations::Vector{Tuple{Float64,Float64}}
    dist::Matrix{Float64}
    flow::Matrix{Float64}
    outvolume::Vector{Float64}
    involume::Vector{Float64}
    fixed_cost::Vector{Float64}
    capacity::Vector{Float64}
    feasible_witness::Union{Nothing,HubCapacitatedWitness}
    infeasibility_certificate::Union{Nothing,CapacityShortfallCertificate}
    feasibility_status::FeasibilityStatus
end

const _HUB_CAPACITY_PROFILES = (:loose, :tight)

function _build_capacitated(n_nodes::Int, n_hubs::Int,
                            feasibility_status::FeasibilityStatus,
                            rng::AbstractRNG)
    n = n_nodes
    h = clamp(n_hubs, 2, n)
    profile = rand(rng, _HUB_CAPACITY_PROFILES)

    populations = _hub_populations(rng, n)
    shape = rand(rng, (:clustered, :corridor, :archipelago))
    locations = _hub_city_locations(rng, n, shape)
    dist = _hub_distance_matrix(locations)
    hubs = _hub_candidate_sites(rng, n, populations, h)

    decay = rand(rng, Uniform(0.5, 1.1))
    noise = rand(rng, Uniform(0.7, 1.2))
    flow = _hub_gravity_flows(rng, n, populations, dist, decay, noise;
                              symmetric=false,
                              scale=rand(rng, Uniform(0.5, 2.0)))
    outvolume = vec(sum(flow; dims=2))
    involume = vec(sum(flow; dims=1))
    total_flow = sum(outvolume)

    chi = rand(rng, Uniform(2.7, 3.3))
    delta = rand(rng, Uniform(1.8, 2.2))
    alpha = rand(rng, Uniform(0.7, 0.8))

    # Fixed costs calibrated against the transport-cost scale.
    flow_cost_scale = sum(flow[i, j] * dist[i, j] for i in 1:n, j in 1:n if i != j)
    base_fixed = flow_cost_scale / h * rand(rng, Uniform(0.2, 0.7))
    fixed_cost = [base_fixed * exp(rand(rng, Uniform(log(0.8), log(1.25))))
                  for _ in hubs]

    witness = nothing
    certificate = nothing
    if feasibility_status == infeasible
        target = total_flow * rand(rng, Uniform(0.55, 0.92))
        capacity = _hub_split_capacity(rng, target, h)
        certificate = CapacityShortfallCertificate(total_flow, sum(capacity))
    else
        if feasibility_status == feasible
            target = total_flow *
                     (profile == :loose ? rand(rng, Uniform(1.35, 1.7)) :
                                         rand(rng, Uniform(1.15, 1.3)))
        else
            target = total_flow *
                     (profile == :loose ? rand(rng, Uniform(1.0, 1.25)) :
                                         rand(rng, Uniform(0.97, 1.06)))
        end
        capacity = _hub_split_capacity(rng, target, h)
        if feasibility_status == feasible
            # Make room for the largest single origin at the roomiest hub,
            # then scale up until the integral best-fit assignment succeeds.
            capacity[argmax(capacity)] = max(maximum(capacity),
                                             1.05 * maximum(outvolume))
            witness = _hub_capacitated_assignment(dist, hubs, outvolume, capacity)
            while witness === nothing
                capacity .*= 1.1
                witness = _hub_capacitated_assignment(dist, hubs, outvolume,
                                                       capacity)
            end
        end
    end

    return CapacitatedHubLocationProblem(n, hubs, profile, chi, alpha, delta,
                                         locations, dist, flow, outvolume,
                                         involume, fixed_cost, capacity,
                                         witness, certificate,
                                         feasibility_status)
end

"""
    _hub_split_capacity(rng, target, h) -> Vector{Float64}

Split a total capacity target across `h` hubs with lognormal variation
(sorting centres of different sizes), preserving the sum.
"""
function _hub_split_capacity(rng::AbstractRNG, target::Float64, h::Int)
    shares = exp.(rand(rng, Normal(0.0, 0.30), h))
    shares ./= sum(shares)
    return round.(target .* shares; digits=3)
end

"""
    _hub_capacitated_assignment(dist, hubs, outvolume, capacity)
    -> Union{Nothing,HubCapacitatedWitness}

Best-fit-by-distance assignment of every node to an open hub such that each
hub's collected outvolume respects its capacity: nodes are placed in
decreasing volume order, each to the nearest hub with enough residual room.
Every candidate is treated as open (opening is only an economic decision
here), so this always succeeds when total capacity suffices and the largest
origin fits somewhere; returns `nothing` otherwise so the caller can scale up.
"""
function _hub_capacitated_assignment(dist::Matrix{Float64}, hubs::Vector{Int},
                                     outvolume::Vector{Float64},
                                     capacity::Vector{Float64})
    n = length(outvolume)
    residual = copy(capacity)
    assignment = zeros(Int, n)
    for i in sortperm(outvolume; rev=true)
        order = sort(hubs; by=k -> dist[i, k])
        placed = false
        for k in order
            t = findfirst(==(k), hubs)
            if residual[t] >= outvolume[i]
                assignment[i] = k
                residual[t] -= outvolume[i]
                placed = true
                break
            end
        end
        placed || return nothing
    end
    return HubCapacitatedWitness(copy(hubs), assignment)
end

"""
    CapacitatedHubLocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a capacitated single-allocation hub location instance.

# Variable count

Per destination `j`: `h` collection variables per origin, `h * (h - 1)`
transfer variables and `h` delivery variables; plus `n * h` allocation
binaries and `h` opening binaries:

    vars = n * h * (n + h - 1) + h * (n + 1)

An iterative re-sizing loop adjusts the node-count hint to land near the
target.

# Feasibility (relaxation-aware)
- `feasible`: a verified best-fit assignment respects every hub capacity
  (`HubCapacitatedWitness`), with total capacity comfortably above total flow.
- `infeasible`: total capacity strictly below total flow
  (`CapacityShortfallCertificate`); summing the capacity rows against the
  single-allocation rows contradicts already in the relaxation.
- `unknown`: total capacity is sampled near total flow (tight profile) or
  moderately above it (loose profile), so capacity may or may not suffice.
"""
function CapacitatedHubLocationProblem(target_variables::Int,
                                       feasibility_status::FeasibilityStatus,
                                       seed::Int)
    target = max(target_variables, 1)
    hint_n = clamp(round(Int, 0.9 * target^(1 / 3)), 4, 80)
    hint_h = clamp(round(Int, hint_n / 3), 2, hint_n)
    best = nothing
    best_score = (1, Inf)
    for attempt in 1:18
        rng = MersenneTwister(seed + 32452843 * attempt)
        candidate = _build_capacitated(hint_n, hint_h, feasibility_status, rng)
        n, h = candidate.n_nodes, length(candidate.hubs)
        total = n * h * (n + h - 1) + h * (n + 1)
        gap = abs(total - target) / target
        score = (gap <= 0.25 || total <= 50 ? 0 : 1, gap)
        if score < best_score
            best_score = score
            best = candidate
        end
        gap <= 0.05 && break
        # The count grows like (n*h)*(n+h), i.e. cubically in a common scale.
        ratio = clamp((target / max(total, 1))^(1 / 3), 0.6, 1.6)
        next_n = round(Int, hint_n * ratio)
        next_h = round(Int, hint_h * ratio)
        if next_n == hint_n && next_h == hint_h
            step = total < target ? 1 : -1
            next_n += step
            next_h += step
        end
        hint_n = clamp(next_n, 4, 80)
        hint_h = clamp(next_h, 2, hint_n)
    end
    return best::CapacitatedHubLocationProblem
end

"""
    build_model(prob::CapacitatedHubLocationProblem)

Build the capacitated single-allocation flow model. Deterministic - uses only
struct fields.

Variables (for each destination `j`; `H` = candidates, `h = |H|`):
- `u[(j,i,k)] >= 0`, `k in H`: collection arc `i -> k` (`i != j`)
- `v[(j,k,m)] >= 0`, `k != m in H`: discounted transfer arc
- `d[(j,k)] >= 0`, `k in H`: delivery arc `k -> j`
- `z[(i,k)] in {0,1}`, `k in H`: node `i` allocated to hub `k`
- `y[k] in {0,1}`, `k in H`: open hub `k`

Objective: `sum_k f_k y_k + sum_j [ chi*d_ik*u + alpha*d_km*v + delta*d_kj*d ]`.

Constraints:
- supply / delivery / hub conservation (as in the multiple-allocation model,
  with every candidate admissible - no reach windows)
- single allocation: `sum_{k in H} z_ik == 1`
- allocation coupling: `sum_j u[(j,i,k)] <= O_i z_ik` and
  `d[(i,k)] <= D_i z_ik` (all of a node's volume uses its own hub)
- hub capacity: `sum_i O_i z_ik <= Gamma_k y_k` (collection inflow)
- linking: `u <= w_ij y_k`, `v <= W^j y_k` (both endpoints), `d <= W^j y_k`,
  `z_ik <= y_k`
"""
function build_model(prob::CapacitatedHubLocationProblem)
    model = Model()
    n = prob.n_nodes
    H = prob.hubs
    h = length(H)

    collections = NTuple{3,Int}[]      # (j, i, k)
    transfers = NTuple{3,Int}[]        # (j, k, m)
    deliveries = NTuple{2,Int}[]       # (j, k)
    for j in 1:n
        for i in 1:n, k in H
            i == j && continue
            push!(collections, (j, i, k))
        end
        for k in H, m in H
            k == m && continue
            push!(transfers, (j, k, m))
        end
        for k in H
            push!(deliveries, (j, k))
        end
    end
    allocations = NTuple{2,Int}[]
    for i in 1:n, k in H
        push!(allocations, (i, k))
    end

    @variable(model, u[collections] >= 0)
    @variable(model, v[transfers] >= 0)
    @variable(model, d[deliveries] >= 0)
    @variable(model, z[allocations], Bin)
    @variable(model, y[H], Bin)

    position = Dict(k => t for (t, k) in enumerate(H))
    fixed_of(k) = prob.fixed_cost[position[k]]
    capacity_of(k) = prob.capacity[position[k]]

    @objective(model, Min,
        sum(fixed_of(k) * y[k] for k in H) +
        sum(prob.chi * prob.dist[i, k] * u[(j, i, k)] for (j, i, k) in collections) +
        sum(prob.alpha * prob.dist[k, m] * v[(j, k, m)] for (j, k, m) in transfers) +
        sum(prob.delta * prob.dist[k, j] * d[(j, k)] for (j, k) in deliveries))

    for j in 1:n
        w_j = sum(prob.flow[i, j] for i in 1:n if i != j)
        for i in 1:n
            i == j && continue
            @constraint(model, sum(u[(j, i, k)] for k in H) == prob.flow[i, j])
        end
        @constraint(model, sum(d[(j, k)] for k in H) == w_j)
        for k in H
            inflow = sum(u[(j, i, k)] for i in 1:n if i != j; init=0.0)
            out_transfer = sum(v[(j, k, m)] for m in H if m != k; init=0.0)
            in_transfer = sum(v[(j, m, k)] for m in H if m != k; init=0.0)
            @constraint(model,
                inflow + in_transfer == out_transfer + d[(j, k)])
        end
        for i in 1:n, k in H
            i == j && continue
            @constraint(model, u[(j, i, k)] <= prob.flow[i, j] * y[k])
        end
        for k in H, m in H
            k == m && continue
            @constraint(model, v[(j, k, m)] <= w_j * y[k])
            @constraint(model, v[(j, k, m)] <= w_j * y[m])
        end
        for k in H
            @constraint(model, d[(j, k)] <= w_j * y[k])
        end
    end

    for i in 1:n
        @constraint(model, sum(z[(i, k)] for k in H) == 1)
    end
    for i in 1:n, k in H
        # All of node i's outbound volume enters the network at its own hub.
        @constraint(model,
            sum(u[(j, i, k)] for j in 1:n if j != i) <=
            prob.outvolume[i] * z[(i, k)])
        # All of node i's inbound volume is delivered from its own hub.
        @constraint(model, d[(i, k)] <= prob.involume[i] * z[(i, k)])
        @constraint(model, z[(i, k)] <= y[k])
    end
    for k in H
        # Hub capacity on collected inflow (AP convention), active when open.
        @constraint(model,
            sum(prob.outvolume[i] * z[(i, k)] for i in 1:n) <=
            capacity_of(k) * y[k])
    end

    return model
end

register_variant(
    :hub_location,
    :capacitated,
    CapacitatedHubLocationProblem,
    "Capacitated single-allocation hub location with fixed costs and collection-inflow capacities in loose/tight AP profiles (per-destination flow formulation)",
)
