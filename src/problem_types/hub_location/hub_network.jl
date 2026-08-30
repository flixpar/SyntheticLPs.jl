using JuMP
using Random
using Distributions

"""
Planted feasible backbone: the open regional gateways, each node's hub, and
the connected set of installed backbone links whose module capacities carry
the planted routing.
"""
struct HubNetworkWitness
    open_hubs::Vector{Int}
    assignment::Vector{Int}
    backbone::Vector{Tuple{Int,Int}}
end

"""
Relaxation-proof infeasibility certificate: a cut of the node set whose
crossing traffic must traverse backbone links whose total capacity (even with
every crossing link built) is strictly below the traffic that must cross.
Reach windows keep the origins' hubs on `side_a` and the destinations' hubs on
the complement, so every unit between the sides crosses on a backbone link;
the shared both-direction capacity rows then cannot hold - already in the LP
relaxation, where `b_l <= 1`.
"""
struct BackboneCutCertificate
    side_a::Vector{Int}
    crossing_flow::Float64
    crossing_capacity::Float64
end

"""
    HubNetworkDesignProblem <: ProblemGenerator

Generator for **single allocation over an incomplete hub network** (Yaman 2009;
O'Kelly & Bryan 1998; Yoon & Current 2008 for the multiple-allocation
counterpart), with modular backbone capacities in the spirit of Yaman & Carello
(2005) - the telecom-backbone member of the family.

# Overview

Access is single-allocation with reach windows: every city feeds its own
regional gateway hub. The backbone between hubs is *designed*: candidate hub
pairs may be built at a fixed cost `g_l` and then carry up to `C_l` (both
directions share the capacity, as in the package's telecom network design
model), with the discount `alpha` applying only on built backbone legs.
Flows route `i -> gateway(i) -> (backbone path) -> gateway(j) -> j`, so the
backbone must effectively be connected and its capacities sized for the
inter-regional traffic - a genuine multicommodity network design layer on top
of the hub layer.

# Data conventions (telecom backbone)
Asymmetric demand with heavy skew; access cost multipliers `chi = delta` in
`[1, 2.5]`; a deep backbone discount `alpha in [0.05, 0.4]` (repeaters and
dense wavelength-division multiplexing make inter-haul transit cheap per
bit-km); link capacities floored at / snapped up to the SONET/SDH module
ladder 155 / 622 / 2488 / 9953 / 39813 (OC-3/12/48/192/768); link build costs
with a distance-proportional component.

# Fields
- `n_nodes`, `hubs::Vector{Int}` (regional gateway candidates)
- `chi`, `alpha`, `delta`: leg cost multipliers
- `locations`, `dist::Matrix{Float64}`, `flow::Matrix{Float64}` (asymmetric)
- `reach::Float64`, `admissible::Vector{Vector{Int}}`: feeder reach windows
- `links::Vector{Tuple{Int,Int}}`: candidate backbone links (`k < m`)
- `link_capacity::Vector{Float64}`, `link_cost::Vector{Float64}` (aligned)
- `fixed_cost::Vector{Float64}`: hub opening costs (aligned with `hubs`)
- `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct HubNetworkDesignProblem <: ProblemGenerator
    n_nodes::Int
    hubs::Vector{Int}
    chi::Float64
    alpha::Float64
    delta::Float64
    locations::Vector{Tuple{Float64,Float64}}
    dist::Matrix{Float64}
    flow::Matrix{Float64}
    reach::Float64
    admissible::Vector{Vector{Int}}
    links::Vector{Tuple{Int,Int}}
    link_capacity::Vector{Float64}
    link_cost::Vector{Float64}
    fixed_cost::Vector{Float64}
    feasible_witness::Union{Nothing,HubNetworkWitness}
    infeasibility_certificate::Union{Nothing,BackboneCutCertificate}
    feasibility_status::FeasibilityStatus
end

const _HUB_BACKBONE_MODULES = (155.0, 622.0, 2488.0, 9953.0, 39813.0)

"""
    _hub_backbone_module(x) -> Float64

Smallest SONET/SDH capacity module of at least `x` (continuous fallback above
the OC-768 ladder).
"""
function _hub_backbone_module(x::Float64)
    for m in _HUB_BACKBONE_MODULES
        m >= x && return m
    end
    return x
end

"""
    _hub_mst_links(dist, nodes) -> Vector{Tuple{Int,Int}}

Minimum spanning tree over `nodes` (Prim's), as canonical `(min, max)` links.
"""
function _hub_mst_links(dist::Matrix{Float64}, nodes::Vector{Int})
    in_tree = Set(nodes[1:1])
    remaining = Set(nodes[2:end])
    links = Tuple{Int,Int}[]
    while !isempty(remaining)
        best_d, best_k, best_m = Inf, 0, 0
        for k in in_tree, m in remaining
            if dist[k, m] < best_d
                best_d, best_k, best_m = dist[k, m], k, m
            end
        end
        push!(links, minmax(best_k, best_m))
        push!(in_tree, best_m)
        delete!(remaining, best_m)
    end
    return links
end

"""
    _hub_tree_link_loads(n, assignment, flow, tree_links)
    -> Dict{Tuple{Int,Int},Float64}

Route every origin-destination pair over the unique tree path between its
endpoints' hubs and accumulate both-direction loads per link (capacities are
shared between directions).
"""
function _hub_tree_link_loads(n::Int, assignment::Vector{Int},
                              flow::Matrix{Float64},
                              tree_links::Vector{Tuple{Int,Int}})
    adj = Dict{Int,Vector{Int}}()
    for (k, m) in tree_links
        push!(get!(adj, k, Int[]), m)
        push!(get!(adj, m, Int[]), k)
    end
    loads = Dict{Tuple{Int,Int},Float64}()
    for i in 1:n, j in 1:n
        (i == j || assignment[i] == assignment[j]) && continue
        a, b = assignment[i], assignment[j]
        prev = Dict(a => 0)
        queue = [a]
        found = false
        while !isempty(queue) && !found
            x = popfirst!(queue)
            for y in get(adj, x, Int[])
                haskey(prev, y) && continue
                prev[y] = x
                y == b && (found = true; break)
                push!(queue, y)
            end
        end
        found || continue
        x = b
        while x != a && x != 0
            p = prev[x]
            key = minmax(x, p)
            loads[key] = get(loads, key, 0.0) + flow[i, j]
            x = p
        end
    end
    return loads
end

function _build_hub_network(n_nodes::Int, n_hubs::Int,
                            feasibility_status::FeasibilityStatus,
                            rng::AbstractRNG)
    n = n_nodes
    h = clamp(n_hubs, 2, n)

    # Regional geography: island groups with anchors on the group centers.
    n_regions = clamp(round(Int, h / 2), 2, 4)
    n_regions = min(n_regions, h, n)
    centers = _hub_ring_centers(rng, n_regions)
    node_region = vcat(collect(1:n_regions), rand(rng, 1:n_regions, max(0, n - n_regions)))
    min_sep = minimum(hypot(centers[a][1] - centers[b][1],
                            centers[a][2] - centers[b][2])
                      for a in 1:n_regions for b in (a + 1):n_regions)
    spread = 0.15 * min_sep
    locations = [g <= n_regions ? centers[g] :
                 (clamp(centers[node_region[g]][1] + rand(rng, Uniform(-spread, spread)),
                        0.0, 100.0),
                  clamp(centers[node_region[g]][2] + rand(rng, Uniform(-spread, spread)),
                        0.0, 100.0))
                 for g in 1:n]
    dist = _hub_distance_matrix(locations)

    populations = _hub_populations(rng, n)
    # Gateway candidates: the anchors plus the largest remaining cities.
    extra = [i for i in sortperm(populations; rev=true)
             if i > n_regions][1:max(0, h - n_regions)]
    hubs = sort(unique(vcat(collect(1:n_regions), extra)))

    # Regional gateway per region: the heaviest candidate of each region.
    planted = Int[]
    for g in 1:n_regions
        members = [k for k in hubs if node_region[k] == g]
        isempty(members) && continue
        push!(planted, members[argmax(populations[members])])
    end

    if feasibility_status == infeasible
        reach = 0.40 * min_sep
    else
        cover = maximum(minimum(dist[i, k] for k in planted) for i in 1:n)
        reach = cover * rand(rng, feasibility_status == feasible ?
                                   Uniform(1.05, 1.2) : Uniform(0.99, 1.1))
    end
    admissible = _hub_reach_admissible(dist, reach; candidates=hubs)

    decay = rand(rng, Uniform(0.5, 1.1))
    noise = rand(rng, Uniform(0.7, 1.2))
    flow = _hub_gravity_flows(rng, n, populations, dist, decay, noise;
                              symmetric=false,
                              scale=rand(rng, Uniform(0.5, 2.0)))
    outvolume = vec(sum(flow; dims=2))
    involume = vec(sum(flow; dims=1))

    chi = delta = rand(rng, Uniform(1.0, 2.5))
    alpha = rand(rng, Uniform(0.05, 0.4))

    # Candidate backbone links: a spanning tree over the planted gateways
    # (so a connected witness design exists) plus distance-limited extras.
    links = _hub_mst_links(dist, planted)
    extra_links = Tuple{Int,Int}[]
    for a in 1:length(hubs), b in (a + 1):length(hubs)
        k, m = hubs[a], hubs[b]
        minmax(k, m) in links && continue
        dist[k, m] <= 75.0 && rand(rng) < 0.25 &&
            push!(extra_links, minmax(k, m))
    end
    append!(links, extra_links)
    unique!(links)
    sort!(links)

    flow_cost_scale = sum(flow[i, j] * dist[i, j] for i in 1:n, j in 1:n if i != j)
    mean_link_dist = sum(dist[k, m] for (k, m) in links) / max(length(links), 1)
    base_link = flow_cost_scale / max(length(links), 1) * rand(rng, Uniform(0.2, 0.6))
    link_cost = [base_link * (0.7 + 0.6 * dist[k, m] / max(mean_link_dist, 1e-9))
                 for (k, m) in links]
    base_fixed = flow_cost_scale / max(length(hubs), 1) * rand(rng, Uniform(0.2, 0.7))
    fixed_cost = [base_fixed * exp(rand(rng, Uniform(log(0.8), log(1.25))))
                  for _ in hubs]

    assignment = _hub_nearest_assignment(dist, planted)
    witness = nothing
    certificate = nothing
    if feasibility_status == infeasible
        # Gateway bottleneck: traffic between the first region and the rest
        # must cross its backbone links, whose total capacity is set below
        # that traffic.
        side_a = findall(==(1), node_region)
        crossing_flow = sum(flow[i, j] + flow[j, i] for i in side_a, j in 1:n
                            if node_region[j] != 1)
        crossing = [t for (t, (k, m)) in enumerate(links)
                    if (node_region[k] == 1) != (node_region[m] == 1)]
        target = crossing_flow * rand(rng, Uniform(0.4, 0.65))
        shares = exp.(rand(rng, Normal(0.0, 0.3), max(length(crossing), 1)))
        shares ./= sum(shares)
        link_capacity = fill(0.0, length(links))
        for (s, t) in enumerate(crossing)
            link_capacity[t] = round(target * shares[s]; digits=3)
        end
        # Generous capacities away from the cut.
        intra_loads = _hub_tree_link_loads(n, assignment, flow, links)
        for (t, (k, m)) in enumerate(links)
            (node_region[k] == 1) != (node_region[m] == 1) && continue
            link_capacity[t] = _hub_backbone_module(
                2.0 * get(intra_loads, (k, m), 0.0))
        end
        certificate = BackboneCutCertificate(side_a, crossing_flow,
                                             sum(link_capacity[t] for t in crossing))
    else
        loads = _hub_tree_link_loads(n, assignment, flow, links)
        link_capacity = [
            _hub_backbone_module(rand(rng, Uniform(1.3, 2.0)) *
                                 get(loads, (k, m), 0.0))
            for (k, m) in links
        ]
        if feasibility_status == feasible
            witness = HubNetworkWitness(planted, assignment, copy(links))
        else
            # Unknown: squeeze the crossing capacity toward the crossing flow.
            side_a = findall(==(1), node_region)
            crossing_flow = sum(flow[i, j] + flow[j, i] for i in side_a, j in 1:n
                                if node_region[j] != 1)
            crossing = [t for (t, (k, m)) in enumerate(links)
                        if (node_region[k] == 1) != (node_region[m] == 1)]
            if !isempty(crossing)
                squeeze = rand(rng, Uniform(0.85, 1.2))
                total_cross = sum(link_capacity[t] for t in crossing)
                if total_cross > crossing_flow * squeeze
                    factor = crossing_flow * squeeze / total_cross
                    for t in crossing
                        link_capacity[t] = round(link_capacity[t] * factor; digits=3)
                    end
                end
            end
        end
    end

    return HubNetworkDesignProblem(n, hubs, chi, alpha, delta, locations, dist,
                                   flow, reach, admissible, links,
                                   link_capacity, link_cost, fixed_cost,
                                   witness, certificate, feasibility_status)
end

"""
    HubNetworkDesignProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an incomplete-hub-network (backbone design) instance.

# Variable count

Per destination `j`: one collection variable per (origin, admissible hub)
pair, two directed transfer variables per candidate backbone link, one
delivery variable per admissible hub; plus per-node allocation binaries, hub
opening binaries and one build binary per candidate link:

    vars = sum_j [ sum_{i != j} |A_i| + 2*|L| + |A_j| ]
          + sum_i |A_i| + h + |L|

An iterative re-sizing loop adjusts the node-count hint to land near the
target.

# Feasibility (relaxation-aware)
- `feasible`: gateways cover every node within the reach window and the
  planted spanning-tree backbone is sized (module-snapped) above its exact
  routed loads (`HubNetworkWitness`).
- `infeasible`: a regional gateway cut whose total crossing capacity is below
  the traffic that must cross it (`BackboneCutCertificate`).
- `unknown`: crossing capacities are squeezed toward the crossing traffic, so
  the backbone may or may not have room for the inter-regional demand.
"""
function HubNetworkDesignProblem(target_variables::Int,
                                 feasibility_status::FeasibilityStatus,
                                 seed::Int)
    target = max(target_variables, 1)
    hint_n = clamp(round(Int, 1.1 * target^(1 / 3)), 4, 80)
    hint_h = clamp(round(Int, hint_n / 3), 2, hint_n)
    best = nothing
    best_score = (1, Inf)
    for attempt in 1:18
        rng = MersenneTwister(seed + 49979687 * attempt)
        candidate = _build_hub_network(hint_n, hint_h, feasibility_status, rng)
        n, h = candidate.n_nodes, length(candidate.hubs)
        A, L = candidate.admissible, length(candidate.links)
        total = sum(sum(length(A[i]) for i in 1:n if i != j) + 2L + length(A[j])
                    for j in 1:n) +
                sum(length(A[i]) for i in 1:n) + h + L
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
        hint_n = clamp(next_n, 4, 80)
        hint_h = clamp(next_h, 2, hint_n)
    end
    return best::HubNetworkDesignProblem
end

"""
    build_model(prob::HubNetworkDesignProblem)

Build the backbone design model. Deterministic - uses only struct fields.

Variables (for each destination `j`; `A_i` admissible gateways, `L` links):
- `u[(j,i,k)] >= 0`, `k in A_i`: collection arc `i -> k` (`i != j`)
- `t[(j,k,m)] >= 0` for both orientations of every candidate link:
  discounted backbone flow
- `d[(j,k)] >= 0`, `k in A_j`: delivery arc `k -> j`
- `z[(i,k)] in {0,1}`, `k in A_i`: node `i` allocated to gateway `k`
- `y[k] in {0,1}`, `k in H`: open gateway `k`
- `b[(k,m)] in {0,1}`: build backbone link `(k, m)`

Objective: `sum f_k y_k + sum g_l b_l + sum_j [ chi*d*u + alpha*d*t + delta*d*d ]`.

Constraints:
- supply / delivery per commodity, hub conservation over the backbone
  adjacency
- link capacity, shared by both directions: `sum_j (t_jkm + t_jmk) <= C_l b_l`
- single allocation with disaggregated coupling to `z` and linking to `y`,
  as in the capacitated model but restricted to the reach windows
- gateway self-allocation: `z_kk == y_k`
"""
function build_model(prob::HubNetworkDesignProblem)
    model = Model()
    n = prob.n_nodes
    H = prob.hubs
    h = length(H)
    A = prob.admissible
    in_A = [Set(a) for a in A]
    links = prob.links

    backbone_adj = Dict{Int,Vector{Int}}()
    for (k, m) in links
        push!(get!(backbone_adj, k, Int[]), m)
        push!(get!(backbone_adj, m, Int[]), k)
    end

    collections = NTuple{3,Int}[]      # (j, i, k)
    transfers = NTuple{3,Int}[]        # (j, k, m), both orientations
    deliveries = NTuple{2,Int}[]       # (j, k)
    allocations = NTuple{2,Int}[]      # (i, k)
    for j in 1:n
        for i in 1:n, k in A[i]
            i == j && continue
            push!(collections, (j, i, k))
        end
        for (k, m) in links
            push!(transfers, (j, k, m))
            push!(transfers, (j, m, k))
        end
        for k in A[j]
            push!(deliveries, (j, k))
        end
    end
    for i in 1:n, k in A[i]
        push!(allocations, (i, k))
    end

    @variable(model, u[collections] >= 0)
    @variable(model, t[transfers] >= 0)
    @variable(model, d[deliveries] >= 0)
    @variable(model, z[allocations], Bin)
    @variable(model, y[H], Bin)
    @variable(model, b[links], Bin)

    position = Dict(k => idx for (idx, k) in enumerate(H))
    link_pos = Dict(l => idx for (idx, l) in enumerate(links))
    fixed_of(k) = prob.fixed_cost[position[k]]
    link_cost_of(l) = prob.link_cost[link_pos[l]]
    outvolume = vec(sum(prob.flow; dims=2))
    involume = vec(sum(prob.flow; dims=1))

    @objective(model, Min,
        sum(fixed_of(k) * y[k] for k in H) +
        sum(link_cost_of(l) * b[l] for l in links) +
        sum(prob.chi * prob.dist[i, k] * u[(j, i, k)] for (j, i, k) in collections) +
        sum(prob.alpha * prob.dist[k, m] * t[(j, k, m)] for (j, k, m) in transfers) +
        sum(prob.delta * prob.dist[k, j] * d[(j, k)] for (j, k) in deliveries))

    for j in 1:n
        w_j = sum(prob.flow[i, j] for i in 1:n if i != j)
        for i in 1:n
            i == j && continue
            @constraint(model, sum(u[(j, i, k)] for k in A[i]) == prob.flow[i, j])
        end
        @constraint(model, sum(d[(j, k)] for k in A[j]) == w_j)
        for k in H
            inflow = sum(u[(j, i, k)] for i in 1:n if i != j && k in in_A[i];
                         init=0.0)
            in_transfer = sum(t[(j, m, k)] for m in get(backbone_adj, k, Int[]);
                              init=0.0)
            out_transfer = sum(t[(j, k, m)] for m in get(backbone_adj, k, Int[]);
                               init=0.0)
            delivered = k in in_A[j] ? d[(j, k)] : 0.0
            @constraint(model, inflow + in_transfer == out_transfer + delivered)
        end
        for i in 1:n, k in A[i]
            i == j && continue
            @constraint(model, u[(j, i, k)] <= prob.flow[i, j] * y[k])
        end
        for l in links
            k, m = l
            @constraint(model, t[(j, k, m)] <= w_j * y[k])
            @constraint(model, t[(j, k, m)] <= w_j * y[m])
            @constraint(model, t[(j, m, k)] <= w_j * y[k])
            @constraint(model, t[(j, m, k)] <= w_j * y[m])
        end
        for k in A[j]
            @constraint(model, d[(j, k)] <= w_j * y[k])
        end
    end

    # Backbone link capacity, shared by both directions, available when built.
    for l in links
        k, m = l
        @constraint(model,
            sum(t[(j, k, m)] + t[(j, m, k)] for j in 1:n) <=
            prob.link_capacity[link_pos[l]] * b[l])
    end

    for i in 1:n
        @constraint(model, sum(z[(i, k)] for k in A[i]) == 1)
    end
    for i in 1:n, k in A[i]
        @constraint(model,
            sum(u[(j, i, k)] for j in 1:n if j != i) <= outvolume[i] * z[(i, k)])
        @constraint(model, d[(i, k)] <= involume[i] * z[(i, k)])
        @constraint(model, z[(i, k)] <= y[k])
    end
    for k in H
        @constraint(model, z[(k, k)] == y[k])
    end

    return model
end

register_variant(
    :hub_location,
    :hub_network,
    HubNetworkDesignProblem,
    "Single allocation over an incomplete hub network: design a capacitated modular backbone between regional gateways with feeder reach windows (telecom conventions)",
)
