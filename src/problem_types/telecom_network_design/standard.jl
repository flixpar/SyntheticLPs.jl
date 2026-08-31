using JuMP
using Random
using Distributions

const TELECOM_EPS = 1e-9

"""
Largest `target_variables` accepted by `TelecomNetworkDesignProblem`. Every
sparse (commodity, directed arc) coordinate is materialised in the JuMP model
and in the planted-routing bookkeeping, so larger targets are rejected with an
`ArgumentError` instead of being silently undersized (same convention as
`supply_chain/network_planning`).
"""
const TELECOM_MAX_VARIABLES = 1_000_000

"""
Planted nominal design: an explicit feasible point of the network design model.

`installed_links` is the support of the design (`y = 1` there, `y = 0`
elsewhere); `routes[k]` lists `(node_path, flow)` pairs whose flows sum to
commodity `k`'s demand and whose paths only use installed links;
`link_loads` is the resulting total (both directions) load per installed link,
which never exceeds that link's capacity; `installation_cost` is the design's
spend, which never exceeds the budget.
"""
struct TelecomRouteWitness
    installed_links::Vector{Tuple{Int,Int}}
    routes::Vector{Vector{Tuple{Vector{Int},Float64}}}
    link_loads::Dict{Tuple{Int,Int},Float64}
    installation_cost::Float64
end

"""
Relaxation-proof capacity-cut certificate: the node set `side` must exchange
`crossing_demand` units with its complement, but the links crossing the cut
can carry at most `crossing_capacity < crossing_demand` even when *all* of them
are installed. The argument only uses flow conservation and
`sum_k f_k(a) <= cap_a * y_a <= cap_a` (valid for `y in [0, 1]`), so it refutes
the LP relaxation as well as the integer model.
"""
struct TelecomCapacityCutCertificate
    side::Vector{Int}
    crossing_links::Vector{Tuple{Int,Int}}
    crossing_demand::Float64
    crossing_capacity::Float64
end

"""
Relaxation-proof budget certificate: the node set `side` exchanges
`crossing_demand` units with its complement, so the crossing links must supply
at least that much installed capacity, `sum_{a in cut} cap_a * y_a >=
crossing_demand`. Every crossing link costs at least `cost_per_capacity =
min_{a in cut} c_a / cap_a` per unit of capacity, hence any solution spends at
least `implied_minimum = crossing_demand * cost_per_capacity > budget`. The
bound uses `y >= 0` only, so it holds for the LP relaxation too.
"""
struct TelecomBudgetCertificate
    side::Vector{Int}
    crossing_links::Vector{Tuple{Int,Int}}
    crossing_demand::Float64
    cost_per_capacity::Float64
    implied_minimum::Float64
    budget::Float64
end

"""
    TelecomNetworkDesignProblem <: ProblemGenerator

Generator for telecommunication network design problems.

This problem models the design of a telecommunications network by deciding which links to install
and how to route multiple traffic demands (commodities) to minimize total cost while satisfying
capacity constraints. It is a multicommodity network design problem with discrete capacity installation.

# Overview
Models fixed-charge telecom network design with multicommodity routing. The
decisions install physical links and route each traffic commodity over directed
arcs. The objective minimizes installation cost plus routing cost. Flow
conservation sends every commodity from its source to its sink, physical link
capacity is available only when the link is installed, and a budget limits total
installation cost.

# Joint calibration

Topology, capacities, demand and budget are *not* sampled independently. A
planted nominal design ties them together:

1. a proximity-driven topology is drawn and each link is given a SONET/OTN
   capacity module sized from the traffic it carries under the nominal routing;
2. the unit-demand traffic matrix is routed over that topology with a
   Frank-Wolfe congestion-balancing routing, giving per-link loads and hence
   `routable_scale`, the largest total demand this planted routing carries;
3. a family of cuts (singletons, geometric sweeps, nearest-neighbour balls)
   gives `cut_bound_scale`, the smallest total demand that provably cannot be
   routed (`routable_scale <= cut_bound_scale` always);
4. the realised total demand is placed relative to those two anchors according
   to the requested feasibility status, and the budget is set relative to
   `nominal_cost`, the spend of the planted design.

# Fields
- `n_nodes::Int`: Number of nodes in the network
- `n_arcs::Int`: Number of potential arcs/links
- `n_commodities::Int`: Number of traffic demands (origin-destination pairs)
- `arcs::Vector{Tuple{Int,Int}}`: Potential physical links (canonical form: i < j)
- `directed_arcs::Vector{Tuple{Int,Int}}`: Directed arcs (both directions for each physical link)
- `node_locations::Vector{Tuple{Float64,Float64}}`: Geographic coordinates of nodes
- `distances::Dict{Tuple{Int,Int},Float64}`: Distance for each arc
- `installation_costs::Dict{Tuple{Int,Int},Float64}`: Cost to install each physical link
- `link_capacities::Dict{Tuple{Int,Int},Float64}`: Capacity of each physical link
- `flow_costs::Dict{Tuple{Int,Int},Float64}`: Cost per unit flow on each directed arc
- `commodities::Vector{Dict{Symbol,Any}}`: Traffic demands with :source, :sink, :demand
- `budget::Float64`: Budget constraint for installation costs
- `outgoing_arcs::Dict{Int,Vector{Tuple{Int,Int}}}`: Outgoing directed arcs for each node
- `incoming_arcs::Dict{Int,Vector{Tuple{Int,Int}}}`: Incoming directed arcs for each node
- `total_demand::Float64`: Realised sum of commodity demands
- `routable_scale::Float64`: Total demand the planted routing carries exactly
- `cut_bound_scale::Float64`: Total demand at which the tightest cut saturates
- `nominal_cost::Float64`: Installation cost of the planted nominal design
- `feasible_witness::Union{Nothing,TelecomRouteWitness}`: planted feasible point
- `infeasibility_certificate::Union{Nothing,TelecomCapacityCutCertificate,TelecomBudgetCertificate}`
- `feasibility_status::FeasibilityStatus`
"""
struct TelecomNetworkDesignProblem <: ProblemGenerator
    n_nodes::Int
    n_arcs::Int
    n_commodities::Int
    arcs::Vector{Tuple{Int,Int}}
    directed_arcs::Vector{Tuple{Int,Int}}
    node_locations::Vector{Tuple{Float64,Float64}}
    distances::Dict{Tuple{Int,Int},Float64}
    installation_costs::Dict{Tuple{Int,Int},Float64}
    link_capacities::Dict{Tuple{Int,Int},Float64}
    flow_costs::Dict{Tuple{Int,Int},Float64}
    commodities::Vector{Dict{Symbol,Any}}
    budget::Float64
    outgoing_arcs::Dict{Int,Vector{Tuple{Int,Int}}}
    incoming_arcs::Dict{Int,Vector{Tuple{Int,Int}}}
    total_demand::Float64
    routable_scale::Float64
    cut_bound_scale::Float64
    nominal_cost::Float64
    feasible_witness::Union{Nothing,TelecomRouteWitness}
    infeasibility_certificate::Union{Nothing,TelecomCapacityCutCertificate,
                                    TelecomBudgetCertificate}
    feasibility_status::FeasibilityStatus
end

# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

@inline _telecom_link(i::Int, j::Int) = i < j ? (i, j) : (j, i)

"""
    _telecom_dimensions(target::Int, rng) -> (n_arcs, n_commodities, n_nodes)

Pick topology dimensions whose exact model size, `n_arcs * (2 * n_commodities +
1)`, lands as close as possible to `target`.

The model has one binary per physical link and one flow variable per
(commodity, directed arc) pair, so the size is a product of two integers. The
old generator rounded `n_arcs`/`n_commodities` inside hard-coded per-scale
bands *before* reconciling the product, which produced plateaus (every target
in `101:290` collapsed onto 315 variables). Here a realistic
commodities-per-link ratio is sampled first, then every commodity count in a
wide window around it is scored by `size error + small ratio penalty`, so the
realised size tracks the target continuously.
"""
function _telecom_dimensions(target::Int, rng::AbstractRNG)
    v = float(target)
    # Commodities per physical link: sparse regional plans through dense
    # national plans where most node pairs exchange traffic.
    ratio = exp(rand(rng, Uniform(log(0.35), log(1.4))))
    # v = A * (2 * ratio * A + 1)  =>  A = (-1 + sqrt(1 + 8 * ratio * v)) / (4 * ratio)
    arcs_hint = (-1.0 + sqrt(1.0 + 8.0 * ratio * v)) / (4.0 * ratio)
    comm_hint = max(2.0, ratio * arcs_hint)

    best = (0, 0)
    best_score = Inf
    lo = max(2, floor(Int, 0.35 * comm_hint))
    hi = max(lo + 1, ceil(Int, 2.6 * comm_hint))
    for k in lo:hi
        a = round(Int, v / (2 * k + 1))
        a < 4 && continue
        size_error = abs(a * (2 * k + 1) - v) / v
        score = size_error + 0.02 * abs(log(k / comm_hint))
        if score < best_score
            best_score = score
            best = (a, k)
        end
    end
    if best == (0, 0)  # tiny targets: fall back to the smallest legal shape
        best = (4, max(2, round(Int, ((v / 4) - 1) / 2)))
    end
    n_arcs, n_commodities = best

    # Nodes from links: telecom access/backbone plans run at 1.5-2.5 links per
    # node. Clamp so the topology is realisable (connected, simple).
    density = rand(rng, Uniform(1.5, 2.5))
    n_nodes = clamp(round(Int, n_arcs / density), 4, n_arcs + 1)
    while div(n_nodes * (n_nodes - 1), 2) < n_arcs
        n_nodes += 1
    end
    return n_arcs, n_commodities, n_nodes
end

# ---------------------------------------------------------------------------
# Topology and traffic data
# ---------------------------------------------------------------------------

"""
    _telecom_node_locations(rng, n_nodes, width, height)

Geographic node placement: population clusters (metro areas) with Gaussian
scatter around each cluster centre.
"""
function _telecom_node_locations(rng::AbstractRNG, n_nodes::Int, width::Float64,
                                 height::Float64)
    n_clusters = max(1, div(n_nodes, 4))
    centers = [(width * rand(rng), height * rand(rng)) for _ in 1:n_clusters]
    locations = Vector{Tuple{Float64,Float64}}(undef, n_nodes)
    for i in 1:n_nodes
        c = centers[rand(rng, 1:n_clusters)]
        x = clamp(c[1] + randn(rng) * (width / 8), 0.0, width)
        y = clamp(c[2] + randn(rng) * (height / 8), 0.0, height)
        locations[i] = (x, y)
    end
    return locations
end

"""
    _telecom_topology(rng, n_nodes, n_arcs, locations) -> Vector{Tuple{Int,Int}}

Proximity-driven topology with exactly `n_arcs` links: a Euclidean minimum
spanning tree (guaranteeing connectivity) plus the shortest remaining pairs,
with a lognormal perturbation of the ranking so a few long-haul links appear.
"""
function _telecom_topology(rng::AbstractRNG, n_nodes::Int, n_arcs::Int,
                           locations::Vector{Tuple{Float64,Float64}})
    dist(i, j) = hypot(locations[i][1] - locations[j][1],
                       locations[i][2] - locations[j][2])

    # Prim's MST.
    in_tree = falses(n_nodes)
    best_cost = fill(Inf, n_nodes)
    best_from = zeros(Int, n_nodes)
    in_tree[1] = true
    for j in 2:n_nodes
        best_cost[j] = dist(1, j)
        best_from[j] = 1
    end
    links = Set{Tuple{Int,Int}}()
    for _ in 2:n_nodes
        u = 0
        best = Inf
        for j in 1:n_nodes
            if !in_tree[j] && best_cost[j] < best
                best = best_cost[j]
                u = j
            end
        end
        u == 0 && break
        in_tree[u] = true
        push!(links, _telecom_link(u, best_from[u]))
        for j in 1:n_nodes
            if !in_tree[j]
                d = dist(u, j)
                if d < best_cost[j]
                    best_cost[j] = d
                    best_from[j] = u
                end
            end
        end
    end

    if length(links) < n_arcs
        candidates = Tuple{Float64,Int,Int}[]
        for i in 1:n_nodes, j in (i + 1):n_nodes
            (i, j) in links && continue
            push!(candidates, (dist(i, j) * exp(randn(rng) * 0.35), i, j))
        end
        sort!(candidates; by=first)
        for (_, i, j) in candidates
            length(links) >= n_arcs && break
            push!(links, (i, j))
        end
    end

    return sort!(collect(links))
end

"""
    _telecom_traffic(rng, n_nodes, n_commodities) -> (sources, sinks, shares)

Origin-destination pairs drawn with a population bias plus lognormal volume
shares that sum to one. The absolute demand scale is fixed later, once the
topology's routing capacity is known.
"""
function _telecom_traffic(rng::AbstractRNG, n_nodes::Int, n_commodities::Int)
    population = [exp(randn(rng) * 0.7) for _ in 1:n_nodes]
    weights = population ./ sum(population)
    cumulative = cumsum(weights)
    draw() = min(n_nodes, searchsortedfirst(cumulative, rand(rng)))

    sources = Vector{Int}(undef, n_commodities)
    sinks = Vector{Int}(undef, n_commodities)
    raw = Vector{Float64}(undef, n_commodities)
    for k in 1:n_commodities
        s = draw()
        t = draw()
        while t == s
            t = draw()
        end
        sources[k] = s
        sinks[k] = t
        # Gravity-style volume: bigger endpoints exchange more traffic.
        raw[k] = sqrt(population[s] * population[t]) * exp(randn(rng) * 0.6)
    end
    return sources, sinks, raw ./ sum(raw)
end

# ---------------------------------------------------------------------------
# Shortest paths (binary-heap Dijkstra over the sparse link list)
# ---------------------------------------------------------------------------

function _telecom_heap_push!(heap_keys::Vector{Float64}, heap_vals::Vector{Int},
                             key::Float64, val::Int)
    push!(heap_keys, key)
    push!(heap_vals, val)
    c = length(heap_keys)
    while c > 1
        p = c >> 1
        heap_keys[p] <= heap_keys[c] && break
        heap_keys[p], heap_keys[c] = heap_keys[c], heap_keys[p]
        heap_vals[p], heap_vals[c] = heap_vals[c], heap_vals[p]
        c = p
    end
    return nothing
end

function _telecom_heap_pop!(heap_keys::Vector{Float64}, heap_vals::Vector{Int})
    top_key, top_val = heap_keys[1], heap_vals[1]
    last_key, last_val = pop!(heap_keys), pop!(heap_vals)
    if !isempty(heap_keys)
        heap_keys[1], heap_vals[1] = last_key, last_val
        n = length(heap_keys)
        p = 1
        while true
            l, r = 2p, 2p + 1
            m = p
            l <= n && heap_keys[l] < heap_keys[m] && (m = l)
            r <= n && heap_keys[r] < heap_keys[m] && (m = r)
            m == p && break
            heap_keys[p], heap_keys[m] = heap_keys[m], heap_keys[p]
            heap_vals[p], heap_vals[m] = heap_vals[m], heap_vals[p]
            p = m
        end
    end
    return top_key, top_val
end

"""
    _telecom_shortest_path_tree(adjacency, lengths, source)

Dijkstra from `source` over the undirected link list. `adjacency[u]` holds
`(neighbour, link_index)` pairs and `lengths[a]` is link `a`'s length. Returns
`(dist, parent_node, parent_link)`.
"""
function _telecom_shortest_path_tree(adjacency::Vector{Vector{Tuple{Int,Int}}},
                                     lengths::Vector{Float64}, source::Int)
    n = length(adjacency)
    dist = fill(Inf, n)
    parent_node = zeros(Int, n)
    parent_link = zeros(Int, n)
    dist[source] = 0.0
    heap_keys = Float64[0.0]
    heap_vals = Int[source]
    while !isempty(heap_vals)
        d, u = _telecom_heap_pop!(heap_keys, heap_vals)
        d > dist[u] + TELECOM_EPS && continue
        for (v, a) in adjacency[u]
            nd = d + lengths[a]
            if nd < dist[v] - TELECOM_EPS
                dist[v] = nd
                parent_node[v] = u
                parent_link[v] = a
                _telecom_heap_push!(heap_keys, heap_vals, nd, v)
            end
        end
    end
    return dist, parent_node, parent_link
end

function _telecom_extract_path(parent_node::Vector{Int}, parent_link::Vector{Int},
                               source::Int, sink::Int)
    path = Int[]
    node = sink
    while node != source
        link = parent_link[node]
        link == 0 && return Int[]
        push!(path, link)
        node = parent_node[node]
    end
    reverse!(path)
    return path
end

# ---------------------------------------------------------------------------
# Planted nominal routing (Frank-Wolfe congestion balancing)
# ---------------------------------------------------------------------------

"""
    _telecom_nominal_routing(adjacency, capacity, link_cost, sources, sinks,
                             shares; iterations)

Route the unit-total-demand traffic matrix over the topology, balancing
congestion. Iteration 1 is the plain cheapest-path routing; later iterations
take Frank-Wolfe steps on `sum_a (load_a / cap_a)^4`, which spreads traffic off
the bottlenecks. The best iterate (lowest maximum utilisation) is returned as

- `loads[a]`: link load per unit of total demand,
- `weights[k]`: `path (link indices) => fraction of commodity k` (sums to 1),
- `ratio`: the maximum utilisation `max_a loads[a] / cap_a`.

Consequently the routing carries a total demand of `1 / ratio` exactly.
"""
function _telecom_nominal_routing(adjacency::Vector{Vector{Tuple{Int,Int}}},
                                  capacity::Vector{Float64},
                                  link_cost::Vector{Float64},
                                  sources::Vector{Int}, sinks::Vector{Int},
                                  shares::Vector{Float64}; iterations::Int=8)
    m = length(capacity)
    n_commodities = length(shares)
    by_source = Dict{Int,Vector{Int}}()
    for k in 1:n_commodities
        push!(get!(by_source, sources[k], Int[]), k)
    end

    cost_scale = maximum(link_cost)
    cost_scale <= 0 && (cost_scale = 1.0)
    normalized_cost = link_cost ./ cost_scale

    loads = zeros(m)
    weights = [Dict{Vector{Int},Float64}() for _ in 1:n_commodities]
    best_loads = zeros(m)
    best_weights = weights
    best_ratio = Inf

    step_loads = zeros(m)
    step_paths = Vector{Vector{Int}}(undef, n_commodities)
    lengths = zeros(m)

    for t in 1:iterations
        gradient_max = 0.0
        for a in 1:m
            g = (loads[a] / capacity[a])^3 / capacity[a]
            lengths[a] = g
            gradient_max = max(gradient_max, g)
        end
        # The cost tie-break keeps early iterates realistic (cheapest paths)
        # and then fades, so the later iterates chase pure min-congestion.
        cost_weight = 0.05 / t
        for a in 1:m
            lengths[a] = (gradient_max > 0 ? lengths[a] / gradient_max : 0.0) +
                         cost_weight * normalized_cost[a] + 1e-9
        end

        fill!(step_loads, 0.0)
        # Sorted source order keeps the floating-point accumulation - and hence
        # the whole instance - bit-for-bit reproducible.
        for s in sort!(collect(keys(by_source)))
            _, parent_node, parent_link = _telecom_shortest_path_tree(adjacency, lengths, s)
            for k in by_source[s]
                path = _telecom_extract_path(parent_node, parent_link, s, sinks[k])
                step_paths[k] = path
                for a in path
                    step_loads[a] += shares[k]
                end
            end
        end

        step = t == 1 ? 1.0 : 2.0 / (t + 1.0)
        for a in 1:m
            loads[a] = (1.0 - step) * loads[a] + step * step_loads[a]
        end
        for k in 1:n_commodities
            w = weights[k]
            if step >= 1.0 - TELECOM_EPS
                empty!(w)
                w[step_paths[k]] = 1.0
            else
                for key in collect(keys(w))
                    w[key] *= (1.0 - step)
                end
                w[step_paths[k]] = get(w, step_paths[k], 0.0) + step
            end
        end

        ratio = 0.0
        for a in 1:m
            ratio = max(ratio, loads[a] / capacity[a])
        end
        if ratio < best_ratio - 1e-12
            best_ratio = ratio
            best_loads = copy(loads)
            best_weights = [copy(w) for w in weights]
        end
    end

    return best_loads, best_weights, best_ratio
end

# ---------------------------------------------------------------------------
# Cut bounds
# ---------------------------------------------------------------------------

"""
    _telecom_cut_bounds(arcs, capacity, install_cost, n_nodes, locations, sources,
                        sinks, shares, rng)

Scan a family of node cuts - every singleton, geometric sweep cuts along random
directions, and nearest-neighbour balls - and return the two extreme ones per
unit of total demand:

- `capacity_cut`: minimises `crossing capacity / crossing demand share`; the
  reciprocal-scaled value `cut_scale` is the smallest total demand that provably
  cannot be routed;
- `budget_cut`: maximises `crossing demand share * min_{a in cut} c_a / cap_a`,
  the strongest per-unit-demand lower bound on installation spend.

Each entry is `(side, crossing_links, crossing_capacity, crossing_share,
cost_per_capacity)`.
"""
function _telecom_cut_bounds(arcs::Vector{Tuple{Int,Int}}, capacity::Vector{Float64},
                             install_cost::Vector{Float64}, n_nodes::Int,
                             locations::Vector{Tuple{Float64,Float64}},
                             sources::Vector{Int}, sinks::Vector{Int},
                             shares::Vector{Float64}, rng::AbstractRNG)
    best_capacity = nothing
    best_capacity_value = Inf
    best_budget = nothing
    best_budget_value = -Inf

    mask = falses(n_nodes)
    function evaluate!(side_mask::BitVector)
        crossing = Int[]
        cap = 0.0
        ratio = Inf
        for (a, (i, j)) in enumerate(arcs)
            if side_mask[i] != side_mask[j]
                push!(crossing, a)
                cap += capacity[a]
                ratio = min(ratio, install_cost[a] / capacity[a])
            end
        end
        isempty(crossing) && return nothing
        share = 0.0
        for k in eachindex(shares)
            if side_mask[sources[k]] != side_mask[sinks[k]]
                share += shares[k]
            end
        end
        share <= TELECOM_EPS && return nothing
        side = [v for v in 1:n_nodes if side_mask[v]]
        links = [arcs[a] for a in crossing]
        if cap / share < best_capacity_value
            best_capacity_value = cap / share
            best_capacity = (side, links, cap, share, ratio)
        end
        if share * ratio > best_budget_value
            best_budget_value = share * ratio
            best_budget = (side, links, cap, share, ratio)
        end
        return nothing
    end

    for v in 1:n_nodes
        fill!(mask, false)
        mask[v] = true
        evaluate!(mask)
    end

    for _ in 1:12
        theta = rand(rng, Uniform(0.0, pi))
        order = sortperm([locations[v][1] * cos(theta) + locations[v][2] * sin(theta)
                          for v in 1:n_nodes])
        step = max(1, div(n_nodes, 16))
        fill!(mask, false)
        for (idx, v) in enumerate(order)
            mask[v] = true
            idx >= n_nodes && break
            (idx % step == 0) && evaluate!(mask)
        end
    end

    for _ in 1:min(12, n_nodes)
        center = rand(rng, 1:n_nodes)
        order = sortperm([hypot(locations[v][1] - locations[center][1],
                                locations[v][2] - locations[center][2])
                          for v in 1:n_nodes])
        ball_size = rand(rng, 2:max(2, div(n_nodes, 2)))
        fill!(mask, false)
        for idx in 1:min(ball_size, n_nodes - 1)
            mask[order[idx]] = true
        end
        evaluate!(mask)
    end

    return best_capacity, best_budget
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
    TelecomNetworkDesignProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a telecommunication network design problem instance.

# Arguments
- `target_variables`: Target number of variables in the LP formulation
  (`n_arcs * (2 * n_commodities + 1)`, at most `TELECOM_MAX_VARIABLES`)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Feasibility (relaxation-aware)
- `feasible`: total demand is set to 55-90% of `routable_scale`, so the planted
  nominal design routes everything inside the installed capacities, and the
  budget exceeds that design's cost. Stored as `feasible_witness`.
- `infeasible`: either a *capacity* shortfall (demand pushed 15-80% past the
  tightest cut, `TelecomCapacityCutCertificate`) or a *budget* shortfall
  (routable demand but a budget below the cut-implied minimum spend,
  `TelecomBudgetCertificate`). Both certificates only use `0 <= y <= 1`, so the
  instance stays infeasible after `relax_integrality`.
- `unknown`: total demand is placed in a +-35% log band just above
  `routable_scale`, which brackets the true routing threshold at every scale,
  so whether the instance is routable is a genuine question - and the position
  inside the band is a low-discrepancy function of the seed, so any block of
  seeds is an even mix rather than a lucky or unlucky draw.
"""
function TelecomNetworkDesignProblem(target_variables::Int,
                                     feasibility_status::FeasibilityStatus, seed::Int)
    if target_variables > TELECOM_MAX_VARIABLES
        throw(ArgumentError(
            "telecom_network_design/standard supports at most " *
            "$(TELECOM_MAX_VARIABLES) variables (requested $(target_variables))"))
    end
    rng = MersenneTwister(seed)
    target = max(target_variables, 12)

    n_arcs, n_commodities, n_nodes = _telecom_dimensions(target, rng)

    # Geographic scale and unit costs grow with the network (metro -> national).
    span = clamp(120.0 * sqrt(n_nodes), 150.0, 9000.0) * rand(rng, Uniform(0.8, 1.25))
    base_installation_cost = 12000.0 * sqrt(n_nodes) * rand(rng, Uniform(0.8, 1.3))
    cost_per_km = rand(rng, Uniform(60.0, 400.0))
    flow_cost_per_unit = rand(rng, Uniform(0.005, 0.08))
    capacity_modules = [155.0, 622.0, 2488.0, 9953.0, 39813.0]  # OC-3 .. OC-768

    node_locations = _telecom_node_locations(rng, n_nodes, span, span)
    arcs = _telecom_topology(rng, n_nodes, n_arcs, node_locations)
    n_arcs = length(arcs)

    link_distance = [hypot(node_locations[i][1] - node_locations[j][1],
                           node_locations[i][2] - node_locations[j][2])
                     for (i, j) in arcs]
    link_flow_cost = [link_distance[a] * flow_cost_per_unit *
                      rand(rng, Uniform(0.9, 1.1)) for a in 1:n_arcs]

    adjacency = [Tuple{Int,Int}[] for _ in 1:n_nodes]
    for (a, (i, j)) in enumerate(arcs)
        push!(adjacency[i], (j, a))
        push!(adjacency[j], (i, a))
    end

    sources, sinks, shares = _telecom_traffic(rng, n_nodes, n_commodities)

    # Pass 1: cheapest-path loads with uniform capacities tell us how much
    # traffic each link is naturally asked to carry; the SONET/OTN module is
    # then sized from that load (long-haul backbone links get the big pipes).
    unit_capacity = ones(n_arcs)
    reference_loads, _, _ = _telecom_nominal_routing(adjacency, unit_capacity,
                                                     link_flow_cost, sources, sinks,
                                                     shares; iterations=3)
    load_ceiling = maximum(reference_loads)
    load_ceiling <= TELECOM_EPS && (load_ceiling = 1.0)
    top_module = capacity_modules[end]
    capacity = Vector{Float64}(undef, n_arcs)
    module_index = Vector{Int}(undef, n_arcs)
    for a in 1:n_arcs
        # A spare floor keeps unused links installable (and useful for reroutes)
        # instead of leaving dead capacity-zero edges in the topology.
        required = max(reference_loads[a], 0.12 * load_ceiling) *
                   rand(rng, Uniform(1.0, 1.5)) / load_ceiling * top_module
        idx = findfirst(c -> c >= required, capacity_modules)
        module_index[a] = idx === nothing ? length(capacity_modules) : idx
        capacity[a] = capacity_modules[module_index[a]]
    end

    install_cost = [base_installation_cost *
                    (0.8 + 0.4 * module_index[a] / length(capacity_modules)) +
                    link_distance[a] * cost_per_km * rand(rng, Uniform(0.9, 1.1))
                    for a in 1:n_arcs]

    # Pass 2: the planted nominal routing over the realised capacities.
    # Small topologies are cheap to route, so spend more Frank-Wolfe steps on
    # them: the planted routing must be near-optimal at every size, otherwise
    # `routable_scale` under-states the true threshold and the unknown profile
    # skews feasible.
    routing_steps = clamp(div(4000, max(n_arcs, 1)), 10, 80)
    unit_loads, path_weights, congestion = _telecom_nominal_routing(
        adjacency, capacity, link_flow_cost, sources, sinks, shares;
        iterations=routing_steps)
    routable_scale = congestion > TELECOM_EPS ? 1.0 / congestion : 1.0

    capacity_cut, budget_cut = _telecom_cut_bounds(arcs, capacity, install_cost,
                                                   n_nodes, node_locations, sources,
                                                   sinks, shares, rng)
    # A connected topology carrying at least one commodity always yields a
    # valid cut (that commodity's source singleton), so the `nothing` branch is
    # a defensive fallback only.
    cut_scale = capacity_cut === nothing ? routable_scale * 4.0 :
                capacity_cut[3] / capacity_cut[4]
    cut_scale = max(cut_scale, routable_scale)

    nominal_links = [a for a in 1:n_arcs if unit_loads[a] > TELECOM_EPS]
    nominal_cost = sum(install_cost[a] for a in nominal_links; init=0.0)

    # --- place the demand scale relative to the two anchors ------------------
    mode = :capacity
    if feasibility_status == feasible
        total_demand = routable_scale * rand(rng, Uniform(0.55, 0.9))
    elseif feasibility_status == infeasible
        mode = rand(rng, Bool) ? :capacity : :budget
        total_demand = mode == :capacity ?
                       cut_scale * rand(rng, Uniform(1.15, 1.8)) :
                       routable_scale * rand(rng, Uniform(0.4, 0.8))
    else
        # `routable_scale` brackets the true routing threshold from below and
        # `cut_scale` from above, and the planted routing is tight: solving the
        # max-concurrent-flow LP over the corpus puts the threshold between
        # 1.00x and 1.11x `routable_scale` at every network size. Placing the
        # demand in a +-35% log band just above `routable_scale` therefore
        # straddles the threshold from both sides at *every* scale - the mix is
        # a property of the planted routing, not of the absolute size.
        #
        # The position inside the band is a golden-ratio (low-discrepancy)
        # function of the seed rather than another uniform draw: consecutive
        # seeds then sweep the band evenly, so any block of seeds is a genuine
        # mix instead of a binomial gamble that can come out 5/30 by chance.
        position = mod(seed * 0.6180339887498949, 1.0)
        total_demand = routable_scale * exp(0.04 + 0.7 * (position - 0.5))
    end

    demands = [round(total_demand * shares[k], digits=4) for k in 1:n_commodities]
    demands = [max(d, 1e-3) for d in demands]

    # --- status-specific repairs on the realised (rounded) demands ----------
    if feasibility_status == feasible
        loads = zeros(n_arcs)
        for k in 1:n_commodities, (path, w) in path_weights[k]
            for a in path
                loads[a] += demands[k] * w
            end
        end
        overflow = maximum(loads[a] / capacity[a] for a in 1:n_arcs)
        if overflow > 1.0
            demands = [round(d / (overflow * 1.001), digits=4) for d in demands]
        end
    elseif feasibility_status == infeasible && mode == :capacity
        side_set = Set(capacity_cut[1])
        crossing_demand = sum(demands[k] for k in 1:n_commodities
                              if (sources[k] in side_set) != (sinks[k] in side_set);
                              init=0.0)
        if crossing_demand <= capacity_cut[3] * 1.05
            factor = capacity_cut[3] * 1.15 / max(crossing_demand, TELECOM_EPS)
            demands = [round(d * factor, digits=4) for d in demands]
        end
    end

    total_demand = sum(demands)

    commodities = [Dict{Symbol,Any}(:source => sources[k], :sink => sinks[k],
                                    :demand => demands[k])
                   for k in 1:n_commodities]

    # --- witness / certificate / budget -------------------------------------
    witness = nothing
    certificate = nothing
    budget = nominal_cost * rand(rng, Uniform(0.85, 1.35))

    if feasibility_status == feasible
        loads = zeros(n_arcs)
        routes = Vector{Vector{Tuple{Vector{Int},Float64}}}(undef, n_commodities)
        for k in 1:n_commodities
            entries = Tuple{Vector{Int},Float64}[]
            for (path, w) in sort!(collect(path_weights[k]); by=first)
                flow = demands[k] * w
                flow <= TELECOM_EPS && continue
                nodes = [sources[k]]
                for a in path
                    i, j = arcs[a]
                    push!(nodes, last(nodes) == i ? j : i)
                    loads[a] += flow
                end
                push!(entries, (nodes, flow))
            end
            routes[k] = entries
        end
        installed = [a for a in 1:n_arcs if loads[a] > TELECOM_EPS]
        installation_cost = sum(install_cost[a] for a in installed; init=0.0)
        budget = installation_cost * rand(rng, Uniform(1.02, 1.35))
        witness = TelecomRouteWitness(
            [arcs[a] for a in installed],
            routes,
            Dict(arcs[a] => loads[a] for a in installed),
            installation_cost,
        )
    elseif feasibility_status == infeasible && mode == :capacity
        side_set = Set(capacity_cut[1])
        crossing_demand = sum(demands[k] for k in 1:n_commodities
                              if (sources[k] in side_set) != (sinks[k] in side_set);
                              init=0.0)
        certificate = TelecomCapacityCutCertificate(capacity_cut[1], capacity_cut[2],
                                                    crossing_demand, capacity_cut[3])
    elseif feasibility_status == infeasible
        side_set = Set(budget_cut[1])
        crossing_demand = sum(demands[k] for k in 1:n_commodities
                              if (sources[k] in side_set) != (sinks[k] in side_set);
                              init=0.0)
        implied_minimum = crossing_demand * budget_cut[5]
        budget = implied_minimum * rand(rng, Uniform(0.45, 0.85))
        certificate = TelecomBudgetCertificate(budget_cut[1], budget_cut[2],
                                               crossing_demand, budget_cut[5],
                                               implied_minimum, budget)
    end

    # --- materialise the dictionary-keyed model data -------------------------
    directed_arcs = Vector{Tuple{Int,Int}}()
    sizehint!(directed_arcs, 2 * n_arcs)
    for (i, j) in arcs
        push!(directed_arcs, (i, j))
        push!(directed_arcs, (j, i))
    end

    distances = Dict{Tuple{Int,Int},Float64}()
    installation_costs = Dict{Tuple{Int,Int},Float64}()
    link_capacities = Dict{Tuple{Int,Int},Float64}()
    flow_costs = Dict{Tuple{Int,Int},Float64}()
    for (a, (i, j)) in enumerate(arcs)
        distances[(i, j)] = link_distance[a]
        distances[(j, i)] = link_distance[a]
        installation_costs[(i, j)] = install_cost[a]
        link_capacities[(i, j)] = capacity[a]
        flow_costs[(i, j)] = link_flow_cost[a]
        flow_costs[(j, i)] = link_flow_cost[a]
    end

    outgoing_arcs = Dict{Int,Vector{Tuple{Int,Int}}}()
    incoming_arcs = Dict{Int,Vector{Tuple{Int,Int}}}()
    for node in 1:n_nodes
        outgoing_arcs[node] = Tuple{Int,Int}[]
        incoming_arcs[node] = Tuple{Int,Int}[]
    end
    for arc in directed_arcs
        push!(outgoing_arcs[arc[1]], arc)
        push!(incoming_arcs[arc[2]], arc)
    end

    return TelecomNetworkDesignProblem(
        n_nodes, n_arcs, n_commodities,
        arcs, directed_arcs,
        node_locations, distances,
        installation_costs, link_capacities, flow_costs,
        commodities, budget,
        outgoing_arcs, incoming_arcs,
        total_demand, routable_scale, cut_scale, nominal_cost,
        witness, certificate, feasibility_status,
    )
end

"""
    build_model(prob::TelecomNetworkDesignProblem)

Build a JuMP model for the telecommunication network design problem.

# Arguments
- `prob`: TelecomNetworkDesignProblem instance

# Returns
- `model`: The JuMP model

# Model Details
Variables:
    - y[arc]: Binary variable, 1 if link is installed on arc
    - f[k,(i,j)]: Continuous flow variable for commodity k on directed arc (i → j)

Objective:
    - Minimize: installation costs + routing costs

Constraints:
    - Flow conservation: at each node, inflow = outflow (except source/sink)
    - Capacity: total flow in both directions on a link ≤ installed capacity * y[arc]
    - Demand satisfaction: each commodity routed from source to destination
    - Budget: total installation cost ≤ budget
"""
function build_model(prob::TelecomNetworkDesignProblem)
    model = Model()

    # Decision variables
    @variable(model, y[arc in prob.arcs], Bin)  # 1 if link is installed
    @variable(model, f[k=1:prob.n_commodities, arc in prob.directed_arcs] >= 0)  # flow of commodity k on directed arc

    # Objective: Minimize total cost (installation + routing)
    @objective(model, Min,
        sum(prob.installation_costs[arc] * y[arc] for arc in prob.arcs) +
        sum(prob.flow_costs[arc] * sum(f[k, arc] for k in 1:prob.n_commodities) for arc in prob.directed_arcs)
    )

    # Flow conservation constraints for each commodity at each node
    for k in 1:prob.n_commodities
        commodity = prob.commodities[k]
        source = commodity[:source]
        sink = commodity[:sink]
        demand = commodity[:demand]

        for node in 1:prob.n_nodes
            # Arcs leaving and entering the node (directed)
            out_arcs = prob.outgoing_arcs[node]
            in_arcs = prob.incoming_arcs[node]

            # Flow balance
            out_flow = isempty(out_arcs) ? 0.0 : sum(f[k, arc] for arc in out_arcs)
            in_flow = isempty(in_arcs) ? 0.0 : sum(f[k, arc] for arc in in_arcs)

            if node == source
                @constraint(model, out_flow - in_flow == demand)
            elseif node == sink
                @constraint(model, out_flow - in_flow == -demand)
            else
                @constraint(model, out_flow - in_flow == 0)
            end
        end
    end

    # Capacity constraints: total flow on each physical link (both directions) ≤ capacity if installed
    for arc in prob.arcs
        forward_arc = arc
        reverse_arc = (arc[2], arc[1])
        @constraint(model,
            sum(f[k, forward_arc] + f[k, reverse_arc] for k in 1:prob.n_commodities) <= prob.link_capacities[arc] * y[arc]
        )
    end

    # Budget constraint
    @constraint(model,
        sum(prob.installation_costs[arc] * y[arc] for arc in prob.arcs) <= prob.budget
    )

    return model
end

# Register the variant
register_variant(
    :telecom_network_design,
    :standard,
    TelecomNetworkDesignProblem,
    "Telecommunication network design problem that minimizes installation and routing costs while satisfying capacity constraints and traffic demands",
)
