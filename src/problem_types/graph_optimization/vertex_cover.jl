using JuMP
using Random

"""
    VertexCoverWitness

A planted capacitated cover: link `e = edges[e]` is monitored from endpoint
`monitor[e]`; the probes installed are `installed = unique(monitor)`, and every
vertex monitors at most `capacity[v]` links.
"""
struct VertexCoverWitness
    monitor::Vector{Int}
end

"""
    VertexCoverDeficitCertificate

Hakimi-type capacity deficit: the links induced by `core` must each be
monitored from one of their two endpoints, both inside `core`, so summing the
capacity rows of `core` (each link of `core_edges` contributes `z_e + (1 - z_e)
= 1` to them) gives
`length(core_edges) <= sum(capacity[core]) == capacity_total`, using LP rows and
`x <= 1` only. `length(core_edges) > capacity_total` makes even the LP relaxation
infeasible; the argument aggregates hundreds of rows, so presolve does not see it.
"""
struct VertexCoverDeficitCertificate
    core::Vector{Int}
    core_edges::Vector{Int}
    capacity_total::Int
end

"""
    VertexCoverProblem <: ProblemGenerator

Capacitated vertex cover — placing link-monitoring probes on a scale-free
communication network (router-level / peering topology). Every link must be
monitored by a probe installed at one of its endpoints, and the probe at vertex
`v` can watch at most `capacity[v]` links (ports on the monitoring card).

# Formulation

    min  sum_v c_v x_v
    s.t. z_e <= x_u,  1 - z_e <= x_v                  for every link e = (u, v)
         sum_{e=(v,·)} z_e + sum_{e=(·,v)} (1 - z_e) <= cap_v x_v   for every vertex v
         x binary, 0 <= z <= 1

`z_e` is the share of link `e = (u, v)` monitored from `u` (the rest from `v`);
a single orientation variable per link avoids the doubleton assignment
equalities `y_eu + y_ev = 1` that presolve would immediately substitute out.

The capacities make this structurally different from (uncapacitated) vertex
cover, which is independent set under `x -> 1 - x`: the LP couples a
b-matching/orientation polytope with fixed-charge capacity linking, so the
relaxation stays meaningful (hubs exceed their capacity and push load onto
their neighbours).

Variables: `n + m` (`n` vertices, `m` links), exactly `target_variables` for
`target_variables >= 6`.

# Feasibility

  - `feasible`: links are oriented greedily toward the endpoint with more spare
    capacity, upgrading the cheaper endpoint's card when both are full; the
    orientation is the `feasible_witness`.
  - `infeasible`: the high-degree core (rich club) runs legacy cards whose total
    capacity is at least 10% below the number of links among core vertices —
    see [`VertexCoverDeficitCertificate`](@ref).
  - `unknown`: capacities are drawn naturally (a random 25–75% of each degree);
    feasibility hinges on whether some dense subgraph is over-subscribed
    (Hakimi's orientation theorem), which is not decided at generation time.
"""
struct VertexCoverProblem <: ProblemGenerator
    n_vertices::Int
    edges::Vector{Tuple{Int, Int}}
    costs::Vector{Float64}
    capacity::Vector{Int}
    feasible_witness::Union{Nothing, VertexCoverWitness}
    infeasibility_certificate::Union{Nothing, VertexCoverDeficitCertificate}
end

# Number of vertices and edges with n + m == target and average degree
# ~avg_degree, keeping the graph connected and simple.
function _vertex_cover_dimensions(target::Int, avg_degree::Float64)
    n = max(3, round(Int, target / (1 + avg_degree / 2)))
    m = target - n
    while m < n - 1 && n > 3
        n -= 1
        m = target - n
    end
    while m > n * (n - 1) ÷ 2
        n += 1
        m = target - n
    end
    return n, m
end

function VertexCoverProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 6 ||
        throw(ArgumentError("capacitated vertex cover needs at least 6 variables"))
    rng = MersenneTwister(seed)
    avg_degree = 4.0 + 4.0 * rand(rng)
    n, m = _vertex_cover_dimensions(target_variables, avg_degree)
    edges = _graph_preferential_attachment(rng, n, m)
    adj = _graph_adjacency(n, edges)
    degree = length.(adj)

    # Natural card capacities: a random 25–75% of the vertex's links.
    capacity = [clamp(round(Int, (0.25 + 0.5 * rand(rng)) * degree[v]), 1, degree[v]) for v in 1:n]

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        load = zeros(Int, n)
        monitor = zeros(Int, length(edges))
        for e in randperm(rng, length(edges))
            u, v = edges[e]
            su, sv = capacity[u] - load[u], capacity[v] - load[v]
            w = if su > 0 || sv > 0
                su >= sv ? u : v
            else
                # Both cards full: upgrade the endpoint with spare ports left.
                w = degree[u] - capacity[u] >= degree[v] - capacity[v] ? u : v
                capacity[w] += 1
                w
            end
            load[w] += 1
            monitor[e] = w
        end
        witness = VertexCoverWitness(monitor)
    elseif feasibility_status == infeasible
        # Rich club: the top 5% of vertices by degree (at least 4).
        h = clamp(round(Int, 0.05 * n), 4, n)
        core = sort!(sortperm(degree; rev=true)[1:h])
        in_core = falses(n)
        in_core[core] .= true
        core_edges = [e for (e, (u, v)) in enumerate(edges) if in_core[u] && in_core[v]]
        if length(core_edges) < 2h
            # Sparse club (small graphs): the whole network is the deficit set.
            core = collect(1:n)
            in_core .= true
            core_edges = collect(eachindex(edges))
        end
        core_degree = zeros(Int, n)
        for e in core_edges
            u, v = edges[e]
            core_degree[u] += 1
            core_degree[v] += 1
        end
        # Legacy cards: total core capacity at least 10% below the core's links.
        budget = floor(Int, 0.9 * length(core_edges))
        scale = budget / (2 * length(core_edges))
        for v in core
            capacity[v] = max(1, floor(Int, scale * core_degree[v] * (0.8 + 0.4 * rand(rng))))
        end
        total = sum(capacity[core])
        # Trim the largest cards until the deficit holds.
        while total > budget
            v = core[argmax(capacity[core])]
            capacity[v] -= 1
            total -= 1
        end
        certificate = VertexCoverDeficitCertificate(core, core_edges, total)
    end

    # Probe cost: installation plus a per-port card cost, with site variation.
    costs = round.(
        [(60.0 + 12.0 * capacity[v]) * exp(0.3 * randn(rng)) for v in 1:n]; digits=2
    )
    return VertexCoverProblem(n, edges, costs, capacity, witness, certificate)
end

function build_model(prob::VertexCoverProblem)
    model = Model()
    n = prob.n_vertices
    m = length(prob.edges)
    @variable(model, x[1:n], Bin)
    # z[e] = share of link e = (u, v) monitored from u; 1 - z[e] from v.
    @variable(model, 0 <= z[1:m] <= 1)
    @objective(model, Min, sum(prob.costs[v] * x[v] for v in 1:n))
    from_tail = [Int[] for _ in 1:n]
    from_head = [Int[] for _ in 1:n]
    for (e, (u, v)) in enumerate(prob.edges)
        @constraint(model, z[e] <= x[u])
        @constraint(model, x[v] + z[e] >= 1)
        push!(from_tail[u], e)
        push!(from_head[v], e)
    end
    for v in 1:n
        # Links monitored at v: z[e] on links where v is the tail, 1 - z[e]
        # where it is the head.
        @constraint(
            model,
            sum(z[e] for e in from_tail[v]; init=0.0) - sum(z[e] for e in from_head[v]; init=0.0) -
            prob.capacity[v] * x[v] <= -length(from_head[v])
        )
    end
    return model
end

register_variant(
    :graph_optimization,
    :vertex_cover,
    VertexCoverProblem,
    "Capacitated vertex cover: link-monitoring probe placement with port capacities on a scale-free network",
)
