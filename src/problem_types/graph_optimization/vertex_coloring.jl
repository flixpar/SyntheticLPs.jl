using JuMP
using Random

"""
    VertexColoringWitness

A planted proper list coloring: `channel[v] in domains[v]` for every access
point, and adjacent access points get different channels, so every
clique-channel row holds with at most one selected member.
"""
struct VertexColoringWitness
    channel::Vector{Int}
end

"""
    OvercrowdedCliqueCertificate

An over-crowded venue: the access points of clique row `cliques[clique]` (all
mutually interfering) can only use the channels `channels`, with
`length(channels) < length(cliques[clique])`. Summing the assignment rows of the
clique (each equals one) and the clique-channel packing rows of `channels` (each
at most one) gives `|K| <= |channels|`, so the LP relaxation is infeasible. The
argument aggregates `|K| + |channels|` rows; presolve does not detect it.
"""
struct OvercrowdedCliqueCertificate
    clique::Int
    channels::Vector{Int}
end

"""
    VertexColoringProblem <: ProblemGenerator

Minimum-interference channel assignment for a wireless LAN — a list-coloring
problem. Access points (vertices) within interference range (edges of a
hotspot unit-disk graph) must use different channels; each access point can use
only the channels in its domain (regulatory DFS/indoor restrictions, radio
capabilities), and each (access point, channel) pair has a cost reflecting
measured external interference on that channel at that location.

# Formulation (clique formulation per channel)

    min  sum_{v, c in D_v} cost[v,c] x[v,c]
    s.t. sum_{c in D_v} x[v,c] == 1                for every access point v
         sum_{v in K, c in D_v} x[v,c] <= 1        for every clique K of a greedy
                                                   edge clique cover and channel c
         x binary

This replaces the compact assignment formulation with color-use variables,
whose LP relaxation collapses (the uniform point `x = 1/k` satisfies every edge
row, so the conflict graph is irrelevant to the bound). Here each clique row
binds per channel and the costs break the channel symmetry, so the LP
relaxation depends on both the clique structure and the interference field.

Variables: one per (access point, allowed channel) pair, adjusted to exactly
`target_variables` by adding or removing non-planted domain entries.

# Feasibility

  - `feasible`: a greedy (largest-degree-first) coloring is planted, the channel
    count is its color count plus 0–2, and each domain contains its planted
    channel (`feasible_witness`).
  - `infeasible`: the largest clique row is an over-crowded venue whose members
    share a channel plan with one channel fewer than the clique size — see
    [`OvercrowdedCliqueCertificate`](@ref).
  - `unknown`: no planted coloring; the channel count is the greedy color count
    minus 2 to plus 1 and domains are random 55–85% subsets, so list
    colorability (and its LP relaxation) is not decided at generation time.
"""
struct VertexColoringProblem <: ProblemGenerator
    n_vertices::Int
    n_channels::Int
    xs::Vector{Float64}
    ys::Vector{Float64}
    edges::Vector{Tuple{Int, Int}}
    cliques::Vector{Vector{Int}}
    domains::Vector{Vector{Int}}
    costs::Vector{Vector{Float64}}
    feasible_witness::Union{Nothing, VertexColoringWitness}
    infeasibility_certificate::Union{Nothing, OvercrowdedCliqueCertificate}
end

# Largest-degree-first greedy coloring; returns the color of every vertex.
function _vertex_coloring_greedy(adj::Vector{Vector{Int}})
    n = length(adj)
    color = zeros(Int, n)
    used = falses(n + 1)
    for v in sortperm(length.(adj); rev=true)
        for u in adj[v]
            color[u] > 0 && (used[color[u]] = true)
        end
        c = findfirst(!, used)
        color[v] = c
        for u in adj[v]
            color[u] > 0 && (used[color[u]] = false)
        end
    end
    return color
end

function VertexColoringProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 12 || throw(ArgumentError("vertex coloring needs at least 12 variables"))
    rng = MersenneTwister(seed)
    avg_degree = 4.0 + 4.0 * rand(rng)
    domain_share = 0.55 + 0.3 * rand(rng)
    slack = feasibility_status == unknown ? rand(rng, -2:1) : rand(rng, 0:2)

    # The channel count follows the graph's greedy chromatic number, which in
    # turn depends on n; two sizing passes settle n against the target.
    n = max(4, round(Int, target_variables / 8))
    local xs, ys, edges, adj, planted, k
    for pass in 1:3
        xs, ys, _ = _graph_geometric_points(rng, n, avg_degree)
        edges = _graph_pairs_within(xs, ys, 1.0)
        adj = _graph_adjacency(n, edges)
        planted = _vertex_coloring_greedy(adj)
        k = max(2, maximum(planted) + slack)
        per_vertex = 1 + domain_share * (k - 1)
        n_next = max(4, ceil(Int, target_variables / per_vertex))
        (pass == 3 || abs(n_next - n) <= 0.03 * n) && break
        n = n_next
    end
    # Small graphs need few channels; keep enough (access point, channel)
    # pairs available to reach the target exactly.
    k = max(k, cld(target_variables, n) + 1)
    cliques = _graph_clique_cover(adj, _graph_cell_groups(xs, ys, 1.0))

    # Domains: the planted channel (feasible) plus a random share of the rest.
    domains = Vector{Vector{Int}}(undef, n)
    fixed = falses(n)                     # entries that must not be removed
    for v in 1:n
        if feasibility_status == feasible
            d = [c for c in 1:k if c == planted[v] || rand(rng) < domain_share]
        else
            d = [c for c in 1:k if rand(rng) < domain_share]
            isempty(d) && push!(d, rand(rng, 1:k))
        end
        domains[v] = d
    end
    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        witness = VertexColoringWitness(planted)
    elseif feasibility_status == infeasible
        r = argmax(length.(cliques))
        venue = cliques[r]
        plan = sort!(randperm(rng, max(k, length(venue)))[1:(length(venue) - 1)])
        k = max(k, maximum(plan))
        for v in venue
            domains[v] = copy(plan)
            fixed[v] = true
        end
        certificate = OvercrowdedCliqueCertificate(r, plan)
    end

    # Exact sizing: add or drop non-planted, non-venue domain entries. Make
    # sure the free access points can absorb the target (a large venue fixes
    # its members' domains).
    unfixed = count(!, fixed)
    fixed_total = sum(length(domains[v]) for v in 1:n if fixed[v]; init=0)
    unfixed > 0 && (k = max(k, cld(target_variables - fixed_total, unfixed) + 1))
    total = sum(length, domains)
    attempts = 0
    while total != target_variables && attempts < 200 * target_variables
        attempts += 1
        v = rand(rng, 1:n)
        fixed[v] && continue
        if total < target_variables
            length(domains[v]) < k || continue
            c = rand(rng, 1:k)
            insorted(c, domains[v]) && continue
            insert!(domains[v], searchsortedfirst(domains[v], c), c)
            total += 1
        else
            length(domains[v]) > 1 || continue
            idx = rand(rng, 1:length(domains[v]))
            witness !== nothing && domains[v][idx] == planted[v] && continue
            deleteat!(domains[v], idx)
            total -= 1
        end
    end

    # Interference field: per channel, a few external sources (neighbouring
    # networks, radar) whose impact decays with distance, on top of a channel
    # base level (DFS channels slightly costlier) and small local noise.
    side = maximum(xs; init=1.0)
    costs = Vector{Vector{Float64}}(undef, n)
    sources = [
        [(side * rand(rng), side * rand(rng), 5.0 + 20.0 * rand(rng)) for _ in 1:rand(rng, 1:4)] for
        _ in 1:k
    ]
    base = [1.0 + (c > 0.6k ? 2.0 : 0.0) + 2.0 * rand(rng) for c in 1:k]
    spread = max(2.0, side / 6)
    for v in 1:n
        costs[v] = [
            round(
                base[c] +
                sum(
                    s * exp(-((xs[v] - sx)^2 + (ys[v] - sy)^2) / (2spread^2)) for
                    (sx, sy, s) in sources[c]
                ) +
                rand(rng);
                digits=2,
            ) for c in domains[v]
        ]
    end

    return VertexColoringProblem(n, k, xs, ys, edges, cliques, domains, costs, witness, certificate)
end

function build_model(prob::VertexColoringProblem)
    model = Model()
    n = prob.n_vertices
    offsets = cumsum([0; length.(prob.domains)])
    @variable(model, x[1:offsets[end]], Bin)
    @objective(
        model,
        Min,
        sum(prob.costs[v][i] * x[offsets[v] + i] for v in 1:n for i in eachindex(prob.domains[v])),
    )
    for v in 1:n
        @constraint(model, sum(x[(offsets[v] + 1):offsets[v + 1]]) == 1)
    end
    members = [Int[] for _ in 1:prob.n_channels]
    for clique in prob.cliques
        foreach(empty!, members)
        for v in clique, (i, c) in enumerate(prob.domains[v])
            push!(members[c], offsets[v] + i)
        end
        for c in 1:prob.n_channels
            length(members[c]) >= 2 && @constraint(model, sum(x[j] for j in members[c]) <= 1)
        end
    end
    return model
end

register_variant(
    :graph_optimization,
    :vertex_coloring,
    VertexColoringProblem,
    "Minimum-interference WLAN channel assignment (list coloring) with per-channel clique rows on a hotspot unit-disk graph";
    tags=[:telecom, :partitioning, :packing],
    min_target_variables=12,
)
