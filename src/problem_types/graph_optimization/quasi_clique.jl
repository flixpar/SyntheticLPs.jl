using JuMP
using Random

"""
    QuasiCliqueWitness

A planted dense community: `vertices` (exactly `selected_vertices` of them)
induce the edges `edges` (indices into the problem's edge list), with
`length(edges) >= required_edges`. Setting `x = 1` on `vertices` and `y = 1` on
`edges` satisfies every row.
"""
struct QuasiCliqueWitness
    vertices::Vector{Int}
    edges::Vector{Int}
end

"""
    DegeneracyBoundCertificate

LP-valid upper bound on the number of activated edges. Every edge `e` is
charged to one endpoint `head[e]` (the endpoint removed first by minimum-degree
peeling), and `y_e <= x_{head[e]}` is one of the model's linking rows. Summing
them gives `sum(y) <= sum_v load[v] x_v`, and with `0 <= x <= 1`,
`sum(x) == k` the right side is at most the sum of the `k` largest loads,
`bound`. Peeling keeps every load at most the graph's degeneracy, so the bound
is tight enough to refute a density request `required_edges > bound` that
aggregates thousands of rows — presolve does not detect it.
"""
struct DegeneracyBoundCertificate
    head::Vector{Int}
    load::Vector{Int}
    bound::Int
end

"""
    QuasiCliqueProblem <: ProblemGenerator

Densest-`k`-subgraph / γ-quasi-clique community extraction on a social or
protein-interaction network with heavy-tailed community sizes and hub vertices:
select exactly `k` members that together carry at least `required_edges`
internal interactions (density `γ = required_edges / C(k,2)`), maximizing member
relevance plus interaction strength.

# Formulation

    max  sum_v w_v x_v + sum_e s_e y_e
    s.t. sum_v x_v == k
         y_e <= x_u,  y_e <= x_v       for every edge e = (u, v) of the network
         sum_e y_e >= required_edges
         x, y binary

Only real edges carry a `y` variable (absent pairs would be dead columns).
Variables: `n + m`, exactly `target_variables` for targets of at least 20.

# Feasibility

  - `feasible`: a hidden community of `k` members with internal density at least
    `γ + 0.05` is planted (`feasible_witness`).
  - `infeasible`: no hidden community; `required_edges` exceeds the degeneracy
    bound of [`DegeneracyBoundCertificate`](@ref).
  - `unknown`: no hidden community; `required_edges` lies between the greedy
    peeling value (an integral solution) and the degeneracy bound.
"""
struct QuasiCliqueProblem <: ProblemGenerator
    n_vertices::Int
    edges::Vector{Tuple{Int, Int}}
    vertex_weights::Vector{Float64}
    edge_weights::Vector{Float64}
    selected_vertices::Int
    required_edges::Int
    feasible_witness::Union{Nothing, QuasiCliqueWitness}
    infeasibility_certificate::Union{Nothing, DegeneracyBoundCertificate}
end

function QuasiCliqueProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 20 || throw(ArgumentError("quasi clique needs at least 20 variables"))
    rng = MersenneTwister(seed)
    avg_degree = 6.0 + 4.0 * rand(rng)
    n = max(8, round(Int, target_variables / (1 + avg_degree / 2)))
    m = target_variables - n
    while m > n * (n - 1) ÷ 4      # keep the graph sparse at tiny targets
        n += 1
        m = target_variables - n
    end
    k = clamp(round(Int, (3.0 + 2.0 * rand(rng)) * avg_degree), 4, max(4, n ÷ 3))
    gamma = 0.5 + 0.3 * rand(rng)

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        members = sort!(randperm(rng, n)[1:k])
        pairs = k * (k - 1) ÷ 2
        required = max(1, ceil(Int, gamma * pairs))
        planted_edges = min(pairs, m, ceil(Int, (gamma + 0.05 + 0.1 * rand(rng)) * pairs))
        required = min(required, planted_edges)
        edges = _graph_community_edges(rng, n, m; planted=members, planted_edges=planted_edges)
        in_members = falses(n)
        in_members[members] .= true
        internal = [e for (e, (u, v)) in enumerate(edges) if in_members[u] && in_members[v]]
        witness = QuasiCliqueWitness(members, internal)
    else
        edges = _graph_community_edges(rng, n, m)
        adj = _graph_adjacency(n, edges)
        order, rank, _ = _graph_peeling(n, adj)
        head = [rank[u] < rank[v] ? u : v for (u, v) in edges]
        load = zeros(Int, n)
        foreach(h -> load[h] += 1, head)
        sorted_load = sort(load; rev=true)
        # Grow k until the degeneracy bound leaves room for a sub-unit density.
        while k < n && sum(@view sorted_load[1:k]) > 0.85 * (k * (k - 1) ÷ 2)
            k += 1
        end
        bound = sum(@view sorted_load[1:k])
        if feasibility_status == infeasible
            required = bound + max(1, ceil(Int, 0.03 * bound))
            certificate = DegeneracyBoundCertificate(head, load, bound)
        else
            peeled = falses(n)
            peeled[order[(n - k + 1):n]] .= true
            greedy = count(e -> peeled[e[1]] && peeled[e[2]], edges)
            required = max(1, _graph_floor_between(rng, greedy, bound; reach=0.8))
        end
    end

    vertex_weights = _graph_lognormal_weights(rng, n; median=10.0, sigma=0.7)
    edge_weights = _graph_lognormal_weights(rng, length(edges); median=2.0, sigma=0.5)
    return QuasiCliqueProblem(
        n, edges, vertex_weights, edge_weights, k, required, witness, certificate
    )
end

function build_model(prob::QuasiCliqueProblem)
    model = Model()
    n = prob.n_vertices
    m = length(prob.edges)
    @variable(model, x[1:n], Bin)
    @variable(model, y[1:m], Bin)
    @objective(
        model,
        Max,
        sum(prob.vertex_weights[v] * x[v] for v in 1:n) +
            sum(prob.edge_weights[e] * y[e] for e in 1:m),
    )
    @constraint(model, sum(x) == prob.selected_vertices)
    for (e, (u, v)) in enumerate(prob.edges)
        @constraint(model, y[e] <= x[u])
        @constraint(model, y[e] <= x[v])
    end
    @constraint(model, sum(y) >= prob.required_edges)
    return model
end

register_variant(
    :graph_optimization,
    :quasi_clique,
    QuasiCliqueProblem,
    "Densest-k-subgraph / quasi-clique community extraction on a heavy-tailed community network",
)
