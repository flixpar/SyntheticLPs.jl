using JuMP
using Random

"""
    QuasiCliqueProblem <: ProblemGenerator

Fixed-cardinality quasi-clique selection with binary vertex variables and binary
edge-activation variables. A planted clique supplies the feasible witness.
"""
struct QuasiCliqueProblem <: ProblemGenerator
    n_vertices::Int
    candidate_edges::Vector{Tuple{Int,Int}}
    edge_present::Vector{Bool}
    vertex_weights::Vector{Float64}
    selected_vertices::Int
    required_edges::Int
end

function QuasiCliqueProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    target_variables >= 10 || throw(ArgumentError("quasi clique needs at least 10 variables"))
    rng = MersenneTwister(seed)
    n = max(4, round(Int, 0.6 * target_variables))
    while target_variables - n > n * (n - 1) ÷ 2
        n += 1
    end
    n_edge_variables = target_variables - n

    k = max(2, floor(Int, sqrt(2 * n_edge_variables)))
    while k * (k - 1) ÷ 2 > n_edge_variables
        k -= 1
    end
    k = min(k, n)
    planted_vertices = sort!(randperm(rng, n)[1:k])
    planted_edges = Tuple{Int,Int}[]
    for a in 1:(k - 1), b in (a + 1):k
        push!(planted_edges, (planted_vertices[a], planted_vertices[b]))
    end
    planted_set = Set(planted_edges)
    remaining = _graph_sample_edges(
        rng, n, n_edge_variables - length(planted_edges); forbidden=planted_set,
    )
    candidate_edges = sort!(vcat(planted_edges, remaining))

    density = 0.75
    required_edges = max(1, ceil(Int, density * (k * (k - 1) ÷ 2)))
    edge_present = if feasibility_status == infeasible
        falses(n_edge_variables)
    else
        present = [rand(rng) < 0.55 for _ in 1:n_edge_variables]
        if feasibility_status == feasible
            planted_lookup = Set(planted_edges)
            for (e, edge) in enumerate(candidate_edges)
                edge in planted_lookup && (present[e] = true)
            end
        end
        present
    end
    weights = _graph_weights(rng, n; low=1, high=50)
    return QuasiCliqueProblem(
        n, candidate_edges, edge_present, weights, k, required_edges,
    )
end

function build_model(prob::QuasiCliqueProblem)
    model = Model()
    @variable(model, x[1:prob.n_vertices], Bin)
    @variable(model, y[1:length(prob.candidate_edges)], Bin)
    @objective(
        model,
        Max,
        sum(prob.vertex_weights[v] * x[v] for v in 1:prob.n_vertices) + sum(y),
    )
    @constraint(model, sum(x) == prob.selected_vertices)
    for (e, (u, v)) in enumerate(prob.candidate_edges)
        @constraint(model, y[e] <= x[u])
        @constraint(model, y[e] <= x[v])
        @constraint(model, y[e] <= Int(prob.edge_present[e]))
    end
    @constraint(model, sum(y) >= prob.required_edges)
    return model
end

register_variant(
    :graph_optimization,
    :quasi_clique,
    QuasiCliqueProblem,
    "Fixed-cardinality quasi-clique selection with explicit edge activations",
)
