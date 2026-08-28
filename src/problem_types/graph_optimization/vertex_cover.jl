using JuMP
using Random

"""
    VertexCoverProblem <: ProblemGenerator

Weighted minimum vertex cover with one binary variable per vertex. Feasible
instances have a planted cover; infeasible instances combine a disjoint matching
with a cover budget below the matching lower bound.
"""
struct VertexCoverProblem <: ProblemGenerator
    n_vertices::Int
    edges::Vector{Tuple{Int,Int}}
    costs::Vector{Float64}
    maximum_selected::Int
end

function VertexCoverProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    target_variables >= 2 || throw(ArgumentError("vertex cover needs at least 2 variables"))
    rng = MersenneTwister(seed)
    n = target_variables
    costs = _graph_weights(rng, n)

    if feasibility_status == infeasible
        edges = [(i, i + 1) for i in 1:2:(n - 1)]
        maximum_selected = length(edges) - 1
    else
        cover_size = max(1, round(Int, 0.4 * n))
        cover = Set(randperm(rng, n)[1:cover_size])
        candidates = Tuple{Int,Int}[]
        for u in 1:(n - 1), v in (u + 1):n
            (u in cover || v in cover) && push!(candidates, (u, v))
        end
        shuffle!(rng, candidates)
        average_degree = rand(rng, 3:8)
        edge_count = min(length(candidates), max(n - 1, round(Int, average_degree * n / 2)))
        edges = sort!(candidates[1:edge_count])
        maximum_selected = feasibility_status == feasible ? cover_size : n
    end

    return VertexCoverProblem(n, edges, costs, maximum_selected)
end

function build_model(prob::VertexCoverProblem)
    model = Model()
    @variable(model, x[1:prob.n_vertices], Bin)
    @objective(model, Min, sum(prob.costs[v] * x[v] for v in 1:prob.n_vertices))
    for (u, v) in prob.edges
        @constraint(model, x[u] + x[v] >= 1)
    end
    if prob.maximum_selected < prob.n_vertices
        @constraint(model, sum(x) <= prob.maximum_selected)
    end
    return model
end

register_variant(
    :graph_optimization,
    :vertex_cover,
    VertexCoverProblem,
    "Weighted minimum vertex cover on sparse graphs with planted-cover generation",
)
