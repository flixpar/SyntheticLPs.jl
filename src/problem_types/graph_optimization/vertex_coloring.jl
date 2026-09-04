using JuMP
using Random

"""
    VertexColoringProblem <: ProblemGenerator

Minimum-color graph coloring with binary vertex-color assignments and color-use
variables. Generated graphs have a planted proper coloring.
"""
struct VertexColoringProblem <: ProblemGenerator
    n_vertices::Int
    n_colors::Int
    edges::Vector{Tuple{Int, Int}}
    color_costs::Vector{Float64}
    assignment_limit::Int
end

function VertexColoringProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 12 || throw(ArgumentError("vertex coloring needs at least 12 variables"))
    rng = MersenneTwister(seed)

    # Choose dimensions whose n*k assignment variables plus k color-use
    # variables are closest to the requested total.
    best = (typemax(Int), 0, 0)
    for k in 3:min(8, target_variables ÷ 3)
        n = max(k + 1, round(Int, target_variables / k) - 1)
        error = abs(k * n + k - target_variables)
        error < best[1] && (best = (error, n, k))
    end
    _, n, k = best

    planted_colors = [mod(i - 1, k) + 1 for i in randperm(rng, n)]
    planted_sets = [Set(findall(==(c), planted_colors)) for c in 1:k]
    forbidden = Set{Tuple{Int, Int}}()
    for color_class in planted_sets
        members = sort!(collect(color_class))
        for a in 1:(length(members) - 1), b in (a + 1):length(members)
            push!(forbidden, (members[a], members[b]))
        end
    end
    maximum_edges = n * (n - 1) ÷ 2 - length(forbidden)
    edge_count = min(maximum_edges, max(n - 1, round(Int, rand(rng, 3:8) * n / 2)))
    edges = _graph_sample_edges(rng, n, edge_count; forbidden=forbidden)
    costs = Float64.(1:k)

    # Every vertex-assignment row sums to one, so this aggregate limit is an
    # LP-valid infeasibility certificate when set below n.
    assignment_limit = feasibility_status == infeasible ? n - 1 : n
    return VertexColoringProblem(n, k, edges, costs, assignment_limit)
end

function build_model(prob::VertexColoringProblem)
    model = Model()
    @variable(model, assign[1:prob.n_vertices, 1:prob.n_colors], Bin)
    @variable(model, used[1:prob.n_colors], Bin)
    @objective(model, Min, sum(prob.color_costs[c] * used[c] for c in 1:prob.n_colors))
    for v in 1:prob.n_vertices
        @constraint(model, sum(assign[v, c] for c in 1:prob.n_colors) == 1)
        for c in 1:prob.n_colors
            @constraint(model, assign[v, c] <= used[c])
        end
    end
    for (u, v) in prob.edges, c in 1:prob.n_colors
        @constraint(model, assign[u, c] + assign[v, c] <= 1)
    end
    for c in 1:(prob.n_colors - 1)
        @constraint(model, used[c] >= used[c + 1])
    end
    if prob.assignment_limit < prob.n_vertices
        @constraint(model, sum(assign) <= prob.assignment_limit)
    end
    return model
end

register_variant(
    :graph_optimization,
    :vertex_coloring,
    VertexColoringProblem,
    "Minimum-color binary vertex coloring on graphs with a planted proper coloring",
)
