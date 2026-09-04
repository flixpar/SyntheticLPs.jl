using JuMP
using Random

"""
    GeneralizedIndependentSetProblem <: ProblemGenerator

Generalized independent set with hard-conflict edges and soft-conflict edges.
Selecting both endpoints of a soft edge activates a binary penalty variable;
the total number of vertex and penalty variables equals `target_variables`.
"""
struct GeneralizedIndependentSetProblem <: ProblemGenerator
    n_vertices::Int
    hard_edges::Vector{Tuple{Int, Int}}
    soft_edges::Vector{Tuple{Int, Int}}
    vertex_benefits::Vector{Float64}
    edge_penalties::Vector{Float64}
    minimum_selected::Int
end

function GeneralizedIndependentSetProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 6 ||
        throw(ArgumentError("generalized independent set needs at least 6 variables"))
    rng = MersenneTwister(seed)

    # Reserve most variables for vertices while retaining a material block of
    # soft-edge activation variables, matching the defining mixed graph model.
    n = max(4, round(Int, 0.6 * target_variables))
    n_soft = target_variables - n
    max_edges = n * (n - 1) ÷ 2
    while n_soft > max_edges && n < target_variables
        n += 1
        n_soft = target_variables - n
        max_edges = n * (n - 1) ÷ 2
    end

    if feasibility_status == infeasible
        hard_edges = [(i, i + 1) for i in 1:2:(n - 1)]
        minimum_selected = cld(n, 2) + 1
    else
        planted_size = max(2, round(Int, 0.15 * n))
        planted = Set(randperm(rng, n)[1:planted_size])
        # Leave at least `n_soft` unused pairs for the penalty variables;
        # otherwise tiny targets (6–10) request more soft edges than remain.
        hard_count = min(
            max_edges - planted_size * (planted_size - 1) ÷ 2, max_edges - n_soft, max(n - 1, 2n)
        )
        hard_count = max(0, hard_count)
        hard_edges = _graph_sample_edges(rng, n, hard_count; planted_independent=planted)
        minimum_selected = feasibility_status == feasible ? planted_size : 0
    end

    hard_set = Set(hard_edges)
    soft_edges = _graph_sample_edges(rng, n, n_soft; forbidden=hard_set)
    vertex_benefits = _graph_weights(rng, n; low=40, high=120)
    edge_penalties = _graph_weights(rng, n_soft; low=5, high=80)

    return GeneralizedIndependentSetProblem(
        n, hard_edges, soft_edges, vertex_benefits, edge_penalties, minimum_selected
    )
end

function build_model(prob::GeneralizedIndependentSetProblem)
    model = Model()
    @variable(model, x[1:prob.n_vertices], Bin)
    @variable(model, y[1:length(prob.soft_edges)], Bin)
    @objective(
        model,
        Max,
        sum(prob.vertex_benefits[v] * x[v] for v in 1:prob.n_vertices) -
            sum(prob.edge_penalties[e] * y[e] for e in eachindex(prob.soft_edges)),
    )
    for (u, v) in prob.hard_edges
        @constraint(model, x[u] + x[v] <= 1)
    end
    for (e, (u, v)) in enumerate(prob.soft_edges)
        @constraint(model, x[u] + x[v] - y[e] <= 1)
    end
    if prob.minimum_selected > 0
        @constraint(model, sum(x) >= prob.minimum_selected)
    end
    return model
end

register_variant(
    :graph_optimization,
    :generalized_independent_set,
    GeneralizedIndependentSetProblem,
    "Generalized independent set with hard conflicts and binary soft-edge penalties",
)
