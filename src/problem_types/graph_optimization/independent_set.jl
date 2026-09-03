using JuMP
using Random

"""
    IndependentSetProblem <: ProblemGenerator

Weighted maximum independent set with one binary variable per vertex. Sparse
feasible instances contain a planted independent set; infeasible instances use
a matching-derived cardinality certificate that also excludes the LP relaxation.
"""
struct IndependentSetProblem <: ProblemGenerator
    n_vertices::Int
    edges::Vector{Tuple{Int, Int}}
    weights::Vector{Float64}
    minimum_selected::Int
end

function IndependentSetProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 2 || throw(ArgumentError("independent set needs at least 2 variables"))
    rng = MersenneTwister(seed)
    n = target_variables
    weights = _graph_weights(rng, n)

    if feasibility_status == infeasible
        # A maximum matching gives sum(x) <= ceil(n/2), including a possible
        # unmatched vertex. The requested floor is therefore impossible even
        # for continuous x in [0,1].
        edges = [(i, i + 1) for i in 1:2:(n - 1)]
        minimum_selected = cld(n, 2) + 1
    else
        planted_size = max(2, round(Int, 0.15 * n))
        planted = Set(randperm(rng, n)[1:planted_size])
        average_degree = rand(rng, 3:8)
        edge_count = min(
            n * (n - 1) ÷ 2 - planted_size * (planted_size - 1) ÷ 2,
            max(n - 1, round(Int, average_degree * n / 2)),
        )
        edges = _graph_sample_edges(rng, n, edge_count; planted_independent=planted)
        minimum_selected = feasibility_status == feasible ? planted_size : 0
    end

    return IndependentSetProblem(n, edges, weights, minimum_selected)
end

function build_model(prob::IndependentSetProblem)
    model = Model()
    @variable(model, x[1:prob.n_vertices], Bin)
    @objective(model, Max, sum(prob.weights[v] * x[v] for v in 1:prob.n_vertices))
    for (u, v) in prob.edges
        @constraint(model, x[u] + x[v] <= 1)
    end
    if prob.minimum_selected > 0
        @constraint(model, sum(x) >= prob.minimum_selected)
    end
    return model
end

register_variant(
    :graph_optimization,
    :independent_set,
    IndependentSetProblem,
    "Weighted maximum independent set on sparse graphs with a planted independent set";
    default=true,
)
