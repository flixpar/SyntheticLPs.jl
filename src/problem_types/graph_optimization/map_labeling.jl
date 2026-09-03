using JuMP
using Random

"""
    MapLabelingProblem <: ProblemGenerator

Maximum-weight map labeling: each feature has several candidate label positions,
at most one position is chosen, and geometrically conflicting candidates cannot
both be used.
"""
struct MapLabelingProblem <: ProblemGenerator
    n_features::Int
    feature_candidates::Vector{Vector{Int}}
    conflicts::Vector{Tuple{Int, Int}}
    label_values::Vector{Float64}
    minimum_placed::Int
end

function MapLabelingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 8 || throw(ArgumentError("map labeling needs at least 8 variables"))
    rng = MersenneTwister(seed)
    n_candidates = target_variables
    n_features = max(2, n_candidates ÷ 4)

    feature_candidates = [Int[] for _ in 1:n_features]
    for candidate in 1:n_candidates
        push!(feature_candidates[mod(candidate - 1, n_features) + 1], candidate)
    end
    planted = Set(first(candidates) for candidates in feature_candidates)
    forbidden = Set{Tuple{Int, Int}}()
    for candidates in feature_candidates
        for a in 1:(length(candidates) - 1), b in (a + 1):length(candidates)
            push!(forbidden, (candidates[a], candidates[b]))
        end
    end
    maximum =
        n_candidates * (n_candidates - 1) ÷ 2 - length(forbidden) -
        length(planted) * (length(planted) - 1) ÷ 2
    conflict_count = min(maximum, max(n_features, 2n_candidates))
    conflicts = _graph_sample_edges(
        rng, n_candidates, conflict_count; forbidden=forbidden, planted_independent=planted
    )
    label_values = _graph_weights(rng, n_candidates; low=10, high=100)

    minimum_placed = if feasibility_status == feasible
        n_features
    elseif feasibility_status == infeasible
        n_features + 1
    else
        0
    end
    return MapLabelingProblem(
        n_features, feature_candidates, conflicts, label_values, minimum_placed
    )
end

function build_model(prob::MapLabelingProblem)
    model = Model()
    n_candidates = length(prob.label_values)
    @variable(model, place[1:n_candidates], Bin)
    @objective(model, Max, sum(prob.label_values[j] * place[j] for j in 1:n_candidates))
    for candidates in prob.feature_candidates
        @constraint(model, sum(place[j] for j in candidates) <= 1)
    end
    for (a, b) in prob.conflicts
        @constraint(model, place[a] + place[b] <= 1)
    end
    if prob.minimum_placed > 0
        @constraint(model, sum(place) >= prob.minimum_placed)
    end
    return model
end

register_variant(
    :graph_optimization,
    :map_labeling,
    MapLabelingProblem,
    "Maximum-weight conflict-free map labeling over candidate label positions",
)
