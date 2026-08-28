using JuMP
using Random

"""
    SetPartitioningProblem <: ProblemGenerator

Minimum-cost exact set partitioning over a generic sparse incidence matrix. A
planted partition proves feasibility; a coverage-count bound certifies requested
infeasibility in both the binary model and its LP relaxation.
"""
struct SetPartitioningProblem <: ProblemGenerator
    n_elements::Int
    columns::Vector{Vector{Int}}
    costs::Vector{Float64}
    maximum_selected::Int
end

function SetPartitioningProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    target_variables >= 4 ||
        throw(ArgumentError("set partitioning needs at least 4 variables"))
    rng = MersenneTwister(seed)
    n_columns = target_variables
    n_elements = max(4, round(Int, 0.4 * n_columns))
    max_size = max(2, min(6, round(Int, sqrt(n_elements)) + 1))
    columns, n_planted = _set_columns_with_partition(
        rng, n_elements, n_columns; max_size=max_size,
    )

    maximum_selected = if feasibility_status == infeasible
        # Summing exact-cover rows gives
        #   n_elements = sum_j |S_j|x_j <= max_size * sum_j x_j.
        # The strict cardinality cap therefore rules out even fractional x.
        cld(n_elements, max_size) - 1
    elseif feasibility_status == feasible
        n_planted
    else
        n_columns
    end

    costs = _set_positive_coefficients(rng, n_columns; low=5, high=100)
    return SetPartitioningProblem(n_elements, columns, costs, maximum_selected)
end

function build_model(prob::SetPartitioningProblem)
    model = Model()
    n_columns = length(prob.columns)
    incidence = _set_elements_to_columns(prob.columns, prob.n_elements)
    @variable(model, x[1:n_columns], Bin)
    @objective(model, Min, sum(prob.costs[j] * x[j] for j in 1:n_columns))
    for i in 1:prob.n_elements
        @constraint(model, sum(x[j] for j in incidence[i]) == 1)
    end
    if prob.maximum_selected < n_columns
        @constraint(model, sum(x) <= prob.maximum_selected)
    end
    return model
end

register_variant(
    :set_system,
    :set_partitioning,
    SetPartitioningProblem,
    "Minimum-cost generic exact set partitioning with a planted partition",
)
