using JuMP
using Random

"""
    SetPackingProblem <: ProblemGenerator

Maximum-value set packing. The leading columns form a planted collection of
pairwise-disjoint sets; infeasible instances share one common element and impose
a contradictory selection floor.
"""
struct SetPackingProblem <: ProblemGenerator
    n_elements::Int
    columns::Vector{Vector{Int}}
    values::Vector{Float64}
    minimum_selected::Int
end

function SetPackingProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    target_variables >= 4 || throw(ArgumentError("set packing needs at least 4 variables"))
    rng = MersenneTwister(seed)
    n_columns = target_variables
    n_elements = max(4, round(Int, 0.4 * n_columns))
    columns, n_planted = _set_columns_with_partition(
        rng, n_elements, n_columns; max_size=max(2, min(6, n_elements)),
    )

    if feasibility_status == infeasible
        for column in columns
            push!(column, 1)
            sort!(unique!(column))
        end
        minimum_selected = 2
    elseif feasibility_status == feasible
        minimum_selected = n_planted
    else
        minimum_selected = 0
    end

    values = _set_positive_coefficients(rng, n_columns; low=5, high=120)
    return SetPackingProblem(n_elements, columns, values, minimum_selected)
end

function build_model(prob::SetPackingProblem)
    model = Model()
    n_columns = length(prob.columns)
    incidence = _set_elements_to_columns(prob.columns, prob.n_elements)
    @variable(model, x[1:n_columns], Bin)
    @objective(model, Max, sum(prob.values[j] * x[j] for j in 1:n_columns))
    for i in 1:prob.n_elements
        @constraint(model, sum(x[j] for j in incidence[i]) <= 1)
    end
    if prob.minimum_selected > 0
        @constraint(model, sum(x) >= prob.minimum_selected)
    end
    return model
end

register_variant(
    :set_system,
    :set_packing,
    SetPackingProblem,
    "Maximum-value binary set packing with planted disjoint columns",
)
