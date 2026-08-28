using JuMP
using Random

"""
    SetCoverProblem <: ProblemGenerator

Minimum-cost set cover with nonempty rows, mixed short and heavy-tailed columns,
and an exact-partition cover planted in the generated set system.
"""
struct SetCoverProblem <: ProblemGenerator
    n_elements::Int
    columns::Vector{Vector{Int}}
    costs::Vector{Float64}
    maximum_selected::Int
end

function SetCoverProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = MersenneTwister(seed)
    n_columns, n_elements = _set_system_size(target_variables, 0.35)
    max_size = max(2, min(n_elements, round(Int, sqrt(n_elements)) + 2))
    columns, n_planted = _set_columns_with_partition(
        rng, n_elements, n_columns; max_size=max_size,
    )

    if feasibility_status == infeasible
        # Give each of k private rows exactly one distinct covering column.
        # Those rows force k distinct x values to one, while the cardinality
        # budget permits only k-1, including in the LP relaxation.
        k = min(n_elements - 1, n_columns, max(2, round(Int, 0.12 * n_columns)))
        private_rows = Set(1:k)
        fallback_row = k + 1
        for j in eachindex(columns)
            filter!(i -> !(i in private_rows), columns[j])
            isempty(columns[j]) && push!(columns[j], fallback_row)
        end
        for i in 1:k
            push!(columns[i], i)
            sort!(unique!(columns[i]))
        end
        maximum_selected = k - 1
    elseif feasibility_status == feasible
        maximum_selected = n_planted
    else
        maximum_selected = n_columns
    end

    costs = _set_positive_coefficients(rng, n_columns; low=5, high=100)
    return SetCoverProblem(n_elements, columns, costs, maximum_selected)
end

function build_model(prob::SetCoverProblem)
    model = Model()
    n_columns = length(prob.columns)
    incidence = _set_elements_to_columns(prob.columns, prob.n_elements)
    @variable(model, x[1:n_columns], Bin)
    @objective(model, Min, sum(prob.costs[j] * x[j] for j in 1:n_columns))
    for i in 1:prob.n_elements
        @constraint(model, sum(x[j] for j in incidence[i]) >= 1)
    end
    if prob.maximum_selected < n_columns
        @constraint(model, sum(x) <= prob.maximum_selected)
    end
    return model
end

register_variant(
    :set_system,
    :set_cover,
    SetCoverProblem,
    "Minimum-cost binary set cover with planted covers and varied column sizes",
    default=true,
)
