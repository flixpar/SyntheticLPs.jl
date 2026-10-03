using JuMP
using Random

"""
    SetPartitioningWitness

A planted exact partition: the columns `columns` are pairwise disjoint and
cover every element, and `length(columns) <= maximum_selected`.
"""
struct SetPartitioningWitness
    columns::Vector{Int}
end

"""
    SetPartitioningCardinalityCertificate

Summing all exact-cover rows gives `n_elements = sum_j |S_j| x_j <= max_size *
sum_j x_j`, where `max_size` is the largest column size, so every LP-feasible
point has `sum(x) >= n_elements / max_size == bound`. A cardinality cap at
least 3% (and at least 0.5) below `bound` is infeasible even in the LP
relaxation, with a margin that survives solver tolerances.
"""
struct SetPartitioningCardinalityCertificate
    max_size::Int
    bound::Float64
end

"""
    SetPartitioningProblem <: ProblemGenerator

Minimum-cost exact set partitioning over a generic sparse incidence matrix with
a cap on the number of selected columns.

# Feasibility

  - `feasible`: the planted partition (`feasible_witness`); the cap equals its size.
  - `infeasible`: the cap is below the cardinality bound of
    [`SetPartitioningCardinalityCertificate`](@ref), which aggregates every row.
  - `unknown`: the cap is drawn from 70–95% of the way from that
    bound to the planted partition's size, which straddles the LP minimum of
    `sum(x)`, so the LP may or may not admit a partition with that few columns.
"""
struct SetPartitioningProblem <: ProblemGenerator
    n_elements::Int
    columns::Vector{Vector{Int}}
    costs::Vector{Float64}
    maximum_selected::Int
    feasible_witness::Union{Nothing, SetPartitioningWitness}
    infeasibility_certificate::Union{Nothing, SetPartitioningCardinalityCertificate}
end

function SetPartitioningProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    n_columns, n_elements = _set_system_size(target_variables, 0.4)
    max_size = max(2, min(6, round(Int, sqrt(n_elements)) + 1))
    columns, n_planted = _set_columns_with_partition(rng, n_elements, n_columns; max_size=max_size)

    largest = maximum(length, columns)
    lower = cld(n_elements, largest)
    witness = nothing
    certificate = nothing
    maximum_selected = if feasibility_status == infeasible
        # Summing exact-cover rows gives
        #   n_elements = sum_j |S_j|x_j <= largest * sum_j x_j.
        # The strict cardinality cap therefore rules out even fractional x.
        bound = n_elements / largest
        certificate = SetPartitioningCardinalityCertificate(largest, bound)
        floor(Int, bound - max(0.5, 0.03 * bound))
    elseif feasibility_status == feasible
        witness = SetPartitioningWitness(collect(1:n_planted))
        n_planted
    else
        # The LP minimum of sum(x) sits at ~85-90% of the way from the
        # cardinality bound to the planted partition; straddle it.
        lower + round(Int, (0.7 + 0.25 * rand(rng)) * max(0, n_planted - lower))
    end

    costs = _set_positive_coefficients(rng, n_columns; low=5, high=100)
    return SetPartitioningProblem(
        n_elements, columns, costs, maximum_selected, witness, certificate
    )
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
