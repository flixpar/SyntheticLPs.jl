using JuMP
using Random

"""
    MixedIntegerKnapsackSetProblem <: ProblemGenerator

A many-row mixed-integer knapsack-set model inspired by the structural regime
of HEM-MIK benchmark instances. Most variables are bounded general integers, a
small block is continuous, and the resource matrix mixes sparse rows with
dense rows. This is deliberately distinct from the category's one-row bounded
knapsack and all-binary multidimensional variants.

# Fields
- `n_integer`, `n_continuous`: Sizes of the general-integer and continuous blocks.
- `n_rows`: Number of packing/resource rows.
- `integer_upper`, `continuous_upper`: Finite variable upper bounds.
- `coefficients`: Row matrix over both variable blocks; structural zeros encode sparsity.
- `capacities`: Packing-row right-hand sides.
- `profits`: Positive objective coefficients over both blocks.
- `minimum_profit`: Optional verification floor. For infeasible instances it
  exceeds the objective's box-bound upper bound.
- `planted_integer`, `planted_continuous`: A nonzero witness used to construct
  every row capacity.
- `dense_rows`: Whether each row was generated in the dense regime.
"""
struct MixedIntegerKnapsackSetProblem <: ProblemGenerator
    n_integer::Int
    n_continuous::Int
    n_rows::Int
    integer_upper::Vector{Int}
    continuous_upper::Vector{Float64}
    coefficients::Matrix{Float64}
    capacities::Vector{Float64}
    profits::Vector{Float64}
    minimum_profit::Float64
    planted_integer::Vector{Int}
    planted_continuous::Vector{Float64}
    dense_rows::BitVector
end

"""
    MixedIntegerKnapsackSetProblem(target_variables, feasibility_status, seed)

Construct a deterministic HEM-MIK-style mixed-integer knapsack set. At corpus
scale (250--500 variables), the continuous block contains 10--20 variables and
the remainder are general integers; the row count is sampled between 60% and
90% of the variable count.

For `feasible`, all capacities and the profit floor admit the stored planted
witness. For `infeasible`, the profit floor is strictly greater than
`sum(profit[j] * upper[j])`, a box-bound certificate that survives integrality
relaxation. `unknown` uses a natural, non-certified profit target against the
planted resource rows.
"""
function MixedIntegerKnapsackSetProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))

    rng = MersenneTwister(seed)
    n_variables = target_variables
    n_continuous = n_variables == 1 ? 0 :
                   clamp(round(Int, 0.04 * n_variables), 2, min(20, n_variables - 1))
    n_integer = n_variables - n_continuous
    n_rows = max(1, round(Int, n_variables * (0.60 + 0.30 * rand(rng))))

    integer_upper = rand(rng, 2:10, n_integer)
    continuous_upper = [2.0 + 8.0 * rand(rng) for _ in 1:n_continuous]

    planted_integer = [rand(rng, 0:integer_upper[i]) for i in 1:n_integer]
    planted_continuous = [continuous_upper[j] * (0.15 + 0.70 * rand(rng))
                          for j in 1:n_continuous]
    # Make the witness nonzero even for the smallest supported instance.
    if all(iszero, planted_integer) && all(iszero, planted_continuous)
        planted_integer[1] = 1
    end
    planted = vcat(Float64.(planted_integer), planted_continuous)

    coefficients = zeros(Float64, n_rows, n_variables)
    capacities = zeros(Float64, n_rows)
    dense_rows = falses(n_rows)

    for row in 1:n_rows
        is_dense = mod(row, 4) == 0
        dense_rows[row] = is_dense
        support_size = if is_dense
            clamp(round(Int, n_variables * (0.45 + 0.35 * rand(rng))), 1, n_variables)
        else
            clamp(round(Int, n_variables * (0.03 + 0.07 * rand(rng))), 1, n_variables)
        end
        support = randperm(rng, n_variables)[1:support_size]
        for column in support
            # Integer-valued resource coefficients dominate, with a small
            # continuous perturbation to retain the mixed numeric regime.
            coefficients[row, column] = rand(rng, 1:50) * (0.9 + 0.2 * rand(rng))
        end
        planted_activity = sum(coefficients[row, :] .* planted)
        # Positive additive slack also handles a row whose planted support is zero.
        row_scale = sum(coefficients[row, :])
        capacities[row] = planted_activity * (1.05 + 0.25 * rand(rng)) +
                          max(1.0, 0.02 * row_scale)
    end

    integer_profit = [Float64(rand(rng, 10:150)) for _ in 1:n_integer]
    continuous_profit = [10.0 + 140.0 * rand(rng) for _ in 1:n_continuous]
    profits = vcat(integer_profit, continuous_profit)
    planted_profit = sum(profits .* planted)
    box_upper = sum(profits[1:n_integer] .* integer_upper) +
                sum(profits[n_integer + j] * continuous_upper[j] for j in 1:n_continuous)

    minimum_profit = if feasibility_status == feasible
        planted_profit * (0.70 + 0.20 * rand(rng))
    elseif feasibility_status == infeasible
        box_upper * (1.05 + 0.15 * rand(rng)) + 1.0
    else
        # A meaningful but uncertified target: resource coupling, rather than a
        # constructor guarantee, decides whether this fraction is attainable.
        box_upper * (0.45 + 0.20 * rand(rng))
    end

    return MixedIntegerKnapsackSetProblem(
        n_integer,
        n_continuous,
        n_rows,
        integer_upper,
        continuous_upper,
        coefficients,
        capacities,
        profits,
        minimum_profit,
        planted_integer,
        planted_continuous,
        dense_rows,
    )
end

"""
    build_model(prob::MixedIntegerKnapsackSetProblem)

Build the many-row mixed-integer knapsack model. Rebuilding from the same
problem object is deterministic and performs no random sampling.
"""
function build_model(prob::MixedIntegerKnapsackSetProblem)
    model = Model()

    @variable(
        model,
        0 <= integer_items[i = 1:prob.n_integer] <= prob.integer_upper[i],
        Int,
    )
    @variable(
        model,
        0 <= continuous_items[j = 1:prob.n_continuous] <= prob.continuous_upper[j],
    )

    for row in 1:prob.n_rows
        @constraint(
            model,
            sum(prob.coefficients[row, i] * integer_items[i] for i in 1:prob.n_integer) +
            sum(
                prob.coefficients[row, prob.n_integer + j] * continuous_items[j]
                for j in 1:prob.n_continuous
            ) <= prob.capacities[row],
        )
    end

    total_profit =
        sum(prob.profits[i] * integer_items[i] for i in 1:prob.n_integer) +
        sum(
            prob.profits[prob.n_integer + j] * continuous_items[j]
            for j in 1:prob.n_continuous
        )
    @constraint(model, total_profit >= prob.minimum_profit)
    @objective(model, Max, total_profit)

    return model
end

register_variant(
    :knapsack,
    :mixed_integer_set,
    MixedIntegerKnapsackSetProblem,
    "HEM-MIK-style many-row knapsack set with bounded general integers, a small continuous block, and mixed sparse/dense rows",
)

