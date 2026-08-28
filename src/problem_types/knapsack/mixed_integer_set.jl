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
- `row_indices`, `row_coefficients`: Sparse supports; only generated nonzeros
  are stored, so mixed sparse/dense rows stay compact at large `n`.
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
    row_indices::Vector{Vector{Int}}
    row_coefficients::Vector{Vector{Float64}}
    capacities::Vector{Float64}
    profits::Vector{Float64}
    minimum_profit::Float64
    planted_integer::Vector{Int}
    planted_continuous::Vector{Float64}
    dense_rows::BitVector
end

# Sample `k` distinct columns. Sparse rows (k ≪ n) use a set; dense rows fall
# back to a permutation prefix, which is cheaper once k is a large fraction of n.
function _mik_sample_support(rng::AbstractRNG, n::Int, k::Int)
    k >= n && return collect(1:n)
    if 3k < n
        picked = Set{Int}()
        sizehint!(picked, k)
        while length(picked) < k
            push!(picked, rand(rng, 1:n))
        end
        return collect(picked)
    end
    return randperm(rng, n)[1:k]
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

    row_indices = Vector{Vector{Int}}(undef, n_rows)
    row_coefficients = Vector{Vector{Float64}}(undef, n_rows)
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
        support = _mik_sample_support(rng, n_variables, support_size)
        coefs = Vector{Float64}(undef, length(support))
        for j in eachindex(support)
            # Integer-valued resource coefficients dominate, with a small
            # continuous perturbation to retain the mixed numeric regime.
            coefs[j] = rand(rng, 1:50) * (0.9 + 0.2 * rand(rng))
        end
        row_indices[row] = support
        row_coefficients[row] = coefs
        planted_activity = sum(coefs[j] * planted[support[j]] for j in eachindex(support); init=0.0)
        # Positive additive slack also handles a row whose planted support is zero.
        row_scale = sum(coefs)
        capacities[row] = planted_activity * (1.05 + 0.25 * rand(rng)) +
                          max(1.0, 0.02 * row_scale)
    end

    integer_profit = [Float64(rand(rng, 10:150)) for _ in 1:n_integer]
    continuous_profit = [10.0 + 140.0 * rand(rng) for _ in 1:n_continuous]
    profits = vcat(integer_profit, continuous_profit)
    planted_profit = sum(profits .* planted)
    box_upper = sum(profits[1:n_integer] .* integer_upper) +
                sum(profits[n_integer + j] * continuous_upper[j] for j in 1:n_continuous; init=0.0)

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
        row_indices,
        row_coefficients,
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
        expr = AffExpr()
        for (column, coefficient) in zip(prob.row_indices[row], prob.row_coefficients[row])
            if column <= prob.n_integer
                add_to_expression!(expr, coefficient, integer_items[column])
            else
                add_to_expression!(expr, coefficient, continuous_items[column - prob.n_integer])
            end
        end
        @constraint(model, expr <= prob.capacities[row])
    end

    total_profit = AffExpr()
    for i in 1:prob.n_integer
        add_to_expression!(total_profit, prob.profits[i], integer_items[i])
    end
    for j in 1:prob.n_continuous
        add_to_expression!(total_profit, prob.profits[prob.n_integer + j], continuous_items[j])
    end
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

