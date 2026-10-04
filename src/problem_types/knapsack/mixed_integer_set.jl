using JuMP
using Random

"""
    LagrangianBoundCertificate

Relaxation-valid infeasibility proof for the profit floor. For any row
multipliers `y >= 0` (`row_multipliers`), weak duality bounds the LP optimum
of the packing rows and box bounds by

    dual_bound = sum_r y_r * cap_r + sum_j upper_j * max(0, profit_j - sum_r y_r * a_rj),

and `minimum_profit >= 1.01 * dual_bound`. Farkas combination: `y` on the
packing rows, the positive reduced profits on the upper bounds, and `-1` on the
profit-floor row. `y` comes from a projected-subgradient Lagrangian heuristic,
so the bound is far below the box bound `sum_j profit_j * upper_j` and
presolve's activity bounds cannot expose the contradiction.
"""
struct LagrangianBoundCertificate
    row_multipliers::Vector{Float64}
    dual_bound::Float64
end

"""
    MixedIntegerKnapsackSetProblem <: ProblemGenerator

A many-row mixed-integer knapsack-set model inspired by the structural regime
of HEM-MIK benchmark instances: most variables are bounded general integers, a
small block is continuous, and the resource matrix mixes many sparse rows with
a few dense rows, plus a profit floor.

# Sizing and sparsity

`n = target_variables` columns (`n_continuous = clamp(round(0.04 n), 2, 20)`,
the rest integer), `n_rows = round(n * U(0.35, 0.60))` packing rows. Sparse
rows have `4-24` nonzeros drawn from a local window of related columns (width
`4k`; with probability 0.15 one far column), and
`n_dense = clamp(round(n_rows / 4), 1, 16)` evenly spaced dense rows cover
`U(0.45, 0.80)` of `min(n, 40_000)` columns. nnz grows linearly (about 15 per
column at 100k). The old generator made 3/4 of the rows cover 3-10% of all
columns and 1/4 cover 45-80% — `O(n^2)` nonzeros, an 18.7 GB MPS file at 50k —
and even with bounded nnz, uniformly random supports at 0.6-0.9 rows per
column filled the LU factors so 10k-column instances exceeded a 60 s simplex
limit; local supports and 0.35-0.6 rows per column solve in seconds.

# Feasibility

Capacities are built around a nonzero planted point (`planted_integer`,
`planted_continuous`), so the packing rows are always satisfiable; the
profit floor decides the status:

  - `feasible`: `minimum_profit = U(0.70, 0.90)` times the planted profit.
  - `infeasible`: `minimum_profit = U(1.01, 1.05)` times a Lagrangian dual bound
    (`LagrangianBoundCertificate`) — not the box bound, so simplex has to work
    to prove it.
  - `unknown`: `minimum_profit` is placed `U(0.45, 1.0)` of the way from the
    planted profit (a valid lower bound on the LP optimum) to the dual bound (an
    upper bound): genuinely undetermined.
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
    infeasibility_certificate::Union{Nothing, LagrangianBoundCertificate}
end

const MIK_DENSE_BASE = 40_000

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
        return sort!(collect(picked))
    end
    return sort!(randperm(rng, n)[1:k])
end

# Local support: `k` distinct columns from a window of width `~4k` around a
# random centre (wrapping); with probability 0.15 one column is
# replaced by a uniformly random one.
function _mik_local_support(rng::AbstractRNG, n::Int, k::Int)
    k >= n && return collect(1:n)
    w = min(n, max(4k, 16))
    c = rand(rng, 1:n)
    picked = Set{Int}()
    while length(picked) < k
        push!(picked, mod(c + rand(rng, 0:(w - 1)) - 1, n) + 1)
    end
    if rand(rng) < 0.15
        far = rand(rng, 1:n)
        far in picked || (delete!(picked, minimum(picked)); push!(picked, far))
    end
    return sort!(collect(picked))
end

"""
    mik_dual_bound(row_indices, row_coefficients, capacities, profits, upper, y)

Weak-duality bound `sum y_r cap_r + sum_j upper_j * max(0, reduced_j)` and the
reduced profits, in O(nnz).
"""
function mik_dual_bound(row_indices, row_coefficients, capacities, profits, upper, y)
    reduced = copy(profits)
    bound = 0.0
    for r in eachindex(row_indices)
        y[r] == 0.0 && continue
        bound += y[r] * capacities[r]
        for (j, a) in zip(row_indices[r], row_coefficients[r])
            reduced[j] -= y[r] * a
        end
    end
    for j in eachindex(reduced)
        reduced[j] > 0 && (bound += upper[j] * reduced[j])
    end
    return bound, reduced
end

# Projected subgradient on the Lagrangian dual of the packing LP. Polyak steps
# aim at an adaptive target `best * (1 - gap)` (the gap shrinks whenever the
# bound stalls), never below the known lower bound `lb`. O(iterations * nnz).
function _mik_lagrangian(row_indices, row_coefficients, capacities, profits, upper, lb; iters=250)
    m = length(capacities)
    y = zeros(m)
    best_y = copy(y)
    best, reduced = mik_dual_bound(row_indices, row_coefficients, capacities, profits, upper, y)
    gap = 0.10
    stall = 0
    g = zeros(m)
    for _ in 1:iters
        val, reduced = mik_dual_bound(row_indices, row_coefficients, capacities, profits, upper, y)
        if val < best - 1e-9 * abs(best)
            best, best_y, stall = val, copy(y), 0
        else
            stall += 1
            if stall >= 8
                gap /= 2
                stall = 0
                y .= best_y
            end
        end
        # Subgradient: cap_r - row activity at the Lagrangian maximiser.
        norm2 = 0.0
        for r in 1:m
            act = 0.0
            for (j, a) in zip(row_indices[r], row_coefficients[r])
                reduced[j] > 0 && (act += a * upper[j])
            end
            g[r] = capacities[r] - act
            (y[r] > 0 || g[r] < 0) && (norm2 += g[r]^2)
        end
        norm2 <= 0 && break
        target = max(lb, best * (1 - gap))
        step = max(val - target, 1e-9 * abs(val)) / norm2
        for r in 1:m
            y[r] = max(0.0, y[r] - step * g[r])
        end
    end
    return best_y, best
end

"""
    MixedIntegerKnapsackSetProblem(target_variables, feasibility_status, seed)

Construct a deterministic HEM-MIK-style mixed-integer knapsack set with exactly
`target_variables` columns.
"""
function MixedIntegerKnapsackSetProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))

    rng = MersenneTwister(seed)
    n_variables = target_variables
    n_continuous =
        n_variables == 1 ? 0 : clamp(round(Int, 0.04 * n_variables), 2, min(20, n_variables - 1))
    n_integer = n_variables - n_continuous
    n_rows = max(1, round(Int, n_variables * (0.35 + 0.25 * rand(rng))))
    n_dense = clamp(round(Int, n_rows / 4), 1, 16)
    stride = max(1, n_rows ÷ n_dense)

    integer_upper = rand(rng, 2:10, n_integer)
    continuous_upper = [2.0 + 8.0 * rand(rng) for _ in 1:n_continuous]
    upper = vcat(Float64.(integer_upper), continuous_upper)

    planted_integer = [rand(rng, 0:integer_upper[i]) for i in 1:n_integer]
    planted_continuous = [continuous_upper[j] * (0.15 + 0.70 * rand(rng)) for j in 1:n_continuous]
    # Make the witness nonzero even for the smallest supported instance.
    if all(iszero, planted_integer) && all(iszero, planted_continuous)
        planted_integer[1] = 1
    end
    planted = vcat(Float64.(planted_integer), planted_continuous)

    row_indices = Vector{Vector{Int}}(undef, n_rows)
    row_coefficients = Vector{Vector{Float64}}(undef, n_rows)
    capacities = zeros(Float64, n_rows)
    dense_rows = falses(n_rows)
    dense_pool = min(n_variables, MIK_DENSE_BASE)

    for row in 1:n_rows
        is_dense = mod(row, stride) == 0 && count(dense_rows) < n_dense
        dense_rows[row] = is_dense
        support_size = if is_dense
            clamp(round(Int, dense_pool * (0.45 + 0.35 * rand(rng))), 1, n_variables)
        else
            min(rand(rng, 4:24), n_variables)
        end
        support = if is_dense
            _mik_sample_support(rng, n_variables, support_size)
        else
            # Sparse rows are local: a resource is shared by a family of
            # related items (a window of nearby columns), plus occasionally
            # one far column. Uniform random supports fill the LU factors.
            _mik_local_support(rng, n_variables, support_size)
        end
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
        capacities[row] = planted_activity * (1.05 + 0.25 * rand(rng)) + max(1.0, 0.02 * row_scale)
    end

    integer_profit = [Float64(rand(rng, 10:150)) for _ in 1:n_integer]
    continuous_profit = [10.0 + 140.0 * rand(rng) for _ in 1:n_continuous]
    profits = vcat(integer_profit, continuous_profit)
    planted_profit = sum(profits .* planted)

    certificate = nothing
    minimum_profit = if feasibility_status == feasible
        planted_profit * (0.70 + 0.20 * rand(rng))
    else
        y, bound = _mik_lagrangian(
            row_indices, row_coefficients, capacities, profits, upper, planted_profit
        )
        if feasibility_status == infeasible
            certificate = LagrangianBoundCertificate(y, bound)
            bound * (1.01 + 0.04 * rand(rng))
        else
            planted_profit + (0.45 + 0.55 * rand(rng)) * (bound - planted_profit)
        end
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
        certificate,
    )
end

"""
    build_model(prob::MixedIntegerKnapsackSetProblem)

Build the many-row mixed-integer knapsack model. Rebuilding from the same
problem object is deterministic and performs no random sampling.
"""
function build_model(prob::MixedIntegerKnapsackSetProblem)
    model = Model()

    @variable(model, 0 <= integer_items[i = 1:prob.n_integer] <= prob.integer_upper[i], Int,)
    @variable(model, 0 <= continuous_items[j = 1:prob.n_continuous] <= prob.continuous_upper[j],)

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
    "HEM-MIK-style many-row knapsack set with bounded general integers, a small continuous block, sparse and a few dense rows, and a profit floor",
)
