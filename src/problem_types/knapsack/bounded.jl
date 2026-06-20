using JuMP
using Random

"""
    BoundedKnapsackProblem <: ProblemGenerator

Generator for bounded knapsack problems with per-item multiplicity limits.

# Overview
A single-budget knapsack in which each item is available in several identical
copies. For item `i` the decision `x[i]` is the integer number of copies taken,
bounded above by a per-item multiplicity `u_i` (the stock available of that
product). The objective maximizes total value subject to a single weight/capacity
budget. The framework relaxes integrality by default, so the solved model is the
box-constrained LP `0 <= x[i] <= u_i` — structurally distinct from the standard
0/1 knapsack relaxation `0 <= x[i] <= 1`: the feasible region is a general box
whose extreme points can be any corner of `[0, u_i]`, not just `{0, 1}`.

Item weights and values are scale-tiered and positively (but noisily) correlated,
so heavier items tend to be worth more without value/weight ratios being constant.
The capacity is set to a fraction of the maximum total weight `sum_i w_i * u_i`,
so the budget genuinely binds and not all copies can be taken.

# Fields
- `n_items::Int`: Number of distinct items (= number of decision variables)
- `capacity::Int`: Single weight/capacity budget
- `values::Vector{Int}`: Value of one copy of each item
- `weights::Vector{Int}`: Weight of one copy of each item
- `upper_bounds::Vector{Int}`: Multiplicity bound `u_i` (max copies) per item
- `min_value::Float64`: Minimum total-value requirement; `0.0` when unused.
  When positive it is set strictly above the LP-relaxation optimum, making the
  relaxed model provably infeasible.
"""
struct BoundedKnapsackProblem <: ProblemGenerator
    n_items::Int
    capacity::Int
    values::Vector{Int}
    weights::Vector{Int}
    upper_bounds::Vector{Int}
    min_value::Float64
end

"""
    BoundedKnapsackProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a bounded knapsack instance.

Variable-count formula (decision variables created by `build_model`):

    total = n_items

There is exactly one decision variable per item, so `n_items = target_variables`.

# Arguments
- `target_variables`: Target number of decision variables (= number of items)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Feasibility
- `feasible` / `unknown`: only the capacity constraint is present. `x = 0` is always
  feasible and the optimum is non-trivial (capacity is a fraction of the maximum
  achievable weight), so the relaxation has a finite optimum.
- `infeasible`: a value floor `sum_i v_i x_i >= min_value` is added with
  `min_value` set strictly above the exact LP-relaxation optimum. That optimum is
  computed here by the classic greedy for the *bounded fractional* knapsack (sort
  by value/weight ratio descending; for each item take `min(u_i, remaining/w_i)`
  copies). Because the relaxation's best total value is `< min_value`, the relaxed
  model is infeasible regardless of integrality.
"""
function BoundedKnapsackProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # One decision variable per item.
    n_items = max(1, target_variables)

    # --- Scale-tiered parameter ranges ---
    # base weights/values plus multiplicity bounds; ranges grow with problem size.
    if n_items <= 100
        weight_lo, weight_hi = 3, rand(18:28)
        value_base_lo, value_base_hi = 10, rand(70:110)
        bound_hi = rand(4:6)
    elseif n_items <= 1000
        weight_lo, weight_hi = 5, rand(30:45)
        value_base_lo, value_base_hi = 20, rand(150:260)
        bound_hi = rand(5:7)
    else
        weight_lo, weight_hi = 10, rand(50:75)
        value_base_lo, value_base_hi = 40, rand(300:550)
        bound_hi = rand(6:8)
    end

    # --- Weights ---
    weights = rand(weight_lo:weight_hi, n_items)

    # --- Values: positively but noisily correlated with weight ---
    # value_i ≈ correlation_slope * weight_i + base_noise, kept strictly positive.
    correlation_slope = value_base_hi / max(weight_hi, 1) * (0.5 + 0.5 * rand())
    values = Vector{Int}(undef, n_items)
    for i in 1:n_items
        signal = correlation_slope * weights[i]
        noise = rand(value_base_lo:value_base_hi)
        v = round(Int, 0.6 * signal + 0.4 * noise + value_base_lo)
        values[i] = max(value_base_lo, v)
    end

    # --- Multiplicity bounds u_i (units of each product available) ---
    upper_bounds = rand(1:bound_hi, n_items)

    # --- Capacity: a fraction of the maximum total weight so the budget binds ---
    max_total_weight = sum(weights[i] * upper_bounds[i] for i in 1:n_items)
    capacity_ratio = 0.30 + 0.30 * rand()  # 30%-60% of the max packable weight
    capacity = max(1, round(Int, max_total_weight * capacity_ratio))

    # --- Resolve feasibility intent ---
    actual_status = feasibility_status
    if feasibility_status == unknown
        # Natural instance: capacity-only models are always feasible; bias is moot,
        # but keep the branch explicit. Treat as feasible (no value floor).
        actual_status = feasible
    end

    min_value = 0.0
    if actual_status == infeasible
        # Exact LP-relaxation optimum via greedy for the bounded fractional knapsack.
        ratios = [values[i] / max(weights[i], 1) for i in 1:n_items]
        order = sortperm(ratios, rev = true)
        remaining = Float64(capacity)
        lp_opt = 0.0
        for i in order
            remaining <= 0 && break
            take = min(Float64(upper_bounds[i]), remaining / weights[i])
            lp_opt += values[i] * take
            remaining -= weights[i] * take
        end
        # Require strictly more value than the relaxation can achieve.
        min_value = lp_opt * (1.1 + 0.3 * rand())
        # Guard against degenerate zero optimum (cannot happen with positive data,
        # but keep the floor strictly positive so the constraint is emitted).
        min_value = max(min_value, 1.0)
    end

    return BoundedKnapsackProblem(n_items, capacity, values, weights, upper_bounds, min_value)
end

"""
    build_model(prob::BoundedKnapsackProblem)

Build a JuMP model for the bounded knapsack problem. Deterministic — uses only the
struct fields.

Decision variables:
- `x[i]`: integer number of copies of item `i` taken, `0 <= x[i] <= u_i`
  (relaxed to continuous on `[0, u_i]` by the framework).

# Returns
- `model`: The JuMP model
"""
function build_model(prob::BoundedKnapsackProblem)
    model = Model()

    n = prob.n_items

    # Variables: integer copies bounded by per-item multiplicity (relaxed -> box [0, u_i]).
    @variable(model, 0 <= x[i = 1:n] <= prob.upper_bounds[i], Int)

    # Objective: maximize total value.
    @objective(model, Max, sum(prob.values[i] * x[i] for i in 1:n))

    # Single capacity / weight budget.
    @constraint(model, sum(prob.weights[i] * x[i] for i in 1:n) <= prob.capacity)

    # Optional value floor (renders the relaxation infeasible when set above LP optimum).
    if prob.min_value > 0
        @constraint(model, sum(prob.values[i] * x[i] for i in 1:n) >= prob.min_value)
    end

    return model
end

# Register the variant
register_variant(
    :knapsack,
    :bounded,
    BoundedKnapsackProblem,
    "Bounded knapsack maximizing value with integer per-item multiplicity limits under a single weight budget",
)
