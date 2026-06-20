using JuMP
using Random
using Distributions
using Statistics

"""
    MultiItemInventoryProblem <: ProblemGenerator

Generator for multi-item lot-sizing inventory problems with a shared per-period
production capacity.

# Overview
Models a deterministic, multi-period production/inventory plan for several items
that compete for a single, shared production resource each period. The decisions
are production quantities `x[i, t]` and end-of-period inventories `I[i, t]` for
every item `i` and period `t`. The objective minimizes total production plus
holding cost. Each item has its own inventory-balance constraints linking
production, demand and carried inventory. The items are coupled through a single
shared resource constraint per period:
`sum_i resource_usage[i] * x[i, t] <= prod_capacity`, where `resource_usage[i]`
is the amount of the shared resource consumed per unit produced of item `i`.

Backlogging is not permitted (inventories are nonnegative), so demand must be met
on time from carried inventory plus current production.

# Fields
- `n_items::Int`: Number of distinct items sharing the production resource
- `n_periods::Int`: Number of planning periods
- `prod_capacity::Float64`: Shared per-period resource capacity
- `item_demands::Matrix{Int}`: Demand per item per period (`n_items × n_periods`)
- `item_production_costs::Matrix{Float64}`: Unit production cost (`n_items × n_periods`)
- `item_holding_costs::Matrix{Float64}`: Unit holding cost (`n_items × n_periods`)
- `item_initial_inventory::Vector{Int}`: Starting inventory per item
- `item_resource_usage::Vector{Float64}`: Shared-resource consumption per unit per item
"""
struct MultiItemInventoryProblem <: ProblemGenerator
    n_items::Int
    n_periods::Int
    prod_capacity::Float64
    item_demands::Matrix{Int}
    item_production_costs::Matrix{Float64}
    item_holding_costs::Matrix{Float64}
    item_initial_inventory::Vector{Int}
    item_resource_usage::Vector{Float64}
end

"""
    MultiItemInventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-item shared-capacity inventory problem instance.

# Variable count
The model creates two variable blocks:
- `x[1:n_items, 1:n_periods]` → `n_items * n_periods` production variables
- `I[1:n_items, 0:n_periods]` → `n_items * (n_periods + 1)` inventory variables

Total variables = `n_items * (2 * n_periods + 1)` ≈ `2 * n_items * n_periods`.
Dimensions are sized so this total lands near `target_variables`.

# Feasibility handling
Feasibility is controlled through the shared per-period capacity relative to the
*resource-usage-weighted* per-period demand load,
`weighted_load[t] = sum_i resource_usage[i] * item_demands[i, t]`:
- `feasible`: `prod_capacity` is set to comfortably exceed `max_t weighted_load[t]`
  (with a positive margin), so each period's demand can be produced just-in-time
  and a feasible plan provably exists.
- `infeasible`: `prod_capacity` is set strictly below `max_t weighted_load[t]`
  (with a margin) while backlogging is disallowed, so the shared resource cannot
  supply enough in the binding period and no feasible plan exists.
- `unknown`: `prod_capacity` is left at a naturally sampled level (no forced
  infeasibility).

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function MultiItemInventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine business scale by target size
    scale = target_variables <= 250 ? :small :
            target_variables <= 1000 ? :medium : :large

    # --- Dimension sizing ---
    # Total variables = n_items * (2 * n_periods + 1) ≈ 2 * n_items * n_periods.
    # Sample n_items ONCE, then size n_periods to hit the target.
    n_items = rand(max(2, target_variables ÷ 50):max(5, target_variables ÷ 20))
    n_periods = max(2, min(500, round(Int, (target_variables / n_items - 1) / 2)))

    # Scale-specific demand / cost ranges
    if scale == :small
        demand_base = round(Int, rand(Uniform(10, 100)))
        prod_cost_base = rand(Uniform(10, 100))
        holding_rate = rand(Uniform(0.05, 0.25)) / 12
    elseif scale == :medium
        demand_base = round(Int, rand(Uniform(50, 1000)))
        prod_cost_base = rand(Uniform(5, 200))
        holding_rate = rand(Uniform(0.03, 0.20)) / 12
    else
        demand_base = round(Int, rand(Uniform(100, 10000)))
        prod_cost_base = rand(Uniform(1, 500))
        holding_rate = rand(Uniform(0.01, 0.15)) / 12
    end

    # --- Per-item demands ---
    item_demands = zeros(Int, n_items, n_periods)
    for i in 1:n_items
        item_base = demand_base * rand(Uniform(0.3, 1.5))
        item_demands[i, :] = round.(Int, clamp.(
            rand(Normal(item_base, item_base * 0.25), n_periods),
            max(1, item_base * 0.3), item_base * 2.0
        ))
    end
    # Keep all demands strictly positive
    item_demands = max.(item_demands, 1)

    # --- Per-item production and holding costs ---
    item_production_costs = zeros(n_items, n_periods)
    item_holding_costs = zeros(n_items, n_periods)
    for i in 1:n_items
        base_cost = prod_cost_base * rand(Uniform(0.5, 2.0))
        item_production_costs[i, :] = clamp.(
            rand(Normal(base_cost, base_cost * 0.1), n_periods),
            base_cost * 0.8, base_cost * 1.2
        )
        item_holding_costs[i, :] = item_production_costs[i, :] .* holding_rate
    end

    # --- Initial inventory and shared-resource usage ---
    item_initial_inventory = round.(Int, [mean(item_demands[i, :]) * rand(Uniform(0.1, 0.4))
                                          for i in 1:n_items])
    item_resource_usage = [rand(Uniform(0.5, 2.0)) for _ in 1:n_items]

    # --- Resource-usage-weighted per-period demand load ---
    # weighted_load[t] = sum_i resource_usage[i] * item_demands[i, t]
    weighted_load = [sum(item_resource_usage[i] * item_demands[i, t] for i in 1:n_items)
                     for t in 1:n_periods]
    peak_load = maximum(weighted_load)

    # --- Feasibility handling (capacity vs. cumulative weighted load) ---
    # With inventory carryover and no backlogging, feasibility is governed by the
    # cumulative (prefix) load against cumulative capacity, NOT the single-period
    # peak: production can be pre-built in slack periods and carried into a spike.
    # The binding per-period rate is
    #   max_t (cumulative_weighted_demand[t] - weighted_initial_inventory) / t.
    weighted_init = sum(item_resource_usage[i] * item_initial_inventory[i] for i in 1:n_items)
    cumulative = 0.0
    binding_rate = 0.0
    for t in 1:n_periods
        cumulative += weighted_load[t]
        binding_rate = max(binding_rate, (cumulative - weighted_init) / t)
    end
    if feasibility_status == infeasible
        # Below the binding cumulative rate: some prefix cannot be supplied even
        # by producing flat-out in every prior period (a true contradiction).
        prod_capacity = max(binding_rate, 1.0) * rand(Uniform(0.4, 0.7))
    else
        # feasible and unknown: capacity comfortably above the peak weighted load
        # (>= the binding cumulative rate), so just-in-time production suffices.
        prod_capacity = peak_load * rand(Uniform(1.3, 2.0))
    end

    return MultiItemInventoryProblem(
        n_items, n_periods, prod_capacity,
        item_demands, item_production_costs, item_holding_costs,
        item_initial_inventory, item_resource_usage,
    )
end

"""
    build_model(prob::MultiItemInventoryProblem)

Build a JuMP model for the multi-item shared-capacity inventory problem.
Deterministic — uses only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::MultiItemInventoryProblem)
    model = Model()

    # Variables: x[i,t] production, I[i,t] end-of-period inventory.
    # Var count = n_items*n_periods + n_items*(n_periods+1) = n_items*(2*n_periods+1).
    @variable(model, x[1:prob.n_items, 1:prob.n_periods] >= 0)
    @variable(model, I[1:prob.n_items, 0:prob.n_periods] >= 0)

    # Objective: total production + holding cost
    @objective(model, Min,
        sum(prob.item_production_costs[i, t] * x[i, t] + prob.item_holding_costs[i, t] * I[i, t]
            for i in 1:prob.n_items, t in 1:prob.n_periods))

    # Initial inventory per item
    for i in 1:prob.n_items
        @constraint(model, I[i, 0] == prob.item_initial_inventory[i])
    end

    # Per-item inventory balance (no backlogging: I >= 0)
    for i in 1:prob.n_items, t in 1:prob.n_periods
        @constraint(model, I[i, t-1] + x[i, t] - prob.item_demands[i, t] == I[i, t])
    end

    # Shared per-period resource capacity
    for t in 1:prob.n_periods
        @constraint(model, sum(prob.item_resource_usage[i] * x[i, t]
                               for i in 1:prob.n_items) <= prob.prod_capacity)
    end

    return model
end

# Register the variant
register_variant(
    :inventory,
    :multi_item,
    MultiItemInventoryProblem,
    "Multi-item lot-sizing inventory with a shared resource-usage-weighted per-period production capacity",
)
