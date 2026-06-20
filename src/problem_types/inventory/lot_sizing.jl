using JuMP
using Random
using Distributions
using Statistics

"""
    LotSizingInventoryProblem <: ProblemGenerator

Generator for capacitated lot-sizing inventory problems (a mixed-integer program).

# Overview
Models single-item production planning where production must occur in fixed batch
("lot") sizes and incurs a fixed setup cost whenever production happens in a
period. The decisions are: the production quantity per period, the integer number
of lots produced per period, a binary setup indicator per period, and the
nonnegative ending inventory per period. The objective minimizes the sum of
per-unit production costs, holding costs on ending inventory, and fixed setup
costs. Inventory-balance constraints link adjacent periods (no backlogging is
allowed, so inventory is always nonnegative), production is forced to an integer
multiple of the period lot size, a big-M setup linking constraint forbids
production without a setup, and a per-period production capacity limits output.

# Fields
- `n_periods::Int`: Number of time periods
- `prod_capacity::Int`: Production capacity per period
- `initial_inventory::Int`: Starting inventory level
- `demands::Vector{Int}`: Demand for each period
- `production_costs::Vector{Float64}`: Per-unit production cost per period
- `holding_costs::Vector{Float64}`: Holding cost per unit of ending inventory per period
- `setup_costs::Vector{Float64}`: Fixed setup cost incurred when producing in a period
- `lot_sizes::Vector{Int}`: Fixed batch (lot) size for each period
"""
struct LotSizingInventoryProblem <: ProblemGenerator
    n_periods::Int
    prod_capacity::Int
    initial_inventory::Int
    demands::Vector{Int}
    production_costs::Vector{Float64}
    holding_costs::Vector{Float64}
    setup_costs::Vector{Float64}
    lot_sizes::Vector{Int}
end

"""
    LotSizingInventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a capacitated lot-sizing inventory problem instance.

Variables in the model are, per period `t`: production `x[t]`, ending inventory
`I[t]` (plus the fixed initial state `I[0]`), the binary setup `y[t]`, and the
integer lot count `n_lots[t]`. The total variable count is therefore
`4 * n_periods + 1` (the `+1` is `I[0]`), so `n_periods` is sized to
`round((target_variables - 1) / 4)` to hit the target.

# Arguments
- `target_variables`: Target number of variables (≈ 4 × n_periods + 1)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function LotSizingInventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Variable count = x[1:n] + I[0:n] + y[1:n] (Bin) + n_lots[1:n] (Int)
    #               = n + (n+1) + n + n = 4*n + 1
    # => n_periods ≈ (target_variables - 1) / 4
    n_periods = max(2, min(2000, round(Int, (target_variables - 1) / 4)))

    # Determine business scale by target size
    scale = target_variables <= 250 ? :small :
            target_variables <= 1000 ? :medium : :large

    # Scale-specific ranges
    if scale == :small
        prod_capacity = round(Int, rand(Uniform(50, 500)))
        demand_base = round(Int, rand(Uniform(10, 100)))
        demand_vol = rand(Uniform(0.2, 0.5))
        prod_cost_base = rand(Uniform(10, 100))
        holding_rate = rand(Uniform(0.05, 0.25)) / 12
    elseif scale == :medium
        prod_capacity = round(Int, rand(Uniform(200, 2000)))
        demand_base = round(Int, rand(Uniform(50, 1000)))
        demand_vol = rand(Uniform(0.15, 0.4))
        prod_cost_base = rand(Uniform(5, 200))
        holding_rate = rand(Uniform(0.03, 0.20)) / 12
    else
        prod_capacity = round(Int, rand(Uniform(1000, 50000)))
        demand_base = round(Int, rand(Uniform(100, 10000)))
        demand_vol = rand(Uniform(0.1, 0.3))
        prod_cost_base = rand(Uniform(1, 500))
        holding_rate = rand(Uniform(0.01, 0.15)) / 12
    end

    demand_min = max(1, round(Int, demand_base * (1 - demand_vol)))
    demand_max = round(Int, demand_base * (1 + demand_vol))
    avgd = (demand_min + demand_max) / 2
    initial_inventory = round(Int, avgd * rand(Uniform(0.1, 0.5)))

    prod_cost_spread = rand(Uniform(0.1, 0.3))
    prod_cost_min = prod_cost_base * (1 - prod_cost_spread)
    prod_cost_max = prod_cost_base * (1 + prod_cost_spread)

    holding_cost_min = max(0.01, prod_cost_base * holding_rate * 0.8)
    holding_cost_max = prod_cost_base * holding_rate * 1.2

    # Generate base demands with seasonality
    demand_mean = (demand_min + demand_max) / 2
    demand_std = (demand_max - demand_min) / 4
    demands = round.(Int, clamp.(rand(Normal(demand_mean, demand_std), n_periods), demand_min, demand_max))

    # Add seasonality
    if rand() < 0.6 && n_periods >= 12
        annual = 1.0 .+ 0.2 * sin.(2π .* (1:n_periods) ./ 12)
        demands = round.(Int, demands .* annual)
    end

    # Production and holding costs
    prod_cost_mean = (prod_cost_min + prod_cost_max) / 2
    prod_cost_std = (prod_cost_max - prod_cost_min) / 4
    production_costs = clamp.(rand(Normal(prod_cost_mean, prod_cost_std), n_periods),
                              prod_cost_min, prod_cost_max)

    holding_cost_mean = (holding_cost_min + holding_cost_max) / 2
    holding_cost_std = (holding_cost_max - holding_cost_min) / 4
    holding_costs = clamp.(rand(Normal(holding_cost_mean, holding_cost_std), n_periods),
                           holding_cost_min, holding_cost_max)

    # Keep demands positive
    demands = max.(demands, 1)

    # --- Lot-sizing specific data ---
    lot_size_options = [5, 10, 20, 25, 50, 100]
    base_lot_size = rand(lot_size_options)
    lot_sizes = fill(base_lot_size, n_periods)
    setup_costs = production_costs .* rand(Uniform(10.0, 50.0))

    # --- Feasibility handling ---
    # No backlogging: inventory must always cover cumulative demand. Production
    # is restricted to integer multiples of the lot size, so a single period may
    # need to produce up to one extra lot beyond raw demand (integer-lot
    # rounding). We control feasibility via the production capacity.
    cum_demands = cumsum(demands)
    max_lot = maximum(lot_sizes)

    # Minimal feasible per-period capacity if all demand had to be produced in
    # one period (a conservative bound), inflated by one lot to absorb rounding.
    peak_demand = maximum(demands)

    if feasibility_status == feasible
        # Guarantee feasibility with margin: every period can produce enough lots
        # to cover its own demand plus carryover rounding. Setting capacity to at
        # least (peak demand + 2 lots) ensures a per-period produce-to-demand
        # policy (rounded up to whole lots) is always within capacity, so the
        # MIP is provably solvable.
        required = peak_demand + 2 * max_lot
        prod_capacity = max(prod_capacity, required)

    elseif feasibility_status == infeasible
        # Guarantee infeasibility with margin: make total production capacity
        # strictly smaller than total net demand. Even producing at full
        # capacity in every period cannot meet cumulative demand, regardless of
        # lot-size rounding.
        total_net_demand = max(1, cum_demands[end] - initial_inventory)
        # Per-period capacity such that n_periods * cap < total_net_demand with
        # a clear ~30% margin below the strict requirement.
        feasible_avg_cap = total_net_demand / n_periods
        prod_capacity = max(0, floor(Int, feasible_avg_cap * 0.7))
        # Ensure capacity is below even a single peak demand and not lot-aligned
        # enough to fully cover any large-demand period.
        prod_capacity = min(prod_capacity, max(0, peak_demand - 1))
    end
    # For unknown, leave capacity as the sampled value (natural instance).

    return LotSizingInventoryProblem(
        n_periods, prod_capacity, initial_inventory,
        demands, production_costs, holding_costs, setup_costs, lot_sizes,
    )
end

"""
    build_model(prob::LotSizingInventoryProblem)

Build a JuMP model for the capacitated lot-sizing inventory problem. Deterministic —
uses only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::LotSizingInventoryProblem)
    model = Model()

    n = prob.n_periods

    # Variables (total = 4*n + 1)
    @variable(model, x[1:n] >= 0)             # production quantity per period
    @variable(model, I[0:n] >= 0)             # ending inventory per period (I[0] = initial)
    @variable(model, y[1:n], Bin)             # setup indicator per period
    @variable(model, n_lots[1:n] >= 0, Int)   # number of lots produced per period

    # Objective: production + holding + setup costs
    @objective(model, Min,
        sum(prob.production_costs[t] * x[t] + prob.holding_costs[t] * I[t] +
            prob.setup_costs[t] * y[t] for t in 1:n))

    # Initial inventory
    @constraint(model, I[0] == prob.initial_inventory)

    M = prob.prod_capacity * 2  # Big-M for setup linking
    for t in 1:n
        # Inventory balance (no backlog: I[t] >= 0 enforced by variable bound)
        @constraint(model, I[t-1] + x[t] - prob.demands[t] == I[t])
        # Production only allowed if setup occurs
        @constraint(model, x[t] <= M * y[t])
        # Production must be an integer number of lots
        @constraint(model, x[t] == prob.lot_sizes[t] * n_lots[t])
        # Production capacity
        @constraint(model, x[t] <= prob.prod_capacity)
    end

    return model
end

# Register the variant
register_variant(
    :inventory,
    :lot_sizing,
    LotSizingInventoryProblem,
    "Capacitated lot-sizing MIP with fixed batch sizes, binary setups, and integer lot counts",
)
