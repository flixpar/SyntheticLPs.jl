using JuMP
using Random
using Distributions
using Statistics

"""
    MultiEchelonInventoryProblem <: ProblemGenerator

Generator for two-level (echelon) multi-location inventory distribution problems.

# Overview
Models a two-echelon distribution network over a planning horizon. A single
production source replenishes a central warehouse (location 1). The central
warehouse holds stock and ships it, via a STAR topology, to retail locations
(locations 2..L); retail stock may also be returned to the central warehouse.
Each location keeps its own inventory and faces its own per-period demand. The
decisions are period production into the central warehouse, transfers along the
star (central <-> retail only), and ending inventory at every location.

The objective minimizes total production, holding, and transfer cost. Per-
location inventory-balance constraints link adjacent periods: at the central
warehouse, production and inbound returns add to stock (inflow, +) while
outbound transfers to retail subtract (outflow, -); at each retail location,
inbound transfers from the central warehouse add to stock while outbound
returns subtract. Production is capacity-limited each period, and every
location has a storage capacity limiting its ending inventory.

Transfers are restricted to the star: only `transfer_to[l, t]` (central -> retail l)
and `transfer_from[l, t]` (retail l -> central) exist for retail locations
`l in 2..L`. No retail-to-retail lanes and no self-transfer (`transfer[i, i]`)
variables are generated.

# Fields
- `n_periods::Int`: Number of time periods
- `n_locations::Int`: Number of locations (location 1 is the central warehouse)
- `prod_capacity::Int`: Production capacity into the central warehouse per period
- `initial_inventory::Vector{Float64}`: Starting inventory at each location
- `location_demands::Matrix{Int}`: Demand at each location per period (L x T)
- `location_capacities::Vector{Float64}`: Storage capacity per location
- `production_costs::Vector{Float64}`: Production cost per period
- `holding_costs::Vector{Float64}`: Holding cost per period (applied to total stock held)
- `transfer_cost_to::Vector{Float64}`: Per-unit cost central -> retail l
- `transfer_cost_from::Vector{Float64}`: Per-unit cost retail l -> central
"""
struct MultiEchelonInventoryProblem <: ProblemGenerator
    n_periods::Int
    n_locations::Int
    prod_capacity::Int
    initial_inventory::Vector{Float64}
    location_demands::Matrix{Int}
    location_capacities::Vector{Float64}
    production_costs::Vector{Float64}
    holding_costs::Vector{Float64}
    transfer_cost_to::Vector{Float64}
    transfer_cost_from::Vector{Float64}
end

"""
    MultiEchelonInventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a two-echelon multi-location inventory distribution problem instance.

Variables (star topology, no diagonal transfers):
- production `x[1:T]`                                  => T
- inventory `I[1:L, 0:T]`                              => L*(T+1)
- transfers central->retail `transfer_to[2:L, 1:T]`   => (L-1)*T
- transfers retail->central `transfer_from[2:L, 1:T]` => (L-1)*T

Total = T + L*(T+1) + 2*(L-1)*T = T*(3L - 1) + L.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function MultiEchelonInventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Business scale by target size
    scale = target_variables <= 250 ? :small :
            target_variables <= 1000 ? :medium : :large

    # Number of locations (central + retail). More for larger problems.
    if scale == :small
        n_locations = rand(2:3)
    elseif scale == :medium
        n_locations = rand(3:5)
    else
        n_locations = rand(4:8)
    end

    # Size n_periods to hit target var count.
    # total_vars = T*(3L - 1) + L  =>  T = (target - L) / (3L - 1)
    per_period = 3 * n_locations - 1
    n_periods = max(2, round(Int, (target_variables - n_locations) / per_period))

    # Iterative refinement toward target var count
    var_count(T) = T * (3 * n_locations - 1) + n_locations
    for _ in 1:20
        current = var_count(n_periods)
        if abs(current - target_variables) / target_variables < 0.05
            break
        end
        if current < target_variables
            n_periods += 1
        elseif n_periods > 2
            n_periods -= 1
        else
            break
        end
    end
    n_periods = max(2, n_periods)

    # Scale-specific parameter ranges
    if scale == :small
        prod_capacity = round(Int, rand(Uniform(100, 800)))
        demand_base = round(Int, rand(Uniform(10, 100)))
        demand_vol = rand(Uniform(0.2, 0.5))
        prod_cost_base = rand(Uniform(10, 100))
        holding_rate = rand(Uniform(0.05, 0.25)) / 12
    elseif scale == :medium
        prod_capacity = round(Int, rand(Uniform(500, 4000)))
        demand_base = round(Int, rand(Uniform(50, 800)))
        demand_vol = rand(Uniform(0.15, 0.4))
        prod_cost_base = rand(Uniform(5, 200))
        holding_rate = rand(Uniform(0.03, 0.20)) / 12
    else
        prod_capacity = round(Int, rand(Uniform(2000, 60000)))
        demand_base = round(Int, rand(Uniform(100, 8000)))
        demand_vol = rand(Uniform(0.1, 0.3))
        prod_cost_base = rand(Uniform(1, 500))
        holding_rate = rand(Uniform(0.01, 0.15)) / 12
    end

    demand_min = max(1, round(Int, demand_base * (1 - demand_vol)))
    demand_max = round(Int, demand_base * (1 + demand_vol))
    demand_mean = (demand_min + demand_max) / 2
    demand_std = (demand_max - demand_min) / 4

    # Per-location demands (each location gets a share of the base demand).
    location_demands = zeros(Int, n_locations, n_periods)
    for l in 1:n_locations
        loc_base = demand_base * rand(Uniform(0.3, 1.2)) / n_locations
        loc_std = max(1.0, loc_base * 0.3)
        series = rand(Normal(loc_base, loc_std), n_periods)
        # Optional seasonality
        if rand() < 0.5 && n_periods >= 12
            annual = 1.0 .+ 0.2 * sin.(2π .* (1:n_periods) ./ 12)
            series = series .* annual
        end
        location_demands[l, :] = round.(Int, clamp.(series, 1, loc_base * 3 + 1))
    end
    location_demands = max.(location_demands, 1)

    # Production & holding cost series
    prod_cost_spread = rand(Uniform(0.1, 0.3))
    prod_cost_min = prod_cost_base * (1 - prod_cost_spread)
    prod_cost_max = prod_cost_base * (1 + prod_cost_spread)
    prod_cost_mean = (prod_cost_min + prod_cost_max) / 2
    prod_cost_std = (prod_cost_max - prod_cost_min) / 4
    production_costs = clamp.(rand(Normal(prod_cost_mean, prod_cost_std), n_periods),
                             prod_cost_min, prod_cost_max)

    holding_cost_min = max(0.01, prod_cost_base * holding_rate * 0.8)
    holding_cost_max = prod_cost_base * holding_rate * 1.2
    holding_cost_mean = (holding_cost_min + holding_cost_max) / 2
    holding_cost_std = (holding_cost_max - holding_cost_min) / 4
    holding_costs = clamp.(rand(Normal(holding_cost_mean, holding_cost_std), n_periods),
                           holding_cost_min, holding_cost_max)

    # Transfer costs (per retail location, both directions). Index 1 unused (central).
    mean_prod_cost = mean(production_costs)
    transfer_cost_to = zeros(n_locations)
    transfer_cost_from = zeros(n_locations)
    for l in 2:n_locations
        transfer_cost_to[l] = mean_prod_cost * rand(Uniform(0.05, 0.20))
        transfer_cost_from[l] = mean_prod_cost * rand(Uniform(0.05, 0.20))
    end

    # Initial inventory split across locations
    avgd_total = sum(demand_mean for _ in 1:n_locations)  # rough total per-period demand
    initial_inventory = zeros(Float64, n_locations)
    for l in 1:n_locations
        loc_avg = mean(location_demands[l, :])
        initial_inventory[l] = loc_avg * rand(Uniform(0.1, 0.5))
    end

    # Location storage capacities (multiple of that location's average demand)
    location_capacities = zeros(Float64, n_locations)
    for l in 1:n_locations
        loc_avg = mean(location_demands[l, :])
        location_capacities[l] = loc_avg * rand(Uniform(3.0, 8.0))
    end

    # ---------- Feasibility handling ----------
    # Aggregate prefix condition: since the star routes everything through the
    # central warehouse, total demand up to period t must be coverable by total
    # initial inventory plus cumulative production:
    #   sum_{s<=t} total_demand[s] <= sum(initial_inventory) + t * prod_capacity
    total_demand_per_period = vec(sum(location_demands, dims=1))
    cum_total = cumsum(total_demand_per_period)
    total_init = sum(initial_inventory)

    function max_prefix_shortfall(cap::Real)
        sf = 0.0
        for t in 1:n_periods
            s = cum_total[t] - (total_init + t * cap)
            if s > sf
                sf = s
            end
        end
        return sf
    end

    actual_status = feasibility_status
    if feasibility_status == unknown
        # leave natural; no forced (in)feasibility
        actual_status = unknown
    end

    if actual_status == feasible
        # Ensure aggregate production can cover all demand with margin.
        required_caps = [ceil(Int, max(0.0, cum_total[t] - total_init) / t) for t in 1:n_periods]
        min_cap_needed = maximum(required_caps)
        prod_capacity = max(prod_capacity, ceil(Int, min_cap_needed * 1.10) + 1)

        # Ensure each location can physically hold a period's worth of supply:
        # capacity must cover both its own peak demand and pass-through staging.
        for l in 1:n_locations
            peak = maximum(location_demands[l, :])
            location_capacities[l] = max(location_capacities[l], peak * 1.5 + initial_inventory[l] + 1.0)
        end
        # Central warehouse may need to stage the whole network's flow.
        peak_total = maximum(total_demand_per_period)
        location_capacities[1] = max(location_capacities[1], peak_total * 1.5 + prod_capacity + total_init + 1.0)

    elseif actual_status == infeasible
        # Deterministic contradiction: shrink production capacity below what the
        # aggregate prefix demand requires, with a clear margin. With no backlog
        # (inventory is nonnegative everywhere), the network cannot meet demand.
        required_caps = [ceil(Int, max(0.0, cum_total[t] - total_init) / t) for t in 1:n_periods]
        min_cap_needed = maximum(required_caps)
        margin = max(1, round(Int, 0.10 * max(1, min_cap_needed)))
        new_cap = max(0, min_cap_needed - margin)
        # Guarantee an actual prefix shortfall remains.
        while new_cap > 0 && max_prefix_shortfall(new_cap) <= 0
            new_cap -= 1
        end
        prod_capacity = new_cap
        # Also shrink initial inventory so the contradiction is robust.
        initial_inventory .*= 0.25
    end

    return MultiEchelonInventoryProblem(
        n_periods, n_locations, prod_capacity, initial_inventory,
        location_demands, location_capacities,
        production_costs, holding_costs,
        transfer_cost_to, transfer_cost_from,
    )
end

"""
    build_model(prob::MultiEchelonInventoryProblem)

Build a JuMP model for the two-echelon inventory distribution problem.
Deterministic — uses only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::MultiEchelonInventoryProblem)
    model = Model()

    T = prob.n_periods
    L = prob.n_locations

    # Variables (star topology, no diagonal transfers)
    @variable(model, x[1:T] >= 0)                       # production into central warehouse
    @variable(model, I[1:L, 0:T] >= 0)                  # inventory at each location
    @variable(model, transfer_to[2:L, 1:T] >= 0)        # central -> retail l
    @variable(model, transfer_from[2:L, 1:T] >= 0)      # retail l -> central

    # Objective: production + holding (total stock) + transfer costs
    @objective(model, Min,
        sum(prob.production_costs[t] * x[t] for t in 1:T) +
        sum(prob.holding_costs[t] * sum(I[l, t] for l in 1:L) for t in 1:T) +
        sum(prob.transfer_cost_to[l] * transfer_to[l, t] +
            prob.transfer_cost_from[l] * transfer_from[l, t]
            for l in 2:L, t in 1:T))

    # Initial inventory at each location
    for l in 1:L
        @constraint(model, I[l, 0] == prob.initial_inventory[l])
    end

    for t in 1:T
        # Production capacity at the source
        @constraint(model, x[t] <= prob.prod_capacity)

        # Central warehouse (location 1) balance:
        #   prev stock + production + inbound returns (+)
        #   - outbound shipments to retail (-) - own demand = ending stock
        @constraint(model,
            I[1, t-1] + x[t]
            + sum(transfer_from[l, t] for l in 2:L)
            - sum(transfer_to[l, t] for l in 2:L)
            - prob.location_demands[1, t] == I[1, t])

        # Retail locations (2..L) balance:
        #   prev stock + inbound from central (+) - outbound return (-) - demand = ending stock
        for l in 2:L
            @constraint(model,
                I[l, t-1] + transfer_to[l, t] - transfer_from[l, t]
                - prob.location_demands[l, t] == I[l, t])
        end

        # Storage capacity per location
        for l in 1:L
            @constraint(model, I[l, t] <= prob.location_capacities[l])
        end
    end

    return model
end

# Register the variant
register_variant(
    :inventory,
    :multi_echelon,
    MultiEchelonInventoryProblem,
    "Two-echelon inventory distribution: production feeds a central warehouse that supplies retail locations over a star transfer network with per-location inventory balance",
)
