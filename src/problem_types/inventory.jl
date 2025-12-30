using JuMP
using Random
using Distributions
using Statistics

"""
Inventory problem variants.

# Variants
- `inv_standard`: Basic inventory with optional backlogging
- `inv_safety_stock`: Minimum inventory levels required
- `inv_warehouse_capacity`: Maximum storage space constraints
- `inv_lot_sizing`: Production in fixed batch sizes
- `inv_multi_item`: Multiple items with shared capacity
- `inv_perishable`: Shelf life/spoilage constraints
- `inv_service_level`: Fill rate requirements
- `inv_multi_echelon`: Two-level warehouse-retail system
"""
@enum InventoryVariant begin
    inv_standard
    inv_safety_stock
    inv_warehouse_capacity
    inv_lot_sizing
    inv_multi_item
    inv_perishable
    inv_service_level
    inv_multi_echelon
end

"""
    InventoryProblem <: ProblemGenerator

Generator for inventory control problems with multiple variants.

# Fields
- `n_periods::Int`: Number of time periods
- `prod_capacity::Int`: Production capacity per period
- `initial_inventory::Int`: Starting inventory level
- `backlog_allowed::Bool`: Whether backorders are permitted
- `demands::Vector{Int}`: Demand for each period
- `production_costs::Vector{Float64}`: Production cost per period
- `holding_costs::Vector{Float64}`: Holding cost per period
- `backlog_costs::Vector{Float64}`: Backlog/shortage cost per period
- `variant::InventoryVariant`: The specific variant type
# Safety stock variant
- `safety_stock_levels::Union{Vector{Float64}, Nothing}`: Minimum inventory levels
# Warehouse capacity variant
- `warehouse_capacity::Union{Float64, Nothing}`: Maximum storage capacity
- `storage_costs::Union{Vector{Float64}, Nothing}`: Per-period storage costs
# Lot sizing variant
- `lot_sizes::Union{Vector{Int}, Nothing}`: Fixed batch sizes per period
- `setup_costs::Union{Vector{Float64}, Nothing}`: Fixed setup costs
# Multi-item variant
- `n_items::Int`: Number of items
- `item_demands::Union{Matrix{Int}, Nothing}`: Demand matrix (items × periods)
- `item_production_costs::Union{Matrix{Float64}, Nothing}`: Production costs (items × periods)
- `item_holding_costs::Union{Matrix{Float64}, Nothing}`: Holding costs (items × periods)
- `item_initial_inventory::Union{Vector{Int}, Nothing}`: Initial inventory per item
- `item_resource_usage::Union{Vector{Float64}, Nothing}`: Resource usage per unit
# Perishable variant
- `shelf_life::Union{Int, Nothing}`: Maximum periods in inventory
- `spoilage_costs::Union{Vector{Float64}, Nothing}`: Cost of spoiled goods
# Service level variant
- `target_fill_rate::Union{Float64, Nothing}`: Target proportion of demand met
- `shortage_penalty::Union{Float64, Nothing}`: High penalty for shortages
# Multi-echelon variant
- `n_locations::Int`: Number of warehouse locations
- `transfer_costs::Union{Matrix{Float64}, Nothing}`: Cost to transfer between locations
- `location_capacities::Union{Vector{Float64}, Nothing}`: Capacity per location
- `location_demands::Union{Matrix{Int}, Nothing}`: Demand at each location
"""
struct InventoryProblem <: ProblemGenerator
    n_periods::Int
    prod_capacity::Int
    initial_inventory::Int
    backlog_allowed::Bool
    demands::Vector{Int}
    production_costs::Vector{Float64}
    holding_costs::Vector{Float64}
    backlog_costs::Vector{Float64}
    variant::InventoryVariant
    # Safety stock variant
    safety_stock_levels::Union{Vector{Float64}, Nothing}
    # Warehouse capacity variant
    warehouse_capacity::Union{Float64, Nothing}
    storage_costs::Union{Vector{Float64}, Nothing}
    # Lot sizing variant
    lot_sizes::Union{Vector{Int}, Nothing}
    setup_costs::Union{Vector{Float64}, Nothing}
    # Multi-item variant
    n_items::Int
    item_demands::Union{Matrix{Int}, Nothing}
    item_production_costs::Union{Matrix{Float64}, Nothing}
    item_holding_costs::Union{Matrix{Float64}, Nothing}
    item_initial_inventory::Union{Vector{Int}, Nothing}
    item_resource_usage::Union{Vector{Float64}, Nothing}
    # Perishable variant
    shelf_life::Union{Int, Nothing}
    spoilage_costs::Union{Vector{Float64}, Nothing}
    # Service level variant
    target_fill_rate::Union{Float64, Nothing}
    shortage_penalty::Union{Float64, Nothing}
    # Multi-echelon variant
    n_locations::Int
    transfer_costs::Union{Matrix{Float64}, Nothing}
    location_capacities::Union{Vector{Float64}, Nothing}
    location_demands::Union{Matrix{Int}, Nothing}
end

# Backwards compatibility
function InventoryProblem(n_periods::Int, prod_capacity::Int, initial_inventory::Int,
                          backlog_allowed::Bool, demands::Vector{Int},
                          production_costs::Vector{Float64}, holding_costs::Vector{Float64},
                          backlog_costs::Vector{Float64})
    InventoryProblem(
        n_periods, prod_capacity, initial_inventory, backlog_allowed,
        demands, production_costs, holding_costs, backlog_costs, inv_standard,
        nothing, nothing, nothing, nothing, nothing,
        1, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing,
        1, nothing, nothing, nothing
    )
end

"""
    InventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                     variant::InventoryVariant=inv_standard)

Construct an inventory control problem instance with the specified variant.
"""
function InventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                          variant::InventoryVariant=inv_standard)
    Random.seed!(seed)

    # Determine business scale by target size
    scale = target_variables <= 250 ? :small :
            target_variables <= 1000 ? :medium : :large

    # Backlog incidence
    backlog_prob = scale == :small ? 0.2 : (scale == :medium ? 0.4 : 0.6)
    backlog_allowed = variant == inv_standard ? rand(Bernoulli(backlog_prob)) : false

    # Calculate n_periods based on variant
    if variant == inv_multi_item
        # Variables: n_items × n_periods (production) + n_items × (n_periods+1) (inventory)
        n_items = rand(max(2, target_variables ÷ 50):max(5, target_variables ÷ 20))
        n_periods = max(2, min(500, target_variables ÷ (2 * n_items)))
    elseif variant == inv_multi_echelon
        # Multiple locations with their own inventories
        n_locations = rand(2:min(5, max(2, target_variables ÷ 100)))
        n_periods = max(2, min(500, target_variables ÷ (3 * n_locations)))
    elseif variant == inv_perishable
        # Additional vintage tracking variables
        n_periods = max(4, min(500, target_variables ÷ 4))
    else
        # Standard variants
        if backlog_allowed
            n_periods = max(2, min(5000, round(Int, (target_variables - 2) / 3)))
        else
            n_periods = max(2, min(5000, round(Int, (target_variables - 1) / 2)))
        end
    end

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

    backlog_cost_factor = rand(Uniform(1.5, 5.0))

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

    backlog_costs = production_costs .* backlog_cost_factor

    # Keep demands positive
    demands = max.(demands, 1)

    # Initialize variant-specific fields
    safety_stock_levels = nothing
    warehouse_capacity = nothing
    storage_costs = nothing
    lot_sizes = nothing
    setup_costs = nothing
    n_items = 1
    item_demands = nothing
    item_production_costs = nothing
    item_holding_costs = nothing
    item_initial_inventory = nothing
    item_resource_usage = nothing
    shelf_life = nothing
    spoilage_costs = nothing
    target_fill_rate = nothing
    shortage_penalty = nothing
    n_locations = 1
    transfer_costs = nothing
    location_capacities = nothing
    location_demands = nothing

    # Generate variant-specific data
    if variant == inv_safety_stock
        # Safety stock as percentage of average demand
        avg_demand = mean(demands)
        safety_ratio = rand(Uniform(0.1, 0.4))
        safety_stock_levels = fill(avg_demand * safety_ratio, n_periods)
        # Add some variability
        for t in 1:n_periods
            safety_stock_levels[t] *= rand(Uniform(0.9, 1.1))
        end

    elseif variant == inv_warehouse_capacity
        # Warehouse can hold some multiple of average demand
        avg_demand = mean(demands)
        warehouse_capacity = avg_demand * rand(Uniform(2.0, 5.0))
        storage_costs = holding_costs .* rand(Uniform(0.2, 0.5))

    elseif variant == inv_lot_sizing
        # Production in fixed batch sizes
        lot_size_options = [5, 10, 20, 25, 50, 100]
        base_lot_size = rand(lot_size_options)
        lot_sizes = fill(base_lot_size, n_periods)
        setup_costs = production_costs .* rand(Uniform(10.0, 50.0))
        # Disable backlogging for lot sizing
        backlog_allowed = false

    elseif variant == inv_multi_item
        n_items = rand(max(2, target_variables ÷ 50):max(5, target_variables ÷ 20))

        # Generate demands for each item
        item_demands = zeros(Int, n_items, n_periods)
        for i in 1:n_items
            item_base = demand_base * rand(Uniform(0.3, 1.5))
            item_demands[i, :] = round.(Int, clamp.(
                rand(Normal(item_base, item_base * 0.25), n_periods),
                max(1, item_base * 0.3), item_base * 2.0
            ))
        end

        # Production and holding costs per item
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

        item_initial_inventory = round.(Int, [mean(item_demands[i, :]) * rand(Uniform(0.1, 0.4))
                                               for i in 1:n_items])
        item_resource_usage = [rand(Uniform(0.5, 2.0)) for _ in 1:n_items]

    elseif variant == inv_perishable
        # Shelf life in periods
        shelf_life = rand(2:min(6, n_periods ÷ 2))
        spoilage_costs = production_costs .* rand(Uniform(0.5, 1.0))
        # No backlogging for perishables
        backlog_allowed = false

    elseif variant == inv_service_level
        # High service level requirements
        target_fill_rate = rand(Uniform(0.90, 0.99))
        shortage_penalty = mean(production_costs) * rand(Uniform(5.0, 20.0))
        backlog_allowed = true  # Allow backlog to measure service

    elseif variant == inv_multi_echelon
        n_locations = rand(2:min(5, max(2, target_variables ÷ 100)))

        # Demands at each location
        location_demands = zeros(Int, n_locations, n_periods)
        for l in 1:n_locations
            loc_base = demand_base * rand(Uniform(0.2, 1.0)) / n_locations
            location_demands[l, :] = round.(Int, clamp.(
                rand(Normal(loc_base, loc_base * 0.3), n_periods),
                1, loc_base * 3
            ))
        end

        # Transfer costs between locations
        transfer_costs = zeros(n_locations, n_locations)
        for i in 1:n_locations, j in 1:n_locations
            if i != j
                transfer_costs[i, j] = mean(production_costs) * rand(Uniform(0.05, 0.2))
            end
        end

        # Location capacities
        avg_loc_demand = mean(location_demands)
        location_capacities = [avg_loc_demand * rand(Uniform(3.0, 8.0)) for _ in 1:n_locations]
    end

    # Handle feasibility
    cum(x) = cumsum(x)
    cum_demands = cum(demands)

    function max_shortfall(cap::Int, init_inv::Int)
        max_sf = 0
        for t in 1:n_periods
            sf = cum_demands[t] - (init_inv + t * cap)
            if sf > max_sf
                max_sf = sf
            end
        end
        return max_sf
    end

    if feasibility_status == feasible
        # Ensure feasibility
        if variant == inv_safety_stock && safety_stock_levels !== nothing
            # Reduce safety stock if needed
            safety_stock_levels .*= 0.5
        end

        if variant == inv_warehouse_capacity && warehouse_capacity !== nothing
            # Increase warehouse capacity
            warehouse_capacity *= 1.5
        end

        if !backlog_allowed
            sf = max_shortfall(prod_capacity, initial_inventory)
            if sf > 0
                # Increase capacity or initial inventory
                if rand() < 0.5
                    prod_capacity = max(prod_capacity, ceil(Int, maximum(cum_demands) / n_periods * 1.2))
                else
                    initial_inventory += sf + rand(1:max(1, sf ÷ 2))
                end
            end
        end

        # For multi-item, ensure total capacity is sufficient
        if variant == inv_multi_item && item_demands !== nothing
            total_demand_per_period = sum(item_demands, dims=1)[:]
            max_total = maximum(total_demand_per_period)
            prod_capacity = max(prod_capacity, round(Int, max_total * 1.3))
        end

    elseif feasibility_status == infeasible
        backlog_allowed = false

        if variant == inv_safety_stock && safety_stock_levels !== nothing
            # Increase safety stock requirements dramatically
            safety_stock_levels .*= 3.0
        elseif variant == inv_warehouse_capacity && warehouse_capacity !== nothing
            # Make warehouse too small
            warehouse_capacity = mean(demands) * 0.3
        elseif variant == inv_multi_item && item_demands !== nothing
            # Make capacity too small for all items
            total_demand = sum(item_demands)
            prod_capacity = round(Int, total_demand / n_periods * 0.3)
        else
            # Standard infeasibility: reduce capacity drastically
            prod_capacity = round(Int, prod_capacity * 0.3)
            initial_inventory = max(0, initial_inventory ÷ 4)
            # Spike demands
            for t in 1:min(3, n_periods)
                demands[t] = round(Int, demands[t] * 2.5)
            end
        end
    end

    return InventoryProblem(
        n_periods, prod_capacity, initial_inventory, backlog_allowed,
        demands, production_costs, holding_costs, backlog_costs, variant,
        safety_stock_levels, warehouse_capacity, storage_costs,
        lot_sizes, setup_costs,
        n_items, item_demands, item_production_costs, item_holding_costs,
        item_initial_inventory, item_resource_usage,
        shelf_life, spoilage_costs,
        target_fill_rate, shortage_penalty,
        n_locations, transfer_costs, location_capacities, location_demands
    )
end

"""
    build_model(prob::InventoryProblem)

Build a JuMP model for the inventory control problem based on its variant.
"""
function build_model(prob::InventoryProblem)
    model = Model()

    if prob.variant == inv_standard
        if prob.backlog_allowed
            @variable(model, x[1:prob.n_periods] >= 0)
            @variable(model, I_plus[0:prob.n_periods] >= 0)
            @variable(model, I_minus[0:prob.n_periods] >= 0)

            @objective(model, Min,
                sum(prob.production_costs[t]*x[t] + prob.holding_costs[t]*I_plus[t] +
                    prob.backlog_costs[t]*I_minus[t] for t in 1:prob.n_periods))

            @constraint(model, I_plus[0] == prob.initial_inventory)
            @constraint(model, I_minus[0] == 0)

            for t in 1:prob.n_periods
                @constraint(model, I_plus[t-1] - I_minus[t-1] + x[t] - prob.demands[t] == I_plus[t] - I_minus[t])
                @constraint(model, x[t] <= prob.prod_capacity)
            end
        else
            @variable(model, x[1:prob.n_periods] >= 0)
            @variable(model, I[0:prob.n_periods] >= 0)

            @objective(model, Min,
                sum(prob.production_costs[t]*x[t] + prob.holding_costs[t]*I[t] for t in 1:prob.n_periods))

            @constraint(model, I[0] == prob.initial_inventory)

            for t in 1:prob.n_periods
                @constraint(model, I[t-1] + x[t] - prob.demands[t] == I[t])
                @constraint(model, x[t] <= prob.prod_capacity)
            end
        end

    elseif prob.variant == inv_safety_stock
        @variable(model, x[1:prob.n_periods] >= 0)
        @variable(model, I[0:prob.n_periods] >= 0)

        @objective(model, Min,
            sum(prob.production_costs[t]*x[t] + prob.holding_costs[t]*I[t] for t in 1:prob.n_periods))

        @constraint(model, I[0] == prob.initial_inventory)

        for t in 1:prob.n_periods
            @constraint(model, I[t-1] + x[t] - prob.demands[t] == I[t])
            @constraint(model, x[t] <= prob.prod_capacity)
            @constraint(model, I[t] >= prob.safety_stock_levels[t])  # Safety stock requirement
        end

    elseif prob.variant == inv_warehouse_capacity
        @variable(model, x[1:prob.n_periods] >= 0)
        @variable(model, I[0:prob.n_periods] >= 0)

        @objective(model, Min,
            sum(prob.production_costs[t]*x[t] + prob.holding_costs[t]*I[t] +
                prob.storage_costs[t]*I[t] for t in 1:prob.n_periods))

        @constraint(model, I[0] == prob.initial_inventory)

        for t in 1:prob.n_periods
            @constraint(model, I[t-1] + x[t] - prob.demands[t] == I[t])
            @constraint(model, x[t] <= prob.prod_capacity)
            @constraint(model, I[t] <= prob.warehouse_capacity)  # Capacity limit
        end

    elseif prob.variant == inv_lot_sizing
        @variable(model, x[1:prob.n_periods] >= 0)
        @variable(model, I[0:prob.n_periods] >= 0)
        @variable(model, y[1:prob.n_periods], Bin)  # Setup indicator
        @variable(model, n_lots[1:prob.n_periods] >= 0, Int)  # Number of lots

        # Minimize production + holding + setup costs
        @objective(model, Min,
            sum(prob.production_costs[t]*x[t] + prob.holding_costs[t]*I[t] +
                prob.setup_costs[t]*y[t] for t in 1:prob.n_periods))

        @constraint(model, I[0] == prob.initial_inventory)

        M = prob.prod_capacity * 2  # Big M
        for t in 1:prob.n_periods
            @constraint(model, I[t-1] + x[t] - prob.demands[t] == I[t])
            @constraint(model, x[t] <= M * y[t])  # Production only if setup
            @constraint(model, x[t] == prob.lot_sizes[t] * n_lots[t])  # Lot sizing
            @constraint(model, x[t] <= prob.prod_capacity)
        end

    elseif prob.variant == inv_multi_item
        @variable(model, x[1:prob.n_items, 1:prob.n_periods] >= 0)
        @variable(model, I[1:prob.n_items, 0:prob.n_periods] >= 0)

        @objective(model, Min,
            sum(prob.item_production_costs[i,t]*x[i,t] + prob.item_holding_costs[i,t]*I[i,t]
                for i in 1:prob.n_items, t in 1:prob.n_periods))

        for i in 1:prob.n_items
            @constraint(model, I[i, 0] == prob.item_initial_inventory[i])
        end

        for i in 1:prob.n_items, t in 1:prob.n_periods
            @constraint(model, I[i, t-1] + x[i, t] - prob.item_demands[i, t] == I[i, t])
        end

        # Shared production capacity
        for t in 1:prob.n_periods
            @constraint(model, sum(prob.item_resource_usage[i] * x[i, t]
                                   for i in 1:prob.n_items) <= prob.prod_capacity)
        end

    elseif prob.variant == inv_perishable
        # Track inventory by vintage (when produced)
        L = prob.shelf_life
        @variable(model, x[1:prob.n_periods] >= 0)  # Production in period t
        @variable(model, I[1:prob.n_periods, 0:prob.n_periods] >= 0)  # Inventory of vintage v at end of period t
        @variable(model, spoiled[1:prob.n_periods] >= 0)  # Spoiled goods

        @objective(model, Min,
            sum(prob.production_costs[t]*x[t] for t in 1:prob.n_periods) +
            sum(prob.holding_costs[t]*sum(I[v, t] for v in max(1, t-L+1):t) for t in 1:prob.n_periods) +
            sum(prob.spoilage_costs[t]*spoiled[t] for t in 1:prob.n_periods))

        # Initial inventory (all from vintage 0, approximated as vintage 1)
        @constraint(model, I[1, 0] == prob.initial_inventory)
        for v in 2:prob.n_periods
            @constraint(model, I[v, 0] == 0)
        end

        for t in 1:prob.n_periods
            # New production creates vintage t inventory
            @constraint(model, x[t] <= prob.prod_capacity)

            # Inventory balance - goods age and some get used
            # Total available = previous inventory (not yet spoiled) + new production
            available = (t == 1 ? prob.initial_inventory : 0) + x[t]
            for v in max(1, t-L+1):t-1
                if t - v < L
                    available = available  # Still valid, counted separately
                end
            end

            # Simplified: total inventory at end of t must satisfy demand
            @constraint(model, sum(I[v, t] for v in max(1, t-L+1):t) >= 0)

            # Spoilage: goods from vintage t-L expire at end of period t
            if t >= L
                @constraint(model, spoiled[t] >= I[t-L+1, t-1])
            end
        end

        # Meet demand with goods of valid age (simplified balance)
        for t in 1:prob.n_periods
            if t == 1
                @constraint(model, prob.initial_inventory + x[t] - prob.demands[t] ==
                            sum(I[v, t] for v in 1:t))
            else
                @constraint(model, sum(I[v, t-1] for v in max(1, t-L):t-1) + x[t] - prob.demands[t] ==
                            sum(I[v, t] for v in max(1, t-L+1):t))
            end
        end

    elseif prob.variant == inv_service_level
        @variable(model, x[1:prob.n_periods] >= 0)
        @variable(model, I_plus[0:prob.n_periods] >= 0)  # On-hand inventory
        @variable(model, I_minus[0:prob.n_periods] >= 0)  # Backlog
        @variable(model, sales[1:prob.n_periods] >= 0)  # Actual sales

        @objective(model, Min,
            sum(prob.production_costs[t]*x[t] + prob.holding_costs[t]*I_plus[t] +
                prob.shortage_penalty*I_minus[t] for t in 1:prob.n_periods))

        @constraint(model, I_plus[0] == prob.initial_inventory)
        @constraint(model, I_minus[0] == 0)

        for t in 1:prob.n_periods
            @constraint(model, I_plus[t-1] - I_minus[t-1] + x[t] - sales[t] == I_plus[t] - I_minus[t])
            @constraint(model, x[t] <= prob.prod_capacity)
            @constraint(model, sales[t] <= prob.demands[t])  # Can't sell more than demanded
            # Service level: sales must be at least target_fill_rate of demand
            @constraint(model, sales[t] >= prob.target_fill_rate * prob.demands[t])
        end

    elseif prob.variant == inv_multi_echelon
        # Central warehouse (location 1) supplies other locations
        @variable(model, x[1:prob.n_periods] >= 0)  # Production to central warehouse
        @variable(model, I[1:prob.n_locations, 0:prob.n_periods] >= 0)  # Inventory at each location
        @variable(model, transfer[1:prob.n_locations, 1:prob.n_locations, 1:prob.n_periods] >= 0)

        @objective(model, Min,
            sum(prob.production_costs[t]*x[t] for t in 1:prob.n_periods) +
            sum(prob.holding_costs[t]*sum(I[l, t] for l in 1:prob.n_locations) for t in 1:prob.n_periods) +
            sum(prob.transfer_costs[i, j]*transfer[i, j, t]
                for i in 1:prob.n_locations, j in 1:prob.n_locations, t in 1:prob.n_periods if i != j))

        # Initial inventory
        for l in 1:prob.n_locations
            @constraint(model, I[l, 0] == prob.initial_inventory / prob.n_locations)
        end

        # Central warehouse receives production
        for t in 1:prob.n_periods
            @constraint(model, x[t] <= prob.prod_capacity)

            # Central warehouse balance (location 1)
            @constraint(model, I[1, t-1] + x[t] -
                        sum(transfer[1, j, t] for j in 2:prob.n_locations) +
                        sum(transfer[j, 1, t] for j in 2:prob.n_locations) -
                        prob.location_demands[1, t] == I[1, t])

            # Other locations balance
            for l in 2:prob.n_locations
                @constraint(model, I[l, t-1] +
                            sum(transfer[i, l, t] for i in 1:prob.n_locations if i != l) -
                            sum(transfer[l, j, t] for j in 1:prob.n_locations if j != l) -
                            prob.location_demands[l, t] == I[l, t])
            end

            # Location capacity constraints
            for l in 1:prob.n_locations
                @constraint(model, I[l, t] <= prob.location_capacities[l])
            end
        end
    end

    return model
end

# Register the problem type
register_problem(
    :inventory,
    InventoryProblem,
    "Inventory control problem with variants including standard, safety stock, warehouse capacity, lot sizing, multi-item, perishable, service level, and multi-echelon"
)
