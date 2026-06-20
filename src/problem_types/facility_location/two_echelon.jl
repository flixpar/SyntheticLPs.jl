using JuMP
using Random
using Distributions

"""
    TwoEchelonFacilityLocationProblem <: ProblemGenerator

Generator for two-echelon capacitated facility location problems with discrete
warehouse sizing.

# Overview
Models a two-echelon distribution network: suppliers → warehouses → customers.
The decisions are which candidate warehouses to open, what discrete capacity size
to install at each opened warehouse, and the flows on both echelons (supplier→
warehouse and warehouse→customer). The objective minimizes fixed opening cost,
size-installation cost, two-echelon transport cost (distance-based), and per-unit
handling cost. Constraints enforce: at most one size per warehouse (linked to the
open decision), supplier supply limits, customer demand satisfaction, warehouse
throughput bounded by the chosen size, flow conservation at warehouses (inflow ≥
outflow), and that inbound flow only occurs at opened warehouses.

# Fields
- `n_warehouses::Int`: Number of candidate warehouse locations
- `n_suppliers::Int`: Number of suppliers
- `n_customers::Int`: Number of customers
- `warehouse_locations::Vector{Tuple{Float64,Float64}}`: Warehouse coordinates
- `supplier_locations::Vector{Tuple{Float64,Float64}}`: Supplier coordinates
- `customer_locations::Vector{Tuple{Float64,Float64}}`: Customer coordinates
- `supplier_capacities::Vector{Float64}`: Supply capacity at each supplier
- `customer_demands::Vector{Float64}`: Demand at each customer
- `warehouse_fixed_costs::Vector{Float64}`: Fixed cost to open each warehouse
- `warehouse_size_options::Vector{Float64}`: Available throughput capacity sizes
- `warehouse_size_costs::Vector{Float64}`: Installation cost for each size option
- `supplier_warehouse_costs::Matrix{Float64}`: Transport cost supplier→warehouse (S×W)
- `warehouse_customer_costs::Matrix{Float64}`: Transport cost warehouse→customer (W×C)
- `handling_costs::Vector{Float64}`: Per-unit handling cost at each warehouse
"""
struct TwoEchelonFacilityLocationProblem <: ProblemGenerator
    n_warehouses::Int
    n_suppliers::Int
    n_customers::Int
    warehouse_locations::Vector{Tuple{Float64,Float64}}
    supplier_locations::Vector{Tuple{Float64,Float64}}
    customer_locations::Vector{Tuple{Float64,Float64}}
    supplier_capacities::Vector{Float64}
    customer_demands::Vector{Float64}
    warehouse_fixed_costs::Vector{Float64}
    warehouse_size_options::Vector{Float64}
    warehouse_size_costs::Vector{Float64}
    supplier_warehouse_costs::Matrix{Float64}
    warehouse_customer_costs::Matrix{Float64}
    handling_costs::Vector{Float64}
end

"""
    TwoEchelonFacilityLocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a two-echelon capacitated facility location and sizing instance.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Details
Variables: y[w] (open, Bin), z[w,k] (size choice, Bin), f1[s,w] (supplier→warehouse
flow), f2[w,c] (warehouse→customer flow). Total variable count:

    n_warehouses × (1 + n_size_options + n_suppliers + n_customers)

Dimensions are sized in the constructor to hit `target_variables`.
"""
function TwoEchelonFacilityLocationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine scale
    if target_variables <= 150
        min_wh, max_wh = 2, 15
        min_supp, max_supp = 2, 10
        min_cust, max_cust = 3, 20
        n_size_options = 3
        grid_size = rand(100.0:20.0:300.0)
        demand_range = (10.0, 100.0)
        supply_range = (50.0, 500.0)
        transport_cost = rand(0.5:0.1:1.5)
    elseif target_variables <= 800
        min_wh, max_wh = 5, 30
        min_supp, max_supp = 3, 20
        min_cust, max_cust = 10, 50
        n_size_options = 4
        grid_size = rand(200.0:50.0:600.0)
        demand_range = (20.0, 200.0)
        supply_range = (100.0, 1000.0)
        transport_cost = rand(0.8:0.1:2.0)
    else
        min_wh, max_wh = 10, 100
        min_supp, max_supp = 5, 50
        min_cust, max_cust = 20, 150
        n_size_options = 5
        grid_size = rand(500.0:100.0:1500.0)
        demand_range = (50.0, 500.0)
        supply_range = (200.0, 3000.0)
        transport_cost = rand(1.0:0.2:3.0)
    end

    # Solve for dimensions to hit target.
    # total_vars = n_wh × (1 + n_size_options + n_supp + n_cust)
    best_config = (min_wh, min_supp, min_cust)
    best_error = Inf

    for n_wh in min_wh:max_wh
        target_per_wh = target_variables / n_wh
        target_flows = target_per_wh - 1 - n_size_options

        if target_flows < (min_supp + min_cust)
            continue
        end

        # Split flow variables between suppliers and customers
        ratio = rand(0.3:0.1:0.7)
        n_supp = clamp(round(Int, target_flows * ratio), min_supp, max_supp)
        n_cust = clamp(round(Int, target_flows * (1 - ratio)), min_cust, max_cust)

        actual_vars = n_wh * (1 + n_size_options + n_supp + n_cust)
        err = abs(actual_vars - target_variables) / target_variables

        if err < best_error
            best_error = err
            best_config = (n_wh, n_supp, n_cust)
        end
    end

    n_warehouses, n_suppliers, n_customers = best_config

    # Generate warehouse locations (uniform over grid)
    warehouse_locations = [(grid_size * rand(), grid_size * rand()) for _ in 1:n_warehouses]

    # Suppliers clustered (e.g. manufacturing zones)
    n_supp_clusters = max(1, n_suppliers ÷ 5)
    supp_centers = [(grid_size * rand(), grid_size * rand()) for _ in 1:n_supp_clusters]
    supplier_locations = Tuple{Float64,Float64}[]
    for _ in 1:n_suppliers
        center = rand(supp_centers)
        x = clamp(center[1] + randn() * grid_size / 10, 0.0, grid_size)
        y = clamp(center[2] + randn() * grid_size / 10, 0.0, grid_size)
        push!(supplier_locations, (x, y))
    end

    # Customers clustered (e.g. cities)
    n_cust_clusters = max(2, n_customers ÷ 8)
    cust_centers = [(grid_size * rand(), grid_size * rand()) for _ in 1:n_cust_clusters]
    customer_locations = Tuple{Float64,Float64}[]
    for _ in 1:n_customers
        center = rand(cust_centers)
        x = clamp(center[1] + randn() * grid_size / 8, 0.0, grid_size)
        y = clamp(center[2] + randn() * grid_size / 8, 0.0, grid_size)
        push!(customer_locations, (x, y))
    end

    # Customer demands (log-normal)
    min_demand, max_demand = demand_range
    log_mean = log(sqrt(min_demand * max_demand))
    log_std = log(max_demand / min_demand) / 4
    customer_demands = [clamp(exp(rand(Normal(log_mean, log_std))), min_demand, max_demand) for _ in 1:n_customers]
    customer_demands = round.(customer_demands, digits=2)

    total_demand = sum(customer_demands)

    # Supplier capacities
    min_supply, max_supply = supply_range
    supplier_capacities = [rand(Uniform(min_supply, max_supply)) for _ in 1:n_suppliers]
    supplier_capacities = round.(supplier_capacities, digits=2)

    # Ensure ample total supply >= total demand by default (feasibility baseline)
    total_supply = sum(supplier_capacities)
    if total_supply < total_demand * 1.2
        scale = (total_demand * 1.3) / total_supply
        supplier_capacities .*= scale
        supplier_capacities = round.(supplier_capacities, digits=2)
    end

    # Warehouse discrete size options (economies of scale)
    base_capacity = total_demand / n_warehouses
    all_mults = [0.5, 1.0, 1.5, 2.0, 3.0]
    warehouse_size_options = [round(base_capacity * mult, digits=2) for mult in all_mults[1:n_size_options]]

    # Sublinear size costs (economies of scale)
    base_size_cost = rand(5000.0:1000.0:20000.0)
    warehouse_size_costs = [round(base_size_cost * (cap / warehouse_size_options[1])^0.85, digits=2)
                            for cap in warehouse_size_options]

    # Fixed warehouse opening costs (location-dependent)
    warehouse_fixed_costs = [rand(Uniform(10000.0, 50000.0)) * (1 + 0.3 * rand()) for _ in 1:n_warehouses]
    warehouse_fixed_costs = round.(warehouse_fixed_costs, digits=2)

    # Distance-based transport costs
    calc_distance(a, b) = sqrt((a[1] - b[1])^2 + (a[2] - b[2])^2)

    supplier_warehouse_costs = zeros(n_suppliers, n_warehouses)
    for i in 1:n_suppliers, j in 1:n_warehouses
        dist = calc_distance(supplier_locations[i], warehouse_locations[j])
        supplier_warehouse_costs[i, j] = round(dist * transport_cost * (0.9 + 0.2 * rand()), digits=2)
    end

    warehouse_customer_costs = zeros(n_warehouses, n_customers)
    for i in 1:n_warehouses, j in 1:n_customers
        dist = calc_distance(warehouse_locations[i], customer_locations[j])
        warehouse_customer_costs[i, j] = round(dist * transport_cost * (0.9 + 0.2 * rand()), digits=2)
    end

    # Handling costs at warehouses
    handling_costs = round.([rand(Uniform(0.5, 2.5)) for _ in 1:n_warehouses], digits=2)

    # --- Feasibility handling ---
    actual_status = feasibility_status
    if feasibility_status == unknown
        # Natural instance; bias toward feasible but do NOT force infeasibility.
        actual_status = feasible
    end

    if actual_status == feasible
        # Guarantee solvability:
        # 1. Aggregate max warehouse throughput must comfortably exceed total demand.
        max_possible_capacity = sum(warehouse_size_options[end] for _ in 1:n_warehouses)
        if max_possible_capacity < total_demand * 1.2
            scale = (total_demand * 1.3) / max_possible_capacity
            warehouse_size_options .*= scale
            warehouse_size_options = round.(warehouse_size_options, digits=2)
        end
        # 2. Aggregate supply must comfortably exceed total demand.
        total_supply = sum(supplier_capacities)
        if total_supply < total_demand * 1.2
            scale = (total_demand * 1.3) / total_supply
            supplier_capacities .*= scale
            supplier_capacities = round.(supplier_capacities, digits=2)
        end

    elseif actual_status == infeasible
        # Reliable infeasibility: make it provably impossible to serve all demand.
        # Aggregate maximum warehouse throughput (all opened at largest size) is
        # forced strictly below total demand with a clear margin, so the customer
        # demand constraints cannot all be satisfied regardless of supply or sizing.
        target_cap_fraction = rand(0.7:0.05:0.9)  # 70%-90% of demand
        target_total_cap = total_demand * target_cap_fraction
        max_size = target_total_cap / n_warehouses
        # Build size options strictly bounded by max_size so even all-largest-size
        # across all warehouses cannot meet demand.
        warehouse_size_options = [round(max_size * mult, digits=2) for mult in all_mults[1:n_size_options]]
        # Normalize so the LARGEST option equals max_size (largest mult may exceed 1).
        warehouse_size_options ./= all_mults[n_size_options]
        warehouse_size_options = round.(warehouse_size_options, digits=2)
        # Recompute size costs consistently.
        base_opt = max(warehouse_size_options[1], 1e-6)
        warehouse_size_costs = [round(base_size_cost * (cap / base_opt)^0.85, digits=2)
                                for cap in warehouse_size_options]
    end

    return TwoEchelonFacilityLocationProblem(
        n_warehouses,
        n_suppliers,
        n_customers,
        warehouse_locations,
        supplier_locations,
        customer_locations,
        supplier_capacities,
        customer_demands,
        warehouse_fixed_costs,
        warehouse_size_options,
        warehouse_size_costs,
        supplier_warehouse_costs,
        warehouse_customer_costs,
        handling_costs,
    )
end

"""
    build_model(prob::TwoEchelonFacilityLocationProblem)

Build a JuMP model for the two-echelon capacitated facility location and sizing
problem. Deterministic — uses only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::TwoEchelonFacilityLocationProblem)
    model = Model()

    W = prob.n_warehouses
    S = prob.n_suppliers
    C = prob.n_customers
    K = length(prob.warehouse_size_options)

    # Decision variables
    @variable(model, y[1:W], Bin)            # open warehouse w
    @variable(model, z[1:W, 1:K], Bin)       # install size k at warehouse w
    @variable(model, f1[1:S, 1:W] >= 0)      # supplier→warehouse flow
    @variable(model, f2[1:W, 1:C] >= 0)      # warehouse→customer flow

    # Objective: total network cost
    @objective(model, Min,
        sum(prob.warehouse_fixed_costs[w] * y[w] for w in 1:W) +
        sum(prob.warehouse_size_costs[k] * z[w, k] for w in 1:W, k in 1:K) +
        sum(prob.supplier_warehouse_costs[s, w] * f1[s, w] for s in 1:S, w in 1:W) +
        sum(prob.warehouse_customer_costs[w, c] * f2[w, c] for w in 1:W, c in 1:C) +
        sum(prob.handling_costs[w] * sum(f1[s, w] for s in 1:S) for w in 1:W)
    )

    # Exactly one size if opened, none otherwise: sum_k z[w,k] == y[w]
    for w in 1:W
        @constraint(model, sum(z[w, k] for k in 1:K) == y[w])
    end

    # Supplier supply limits
    for s in 1:S
        @constraint(model, sum(f1[s, w] for w in 1:W) <= prob.supplier_capacities[s])
    end

    # Customer demand satisfaction
    for c in 1:C
        @constraint(model, sum(f2[w, c] for w in 1:W) >= prob.customer_demands[c])
    end

    # Warehouse throughput bounded by chosen size
    for w in 1:W
        @constraint(model,
            sum(f1[s, w] for s in 1:S) <=
            sum(prob.warehouse_size_options[k] * z[w, k] for k in 1:K)
        )
    end

    # Flow conservation at warehouses: inflow >= outflow
    for w in 1:W
        @constraint(model, sum(f1[s, w] for s in 1:S) >= sum(f2[w, c] for c in 1:C))
    end

    # Inbound flow only at opened warehouses
    for w in 1:W, s in 1:S
        @constraint(model, f1[s, w] <= prob.supplier_capacities[s] * y[w])
    end

    return model
end

# Register the variant
register_variant(
    :facility_location,
    :two_echelon,
    TwoEchelonFacilityLocationProblem,
    "Two-echelon capacitated facility location with discrete warehouse sizing over a supplier→warehouse→customer network",
)
