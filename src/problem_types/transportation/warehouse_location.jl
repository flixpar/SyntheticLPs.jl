using JuMP
using Random
using Distributions

"""
    WarehouseLocation <: ProblemGenerator

Generator for warehouse location and sizing problems.

This problem combines facility location decisions with capacity sizing choices,
optimizing the network of suppliers → warehouses → customers.

# Fields
- `n_warehouses::Int`: Number of potential warehouse locations
- `n_suppliers::Int`: Number of suppliers
- `n_customers::Int`: Number of customers
- `warehouse_locations::Vector{Tuple{Float64,Float64}}`: Warehouse coordinates
- `supplier_locations::Vector{Tuple{Float64,Float64}}`: Supplier coordinates
- `customer_locations::Vector{Tuple{Float64,Float64}}`: Customer coordinates
- `supplier_capacities::Vector{Float64}`: Supply capacity at each supplier
- `customer_demands::Vector{Float64}`: Demand at each customer
- `warehouse_fixed_costs::Vector{Float64}`: Fixed cost to open each warehouse
- `warehouse_size_options::Vector{Float64}`: Available capacity sizes
- `warehouse_size_costs::Vector{Float64}`: Cost for each size option
- `supplier_warehouse_costs::Matrix{Float64}`: Transport cost supplier→warehouse
- `warehouse_customer_costs::Matrix{Float64}`: Transport cost warehouse→customer
- `handling_costs::Vector{Float64}`: Per-unit handling cost at each warehouse
"""
struct WarehouseLocation <: ProblemGenerator
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
    WarehouseLocation(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a warehouse location and sizing problem instance.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Details
Variables: warehouse opening (binary), size selection (binary), and flows
Target: n_warehouses × (1 + n_sizes + n_suppliers + n_customers)
"""
function WarehouseLocation(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
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

    # Solve for dimensions
    # target_vars ≈ n_wh × (1 + n_sizes + n_supp + n_cust)
    best_config = (min_wh, min_supp, min_cust)
    best_error = Inf

    for n_wh in min_wh:max_wh
        # Given n_wh, solve for n_supp and n_cust
        # target ≈ n_wh × (1 + n_sizes + n_supp + n_cust)
        target_per_wh = target_variables / n_wh
        target_flows = target_per_wh - 1 - n_size_options

        if target_flows < (min_supp + min_cust)
            continue
        end

        # Split between suppliers and customers
        ratio = rand(0.3:0.1:0.7)
        n_supp = clamp(round(Int, target_flows * ratio), min_supp, max_supp)
        n_cust = clamp(round(Int, target_flows * (1 - ratio)), min_cust, max_cust)

        actual_vars = n_wh * (1 + n_size_options + n_supp + n_cust)
        error = abs(actual_vars - target_variables) / target_variables

        if error < best_error
            best_error = error
            best_config = (n_wh, n_supp, n_cust)
        end
    end

    n_warehouses, n_suppliers, n_customers = best_config

    # Generate locations
    warehouse_locations = [(grid_size * rand(), grid_size * rand()) for _ in 1:n_warehouses]

    # Suppliers often clustered (e.g., manufacturing zones)
    n_supp_clusters = max(1, n_suppliers ÷ 5)
    supp_centers = [(grid_size * rand(), grid_size * rand()) for _ in 1:n_supp_clusters]
    supplier_locations = Tuple{Float64,Float64}[]
    for i in 1:n_suppliers
        center = rand(supp_centers)
        x = clamp(center[1] + randn() * grid_size / 10, 0.0, grid_size)
        y = clamp(center[2] + randn() * grid_size / 10, 0.0, grid_size)
        push!(supplier_locations, (x, y))
    end

    # Customers often clustered (e.g., cities)
    n_cust_clusters = max(2, n_customers ÷ 8)
    cust_centers = [(grid_size * rand(), grid_size * rand()) for _ in 1:n_cust_clusters]
    customer_locations = Tuple{Float64,Float64}[]
    for i in 1:n_customers
        center = rand(cust_centers)
        x = clamp(center[1] + randn() * grid_size / 8, 0.0, grid_size)
        y = clamp(center[2] + randn() * grid_size / 8, 0.0, grid_size)
        push!(customer_locations, (x, y))
    end

    # Generate demands
    min_demand, max_demand = demand_range
    log_mean = log(sqrt(min_demand * max_demand))
    log_std = log(max_demand / min_demand) / 4
    customer_demands = [clamp(exp(rand(Normal(log_mean, log_std))), min_demand, max_demand) for _ in 1:n_customers]
    customer_demands = round.(customer_demands, digits=2)

    total_demand = sum(customer_demands)

    # Generate supplier capacities
    min_supply, max_supply = supply_range
    supplier_capacities = [rand(Uniform(min_supply, max_supply)) for _ in 1:n_suppliers]
    supplier_capacities = round.(supplier_capacities, digits=2)

    # Ensure total supply >= total demand (adjust later for feasibility)
    total_supply = sum(supplier_capacities)
    if total_supply < total_demand
        scale = (total_demand * 1.1) / total_supply
        supplier_capacities .*= scale
        supplier_capacities = round.(supplier_capacities, digits=2)
    end

    # Warehouse size options (with economies of scale)
    base_capacity = total_demand / n_warehouses
    warehouse_size_options = [round(base_capacity * mult, digits=2) for mult in [0.5, 1.0, 1.5, 2.0, 3.0]]
    warehouse_size_options = warehouse_size_options[1:n_size_options]

    # Costs for each size (sublinear to reflect economies of scale)
    base_size_cost = rand(5000.0:1000.0:20000.0)
    warehouse_size_costs = [round(base_size_cost * (cap / warehouse_size_options[1])^0.85, digits=2)
                            for cap in warehouse_size_options]

    # Fixed costs for warehouse locations (vary by location)
    warehouse_fixed_costs = [rand(Uniform(10000.0, 50000.0)) * (1 + 0.3 * rand()) for _ in 1:n_warehouses]
    warehouse_fixed_costs = round.(warehouse_fixed_costs, digits=2)

    # Transport costs (distance-based)
    function calc_distance(loc1, loc2)
        return sqrt((loc1[1] - loc2[1])^2 + (loc1[2] - loc2[2])^2)
    end

    supplier_warehouse_costs = zeros(n_suppliers, n_warehouses)
    for i in 1:n_suppliers
        for j in 1:n_warehouses
            dist = calc_distance(supplier_locations[i], warehouse_locations[j])
            supplier_warehouse_costs[i, j] = round(dist * transport_cost * (0.9 + 0.2 * rand()), digits=2)
        end
    end

    warehouse_customer_costs = zeros(n_warehouses, n_customers)
    for i in 1:n_warehouses
        for j in 1:n_customers
            dist = calc_distance(warehouse_locations[i], customer_locations[j])
            warehouse_customer_costs[i, j] = round(dist * transport_cost * (0.9 + 0.2 * rand()), digits=2)
        end
    end

    # Handling costs at warehouses
    handling_costs = [rand(Uniform(0.5, 2.5)) for _ in 1:n_warehouses]
    handling_costs = round.(handling_costs, digits=2)

    # Adjust for feasibility
    if feasibility_status == feasible
        # Ensure enough warehouse capacity is available
        max_possible_capacity = sum(warehouse_size_options[end] for _ in 1:n_warehouses)
        if max_possible_capacity < total_demand
            scale = (total_demand * 1.2) / max_possible_capacity
            warehouse_size_options .*= scale
            warehouse_size_options = round.(warehouse_size_options, digits=2)
        end

    elseif feasibility_status == infeasible
        # Create infeasibility by limiting warehouse capacity
        if rand() < 0.5
            # Reduce warehouse size options
            scale = rand(0.3:0.05:0.7)
            warehouse_size_options .*= scale
            warehouse_size_options = round.(warehouse_size_options, digits=2)
        else
            # Reduce supplier capacity
            scale = rand(0.5:0.05:0.8)
            supplier_capacities .*= scale
            supplier_capacities = round.(supplier_capacities, digits=2)
        end
    end

    return WarehouseLocation(
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
        handling_costs
    )
end

"""
    build_model(prob::WarehouseLocation)

Build a JuMP model for the warehouse location and sizing problem.

# Arguments
- `prob`: WarehouseLocation instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::WarehouseLocation)
    model = Model()

    W = prob.n_warehouses
    S = prob.n_suppliers
    C = prob.n_customers
    K = length(prob.warehouse_size_options)

    # Decision variables
    # y[w] = 1 if warehouse w is opened
    @variable(model, y[1:W], Bin)

    # z[w,k] = 1 if warehouse w is opened with size option k
    @variable(model, z[1:W, 1:K], Bin)

    # f1[s,w] = flow from supplier s to warehouse w
    @variable(model, f1[1:S, 1:W] >= 0)

    # f2[w,c] = flow from warehouse w to customer c
    @variable(model, f2[1:W, 1:C] >= 0)

    # Objective: minimize total cost
    @objective(model, Min,
        # Fixed costs for opening warehouses
        sum(prob.warehouse_fixed_costs[w] * y[w] for w in 1:W) +
        # Size costs
        sum(prob.warehouse_size_costs[k] * z[w, k] for w in 1:W, k in 1:K) +
        # Transport costs supplier→warehouse
        sum(prob.supplier_warehouse_costs[s, w] * f1[s, w] for s in 1:S, w in 1:W) +
        # Transport costs warehouse→customer
        sum(prob.warehouse_customer_costs[w, c] * f2[w, c] for w in 1:W, c in 1:C) +
        # Handling costs
        sum(prob.handling_costs[w] * sum(f1[s, w] for s in 1:S) for w in 1:W)
    )

    # Constraints

    # Each warehouse can have at most one size if opened
    for w in 1:W
        @constraint(model, sum(z[w, k] for k in 1:K) == y[w])
    end

    # Supplier capacity constraints
    for s in 1:S
        @constraint(model, sum(f1[s, w] for w in 1:W) <= prob.supplier_capacities[s])
    end

    # Customer demand constraints
    for c in 1:C
        @constraint(model, sum(f2[w, c] for w in 1:W) >= prob.customer_demands[c])
    end

    # Warehouse capacity constraints (based on size selected)
    for w in 1:W
        @constraint(model,
            sum(f1[s, w] for s in 1:S) <=
            sum(prob.warehouse_size_options[k] * z[w, k] for k in 1:K)
        )
    end

    # Flow balance at warehouses
    for w in 1:W
        @constraint(model,
            sum(f1[s, w] for s in 1:S) >= sum(f2[w, c] for c in 1:C)
        )
    end

    # Warehouses can only operate if opened
    for w in 1:W
        for s in 1:S
            @constraint(model, f1[s, w] <= prob.supplier_capacities[s] * y[w])
        end
    end

    return model
end

# Register the problem type
register_problem(
    :warehouse_location,
    WarehouseLocation,
    "Warehouse location and sizing problem optimizing multi-echelon distribution network"
)
