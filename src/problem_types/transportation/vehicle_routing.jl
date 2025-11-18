using JuMP
using Random
using Distributions

"""
    VehicleRouting <: ProblemGenerator

Generator for vehicle routing problems (VRP) that optimize delivery routes to serve customers.

This implements a capacitated vehicle routing problem where vehicles with limited capacity
must serve customers with known demands, minimizing total travel distance/cost.

# Fields
- `n_customers::Int`: Number of customers to serve
- `n_vehicles::Int`: Number of available vehicles
- `depot_location::Tuple{Float64,Float64}`: Depot coordinates (x, y)
- `customer_locations::Vector{Tuple{Float64,Float64}}`: Customer coordinates
- `demands::Vector{Float64}`: Demand at each customer
- `vehicle_capacity::Float64`: Capacity of each vehicle
- `distances::Matrix{Float64}`: Distance matrix (0=depot, 1:n_customers)
- `travel_costs::Matrix{Float64}`: Cost matrix for travel
"""
struct VehicleRouting <: ProblemGenerator
    n_customers::Int
    n_vehicles::Int
    depot_location::Tuple{Float64,Float64}
    customer_locations::Vector{Tuple{Float64,Float64}}
    demands::Vector{Float64}
    vehicle_capacity::Float64
    distances::Matrix{Float64}
    travel_costs::Matrix{Float64}
end

"""
    VehicleRouting(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a vehicle routing problem instance.

# Arguments
- `target_variables`: Target number of variables (approximately n_customers × n_vehicles + edges)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Details
Variables include customer-vehicle assignments and flow variables for routing.
Target: n_customers × n_vehicles + n_customers × n_customers (approximately)
"""
function VehicleRouting(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Estimate dimensions
    # Variables: n_customers × n_vehicles (assignment) + flow variables
    # Approximate: target_vars ≈ n_customers × (n_vehicles + avg_degree)

    if target_variables <= 100
        min_customers, max_customers = 3, 20
        vehicles_ratio = rand(0.2:0.05:0.5)  # 20-50% of customers
        grid_size = rand(50.0:10.0:200.0)
        demand_range = (5.0, 50.0)
        cost_per_km = rand(0.5:0.1:1.5)
        capacity_utilization = rand(0.65:0.05:0.85)
    elseif target_variables <= 500
        min_customers, max_customers = 10, 50
        vehicles_ratio = rand(0.15:0.05:0.4)
        grid_size = rand(100.0:20.0:500.0)
        demand_range = (10.0, 100.0)
        cost_per_km = rand(0.8:0.1:2.0)
        capacity_utilization = rand(0.7:0.05:0.85)
    else
        min_customers, max_customers = 20, 150
        vehicles_ratio = rand(0.1:0.05:0.3)
        grid_size = rand(200.0:50.0:1000.0)
        demand_range = (20.0, 200.0)
        cost_per_km = rand(1.0:0.2:2.5)
        capacity_utilization = rand(0.75:0.05:0.9)
    end

    # Solve for n_customers considering both assignment and routing variables
    # target_vars ≈ n_customers × n_vehicles + 0.3 × n_customers²
    best_n_customers = min_customers
    best_error = Inf

    for n_cust in min_customers:max_customers
        n_veh = max(1, round(Int, n_cust * vehicles_ratio))
        # Approximate number of routing edges (sparse graph)
        approx_vars = n_cust * n_veh + round(Int, 0.3 * n_cust * n_cust)
        error = abs(approx_vars - target_variables) / target_variables

        if error < best_error
            best_error = error
            best_n_customers = n_cust
        end
    end

    n_customers = best_n_customers
    n_vehicles = max(1, round(Int, n_customers * vehicles_ratio))

    # Generate locations with realistic clustering
    depot_location = (grid_size * rand(), grid_size * rand())

    # Create customer clusters for realism
    n_clusters = max(2, min(5, n_customers ÷ 4))
    cluster_centers = [(grid_size * rand(), grid_size * rand()) for _ in 1:n_clusters]

    customer_locations = Tuple{Float64,Float64}[]
    for i in 1:n_customers
        # Assign to random cluster
        center = rand(cluster_centers)
        # Add noise around cluster center
        cluster_radius = grid_size / (2 * sqrt(n_clusters))
        x = clamp(center[1] + randn() * cluster_radius, 0.0, grid_size)
        y = clamp(center[2] + randn() * cluster_radius, 0.0, grid_size)
        push!(customer_locations, (x, y))
    end

    # Generate demands using log-normal distribution for realism
    min_demand, max_demand = demand_range
    log_mean = log(sqrt(min_demand * max_demand))
    log_std = log(max_demand / min_demand) / 4

    demands = Float64[]
    for i in 1:n_customers
        demand = exp(rand(Normal(log_mean, log_std)))
        demand = clamp(demand, min_demand, max_demand)
        push!(demands, round(demand, digits=2))
    end

    # Calculate distances (Euclidean)
    # Matrix indices: 0=depot (use index 1), customers are 2:n_customers+1
    n_locations = n_customers + 1
    distances = zeros(Float64, n_locations, n_locations)

    locations = vcat([depot_location], customer_locations)
    for i in 1:n_locations
        for j in 1:n_locations
            if i != j
                dx = locations[i][1] - locations[j][1]
                dy = locations[i][2] - locations[j][2]
                distances[i, j] = sqrt(dx^2 + dy^2)
            end
        end
    end

    # Travel costs with some variation
    travel_costs = zeros(Float64, n_locations, n_locations)
    for i in 1:n_locations
        for j in 1:n_locations
            if i != j
                # Cost is distance-based with small random variation for road conditions
                variation = 0.9 + 0.2 * rand()
                travel_costs[i, j] = round(distances[i, j] * cost_per_km * variation, digits=2)
            end
        end
    end

    # Determine vehicle capacity
    total_demand = sum(demands)

    if feasibility_status == feasible
        # Ensure vehicles can serve all customers
        # Each vehicle should handle approximately total_demand/n_vehicles
        avg_capacity = (total_demand / n_vehicles) / capacity_utilization
        # Add buffer to ensure feasibility
        vehicle_capacity = round(avg_capacity * 1.1, digits=2)

        # Verify total capacity is sufficient
        total_capacity = vehicle_capacity * n_vehicles
        if total_capacity < total_demand
            vehicle_capacity = round((total_demand / n_vehicles) * 1.15, digits=2)
        end

    elseif feasibility_status == infeasible
        # Make capacity insufficient to serve all customers
        # Option 1: Reduce vehicle capacity
        if rand() < 0.5
            avg_capacity = (total_demand / n_vehicles) / capacity_utilization
            shortage_factor = rand(0.6:0.05:0.85)  # 15-40% shortage
            vehicle_capacity = round(avg_capacity * shortage_factor, digits=2)
        else
            # Option 2: Set capacity such that even optimal packing can't work
            max_demand = maximum(demands)
            # Make capacity less than what's needed
            required_capacity = total_demand / n_vehicles
            vehicle_capacity = round(required_capacity * rand(0.5:0.05:0.8), digits=2)
        end

    else  # unknown
        # Natural capacity based on utilization
        avg_capacity = (total_demand / n_vehicles) / capacity_utilization
        variation = 0.8 + 0.4 * rand()
        vehicle_capacity = round(avg_capacity * variation, digits=2)
    end

    return VehicleRouting(
        n_customers,
        n_vehicles,
        depot_location,
        customer_locations,
        demands,
        vehicle_capacity,
        distances,
        travel_costs
    )
end

"""
    build_model(prob::VehicleRouting)

Build a JuMP model for the vehicle routing problem.

Uses a flow-based formulation suitable for LP relaxation.

# Arguments
- `prob`: VehicleRouting instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::VehicleRouting)
    model = Model()

    n = prob.n_customers
    m = prob.n_vehicles
    n_loc = n + 1  # customers + depot

    # Decision variables
    # y[i,k] = fraction of customer i's demand served by vehicle k
    @variable(model, 0 <= y[1:n, 1:m] <= 1)

    # x[i,j,k] = fraction of arc (i,j) used by vehicle k in solution
    # i,j ∈ {0=depot (index 1), 1:n customers (indices 2:n+1)}
    @variable(model, 0 <= x[1:n_loc, 1:n_loc, 1:m] <= 1)

    # Objective: minimize total travel cost
    @objective(model, Min,
        sum(prob.travel_costs[i, j] * x[i, j, k]
            for i in 1:n_loc, j in 1:n_loc, k in 1:m if i != j)
    )

    # Constraint: Each customer must be fully served
    for i in 1:n
        @constraint(model, sum(y[i, k] for k in 1:m) == 1)
    end

    # Constraint: Vehicle capacity
    for k in 1:m
        @constraint(model,
            sum(prob.demands[i] * y[i, k] for i in 1:n) <= prob.vehicle_capacity
        )
    end

    # Constraint: Flow conservation for each vehicle
    for k in 1:m
        # At depot (index 1): vehicle must leave and return
        @constraint(model,
            sum(x[1, j, k] for j in 2:n_loc) <= 1  # Leave depot
        )
        @constraint(model,
            sum(x[i, 1, k] for i in 2:n_loc) <= 1  # Return to depot
        )
        @constraint(model,
            sum(x[1, j, k] for j in 2:n_loc) == sum(x[i, 1, k] for i in 2:n_loc)  # Balance
        )

        # At each customer: flow in = flow out (if visited)
        for i in 1:n
            loc_idx = i + 1  # Customer i is at location index i+1
            @constraint(model,
                sum(x[j, loc_idx, k] for j in 1:n_loc if j != loc_idx) ==
                sum(x[loc_idx, j, k] for j in 1:n_loc if j != loc_idx)
            )
        end
    end

    # Constraint: Link assignment to routing
    # If customer i is served by vehicle k, vehicle k must visit customer i
    for i in 1:n
        for k in 1:m
            loc_idx = i + 1
            @constraint(model,
                y[i, k] <= sum(x[j, loc_idx, k] for j in 1:n_loc if j != loc_idx)
            )
        end
    end

    # Constraint: No self-loops
    for i in 1:n_loc
        for k in 1:m
            @constraint(model, x[i, i, k] == 0)
        end
    end

    return model
end

# Register the problem type
register_problem(
    :vehicle_routing,
    VehicleRouting,
    "Vehicle routing problem optimizing delivery routes with capacity constraints"
)
