using JuMP
using Random
using Distributions

"""
    VehicleRoutingProblem <: ProblemGenerator

Generator for capacitated vehicle routing problems (CVRP) with LP relaxation.

The problem involves routing a fleet of vehicles from a central depot to serve
customers with known demands, minimizing total travel cost while respecting
vehicle capacity constraints.

# Fields
- `n_customers::Int`: Number of customers to serve
- `n_vehicles::Int`: Number of vehicles in the fleet
- `depot_location::Tuple{Float64,Float64}`: Coordinates of the depot
- `customer_locations::Vector{Tuple{Float64,Float64}}`: Coordinates of each customer
- `demands::Vector{Float64}`: Demand at each customer location
- `vehicle_capacities::Vector{Float64}`: Capacity of each vehicle
- `distances::Dict{Tuple{Int,Int},Float64}`: Distance/cost between each pair of locations (0=depot, 1..n=customers)
"""
struct VehicleRoutingProblem <: ProblemGenerator
    n_customers::Int
    n_vehicles::Int
    depot_location::Tuple{Float64,Float64}
    customer_locations::Vector{Tuple{Float64,Float64}}
    demands::Vector{Float64}
    vehicle_capacities::Vector{Float64}
    distances::Dict{Tuple{Int,Int},Float64}
end

"""
    VehicleRoutingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a vehicle routing problem instance.

# Arguments
- `target_variables`: Target number of variables in the LP formulation
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Details
For VRP: target_variables = m × ((n+1)² + n) where n=n_customers, m=n_vehicles
This accounts for x[i,j,k] flow variables and y[j,k] service variables.
"""
function VehicleRoutingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem scale based on target variables
    params = sample_vrp_parameters(target_variables)

    n_customers = params[:n_customers]
    n_vehicles = params[:n_vehicles]
    grid_width = params[:grid_width]
    grid_height = params[:grid_height]
    demand_range = params[:demand_range]
    capacity_factor = params[:capacity_factor]
    n_clusters = params[:n_clusters]

    # Generate depot location (typically at center or edge)
    depot_location = if rand() < 0.6
        # Central depot (60% of cases)
        (grid_width * (0.4 + 0.2 * rand()), grid_height * (0.4 + 0.2 * rand()))
    else
        # Edge depot (40% of cases - e.g., warehouse at city edge)
        if rand() < 0.5
            (grid_width * rand(0.0:0.1:0.15), grid_height * rand())
        else
            (grid_width * rand(), grid_height * rand(0.0:0.1:0.15))
        end
    end

    # Generate customer locations with clustering (realistic urban patterns)
    customer_locations = generate_clustered_customers(n_customers, n_clusters, grid_width, grid_height)

    # Generate customer demands using log-normal distribution
    # (realistic: few large shipments, many small ones)
    min_demand, max_demand = demand_range
    demands = generate_lognormal_demands(n_customers, min_demand, max_demand)

    total_demand = sum(demands)

    # Generate heterogeneous vehicle fleet
    vehicle_capacities = generate_vehicle_fleet(n_vehicles, total_demand, capacity_factor)

    total_capacity = sum(vehicle_capacities)

    # Calculate distances between all location pairs
    # Locations: 0=depot, 1..n_customers=customers
    distances = calculate_distances(depot_location, customer_locations)

    # Adjust for feasibility status
    solution_status = feasibility_status == feasible ? :feasible :
                     feasibility_status == infeasible ? :infeasible : :unknown

    if solution_status == :feasible
        # Ensure total capacity is sufficient with margin
        if total_capacity < total_demand * 1.05
            # Scale up vehicle capacities
            scale_factor = (total_demand * rand(Uniform(1.15, 1.35))) / total_capacity
            vehicle_capacities = [cap * scale_factor for cap in vehicle_capacities]
        end

    elseif solution_status == :infeasible
        # Create infeasibility through capacity shortage
        target_capacity = total_demand * rand(Uniform(0.65, 0.92))

        if total_capacity > target_capacity
            # Scale down capacities to create infeasibility
            scale_factor = target_capacity / total_capacity
            vehicle_capacities = [cap * scale_factor for cap in vehicle_capacities]
        end

        # Alternatively, increase demands beyond capacity
        if rand() < 0.3
            demand_scale = rand(Uniform(1.1, 1.4))
            demands = [d * demand_scale for d in demands]
        end
    end
    # For :unknown, leave as-is

    return VehicleRoutingProblem(
        n_customers,
        n_vehicles,
        depot_location,
        customer_locations,
        demands,
        vehicle_capacities,
        distances
    )
end

"""
    build_model(prob::VehicleRoutingProblem)

Build a JuMP model for the vehicle routing problem (LP relaxation).

# Arguments
- `prob`: VehicleRoutingProblem instance

# Returns
- `model`: The JuMP model

# Formulation
Variables:
- x[i,j,k]: Flow from location i to j using vehicle k (continuous, 0 ≤ x ≤ 1)
- y[j,k]: Whether vehicle k serves customer j (continuous, 0 ≤ y ≤ 1)

Objective: Minimize total routing cost

Constraints:
- Each customer is served exactly once
- Flow conservation at each node for each vehicle
- Vehicle capacity limits
- Depot flow constraints (vehicles start and end at depot)
"""
function build_model(prob::VehicleRoutingProblem)
    model = Model()

    n = prob.n_customers
    m = prob.n_vehicles

    # All locations: 0=depot, 1..n=customers
    locations = 0:n
    customers = 1:n
    vehicles = 1:m

    # Decision variables: flow from location i to j using vehicle k
    # For LP relaxation, these are continuous [0,1]
    @variable(model, 0 <= x[i in locations, j in locations, k in vehicles] <= 1)

    # Auxiliary variables: whether vehicle k serves customer j
    @variable(model, 0 <= y[j in customers, k in vehicles] <= 1)

    # Objective: Minimize total distance traveled
    @objective(model, Min,
        sum(prob.distances[(i,j)] * x[i,j,k]
            for i in locations, j in locations, k in vehicles
            if i != j)
    )

    # Constraint 1: Each customer is served exactly once
    for j in customers
        @constraint(model, sum(y[j,k] for k in vehicles) == 1)
    end

    # Constraint 2: Flow conservation at customers
    # If a vehicle visits a customer, it must arrive and leave
    for j in customers, k in vehicles
        # Incoming flow
        inflow = sum(x[i,j,k] for i in locations if i != j)
        # Outgoing flow
        outflow = sum(x[j,i,k] for i in locations if i != j)

        # Flow conservation: what comes in must go out
        @constraint(model, inflow == outflow)

        # Link flow to service
        @constraint(model, inflow == y[j,k])
    end

    # Constraint 3: Vehicle capacity constraints
    for k in vehicles
        @constraint(model,
            sum(prob.demands[j] * y[j,k] for j in customers) <= prob.vehicle_capacities[k]
        )
    end

    # Constraint 4: Each vehicle starts and ends at depot
    for k in vehicles
        # Number of times vehicle leaves depot
        depot_outflow = sum(x[0,j,k] for j in customers)
        # Number of times vehicle returns to depot
        depot_inflow = sum(x[j,0,k] for j in customers)

        # Vehicle makes at most one tour (for LP relaxation, this is ≤ 1)
        @constraint(model, depot_outflow <= 1)
        @constraint(model, depot_inflow <= 1)

        # What leaves must return
        @constraint(model, depot_outflow == depot_inflow)
    end

    # Constraint 5: No self-loops
    for i in locations, k in vehicles
        @constraint(model, x[i,i,k] == 0)
    end

    return model
end

# Helper functions

"""
    sample_vrp_parameters(target_variables::Int)

Sample realistic parameters for a VRP instance targeting the specified number of variables.

For VRP: target_variables = m × ((n+1)² + n) where n=n_customers, m=n_vehicles
This accounts for x[i,j,k] variables ((n+1)² × m) and y[j,k] variables (n × m)
"""
function sample_vrp_parameters(target_variables::Int)
    params = Dict{Symbol,Any}()

    # Set ranges based on problem size
    if target_variables <= 100
        # Small problems: local delivery
        min_customers, max_customers = 3, 10
        min_vehicles, max_vehicles = 2, 4
        grid_size = rand(Uniform(50.0, 150.0))
        demand_range = (5.0, 50.0)
        capacity_factor = rand(Uniform(1.25, 1.6))
        n_clusters = rand(2:3)

    elseif target_variables <= 500
        # Medium problems: regional distribution
        min_customers, max_customers = 8, 25
        min_vehicles, max_vehicles = 3, 8
        grid_size = rand(Uniform(100.0, 300.0))
        demand_range = (10.0, 100.0)
        capacity_factor = rand(Uniform(1.2, 1.5))
        n_clusters = rand(3:5)

    elseif target_variables <= 2000
        # Large problems: city-wide delivery
        min_customers, max_customers = 20, 50
        min_vehicles, max_vehicles = 5, 15
        grid_size = rand(Uniform(200.0, 500.0))
        demand_range = (20.0, 200.0)
        capacity_factor = rand(Uniform(1.15, 1.4))
        n_clusters = rand(4:7)

    else
        # Very large problems: multi-city logistics
        min_customers, max_customers = 40, 100
        min_vehicles, max_vehicles = 8, 25
        grid_size = rand(Uniform(300.0, 800.0))
        demand_range = (30.0, 300.0)
        capacity_factor = rand(Uniform(1.1, 1.35))
        n_clusters = rand(5:10)
    end

    # Helper function to calculate actual variable count
    calc_vars(n_cust::Int, n_veh::Int) = n_veh * ((n_cust + 1)^2 + n_cust)

    # Find optimal combination of n_customers and n_vehicles
    best_n_customers = min_customers
    best_n_vehicles = min_vehicles
    best_error = Inf

    for n_cust in min_customers:max_customers
        # Calculate required n_vehicles for this n_customers
        # target_variables = m × ((n+1)² + n)
        # m = target_variables / ((n+1)² + n)
        vars_per_vehicle = (n_cust + 1)^2 + n_cust
        target_vehicles = target_variables / vars_per_vehicle

        if target_vehicles >= min_vehicles && target_vehicles <= max_vehicles
            n_veh = round(Int, target_vehicles)
            n_veh = clamp(n_veh, min_vehicles, max_vehicles)

            actual_vars = calc_vars(n_cust, n_veh)
            error = abs(actual_vars - target_variables) / target_variables

            if error < best_error
                best_error = error
                best_n_customers = n_cust
                best_n_vehicles = n_veh
            end
        end
    end

    # If error still too high, use heuristic
    if best_error > 0.1
        # Choose vehicle count from middle of range
        n_veh = div(min_vehicles + max_vehicles, 2)

        # Solve for n: target ≈ m × ((n+1)² + n) = m × (n² + 3n + 1)
        # Using quadratic formula: n² + 3n + (1 - target/m) = 0
        # n = (-3 + sqrt(9 - 4(1 - target/m))) / 2
        discriminant = 9 - 4 * (1 - target_variables / n_veh)
        if discriminant > 0
            n_cust = round(Int, (-3 + sqrt(discriminant)) / 2)
        else
            n_cust = round(Int, sqrt(target_variables / n_veh) - 1)
        end
        n_cust = clamp(n_cust, min_customers, max_customers)

        # Refine n_vehicles with the calculated n_cust
        vars_per_vehicle = (n_cust + 1)^2 + n_cust
        n_veh = round(Int, target_variables / vars_per_vehicle)
        n_veh = clamp(n_veh, min_vehicles, max_vehicles)

        best_n_customers = n_cust
        best_n_vehicles = n_veh
    end

    params[:n_customers] = best_n_customers
    params[:n_vehicles] = best_n_vehicles
    params[:grid_width] = grid_size * rand(Uniform(0.9, 1.1))
    params[:grid_height] = grid_size * rand(Uniform(0.9, 1.1))
    params[:demand_range] = demand_range
    params[:capacity_factor] = capacity_factor
    params[:n_clusters] = n_clusters

    return params
end

"""
    generate_clustered_customers(n_customers::Int, n_clusters::Int, grid_width::Float64, grid_height::Float64)

Generate customer locations with geographic clustering (realistic urban delivery patterns).
"""
function generate_clustered_customers(n_customers::Int, n_clusters::Int, grid_width::Float64, grid_height::Float64)
    # Generate cluster centers
    cluster_centers = [(grid_width * rand(), grid_height * rand()) for _ in 1:n_clusters]

    # Assign customers to clusters
    customer_locations = Tuple{Float64,Float64}[]

    for i in 1:n_customers
        # Pick a random cluster (with slight preference for first clusters)
        cluster_idx = min(n_clusters, 1 + floor(Int, rand() * n_clusters * 1.2))
        center = cluster_centers[cluster_idx]

        # Add customer near this cluster with normal distribution
        cluster_spread = min(grid_width, grid_height) / (2 * n_clusters)
        x = clamp(center[1] + randn() * cluster_spread, 0.0, grid_width)
        y = clamp(center[2] + randn() * cluster_spread, 0.0, grid_height)

        push!(customer_locations, (x, y))
    end

    return customer_locations
end

"""
    generate_lognormal_demands(n_customers::Int, min_demand::Float64, max_demand::Float64)

Generate customer demands using log-normal distribution for realism.
"""
function generate_lognormal_demands(n_customers::Int, min_demand::Float64, max_demand::Float64)
    demands = Float64[]

    # Parameters for log-normal distribution
    log_mean = log(sqrt(min_demand * max_demand))
    log_std = log(max_demand / min_demand) / 4

    for i in 1:n_customers
        demand = exp(rand(Normal(log_mean, log_std)))
        demand = clamp(demand, min_demand, max_demand)
        push!(demands, round(demand, digits=2))
    end

    return demands
end

"""
    generate_vehicle_fleet(n_vehicles::Int, total_demand::Float64, capacity_factor::Float64)

Generate heterogeneous vehicle fleet with varying capacities.
"""
function generate_vehicle_fleet(n_vehicles::Int, total_demand::Float64, capacity_factor::Float64)
    # Target total capacity
    target_total_capacity = total_demand * capacity_factor

    # Average capacity per vehicle
    avg_capacity = target_total_capacity / n_vehicles

    capacities = Float64[]

    if n_vehicles == 1
        push!(capacities, round(target_total_capacity, digits=2))
    else
        # Generate heterogeneous fleet
        # Mix of vehicle sizes: some large (1.5x avg), some medium (1x avg), some small (0.6x avg)
        for i in 1:n_vehicles
            if i <= n_vehicles ÷ 3
                # Large vehicles
                capacity = avg_capacity * rand(Uniform(1.2, 1.6))
            elseif i <= 2 * n_vehicles ÷ 3
                # Medium vehicles
                capacity = avg_capacity * rand(Uniform(0.8, 1.2))
            else
                # Small vehicles
                capacity = avg_capacity * rand(Uniform(0.5, 0.9))
            end
            push!(capacities, round(capacity, digits=2))
        end

        # Adjust to match target total capacity
        current_total = sum(capacities)
        scale = target_total_capacity / current_total
        capacities = [round(cap * scale, digits=2) for cap in capacities]
    end

    return capacities
end

"""
    calculate_distances(depot_location::Tuple{Float64,Float64}, customer_locations::Vector{Tuple{Float64,Float64}})

Calculate Euclidean distances between all location pairs.
Location 0 is depot, locations 1..n are customers.
"""
function calculate_distances(depot_location::Tuple{Float64,Float64}, customer_locations::Vector{Tuple{Float64,Float64}})
    n_customers = length(customer_locations)
    distances = Dict{Tuple{Int,Int},Float64}()

    # All locations including depot
    all_locations = [depot_location; customer_locations]
    n_locations = length(all_locations)

    for i in 0:(n_locations-1)
        for j in 0:(n_locations-1)
            if i == j
                distances[(i,j)] = 0.0
            else
                loc_i = all_locations[i+1]
                loc_j = all_locations[j+1]

                # Euclidean distance with small random variation for realism
                base_distance = sqrt((loc_i[1] - loc_j[1])^2 + (loc_i[2] - loc_j[2])^2)
                variation = 1.0 + rand(Uniform(-0.05, 0.05))
                distances[(i,j)] = round(base_distance * variation, digits=2)
            end
        end
    end

    return distances
end

# Register the problem type
register_problem(
    :vehicle_routing,
    VehicleRoutingProblem,
    "Capacitated vehicle routing problem (CVRP) with LP relaxation that routes vehicles from a depot to serve customers while minimizing travel cost"
)
