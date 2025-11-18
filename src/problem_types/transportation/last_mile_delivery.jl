using JuMP
using Random
using Distributions

"""
    LastMileDelivery <: ProblemGenerator

Generator for last-mile delivery problems in urban environments.

This problem optimizes urban package delivery with time windows, congestion,
and zone-based constraints typical of last-mile logistics.

# Fields
- `n_customers::Int`: Number of delivery customers
- `n_vehicles::Int`: Number of delivery vehicles
- `n_zones::Int`: Number of urban zones
- `depot_location::Tuple{Float64,Float64}`: Depot coordinates
- `customer_locations::Vector{Tuple{Float64,Float64}}`: Customer coordinates
- `customer_zones::Vector{Int}`: Zone assignment for each customer
- `package_sizes::Vector{Float64}`: Package size/weight for each customer
- `time_windows::Vector{Tuple{Float64,Float64}}`: Delivery time window (start, end)
- `service_times::Vector{Float64}`: Service time at each customer
- `vehicle_capacity::Float64`: Vehicle capacity (packages)
- `max_route_time::Float64`: Maximum route duration per vehicle
- `travel_times::Matrix{Float64}`: Travel time matrix (includes congestion)
- `travel_costs::Matrix{Float64}`: Travel cost matrix
- `zone_penalties::Vector{Float64}`: Penalty for inter-zone travel
- `congestion_factors::Vector{Float64}`: Congestion multiplier by zone
"""
struct LastMileDelivery <: ProblemGenerator
    n_customers::Int
    n_vehicles::Int
    n_zones::Int
    depot_location::Tuple{Float64,Float64}
    customer_locations::Vector{Tuple{Float64,Float64}}
    customer_zones::Vector{Int}
    package_sizes::Vector{Float64}
    time_windows::Vector{Tuple{Float64,Float64}}
    service_times::Vector{Float64}
    vehicle_capacity::Float64
    max_route_time::Float64
    travel_times::Matrix{Float64}
    travel_costs::Matrix{Float64}
    zone_penalties::Vector{Float64}
    congestion_factors::Vector{Float64}
end

"""
    LastMileDelivery(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a last-mile delivery problem instance.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Details
Variables: customer-vehicle assignments + routing + time variables
Target: n_customers × (n_vehicles + time_slots) + routing edges
"""
function LastMileDelivery(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine scale
    if target_variables <= 150
        min_customers, max_customers = 5, 25
        n_zones = rand(2:4)
        vehicles_ratio = rand(0.2:0.05:0.4)
        grid_size = rand(10.0:2.0:30.0)  # Smaller grid for urban area (km)
        package_range = (1.0, 20.0)
        time_window_width = rand(60.0:15.0:180.0)  # minutes
        max_route_time = rand(240.0:30.0:480.0)  # 4-8 hours
        avg_speed = rand(20.0:5.0:40.0)  # km/h in urban area
    elseif target_variables <= 800
        min_customers, max_customers = 15, 60
        n_zones = rand(3:6)
        vehicles_ratio = rand(0.15:0.05:0.35)
        grid_size = rand(20.0:5.0:50.0)
        package_range = (1.0, 30.0)
        time_window_width = rand(90.0:15.0:240.0)
        max_route_time = rand(300.0:30.0:540.0)
        avg_speed = rand(25.0:5.0:45.0)
    else
        min_customers, max_customers = 30, 120
        n_zones = rand(4:8)
        vehicles_ratio = rand(0.1:0.05:0.3)
        grid_size = rand(30.0:10.0:80.0)
        package_range = (1.0, 50.0)
        time_window_width = rand(120.0:20.0:300.0)
        max_route_time = rand(360.0:30.0:600.0)
        avg_speed = rand(30.0:5.0:50.0)
    end

    # Solve for dimensions
    # Variables: n_customers × n_vehicles + routing variables
    best_n_customers = min_customers
    best_error = Inf

    for n_cust in min_customers:max_customers
        n_veh = max(1, round(Int, n_cust * vehicles_ratio))
        # Assignment variables + sparse routing
        approx_vars = n_cust * n_veh + round(Int, 0.2 * n_cust * n_cust)
        error = abs(approx_vars - target_variables) / target_variables

        if error < best_error
            best_error = error
            best_n_customers = n_cust
        end
    end

    n_customers = best_n_customers
    n_vehicles = max(1, round(Int, n_customers * vehicles_ratio))

    # Generate urban zones
    zone_centers = [(grid_size * rand(), grid_size * rand()) for _ in 1:n_zones]

    # Depot typically central or edge of service area
    depot_location = if rand() < 0.5
        # Central depot
        (grid_size / 2 + randn() * grid_size / 10, grid_size / 2 + randn() * grid_size / 10)
    else
        # Edge depot
        (grid_size * rand() * 0.3, grid_size * rand() * 0.3)
    end
    depot_location = (clamp(depot_location[1], 0.0, grid_size), clamp(depot_location[2], 0.0, grid_size))

    # Generate customer locations clustered in zones
    customer_locations = Tuple{Float64,Float64}[]
    customer_zones = Int[]

    for i in 1:n_customers
        zone = rand(1:n_zones)
        push!(customer_zones, zone)

        # Cluster around zone center
        center = zone_centers[zone]
        zone_radius = grid_size / (2.5 * sqrt(n_zones))
        x = clamp(center[1] + randn() * zone_radius, 0.0, grid_size)
        y = clamp(center[2] + randn() * zone_radius, 0.0, grid_size)
        push!(customer_locations, (x, y))
    end

    # Package sizes (small for last-mile)
    min_pkg, max_pkg = package_range
    log_mean = log(sqrt(min_pkg * max_pkg))
    log_std = log(max_pkg / min_pkg) / 3
    package_sizes = [clamp(exp(rand(Normal(log_mean, log_std))), min_pkg, max_pkg) for _ in 1:n_customers]
    package_sizes = round.(package_sizes, digits=2)

    # Time windows (narrow for last-mile)
    # Operating hours: 8 AM to 8 PM (720 minutes)
    operating_start = 0.0
    operating_end = 720.0

    time_windows = Tuple{Float64,Float64}[]
    for i in 1:n_customers
        # Random start within operating hours
        window_start = operating_start + rand() * (operating_end - operating_start - time_window_width)
        window_end = window_start + time_window_width + rand() * time_window_width * 0.5
        window_end = min(window_end, operating_end)
        push!(time_windows, (round(window_start, digits=1), round(window_end, digits=1)))
    end

    # Service times (time to deliver package)
    service_times = [round(rand(Uniform(3.0, 15.0)), digits=1) for _ in 1:n_customers]

    # Calculate distances and travel times
    function calc_distance(loc1, loc2)
        return sqrt((loc1[1] - loc2[1])^2 + (loc1[2] - loc2[2])^2)
    end

    n_locations = n_customers + 1
    locations = vcat([depot_location], customer_locations)

    # Congestion factors by zone (higher in busy zones)
    congestion_factors = [rand(Uniform(1.0, 2.5)) for _ in 1:n_zones]
    congestion_factors = round.(congestion_factors, digits=2)

    # Travel times with congestion
    travel_times = zeros(n_locations, n_locations)
    for i in 1:n_locations
        for j in 1:n_locations
            if i != j
                dist = calc_distance(locations[i], locations[j])
                base_time = (dist / avg_speed) * 60.0  # Convert to minutes

                # Apply congestion based on destination zone
                if j > 1  # Not depot
                    zone = customer_zones[j - 1]
                    congestion = congestion_factors[zone]
                else
                    congestion = 1.0
                end

                travel_times[i, j] = round(base_time * congestion, digits=1)
            end
        end
    end

    # Travel costs (fuel + driver cost per minute)
    cost_per_minute = rand(0.5:0.1:1.5)
    travel_costs = travel_times * cost_per_minute
    travel_costs = round.(travel_costs, digits=2)

    # Zone penalties (cost for crossing zones)
    zone_penalties = [round(rand(Uniform(5.0, 30.0)), digits=2) for _ in 1:n_zones]

    # Vehicle capacity
    total_packages = sum(package_sizes)
    capacity_utilization = rand(0.65:0.05:0.85)

    if feasibility_status == feasible
        # Ensure vehicles can handle all packages
        avg_capacity = (total_packages / n_vehicles) / capacity_utilization
        vehicle_capacity = round(avg_capacity * 1.15, digits=2)

        # Verify feasibility
        if vehicle_capacity * n_vehicles < total_packages
            vehicle_capacity = round((total_packages / n_vehicles) * 1.2, digits=2)
        end

        # Relax time windows slightly to ensure feasibility
        for i in 1:n_customers
            start_time, end_time = time_windows[i]
            expanded_end = min(operating_end, end_time + time_window_width * 0.3)
            time_windows[i] = (start_time, round(expanded_end, digits=1))
        end

    elseif feasibility_status == infeasible
        # Create infeasibility
        choice = rand()
        if choice < 0.4
            # Insufficient capacity
            avg_capacity = (total_packages / n_vehicles) / capacity_utilization
            vehicle_capacity = round(avg_capacity * rand(0.5:0.05:0.8), digits=2)
        elseif choice < 0.7
            # Impossible time windows
            for i in 1:n_customers
                # Make some windows very tight or conflicting with route time
                if rand() < 0.3
                    tight_width = time_window_width * rand(0.2:0.05:0.5)
                    start_time = rand() * (operating_end - tight_width)
                    time_windows[i] = (round(start_time, digits=1), round(start_time + tight_width, digits=1))
                end
            end
            avg_capacity = (total_packages / n_vehicles) / capacity_utilization
            vehicle_capacity = round(avg_capacity, digits=2)
        else
            # Insufficient route time
            max_route_time = max_route_time * rand(0.4:0.05:0.7)
            avg_capacity = (total_packages / n_vehicles) / capacity_utilization
            vehicle_capacity = round(avg_capacity, digits=2)
        end

    else  # unknown
        avg_capacity = (total_packages / n_vehicles) / capacity_utilization
        vehicle_capacity = round(avg_capacity * (0.8 + 0.4 * rand()), digits=2)
    end

    return LastMileDelivery(
        n_customers,
        n_vehicles,
        n_zones,
        depot_location,
        customer_locations,
        customer_zones,
        package_sizes,
        time_windows,
        service_times,
        vehicle_capacity,
        max_route_time,
        travel_times,
        travel_costs,
        zone_penalties,
        congestion_factors
    )
end

"""
    build_model(prob::LastMileDelivery)

Build a JuMP model for the last-mile delivery problem.

# Arguments
- `prob`: LastMileDelivery instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::LastMileDelivery)
    model = Model()

    n = prob.n_customers
    m = prob.n_vehicles
    n_loc = n + 1  # customers + depot

    # Decision variables

    # y[i,k] = 1 if customer i is served by vehicle k
    @variable(model, 0 <= y[1:n, 1:m] <= 1)

    # x[i,j,k] = 1 if vehicle k travels from location i to j
    @variable(model, 0 <= x[1:n_loc, 1:n_loc, 1:m] <= 1)

    # t[i,k] = arrival time at customer i by vehicle k
    @variable(model, t[1:n, 1:m] >= 0)

    # Objective: minimize total cost (travel + zone crossing penalties)
    @objective(model, Min,
        sum(prob.travel_costs[i, j] * x[i, j, k]
            for i in 1:n_loc, j in 1:n_loc, k in 1:m if i != j)
    )

    # Constraints

    # Each customer must be served exactly once
    for i in 1:n
        @constraint(model, sum(y[i, k] for k in 1:m) == 1)
    end

    # Vehicle capacity
    for k in 1:m
        @constraint(model,
            sum(prob.package_sizes[i] * y[i, k] for i in 1:n) <= prob.vehicle_capacity
        )
    end

    # Flow conservation for each vehicle
    for k in 1:m
        # Depot: leave and return
        @constraint(model,
            sum(x[1, j, k] for j in 2:n_loc) <= 1
        )
        @constraint(model,
            sum(x[i, 1, k] for i in 2:n_loc) <= 1
        )
        @constraint(model,
            sum(x[1, j, k] for j in 2:n_loc) == sum(x[i, 1, k] for i in 2:n_loc)
        )

        # Customer nodes: flow balance
        for i in 1:n
            loc_idx = i + 1
            @constraint(model,
                sum(x[j, loc_idx, k] for j in 1:n_loc if j != loc_idx) ==
                sum(x[loc_idx, j, k] for j in 1:n_loc if j != loc_idx)
            )
        end
    end

    # Link assignment to routing
    for i in 1:n
        for k in 1:m
            loc_idx = i + 1
            @constraint(model,
                y[i, k] <= sum(x[j, loc_idx, k] for j in 1:n_loc if j != loc_idx)
            )
        end
    end

    # Time window constraints (relaxed for LP)
    for i in 1:n
        for k in 1:m
            start_time, end_time = prob.time_windows[i]
            @constraint(model, t[i, k] >= start_time * y[i, k])
            @constraint(model, t[i, k] <= end_time * y[i, k] + (1 - y[i, k]) * 10000)
        end
    end

    # Time consistency (simplified for LP)
    for k in 1:m
        for i in 1:n
            for j in 1:n
                if i != j
                    loc_i = i + 1
                    loc_j = j + 1
                    # If vehicle k goes from i to j, time must be consistent
                    @constraint(model,
                        t[j, k] >= t[i, k] + prob.service_times[i] +
                        prob.travel_times[loc_i, loc_j] - (1 - x[loc_i, loc_j, k]) * 10000
                    )
                end
            end
        end
    end

    # Maximum route time
    for k in 1:m
        # Total route time ≤ max_route_time
        @constraint(model,
            sum(prob.travel_times[i, j] * x[i, j, k] for i in 1:n_loc, j in 1:n_loc if i != j) +
            sum(prob.service_times[i] * y[i, k] for i in 1:n) <=
            prob.max_route_time
        )
    end

    # No self-loops
    for i in 1:n_loc
        for k in 1:m
            @constraint(model, x[i, i, k] == 0)
        end
    end

    return model
end

# Register the problem type
register_problem(
    :last_mile_delivery,
    LastMileDelivery,
    "Last-mile delivery problem with urban constraints, time windows, and congestion"
)
