using JuMP
using Random
using StatsBase
using Dates
using LinearAlgebra
using Distributions

"""
    generate_airline_crew_problem(params::Dict=Dict(); seed::Int=0)

Generate an airline crew pairing problem instance.

# Arguments
- `params`: Dictionary of problem parameters
  - `:num_airports`: Number of airports in the network (default: 15)
  - `:num_bases`: Number of crew bases (default: 4)
  - `:num_flights`: Number of flights in the planning horizon (default: 120)
  - `:planning_horizon_days`: Length of planning horizon in days (default: 5)
  - `:max_pairings`: Maximum number of pairings to generate (default: 200)
  - `:flight_cost_mean`: Mean flight operating cost (default: 2500)
  - `:flight_cost_std`: Standard deviation of flight costs (default: 800)
  - `:pairing_overhead_mean`: Mean overhead factor for pairings (default: 0.25)
  - `:pairing_overhead_std`: Standard deviation of pairing overhead (default: 0.15)
  - `:max_flights_per_pairing`: Maximum flights per pairing (default: 6)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- `model`: The JuMP model
- `params`: Dictionary of all parameters used (including defaults)
"""
function generate_airline_crew_problem(params::Dict=Dict(); seed::Int=0)
    # Set random seed
    Random.seed!(seed)
    
    # Extract parameters with defaults
    num_airports = get(params, :num_airports, 15)
    num_bases = get(params, :num_bases, 4)
    num_flights = get(params, :num_flights, 120)
    planning_horizon_days = get(params, :planning_horizon_days, 5)
    max_pairings = get(params, :max_pairings, 200)
    flight_cost_mean = get(params, :flight_cost_mean, 2500.0)
    flight_cost_std = get(params, :flight_cost_std, 800.0)
    pairing_overhead_mean = get(params, :pairing_overhead_mean, 0.25)
    pairing_overhead_std = get(params, :pairing_overhead_std, 0.15)
    max_flights_per_pairing = get(params, :max_flights_per_pairing, 6)
    
    # Save actual parameters used
    actual_params = Dict{Symbol, Any}(
        :num_airports => num_airports,
        :num_bases => num_bases,
        :num_flights => num_flights,
        :planning_horizon_days => planning_horizon_days,
        :max_pairings => max_pairings,
        :flight_cost_mean => flight_cost_mean,
        :flight_cost_std => flight_cost_std,
        :pairing_overhead_mean => pairing_overhead_mean,
        :pairing_overhead_std => pairing_overhead_std,
        :max_flights_per_pairing => max_flights_per_pairing
    )
    
    # Model
    model = Model()
    
    # Generate the flight data
    flight_ids = 1:num_flights
    
    # Generate crew bases and other airports
    base_locations = collect(1:num_bases)
    non_base_locations = collect((num_bases+1):num_airports)
    
    # Generate realistic flight network using distributions
    flight_origins = Int[]
    flight_destinations = Int[]
    flight_costs = Float64[]
    
    # Flight cost distribution (truncated normal to ensure positive costs)
    flight_cost_dist = truncated(Normal(flight_cost_mean, flight_cost_std), 500.0, 10000.0)
    
    # Generate flights with realistic patterns
    flights_per_route = max(1, round(Int, num_flights / (num_airports * 0.6)))
    
    # Generate hub-and-spoke patterns (common in airline operations)
    for base in base_locations
        # Flights from base to non-base destinations
        for destination in non_base_locations
            if rand() < 0.7  # 70% chance of route existing
                num_flights_on_route = rand(Poisson(flights_per_route))
                for _ in 1:num_flights_on_route
                    if length(flight_costs) < num_flights
                        push!(flight_origins, base)
                        push!(flight_destinations, destination)
                        # Hub flights tend to be more expensive
                        cost = rand(flight_cost_dist) * (1 + rand(Beta(2, 5)) * 0.5)
                        push!(flight_costs, cost)
                    end
                end
            end
        end
        
        # Return flights from non-base to base
        for origin in non_base_locations
            if rand() < 0.6  # 60% chance of return route
                num_flights_on_route = rand(Poisson(flights_per_route))
                for _ in 1:num_flights_on_route
                    if length(flight_costs) < num_flights
                        push!(flight_origins, origin)
                        push!(flight_destinations, base)
                        cost = rand(flight_cost_dist)
                        push!(flight_costs, cost)
                    end
                end
            end
        end
    end
    
    # Fill remaining flights with point-to-point routes
    while length(flight_costs) < num_flights
        origin = rand(1:num_airports)
        destination = rand(1:num_airports)
        if origin != destination
            push!(flight_origins, origin)
            push!(flight_destinations, destination)
            # Point-to-point flights have variable costs
            cost = rand(flight_cost_dist) * (1 + rand(Beta(1, 3)) * 0.3)
            push!(flight_costs, cost)
        end
    end
    
    # Trim to exact number of flights
    if length(flight_costs) > num_flights
        flight_origins = flight_origins[1:num_flights]
        flight_destinations = flight_destinations[1:num_flights]
        flight_costs = flight_costs[1:num_flights]
    end
    
    actual_num_flights = length(flight_costs)
    
    # Generate realistic pairings (sequences of flights that crews can operate)
    num_pairings = min(max_pairings, actual_num_flights * 2)
    pairing_costs = Float64[]
    flights_in_pairing = Vector{Int}[]
    
    # Pairing overhead distribution
    overhead_dist = truncated(Normal(pairing_overhead_mean, pairing_overhead_std), 0.05, 0.8)
    
    # Generate pairings with realistic flight sequences
    for _ in 1:num_pairings
        if actual_num_flights == 0
            break
        end
        
        # Realistic number of flights per pairing (weighted toward shorter pairings)
        pairing_length_weights = [0.4, 0.3, 0.15, 0.1, 0.04, 0.01]  # Favor shorter pairings
        max_length = min(max_flights_per_pairing, actual_num_flights, length(pairing_length_weights))
        pairing_length = sample(1:max_length, Weights(pairing_length_weights[1:max_length]))
        
        # Try to create connected flight sequences (more realistic)
        pairing_flights = Int[]
        available_flights = collect(1:actual_num_flights)
        
        if !isempty(available_flights)
            # Start with a random flight
            current_flight = sample(available_flights)
            push!(pairing_flights, current_flight)
            filter!(f -> f != current_flight, available_flights)
            
            # Try to find connecting flights
            for _ in 2:pairing_length
                if isempty(available_flights)
                    break
                end
                
                current_dest = flight_destinations[current_flight]
                
                # Look for flights that start where the current flight ends
                connecting_flights = filter(f -> flight_origins[f] == current_dest, available_flights)
                
                if !isempty(connecting_flights)
                    # Prefer connecting flights (80% chance)
                    if rand() < 0.8 && !isempty(connecting_flights)
                        current_flight = sample(connecting_flights)
                    else
                        current_flight = sample(available_flights)
                    end
                else
                    # No connecting flights, pick any available flight
                    current_flight = sample(available_flights)
                end
                
                push!(pairing_flights, current_flight)
                filter!(f -> f != current_flight, available_flights)
            end
        end
        
        # If we couldn't find enough flights, fill with random ones
        while length(pairing_flights) < pairing_length && !isempty(available_flights)
            flight = sample(available_flights)
            push!(pairing_flights, flight)
            filter!(f -> f != flight, available_flights)
        end
        
        if !isempty(pairing_flights)
            push!(flights_in_pairing, pairing_flights)
            
            # Calculate realistic pairing cost
            base_cost = sum(flight_costs[f] for f in pairing_flights)
            overhead_factor = rand(overhead_dist)
            
            # Longer pairings have higher overhead (crew hotels, meals, etc.)
            length_multiplier = 1.0 + (length(pairing_flights) - 1) * 0.1
            
            # Base location factor (pairings starting/ending at bases are cheaper)
            base_factor = 1.0
            if flight_origins[pairing_flights[1]] in base_locations
                base_factor *= 0.9
            end
            if flight_destinations[pairing_flights[end]] in base_locations
                base_factor *= 0.9
            end
            
            pairing_cost = base_cost * (1 + overhead_factor) * length_multiplier * base_factor
            push!(pairing_costs, pairing_cost)
        end
    end
    
    # Update actual number of pairings
    actual_num_pairings = length(pairing_costs)
    
    # Store generated data in params
    actual_params[:flight_origins] = flight_origins
    actual_params[:flight_destinations] = flight_destinations
    actual_params[:flight_costs] = flight_costs
    actual_params[:pairing_costs] = pairing_costs
    actual_params[:flights_in_pairing] = flights_in_pairing
    actual_params[:actual_num_pairings] = actual_num_pairings
    
    # Decision variables: which pairings to use
    @variable(model, x[1:actual_num_pairings], Bin)
    
    # Objective: minimize total cost
    @objective(model, Min, sum(pairing_costs[p] * x[p] for p in 1:actual_num_pairings))
    
    # Constraints: each flight must be covered by exactly one pairing
    for f in 1:actual_num_flights
        covering_pairings = findall(p -> f in flights_in_pairing[p], 1:actual_num_pairings)
        if !isempty(covering_pairings)
            @constraint(model, sum(x[p] for p in covering_pairings) == 1)
        end
    end
    
    return model, actual_params
end

"""
    sample_airline_crew_parameters(size::Symbol=:medium; seed::Int=0)

Sample realistic parameters for an airline crew pairing problem.

# Arguments
- `size`: Symbol specifying the problem size (:small, :medium, :large)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_airline_crew_parameters(size::Symbol=:medium; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Set size-dependent parameters based on realistic airline scales
    if size == :small  # Regional airline (50-250 variables)
        params[:num_airports] = rand(5:15)
        params[:num_bases] = rand(2:5)
        params[:num_flights] = rand(40:120)
        params[:planning_horizon_days] = rand(3:5)
        params[:max_pairings] = rand(50:250)
        params[:flight_cost_mean] = rand(1200:2000)  # Regional flights cheaper
        params[:flight_cost_std] = rand(300:600)
        params[:pairing_overhead_mean] = rand(truncated(Normal(0.2, 0.05), 0.1, 0.3))
        params[:pairing_overhead_std] = rand(truncated(Normal(0.12, 0.03), 0.05, 0.2))
        params[:max_flights_per_pairing] = rand(3:5)
    elseif size == :medium  # National carrier (250-1000 variables)
        params[:num_airports] = rand(10:30)
        params[:num_bases] = rand(3:8)
        params[:num_flights] = rand(80:300)
        params[:planning_horizon_days] = rand(5:7)
        params[:max_pairings] = rand(200:800)
        params[:flight_cost_mean] = rand(2000:3500)  # National flights
        params[:flight_cost_std] = rand(600:1200)
        params[:pairing_overhead_mean] = rand(truncated(Normal(0.25, 0.05), 0.15, 0.35))
        params[:pairing_overhead_std] = rand(truncated(Normal(0.15, 0.03), 0.08, 0.25))
        params[:max_flights_per_pairing] = rand(4:6)
    elseif size == :large  # International airline (1000-10000 variables)
        params[:num_airports] = rand(20:100)
        params[:num_bases] = rand(5:20)
        params[:num_flights] = rand(200:1000)
        params[:planning_horizon_days] = rand(7:14)
        params[:max_pairings] = rand(800:5000)
        params[:flight_cost_mean] = rand(3000:6000)  # International flights expensive
        params[:flight_cost_std] = rand(1000:2500)
        params[:pairing_overhead_mean] = rand(truncated(Normal(0.35, 0.08), 0.2, 0.5))
        params[:pairing_overhead_std] = rand(truncated(Normal(0.18, 0.05), 0.1, 0.3))
        params[:max_flights_per_pairing] = rand(5:8)
    else
        error("Unknown size: $size. Must be :small, :medium, or :large")
    end
    
    return params
end

"""
    sample_airline_crew_parameters(target_variables::Int; seed::Int=0)

Sample realistic parameters for an airline crew pairing problem targeting approximately the specified number of variables.

# Arguments
- `target_variables`: Target number of variables in the LP formulation (approximately within ±10%)
- `seed`: Random seed for reproducibility (default: 0)

# Returns
- Dictionary of sampled parameters
"""
function sample_airline_crew_parameters(target_variables::Int; seed::Int=0)
    Random.seed!(seed)
    
    params = Dict{Symbol, Any}()
    
    # Determine scale based on target variables
    if target_variables <= 300
        # Small scale - regional airline
        scale = :small
        base_airports = rand(5:12)
        base_bases = rand(2:4)
        base_flights = rand(40:80)
        base_pairings = rand(50:150)
    elseif target_variables <= 1500
        # Medium scale - national carrier
        scale = :medium
        base_airports = rand(10:25)
        base_bases = rand(3:6)
        base_flights = rand(80:200)
        base_pairings = rand(150:500)
    else
        # Large scale - international airline
        scale = :large
        base_airports = rand(20:60)
        base_bases = rand(5:15)
        base_flights = rand(150:600)
        base_pairings = rand(400:2000)
    end
    
    # Set realistic parameters based on scale
    params[:num_airports] = base_airports
    params[:num_bases] = base_bases
    params[:num_flights] = base_flights
    params[:max_pairings] = base_pairings
    params[:planning_horizon_days] = scale == :small ? rand(3:5) : 
                                     scale == :medium ? rand(5:7) : rand(7:12)
    
    # Set cost parameters based on scale
    if scale == :small
        params[:flight_cost_mean] = rand(1200:2000)
        params[:flight_cost_std] = rand(300:600)
        params[:pairing_overhead_mean] = rand(truncated(Normal(0.2, 0.05), 0.1, 0.3))
        params[:pairing_overhead_std] = rand(truncated(Normal(0.12, 0.03), 0.05, 0.2))
        params[:max_flights_per_pairing] = rand(3:5)
    elseif scale == :medium
        params[:flight_cost_mean] = rand(2000:3500)
        params[:flight_cost_std] = rand(600:1200)
        params[:pairing_overhead_mean] = rand(truncated(Normal(0.25, 0.05), 0.15, 0.35))
        params[:pairing_overhead_std] = rand(truncated(Normal(0.15, 0.03), 0.08, 0.25))
        params[:max_flights_per_pairing] = rand(4:6)
    else
        params[:flight_cost_mean] = rand(3000:6000)
        params[:flight_cost_std] = rand(1000:2500)
        params[:pairing_overhead_mean] = rand(truncated(Normal(0.35, 0.08), 0.2, 0.5))
        params[:pairing_overhead_std] = rand(truncated(Normal(0.18, 0.05), 0.1, 0.3))
        params[:max_flights_per_pairing] = rand(5:8)
    end
    
    # Iteratively adjust parameters to reach target
    for iteration in 1:15
        current_vars = calculate_airline_crew_variable_count(params)
        
        if abs(current_vars - target_variables) / target_variables < 0.1
            break  # Within 10% tolerance
        end
        
        # Adjust parameters based on current vs target
        ratio = target_variables / current_vars
        
        if ratio > 1.3  # Need significantly more variables
            params[:num_flights] = round(Int, params[:num_flights] * sqrt(ratio))
            params[:max_pairings] = round(Int, params[:max_pairings] * sqrt(ratio))
            if ratio > 2.0
                params[:num_airports] = min(100, round(Int, params[:num_airports] * 1.2))
                params[:num_bases] = min(20, round(Int, params[:num_bases] * 1.1))
            end
        elseif ratio < 0.7  # Need significantly fewer variables
            params[:num_flights] = max(20, round(Int, params[:num_flights] * sqrt(ratio)))
            params[:max_pairings] = max(30, round(Int, params[:max_pairings] * sqrt(ratio)))
            if ratio < 0.5
                params[:num_airports] = max(3, round(Int, params[:num_airports] * 0.9))
                params[:num_bases] = max(2, round(Int, params[:num_bases] * 0.9))
            end
        else  # Fine-tune
            params[:max_pairings] = max(30, round(Int, params[:max_pairings] * ratio))
        end
    end
    
    return params
end

function calculate_airline_crew_variable_count(params::Dict)
    # Extract parameters with defaults
    num_airports = get(params, :num_airports, 15)
    num_bases = get(params, :num_bases, 4)
    num_flights = get(params, :num_flights, 120)
    max_pairings = get(params, :max_pairings, 200)
    
    # Calculate number of pairings (same logic as in generate function)
    num_pairings = min(max_pairings, num_flights * 2)
    
    # Variables: x[1:num_pairings] - binary variables for pairing selection
    return num_pairings
end

# Register the problem type
register_problem(
    :airline_crew,
    generate_airline_crew_problem,
    sample_airline_crew_parameters,
    "Airline crew pairing problem that optimizes the assignment of crews to flight sequences with realistic cost structures and operational constraints"
)