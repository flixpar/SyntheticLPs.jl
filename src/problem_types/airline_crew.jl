using JuMP
using Random
using StatsBase
using LinearAlgebra
using Distributions

"""
    AirlineCrewProblem <: ProblemGenerator

Generator for airline crew pairing problems with realistic cost structures and operational constraints.

# Fields
- `num_flights::Int`: Number of flights in planning horizon
- `flight_origins::Vector{Int}`: Origin airport for each flight
- `flight_destinations::Vector{Int}`: Destination airport for each flight
- `pairing_costs::Vector{Float64}`: Cost of each pairing
- `flights_in_pairing::Vector{Vector{Int}}`: Flights covered by each pairing
"""
struct AirlineCrewProblem <: ProblemGenerator
    num_flights::Int
    flight_origins::Vector{Int}
    flight_destinations::Vector{Int}
    pairing_costs::Vector{Float64}
    flights_in_pairing::Vector{Vector{Int}}
end

"""
    AirlineCrewProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an airline crew pairing problem instance.

# Arguments
- `target_variables`: Target number of variables (pairings)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function AirlineCrewProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine scale
    if target_variables <= 300
        scale = :small
        num_airports = rand(5:12)
        num_bases = rand(2:4)
        num_flights = rand(40:80)
        max_pairings = rand(50:150)
        flight_cost_mean = rand(1200:2000)
        flight_cost_std = rand(300:600)
        pairing_overhead_mean = rand(truncated(Normal(0.2, 0.05), 0.1, 0.3))
        pairing_overhead_std = rand(truncated(Normal(0.12, 0.03), 0.05, 0.2))
        max_flights_per_pairing = rand(3:5)
    elseif target_variables <= 1500
        scale = :medium
        num_airports = rand(10:25)
        num_bases = rand(3:6)
        num_flights = rand(80:200)
        max_pairings = rand(150:500)
        flight_cost_mean = rand(2000:3500)
        flight_cost_std = rand(600:1200)
        pairing_overhead_mean = rand(truncated(Normal(0.25, 0.05), 0.15, 0.35))
        pairing_overhead_std = rand(truncated(Normal(0.15, 0.03), 0.08, 0.25))
        max_flights_per_pairing = rand(4:6)
    else
        scale = :large
        num_airports = rand(20:60)
        num_bases = rand(5:15)
        num_flights = rand(150:600)
        max_pairings = rand(400:2000)
        flight_cost_mean = rand(3000:6000)
        flight_cost_std = rand(1000:2500)
        pairing_overhead_mean = rand(truncated(Normal(0.35, 0.08), 0.2, 0.5))
        pairing_overhead_std = rand(truncated(Normal(0.18, 0.05), 0.1, 0.3))
        max_flights_per_pairing = rand(5:8)
    end

    # Adjust for target
    for iteration in 1:15
        current_vars = min(max_pairings, num_flights * 2)
        if abs(current_vars - target_variables) / target_variables < 0.1
            break
        end

        ratio = target_variables / current_vars
        if ratio > 1.3
            num_flights = round(Int, num_flights * sqrt(ratio))
            max_pairings = round(Int, max_pairings * sqrt(ratio))
            if ratio > 2.0
                num_airports = min(100, round(Int, num_airports * 1.2))
                num_bases = min(20, round(Int, num_bases * 1.1))
            end
        elseif ratio < 0.7
            num_flights = max(20, round(Int, num_flights * sqrt(ratio)))
            max_pairings = max(30, round(Int, max_pairings * sqrt(ratio)))
            if ratio < 0.5
                num_airports = max(3, round(Int, num_airports * 0.9))
                num_bases = max(2, round(Int, num_bases * 0.9))
            end
        else
            max_pairings = max(30, round(Int, max_pairings * ratio))
        end
    end

    # Generate flight network
    base_locations = collect(1:num_bases)
    non_base_locations = collect((num_bases+1):num_airports)

    flight_origins = Int[]
    flight_destinations = Int[]
    flight_costs = Float64[]

    flight_cost_dist = truncated(Normal(flight_cost_mean, flight_cost_std), 500.0, 10000.0)
    flights_per_route = max(1, round(Int, num_flights / (num_airports * 0.6)))

    # Hub-and-spoke patterns
    for base in base_locations
        for destination in non_base_locations
            if rand() < 0.7
                n_flights_on_route = rand(Poisson(flights_per_route))
                for _ in 1:n_flights_on_route
                    if length(flight_costs) < num_flights
                        push!(flight_origins, base)
                        push!(flight_destinations, destination)
                        cost = rand(flight_cost_dist) * (1 + rand(Beta(2, 5)) * 0.5)
                        push!(flight_costs, cost)
                    end
                end
            end
        end

        for origin in non_base_locations
            if rand() < 0.6
                n_flights_on_route = rand(Poisson(flights_per_route))
                for _ in 1:n_flights_on_route
                    if length(flight_costs) < num_flights
                        push!(flight_origins, origin)
                        push!(flight_destinations, base)
                        push!(flight_costs, rand(flight_cost_dist))
                    end
                end
            end
        end
    end

    # Fill with point-to-point
    while length(flight_costs) < num_flights
        origin = rand(1:num_airports)
        destination = rand(1:num_airports)
        if origin != destination
            push!(flight_origins, origin)
            push!(flight_destinations, destination)
            cost = rand(flight_cost_dist) * (1 + rand(Beta(1, 3)) * 0.3)
            push!(flight_costs, cost)
        end
    end

    if length(flight_costs) > num_flights
        flight_origins = flight_origins[1:num_flights]
        flight_destinations = flight_destinations[1:num_flights]
        flight_costs = flight_costs[1:num_flights]
    end

    actual_num_flights = length(flight_costs)
    num_pairings = min(max_pairings, actual_num_flights * 2)

    pairing_costs = Float64[]
    flights_in_pairing = Vector{Int}[]

    overhead_dist = truncated(Normal(pairing_overhead_mean, pairing_overhead_std), 0.05, 0.8)

    function compute_pairing_cost(pairing_flights::Vector{Int})
        base_cost = sum(flight_costs[f] for f in pairing_flights)
        overhead_factor = rand(overhead_dist)
        length_multiplier = 1.0 + (length(pairing_flights) - 1) * 0.1
        base_factor = 1.0
        if flight_origins[pairing_flights[1]] in base_locations
            base_factor *= 0.9
        end
        if flight_destinations[pairing_flights[end]] in base_locations
            base_factor *= 0.9
        end
        return base_cost * (1 + overhead_factor) * length_multiplier * base_factor
    end

    # Build adjacency
    flights_from_origin = Dict{Int, Vector{Int}}()
    for (fid, o) in enumerate(flight_origins)
        if !haskey(flights_from_origin, o)
            flights_from_origin[o] = Int[]
        end
        push!(flights_from_origin[o], fid)
    end

    function generate_connected_sequence(max_length::Int; avoid::Set{Int}=Set{Int}())
        candidates = [f for f in 1:actual_num_flights if !(f in avoid)]
        if isempty(candidates)
            return Int[]
        end
        base_starts = [f for f in candidates if flight_origins[f] in base_locations]
        current_flight = isempty(base_starts) ? sample(candidates) : sample(base_starts)
        sequence = Int[current_flight]
        used = Set([current_flight])
        for _ in 2:max_length
            current_dest = flight_destinations[current_flight]
            next_options = haskey(flights_from_origin, current_dest) ? [f for f in flights_from_origin[current_dest] if !(f in used) && !(f in avoid)] : Int[]
            if isempty(next_options)
                break
            end
            if rand() < 0.8
                current_flight = sample(next_options)
            else
                any_unused = [f for f in candidates if !(f in used)]
                if isempty(any_unused)
                    break
                end
                current_flight = sample(any_unused)
            end
            push!(sequence, current_flight)
            push!(used, current_flight)
        end
        return sequence
    end

    target_feasible = feasibility_status == feasible ? true :
                     feasibility_status == infeasible ? false :
                     rand() < 0.5

    if target_feasible
        # Build exact cover (partition)
        unassigned = Set(1:actual_num_flights)
        while !isempty(unassigned)
            max_length = min(max_flights_per_pairing, length(unassigned))
            pairing_length_weights = [0.4, 0.3, 0.15, 0.1, 0.04, 0.01]
            max_weight_len = min(max_length, length(pairing_length_weights))
            desired_len = sample(1:max_weight_len, Weights(pairing_length_weights[1:max_weight_len]))
            seq = generate_connected_sequence(desired_len; avoid=Set{Int}())
            if isempty(intersect(collect(unassigned), seq))
                start = first(unassigned)
                seq = [start]
                current = start
                while length(seq) < desired_len
                    current_dest = flight_destinations[current]
                    options = haskey(flights_from_origin, current_dest) ? [f for f in flights_from_origin[current_dest] if f in unassigned && !(f in seq)] : Int[]
                    if isempty(options)
                        break
                    end
                    nxt = sample(options)
                    push!(seq, nxt)
                    current = nxt
                end
            end
            seq = [f for f in seq if f in unassigned]
            if isempty(seq)
                start = first(unassigned)
                seq = [start]
            end
            if length(seq) > max_flights_per_pairing
                seq = seq[1:max_flights_per_pairing]
            end
            push!(flights_in_pairing, seq)
            push!(pairing_costs, compute_pairing_cost(seq))
            for f in seq
                pop!(unassigned, f)
            end
        end

        # Add additional pairings
        while length(flights_in_pairing) < num_pairings
            max_length = min(max_flights_per_pairing, actual_num_flights)
            pairing_length_weights = [0.4, 0.3, 0.15, 0.1, 0.04, 0.01]
            max_weight_len = min(max_length, length(pairing_length_weights))
            desired_len = sample(1:max_weight_len, Weights(pairing_length_weights[1:max_weight_len]))
            seq = generate_connected_sequence(desired_len)
            if !isempty(seq)
                push!(flights_in_pairing, seq)
                push!(pairing_costs, compute_pairing_cost(seq))
            end
        end
    else
        # Generate without guaranteeing exact cover
        avoid_set = Set{Int}()
        if actual_num_flights > 0
            push!(avoid_set, rand(1:actual_num_flights))
        end
        pairing_length_weights = [0.4, 0.3, 0.15, 0.1, 0.04, 0.01]
        for _ in 1:num_pairings
            if actual_num_flights == 0
                break
            end
            max_length = min(max_flights_per_pairing, actual_num_flights, length(pairing_length_weights))
            desired_len = sample(1:max_length, Weights(pairing_length_weights[1:max_length]))
            seq = generate_connected_sequence(desired_len; avoid=avoid_set)
            if isempty(seq)
                candidates = [f for f in 1:actual_num_flights if !(f in avoid_set)]
                if isempty(candidates)
                    continue
                end
                seq = [sample(candidates)]
            end
            push!(flights_in_pairing, seq)
            push!(pairing_costs, compute_pairing_cost(seq))
        end
    end

    return AirlineCrewProblem(actual_num_flights, flight_origins, flight_destinations,
                              pairing_costs, flights_in_pairing)
end

"""
    build_model(prob::AirlineCrewProblem)

Build a JuMP model for the airline crew pairing problem.

# Arguments
- `prob`: AirlineCrewProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::AirlineCrewProblem)
    model = Model()

    actual_num_pairings = length(prob.pairing_costs)

    @variable(model, x[1:actual_num_pairings], Bin)

    @objective(model, Min, sum(prob.pairing_costs[p] * x[p] for p in 1:actual_num_pairings))

    # Each flight must be covered exactly once
    for f in 1:prob.num_flights
        covering_pairings = findall(p -> f in prob.flights_in_pairing[p], 1:actual_num_pairings)
        @constraint(model, sum(x[p] for p in covering_pairings) == 1)
    end

    return model
end

# Register the problem type
register_problem(
    :airline_crew,
    AirlineCrewProblem,
    "Airline crew pairing problem that optimizes the assignment of crews to flight sequences with realistic cost structures and operational constraints"
)
