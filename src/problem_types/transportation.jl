using JuMP
using Random
using Distributions

"""
Transportation problem variants that control the constraint and objective structure.

# Variants
- `standard`: Classic transportation with supply/demand constraints (default)
- `balanced`: Force total supply = total demand (equality constraints)
- `capacitated`: Add route capacity limits on individual source-destination pairs
- `multi_commodity`: Multiple product types with separate demands
- `transshipment`: Add intermediate transshipment nodes
- `time_windows`: Add time-based constraints on deliveries
- `service_level`: Minimum percentage of demand that must be satisfied
- `emission_constrained`: Total CO2 emissions from transportation must be below threshold
"""
@enum TransportationVariant begin
    transport_standard
    transport_balanced
    transport_capacitated
    transport_multi_commodity
    transport_transshipment
    transport_time_windows
    transport_service_level
    transport_emission_constrained
end

"""
    TransportationProblem <: ProblemGenerator

Generator for transportation problems that optimize shipping goods from sources to destinations.

# Fields
- `n_sources::Int`: Number of supply sources
- `n_destinations::Int`: Number of demand destinations
- `supplies::Vector{Int}`: Supply at each source
- `demands::Vector{Int}`: Demand at each destination
- `costs::Matrix{Float64}`: Transportation cost from each source to each destination
- `variant::TransportationVariant`: Problem variant type
- `route_capacities::Union{Matrix{Float64}, Nothing}`: Route capacity limits (for capacitated)
- `n_commodities::Int`: Number of commodities (for multi_commodity)
- `commodity_demands::Union{Matrix{Int}, Nothing}`: Demand per commodity (commodities × destinations)
- `commodity_supplies::Union{Matrix{Int}, Nothing}`: Supply per commodity (commodities × sources)
- `n_transship::Int`: Number of transshipment nodes (for transshipment)
- `transship_costs::Union{Dict{Tuple{Int,Int,Int},Float64}, Nothing}`: Transshipment arc costs
- `transship_capacities::Union{Dict{Tuple{Int,Int,Int},Float64}, Nothing}`: Transshipment arc capacities
- `time_windows::Union{Matrix{Tuple{Float64,Float64}}, Nothing}`: (earliest, latest) delivery times
- `transit_times::Union{Matrix{Float64}, Nothing}`: Transit time for each route
- `service_level_min::Float64`: Minimum service level (fraction of demand to satisfy)
- `emission_rates::Union{Matrix{Float64}, Nothing}`: CO2 emission per unit shipped on each route
- `emission_limit::Float64`: Maximum total emissions allowed
- `distances::Union{Matrix{Float64}, Nothing}`: Geographic distances between nodes
"""
struct TransportationProblem <: ProblemGenerator
    n_sources::Int
    n_destinations::Int
    supplies::Vector{Int}
    demands::Vector{Int}
    costs::Matrix{Float64}
    variant::TransportationVariant
    # Capacitated variant
    route_capacities::Union{Matrix{Float64}, Nothing}
    # Multi-commodity variant
    n_commodities::Int
    commodity_demands::Union{Matrix{Int}, Nothing}
    commodity_supplies::Union{Matrix{Int}, Nothing}
    # Transshipment variant
    n_transship::Int
    transship_costs::Union{Dict{Tuple{Int,Int,Int},Float64}, Nothing}
    transship_capacities::Union{Dict{Tuple{Int,Int,Int},Float64}, Nothing}
    # Time windows variant
    time_windows::Union{Matrix{Tuple{Float64,Float64}}, Nothing}
    transit_times::Union{Matrix{Float64}, Nothing}
    # Service level variant
    service_level_min::Float64
    # Emission constrained variant
    emission_rates::Union{Matrix{Float64}, Nothing}
    emission_limit::Float64
    # Geographic data (used by multiple variants)
    distances::Union{Matrix{Float64}, Nothing}
end

# Convenience constructor for backwards compatibility
function TransportationProblem(n_sources::Int, n_destinations::Int,
                               supplies::Vector{Int}, demands::Vector{Int},
                               costs::Matrix{Int})
    TransportationProblem(
        n_sources, n_destinations, supplies, demands, Float64.(costs),
        transport_standard, nothing, 1, nothing, nothing, 0, nothing, nothing,
        nothing, nothing, 1.0, nothing, 0.0, nothing
    )
end

"""
    TransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                          variant::TransportationVariant=transport_standard)

Construct a transportation problem instance with the specified variant.

# Arguments
- `target_variables`: Target number of variables (n_sources × n_destinations)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
- `variant`: Problem variant (default: transport_standard)
"""
function TransportationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                               variant::TransportationVariant=transport_standard)
    Random.seed!(seed)

    # Calculate dimensions to achieve target number of variables
    sqrt_target = sqrt(target_variables)
    ratio = 0.5 + rand() * 1.0

    n_sources = max(2, round(Int, sqrt_target * ratio))
    n_destinations = max(2, round(Int, target_variables / n_sources))

    # Fine-tune to get closer to target
    current_vars = n_sources * n_destinations
    if current_vars < target_variables * 0.9
        if n_sources >= n_destinations
            n_sources = max(n_sources, round(Int, target_variables / n_destinations))
        else
            n_destinations = max(n_destinations, round(Int, target_variables / n_sources))
        end
    elseif current_vars > target_variables * 1.1
        if n_sources >= n_destinations
            n_sources = max(2, round(Int, target_variables / n_destinations))
        else
            n_destinations = max(2, round(Int, target_variables / n_sources))
        end
    end

    # Set realistic parameter ranges based on problem size
    total_vars = n_sources * n_destinations
    if total_vars <= 250
        supply_range = (rand(50:100), rand(200:500))
        demand_range = (rand(30:80), rand(150:300))
        cost_range = (rand(5.0:0.5:15.0), rand(25.0:0.5:60.0))
    elseif total_vars <= 1000
        supply_range = (rand(100:500), rand(1000:5000))
        demand_range = (rand(80:300), rand(800:3000))
        cost_range = (rand(10.0:0.5:30.0), rand(50.0:0.5:150.0))
    else
        supply_range = (rand(500:2000), rand(5000:50000))
        demand_range = (rand(300:1500), rand(3000:30000))
        cost_range = (rand(20.0:0.5:100.0), rand(100.0:0.5:500.0))
    end

    # Generate random data
    min_supply, max_supply = supply_range
    supplies = rand(min_supply:max_supply, n_sources)

    min_demand, max_demand = demand_range
    demands = rand(min_demand:max_demand, n_destinations)

    # Generate geographic positions for realistic distance-based costs
    source_positions = [(rand() * 100.0, rand() * 100.0) for _ in 1:n_sources]
    dest_positions = [(rand() * 100.0, rand() * 100.0) for _ in 1:n_destinations]

    distances = zeros(n_sources, n_destinations)
    for i in 1:n_sources, j in 1:n_destinations
        distances[i, j] = sqrt((source_positions[i][1] - dest_positions[j][1])^2 +
                               (source_positions[i][2] - dest_positions[j][2])^2)
    end

    # Generate costs based on distances with variation
    min_cost, max_cost = cost_range
    cost_per_distance = (max_cost - min_cost) / maximum(distances)
    costs = zeros(n_sources, n_destinations)
    for i in 1:n_sources, j in 1:n_destinations
        base_cost = min_cost + distances[i, j] * cost_per_distance
        costs[i, j] = base_cost * (0.8 + 0.4 * rand())
    end

    # Initialize variant-specific fields
    route_capacities = nothing
    n_commodities = 1
    commodity_demands = nothing
    commodity_supplies = nothing
    n_transship = 0
    transship_costs = nothing
    transship_capacities = nothing
    time_windows = nothing
    transit_times = nothing
    service_level_min = 1.0
    emission_rates = nothing
    emission_limit = 0.0

    # Generate variant-specific data
    if variant == transport_capacitated
        # Route capacity limits - some routes have limited capacity
        avg_flow = sum(demands) / (n_sources * n_destinations)
        route_capacities = zeros(n_sources, n_destinations)
        for i in 1:n_sources, j in 1:n_destinations
            if rand() < 0.7  # 70% of routes have capacity limits
                route_capacities[i, j] = avg_flow * rand(Uniform(0.5, 3.0))
            else
                route_capacities[i, j] = Inf  # Unlimited
            end
        end

    elseif variant == transport_multi_commodity
        # Multiple commodity types with different demands
        n_commodities = rand(2:min(5, max(2, n_destinations ÷ 2)))
        commodity_demands = zeros(Int, n_commodities, n_destinations)
        commodity_supplies = zeros(Int, n_commodities, n_sources)

        for k in 1:n_commodities
            # Split original demand among commodities
            for j in 1:n_destinations
                commodity_demands[k, j] = round(Int, demands[j] / n_commodities * (0.5 + rand()))
            end
            for i in 1:n_sources
                commodity_supplies[k, i] = round(Int, supplies[i] / n_commodities * (0.5 + rand()))
            end
        end

    elseif variant == transport_transshipment
        # Add intermediate transshipment nodes
        n_transship = rand(1:max(1, min(n_sources, n_destinations) ÷ 2))
        transship_costs = Dict{Tuple{Int,Int,Int},Float64}()
        transship_capacities = Dict{Tuple{Int,Int,Int},Float64}()

        transship_positions = [(rand() * 100.0, rand() * 100.0) for _ in 1:n_transship]

        # Costs from sources to transshipment nodes
        for i in 1:n_sources, t in 1:n_transship
            dist = sqrt((source_positions[i][1] - transship_positions[t][1])^2 +
                       (source_positions[i][2] - transship_positions[t][2])^2)
            transship_costs[(1, i, t)] = (min_cost + dist * cost_per_distance) * 0.8
            transship_capacities[(1, i, t)] = supplies[i] * rand(Uniform(0.3, 0.8))
        end

        # Costs from transshipment nodes to destinations
        for t in 1:n_transship, j in 1:n_destinations
            dist = sqrt((transship_positions[t][1] - dest_positions[j][1])^2 +
                       (transship_positions[t][2] - dest_positions[j][2])^2)
            transship_costs[(2, t, j)] = (min_cost + dist * cost_per_distance) * 0.8
            transship_capacities[(2, t, j)] = demands[j] * rand(Uniform(0.5, 1.5))
        end

    elseif variant == transport_time_windows
        # Add time window constraints for deliveries
        time_windows = Matrix{Tuple{Float64,Float64}}(undef, n_sources, n_destinations)
        transit_times = zeros(n_sources, n_destinations)

        base_time = 0.0
        time_horizon = 24.0  # 24 hour planning horizon

        for i in 1:n_sources, j in 1:n_destinations
            # Transit time proportional to distance
            transit_times[i, j] = distances[i, j] / 20.0 * (0.8 + 0.4 * rand())

            # Time windows with some flexibility
            earliest = rand(Uniform(0.0, time_horizon * 0.5))
            latest = earliest + rand(Uniform(transit_times[i, j] * 1.5, time_horizon * 0.5))
            time_windows[i, j] = (earliest, min(latest, time_horizon))
        end

    elseif variant == transport_service_level
        # Minimum service level constraint (allow partial demand satisfaction)
        service_level_min = rand(Uniform(0.85, 0.98))

    elseif variant == transport_emission_constrained
        # CO2 emission constraints
        emission_rates = zeros(n_sources, n_destinations)
        for i in 1:n_sources, j in 1:n_destinations
            # Emissions proportional to distance with vehicle type variation
            emission_rates[i, j] = distances[i, j] * rand(Uniform(0.01, 0.05))
        end

        # Total emission limit based on minimum possible emissions
        total_demand = sum(demands)
        min_emissions = minimum(emission_rates) * total_demand
        max_emissions = maximum(emission_rates) * total_demand
        emission_limit = rand(Uniform(min_emissions * 1.5, max_emissions * 0.7))
    end

    # Helper function to distribute additions across a vector
    function distribute_additions!(vec::Vector{Int}, amount::Int)
        if amount <= 0
            return
        end
        w = rand(length(vec))
        w_sum = sum(w)
        base = floor.(Int, (w ./ w_sum) .* amount)
        remainder = amount - sum(base)
        if remainder > 0
            for idx in randperm(length(vec))[1:min(remainder, length(vec))]
                base[idx] += 1
            end
        end
        vec .+= base
    end

    # Adjust for feasibility
    total_supply = sum(supplies)
    total_demand = sum(demands)

    if feasibility_status == feasible
        if variant == transport_balanced
            # For balanced, make supply exactly equal demand
            if total_supply > total_demand
                distribute_additions!(demands, total_supply - total_demand)
            elseif total_demand > total_supply
                distribute_additions!(supplies, total_demand - total_supply)
            end
        else
            # Guarantee feasibility: ensure total_supply >= total_demand
            if total_supply < total_demand
                shortage = total_demand - total_supply
                distribute_additions!(supplies, shortage)
            end
        end

        # Adjust variant-specific constraints for feasibility
        if variant == transport_capacitated && route_capacities !== nothing
            # Ensure enough route capacity exists
            total_route_capacity = sum(route_capacities[isfinite.(route_capacities)])
            if total_route_capacity < total_demand
                scale = total_demand / total_route_capacity * 1.2
                route_capacities[isfinite.(route_capacities)] .*= scale
            end
        end

        if variant == transport_emission_constrained && emission_rates !== nothing
            # Ensure emission limit is achievable
            min_possible_emissions = sum(minimum(emission_rates, dims=1)) .* demands'
            if emission_limit < sum(min_possible_emissions)
                emission_limit = sum(min_possible_emissions) * 1.1
            end
        end

    elseif feasibility_status == infeasible
        if variant == transport_balanced
            # Make it impossible to balance
            distribute_additions!(demands, round(Int, total_supply * 0.3))
            supplies = max.(1, supplies .- round.(Int, supplies .* 0.2))
        else
            # Guarantee infeasibility: ensure total_demand > total_supply with margin
            target_margin = max(1, round(Int, (0.02 + 0.08 * rand()) * max(total_supply, 1)))
            missing = (total_supply + target_margin) - total_demand
            if missing > 0
                distribute_additions!(demands, missing)
            end
        end

        # Make variant-specific constraints infeasible
        if variant == transport_capacitated && route_capacities !== nothing
            # Reduce capacities below demand
            route_capacities .*= 0.3
        end

        if variant == transport_emission_constrained && emission_rates !== nothing
            # Set emission limit too low
            emission_limit = minimum(emission_rates) * sum(demands) * 0.5
        end
    end

    return TransportationProblem(
        n_sources, n_destinations, supplies, demands, costs, variant,
        route_capacities, n_commodities, commodity_demands, commodity_supplies,
        n_transship, transship_costs, transship_capacities,
        time_windows, transit_times, service_level_min,
        emission_rates, emission_limit, distances
    )
end

"""
    build_model(prob::TransportationProblem)

Build a JuMP model for the transportation problem based on its variant.

# Arguments
- `prob`: TransportationProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::TransportationProblem)
    model = Model()

    if prob.variant == transport_standard || prob.variant == transport_balanced
        # Standard/balanced transportation problem
        @variable(model, x[1:prob.n_sources, 1:prob.n_destinations] >= 0)

        @objective(model, Min, sum(prob.costs[i, j] * x[i, j]
                                   for i in 1:prob.n_sources, j in 1:prob.n_destinations))

        # Supply constraints
        for i in 1:prob.n_sources
            if prob.variant == transport_balanced
                @constraint(model, sum(x[i, j] for j in 1:prob.n_destinations) == prob.supplies[i])
            else
                @constraint(model, sum(x[i, j] for j in 1:prob.n_destinations) <= prob.supplies[i])
            end
        end

        # Demand constraints
        for j in 1:prob.n_destinations
            if prob.variant == transport_balanced
                @constraint(model, sum(x[i, j] for i in 1:prob.n_sources) == prob.demands[j])
            else
                @constraint(model, sum(x[i, j] for i in 1:prob.n_sources) >= prob.demands[j])
            end
        end

    elseif prob.variant == transport_capacitated
        @variable(model, x[1:prob.n_sources, 1:prob.n_destinations] >= 0)

        @objective(model, Min, sum(prob.costs[i, j] * x[i, j]
                                   for i in 1:prob.n_sources, j in 1:prob.n_destinations))

        for i in 1:prob.n_sources
            @constraint(model, sum(x[i, j] for j in 1:prob.n_destinations) <= prob.supplies[i])
        end

        for j in 1:prob.n_destinations
            @constraint(model, sum(x[i, j] for i in 1:prob.n_sources) >= prob.demands[j])
        end

        # Route capacity constraints
        for i in 1:prob.n_sources, j in 1:prob.n_destinations
            if prob.route_capacities !== nothing && isfinite(prob.route_capacities[i, j])
                @constraint(model, x[i, j] <= prob.route_capacities[i, j])
            end
        end

    elseif prob.variant == transport_multi_commodity
        @variable(model, x[1:prob.n_commodities, 1:prob.n_sources, 1:prob.n_destinations] >= 0)

        # Objective: minimize total cost (same cost regardless of commodity)
        @objective(model, Min, sum(prob.costs[i, j] * x[k, i, j]
                                   for k in 1:prob.n_commodities,
                                   i in 1:prob.n_sources, j in 1:prob.n_destinations))

        # Supply constraints per commodity
        for k in 1:prob.n_commodities, i in 1:prob.n_sources
            @constraint(model, sum(x[k, i, j] for j in 1:prob.n_destinations) <= prob.commodity_supplies[k, i])
        end

        # Demand constraints per commodity
        for k in 1:prob.n_commodities, j in 1:prob.n_destinations
            @constraint(model, sum(x[k, i, j] for i in 1:prob.n_sources) >= prob.commodity_demands[k, j])
        end

    elseif prob.variant == transport_transshipment
        # Variables for direct shipping and transshipment
        @variable(model, x_direct[1:prob.n_sources, 1:prob.n_destinations] >= 0)
        @variable(model, x_to_trans[1:prob.n_sources, 1:prob.n_transship] >= 0)
        @variable(model, x_from_trans[1:prob.n_transship, 1:prob.n_destinations] >= 0)

        # Objective: minimize total cost
        @objective(model, Min,
            sum(prob.costs[i, j] * x_direct[i, j] for i in 1:prob.n_sources, j in 1:prob.n_destinations) +
            sum(prob.transship_costs[(1, i, t)] * x_to_trans[i, t]
                for i in 1:prob.n_sources, t in 1:prob.n_transship
                if haskey(prob.transship_costs, (1, i, t))) +
            sum(prob.transship_costs[(2, t, j)] * x_from_trans[t, j]
                for t in 1:prob.n_transship, j in 1:prob.n_destinations
                if haskey(prob.transship_costs, (2, t, j)))
        )

        # Supply constraints
        for i in 1:prob.n_sources
            @constraint(model,
                sum(x_direct[i, j] for j in 1:prob.n_destinations) +
                sum(x_to_trans[i, t] for t in 1:prob.n_transship) <= prob.supplies[i])
        end

        # Demand constraints
        for j in 1:prob.n_destinations
            @constraint(model,
                sum(x_direct[i, j] for i in 1:prob.n_sources) +
                sum(x_from_trans[t, j] for t in 1:prob.n_transship) >= prob.demands[j])
        end

        # Transshipment flow balance
        for t in 1:prob.n_transship
            @constraint(model,
                sum(x_to_trans[i, t] for i in 1:prob.n_sources) ==
                sum(x_from_trans[t, j] for j in 1:prob.n_destinations))
        end

        # Transshipment capacity constraints
        for i in 1:prob.n_sources, t in 1:prob.n_transship
            if haskey(prob.transship_capacities, (1, i, t))
                @constraint(model, x_to_trans[i, t] <= prob.transship_capacities[(1, i, t)])
            end
        end
        for t in 1:prob.n_transship, j in 1:prob.n_destinations
            if haskey(prob.transship_capacities, (2, t, j))
                @constraint(model, x_from_trans[t, j] <= prob.transship_capacities[(2, t, j)])
            end
        end

    elseif prob.variant == transport_time_windows
        @variable(model, x[1:prob.n_sources, 1:prob.n_destinations] >= 0)
        @variable(model, arrival_time[1:prob.n_sources, 1:prob.n_destinations] >= 0)
        @variable(model, departure_time[1:prob.n_sources] >= 0)

        @objective(model, Min, sum(prob.costs[i, j] * x[i, j]
                                   for i in 1:prob.n_sources, j in 1:prob.n_destinations))

        for i in 1:prob.n_sources
            @constraint(model, sum(x[i, j] for j in 1:prob.n_destinations) <= prob.supplies[i])
        end

        for j in 1:prob.n_destinations
            @constraint(model, sum(x[i, j] for i in 1:prob.n_sources) >= prob.demands[j])
        end

        # Time window constraints (linearized)
        M = 1000.0  # Big-M constant
        for i in 1:prob.n_sources, j in 1:prob.n_destinations
            earliest, latest = prob.time_windows[i, j]
            transit = prob.transit_times[i, j]

            # Arrival time = departure + transit (when flow exists)
            @constraint(model, arrival_time[i, j] >= departure_time[i] + transit - M * (1 - x[i, j] / max(prob.supplies[i], 1)))

            # Time window bounds
            @constraint(model, arrival_time[i, j] >= earliest)
            @constraint(model, arrival_time[i, j] <= latest)
        end

    elseif prob.variant == transport_service_level
        @variable(model, x[1:prob.n_sources, 1:prob.n_destinations] >= 0)
        @variable(model, satisfied[1:prob.n_destinations] >= 0)
        @variable(model, unsatisfied[1:prob.n_destinations] >= 0)

        @objective(model, Min, sum(prob.costs[i, j] * x[i, j]
                                   for i in 1:prob.n_sources, j in 1:prob.n_destinations))

        for i in 1:prob.n_sources
            @constraint(model, sum(x[i, j] for j in 1:prob.n_destinations) <= prob.supplies[i])
        end

        # Demand satisfaction tracking
        for j in 1:prob.n_destinations
            @constraint(model, satisfied[j] == sum(x[i, j] for i in 1:prob.n_sources))
            @constraint(model, satisfied[j] + unsatisfied[j] == prob.demands[j])
            @constraint(model, satisfied[j] <= prob.demands[j])
        end

        # Service level constraint
        total_demand = sum(prob.demands)
        @constraint(model, sum(satisfied[j] for j in 1:prob.n_destinations) >= prob.service_level_min * total_demand)

    elseif prob.variant == transport_emission_constrained
        @variable(model, x[1:prob.n_sources, 1:prob.n_destinations] >= 0)

        @objective(model, Min, sum(prob.costs[i, j] * x[i, j]
                                   for i in 1:prob.n_sources, j in 1:prob.n_destinations))

        for i in 1:prob.n_sources
            @constraint(model, sum(x[i, j] for j in 1:prob.n_destinations) <= prob.supplies[i])
        end

        for j in 1:prob.n_destinations
            @constraint(model, sum(x[i, j] for i in 1:prob.n_sources) >= prob.demands[j])
        end

        # Emission constraint
        @constraint(model, sum(prob.emission_rates[i, j] * x[i, j]
                              for i in 1:prob.n_sources, j in 1:prob.n_destinations) <= prob.emission_limit)
    end

    return model
end

# Register the problem type
register_problem(
    :transportation,
    TransportationProblem,
    "Transportation problem with multiple variants including standard, balanced, capacitated, multi-commodity, transshipment, time windows, service level, and emission-constrained formulations"
)
