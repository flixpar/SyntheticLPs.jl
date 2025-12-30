using JuMP
using Random
using StatsBase
using Distributions

"""
Energy problem variants.

# Variants
- `energy_standard`: Basic energy generation mix optimization
- `energy_ramping`: Include ramping constraints between periods
- `energy_reserves`: Include spinning/non-spinning reserve requirements
- `energy_storage`: Include battery storage with charge/discharge
- `energy_unit_commit`: Include startup/shutdown costs (unit commitment)
- `energy_min_emissions`: Minimize emissions as primary objective
- `energy_curtailment`: Allow renewable curtailment with penalty
- `energy_transmission`: Multi-zone with transmission constraints
"""
@enum EnergyVariant begin
    energy_standard
    energy_ramping
    energy_reserves
    energy_storage
    energy_unit_commit
    energy_min_emissions
    energy_curtailment
    energy_transmission
end

"""
    EnergyProblem <: ProblemGenerator

Generator for energy generation mix optimization problems with multiple variants.

# Fields
- `n_sources::Int`: Number of power generation sources
- `n_periods::Int`: Number of time periods
- `sources::Vector{String}`: Names of energy sources
- `time_periods::Vector{Int}`: Time period indices
- `generation_costs::Dict{String,Float64}`: Cost per MWh for each source
- `capacities::Dict{String,Float64}`: Maximum capacity (MW) for each source
- `demands::Vector{Float64}`: Demand (MW) in each period
- `emission_limits::Dict{String,Float64}`: Emission rate per MWh for each source
- `renewable_fraction::Float64`: Minimum fraction of generation from renewables
- `variant::EnergyVariant`: The specific variant type
# Ramping variant
- `ramp_up_limits::Union{Dict{String,Float64}, Nothing}`: Maximum ramp up per period
- `ramp_down_limits::Union{Dict{String,Float64}, Nothing}`: Maximum ramp down per period
# Reserves variant
- `spinning_reserve_req::Union{Vector{Float64}, Nothing}`: Spinning reserve per period
- `non_spinning_reserve_req::Union{Vector{Float64}, Nothing}`: Non-spinning reserve per period
# Storage variant
- `storage_capacity::Union{Float64, Nothing}`: Maximum battery storage (MWh)
- `storage_power::Union{Float64, Nothing}`: Maximum charge/discharge rate (MW)
- `storage_efficiency::Union{Float64, Nothing}`: Round-trip efficiency
# Unit commitment variant
- `startup_costs::Union{Dict{String,Float64}, Nothing}`: Cost to start each source
- `shutdown_costs::Union{Dict{String,Float64}, Nothing}`: Cost to shut down each source
- `min_up_time::Union{Dict{String,Int}, Nothing}`: Minimum periods after startup
- `min_down_time::Union{Dict{String,Int}, Nothing}`: Minimum periods after shutdown
# Curtailment variant
- `curtailment_penalty::Union{Float64, Nothing}`: Cost per MW curtailed
- `max_curtailment_fraction::Union{Float64, Nothing}`: Maximum fraction that can be curtailed
# Transmission variant
- `n_zones::Int`: Number of transmission zones
- `zone_demands::Union{Matrix{Float64}, Nothing}`: Demand per zone per period
- `zone_sources::Union{Dict{String,Int}, Nothing}`: Which zone each source is in
- `transmission_capacity::Union{Matrix{Float64}, Nothing}`: Capacity between zones
- `transmission_loss::Union{Float64, Nothing}`: Transmission loss factor
"""
struct EnergyProblem <: ProblemGenerator
    n_sources::Int
    n_periods::Int
    sources::Vector{String}
    time_periods::Vector{Int}
    generation_costs::Dict{String,Float64}
    capacities::Dict{String,Float64}
    demands::Vector{Float64}
    emission_limits::Dict{String,Float64}
    renewable_fraction::Float64
    variant::EnergyVariant
    # Ramping variant
    ramp_up_limits::Union{Dict{String,Float64}, Nothing}
    ramp_down_limits::Union{Dict{String,Float64}, Nothing}
    # Reserves variant
    spinning_reserve_req::Union{Vector{Float64}, Nothing}
    non_spinning_reserve_req::Union{Vector{Float64}, Nothing}
    # Storage variant
    storage_capacity::Union{Float64, Nothing}
    storage_power::Union{Float64, Nothing}
    storage_efficiency::Union{Float64, Nothing}
    # Unit commitment variant
    startup_costs::Union{Dict{String,Float64}, Nothing}
    shutdown_costs::Union{Dict{String,Float64}, Nothing}
    min_up_time::Union{Dict{String,Int}, Nothing}
    min_down_time::Union{Dict{String,Int}, Nothing}
    # Curtailment variant
    curtailment_penalty::Union{Float64, Nothing}
    max_curtailment_fraction::Union{Float64, Nothing}
    # Transmission variant
    n_zones::Int
    zone_demands::Union{Matrix{Float64}, Nothing}
    zone_sources::Union{Dict{String,Int}, Nothing}
    transmission_capacity::Union{Matrix{Float64}, Nothing}
    transmission_loss::Union{Float64, Nothing}
end

# Backwards compatibility
function EnergyProblem(n_sources::Int, n_periods::Int, sources::Vector{String},
                       time_periods::Vector{Int}, generation_costs::Dict{String,Float64},
                       capacities::Dict{String,Float64}, demands::Vector{Float64},
                       emission_limits::Dict{String,Float64}, renewable_fraction::Float64)
    EnergyProblem(
        n_sources, n_periods, sources, time_periods, generation_costs,
        capacities, demands, emission_limits, renewable_fraction, energy_standard,
        nothing, nothing, nothing, nothing,
        nothing, nothing, nothing,
        nothing, nothing, nothing, nothing,
        nothing, nothing,
        1, nothing, nothing, nothing, nothing
    )
end

"""
    EnergyProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                  variant::EnergyVariant=energy_standard)

Construct an energy generation mix problem instance with the specified variant.
"""
function EnergyProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
                       variant::EnergyVariant=energy_standard)
    Random.seed!(seed)

    # Determine scale
    if target_variables < 250
        scale = :small
        min_sources, max_sources = 3, 8
        min_periods, max_periods = 12, 48
        peak_demand_range = (10.0, 100.0)
    elseif target_variables < 1000
        scale = :medium
        min_sources, max_sources = 5, 12
        min_periods, max_periods = 24, 72
        peak_demand_range = (100.0, 1000.0)
    else
        scale = :large
        min_sources, max_sources = 8, 20
        min_periods, max_periods = 48, 200
        peak_demand_range = (1000.0, 10000.0)
    end

    # Adjust for variant-specific variable counts
    if variant == energy_transmission
        n_zones = rand(2:min(5, max(2, target_variables ÷ 100)))
        n_sources = min_sources + 1
        n_periods = max(min_periods, target_variables ÷ (n_sources + n_zones * n_zones))
    elseif variant == energy_storage
        n_sources = min_sources + 1
        n_periods = max(min_periods, target_variables ÷ (n_sources + 3))  # Extra for storage
    else
        n_sources = min_sources + 2
        n_periods = min_periods + 12
    end

    # Iteratively adjust to reach target
    for _ in 1:15
        current_vars = n_sources * n_periods
        if abs(current_vars - target_variables) / target_variables < 0.15
            break
        end

        ratio = target_variables / current_vars
        if ratio > 1.1
            if n_periods < max_periods
                n_periods = min(max_periods, round(Int, n_periods * sqrt(ratio)))
            elseif n_sources < max_sources
                n_sources = min(max_sources, round(Int, n_sources * sqrt(ratio)))
            end
        elseif ratio < 0.9
            if n_periods > min_periods
                n_periods = max(min_periods, round(Int, n_periods * sqrt(ratio)))
            elseif n_sources > min_sources
                n_sources = max(min_sources, round(Int, n_sources * sqrt(ratio)))
            end
        end
    end

    # Sample parameters
    renewable_fraction_target = rand(Beta(2, 3))
    demand_variation = rand(Beta(2, 3))
    peak_demand = rand(Uniform(peak_demand_range...))
    base_generation_cost = rand(LogNormal(log(50.0), 0.3))
    renewable_cost_factor = rand(Gamma(2.5, 0.4))
    capacity_margin = max(1.15, min(1.6, rand(Normal(1.3, 0.08))))
    emission_limit = rand(Beta(2, 2))

    # Source types
    source_types = [
        ("coal", false, 0.95, 0.9, 1.0),
        ("gas", false, 0.98, 0.85, 1.2),
        ("nuclear", false, 0.92, 0.95, 0.8),
        ("solar", true, 0.99, 0.25, 0.3),
        ("wind", true, 0.95, 0.35, 0.4),
        ("hydro", true, 0.90, 0.50, 0.6),
        ("biomass", true, 0.88, 0.75, 1.1)
    ]

    # Select sources
    n_renewables = max(1, ceil(Int, n_sources * renewable_fraction_target))
    n_conventional = n_sources - n_renewables

    renewable_indices = findall(s -> s[2], source_types)
    conventional_indices = findall(s -> !s[2], source_types)

    n_renewables = min(n_renewables, length(renewable_indices))
    n_conventional = min(n_conventional, length(conventional_indices))

    renewable_sources = source_types[sample(renewable_indices, n_renewables, replace=false)]
    conventional_sources = source_types[sample(conventional_indices, n_conventional, replace=false)]
    selected_sources = vcat(renewable_sources, conventional_sources)

    sources = [s[1] for s in selected_sources]
    time_periods = collect(1:n_periods)

    # Generate costs
    generation_costs = Dict{String, Float64}()
    for (name, is_renewable, _, _, cost_factor) in selected_sources
        base_cost = base_generation_cost * cost_factor
        if is_renewable
            base_cost *= renewable_cost_factor
        end
        variation = rand(Normal(1.0, 0.12))
        generation_costs[name] = max(5.0, base_cost * variation)
    end

    # Generate capacities
    capacities = Dict{String, Float64}()
    total_required_capacity = peak_demand * capacity_margin

    capacity_shares = Float64[]
    for (name, is_renewable, availability, capacity_factor, _) in selected_sources
        share = if name == "coal"
            rand(Gamma(2, 0.25))
        elseif name == "gas"
            rand(Gamma(3, 0.15))
        elseif name == "nuclear"
            rand(Gamma(1.5, 0.4))
        elseif name in ["solar", "wind"]
            rand(Beta(2, 4))
        else
            rand(Beta(2, 5))
        end
        push!(capacity_shares, share)
    end

    total_share = sum(capacity_shares)
    for (i, (name, _, availability, capacity_factor, _)) in enumerate(selected_sources)
        normalized_share = capacity_shares[i] / total_share
        effective_capacity = total_required_capacity * normalized_share / (availability * capacity_factor)
        capacities[name] = max(10.0, effective_capacity)
    end

    # Generate demands
    demands = Float64[]
    hour_factors = [0.6, 0.55, 0.5, 0.5, 0.55, 0.7, 0.85, 1.0, 0.95, 0.9,
                    0.85, 0.9, 0.95, 1.0, 0.9, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.8, 0.7, 0.65]
    base_demand = peak_demand * (1 - demand_variation)

    for p in 1:n_periods
        hour_idx = 1 + (p - 1) % 24
        pattern_demand = base_demand + (peak_demand - base_demand) * hour_factors[hour_idx]
        noise = rand(Normal(1.0, 0.05))
        push!(demands, pattern_demand * max(0.7, min(1.3, noise)))
    end

    # Generate emission limits
    emission_limits = Dict{String, Float64}()
    for (name, is_renewable, _, _, _) in selected_sources
        if is_renewable
            emission_limits[name] = 0.0
        else
            if name == "coal"
                emission_limits[name] = emission_limit
            elseif name == "gas"
                emission_limits[name] = emission_limit * 0.5
            else
                emission_limits[name] = 0.0
            end
        end
    end

    renewable_fraction = renewable_fraction_target

    # Initialize variant-specific fields
    ramp_up_limits = nothing
    ramp_down_limits = nothing
    spinning_reserve_req = nothing
    non_spinning_reserve_req = nothing
    storage_capacity = nothing
    storage_power = nothing
    storage_efficiency = nothing
    startup_costs = nothing
    shutdown_costs = nothing
    min_up_time = nothing
    min_down_time = nothing
    curtailment_penalty = nothing
    max_curtailment_fraction = nothing
    n_zones = 1
    zone_demands = nothing
    zone_sources = nothing
    transmission_capacity = nothing
    transmission_loss = nothing

    # Generate variant-specific data
    if variant == energy_ramping
        ramp_up_limits = Dict{String, Float64}()
        ramp_down_limits = Dict{String, Float64}()
        for (name, is_renewable, _, _, _) in selected_sources
            cap = capacities[name]
            if is_renewable
                # Renewables have high ramp rates (can change quickly)
                ramp_up_limits[name] = cap * rand(Uniform(0.8, 1.0))
                ramp_down_limits[name] = cap * rand(Uniform(0.8, 1.0))
            elseif name == "nuclear"
                # Nuclear has slow ramp rates
                ramp_up_limits[name] = cap * rand(Uniform(0.02, 0.05))
                ramp_down_limits[name] = cap * rand(Uniform(0.02, 0.05))
            elseif name == "coal"
                # Coal is moderate
                ramp_up_limits[name] = cap * rand(Uniform(0.1, 0.2))
                ramp_down_limits[name] = cap * rand(Uniform(0.1, 0.2))
            else  # gas
                # Gas is flexible
                ramp_up_limits[name] = cap * rand(Uniform(0.3, 0.6))
                ramp_down_limits[name] = cap * rand(Uniform(0.3, 0.6))
            end
        end

    elseif variant == energy_reserves
        # Reserve requirements as fraction of demand
        reserve_fraction = rand(Uniform(0.05, 0.15))
        spinning_reserve_req = demands .* reserve_fraction
        non_spinning_reserve_req = demands .* reserve_fraction .* rand(Uniform(0.5, 1.0))

    elseif variant == energy_storage
        # Battery storage parameters
        storage_capacity = peak_demand * rand(Uniform(0.5, 2.0))  # MWh
        storage_power = storage_capacity / rand(Uniform(2.0, 6.0))  # MW (2-6 hour duration)
        storage_efficiency = rand(Uniform(0.85, 0.95))

    elseif variant == energy_unit_commit
        startup_costs = Dict{String, Float64}()
        shutdown_costs = Dict{String, Float64}()
        min_up_time = Dict{String, Int}()
        min_down_time = Dict{String, Int}()

        for (name, is_renewable, _, _, _) in selected_sources
            if is_renewable
                # Renewables have no startup costs
                startup_costs[name] = 0.0
                shutdown_costs[name] = 0.0
                min_up_time[name] = 0
                min_down_time[name] = 0
            elseif name == "nuclear"
                startup_costs[name] = generation_costs[name] * rand(Uniform(100, 200))
                shutdown_costs[name] = generation_costs[name] * rand(Uniform(50, 100))
                min_up_time[name] = rand(12:min(24, n_periods ÷ 2))
                min_down_time[name] = rand(8:min(16, n_periods ÷ 3))
            elseif name == "coal"
                startup_costs[name] = generation_costs[name] * rand(Uniform(20, 50))
                shutdown_costs[name] = generation_costs[name] * rand(Uniform(10, 25))
                min_up_time[name] = rand(4:min(8, n_periods ÷ 4))
                min_down_time[name] = rand(4:min(8, n_periods ÷ 4))
            else  # gas
                startup_costs[name] = generation_costs[name] * rand(Uniform(5, 15))
                shutdown_costs[name] = generation_costs[name] * rand(Uniform(2, 8))
                min_up_time[name] = rand(1:min(3, n_periods ÷ 6))
                min_down_time[name] = rand(1:min(2, n_periods ÷ 8))
            end
        end

    elseif variant == energy_min_emissions
        # Emissions objective - adjust emission weights
        # Uses same data, just different objective

    elseif variant == energy_curtailment
        # Renewable curtailment parameters
        curtailment_penalty = mean(values(generation_costs)) * rand(Uniform(0.1, 0.5))
        max_curtailment_fraction = rand(Uniform(0.1, 0.3))

    elseif variant == energy_transmission
        n_zones = rand(2:min(5, max(2, target_variables ÷ 100)))

        # Distribute sources to zones
        zone_sources = Dict{String, Int}()
        for (i, source) in enumerate(sources)
            zone_sources[source] = ((i - 1) % n_zones) + 1
        end

        # Zone demands (distribute total demand)
        zone_demands = zeros(n_zones, n_periods)
        zone_weights = rand(Dirichlet(ones(n_zones)))
        for z in 1:n_zones
            for t in 1:n_periods
                zone_demands[z, t] = demands[t] * zone_weights[z] * rand(Uniform(0.9, 1.1))
            end
        end

        # Transmission capacity between zones
        transmission_capacity = zeros(n_zones, n_zones)
        avg_zone_demand = sum(demands) / n_periods / n_zones
        for i in 1:n_zones, j in 1:n_zones
            if i != j
                transmission_capacity[i, j] = avg_zone_demand * rand(Uniform(0.3, 0.8))
            end
        end

        transmission_loss = rand(Uniform(0.02, 0.05))  # 2-5% loss
    end

    # Handle feasibility
    if feasibility_status == feasible
        # Ensure sufficient capacity
        total_capacity = sum(values(capacities))
        max_demand = maximum(demands)
        if total_capacity < max_demand * 1.2
            scale_factor = max_demand * 1.3 / total_capacity
            for source in sources
                capacities[source] *= scale_factor
            end
        end

        # For reserves, ensure capacity covers reserves too
        if variant == energy_reserves && spinning_reserve_req !== nothing
            max_reserve = maximum(spinning_reserve_req) + maximum(non_spinning_reserve_req)
            if total_capacity < maximum(demands) + max_reserve
                scale_factor = (maximum(demands) + max_reserve * 1.2) / total_capacity
                for source in sources
                    capacities[source] *= scale_factor
                end
            end
        end

    elseif feasibility_status == infeasible
        # Create infeasibility
        scenario = rand(1:3)

        if scenario == 1
            # Capacity shortage
            for source in sources
                capacities[source] *= 0.5
            end
        elseif scenario == 2
            # Impossible renewable requirement
            renewable_fraction = 0.95
            renewable_sources_list = [s for s in sources if emission_limits[s] == 0.0]
            for source in renewable_sources_list
                capacities[source] *= 0.3
            end
        else
            # Transmission bottleneck (for transmission variant)
            if variant == energy_transmission && transmission_capacity !== nothing
                transmission_capacity .*= 0.1
            else
                # Otherwise use demand surge
                demands .*= 3.0
            end
        end
    end

    return EnergyProblem(
        n_sources, n_periods, sources, time_periods, generation_costs,
        capacities, demands, emission_limits, renewable_fraction, variant,
        ramp_up_limits, ramp_down_limits,
        spinning_reserve_req, non_spinning_reserve_req,
        storage_capacity, storage_power, storage_efficiency,
        startup_costs, shutdown_costs, min_up_time, min_down_time,
        curtailment_penalty, max_curtailment_fraction,
        n_zones, zone_demands, zone_sources, transmission_capacity, transmission_loss
    )
end

"""
    build_model(prob::EnergyProblem)

Build a JuMP model for the energy generation mix problem based on its variant.
"""
function build_model(prob::EnergyProblem)
    model = Model()

    if prob.variant == energy_standard
        @variable(model, 0 <= x[s in prob.sources, t in prob.time_periods] <= prob.capacities[s])

        @objective(model, Min,
            sum(prob.generation_costs[s] * x[s,t] for s in prob.sources, t in prob.time_periods))

        # Meet demand
        for t in prob.time_periods
            @constraint(model, sum(x[s,t] for s in prob.sources) >= prob.demands[t])
        end

        # Emissions constraint
        max_emission = maximum(values(prob.emission_limits))
        for t in prob.time_periods
            @constraint(model,
                sum(prob.emission_limits[s] * x[s,t] for s in prob.sources) <=
                sum(x[s,t] for s in prob.sources) * max_emission)
        end

        # Renewable fraction
        renewable_sources = [s for s in prob.sources if prob.emission_limits[s] == 0.0]
        for t in prob.time_periods
            @constraint(model,
                sum(x[s,t] for s in renewable_sources) >=
                prob.renewable_fraction * sum(x[s,t] for s in prob.sources))
        end

    elseif prob.variant == energy_ramping
        @variable(model, 0 <= x[s in prob.sources, t in prob.time_periods] <= prob.capacities[s])

        @objective(model, Min,
            sum(prob.generation_costs[s] * x[s,t] for s in prob.sources, t in prob.time_periods))

        for t in prob.time_periods
            @constraint(model, sum(x[s,t] for s in prob.sources) >= prob.demands[t])
        end

        # Ramping constraints
        for s in prob.sources, t in prob.time_periods[2:end]
            @constraint(model, x[s,t] - x[s,t-1] <= prob.ramp_up_limits[s])
            @constraint(model, x[s,t-1] - x[s,t] <= prob.ramp_down_limits[s])
        end

        # Emissions and renewables
        max_emission = maximum(values(prob.emission_limits))
        renewable_sources = [s for s in prob.sources if prob.emission_limits[s] == 0.0]
        for t in prob.time_periods
            @constraint(model,
                sum(prob.emission_limits[s] * x[s,t] for s in prob.sources) <=
                sum(x[s,t] for s in prob.sources) * max_emission)
            @constraint(model,
                sum(x[s,t] for s in renewable_sources) >=
                prob.renewable_fraction * sum(x[s,t] for s in prob.sources))
        end

    elseif prob.variant == energy_reserves
        @variable(model, 0 <= x[s in prob.sources, t in prob.time_periods] <= prob.capacities[s])
        @variable(model, spin_res[s in prob.sources, t in prob.time_periods] >= 0)  # Spinning reserve contribution
        @variable(model, nonspin_res[s in prob.sources, t in prob.time_periods] >= 0)  # Non-spinning reserve

        @objective(model, Min,
            sum(prob.generation_costs[s] * x[s,t] for s in prob.sources, t in prob.time_periods))

        for t in prob.time_periods
            # Meet demand
            @constraint(model, sum(x[s,t] for s in prob.sources) >= prob.demands[t])

            # Meet spinning reserve requirement
            @constraint(model, sum(spin_res[s,t] for s in prob.sources) >= prob.spinning_reserve_req[t])

            # Meet non-spinning reserve requirement
            @constraint(model, sum(nonspin_res[s,t] for s in prob.sources) >= prob.non_spinning_reserve_req[t])
        end

        # Reserve capacity limits (reserve + generation <= capacity)
        for s in prob.sources, t in prob.time_periods
            @constraint(model, x[s,t] + spin_res[s,t] <= prob.capacities[s])
            @constraint(model, nonspin_res[s,t] <= prob.capacities[s] - x[s,t])
        end

    elseif prob.variant == energy_storage
        @variable(model, 0 <= x[s in prob.sources, t in prob.time_periods] <= prob.capacities[s])
        @variable(model, 0 <= storage_level[t in 0:prob.n_periods] <= prob.storage_capacity)
        @variable(model, 0 <= charge[t in prob.time_periods] <= prob.storage_power)
        @variable(model, 0 <= discharge[t in prob.time_periods] <= prob.storage_power)

        @objective(model, Min,
            sum(prob.generation_costs[s] * x[s,t] for s in prob.sources, t in prob.time_periods))

        @constraint(model, storage_level[0] == prob.storage_capacity / 2)  # Start half-charged

        for t in prob.time_periods
            # Meet demand (generation + discharge - charge)
            @constraint(model,
                sum(x[s,t] for s in prob.sources) + discharge[t] - charge[t] >= prob.demands[t])

            # Storage balance
            @constraint(model,
                storage_level[t] == storage_level[t-1] +
                prob.storage_efficiency * charge[t] - discharge[t])
        end

        # Emissions and renewables
        renewable_sources = [s for s in prob.sources if prob.emission_limits[s] == 0.0]
        for t in prob.time_periods
            @constraint(model,
                sum(x[s,t] for s in renewable_sources) >=
                prob.renewable_fraction * sum(x[s,t] for s in prob.sources))
        end

    elseif prob.variant == energy_unit_commit
        @variable(model, 0 <= x[s in prob.sources, t in prob.time_periods] <= prob.capacities[s])
        @variable(model, on[s in prob.sources, t in prob.time_periods], Bin)  # Unit on/off
        @variable(model, startup[s in prob.sources, t in prob.time_periods], Bin)
        @variable(model, shutdown[s in prob.sources, t in prob.time_periods], Bin)

        # Minimize generation + startup + shutdown costs
        @objective(model, Min,
            sum(prob.generation_costs[s] * x[s,t] for s in prob.sources, t in prob.time_periods) +
            sum(prob.startup_costs[s] * startup[s,t] for s in prob.sources, t in prob.time_periods) +
            sum(prob.shutdown_costs[s] * shutdown[s,t] for s in prob.sources, t in prob.time_periods))

        for t in prob.time_periods
            @constraint(model, sum(x[s,t] for s in prob.sources) >= prob.demands[t])
        end

        # Linking constraints
        for s in prob.sources, t in prob.time_periods
            # Can only generate if on
            @constraint(model, x[s,t] <= prob.capacities[s] * on[s,t])

            # Startup/shutdown logic
            if t > 1
                @constraint(model, startup[s,t] >= on[s,t] - on[s,t-1])
                @constraint(model, shutdown[s,t] >= on[s,t-1] - on[s,t])
            end
        end

        # Minimum up/down time constraints (simplified)
        for s in prob.sources
            for t in prob.time_periods
                if t >= prob.min_up_time[s] && prob.min_up_time[s] > 0
                    @constraint(model,
                        sum(startup[s, tau] for tau in (t - prob.min_up_time[s] + 1):t) <= on[s,t])
                end
            end
        end

    elseif prob.variant == energy_min_emissions
        @variable(model, 0 <= x[s in prob.sources, t in prob.time_periods] <= prob.capacities[s])

        # Minimize emissions instead of cost
        @objective(model, Min,
            sum(prob.emission_limits[s] * x[s,t] for s in prob.sources, t in prob.time_periods))

        for t in prob.time_periods
            @constraint(model, sum(x[s,t] for s in prob.sources) >= prob.demands[t])
        end

        # Renewable fraction still applies
        renewable_sources = [s for s in prob.sources if prob.emission_limits[s] == 0.0]
        for t in prob.time_periods
            @constraint(model,
                sum(x[s,t] for s in renewable_sources) >=
                prob.renewable_fraction * sum(x[s,t] for s in prob.sources))
        end

    elseif prob.variant == energy_curtailment
        @variable(model, 0 <= x[s in prob.sources, t in prob.time_periods] <= prob.capacities[s])
        renewable_sources = [s for s in prob.sources if prob.emission_limits[s] == 0.0]
        @variable(model, curtailed[s in renewable_sources, t in prob.time_periods] >= 0)

        # Minimize cost + curtailment penalty
        @objective(model, Min,
            sum(prob.generation_costs[s] * x[s,t] for s in prob.sources, t in prob.time_periods) +
            prob.curtailment_penalty * sum(curtailed[s,t] for s in renewable_sources, t in prob.time_periods))

        for t in prob.time_periods
            @constraint(model, sum(x[s,t] for s in prob.sources) >= prob.demands[t])
        end

        # Curtailment limited by capacity and max fraction
        for s in renewable_sources, t in prob.time_periods
            @constraint(model, x[s,t] + curtailed[s,t] <= prob.capacities[s])
            @constraint(model, curtailed[s,t] <= prob.max_curtailment_fraction * prob.capacities[s])
        end

    elseif prob.variant == energy_transmission
        @variable(model, 0 <= x[s in prob.sources, t in prob.time_periods] <= prob.capacities[s])
        @variable(model, flow[i in 1:prob.n_zones, j in 1:prob.n_zones, t in prob.time_periods] >= 0)

        @objective(model, Min,
            sum(prob.generation_costs[s] * x[s,t] for s in prob.sources, t in prob.time_periods))

        for t in prob.time_periods
            # Zonal balance: generation + inflow - outflow = demand
            for z in 1:prob.n_zones
                zone_gen = sum(x[s,t] for s in prob.sources if prob.zone_sources[s] == z; init=0.0)
                inflow = sum((1 - prob.transmission_loss) * flow[i, z, t]
                             for i in 1:prob.n_zones if i != z)
                outflow = sum(flow[z, j, t] for j in 1:prob.n_zones if j != z)
                @constraint(model, zone_gen + inflow - outflow >= prob.zone_demands[z, t])
            end
        end

        # Transmission capacity limits
        for i in 1:prob.n_zones, j in 1:prob.n_zones, t in prob.time_periods
            if i != j
                @constraint(model, flow[i, j, t] <= prob.transmission_capacity[i, j])
            else
                @constraint(model, flow[i, j, t] == 0)
            end
        end
    end

    return model
end

# Register the problem type
register_problem(
    :energy,
    EnergyProblem,
    "Energy generation mix problem with variants including standard, ramping, reserves, storage, unit commitment, min emissions, curtailment, and transmission"
)
