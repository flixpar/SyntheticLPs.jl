using JuMP
using Random
using StatsBase
using Distributions

"""
    EnergyProblem <: ProblemGenerator

Generator for energy generation mix optimization problems.

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
end

"""
    EnergyProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an energy generation mix problem instance.

# Arguments
- `target_variables`: Target number of variables (n_sources × n_periods)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function EnergyProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
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

    # Start with reasonable defaults
    n_sources = min_sources + 2
    n_periods = min_periods + 12

    # Iteratively adjust to reach target
    for _ in 1:15
        current_vars = n_sources * n_periods

        if abs(current_vars - target_variables) / target_variables < 0.1
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

    # Sample parameters based on scale
    if scale == :small
        renewable_fraction_target = rand(Beta(2, 3))
        demand_variation = rand(Beta(2, 3))
        peak_demand = rand(Uniform(peak_demand_range...))
        base_generation_cost = rand(LogNormal(log(60.0), 0.4))
        renewable_cost_factor = rand(Gamma(3, 0.4))
        capacity_margin = max(1.15, min(1.6, rand(Normal(1.35, 0.08))))
        emission_limit = rand(Beta(2, 2))
    elseif scale == :medium
        renewable_fraction_target = rand(Beta(3, 4))
        demand_variation = rand(Beta(3, 5))
        peak_demand = rand(Uniform(peak_demand_range...))
        base_generation_cost = rand(LogNormal(log(45.0), 0.3))
        renewable_cost_factor = rand(Gamma(2.5, 0.35))
        capacity_margin = max(1.1, min(1.5, rand(Normal(1.25, 0.05))))
        emission_limit = rand(Beta(3, 3))
    else
        renewable_fraction_target = rand(Beta(4, 5))
        demand_variation = rand(Beta(4, 8))
        peak_demand = rand(Uniform(peak_demand_range...))
        base_generation_cost = rand(LogNormal(log(35.0), 0.25))
        renewable_cost_factor = rand(Gamma(2, 0.3))
        capacity_margin = max(1.05, min(1.3, rand(Normal(1.15, 0.04))))
        emission_limit = rand(Beta(4, 6))
    end

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

        variation = if name in ["coal", "gas"]
            rand(LogNormal(log(1.0), 0.15))
        elseif name == "nuclear"
            rand(Normal(1.0, 0.08))
        elseif name in ["solar", "wind"]
            rand(Gamma(8, 0.12))
        else
            rand(Normal(1.0, 0.12))
        end

        generation_costs[name] = base_cost * max(0.3, variation)
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
        elseif name == "solar"
            rand(Beta(2, 4))
        elseif name == "wind"
            rand(Beta(3, 3))
        elseif name == "hydro"
            rand(LogNormal(log(0.3), 0.6))
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

    residential_pattern = [0.6, 0.55, 0.5, 0.5, 0.55, 0.7, 0.85, 1.0, 0.95, 0.9, 0.85, 0.9, 0.95, 1.0, 0.9, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.8, 0.7, 0.65]
    commercial_pattern = [0.4, 0.35, 0.3, 0.3, 0.35, 0.5, 0.7, 0.9, 1.0, 1.0, 0.95, 0.9, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.75, 0.6, 0.5, 0.45, 0.4, 0.35]
    industrial_pattern = [0.8, 0.75, 0.7, 0.7, 0.75, 0.85, 0.95, 1.0, 1.0, 0.95, 0.9, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.7, 0.75, 0.8]

    hour_factors = if peak_demand < 100
        residential_pattern
    elseif peak_demand < 1000
        commercial_pattern
    else
        industrial_pattern
    end

    base_demand = peak_demand * (1 - demand_variation)

    if n_periods == 24
        for h in 1:24
            pattern_demand = base_demand + (peak_demand - base_demand) * hour_factors[h]

            weather_effect = rand(Normal(1.0, 0.03))
            economic_effect = rand(Normal(1.0, 0.02))
            random_effect = rand(Normal(1.0, 0.025))
            seasonal_effect = rand(Normal(1.0, 0.01))

            total_effect = weather_effect * economic_effect * random_effect * seasonal_effect
            demand = pattern_demand * max(0.7, min(1.4, total_effect))

            push!(demands, demand)
        end
    else
        for p in 1:n_periods
            relative_hour = (p - 1) * 24 / n_periods
            hour_idx = 1 + floor(Int, relative_hour % 24)
            if hour_idx > 24
                hour_idx = 24
            end

            pattern_demand = base_demand + (peak_demand - base_demand) * hour_factors[hour_idx]

            variability_scale = sqrt(24 / n_periods)

            weather_effect = rand(Normal(1.0, 0.03 * variability_scale))
            economic_effect = rand(Normal(1.0, 0.02 * variability_scale))
            random_effect = rand(Normal(1.0, 0.025 * variability_scale))
            seasonal_effect = rand(Normal(1.0, 0.01 * variability_scale))

            total_effect = weather_effect * economic_effect * random_effect * seasonal_effect
            demand = pattern_demand * max(0.7, min(1.4, total_effect))

            push!(demands, demand)
        end
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

    # Adjust for feasibility
    solution_status = feasibility_status == feasible ? :feasible :
                     feasibility_status == infeasible ? :infeasible : :all
    actual_status = solution_status
    if solution_status == :all
        actual_status = rand() < 0.7 ? :feasible : :infeasible
    end

    renewable_fraction = renewable_fraction_target

    if actual_status == :infeasible
        scenario = rand(1:4)

        if scenario == 1
            # Capacity crisis
            reduction_factor = 0.6 + rand() * 0.2
            target_total_capacity = peak_demand * reduction_factor
            current_total_capacity = sum(values(capacities))
            capacity_scale = target_total_capacity / current_total_capacity

            for source in sources
                capacities[source] *= capacity_scale
            end

        elseif scenario == 2
            # Renewable intermittency
            renewable_fraction = 0.7 + rand() * 0.2

            renewable_sources_list = [s for s in sources if emission_limits[s] == 0.0]
            total_renewable_capacity = sum(capacities[s] for s in renewable_sources_list)

            required_renewable_capacity = peak_demand * renewable_fraction
            target_renewable_capacity = required_renewable_capacity * (0.4 + rand() * 0.2)

            if total_renewable_capacity > 0
                renewable_scale = target_renewable_capacity / total_renewable_capacity
                for source in renewable_sources_list
                    capacities[source] *= renewable_scale
                end
            end

        elseif scenario == 3
            # Emission impossibility
            new_emission_limit = emission_limit * (0.01 + rand() * 0.05)

            for (name, is_renewable, _, _, _) in selected_sources
                if !is_renewable && name != "nuclear"
                    emission_limits[name] = new_emission_limit
                end
            end

            clean_sources = [s for s in sources if emission_limits[s] == 0.0]
            total_clean_capacity = sum(capacities[s] for s in clean_sources)

            target_clean_capacity = peak_demand * (0.5 + rand() * 0.2)

            if total_clean_capacity > target_clean_capacity
                clean_scale = target_clean_capacity / total_clean_capacity
                for source in clean_sources
                    capacities[source] *= clean_scale
                end
            end

        else  # scenario == 4
            # Demand surge
            current_total_capacity = sum(values(capacities))

            surge_factor = (current_total_capacity * (1.1 + rand() * 0.2)) / peak_demand

            for i in 1:length(demands)
                demands[i] *= surge_factor
            end

            peak_demand *= surge_factor
        end
    else
        # Feasible: ensure capacity
        total_capacity = sum(values(capacities))
        required_capacity = peak_demand * capacity_margin

        if total_capacity < required_capacity
            scale_factor = required_capacity / total_capacity
            for source in sources
                capacities[source] *= scale_factor
            end
        end
    end

    return EnergyProblem(n_sources, n_periods, sources, time_periods, generation_costs, capacities,
                        demands, emission_limits, renewable_fraction)
end

"""
    build_model(prob::EnergyProblem)

Build a JuMP model for the energy generation mix problem.

# Arguments
- `prob`: EnergyProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::EnergyProblem)
    model = Model()

    # Variables
    @variable(model, 0 <= x[s in prob.sources, t in prob.time_periods] <= prob.capacities[s])

    # Objective
    @objective(model, Min,
        sum(prob.generation_costs[s] * x[s,t] for s in prob.sources, t in prob.time_periods)
    )

    # Meet demand
    for t in prob.time_periods
        @constraint(model, sum(x[s,t] for s in prob.sources) >= prob.demands[t])
    end

    # Emissions
    max_emission = maximum(values(prob.emission_limits))
    for t in prob.time_periods
        @constraint(model,
            sum(prob.emission_limits[s] * x[s,t] for s in prob.sources) <=
            sum(x[s,t] for s in prob.sources) * max_emission
        )
    end

    # Renewables
    renewable_sources = [s for s in prob.sources if prob.emission_limits[s] == 0.0]
    for t in prob.time_periods
        @constraint(model,
            sum(x[s,t] for s in renewable_sources) >=
            prob.renewable_fraction * sum(x[s,t] for s in prob.sources)
        )
    end

    return model
end

# Register the problem type
register_problem(
    :energy,
    EnergyProblem,
    "Energy generation mix problem that optimizes the allocation of different energy sources to meet demand while minimizing costs and emissions"
)
