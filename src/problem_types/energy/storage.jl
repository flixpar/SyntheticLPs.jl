using JuMP
using Random
using StatsBase
using Distributions

"""
    StorageEnergyProblem <: ProblemGenerator

Generator for energy generation mix optimization with battery storage.

# Overview
Models a power dispatch problem over a sequence of time periods where, in addition
to dispatchable generation sources, a single battery can charge, discharge, and
carry a state of charge between periods. The objective minimizes total generation
cost. Demand in each period must be met by generation plus net discharge, the
battery state of charge follows a balance equation with round-trip efficiency
applied on charging, and a minimum renewable fraction must be met each period.

A terminal state-of-charge constraint requires the battery to end no lower than it
started, so the battery cannot be drained "for free" to subsidize generation.

# Fields
- `n_sources::Int`: Number of power generation sources
- `n_periods::Int`: Number of time periods
- `sources::Vector{String}`: Names of energy sources
- `time_periods::Vector{Int}`: Time period indices (1:n_periods)
- `generation_costs::Dict{String,Float64}`: Cost per MWh for each source
- `capacities::Dict{String,Float64}`: Maximum capacity (MW) for each source
- `demands::Vector{Float64}`: Demand (MW) in each period
- `renewable_sources::Vector{String}`: Subset of sources that count as renewable
- `renewable_fraction::Float64`: Minimum fraction of generation from renewables
- `storage_capacity::Float64`: Maximum battery energy storage (MWh)
- `storage_power::Float64`: Maximum charge/discharge rate (MW)
- `storage_efficiency::Float64`: Round-trip (charging) efficiency in (0, 1]
- `initial_level::Float64`: Initial battery state of charge (MWh)
"""
struct StorageEnergyProblem <: ProblemGenerator
    n_sources::Int
    n_periods::Int
    sources::Vector{String}
    time_periods::Vector{Int}
    generation_costs::Dict{String,Float64}
    capacities::Dict{String,Float64}
    demands::Vector{Float64}
    renewable_sources::Vector{String}
    renewable_fraction::Float64
    storage_capacity::Float64
    storage_power::Float64
    storage_efficiency::Float64
    initial_level::Float64
end

"""
    StorageEnergyProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an energy-with-storage problem instance.

Variable count in `build_model`:
- generation `x[s, t]`            => n_sources * n_periods
- `storage_level[t in 0:T]`       => n_periods + 1
- `charge[t]`                     => n_periods
- `discharge[t]`                  => n_periods
Total = n_sources * n_periods + 3 * n_periods + 1
      = (n_sources + 3) * n_periods + 1
The dimensions are sized so this total lands near `target_variables`.

# Arguments
- `target_variables`: Target number of decision variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function StorageEnergyProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine scale
    if target_variables < 250
        min_sources, max_sources = 2, 8
        min_periods, max_periods = 4, 60
        peak_demand_range = (10.0, 100.0)
    elseif target_variables < 1000
        min_sources, max_sources = 5, 12
        min_periods, max_periods = 24, 120
        peak_demand_range = (100.0, 1000.0)
    else
        min_sources, max_sources = 8, 40
        min_periods, max_periods = 48, 300
        peak_demand_range = (1000.0, 10000.0)
    end

    # Sizing: total = (n_sources + 3) * n_periods + 1
    n_sources = min_sources + 1
    n_sources = clamp(n_sources, min_sources, max_sources)
    # Clamp to max_periods up front: otherwise an over-cap initial estimate makes
    # the loop believe it is already on target and break before scaling n_sources.
    n_periods = clamp(round(Int, (target_variables - 1) / (n_sources + 3)), min_periods, max_periods)

    # Iteratively adjust to reach target (accounting for the +3 storage var sets)
    for _ in 1:20
        current_vars = (n_sources + 3) * n_periods + 1
        if abs(current_vars - target_variables) / target_variables < 0.10
            break
        end
        ratio = target_variables / current_vars
        if ratio > 1.05
            if n_periods < max_periods
                n_periods = min(max_periods, max(min_periods,
                    round(Int, (target_variables - 1) / (n_sources + 3))))
            elseif n_sources < max_sources
                n_sources = min(max_sources, n_sources + 1)
            else
                break
            end
        elseif ratio < 0.95
            if n_periods > min_periods
                n_periods = max(min_periods,
                    round(Int, (target_variables - 1) / (n_sources + 3)))
            elseif n_sources > min_sources
                n_sources = max(min_sources, n_sources - 1)
            else
                break
            end
        else
            break
        end
    end
    n_periods = clamp(n_periods, min_periods, max_periods)

    # Sample parameters
    renewable_fraction_target = rand(Beta(2, 3))
    demand_variation = rand(Beta(2, 3))
    peak_demand = rand(Uniform(peak_demand_range...))
    base_generation_cost = rand(LogNormal(log(50.0), 0.3))
    renewable_cost_factor = rand(Gamma(2.5, 0.4))
    capacity_margin = max(1.15, min(1.6, rand(Normal(1.3, 0.08))))

    # Source types: (name, is_renewable, availability, capacity_factor, cost_factor)
    source_types = [
        ("coal", false, 0.95, 0.9, 1.0),
        ("gas", false, 0.98, 0.85, 1.2),
        ("nuclear", false, 0.92, 0.95, 0.8),
        ("solar", true, 0.99, 0.25, 0.3),
        ("wind", true, 0.95, 0.35, 0.4),
        ("hydro", true, 0.90, 0.50, 0.6),
        ("biomass", true, 0.88, 0.75, 1.1),
    ]

    # Build the generation fleet. Real grids run many units per technology, so the
    # fleet may exceed the number of distinct catalogue types. Distinct types are
    # used first (diversity at small sizes); beyond that, types repeat with unique
    # names and jittered techno-economic attributes, so the fleet — and the
    # variable count — scales with n_sources instead of being capped at the
    # catalogue size.
    n_renewables = clamp(ceil(Int, n_sources * renewable_fraction_target), 1, n_sources - 1)
    n_conventional = n_sources - n_renewables

    renewable_indices = findall(s -> s[2], source_types)
    conventional_indices = findall(s -> !s[2], source_types)

    name_counter = Dict{String,Int}()
    function build_fleet(indices, n)
        fleet = Tuple{String,String,Bool,Float64,Float64,Float64}[]
        (isempty(indices) || n <= 0) && return fleet
        order = shuffle(indices)
        for k in 1:n
            base = source_types[order[((k - 1) % length(order)) + 1]]
            name_counter[base[1]] = get(name_counter, base[1], 0) + 1
            uname = name_counter[base[1]] == 1 ? base[1] :
                    string(base[1], "_", name_counter[base[1]])
            push!(fleet, (uname, base[1], base[2],
                clamp(base[3] * rand(Uniform(0.97, 1.03)), 0.5, 1.0),
                base[4] * rand(Uniform(0.9, 1.1)),
                base[5] * rand(Uniform(0.9, 1.1))))
        end
        return fleet
    end
    selected_sources = vcat(build_fleet(renewable_indices, n_renewables),
                            build_fleet(conventional_indices, n_conventional))
    # Keep total sources consistent with the variable-count sizing
    n_sources = length(selected_sources)

    sources = [s[1] for s in selected_sources]
    renewable_sources = [s[1] for s in selected_sources if s[3]]
    time_periods = collect(1:n_periods)

    # Generation costs
    generation_costs = Dict{String,Float64}()
    for (name, _technology, is_renewable, _, _, cost_factor) in selected_sources
        base_cost = base_generation_cost * cost_factor
        if is_renewable
            base_cost *= renewable_cost_factor
        end
        variation = rand(Normal(1.0, 0.12))
        generation_costs[name] = max(5.0, base_cost * variation)
    end

    # Capacities
    capacities = Dict{String,Float64}()
    total_required_capacity = peak_demand * capacity_margin
    capacity_shares = Float64[]
    for (_name, technology, _, _, _, _) in selected_sources
        share = if technology == "coal"
            rand(Gamma(2, 0.25))
        elseif technology == "gas"
            rand(Gamma(3, 0.15))
        elseif technology == "nuclear"
            rand(Gamma(1.5, 0.4))
        elseif technology in ("solar", "wind")
            rand(Beta(2, 4))
        else
            rand(Beta(2, 5))
        end
        push!(capacity_shares, share)
    end
    total_share = sum(capacity_shares)
    for (i, (name, _technology, _, availability, capacity_factor, _)) in enumerate(selected_sources)
        normalized_share = capacity_shares[i] / total_share
        effective_capacity = total_required_capacity * normalized_share / (availability * capacity_factor)
        capacities[name] = max(10.0, effective_capacity)
    end

    # Demands (daily load pattern with mild noise)
    demands = Float64[]
    hour_factors = [0.6, 0.55, 0.5, 0.5, 0.55, 0.7, 0.85, 1.0, 0.95, 0.9,
                    0.85, 0.9, 0.95, 1.0, 0.9, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.8, 0.7, 0.65]
    base_demand = peak_demand * (1 - demand_variation)
    for p in 1:n_periods
        hour_idx = 1 + (p - 1) % 24
        pattern_demand = base_demand + (peak_demand - base_demand) * hour_factors[hour_idx]
        noise = rand(Normal(1.0, 0.05))
        push!(demands, pattern_demand * clamp(noise, 0.7, 1.3))
    end

    renewable_fraction = renewable_fraction_target

    # Battery storage parameters
    storage_capacity = peak_demand * rand(Uniform(0.5, 2.0))      # MWh
    storage_power = storage_capacity / rand(Uniform(2.0, 6.0))    # MW (2-6 hour duration)
    storage_efficiency = rand(Uniform(0.85, 0.95))
    initial_level = storage_capacity / 2                          # start half-charged

    # Feasibility handling
    actual_status = feasibility_status
    if feasibility_status == unknown
        # Natural instance: no forced infeasibility either way.
        actual_status = unknown
    end

    if actual_status == feasible
        max_demand = maximum(demands)
        # Ensure total generation can meet peak demand with margin.
        total_capacity = sum(values(capacities))
        if total_capacity < max_demand * 1.3
            scale_factor = (max_demand * 1.3) / total_capacity
            for s in sources
                capacities[s] *= scale_factor
            end
        end
        # Ensure renewable capacity alone can satisfy the renewable-fraction
        # requirement at peak demand (renewable gen >= rf * total gen, and total
        # gen >= demand, so renewables must be able to supply rf * demand).
        renewable_capacity = sum(capacities[s] for s in renewable_sources)
        required_renewable = renewable_fraction * max_demand * 1.05
        if renewable_capacity < required_renewable
            scale_factor = required_renewable / renewable_capacity
            for s in renewable_sources
                capacities[s] *= scale_factor
            end
        end
    elseif actual_status == infeasible
        # Force a deterministic capacity shortage: total generation cannot meet
        # peak demand even with full discharge, since discharge is bounded by
        # storage_power and the terminal SoC constraint forbids net draindown.
        max_demand = maximum(demands)
        # Shrink all generation so that gen + storage_power < demand with margin.
        target_total = (max_demand - storage_power) * 0.5
        target_total = max(target_total, 1.0)
        total_capacity = sum(values(capacities))
        scale_factor = target_total / total_capacity
        for s in sources
            capacities[s] *= scale_factor
        end
    end

    return StorageEnergyProblem(
        n_sources, n_periods, sources, time_periods, generation_costs,
        capacities, demands, renewable_sources, renewable_fraction,
        storage_capacity, storage_power, storage_efficiency, initial_level,
    )
end

"""
    build_model(prob::StorageEnergyProblem)

Build a JuMP model for the energy-with-storage problem. Completely deterministic —
uses only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::StorageEnergyProblem)
    model = Model()

    sources = prob.sources
    T = prob.time_periods

    # Variables
    # x[s, t]: generation; storage_level[t]: state of charge (t = 0..n_periods);
    # charge[t], discharge[t]: battery power flows.
    # Total = n_sources*n_periods + (n_periods+1) + n_periods + n_periods
    @variable(model, 0 <= x[s in sources, t in T] <= prob.capacities[s])
    @variable(model, 0 <= storage_level[t in 0:prob.n_periods] <= prob.storage_capacity)
    @variable(model, 0 <= charge[t in T] <= prob.storage_power)
    @variable(model, 0 <= discharge[t in T] <= prob.storage_power)

    # Objective: minimize total generation cost
    @objective(model, Min,
        sum(prob.generation_costs[s] * x[s, t] for s in sources, t in T))

    # Initial state of charge
    @constraint(model, storage_level[0] == prob.initial_level)

    for t in T
        # Meet demand: generation + net discharge >= demand
        @constraint(model,
            sum(x[s, t] for s in sources) + discharge[t] - charge[t] >= prob.demands[t])

        # Storage balance: round-trip efficiency applied on charging
        @constraint(model,
            storage_level[t] == storage_level[t-1] +
                prob.storage_efficiency * charge[t] - discharge[t])
    end

    # Terminal state-of-charge: battery cannot end below where it started,
    # preventing a "free" draindown of stored energy.
    @constraint(model, storage_level[prob.n_periods] >= storage_level[0])

    # Minimum renewable fraction each period
    for t in T
        @constraint(model,
            sum(x[s, t] for s in prob.renewable_sources) >=
                prob.renewable_fraction * sum(x[s, t] for s in sources))
    end

    return model
end

# Register the variant
register_variant(
    :energy,
    :storage,
    StorageEnergyProblem,
    "Energy generation mix with battery storage: charge/discharge, state-of-charge balance with round-trip efficiency, and a terminal SoC floor",
)
