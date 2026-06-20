using JuMP
using Random
using StatsBase
using Distributions

"""
    RampingEnergyProblem <: ProblemGenerator

Generator for economic dispatch problems with inter-temporal ramping constraints.

# Overview
Models multi-period economic dispatch where each generation source produces power
in every time period to meet demand at minimum cost. The distinguishing feature is
that consecutive periods are linked by per-generator ramp-up and ramp-down limits:
a source cannot increase or decrease its output between adjacent periods by more
than its technology-dependent ramp rate. Ramp fractions are technology-correlated
(nuclear ramps slowly, gas is flexible, renewables can change quickly). A renewable
generation floor is also imposed in every period.

# Fields
- `n_sources::Int`: Number of power generation sources
- `n_periods::Int`: Number of time periods
- `sources::Vector{String}`: Names of energy sources
- `time_periods::Vector{Int}`: Time period indices
- `generation_costs::Dict{String,Float64}`: Cost per MWh for each source
- `capacities::Dict{String,Float64}`: Maximum capacity (MW) for each source
- `demands::Vector{Float64}`: Demand (MW) in each period
- `renewable_set::Vector{String}`: Subset of sources that count as renewable
- `renewable_fraction::Float64`: Minimum fraction of generation from renewables each period
- `ramp_up_limits::Dict{String,Float64}`: Maximum increase in output per period (MW) per source
- `ramp_down_limits::Dict{String,Float64}`: Maximum decrease in output per period (MW) per source
"""
struct RampingEnergyProblem <: ProblemGenerator
    n_sources::Int
    n_periods::Int
    sources::Vector{String}
    time_periods::Vector{Int}
    generation_costs::Dict{String,Float64}
    capacities::Dict{String,Float64}
    demands::Vector{Float64}
    renewable_set::Vector{String}
    renewable_fraction::Float64
    ramp_up_limits::Dict{String,Float64}
    ramp_down_limits::Dict{String,Float64}
end

"""
    RampingEnergyProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a ramping economic dispatch problem instance.

The only decision variables are the dispatch levels `x[s, t]`, so the total
variable count is exactly `n_sources * n_periods`. The constructor sizes these two
dimensions so that their product lands near `target_variables`.

# Arguments
- `target_variables`: Target number of variables (n_sources × n_periods)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function RampingEnergyProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # Variable count formula: n_sources * n_periods (only x[s,t] decision variables).
    if target_variables < 250
        min_sources, max_sources = 3, 8
        min_periods, max_periods = 12, 48
        peak_demand_range = (10.0, 100.0)
    elseif target_variables < 1000
        min_sources, max_sources = 5, 12
        min_periods, max_periods = 24, 72
        peak_demand_range = (100.0, 1000.0)
    else
        min_sources, max_sources = 8, 40
        min_periods, max_periods = 48, 260
        peak_demand_range = (1000.0, 10000.0)
    end

    n_sources = min_sources + 2
    n_periods = min_periods + 12

    # Iteratively adjust n_sources * n_periods toward the target.
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

    # --- Sample high-level parameters ---
    renewable_fraction_target = rand(Beta(2, 3))
    demand_variation = rand(Beta(2, 3))
    peak_demand = rand(Uniform(peak_demand_range...))
    base_generation_cost = rand(LogNormal(log(50.0), 0.3))
    renewable_cost_factor = rand(Gamma(2.5, 0.4))
    capacity_margin = max(1.15, min(1.6, rand(Normal(1.3, 0.08))))

    # Source catalogue: (name, is_renewable, availability, capacity_factor, cost_factor)
    source_types = [
        ("coal", false, 0.95, 0.9, 1.0),
        ("gas", false, 0.98, 0.85, 1.2),
        ("nuclear", false, 0.92, 0.95, 0.8),
        ("solar", true, 0.99, 0.25, 0.3),
        ("wind", true, 0.95, 0.35, 0.4),
        ("hydro", true, 0.90, 0.50, 0.6),
        ("biomass", true, 0.88, 0.75, 1.1),
    ]

    # --- Build the generation fleet ---
    # Real grids run many units per technology, so the fleet may exceed the number
    # of distinct catalogue types. Distinct types are used first (diversity at small
    # sizes); beyond that, types repeat with unique names and jittered
    # techno-economic attributes, so the fleet — and the variable count — scales
    # with n_sources instead of being capped at the catalogue size.
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

    n_sources = length(selected_sources)
    sources = [s[1] for s in selected_sources]
    time_periods = collect(1:n_periods)

    # --- Generation costs ---
    generation_costs = Dict{String,Float64}()
    for (name, _technology, is_renewable, _, _, cost_factor) in selected_sources
        base_cost = base_generation_cost * cost_factor
        if is_renewable
            base_cost *= renewable_cost_factor
        end
        variation = rand(Normal(1.0, 0.12))
        generation_costs[name] = max(5.0, base_cost * variation)
    end

    # --- Capacities ---
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

    # --- Demand profile (daily load pattern + noise) ---
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

    # --- Renewable set and floor ---
    renewable_set = [s[1] for s in selected_sources if s[3]]
    renewable_fraction = renewable_fraction_target

    # --- Technology-differentiated ramp limits (fractions of capacity per period) ---
    ramp_up_limits = Dict{String,Float64}()
    ramp_down_limits = Dict{String,Float64}()
    for (name, technology, is_renewable, _, _, _) in selected_sources
        cap = capacities[name]
        if is_renewable
            # Renewables / hydro / biomass can change output quickly.
            ramp_up_limits[name] = cap * rand(Uniform(0.8, 1.0))
            ramp_down_limits[name] = cap * rand(Uniform(0.8, 1.0))
        elseif technology == "nuclear"
            # Nuclear ramps very slowly (baseload).
            ramp_up_limits[name] = cap * rand(Uniform(0.02, 0.05))
            ramp_down_limits[name] = cap * rand(Uniform(0.02, 0.05))
        elseif technology == "coal"
            # Coal is moderately flexible.
            ramp_up_limits[name] = cap * rand(Uniform(0.1, 0.2))
            ramp_down_limits[name] = cap * rand(Uniform(0.1, 0.2))
        else  # gas
            # Gas peakers are highly flexible.
            ramp_up_limits[name] = cap * rand(Uniform(0.3, 0.6))
            ramp_down_limits[name] = cap * rand(Uniform(0.3, 0.6))
        end
    end

    # --- Feasibility handling ---
    actual_status = feasibility_status
    if actual_status == unknown
        # Generate a natural instance; do not force infeasibility.
        actual_status = feasible
    end

    total_capacity = sum(values(capacities))
    max_demand = maximum(demands)

    if actual_status == feasible
        # 1. Ensure aggregate capacity comfortably covers peak demand.
        if total_capacity < max_demand * 1.3
            scale_factor = max_demand * 1.3 / total_capacity
            for s in sources
                capacities[s] *= scale_factor
                ramp_up_limits[s] *= scale_factor
                ramp_down_limits[s] *= scale_factor
            end
            total_capacity = sum(values(capacities))
        end

        # 2. Ensure aggregate ramp capacity can track demand swings between periods.
        max_demand_swing = 0.0
        for t in 2:n_periods
            max_demand_swing = max(max_demand_swing, abs(demands[t] - demands[t-1]))
        end
        total_ramp_up = sum(values(ramp_up_limits))
        total_ramp_down = sum(values(ramp_down_limits))
        min_total_ramp = min(total_ramp_up, total_ramp_down)
        if min_total_ramp < max_demand_swing * 1.5
            # Boost flexible sources' ramp limits (capped at their capacity).
            for s in sources
                boost = 2.0
                ramp_up_limits[s] = min(capacities[s], ramp_up_limits[s] * boost)
                ramp_down_limits[s] = min(capacities[s], ramp_down_limits[s] * boost)
            end
            # If still short, give every source full-capacity ramp freedom.
            if min(sum(values(ramp_up_limits)), sum(values(ramp_down_limits))) < max_demand_swing * 1.5
                for s in sources
                    ramp_up_limits[s] = capacities[s]
                    ramp_down_limits[s] = capacities[s]
                end
            end
        end

        # 3. Make the renewable floor satisfiable: renewable capacity must be able to
        #    supply at least renewable_fraction of total generation. Cap the floor at
        #    a value the available renewable capacity can plausibly meet.
        renewable_capacity = isempty(renewable_set) ? 0.0 : sum(capacities[s] for s in renewable_set)
        if renewable_capacity <= 0.0
            renewable_fraction = 0.0
        else
            # Each period total generation can be as low as that period's demand.
            # Require fraction * demand <= renewable_capacity for all periods.
            feasible_fraction = min(renewable_fraction, 0.95 * renewable_capacity / max_demand)
            renewable_fraction = max(0.0, feasible_fraction)
        end

    elseif actual_status == infeasible
        # Deterministic capacity shortage with a clear margin: total capacity is
        # forced strictly below peak demand, so demand can never be met.
        scale_factor = (0.6 * max_demand) / total_capacity
        for s in sources
            capacities[s] *= scale_factor
            # Ramp limits cannot exceed the (now small) capacity.
            ramp_up_limits[s] = min(ramp_up_limits[s], capacities[s])
            ramp_down_limits[s] = min(ramp_down_limits[s], capacities[s])
        end
    end

    return RampingEnergyProblem(
        n_sources, n_periods, sources, time_periods, generation_costs,
        capacities, demands, renewable_set, renewable_fraction,
        ramp_up_limits, ramp_down_limits,
    )
end

"""
    build_model(prob::RampingEnergyProblem)

Build a JuMP model for the ramping economic dispatch problem. Deterministic — uses
only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::RampingEnergyProblem)
    model = Model()

    # Decision variables: dispatch level of each source in each period (bounded by capacity).
    # Total variable count = n_sources * n_periods.
    @variable(model, 0 <= x[s in prob.sources, t in prob.time_periods] <= prob.capacities[s])

    # Objective: minimize total generation cost.
    @objective(model, Min,
        sum(prob.generation_costs[s] * x[s, t] for s in prob.sources, t in prob.time_periods))

    # Meet demand in every period.
    for t in prob.time_periods
        @constraint(model, sum(x[s, t] for s in prob.sources) >= prob.demands[t])
    end

    # Inter-temporal ramp limits linking consecutive periods.
    for s in prob.sources, t in prob.time_periods[2:end]
        @constraint(model, x[s, t] - x[s, t-1] <= prob.ramp_up_limits[s])
        @constraint(model, x[s, t-1] - x[s, t] <= prob.ramp_down_limits[s])
    end

    # Renewable generation floor in every period.
    if !isempty(prob.renewable_set) && prob.renewable_fraction > 0.0
        for t in prob.time_periods
            @constraint(model,
                sum(x[s, t] for s in prob.renewable_set) >=
                prob.renewable_fraction * sum(x[s, t] for s in prob.sources))
        end
    end

    return model
end

# Register the variant
register_variant(
    :energy,
    :ramping,
    RampingEnergyProblem,
    "Economic dispatch over time with per-generator ramp up/down limits linking consecutive periods",
)
