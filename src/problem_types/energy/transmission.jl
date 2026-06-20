using JuMP
using Random
using StatsBase
using Distributions

"""
    TransmissionEnergyProblem <: ProblemGenerator

Generator for multi-zone energy dispatch problems with inter-zone transmission.

# Overview
Models economic dispatch across several transmission zones connected by capacitated
lines. Each generation source lives in a specific zone, and power can be transmitted
between zones subject to line capacity limits and transmission losses. The decisions
are per-source generation in each period and directed inter-zone power flows in each
period. The objective minimizes total generation cost. Each zone must meet its local
demand from a combination of in-zone generation and net imports.

# Line / loss model convention
For every ordered pair of distinct zones `(i, j)` and period `t` there is a single
non-negative directed flow variable `flow[i, j, t]` representing power dispatched FROM
zone `i` TOWARD zone `j`. To avoid double-counting losses, the loss factor is applied
exactly ONCE, on the DELIVERED power at the receiving zone: zone `i` sends `flow[i,j,t]`
(counted as outflow at `i`), and zone `j` receives `(1 - transmission_loss) * flow[i,j,t]`
(counted as inflow at `j`). Capacity `transmission_capacity[i, j]` caps the SENT power
`flow[i, j, t]`. Diagonal (self) flows are fixed to zero.

# Fields
- `n_sources::Int`: Number of power generation sources
- `n_periods::Int`: Number of time periods
- `n_zones::Int`: Number of transmission zones
- `sources::Vector{String}`: Names of energy sources
- `time_periods::Vector{Int}`: Time period indices
- `generation_costs::Dict{String,Float64}`: Cost per MWh for each source
- `capacities::Dict{String,Float64}`: Maximum capacity (MW) for each source
- `zone_sources::Dict{String,Int}`: Zone index that each source belongs to
- `zone_demands::Matrix{Float64}`: Demand (MW) per zone (rows) per period (cols)
- `transmission_capacity::Matrix{Float64}`: Sent-power capacity (MW) between zones
- `transmission_loss::Float64`: Fractional loss applied once on delivered power
"""
struct TransmissionEnergyProblem <: ProblemGenerator
    n_sources::Int
    n_periods::Int
    n_zones::Int
    sources::Vector{String}
    time_periods::Vector{Int}
    generation_costs::Dict{String,Float64}
    capacities::Dict{String,Float64}
    zone_sources::Dict{String,Int}
    zone_demands::Matrix{Float64}
    transmission_capacity::Matrix{Float64}
    transmission_loss::Float64
end

"""
    TransmissionEnergyProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-zone transmission energy dispatch problem instance.

# Variable count
`build_model` creates `x[sources, periods]` (n_sources * n_periods) plus
`flow[zones, zones, periods]` (n_zones^2 * n_periods, including fixed diagonal flows).
Total decision variables = n_periods * (n_sources + n_zones^2). Dimensions are sized
so this product lands near `target_variables`; the flow set (dominated by
n_zones^2 * n_periods) is explicitly accounted for so it is not ignored.

# Arguments
- `target_variables`: Target number of variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function TransmissionEnergyProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # Total vars = n_periods * (n_sources + n_zones^2).
    # Pick a modest number of zones, then a number of sources, then solve for periods
    # so the total matches target_variables (accounting for BOTH var sets).
    if target_variables < 250
        n_zones = rand(2:3)
        n_sources = rand(3:5)
    elseif target_variables < 1000
        n_zones = rand(3:4)
        n_sources = rand(5:8)
    else
        n_zones = rand(4:5)
        n_sources = rand(8:12)
    end

    # Cap sources at the number of distinct source types available (7).
    n_sources = min(n_sources, 7)

    vars_per_period = n_sources + n_zones * n_zones
    n_periods = max(2, round(Int, target_variables / vars_per_period))

    # --- Source catalog (name, is_renewable, cost_factor) ---
    source_catalog = [
        ("nuclear", false, 0.8),
        ("coal", false, 1.0),
        ("gas", false, 1.2),
        ("biomass", true, 1.1),
        ("hydro", true, 0.6),
        ("wind", true, 0.4),
        ("solar", true, 0.3),
    ]
    selected = source_catalog[sample(1:length(source_catalog), n_sources, replace=false)]
    sources = [s[1] for s in selected]
    time_periods = collect(1:n_periods)

    # --- Generation costs ---
    base_generation_cost = rand(LogNormal(log(50.0), 0.3))
    generation_costs = Dict{String,Float64}()
    for (name, _, cost_factor) in selected
        variation = rand(Normal(1.0, 0.12))
        generation_costs[name] = max(5.0, base_generation_cost * cost_factor * variation)
    end

    # --- Demand profile ---
    peak_demand = rand(Uniform(50.0, 500.0))
    demand_variation = rand(Beta(2, 3))
    base_demand = peak_demand * (1 - demand_variation)
    hour_factors = [0.6, 0.55, 0.5, 0.5, 0.55, 0.7, 0.85, 1.0, 0.95, 0.9,
                    0.85, 0.9, 0.95, 1.0, 0.9, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.8, 0.7, 0.65]
    demands = Float64[]
    for p in 1:n_periods
        hour_idx = 1 + (p - 1) % 24
        pattern_demand = base_demand + (peak_demand - base_demand) * hour_factors[hour_idx]
        noise = rand(Normal(1.0, 0.05))
        push!(demands, pattern_demand * max(0.7, min(1.3, noise)))
    end

    # --- Assign sources to zones (round-robin so every zone with index <= n_sources
    #     gets at least one source; extras wrap around) ---
    zone_sources = Dict{String,Int}()
    for (i, source) in enumerate(sources)
        zone_sources[source] = ((i - 1) % n_zones) + 1
    end

    # --- Zone demands: split total demand across zones each period ---
    zone_weights = rand(Dirichlet(ones(n_zones)))
    zone_demands = zeros(n_zones, n_periods)
    for z in 1:n_zones, t in 1:n_periods
        zone_demands[z, t] = demands[t] * zone_weights[z] * rand(Uniform(0.9, 1.1))
    end

    # --- Generator capacities ---
    # Size capacities so that, system-wide, generation can comfortably cover peak
    # total demand plus a margin. Distribute capacity across sources by random shares.
    capacity_margin = rand(Uniform(1.25, 1.6))
    max_total_demand = maximum(sum(zone_demands, dims=1))
    total_required_capacity = max_total_demand * capacity_margin
    shares = rand(Dirichlet(ones(n_sources)))
    capacities = Dict{String,Float64}()
    for (i, name) in enumerate(sources)
        capacities[name] = max(10.0, total_required_capacity * shares[i])
    end

    # --- Transmission ---
    transmission_loss = rand(Uniform(0.02, 0.05))  # 2-5% loss on delivered power
    avg_zone_demand = max_total_demand / n_zones
    transmission_capacity = zeros(n_zones, n_zones)
    for i in 1:n_zones, j in 1:n_zones
        if i != j
            transmission_capacity[i, j] = avg_zone_demand * rand(Uniform(0.6, 1.2))
        end
    end

    # --- Feasibility handling ---
    actual_status = feasibility_status
    if feasibility_status == unknown
        # Natural instance; no forced contradiction.
        actual_status = unknown
    end

    if actual_status == feasible
        # Guarantee feasibility: ensure EACH zone can independently meet its own peak
        # demand from in-zone generation alone (so transmission is never required).
        # This makes a feasible point provably exist regardless of line capacities.
        for z in 1:n_zones
            in_zone = [s for s in sources if zone_sources[s] == z]
            zone_peak = maximum(zone_demands[z, :])
            if isempty(in_zone)
                # Every zone must host at least one source to be self-sufficient;
                # reassign the first source to this zone.
                src = sources[1]
                zone_sources[src] = z
                in_zone = [src]
            end
            in_zone_cap = sum(capacities[s] for s in in_zone)
            if in_zone_cap < zone_peak * 1.1
                scale = (zone_peak * 1.2) / in_zone_cap
                for s in in_zone
                    capacities[s] *= scale
                end
            end
        end
    elseif actual_status == infeasible
        # Force a deterministic contradiction with a clear margin: give zone 1 far more
        # demand than its in-zone generation can supply, and choke the lines feeding it
        # so imports cannot close the gap (sent power capped near zero into zone 1).
        in_zone1 = [s for s in sources if zone_sources[s] == 1]
        if isempty(in_zone1)
            zone_sources[sources[1]] = 1
            in_zone1 = [sources[1]]
        end
        in_zone1_cap = sum(capacities[s] for s in in_zone1)
        # Demand zone 1 to require 3x its local capacity in every period.
        for t in 1:n_periods
            zone_demands[1, t] = in_zone1_cap * 3.0
        end
        # Even with full delivered imports from all other zones, total deliverable
        # power is bounded by line capacities; make those tiny so the gap cannot close.
        for i in 1:n_zones
            if i != 1
                transmission_capacity[i, 1] = in_zone1_cap * 0.01
            end
        end
    end

    return TransmissionEnergyProblem(
        n_sources, n_periods, n_zones, sources, time_periods,
        generation_costs, capacities, zone_sources,
        zone_demands, transmission_capacity, transmission_loss,
    )
end

"""
    build_model(prob::TransmissionEnergyProblem)

Build a JuMP model for the multi-zone transmission energy dispatch problem.
Deterministic — uses only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::TransmissionEnergyProblem)
    model = Model()

    # Variables:
    #   x[s, t]        generation per source per period   -> n_sources * n_periods
    #   flow[i, j, t]  sent power from zone i toward j     -> n_zones^2 * n_periods
    # Total = n_periods * (n_sources + n_zones^2).
    @variable(model, 0 <= x[s in prob.sources, t in prob.time_periods] <= prob.capacities[s])
    @variable(model, flow[i in 1:prob.n_zones, j in 1:prob.n_zones, t in prob.time_periods] >= 0)

    # Objective: minimize total generation cost.
    @objective(model, Min,
        sum(prob.generation_costs[s] * x[s, t] for s in prob.sources, t in prob.time_periods))

    # Zonal power balance per period:
    #   in-zone generation + delivered inflow - sent outflow >= zone demand.
    # Loss is applied ONCE, on the delivered inflow at the receiving zone.
    for t in prob.time_periods
        for z in 1:prob.n_zones
            zone_gen = sum(x[s, t] for s in prob.sources if prob.zone_sources[s] == z; init = 0.0)
            inflow = sum((1 - prob.transmission_loss) * flow[i, z, t]
                         for i in 1:prob.n_zones if i != z; init = 0.0)
            outflow = sum(flow[z, j, t] for j in 1:prob.n_zones if j != z; init = 0.0)
            @constraint(model, zone_gen + inflow - outflow >= prob.zone_demands[z, t])
        end
    end

    # Transmission limits: cap sent power; fix self-flows to zero.
    for i in 1:prob.n_zones, j in 1:prob.n_zones, t in prob.time_periods
        if i != j
            @constraint(model, flow[i, j, t] <= prob.transmission_capacity[i, j])
        else
            @constraint(model, flow[i, j, t] == 0)
        end
    end

    return model
end

# Register the variant
register_variant(
    :energy,
    :transmission,
    TransmissionEnergyProblem,
    "Multi-zone energy dispatch with capacitated inter-zone transmission flows and losses",
)
