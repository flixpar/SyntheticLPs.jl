using JuMP
using Random
using StatsBase
using Distributions

"""
    ReservesEnergyProblem <: ProblemGenerator

Generator for energy generation dispatch problems with operating reserve requirements.

# Overview
Models economic dispatch of a generation fleet over a time horizon while procuring
spinning and non-spinning operating reserves in every period. The decisions are,
for each source and period, the energy generated, the spinning reserve provided,
and the non-spinning reserve provided. The objective minimizes total generation
cost plus a small reserve provision (opportunity) cost so that reserve allocation
is non-degenerate. Constraints meet demand, meet spinning and non-spinning reserve
requirements, and couple reserves to remaining headroom: generation plus spinning
reserve cannot exceed capacity, and non-spinning reserve cannot exceed the unused
capacity after generation.

# Fields
- `n_sources::Int`: Number of power generation sources
- `n_periods::Int`: Number of time periods
- `sources::Vector{String}`: Names of energy sources
- `time_periods::Vector{Int}`: Time period indices
- `generation_costs::Dict{String,Float64}`: Cost per MWh for each source
- `capacities::Dict{String,Float64}`: Maximum capacity (MW) for each source
- `demands::Vector{Float64}`: Demand (MW) in each period
- `spinning_reserve_req::Vector{Float64}`: Required spinning reserve (MW) per period
- `non_spinning_reserve_req::Vector{Float64}`: Required non-spinning reserve (MW) per period
- `spinning_reserve_cost::Dict{String,Float64}`: Provision cost per MW of spinning reserve for each source
- `non_spinning_reserve_cost::Dict{String,Float64}`: Provision cost per MW of non-spinning reserve for each source
"""
struct ReservesEnergyProblem <: ProblemGenerator
    n_sources::Int
    n_periods::Int
    sources::Vector{String}
    time_periods::Vector{Int}
    generation_costs::Dict{String,Float64}
    capacities::Dict{String,Float64}
    demands::Vector{Float64}
    spinning_reserve_req::Vector{Float64}
    non_spinning_reserve_req::Vector{Float64}
    spinning_reserve_cost::Dict{String,Float64}
    non_spinning_reserve_cost::Dict{String,Float64}
end

"""
    ReservesEnergyProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an energy dispatch-with-reserves problem instance.

The model creates 3 decision variables per source-period (generation, spinning
reserve, non-spinning reserve), so the dimensions are sized as
`n_sources * n_periods ≈ target_variables / 3`.

# Arguments
- `target_variables`: Target number of variables (≈ 3 * n_sources * n_periods)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function ReservesEnergyProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Variable count = 3 * n_sources * n_periods (gen + spin + nonspin per source-period).
    # Size dimensions so that n_sources * n_periods ≈ target_variables / 3.
    sp_target = max(1, round(Int, target_variables / 3))

    # Scale-dependent ranges
    if target_variables < 250
        min_sources, max_sources = 2, 8
        min_periods, max_periods = 4, 48
        peak_demand_range = (10.0, 100.0)
    elseif target_variables < 1000
        min_sources, max_sources = 5, 12
        min_periods, max_periods = 16, 96
        peak_demand_range = (100.0, 1000.0)
    else
        min_sources, max_sources = 8, 40
        min_periods, max_periods = 32, 250
        peak_demand_range = (1000.0, 10000.0)
    end

    # Pick a source count, then derive periods to hit the source-period target.
    n_sources = clamp(min_sources + 2, min_sources, max_sources)
    n_periods = clamp(round(Int, sp_target / n_sources), min_periods, max_periods)

    # Iteratively adjust to land near the source-period target.
    for _ in 1:15
        current = n_sources * n_periods
        if abs(current - sp_target) / sp_target < 0.10
            break
        end
        ratio = sp_target / current
        if ratio > 1.05
            if n_periods < max_periods
                n_periods = clamp(round(Int, n_periods * ratio), min_periods, max_periods)
            elseif n_sources < max_sources
                n_sources = clamp(round(Int, n_sources * ratio), min_sources, max_sources)
            else
                break
            end
        elseif ratio < 0.95
            if n_periods > min_periods
                n_periods = clamp(round(Int, n_periods * ratio), min_periods, max_periods)
            elseif n_sources > min_sources
                n_sources = clamp(round(Int, n_sources * ratio), min_sources, max_sources)
            else
                break
            end
        end
    end

    # Sample high-level parameters
    renewable_fraction_target = rand(Beta(2, 3))
    demand_variation = rand(Beta(2, 3))
    peak_demand = rand(Uniform(peak_demand_range...))
    base_generation_cost = rand(LogNormal(log(50.0), 0.3))
    renewable_cost_factor = rand(Gamma(2.5, 0.4))
    capacity_margin = max(1.15, min(1.6, rand(Normal(1.3, 0.08))))

    # Candidate source types: (name, is_renewable, availability, capacity_factor, cost_factor)
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

    sources = [s[1] for s in selected_sources]
    n_sources = length(sources)
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

    # Capacities sized to cover peak demand with margin
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

    # Demand profile (daily pattern with noise)
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

    # Reserve requirements as fractions of demand
    spin_fraction = rand(Uniform(0.05, 0.15))
    nonspin_fraction = spin_fraction * rand(Uniform(0.5, 1.0))
    spinning_reserve_req = demands .* spin_fraction
    non_spinning_reserve_req = demands .* nonspin_fraction

    # Small reserve provision (opportunity) costs so reserve allocation is non-degenerate.
    # Spinning reserve is held on synchronized headroom and is pricier than non-spinning.
    spinning_reserve_cost = Dict{String,Float64}()
    non_spinning_reserve_cost = Dict{String,Float64}()
    for name in sources
        gc = generation_costs[name]
        spinning_reserve_cost[name] = gc * rand(Uniform(0.08, 0.20))
        non_spinning_reserve_cost[name] = gc * rand(Uniform(0.02, 0.08))
    end

    # Feasibility handling.
    # Feasible requires, in the worst period, total capacity to cover
    # demand + spinning reserve + non-spinning reserve (all draw on the same headroom).
    worst_need = maximum(demands .+ spinning_reserve_req .+ non_spinning_reserve_req)
    total_capacity = sum(values(capacities))

    if feasibility_status == feasible
        # Guarantee a feasible point exists with comfortable margin.
        required = worst_need * 1.25
        if total_capacity < required
            scale_factor = required / total_capacity
            for s in sources
                capacities[s] *= scale_factor
            end
        end
    elseif feasibility_status == infeasible
        # Force a deterministic contradiction: shrink total capacity well below the
        # worst-period demand alone so the demand constraint cannot be met.
        max_demand = maximum(demands)
        # Target total capacity at ~70% of peak demand (clear margin of infeasibility).
        target_capacity = max_demand * 0.7
        scale_factor = target_capacity / total_capacity
        for s in sources
            capacities[s] *= scale_factor
        end
    end
    # For unknown, leave the naturally-generated instance as-is (no forced infeasibility).

    return ReservesEnergyProblem(
        n_sources, n_periods, sources, time_periods, generation_costs,
        capacities, demands, spinning_reserve_req, non_spinning_reserve_req,
        spinning_reserve_cost, non_spinning_reserve_cost,
    )
end

"""
    build_model(prob::ReservesEnergyProblem)

Build a JuMP model for the energy dispatch-with-reserves problem. Deterministic —
uses only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::ReservesEnergyProblem)
    model = Model()

    # Variables: 3 per source-period (generation, spinning reserve, non-spinning reserve).
    @variable(model, 0 <= x[s in prob.sources, t in prob.time_periods] <= prob.capacities[s])
    @variable(model, spin_res[s in prob.sources, t in prob.time_periods] >= 0)
    @variable(model, nonspin_res[s in prob.sources, t in prob.time_periods] >= 0)

    # Objective: generation cost + small reserve provision (opportunity) cost.
    @objective(model, Min,
        sum(prob.generation_costs[s] * x[s, t] for s in prob.sources, t in prob.time_periods) +
        sum(prob.spinning_reserve_cost[s] * spin_res[s, t] for s in prob.sources, t in prob.time_periods) +
        sum(prob.non_spinning_reserve_cost[s] * nonspin_res[s, t] for s in prob.sources, t in prob.time_periods))

    for t in prob.time_periods
        # Meet demand
        @constraint(model, sum(x[s, t] for s in prob.sources) >= prob.demands[t])

        # Meet spinning reserve requirement
        @constraint(model, sum(spin_res[s, t] for s in prob.sources) >= prob.spinning_reserve_req[t])

        # Meet non-spinning reserve requirement
        @constraint(model, sum(nonspin_res[s, t] for s in prob.sources) >= prob.non_spinning_reserve_req[t])
    end

    # Capacity coupling: reserves draw on unused headroom above generation.
    for s in prob.sources, t in prob.time_periods
        # Generation and both reserve products draw on the same physical
        # headroom, so their sum cannot exceed the unit's capacity. Splitting
        # this into separate `x + spin <= cap` and `nonspin <= cap - x` bounds
        # would let the same unused MW back both reserve products at once,
        # certifying up to twice the real headroom.
        @constraint(model, x[s, t] + spin_res[s, t] + nonspin_res[s, t] <= prob.capacities[s])
    end

    return model
end

# Register the variant
register_variant(
    :energy,
    :reserves,
    ReservesEnergyProblem,
    "Energy dispatch with spinning and non-spinning operating reserve requirements and capacity coupling",
)
