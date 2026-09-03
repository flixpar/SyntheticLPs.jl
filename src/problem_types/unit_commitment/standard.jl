using JuMP
using Random
using StatsBase
using Distributions

"""
    UnitCommitmentProblem <: ProblemGenerator

Generator for realistic unit commitment power system planning problems that capture
thermal, renewable, and peaking generation fleets across multiple time periods.

# Overview

Models the operational scheduling of a heterogeneous generation fleet over a
multi-period horizon. The fleet mixes baseload units (nuclear/coal), cycling
combined-cycle gas, fast-ramping combustion turbines, hydro, and variable
renewables (wind), each with distinct capacities, minimum stable outputs, ramp
limits, costs, and minimum up/down times. The objective minimizes total
operating cost (variable generation cost, no-load cost, startup cost, and
shutdown cost). Constraints enforce generation bounds tied to commitment status
and availability, system demand balance, spinning reserve requirements, ramp
limits between consecutive periods, commitment state logic linking on/startup/
shutdown, and minimum up/down time windows.

The natural formulation is a MILP: commitment, startup, and shutdown are binary.
The package-level default `relax_integer=true` exposes its LP relaxation, while
`relax_integer=false` retains the implementable commitment schedule.

Requested-feasible instances store a complete primal witness covering every
variable family. Requested-infeasible instances store an aggregate capacity
certificate for one period. These artifacts make the requested status checkable
without solving the model.

# Fields

  - `n_units::Int`: Number of generation units
  - `n_periods::Int`: Number of time periods in the horizon
  - `units::Vector{String}`: Unit identifiers
  - `time_periods::Vector{Int}`: Time period indices
  - `demand::Vector{Float64}`: System demand per period
  - `reserve_requirements::Vector{Float64}`: Spinning reserve requirement per period
  - `max_output::Dict{String,Float64}`: Maximum output per unit
  - `min_output::Dict{String,Float64}`: Minimum stable output per unit (when committed)
  - `ramp_up::Dict{String,Float64}`: Ramp-up limit per unit per period
  - `ramp_down::Dict{String,Float64}`: Ramp-down limit per unit per period
  - `variable_costs::Dict{String,Float64}`: Variable (per-MW) generation cost per unit
  - `no_load_costs::Dict{String,Float64}`: No-load (commitment) cost per unit per period
  - `startup_costs::Dict{String,Float64}`: Startup cost per unit
  - `shutdown_costs::Dict{String,Float64}`: Shutdown cost per unit
  - `min_up_times::Dict{String,Int}`: Minimum up time per unit (periods)
  - `min_down_times::Dict{String,Int}`: Minimum down time per unit (periods)
  - `availability_factors::Dict{String,Vector{Float64}}`: Per-period availability fraction per unit
  - `initial_on::Dict{String,Float64}`: Initial commitment state per unit (0 or 1)
  - `initial_generation::Dict{String,Float64}`: Initial generation per unit
  - `unit_types::Dict{String,Symbol}`: Sampled fleet archetype for each unit
  - `resolved_status::FeasibilityStatus`: Status actually constructed (`unknown` is resolved)
  - `feasible_witness`: Complete feasible point, present exactly for feasible instances
  - `infeasibility_certificate`: Aggregate capacity contradiction, present exactly for infeasible instances
"""
struct UnitCommitmentWitness
    generation::Matrix{Float64}
    commitment::Matrix{Float64}
    startup::Matrix{Float64}
    shutdown::Matrix{Float64}
end

Base.:(==)(a::UnitCommitmentWitness, b::UnitCommitmentWitness) =
    a.generation == b.generation &&
    a.commitment == b.commitment &&
    a.startup == b.startup &&
    a.shutdown == b.shutdown
Base.isequal(a::UnitCommitmentWitness, b::UnitCommitmentWitness) = a == b
Base.hash(a::UnitCommitmentWitness, h::UInt) =
    hash((a.generation, a.commitment, a.startup, a.shutdown), h)

struct UnitCommitmentCapacityCertificate
    period::Int
    available_capacity::Float64
    required_capacity::Float64
    excess::Float64
end

Base.:(==)(a::UnitCommitmentCapacityCertificate, b::UnitCommitmentCapacityCertificate) =
    a.period == b.period &&
    a.available_capacity == b.available_capacity &&
    a.required_capacity == b.required_capacity &&
    a.excess == b.excess
Base.isequal(a::UnitCommitmentCapacityCertificate, b::UnitCommitmentCapacityCertificate) = a == b
Base.hash(a::UnitCommitmentCapacityCertificate, h::UInt) =
    hash((a.period, a.available_capacity, a.required_capacity, a.excess), h)

struct UnitCommitmentProblem <: ProblemGenerator
    n_units::Int
    n_periods::Int
    units::Vector{String}
    time_periods::Vector{Int}
    demand::Vector{Float64}
    reserve_requirements::Vector{Float64}
    max_output::Dict{String, Float64}
    min_output::Dict{String, Float64}
    ramp_up::Dict{String, Float64}
    ramp_down::Dict{String, Float64}
    variable_costs::Dict{String, Float64}
    no_load_costs::Dict{String, Float64}
    startup_costs::Dict{String, Float64}
    shutdown_costs::Dict{String, Float64}
    min_up_times::Dict{String, Int}
    min_down_times::Dict{String, Int}
    availability_factors::Dict{String, Vector{Float64}}
    initial_on::Dict{String, Float64}
    initial_generation::Dict{String, Float64}
    unit_types::Dict{String, Symbol}
    resolved_status::FeasibilityStatus
    feasible_witness::Union{Nothing, UnitCommitmentWitness}
    infeasibility_certificate::Union{Nothing, UnitCommitmentCapacityCertificate}
end

"""
    UnitCommitmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a unit commitment problem instance.

Decision variables in `build_model`: `g`, `on`, `startup`, `shutdown`, each indexed
by (unit, period). Total = 4 * n_units * n_periods. The constructor sizes
`n_units` and `n_periods` so this product lands near `target_variables`.

# Arguments

  - `target_variables`: Target number of variables (4 × n_units × n_periods)
  - `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
  - `seed`: Random seed for reproducibility

For `feasible`, the constructor builds commitment, transition, and dispatch
trajectories first, then defines demand and reserve from that trajectory. The
stored witness satisfies every model row by construction. For `infeasible`, one
period has demand plus reserve strictly above all available nameplate capacity;
the stored certificate records the corresponding aggregate cut. For `unknown`,
the constructor resolves to one of these two profiles and records the result.
"""
function UnitCommitmentProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    resolved_status = if feasibility_status == unknown
        (rand(rng) < 0.65 ? feasible : infeasible)
    else
        feasibility_status
    end
    sizing_target = max(target_variables, 48)

    # Variable-count formula: 4 * n_units * n_periods (g, on, startup, shutdown
    # are each indexed by unit × period). Dimensions are sized below so this
    # product lands within ~10% of target_variables.
    # Switch bands at the smallest formulation supported by the next band. This
    # avoids artificial jumps (for example, 3,000 requested variables formerly
    # jumped to the 3,840-variable large-band floor).
    scale = if sizing_target < 192
        :tiny
    elseif sizing_target < 960
        :small
    elseif sizing_target < 3840
        :medium
    else
        :large
    end

    if scale == :tiny
        # Small synthetic instances: allow short horizons / few units so the
        # variable count (4 * n_units * n_periods) can reach low targets.
        unit_range = (2, 6)
        period_range = (6, 24)
    elseif scale == :small
        unit_range = (4, 9)
        period_range = (12, 36)
    elseif scale == :medium
        unit_range = (10, 22)
        period_range = (24, 72)
    else
        # Keep operational horizons at one week or less and grow the fleet for
        # large matrix requests instead of silently saturating near 32k variables.
        max_large_units = max(48, ceil(Int, sizing_target / (4 * 48)))
        unit_range = (20, max_large_units)
        period_range = (48, 168)
    end

    n_units = unit_range[1]
    n_periods = period_range[1]

    for _ in 1:20
        current_vars = n_units * n_periods * 4
        if abs(current_vars - sizing_target) / sizing_target <= 0.1
            break
        end

        ratio = sqrt(sizing_target / max(current_vars, 1))
        if ratio > 1.05
            if n_periods < period_range[2]
                n_periods = min(
                    period_range[2], max(period_range[1], round(Int, n_periods * ratio))
                )
            elseif n_units < unit_range[2]
                n_units = min(unit_range[2], max(unit_range[1], round(Int, n_units * ratio)))
            end
        elseif ratio < 0.95
            if n_periods > period_range[1]
                n_periods = max(period_range[1], round(Int, n_periods * ratio))
            elseif n_units > unit_range[1]
                n_units = max(unit_range[1], round(Int, n_units * ratio))
            end
        else
            break
        end
    end

    units = ["GEN$(i)" for i in 1:n_units]
    time_periods = collect(1:n_periods)

    unit_profiles = [
        (;
            type=:nuclear,
            weight=0.08,
            capacity_range=(700.0, 1300.0),
            min_ratio=(0.6, 0.85),
            ramp_fraction=(0.03, 0.07),
            var_cost=(10.0, 22.0),
            no_load=(10000.0, 18000.0),
            startup=(50000.0, 120000.0),
            shutdown=(10000.0, 25000.0),
            min_up=(36, 80),
            min_down=(24, 60),
        ),
        (;
            type=:coal,
            weight=0.22,
            capacity_range=(200.0, 800.0),
            min_ratio=(0.4, 0.7),
            ramp_fraction=(0.05, 0.12),
            var_cost=(20.0, 35.0),
            no_load=(6000.0, 12000.0),
            startup=(20000.0, 70000.0),
            shutdown=(5000.0, 15000.0),
            min_up=(12, 48),
            min_down=(10, 36),
        ),
        (;
            type=:ccgt,
            weight=0.28,
            capacity_range=(150.0, 500.0),
            min_ratio=(0.35, 0.6),
            ramp_fraction=(0.1, 0.2),
            var_cost=(28.0, 52.0),
            no_load=(4000.0, 9000.0),
            startup=(10000.0, 35000.0),
            shutdown=(3000.0, 10000.0),
            min_up=(6, 24),
            min_down=(6, 24),
        ),
        (;
            type=:gas_ct,
            weight=0.18,
            capacity_range=(40.0, 180.0),
            min_ratio=(0.0, 0.2),
            ramp_fraction=(0.5, 1.0),
            var_cost=(60.0, 120.0),
            no_load=(500.0, 2000.0),
            startup=(2000.0, 8000.0),
            shutdown=(1000.0, 4000.0),
            min_up=(1, 4),
            min_down=(1, 4),
        ),
        (;
            type=:hydro,
            weight=0.12,
            capacity_range=(80.0, 400.0),
            min_ratio=(0.1, 0.4),
            ramp_fraction=(0.3, 0.6),
            var_cost=(5.0, 18.0),
            no_load=(1000.0, 4000.0),
            startup=(5000.0, 20000.0),
            shutdown=(2000.0, 6000.0),
            min_up=(4, 12),
            min_down=(2, 10),
        ),
        (;
            type=:wind,
            weight=0.12,
            capacity_range=(50.0, 250.0),
            min_ratio=(0.0, 0.05),
            ramp_fraction=(0.8, 1.2),
            var_cost=(0.0, 5.0),
            no_load=(0.0, 0.0),
            startup=(0.0, 0.0),
            shutdown=(0.0, 0.0),
            min_up=(1, 2),
            min_down=(1, 2),
        ),
    ]

    profile_weights = Weights([profile.weight for profile in unit_profiles])

    max_output = Dict{String, Float64}()
    min_output = Dict{String, Float64}()
    ramp_up = Dict{String, Float64}()
    ramp_down = Dict{String, Float64}()
    variable_costs = Dict{String, Float64}()
    no_load_costs = Dict{String, Float64}()
    startup_costs = Dict{String, Float64}()
    shutdown_costs = Dict{String, Float64}()
    min_up_times = Dict{String, Int}()
    min_down_times = Dict{String, Int}()
    availability_factors = Dict{String, Vector{Float64}}()
    initial_on = Dict{String, Float64}()
    initial_generation = Dict{String, Float64}()
    unit_types = Dict{String, Symbol}()

    total_capacity = 0.0

    # Sample a value from an archetype range, tolerating zero-width ranges
    # (e.g. wind has no_load = (0.0, 0.0)) which `Uniform` rejects.
    runif(r) = r[1] < r[2] ? rand(rng, Uniform(r[1], r[2])) : float(r[1])

    for u in units
        profile = sample(rng, unit_profiles, profile_weights)
        unit_types[u] = profile.type

        cap = runif(profile.capacity_range)
        max_output[u] = cap
        total_capacity += cap

        min_ratio = runif(profile.min_ratio)
        min_output[u] = cap * min_ratio

        ramp_fraction = runif(profile.ramp_fraction)
        ramp_up[u] = max(5.0, cap * ramp_fraction)
        ramp_down[u] = max(5.0, cap * ramp_fraction * rand(rng, Uniform(0.8, 1.2)))

        variable_costs[u] = runif(profile.var_cost)
        no_load_costs[u] = runif(profile.no_load)
        startup_costs[u] = runif(profile.startup)
        shutdown_costs[u] = runif(profile.shutdown)
        min_up_times[u] = max(1, rand(rng, profile.min_up[1]:profile.min_up[2]))
        min_down_times[u] = max(1, rand(rng, profile.min_down[1]:profile.min_down[2]))

        availability = ones(Float64, n_periods)

        if profile.type in (:nuclear, :coal, :ccgt)
            if rand(rng) < 0.35
                outage_length = max(2, round(Int, rand(rng, Uniform(0.05, 0.15)) * n_periods))
                outage_length = min(outage_length, n_periods)
                start_period = rand(rng, 1:max(1, n_periods - outage_length + 1))
                for t in start_period:(start_period + outage_length - 1)
                    availability[t] = 0.0
                end
            end
        elseif profile.type == :gas_ct
            for t in eachindex(availability)
                if rand(rng) < 0.03
                    availability[t] = 0.0
                end
            end
        elseif profile.type == :hydro
            seasonality = rand(rng, Uniform(0.6, 1.0))
            for t in eachindex(availability)
                hour = (t - 1) % 24 + 1
                pattern = hour in 7:20 ? 1.0 : 0.8
                availability[t] = pattern * seasonality
            end
        else
            base_speed = rand(rng, Uniform(0.4, 0.8))
            gust_factor = rand(rng, Uniform(0.1, 0.35))
            for t in eachindex(availability)
                hour = (t - 1) % 24 + 1
                diurnal = 0.4 + 0.6 * sin(2π * hour / 24)
                noise = rand(rng, Uniform(-gust_factor, gust_factor))
                availability[t] = clamp(base_speed + diurnal * gust_factor + noise, 0.0, 1.0)
            end
        end

        if resolved_status == feasible
            # The feasible witness keeps the fleet online throughout the horizon.
            # Preserve derating/renewable variation, but leave enough available
            # output above the unit's stable minimum to carry load and reserve.
            minimum_fraction = min_output[u] / cap
            operating_floor = minimum_fraction + 0.12 * (1.0 - minimum_fraction)
            availability .= max.(availability, operating_floor)
        end
        availability_factors[u] = availability

        if resolved_status == feasible
            initial_on[u] = 1.0
            initial_generation[u] = 0.0  # filled from the constructed dispatch below
        else
            preferred_on = if profile.type in (:nuclear, :coal)
                rand(rng) < 0.8
            elseif profile.type == :wind
                true
            else
                rand(rng) < 0.6
            end
            available_first = cap * availability[1]
            # Strict: `Uniform(min_output, available_first)` below throws unless
            # the unit has real headroom above its stable minimum.
            initial_on[u] = preferred_on && available_first > min_output[u] + 1e-9 ? 1.0 : 0.0
            initial_generation[u] =
                initial_on[u] > 0.5 ? rand(rng, Uniform(min_output[u], available_first)) : 0.0
        end
    end

    # Demand shape used both as a natural load profile and as the common dispatch
    # signal for the constructive feasible branch.
    seasonal_trend = rand(rng, Uniform(0.85, 1.15))
    reserve_fraction = rand(rng, Uniform(0.08, 0.18))

    daily_profiles = [
        [
            0.55,
            0.5,
            0.48,
            0.47,
            0.5,
            0.6,
            0.75,
            0.9,
            0.95,
            1.0,
            0.98,
            0.95,
            0.92,
            0.94,
            0.97,
            1.0,
            0.98,
            0.93,
            0.85,
            0.78,
            0.72,
            0.68,
            0.62,
            0.58,
        ],
        [
            0.45,
            0.43,
            0.42,
            0.41,
            0.42,
            0.5,
            0.65,
            0.85,
            1.0,
            1.0,
            0.98,
            0.95,
            0.92,
            0.9,
            0.92,
            0.95,
            0.9,
            0.8,
            0.65,
            0.55,
            0.5,
            0.48,
            0.47,
            0.46,
        ],
        [
            0.6,
            0.58,
            0.57,
            0.56,
            0.58,
            0.7,
            0.85,
            0.95,
            1.0,
            1.0,
            0.98,
            0.97,
            0.95,
            0.93,
            0.94,
            0.96,
            0.94,
            0.92,
            0.88,
            0.82,
            0.75,
            0.7,
            0.65,
            0.62,
        ],
    ]
    demand_profile = daily_profiles[rand(rng, eachindex(daily_profiles))]

    demand = zeros(Float64, n_periods)
    reserve_requirements = zeros(Float64, n_periods)

    base_peak = total_capacity * rand(rng, Uniform(0.55, 0.85))
    day_count = max(1, ceil(Int, n_periods / 24))
    weekly_shape = [rand(rng, Uniform(0.9, 1.1)) for _ in 1:day_count]

    for t in 1:n_periods
        day_index = ceil(Int, t / 24)
        hour = (t - 1) % 24 + 1
        random_effect = rand(rng, Normal(1.0, 0.03))
        raw_demand =
            base_peak *
            demand_profile[hour] *
            weekly_shape[day_index] *
            seasonal_trend *
            random_effect
        demand[t] = max(0.2 * base_peak, raw_demand)
        reserve_requirements[t] = demand[t] * reserve_fraction
    end

    feasible_witness = nothing
    infeasibility_certificate = nothing

    if resolved_status == feasible
        n_u = length(units)
        generation = zeros(Float64, n_u, n_periods)
        commitment = ones(Float64, n_u, n_periods)
        startup = zeros(Float64, n_u, n_periods)
        shutdown = zeros(Float64, n_u, n_periods)
        shape_max = maximum(demand)

        for (u_idx, u) in enumerate(units)
            cap = max_output[u]
            lower = min_output[u]
            previous = lower
            for t in 1:n_periods
                available = cap * availability_factors[u][t]
                if t > 1
                    # A steep stochastic derating can otherwise fall faster than
                    # the unit's ramp-down capability. Raise only that availability
                    # point enough to keep the stored dispatch physically reachable.
                    reachable_floor = max(lower, previous - ramp_down[u])
                    if available + 1e-10 < reachable_floor
                        availability_factors[u][t] = min(1.0, reachable_floor / cap)
                        available = cap * availability_factors[u][t]
                    end
                end

                utilization = 0.34 + 0.38 * demand[t] / max(shape_max, eps())
                target = lower + utilization * (available - lower)
                if t == 1
                    generation[u_idx, t] = clamp(target, lower, available)
                else
                    lo = max(lower, previous - ramp_down[u])
                    hi = min(available, previous + ramp_up[u])
                    generation[u_idx, t] = clamp(target, lo, hi)
                end
                previous = generation[u_idx, t]
            end
            initial_generation[u] = generation[u_idx, 1]
        end

        # Define load from the dispatch so demand balance is exact. Reserve is a
        # realistic percentage of load, capped below the witness's online headroom.
        for t in 1:n_periods
            demand[t] = sum(generation[:, t])
            available = sum(max_output[u] * availability_factors[u][t] for u in units)
            headroom = max(0.0, available - demand[t])
            reserve_requirements[t] = min(reserve_fraction * demand[t], 0.85 * headroom)
        end
        feasible_witness = UnitCommitmentWitness(generation, commitment, startup, shutdown)
    else
        # Retain several operational stress profiles for data diversity, then add
        # one explicit aggregate certificate rather than relying on the scenario to
        # happen to be infeasible.
        scenario = rand(rng, 1:3)
        if scenario == 1
            shortage_factor = rand(rng, Uniform(1.08, 1.22))
            demand .= demand .* shortage_factor
            reserve_requirements .= reserve_requirements .* shortage_factor
        elseif scenario == 2
            affected = sample(
                rng,
                units,
                min(length(units), max(1, round(Int, 0.3 * length(units))));
                replace=false,
            )
            outage_len = max(1, round(Int, 0.2 * n_periods))
            for u in affected
                outage_periods = sample(rng, 1:n_periods, outage_len; replace=false)
                for t in outage_periods
                    availability_factors[u][t] = 0.0
                end
            end
        else
            reserve_requirements .= demand .* rand(rng, Uniform(0.20, 0.35))
        end

        capacity_per_period = [
            sum(max_output[u] * availability_factors[u][t] for u in units) for t in 1:n_periods
        ]
        stress_ratio = [
            (demand[t] + reserve_requirements[t]) / max(capacity_per_period[t], 1.0) for
            t in 1:n_periods
        ]
        critical_period = argmax(stress_ratio)
        available = capacity_per_period[critical_period]
        local_reserve_fraction =
            reserve_requirements[critical_period] / max(demand[critical_period], eps())
        local_reserve_fraction = clamp(local_reserve_fraction, 0.08, 0.35)
        required = if available > 0
            available * (1.0 + 0.35 * local_reserve_fraction)
        else
            max(1.0, demand[critical_period] + reserve_requirements[critical_period])
        end
        demand[critical_period] = required / (1.0 + local_reserve_fraction)
        reserve_requirements[critical_period] = required - demand[critical_period]
        excess = required - available
        infeasibility_certificate = UnitCommitmentCapacityCertificate(
            critical_period, available, required, excess
        )
    end

    problem = UnitCommitmentProblem(
        n_units,
        n_periods,
        units,
        time_periods,
        demand,
        reserve_requirements,
        max_output,
        min_output,
        ramp_up,
        ramp_down,
        variable_costs,
        no_load_costs,
        startup_costs,
        shutdown_costs,
        min_up_times,
        min_down_times,
        availability_factors,
        initial_on,
        initial_generation,
        unit_types,
        resolved_status,
        feasible_witness,
        infeasibility_certificate,
    )
    if resolved_status == feasible
        @assert _unit_commitment_witness_is_valid(problem)
    else
        @assert _unit_commitment_certificate_is_valid(problem)
    end
    return problem
end

"""
    _unit_commitment_witness_is_valid(prob; atol=1e-7)

Check the stored primal witness against every constraint family built by
`build_model`. This is intentionally solver-independent so generation and tests can
audit the feasibility contract directly.
"""
function _unit_commitment_witness_is_valid(prob::UnitCommitmentProblem; atol::Float64=1e-7)
    prob.resolved_status == feasible || return false
    prob.infeasibility_certificate === nothing || return false
    witness = prob.feasible_witness
    witness === nothing && return false

    expected_size = (prob.n_units, prob.n_periods)
    size(witness.generation) == expected_size || return false
    size(witness.commitment) == expected_size || return false
    size(witness.startup) == expected_size || return false
    size(witness.shutdown) == expected_size || return false

    for (u_idx, u) in enumerate(prob.units)
        for t in prob.time_periods
            generation = witness.generation[u_idx, t]
            commitment = witness.commitment[u_idx, t]
            startup = witness.startup[u_idx, t]
            shutdown = witness.shutdown[u_idx, t]
            available = prob.max_output[u] * prob.availability_factors[u][t]

            generation >= -atol || return false
            -atol <= commitment <= 1.0 + atol || return false
            -atol <= startup <= 1.0 + atol || return false
            -atol <= shutdown <= 1.0 + atol || return false
            abs(commitment - round(commitment)) <= atol || return false
            abs(startup - round(startup)) <= atol || return false
            abs(shutdown - round(shutdown)) <= atol || return false
            startup + shutdown <= 1.0 + atol || return false
            generation <= available + atol || return false
            generation <= prob.max_output[u] * commitment + atol || return false
            generation + atol >= prob.min_output[u] * commitment || return false

            if t == first(prob.time_periods)
                generation - prob.initial_generation[u] <=
                prob.ramp_up[u] * prob.initial_on[u] + prob.max_output[u] * startup + atol ||
                    return false
                prob.initial_generation[u] - generation <=
                prob.ramp_down[u] * commitment + prob.max_output[u] * shutdown + atol ||
                    return false
                abs(commitment - prob.initial_on[u] - startup + shutdown) <= atol || return false
            else
                previous = t - 1
                previous_generation = witness.generation[u_idx, previous]
                previous_commitment = witness.commitment[u_idx, previous]
                generation - previous_generation <=
                prob.ramp_up[u] * previous_commitment + prob.max_output[u] * startup + atol ||
                    return false
                previous_generation - generation <=
                prob.ramp_down[u] * commitment + prob.max_output[u] * shutdown + atol ||
                    return false
                abs(commitment - previous_commitment - startup + shutdown) <= atol || return false
            end

            up_start = max(1, t - prob.min_up_times[u] + 1)
            sum(witness.startup[u_idx, j] for j in up_start:t) <= commitment + atol || return false
            down_start = max(1, t - prob.min_down_times[u] + 1)
            sum(witness.shutdown[u_idx, j] for j in down_start:t) <= 1.0 - commitment + atol ||
                return false
        end
    end

    for t in prob.time_periods
        generation = sum(witness.generation[:, t])
        abs(generation - prob.demand[t]) <= atol * max(1.0, prob.demand[t]) || return false
        headroom = sum(
            prob.max_output[u] * prob.availability_factors[u][t] * witness.commitment[u_idx, t] -
            witness.generation[u_idx, t] for (u_idx, u) in enumerate(prob.units)
        )
        headroom + atol >= prob.reserve_requirements[t] || return false
    end
    return true
end

"""
    _unit_commitment_certificate_is_valid(prob; atol=1e-7)

Check the aggregate capacity certificate stored for an infeasible instance.
Demand balance and the reserve row imply
`demand[t] + reserve[t] <= available_capacity[t]`; the certificate records a
strict violation of that necessary condition.
"""
function _unit_commitment_certificate_is_valid(prob::UnitCommitmentProblem; atol::Float64=1e-7)
    prob.resolved_status == infeasible || return false
    prob.feasible_witness === nothing || return false
    certificate = prob.infeasibility_certificate
    certificate === nothing && return false
    1 <= certificate.period <= prob.n_periods || return false

    t = certificate.period
    available = sum(prob.max_output[u] * prob.availability_factors[u][t] for u in prob.units)
    required = prob.demand[t] + prob.reserve_requirements[t]
    isapprox(certificate.available_capacity, available; atol=atol, rtol=1e-10) || return false
    isapprox(certificate.required_capacity, required; atol=atol, rtol=1e-10) || return false
    isapprox(certificate.excess, required - available; atol=atol, rtol=1e-10) || return false
    return certificate.excess > atol
end

"""
    build_model(prob::UnitCommitmentProblem)

Build the natural unit-commitment MILP. Deterministic — uses only data from the
struct fields. The public generation API relaxes the three binary families when
`relax_integer=true` (the package default).

# Returns

  - `model`: The JuMP model
"""
function build_model(prob::UnitCommitmentProblem)
    model = Model()

    units = prob.units
    periods = prob.time_periods

    # Decision variables: 4 * n_units * n_periods total.
    @variable(model, g[u in units, t in periods] >= 0)
    @variable(model, on[u in units, t in periods], Bin)
    @variable(model, startup[u in units, t in periods], Bin)
    @variable(model, shutdown[u in units, t in periods], Bin)

    if prob.feasible_witness !== nothing
        witness = prob.feasible_witness
        for (u_idx, u) in enumerate(units), t in periods
            set_start_value(g[u, t], witness.generation[u_idx, t])
            set_start_value(on[u, t], witness.commitment[u_idx, t])
            set_start_value(startup[u, t], witness.startup[u_idx, t])
            set_start_value(shutdown[u, t], witness.shutdown[u_idx, t])
        end
    end

    @objective(
        model,
        Min,
        sum(
            prob.variable_costs[u] * g[u, t] +
            prob.no_load_costs[u] * on[u, t] +
            prob.startup_costs[u] * startup[u, t] +
            prob.shutdown_costs[u] * shutdown[u, t] for u in units, t in periods
        )
    )

    for u in units
        for (idx, t) in enumerate(periods)
            max_cap = prob.max_output[u] * prob.availability_factors[u][idx]
            @constraint(model, g[u, t] <= max_cap)
            @constraint(model, g[u, t] <= prob.max_output[u] * on[u, t])
            @constraint(model, g[u, t] >= prob.min_output[u] * on[u, t])
        end
    end

    @constraint(model, demand_balance[t in periods], sum(g[u, t] for u in units) == prob.demand[t],)
    @constraint(
        model,
        reserve_requirement[t in periods],
        sum(
            prob.max_output[u] * prob.availability_factors[u][t] * on[u, t] - g[u, t] for u in units
        ) >= prob.reserve_requirements[t],
    )

    for u in units
        for idx in 2:length(periods)
            t = periods[idx]
            prev = periods[idx - 1]
            @constraint(
                model,
                g[u, t] - g[u, prev] <=
                    prob.ramp_up[u] * on[u, prev] + prob.max_output[u] * startup[u, t]
            )
            @constraint(
                model,
                g[u, prev] - g[u, t] <=
                    prob.ramp_down[u] * on[u, t] + prob.max_output[u] * shutdown[u, t]
            )
            @constraint(model, on[u, t] - on[u, prev] == startup[u, t] - shutdown[u, t])
        end

        first_idx = periods[1]
        @constraint(
            model,
            g[u, first_idx] - prob.initial_generation[u] <=
                prob.ramp_up[u] * prob.initial_on[u] + prob.max_output[u] * startup[u, first_idx]
        )
        @constraint(
            model,
            prob.initial_generation[u] - g[u, first_idx] <=
                prob.ramp_down[u] * on[u, first_idx] + prob.max_output[u] * shutdown[u, first_idx]
        )
        @constraint(
            model,
            on[u, first_idx] - prob.initial_on[u] == startup[u, first_idx] - shutdown[u, first_idx]
        )

        for t in periods
            @constraint(model, startup[u, t] + shutdown[u, t] <= 1)
        end

        min_up = prob.min_up_times[u]
        min_down = prob.min_down_times[u]
        for idx in 1:length(periods)
            window_start_up = max(1, idx - min_up + 1)
            if idx - window_start_up + 1 > 0
                @constraint(
                    model,
                    sum(startup[u, periods[j]] for j in window_start_up:idx) <= on[u, periods[idx]]
                )
            end
            window_start_down = max(1, idx - min_down + 1)
            if idx - window_start_down + 1 > 0
                @constraint(
                    model,
                    sum(shutdown[u, periods[j]] for j in window_start_down:idx) <=
                        1 - on[u, periods[idx]]
                )
            end
        end
    end

    return model
end

# Register the variant
register_variant(
    :unit_commitment,
    :standard,
    UnitCommitmentProblem,
    "Unit-commitment MILP (LP-relaxed by default through the package API) with exact demand balance, ramping, reserves, minimum up/down times, and auditable status artifacts",
)
