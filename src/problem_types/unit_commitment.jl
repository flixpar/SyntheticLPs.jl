using JuMP
using Random
using StatsBase
using Distributions

"""
    UnitCommitmentProblem <: ProblemGenerator

Generator for realistic unit commitment power system planning problems that capture
thermal, renewable, and peaking generation fleets across multiple time periods.

Research summary and design plan:
- Realistic unit commitment studies typically model portfolios that mix baseload
  (nuclear/coal), cycling combined-cycle gas, fast-ramping combustion turbines,
  and variable renewable units (wind/solar/hydro). Each category has distinct
  operating characteristics: baseload units carry high minimum stable output,
  long minimum up/down times, and slow ramping. Peakers have small capacities
  but fast ramps and high marginal costs. Renewables exhibit time-varying
  availability profiles with zero fuel cost but limited dispatchability.
- Operational constraints include generation bounds tied to commitment status,
  ramp limits linking consecutive periods, startup/shutdown costs, minimum
  up/down times, and system-wide reserve requirements proportional to demand.
- To stay within ±10% of the target variable count, the generator calibrates the
  number of units and periods based on scale buckets, then iteratively adjusts
  until the estimated variable count (4 variables per unit-period: generation,
  commitment, startup, shutdown) is close to the request.
- Feasibility control is enforced by scaling demand/reserve relative to the
  available capacity profile. Feasible instances guarantee at least a 10% margin
  between capacity and requirement in every period, while infeasible instances
  enforce demand+reserve that exceed available capacity by at least 10% in at
  least one period (often many periods). Unknown feasibility randomly selects
  between these scenarios.
"""
struct UnitCommitmentProblem <: ProblemGenerator
    n_units::Int
    n_periods::Int
    units::Vector{String}
    time_periods::Vector{Int}
    demand::Vector{Float64}
    reserve_requirements::Vector{Float64}
    max_output::Dict{String,Float64}
    min_output::Dict{String,Float64}
    ramp_up::Dict{String,Float64}
    ramp_down::Dict{String,Float64}
    variable_costs::Dict{String,Float64}
    no_load_costs::Dict{String,Float64}
    startup_costs::Dict{String,Float64}
    shutdown_costs::Dict{String,Float64}
    min_up_times::Dict{String,Int}
    min_down_times::Dict{String,Int}
    availability_factors::Dict{String,Vector{Float64}}
    initial_on::Dict{String,Float64}
    initial_generation::Dict{String,Float64}
    solution_status::Symbol
end

function UnitCommitmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    scale = target_variables < 800 ? :small : target_variables < 3000 ? :medium : :large

    if scale == :small
        unit_range = (4, 9)
        period_range = (12, 36)
    elseif scale == :medium
        unit_range = (10, 22)
        period_range = (24, 72)
    else
        unit_range = (20, 48)
        period_range = (48, 168)
    end

    n_units = unit_range[1]
    n_periods = period_range[1]

    for _ in 1:20
        current_vars = n_units * n_periods * 4
        if abs(current_vars - target_variables) / max(target_variables, 1) <= 0.1
            break
        end

        ratio = sqrt(target_variables / max(current_vars, 1))
        if ratio > 1.05
            if n_periods < period_range[2]
                n_periods = min(period_range[2], max(period_range[1], round(Int, n_periods * ratio)))
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
        (; type=:nuclear, weight=0.08, capacity_range=(700.0, 1300.0), min_ratio=(0.6, 0.85),
           ramp_fraction=(0.03, 0.07), var_cost=(10.0, 22.0), no_load=(10000.0, 18000.0),
           startup=(50000.0, 120000.0), shutdown=(10000.0, 25000.0), min_up=(36, 80), min_down=(24, 60)),
        (; type=:coal, weight=0.22, capacity_range=(200.0, 800.0), min_ratio=(0.4, 0.7),
           ramp_fraction=(0.05, 0.12), var_cost=(20.0, 35.0), no_load=(6000.0, 12000.0),
           startup=(20000.0, 70000.0), shutdown=(5000.0, 15000.0), min_up=(12, 48), min_down=(10, 36)),
        (; type=:ccgt, weight=0.28, capacity_range=(150.0, 500.0), min_ratio=(0.35, 0.6),
           ramp_fraction=(0.1, 0.2), var_cost=(28.0, 52.0), no_load=(4000.0, 9000.0),
           startup=(10000.0, 35000.0), shutdown=(3000.0, 10000.0), min_up=(6, 24), min_down=(6, 24)),
        (; type=:gas_ct, weight=0.18, capacity_range=(40.0, 180.0), min_ratio=(0.0, 0.2),
           ramp_fraction=(0.5, 1.0), var_cost=(60.0, 120.0), no_load=(500.0, 2000.0),
           startup=(2000.0, 8000.0), shutdown=(1000.0, 4000.0), min_up=(1, 4), min_down=(1, 4)),
        (; type=:hydro, weight=0.12, capacity_range=(80.0, 400.0), min_ratio=(0.1, 0.4),
           ramp_fraction=(0.3, 0.6), var_cost=(5.0, 18.0), no_load=(1000.0, 4000.0),
           startup=(5000.0, 20000.0), shutdown=(2000.0, 6000.0), min_up=(4, 12), min_down=(2, 10)),
        (; type=:wind, weight=0.12, capacity_range=(50.0, 250.0), min_ratio=(0.0, 0.05),
           ramp_fraction=(0.8, 1.2), var_cost=(0.0, 5.0), no_load=(0.0, 0.0),
           startup=(0.0, 0.0), shutdown=(0.0, 0.0), min_up=(1, 2), min_down=(1, 2))
    ]

    weights = Weights([profile.weight for profile in unit_profiles])

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

    total_capacity = 0.0

    for u in units
        profile = sample(unit_profiles, weights, 1)[1]

        cap = rand(Uniform(profile.capacity_range...))
        max_output[u] = cap
        total_capacity += cap

        min_ratio = rand(Uniform(profile.min_ratio...))
        min_output[u] = cap * min_ratio

        ramp_fraction = rand(Uniform(profile.ramp_fraction...))
        ramp_up[u] = max(5.0, cap * ramp_fraction)
        ramp_down[u] = max(5.0, cap * ramp_fraction * rand(Uniform(0.8, 1.2)))

        variable_costs[u] = rand(Uniform(profile.var_cost...))
        no_load_costs[u] = rand(Uniform(profile.no_load...))
        startup_costs[u] = rand(Uniform(profile.startup...))
        shutdown_costs[u] = rand(Uniform(profile.shutdown...))
        min_up_times[u] = max(1, rand(profile.min_up[1]:profile.min_up[2]))
        min_down_times[u] = max(1, rand(profile.min_down[1]:profile.min_down[2]))

        availability = ones(Float64, n_periods)

        if profile.type in (:nuclear, :coal, :ccgt)
            if rand() < 0.35
                outage_length = max(2, round(Int, rand(Uniform(0.05, 0.15)) * n_periods))
                start_period = rand(1:max(1, n_periods - outage_length + 1))
                for t in start_period:(start_period + outage_length - 1)
                    availability[t] = 0.0
                end
            end
        elseif profile.type == :gas_ct
            for t in eachindex(availability)
                if rand() < 0.03
                    availability[t] = 0.0
                end
            end
        elseif profile.type == :hydro
            seasonality = rand(Uniform(0.6, 1.0))
            for t in eachindex(availability)
                hour = (t - 1) % 24 + 1
                pattern = hour in 7:20 ? 1.0 : 0.8
                availability[t] = pattern * seasonality
            end
        else
            base_speed = rand(Uniform(0.4, 0.8))
            gust_factor = rand(Uniform(0.1, 0.35))
            for t in eachindex(availability)
                hour = (t - 1) % 24 + 1
                diurnal = 0.4 + 0.6 * sin(2π * hour / 24)
                noise = rand(Uniform(-gust_factor, gust_factor))
                availability[t] = clamp(base_speed + diurnal * gust_factor + noise, 0.0, 1.0)
            end
        end
        availability_factors[u] = availability

        if profile.type in (:nuclear, :coal)
            initial_on[u] = rand() < 0.8 ? 1.0 : 0.0
        elseif profile.type == :wind
            initial_on[u] = 1.0
        else
            initial_on[u] = rand() < 0.6 ? 1.0 : 0.0
        end

        if initial_on[u] > 0.5
            availability_first = availability[1]
            initial_generation[u] = availability_first == 0.0 ? 0.0 : rand(Uniform(min_output[u], cap * availability_first))
        else
            initial_generation[u] = 0.0
        end
    end

    # Demand pattern
    seasonal_trend = rand(Uniform(0.85, 1.15))
    reserve_fraction = rand(Uniform(0.08, 0.18))

    daily_profiles = [
        [0.55, 0.5, 0.48, 0.47, 0.5, 0.6, 0.75, 0.9, 0.95, 1.0, 0.98, 0.95, 0.92, 0.94, 0.97, 1.0, 0.98, 0.93, 0.85, 0.78, 0.72, 0.68, 0.62, 0.58],
        [0.45, 0.43, 0.42, 0.41, 0.42, 0.5, 0.65, 0.85, 1.0, 1.0, 0.98, 0.95, 0.92, 0.9, 0.92, 0.95, 0.9, 0.8, 0.65, 0.55, 0.5, 0.48, 0.47, 0.46],
        [0.6, 0.58, 0.57, 0.56, 0.58, 0.7, 0.85, 0.95, 1.0, 1.0, 0.98, 0.97, 0.95, 0.93, 0.94, 0.96, 0.94, 0.92, 0.88, 0.82, 0.75, 0.7, 0.65, 0.62]
    ]
    profile = daily_profiles[rand(1:length(daily_profiles))]

    demand = zeros(Float64, n_periods)
    reserve_requirements = zeros(Float64, n_periods)

    base_peak = total_capacity * rand(Uniform(0.55, 0.85))
    day_count = max(1, ceil(Int, n_periods / 24))
    weekly_shape = [rand(Uniform(0.9, 1.1)) for _ in 1:day_count]

    for t in 1:n_periods
        day_index = ceil(Int, t / 24)
        hour = (t - 1) % 24 + 1
        random_effect = rand(Normal(1.0, 0.03))
        raw_demand = base_peak * profile[hour] * weekly_shape[day_index] * seasonal_trend * random_effect
        demand[t] = max(0.2 * base_peak, raw_demand)
        reserve_requirements[t] = demand[t] * reserve_fraction
    end

    solution_status = feasibility_status == feasible ? :feasible :
                      feasibility_status == infeasible ? :infeasible : :all
    actual_status = solution_status == :all ? (rand() < 0.6 ? :feasible : :infeasible) : solution_status

    capacity_per_period = [sum(max_output[u] * availability_factors[u][t] for u in units) for t in 1:n_periods]

    required = demand .+ reserve_requirements

    if actual_status == :feasible
        ratios = [required[t] / max(capacity_per_period[t], 1.0) for t in 1:n_periods]
        max_ratio = maximum(ratios)
        if max_ratio > 0.85 && max_ratio > 0
            scale_factor = 0.85 / max_ratio
            demand .= demand .* scale_factor
            reserve_requirements .= reserve_requirements .* scale_factor
            required = demand .+ reserve_requirements
        end

        min_margin = minimum(capacity_per_period[t] - required[t] for t in 1:n_periods)
        if min_margin < 0.1 * mean(demand)
            adjust = (0.12 * mean(demand) - min_margin) / maximum(capacity_per_period)
            for u in units
                scaling = 1 + adjust
                max_output[u] *= scaling
                min_output[u] *= scaling
                ramp_up[u] *= scaling
                ramp_down[u] *= scaling
            end
            capacity_per_period = [sum(max_output[u] * availability_factors[u][t] for u in units) for t in 1:n_periods]
        end
    else
        scenario = rand(1:3)
        if scenario == 1
            # Demand spike
            shortage_factor = rand(Uniform(1.12, 1.3))
            demand .= demand .* shortage_factor
            reserve_requirements .= reserve_requirements .* shortage_factor
        elseif scenario == 2
            # Forced outages on mid-merit units
            affected = sample(units, min(length(units), max(1, round(Int, 0.3 * length(units)))), replace=false)
            outage_len = max(1, round(Int, 0.2 * n_periods))
            for u in affected
                outage_periods = sample(1:n_periods, outage_len; replace=false)
                for t in outage_periods
                    availability_factors[u][t] = 0.0
                end
            end
        else
            # Reserve tightening
            reserve_requirements .= demand .* rand(Uniform(0.3, 0.5))
        end
        required = demand .+ reserve_requirements
        capacity_per_period = [sum(max_output[u] * availability_factors[u][t] for u in units) for t in 1:n_periods]
        min_deficit = minimum(capacity_per_period[t] - required[t] for t in 1:n_periods)
        if min_deficit > -0.1 * mean(demand)
            max_required, idx = findmax(required)
            limiting_capacity = capacity_per_period[idx]
            if limiting_capacity <= 0.0
                limiting_capacity = maximum(capacity_per_period)
            end
            desired_requirement = max(limiting_capacity, 1.0) * rand(Uniform(1.12, 1.3))
            shortage_multiplier = desired_requirement / max(max_required, 1.0)
            demand .= demand .* shortage_multiplier
            reserve_requirements .= reserve_requirements .* shortage_multiplier
        end
    end

    capacity_per_period = [sum(max_output[u] * availability_factors[u][t] for u in units) for t in 1:n_periods]
    required = demand .+ reserve_requirements
    ratios = [required[t] / max(capacity_per_period[t], 1.0) for t in 1:n_periods]

    if actual_status == :feasible
        max_ratio = maximum(ratios)
        if max_ratio > 0.9 && max_ratio > 0
            scale_factor = 0.9 / max_ratio
            demand .= demand .* scale_factor
            reserve_requirements .= reserve_requirements .* scale_factor
        end
    else
        max_ratio = maximum(ratios)
        if max_ratio < 1.1
            safe_ratio = max(max_ratio, 1e-6)
            scale_factor = 1.1 / safe_ratio
            demand .= demand .* scale_factor
            reserve_requirements .= reserve_requirements .* scale_factor
        end
    end

    return UnitCommitmentProblem(
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
        actual_status
    )
end

function build_model(prob::UnitCommitmentProblem)
    model = Model()

    units = prob.units
    periods = prob.time_periods

    @variable(model, g[u in units, t in periods] >= 0)
    @variable(model, 0 <= on[u in units, t in periods] <= 1)
    @variable(model, 0 <= startup[u in units, t in periods] <= 1)
    @variable(model, 0 <= shutdown[u in units, t in periods] <= 1)

    @objective(model, Min,
        sum(prob.variable_costs[u] * g[u,t] +
            prob.no_load_costs[u] * on[u,t] +
            prob.startup_costs[u] * startup[u,t] +
            prob.shutdown_costs[u] * shutdown[u,t]
            for u in units, t in periods)
    )

    for u in units
        for (idx, t) in enumerate(periods)
            max_cap = prob.max_output[u] * prob.availability_factors[u][idx]
            @constraint(model, g[u,t] <= max_cap)
            @constraint(model, g[u,t] <= prob.max_output[u] * on[u,t])
            @constraint(model, g[u,t] >= prob.min_output[u] * on[u,t])
        end
    end

    for (idx, t) in enumerate(periods)
        @constraint(model, sum(g[u,t] for u in units) >= prob.demand[idx])
        @constraint(model,
            sum(prob.max_output[u] * prob.availability_factors[u][idx] * on[u,t] - g[u,t] for u in units) >=
            prob.reserve_requirements[idx]
        )
    end

    for u in units
        for idx in 2:length(periods)
            t = periods[idx]
            prev = periods[idx-1]
            @constraint(model,
                g[u,t] - g[u,prev] <= prob.ramp_up[u] * on[u,prev] + prob.max_output[u] * startup[u,t]
            )
            @constraint(model,
                g[u,prev] - g[u,t] <= prob.ramp_down[u] * on[u,t] + prob.max_output[u] * shutdown[u,t]
            )
            @constraint(model, on[u,t] - on[u,prev] == startup[u,t] - shutdown[u,t])
        end

        first_idx = periods[1]
        @constraint(model,
            g[u,first_idx] - prob.initial_generation[u] <= prob.ramp_up[u] * prob.initial_on[u] + prob.max_output[u] * startup[u,first_idx]
        )
        @constraint(model,
            prob.initial_generation[u] - g[u,first_idx] <= prob.ramp_down[u] * on[u,first_idx] + prob.max_output[u] * shutdown[u,first_idx]
        )
        @constraint(model, on[u,first_idx] - prob.initial_on[u] == startup[u,first_idx] - shutdown[u,first_idx])

        min_up = prob.min_up_times[u]
        min_down = prob.min_down_times[u]
        for idx in 1:length(periods)
            window_start_up = max(1, idx - min_up + 1)
            if idx - window_start_up + 1 > 0
                @constraint(model, sum(startup[u,periods[j]] for j in window_start_up:idx) <= on[u,periods[idx]])
            end
            window_start_down = max(1, idx - min_down + 1)
            if idx - window_start_down + 1 > 0
                @constraint(model, sum(shutdown[u,periods[j]] for j in window_start_down:idx) <= 1 - on[u,periods[idx]])
            end
        end
    end

    return model
end

register_problem(
    :unit_commitment,
    UnitCommitmentProblem,
    "Unit commitment power system planning problem with ramping, startup/shutdown, reserves, and realistic generator portfolios"
)
