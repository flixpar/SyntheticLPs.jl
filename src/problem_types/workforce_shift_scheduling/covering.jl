using JuMP
using Random

"""
    WorkforceShiftCoveringProblem <: ProblemGenerator

Continuous multi-skill shift-pattern covering and staffing problem.

Each decision column assigns workers from one labor pool to one shift pattern
and one service skill. Cross-trained pools can supply several skills, with
skill-specific productivity. Pool-capacity rows prevent the same workers from
being assigned to several patterns or skills. Coverage rows require effective
worker-equivalents in every skill-period.

The sampled `profile` is stored explicitly and is one of `:contact_center`,
`:retail`, or `:continuous_operations`. Profiles alter the horizon, period
length, skill names, demand curve, shift lengths, pool availability, and wage
range. There are no undercoverage variables: unmet demand makes the LP
infeasible rather than providing an always-feasible penalized escape.

For a `feasible` request, `feasible_staffing` stores a planted feasible point
and pool capacities are derived from its usage. It is `nothing` for other
statuses. For `infeasible`, `infeasible_skill` and
`infeasibility_capacity_bound` identify the aggregate skill-capacity
certificate; both are `nothing` otherwise. For `unknown`, capacities and
demand receive independent market/load shocks and no status is forced.
"""
struct WorkforceShiftCoveringProblem <: ProblemGenerator
    profile::Symbol
    feasibility_status::FeasibilityStatus
    period_minutes::Int
    n_periods::Int
    skill_names::Vector{Symbol}
    pool_names::Vector{Symbol}
    pool_types::Vector{Symbol}
    pattern_starts::Vector{Int}
    pattern_span_periods::Vector{Int}
    pattern_break_periods::Vector{Int}
    pattern_wraps::BitVector
    pattern_coverage::BitMatrix
    pool_qualifications::BitMatrix
    pool_productivity::Matrix{Float64}
    pool_availability::BitMatrix
    pattern_eligibility::BitMatrix
    hourly_wages::Vector{Float64}
    pool_capacities::Vector{Float64}
    column_pools::Vector{Int}
    column_patterns::Vector{Int}
    column_skills::Vector{Int}
    staffing_costs::Vector{Float64}
    demand::Matrix{Float64}
    feasible_staffing::Union{Nothing,Vector{Float64}}
    infeasible_skill::Union{Nothing,Int}
    infeasibility_capacity_bound::Union{Nothing,Float64}
end

const _WORKFORCE_PROFILES = (
    :contact_center,
    :retail,
    :continuous_operations,
)

function _workforce_profile_spec(profile::Symbol)
    if profile == :contact_center
        return (
            period_minutes=30,
            n_periods=24,
            skill_names=[:general_support, :billing, :technical, :retention],
            shift_spans=[8, 12, 16],
            pool_types=[:core, :early, :late, :part_time, :remote],
            wage_range=(19.0, 34.0),
        )
    elseif profile == :retail
        return (
            period_minutes=60,
            n_periods=14,
            skill_names=[:sales, :checkout, :inventory],
            shift_spans=[4, 6, 8, 10],
            pool_types=[:full_time, :opening, :closing, :part_time, :seasonal],
            wage_range=(15.0, 25.0),
        )
    elseif profile == :continuous_operations
        return (
            period_minutes=60,
            n_periods=24,
            skill_names=[:operator, :maintenance, :quality, :control_room],
            shift_spans=[6, 8, 10, 12],
            pool_types=[:rotating, :day_crew, :evening_crew, :night_crew, :contractor],
            wage_range=(27.0, 49.0),
        )
    end
    error("Unknown workforce profile: $profile")
end

function _workforce_skill_count(profile::Symbol, target::Int)
    target < 120 && return 2
    if profile == :retail
        return target < 700 ? 3 : 3
    end
    return target < 700 ? 3 : 4
end

function _workforce_mark_window!(
    availability::BitVector,
    start::Int,
    width::Int;
    wrap::Bool=false,
)
    n_periods = length(availability)
    for offset in 0:(width - 1)
        period = start + offset
        if wrap
            availability[mod1(period, n_periods)] = true
        elseif period <= n_periods
            availability[period] = true
        end
    end
    return availability
end

# Build realistic contiguous spans. A break is a hole inside the span, not a
# paid coverage period. Pattern supports are deduplicated so a labor pool never
# receives two structurally identical columns for the same skill.
function _workforce_patterns(rng::AbstractRNG, profile::Symbol, spec)
    n_periods = spec.n_periods
    supports = BitVector[]
    starts = Int[]
    spans = Int[]
    breaks = Int[]
    wraps = Bool[]
    seen = Set{Tuple}()

    for span in spec.shift_spans
        candidate_starts = profile == :continuous_operations ?
                           collect(1:n_periods) :
                           collect(1:(n_periods - span + 1))
        for start in candidate_starts
            break_period = 0
            paid_hours = span * spec.period_minutes / 60
            if paid_hours >= 6
                center = max(1, span ÷ 2)
                break_offset = clamp(center + rand(rng, -1:1), 1, span - 2)
                break_period = profile == :continuous_operations ?
                               mod1(start + break_offset, n_periods) :
                               start + break_offset
            end

            active = falses(n_periods)
            for offset in 0:(span - 1)
                period = profile == :continuous_operations ?
                         mod1(start + offset, n_periods) :
                         start + offset
                period == break_period || (active[period] = true)
            end
            signature = Tuple(findall(active))
            signature in seen && continue
            push!(seen, signature)
            push!(supports, active)
            push!(starts, start)
            push!(spans, span)
            push!(breaks, break_period)
            push!(wraps, start + span - 1 > n_periods)
        end
    end

    # Every profile has both short and break-bearing shifts. Continuous
    # operations additionally has starts around the clock and therefore
    # wraparound/night patterns.
    n_patterns = length(supports)
    coverage = falses(n_periods, n_patterns)
    for pattern in 1:n_patterns
        coverage[:, pattern] .= supports[pattern]
    end
    return starts, spans, breaks, BitVector(wraps), coverage
end

function _workforce_pool_availability(
    rng::AbstractRNG,
    profile::Symbol,
    pool_type::Symbol,
    n_periods::Int,
    pool_index::Int,
)
    availability = falses(n_periods)
    if pool_type in (:core, :remote, :full_time, :seasonal, :rotating, :contractor)
        availability .= true
        # Most flexible pools remain fully available; occasional pools have a
        # short unavailability window, producing different pattern menus.
        if pool_index % 3 == 0
            blocked = rand(rng, 1:n_periods)
            availability[blocked] = false
        end
    elseif pool_type in (:early, :opening, :day_crew)
        width = max(8, round(Int, 0.70 * n_periods))
        _workforce_mark_window!(availability, 1, width)
    elseif pool_type in (:late, :closing, :evening_crew)
        width = max(8, round(Int, 0.70 * n_periods))
        _workforce_mark_window!(availability, n_periods - width + 1, width)
    elseif pool_type == :night_crew
        width = 12
        _workforce_mark_window!(availability, 19, width; wrap=true)
    else
        width = max(6, round(Int, rand(rng, 0.42:0.04:0.62) * n_periods))
        start = rand(rng, 1:max(1, n_periods - width + 1))
        _workforce_mark_window!(availability, start, width)
    end
    return availability
end

function _workforce_pool_data(
    rng::AbstractRNG,
    profile::Symbol,
    spec,
    n_skills::Int,
    pool_index::Int,
)
    type_index = mod1(pool_index, length(spec.pool_types))
    pool_type = spec.pool_types[type_index]
    pool_name = Symbol("$(pool_type)_$(pool_index)")
    availability = _workforce_pool_availability(
        rng, profile, pool_type, spec.n_periods, pool_index,
    )

    qualified = falses(n_skills)
    primary = mod1(pool_index, n_skills)
    qualified[primary] = true
    cross_training = pool_type in (:core, :remote, :full_time, :rotating, :contractor) ?
                     0.58 : 0.28
    for skill in 1:n_skills
        if skill != primary && rand(rng) < cross_training
            qualified[skill] = true
        end
    end
    # A flexible anchor pool makes every skill coverable, while the remaining
    # pools retain distinct qualification sets.
    pool_index == 1 && (qualified .= true)

    productivity = zeros(Float64, n_skills)
    type_factor = pool_type in (:remote, :seasonal, :contractor) ? 0.92 :
                  pool_type in (:core, :full_time, :rotating) ? 1.05 : 0.98
    for skill in 1:n_skills
        if qualified[skill]
            primary_factor = skill == primary ? 1.06 : 0.90
            productivity[skill] = round(
                clamp(type_factor * primary_factor * (0.94 + 0.12 * rand(rng)),
                      0.72, 1.20),
                digits=3,
            )
        end
    end

    low, high = spec.wage_range
    wage = low + (high - low) * rand(rng)
    pool_type in (:remote, :seasonal) && (wage *= 0.94)
    pool_type == :contractor && (wage *= 1.22)
    wage = round(wage, digits=2)
    return pool_name, pool_type, qualified, productivity, availability, wage
end

function _workforce_pattern_eligibility(
    availability::BitVector,
    pattern_coverage::BitMatrix,
)
    n_patterns = size(pattern_coverage, 2)
    eligible = falses(n_patterns)
    for pattern in 1:n_patterns
        eligible[pattern] = all(
            !pattern_coverage[period, pattern] || availability[period]
            for period in axes(pattern_coverage, 1)
        )
    end
    return eligible
end

function _workforce_candidate_columns(
    qualifications::Vector{BitVector},
    eligibility::Vector{BitVector},
)
    candidates = NTuple{3,Int}[]
    for pool in eachindex(qualifications)
        for pattern in eachindex(eligibility[pool])
            eligibility[pool][pattern] || continue
            for skill in eachindex(qualifications[pool])
                qualifications[pool][skill] &&
                    push!(candidates, (pool, pattern, skill))
            end
        end
    end
    return candidates
end

function _workforce_select_columns(
    rng::AbstractRNG,
    candidates::Vector{NTuple{3,Int}},
    pattern_coverage::BitMatrix,
    n_pools::Int,
    n_skills::Int,
    requested::Int,
)
    selected = NTuple{3,Int}[]
    selected_set = Set{NTuple{3,Int}}()
    n_periods = size(pattern_coverage, 1)

    # First choose a compact cover for every skill-period. This guarantees
    # nonempty row support even for small target sizes.
    for skill in 1:n_skills
        uncovered = trues(n_periods)
        while any(uncovered)
            best_score = -1
            ties = NTuple{3,Int}[]
            for column in candidates
                column[3] == skill || continue
                column in selected_set && continue
                score = count(
                    period -> uncovered[period] &&
                              pattern_coverage[period, column[2]],
                    1:n_periods,
                )
                if score > best_score
                    empty!(ties)
                    push!(ties, column)
                    best_score = score
                elseif score == best_score
                    push!(ties, column)
                end
            end
            best_score > 0 ||
                error("Generated workforce columns do not cover skill $skill")
            chosen = rand(rng, ties)
            push!(selected, chosen)
            push!(selected_set, chosen)
            for period in 1:n_periods
                pattern_coverage[period, chosen[2]] &&
                    (uncovered[period] = false)
            end
        end
    end

    # Keep generated pools meaningful by including at least one decision
    # column from each pool when the target permits it.
    for pool in 1:n_pools
        any(column -> column[1] == pool, selected) && continue
        options = [column for column in candidates
                   if column[1] == pool && !(column in selected_set)]
        isempty(options) && continue
        chosen = rand(rng, options)
        push!(selected, chosen)
        push!(selected_set, chosen)
    end

    delivered = max(requested, length(selected))
    remaining = [column for column in candidates if !(column in selected_set)]
    shuffle!(rng, remaining)
    needed = delivered - length(selected)
    needed <= length(remaining) ||
        error("Insufficient distinct workforce columns for target $requested")
    append!(selected, remaining[1:needed])
    shuffle!(rng, selected)
    return selected
end

function _workforce_undesirable_fraction(
    profile::Symbol,
    pattern::Int,
    pattern_coverage::BitMatrix,
)
    periods = findall(pattern_coverage[:, pattern])
    isempty(periods) && return 0.0
    n_periods = size(pattern_coverage, 1)
    undesirable = if profile == :contact_center
        count(period -> period <= 2 || period >= n_periods - 2, periods)
    elseif profile == :retail
        count(period -> period >= n_periods - 3, periods)
    else
        count(period -> period <= 6 || period >= 19, periods)
    end
    return undesirable / length(periods)
end

function _workforce_demand(
    rng::AbstractRNG,
    profile::Symbol,
    n_periods::Int,
    n_skills::Int,
    target::Int,
)
    demand = zeros(Float64, n_periods, n_skills)
    workforce_scale = max(5.0, 1.8 * sqrt(max(target, 1)))
    for period in 1:n_periods
        x = (period - 0.5) / n_periods
        if profile == :contact_center
            shape = 0.42 +
                    0.85 * exp(-((x - 0.28) / 0.17)^2) +
                    1.05 * exp(-((x - 0.72) / 0.15)^2)
        elseif profile == :retail
            shape = 0.52 +
                    0.35 * exp(-((x - 0.25) / 0.19)^2) +
                    1.10 * exp(-((x - 0.76) / 0.18)^2)
        else
            shape = 0.82 +
                    0.13 * sin(2π * x - 0.4) +
                    0.12 * exp(-((x - 0.55) / 0.20)^2)
        end
        for skill in 1:n_skills
            skill_share = if profile == :contact_center
                (0.50, 0.22, 0.18, 0.10)[skill]
            elseif profile == :retail
                (0.46, 0.36, 0.18)[skill]
            else
                (0.44, 0.22, 0.20, 0.14)[skill]
            end
            skill_shape = 1.0
            if profile == :retail && skill == min(2, n_skills)
                skill_shape += 0.40 * exp(-((x - 0.78) / 0.14)^2)
            elseif profile == :continuous_operations && skill == min(2, n_skills)
                skill_shape += 0.55 * exp(-((x - 0.50) / 0.16)^2)
            elseif profile == :contact_center && skill == min(3, n_skills)
                skill_shape += 0.30 * exp(-((x - 0.68) / 0.18)^2)
            end
            noise = 0.92 + 0.16 * rand(rng)
            demand[period, skill] = round(
                max(0.35, workforce_scale * shape * skill_share *
                          skill_shape * noise),
                digits=3,
            )
        end
    end
    return demand
end

# Greedily construct a continuous staffing witness for the generated demand.
# Each update closes the largest remaining gap and can only improve other
# periods covered by the same shift.
function _workforce_construction_staffing(
    demand::Matrix{Float64},
    column_pools::Vector{Int},
    column_patterns::Vector{Int},
    column_skills::Vector{Int},
    pattern_coverage::BitMatrix,
    productivity::Matrix{Float64},
    costs::Vector{Float64},
)
    n_columns = length(column_pools)
    reference = zeros(Float64, n_columns)
    covered = zeros(Float64, size(demand))
    n_periods, n_skills = size(demand)

    for skill in 1:n_skills
        while true
            relative_gaps = [
                max(0.0, demand[period, skill] - covered[period, skill]) /
                demand[period, skill]
                for period in 1:n_periods
            ]
            period = argmax(relative_gaps)
            relative_gaps[period] <= 1e-9 && break
            options = [
                column for column in 1:n_columns
                if column_skills[column] == skill &&
                   pattern_coverage[period, column_patterns[column]]
            ]
            isempty(options) &&
                error("No selected workforce column covers ($period, $skill)")
            best = options[1]
            best_score = -Inf
            for column in options
                pool = column_pools[column]
                pattern = column_patterns[column]
                effective = productivity[pool, skill]
                useful = sum(
                    max(0.0, demand[t, skill] - covered[t, skill])
                    for t in 1:n_periods if pattern_coverage[t, pattern]
                )
                score = effective * useful / max(costs[column], 1e-9)
                if score > best_score
                    best = column
                    best_score = score
                end
            end
            pool = column_pools[best]
            pattern = column_patterns[best]
            effective = productivity[pool, skill]
            addition = 1.015 *
                       (demand[period, skill] - covered[period, skill]) /
                       effective
            reference[best] += addition
            for t in 1:n_periods
                if pattern_coverage[t, pattern]
                    covered[t, skill] += effective * addition
                end
            end
        end
    end
    return reference
end

function _workforce_skill_capacity_bound(
    skill::Int,
    column_pools::Vector{Int},
    column_patterns::Vector{Int},
    column_skills::Vector{Int},
    pattern_coverage::BitMatrix,
    productivity::Matrix{Float64},
    pool_capacities::Vector{Float64},
)
    bound = 0.0
    for pool in eachindex(pool_capacities)
        max_paid_periods = 0
        for column in eachindex(column_pools)
            if column_pools[column] == pool && column_skills[column] == skill
                paid_periods = count(pattern_coverage[:, column_patterns[column]])
                max_paid_periods = max(max_paid_periods, paid_periods)
            end
        end
        bound += pool_capacities[pool] * productivity[pool, skill] *
                 max_paid_periods
    end
    return bound
end

"""
    WorkforceShiftCoveringProblem(target_variables, feasibility_status, seed)

Generate a workforce covering LP. The model has one variable for each selected
`(pool, pattern, skill)` staffing column and no auxiliary variable block, so the
delivered variable count equals `target_variables` for normal targets (small
targets may be raised to the minimum number of columns needed to cover every
skill-period).
"""
function WorkforceShiftCoveringProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)
    profile = rand(rng, _WORKFORCE_PROFILES)
    spec = _workforce_profile_spec(profile)
    n_skills = _workforce_skill_count(profile, target)
    skill_names = spec.skill_names[1:n_skills]

    pattern_starts, pattern_spans, pattern_breaks, pattern_wraps,
        pattern_coverage = _workforce_patterns(rng, profile, spec)
    n_patterns = size(pattern_coverage, 2)

    pool_names = Symbol[]
    pool_types = Symbol[]
    qualifications = BitVector[]
    productivities = Vector{Float64}[]
    availabilities = BitVector[]
    eligibilities = BitVector[]
    hourly_wages = Float64[]

    candidates = NTuple{3,Int}[]
    min_pools = max(4, n_skills + 1)
    while length(pool_names) < min_pools || length(candidates) < target
        pool_index = length(pool_names) + 1
        name, pool_type, qualified, productivity, availability, wage =
            _workforce_pool_data(rng, profile, spec, n_skills, pool_index)
        eligibility = _workforce_pattern_eligibility(
            availability, pattern_coverage,
        )
        push!(pool_names, name)
        push!(pool_types, pool_type)
        push!(qualifications, qualified)
        push!(productivities, productivity)
        push!(availabilities, availability)
        push!(eligibilities, eligibility)
        push!(hourly_wages, wage)
        candidates = _workforce_candidate_columns(
            qualifications, eligibilities,
        )
    end

    n_pools = length(pool_names)
    selected = _workforce_select_columns(
        rng, candidates, pattern_coverage, n_pools, n_skills, target,
    )
    column_pools = [column[1] for column in selected]
    column_patterns = [column[2] for column in selected]
    column_skills = [column[3] for column in selected]

    pool_qualifications = falses(n_pools, n_skills)
    pool_productivity = zeros(Float64, n_pools, n_skills)
    pool_availability = falses(n_pools, spec.n_periods)
    pattern_eligibility = falses(n_pools, n_patterns)
    for pool in 1:n_pools
        pool_qualifications[pool, :] .= qualifications[pool]
        pool_productivity[pool, :] .= productivities[pool]
        pool_availability[pool, :] .= availabilities[pool]
        pattern_eligibility[pool, :] .= eligibilities[pool]
    end

    staffing_costs = zeros(Float64, length(selected))
    for column in eachindex(selected)
        pool = column_pools[column]
        pattern = column_patterns[column]
        skill = column_skills[column]
        paid_periods = count(pattern_coverage[:, pattern])
        paid_hours = paid_periods * spec.period_minutes / 60
        undesirable = _workforce_undesirable_fraction(
            profile, pattern, pattern_coverage,
        )
        skill_premium = 1.0 + 0.035 * (skill - 1)
        quote_variation = 0.98 + 0.04 * rand(rng)
        staffing_costs[column] = round(
            hourly_wages[pool] * paid_hours * (1.0 + 0.30 * undesirable) *
            skill_premium * quote_variation,
            digits=2,
        )
    end

    demand = _workforce_demand(
        rng, profile, spec.n_periods, n_skills, target,
    )
    construction_staffing = _workforce_construction_staffing(
        demand, column_pools, column_patterns, column_skills,
        pattern_coverage, pool_productivity, staffing_costs,
    )
    construction_usage = zeros(Float64, n_pools)
    for column in eachindex(construction_staffing)
        construction_usage[column_pools[column]] += construction_staffing[column]
    end

    pool_capacities = zeros(Float64, n_pools)
    for pool in 1:n_pools
        reserve = max(0.35, construction_usage[pool] * (0.05 + 0.08 * rand(rng)))
        pool_capacities[pool] = round(
            max(construction_usage[pool] + reserve, 0.75 + 1.75 * rand(rng)),
            digits=3,
        )
    end

    feasible_staffing = feasibility_status == feasible ?
                        construction_staffing : nothing
    infeasible_skill = nothing
    infeasibility_capacity_bound = nothing

    if feasibility_status == unknown
        # Independent labor-market and workload shocks deliberately make no
        # feasibility claim.
        for pool in 1:n_pools
            pool_capacities[pool] = round(
                pool_capacities[pool] * (0.72 + 0.48 * rand(rng)),
                digits=3,
            )
        end
        for period in 1:spec.n_periods
            load_shock = 0.88 + 0.27 * rand(rng)
            for skill in 1:n_skills
                demand[period, skill] = round(
                    demand[period, skill] * load_shock *
                    (0.95 + 0.10 * rand(rng)),
                    digits=3,
                )
            end
        end
    elseif feasibility_status == infeasible
        # Sum all coverage rows for one skill. A pool can assign at most its
        # capacity to one eligible pattern, so its total contribution is at
        # most capacity × productivity × longest paid pattern. Scale the
        # skill's whole demand curve just above the smallest such bound ratio.
        bounds = [
            _workforce_skill_capacity_bound(
                skill, column_pools, column_patterns, column_skills,
                pattern_coverage, pool_productivity, pool_capacities,
            )
            for skill in 1:n_skills
        ]
        ratios = [bounds[skill] / sum(demand[:, skill])
                  for skill in 1:n_skills]
        certificate_skill = argmin(ratios)
        infeasible_skill = certificate_skill
        infeasibility_capacity_bound = bounds[certificate_skill]
        required_total = bounds[certificate_skill] +
                         max(0.5, 0.03 * bounds[certificate_skill])
        scale = required_total / sum(demand[:, certificate_skill])
        demand[:, certificate_skill] .=
            round.(demand[:, certificate_skill] .* scale, digits=3)
        # Rounding down could erase a very small strict margin.
        shortfall = required_total - sum(demand[:, certificate_skill])
        shortfall >= 0 &&
            (demand[argmax(demand[:, certificate_skill]), certificate_skill] +=
             round(shortfall + 0.001, digits=3))
    end

    return WorkforceShiftCoveringProblem(
        profile,
        feasibility_status,
        spec.period_minutes,
        spec.n_periods,
        skill_names,
        pool_names,
        pool_types,
        pattern_starts,
        pattern_spans,
        pattern_breaks,
        pattern_wraps,
        pattern_coverage,
        pool_qualifications,
        pool_productivity,
        pool_availability,
        pattern_eligibility,
        hourly_wages,
        pool_capacities,
        column_pools,
        column_patterns,
        column_skills,
        staffing_costs,
        demand,
        feasible_staffing,
        infeasible_skill,
        infeasibility_capacity_bound,
    )
end

"""
    build_model(prob::WorkforceShiftCoveringProblem)

Build the deterministic continuous staffing LP. `assigned_workers[j]` is the
number of workers assigned through staffing column `j`, not a number of hours.
"""
function build_model(prob::WorkforceShiftCoveringProblem)
    model = Model()
    n_columns = length(prob.column_pools)
    n_pools = length(prob.pool_names)
    n_skills = length(prob.skill_names)

    @variable(model, assigned_workers[1:n_columns] >= 0)
    @objective(
        model,
        Min,
        sum(prob.staffing_costs[column] * assigned_workers[column]
            for column in 1:n_columns),
    )

    @constraint(
        model,
        pool_capacity[pool=1:n_pools],
        sum(assigned_workers[column] for column in 1:n_columns
            if prob.column_pools[column] == pool) <=
        prob.pool_capacities[pool],
    )

    @constraint(
        model,
        skill_coverage[period=1:prob.n_periods, skill=1:n_skills],
        sum(
            prob.pool_productivity[prob.column_pools[column], skill] *
            assigned_workers[column]
            for column in 1:n_columns
            if prob.column_skills[column] == skill &&
               prob.pattern_coverage[period, prob.column_patterns[column]]
        ) >= prob.demand[period, skill],
    )
    return model
end

register_variant(
    :workforce_shift_scheduling,
    :covering,
    WorkforceShiftCoveringProblem,
    "Multi-skill shift-pattern covering LP with profile-specific demand, cross-trained labor pools, breaks, availability, and capacity-certified feasibility controls",
)
