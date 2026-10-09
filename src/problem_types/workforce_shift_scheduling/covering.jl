using JuMP
using Random

"""
    WorkforceShiftCoveringProblem <: ProblemGenerator

Continuous multi-site, multi-day, multi-skill shift-pattern covering and
staffing problem.

Each decision column assigns workers from one labor pool to one site, one day
of the planning week, one shift pattern and one service skill. Labor pools
belong to a home site; *float* pools (remote agents, seasonal store staff,
contractors) also serve neighbouring sites, coupling the sites' otherwise
separate staffing problems. Cross-trained pools can supply several skills with
skill-specific productivity. Pool-day capacity rows prevent the same workers
from being assigned to several patterns, skills or sites on one day; weekly
pool rows cap the shifts a pool works over the week (rest days). Coverage rows
require effective worker-equivalents in every site-day-period-skill.

The sampled `profile` is stored explicitly and is one of `:contact_center`,
`:retail`, or `:continuous_operations`. Profiles alter the intraday horizon,
period length, skill names, intraday and weekly demand curves, shift lengths,
pool availability, float pools and wage range. There are no undercoverage
variables: unmet demand makes the LP infeasible rather than providing an
always-feasible penalized escape.

For a `feasible` request, `feasible_staffing` stores a planted feasible point
and pool capacities are derived from its usage. It is `nothing` for other
statuses. For `infeasible`, `infeasible_group = (site, day, skill)` and
`infeasibility_capacity_bound` identify an aggregate capacity certificate over
one site-day-skill demand curve; both are `nothing` otherwise. For `unknown`,
capacities and demand receive two-sided market/load shocks and no status is
forced.

# Fields

  - `profile`, `feasibility_status`, `period_minutes`, `n_periods` (intraday
    periods), `n_days` (days in the planning horizon), `n_sites`
  - `day_factors::Vector{Float64}`: weekly demand multiplier per day
  - `skill_names`, `pool_names`, `pool_types`
  - `pool_home_site::Vector{Int}`; `pool_sites::BitMatrix` (pools x sites the
    pool can serve); `pool_days::BitMatrix` (pools x days the pool works)
  - `pattern_starts`, `pattern_span_periods`, `pattern_break_periods`,
    `pattern_wraps`, `pattern_coverage::BitMatrix` (periods x patterns)
  - `pool_qualifications`, `pool_productivity` (pools x skills),
    `pool_availability` (pools x periods), `pattern_eligibility` (pools x patterns)
  - `hourly_wages::Vector{Float64}`
  - `pool_day_capacity::Matrix{Float64}`: workers pool `q` can field on day `d`
    (0 when the pool has no column that day)
  - `pool_week_capacity::Vector{Float64}`: worker-shifts per pool and week
  - `column_pools`, `column_sites`, `column_days`, `column_patterns`,
    `column_skills`, `staffing_costs`
  - `demand::Array{Float64,4}`: worker-equivalents per (site, day, period, skill)
  - `feasible_staffing::Union{Nothing,Vector{Float64}}`
  - `infeasible_group::Union{Nothing,NTuple{3,Int}}`
  - `infeasibility_capacity_bound::Union{Nothing,Float64}`
"""
struct WorkforceShiftCoveringProblem <: ProblemGenerator
    profile::Symbol
    feasibility_status::FeasibilityStatus
    period_minutes::Int
    n_periods::Int
    n_days::Int
    n_sites::Int
    day_factors::Vector{Float64}
    skill_names::Vector{Symbol}
    pool_names::Vector{Symbol}
    pool_types::Vector{Symbol}
    pool_home_site::Vector{Int}
    pool_sites::BitMatrix
    pool_days::BitMatrix
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
    pool_day_capacity::Matrix{Float64}
    pool_week_capacity::Vector{Float64}
    column_pools::Vector{Int}
    column_sites::Vector{Int}
    column_days::Vector{Int}
    column_patterns::Vector{Int}
    column_skills::Vector{Int}
    staffing_costs::Vector{Float64}
    demand::Array{Float64, 4}
    feasible_staffing::Union{Nothing, Vector{Float64}}
    infeasible_group::Union{Nothing, NTuple{3, Int}}
    infeasibility_capacity_bound::Union{Nothing, Float64}
end

const _WORKFORCE_PROFILES = (:contact_center, :retail, :continuous_operations)

function _workforce_profile_spec(profile::Symbol)
    if profile == :contact_center
        return (
            period_minutes=30,
            n_periods=24,
            skill_names=[:general_support, :billing, :technical, :retention],
            shift_spans=[8, 12, 16],
            pool_types=[:core, :early, :late, :part_time, :remote],
            float_types=(:remote,),
            weekend_types=(:part_time,),
            # Monday..Sunday call volume.
            day_factors=[1.08, 1.0, 0.97, 0.98, 0.94, 0.55, 0.42],
            wage_range=(19.0, 34.0),
        )
    elseif profile == :retail
        return (
            period_minutes=60,
            n_periods=14,
            skill_names=[:sales, :checkout, :inventory],
            shift_spans=[4, 6, 8, 10],
            pool_types=[:full_time, :opening, :closing, :part_time, :seasonal],
            float_types=(:seasonal,),
            weekend_types=(:part_time, :seasonal),
            day_factors=[0.78, 0.74, 0.80, 0.88, 1.05, 1.30, 1.02],
            wage_range=(15.0, 25.0),
        )
    elseif profile == :continuous_operations
        return (
            period_minutes=60,
            n_periods=24,
            skill_names=[:operator, :maintenance, :quality, :control_room],
            shift_spans=[6, 8, 10, 12],
            pool_types=[:rotating, :day_crew, :evening_crew, :night_crew, :contractor],
            float_types=(:contractor,),
            weekend_types=(:rotating,),
            day_factors=[1.0, 1.0, 1.0, 1.0, 1.0, 0.93, 0.90],
            wage_range=(27.0, 49.0),
        )
    end
    error("Unknown workforce profile: $profile")
end

function _workforce_skill_count(profile::Symbol, target::Int)
    target < 120 && return 2
    profile == :retail && return 3
    return target < 700 ? 3 : 4
end

"""
    _workforce_horizon(target) -> (n_days, n_sites)

Planning horizon and site count for a variable target: one day for tiny
targets, a full week from about 2,000 columns, and one more site per ~5,000
columns, so coverage rows (`sites * days * periods * skills`) grow in
proportion to the column count.
"""
function _workforce_horizon(target::Int)
    n_days = clamp(round(Int, target / 300), 1, 7)
    n_sites = clamp(round(Int, target / 5000), 1, 1000)
    return n_days, n_sites
end

function _workforce_mark_window!(availability::BitVector, start::Int, width::Int; wrap::Bool=false)
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
        candidate_starts = if profile == :continuous_operations
            collect(1:n_periods)
        else
            collect(1:(n_periods - span + 1))
        end
        for start in candidate_starts
            break_period = 0
            paid_hours = span * spec.period_minutes / 60
            if paid_hours >= 6
                center = max(1, span ÷ 2)
                break_offset = clamp(center + rand(rng, -1:1), 1, span - 2)
                break_period = if profile == :continuous_operations
                    mod1(start + break_offset, n_periods)
                else
                    start + break_offset
                end
            end

            active = falses(n_periods)
            for offset in 0:(span - 1)
                period = if profile == :continuous_operations
                    mod1(start + offset, n_periods)
                else
                    start + offset
                end
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
    rng::AbstractRNG, profile::Symbol, pool_type::Symbol, n_periods::Int, pool_index::Int
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

function _workforce_pattern_eligibility(availability::BitVector, pattern_coverage::BitMatrix)
    n_patterns = size(pattern_coverage, 2)
    eligible = falses(n_patterns)
    for pattern in 1:n_patterns
        eligible[pattern] = all(
            !pattern_coverage[period, pattern] || availability[period] for
            period in axes(pattern_coverage, 1)
        )
    end
    return eligible
end

function _workforce_undesirable_fraction(profile::Symbol, pattern::Int, pattern_coverage::BitMatrix)
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

"""
    _workforce_pool_data(rng, profile, spec, n_skills, n_days, n_sites, home_site, local_index)

Sample one labor pool at `home_site`. `local_index` is the pool's position
among its site's pools: it cycles the pool type, and the first pool of every
site is a flexible anchor (all skills, every period and day) so each site's
skill-period-days stay coverable. Float pool types also serve neighbouring
sites; weekend pool types work the weekend plus a few weekdays.
"""
function _workforce_pool_data(
    rng::AbstractRNG,
    profile::Symbol,
    spec,
    n_skills::Int,
    n_days::Int,
    n_sites::Int,
    home_site::Int,
    local_index::Int,
)
    anchor = local_index == 1
    pool_type = spec.pool_types[mod1(local_index, length(spec.pool_types))]
    availability = if anchor
        trues(spec.n_periods)
    else
        _workforce_pool_availability(rng, profile, pool_type, spec.n_periods, local_index)
    end

    days = falses(n_days)
    if anchor || !(pool_type in spec.weekend_types)
        for d in 1:n_days
            days[d] = anchor || rand(rng) < 0.85
        end
    else
        for d in 1:n_days
            days[d] = d >= 6 || rand(rng) < 0.4
        end
    end
    any(days) || (days[rand(rng, 1:n_days)] = true)

    sites = falses(n_sites)
    sites[home_site] = true
    if pool_type in spec.float_types && n_sites > 1
        sites[mod1(home_site + 1, n_sites)] = true
        rand(rng) < 0.5 && (sites[mod1(home_site - 1, n_sites)] = true)
    end

    qualified = falses(n_skills)
    primary = mod1(local_index, n_skills)
    qualified[primary] = true
    cross_training = pool_type in (:core, :remote, :full_time, :rotating, :contractor) ? 0.58 : 0.28
    for skill in 1:n_skills
        if skill != primary && rand(rng) < cross_training
            qualified[skill] = true
        end
    end
    anchor && (qualified .= true)

    productivity = zeros(Float64, n_skills)
    type_factor = if pool_type in (:remote, :seasonal, :contractor)
        0.92
    elseif pool_type in (:core, :full_time, :rotating)
        1.05
    else
        0.98
    end
    for skill in 1:n_skills
        if qualified[skill]
            primary_factor = skill == primary ? 1.06 : 0.90
            productivity[skill] = round(
                clamp(type_factor * primary_factor * (0.94 + 0.12 * rand(rng)), 0.72, 1.20);
                digits=3,
            )
        end
    end

    low, high = spec.wage_range
    wage = low + (high - low) * rand(rng)
    pool_type in (:remote, :seasonal) && (wage *= 0.94)
    pool_type == :contractor && (wage *= 1.22)
    wage = round(wage; digits=2)
    name = Symbol("$(pool_type)_s$(home_site)_$(local_index)")
    return name, pool_type, qualified, productivity, availability, days, sites, wage
end

"""
    _workforce_pool_candidates(pool, sites, days, eligible, qualified)

All `(pool, site, day, pattern, skill)` staffing columns a pool can offer.
"""
function _workforce_pool_candidates(
    pool::Int, sites::BitVector, days::BitVector, eligible::BitVector, qualified::BitVector
)
    out = NTuple{5, Int}[]
    for site in findall(sites),
        day in findall(days), pattern in findall(eligible),
        skill in findall(qualified)

        push!(out, (pool, site, day, pattern, skill))
    end
    return out
end

"""
    _workforce_select_columns(rng, candidates, pattern_coverage, n_pools, requested)

Choose the staffing columns. First a greedy cover of every period of every
`(site, day, skill)` group, so each coverage row has support; then one column
for any pool still unrepresented; then a uniform sample of the remaining
candidates up to `requested`. Tiny targets are raised to the cover size.
"""
function _workforce_select_columns(
    rng::AbstractRNG,
    candidates::Vector{NTuple{5, Int}},
    pattern_coverage::BitMatrix,
    n_pools::Int,
    requested::Int,
)
    n_periods = size(pattern_coverage, 1)
    groups = Dict{NTuple{3, Int}, Vector{Int}}()
    for (i, (_, site, day, _, skill)) in enumerate(candidates)
        push!(get!(groups, (site, day, skill), Int[]), i)
    end
    taken = falses(length(candidates))
    selected = Int[]
    uncovered = falses(n_periods)
    for key in sort!(collect(keys(groups)))
        members = groups[key]
        uncovered .= true
        while any(uncovered)
            best_score = 0
            ties = Int[]
            for i in members
                taken[i] && continue
                pattern = candidates[i][4]
                score = 0
                for t in 1:n_periods
                    (uncovered[t] && pattern_coverage[t, pattern]) && (score += 1)
                end
                if score > best_score
                    empty!(ties)
                    push!(ties, i)
                    best_score = score
                elseif score == best_score && score > 0
                    push!(ties, i)
                end
            end
            best_score > 0 || error("Generated workforce columns do not cover group $key")
            chosen = rand(rng, ties)
            taken[chosen] = true
            push!(selected, chosen)
            for t in 1:n_periods
                pattern_coverage[t, candidates[chosen][4]] && (uncovered[t] = false)
            end
        end
    end

    has_pool = falses(n_pools)
    for i in selected
        has_pool[candidates[i][1]] = true
    end
    by_pool = [Int[] for _ in 1:n_pools]
    for (i, c) in enumerate(candidates)
        taken[i] || push!(by_pool[c[1]], i)
    end
    for pool in 1:n_pools
        (has_pool[pool] || isempty(by_pool[pool])) && continue
        chosen = rand(rng, by_pool[pool])
        taken[chosen] = true
        push!(selected, chosen)
    end

    remaining = findall(.!taken)
    shuffle!(rng, remaining)
    needed = max(requested, length(selected)) - length(selected)
    needed <= length(remaining) ||
        error("Insufficient distinct workforce columns for target $requested")
    append!(selected, remaining[1:needed])
    shuffle!(rng, selected)
    return candidates[selected]
end

"""
    _workforce_construction_staffing(demand, column_groups, column_pools,
                                     column_patterns, pattern_coverage,
                                     productivity, costs)

Greedily construct a continuous staffing witness, one `(site, day, skill)`
group at a time: repeatedly close the largest relative gap with the column
whose productive, still-needed coverage per unit cost is best, adding 1.5%
slack. Each update can only improve the other periods its shift covers.
"""
function _workforce_construction_staffing(
    demand::Array{Float64, 4},
    groups::Dict{NTuple{3, Int}, Vector{Int}},
    column_pools::Vector{Int},
    column_patterns::Vector{Int},
    pattern_coverage::BitMatrix,
    productivity::Matrix{Float64},
    costs::Vector{Float64},
)
    reference = zeros(Float64, length(column_pools))
    n_periods = size(pattern_coverage, 1)
    covered = zeros(Float64, n_periods)
    gaps = zeros(Float64, n_periods)
    for key in sort!(collect(keys(groups)))
        site, day, skill = key
        options_all = groups[key]
        fill!(covered, 0.0)
        while true
            for t in 1:n_periods
                gaps[t] =
                    max(0.0, demand[site, day, t, skill] - covered[t]) / demand[site, day, t, skill]
            end
            period = argmax(gaps)
            gaps[period] <= 1e-9 && break
            best = 0
            best_score = -Inf
            for column in options_all
                pattern = column_patterns[column]
                pattern_coverage[period, pattern] || continue
                effective = productivity[column_pools[column], skill]
                useful = 0.0
                for t in 1:n_periods
                    pattern_coverage[t, pattern] &&
                        (useful += max(0.0, demand[site, day, t, skill] - covered[t]))
                end
                score = effective * useful / max(costs[column], 1e-9)
                if score > best_score
                    best = column
                    best_score = score
                end
            end
            best == 0 && error("No selected workforce column covers ($key, $period)")
            effective = productivity[column_pools[best], skill]
            addition = 1.015 * (demand[site, day, period, skill] - covered[period]) / effective
            reference[best] += addition
            pattern = column_patterns[best]
            for t in 1:n_periods
                pattern_coverage[t, pattern] && (covered[t] += effective * addition)
            end
        end
    end
    return reference
end

"""
    _workforce_group_capacity_bound(members, day, skill, column_pools, column_patterns,
                                    pattern_coverage, productivity, day_capacity)

Upper bound on the summed coverage rows of one `(site, day, skill)` group: a
pool can field at most `day_capacity[pool, day]` workers that day, each
covering at most its longest paid pattern among the group's columns, at the
pool's productivity for the skill.
"""
function _workforce_group_capacity_bound(
    members::Vector{Int},
    day::Int,
    skill::Int,
    column_pools::Vector{Int},
    column_patterns::Vector{Int},
    pattern_coverage::BitMatrix,
    productivity::Matrix{Float64},
    day_capacity::Matrix{Float64},
)
    longest = Dict{Int, Int}()
    for column in members
        pool = column_pools[column]
        paid = count(view(pattern_coverage, :, column_patterns[column]))
        longest[pool] = max(get(longest, pool, 0), paid)
    end
    return sum(
        day_capacity[pool, day] * productivity[pool, skill] * paid for (pool, paid) in longest
    )
end

function _workforce_demand(
    rng::AbstractRNG, profile::Symbol, spec, n_sites::Int, n_days::Int, n_skills::Int, target::Int
)
    n_periods = spec.n_periods
    demand = zeros(Float64, n_sites, n_days, n_periods, n_skills)
    base_scale = max(5.0, 1.8 * sqrt(max(target, 1) / (n_sites * n_days)))
    for site in 1:n_sites
        site_scale = base_scale * (0.75 + 0.5 * rand(rng))
        # Sites differ in when their peaks fall (time zones, local habits).
        shift = 0.04 * (rand(rng) - 0.5)
        for day in 1:n_days
            day_scale = spec.day_factors[day] * (0.95 + 0.10 * rand(rng))
            for period in 1:n_periods
                x = (period - 0.5) / n_periods + shift
                if profile == :contact_center
                    shape =
                        0.42 +
                        0.85 * exp(-((x - 0.28) / 0.17)^2) +
                        1.05 * exp(-((x - 0.72) / 0.15)^2)
                elseif profile == :retail
                    shape =
                        0.52 +
                        0.35 * exp(-((x - 0.25) / 0.19)^2) +
                        1.10 * exp(-((x - 0.76) / 0.18)^2)
                else
                    shape = 0.82 + 0.13 * sin(2π * x - 0.4) + 0.12 * exp(-((x - 0.55) / 0.20)^2)
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
                    demand[site, day, period, skill] = round(
                        max(
                            0.35, site_scale * day_scale * shape * skill_share * skill_shape * noise
                        );
                        digits=3,
                    )
                end
            end
        end
    end
    return demand
end

"""
    WorkforceShiftCoveringProblem(target_variables, feasibility_status, seed)

Generate a workforce covering LP. The model has one variable for each selected
`(pool, site, day, pattern, skill)` staffing column and no auxiliary variable
block, so the delivered variable count equals `target_variables` for normal
targets (tiny targets may be raised to the minimum number of columns needed to
cover every site-day-period-skill).

# Sizing

The horizon is `clamp(round(target/300), 1, 7)` days and there are
`clamp(round(target/5000), 1, 1000)` sites, so the
`sites * days * periods * skills` coverage rows, plus one capacity row per
pool-day and one weekly row per pool, grow in proportion to the column count
(about 10-20% of it). Pools are added round-robin over sites until the
candidate columns exceed `1.3 * target`.
"""
function WorkforceShiftCoveringProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)
    profile = rand(rng, _WORKFORCE_PROFILES)
    spec = _workforce_profile_spec(profile)
    n_skills = _workforce_skill_count(profile, target)
    skill_names = spec.skill_names[1:n_skills]
    n_days, n_sites = _workforce_horizon(target)
    day_factors = spec.day_factors[1:n_days]

    pattern_starts, pattern_spans, pattern_breaks, pattern_wraps, pattern_coverage = _workforce_patterns(
        rng, profile, spec
    )
    n_patterns = size(pattern_coverage, 2)

    pool_names = Symbol[]
    pool_types = Symbol[]
    pool_home = Int[]
    qualifications = BitVector[]
    productivities = Vector{Float64}[]
    availabilities = BitVector[]
    day_sets = BitVector[]
    site_sets = BitVector[]
    eligibilities = BitVector[]
    hourly_wages = Float64[]
    candidates = NTuple{5, Int}[]
    site_pools = zeros(Int, n_sites)

    min_pools = max(4, n_skills + 1)
    wanted = ceil(Int, 1.3 * target)
    next_site = 1
    while minimum(site_pools) < min_pools || length(candidates) < wanted
        site = next_site
        next_site = mod1(next_site + 1, n_sites)
        site_pools[site] += 1
        name, pool_type, qualified, productivity, availability, days, sites, wage = _workforce_pool_data(
            rng, profile, spec, n_skills, n_days, n_sites, site, site_pools[site]
        )
        eligibility = _workforce_pattern_eligibility(availability, pattern_coverage)
        push!(pool_names, name)
        push!(pool_types, pool_type)
        push!(pool_home, site)
        push!(qualifications, qualified)
        push!(productivities, productivity)
        push!(availabilities, availability)
        push!(day_sets, days)
        push!(site_sets, sites)
        push!(eligibilities, eligibility)
        push!(hourly_wages, wage)
        append!(
            candidates,
            _workforce_pool_candidates(length(pool_names), sites, days, eligibility, qualified),
        )
    end

    n_pools = length(pool_names)
    selected = _workforce_select_columns(rng, candidates, pattern_coverage, n_pools, target)
    column_pools = [c[1] for c in selected]
    column_sites = [c[2] for c in selected]
    column_days = [c[3] for c in selected]
    column_patterns = [c[4] for c in selected]
    column_skills = [c[5] for c in selected]
    n_columns = length(selected)

    pool_qualifications = falses(n_pools, n_skills)
    pool_productivity = zeros(Float64, n_pools, n_skills)
    pool_availability = falses(n_pools, spec.n_periods)
    pattern_eligibility = falses(n_pools, n_patterns)
    pool_sites = falses(n_pools, n_sites)
    pool_days = falses(n_pools, n_days)
    for pool in 1:n_pools
        pool_qualifications[pool, :] .= qualifications[pool]
        pool_productivity[pool, :] .= productivities[pool]
        pool_availability[pool, :] .= availabilities[pool]
        pattern_eligibility[pool, :] .= eligibilities[pool]
        pool_sites[pool, :] .= site_sets[pool]
        pool_days[pool, :] .= day_sets[pool]
    end

    staffing_costs = zeros(Float64, n_columns)
    for column in 1:n_columns
        pool = column_pools[column]
        pattern = column_patterns[column]
        paid_hours = count(view(pattern_coverage, :, pattern)) * spec.period_minutes / 60
        undesirable = _workforce_undesirable_fraction(profile, pattern, pattern_coverage)
        skill_premium = 1.0 + 0.035 * (column_skills[column] - 1)
        weekend_premium = column_days[column] >= 6 ? 1.12 : 1.0
        travel_premium = column_sites[column] == pool_home[pool] ? 1.0 : 1.06
        quote_variation = 0.98 + 0.04 * rand(rng)
        staffing_costs[column] = round(
            hourly_wages[pool] *
            paid_hours *
            (1.0 + 0.30 * undesirable) *
            skill_premium *
            weekend_premium *
            travel_premium *
            quote_variation;
            digits=2,
        )
    end

    demand = _workforce_demand(rng, profile, spec, n_sites, n_days, n_skills, target)
    groups = Dict{NTuple{3, Int}, Vector{Int}}()
    for column in 1:n_columns
        push!(
            get!(groups, (column_sites[column], column_days[column], column_skills[column]), Int[]),
            column,
        )
    end
    construction_staffing = _workforce_construction_staffing(
        demand,
        groups,
        column_pools,
        column_patterns,
        pattern_coverage,
        pool_productivity,
        staffing_costs,
    )
    day_usage = zeros(Float64, n_pools, n_days)
    has_column = falses(n_pools, n_days)
    for column in 1:n_columns
        day_usage[column_pools[column], column_days[column]] += construction_staffing[column]
        has_column[column_pools[column], column_days[column]] = true
    end

    pool_day_capacity = zeros(Float64, n_pools, n_days)
    pool_week_capacity = zeros(Float64, n_pools)
    for pool in 1:n_pools
        for day in 1:n_days
            has_column[pool, day] || continue
            usage = day_usage[pool, day]
            reserve = max(0.35, usage * (0.05 + 0.08 * rand(rng)))
            pool_day_capacity[pool, day] = round(
                max(usage + reserve, 0.75 + 1.75 * rand(rng)); digits=3
            )
        end
        # Rest days: the week allows less than every daily maximum at once.
        week_usage = sum(day_usage[pool, :])
        week_reserve = max(0.35, week_usage * (0.03 + 0.06 * rand(rng)))
        pool_week_capacity[pool] = round(
            min(week_usage + week_reserve + 0.5, sum(pool_day_capacity[pool, :])); digits=3
        )
        pool_week_capacity[pool] = max(pool_week_capacity[pool], round(week_usage + 1e-3; digits=3))
    end

    feasible_staffing = feasibility_status == feasible ? construction_staffing : nothing
    infeasible_group = nothing
    infeasibility_capacity_bound = nothing

    if feasibility_status == unknown
        # A labor-market factor plus independent pool and workload noise; no
        # status is forced. The greedy planted staffing is generous, so the
        # critical uniform capacity factor (the smallest one keeping the LP
        # feasible) was measured at 0.53-0.88 (2k-20k) and 0.58-0.87 (100k); the
        # market factor straddles that range.
        market = 0.60 + 0.50 * rand(rng)
        for pool in 1:n_pools
            pool_shock = market * (0.93 + 0.14 * rand(rng))
            for day in 1:n_days
                pool_day_capacity[pool, day] = round(
                    pool_day_capacity[pool, day] * pool_shock; digits=3
                )
            end
            pool_week_capacity[pool] = round(pool_week_capacity[pool] * pool_shock; digits=3)
        end
        for site in 1:n_sites, day in 1:n_days, period in 1:spec.n_periods
            load_shock = 0.96 + 0.08 * rand(rng)
            for skill in 1:n_skills
                demand[site, day, period, skill] = round(
                    demand[site, day, period, skill] * load_shock * (0.97 + 0.06 * rand(rng));
                    digits=3,
                )
            end
        end
    elseif feasibility_status == infeasible
        # Sum the coverage rows of one (site, day, skill) group. Scale that
        # group's demand curve just above the group's capacity bound.
        best_key = (0, 0, 0)
        best_ratio = Inf
        best_bound = 0.0
        for key in sort!(collect(keys(groups)))
            site, day, skill = key
            bound = _workforce_group_capacity_bound(
                groups[key],
                day,
                skill,
                column_pools,
                column_patterns,
                pattern_coverage,
                pool_productivity,
                pool_day_capacity,
            )
            ratio = bound / sum(demand[site, day, :, skill])
            if ratio < best_ratio
                best_ratio = ratio
                best_key = key
                best_bound = bound
            end
        end
        site, day, skill = best_key
        infeasible_group = best_key
        infeasibility_capacity_bound = best_bound
        required_total = best_bound + max(0.5, 0.03 * best_bound)
        scale = required_total / sum(demand[site, day, :, skill])
        demand[site, day, :, skill] .= round.(demand[site, day, :, skill] .* scale; digits=3)
        # Rounding down could erase a very small strict margin.
        shortfall = required_total - sum(demand[site, day, :, skill])
        if shortfall >= 0
            peak = argmax(demand[site, day, :, skill])
            demand[site, day, peak, skill] += round(shortfall + 0.001; digits=3)
        end
    end

    return WorkforceShiftCoveringProblem(
        profile,
        feasibility_status,
        spec.period_minutes,
        spec.n_periods,
        n_days,
        n_sites,
        day_factors,
        skill_names,
        pool_names,
        pool_types,
        pool_home,
        pool_sites,
        pool_days,
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
        pool_day_capacity,
        pool_week_capacity,
        column_pools,
        column_sites,
        column_days,
        column_patterns,
        column_skills,
        staffing_costs,
        demand,
        feasible_staffing,
        infeasible_group,
        infeasibility_capacity_bound,
    )
end

"""
    _workforce_pool_rows(prob) -> (day_rows, week_rows, single_bounds)

Row structure of the capacity block, shared by `build_model` and the tests.
`day_rows` lists `(pool, day, columns)` for pool-days with at least two
columns, `week_rows` lists `(pool, columns)` for pools working on at least two
days, and `single_bounds` maps a column that is alone in its pool-day to its
upper bound (the pool-day capacity, tightened by the weekly capacity when the
pool works that day only) - a single-column row is emitted as a bound.
"""
function _workforce_pool_rows(prob::WorkforceShiftCoveringProblem)
    n_pools = length(prob.pool_names)
    by_pool_day = Dict{Tuple{Int, Int}, Vector{Int}}()
    for column in eachindex(prob.column_pools)
        push!(
            get!(by_pool_day, (prob.column_pools[column], prob.column_days[column]), Int[]), column
        )
    end
    days_of = [Int[] for _ in 1:n_pools]
    for (pool, day) in keys(by_pool_day)
        push!(days_of[pool], day)
    end
    day_rows = Tuple{Int, Int, Vector{Int}}[]
    single_bounds = Dict{Int, Float64}()
    for key in sort!(collect(keys(by_pool_day)))
        pool, day = key
        columns = by_pool_day[key]
        cap = prob.pool_day_capacity[pool, day]
        length(days_of[pool]) == 1 && (cap = min(cap, prob.pool_week_capacity[pool]))
        if length(columns) >= 2
            push!(day_rows, (pool, day, columns))
        else
            single_bounds[only(columns)] = cap
        end
    end
    week_rows = Tuple{Int, Vector{Int}}[]
    for pool in 1:n_pools
        length(days_of[pool]) >= 2 || continue
        push!(week_rows, (pool, sort!(vcat((by_pool_day[(pool, d)] for d in days_of[pool])...))))
    end
    return day_rows, week_rows, single_bounds
end

"""
    build_model(prob::WorkforceShiftCoveringProblem)

Build the deterministic continuous staffing LP. `assigned_workers[j]` is the
number of workers assigned through staffing column `j`, not a number of hours.

  - `skill_coverage[site, day, period, skill]`: effective staffing >= demand
  - `pool_day_capacity`: workers a pool fields on one day, across all its
    sites, patterns and skills, <= its daily capacity (a pool-day with a
    single column becomes that column's upper bound instead)
  - `pool_week_capacity`: worker-shifts a pool works over the horizon <= its
    weekly capacity (pools working two or more days)
"""
function build_model(prob::WorkforceShiftCoveringProblem)
    model = Model()
    n_columns = length(prob.column_pools)
    n_skills = length(prob.skill_names)

    @variable(model, assigned_workers[1:n_columns] >= 0)
    x = assigned_workers
    @objective(model, Min, sum(prob.staffing_costs[column] * x[column] for column in 1:n_columns))

    day_rows, week_rows, single_bounds = _workforce_pool_rows(prob)
    for column in sort!(collect(keys(single_bounds)))
        set_upper_bound(x[column], single_bounds[column])
    end
    day_refs = [
        @constraint(model, sum(x[c] for c in columns) <= prob.pool_day_capacity[pool, day]) for
        (pool, day, columns) in day_rows
    ]
    week_refs = [
        @constraint(model, sum(x[c] for c in columns) <= prob.pool_week_capacity[pool]) for
        (pool, columns) in week_rows
    ]
    model[:pool_day_capacity] = day_refs
    model[:pool_week_capacity] = week_refs

    dims = (prob.n_sites, prob.n_days, prob.n_periods, n_skills)
    supply = [Int[] for _ in 1:prod(dims)]
    lin = LinearIndices(dims)
    for column in 1:n_columns
        site, day, skill = prob.column_sites[column],
        prob.column_days[column],
        prob.column_skills[column]
        for period in findall(view(prob.pattern_coverage, :, prob.column_patterns[column]))
            push!(supply[lin[site, day, period, skill]], column)
        end
    end
    coverage = Array{ConstraintRef}(undef, dims)
    for site in 1:dims[1], day in 1:dims[2], period in 1:dims[3], skill in 1:dims[4]
        columns = supply[lin[site, day, period, skill]]
        coverage[site, day, period, skill] = @constraint(
            model,
            sum(
                prob.pool_productivity[prob.column_pools[c], skill] * x[c] for c in columns;
                init=0.0,
            ) >= prob.demand[site, day, period, skill]
        )
    end
    model[:skill_coverage] = coverage
    return model
end

register_variant(
    :workforce_shift_scheduling,
    :covering,
    WorkforceShiftCoveringProblem,
    "Multi-site, multi-day, multi-skill shift-pattern covering LP with profile-specific weekly and intraday demand, cross-trained and floating labor pools, breaks, pool-day and weekly capacities, and capacity-certified feasibility controls";
    tags=[:scheduling, :covering],
)
