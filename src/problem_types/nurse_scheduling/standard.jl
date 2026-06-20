using JuMP
using Random
using Distributions

"""
    NurseSchedulingProblem <: ProblemGenerator

Generator for nurse scheduling problems with realistic labor-contract rules.

# Overview
Models the assignment of nurses to shifts across a planning horizon. This is the
**LP relaxation** of the nurse-rostering MIP: the natural model uses binary
assignments, but here the decision variables are continuous assignment fractions
`x[n, d, s]` in `[0, 1]` indicating the extent to which nurse `n` works shift `s`
on day `d`. The objective minimizes total labor cost (with night/weekend/penalty
multipliers). As an LP it is a realistic, structurally rich workforce-scheduling
*relaxation* (fractional rosters), not a directly implementable integer roster.

Constraints capture a rich set of labor rules:
- Shift coverage: each (day, shift) must be staffed to its demand level.
- Skill mix: enough qualified nurses must cover specialty skill requirements.
- Availability: nurses can only be assigned to shifts they are available for.
- One shift per day per nurse.
- Per-nurse min/max total shifts over the horizon.
- Per-nurse weekend shift bounds.
- Per-nurse night-shift limits.
- Maximum consecutive working days.
- Mandatory rest (no early shifts) after a night shift.

# Fields
- `n_nurses::Int`: Number of nurses
- `n_days::Int`: Number of days in the planning horizon
- `n_shifts::Int`: Number of shift types per day
- `total_shifts::Int`: `n_days * n_shifts` (shift slots per nurse)
- `shift_labels::Vector{Symbol}`: Label for each shift type (e.g. `:day`, `:night`)
- `demand::Matrix{Int}`: Required coverage per (day, shift)
- `skill_requirements::Array{Int,3}`: Required qualified nurses per (day, shift, skill)
- `availability::Array{Int,3}`: 1 if nurse `n` is available for (day, shift), else 0
- `min_shifts::Vector{Int}`: Minimum total shifts per nurse
- `max_shifts::Vector{Int}`: Maximum total shifts per nurse
- `weekend_bounds::Vector{Tuple{Int,Int}}`: (lower, upper) weekend shift bounds per nurse
- `night_limits::Vector{Int}`: Maximum night shifts per nurse
- `max_consecutive_days::Vector{Int}`: Maximum consecutive working days per nurse
- `rest_after_night::Vector{Int}`: Required rest days (no early shifts) after a night shift
- `nurse_skills::Matrix{Int}`: 1 if nurse `n` has skill `k`, else 0
- `nurse_types::Vector{Symbol}`: Contract type per nurse (`:core`, `:float_pool`, etc.)
- `costs::Array{Float64,3}`: Cost of assigning nurse `n` to (day, shift)
- `weekend_days::Vector{Int}`: Indices of weekend days in the horizon
- `feasibility_status::FeasibilityStatus`: Resolved feasibility status of the instance
"""
struct NurseSchedulingProblem <: ProblemGenerator
    n_nurses::Int
    n_days::Int
    n_shifts::Int
    total_shifts::Int
    shift_labels::Vector{Symbol}
    demand::Matrix{Int}
    skill_requirements::Array{Int,3}
    availability::Array{Int,3}
    min_shifts::Vector{Int}
    max_shifts::Vector{Int}
    weekend_bounds::Vector{Tuple{Int,Int}}
    night_limits::Vector{Int}
    max_consecutive_days::Vector{Int}
    rest_after_night::Vector{Int}
    nurse_skills::Matrix{Int}
    nurse_types::Vector{Symbol}
    costs::Array{Float64,3}
    weekend_days::Vector{Int}
    feasibility_status::FeasibilityStatus
end

const NURSE_SHIFT_ALIASES = Dict(
    1 => [:day],
    2 => [:day, :night],
    3 => [:day, :evening, :night],
    4 => [:day, :swing, :evening, :night],
)

is_nurse_weekend(day::Int) = mod1(day, 7) in (6, 7)

# Choose (n_nurses, n_days, n_shifts) so that n_nurses * n_days * n_shifts ~ target.
function select_nurse_dimensions(target_variables::Int)
    target = max(target_variables, 1)
    low = max(1, floor(Int, 0.9 * target))
    high = ceil(Int, 1.1 * target)
    best = nothing
    best_diff = typemax(Int)
    for days in 1:56
        for shifts in 1:4
            for nurses in 2:2000
                actual = nurses * days * shifts
                diff = abs(actual - target)
                if diff < best_diff
                    best = (nurses, days, shifts, actual)
                    best_diff = diff
                end
                if actual >= low && actual <= high
                    return nurses, days, shifts
                end
            end
        end
    end
    nurses, days, shifts, _ = best
    return nurses, days, shifts
end

function build_nurse_shift_labels(n_shifts::Int)
    if haskey(NURSE_SHIFT_ALIASES, n_shifts)
        return copy(NURSE_SHIFT_ALIASES[n_shifts])
    end
    labels = [:day, :swing, :evening]
    push!(labels, :night)
    while length(labels) < n_shifts
        push!(labels, Symbol("shift$(length(labels)+1)"))
    end
    return labels[1:n_shifts]
end

function sample_nurse_types(n_nurses::Int, scenario::Symbol)
    probs = scenario == :small ? [0.55, 0.18, 0.17, 0.10] :
            scenario == :medium ? [0.5, 0.2, 0.2, 0.1] :
            [0.48, 0.27, 0.18, 0.07]
    types = [:core, :float_pool, :part_time, :day_only]
    assignments = Vector{Symbol}(undef, n_nurses)
    for n in 1:n_nurses
        r = rand()
        cumulative = 0.0
        for (idx, p) in enumerate(probs)
            cumulative += p
            if r <= cumulative || idx == length(probs)
                assignments[n] = types[idx]
                break
            end
        end
    end
    return assignments
end

function sample_nurse_base_rates(nurse_types::Vector{Symbol}, scenario::Symbol)
    base_range = scenario == :small ? (32.0, 45.0) :
                 scenario == :medium ? (35.0, 52.0) :
                 (38.0, 60.0)
    rates = zeros(Float64, length(nurse_types))
    for (idx, t) in enumerate(nurse_types)
        premium = t == :float_pool ? 1.08 : t == :part_time ? 0.95 : 1.0
        rates[idx] = rand(Uniform(base_range[1], base_range[2])) * premium
    end
    return rates
end

function sample_nurse_skill_matrix(n_nurses::Int, n_skills::Int, scenario::Symbol)
    skills = zeros(Int, n_nurses, n_skills)
    skills[:, 1] .= 1  # every nurse has the base skill
    base_prob = scenario == :small ? 0.25 : scenario == :medium ? 0.32 : 0.4
    for n in 1:n_nurses
        for k in 2:n_skills
            prob = base_prob * rand(Uniform(0.8, 1.2))
            if rand() < min(0.95, prob)
                skills[n, k] = 1
            end
        end
        if n_skills > 1 && all(skills[n, 2:end] .== 0)
            idx = rand(2:n_skills)
            skills[n, idx] = 1
        end
    end
    for k in 2:n_skills
        if all(skills[:, k] .== 0)
            idx = rand(1:n_nurses)
            skills[idx, k] = 1
        end
    end
    return skills
end

function build_nurse_availability(
    n_nurses::Int, n_days::Int, n_shifts::Int,
    shift_labels::Vector{Symbol}, nurse_types::Vector{Symbol},
)
    availability = zeros(Int, n_nurses, n_days, n_shifts)
    for n in 1:n_nurses
        base_density = rand(Beta(7, 2))
        for d in 1:n_days
            weekend_flag = is_nurse_weekend(d)
            for s in 1:n_shifts
                label = shift_labels[s]
                prob = label == :day ? 0.85 : label == :evening || label == :swing ? 0.65 : 0.42
                prob *= base_density
                if weekend_flag
                    if nurse_types[n] == :core
                        prob *= 0.85
                    elseif nurse_types[n] == :float_pool
                        prob *= 1.1
                    else
                        prob *= 0.95
                    end
                end
                if nurse_types[n] == :day_only && label != :day
                    prob *= 0.1
                elseif nurse_types[n] == :part_time
                    prob *= 0.8
                end
                prob = clamp(prob, 0.02, 0.98)
                availability[n, d, s] = rand() < prob ? 1 : 0
            end
        end
    end
    # Guarantee at least 2 available nurses per shift slot
    for d in 1:n_days, s in 1:n_shifts
        if sum(availability[:, d, s]) < 2
            needed = 2 - sum(availability[:, d, s])
            idxs = randperm(n_nurses)[1:needed]
            for idx in idxs
                availability[idx, d, s] = 1
            end
        end
    end
    return availability
end

function build_base_nurse_demand(
    n_days::Int, n_shifts::Int, n_nurses::Int,
    shift_labels::Vector{Symbol}, scenario::Symbol,
)
    demand = zeros(Int, n_days, n_shifts)
    avg_ratio = scenario == :small ? 0.35 : scenario == :medium ? 0.42 : 0.5
    for d in 1:n_days
        season = 0.9 + 0.2 * sin(2π * d / max(7, n_days))
        weekend_factor = is_nurse_weekend(d) ? 0.95 : 1.05
        for s in 1:n_shifts
            label = shift_labels[s]
            shift_factor = label == :day ? 1.1 : label == :night ? 0.7 : 0.9
            base = n_nurses * avg_ratio * season * weekend_factor * shift_factor
            noise = rand(Uniform(0.85, 1.15))
            demand[d, s] = max(1, round(Int, base * noise))
        end
        total_day = sum(demand[d, :])
        if total_day > n_nurses
            scale = n_nurses / total_day
            demand[d, :] .= max.(1, round.(Int, demand[d, :] .* scale))
        end
    end
    return demand
end

function select_nurse(candidates::Vector{Int}, assigned_total::Vector{Int}, targets::Vector{Int})
    best_score = -typemax(Int)
    best = candidates[1]
    for n in candidates
        score = targets[n] - assigned_total[n]
        if score == best_score
            if assigned_total[n] < assigned_total[best]
                best = n
            elseif assigned_total[n] == assigned_total[best] && rand() < 0.5
                best = n
            end
        elseif score > best_score
            best_score = score
            best = n
        end
    end
    return best
end

# Heuristically build a feasible assignment pattern that respects availability,
# consecutive-day limits, night cooldowns and night qualifications. The resulting
# pattern is used to set demand/skill/labor parameters so that a feasible point
# provably exists.
function build_nurse_assignments(
    base_demand::Matrix{Int},
    availability::Array{Int,3},
    shift_labels::Vector{Symbol},
    weekend_days::Vector{Int},
    night_qualified::Vector{Bool},
    max_consec::Vector{Int},
    rest_after_night::Vector{Int},
    target_totals::Vector{Int},
)
    n_nurses, n_days, n_shifts = size(availability)
    assignments = zeros(Int, n_nurses, n_days, n_shifts)
    assigned_total = zeros(Int, n_nurses)
    night_counts = zeros(Int, n_nurses)
    weekend_counts = zeros(Int, n_nurses)
    consecutive = zeros(Int, n_nurses)
    worked_prev_day = falses(n_nurses)
    night_cooldown = zeros(Int, n_nurses)
    weekend_set = Set(weekend_days)
    for d in 1:n_days
        worked_today = falses(n_nurses)
        for n in 1:n_nurses
            if night_cooldown[n] > 0
                night_cooldown[n] -= 1
            end
        end
        weekend_flag = d in weekend_set
        for s in 1:n_shifts
            req = max(1, base_demand[d, s])
            assigned = 0
            attempts = 0
            while assigned < req && attempts < 6 * n_nurses
                eligible = Int[]
                for n in 1:n_nurses
                    if availability[n, d, s] == 0 || worked_today[n]
                        continue
                    end
                    if worked_prev_day[n] && consecutive[n] >= max_consec[n]
                        continue
                    end
                    if night_cooldown[n] > 0 && s == 1
                        continue
                    end
                    if shift_labels[s] == :night && !night_qualified[n]
                        continue
                    end
                    push!(eligible, n)
                end
                if isempty(eligible)
                    break
                end
                chosen = select_nurse(eligible, assigned_total, target_totals)
                assignments[chosen, d, s] = 1
                worked_today[chosen] = true
                assigned_total[chosen] += 1
                if shift_labels[s] == :night
                    night_counts[chosen] += 1
                    night_cooldown[chosen] = rest_after_night[chosen]
                end
                if weekend_flag
                    weekend_counts[chosen] += 1
                end
                assigned += 1
                attempts += 1
            end
            if assigned == 0
                fallback = Int[]
                for n in 1:n_nurses
                    if availability[n, d, s] == 1 && !worked_today[n]
                        if night_cooldown[n] > 0 && s == 1
                            continue
                        end
                        if shift_labels[s] == :night && !night_qualified[n]
                            continue
                        end
                        push!(fallback, n)
                    end
                end
                if !isempty(fallback)
                    chosen = select_nurse(fallback, assigned_total, target_totals)
                    assignments[chosen, d, s] = 1
                    worked_today[chosen] = true
                    assigned_total[chosen] += 1
                    if shift_labels[s] == :night
                        night_counts[chosen] += 1
                        night_cooldown[chosen] = rest_after_night[chosen]
                    end
                    if weekend_flag
                        weekend_counts[chosen] += 1
                    end
                end
            end
        end
        for n in 1:n_nurses
            if worked_today[n]
                consecutive[n] = worked_prev_day[n] ? min(max_consec[n], consecutive[n] + 1) : 1
            else
                consecutive[n] = 0
            end
            worked_prev_day[n] = worked_today[n]
        end
    end
    return assignments, assigned_total, night_counts, weekend_counts
end

# Set demand slightly below the achieved coverage so the heuristic pattern is feasible.
function finalize_nurse_demand(assignments::Array{Int,3})
    _, n_days, n_shifts = size(assignments)
    demand = zeros(Int, n_days, n_shifts)
    for d in 1:n_days, s in 1:n_shifts
        coverage = max(1, sum(assignments[:, d, s]))
        slack = rand(Uniform(0.85, 0.98))
        demand[d, s] = max(1, min(coverage, round(Int, coverage * slack)))
    end
    return demand
end

# Skill requirements are capped at the number of skilled nurses actually assigned,
# so the heuristic pattern stays feasible. Skill 1 (base) is implied by the coverage
# constraint and is left at the demand level (build_model skips it to avoid duplication).
function compute_nurse_skill_requirements(
    demand::Matrix{Int},
    assignments::Array{Int,3},
    nurse_skills::Matrix{Int},
    scenario::Symbol,
)
    n_days, n_shifts = size(demand)
    n_skills = size(nurse_skills, 2)
    requirements = zeros(Int, n_days, n_shifts, n_skills)
    ratios = scenario == :small ? [1.0, 0.2, 0.12, 0.08] :
             scenario == :medium ? [1.0, 0.25, 0.18, 0.12] :
             [1.0, 0.3, 0.22, 0.15]
    for d in 1:n_days, s in 1:n_shifts
        requirements[d, s, 1] = demand[d, s]
        for k in 2:n_skills
            desired = round(Int, demand[d, s] * ratios[k])
            skilled = 0
            for n in 1:size(nurse_skills, 1)
                if assignments[n, d, s] == 1 && nurse_skills[n, k] == 1
                    skilled += 1
                end
            end
            requirements[d, s, k] = min(desired, skilled)
        end
    end
    return requirements
end

function build_nurse_cost_tensor(
    base_rates::Vector{Float64},
    n_days::Int,
    n_shifts::Int,
    shift_labels::Vector{Symbol},
    weekend_days::Vector{Int},
    nurse_types::Vector{Symbol},
)
    costs = zeros(Float64, length(base_rates), n_days, n_shifts)
    weekend_set = Set(weekend_days)
    for n in 1:length(base_rates)
        for d in 1:n_days
            weekend_flag = d in weekend_set
            for s in 1:n_shifts
                label = shift_labels[s]
                shift_mult = label == :night ? 1.28 : label == :evening || label == :swing ? 1.12 : 1.0
                weekend_mult = weekend_flag ? 1.08 : 1.0
                penalty = 1.0
                if nurse_types[n] == :part_time && label == :night
                    penalty += 0.4
                elseif nurse_types[n] == :day_only && label != :day
                    penalty += 0.5
                end
                costs[n, d, s] = base_rates[n] * shift_mult * weekend_mult * penalty
            end
        end
    end
    return costs
end

function build_nurse_weekend_bounds(weekend_counts::Vector{Int})
    bounds = Vector{Tuple{Int,Int}}(undef, length(weekend_counts))
    for n in 1:length(weekend_counts)
        lower = max(0, weekend_counts[n] - 1)
        upper = weekend_counts[n] + 1
        bounds[n] = (lower, upper)
    end
    return bounds
end

function build_nurse_target_totals(n_nurses::Int, n_days::Int, nurse_types::Vector{Symbol})
    targets = zeros(Int, n_nurses)
    for n in 1:n_nurses
        ratio = nurse_types[n] == :core ? rand(Uniform(0.65, 0.9)) :
                nurse_types[n] == :float_pool ? rand(Uniform(0.55, 0.8)) :
                nurse_types[n] == :part_time ? rand(Uniform(0.3, 0.6)) :
                rand(Uniform(0.4, 0.55))
        targets[n] = max(1, min(n_days, round(Int, ratio * n_days)))
    end
    return targets
end

function build_nurse_night_qualification(nurse_types::Vector{Symbol}, shift_labels::Vector{Symbol})
    has_night = any(label -> label == :night, shift_labels)
    quals = [has_night && nurse_types[n] != :day_only && rand() < 0.8 for n in 1:length(nurse_types)]
    if has_night && all(!q for q in quals)
        idx = findfirst(t -> t != :day_only, nurse_types)
        if idx === nothing
            idx = 1
        end
        quals[idx] = true
    end
    return quals
end

function build_nurse_consecutive_limits(n_nurses::Int, n_days::Int, nurse_types::Vector{Symbol})
    limits = zeros(Int, n_nurses)
    for n in 1:n_nurses
        base = nurse_types[n] == :core ? rand(3:5) : nurse_types[n] == :float_pool ? rand(2:4) : rand(2:3)
        limits[n] = min(max(2, base), n_days)
    end
    return limits
end

function build_nurse_rest_requirements(n_nurses::Int, night_qualified::Vector{Bool})
    rest = zeros(Int, n_nurses)
    for n in 1:n_nurses
        rest[n] = night_qualified[n] ? rand(1:2) : 0
    end
    return rest
end

function observed_nurse_consecutive_days(assignments::Array{Int,3})
    n_nurses, n_days, _ = size(assignments)
    observed = zeros(Int, n_nurses)
    for n in 1:n_nurses
        current = 0
        best = 0
        for d in 1:n_days
            worked = sum(assignments[n, d, :]) > 0
            if worked
                current += 1
                best = max(best, current)
            else
                current = 0
            end
        end
        observed[n] = best
    end
    return observed
end

function finalize_nurse_shift_bounds(assigned_total::Vector{Int}, n_days::Int)
    min_shifts = Vector{Int}(undef, length(assigned_total))
    max_shifts = Vector{Int}(undef, length(assigned_total))
    for n in 1:length(assigned_total)
        min_shifts[n] = max(0, round(Int, assigned_total[n] * 0.7))
        buffer = max(1, round(Int, n_days * 0.15))
        max_shifts[n] = min(n_days, assigned_total[n] + buffer)
        if max_shifts[n] < min_shifts[n] + 1
            max_shifts[n] = min(n_days, min_shifts[n] + 1)
        end
    end
    return min_shifts, max_shifts
end

function build_nurse_night_limits(night_counts::Vector{Int}, night_qualified::Vector{Bool})
    limits = zeros(Int, length(night_counts))
    for n in 1:length(night_counts)
        if night_qualified[n]
            limits[n] = night_counts[n] + (night_counts[n] == 0 ? 1 : rand(0:1))
        else
            limits[n] = 0
        end
    end
    return limits
end

# Force a deterministic contradiction: require more nurses of a specialty skill on
# some (day, shift) than exist in the entire roster.
function inject_nurse_infeasibility!(
    demand::Matrix{Int},
    skill_requirements::Array{Int,3},
    nurse_skills::Matrix{Int},
)
    n_days, n_shifts = size(demand)
    n_skills = size(nurse_skills, 2)
    day = rand(1:n_days)
    shift = rand(1:n_shifts)
    skill = n_skills > 1 ? rand(2:n_skills) : 1
    available = sum(nurse_skills[:, skill])
    skill_requirements[day, shift, skill] = available + 1
    demand[day, shift] = min(size(nurse_skills, 1), demand[day, shift] + max(1, available))
end

"""
    NurseSchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a nurse scheduling problem instance.

The model has exactly one decision-variable block, `x[1:n_nurses, 1:n_days, 1:n_shifts]`,
so the variable count is `n_nurses * n_days * n_shifts`. Dimensions are chosen by
`select_nurse_dimensions` to land within ~10% of `target_variables`.

For `feasible` (and `unknown` resolved to feasible) instances, a heuristic schedule
respecting availability, consecutive-day limits and night rules is built first, and
the demand / skill / labor parameters are then derived from that achieved schedule so
a feasible point provably exists. For `infeasible` instances a specialty-skill
requirement is forced above the number of qualified nurses, creating a guaranteed
contradiction.

# Arguments
- `target_variables`: Target number of variables (`n_nurses * n_days * n_shifts`)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function NurseSchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    actual_status = feasibility_status == unknown ? (rand() < 0.5 ? feasible : infeasible) : feasibility_status

    n_nurses, n_days, n_shifts = select_nurse_dimensions(target_variables)
    total_shifts = n_days * n_shifts
    actual_vars = n_nurses * total_shifts

    shift_labels = build_nurse_shift_labels(n_shifts)
    weekend_days = [d for d in 1:n_days if is_nurse_weekend(d)]
    scenario = actual_vars <= 600 ? :small : actual_vars <= 4000 ? :medium : :large

    nurse_types = sample_nurse_types(n_nurses, scenario)
    base_rates = sample_nurse_base_rates(nurse_types, scenario)
    nurse_skills = sample_nurse_skill_matrix(n_nurses, scenario == :large ? 4 : 3, scenario)
    night_qualified = build_nurse_night_qualification(nurse_types, shift_labels)
    rest_after_night = build_nurse_rest_requirements(n_nurses, night_qualified)
    max_consecutive_days = build_nurse_consecutive_limits(n_nurses, n_days, nurse_types)
    availability = build_nurse_availability(n_nurses, n_days, n_shifts, shift_labels, nurse_types)
    base_demand = build_base_nurse_demand(n_days, n_shifts, n_nurses, shift_labels, scenario)
    target_totals = build_nurse_target_totals(n_nurses, n_days, nurse_types)

    assignments, assigned_total, night_counts, weekend_counts = build_nurse_assignments(
        base_demand,
        availability,
        shift_labels,
        weekend_days,
        night_qualified,
        max_consecutive_days,
        rest_after_night,
        target_totals,
    )

    observed = observed_nurse_consecutive_days(assignments)
    max_consecutive_days = max.(max_consecutive_days, observed)

    demand = finalize_nurse_demand(assignments)
    skill_requirements = compute_nurse_skill_requirements(demand, assignments, nurse_skills, scenario)
    min_shifts, max_shifts = finalize_nurse_shift_bounds(assigned_total, n_days)
    weekend_bounds = build_nurse_weekend_bounds(weekend_counts)
    night_limits = build_nurse_night_limits(night_counts, night_qualified)
    costs = build_nurse_cost_tensor(base_rates, n_days, n_shifts, shift_labels, weekend_days, nurse_types)

    if actual_status == infeasible
        inject_nurse_infeasibility!(demand, skill_requirements, nurse_skills)
    end

    return NurseSchedulingProblem(
        n_nurses,
        n_days,
        n_shifts,
        total_shifts,
        shift_labels,
        demand,
        skill_requirements,
        availability,
        min_shifts,
        max_shifts,
        weekend_bounds,
        night_limits,
        max_consecutive_days,
        rest_after_night,
        nurse_skills,
        nurse_types,
        costs,
        weekend_days,
        actual_status,
    )
end

"""
    build_model(prob::NurseSchedulingProblem)

Build a JuMP model for the nurse scheduling problem. Deterministic — uses only data
from the struct fields.

The single variable block `x[1:n_nurses, 1:n_days, 1:n_shifts] ∈ [0, 1]` gives a
variable count of `n_nurses * n_days * n_shifts`.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::NurseSchedulingProblem)
    model = Model()

    n_nurses = prob.n_nurses
    n_days = prob.n_days
    n_shifts = prob.n_shifts
    shift_labels = prob.shift_labels
    night_idx = findfirst(label -> lowercase(String(label)) == "night", shift_labels)

    # Early shifts that must be rested after a night shift.
    early_indices = Int[]
    if n_shifts >= 1
        push!(early_indices, 1)
    end
    if n_shifts >= 3
        push!(early_indices, 2)
    end

    # Decision variables: assignment fraction of nurse n to (day d, shift s).
    @variable(model, 0 <= x[1:n_nurses, 1:n_days, 1:n_shifts] <= 1)

    # Objective: minimize total labor cost.
    @objective(model, Min, sum(prob.costs[n, d, s] * x[n, d, s] for n in 1:n_nurses, d in 1:n_days, s in 1:n_shifts))

    # Coverage and skill-mix constraints.
    n_skills = size(prob.nurse_skills, 2)
    for d in 1:n_days, s in 1:n_shifts
        @constraint(model, sum(x[n, d, s] for n in 1:n_nurses) >= prob.demand[d, s])
        # Skip skill 1 (base): every nurse has it, so its constraint duplicates coverage.
        for k in 2:n_skills
            req = prob.skill_requirements[d, s, k]
            if req > 0
                @constraint(model, sum(prob.nurse_skills[n, k] * x[n, d, s] for n in 1:n_nurses) >= req)
            end
        end
    end

    # Availability: nurses can only be assigned to shifts they are available for.
    # (When available, the [0,1] variable bound already caps the value at 1.)
    for n in 1:n_nurses, d in 1:n_days, s in 1:n_shifts
        if prob.availability[n, d, s] == 0
            @constraint(model, x[n, d, s] == 0)
        end
    end

    # At most one shift per nurse per day.
    for n in 1:n_nurses, d in 1:n_days
        @constraint(model, sum(x[n, d, s] for s in 1:n_shifts) <= 1)
    end

    # Per-nurse total shift bounds.
    for n in 1:n_nurses
        total_expr = sum(x[n, d, s] for d in 1:n_days, s in 1:n_shifts)
        @constraint(model, total_expr >= prob.min_shifts[n])
        @constraint(model, total_expr <= prob.max_shifts[n])
    end

    # Per-nurse weekend shift bounds.
    if !isempty(prob.weekend_days)
        for n in 1:n_nurses
            expr = sum(x[n, d, s] for d in prob.weekend_days, s in 1:n_shifts)
            lower, upper = prob.weekend_bounds[n]
            @constraint(model, expr >= lower)
            @constraint(model, expr <= upper)
        end
    end

    # Per-nurse night shift limits.
    if night_idx !== nothing
        for n in 1:n_nurses
            @constraint(model, sum(x[n, d, night_idx] for d in 1:n_days) <= prob.night_limits[n])
        end
    end

    # Maximum consecutive working days.
    for n in 1:n_nurses
        limit = prob.max_consecutive_days[n]
        if limit < n_days
            for start_day in 1:(n_days - limit)
                window = sum(sum(x[n, d, s] for s in 1:n_shifts) for d in start_day:(start_day + limit))
                @constraint(model, window <= limit)
            end
        end
    end

    # Mandatory rest (no early shifts) after a night shift.
    if night_idx !== nothing && !isempty(early_indices)
        for n in 1:n_nurses
            rest = prob.rest_after_night[n]
            if rest > 0
                for d in 1:(n_days - rest)
                    expr = x[n, d, night_idx]
                    for offset in 1:rest
                        for idx in early_indices
                            if d + offset <= n_days
                                expr += x[n, d + offset, idx]
                            end
                        end
                    end
                    @constraint(model, expr <= 1)
                end
            end
        end
    end

    return model
end

# Register the variant
register_variant(
    :nurse_scheduling,
    :standard,
    NurseSchedulingProblem,
    "Nurse scheduling with skill mix, shift coverage, and realistic labor-contract rules",
)
