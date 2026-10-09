using JuMP
using Random
using Distributions

"""
Planted integral roster for a `feasible` instance: the set of assignment
variables (indices into `assignment_slots`) the demand, skill and labor-contract
parameters are derived from, together with the per-nurse aggregates those
parameters are bounded against. Every constraint of the *integer* model is
satisfied by the roster, so `feasible` instances are feasible both for the
natural MIP and for its LP relaxation.

# Fields

  - `assigned::Vector{Int}`: indices of the assignment variables set to one
  - `shift_totals::Vector{Int}`: total shifts worked per nurse
  - `night_counts::Vector{Int}`: night shifts worked per nurse
  - `weekend_counts::Vector{Int}`: weekend shifts worked per nurse
  - `max_consecutive::Vector{Int}`: longest run of consecutive working days per nurse
"""
struct NurseRosterWitness
    assigned::Vector{Int}
    shift_totals::Vector{Int}
    night_counts::Vector{Int}
    weekend_counts::Vector{Int}
    max_consecutive::Vector{Int}
end

"""
Relaxation-proof infeasibility certificate: a hospital-wide night-shift
shortage. Summing every night coverage row gives
`sum_{nurse, ward, day} x[nurse, ward, day, night] >= night_demand`, while
summing every nurse's night-limit row gives the same left-hand side
`<= night_capacity`. The generator sets `night_capacity <= night_demand - 1`
(and at most 95% of it), a contradiction for any `x >= 0`: the LP relaxation is
infeasible as well as the MIP. The argument spans every night coverage row and
every night-limit row, so presolve alone does not detect it.

# Fields

  - `night_demand::Int`: total night coverage required over all wards and days
  - `night_capacity::Int`: sum of the nurses' night limits
"""
struct NurseNightShortageCertificate
    night_demand::Int
    night_capacity::Int
end

"""
    NurseSchedulingProblem <: ProblemGenerator

Generator for multi-ward nurse rostering with realistic labor-contract rules.

# Overview

Models the assignment of nurses to wards and shifts across a planning horizon of
whole weeks. This is a genuine **mixed-integer rostering formulation**: the
binary variable for assignment slot `(nurse, ward, day, shift)` says whether the
nurse works that shift on that ward. Only *available* slots get a variable: a
nurse who is off, not qualified for nights, or not attached to a ward simply has
no column there (rather than a column fixed to zero by a one-variable row).
Nurses belong to a home ward; float-pool nurses also serve one or two
neighbouring wards, coupling the wards' rosters. Because the public generation
API defaults to `relax_integer=true`, the *default* model is the LP relaxation
of this MIP; pass `relax_integer=false` for the natural integer roster, which
`feasible` instances also satisfy by construction.

Constraints capture a rich set of labor rules:

  - Shift coverage per (ward, day, shift) and specialty skill mix.
  - At most one shift per nurse per day (across wards).
  - Per-nurse min/max total shifts (one ranged row) and weekend shift bounds.
  - Per-nurse night-shift limits.
  - Maximum consecutive working days (sliding windows).
  - Mandatory rest after a night: a night on day `d` excludes every early
    shift on days `d+1..d+rest` (one aggregated row per night-day and rest day).

# Fields

  - `n_nurses`, `n_wards`, `n_days`, `n_shifts`: dimensions
  - `shift_labels::Vector{Symbol}`: label of each shift type (e.g. `:day`, `:night`)
  - `weekend_days::Vector{Int}`: weekend day indices
  - `nurse_types::Vector{Symbol}`: contract type (`:core`, `:float_pool`, `:part_time`, `:day_only`)
  - `nurse_wards::Vector{Vector{Int}}`: wards each nurse serves (home ward first)
  - `nurse_skills::Matrix{Int}`: 1 if nurse `n` has skill `k` (skill 1 is the base skill)
  - `night_qualified::Vector{Bool}`
  - `assignment_slots::Vector{NTuple{4,Int}}`: `(nurse, ward, day, shift)` of each variable
  - `costs::Vector{Float64}`: cost of each assignment variable
  - `demand::Array{Int,3}`: required nurses per (ward, day, shift)
  - `skill_requirements::Array{Int,4}`: required qualified nurses per (ward, day, shift, skill)
  - `min_shifts`, `max_shifts::Vector{Int}`: total-shift bounds per nurse
  - `weekend_bounds::Vector{Tuple{Int,Int}}`: weekend-shift bounds per nurse
  - `night_limits::Vector{Int}`: maximum night shifts per nurse
  - `max_consecutive_days::Vector{Int}`: maximum consecutive working days per nurse
  - `rest_after_night::Vector{Int}`: rest days (no early shifts) after a night shift
  - `min_available_per_slot::Int`: every `(ward, day, shift)` has at least this
    many assignment variables
  - `feasible_witness::Union{Nothing,NurseRosterWitness}`
  - `infeasibility_certificate::Union{Nothing,NurseNightShortageCertificate}`
  - `feasibility_status::FeasibilityStatus`
"""
struct NurseSchedulingProblem <: ProblemGenerator
    n_nurses::Int
    n_wards::Int
    n_days::Int
    n_shifts::Int
    shift_labels::Vector{Symbol}
    weekend_days::Vector{Int}
    nurse_types::Vector{Symbol}
    nurse_wards::Vector{Vector{Int}}
    nurse_skills::Matrix{Int}
    night_qualified::Vector{Bool}
    assignment_slots::Vector{NTuple{4, Int}}
    costs::Vector{Float64}
    demand::Array{Int, 3}
    skill_requirements::Array{Int, 4}
    min_shifts::Vector{Int}
    max_shifts::Vector{Int}
    weekend_bounds::Vector{Tuple{Int, Int}}
    night_limits::Vector{Int}
    max_consecutive_days::Vector{Int}
    rest_after_night::Vector{Int}
    min_available_per_slot::Int
    feasible_witness::Union{Nothing, NurseRosterWitness}
    infeasibility_certificate::Union{Nothing, NurseNightShortageCertificate}
    feasibility_status::FeasibilityStatus
end

# Minimum number of assignment variables for every (ward, day, shift) slot.
const NURSE_MIN_AVAILABLE_PER_SHIFT = 2

# Nurses per ward (home-ward roster size) the ward count is derived from.
const NURSE_WARD_SIZE = 32

const NURSE_SHIFT_ALIASES = Dict(
    1 => [:day],
    2 => [:day, :night],
    3 => [:day, :evening, :night],
    4 => [:day, :swing, :evening, :night],
)

is_nurse_weekend(day::Int) = mod1(day, 7) in (6, 7)

"""
    select_nurse_dimensions(target_variables) -> (n_days, n_shifts)

Horizon and shift structure for a variable target: every horizon spans whole
weeks (so weekends appear) and has at least two shift types (so a night shift,
with its rest rules, is always present).
"""
function select_nurse_dimensions(target_variables::Int)
    target = max(target_variables, 1)
    return if target <= 150
        (7, 2)
    elseif target <= 600
        (7, 3)
    elseif target <= 2000
        (14, 3)
    else
        (28, 3)
    end
end

function build_nurse_shift_labels(n_shifts::Int)
    haskey(NURSE_SHIFT_ALIASES, n_shifts) && return copy(NURSE_SHIFT_ALIASES[n_shifts])
    labels = [:day, :swing, :evening, :night]
    while length(labels) < n_shifts
        push!(labels, Symbol("shift$(length(labels)+1)"))
    end
    return labels[1:n_shifts]
end

# Early shifts that must be rested after a night shift: shift 1 always, and shift 2
# once there are at least three shifts (e.g. day + evening).
function nurse_early_shift_indices(n_shifts::Int)
    indices = Int[]
    n_shifts >= 1 && push!(indices, 1)
    n_shifts >= 3 && push!(indices, 2)
    return indices
end

function nurse_scenario(target::Int)
    return target <= 600 ? :small : (target <= 4000 ? :medium : :large)
end

function sample_nurse_type(rng::AbstractRNG, scenario::Symbol)
    probs = if scenario == :small
        (0.55, 0.18, 0.17, 0.10)
    elseif scenario == :medium
        (0.5, 0.2, 0.2, 0.1)
    else
        (0.48, 0.27, 0.18, 0.07)
    end
    r = rand(rng)
    types = (:core, :float_pool, :part_time, :day_only)
    cumulative = 0.0
    for (idx, p) in enumerate(probs)
        cumulative += p
        r <= cumulative && return types[idx]
    end
    return :core
end

function sample_nurse_base_rate(rng::AbstractRNG, nurse_type::Symbol, scenario::Symbol)
    lo, hi = scenario == :small ? (32.0, 45.0) : (scenario == :medium ? (35.0, 52.0) : (38.0, 60.0))
    premium = nurse_type == :float_pool ? 1.08 : (nurse_type == :part_time ? 0.95 : 1.0)
    return rand(rng, Uniform(lo, hi)) * premium
end

function sample_nurse_skills(rng::AbstractRNG, n_skills::Int, scenario::Symbol)
    skills = zeros(Int, n_skills)
    skills[1] = 1
    base_prob = scenario == :small ? 0.25 : (scenario == :medium ? 0.32 : 0.4)
    for k in 2:n_skills
        rand(rng) < min(0.95, base_prob * rand(rng, Uniform(0.8, 1.2))) && (skills[k] = 1)
    end
    n_skills > 1 && all(skills[2:end] .== 0) && (skills[rand(rng, 2:n_skills)] = 1)
    return skills
end

"""
    sample_nurse_availability(rng, n_days, shift_labels, nurse_type, night_qualified)

0/1 availability per (day, shift) for one nurse: a personal density (Beta(7,2))
times shift-type propensities, weekend effects by contract type, day-only and
part-time reductions; no nights for nurses who are not night-qualified.
"""
function sample_nurse_availability(
    rng::AbstractRNG,
    n_days::Int,
    shift_labels::Vector{Symbol},
    nurse_type::Symbol,
    night_qualified::Bool,
)
    n_shifts = length(shift_labels)
    availability = falses(n_days, n_shifts)
    base_density = rand(rng, Beta(7, 2))
    for d in 1:n_days, s in 1:n_shifts
        label = shift_labels[s]
        (label == :night && !night_qualified) && continue
        prob = label == :day ? 0.85 : ((label == :evening || label == :swing) ? 0.65 : 0.42)
        prob *= base_density
        if is_nurse_weekend(d)
            prob *= nurse_type == :core ? 0.85 : (nurse_type == :float_pool ? 1.1 : 0.95)
        end
        if nurse_type == :day_only && label != :day
            prob *= 0.1
        elseif nurse_type == :part_time
            prob *= 0.8
        end
        availability[d, s] = rand(rng) < clamp(prob, 0.02, 0.98)
    end
    return availability
end

function sample_nurse_consecutive_limit(rng::AbstractRNG, nurse_type::Symbol, n_days::Int)
    base = if nurse_type == :core
        rand(rng, 3:5)
    else
        (nurse_type == :float_pool ? rand(rng, 2:4) : rand(rng, 2:3))
    end
    return min(max(2, base), n_days)
end

function sample_nurse_target_total(rng::AbstractRNG, nurse_type::Symbol, n_days::Int)
    ratio = if nurse_type == :core
        rand(rng, Uniform(0.65, 0.9))
    elseif nurse_type == :float_pool
        rand(rng, Uniform(0.55, 0.8))
    elseif nurse_type == :part_time
        rand(rng, Uniform(0.3, 0.6))
    else
        rand(rng, Uniform(0.4, 0.55))
    end
    return max(1, min(n_days, round(Int, ratio * n_days)))
end

function select_nurse(
    rng::AbstractRNG, candidates::Vector{Int}, assigned_total::Vector{Int}, targets::Vector{Int}
)
    best_score = -typemax(Int)
    best = candidates[1]
    for n in candidates
        score = targets[n] - assigned_total[n]
        if score == best_score
            if assigned_total[n] < assigned_total[best]
                best = n
            elseif assigned_total[n] == assigned_total[best] && rand(rng) < 0.5
                best = n
            end
        elseif score > best_score
            best_score = score
            best = n
        end
    end
    return best
end

"""
    build_nurse_roster(rng, slots, slot_nurses, base_demand, shift_labels, weekend_days,
                       max_consec, rest_after_night, target_totals, n_nurses)

Greedily build an integral roster day by day, shift by shift, ward by ward:
each `(ward, day, shift)` takes up to its base demand from the nurses holding a
variable there, skipping nurses who already work that day, would exceed their
consecutive-day limit, or are in a post-night rest window for an early shift.
Returns the chosen variable indices and per-nurse aggregates.
"""
function build_nurse_roster(
    rng::AbstractRNG,
    slot_nurses::Array{Vector{Tuple{Int, Int}}, 3},
    base_demand::Array{Int, 3},
    shift_labels::Vector{Symbol},
    weekend_days::Vector{Int},
    max_consec::Vector{Int},
    rest_after_night::Vector{Int},
    target_totals::Vector{Int},
    n_nurses::Int,
)
    n_wards, n_days, n_shifts = size(base_demand)
    chosen_vars = Int[]
    assigned_total = zeros(Int, n_nurses)
    night_counts = zeros(Int, n_nurses)
    weekend_counts = zeros(Int, n_nurses)
    consecutive = zeros(Int, n_nurses)
    worked_prev_day = falses(n_nurses)
    night_block_until = zeros(Int, n_nurses)
    early = nurse_early_shift_indices(n_shifts)
    weekend_set = Set(weekend_days)
    worked_today = falses(n_nurses)
    eligible = Int[]
    var_of = Dict{Int, Int}()
    for d in 1:n_days
        fill!(worked_today, false)
        for s in 1:n_shifts, w in shuffle(rng, collect(1:n_wards))
            req = base_demand[w, d, s]
            assigned = 0
            while assigned < req
                empty!(eligible)
                empty!(var_of)
                for (n, v) in slot_nurses[w, d, s]
                    worked_today[n] && continue
                    (worked_prev_day[n] && consecutive[n] >= max_consec[n]) && continue
                    (d <= night_block_until[n] && s in early) && continue
                    push!(eligible, n)
                    var_of[n] = v
                end
                isempty(eligible) && break
                n = select_nurse(rng, eligible, assigned_total, target_totals)
                push!(chosen_vars, var_of[n])
                worked_today[n] = true
                assigned_total[n] += 1
                if shift_labels[s] == :night
                    night_counts[n] += 1
                    night_block_until[n] = d + rest_after_night[n]
                end
                d in weekend_set && (weekend_counts[n] += 1)
                assigned += 1
            end
        end
        for n in 1:n_nurses
            consecutive[n] = worked_today[n] ? (worked_prev_day[n] ? consecutive[n] + 1 : 1) : 0
            worked_prev_day[n] = worked_today[n]
        end
    end
    return sort!(chosen_vars), assigned_total, night_counts, weekend_counts
end

function observed_nurse_consecutive_days(
    assigned::Vector{Int}, slots::Vector{NTuple{4, Int}}, n_nurses::Int, n_days::Int
)
    worked = falses(n_nurses, n_days)
    for v in assigned
        n, _, d, _ = slots[v]
        worked[n, d] = true
    end
    observed = zeros(Int, n_nurses)
    for n in 1:n_nurses
        current = 0
        for d in 1:n_days
            current = worked[n, d] ? current + 1 : 0
            observed[n] = max(observed[n], current)
        end
    end
    return observed
end

"""
    NurseSchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-ward nurse rostering instance with exactly `target_variables`
assignment variables (for targets of at least about 30; tiny targets are raised
to the structural minimum of two variables per `(ward, day, shift)`).

# Sizing

The horizon is 7, 14 or 28 days with 2-3 shift types (see
[`select_nurse_dimensions`](@ref)); wards hold about $(NURSE_WARD_SIZE) home
nurses each, `n_wards = clamp(round(target / (32 * days * shifts * 0.5)), 1, ...)`.
Nurses are sampled one by one (type, skills, ward attachments, availability)
until their available slots reach the target; every `(ward, day, shift)` is
topped up to $(NURSE_MIN_AVAILABLE_PER_SHIFT) variables, and surplus slots are
then trimmed (never below that minimum) to land exactly on the target.

# Feasibility

  - `feasible`: a greedy integral roster is built first and demand, skill
    requirements and every per-nurse bound are derived from it, so it is a
    feasible 0/1 point (`feasible_witness`).
  - `infeasible`: the same construction, then nurses' night limits are cut
    (largest first, keeping at least one night where possible) until their sum
    is at most `min(0.95 * night_demand, night_demand - 1)` - a hospital-wide
    night shortage ([`NurseNightShortageCertificate`]) that holds for the LP
    relaxation and needs an aggregation of many rows to see.
  - `unknown`: the same construction with natural, two-sided perturbations -
    demand redrawn as the planted coverage times a global census factor in
    `[1.00, 1.20]` with ±3% per-slot noise (capped one below the slot's
    variable count), nurses' maximum totals tightened by 0-2 shifts and night
    limits by 0-1 (never below one for a nurse who worked nights) - so the
    instance may or may not be feasible. No metadata is attached.
"""
function NurseSchedulingProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)
    n_days, n_shifts = select_nurse_dimensions(target)
    shift_labels = build_nurse_shift_labels(n_shifts)
    weekend_days = [d for d in 1:n_days if is_nurse_weekend(d)]
    night_idx = findfirst(==(:night), shift_labels)
    scenario = nurse_scenario(target)
    n_skills = scenario == :large ? 4 : 3
    n_wards = clamp(round(Int, target / (NURSE_WARD_SIZE * n_days * n_shifts * 0.5)), 1, 100_000)

    nurse_types = Symbol[]
    nurse_wards = Vector{Int}[]
    skill_rows = Vector{Int}[]
    night_qualified = Bool[]
    base_rates = Float64[]
    availability = BitMatrix[]
    slots = NTuple{4, Int}[]

    # Sample nurses until their available slots reach the target.
    while length(slots) < target || length(nurse_types) < 2
        n = length(nurse_types) + 1
        t = sample_nurse_type(rng, scenario)
        home = mod1(n, n_wards)
        wards = [home]
        if t == :float_pool && n_wards > 1
            push!(wards, mod1(home + 1, n_wards))
            (n_wards > 2 && rand(rng) < 0.5) && push!(wards, mod1(home - 1, n_wards))
        end
        nq = night_idx !== nothing && t != :day_only && rand(rng) < 0.8
        avail = sample_nurse_availability(rng, n_days, shift_labels, t, nq)
        push!(nurse_types, t)
        push!(nurse_wards, wards)
        push!(skill_rows, sample_nurse_skills(rng, n_skills, scenario))
        push!(night_qualified, nq)
        push!(base_rates, sample_nurse_base_rate(rng, t, scenario))
        push!(availability, avail)
        for w in wards, d in 1:n_days, s in 1:n_shifts
            avail[d, s] && push!(slots, (n, w, d, s))
        end
    end
    n_nurses = length(nurse_types)
    # Every night shift needs a night-qualified nurse somewhere.
    if night_idx !== nothing && !any(night_qualified)
        idx = something(findfirst(!=(:day_only), nurse_types), 1)
        night_qualified[idx] = true
    end

    # Top up every (ward, day, shift) to the structural minimum.
    present = Set(slots)
    count_at = zeros(Int, n_wards, n_days, n_shifts)
    for (_, w, d, s) in slots
        count_at[w, d, s] += 1
    end
    ward_members = [Int[] for _ in 1:n_wards]
    for n in 1:n_nurses, w in nurse_wards[n]
        push!(ward_members[w], n)
    end
    for w in 1:n_wards, d in 1:n_days, s in 1:n_shifts
        needed = NURSE_MIN_AVAILABLE_PER_SHIFT - count_at[w, d, s]
        needed <= 0 && continue
        pool = [
            n for n in ward_members[w] if
            !((n, w, d, s) in present) && (s != night_idx || night_qualified[n])
        ]
        if length(pool) < needed
            # Promote nurses to night duty if the ward lacks night-qualified staff.
            extra = [n for n in ward_members[w] if !((n, w, d, s) in present) && !(n in pool)]
            for n in extra
                night_qualified[n] = true
            end
            append!(pool, extra)
        end
        for n in pool[randperm(rng, length(pool))[1:min(needed, length(pool))]]
            push!(slots, (n, w, d, s))
            push!(present, (n, w, d, s))
            count_at[w, d, s] += 1
        end
    end
    # Trim surplus slots (never below the per-slot minimum) to hit the target.
    surplus = length(slots) - target
    if surplus > 0
        order = randperm(rng, length(slots))
        keep = trues(length(slots))
        for i in order
            surplus == 0 && break
            n, w, d, s = slots[i]
            count_at[w, d, s] > NURSE_MIN_AVAILABLE_PER_SHIFT || continue
            keep[i] = false
            count_at[w, d, s] -= 1
            surplus -= 1
        end
        slots = slots[keep]
    end
    sort!(slots)
    n_vars = length(slots)

    nurse_skills = zeros(Int, n_nurses, n_skills)
    for n in 1:n_nurses
        nurse_skills[n, :] .= skill_rows[n]
    end
    for k in 2:n_skills
        any(nurse_skills[:, k] .== 1) || (nurse_skills[rand(rng, 1:n_nurses), k] = 1)
    end
    rest_after_night = [night_qualified[n] ? rand(rng, 1:2) : 0 for n in 1:n_nurses]
    max_consec = [sample_nurse_consecutive_limit(rng, nurse_types[n], n_days) for n in 1:n_nurses]
    target_totals = [sample_nurse_target_total(rng, nurse_types[n], n_days) for n in 1:n_nurses]

    slot_nurses = [Tuple{Int, Int}[] for _ in 1:n_wards, _ in 1:n_days, _ in 1:n_shifts]
    for (v, (n, w, d, s)) in enumerate(slots)
        push!(slot_nurses[w, d, s], (n, v))
    end

    # Base demand: a share of each ward's home roster, by shift type, weekday
    # and a seasonal swing.
    avg_ratio = scenario == :small ? 0.35 : (scenario == :medium ? 0.42 : 0.5)
    home_count = zeros(Int, n_wards)
    for n in 1:n_nurses
        home_count[nurse_wards[n][1]] += 1
    end
    base_demand = zeros(Int, n_wards, n_days, n_shifts)
    for w in 1:n_wards, d in 1:n_days
        season = 0.9 + 0.2 * sin(2π * d / max(7, n_days) + w)
        weekend_factor = is_nurse_weekend(d) ? 0.95 : 1.05
        for s in 1:n_shifts
            label = shift_labels[s]
            shift_factor = label == :day ? 1.1 : (label == :night ? 0.7 : 0.9)
            base =
                home_count[w] * avg_ratio * season * weekend_factor * shift_factor / n_shifts * 1.6
            base_demand[w, d, s] = max(1, round(Int, base * rand(rng, Uniform(0.85, 1.15))))
        end
    end

    assigned, assigned_total, night_counts, weekend_counts = build_nurse_roster(
        rng,
        slot_nurses,
        base_demand,
        shift_labels,
        weekend_days,
        max_consec,
        rest_after_night,
        target_totals,
        n_nurses,
    )
    observed = observed_nurse_consecutive_days(assigned, slots, n_nurses, n_days)
    max_consec = max.(max_consec, observed)

    # Demand slightly below the planted coverage; skill requirements capped at
    # the skilled nurses the roster fields.
    coverage = zeros(Int, n_wards, n_days, n_shifts)
    skilled = zeros(Int, n_wards, n_days, n_shifts, n_skills)
    for v in assigned
        n, w, d, s = slots[v]
        coverage[w, d, s] += 1
        for k in 1:n_skills
            nurse_skills[n, k] == 1 && (skilled[w, d, s, k] += 1)
        end
    end
    demand = zeros(Int, n_wards, n_days, n_shifts)
    skill_requirements = zeros(Int, n_wards, n_days, n_shifts, n_skills)
    ratios = if scenario == :small
        (1.0, 0.2, 0.12, 0.08)
    else
        (scenario == :medium ? (1.0, 0.25, 0.18, 0.12) : (1.0, 0.3, 0.22, 0.15))
    end
    for w in 1:n_wards, d in 1:n_days, s in 1:n_shifts
        c = coverage[w, d, s]
        c == 0 && continue
        demand[w, d, s] = max(1, min(c, round(Int, c * rand(rng, Uniform(0.85, 0.98)))))
        skill_requirements[w, d, s, 1] = demand[w, d, s]
        for k in 2:n_skills
            skill_requirements[w, d, s, k] = min(
                round(Int, demand[w, d, s] * ratios[k]), skilled[w, d, s, k]
            )
        end
    end

    min_shifts = [max(0, round(Int, assigned_total[n] * 0.7)) for n in 1:n_nurses]
    buffer = max(1, round(Int, n_days * 0.15))
    max_shifts = [min(n_days, assigned_total[n] + buffer) for n in 1:n_nurses]
    for n in 1:n_nurses
        max_shifts[n] < min_shifts[n] + 1 && (max_shifts[n] = min(n_days, min_shifts[n] + 1))
    end
    weekend_bounds = [(max(0, weekend_counts[n] - 1), weekend_counts[n] + 1) for n in 1:n_nurses]
    night_limits = [
        night_qualified[n] ? night_counts[n] + (night_counts[n] == 0 ? 1 : rand(rng, 0:1)) : 0 for
        n in 1:n_nurses
    ]

    costs = zeros(n_vars)
    for (v, (n, w, d, s)) in enumerate(slots)
        label = shift_labels[s]
        shift_mult = label == :night ? 1.28 : ((label == :evening || label == :swing) ? 1.12 : 1.0)
        weekend_mult = is_nurse_weekend(d) ? 1.08 : 1.0
        penalty = 1.0
        if nurse_types[n] == :part_time && label == :night
            penalty += 0.4
        elseif nurse_types[n] == :day_only && label != :day
            penalty += 0.5
        end
        away = w == nurse_wards[n][1] ? 1.0 : 1.05
        costs[v] = round(base_rates[n] * shift_mult * weekend_mult * penalty * away; digits=3)
    end

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        observed = observed_nurse_consecutive_days(assigned, slots, n_nurses, n_days)
        witness = NurseRosterWitness(
            assigned, assigned_total, night_counts, weekend_counts, observed
        )
    elseif feasibility_status == infeasible
        night_demand = sum(demand[:, :, night_idx])
        goal = min(floor(Int, 0.95 * night_demand), night_demand - 1)
        has_night_var = falses(n_nurses)
        for (n, _, _, s) in slots
            s == night_idx && (has_night_var[n] = true)
        end
        for n in 1:n_nurses
            has_night_var[n] || (night_limits[n] = 0)
        end
        # Cut the largest limits first, keeping one night per nurse while possible.
        for floor_value in (1, 0)
            while sum(night_limits) > goal
                candidates = [n for n in 1:n_nurses if night_limits[n] > floor_value]
                isempty(candidates) && break
                top = maximum(night_limits[candidates])
                pick = rand(rng, [n for n in candidates if night_limits[n] == top])
                night_limits[pick] -= 1
            end
        end
        certificate = NurseNightShortageCertificate(night_demand, sum(night_limits))
    else
        # Natural, two-sided perturbations of the planted instance: a global
        # census factor on demand with mild per-slot noise (never more than
        # one below the slot's variable count, so no slot is contradictory on
        # its own), and tighter contracts. Night limits never drop below one
        # for a nurse who worked nights.
        census = rand(rng, Uniform(1.0, 1.2))
        for w in 1:n_wards, d in 1:n_days, s in 1:n_shifts
            c = coverage[w, d, s]
            c == 0 && continue
            target_demand = round(Int, c * census * rand(rng, Uniform(0.97, 1.03)))
            cap = max(1, length(slot_nurses[w, d, s]) - 1)
            demand[w, d, s] = clamp(target_demand, 1, max(cap, demand[w, d, s]))
            skill_requirements[w, d, s, 1] = demand[w, d, s]
        end
        for n in 1:n_nurses
            max_shifts[n] = max(min_shifts[n], max_shifts[n] - rand(rng, 0:2))
            night_limits[n] = max(min(1, night_limits[n]), night_limits[n] - rand(rng, 0:1))
        end
    end

    return NurseSchedulingProblem(
        n_nurses,
        n_wards,
        n_days,
        n_shifts,
        shift_labels,
        weekend_days,
        nurse_types,
        nurse_wards,
        nurse_skills,
        night_qualified,
        slots,
        costs,
        demand,
        skill_requirements,
        min_shifts,
        max_shifts,
        weekend_bounds,
        night_limits,
        max_consec,
        rest_after_night,
        NURSE_MIN_AVAILABLE_PER_SHIFT,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    nurse_model_rows(prob) -> NamedTuple

The row structure of the model as index lists over `assignment_slots`, shared by
`build_model` and the tests: coverage and skill rows per `(ward, day, shift)`,
one-shift-per-day rows, per-nurse total/weekend/night rows, consecutive-day
windows and rest rows. Rows that cannot bind are omitted: a one-shift-per-day
row needs two variables, a window needs more variables than its limit, a rest
row needs a night variable and an early variable.
"""
function nurse_model_rows(prob::NurseSchedulingProblem)
    slots = prob.assignment_slots
    n_nurses, n_days, n_shifts = prob.n_nurses, prob.n_days, prob.n_shifts
    night_idx = findfirst(==(:night), prob.shift_labels)
    early = nurse_early_shift_indices(n_shifts)
    weekend = Set(prob.weekend_days)

    by_slot = Dict{NTuple{3, Int}, Vector{Int}}()
    by_nurse_day = [Int[] for _ in 1:n_nurses, _ in 1:n_days]
    night_of = [Int[] for _ in 1:n_nurses, _ in 1:n_days]
    early_of = [Int[] for _ in 1:n_nurses, _ in 1:n_days]
    for (v, (n, w, d, s)) in enumerate(slots)
        push!(get!(by_slot, (w, d, s), Int[]), v)
        push!(by_nurse_day[n, d], v)
        s == night_idx && push!(night_of[n, d], v)
        s in early && push!(early_of[n, d], v)
    end

    coverage = Tuple{NTuple{3, Int}, Vector{Int}, Int}[]
    skill = Tuple{NTuple{4, Int}, Vector{Int}, Int}[]
    for key in sort!(collect(keys(by_slot)))
        w, d, s = key
        vars = by_slot[key]
        prob.demand[w, d, s] > 0 && push!(coverage, (key, vars, prob.demand[w, d, s]))
        for k in 2:size(prob.nurse_skills, 2)
            req = prob.skill_requirements[w, d, s, k]
            req > 0 || continue
            push!(
                skill,
                ((w, d, s, k), [v for v in vars if prob.nurse_skills[slots[v][1], k] == 1], req),
            )
        end
    end

    one_per_day = [
        (n, d, by_nurse_day[n, d]) for
        n in 1:n_nurses, d in 1:n_days if length(by_nurse_day[n, d]) >= 2
    ]
    totals = Tuple{Int, Vector{Int}}[]
    weekends = Tuple{Int, Vector{Int}}[]
    nights = Tuple{Int, Vector{Int}}[]
    windows = Tuple{Int, Int, Vector{Int}}[]
    rests = Tuple{Int, Int, Int, Vector{Int}}[]
    for n in 1:n_nurses
        all_vars = reduce(vcat, (by_nurse_day[n, d] for d in 1:n_days); init=Int[])
        isempty(all_vars) && continue
        push!(totals, (n, all_vars))
        wk = [v for v in all_vars if slots[v][3] in weekend]
        isempty(wk) || push!(weekends, (n, wk))
        nv = reduce(vcat, (night_of[n, d] for d in 1:n_days); init=Int[])
        isempty(nv) || push!(nights, (n, nv))
        limit = prob.max_consecutive_days[n]
        for start in 1:(n_days - limit)
            vars = reduce(vcat, (by_nurse_day[n, d] for d in start:(start + limit)); init=Int[])
            # A window binds only if more than `limit` of its days have a variable.
            count(d -> !isempty(by_nurse_day[n, d]), start:(start + limit)) > limit &&
                push!(windows, (n, start, vars))
        end
        for d in 1:(n_days - 1), offset in 1:prob.rest_after_night[n]
            d + offset <= n_days || continue
            (isempty(night_of[n, d]) || isempty(early_of[n, d + offset])) && continue
            push!(rests, (n, d, offset, vcat(night_of[n, d], early_of[n, d + offset])))
        end
    end
    return (
        coverage=coverage,
        skill=skill,
        one_per_day=one_per_day,
        totals=totals,
        weekends=weekends,
        nights=nights,
        windows=windows,
        rests=rests,
    )
end

"""
    build_model(prob::NurseSchedulingProblem)

Build the JuMP model. Deterministic - uses only the struct's fields.

`x[v]` is binary for assignment slot `v = (nurse, ward, day, shift)`; integrality
is relaxed centrally by `generate_problem` when `relax_integer=true` (the
default). Rows (see [`nurse_model_rows`](@ref)):

  - coverage `sum_{v at (w,d,s)} x[v] >= demand[w,d,s]` and skill mix
    `sum_{skilled v at (w,d,s)} x[v] >= skill_requirements[w,d,s,k]` (k >= 2)
  - one shift per nurse per day: `sum_{v of (n,d)} x[v] <= 1`
  - `min_shifts[n] <= sum_{v of n} x[v] <= max_shifts[n]` (ranged)
  - weekend `lo <= sum_{weekend v of n} x[v] <= hi` (ranged)
  - night limit `sum_{night v of n} x[v] <= night_limits[n]`
  - consecutive days: every window of `limit + 1` days sums to at most `limit`
  - rest: `x[night of (n,d)] + sum x[early shifts of (n, d + o)] <= 1`, `o = 1..rest`
"""
function build_model(prob::NurseSchedulingProblem)
    model = Model()
    n_vars = length(prob.assignment_slots)
    @variable(model, x[1:n_vars], Bin)
    @objective(model, Min, sum(prob.costs[v] * x[v] for v in 1:n_vars))

    rows = nurse_model_rows(prob)
    for (_, vars, rhs) in rows.coverage
        @constraint(model, sum(x[v] for v in vars) >= rhs)
    end
    for (_, vars, rhs) in rows.skill
        @constraint(model, sum(x[v] for v in vars; init=0.0) >= rhs)
    end
    for (_, _, vars) in rows.one_per_day
        @constraint(model, sum(x[v] for v in vars) <= 1)
    end
    for (n, vars) in rows.totals
        @constraint(model, prob.min_shifts[n] <= sum(x[v] for v in vars) <= prob.max_shifts[n])
    end
    for (n, vars) in rows.weekends
        lo, hi = prob.weekend_bounds[n]
        @constraint(model, lo <= sum(x[v] for v in vars) <= hi)
    end
    for (n, vars) in rows.nights
        @constraint(model, sum(x[v] for v in vars) <= prob.night_limits[n])
    end
    for (n, _, vars) in rows.windows
        @constraint(model, sum(x[v] for v in vars) <= prob.max_consecutive_days[n])
    end
    for (_, _, _, vars) in rows.rests
        @constraint(model, sum(x[v] for v in vars) <= 1)
    end
    return model
end

# Register the variant
register_variant(
    :nurse_scheduling,
    :standard,
    NurseSchedulingProblem,
    "Multi-ward nurse rostering MIP over available assignment slots with float-pool nurses, skill mix, shift coverage, and realistic labor-contract rules";
    tags=[:healthcare, :covering, :packing],
)
