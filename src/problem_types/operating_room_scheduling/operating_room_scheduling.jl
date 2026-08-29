# operating_room_scheduling category
#
# Operating room (OR) planning and scheduling — one of the most studied
# healthcare-operations MILP problem classes (see the survey of Cardoen,
# Demeulemeester & Beliën, "Operating room planning and scheduling: A
# literature review", EJOR 2010). The category covers the two operational
# levels distinguished by that survey:
#
# - advance scheduling (`elective_assignment`): assign elective surgeries from
#   a waiting list to OR blocks over a 1-2 week horizon under a master surgical
#   schedule (Marques, Captivo & Vaz Pato, OR Spectrum 2012);
# - allocation scheduling (`case_sequencing`): allocate a day's surgeries to
#   ORs and surgeons and sequence them with turnover times (the MILP of
#   Maaroufi, Camus & Korbaa, IEEE SMC 2016);
# - multi-day planning with downstream resources (`weekly_planning`):
#   surgery-to-day assignment leveled against ward/ICU bed capacity.
#
# Shared data-generation helpers live in this file (all `_orsched_`-prefixed);
# they consume the global RNG, so they are called from constructors only —
# never from `build_model`. Data distributions are grounded in the empirical
# literature: surgery durations are log-normally distributed per specialty
# (Strum, May & Vargas, Anesthesiology 2000; confirmed per specialty group by
# goodness-of-fit tests in applied studies), OR sessions are 480-minute blocks
# (with 240-minute half sessions and occasional long sessions), OR turnover
# between cases is 15-35 minutes, and the case mix spans the 11 surgical
# specialties used by the Leeftink & Hans (2018) benchmark set.

using Random
using Distributions

# Category-level description (it groups several formulations).
register_category(:operating_room_scheduling,
    "Operating room planning and scheduling: elective surgery assignment under a master surgical schedule, daily surgical case sequencing with surgeon conflicts, and multi-day surgery planning with downstream ward/ICU beds")

# --- Shared data helpers -----------------------------------------------------

# Empirical per-specialty surgery-duration profiles. Durations of elective
# surgeries are well described by a lognormal distribution (Strum, May & Vargas
# 2000); means and coefficients of variation below follow the ranges reported
# for the 11 specialties of the Leeftink & Hans (2018) benchmark case mixes.
# `weight` is a typical case-mix volume share, `day_case` the probability a
# case needs no overnight ward bed, `los` the ward length-of-stay range (days)
# for admitted cases, and `icu` the probability of needing ICU bed days.
const _ORSCHED_SPECIALTIES = (
    (name=:ophthalmology,     mean=50.0,  cv=0.30, weight=1.3, day_case=0.95, los=(0, 1), icu=0.00),
    (name=:oral_maxillofacial, mean=65.0, cv=0.35, weight=0.8, day_case=0.85, los=(0, 2), icu=0.02),
    (name=:urology,           mean=75.0,  cv=0.40, weight=1.0, day_case=0.65, los=(0, 3), icu=0.05),
    (name=:ent,               mean=85.0,  cv=0.40, weight=1.0, day_case=0.75, los=(0, 2), icu=0.02),
    (name=:gynecology,        mean=85.0,  cv=0.40, weight=0.9, day_case=0.70, los=(0, 3), icu=0.03),
    (name=:general_surgery,   mean=115.0, cv=0.40, weight=1.2, day_case=0.45, los=(1, 4), icu=0.08),
    (name=:plastics,          mean=120.0, cv=0.45, weight=0.8, day_case=0.55, los=(1, 4), icu=0.03),
    (name=:orthopedics,       mean=135.0, cv=0.40, weight=1.1, day_case=0.25, los=(2, 5), icu=0.08),
    (name=:vascular,          mean=165.0, cv=0.45, weight=0.6, day_case=0.15, los=(2, 6), icu=0.30),
    (name=:neurosurgery,      mean=215.0, cv=0.45, weight=0.5, day_case=0.05, los=(3, 7), icu=0.50),
    (name=:cardiothoracic,    mean=240.0, cv=0.45, weight=0.4, day_case=0.00, los=(4, 8), icu=0.90),
)

# Draw an index from a cumulative probability vector with a single uniform draw.
function _orsched_pick(cum_probs)
    u = rand()
    idx = findfirst(c -> u <= c, cum_probs)
    return idx === nothing ? length(cum_probs) : idx
end

# Lognormal distribution with the given arithmetic mean and coefficient of
# variation (moment matching, the standard parameterization used to fit
# surgical procedure times).
function _orsched_lognormal(mean_target::Float64, cv::Float64)
    sigma2 = log(1 + cv^2)
    mu = log(mean_target) - sigma2 / 2
    return LogNormal(mu, sqrt(sigma2))
end

# Sample one planned surgery duration (minutes): a lognormal draw rounded to
# the 5-minute granularity hospitals use for case-duration estimates, clamped
# to the range that fits a single OR session.
function _orsched_sample_duration(mean_target::Float64, cv::Float64)
    raw = rand(_orsched_lognormal(mean_target, cv))
    return clamp(5.0 * round(raw / 5.0), 20.0, 480.0)
end

# Pick `n` distinct specialty indices (into `_ORSCHED_SPECIALTIES`) forming a
# plausible hospital case mix: volume-weighted sampling that always spans both
# short-duration (e.g. ophthalmology) and long-duration (e.g. neurosurgery)
# profiles when at least three specialties are present.
function _orsched_case_mix(n_specialties::Int)
    table = _ORSCHED_SPECIALTIES
    n = clamp(n_specialties, 1, length(table))
    weights = [s.weight for s in table]
    chosen = Int[]
    pool = collect(1:length(table))
    while length(chosen) < n
        probs = weights[pool] ./ sum(weights[pool])
        pick = pool[_orsched_pick(cumsum(probs))]
        push!(chosen, pick)
        deleteat!(pool, findfirst(==(pick), pool))
    end
    if n >= 3
        if !any(table[k].mean <= 90.0 for k in chosen)
            shortest = pool[argmin([table[k].mean for k in pool])]
            chosen[rand(1:n)] = shortest
        end
        remaining = [k for k in 1:length(table) if !(k in chosen)]
        if !isempty(remaining) && !any(table[k].mean >= 160.0 for k in chosen)
            longest = remaining[argmax([table[k].mean for k in remaining])]
            chosen[rand(1:n)] = longest
        end
    end
    return sort(chosen)
end

# Master surgical schedule: for each (room, day) either closed (0) or open and
# dedicated to one specialty (its index into `spec_ids`, i.e. positions 1..n_specs
# map to `spec_ids`). Returns `(mss, session)` where `session[r, d]` is the OR
# session length in minutes (0 when closed). Real MSSs open ~85-97% of weekday
# OR-days, run mostly 480-minute sessions with some 240-minute half sessions
# and rare long (780-minute) sessions, and guarantee every specialty at least
# one block per week.
function _orsched_master_schedule(n_rooms::Int, n_days::Int, spec_ids::Vector{Int})
    n_specs = length(spec_ids)
    weights = [_ORSCHED_SPECIALTIES[k].weight * _ORSCHED_SPECIALTIES[k].mean for k in spec_ids]
    mss = zeros(Int, n_rooms, n_days)
    session = zeros(Float64, n_rooms, n_days)
    open_rate = rand(Uniform(0.85, 0.97))
    for d in 1:n_days, r in 1:n_rooms
        rand() > open_rate && continue
        u = rand()
        session[r, d] = u < 0.78 ? 480.0 : u < 0.95 ? 240.0 : 780.0
        mss[r, d] = _orsched_pick(cumsum(weights ./ sum(weights)))
    end
    # Guarantee each specialty at least one block per (partial) week.
    blocks_needed = max(1, n_days ÷ 5)
    open_slots = [(r, d) for d in 1:n_days for r in 1:n_rooms if session[r, d] > 0]
    for k in 1:n_specs
        while count(==(k), mss) < blocks_needed && !isempty(open_slots)
            idx = rand(1:length(open_slots))
            (r, d) = open_slots[idx]
            mss[r, d] = k
            deleteat!(open_slots, idx)
        end
    end
    # Pathological all-closed sample: reopen one full-session block.
    if all(session .== 0)
        session[1, 1] = 480.0
        mss[1, 1] = 1
    end
    return mss, session
end

# Surgeon pool for a set of specialties. `cases_per_spec[k]` is the number of
# waiting-list cases of specialty `k`; each specialty gets one surgeon per
# ~4-7 cases (clamped to [1, 5]). A surgeon works a subset of the days on which
# their specialty has a block (they can only operate when a block exists), with
# a daily operating-time budget of 240-480 minutes (0 on days they do not work).
# Returns `(surgeon_specialty, surgeon_budget)` where `surgeon_budget[s, d]`
# is surgeon s's available operating minutes on day d (0 = not working).
function _orsched_surgeon_pool(cases_per_spec::Vector{Int}, n_days::Int, mss::Matrix{Int})
    surgeon_specialty = Int[]
    for k in 1:length(cases_per_spec)
        n_surgeons = clamp(round(Int, cases_per_spec[k] / rand(Uniform(4.0, 7.0))), 1, 5)
        append!(surgeon_specialty, fill(k, n_surgeons))
    end
    n_surgeons = length(surgeon_specialty)
    budget = zeros(Float64, n_surgeons, n_days)
    for s in 1:n_surgeons
        k = surgeon_specialty[s]
        block_days = [d for d in 1:n_days if any(mss[:, d] .== k)]
        isempty(block_days) && continue
        keep = rand(Uniform(0.5, 0.85))
        working = [d for d in block_days if rand() < keep]
        isempty(working) && (working = [rand(block_days)])
        for d in working
            budget[s, d] = 5.0 * round(rand(Uniform(240.0, 480.0)) / 5.0)
        end
    end
    return surgeon_specialty, budget
end

# Elective-surgery waiting list. Each case draws a specialty (case-mix
# weighted), a lognormal planned duration, and an urgency class:
# `:urgent` (~8-18%, clinically mandated to be scheduled within a few days),
# `:semi_urgent` (~22-40%, target within about two thirds of the horizon), or
# `:routine` (deadline at the horizon end). Postponement penalties scale with
# urgency; a fraction of routine cases are long waiters whose postponement is
# politically costly. With `with_los=true`, ward length-of-stay and ICU bed-day
# needs are sampled as well (for downstream-resource models).
# Returns a NamedTuple of per-surgery vectors.
function _orsched_waiting_list(n_surgeries::Int, spec_ids::Vector{Int}, n_days::Int;
                               with_los::Bool=false)
    table = _ORSCHED_SPECIALTIES
    weights = [table[k].weight for k in spec_ids]
    probs = weights ./ sum(weights)
    cum = cumsum(probs)

    p_urgent = rand(Uniform(0.08, 0.18))
    p_semi = rand(Uniform(0.22, 0.40))
    urgent_max = max(1, n_days ÷ 3)
    semi_max = min(max(urgent_max + 1, (2 * n_days) ÷ 3), n_days)

    specialty = Vector{Int}(undef, n_surgeries)
    duration = Vector{Float64}(undef, n_surgeries)
    urgency = Vector{Symbol}(undef, n_surgeries)
    deadline = Vector{Int}(undef, n_surgeries)
    penalty = Vector{Float64}(undef, n_surgeries)
    ward_los = with_los ? zeros(Int, n_surgeries) : Int[]
    icu_los = with_los ? zeros(Int, n_surgeries) : Int[]

    for i in 1:n_surgeries
        specialty[i] = _orsched_pick(cum)
        k = specialty[i]
        profile = table[spec_ids[k]]
        duration[i] = _orsched_sample_duration(profile.mean, profile.cv)

        u = rand()
        if u < p_urgent
            urgency[i] = :urgent
            deadline[i] = rand(1:urgent_max)
            penalty[i] = rand(Uniform(300.0, 600.0))
        elseif u < p_urgent + p_semi
            urgency[i] = :semi_urgent
            deadline[i] = rand(min(urgent_max + 1, semi_max):semi_max)
            penalty[i] = rand(Uniform(5.0, 25.0)) * rand(Uniform(2.0, 4.0))
        else
            urgency[i] = :routine
            deadline[i] = n_days
            penalty[i] = rand(Uniform(5.0, 25.0))
            rand() < 0.25 && (penalty[i] *= rand(Uniform(2.0, 3.0)))  # long waiter
        end

        if with_los
            if rand() < profile.day_case
                ward_los[i] = 0
            else
                ward_los[i] = rand(profile.los[1]:profile.los[2])
            end
            icu_los[i] = rand() < profile.icu ? rand(1:2) : 0
            ward_los[i] = max(ward_los[i], icu_los[i])
        end
    end

    if with_los
        return (specialty=specialty, duration=duration, urgency=urgency,
                deadline=deadline, penalty=penalty, ward_los=ward_los, icu_los=icu_los)
    end
    return (specialty=specialty, duration=duration, urgency=urgency,
            deadline=deadline, penalty=penalty)
end

# Scale tier for the hospital size given a target variable count. Returns
# `(n_rooms, n_days, n_specialties)` spanning small outpatient surgery centers
# (2-3 ORs, one week) through large academic medical centers (12-16 ORs, two
# weeks), mirroring the sizes in the literature (4-6 ORs in Maaroufi et al.
# 2016; 14 ORs over two weeks in the Leeftink & Hans benchmark case study).
function _orsched_hospital_scale(target_variables::Int)
    target = max(target_variables, 1)
    if target <= 120
        return rand(2:3), 5, rand(2:3)
    elseif target <= 600
        return rand(3:6), 5, rand(3:5)
    elseif target <= 2500
        return rand(5:9), rand(5:10), rand(4:7)
    else
        return rand(8:16), 10, rand(6:11)
    end
end

# Greedy earliest-deadline / best-fit scheduler used to construct a provably
# feasible point for `feasible` instances. Surgeries are considered in
# clinical-priority order (urgent first by deadline, then semi-urgent, then
# routine; longer cases first within a class) and each is placed in the
# earliest admissible slot whose remaining capacities all fit. `slots_for[i]`
# lists the admissible slot indices for surgery i; `consume!(remaining, slot, i)`
# returns true and decrements capacities when surgery i fits in `slot`.
function _orsched_greedy_schedule(n_surgeries::Int, urgency::Vector{Symbol},
                                  deadline::Vector{Int}, duration::Vector{Float64},
                                  slots_for::Vector{Vector{Int}}, n_slots::Int,
                                  consume!::Function)
    rank = Dict(:urgent => 1, :semi_urgent => 2, :routine => 3)
    order = sort(collect(1:n_surgeries),
                 by=i -> (rank[urgency[i]], deadline[i], -duration[i]))
    assignment = zeros(Int, n_surgeries)
    for i in order
        for slot in slots_for[i]
            if consume!(slot, i)
                assignment[i] = slot
                break
            end
        end
    end
    return assignment
end

# Deterministic LP-level infeasibility certificate shared by the assignment
# variants: make surgery `victim` mandatory and shrink its surgeon's daily
# budgets so the budgets over the victim's admissible days sum to strictly less
# than the surgery's duration. Then summing the victim's surgeon-day rows gives
# duration[victim] * sum(x) <= total budget < duration[victim], contradicting
# the mandatory row sum(x) = 1 — a contradiction that already holds in the LP
# relaxation. Returns nothing; mutates `surgeon_budget`.
function _orsched_inject_surgeon_shortage!(surgeon_budget::Matrix{Float64}, surgeon::Int,
                                           duration::Float64, working_days::Vector{Int})
    total = 0.5 * duration
    n_days_worked = max(1, length(working_days))
    for d in working_days
        surgeon_budget[surgeon, d] = floor(total / n_days_worked / 5.0) * 5.0
    end
    return nothing
end

include("elective_assignment.jl")
include("case_sequencing.jl")
include("weekly_planning.jl")
