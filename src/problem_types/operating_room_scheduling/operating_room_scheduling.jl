# Operating-room planning and scheduling generators.
#
# Shared random-data helpers are deliberately pure with respect to Julia's
# global RNG: every constructor owns a local MersenneTwister and `build_model`
# is deterministic.

using Random
using Distributions

register_category(
    :operating_room_scheduling,
    "Operating-room tactical, advance, robust, and operational (time-indexed) allocation and scheduling models with empirically calibrated duration uncertainty",
)

include("leeftink_hans_data.jl")

function _orsched_pick(rng::AbstractRNG, cum_probs)
    u = rand(rng) * cum_probs[end]
    idx = findfirst(c -> u <= c, cum_probs)
    return idx === nothing ? length(cum_probs) : idx
end

function _orsched_lognormal(mean_target::Real, cv::Real)
    sigma2 = log(1 + cv^2)
    mu = log(mean_target) - sigma2 / 2
    return LogNormal(mu, sqrt(sigma2))
end

function _orsched_sample_duration(rng::AbstractRNG, mean_target::Real, cv::Real)
    raw = rand(rng, _orsched_lognormal(mean_target, cv))
    return clamp(5.0 * round(raw / 5.0), 20.0, 480.0)
end

_orsched_type_mean(t) = t.gamma + exp(t.mu + t.sigma^2 / 2)
_orsched_type_sd(t) = exp(t.mu + t.sigma^2 / 2) * sqrt(exp(t.sigma^2) - 1)

function _orsched_sample_benchmark_type(rng::AbstractRNG, specialty_id::Int)
    profile = _ORSCHED_SPECIALTIES[specialty_id]
    return profile.types[rand(rng, eachindex(profile.types))]
end

# Pick distinct specialties and preserve short- and long-duration services.
# The two replacement positions are disjoint, fixing the old repair in which
# inserting the long service could overwrite the short service.
function _orsched_case_mix(rng::AbstractRNG, n_specialties::Int)
    table = _ORSCHED_SPECIALTIES
    n = clamp(n_specialties, 1, length(table))
    pool = collect(eachindex(table))
    chosen = Int[]
    while length(chosen) < n
        weights = [table[k].weight for k in pool]
        pos = _orsched_pick(rng, cumsum(weights))
        push!(chosen, pool[pos])
        deleteat!(pool, pos)
    end
    if n >= 3
        short_ids = [k for k in eachindex(table) if table[k].aggregate_mean <= 90]
        long_ids = [k for k in eachindex(table) if table[k].aggregate_mean >= 160]
        if !any(k -> k in short_ids, chosen)
            candidate = short_ids[argmax([table[k].weight for k in short_ids])]
            protected = findfirst(k -> k in long_ids, chosen)
            replaceable = if protected === nothing
                collect(eachindex(chosen))
            else
                [j for j in eachindex(chosen) if j != protected]
            end
            chosen[first(replaceable)] = candidate
        end
        if !any(k -> k in long_ids, chosen)
            candidate = long_ids[argmax([table[k].weight for k in long_ids])]
            protected = findfirst(k -> k in short_ids, chosen)
            replaceable = [j for j in eachindex(chosen) if j != protected]
            chosen[last(replaceable)] = candidate
        end
    end
    @assert length(unique(chosen)) == n
    return sort!(chosen)
end

# Create an MSS and guarantee every service its quota without taking the only
# guaranteed block from a previous service.  A donor is unassigned or strictly
# above quota; a closed slot is opened if necessary.
function _orsched_master_schedule(
    rng::AbstractRNG, n_rooms::Int, n_days::Int, spec_ids::Vector{Int}
)
    n_specs = length(spec_ids)
    weights = [
        _ORSCHED_SPECIALTIES[k].weight * _ORSCHED_SPECIALTIES[k].aggregate_mean for k in spec_ids
    ]
    mss = zeros(Int, n_rooms, n_days)
    session = zeros(Float64, n_rooms, n_days)
    open_rate = rand(rng, Uniform(0.85, 0.97))
    for d in 1:n_days, r in 1:n_rooms
        rand(rng) > open_rate && continue
        u = rand(rng)
        session[r, d] = if u < 0.78
            480.0
        elseif u < 0.95
            240.0
        else
            780.0
        end
        mss[r, d] = _orsched_pick(rng, cumsum(weights))
    end
    quota = max(1, n_days ÷ 5)
    for k in 1:n_specs
        while count(==(k), mss) < quota
            counts = [count(==(j), mss) for j in 1:n_specs]
            donors = [
                (r, d) for d in 1:n_days for
                r in 1:n_rooms if session[r, d] > 0 && (mss[r, d] == 0 || counts[mss[r, d]] > quota)
            ]
            if isempty(donors)
                closed = [(r, d) for d in 1:n_days for r in 1:n_rooms if session[r, d] == 0]
                isempty(closed) && error("MSS dimensions cannot satisfy specialty quotas")
                r, d = rand(rng, closed)
                session[r, d] = 480.0
                mss[r, d] = k
            else
                r, d = rand(rng, donors)
                mss[r, d] = k
            end
        end
    end
    @assert all(count(==(k), mss) >= quota for k in 1:n_specs)
    @assert all((mss[r, d] == 0) == (session[r, d] == 0) for r in 1:n_rooms, d in 1:n_days)
    return mss, session
end

function _orsched_surgeon_pool(
    rng::AbstractRNG, cases_per_spec::Vector{Int}, n_days::Int, mss::Matrix{Int}
)
    surgeon_specialty = Int[]
    for k in eachindex(cases_per_spec)
        # A surgeon carries 4-7 waiting-list cases; large hospitals have many
        # surgeons per service.
        n_surgeons = clamp(round(Int, cases_per_spec[k] / rand(rng, Uniform(4.0, 7.0))), 1, 1000)
        append!(surgeon_specialty, fill(k, n_surgeons))
    end
    budget = zeros(Float64, length(surgeon_specialty), n_days)
    block_days_of = [[d for d in 1:n_days if any(view(mss, :, d) .== k)] for k in eachindex(cases_per_spec)]
    for s in eachindex(surgeon_specialty)
        block_days = block_days_of[surgeon_specialty[s]]
        keep = rand(rng, Uniform(0.55, 0.90))
        working = [d for d in block_days if rand(rng) < keep]
        isempty(working) && (working = [rand(rng, block_days)])
        for d in working
            budget[s, d] = 5.0 * round(rand(rng, Uniform(240.0, 480.0)) / 5.0)
        end
    end
    return surgeon_specialty, budget
end

# Planned durations are expected values of sampled empirical benchmark
# archetypes; `duration_sd` preserves uncertainty for the robust formulation.
# With `allow_urgent=false`, mandatory cases are designated only after a
# feasible witness is planted, so clinical labels are never downgraded.
function _orsched_waiting_list(
    rng::AbstractRNG,
    n_surgeries::Int,
    spec_ids::Vector{Int},
    n_days::Int;
    with_los::Bool=false,
    allow_urgent::Bool=true,
)
    table = _ORSCHED_SPECIALTIES
    cum = cumsum([table[k].weight for k in spec_ids])
    p_urgent = rand(rng, Uniform(0.08, 0.18))
    p_semi = rand(rng, Uniform(0.22, 0.40))
    urgent_max = max(1, n_days ÷ 3)
    semi_lo = min(urgent_max + 1, n_days)
    semi_max = min(max(semi_lo, (2 * n_days) ÷ 3), n_days)

    specialty = Vector{Int}(undef, n_surgeries)
    source_type = Vector{Int}(undef, n_surgeries)
    duration = Vector{Float64}(undef, n_surgeries)
    duration_sd = Vector{Float64}(undef, n_surgeries)
    urgency = Vector{Symbol}(undef, n_surgeries)
    deadline = Vector{Int}(undef, n_surgeries)
    penalty = Vector{Float64}(undef, n_surgeries)
    ward_los = with_los ? zeros(Int, n_surgeries) : Int[]
    icu_los = with_los ? zeros(Int, n_surgeries) : Int[]

    for i in 1:n_surgeries
        specialty[i] = _orsched_pick(rng, cum)
        profile = table[spec_ids[specialty[i]]]
        surgery_type = _orsched_sample_benchmark_type(rng, spec_ids[specialty[i]])
        source_type[i] = surgery_type.id
        duration[i] = clamp(5.0 * round(_orsched_type_mean(surgery_type) / 5.0), 20.0, 480.0)
        duration_sd[i] = max(5.0, 5.0 * round(_orsched_type_sd(surgery_type) / 5.0))

        u = rand(rng)
        if allow_urgent && u < p_urgent
            urgency[i] = :urgent
            deadline[i] = rand(rng, 1:urgent_max)
            penalty[i] = rand(rng, Uniform(300.0, 600.0))
        elseif u < p_urgent + p_semi
            urgency[i] = :semi_urgent
            deadline[i] = rand(rng, semi_lo:semi_max)
            penalty[i] = rand(rng, Uniform(10.0, 80.0))
        else
            urgency[i] = :routine
            deadline[i] = n_days
            penalty[i] = rand(rng, Uniform(5.0, 25.0))
            rand(rng) < 0.25 && (penalty[i] *= rand(rng, Uniform(2.0, 3.0)))
        end

        if with_los
            if rand(rng) < profile.icu
                icu_los[i] = rand(rng, 1:2)
                ward_los[i] = max(1, rand(rng, profile.ward_los[1]:profile.ward_los[2]))
            elseif rand(rng) < profile.day_case
                ward_los[i] = 0
            else
                ward_los[i] = rand(rng, profile.ward_los[1]:profile.ward_los[2])
            end
        end
    end

    base = (
        specialty=specialty,
        source_type=source_type,
        duration=duration,
        duration_sd=duration_sd,
        urgency=urgency,
        deadline=deadline,
        penalty=penalty,
        requested_urgent_fraction=p_urgent,
    )
    return with_los ? merge(base, (ward_los=ward_los, icu_los=icu_los)) : base
end

function _orsched_designate_mandatory!(
    rng::AbstractRNG,
    urgency::Vector{Symbol},
    deadline::Vector{Int},
    penalty::Vector{Float64},
    assignment::Vector{Int},
    fraction::Real,
)
    scheduled = findall(>(0), assignment)
    isempty(scheduled) && return falses(length(assignment))
    n_mandatory = clamp(round(Int, fraction * length(assignment)), 1, length(scheduled))
    # Prefer already-early cases; random tie-breaking avoids a fixed case-id
    # pattern without changing any clinical deadline.
    shuffled = shuffle(rng, scheduled)
    candidates = sort(shuffled; by=i -> deadline[i])[1:n_mandatory]
    mandatory = falses(length(assignment))
    for i in candidates
        urgency[i] = :urgent
        penalty[i] = rand(rng, Uniform(300.0, 600.0))
        mandatory[i] = true
    end
    return mandatory
end

"""
    _orsched_add_referrals!(rng, mandatory, urgency, penalty, assignment, has_option)

`unknown` instances: on top of the urgent cases designated from the greedy
plan, a random 0-100% of the cases the plan could not place (but that fit some
admissible slot on their own, `has_option`) arrive as urgent referrals and
become mandatory. Whether the
LP can fit them depends on how much slack the greedy plan left, so the
instance may or may not be feasible.
"""
function _orsched_add_referrals!(
    rng::AbstractRNG,
    mandatory::BitVector,
    urgency::Vector{Symbol},
    penalty::Vector{Float64},
    assignment::Vector{Int},
    has_option::Vector{Bool},
)
    pool = [i for i in eachindex(assignment) if assignment[i] == 0 && has_option[i] && !mandatory[i]]
    isempty(pool) && return mandatory
    n_referrals = round(Int, rand(rng, Uniform(0.0, 1.0)) * length(pool))
    for i in shuffle(rng, pool)[1:n_referrals]
        mandatory[i] = true
        urgency[i] = :urgent
        penalty[i] = rand(rng, Uniform(300.0, 600.0))
    end
    return mandatory
end

"""
    _orsched_hospital_scale(rng, target_variables; growth=:quadratic) -> (rooms, days, specialties)

Hospital dimensions for a variable target. Small targets use fixed tiers. From
2,500 variables up the suite grows with the target so the waiting list stays
proportionate to OR capacity (a load near one) instead of piling thousands of
cases onto 16 rooms: with `growth=:quadratic` (room-level assignment, whose
variables grow with cases x rooms) the room count is about
`sqrt(target * specialties / 150)`; with `growth=:linear` (day-level planning, a handful of
variables per case) it is about `target / 190`. Specialties: 6-11; horizon: 10
days.
"""
function _orsched_hospital_scale(rng::AbstractRNG, target_variables::Int; growth::Symbol=:quadratic)
    target = max(target_variables, 1)
    if target <= 120
        return rand(rng, 2:3), 5, rand(rng, 2:3)
    elseif target <= 600
        return rand(rng, 3:6), 5, rand(rng, 3:5)
    elseif target <= 2500
        return rand(rng, 5:9), rand(rng, 5:10), rand(rng, 4:7)
    end
    specs = rand(rng, 6:11)
    jitter = rand(rng, Uniform(0.85, 1.15))
    rooms = if growth == :linear
        clamp(round(Int, target / 190 * jitter), 8, 20_000)
    else
        # Each case is admissible to about 5.2 * rooms / specs rooms over the
        # horizon, so variables ~ 146 * load / specs * rooms^2 at a load near 1.
        clamp(round(Int, sqrt(target * specs / 150) * jitter), 8, 2_000)
    end
    return rooms, 10, specs
end

_orsched_load_target(rng::AbstractRNG) = rand(rng, _ORSCHED_BENCHMARK_LOADS)

function _orsched_postop_days(surgery_day::Int, icu_los::Int, ward_los::Int, bed_horizon::Int)
    icu_days =
        icu_los == 0 ? Int[] : collect(surgery_day:min(bed_horizon, surgery_day + icu_los - 1))
    ward_start = surgery_day + icu_los
    ward_days =
        ward_los == 0 ? Int[] : collect(ward_start:min(bed_horizon, ward_start + ward_los - 1))
    return icu_days, ward_days
end

function _orsched_greedy_schedule(
    n_surgeries::Int,
    urgency::Vector{Symbol},
    deadline::Vector{Int},
    duration::Vector{Float64},
    slots_for::Vector{Vector{Int}},
    n_slots::Int,
    consume!::Function,
)
    rank = Dict(:urgent => 1, :semi_urgent => 2, :routine => 3)
    order = sort(collect(1:n_surgeries); by=i -> (rank[urgency[i]], deadline[i], -duration[i]))
    assignment = zeros(Int, n_surgeries)
    for i in order, slot in slots_for[i]
        if consume!(slot, i)
            assignment[i] = slot
            break
        end
    end
    return assignment
end

"""
    SurgeonOverloadCertificate

Relaxation-proof infeasibility certificate shared by the waiting-list
variants: surgeon `surgeon` must operate every case in `cases` (all mandatory,
each admissible only on days in `days`), but the operating minutes budgeted
over `days` total `budget_minutes <= 0.9 * case_minutes`. Summing the
surgeon's day-budget rows over `days` (`sum duration_i * assign <= budget[s, d]`)
and the cases' assignment rows (`sum assign = 1`, no postponement) gives
`case_minutes <= budget_minutes`, a contradiction for any fractional
assignment.

Every day in `days` gets the same budget, the longest of the cases, and
(except in the small-hospital fallback) every case is admissible on at least
two of those days and fits the room (or specialty) capacity on each of them. So no variable bound is tightened by a
single row, no single row is contradictory, and presolve has to aggregate the
rows to see the shortage.
"""
struct SurgeonOverloadCertificate
    surgeon::Int
    days::Vector{Int}
    cases::Vector{Int}
    case_minutes::Float64
    budget_minutes::Float64
end

"""
    _orsched_plant_surgeon_overload!(rng, surgeon_budget, surgery_surgeon, duration,
                                     case_days, fits, mandatory, urgency, penalty)

Plant a [`SurgeonOverloadCertificate`](@ref). `case_days[i]` lists the days
case `i` is admissible; `fits(i, d)` says whether it fits the non-surgeon
capacity of day `d` on its own. For each surgeon (in random order) and each
set `D` of two or three of its working days, the candidate cases are those
admissible only on days of `D`, on at least two of them, fitting on each; the
first `D` whose cases' minutes reach `|D| * longest / 0.9` (and that leaves no
other mandatory case of the surgeon stranded on `D`) is used, with every day
of `D` budgeted at the longest case. Small hospitals without such a surgeon
fall back to single shared days and pairs of cases (still a valid
certificate). Errors if no surgeon qualifies at all.
"""
function _orsched_plant_surgeon_overload!(
    rng::AbstractRNG,
    surgeon_budget::Matrix{Float64},
    surgery_surgeon::Vector{Int},
    duration::Vector{Float64},
    case_days::Vector{Vector{Int}},
    fits::Function,
    mandatory::BitVector,
    urgency::Vector{Symbol},
    penalty::Vector{Float64},
)
    n_surgeons, n_days = size(surgeon_budget)
    cases_of = [Int[] for _ in 1:n_surgeons]
    for i in eachindex(surgery_surgeon)
        isempty(case_days[i]) || push!(cases_of[surgery_surgeon[i]], i)
    end
    order = shuffle(rng, collect(1:n_surgeons))
    # Strict mode first (every case on >= 2 days, >= 3 cases: presolve-proof);
    # small hospitals fall back to single shared days and pairs of cases.
    for (min_days, min_group) in ((2, 3), (1, 2))
        for s in order
            length(cases_of[s]) >= min_group || continue
            good = [
                i for i in cases_of[s] if
                length(case_days[i]) >= min_days && all(fits(i, d) for d in case_days[i])
            ]
            length(good) >= min_group || continue
            working = sort(unique(reduce(vcat, case_days[good])))
            sets = vcat(
                min_days == 1 ? [[d] for d in working] : Vector{Int}[],
                [[d1, d2] for d1 in working for d2 in working if d1 < d2],
                [[d1, d2, d3] for d1 in working for d2 in working for d3 in working if d1 < d2 < d3],
            )
            best = nothing
            for D in sets
                group = [i for i in good if issubset(case_days[i], D)]
                length(group) >= min_group || continue
                longest = maximum(duration[group])
                sum(duration[group]) * 0.9 >= length(D) * longest || continue
                # Other mandatory cases confined to D would face the cut budgets.
                any(
                    mandatory[i] && !(i in group) && issubset(case_days[i], D) for i in cases_of[s]
                ) && continue
                # Prefer three days (two-day sets of doubleton assignment rows
                # can be aggregated by presolve), then the larger group.
                if best === nothing || (length(D), length(group)) > (length(best[1]), length(best[2]))
                    best = (D, group, longest)
                end
            end
            best === nothing && continue
            D, group, longest = best
            for d in D
                surgeon_budget[s, d] = longest
            end
            for i in group
                mandatory[i] = true
                if urgency[i] != :urgent
                    urgency[i] = :urgent
                    penalty[i] = rand(rng, Uniform(300.0, 600.0))
                end
            end
            return SurgeonOverloadCertificate(s, D, sort(group), sum(duration[group]), length(D) * longest)
        end
    end
    # Tiny instances: overload one surgeon's whole list by spreading 90% of
    # its minutes over its days (single rows may then be contradictory, which
    # presolve can see, but the certificate is still valid).
    s = argmax(length.(cases_of))
    group = sort(cases_of[s])
    isempty(group) && error("no schedulable case for an overload certificate")
    D = sort(unique(reduce(vcat, case_days[group])))
    share = max(1.0, floor(0.9 * sum(duration[group]) / length(D)))
    for d in D
        surgeon_budget[s, d] = share
    end
    for i in group
        mandatory[i] = true
        if urgency[i] != :urgent
            urgency[i] = :urgent
            penalty[i] = rand(rng, Uniform(300.0, 600.0))
        end
    end
    return SurgeonOverloadCertificate(s, D, group, sum(duration[group]), share * length(D))
end

include("elective_assignment.jl")
include("case_sequencing.jl")
include("weekly_planning.jl")
include("master_surgical_schedule.jl")
include("robust_elective.jl")
include("benchmark_loading.jl")
