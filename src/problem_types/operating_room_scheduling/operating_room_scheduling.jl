# Operating-room planning and scheduling generators.
#
# Shared random-data helpers are deliberately pure with respect to Julia's
# global RNG: every constructor owns a local MersenneTwister and `build_model`
# is deterministic.

using Random
using Distributions

register_category(
    :operating_room_scheduling,
    "Operating-room tactical, advance, robust, and daily allocation/scheduling models with empirically calibrated duration uncertainty",
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
        n_surgeons = clamp(round(Int, cases_per_spec[k] / rand(rng, Uniform(4.0, 7.0))), 1, 5)
        append!(surgeon_specialty, fill(k, n_surgeons))
    end
    budget = zeros(Float64, length(surgeon_specialty), n_days)
    for s in eachindex(surgeon_specialty)
        k = surgeon_specialty[s]
        block_days = [d for d in 1:n_days if any(mss[:, d] .== k)]
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

function _orsched_hospital_scale(rng::AbstractRNG, target_variables::Int)
    target = max(target_variables, 1)
    if target <= 120
        return rand(rng, 2:3), 5, rand(rng, 2:3)
    elseif target <= 600
        return rand(rng, 3:6), 5, rand(rng, 3:5)
    elseif target <= 2500
        return rand(rng, 5:9), rand(rng, 5:10), rand(rng, 4:7)
    end
    return rand(rng, 8:16), 10, rand(rng, 6:11)
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

function _orsched_inject_surgeon_shortage!(
    surgeon_budget::Matrix{Float64}, surgeon::Int, duration::Float64, working_days::Vector{Int}
)
    n_days_worked = max(1, length(working_days))
    share = 0.5 * duration / n_days_worked
    for d in working_days
        # Keep the day admissible (> 0) while making the summed budget strictly
        # less than one mandatory case.  This preserves the sized sparse graph.
        surgeon_budget[surgeon, d] = share
    end
    return nothing
end

include("elective_assignment.jl")
include("case_sequencing.jl")
include("weekly_planning.jl")
include("master_surgical_schedule.jl")
include("robust_elective.jl")
include("benchmark_loading.jl")
