using JuMP
using Random
using Distributions

"""
    MSSWardShortageCertificate

Relaxation-proof infeasibility certificate for the master surgical schedule:
the specialty ward `ward` cannot absorb the patients of its services' minimum
block quotas. Every block of service `g` puts `bed_days_per_block[g]` expected
patient-days into the ward over one cycle (the sum of its periodized ward
profile). Summing the ward's occupancy-definition rows over the cycle and
using each service's minimum-quota row gives
`sum_d ward_occupancy[d] >= required_bed_days = sum_g bed_days_per_block[g] * min_blocks[g]`,
while the ward's occupancy bounds give `sum_d ward_occupancy[d] <= capacity_bed_days`.
The capacities are cut so `capacity_bed_days <= 0.9 * required_bed_days`. No
single day's bound is contradictory, so presolve does not detect it.
"""
struct MSSWardShortageCertificate
    ward::Int
    services::Vector{Int}
    required_bed_days::Float64
    capacity_bed_days::Float64
end

"""
    OperatingRoomMasterScheduleProblem <: ProblemGenerator

Tactical cyclic master-surgical-schedule (MSS) design for a hospital (or
hospital group) whose surgical *services* - surgeon groups, each belonging to
a specialty - share operating rooms. Blocks `(service, room, day)` are
assigned subject to minimum/maximum block quotas, a soft target quota, daily
concentration limits, and room exclusivity. Each specialty has its own ward;
expected ICU (hospital-wide) and post-ICU ward occupancy profiles are
convolved cyclically with the block plan, capped, and the peaks levelled.

Rooms are clustered by specialty (in proportion to workload); every service
is compatible with 4-8 rooms of its specialty's cluster, so only compatible
assignment variables are created and the model scales to 100k+ variables by
adding services and rooms (a 10-day cycle from 600 variables).

Feasible instances retain the complete planted block plan (admissible-block
indices). Infeasible instances carry an LP-level ward-shortage certificate
([`MSSWardShortageCertificate`]). Unknown instances perturb the quotas and
capacities around the planted plan.

# Fields

  - `n_services`, `n_rooms`, `n_days`, `n_wards` (specialties present)
  - `specialty_names::Vector{Symbol}`: name of each ward's specialty
  - `service_ward::Vector{Int}`: ward (specialty) of each service
  - `target_blocks`, `min_blocks`, `max_blocks`, `max_daily_rooms` (per service)
  - `service_rooms::Vector{Vector{Int}}`: compatible rooms per service
  - `admissible_blocks::Vector{NTuple{3,Int}}`: `(service, room, day)` per assignment variable
  - `preference_cost::Vector{Float64}`, `room_open_cost::Matrix{Float64}` (rooms x days)
  - `ward_profile::Matrix{Float64}`, `icu_profile::Matrix{Float64}` (wards x days, by lag)
  - `ward_capacity::Matrix{Float64}` (wards x days), `icu_capacity::Vector{Float64}`
  - `under_penalty`, `over_penalty` (per service), `peak_ward_weight`, `peak_icu_weight`
  - `feasible_witness::Union{Nothing,Vector{Int}}`: planted admissible-block indices
  - `infeasibility_certificate::Union{Nothing,MSSWardShortageCertificate}`
  - `feasibility_status`
"""
struct OperatingRoomMasterScheduleProblem <: ProblemGenerator
    n_services::Int
    n_rooms::Int
    n_days::Int
    n_wards::Int
    specialty_names::Vector{Symbol}
    service_ward::Vector{Int}
    target_blocks::Vector{Int}
    min_blocks::Vector{Int}
    max_blocks::Vector{Int}
    max_daily_rooms::Vector{Int}
    service_rooms::Vector{Vector{Int}}
    admissible_blocks::Vector{NTuple{3, Int}}
    preference_cost::Vector{Float64}
    room_open_cost::Matrix{Float64}
    ward_profile::Matrix{Float64}
    icu_profile::Matrix{Float64}
    ward_capacity::Matrix{Float64}
    icu_capacity::Vector{Float64}
    under_penalty::Vector{Float64}
    over_penalty::Vector{Float64}
    peak_ward_weight::Float64
    peak_icu_weight::Float64
    feasible_witness::Union{Nothing, Vector{Int}}
    infeasibility_certificate::Union{Nothing, MSSWardShortageCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _mss_dimensions(target) -> (days, services, rooms)

A 5-day cycle up to 600 variables, then 10 days. Each service contributes
about `6 * days` assignment columns plus its share of room-day columns
(services need ~5.5 blocks; rooms are ~85% booked), so
`services ~ target / (6.6 * days + 2)` and `rooms ~ 5.5 * services / (0.85 * days)`.
"""
function _mss_dimensions(target::Int)
    days = target <= 600 ? 5 : 10
    services = max(2, round(Int, target / (6.6 * days + 2)))
    rooms = max(2, round(Int, 5.5 * services / (0.85 * days)))
    return days, services, rooms
end

function _mss_uniform_los_survival(los::Tuple{Int, Int}, age::Int; minimum_los::Int=0)
    age < 0 && return 0.0
    lo, hi = los
    return count(length_of_stay -> max(minimum_los, length_of_stay) > age, lo:hi) / (hi - lo + 1)
end
function _mss_profile_components(profile, n_days::Int)
    cases_per_block = 480.0 / (profile.aggregate_mean + 25.0)
    direct_ward = zeros(Float64, n_days)
    post_icu_ward = zeros(Float64, n_days)
    icu = zeros(Float64, n_days)
    max_lag = max(2, profile.ward_los[2] + 1)

    for lag in 0:max_lag
        phase = mod(lag, n_days) + 1

        # Half the ICU cohort has LOS 1 and half LOS 2. At lag 1, the former
        # half is in the ward while the latter half remains in ICU.
        icu_survival = if lag == 0
            1.0
        elseif lag == 1
            0.5
        else
            0.0
        end
        icu[phase] += cases_per_block * profile.icu * icu_survival

        direct_survival = _mss_uniform_los_survival(profile.ward_los, lag)
        direct_ward[phase] +=
            cases_per_block * (1 - profile.icu) * (1 - profile.day_case) * direct_survival

        if lag >= 1
            # Discharge after one ICU day (probability 1/2).
            post_icu_ward[phase] +=
                cases_per_block *
                profile.icu *
                0.5 *
                _mss_uniform_los_survival(profile.ward_los, lag - 1; minimum_los=1)
        end
        if lag >= 2
            # Discharge after two ICU days (probability 1/2).
            post_icu_ward[phase] +=
                cases_per_block *
                profile.icu *
                0.5 *
                _mss_uniform_los_survival(profile.ward_los, lag - 2; minimum_los=1)
        end
    end
    return (direct_ward=direct_ward, post_icu_ward=post_icu_ward, icu=icu)
end
function _mss_profiles(spec_ids::Vector{Int}, n_days::Int)
    S = length(spec_ids)
    ward = zeros(Float64, S, n_days)
    icu = zeros(Float64, S, n_days)
    for s in 1:S
        components = _mss_profile_components(_ORSCHED_SPECIALTIES[spec_ids[s]], n_days)
        ward[s, :] .= components.direct_ward .+ components.post_icu_ward
        icu[s, :] .= components.icu
    end
    return ward, icu
end

"""
    _mss_layout(rng, n_services, n_rooms, n_days) -> NamedTuple

Sample one MSS layout: specialties (case mix), services per specialty, room
clusters per specialty (proportional to services, at least one room each),
compatible rooms per service (4-8 of its cluster) and the planted block plan.
"""
function _mss_layout(rng::AbstractRNG, n_services::Int, n_rooms::Int, n_days::Int)
    n_specs = clamp(round(Int, n_services / 3), 2, min(11, n_services))
    spec_ids = _orsched_case_mix(rng, n_specs)
    weights = [
        _ORSCHED_SPECIALTIES[k].weight * _ORSCHED_SPECIALTIES[k].aggregate_mean for k in spec_ids
    ]
    # Every specialty gets at least one service, the rest by weight.
    service_ward = collect(1:n_specs)
    cum = cumsum(weights)
    while length(service_ward) < n_services
        push!(service_ward, _orsched_pick(rng, cum))
    end
    shuffle!(rng, service_ward)
    services_of = [findall(==(w), service_ward) for w in 1:n_specs]

    # Room clusters in proportion to the specialty's services.
    n_rooms = max(n_rooms, n_specs)
    cluster_size = ones(Int, n_specs)
    rest = n_rooms - n_specs
    shares = rest .* length.(services_of) ./ n_services
    extra = floor.(Int, shares)
    order = sortperm(shares .- extra; rev=true)
    for j in 1:(rest - sum(extra))
        extra[order[j]] += 1
    end
    cluster_size .+= extra
    room_ids = shuffle(rng, collect(1:n_rooms))
    clusters = Vector{Vector{Int}}(undef, n_specs)
    offset = 0
    for w in 1:n_specs
        clusters[w] = sort(room_ids[(offset + 1):(offset + cluster_size[w])])
        offset += cluster_size[w]
    end
    service_rooms = Vector{Vector{Int}}(undef, n_services)
    for g in 1:n_services
        cluster = clusters[service_ward[g]]
        k = min(length(cluster), rand(rng, 4:8))
        service_rooms[g] = sort(shuffle(rng, cluster)[1:k])
    end

    # Planted plan: each service gets a quota of 3-8 blocks, at most
    # `max_daily` per day, placed in random free compatible room-days.
    max_daily = [rand(rng, 1:max(1, min(3, length(service_rooms[g])))) for g in 1:n_services]
    occupied = falses(n_rooms, n_days)
    planted = NTuple{3, Int}[]
    daily = zeros(Int, n_services, n_days)
    for g in shuffle(rng, collect(1:n_services))
        quota = rand(rng, 3:8)
        placed = 0
        for (r, d) in shuffle(rng, [(r, d) for r in service_rooms[g] for d in 1:n_days])
            placed >= quota && break
            (occupied[r, d] || daily[g, d] >= max_daily[g]) && continue
            occupied[r, d] = true
            daily[g, d] += 1
            push!(planted, (g, r, d))
            placed += 1
        end
    end
    return (
        spec_ids=spec_ids,
        service_ward=service_ward,
        service_rooms=service_rooms,
        max_daily=max_daily,
        planted=planted,
        n_rooms=n_rooms,
    )
end

_mss_variable_count(layout, n_days, n_wards) =
    sum(length, layout.service_rooms) * n_days +
    layout.n_rooms * n_days +
    2length(layout.service_ward) +
    n_wards * n_days +
    n_days +
    n_wards +
    1

function OperatingRoomMasterScheduleProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 20)
    D, S, R = _mss_dimensions(target)

    layout = nothing
    best_gap = Inf
    for _ in 1:6
        candidate = _mss_layout(rng, S, R, D)
        W = length(candidate.spec_ids)
        total = _mss_variable_count(candidate, D, W)
        gap = abs(total - target) / target
        if gap < best_gap
            best_gap = gap
            layout = candidate
        end
        gap <= 0.03 && break
        S = max(2, round(Int, S * target / total))
        R = max(2, round(Int, 5.5 * S / (0.85 * D)))
    end
    spec_ids = layout.spec_ids
    W = length(spec_ids)
    service_ward = layout.service_ward
    service_rooms = layout.service_rooms
    max_daily = layout.max_daily
    S = length(service_ward)
    R = layout.n_rooms

    admissible = [(g, r, d) for g in 1:S for r in service_rooms[g] for d in 1:D]
    index_of = Dict(b => a for (a, b) in enumerate(admissible))
    planted_idx = sort([index_of[b] for b in layout.planted])

    counts = zeros(Int, S)
    for (g, _, _) in layout.planted
        counts[g] += 1
    end
    target_blocks = copy(counts)
    min_blocks = [max(min(1, counts[g]), counts[g] - rand(rng, 0:1)) for g in 1:S]
    max_blocks = [counts[g] + rand(rng, 0:2) for g in 1:S]
    ward_profile, icu_profile = _mss_profiles(spec_ids, D)

    planted_ward = zeros(W, D)
    planted_icu = zeros(D)
    for (g, _, dp) in layout.planted, d in 1:D
        w = service_ward[g]
        planted_ward[w, d] += ward_profile[w, mod(d - dp, D) + 1]
        planted_icu[d] += icu_profile[w, mod(d - dp, D) + 1]
    end
    ward_capacity = ceil.(1.10 .* planted_ward .+ 1.0)
    icu_capacity = ceil.(1.15 .* planted_icu .+ 1.0)

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        witness = planted_idx
    elseif feasibility_status == infeasible
        # Ward shortage: the busiest specialty ward cannot absorb its
        # services' minimum quotas over the cycle.
        mass = [sum(ward_profile[w, :]) for w in 1:W]
        required = [
            sum(mass[w] * min_blocks[g] for g in 1:S if service_ward[g] == w; init=0.0) for w in 1:W
        ]
        w = argmax(required)
        services = [g for g in 1:S if service_ward[g] == w]
        goal = 0.9 * required[w]
        shape = planted_ward[w, :] .+ 0.05
        scaled = floor.(goal .* shape ./ sum(shape); digits=2)
        ward_capacity[w, :] .= scaled
        certificate = MSSWardShortageCertificate(w, services, required[w], sum(ward_capacity[w, :]))
    else
        # Natural scenario: quotas loosen or tighten around the planted plan
        # and a hospital-wide bed-pressure factor in [0.70, 1.05] scales every
        # capacity (measured critical factor ~0.85).
        for g in 1:S
            min_blocks[g] = max(min(1, counts[g]), target_blocks[g] - rand(rng, 0:1))
            max_blocks[g] = max(min_blocks[g], target_blocks[g] + rand(rng, -1:2))
        end
        pressure = rand(rng, Uniform(0.70, 1.05))
        ward_capacity .= round.(
            ward_capacity .* pressure .* rand(rng, Uniform(0.95, 1.05), W, D); digits=2
        )
        icu_capacity .= round.(
            icu_capacity .* pressure .* rand(rng, Uniform(0.95, 1.05), D); digits=2
        )
    end

    preference = [rand(rng, Uniform(0.0, 50.0)) for _ in admissible]
    open_cost = [rand(rng, Uniform(300.0, 900.0)) for _ in 1:R, _ in 1:D]
    under = [rand(rng, Uniform(300.0, 700.0)) for _ in 1:S]
    over = [rand(rng, Uniform(100.0, 300.0)) for _ in 1:S]

    return OperatingRoomMasterScheduleProblem(
        S,
        R,
        D,
        W,
        [_ORSCHED_SPECIALTIES[k].name for k in spec_ids],
        service_ward,
        target_blocks,
        min_blocks,
        max_blocks,
        max_daily,
        service_rooms,
        admissible,
        preference,
        open_cost,
        ward_profile,
        icu_profile,
        ward_capacity,
        icu_capacity,
        under,
        over,
        rand(rng, Uniform(10.0, 30.0)),
        rand(rng, Uniform(20.0, 50.0)),
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::OperatingRoomMasterScheduleProblem)

Variables: `assign_block[a]` (binary, per admissible block), `open_room[r, d]`
(binary), `under_blocks[g]`, `over_blocks[g]`, `ward_occupancy[w, d]`
(bounded by the ward capacity), `icu_occupancy[d]` (bounded by the ICU
capacity), `peak_ward[w]`, `peak_icu`. Rows: room exclusivity per room-day,
soft target and ranged min/max quota per service, daily concentration per
service-day (when it can bind), occupancy definitions per ward-day and
ICU-day (zero profile coefficients omitted), and peak rows.
"""
function build_model(prob::OperatingRoomMasterScheduleProblem)
    model = Model()
    S, R, D, W = prob.n_services, prob.n_rooms, prob.n_days, prob.n_wards
    A = length(prob.admissible_blocks)
    @variable(model, assign_block[1:A], Bin)
    @variable(model, open_room[1:R, 1:D], Bin)
    @variable(model, under_blocks[1:S] >= 0)
    @variable(model, over_blocks[1:S] >= 0)
    @variable(model, 0 <= ward_occupancy[w = 1:W, d = 1:D] <= prob.ward_capacity[w, d])
    @variable(model, 0 <= icu_occupancy[d = 1:D] <= prob.icu_capacity[d])
    @variable(model, peak_ward[1:W] >= 0)
    @variable(model, peak_icu >= 0)

    by_room_day = [Int[] for _ in 1:R, _ in 1:D]
    by_service = [Int[] for _ in 1:S]
    by_service_day = [Int[] for _ in 1:S, _ in 1:D]
    for (a, (g, r, d)) in enumerate(prob.admissible_blocks)
        push!(by_room_day[r, d], a)
        push!(by_service[g], a)
        push!(by_service_day[g, d], a)
    end

    for r in 1:R, d in 1:D
        @constraint(
            model,
            sum(assign_block[a] for a in by_room_day[r, d]; init=AffExpr(0.0)) == open_room[r, d]
        )
    end
    for g in 1:S
        blocks = sum(assign_block[a] for a in by_service[g])
        @constraint(model, blocks + under_blocks[g] - over_blocks[g] == prob.target_blocks[g])
        @constraint(model, prob.min_blocks[g] <= blocks <= prob.max_blocks[g])
        for d in 1:D
            vars = by_service_day[g, d]
            length(vars) > prob.max_daily_rooms[g] || continue
            @constraint(model, sum(assign_block[a] for a in vars) <= prob.max_daily_rooms[g])
        end
    end

    ward_terms = [AffExpr(0.0) for _ in 1:W, _ in 1:D]
    icu_terms = [AffExpr(0.0) for _ in 1:D]
    for (a, (g, _, dp)) in enumerate(prob.admissible_blocks)
        w = prob.service_ward[g]
        for d in 1:D
            lag = mod(d - dp, D) + 1
            c = prob.ward_profile[w, lag]
            c > 0 && add_to_expression!(ward_terms[w, d], c, assign_block[a])
            c = prob.icu_profile[w, lag]
            c > 0 && add_to_expression!(icu_terms[d], c, assign_block[a])
        end
    end
    for w in 1:W, d in 1:D
        @constraint(model, ward_occupancy[w, d] == ward_terms[w, d])
        @constraint(model, peak_ward[w] >= ward_occupancy[w, d])
    end
    for d in 1:D
        @constraint(model, icu_occupancy[d] == icu_terms[d])
        @constraint(model, peak_icu >= icu_occupancy[d])
    end

    @objective(
        model,
        Min,
        sum(prob.preference_cost[a] * assign_block[a] for a in 1:A) +
            sum(prob.room_open_cost[r, d] * open_room[r, d] for r in 1:R, d in 1:D) +
            sum(
                prob.under_penalty[g] * under_blocks[g] + prob.over_penalty[g] * over_blocks[g] for
                g in 1:S
            ) +
            prob.peak_ward_weight * sum(peak_ward) +
            prob.peak_icu_weight * peak_icu
    )
    return model
end

register_variant(
    :operating_room_scheduling,
    :master_surgical_schedule,
    OperatingRoomMasterScheduleProblem,
    "Sparse tactical cyclic master-surgical-schedule block allocation for surgical services over specialty room clusters, with quotas, specialty wards and ICU occupancy leveling";
    tags=[:healthcare, :packing],
)
