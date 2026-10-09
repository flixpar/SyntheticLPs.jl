using JuMP
using Random
using Distributions

"""
    SurgeonDayOverbookingCertificate

Relaxation-proof infeasibility certificate for `case_sequencing`: every case in
`cases` belongs to surgeon `surgeon` and is admissible only on day `day`.
A start variable of case `o` keeps the surgeon busy for `case_slots[o] +
surgeon_turnover` consecutive surgeon-slot rows, all inside the surgeon's
window `[window_start, window_end + surgeon_turnover)` of
`available_slots = window_end - window_start + surgeon_turnover` slots.
Summing those rows (each `<= 1`) against the cases' assignment rows (`= 1`)
gives `busy_slots = sum_o (case_slots[o] + surgeon_turnover) <= available_slots`,
which the generator violates by at least 5% - for any fractional schedule.
Each case fits the window on its own, so no single row is contradictory.
"""
struct SurgeonDayOverbookingCertificate
    surgeon::Int
    day::Int
    cases::Vector{Int}
    busy_slots::Int
    available_slots::Int
end

"""
    SurgicalCaseSequencingProblem <: ProblemGenerator

Time-indexed surgical case scheduling over a week of operating days: assign
each elective case a room, a day and a start slot (15 minutes) so that rooms
and surgeons are never double-booked, cases stay inside their surgeon's
operating window, and overruns past the regular session close are penalized.

# Why time-indexed

The previous formulation (allocation plus big-M disjunctive ordering
binaries, as in Maaroufi et al. 2016) has an empty LP relaxation: with
fractional ordering variables every disjunction is slack, so the relaxed
optimum starts every case at time zero - the same collapse as
`job_shop_scheduling`. The time-indexed formulation (Pritsker et al.; the
standard strong formulation for OR scheduling) keeps the resource conflicts
in the LP: each start variable occupies the room for the case duration plus
the room turnover, and the surgeon for the duration plus the surgeon
turnover, in per-slot capacity rows.

# Model

  - `x[c] ∈ {0,1}` for each column `c = (case, room, day, start)`: case
    starts in that room on that day at that slot; columns exist only for the
    case's eligible rooms, its surgeon's operating days it may be booked on,
    and starts inside the surgeon's window;
  - every case is scheduled exactly once: `sum_c x[c] = 1`;
  - room capacity per (room, day, slot): `<= 1` over the columns occupying it;
  - surgeon capacity per (surgeon, day, slot): `<= 1`;
  - objective: weighted tardiness minutes past the regular close plus a small
    completion-time term and room preferences.

# Data

Surgeons have a specialty, 1-3 operating days per week and a full-day,
morning or afternoon window (in 15-minute slots of a 480-minute session) with
0-60 minutes of allowed overtime. Rooms form specialty clusters that grow as
needed. Durations come from the empirical Leeftink--Hans surgery types. Each
case is eligible for its planted room plus 1-3 other rooms of its specialty
cluster, and for its planted day plus each other operating day of its surgeon
with probability 1/2. Surgeons are added until the columns reach the target;
the surplus columns (never a planted one, never a case's last) are then
dropped at random, so the variable count equals the target.

# Feasibility

The schedule is *planted*: each surgeon-day fills one room sequentially from
the window start (room turnover between cases), mostly within the regular
session and occasionally one case into the allowed overtime.

  - `feasible`: the planted schedule is the witness (`feasible_witness`, the
    column of each case).
  - `infeasible`: one surgeon-day is overbooked with add-on cases restricted
    to that day until the surgeon's busy slots exceed the window by at least
    5% ([`SurgeonDayOverbookingCertificate`]); presolve cannot see it.
  - `unknown`: up to ~20 surgeons (a random share of at most 40, whatever
    the hospital size) receive one short (at most 90 minutes) add-on case
    bookable on any of their operating days; whether the add-ons fit depends on the slack the plan left
    in rooms and windows. No metadata is attached.
"""
struct SurgicalCaseSequencingProblem <: ProblemGenerator
    n_cases::Int
    n_rooms::Int
    n_surgeons::Int
    n_days::Int
    slot_minutes::Int
    regular_slots::Int
    horizon_slots::Int
    case_specialty::Vector{Symbol}
    case_duration::Vector{Float64}
    case_slots::Vector{Int}
    case_surgeon::Vector{Int}
    case_days::Vector{Vector{Int}}
    case_rooms::Vector{Vector{Int}}
    tardiness_weight::Vector{Float64}
    surgeon_specialty::Vector{Int}
    surgeon_days::Vector{Vector{Int}}
    surgeon_window_start::Vector{Int}
    surgeon_window_end::Vector{Int}
    room_specialty::Vector{Int}
    room_turnover::Int
    surgeon_turnover::Int
    columns::Vector{NTuple{4, Int}}
    costs::Vector{Float64}
    feasible_witness::Union{Nothing, Vector{Int}}
    infeasibility_certificate::Union{Nothing, SurgeonDayOverbookingCertificate}
    feasibility_status::FeasibilityStatus
end

const _SEQ_SLOT = 15
const _SEQ_REGULAR = 32   # 480-minute session
const _SEQ_HORIZON = 40   # latest completion: 600 minutes

"""
    _sequencing_columns(case_slots, case_surgeon, case_days, case_rooms, window_start, window_end)

All `(case, room, day, start)` columns: eligible rooms and days, starts with
the whole case inside the surgeon's window.
"""
function _sequencing_columns(
    case_slots::Vector{Int},
    case_surgeon::Vector{Int},
    case_days::Vector{Vector{Int}},
    case_rooms::Vector{Vector{Int}},
    window_start::Vector{Int},
    window_end::Vector{Int},
)
    cols = NTuple{4, Int}[]
    for o in eachindex(case_slots)
        s = case_surgeon[o]
        for d in case_days[o],
            r in case_rooms[o],
            t in window_start[s]:(window_end[s] - case_slots[o])

            push!(cols, (o, r, d, t))
        end
    end
    return cols
end

function SurgicalCaseSequencingProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 20)
    n_days = target <= 400 ? 1 : (target <= 3000 ? 3 : 5)
    n_specs = clamp(round(Int, 2 + log2(target / 200)), 2, 11)
    spec_ids = _orsched_case_mix(rng, n_specs)
    spec_cum = cumsum([_ORSCHED_SPECIALTIES[k].weight for k in spec_ids])
    room_turnover = rand(rng, 1:2)
    surgeon_turnover = rand(rng, 0:1)

    surgeon_specialty = Int[]
    surgeon_days = Vector{Int}[]
    window_start = Int[]
    regular_end = Int[]
    window_end = Int[]
    room_specialty = Int[]
    room_free = Vector{Int}[]          # per room: next free slot per day

    case_spec = Int[]
    case_duration = Float64[]
    case_slots = Int[]
    case_surgeon = Int[]
    case_rooms = Vector{Int}[]
    case_days = Vector{Int}[]
    planted_day = Int[]
    planted_room = Int[]
    planted_start = Int[]
    surgeon_day_cases = Dict{Tuple{Int, Int}, Vector{Int}}()
    n_vars = 0

    function new_room(k::Int)
        push!(room_specialty, k)
        push!(room_free, zeros(Int, n_days))
        return length(room_specialty)
    end

    # Add one surgeon (and the rooms they need) with a planted list per
    # operating day; stops early once the columns reach the target.
    function plant_surgeon!()
        k = _orsched_pick(rng, spec_cum)
        u = rand(rng)
        ws, we = if u < 0.6
            (0, _SEQ_REGULAR)
        elseif u < 0.8
            (0, rand(rng, 14:20))
        else
            (rand(rng, 14:18), _SEQ_REGULAR)
        end
        overtime = we == _SEQ_REGULAR ? rand(rng, (0, 2, 4)) : rand(rng, 0:2)
        days = sort(shuffle(rng, collect(1:n_days))[1:rand(rng, 1:min(3, n_days))])
        push!(surgeon_specialty, k)
        push!(surgeon_days, days)
        push!(window_start, ws)
        push!(regular_end, we)
        push!(window_end, min(_SEQ_HORIZON, we + overtime))
        s = length(surgeon_specialty)

        for d in days
            n_vars >= target && break
            cluster = [r for r in eachindex(room_specialty) if room_specialty[r] == spec_ids[k]]
            free = [room_free[r][d] for r in cluster]
            r = if !isempty(cluster) && minimum(free) <= ws
                cluster[argmin(free)]
            else
                new_room(spec_ids[k])
            end
            t = max(ws, room_free[r][d])
            booked = Int[]
            overtime_used = false
            for _ in 1:12
                n_vars >= target && break
                surgery_type = _orsched_sample_benchmark_type(rng, spec_ids[k])
                dur = clamp(5.0 * round(_orsched_type_mean(surgery_type) / 5.0), 20.0, 480.0)
                p = cld(Int(dur), _SEQ_SLOT)
                if t + p <= we
                    # within the regular window
                elseif !overtime_used && t + p <= window_end[s] && rand(rng) < 0.3
                    overtime_used = true
                else
                    break
                end
                push!(case_spec, k)
                push!(case_duration, dur)
                push!(case_slots, p)
                push!(case_surgeon, s)
                push!(planted_day, d)
                push!(planted_room, r)
                push!(planted_start, t)
                push!(booked, length(case_slots))
                # Eligibility: planted room plus 1-3 others of the cluster;
                # planted day plus each other operating day with prob. 1/2.
                others = [
                    q for
                    q in eachindex(room_specialty) if room_specialty[q] == spec_ids[k] && q != r
                ]
                extra = shuffle(rng, others)[1:min(length(others), rand(rng, 1:3))]
                push!(case_rooms, sort(vcat(r, extra)))
                case_day_list = [d]
                for e in days
                    e != d && rand(rng) < 0.5 && push!(case_day_list, e)
                end
                push!(case_days, sort(case_day_list))
                n_vars +=
                    length(case_rooms[end]) * length(case_days[end]) * (window_end[s] - ws - p + 1)
                t += p + max(room_turnover, surgeon_turnover)
                overtime_used && break
            end
            room_free[r][d] = if isempty(booked)
                room_free[r][d]
            else
                t - max(room_turnover, surgeon_turnover) + room_turnover
            end
            surgeon_day_cases[(s, d)] = booked
        end
        return nothing
    end
    exact_columns() = sum(
        length(case_rooms[o]) *
        length(case_days[o]) *
        (window_end[case_surgeon[o]] - window_start[case_surgeon[o]] - case_slots[o] + 1) for
        o in eachindex(case_slots);
        init=0,
    )
    while n_vars < target
        plant_surgeon!()
    end

    certificate = nothing
    overbooked = nothing
    if feasibility_status == infeasible
        # Restrict the planted cases of the surgeon-day with the most planted
        # minutes to that day, then top the columns back up with more surgeons.
        keys_sorted = sort!(collect(keys(surgeon_day_cases)))
        load(key) = sum(case_slots[o] for o in surgeon_day_cases[key]; init=0)
        overbooked = keys_sorted[argmax([load(key) for key in keys_sorted])]
        for o in surgeon_day_cases[overbooked]
            case_days[o] = [overbooked[2]]
        end
        n_vars = exact_columns()
        while n_vars < target
            plant_surgeon!()
        end
    end

    n_cases = length(case_slots)
    n_rooms = length(room_specialty)
    n_surgeons = length(surgeon_specialty)

    cluster_of = Dict(
        k => [r for r in 1:n_rooms if room_specialty[r] == k] for k in unique(room_specialty)
    )

    add_on(s, d; max_minutes=480.0) = begin
        k = surgeon_specialty[s]
        surgery_type = _orsched_sample_benchmark_type(rng, spec_ids[k])
        dur = clamp(5.0 * round(_orsched_type_mean(surgery_type) / 5.0), 20.0, max_minutes)
        p = min(cld(Int(dur), _SEQ_SLOT), window_end[s] - window_start[s])
        push!(case_spec, k)
        push!(case_duration, dur)
        push!(case_slots, p)
        push!(case_surgeon, s)
        cluster = cluster_of[spec_ids[k]]
        push!(case_rooms, sort(shuffle(rng, cluster)[1:min(length(cluster), rand(rng, 2:4))]))
        push!(case_days, [d])
        length(case_slots)
    end
    if feasibility_status == infeasible
        # Overbook that surgeon-day with add-on cases.
        s, d = overbooked
        cases = copy(surgeon_day_cases[(s, d)])
        available = window_end[s] - window_start[s] + surgeon_turnover
        busy() = sum(case_slots[o] + surgeon_turnover for o in cases; init=0)
        while busy() < 1.05 * available
            push!(cases, add_on(s, d))
        end
        certificate = SurgeonDayOverbookingCertificate(s, d, sort(cases), busy(), available)
    elseif feasibility_status == unknown
        # Add-on cases: up to ~20 surgeons (independent of hospital size, as
        # a week's urgent add-ons are) receive one extra case, bookable on any
        # of the surgeon's operating days.
        n_add = round(Int, rand(rng, Uniform(0.0, 0.5)) * min(n_surgeons, 40))
        for s in sort(shuffle(rng, collect(1:n_surgeons))[1:n_add])
            o = add_on(s, surgeon_days[s][1]; max_minutes=90.0)
            case_days[o] = copy(surgeon_days[s])
        end
    end
    n_cases = length(case_slots)

    tardiness_weight = Vector{Float64}(undef, n_cases)
    for o in 1:n_cases
        u = rand(rng)
        tardiness_weight[o] = if u < 0.12
            rand(rng, Uniform(4.0, 8.0))
        else
            (u < 0.40 ? rand(rng, Uniform(2.0, 4.0)) : rand(rng, Uniform(0.5, 2.0)))
        end
    end

    columns = _sequencing_columns(
        case_slots, case_surgeon, case_days, case_rooms, window_start, window_end
    )
    # Exact sizing: drop surplus columns at random (room/start combinations
    # unavailable for equipment reasons), never a case's planted column and
    # never a case's last column. Dropping columns only restricts the model,
    # so the overbooking certificate stays valid.
    if length(columns) > target
        protected = Set{NTuple{4, Int}}(
            (o, planted_room[o], planted_day[o], planted_start[o]) for o in eachindex(planted_day)
        )
        remaining = zeros(Int, n_cases)
        for (o, _, _, _) in columns
            remaining[o] += 1
        end
        keep = trues(length(columns))
        surplus = length(columns) - target
        for c in randperm(rng, length(columns))
            surplus == 0 && break
            o = columns[c][1]
            (columns[c] in protected || remaining[o] <= 1) && continue
            keep[c] = false
            remaining[o] -= 1
            surplus -= 1
        end
        columns = columns[keep]
    end
    room_bias = [rand(rng, Uniform(0.0, 3.0)) for _ in 1:n_rooms]
    costs = Vector{Float64}(undef, length(columns))
    for (c, (o, r, d, t)) in enumerate(columns)
        finish = t + case_slots[o]
        tardy = max(0, finish - _SEQ_REGULAR) * _SEQ_SLOT
        costs[c] = round(
            tardiness_weight[o] * tardy + 0.05 * finish * _SEQ_SLOT + room_bias[r] + 0.5 * (d - 1);
            digits=4,
        )
    end

    witness = nothing
    if feasibility_status == feasible
        index_of = Dict(col => c for (c, col) in enumerate(columns))
        witness = [
            index_of[(o, planted_room[o], planted_day[o], planted_start[o])] for o in 1:n_cases
        ]
    end

    return SurgicalCaseSequencingProblem(
        n_cases,
        n_rooms,
        n_surgeons,
        n_days,
        _SEQ_SLOT,
        _SEQ_REGULAR,
        _SEQ_HORIZON,
        [_ORSCHED_SPECIALTIES[spec_ids[k]].name for k in case_spec],
        case_duration,
        case_slots,
        case_surgeon,
        case_days,
        case_rooms,
        tardiness_weight,
        surgeon_specialty,
        surgeon_days,
        window_start,
        window_end,
        room_specialty,
        room_turnover,
        surgeon_turnover,
        columns,
        costs,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    _sequencing_rows(prob) -> (assignment, room_rows, surgeon_rows)

Row structure shared by `build_model` and the tests: one assignment row per
case, and capacity rows per (room, day, slot) and (surgeon, day, slot) listing
the columns that occupy the slot (room: duration plus room turnover;
surgeon: duration plus surgeon turnover). Capacity rows with fewer than two
columns cannot bind (`x <= 1` already) and are omitted.
"""
function _sequencing_rows(prob::SurgicalCaseSequencingProblem)
    assignment = [Int[] for _ in 1:prob.n_cases]
    room_rows = Dict{NTuple{3, Int}, Vector{Int}}()
    surgeon_rows = Dict{NTuple{3, Int}, Vector{Int}}()
    for (c, (o, r, d, t)) in enumerate(prob.columns)
        push!(assignment[o], c)
        p = prob.case_slots[o]
        for τ in t:(t + p + prob.room_turnover - 1)
            push!(get!(room_rows, (r, d, τ), Int[]), c)
        end
        s = prob.case_surgeon[o]
        for τ in t:(t + p + prob.surgeon_turnover - 1)
            push!(get!(surgeon_rows, (s, d, τ), Int[]), c)
        end
    end
    keep(rows) = [(key, rows[key]) for key in sort!(collect(keys(rows))) if length(rows[key]) >= 2]
    return assignment, keep(room_rows), keep(surgeon_rows)
end

"""
    build_model(prob::SurgicalCaseSequencingProblem)

Build the time-indexed case scheduling model. Deterministic - uses only the
struct's fields (see [`_sequencing_rows`](@ref) for the row structure).
"""
function build_model(prob::SurgicalCaseSequencingProblem)
    model = Model()
    n = length(prob.columns)
    @variable(model, x[1:n], Bin)
    @objective(model, Min, sum(prob.costs[c] * x[c] for c in 1:n))
    assignment, room_rows, surgeon_rows = _sequencing_rows(prob)
    for cols in assignment
        @constraint(model, sum(x[c] for c in cols; init=AffExpr(0.0)) == 1)
    end
    for (_, cols) in room_rows
        @constraint(model, sum(x[c] for c in cols) <= 1)
    end
    for (_, cols) in surgeon_rows
        @constraint(model, sum(x[c] for c in cols) <= 1)
    end
    return model
end

# Register the variant
register_variant(
    :operating_room_scheduling,
    :case_sequencing,
    SurgicalCaseSequencingProblem,
    "Time-indexed weekly surgical case scheduling: rooms, days and 15-minute start slots with room and surgeon capacity per slot, turnovers, surgeon windows, and weighted tardiness";
    tags=[:healthcare, :time_indexed, :partitioning, :packing],
)
