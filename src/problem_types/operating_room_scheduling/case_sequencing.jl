using JuMP
using Random
using Distributions

"""
    SurgicalCaseSequencingProblem <: ProblemGenerator

Generator for the daily surgical case scheduling problem: allocate a day's
elective surgeries to operating rooms and surgeons and sequence them within
each resource. This is the allocation-plus-sequencing MILP of Maaroufi, Camus
& Korbaa (IEEE SMC 2016), closely related to surgical case scheduling as a
generalized job shop (Pham & Klinkert, EJOR 2008).

# Overview
A surgical suite opens `n_rooms` ORs for one day (480-minute regular sessions)
with `n_surgeons` surgeons, each available in a time window (full day, morning,
or afternoon; occasionally extending past the session close when a surgeon
stays late). Every surgery has a planned (lognormal, per-specialty) duration,
a set of eligible rooms (equipment restrictions), and one or two eligible
surgeons of its specialty. Two surgeries sharing a room may not overlap and
need an OR turnover (15-35 min) between them; two surgeries sharing a surgeon
may not overlap and need a surgeon turnover (10-20 min). Room and surgeon
no-overlap are modeled with big-M disjunctive constraints and one binary
ordering variable per shared-resource pair, exactly as in the reference MILP.
Each surgery has a soft target end (the regular session close): overruns are
absorbed by a tardiness variable and penalized by an urgency weight. The
objective is weighted tardiness plus a small makespan term.

# Model
- `assign_room[o, r] ∈ {0,1}` for eligible rooms, `assign_surgeon[o, s] ∈ {0,1}`
  for eligible surgeons (each surgery gets exactly one of each);
- `room_order` / `surgeon_order` binaries: one per unordered surgery pair per
  shared eligible room / shared eligible surgeon, selecting the pair's order;
- `start[o] ≥ 0` (minutes into the day), `tardiness[o] ≥ 0`, `makespan ≥ 0`.

Constraints: exactly-one room and surgeon per surgery; big-M disjunctions per
shared room (`+ turnover`) and per shared surgeon (`+ surgeon_turnover`);
surgeon availability windows (start no earlier than the window start, the case
completed by the window end, when that surgeon is chosen); soft target-end
rows `start[o] + duration[o] - tardiness[o] ≤ target_end[o]`; makespan rows.

# Feasibility control
Surgeon availability windows are HARD constraints, so feasibility is not
automatic: a specialty's caseload could exceed its eligible surgeons' window
capacity. The constructor therefore builds a full feasible schedule: surgeries
are assigned to surgeons window-capacity-aware (extending eligibility or a
window only when no eligible surgeon fits, and recording window extensions as
surgeons staying late), rooms are load-balanced, and a serial schedule
simulation produces start times that respect every room/surgeon disjunction
and window. For `feasible` instances that schedule is stored in
`feasible_witness` as `(assigned_room, assigned_surgeon, start)` per surgery —
a provably feasible point of the MIP. `unknown` uses the identical always-
feasible construction without storing the witness. For `infeasible` instances
one surgery additionally receives a HARD completion deadline
(`infeasible_surgery`, `hard_deadline`) set below its own duration; since
`start ≥ 0` its completion is at least its duration, contradicting the
deadline already in the LP relaxation (the same certificate pattern as
`job_shop_scheduling`).

# Fields
- `n_surgeries::Int`, `n_rooms::Int`, `n_surgeons::Int`: dimensions
- `surgery_specialty::Vector{Symbol}`: specialty name per surgery
- `surgery_duration::Vector{Float64}`: planned duration (minutes) per surgery
- `eligible_rooms::Vector{Vector{Int}}`: rooms each surgery may use
- `eligible_surgeons::Vector{Vector{Int}}`: surgeons who may perform each surgery
- `surgeon_window_start::Vector{Float64}` / `surgeon_window_end::Vector{Float64}`:
  availability window per surgeon (minutes into the day)
- `room_turnover::Float64`: OR turnover between consecutive cases (minutes)
- `surgeon_turnover::Float64`: surgeon turnover between consecutive cases (minutes)
- `target_end::Vector{Float64}`: soft target completion time per surgery
- `tardiness_weight::Vector{Float64}`: per-minute overrun weight per surgery
- `big_m::Float64`: big-M constant for the disjunctive constraints
- `room_pairs::Vector{Tuple{Int,Int,Int}}`: `(surgery_o, surgery_p, room)` triples
  (o < p) sharing an eligible room — indices of `room_order`
- `surgeon_pairs::Vector{Tuple{Int,Int,Int}}`: `(surgery_o, surgery_p, surgeon)`
  triples (o < p) sharing an eligible surgeon — indices of `surgeon_order`
- `feasible_witness::Union{Nothing,Vector{Tuple{Int,Int,Float64}}}`:
  `(assigned_room, assigned_surgeon, start)` per surgery of the planted
  feasible schedule (only for `feasible` instances)
- `infeasible_surgery::Union{Nothing,Int}`: surgery carrying the hard deadline
  (only for `infeasible` instances)
- `hard_deadline::Union{Nothing,Float64}`: the contradictory hard deadline
- `feasibility_status::FeasibilityStatus`: resolved feasibility status
"""
struct SurgicalCaseSequencingProblem <: ProblemGenerator
    n_surgeries::Int
    n_rooms::Int
    n_surgeons::Int
    surgery_specialty::Vector{Symbol}
    surgery_duration::Vector{Float64}
    eligible_rooms::Vector{Vector{Int}}
    eligible_surgeons::Vector{Vector{Int}}
    surgeon_window_start::Vector{Float64}
    surgeon_window_end::Vector{Float64}
    room_turnover::Float64
    surgeon_turnover::Float64
    target_end::Vector{Float64}
    tardiness_weight::Vector{Float64}
    big_m::Float64
    room_pairs::Vector{Tuple{Int,Int,Int}}
    surgeon_pairs::Vector{Tuple{Int,Int,Int}}
    feasible_witness::Union{Nothing,Vector{Tuple{Int,Int,Float64}}}
    infeasible_surgery::Union{Nothing,Int}
    hard_deadline::Union{Nothing,Float64}
    feasibility_status::FeasibilityStatus
end

# Exact decision-variable count for a sampled candidate: one room- and one
# surgeon-assignment binary per eligibility entry, one ordering binary per
# shared-resource pair, plus start/tardiness per surgery and one makespan.
function _sequencing_var_count(eligible_rooms::Vector{Vector{Int}},
                               eligible_surgeons::Vector{Vector{Int}})
    n_surgeries = length(eligible_rooms)
    n = sum(length, eligible_rooms) + sum(length, eligible_surgeons) +
        2 * n_surgeries + 1
    for o in 1:n_surgeries, p in (o + 1):n_surgeries
        n += length(intersect(eligible_rooms[o], eligible_rooms[p]))
        n += length(intersect(eligible_surgeons[o], eligible_surgeons[p]))
    end
    return n
end

# Capacity-aware assignment + serial schedule simulation producing a provably
# feasible point: each surgery is assigned a surgeon whose remaining window
# capacity fits the case (extending eligibility to a same-specialty surgeon,
# or that surgeon's window — a surgeon staying late — only when no eligible
# surgeon fits), rooms are load-balanced across eligible rooms, and start
# times come from a serial pass respecting room/surgeon disjunctions and
# windows. Returns `(assigned_room, assigned_surgeon, start, window_end,
# eligible_surgeons)` with windows/eligibility possibly repaired.
function _sequencing_build_schedule(durations::Vector{Float64},
                                    eligible_rooms::Vector{Vector{Int}},
                                    eligible_surgeons::Vector{Vector{Int}},
                                    surgeon_specialty_idx::Vector{Int},
                                    surgery_specialty_idx::Vector{Int},
                                    window_start::Vector{Float64},
                                    window_end::Vector{Float64},
                                    room_turnover::Float64,
                                    surgeon_turnover::Float64)
    n_surgeries = length(durations)
    n_surgeons = length(window_start)
    eligible_surgeons = deepcopy(eligible_surgeons)
    window_end = copy(window_end)

    # Surgeon assignment: longest cases first, to the eligible surgeon with
    # the most remaining window capacity (92% utilization cap keeps a buffer
    # for turnovers and makes window extensions rare).
    load = zeros(Float64, n_surgeons)
    assigned_surgeon = zeros(Int, n_surgeries)
    for o in sortperm(durations; rev=true)
        best_s, best_slack = 0, -Inf
        for s in eligible_surgeons[o]
            slack = 0.92 * (window_end[s] - window_start[s]) -
                    (load[s] + (load[s] > 0 ? surgeon_turnover : 0.0) + durations[o])
            if slack > best_slack
                best_s, best_slack = s, slack
            end
        end
        if best_slack < 0
            # No eligible surgeon fits: extend eligibility to the same-specialty
            # surgeon with the most remaining capacity.
            pool = findall(==(surgery_specialty_idx[o]), surgeon_specialty_idx)
            for s in shuffle(pool)
                slack = 0.92 * (window_end[s] - window_start[s]) -
                        (load[s] + (load[s] > 0 ? surgeon_turnover : 0.0) + durations[o])
                if slack > best_slack
                    best_s, best_slack = s, slack
                end
            end
            best_s ∉ eligible_surgeons[o] && push!(eligible_surgeons[o], best_s)
            sort!(eligible_surgeons[o])
        end
        if best_slack < 0
            # Still no fit: the chosen surgeon stays late — extend their window
            # (and capacity) to take the case.
            needed = load[best_s] + (load[best_s] > 0 ? surgeon_turnover : 0.0) + durations[o]
            window_end[best_s] = max(window_end[best_s],
                                     5.0 * ceil((window_start[best_s] + needed / 0.92) / 5.0))
        end
        assigned_surgeon[o] = best_s
        load[best_s] += (load[best_s] > 0 ? surgeon_turnover : 0.0) + durations[o]
    end

    # Room assignment: anchor each surgeon to a home room (the dominant real
    # pattern — a surgeon works one OR per day), chosen among the rooms all of
    # the surgeon's cases can use so per-room loads stay balanced; cases with
    # incompatible room eligibilities are placed individually.
    n_rooms = maximum(maximum.(eligible_rooms))
    room_load = zeros(Float64, n_rooms)
    assigned_room = zeros(Int, n_surgeries)
    cases_of = [Int[] for _ in 1:n_surgeons]
    for o in 1:n_surgeries
        push!(cases_of[assigned_surgeon[o]], o)
    end
    # Half-day surgeons pick home rooms first (they need lightly loaded rooms
    # to fit their windows), then full-day surgeons by descending caseload.
    home_order = sort(collect(1:n_surgeons),
                      by=s -> (window_end[s] - window_start[s],
                               -sum(durations[o] for o in cases_of[s]; init=0.0)))
    for s in home_order
        isempty(cases_of[s]) && continue
        common = intersect([eligible_rooms[o] for o in cases_of[s]]...)
        if !isempty(common)
            r = common[argmin([room_load[r] for r in common])]
            for o in cases_of[s]
                assigned_room[o] = r
                room_load[r] += durations[o] + room_turnover
            end
        else
            for o in cases_of[s]
                r = eligible_rooms[o][argmin([room_load[r] for r in eligible_rooms[o]])]
                assigned_room[o] = r
                room_load[r] += durations[o] + room_turnover
            end
        end
    end

    # Per-room sequencing: each room's cases are packed contiguously, ordered
    # by surgeon window start (full-day surgeons first, half-day surgeons
    # once their window opens), so rooms stay gap-free. Surgeon turnovers are
    # tracked across rooms. Cases that would overrun their surgeon's window
    # are deferred and re-placed afterwards wherever they fit; the surgeon
    # stays late (window extended) only as a last resort.
    start = zeros(Float64, n_surgeries)
    surgeon_free = copy(window_start)
    room_free = zeros(Float64, n_rooms)
    deferred = Int[]
    for r in 1:n_rooms
        cases_r = [o for o in 1:n_surgeries if assigned_room[o] == r]
        sort!(cases_r, by=o -> (window_start[assigned_surgeon[o]],
                                window_end[assigned_surgeon[o]], -durations[o]))
        for o in cases_r
            s = assigned_surgeon[o]
            t = max(room_free[r], window_start[s], surgeon_free[s])
            if t + durations[o] <= window_end[s] + 1e-9
                start[o] = t
                surgeon_free[s] = t + durations[o] + surgeon_turnover
                room_free[r] = t + durations[o] + room_turnover
            else
                push!(deferred, o)
            end
        end
    end
    for o in deferred
        s0 = assigned_surgeon[o]
        best = nothing  # (start, surgeon, room)
        for s in eligible_surgeons[o], r in eligible_rooms[o]
            t = max(room_free[r], window_start[s], surgeon_free[s])
            t + durations[o] <= window_end[s] + 1e-9 || continue
            if best === nothing || t < best[1] - 1e-9
                best = (t, s, r)
            end
        end
        if best === nothing
            r = eligible_rooms[o][argmin([room_free[r] for r in eligible_rooms[o]])]
            t = max(room_free[r], window_start[s0], surgeon_free[s0])
            window_end[s0] = 5.0 * ceil((t + durations[o]) / 5.0)
            best = (t, s0, r)
        end
        (t, s, r) = best
        start[o] = t
        assigned_surgeon[o] = s
        assigned_room[o] = r
        surgeon_free[s] = t + durations[o] + surgeon_turnover
        room_free[r] = t + durations[o] + room_turnover
    end

    return assigned_room, assigned_surgeon, start, window_end, eligible_surgeons
end

# Exact count restricted to the pair binaries (used for the struct fields).
function _sequencing_pair_lists(eligible_rooms::Vector{Vector{Int}},
                                eligible_surgeons::Vector{Vector{Int}})
    room_pairs = Tuple{Int,Int,Int}[]
    surgeon_pairs = Tuple{Int,Int,Int}[]
    n_surgeries = length(eligible_rooms)
    for o in 1:n_surgeries, p in (o + 1):n_surgeries
        for r in intersect(eligible_rooms[o], eligible_rooms[p])
            push!(room_pairs, (o, p, r))
        end
        for s in intersect(eligible_surgeons[o], eligible_surgeons[p])
            push!(surgeon_pairs, (o, p, s))
        end
    end
    return room_pairs, surgeon_pairs
end

"""
    SurgicalCaseSequencingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a daily surgical case sequencing instance near `target_variables`
decision variables (exact count: assignment binaries + shared-resource
ordering binaries + 2 per surgery + 1 makespan). The constructor iterates over
surgery counts, computing the exact variable count of each sampled instance,
and keeps the closest one.

# Arguments
- `target_variables`: Target number of decision variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function SurgicalCaseSequencingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    target = max(target_variables, 20)

    # Scale tier: 4-6 ORs in the reference MILP (Maaroufi et al. 2016); the
    # specialty count grows with the requested size. The room count itself
    # follows the sampled caseload (daily lists run at 75-95% room
    # utilization), with the tier value as a minimum suite size.
    if target <= 150
        n_rooms_min = 2
        n_specs = rand(2:3)
    elseif target <= 700
        n_rooms_min = rand(2:4)
        n_specs = rand(2:4)
    elseif target <= 3000
        n_rooms_min = rand(3:5)
        n_specs = rand(3:5)
    else
        n_rooms_min = rand(4:7)
        n_specs = rand(4:7)
    end

    room_turnover = 5.0 * round(rand(Uniform(15.0, 35.0)) / 5.0)
    surgeon_turnover = 5.0 * round(rand(Uniform(10.0, 20.0)) / 5.0)

    best = nothing
    best_gap = Inf
    # Ordering binaries dominate: ~C(O,2)*(shared rooms + shared surgeons).
    n_surgeries = max(4, round(Int, sqrt(target / max(1.0, n_rooms_min * 0.6))))

    for _ in 1:60
        spec_ids = _orsched_case_mix(n_specs)

        durations = Vector{Float64}(undef, n_surgeries)
        specialties = Vector{Symbol}(undef, n_surgeries)
        spec_index = Vector{Int}(undef, n_surgeries)
        weights = [_ORSCHED_SPECIALTIES[k].weight for k in spec_ids]
        cum = cumsum(weights ./ sum(weights))
        for o in 1:n_surgeries
            k = _orsched_pick(cum)
            profile = _ORSCHED_SPECIALTIES[spec_ids[k]]
            durations[o] = _orsched_sample_duration(profile.mean, profile.cv)
            specialties[o] = profile.name
            spec_index[o] = spec_ids[k]
        end

        # Surgeon pool: per-specialty surgeon counts follow the specialty's
        # total OR-time demand including turnovers (a full-day surgeon covers
        # 0.85 x 480 = 408 minutes), as in real daily block staffing. Counts
        # are ceiled so aggregate capacity always covers the caseload even
        # before the window repair below.
        surgeon_specialty = Int[]
        for k in spec_ids
            cases_k = [o for o in 1:n_surgeries if spec_index[o] == k]
            isempty(cases_k) && continue
            eff_load_k = sum(durations[o] for o in cases_k) +
                         length(cases_k) * surgeon_turnover
            n_k = max(1, ceil(Int, eff_load_k / (0.85 * 480.0) * rand(Uniform(1.0, 1.2))))
            append!(surgeon_specialty, fill(k, n_k))
        end
        n_surgeons = length(surgeon_specialty)

        # Surgeon availability windows: the first surgeon of each specialty is
        # full-day (so every specialty has full-day coverage); the rest are
        # full day, morning, or afternoon.
        window_start = Vector{Float64}(undef, n_surgeons)
        window_end = Vector{Float64}(undef, n_surgeons)
        seen_specialty = Set{Int}()
        for s in 1:n_surgeons
            u = rand()
            if !(surgeon_specialty[s] in seen_specialty) || u < 0.55
                window_start[s], window_end[s] = 0.0, 480.0
            elseif u < 0.75
                window_start[s], window_end[s] = 0.0, 5.0 * round(rand(Uniform(210.0, 300.0)) / 5.0)
            else
                window_start[s], window_end[s] = 5.0 * round(rand(Uniform(180.0, 300.0)) / 5.0), 480.0
            end
            push!(seen_specialty, surgeon_specialty[s])
        end
        # Capacity sufficiency repair: each specialty's total window capacity
        # (at 85% utilization) must cover its caseload including turnovers;
        # flip half-day surgeons to full-day until it does.
        for k in spec_ids
            pool = findall(==(k), surgeon_specialty)
            isempty(pool) && continue
            cases_k = [o for o in 1:n_surgeries if spec_index[o] == k]
            eff_load_k = sum(durations[o] for o in cases_k; init=0.0) +
                         length(cases_k) * surgeon_turnover
            cap_k() = sum(0.85 * (window_end[s] - window_start[s]) for s in pool)
            for s in shuffle(pool)
                cap_k() >= 1.02 * eff_load_k && break
                window_start[s], window_end[s] = 0.0, 480.0
            end
        end

        # Room count follows the caseload: enough ORs that the day's list
        # (durations plus turnovers plus the unavoidable morning idle time of
        # rooms hosting afternoon surgeons) fills 75-95% of session minutes.
        idle_allowance = sum(window_start[s] for s in 1:n_surgeons
                             if window_start[s] > 0; init=0.0)
        total_minutes = sum(durations) + n_surgeries * room_turnover +
                        0.7 * idle_allowance
        n_rooms = max(n_rooms_min,
                      ceil(Int, total_minutes / (480.0 * rand(Uniform(0.75, 0.95)))))

        # Eligible surgeons: one or two surgeons of the surgery's specialty.
        eligible_surgeons = Vector{Vector{Int}}(undef, n_surgeries)
        for o in 1:n_surgeries
            pool = findall(==(spec_index[o]), surgeon_specialty)
            eligible_surgeons[o] = rand() < 0.65 ? [rand(pool)] :
                                   shuffle(pool)[1:min(2, length(pool))]
        end

        # Eligible rooms: most surgeries can use any room; some need specially
        # equipped rooms (hybrid OR, robotics) and are restricted to a subset.
        eligible_rooms = Vector{Vector{Int}}(undef, n_surgeries)
        for o in 1:n_surgeries
            if rand() < 0.75 || n_rooms == 1
                eligible_rooms[o] = collect(1:n_rooms)
            else
                k = rand(1:max(1, n_rooms ÷ 2))
                eligible_rooms[o] = sort(randperm(n_rooms)[1:k])
            end
        end

        # Feasible schedule (witness): capacity-aware assignment + simulation;
        # may repair eligibility and window ends.
        assigned_room, assigned_surgeon, start, final_window_end, final_eligible =
            _sequencing_build_schedule(durations, eligible_rooms, eligible_surgeons,
                                       surgeon_specialty, spec_index,
                                       window_start, window_end,
                                       room_turnover, surgeon_turnover)

        total = _sequencing_var_count(eligible_rooms, final_eligible)
        gap = abs(total - target) / target
        if gap < best_gap
            best_gap = gap
            best = (durations=durations, specialties=specialties,
                    eligible_rooms=eligible_rooms, eligible_surgeons=final_eligible,
                    window_start=window_start, window_end=final_window_end,
                    n_rooms=n_rooms, n_surgeons=n_surgeons, assigned_room=assigned_room,
                    assigned_surgeon=assigned_surgeon, start=start)
            gap <= 0.05 && break
        end
        scale = sqrt(target / max(total, 1))
        new_n = clamp(round(Int, n_surgeries * scale), 4, 3 * n_surgeries)
        new_n == n_surgeries && (new_n = total < target ? n_surgeries + 1 : max(4, n_surgeries - 1))
        n_surgeries = new_n
    end

    durations = best.durations
    n_rooms = best.n_rooms
    n_surgeons = best.n_surgeons
    eligible_rooms = best.eligible_rooms
    eligible_surgeons = best.eligible_surgeons
    window_start = best.window_start
    window_end = best.window_end
    n_surgeries = length(durations)

    room_pairs, surgeon_pairs = _sequencing_pair_lists(eligible_rooms, eligible_surgeons)

    # Soft target ends at the regular session close; overrun weights scale with
    # clinical urgency (most cases are routine electives).
    target_end = fill(480.0, n_surgeries)
    tardiness_weight = Vector{Float64}(undef, n_surgeries)
    for o in 1:n_surgeries
        u = rand()
        tardiness_weight[o] = u < 0.12 ? rand(Uniform(4.0, 8.0)) :
                              u < 0.40 ? rand(Uniform(2.0, 4.0)) :
                              rand(Uniform(0.5, 2.0))
    end

    # Big-M valid for every feasible start: starts complete within a surgeon
    # window (window ends may reach past the session close when a surgeon stays
    # late) plus one duration and turnover.
    big_m = max(480.0, maximum(window_end)) + maximum(durations) +
            room_turnover + surgeon_turnover

    witness = nothing
    infeasible_surgery = nothing
    hard_deadline = nothing
    if feasibility_status == feasible
        witness = [(best.assigned_room[o], best.assigned_surgeon[o], best.start[o])
                   for o in 1:n_surgeries]
    elseif feasibility_status == infeasible
        # Hard completion deadline below the surgery's own duration. Completion
        # is start + duration ≥ duration (start ≥ 0), so the deadline is
        # contradictory already in the LP relaxation.
        victim = argmax(durations)
        infeasible_surgery = victim
        hard_deadline = floor(0.5 * durations[victim] / 5.0) * 5.0
    end

    return SurgicalCaseSequencingProblem(
        n_surgeries, n_rooms, n_surgeons,
        best.specialties, durations, eligible_rooms, eligible_surgeons,
        window_start, window_end, room_turnover, surgeon_turnover,
        target_end, tardiness_weight, big_m, room_pairs, surgeon_pairs,
        witness, infeasible_surgery, hard_deadline, feasibility_status,
    )
end

"""
    build_model(prob::SurgicalCaseSequencingProblem)

Build the JuMP model for the daily surgical case sequencing problem.
Deterministic — uses only data from the struct fields.

# Returns
- `model`: The JuMP model
"""
function build_model(prob::SurgicalCaseSequencingProblem)
    model = Model()

    n_surgeries = prob.n_surgeries
    n_rooms = prob.n_rooms
    n_surgeons = prob.n_surgeons
    dur = prob.surgery_duration
    M = prob.big_m

    @variable(model, assign_room[o in 1:n_surgeries, r in 1:n_rooms;
                                 r in prob.eligible_rooms[o]], Bin)
    @variable(model, assign_surgeon[o in 1:n_surgeries, s in 1:n_surgeons;
                                    s in prob.eligible_surgeons[o]], Bin)
    @variable(model, room_order[1:length(prob.room_pairs)], Bin)
    @variable(model, surgeon_order[1:length(prob.surgeon_pairs)], Bin)
    @variable(model, start[1:n_surgeries] >= 0)
    @variable(model, tardiness[1:n_surgeries] >= 0)
    @variable(model, makespan >= 0)

    # Exactly one room and one surgeon per surgery.
    for o in 1:n_surgeries
        @constraint(model, sum(assign_room[o, r] for r in prob.eligible_rooms[o]) == 1)
        @constraint(model, sum(assign_surgeon[o, s] for s in prob.eligible_surgeons[o]) == 1)
    end

    # Room no-overlap with OR turnover (big-M disjunctions per shared room).
    # `room_order[idx] = 1` orders o before p in room r; both disjunctions are
    # vacuous unless both surgeries take that room.
    for (idx, (o, p, r)) in enumerate(prob.room_pairs)
        @constraint(model, start[p] >= start[o] + dur[o] + prob.room_turnover -
            M * (3 - room_order[idx] - assign_room[o, r] - assign_room[p, r]))
        @constraint(model, start[o] >= start[p] + dur[p] + prob.room_turnover -
            M * (2 + room_order[idx] - assign_room[o, r] - assign_room[p, r]))
    end

    # Surgeon no-overlap with surgeon turnover (per shared surgeon).
    for (idx, (o, p, s)) in enumerate(prob.surgeon_pairs)
        @constraint(model, start[p] >= start[o] + dur[o] + prob.surgeon_turnover -
            M * (3 - surgeon_order[idx] - assign_surgeon[o, s] - assign_surgeon[p, s]))
        @constraint(model, start[o] >= start[p] + dur[p] + prob.surgeon_turnover -
            M * (2 + surgeon_order[idx] - assign_surgeon[o, s] - assign_surgeon[p, s]))
    end

    # Surgeon availability windows bind when the surgeon is chosen.
    for o in 1:n_surgeries, s in prob.eligible_surgeons[o]
        @constraint(model, start[o] >= prob.surgeon_window_start[s] -
            M * (1 - assign_surgeon[o, s]))
        @constraint(model, start[o] + dur[o] <= prob.surgeon_window_end[s] +
            M * (1 - assign_surgeon[o, s]))
    end

    # Soft target ends (regular session close) and makespan.
    for o in 1:n_surgeries
        if prob.infeasible_surgery == o
            # Hard completion deadline (infeasibility certificate).
            @constraint(model, start[o] + dur[o] <= prob.hard_deadline)
        end
        @constraint(model, start[o] + dur[o] - tardiness[o] <= prob.target_end[o])
        @constraint(model, makespan >= start[o] + dur[o])
    end

    @objective(model, Min,
        sum(prob.tardiness_weight[o] * tardiness[o] for o in 1:n_surgeries) +
        0.05 * makespan)

    return model
end

# Register the variant
register_variant(
    :operating_room_scheduling,
    :case_sequencing,
    SurgicalCaseSequencingProblem,
    "Daily surgical case allocation and sequencing across operating rooms with surgeon conflicts, turnover times, and weighted tardiness",
)
