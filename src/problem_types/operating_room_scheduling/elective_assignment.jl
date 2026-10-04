using JuMP
using Random
using Distributions

"""
    ElectiveSurgeryAssignmentProblem <: ProblemGenerator

Generator for elective surgery assignment (advance scheduling) under a master
surgical schedule — the most common MILP in the operating-room planning
literature (Marques, Captivo & Vaz Pato, OR Spectrum 2012; Cardoen,
Demeulemeester & Beliën, EJOR 2010 survey).

# Overview

A hospital runs a master surgical schedule (MSS): each open `(room, day)` block
is dedicated to a single surgical specialty and has a session length (mostly
480-minute full sessions, some 240-minute half sessions, rare 780-minute long
sessions). A waiting list of elective surgeries must be assigned to compatible
blocks over a 1-2 week horizon. A surgery is admissible to a block when the
block's specialty matches, the surgery's surgeon works that day, and the day is
no later than the surgery's clinical deadline. Each case occupies its planned
(lognormal, per-specialty) duration plus an OR turnover time (15-35 minutes).
Block load may exceed the session length only through costly overtime.
Surgeries need not be scheduled: postponement incurs an urgency-weighted
penalty, except clinically urgent cases, which are mandatory.

# Model

  - `assign[a] ∈ {0,1}` for each admissible `(surgery, room, day)` triple;
  - `postpone[i] ∈ {0,1}` per non-mandatory surgery (mandatory cases get no
    postponement column at all, rather than a column fixed to zero);
  - `overtime[q] ∈ [0, max_overtime]` per open block.

Minimize total postponement penalty plus overtime cost, subject to:

  - each surgery is either assigned to exactly one block or postponed;
  - block capacity: assigned durations + turnovers ≤ session length + overtime;
  - surgeon-day budgets: a surgeon's assigned minutes per day ≤ their budget.

# Feasibility control

For `feasible` instances a greedy earliest-deadline/best-fit schedule is
constructed first; mandatory urgent cases are then designated from cases in
that schedule, so urgency is never weakened to repair feasibility. The
assignment is stored in `feasible_witness` as admissible-triple indices — a
provably feasible point. Every status designates its urgent (mandatory)
cases from such a plan, so no urgent case is ever stranded. For `infeasible`
instances a surgeon overload is planted on top ([`SurgeonOverloadCertificate`],
`infeasibility_certificate`): three or more of one surgeon's cases, each
admissible on two or three shared days and fitting each of them, become
mandatory while every one of those days is budgeted only the longest case, so
the total budget is at most 90% of the cases' minutes. Summing the surgeon's
budget rows against the cases' assignment rows refutes the LP relaxation, but
no single row is contradictory and no bound is tightened, so presolve does not
see it. For `unknown`, overtime availability is a global draw (the cap is
scaled by U(0, 1)), a share U(0.5, 1.0) of the planned cases are already booked
(mandatory), and urgent referrals - a random 0-100% of the cases the plan could
not place - are added ([`_orsched_add_referrals!`]); booked cases and
referrals must fit at least two days on their own. Whether the LP can still
fit everything depends on these global draws, so both outcomes occur at every
size (measured 2/6, 4/6, 1/6 infeasible at 3k, 30k, 100k).

Admissible triples require the case to fit its surgeon's day budget and the
session plus overtime on its own, and surgeon-day rows that cannot bind are
omitted (also in `robust_elective`), so presolve keeps 98-100% of rows at 10k+.

The hospital grows with the target from 2,500 variables up (about
`sqrt(target * specialties / 150)` rooms, see [`_orsched_hospital_scale`]), and the surgeon
pool with the waiting list, so the list stays proportionate to OR capacity.

# Fields

  - `n_surgeries::Int`, `n_rooms::Int`, `n_days::Int`, `n_specialties::Int`: dimensions
  - `specialty_names::Vector{Symbol}`: specialty of each position 1..n_specialties
  - `surgery_specialty::Vector{Int}`: specialty index per surgery
  - `surgery_duration::Vector{Float64}`: planned duration (minutes) per surgery
  - `surgery_urgency::Vector{Symbol}`: `:urgent` / `:semi_urgent` / `:routine`
  - `surgery_deadline::Vector{Int}`: latest day the surgery may be scheduled
  - `postponement_penalty::Vector{Float64}`: cost of postponing each surgery
  - `surgery_surgeon::Vector{Int}`: surgeon performing each surgery
  - `mandatory::BitVector`: surgeries that must be scheduled (postpone == 0)
  - `surgeon_specialty::Vector{Int}`: specialty index per surgeon
  - `surgeon_budget::Matrix{Float64}`: operating minutes of surgeon s on day d (0 = off)
  - `mss::Matrix{Int}`: specialty index of each (room, day) block (0 = closed)
  - `session_length::Matrix{Float64}`: session minutes of each (room, day) (0 = closed)
  - `turnover::Float64`: OR turnover time between consecutive cases (minutes)
  - `max_overtime::Float64`: overtime cap per block (minutes)
  - `overtime_cost::Float64`: cost per overtime minute
  - `admissible::Vector{Tuple{Int,Int,Int}}`: admissible `(surgery, room, day)` triples
  - `open_blocks::Vector{Tuple{Int,Int}}`: open `(room, day)` blocks (overtime indices)
  - `feasible_witness::Union{Nothing,Vector{Int}}`: triple indices of the planted
    feasible schedule (only for `feasible` instances)
  - `infeasibility_certificate::Union{Nothing,SurgeonOverloadCertificate}`
    (only for `infeasible` instances)
  - `feasibility_status::FeasibilityStatus`: resolved feasibility status
"""
struct ElectiveSurgeryAssignmentProblem <: ProblemGenerator
    n_surgeries::Int
    n_rooms::Int
    n_days::Int
    n_specialties::Int
    specialty_names::Vector{Symbol}
    surgery_specialty::Vector{Int}
    surgery_duration::Vector{Float64}
    surgery_duration_sd::Vector{Float64}
    surgery_type_id::Vector{Int}
    surgery_urgency::Vector{Symbol}
    surgery_deadline::Vector{Int}
    postponement_penalty::Vector{Float64}
    surgery_surgeon::Vector{Int}
    mandatory::BitVector
    surgeon_specialty::Vector{Int}
    surgeon_budget::Matrix{Float64}
    mss::Matrix{Int}
    session_length::Matrix{Float64}
    turnover::Float64
    max_overtime::Float64
    overtime_cost::Float64
    admissible::Vector{Tuple{Int, Int, Int}}
    open_blocks::Vector{Tuple{Int, Int}}
    feasible_witness::Union{Nothing, Vector{Int}}
    infeasibility_certificate::Union{Nothing, SurgeonOverloadCertificate}
    feasibility_status::FeasibilityStatus
end

# Admissible (surgery, room, day) triples: specialty match with the block, the
# surgery's surgeon works that day, and the day is within the clinical deadline.
function _elective_admissible_triples(
    n_surgeries::Int,
    surgery_specialty::Vector{Int},
    surgery_deadline::Vector{Int},
    surgery_surgeon::Vector{Int},
    surgeon_budget::Matrix{Float64},
    mss::Matrix{Int},
    duration::Vector{Float64},
    session::Matrix{Float64},
    room_allowance::Float64,
)
    # A triple is admissible only if the case fits its surgeon's day budget
    # and the session (plus `room_allowance` minutes of overtime, net of
    # turnover) on its own; otherwise the column could never be fully used
    # and only adds singleton rows that presolve turns into bounds.
    triples = Tuple{Int, Int, Int}[]
    n_rooms, n_days = size(mss)
    for i in 1:n_surgeries, d in 1:min(n_days, surgery_deadline[i])
        surgeon_budget[surgery_surgeon[i], d] >= duration[i] || continue
        for r in 1:n_rooms
            (mss[r, d] == surgery_specialty[i] && session[r, d] + room_allowance >= duration[i]) &&
                push!(triples, (i, r, d))
        end
    end
    return triples
end

# Assign each surgery a surgeon of its specialty, rotating so surgeon caseloads
# stay balanced. Specialties without a surgeon cannot occur: the surgeon pool
# creates at least one surgeon per specialty that has cases.
function _elective_assign_surgeons(
    rng::AbstractRNG, surgery_specialty::Vector{Int}, surgeon_specialty::Vector{Int}
)
    pools = Dict(k => findall(==(k), surgeon_specialty) for k in unique(surgeon_specialty))
    assignment = Vector{Int}(undef, length(surgery_specialty))
    for k in keys(pools)
        cases = shuffle(rng, findall(==(k), surgery_specialty))
        pool = pools[k]
        for (offset, i) in enumerate(cases)
            assignment[i] = pool[mod1(offset, length(pool))]
        end
    end
    return assignment
end

"""
    ElectiveSurgeryAssignmentProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an elective surgery assignment instance near `target_variables`
decision variables (`|admissible| + #non-mandatory + |open_blocks|`). The
constructor iterates over waiting-list sizes, computing the exact variable
count of each sampled instance, and keeps the closest one.

# Arguments

  - `target_variables`: Target number of decision variables
  - `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
  - `seed`: Random seed for reproducibility
"""
function ElectiveSurgeryAssignmentProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)

    target = max(target_variables, 20)
    n_rooms, n_days, n_specs = _orsched_hospital_scale(rng, target)
    turnover = 5.0 * round(rand(rng, Uniform(15.0, 35.0)) / 5.0)
    max_overtime = 5.0 * round(rand(rng, Uniform(60.0, 120.0)) / 5.0)
    overtime_cost = rand(rng, Uniform(2.0, 6.0))
    n_open_estimate = round(Int, n_rooms * n_days * 0.9)

    best = nothing
    best_gap = Inf
    n_surgeries = max(4, round(Int, (target - n_open_estimate) / 4.0))

    for _ in 1:60
        spec_ids = _orsched_case_mix(rng, n_specs)
        mss, session = _orsched_master_schedule(rng, n_rooms, n_days, spec_ids)
        wl = _orsched_waiting_list(
            rng, n_surgeries, spec_ids, n_days; allow_urgent=false
        )
        counts = [count(==(k), wl.specialty) for k in 1:n_specs]
        surgeon_specialty, surgeon_budget = _orsched_surgeon_pool(rng, counts, n_days, mss)
        surgery_surgeon = _elective_assign_surgeons(rng, wl.specialty, surgeon_specialty)
        admissible = _elective_admissible_triples(
            n_surgeries,
            wl.specialty,
            wl.deadline,
            surgery_surgeon,
            surgeon_budget,
            mss,
            wl.duration,
            session,
            max_overtime - turnover,
        )
        open_blocks = [(r, d) for d in 1:n_days for r in 1:n_rooms if session[r, d] > 0]
        # Mandatory cases carry no postponement column: about the urgent share
        # designated from the planted schedule.
        n_mandatory = round(Int, wl.requested_urgent_fraction * n_surgeries)
        total = length(admissible) + n_surgeries - n_mandatory + length(open_blocks)
        gap = abs(total - target) / target
        if gap < best_gap
            best_gap = gap
            best = (
                spec_ids=spec_ids,
                mss=mss,
                session=session,
                wl=wl,
                surgeon_specialty=surgeon_specialty,
                surgeon_budget=surgeon_budget,
                surgery_surgeon=surgery_surgeon,
                admissible=admissible,
                open_blocks=open_blocks,
                n_surgeries=n_surgeries,
            )
            gap <= 0.05 && break
        end
        # Adjust the waiting-list size toward the target.
        scale = target / max(total, 1)
        new_n = clamp(round(Int, n_surgeries * scale), 4, 4 * n_surgeries)
        new_n == n_surgeries && (new_n = total < target ? n_surgeries + 1 : max(4, n_surgeries - 1))
        n_surgeries = new_n
    end

    spec_ids = best.spec_ids
    mss, session = best.mss, best.session
    wl = best.wl
    surgeon_specialty, surgeon_budget = best.surgeon_specialty, best.surgeon_budget
    surgery_surgeon = best.surgery_surgeon
    admissible = best.admissible
    open_blocks = best.open_blocks
    n_surgeries = best.n_surgeries

    specialty_names = [_ORSCHED_SPECIALTIES[k].name for k in spec_ids]
    urgency = collect(wl.urgency)
    deadline = collect(wl.deadline)
    penalty = collect(wl.penalty)

    witness = nothing
    certificate = nothing
    # `room_fits(i, d)`: case i fits some admissible room session (with
    # overtime) on day d on its own; `has_option[i]`: it also fits its
    # surgeon's budget on such a day. Urgent cases without such a slot are
    # referred elsewhere rather than made mandatory: a mandatory case that
    # cannot fit any single slot is a one-row contradiction presolve spots,
    # not a scheduling problem.
    if feasibility_status == unknown
        # Overtime availability (anaesthesia/nursing cover) is a global draw:
        # with little overtime the booked list plus referrals may no longer
        # fit, with plenty they do - at every hospital size.
        max_overtime = 5.0 * round(max_overtime * rand(rng) / 5.0)
    end
    room_fit_days = [Int[] for _ in 1:n_surgeries]
    for (i, r, d) in admissible
        if session[r, d] + max_overtime >= wl.duration[i] + turnover && !(d in room_fit_days[i])
            push!(room_fit_days[i], d)
        end
    end
    room_fits(i, d) = d in room_fit_days[i]
    # Referral candidates need two fitting days, so presolve cannot pin them.
    has_option = [
        count(surgeon_budget[surgery_surgeon[i], d] >= wl.duration[i] for d in room_fit_days[i]) >= 2
        for i in 1:n_surgeries
    ]
    mandatory = BitVector(urgency[i] == :urgent && has_option[i] for i in 1:n_surgeries)

    let
        # The urgent list is designated from a greedy plan that respects every
        # capacity (for every status), so for `feasible` the plan is a witness.
        # For `unknown`, referrals add urgent cases the plan could not place;
        # for `infeasible`, the surgeon overload is planted on top.
        # Plant a schedule first and then designate mandatory cases from the
        # scheduled set.  Clinical urgency is never weakened to repair the
        # heuristic.
        slots_for = [Int[] for _ in 1:n_surgeries]
        for (a, (i, _, _)) in enumerate(admissible)
            push!(slots_for[i], a)
        end
        rem_room = copy(session)
        rem_surg = copy(surgeon_budget)
        function consume!(a::Int, i::Int)
            (_, r, d) = admissible[a]
            need_room = wl.duration[i] + turnover
            need_surg = wl.duration[i]
            if rem_room[r, d] >= need_room && rem_surg[surgery_surgeon[i], d] >= need_surg
                rem_room[r, d] -= need_room
                rem_surg[surgery_surgeon[i], d] -= need_surg
                return true
            end
            return false
        end
        assignment = _orsched_greedy_schedule(
            n_surgeries, urgency, deadline, wl.duration, slots_for, length(admissible), consume!
        )
        mandatory = _orsched_designate_mandatory!(
            rng, urgency, deadline, penalty, assignment, wl.requested_urgent_fraction
        )
        if feasibility_status == feasible
            witness = [assignment[i] for i in 1:n_surgeries if assignment[i] > 0]
        elseif feasibility_status == unknown
            # Most planned cases are already booked with the patient.
            booked = rand(rng, Uniform(0.5, 1.0))
            for i in 1:n_surgeries
                (assignment[i] > 0 && has_option[i] && rand(rng) < booked) && (mandatory[i] = true)
            end
            _orsched_add_referrals!(rng, mandatory, urgency, penalty, assignment, has_option)
        end
    end
    if feasibility_status == infeasible
        case_days = [Int[] for _ in 1:n_surgeries]
        for (i, _, d) in admissible
            d in case_days[i] || push!(case_days[i], d)
        end
        certificate = _orsched_plant_surgeon_overload!(
            rng,
            surgeon_budget,
            surgery_surgeon,
            wl.duration,
            case_days,
            room_fits,
            mandatory,
            urgency,
            penalty,
        )
    end

    return ElectiveSurgeryAssignmentProblem(
        n_surgeries,
        n_rooms,
        n_days,
        n_specs,
        specialty_names,
        wl.specialty,
        wl.duration,
        wl.duration_sd,
        wl.source_type,
        urgency,
        deadline,
        penalty,
        surgery_surgeon,
        mandatory,
        surgeon_specialty,
        surgeon_budget,
        mss,
        session,
        turnover,
        max_overtime,
        overtime_cost,
        admissible,
        open_blocks,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::ElectiveSurgeryAssignmentProblem)

Build the JuMP model for the elective surgery assignment problem.
Deterministic — uses only data from the struct fields.

# Returns

  - `model`: The JuMP model
"""
function build_model(prob::ElectiveSurgeryAssignmentProblem)
    model = Model()

    n_surgeries = prob.n_surgeries
    admissible = prob.admissible
    n_adm = length(admissible)
    n_open = length(prob.open_blocks)
    open_index = Dict(q => idx for (idx, q) in enumerate(prob.open_blocks))

    @variable(model, assign[1:n_adm], Bin)
    # Mandatory (clinically urgent) cases get no postponement column.
    optional = [i for i in 1:n_surgeries if !prob.mandatory[i]]
    @variable(model, postpone[optional], Bin)
    @variable(model, 0 <= overtime[1:n_open] <= prob.max_overtime)

    # Index the admissible triples by surgery, by block, and by (surgeon, day).
    by_surgery = [Int[] for _ in 1:n_surgeries]
    by_block = [Int[] for _ in 1:n_open]
    by_surgeon_day = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (a, (i, r, d)) in enumerate(admissible)
        push!(by_surgery[i], a)
        push!(by_block[open_index[(r, d)]], a)
        push!(get!(by_surgeon_day, (prob.surgery_surgeon[i], d), Int[]), a)
    end

    # Each surgery is either assigned to exactly one admissible block or postponed.
    for i in 1:n_surgeries
        lhs = sum(assign[a] for a in by_surgery[i]; init=AffExpr(0.0))
        prob.mandatory[i] || (lhs += postpone[i])
        @constraint(model, lhs == 1)
    end

    # Block capacity: case durations plus turnovers fit the session, allowing
    # costly overtime up to its cap.
    for q in 1:n_open
        (r, d) = prob.open_blocks[q]
        @constraint(
            model,
            sum(
                (prob.surgery_duration[admissible[a][1]] + prob.turnover) * assign[a] for
                a in by_block[q];
                init=0.0,
            ) - overtime[q] <= prob.session_length[r, d]
        )
    end

    # Surgeon-day operating-time budgets (sorted keys for deterministic builds).
    for (s, d) in sort!(collect(keys(by_surgeon_day)))
        idxs = by_surgeon_day[(s, d)]
        # Rows that cannot bind (all admissible cases fit at once) are omitted.
        sum(prob.surgery_duration[admissible[a][1]] for a in idxs) <= prob.surgeon_budget[s, d] && continue
        @constraint(
            model,
            sum(prob.surgery_duration[admissible[a][1]] * assign[a] for a in idxs) <=
                prob.surgeon_budget[s, d]
        )
    end

    @objective(
        model,
        Min,
        sum(prob.postponement_penalty[i] * postpone[i] for i in optional; init=0.0) +
            prob.overtime_cost * sum(overtime[q] for q in 1:n_open)
    )

    return model
end

# Register the variant (category default).
register_variant(
    :operating_room_scheduling,
    :elective_assignment,
    ElectiveSurgeryAssignmentProblem,
    "Elective surgery assignment to OR blocks under a master surgical schedule with surgeon availability, overtime, and urgency-weighted postponement";
    default=true,
)
