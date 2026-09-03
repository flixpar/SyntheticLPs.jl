using JuMP
using Random
using Distributions

"""
    WeeklySurgeryPlanningProblem <: ProblemGenerator

Generator for multi-day surgery planning with downstream resources — the
bed-leveled advance scheduling MILP in which surgeries are assigned to days
(rather than specific ORs) against aggregate per-specialty OR capacity, while
ward and ICU bed occupancy stays within capacity on every day (Cardoen,
Demeulemeester & Beliën, EJOR 2010 survey; the downstream-resource models of
Hans et al. and the master-surgical-scheduling-with-bed-leveling literature).

# Overview

A hospital plans its elective waiting list over a 1-2 week horizon. Each day
offers aggregate OR minutes per specialty (the sum of that day's master
surgical schedule blocks for the specialty) and each surgeon a daily
operating-time budget. A surgery is admissible to a day when its specialty has
capacity, its surgeon works, and the day is within its clinical deadline.
Scheduled cases follow a sequential ICU-to-ward path: cases needing ICU use an
ICU bed for 1-2 days and start their post-ICU ward stay only after discharge;
other admitted cases enter the ward directly. Occupancy is constrained through
discharge beyond the last surgery day. Surgeries may be
postponed at an urgency-weighted penalty, except clinically urgent cases,
which are mandatory. Scheduling earlier in the horizon is mildly preferred
(waiting-time reduction), so the objective combines postponement penalties
with small day-preference costs.

# Model

  - `assign_day[i, d] ∈ {0,1}` for each admissible (surgery, day);
  - `postpone[i] ∈ {0,1}` per surgery (fixed to 0 for mandatory/urgent cases).

Minimize postponement penalties plus day-preference costs, subject to:

  - each surgery is assigned to exactly one admissible day or postponed;
  - per-specialty-day OR capacity (durations + turnovers);
  - per-surgeon-day operating-time budgets;
  - ward bed occupancy per day ≤ effective ward capacity;
  - ICU bed occupancy per day ≤ ICU capacity.

# Feasibility control

Same pattern as the `elective_assignment` variant: for `feasible` instances a
greedy earliest-deadline schedule respecting all four capacity families is
built first (stored as `feasible_witness`, the scheduled day per surgery), then
mandatory urgent cases are designated from the scheduled set. For `infeasible`
instances one surgery
is made mandatory and its surgeon's budgets are shrunk below what its
admissible days would need — an LP-level contradiction with the mandatory
assignment row (`infeasible_surgery` records the case). For `unknown`, urgent
cases stay mandatory regardless, so feasibility is genuinely uncertain.

# Fields

  - `n_surgeries::Int`, `n_days::Int`, `n_specialties::Int`: dimensions
  - `specialty_names::Vector{Symbol}`: specialty of each position 1..n_specialties
  - `surgery_specialty::Vector{Int}`: specialty index per surgery
  - `surgery_duration::Vector{Float64}`: planned duration (minutes) per surgery
  - `surgery_urgency::Vector{Symbol}`: `:urgent` / `:semi_urgent` / `:routine`
  - `surgery_deadline::Vector{Int}`: latest admissible day per surgery
  - `postponement_penalty::Vector{Float64}`: postponement cost per surgery
  - `day_cost::Vector{Float64}`: per-day scheduling cost slope (earlier preferred)
  - `surgery_surgeon::Vector{Int}`: surgeon performing each surgery
  - `ward_los::Vector{Int}`: ward length of stay (days) per surgery (0 = day case)
  - `icu_los::Vector{Int}`: ICU length of stay (days) per surgery (0 = no ICU)
  - `mandatory::BitVector`: surgeries that must be scheduled
  - `surgeon_specialty::Vector{Int}`: specialty index per surgeon
  - `surgeon_budget::Matrix{Float64}`: operating minutes of surgeon s on day d
  - `specialty_capacity::Matrix{Float64}`: aggregate OR minutes per (specialty, day)
  - `turnover::Float64`: OR turnover time per case (minutes)
  - `ward_capacity::Vector{Float64}`: effective ward beds available per day
  - `icu_capacity::Vector{Float64}`: ICU beds available per day
  - `admissible_days::Vector{Vector{Int}}`: admissible days per surgery
  - `feasible_witness::Union{Nothing,Vector{Int}}`: scheduled day per surgery
    (0 = postponed; only for `feasible` instances)
  - `infeasible_surgery::Union{Nothing,Int}`: mandatory surgery of the
    infeasibility certificate (only for `infeasible` instances)
  - `feasibility_status::FeasibilityStatus`: resolved feasibility status
"""
struct WeeklySurgeryPlanningProblem <: ProblemGenerator
    n_surgeries::Int
    n_days::Int
    bed_horizon::Int
    n_specialties::Int
    specialty_names::Vector{Symbol}
    surgery_specialty::Vector{Int}
    surgery_duration::Vector{Float64}
    surgery_urgency::Vector{Symbol}
    surgery_deadline::Vector{Int}
    postponement_penalty::Vector{Float64}
    day_cost::Vector{Float64}
    surgery_surgeon::Vector{Int}
    ward_los::Vector{Int}
    icu_los::Vector{Int}
    mandatory::BitVector
    surgeon_specialty::Vector{Int}
    surgeon_budget::Matrix{Float64}
    specialty_capacity::Matrix{Float64}
    turnover::Float64
    ward_capacity::Vector{Float64}
    icu_capacity::Vector{Float64}
    admissible_days::Vector{Vector{Int}}
    feasible_witness::Union{Nothing, Vector{Int}}
    infeasible_surgery::Union{Nothing, Int}
    feasibility_status::FeasibilityStatus
end

# Admissible days: specialty has aggregate OR capacity, the surgeon works, and
# the day is within the clinical deadline.
function _weekly_admissible_days(
    n_surgeries::Int,
    surgery_specialty::Vector{Int},
    surgery_deadline::Vector{Int},
    surgery_surgeon::Vector{Int},
    surgeon_budget::Matrix{Float64},
    specialty_capacity::Matrix{Float64},
    n_days::Int,
)
    days = [Int[] for _ in 1:n_surgeries]
    for i in 1:n_surgeries, d in 1:min(n_days, surgery_deadline[i])
        if specialty_capacity[surgery_specialty[i], d] > 0 &&
            surgeon_budget[surgery_surgeon[i], d] > 0
            push!(days[i], d)
        end
    end
    return days
end

"""
    WeeklySurgeryPlanningProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-day surgery planning instance near `target_variables`
decision variables (`sum |admissible_days| + n_surgeries`). The constructor
iterates over waiting-list sizes, computing the exact variable count of each
sampled instance, and keeps the closest one.

# Arguments

  - `target_variables`: Target number of decision variables
  - `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
  - `seed`: Random seed for reproducibility
"""
function WeeklySurgeryPlanningProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)

    target = max(target_variables, 20)
    n_rooms, n_days, n_specs = _orsched_hospital_scale(rng, target)
    turnover = 5.0 * round(rand(rng, Uniform(15.0, 35.0)) / 5.0)

    best = nothing
    best_gap = Inf
    n_surgeries = max(4, round(Int, target / 5.0))

    for _ in 1:60
        spec_ids = _orsched_case_mix(rng, n_specs)
        mss, session = _orsched_master_schedule(rng, n_rooms, n_days, spec_ids)
        wl = _orsched_waiting_list(
            rng,
            n_surgeries,
            spec_ids,
            n_days;
            with_los=true,
            allow_urgent=feasibility_status != feasible,
        )
        counts = [count(==(k), wl.specialty) for k in 1:n_specs]
        surgeon_specialty, surgeon_budget = _orsched_surgeon_pool(rng, counts, n_days, mss)
        surgery_surgeon = _elective_assign_surgeons(rng, wl.specialty, surgeon_specialty)

        # Aggregate OR minutes per (specialty, day) from the master schedule.
        specialty_capacity = zeros(Float64, n_specs, n_days)
        for d in 1:n_days, r in 1:n_rooms
            if session[r, d] > 0
                specialty_capacity[mss[r, d], d] += session[r, d]
            end
        end

        admissible_days = _weekly_admissible_days(
            n_surgeries,
            wl.specialty,
            wl.deadline,
            surgery_surgeon,
            surgeon_budget,
            specialty_capacity,
            n_days,
        )
        total = sum(length, admissible_days) + n_surgeries
        gap = abs(total - target) / target
        if gap < best_gap
            best_gap = gap
            best = (
                spec_ids=spec_ids,
                wl=wl,
                surgeon_specialty=surgeon_specialty,
                surgeon_budget=surgeon_budget,
                surgery_surgeon=surgery_surgeon,
                specialty_capacity=specialty_capacity,
                admissible_days=admissible_days,
                n_surgeries=n_surgeries,
            )
            gap <= 0.05 && break
        end
        scale = target / max(total, 1)
        new_n = clamp(round(Int, n_surgeries * scale), 4, 4 * n_surgeries)
        new_n == n_surgeries && (new_n = total < target ? n_surgeries + 1 : max(4, n_surgeries - 1))
        n_surgeries = new_n
    end

    spec_ids = best.spec_ids
    wl = best.wl
    surgeon_specialty, surgeon_budget = best.surgeon_specialty, best.surgeon_budget
    surgery_surgeon = best.surgery_surgeon
    specialty_capacity = best.specialty_capacity
    admissible_days = best.admissible_days
    n_surgeries = best.n_surgeries

    specialty_names = [_ORSCHED_SPECIALTIES[k].name for k in spec_ids]
    urgency = collect(wl.urgency)
    deadline = collect(wl.deadline)
    penalty = collect(wl.penalty)
    day_cost = [0.02 * penalty[i] for i in 1:n_surgeries]

    # Keep post-operative occupancy through discharge even for cases placed on
    # the last surgery day.  Ward occupancy starts after ICU discharge; the two
    # resources are never consumed simultaneously by one patient.
    max_path = maximum(wl.icu_los .+ wl.ward_los)
    bed_horizon = n_days + max(0, max_path - 1)

    # Effective bed capacities: supply slightly above average daily demand,
    # with day-to-day fluctuation from background occupancy.
    ward_demand = sum(wl.ward_los) / n_days
    ward_capacity = [
        max(
            1.0,
            round(ward_demand * rand(rng, Uniform(1.05, 1.35)) * rand(rng, Uniform(0.85, 1.15))),
        ) for _ in 1:bed_horizon
    ]
    icu_demand = sum(wl.icu_los) / n_days
    icu_capacity = if any(>(0), wl.icu_los)
        [
            max(
                1.0,
                round(icu_demand * rand(rng, Uniform(1.1, 1.5)) * rand(rng, Uniform(0.85, 1.15))),
            ) for _ in 1:bed_horizon
        ]
    else
        zeros(Float64, bed_horizon)
    end

    witness = nothing
    infeasible_surgery = nothing
    mandatory = BitVector(urgency[i] == :urgent for i in 1:n_surgeries)

    if feasibility_status == feasible
        # Greedy earliest-deadline schedule respecting specialty-day OR
        # capacity, surgeon budgets, and day-by-day ward/ICU bed occupancy.
        rem_spec = copy(specialty_capacity)
        rem_surg = copy(surgeon_budget)
        rem_ward = copy(ward_capacity)
        rem_icu = copy(icu_capacity)
        function consume!(day::Int, i::Int)
            need_or = wl.duration[i] + turnover
            need_surg = wl.duration[i]
            icu_days, ward_days = _orsched_postop_days(
                day, wl.icu_los[i], wl.ward_los[i], bed_horizon
            )
            if rem_spec[wl.specialty[i], day] >= need_or &&
                rem_surg[surgery_surgeon[i], day] >= need_surg &&
                all(rem_ward[t] >= 1 for t in ward_days) &&
                all(rem_icu[t] >= 1 for t in icu_days)
                rem_spec[wl.specialty[i], day] -= need_or
                rem_surg[surgery_surgeon[i], day] -= need_surg
                for t in ward_days
                    rem_ward[t] -= 1
                end
                for t in icu_days
                    rem_icu[t] -= 1
                end
                return true
            end
            return false
        end
        assignment = _orsched_greedy_schedule(
            n_surgeries, urgency, deadline, wl.duration, admissible_days, n_days, consume!
        )
        mandatory = _orsched_designate_mandatory!(
            rng, urgency, deadline, penalty, assignment, wl.requested_urgent_fraction
        )
        day_cost .= 0.02 .* penalty
        witness = assignment
    elseif feasibility_status == infeasible
        candidates = [i for i in 1:n_surgeries if !isempty(admissible_days[i])]
        if isempty(candidates)
            deadline[1] = n_days
            days_with_capacity = [d for d in 1:n_days if specialty_capacity[wl.specialty[1], d] > 0]
            surgeon_budget[surgery_surgeon[1], first(days_with_capacity)] = wl.duration[1]
            admissible_days = _weekly_admissible_days(
                n_surgeries,
                wl.specialty,
                deadline,
                surgery_surgeon,
                surgeon_budget,
                specialty_capacity,
                n_days,
            )
            candidates = [i for i in 1:n_surgeries if !isempty(admissible_days[i])]
        end
        victim = candidates[argmax([wl.duration[i] for i in candidates])]
        surgeon = surgery_surgeon[victim]
        _orsched_inject_surgeon_shortage!(
            surgeon_budget, surgeon, wl.duration[victim], admissible_days[victim]
        )
        urgency[victim] = :urgent
        mandatory[victim] = true
        infeasible_surgery = victim
        admissible_days = _weekly_admissible_days(
            n_surgeries,
            wl.specialty,
            deadline,
            surgery_surgeon,
            surgeon_budget,
            specialty_capacity,
            n_days,
        )
    end

    return WeeklySurgeryPlanningProblem(
        n_surgeries,
        n_days,
        bed_horizon,
        n_specs,
        specialty_names,
        wl.specialty,
        wl.duration,
        urgency,
        deadline,
        penalty,
        day_cost,
        surgery_surgeon,
        wl.ward_los,
        wl.icu_los,
        mandatory,
        surgeon_specialty,
        surgeon_budget,
        specialty_capacity,
        turnover,
        ward_capacity,
        icu_capacity,
        admissible_days,
        witness,
        infeasible_surgery,
        feasibility_status,
    )
end

"""
    build_model(prob::WeeklySurgeryPlanningProblem)

Build the JuMP model for the multi-day surgery planning problem with
downstream beds. Deterministic — uses only data from the struct fields.

# Returns

  - `model`: The JuMP model
"""
function build_model(prob::WeeklySurgeryPlanningProblem)
    model = Model()

    n_surgeries = prob.n_surgeries
    n_days = prob.n_days
    admissible_days = prob.admissible_days

    @variable(model, assign_day[i in 1:n_surgeries, d in admissible_days[i]], Bin)
    @variable(model, postpone[1:n_surgeries], Bin)

    # Each surgery is either assigned to exactly one admissible day or postponed.
    for i in 1:n_surgeries
        @constraint(
            model, sum(assign_day[i, d] for d in admissible_days[i]; init=0.0) + postpone[i] == 1
        )
    end

    # Mandatory (clinically urgent) cases may not be postponed.
    for i in 1:n_surgeries
        prob.mandatory[i] && @constraint(model, postpone[i] == 0)
    end

    # Aggregate OR capacity per specialty-day (durations plus turnovers).
    for k in 1:prob.n_specialties, d in 1:n_days
        cases = [
            i for i in 1:n_surgeries if prob.surgery_specialty[i] == k && d in admissible_days[i]
        ]
        isempty(cases) && continue
        @constraint(
            model,
            sum((prob.surgery_duration[i] + prob.turnover) * assign_day[i, d] for i in cases) <=
                prob.specialty_capacity[k, d]
        )
    end

    # Surgeon-day operating-time budgets.
    for s in eachindex(prob.surgeon_specialty), d in 1:n_days
        cases = [
            i for i in 1:n_surgeries if prob.surgery_surgeon[i] == s && d in admissible_days[i]
        ]
        isempty(cases) && continue
        @constraint(
            model,
            sum(prob.surgery_duration[i] * assign_day[i, d] for i in cases) <=
                prob.surgeon_budget[s, d]
        )
    end

    # Sequential ICU -> ward patient path.  The extended bed horizon prevents
    # last-day cases from escaping downstream capacity constraints.
    for t in 1:prob.bed_horizon
        terms = Tuple{Int, Int}[]
        for i in 1:n_surgeries
            prob.ward_los[i] > 0 || continue
            for d in admissible_days[i]
                ward_start = d + prob.icu_los[i]
                if ward_start <= t <= ward_start + prob.ward_los[i] - 1
                    push!(terms, (i, d))
                end
            end
        end
        isempty(terms) && continue
        @constraint(model, sum(assign_day[i, d] for (i, d) in terms) <= prob.ward_capacity[t])
    end

    # ICU bed occupancy per day.
    for t in 1:prob.bed_horizon
        terms = Tuple{Int, Int}[]
        for i in 1:n_surgeries
            prob.icu_los[i] > 0 || continue
            for d in admissible_days[i]
                if d <= t <= d + prob.icu_los[i] - 1
                    push!(terms, (i, d))
                end
            end
        end
        isempty(terms) && continue
        @constraint(model, sum(assign_day[i, d] for (i, d) in terms) <= prob.icu_capacity[t])
    end

    @objective(
        model,
        Min,
        sum(prob.postponement_penalty[i] * postpone[i] for i in 1:n_surgeries) + sum(
            prob.day_cost[i] * (d - 1) * assign_day[i, d] for
            i in 1:n_surgeries, d in admissible_days[i]
        )
    )

    return model
end

# Register the variant
register_variant(
    :operating_room_scheduling,
    :weekly_planning,
    WeeklySurgeryPlanningProblem,
    "Multi-day surgery planning with aggregate specialty OR capacity, surgeon budgets, and downstream ward/ICU bed leveling",
)
