using JuMP
using Random
using Distributions
using StatsBase

"""
    OperatingRoomSchedulingProblem <: ProblemGenerator

Generator for advance elective surgical case scheduling problems across operating rooms
and days with surgeon availability, specialized equipment, regular/overtime capacity,
PACU recovery beds, and patient urgency weighting.

# Overview
Models multi-day advance scheduling of elective surgical procedures in a hospital surgical suite.
The decisions assign surgical cases to specific operating rooms (ORs) and days in the planning
horizon, decide which ORs to open, track regular time, overtime, and undertime (idle time),
and manage potential case postponements.

Constraints enforce:
- Each elective surgery is scheduled at most once across the horizon (or postponed).
- Room-specialty eligibility: procedures requiring specialized equipment (e.g., laminar airflow,
  robotic systems, hybrid imaging) can only be performed in equipped ORs.
- Operating room capacity: daily procedure times plus turnovers balance against regular open hours,
  overtime (capped at maximum allowable overtime), and undertime.
- Surgeon daily availability and operating time limits.
- Downstream Post-Anesthesia Care Unit (PACU) recovery bed-hour capacity per day.
- Maximum open operating rooms per day (anesthesia / nursing staffing limits).

# Fields
- `n_surgeries::Int`: Total number of elective surgeries
- `n_rooms::Int`: Number of operating rooms in the suite
- `n_days::Int`: Number of days in planning horizon (e.g. 5 days)
- `n_specialties::Int`: Number of surgical specialties
- `n_surgeons::Int`: Number of surgeons
- `surgery_specialty::Vector{Int}`: Specialty index for each surgery
- `surgery_surgeon::Vector{Int}`: Surgeon assigned to each surgery
- `duration::Vector{Float64}`: Expected procedure duration (minutes)
- `cleaning_time::Vector{Float64}`: Turnover/cleaning time between cases (minutes)
- `pacu_stay::Vector{Float64}`: Expected PACU recovery duration (hours)
- `urgency_weight::Vector{Float64}`: Priority/urgency waiting penalty weight
- `postpone_cost::Vector{Float64}`: Penalty cost if case is deferred beyond horizon
- `room_specialty_matrix::Matrix{Int}`: Binary compatibility matrix (1 if room r can host specialty s)
- `regular_capacity::Matrix{Float64}`: Regular open hours (minutes) for room r on day d
- `max_overtime::Matrix{Float64}`: Maximum allowed overtime (minutes) for room r on day d
- `fixed_open_cost::Matrix{Float64}`: Fixed cost to open room r on day d
- `overtime_cost::Matrix{Float64}`: Cost per minute of overtime for room r on day d
- `undertime_cost::Matrix{Float64}`: Cost per minute of undertime for room r on day d
- `surgeon_availability::Matrix{Int}`: 1 if surgeon j is available on day d, 0 otherwise
- `surgeon_max_hours::Matrix{Float64}`: Max operating minutes for surgeon j on day d
- `pacu_capacity::Vector{Float64}`: Daily PACU bed-hour capacity on day d
- `max_open_rooms_per_day::Vector{Int}`: Max rooms that can be staffed on day d
- `force_all_scheduled::Bool`: If true, no postponements allowed (postpone[i] == 0)
- `feasibility_status::FeasibilityStatus`: Resolved feasibility status
"""
struct OperatingRoomSchedulingProblem <: ProblemGenerator
    n_surgeries::Int
    n_rooms::Int
    n_days::Int
    n_specialties::Int
    n_surgeons::Int
    surgery_specialty::Vector{Int}
    surgery_surgeon::Vector{Int}
    duration::Vector{Float64}
    cleaning_time::Vector{Float64}
    pacu_stay::Vector{Float64}
    urgency_weight::Vector{Float64}
    postpone_cost::Vector{Float64}
    room_specialty_matrix::Matrix{Int}
    regular_capacity::Matrix{Float64}
    max_overtime::Matrix{Float64}
    fixed_open_cost::Matrix{Float64}
    overtime_cost::Matrix{Float64}
    undertime_cost::Matrix{Float64}
    surgeon_availability::Matrix{Int}
    surgeon_max_hours::Matrix{Float64}
    pacu_capacity::Vector{Float64}
    max_open_rooms_per_day::Vector{Int}
    force_all_scheduled::Bool
    feasibility_status::FeasibilityStatus
end

# Specialty reference profiles based on hospital empirical data
const OR_SPECIALTY_PROFILES = [
    (name="General", mean_dur=110.0, logn_sigma=0.35, clean_min=20.0, pacu_hours=2.0, room_prob=1.0),
    (name="Orthopedics", mean_dur=130.0, logn_sigma=0.30, clean_min=25.0, pacu_hours=2.5, room_prob=0.75),
    (name="Neurosurgery", mean_dur=200.0, logn_sigma=0.40, clean_min=30.0, pacu_hours=3.5, room_prob=0.45),
    (name="Cardiovascular", mean_dur=220.0, logn_sigma=0.35, clean_min=30.0, pacu_hours=4.0, room_prob=0.40),
    (name="Ophthalmology", mean_dur=45.0, logn_sigma=0.25, clean_min=15.0, pacu_hours=1.0, room_prob=0.60),
    (name="Urology", mean_dur=85.0, logn_sigma=0.30, clean_min=20.0, pacu_hours=1.8, room_prob=0.70),
    (name="ENT", mean_dur=75.0, logn_sigma=0.30, clean_min=20.0, pacu_hours=1.5, room_prob=0.65),
    (name="Plastic", mean_dur=140.0, logn_sigma=0.35, clean_min=20.0, pacu_hours=2.2, room_prob=0.55),
]

"""
    OperatingRoomSchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an advance elective case scheduling problem instance targeting `target_variables`.

# Variable-count formula
`build_model` defines:
- `x[1:n_surgeries, 1:n_rooms, 1:n_days]` (Bin): `n_surgeries * n_rooms * n_days`
- `y[1:n_rooms, 1:n_days]` (Bin): `n_rooms * n_days`
- `overtime[1:n_rooms, 1:n_days]` (Cont >= 0): `n_rooms * n_days`
- `undertime[1:n_rooms, 1:n_days]` (Cont >= 0): `n_rooms * n_days`
- `postpone[1:n_surgeries]` (Bin): `n_surgeries`
Total variables = `(n_surgeries + 3) * n_rooms * n_days + n_surgeries`.
"""
function OperatingRoomSchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    target = max(20, target_variables)

    # Scale dimensions based on target
    if target <= 120
        n_days = 4
        n_rooms = 2
        n_specialties = 3
        n_surgeons = 4
    elseif target <= 500
        n_days = 5
        n_rooms = 4
        n_specialties = 4
        n_surgeons = 8
    elseif target <= 2000
        n_days = 5
        n_rooms = 7
        n_specialties = 6
        n_surgeons = 14
    else
        n_days = min(10, 5 + round(Int, log2(target / 2000)))
        n_rooms = clamp(round(Int, sqrt(target / (n_days * 3))), 6, 20)
        n_specialties = min(8, max(4, round(Int, n_rooms * 0.7)))
        n_surgeons = max(n_rooms * 2, round(Int, n_rooms * 2.5))
    end

    rd = n_rooms * n_days
    # Solve (N + 3) * rd + N ≈ target => N * (rd + 1) + 3*rd ≈ target => N = round((target - 3*rd) / (rd + 1))
    n_surgeries = max(2, round(Int, (target - 3 * rd) / (rd + 1)))

    actual_status = feasibility_status

    # Specialties setup
    profiles = OR_SPECIALTY_PROFILES[1:n_specialties]

    # Room-specialty compatibility matrix
    room_specialty_matrix = zeros(Int, n_rooms, n_specialties)
    for s in 1:n_specialties
        prob = profiles[s].room_prob
        for r in 1:n_rooms
            room_specialty_matrix[r, s] = (rand() < prob || s == 1) ? 1 : 0
        end
        # Guarantee every specialty can be hosted in at least one room
        if sum(room_specialty_matrix[:, s]) == 0
            room_specialty_matrix[rand(1:n_rooms), s] = 1
        end
    end
    # Guarantee every room can host at least one specialty
    for r in 1:n_rooms
        if sum(room_specialty_matrix[r, :]) == 0
            room_specialty_matrix[r, 1] = 1
        end
    end

    # Assign each surgeon a primary specialty
    surgeon_specialty = Vector{Int}(undef, n_surgeons)
    for j in 1:n_surgeons
        surgeon_specialty[j] = mod1(j, n_specialties)
    end

    # Generate surgeries
    surgery_specialty = Vector{Int}(undef, n_surgeries)
    surgery_surgeon = Vector{Int}(undef, n_surgeries)
    duration = Vector{Float64}(undef, n_surgeries)
    cleaning_time = Vector{Float64}(undef, n_surgeries)
    pacu_stay = Vector{Float64}(undef, n_surgeries)
    urgency_weight = Vector{Float64}(undef, n_surgeries)
    postpone_cost = Vector{Float64}(undef, n_surgeries)

    for i in 1:n_surgeries
        s = rand(1:n_specialties)
        surgery_specialty[i] = s

        # Pick a surgeon with matching specialty
        eligible_surgeons = findall(==(s), surgeon_specialty)
        surgery_surgeon[i] = isempty(eligible_surgeons) ? rand(1:n_surgeons) : rand(eligible_surgeons)

        # Sample duration from specialty lognormal distribution
        prof = profiles[s]
        # Lognormal parameters so that mean ≈ prof.mean_dur
        sigma = prof.logn_sigma
        mu = log(prof.mean_dur) - 0.5 * sigma^2
        duration[i] = clamp(exp(rand(Normal(mu, sigma))), 20.0, 360.0)
        cleaning_time[i] = prof.clean_min * rand(Uniform(0.85, 1.25))
        pacu_stay[i] = prof.pacu_hours * rand(Uniform(0.8, 1.3))

        # Urgency weight: 1.0 (routine elective) to 10.0 (high priority)
        urgency_weight[i] = rand(Uniform(1.0, 10.0))
        # Postponement cost scales with urgency
        postpone_cost[i] = 1500.0 + urgency_weight[i] * rand(Uniform(300.0, 800.0))
    end

    # Room capacities and costs
    regular_capacity = fill(480.0, n_rooms, n_days) # 8 hours standard
    max_overtime = fill(120.0, n_rooms, n_days)      # Up to 2 hours overtime

    fixed_open_cost = Matrix{Float64}(undef, n_rooms, n_days)
    overtime_cost = Matrix{Float64}(undef, n_rooms, n_days)
    undertime_cost = Matrix{Float64}(undef, n_rooms, n_days)

    for r in 1:n_rooms, d in 1:n_days
        fixed_open_cost[r, d] = rand(Uniform(800.0, 1800.0))
        overtime_cost[r, d] = rand(Uniform(25.0, 50.0))     # per minute ($1500-$3000/hr)
        undertime_cost[r, d] = rand(Uniform(5.0, 15.0))      # per minute idle penalty
    end

    # Surgeon availability and hours
    surgeon_availability = zeros(Int, n_surgeons, n_days)
    surgeon_max_hours = Matrix{Float64}(undef, n_surgeons, n_days)
    for j in 1:n_surgeons
        for d in 1:n_days
            avail = rand() < 0.75 ? 1 : 0
            surgeon_availability[j, d] = avail
            surgeon_max_hours[j, d] = 480.0
        end
        # Guarantee surgeon is available on at least one day
        if sum(surgeon_availability[j, :]) == 0
            surgeon_availability[j, rand(1:n_days)] = 1
        end
    end

    # Daily PACU capacity
    total_pacu_hours_expected = sum(pacu_stay)
    avg_pacu_per_day = total_pacu_hours_expected / n_days
    pacu_capacity = [avg_pacu_per_day * rand(Uniform(1.1, 1.6)) + 5.0 for _ in 1:n_days]

    # Staffed room limit per day
    max_open_rooms_per_day = fill(n_rooms, n_days)

    force_all_scheduled = false

    # Feasibility handling
    if actual_status == feasible
        force_all_scheduled = true

        # Build a valid witness schedule by greedily packing surgeries into compatible rooms/days
        room_day_load = zeros(Float64, n_rooms, n_days)
        surgeon_day_load = zeros(Float64, n_surgeons, n_days)
        pacu_day_load = zeros(Float64, n_days)

        for i in 1:n_surgeries
            s = surgery_specialty[i]
            j = surgery_surgeon[i]
            dur = duration[i]
            clean = cleaning_time[i]
            pacu = pacu_stay[i]

            # Find candidate (r, d)
            best_rd = nothing
            best_load = Inf

            for d in 1:n_days, r in 1:n_rooms
                room_specialty_matrix[r, s] == 1 || continue
                load = room_day_load[r, d]
                if load < best_load
                    best_load = load
                    best_rd = (r, d)
                end
            end

            if best_rd === nothing
                # Make room 1 compatible with s as fallback
                room_specialty_matrix[1, s] = 1
                best_rd = (1, 1)
            end

            r_chosen, d_chosen = best_rd
            room_day_load[r_chosen, d_chosen] += dur + clean
            surgeon_day_load[j, d_chosen] += dur
            pacu_day_load[d_chosen] += pacu
            surgeon_availability[j, d_chosen] = 1
        end

        # Adjust capacities with generous slack above the witness loads
        for r in 1:n_rooms, d in 1:n_days
            needed = room_day_load[r, d]
            if needed > 0
                regular_capacity[r, d] = max(480.0, ceil(needed * 0.85))
                max_overtime[r, d] = max(120.0, ceil(needed * 0.40))
            end
        end

        for j in 1:n_surgeons, d in 1:n_days
            needed = surgeon_day_load[j, d]
            if needed > 0
                surgeon_max_hours[j, d] = max(480.0, ceil(needed * 1.30))
            end
        end

        for d in 1:n_days
            needed = pacu_day_load[d]
            pacu_capacity[d] = max(pacu_capacity[d], ceil(needed * 1.35) + 5.0)
        end

    elseif actual_status == infeasible
        # Deterministic contradiction:
        # Require all surgeries to be scheduled, pick specialty 1, and make total required
        # operating duration of specialty 1 strictly greater than total available capacity
        # across ALL compatible rooms over the entire horizon combined.
        force_all_scheduled = true
        target_spec = 1

        # Compatible rooms for target_spec
        compat_rooms = [r for r in 1:n_rooms if room_specialty_matrix[r, target_spec] == 1]
        if isempty(compat_rooms)
            room_specialty_matrix[1, target_spec] = 1
            compat_rooms = [1]
        end

        total_horizon_room_capacity = sum(regular_capacity[r, d] + max_overtime[r, d]
                                          for r in compat_rooms, d in 1:n_days)

        # Force all surgeries to target_spec and inflate duration so sum > total capacity
        for i in 1:n_surgeries
            surgery_specialty[i] = target_spec
        end
        boost_per_surgery = (total_horizon_room_capacity * 1.5) / n_surgeries
        for i in 1:n_surgeries
            duration[i] = max(duration[i], boost_per_surgery)
        end
    end

    return OperatingRoomSchedulingProblem(
        n_surgeries,
        n_rooms,
        n_days,
        n_specialties,
        n_surgeons,
        surgery_specialty,
        surgery_surgeon,
        duration,
        cleaning_time,
        pacu_stay,
        urgency_weight,
        postpone_cost,
        room_specialty_matrix,
        regular_capacity,
        max_overtime,
        fixed_open_cost,
        overtime_cost,
        undertime_cost,
        surgeon_availability,
        surgeon_max_hours,
        pacu_capacity,
        max_open_rooms_per_day,
        force_all_scheduled,
        actual_status,
    )
end

"""
    build_model(prob::OperatingRoomSchedulingProblem)

Build a JuMP model for the advance elective operating room scheduling problem.
Deterministic implementation.
"""
function build_model(prob::OperatingRoomSchedulingProblem)
    model = Model()

    N = prob.n_surgeries
    R = prob.n_rooms
    D = prob.n_days
    J = prob.n_surgeons

    # Decision variables
    @variable(model, x[1:N, 1:R, 1:D], Bin)
    @variable(model, y[1:R, 1:D], Bin)
    @variable(model, overtime[1:R, 1:D] >= 0)
    @variable(model, undertime[1:R, 1:D] >= 0)
    @variable(model, postpone[1:N], Bin)

    # Objective: Minimize fixed opening costs + overtime penalties + undertime penalties + postponement + waiting time
    @objective(
        model,
        Min,
        sum(prob.fixed_open_cost[r, d] * y[r, d] for r in 1:R, d in 1:D) +
        sum(prob.overtime_cost[r, d] * overtime[r, d] for r in 1:R, d in 1:D) +
        sum(prob.undertime_cost[r, d] * undertime[r, d] for r in 1:R, d in 1:D) +
        sum(prob.postpone_cost[i] * postpone[i] for i in 1:N) +
        sum(prob.urgency_weight[i] * d * x[i, r, d] for i in 1:N, r in 1:R, d in 1:D)
    )

    # 1. Surgery assignment: scheduled at most once across the horizon or postponed
    for i in 1:N
        @constraint(model, sum(x[i, r, d] for r in 1:R, d in 1:D) + postpone[i] == 1)
        if prob.force_all_scheduled
            @constraint(model, postpone[i] == 0)
        end
    end

    # 2. Room-specialty eligibility
    for i in 1:N, r in 1:R, d in 1:D
        s = prob.surgery_specialty[i]
        if prob.room_specialty_matrix[r, s] == 0
            @constraint(model, x[i, r, d] == 0)
        end
    end

    # 3. Operating room daily duration balance and overtime/undertime linking
    for r in 1:R, d in 1:D
        total_time_expr = sum((prob.duration[i] + prob.cleaning_time[i]) * x[i, r, d] for i in 1:N)
        @constraint(model, total_time_expr == prob.regular_capacity[r, d] * y[r, d] + overtime[r, d] - undertime[r, d])
        @constraint(model, overtime[r, d] <= prob.max_overtime[r, d] * y[r, d])
        @constraint(model, undertime[r, d] <= prob.regular_capacity[r, d] * y[r, d])
    end

    # 4. Surgeon daily availability and operating hours
    for j in 1:J, d in 1:D
        surgeries_j = findall(==(j), prob.surgery_surgeon)
        if !isempty(surgeries_j)
            surg_time_expr = sum(prob.duration[i] * sum(x[i, r, d] for r in 1:R) for i in surgeries_j)
            max_surg = prob.surgeon_max_hours[j, d] * prob.surgeon_availability[j, d]
            @constraint(model, surg_time_expr <= max_surg)
        end
    end

    # 5. Downstream PACU recovery bed capacity
    for d in 1:D
        @constraint(model, sum(prob.pacu_stay[i] * sum(x[i, r, d] for r in 1:R) for i in 1:N) <= prob.pacu_capacity[d])
    end

    # 6. Staffed operating rooms ceiling
    for d in 1:D
        @constraint(model, sum(y[r, d] for r in 1:R) <= prob.max_open_rooms_per_day[d])
    end

    return model
end

# Register the standard variant
register_variant(
    :operating_room_scheduling,
    :standard,
    OperatingRoomSchedulingProblem,
    "Advance elective surgical case scheduling across operating rooms and days with surgeon availability, specialized equipment, regular/overtime capacity, and PACU limits",
    default=true,
)
