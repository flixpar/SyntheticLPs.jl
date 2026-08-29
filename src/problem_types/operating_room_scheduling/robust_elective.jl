using JuMP
using Random
using Distributions

"""
    OperatingRoomRobustSchedulingProblem <: ProblemGenerator

Generator for robust advance elective surgery scheduling problems under duration
uncertainty using a budget-of-uncertainty formulation.

# Overview
Models advance scheduling of elective surgeries where procedure durations are subject
to significant uncertainty. Protects operating room schedules against worst-case duration
overruns via Bertsimas-Simchi-Levi robust optimization with uncertainty budget parameters
`Gamma[r, d]`, which bound the number of surgeries in room `r` on day `d` that can
simultaneously achieve their peak duration deviation without causing unscheduled overtime.

The robust capacity constraint:
    sum((p_bar_i + clean_i) * x_{i,r,d}) + max_{|U| <= Gamma} sum_{i in U} p_hat_i * x_{i,r,d} <= Cap * y_{r,d} + OT_{r,d}
is reformulated as a linear program using strong duality via auxiliary variables `theta_{r,d}`
and `mu_{i,r,d}`:
    sum((p_bar_i + clean_i) * x_{i,r,d}) + Gamma_{r,d} * theta_{r,d} + sum(mu_{i,r,d}) <= Cap * y_{r,d} + OT_{r,d}
    theta_{r,d} + mu_{i,r,d} >= p_hat_i * x_{i,r,d}

# Fields
- `n_surgeries::Int`: Number of elective surgeries
- `n_rooms::Int`: Number of operating rooms
- `n_days::Int`: Number of days in planning horizon
- `n_specialties::Int`: Number of surgical specialties
- `n_surgeons::Int`: Number of surgeons
- `surgery_specialty::Vector{Int}`: Specialty index of each surgery
- `surgery_surgeon::Vector{Int}`: Assigned surgeon for each surgery
- `nominal_duration::Vector{Float64}`: Nominal procedure duration (minutes)
- `duration_deviation::Vector{Float64}`: Maximum duration deviation (minutes)
- `cleaning_time::Vector{Float64}`: Turnover/cleaning time (minutes)
- `urgency_weight::Vector{Float64}`: Patient waiting penalty weight
- `postpone_cost::Vector{Float64}`: Cost to postpone surgery beyond horizon
- `uncertainty_budget::Matrix{Float64}`: Gamma budget parameter for room r on day d
- `room_specialty_matrix::Matrix{Int}`: Compatibility matrix (R x S)
- `regular_capacity::Matrix{Float64}`: Regular hours (minutes) for room r on day d
- `max_overtime::Matrix{Float64}`: Maximum overtime (minutes) for room r on day d
- `fixed_open_cost::Matrix{Float64}`: Fixed cost to open room r on day d
- `overtime_cost::Matrix{Float64}`: Cost per minute of overtime
- `surgeon_availability::Matrix{Int}`: Surgeon availability (J x D)
- `surgeon_max_hours::Matrix{Float64}`: Max operating minutes for surgeon j on day d
- `force_all_scheduled::Bool`: If true, postponement is disallowed (postpone[i] == 0)
- `feasibility_status::FeasibilityStatus`: Resolved feasibility status
"""
struct OperatingRoomRobustSchedulingProblem <: ProblemGenerator
    n_surgeries::Int
    n_rooms::Int
    n_days::Int
    n_specialties::Int
    n_surgeons::Int
    surgery_specialty::Vector{Int}
    surgery_surgeon::Vector{Int}
    nominal_duration::Vector{Float64}
    duration_deviation::Vector{Float64}
    cleaning_time::Vector{Float64}
    urgency_weight::Vector{Float64}
    postpone_cost::Vector{Float64}
    uncertainty_budget::Matrix{Float64}
    room_specialty_matrix::Matrix{Int}
    regular_capacity::Matrix{Float64}
    max_overtime::Matrix{Float64}
    fixed_open_cost::Matrix{Float64}
    overtime_cost::Matrix{Float64}
    surgeon_availability::Matrix{Int}
    surgeon_max_hours::Matrix{Float64}
    force_all_scheduled::Bool
    feasibility_status::FeasibilityStatus
end

"""
    OperatingRoomRobustSchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a robust elective case scheduling instance targeting `target_variables`.

# Variable-count formula
- `x[1:n_surgeries, 1:n_rooms, 1:n_days]`: `n_surgeries * n_rooms * n_days` (Bin)
- `mu[1:n_surgeries, 1:n_rooms, 1:n_days]`: `n_surgeries * n_rooms * n_days` (Cont >= 0)
- `theta[1:n_rooms, 1:n_days]`: `n_rooms * n_days` (Cont >= 0)
- `y[1:n_rooms, 1:n_days]`: `n_rooms * n_days` (Bin)
- `overtime[1:n_rooms, 1:n_days]`: `n_rooms * n_days` (Cont >= 0)
- `postpone[1:n_surgeries]`: `n_surgeries` (Bin)
Total variables = `2 * n_surgeries * n_rooms * n_days + 3 * n_rooms * n_days + n_surgeries`.
"""
function OperatingRoomRobustSchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    target = max(20, target_variables)

    # Scale dimensions based on target
    if target <= 120
        n_days = 3
        n_rooms = 2
        n_specialties = 3
        n_surgeons = 4
    elseif target <= 600
        n_days = 4
        n_rooms = 3
        n_specialties = 4
        n_surgeons = 8
    elseif target <= 2500
        n_days = 5
        n_rooms = 5
        n_specialties = 5
        n_surgeons = 12
    else
        n_days = min(8, 4 + round(Int, log2(target / 2000)))
        n_rooms = clamp(round(Int, sqrt(target / (n_days * 4))), 5, 16)
        n_specialties = min(8, max(4, round(Int, n_rooms * 0.7)))
        n_surgeons = max(n_rooms * 2, round(Int, n_rooms * 2.5))
    end

    rd = n_rooms * n_days
    # Solve 2 * N * rd + 3 * rd + N ≈ target => N * (2*rd + 1) + 3*rd ≈ target => N = round((target - 3*rd) / (2*rd + 1))
    n_surgeries = max(2, round(Int, (target - 3 * rd) / (2 * rd + 1)))

    actual_status = feasibility_status

    # Room-specialty compatibility
    room_specialty_matrix = ones(Int, n_rooms, n_specialties)
    if n_specialties > 2 && n_rooms > 2
        for s in 2:n_specialties, r in 1:n_rooms
            if rand() < 0.30 && sum(room_specialty_matrix[:, s]) > 1
                room_specialty_matrix[r, s] = 0
            end
        end
    end
    for s in 1:n_specialties
        if sum(room_specialty_matrix[:, s]) == 0
            room_specialty_matrix[rand(1:n_rooms), s] = 1
        end
    end

    # Assign each surgeon a primary specialty
    surgeon_specialty = [mod1(j, n_specialties) for j in 1:n_surgeons]

    # Generate surgeries
    surgery_specialty = Vector{Int}(undef, n_surgeries)
    surgery_surgeon = Vector{Int}(undef, n_surgeries)
    nominal_duration = Vector{Float64}(undef, n_surgeries)
    duration_deviation = Vector{Float64}(undef, n_surgeries)
    cleaning_time = Vector{Float64}(undef, n_surgeries)
    urgency_weight = Vector{Float64}(undef, n_surgeries)
    postpone_cost = Vector{Float64}(undef, n_surgeries)

    base_durations = [110.0, 130.0, 190.0, 210.0, 50.0, 85.0, 75.0, 140.0]

    for i in 1:n_surgeries
        s = rand(1:n_specialties)
        surgery_specialty[i] = s

        eligible_surgeons = findall(==(s), surgeon_specialty)
        surgery_surgeon[i] = isempty(eligible_surgeons) ? rand(1:n_surgeons) : rand(eligible_surgeons)

        base_mean = base_durations[min(s, length(base_durations))]
        nom = clamp(rand(Normal(base_mean, base_mean * 0.15)), 20.0, 300.0)
        nominal_duration[i] = nom
        # Deviation represents ~25-45% worst-case duration stretch
        duration_deviation[i] = nom * rand(Uniform(0.25, 0.45))
        cleaning_time[i] = rand(Uniform(15.0, 25.0))
        urgency_weight[i] = rand(Uniform(1.0, 8.0))
        postpone_cost[i] = 1200.0 + urgency_weight[i] * rand(Uniform(250.0, 600.0))
    end

    # Uncertainty budget Gamma per room-day (protecting against up to ~1.5 to 2.5 concurrent overruns)
    uncertainty_budget = Matrix{Float64}(undef, n_rooms, n_days)
    for r in 1:n_rooms, d in 1:n_days
        uncertainty_budget[r, d] = rand(Uniform(1.2, 2.5))
    end

    regular_capacity = fill(480.0, n_rooms, n_days)
    max_overtime = fill(120.0, n_rooms, n_days)

    fixed_open_cost = Matrix{Float64}(undef, n_rooms, n_days)
    overtime_cost = Matrix{Float64}(undef, n_rooms, n_days)
    for r in 1:n_rooms, d in 1:n_days
        fixed_open_cost[r, d] = rand(Uniform(750.0, 1600.0))
        overtime_cost[r, d] = rand(Uniform(25.0, 50.0))
    end

    surgeon_availability = zeros(Int, n_surgeons, n_days)
    surgeon_max_hours = Matrix{Float64}(undef, n_surgeons, n_days)
    for j in 1:n_surgeons
        for d in 1:n_days
            avail = rand() < 0.75 ? 1 : 0
            surgeon_availability[j, d] = avail
            surgeon_max_hours[j, d] = 480.0
        end
        if sum(surgeon_availability[j, :]) == 0
            surgeon_availability[j, rand(1:n_days)] = 1
        end
    end

    force_all_scheduled = false

    if actual_status == feasible
        force_all_scheduled = true

        # Build witness schedule with robust protection
        room_day_nom = zeros(Float64, n_rooms, n_days)
        room_day_dev = zeros(Float64, n_rooms, n_days)
        surgeon_day_load = zeros(Float64, n_surgeons, n_days)

        for i in 1:n_surgeries
            s = surgery_specialty[i]
            j = surgery_surgeon[i]
            nom = nominal_duration[i]
            dev = duration_deviation[i]
            clean = cleaning_time[i]

            best_rd = nothing
            best_load = Inf

            for d in 1:n_days, r in 1:n_rooms
                room_specialty_matrix[r, s] == 1 || continue
                load = room_day_nom[r, d]
                if load < best_load
                    best_load = load
                    best_rd = (r, d)
                end
            end

            if best_rd === nothing
                room_specialty_matrix[1, s] = 1
                best_rd = (1, 1)
            end

            r_c, d_c = best_rd
            room_day_nom[r_c, d_c] += nom + clean
            room_day_dev[r_c, d_c] += dev
            surgeon_day_load[j, d_c] += nom
            surgeon_availability[j, d_c] = 1
        end

        for r in 1:n_rooms, d in 1:n_days
            needed_nom = room_day_nom[r, d]
            needed_dev = room_day_dev[r, d]
            gamma = uncertainty_budget[r, d]
            worst_case_robust_req = needed_nom + gamma * (needed_dev / max(1.0, needed_nom / 100.0))
            if worst_case_robust_req > 0
                regular_capacity[r, d] = max(480.0, ceil(worst_case_robust_req * 1.05))
                max_overtime[r, d] = max(120.0, ceil(worst_case_robust_req * 0.35))
            end
        end

        for j in 1:n_surgeons, d in 1:n_days
            needed = surgeon_day_load[j, d]
            if needed > 0
                surgeon_max_hours[j, d] = max(480.0, ceil(needed * 1.30))
            end
        end

    elseif actual_status == infeasible
        # Deterministic contradiction
        force_all_scheduled = true
        target_spec = 1
        compat_rooms = [r for r in 1:n_rooms if room_specialty_matrix[r, target_spec] == 1]
        if isempty(compat_rooms)
            room_specialty_matrix[1, target_spec] = 1
            compat_rooms = [1]
        end

        total_cap = sum(regular_capacity[r, d] + max_overtime[r, d] for r in compat_rooms, d in 1:n_days)
        for i in 1:n_surgeries
            surgery_specialty[i] = target_spec
        end
        boost_per_surgery = (total_cap * 1.6) / n_surgeries
        for i in 1:n_surgeries
            nominal_duration[i] = max(nominal_duration[i], boost_per_surgery)
        end
    end

    return OperatingRoomRobustSchedulingProblem(
        n_surgeries,
        n_rooms,
        n_days,
        n_specialties,
        n_surgeons,
        surgery_specialty,
        surgery_surgeon,
        nominal_duration,
        duration_deviation,
        cleaning_time,
        urgency_weight,
        postpone_cost,
        uncertainty_budget,
        room_specialty_matrix,
        regular_capacity,
        max_overtime,
        fixed_open_cost,
        overtime_cost,
        surgeon_availability,
        surgeon_max_hours,
        force_all_scheduled,
        actual_status,
    )
end

"""
    build_model(prob::OperatingRoomRobustSchedulingProblem)

Build a JuMP model for robust advance elective surgery scheduling.
Deterministic implementation.
"""
function build_model(prob::OperatingRoomRobustSchedulingProblem)
    model = Model()

    N = prob.n_surgeries
    R = prob.n_rooms
    D = prob.n_days
    J = prob.n_surgeons

    # Decision variables
    @variable(model, x[1:N, 1:R, 1:D], Bin)
    @variable(model, mu[1:N, 1:R, 1:D] >= 0)
    @variable(model, theta[1:R, 1:D] >= 0)
    @variable(model, y[1:R, 1:D], Bin)
    @variable(model, overtime[1:R, 1:D] >= 0)
    @variable(model, postpone[1:N], Bin)

    # Objective: Minimize fixed opening costs + overtime + postponement + waiting penalties
    @objective(
        model,
        Min,
        sum(prob.fixed_open_cost[r, d] * y[r, d] for r in 1:R, d in 1:D) +
        sum(prob.overtime_cost[r, d] * overtime[r, d] for r in 1:R, d in 1:D) +
        sum(prob.postpone_cost[i] * postpone[i] for i in 1:N) +
        sum(prob.urgency_weight[i] * d * x[i, r, d] for i in 1:N, r in 1:R, d in 1:D)
    )

    # 1. Surgery assignment or postponement
    for i in 1:N
        @constraint(model, sum(x[i, r, d] for r in 1:R, d in 1:D) + postpone[i] == 1)
        if prob.force_all_scheduled
            @constraint(model, postpone[i] == 0)
        end
    end

    # 2. Room-specialty compatibility
    for i in 1:N, r in 1:R, d in 1:D
        s = prob.surgery_specialty[i]
        if prob.room_specialty_matrix[r, s] == 0
            @constraint(model, x[i, r, d] == 0)
        end
    end

    # 3. Robust capacity constraint (dualized budget-of-uncertainty)
    for r in 1:R, d in 1:D
        nominal_load = sum((prob.nominal_duration[i] + prob.cleaning_time[i]) * x[i, r, d] for i in 1:N)
        robust_term = prob.uncertainty_budget[r, d] * theta[r, d] + sum(mu[i, r, d] for i in 1:N)
        @constraint(model, nominal_load + robust_term <= prob.regular_capacity[r, d] * y[r, d] + overtime[r, d])
        @constraint(model, overtime[r, d] <= prob.max_overtime[r, d] * y[r, d])
    end

    # 4. Dual uncertainty protection linking
    for i in 1:N, r in 1:R, d in 1:D
        @constraint(model, theta[r, d] + mu[i, r, d] >= prob.duration_deviation[i] * x[i, r, d])
    end

    # 5. Surgeon daily capacity and availability
    for j in 1:J, d in 1:D
        surgeries_j = findall(==(j), prob.surgery_surgeon)
        if !isempty(surgeries_j)
            surg_time_expr = sum(prob.nominal_duration[i] * sum(x[i, r, d] for r in 1:R) for i in surgeries_j)
            max_surg = prob.surgeon_max_hours[j, d] * prob.surgeon_availability[j, d]
            @constraint(model, surg_time_expr <= max_surg)
        end
    end

    return model
end

# Register the robust elective variant
register_variant(
    :operating_room_scheduling,
    :robust_elective,
    OperatingRoomRobustSchedulingProblem,
    "Robust advance elective case scheduling under duration uncertainty with budget-of-uncertainty protection",
)
