using JuMP
using Random
using Distributions

"""
    OperatingRoomDailySequencingProblem <: ProblemGenerator

Generator for daily operational Operating Room (OR) scheduling and sequencing problems.

# Overview
Models the intra-day scheduling and sequencing of surgical procedures across operating
rooms, tracking pre-operative preparation (holding/anesthesia induction), surgical procedure
(with room turnover/cleaning times), and post-operative PACU recovery.

Constraints model:
- Room assignment and equipment/specialty compatibility.
- Pre-op -> Intra-op -> Post-op sequential stage progression for each patient.
- Disjunctive no-overlap between surgeries assigned to the same room (with turnover cleaning times).
- Disjunctive no-overlap for surgeries performed by the same surgeon across different rooms.
- Operating room overtime tracking against regular working shift duration.
- Makespan and patient completion (flow time) minimization.

# Fields
- `n_surgeries::Int`: Number of surgical cases to be performed
- `n_rooms::Int`: Number of operating rooms
- `n_surgeons::Int`: Number of surgeons
- `n_specialties::Int`: Number of surgical specialties
- `surgery_specialty::Vector{Int}`: Specialty of each surgery
- `surgery_surgeon::Vector{Int}`: Surgeon assigned to each surgery
- `preop_duration::Vector{Float64}`: Pre-operative preparation duration (minutes)
- `surgery_duration::Vector{Float64}`: Surgical procedure duration (minutes)
- `cleaning_time::Vector{Float64}`: Turnover/cleaning duration (minutes)
- `pacu_duration::Vector{Float64}`: PACU recovery duration (minutes)
- `room_specialty_matrix::Matrix{Int}`: Binary compatibility matrix (R x S)
- `urgency_weights::Vector{Float64}`: Priority weights for patient completion time
- `regular_day_length::Float64`: Standard shift duration (minutes, e.g. 480 min)
- `overtime_penalty::Float64`: Weight on room overtime
- `makespan_penalty::Float64`: Weight on overall makespan
- `surgeon_pairs::Vector{Tuple{Int,Int}}`: Pairs of surgeries sharing the same surgeon
- `surgery_pairs::Vector{Tuple{Int,Int}}`: All unordered pairs of surgeries (i < i')
- `big_m::Float64`: Big-M constant for disjunctive constraints
- `max_makespan_limit::Float64`: Upper bound limit on makespan
- `feasibility_status::FeasibilityStatus`: Resolved feasibility status
"""
struct OperatingRoomDailySequencingProblem <: ProblemGenerator
    n_surgeries::Int
    n_rooms::Int
    n_surgeons::Int
    n_specialties::Int
    surgery_specialty::Vector{Int}
    surgery_surgeon::Vector{Int}
    preop_duration::Vector{Float64}
    surgery_duration::Vector{Float64}
    cleaning_time::Vector{Float64}
    pacu_duration::Vector{Float64}
    room_specialty_matrix::Matrix{Int}
    urgency_weights::Vector{Float64}
    regular_day_length::Float64
    overtime_penalty::Float64
    makespan_penalty::Float64
    surgeon_pairs::Vector{Tuple{Int,Int}}
    surgery_pairs::Vector{Tuple{Int,Int}}
    big_m::Float64
    max_makespan_limit::Float64
    feasibility_status::FeasibilityStatus
end

"""
    OperatingRoomDailySequencingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a daily OR scheduling and sequencing instance targeting `target_variables`.

# Variable-count formula
- `x[1:n_surgeries, 1:n_rooms]`: `n_surgeries * n_rooms`
- `t_pre`, `t_surg`, `t_pacu`, `completion`: `4 * n_surgeries`
- `room_order[1:length(surgery_pairs)]`: `n_surgeries * (n_surgeries - 1) ÷ 2`
- `surgeon_order[1:length(surgeon_pairs)]`: `length(surgeon_pairs)`
- `room_overtime[1:n_rooms]`: `n_rooms`
- `makespan`: 1
"""
function OperatingRoomDailySequencingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    target = max(20, target_variables)

    # Search for (N, R) combinations that best match the target variable count
    best_n = 4
    best_r = 2
    best_err = Inf

    r_candidates = target <= 100 ? (2:3) :
                   target <= 600 ? (2:5) :
                   target <= 2500 ? (4:8) : (6:12)

    for r in r_candidates
        # Estimate N from quadratic approx: N^2/2 + N*(R + 4) + R + 1 ≈ target
        # 0.5 N^2 + (R + 4.5) N + R + 1 - target ≈ 0
        a = 0.5
        b = r + 4.5
        c = r + 1 - target
        disc = b^2 - 4 * a * c
        disc > 0 || continue
        n_est = (-b + sqrt(disc)) / (2 * a)
        n_int = max(2, round(Int, n_est))

        for n in max(2, n_int - 2):(n_int + 2)
            # Estimate surgeon pairs as approx ~ n*(n-1)/(2 * (r+1))
            pair_count = n * (n - 1) ÷ 2
            est_surg_pairs = round(Int, pair_count * 0.25)
            vc = n * r + 4 * n + pair_count + est_surg_pairs + r + 1
            err = abs(vc - target) / target
            if err < best_err
                best_err = err
                best_n = n
                best_r = r
            end
        end
    end

    n_surgeries = best_n
    n_rooms = best_r
    n_specialties = min(6, max(2, round(Int, n_rooms * 0.8)))
    n_surgeons = max(2, round(Int, n_rooms * 1.5))

    actual_status = feasibility_status

    # Room-specialty compatibility
    room_specialty_matrix = ones(Int, n_rooms, n_specialties)
    if n_specialties > 2 && n_rooms > 2
        for s in 2:n_specialties, r in 1:n_rooms
            if rand() < 0.35 && sum(room_specialty_matrix[:, s]) > 1
                room_specialty_matrix[r, s] = 0
            end
        end
    end
    # Ensure every specialty can be done in at least one room
    for s in 1:n_specialties
        if sum(room_specialty_matrix[:, s]) == 0
            room_specialty_matrix[rand(1:n_rooms), s] = 1
        end
    end

    # Assign specialties and surgeons to surgeries
    surgery_specialty = [rand(1:n_specialties) for _ in 1:n_surgeries]
    surgery_surgeon = [rand(1:n_surgeons) for _ in 1:n_surgeries]

    # Durations (minutes)
    preop_duration = Vector{Float64}(undef, n_surgeries)
    surgery_duration = Vector{Float64}(undef, n_surgeries)
    cleaning_time = Vector{Float64}(undef, n_surgeries)
    pacu_duration = Vector{Float64}(undef, n_surgeries)
    urgency_weights = Vector{Float64}(undef, n_surgeries)

    for i in 1:n_surgeries
        s = surgery_specialty[i]
        preop_duration[i] = rand(Uniform(15.0, 35.0))
        surgery_duration[i] = rand(Gamma(3.0, 30.0)) + rand(Uniform(20.0, 60.0))
        cleaning_time[i] = rand(Uniform(15.0, 25.0))
        pacu_duration[i] = rand(Uniform(60.0, 150.0))
        urgency_weights[i] = rand(Uniform(1.0, 5.0))
    end

    # Build unordered pairs of all surgeries
    surgery_pairs = Tuple{Int,Int}[]
    for i in 1:(n_surgeries - 1)
        for k in (i + 1):n_surgeries
            push!(surgery_pairs, (i, k))
        end
    end

    # Build pairs of surgeries sharing the same surgeon
    surgeon_pairs = Tuple{Int,Int}[]
    for j in 1:n_surgeons
        surgeries_j = findall(==(j), surgery_surgeon)
        for a in 1:(length(surgeries_j) - 1)
            for b in (a + 1):length(surgeries_j)
                push!(surgeon_pairs, (surgeries_j[a], surgeries_j[b]))
            end
        end
    end

    regular_day_length = 480.0 # 8 hours
    overtime_penalty = 2.5
    makespan_penalty = 0.5

    total_proc_time = sum(preop_duration) + sum(surgery_duration) + sum(cleaning_time) + sum(pacu_duration)
    big_m = 2.0 * total_proc_time + regular_day_length + 1000.0

    max_makespan_limit = big_m

    if actual_status == infeasible
        # Force a deterministic contradiction:
        # Minimum completion time of surgery 1 alone is preop + surgery + pacu.
        # Set max_makespan_limit strictly below this minimum.
        min_comp_1 = preop_duration[1] + surgery_duration[1] + pacu_duration[1]
        max_makespan_limit = 0.5 * min_comp_1
    end

    return OperatingRoomDailySequencingProblem(
        n_surgeries,
        n_rooms,
        n_surgeons,
        n_specialties,
        surgery_specialty,
        surgery_surgeon,
        preop_duration,
        surgery_duration,
        cleaning_time,
        pacu_duration,
        room_specialty_matrix,
        urgency_weights,
        regular_day_length,
        overtime_penalty,
        makespan_penalty,
        surgeon_pairs,
        surgery_pairs,
        big_m,
        max_makespan_limit,
        actual_status,
    )
end

"""
    build_model(prob::OperatingRoomDailySequencingProblem)

Build a JuMP model for daily multi-stage operating room scheduling and sequencing.
Deterministic implementation.
"""
function build_model(prob::OperatingRoomDailySequencingProblem)
    model = Model()

    N = prob.n_surgeries
    R = prob.n_rooms
    bigM = prob.big_m

    # Decision variables
    @variable(model, x[1:N, 1:R], Bin)
    @variable(model, t_pre[1:N] >= 0)
    @variable(model, t_surg[1:N] >= 0)
    @variable(model, t_pacu[1:N] >= 0)
    @variable(model, completion[1:N] >= 0)
    @variable(model, room_order[1:length(prob.surgery_pairs)], Bin)
    @variable(model, surgeon_order[1:length(prob.surgeon_pairs)], Bin)
    @variable(model, room_overtime[1:R] >= 0)
    @variable(model, makespan >= 0)

    # Objective: Minimize patient completion flow time + overtime penalties + makespan
    @objective(
        model,
        Min,
        sum(prob.urgency_weights[i] * completion[i] for i in 1:N) +
        prob.overtime_penalty * sum(room_overtime[r] for r in 1:R) +
        prob.makespan_penalty * makespan
    )

    # 1. Each surgery assigned to exactly one compatible room
    for i in 1:N
        @constraint(model, sum(x[i, r] for r in 1:R) == 1)
        s = prob.surgery_specialty[i]
        for r in 1:R
            if prob.room_specialty_matrix[r, s] == 0
                @constraint(model, x[i, r] == 0)
            end
        end
    end

    # 2. Pre-op -> Intra-op -> Post-op sequential stage progression
    for i in 1:N
        @constraint(model, t_surg[i] >= t_pre[i] + prob.preop_duration[i])
        @constraint(model, t_pacu[i] >= t_surg[i] + prob.surgery_duration[i])
        @constraint(model, completion[i] >= t_pacu[i] + prob.pacu_duration[i])
    end

    # 3. Disjunctive no-overlap for surgeries assigned to the same room
    for (k, (i, ip)) in enumerate(prob.surgery_pairs)
        for r in 1:R
            # If both i and ip are in room r:
            # room_order[k] == 1 => i precedes ip: t_surg[ip] >= t_surg[i] + dur[i] + clean[i]
            # room_order[k] == 0 => ip precedes i: t_surg[i] >= t_surg[ip] + dur[ip] + clean[ip]
            @constraint(
                model,
                t_surg[ip] >= t_surg[i] + prob.surgery_duration[i] + prob.cleaning_time[i] -
                              bigM * (1 - room_order[k]) - bigM * (2 - x[i, r] - x[ip, r])
            )
            @constraint(
                model,
                t_surg[i] >= t_surg[ip] + prob.surgery_duration[ip] + prob.cleaning_time[ip] -
                             bigM * room_order[k] - bigM * (2 - x[i, r] - x[ip, r])
            )
        end
    end

    # 4. Disjunctive no-overlap for surgeries performed by the same surgeon
    for (k, (i, ip)) in enumerate(prob.surgeon_pairs)
        @constraint(
            model,
            t_surg[ip] >= t_surg[i] + prob.surgery_duration[i] - bigM * (1 - surgeon_order[k])
        )
        @constraint(
            model,
            t_surg[i] >= t_surg[ip] + prob.surgery_duration[ip] - bigM * surgeon_order[k]
        )
    end

    # 5. Room overtime tracking
    for r in 1:R, i in 1:N
        @constraint(
            model,
            room_overtime[r] >= t_surg[i] + prob.surgery_duration[i] + prob.cleaning_time[i] -
                                prob.regular_day_length - bigM * (1 - x[i, r])
        )
    end

    # 6. Makespan tracking and bound
    for i in 1:N
        @constraint(model, makespan >= completion[i])
    end
    @constraint(model, makespan <= prob.max_makespan_limit)

    return model
end

# Register the daily sequencing variant
register_variant(
    :operating_room_scheduling,
    :daily_sequencing,
    OperatingRoomDailySequencingProblem,
    "Daily multi-stage surgical case scheduling and sequencing with pre-op, intra-op OR, post-op PACU, and surgeon disjunctions",
)
