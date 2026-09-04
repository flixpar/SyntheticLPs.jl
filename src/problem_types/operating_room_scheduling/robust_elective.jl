using JuMP
using Random
using Distributions

"""
    RobustElectiveSurgeryAssignmentProblem <: ProblemGenerator

Sparse robust counterpart of `elective_assignment` using the Bertsimas--Sim
budget of uncertainty.  Nominal case duration is the empirical benchmark type
mean and its nonnegative deviation is calibrated from the fitted type standard
deviation.  For each open block `q`, `Gamma[q]` controls how many assigned cases
may simultaneously realize their maximum deviation.  One `mu` variable is
created per admissible assignment, rather than per dense surgery/room/day
combination.
"""
struct RobustElectiveSurgeryAssignmentProblem <: ProblemGenerator
    n_surgeries::Int
    n_rooms::Int
    n_days::Int
    specialty_names::Vector{Symbol}
    surgery_specialty::Vector{Int}
    surgery_type_id::Vector{Int}
    nominal_duration::Vector{Float64}
    duration_deviation::Vector{Float64}
    surgery_deadline::Vector{Int}
    postponement_penalty::Vector{Float64}
    surgery_surgeon::Vector{Int}
    mandatory::BitVector
    surgeon_budget::Matrix{Float64}
    session_length::Matrix{Float64}
    turnover::Float64
    max_overtime::Vector{Float64}
    overtime_cost::Float64
    uncertainty_budget::Vector{Float64}
    admissible::Vector{Tuple{Int, Int, Int}}
    open_blocks::Vector{Tuple{Int, Int}}
    feasible_witness::Union{Nothing, Vector{Int}}
    infeasible_surgery::Union{Nothing, Int}
    feasibility_status::FeasibilityStatus
end

function _robust_extra_capacity(deviations::Vector{Float64}, gamma::Float64)
    isempty(deviations) && return 0.0
    sorted = sort(deviations; rev=true)
    whole = min(floor(Int, gamma), length(sorted))
    value = sum(sorted[1:whole]; init=0.0)
    if whole < length(sorted)
        value += (gamma - whole) * sorted[whole + 1]
    end
    return value
end

function RobustElectiveSurgeryAssignmentProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target = max(target_variables, 20)
    base_target = max(20, round(Int, target / 1.7))
    best = nothing
    best_gap = Inf
    for _ in 1:16
        base = ElectiveSurgeryAssignmentProblem(base_target, feasibility_status, seed)
        total = 2length(base.admissible) + base.n_surgeries + 2length(base.open_blocks)
        gap = abs(total - target) / target
        if gap < best_gap
            best_gap = gap
            best = base
        end
        gap <= 0.05 && break
        next_target = max(20, round(Int, base_target * target / max(total, 1)))
        next_target == base_target && (next_target += total < target ? 1 : -1)
        base_target = max(20, next_target)
    end
    base = best
    rng = MersenneTwister(seed + 23063)
    deviation = [
        clamp(1.2816 * base.surgery_duration_sd[i], 5.0, 0.75 * base.surgery_duration[i]) for
        i in 1:base.n_surgeries
    ]
    gamma = [rand(rng, Uniform(1.0, 3.0)) for _ in base.open_blocks]
    max_overtime = fill(base.max_overtime, length(base.open_blocks))

    if feasibility_status == feasible
        open_index = Dict(q => k for (k, q) in enumerate(base.open_blocks))
        assigned_by_block = [Int[] for _ in base.open_blocks]
        for a in something(base.feasible_witness)
            i, r, d = base.admissible[a]
            push!(assigned_by_block[open_index[(r, d)]], i)
        end
        for q in eachindex(base.open_blocks)
            r, d = base.open_blocks[q]
            nominal = sum(
                base.surgery_duration[i] + base.turnover for i in assigned_by_block[q]; init=0.0
            )
            robust = _robust_extra_capacity(deviation[assigned_by_block[q]], gamma[q])
            max_overtime[q] = max(
                max_overtime[q], ceil(nominal + robust - base.session_length[r, d])
            )
        end
    end

    return RobustElectiveSurgeryAssignmentProblem(
        base.n_surgeries,
        base.n_rooms,
        base.n_days,
        base.specialty_names,
        base.surgery_specialty,
        base.surgery_type_id,
        base.surgery_duration,
        deviation,
        base.surgery_deadline,
        base.postponement_penalty,
        base.surgery_surgeon,
        base.mandatory,
        base.surgeon_budget,
        base.session_length,
        base.turnover,
        max_overtime,
        base.overtime_cost,
        gamma,
        base.admissible,
        base.open_blocks,
        base.feasible_witness,
        base.infeasible_surgery,
        feasibility_status,
    )
end

function build_model(prob::RobustElectiveSurgeryAssignmentProblem)
    model = Model()
    N = prob.n_surgeries
    A = length(prob.admissible)
    Q = length(prob.open_blocks)
    open_index = Dict(block => q for (q, block) in enumerate(prob.open_blocks))
    by_surgery = [Int[] for _ in 1:N]
    by_block = [Int[] for _ in 1:Q]
    by_surgeon_day = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (a, (i, r, d)) in enumerate(prob.admissible)
        push!(by_surgery[i], a)
        push!(by_block[open_index[(r, d)]], a)
        push!(get!(by_surgeon_day, (prob.surgery_surgeon[i], d), Int[]), a)
    end

    @variable(model, assign[1:A], Bin)
    @variable(model, postpone[1:N], Bin)
    @variable(model, 0 <= overtime[q = 1:Q] <= prob.max_overtime[q])
    @variable(model, theta[1:Q] >= 0)
    @variable(model, mu[1:A] >= 0)

    @constraint(
        model,
        case_assignment[i = 1:N],
        sum(assign[a] for a in by_surgery[i]; init=0.0) + postpone[i] == 1
    )
    for i in 1:N
        prob.mandatory[i] && @constraint(model, postpone[i] == 0)
    end
    for q in 1:Q
        r, d = prob.open_blocks[q]
        @constraint(
            model,
            sum(
                (prob.nominal_duration[prob.admissible[a][1]] + prob.turnover) * assign[a] for
                a in by_block[q];
                init=0.0,
            ) +
            prob.uncertainty_budget[q] * theta[q] +
            sum(mu[a] for a in by_block[q]; init=0.0) - overtime[q] <= prob.session_length[r, d]
        )
        for a in by_block[q]
            i = prob.admissible[a][1]
            @constraint(model, theta[q] + mu[a] >= prob.duration_deviation[i] * assign[a])
        end
    end
    for (s, d) in sort!(collect(keys(by_surgeon_day)))
        idxs = by_surgeon_day[(s, d)]
        @constraint(
            model,
            sum(prob.nominal_duration[prob.admissible[a][1]] * assign[a] for a in idxs) <=
                prob.surgeon_budget[s, d]
        )
    end

    @objective(
        model,
        Min,
        sum(prob.postponement_penalty[i] * postpone[i] for i in 1:N) +
            prob.overtime_cost * sum(overtime[q] for q in 1:Q)
    )
    return model
end

register_variant(
    :operating_room_scheduling,
    :robust_elective,
    RobustElectiveSurgeryAssignmentProblem,
    "Sparse Bertsimas--Sim robust elective assignment with empirical duration deviations, surgeon availability, capped overtime, and postponement",
)
