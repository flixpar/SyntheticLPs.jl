using JuMP
using Random
using Distributions

"""
    LeeftinkHansORSchedulingProblem <: ProblemGenerator

Benchmark-informed elective-case loading across identical 480-minute OR-days.
The case list follows the Leeftink--Hans benchmark design: a load factor from
`0.80:0.05:1.20`, expected durations from empirical three-parameter-lognormal
surgery types, and a case list whose expected workload is within 2.5% of the
target whenever the requested scale permits it.  The complete empirical files
are compressed to documented weighted-quantile archetypes in
`leeftink_hans_data.jl`; source type IDs and fitted parameters remain visible in
every generated instance.

This is intentionally a loading/assignment model, not a waiting-list model:
the public benchmark has no surgeons, deadlines, urgency, or bed data.  The
model assigns cases to OR-days, permits explicitly synthetic cancellation and
overtime recourse, and minimizes cancellation plus overtime cost.
"""
struct LeeftinkHansORSchedulingProblem <: ProblemGenerator
    n_surgeries::Int
    n_or_days::Int
    session_length::Float64
    target_load::Float64
    achieved_load::Float64
    benchmark_scale::Symbol
    specialty_code::Vector{Symbol}
    surgery_type_id::Vector{Int}
    duration_mu::Vector{Float64}
    duration_sigma::Vector{Float64}
    duration_gamma::Vector{Float64}
    expected_duration::Vector{Float64}
    realized_duration::Vector{Float64}
    cancellation_cost::Vector{Float64}
    overtime_cost::Vector{Float64}
    max_overtime::Vector{Float64}
    mandatory::BitVector
    feasible_witness::Union{Nothing,Vector{Int}}
    infeasibility_excess::Union{Nothing,Float64}
    feasibility_status::FeasibilityStatus
end

function _benchmark_case_list(rng::AbstractRNG, n_or_days::Int, load::Float64)
    target_minutes = load * 480.0 * n_or_days
    tolerance_minutes = 0.025 * 480.0 * n_or_days
    codes = Symbol[]
    ids = Int[]
    mus = Float64[]
    sigmas = Float64[]
    gammas = Float64[]
    means = Float64[]
    realized = Float64[]
    specialty_cum = cumsum([p.weight for p in _ORSCHED_SPECIALTIES])

    # Greedy residual fitting from the empirical archetype support.  Near the
    # end, choose the closest of a random candidate batch; this mirrors the
    # benchmark's proximity-based list selection and reliably meets 2.5%.
    while sum(means) < target_minutes - tolerance_minutes
        residual = target_minutes - sum(means)
        candidates = Tuple{Int,Any}[]
        for _ in 1:32
            k = _orsched_pick(rng, specialty_cum)
            push!(candidates, (k, _orsched_sample_benchmark_type(rng, k)))
        end
        feasible_candidates = [(k, t) for (k, t) in candidates
                               if _orsched_type_mean(t) <=
                                  residual + tolerance_minutes]
        chosen = isempty(feasible_candidates) ?
                 candidates[argmin([abs(_orsched_type_mean(t) - residual)
                                    for (_, t) in candidates])] :
                 feasible_candidates[argmin([abs(_orsched_type_mean(t) - residual)
                                             for (_, t) in feasible_candidates])]
        k, t = chosen
        push!(codes, _ORSCHED_SPECIALTIES[k].code)
        push!(ids, t.id)
        push!(mus, t.mu)
        push!(sigmas, t.sigma)
        push!(gammas, t.gamma)
        push!(means, _orsched_type_mean(t))
        push!(realized, t.gamma + rand(rng, LogNormal(t.mu, t.sigma)))
        length(means) > 20_000 && error("benchmark case-list generation did not converge")
    end
    return (codes=codes, ids=ids, mus=mus, sigmas=sigmas, gammas=gammas,
            means=means, realized=realized)
end

function _benchmark_lpt_assignment(duration::Vector{Float64}, n_or_days::Int)
    load = zeros(Float64, n_or_days)
    assignment = zeros(Int, length(duration))
    for i in sortperm(duration; rev=true)
        q = argmin(load)
        assignment[i] = q
        load[q] += duration[i]
    end
    return assignment, load
end

function LeeftinkHansORSchedulingProblem(target_variables::Int,
                                         feasibility_status::FeasibilityStatus,
                                         seed::Int)
    rng = MersenneTwister(seed)
    target = max(target_variables, 20)
    requested_load = feasibility_status == infeasible ?
                     rand(rng, _ORSCHED_BENCHMARK_LOADS[6:end]) :
                     _orsched_load_target(rng)

    best = nothing
    best_gap = Inf
    max_days = max(40, ceil(Int, sqrt(target)))
    for q in 1:max_days
        # Case count is endogenous to sampled empirical types.  A few
        # independent lists per OR-day count materially improve the package's
        # size match without altering the published load construction.
        for _ in 1:4
            cases = _benchmark_case_list(rng, q, requested_load)
            n = length(cases.means)
            variables = n * q + n + q
            gap = abs(variables - target) / target
            if gap < best_gap
                best_gap = gap
                best = (q=q, cases=cases)
            end
            best_gap <= 0.025 && break
        end
        best_gap <= 0.025 && break
    end

    n_or_days = best.q
    cases = best.cases
    expected = cases.means
    n_surgeries = length(expected)
    achieved_load = sum(expected) / (480.0 * n_or_days)
    assignment, room_load = _benchmark_lpt_assignment(expected, n_or_days)
    cancellation_cost = [rand(rng, Uniform(800.0, 2400.0)) for _ in 1:n_surgeries]
    overtime_cost = [rand(rng, Uniform(3.0, 8.0)) for _ in 1:n_or_days]
    mandatory = falses(n_surgeries)
    max_overtime = fill(60.0, n_or_days)
    witness = nothing
    excess = nothing

    if feasibility_status == feasible
        mandatory .= true
        max_overtime = [max(0.0, ceil(room_load[q] - 480.0)) for q in 1:n_or_days]
        witness = assignment
    elseif feasibility_status == infeasible
        mandatory .= true
        max_overtime .= 0.0
        excess = sum(expected) - 480.0 * n_or_days
        @assert excess > 0
    else
        n_mandatory = clamp(round(Int, rand(rng, Uniform(0.08, 0.18)) * n_surgeries),
                            0, n_surgeries)
        n_mandatory > 0 && (mandatory[shuffle(rng, 1:n_surgeries)[1:n_mandatory]] .= true)
    end

    scale = n_or_days in _ORSCHED_BENCHMARK_OR_DAYS ? :published_or_days : :scaled_or_days
    return LeeftinkHansORSchedulingProblem(
        n_surgeries, n_or_days, 480.0, requested_load, achieved_load, scale,
        cases.codes, cases.ids, cases.mus, cases.sigmas, cases.gammas,
        expected, cases.realized, cancellation_cost, overtime_cost,
        max_overtime, mandatory, witness, excess, feasibility_status,
    )
end

function build_model(prob::LeeftinkHansORSchedulingProblem)
    model = Model()
    N, Q = prob.n_surgeries, prob.n_or_days
    @variable(model, assign[1:N, 1:Q], Bin)
    @variable(model, cancel[1:N], Bin)
    @variable(model, 0 <= overtime[q=1:Q] <= prob.max_overtime[q])

    @constraint(model, case_assignment[i=1:N],
                sum(assign[i, q] for q in 1:Q) + cancel[i] == 1)
    for i in 1:N
        prob.mandatory[i] && @constraint(model, cancel[i] == 0)
    end
    @constraint(model, or_day_capacity[q=1:Q],
                sum(prob.expected_duration[i] * assign[i, q] for i in 1:N) -
                overtime[q] <= prob.session_length)

    @objective(model, Min,
        sum(prob.cancellation_cost[i] * cancel[i] for i in 1:N) +
        sum(prob.overtime_cost[q] * overtime[q] for q in 1:Q))
    return model
end

register_variant(
    :operating_room_scheduling,
    :benchmark_loading,
    LeeftinkHansORSchedulingProblem,
    "Leeftink--Hans benchmark-informed 480-minute OR-day loading with empirical three-parameter-lognormal surgery types and calibrated 0.80--1.20 load",
)
