using JuMP
using Random
using Distributions

"""
    LeeftinkHansORSchedulingProblem <: ProblemGenerator

Benchmark-informed elective-case loading across 480-minute OR-days.
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

# Sparse block structure (package extension)

The OR-days are laid out as `rooms_per_day` rooms over `n_calendar_days`
days, and each OR-day is a specialty block (blocks are dealt to specialties in
proportion to their share of the case workload). A case may only be loaded
into blocks of its own specialty inside its scheduling window
`[case_release, case_due]`, sized so a case has about ten admissible OR-days.
The previous dense `cases x OR-days` model had only `cases + OR-days` rows (327
rows at 10k columns); the sparse model keeps rows near 10% of the columns at
every size. Mandatory cases have no cancellation column.

# Feasibility

  - `feasible`: every case is mandatory; a longest-processing-time assignment
    over each case's admissible blocks is planted (`feasible_witness`, the
    OR-day per case) and each block's overtime cap is set to the witness's
    excess load.
  - `infeasible`: the load factor is drawn from `1.05:0.05:1.20`, every case
    is mandatory and no overtime is allowed: summing all assignment rows
    against all OR-day capacity rows gives
    `sum expected_duration <= 480 * n_or_days`, violated by
    `infeasibility_excess > 0`. The refutation aggregates every row, so
    presolve does not detect it.
  - `unknown`: 60-100% of cases are mandatory and each block's overtime cap is
    a global factor in `[0.3, 1.3]` times the excess load of a reference LPT
    plan, so feasibility depends on how much better than LPT the LP can pack
    the mandatory cases into their windows and specialty blocks.
"""
struct LeeftinkHansORSchedulingProblem <: ProblemGenerator
    n_surgeries::Int
    n_or_days::Int
    n_calendar_days::Int
    rooms_per_day::Int
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
    or_day_specialty::Vector{Symbol}
    or_day_calendar::Vector{Int}
    case_release::Vector{Int}
    case_due::Vector{Int}
    admissible::Vector{Vector{Int}}
    cancellation_cost::Vector{Float64}
    overtime_cost::Vector{Float64}
    max_overtime::Vector{Float64}
    mandatory::BitVector
    feasible_witness::Union{Nothing, Vector{Int}}
    infeasibility_excess::Union{Nothing, Float64}
    feasibility_status::FeasibilityStatus
end

function _benchmark_case_list(
    rng::AbstractRNG, n_or_days::Int, load::Float64, allowed::Vector{Int}=collect(eachindex(_ORSCHED_SPECIALTIES))
)
    target_minutes = load * 480.0 * n_or_days
    tolerance_minutes = 0.025 * 480.0 * n_or_days
    codes = Symbol[]
    specs = Int[]
    ids = Int[]
    mus = Float64[]
    sigmas = Float64[]
    gammas = Float64[]
    means = Float64[]
    realized = Float64[]
    specialty_cum = cumsum([_ORSCHED_SPECIALTIES[k].weight for k in allowed])
    total = 0.0

    # Greedy residual fitting from the empirical archetype support.  Near the
    # end, choose the closest of a random candidate batch; this mirrors the
    # benchmark's proximity-based list selection and reliably meets 2.5%.
    while total < target_minutes - tolerance_minutes
        residual = target_minutes - total
        candidates = Tuple{Int, Any}[]
        for _ in 1:32
            k = allowed[_orsched_pick(rng, specialty_cum)]
            push!(candidates, (k, _orsched_sample_benchmark_type(rng, k)))
        end
        feasible_candidates = [
            (k, t) for (k, t) in candidates if _orsched_type_mean(t) <= residual + tolerance_minutes
        ]
        chosen = if isempty(feasible_candidates)
            candidates[argmin([abs(_orsched_type_mean(t) - residual) for (_, t) in candidates])]
        else
            feasible_candidates[argmin([
                abs(_orsched_type_mean(t) - residual) for (_, t) in feasible_candidates
            ])]
        end
        k, t = chosen
        push!(codes, _ORSCHED_SPECIALTIES[k].code)
        push!(specs, k)
        push!(ids, t.id)
        push!(mus, t.mu)
        push!(sigmas, t.sigma)
        push!(gammas, t.gamma)
        push!(means, _orsched_type_mean(t))
        push!(realized, t.gamma + rand(rng, LogNormal(t.mu, t.sigma)))
        total += means[end]
    end
    return (
        codes=codes,
        specs=specs,
        ids=ids,
        mus=mus,
        sigmas=sigmas,
        gammas=gammas,
        means=means,
        realized=realized,
    )
end

"""
    _benchmark_blocks(rng, specs, means, n_or_days)

Lay the OR-days out as `rooms_per_day` rooms over `n_calendar_days` days, deal
them to specialties in proportion to workload (every specialty with cases gets
at least one block), and give each case a window over the calendar sized for
about ten admissible blocks of its specialty (widened until it holds at least
one). Returns `(or_spec, or_cal, n_cal, rooms, release, due, admissible)`.
"""
function _benchmark_blocks(
    rng::AbstractRNG, specs::Vector{Int}, means::Vector{Float64}, n_or_days::Int
)
    present = sort(unique(specs))
    load = [sum(means[i] for i in eachindex(specs) if specs[i] == k) for k in present]
    n_or_days = max(n_or_days, length(present))
    n_cal = clamp(round(Int, sqrt(n_or_days / 2)), 1, 60)
    rooms = cld(n_or_days, n_cal)
    # Largest-remainder block quotas, at least one per present specialty.
    quota = ones(Int, length(present))
    rest = n_or_days - length(present)
    if rest > 0
        shares = rest .* load ./ sum(load)
        extra = floor.(Int, shares)
        order = sortperm(shares .- extra; rev=true)
        for j in 1:(rest - sum(extra))
            extra[order[j]] += 1
        end
        quota .+= extra
    end
    or_spec = shuffle(rng, reduce(vcat, [fill(present[j], quota[j]) for j in eachindex(present)]))
    or_cal = [cld(q, rooms) for q in 1:n_or_days]
    n_cal = maximum(or_cal)
    blocks_of = Dict(k => findall(==(k), or_spec) for k in present)

    release = zeros(Int, length(specs))
    due = zeros(Int, length(specs))
    admissible = Vector{Vector{Int}}(undef, length(specs))
    for i in eachindex(specs)
        blocks = blocks_of[specs[i]]
        per_day = length(blocks) / n_cal
        width = clamp(round(Int, 10 / per_day), 1, n_cal)
        start = rand(rng, 1:(n_cal - width + 1))
        while true
            adm = [q for q in blocks if start <= or_cal[q] <= start + width - 1]
            if !isempty(adm) || width >= n_cal
                # In a large suite a one-day window can still hold dozens of
                # same-specialty blocks; a case is booked into its surgical
                # team's blocks, 8-12 of them.
                length(adm) > 12 && (adm = sort(shuffle(rng, adm)[1:rand(rng, 8:12)]))
                release[i], due[i], admissible[i] = start, start + width - 1, adm
                break
            end
            width += 1
            start = max(1, min(start, n_cal - width + 1))
        end
    end
    return or_spec, or_cal, n_cal, rooms, release, due, admissible
end

"""
    _benchmark_lpt_assignment(duration, admissible, n_or_days)

Longest processing time first, each case to its least-loaded admissible block.
"""
function _benchmark_lpt_assignment(
    duration::Vector{Float64}, admissible::Vector{Vector{Int}}, n_or_days::Int
)
    load = zeros(Float64, n_or_days)
    assignment = zeros(Int, length(duration))
    for i in sortperm(duration; rev=true)
        q = admissible[i][argmin([load[q] for q in admissible[i]])]
        assignment[i] = q
        load[q] += duration[i]
    end
    return assignment, load
end

"""
    LeeftinkHansORSchedulingProblem(target_variables, feasibility_status, seed)

Iterates the OR-day count (cases follow from the load factor; ~10 admissible
blocks plus at most one cancellation column per case) until the exact variable
count `sum |admissible| + #non-mandatory + n_or_days` is within 2% of the
target, keeping the closest instance.
"""
function LeeftinkHansORSchedulingProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 20)
    requested_load = if feasibility_status == infeasible
        rand(rng, _ORSCHED_BENCHMARK_LOADS[6:end])
    else
        _orsched_load_target(rng)
    end

    # About ten admissible blocks per case plus one cancellation column; a
    # case averages ~140 minutes.
    n_or_days = max(1, round(Int, target / 11 * 140 / (480 * requested_load)))
    best = nothing
    best_gap = Inf
    for _ in 1:20
        # Smaller suites run fewer services (about one per three OR-days).
        allowed = _orsched_case_mix(rng, clamp(n_or_days ÷ 3, 2, length(_ORSCHED_SPECIALTIES)))
        cases = _benchmark_case_list(rng, n_or_days, requested_load, allowed)
        blocks = _benchmark_blocks(rng, cases.specs, cases.means, n_or_days)
        q_actual = length(blocks[1])
        n = length(cases.means)
        optional_est = feasibility_status == unknown ? round(Int, 0.2 * n) : 0
        variables = sum(length, blocks[7]) + optional_est + q_actual
        gap = abs(variables - target) / target
        if gap < best_gap
            best_gap = gap
            best = (cases=cases, blocks=blocks)
        end
        gap <= 0.02 && break
        next_q = max(1, round(Int, q_actual * target / max(variables, 1)))
        next_q == n_or_days && (next_q = variables < target ? n_or_days + 1 : max(1, n_or_days - 1))
        n_or_days = next_q
    end

    cases = best.cases
    or_spec, or_cal, n_cal, rooms, release, due, admissible = best.blocks
    n_or_days = length(or_spec)
    expected = cases.means
    n_surgeries = length(expected)
    achieved_load = sum(expected) / (480.0 * n_or_days)
    cancellation_cost = [rand(rng, Uniform(800.0, 2400.0)) for _ in 1:n_surgeries]
    overtime_cost = [rand(rng, Uniform(3.0, 8.0)) for _ in 1:n_or_days]
    mandatory = falses(n_surgeries)
    max_overtime = fill(60.0, n_or_days)
    witness = nothing
    excess = nothing

    if feasibility_status == feasible
        mandatory .= true
        assignment, room_load = _benchmark_lpt_assignment(expected, admissible, n_or_days)
        max_overtime = [max(0.0, ceil(room_load[q] - 480.0)) for q in 1:n_or_days]
        witness = assignment
    elseif feasibility_status == infeasible
        mandatory .= true
        max_overtime .= 0.0
        excess = sum(expected) - 480.0 * n_or_days
        @assert excess > 0
    else
        # Overtime is budgeted against a reference LPT plan scaled by a global
        # factor on either side of what that plan needs, and a large share of
        # the list is mandatory: feasible or not depending on how much better
        # than LPT the LP can pack the blocks.
        n_mandatory = clamp(
            round(Int, rand(rng, Uniform(0.6, 1.0)) * n_surgeries), 0, n_surgeries
        )
        n_mandatory > 0 && (mandatory[shuffle(rng, 1:n_surgeries)[1:n_mandatory]] .= true)
        _, room_load = _benchmark_lpt_assignment(expected, admissible, n_or_days)
        budget = rand(rng, Uniform(0.3, 1.3))
        max_overtime = [max(0.0, ceil(budget * (room_load[q] - 480.0))) for q in 1:n_or_days]
    end

    scale = n_or_days in _ORSCHED_BENCHMARK_OR_DAYS ? :published_or_days : :scaled_or_days
    return LeeftinkHansORSchedulingProblem(
        n_surgeries,
        n_or_days,
        n_cal,
        rooms,
        480.0,
        requested_load,
        achieved_load,
        scale,
        cases.codes,
        cases.ids,
        cases.mus,
        cases.sigmas,
        cases.gammas,
        expected,
        cases.realized,
        [_ORSCHED_SPECIALTIES[k].code for k in or_spec],
        or_cal,
        release,
        due,
        admissible,
        cancellation_cost,
        overtime_cost,
        max_overtime,
        mandatory,
        witness,
        excess,
        feasibility_status,
    )
end

function build_model(prob::LeeftinkHansORSchedulingProblem)
    model = Model()
    N, Q = prob.n_surgeries, prob.n_or_days
    @variable(model, assign[i = 1:N, q = prob.admissible[i]], Bin)
    optional = [i for i in 1:N if !prob.mandatory[i]]
    @variable(model, cancel[optional], Bin)
    @variable(model, 0 <= overtime[q = 1:Q] <= prob.max_overtime[q])

    for i in 1:N
        lhs = sum(assign[i, q] for q in prob.admissible[i]; init=AffExpr(0.0))
        prob.mandatory[i] || (lhs += cancel[i])
        @constraint(model, lhs == 1)
    end
    cases_in = [Int[] for _ in 1:Q]
    for i in 1:N, q in prob.admissible[i]
        push!(cases_in[q], i)
    end
    for q in 1:Q
        @constraint(
            model,
            sum(prob.expected_duration[i] * assign[i, q] for i in cases_in[q]; init=AffExpr(0.0)) -
            overtime[q] <= prob.session_length
        )
    end

    @objective(
        model,
        Min,
        sum(prob.cancellation_cost[i] * cancel[i] for i in optional; init=0.0) +
            sum(prob.overtime_cost[q] * overtime[q] for q in 1:Q)
    )
    return model
end

register_variant(
    :operating_room_scheduling,
    :benchmark_loading,
    LeeftinkHansORSchedulingProblem,
    "Leeftink--Hans benchmark-informed 480-minute OR-day loading with empirical three-parameter-lognormal surgery types, calibrated 0.80--1.20 load, and sparse specialty-block scheduling windows",
)
