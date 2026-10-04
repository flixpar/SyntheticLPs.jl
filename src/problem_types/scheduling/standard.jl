using JuMP
using Random
using Distributions

"""
Planted roster: `assignment[w]` lists the `(day, template, department)` shifts
worker `w` works. It respects every worker rule of the model (one shift per
day, contract hours per week, consecutive-day limit, quick-return rest rule),
and the staffing requirements are derived from the coverage it provides, so
it is a 0/1 feasible point.
"""
struct ScheduleRosterWitness
    assignment::Vector{Vector{NTuple{3, Int}}}
end

"""
Department-week staffing certificate. Department `department`'s coverage rows
over the days of week `week` require `required` productive shifts in total.
Worker `w` can contribute at most `efficiency[w, m] · cap_w` of them, where
`cap_w = min(available days in the week, max_hours[w] / shortest shift)` follows
from the worker's one-shift-per-day and weekly-hours rows (valid in the LP
relaxation). `workers`/`caps` list the eligible workers and their caps;
`available = Σ efficiency · cap < required`. Each single coverage row stays
satisfiable, so only the aggregate (coverage rows of the week plus every
eligible worker's rows) refutes the instance.
"""
struct DepartmentWeekCertificate
    department::Int
    week::Int
    workers::Vector{Int}
    caps::Vector{Float64}
    required::Float64
    available::Float64
end

"""
    SchedulingProblem <: ProblemGenerator

Multi-department staff rostering with cross-training, contract hours, and rest
rules (retail stores, hospitality, hospital support services, contact centers).

# Overview

`n_workers` named employees are rostered over `n_days` days (whole weeks) onto
shift templates (early, day, late, night, short evening; different lengths) in
`n_departments` departments. Each worker has a home department (full
productivity) and may be cross-trained for one or two others at reduced
efficiency, a full- or part-time contract (weekly hour band), a set of shift
templates they can work, and requested days off. Columns: `x[w, d, k, m] ∈ {0,1}`
for every available (worker, day, template) and eligible department.

Rows:

  - coverage, per day, template, and department:
    `Σ_w efficiency[w,m] x[w,d,k,m] >= requirement[d,k,m]` (productivity-weighted);
  - one shift per worker per day;
  - contract hours, per worker and week: `min_hours[w] <= Σ length[k] x <= max_hours[w]`
    (a ranged row);
  - consecutive working days: every window of `max_consecutive + 1` days holds
    at most `max_consecutive` shifts;
  - quick-return rest rule: a closing shift (late/night) on day `d` and an
    opening shift on day `d + 1` cannot both be worked.

Objective: wage cost (hourly wage × hours × night/weekend premiums) minus a
small preference bonus for home-department work.

Compared with `nurse_scheduling` (a single ward with skill mix and night/weekend
fairness) and `workforce_shift_scheduling` (anonymous pools covering demand
intervals), the structure here is departmental cross-training with
productivity-weighted coverage, hour-banded contracts, and rest rules.

# Feasibility control

  - `feasible`: a roster is planted first (each worker's working days, shifts,
    and departments respect every rule), and requirements are 80–100% of the
    productivity it supplies per slot — [`ScheduleRosterWitness`](@ref).
  - `infeasible`: a peak week (holiday trading, an audit) in one department
    raises requirements so the department needs 10–35% more productive shifts
    than its eligible staff can supply — [`DepartmentWeekCertificate`](@ref);
    the surge is spread so no single coverage row is unattainable.
  - `unknown`: the same surge with ratio `1 ± U(0.03, 0.30)`.

# Fields

  - `n_workers`, `n_days`, `n_departments`, `n_templates::Int`
  - `template_length::Vector{Float64}`, `template_start::Vector{Int}` (hour of day)
  - `closing::Vector{Bool}`, `opening::Vector{Bool}`: rest-rule template classes
  - `home::Vector{Int}`, `departments::Vector{Vector{Int}}`, `efficiency::Matrix{Float64}`
  - `templates::Vector{Vector{Int}}`: templates each worker can work
  - `available::Matrix{Bool}`: `n_workers × n_days`
  - `min_hours`, `max_hours`, `wage::Vector{Float64}`, `max_consecutive::Int`
  - `requirement::Array{Float64,3}`: `n_days × n_templates × n_departments`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct SchedulingProblem <: ProblemGenerator
    n_workers::Int
    n_days::Int
    n_departments::Int
    n_templates::Int
    template_length::Vector{Float64}
    template_start::Vector{Int}
    closing::Vector{Bool}
    opening::Vector{Bool}
    home::Vector{Int}
    departments::Vector{Vector{Int}}
    efficiency::Matrix{Float64}
    templates::Vector{Vector{Int}}
    available::Matrix{Bool}
    min_hours::Vector{Float64}
    max_hours::Vector{Float64}
    wage::Vector{Float64}
    max_consecutive::Int
    requirement::Array{Float64, 3}
    feasible_witness::Union{Nothing, ScheduleRosterWitness}
    infeasibility_certificate::Union{Nothing, DepartmentWeekCertificate}
    feasibility_status::FeasibilityStatus
end

# (name, start hour, length, closing, opening)
const _SCHEDULING_TEMPLATES = (
    (:early, 6, 8.0, false, true),
    (:day, 9, 8.0, false, true),
    (:late, 14, 8.0, true, false),
    (:night, 22, 10.0, true, true),
    (:evening, 17, 4.0, false, false),
)

"""
    _scheduling_columns(prob) -> Vector{NTuple{4,Int}}

Model columns `(worker, day, template, department)` in worker / day / template
/ department order.
"""
function _scheduling_columns(prob::SchedulingProblem)
    cols = NTuple{4, Int}[]
    for w in 1:prob.n_workers, d in 1:prob.n_days
        prob.available[w, d] || continue
        for k in prob.templates[w], m in prob.departments[w]
            push!(cols, (w, d, k, m))
        end
    end
    return cols
end

"""
    _scheduling_week_cap(prob, w, week) -> Float64

LP upper bound on the shifts worker `w` can work in `week`: available days, and
weekly maximum hours over the worker's shortest template.
"""
function _scheduling_week_cap(prob::SchedulingProblem, w::Int, week::Int)
    days = ((week - 1) * 7 + 1):min(week * 7, prob.n_days)
    avail = count(d -> prob.available[w, d], days)
    shortest = minimum(prob.template_length[k] for k in prob.templates[w])
    return min(Float64(avail), prob.max_hours[w] / shortest)
end

"""
    SchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a rostering instance with about `target_variables` columns (workers
are added until the budget is used).
"""
function SchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    rng = MersenneTwister(seed)
    target = max(target_variables, 20)

    weeks = target <= 300 ? 1 : (target <= 5_000 ? rand(rng, 1:2) : rand(rng, 2:4))
    D = 7 * weeks
    profile = rand(rng, (:retail, :hospital_support, :contact_center))
    # Template mix per profile.
    tmpl_ids = if profile == :retail
        [1, 2, 3, 5]
    elseif profile == :hospital_support
        [1, 3, 4]
    else
        [1, 2, 3, 4, 5]
    end
    K = length(tmpl_ids)
    template_length = [_SCHEDULING_TEMPLATES[i][3] for i in tmpl_ids]
    template_start = [_SCHEDULING_TEMPLATES[i][2] for i in tmpl_ids]
    closing = [_SCHEDULING_TEMPLATES[i][4] for i in tmpl_ids]
    opening = [_SCHEDULING_TEMPLATES[i][5] for i in tmpl_ids]
    night_k = [i == 4 for i in tmpl_ids]
    max_consecutive = rand(rng, 5:6)

    # Departments sized so each has a crew of ~8-25 workers.
    per_worker_cols = D * 0.85 * (K * 0.7) * 1.4
    est_workers = max(3, round(Int, target / per_worker_cols))
    M = max(1, round(Int, est_workers / rand(rng, 8:25)))

    # --- Workers until the column budget is used --------------------------------
    home = Int[]
    departments = Vector{Int}[]
    eff_rows = Vector{Float64}[]
    templates = Vector{Int}[]
    avail_rows = Vector{Bool}[]
    min_hours, max_hours, wage = Float64[], Float64[], Float64[]
    cols = 0
    while cols < target || length(home) < 2
        h = rand(rng, 1:M)
        deps = [h]
        if M > 1
            for _ in 1:(rand(rng) < 0.5 ? (rand(rng) < 0.3 ? 2 : 1) : 0)
                o = rand(rng, 1:M)
                o in deps || push!(deps, o)
            end
        end
        sort!(deps)
        productivity = rand(rng, LogNormal(0.0, 0.12))
        eff = zeros(M)
        for m in deps
            eff[m] = round(productivity * (m == h ? 1.0 : rand(rng, Uniform(0.6, 0.9))); digits=3)
        end
        full_time = rand(rng) < 0.6
        ks = full_time ? [k for k in 1:K if rand(rng) < 0.8] : [k for k in 1:K if rand(rng) < 0.5]
        isempty(ks) && (ks = [rand(rng, 1:K)])
        av = [rand(rng) > (full_time ? 0.12 : 0.3) for _ in 1:D]
        sum(av) == 0 && (av[rand(rng, 1:D)] = true)
        push!(home, h)
        push!(departments, deps)
        push!(eff_rows, eff)
        push!(templates, ks)
        push!(avail_rows, av)
        push!(max_hours, full_time ? 40.0 : rand(rng, [16.0, 20.0, 24.0]))
        push!(min_hours, full_time ? 32.0 : 8.0)
        push!(wage, rand(rng, LogNormal(log(full_time ? 22.0 : 17.0), 0.15)))
        cols += sum(av) * length(ks) * length(deps)
    end
    W = length(home)
    efficiency = permutedims(reduce(hcat, eff_rows))
    available = permutedims(reduce(hcat, avail_rows))

    # --- Planted roster ------------------------------------------------------------
    # Workers in random order pick days greedily (respecting availability, the
    # consecutive-day limit, weekly hours, and the quick-return rule), with a
    # target weekly load inside their contract band; mostly home department.
    assignment = [NTuple{3, Int}[] for _ in 1:W]
    supply = zeros(D, K, M)
    for w in shuffle(rng, 1:W)
        worked = falses(D)
        last_k = 0
        for wk in 1:weeks
            days = ((wk - 1) * 7 + 1):(wk * 7)
            goal = rand(rng, Uniform(min_hours[w], max_hours[w]))
            hours = 0.0
            for d in shuffle(rng, collect(days))
                available[w, d] || continue
                # Consecutive-day limit: the run through d must stay short.
                run_lo = d
                while run_lo > 1 && worked[run_lo - 1]
                    run_lo -= 1
                end
                run_hi = d
                while run_hi < D && worked[run_hi + 1]
                    run_hi += 1
                end
                run_hi - run_lo + 1 > max_consecutive && continue
                # Quick-return rule against neighbours already worked.
                prev = d > 1 && worked[d - 1] ? assignment[w][findfirst(a -> a[1] == d - 1, assignment[w])][2] : 0
                nxt = d < D && worked[d + 1] ? assignment[w][findfirst(a -> a[1] == d + 1, assignment[w])][2] : 0
                options = [
                    k for k in templates[w] if hours + template_length[k] <= goal + 1e-9 &&
                    !(prev > 0 && closing[prev] && opening[k]) && !(nxt > 0 && closing[k] && opening[nxt])
                ]
                isempty(options) && continue
                k = rand(rng, options)
                m = rand(rng) < 0.8 ? home[w] : rand(rng, departments[w])
                push!(assignment[w], (d, k, m))
                worked[d] = true
                hours += template_length[k]
                supply[d, k, m] += efficiency[w, m]
            end
            # Contract floor: lower it to what the planted roster achieves when
            # availability made the band unreachable (a short-hours week).
            if hours < min_hours[w]
                min_hours[w] = min(min_hours[w], hours)
            end
        end
        sort!(assignment[w])
    end
    requirement = zeros(D, K, M)
    for d in 1:D, k in 1:K, m in 1:M
        supply[d, k, m] > 0 || continue
        requirement[d, k, m] = floor(supply[d, k, m] * rand(rng, Uniform(0.8, 1.0)) * 100) / 100
    end

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        witness = ScheduleRosterWitness(assignment)
    else
        tmp = SchedulingProblem(
            W, D, M, K, template_length, template_start, closing, opening, home, departments,
            efficiency, templates, available, min_hours, max_hours, wage, max_consecutive,
            requirement, nothing, nothing, feasibility_status,
        )
        # Department-week aggregate: requirement vs. eligible productive capacity.
        best = (0, 0, -1.0, 0.0, 0.0)
        for m in 1:M, wk in 1:weeks
            elig = [w for w in 1:W if efficiency[w, m] > 0]
            isempty(elig) && continue
            avail_cap = sum(efficiency[w, m] * _scheduling_week_cap(tmp, w, wk) for w in elig)
            days = ((wk - 1) * 7 + 1):(wk * 7)
            req = sum(requirement[d, k, m] for d in days, k in 1:K)
            req / avail_cap > best[3] && (best = (m, wk, req / avail_cap, req, avail_cap))
        end
        m, wk, _, req, avail_cap = best
        ratio = _scheduling_ratio(rng, feasibility_status)
        # Spread the surge over the week's slots in proportion to each slot's
        # own attainable coverage (never above 90% of it).
        days = ((wk - 1) * 7 + 1):(wk * 7)
        slot_cap = zeros(D, K)
        for w in 1:W
            efficiency[w, m] > 0 || continue
            for d in days, k in templates[w]
                available[w, d] && (slot_cap[d, k] += efficiency[w, m])
            end
        end
        goal = ratio * avail_cap
        extra = goal - req
        room = sum(max(0.0, 0.9 * slot_cap[d, k] - requirement[d, k, m]) for d in days, k in 1:K)
        frac = room > 0 ? min(1.0, max(extra, 0.0) / room) : 0.0
        for d in days, k in 1:K
            r = requirement[d, k, m]
            if extra >= 0
                requirement[d, k, m] = r + frac * max(0.0, 0.9 * slot_cap[d, k] - r)
            else
                requirement[d, k, m] = r * goal / req
            end
        end
        if feasibility_status == infeasible
            elig = [w for w in 1:W if efficiency[w, m] > 0]
            caps = [_scheduling_week_cap(tmp, w, wk) for w in elig]
            certificate = DepartmentWeekCertificate(
                m,
                wk,
                elig,
                caps,
                sum(requirement[d, k, m] for d in days, k in 1:K),
                sum(efficiency[w, m] * c for (w, c) in zip(elig, caps)),
            )
        end
    end

    return SchedulingProblem(
        W,
        D,
        M,
        K,
        template_length,
        template_start,
        closing,
        opening,
        home,
        departments,
        efficiency,
        templates,
        available,
        min_hours,
        max_hours,
        wage,
        max_consecutive,
        requirement,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    _scheduling_ratio(rng, status) -> Float64

`1.10–1.35` for `infeasible`; `1 ± U(0.03, 0.30)` for `unknown`.
"""
function _scheduling_ratio(rng::AbstractRNG, status::FeasibilityStatus)
    status == infeasible && return 1.1 + 0.25 * rand(rng)
    m = 0.03 + 0.27 * rand(rng)
    return rand(rng) < 0.5 ? 1.0 - m : 1.0 + m
end

"""
    build_model(prob::SchedulingProblem)

Build the rostering model (binary `x`, one per column of
[`_scheduling_columns`](@ref)). Deterministic.
"""
function build_model(prob::SchedulingProblem)
    model = Model()
    cols = _scheduling_columns(prob)
    n = length(cols)
    @variable(model, x[1:n], Bin)

    D, K, M = prob.n_days, prob.n_templates, prob.n_departments
    slot_cols = [Int[] for _ in 1:(D * K * M)]
    slot_index(d, k, m) = ((d - 1) * K + (k - 1)) * M + m
    worker_day = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (c, (w, d, k, m)) in enumerate(cols)
        push!(slot_cols[slot_index(d, k, m)], c)
        push!(get!(worker_day, (w, d), Int[]), c)
    end

    # Coverage.
    for d in 1:D, k in 1:K, m in 1:M
        req = prob.requirement[d, k, m]
        req > 0 || continue
        cs = slot_cols[slot_index(d, k, m)]
        @constraint(model, sum(prob.efficiency[cols[c][1], m] * x[c] for c in cs) >= req)
    end

    weeks = cld(D, 7)
    for w in 1:prob.n_workers
        # One shift per day (rows with a single column are just the 0/1 bound).
        for d in 1:D
            cs = get(worker_day, (w, d), Int[])
            length(cs) > 1 && @constraint(model, sum(x[c] for c in cs) <= 1)
        end
        # Contract hours per week (ranged).
        for wk in 1:weeks
            cs = reduce(
                vcat, [get(worker_day, (w, d), Int[]) for d in ((wk - 1) * 7 + 1):min(wk * 7, D)]; init=Int[]
            )
            isempty(cs) && continue
            expr = @expression(model, sum(prob.template_length[cols[c][3]] * x[c] for c in cs))
            if prob.min_hours[w] > 0
                @constraint(model, prob.min_hours[w] <= expr <= prob.max_hours[w])
            else
                @constraint(model, expr <= prob.max_hours[w])
            end
        end
        # Consecutive working days.
        c_max = prob.max_consecutive
        for start in 1:(D - c_max)
            window = start:(start + c_max)
            count(d -> haskey(worker_day, (w, d)), window) > c_max || continue
            cs = reduce(vcat, [get(worker_day, (w, d), Int[]) for d in window])
            @constraint(model, sum(x[c] for c in cs) <= c_max)
        end
        # Quick-return rest rule.
        for d in 1:(D - 1)
            close_cs = [c for c in get(worker_day, (w, d), Int[]) if prob.closing[cols[c][3]]]
            open_cs = [c for c in get(worker_day, (w, d + 1), Int[]) if prob.opening[cols[c][3]]]
            (isempty(close_cs) || isempty(open_cs)) && continue
            @constraint(model, sum(x[c] for c in close_cs) + sum(x[c] for c in open_cs) <= 1)
        end
    end

    # Wage cost with night (k start 22h) and weekend premiums, minus a small
    # home-department preference bonus.
    obj = AffExpr(0.0)
    for (c, (w, d, k, m)) in enumerate(cols)
        premium = (prob.template_start[k] >= 22 ? 0.25 : 0.0) + (mod1(d, 7) >= 6 ? 0.2 : 0.0)
        coef = prob.wage[w] * prob.template_length[k] * (1 + premium) - (m == prob.home[w] ? 5.0 : 0.0)
        add_to_expression!(obj, coef, x[c])
    end
    @objective(model, Min, obj)
    return model
end

register_variant(
    :scheduling,
    :standard,
    SchedulingProblem,
    "Multi-department staff rostering with cross-training efficiencies, shift templates, ranged weekly contract hours, consecutive-day and quick-return rest rules, a planted roster, and a department-week staffing certificate";
    tags=[:scheduling, :block_angular, :covering, :packing],
)
