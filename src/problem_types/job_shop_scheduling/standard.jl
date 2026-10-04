using JuMP
using Random
using Distributions

"""
Planted schedule: an integral start slot for every operation (a genuine 0/1
point of the time-indexed model — `x[o, start[o]] = 1`) built by a serial
list-scheduling heuristic that respects releases, job routings, and
work-center capacities. `completion[j]` is the last slot job `j` occupies.
"""
struct JobShopScheduleWitness
    start::Vector{Int}
    completion::Vector{Int}
end

"""
Energetic (interval-load) infeasibility certificate. Every operation in
`operations` runs on work center `work_center` and its whole start window lies
in `[interval_start, interval_end - p_o + 1]`, so wherever it starts it occupies
`p_o` slots of the interval. Summing the work center's capacity rows over the
interval gives `Σ_o p_o Σ_s x[o,s] <= capacity * (interval_end - interval_start + 1)`
for those operations, and their assignment rows force `Σ_s x[o,s] = 1`, so the
interval must absorb `required_load = Σ p_o` slot-units of work while only
`available_capacity` exist. The argument uses LP rows only — it refutes the
LP relaxation, not just the integer program — and needs
`interval length + |operations|` rows at once, so presolve does not see it.
"""
struct JobShopEnergyCertificate
    work_center::Int
    interval_start::Int
    interval_end::Int
    operations::Vector{Int}
    required_load::Int
    available_capacity::Int
end

"""
    JobShopSchedulingProblem <: ProblemGenerator

Time-indexed job shop scheduling with parallel-machine work centers, release
dates, hard deadlines, and weighted tardiness.

# Overview

Jobs arrive over time (a dynamic shop) and each follows a routing through a
subset of work centers — flow-dominant, with occasional out-of-order visits.
Work center `w` holds `capacity[w]` identical machines (mostly 1). Time is
discretised into slots; operation `o` needs `proc[o]` consecutive slots.

The formulation is the classical time-indexed (Pritsker–Watters–Wolfe)
model:

  - `x[o, s] ∈ {0,1}` — operation `o` starts in slot `s`, for every slot of its
    start window `[earliest[o], latest[o]]` (release plus routing head; job
    deadline minus routing tail);
  - assignment rows `Σ_s x[o,s] = 1`;
  - work-center capacity rows, one per work center and slot `t`:
    `Σ_{o at w} Σ_{s = t - proc[o] + 1}^{t} x[o,s] <= capacity[w]` (rows that
    can never bind — at most `capacity[w]` operations could be running — are
    omitted);
  - routing precedence between consecutive operations of a job:
    `Σ_s s·x[o',s] - Σ_s s·x[o,s] >= proc[o]`.

Objective: weighted tardiness of each job's completion against its due date
(a piecewise-linear cost placed directly on the last operation's start
columns) plus a small work-in-process cost on every operation's start time.

Unlike the big-M disjunctive formulation this replaces — whose LP relaxation
collapses to the no-contention closed form — the time-indexed relaxation keeps
machine contention: the capacity rows bind wherever several operations compete
for a slot, so the relaxed optimum sits strictly above the no-contention bound.

# Data grounding

Releases follow a Poisson arrival stream whose rate puts the bottleneck work
center at 75–92% utilisation; processing times are lognormal around
work-center-specific means (a few slow bottleneck centers); due dates use the
total-work-content rule `release + F × total processing` with `F ∈ [1.3, 3.5]`;
job weights come from priority classes. Start windows come from a planted
list schedule (queueing delays) plus deadline slack, so window widths vary
across jobs the way real flow allowances do.

# Feasibility control

  - `feasible`: job deadlines are at or beyond the planted schedule's
    completions, so the planted schedule (a [`JobShopScheduleWitness`](@ref))
    is a 0/1 point of the model.
  - `infeasible`: a batch of expedited orders (3–8 jobs) that all visit the
    bottleneck work center gets releases and hard deadlines confining their
    bottleneck operations to one interval whose capacity is 10–35% short of
    their total processing time — certified by a
    [`JobShopEnergyCertificate`](@ref).
  - `unknown`: the same expedited batch with its load-to-capacity ratio drawn
    from `1 ± U(0.03, 0.30)`: above 1 provably infeasible, below 1 decided by
    how the batch interacts with the rest of the shop.

# Fields

  - `n_jobs::Int`, `n_work_centers::Int`, `n_ops::Int`
  - `capacity::Vector{Int}`: machines per work center
  - `job_ops::Vector{Vector{Int}}`: operations of each job, in routing order
  - `op_job::Vector{Int}`, `op_work_center::Vector{Int}`, `proc::Vector{Int}`
  - `release::Vector{Int}`, `deadline::Vector{Int}`, `due::Vector{Int}`: per job (slots)
  - `weight::Vector{Float64}`: tardiness weight per job
  - `wip_cost::Float64`: per-slot work-in-process cost factor
  - `earliest::Vector{Int}`, `latest::Vector{Int}`: start window per operation
  - `horizon::Int`: last slot any operation can occupy
  - `expedited_jobs::Vector{Int}`: the expedited batch (`infeasible`/`unknown`)
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct JobShopSchedulingProblem <: ProblemGenerator
    n_jobs::Int
    n_work_centers::Int
    n_ops::Int
    capacity::Vector{Int}
    job_ops::Vector{Vector{Int}}
    op_job::Vector{Int}
    op_work_center::Vector{Int}
    proc::Vector{Int}
    release::Vector{Int}
    deadline::Vector{Int}
    due::Vector{Int}
    weight::Vector{Float64}
    wip_cost::Float64
    earliest::Vector{Int}
    latest::Vector{Int}
    horizon::Int
    expedited_jobs::Vector{Int}
    feasible_witness::Union{Nothing, JobShopScheduleWitness}
    infeasibility_certificate::Union{Nothing, JobShopEnergyCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _job_shop_place!(usage, w, cap, earliest, p) -> Int

Earliest slot `s >= earliest` such that work center `w` has a free machine in
every slot `s:(s + p - 1)`, booking it. `usage[w]` grows as needed.
"""
function _job_shop_place!(usage::Vector{Vector{Int}}, w::Int, cap::Int, earliest::Int, p::Int)
    u = usage[w]
    s = earliest
    while true
        need = s + p - 1
        length(u) < need && append!(u, zeros(Int, need - length(u) + 64))
        blocked = 0
        for t in s:need
            if u[t] >= cap
                blocked = t
            end
        end
        if blocked == 0
            for t in s:need
                u[t] += 1
            end
            return s
        end
        s = blocked + 1
    end
end

"""
    _job_shop_route(rng, n_wc, n_ops, order) -> Vector{Int}

A routing of `n_ops` distinct work centers: a random subset visited in the
shop's dominant flow `order`, with one adjacent swap 30% of the time.
"""
function _job_shop_route(rng::AbstractRNG, n_wc::Int, n_ops::Int, rank::Vector{Int})
    chosen = shuffle(rng, 1:n_wc)[1:n_ops]
    sort!(chosen; by=w -> rank[w])
    if n_ops >= 2 && rand(rng) < 0.3
        i = rand(rng, 1:(n_ops - 1))
        chosen[i], chosen[i + 1] = chosen[i + 1], chosen[i]
    end
    return chosen
end

"""
    JobShopSchedulingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a time-indexed job shop instance whose start-variable count is within
a few columns of `target_variables` (targets below ~60 round up to a minimal
three-job shop).
"""
function JobShopSchedulingProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    rng = MersenneTwister(seed)
    target = target_variables

    # --- Shop layout --------------------------------------------------------
    n_wc = if target <= 2_000
        rand(rng, 4:7)
    elseif target <= 20_000
        rand(rng, 6:12)
    else
        rand(rng, 10:20)
    end
    capacity = [rand(rng) < 0.75 ? 1 : rand(rng, 2:3) for _ in 1:n_wc]
    mean_proc = [rand(rng, LogNormal(log(3.0), 0.35)) for _ in 1:n_wc]
    for w in shuffle(rng, 1:n_wc)[1:max(1, n_wc ÷ 5)]
        mean_proc[w] *= 1.6                   # slow bottleneck centers
    end
    rank = randperm(rng, n_wc)                # dominant flow order
    ops_lo = min(2, n_wc)
    ops_hi = min(n_wc, target <= 2_000 ? 5 : 8)
    # Mean start-window width (slots) at this scale.
    mean_window = target <= 2_000 ? rand(rng, 8.0:1.0:16.0) : rand(rng, 14.0:1.0:30.0)

    # Arrival rate: bottleneck utilisation rho. Expected load per job on w is
    # P(visit w) * E[proc at w]; visits are uniform over work centers.
    mean_ops = (ops_lo + ops_hi) / 2
    rho = 0.75 + 0.17 * rand(rng)
    load_per_job = [mean_ops / n_wc * mean_proc[w] / capacity[w] for w in 1:n_wc]
    rate = rho / maximum(load_per_job)

    # --- Jobs, generated and list-scheduled in arrival order ------------------
    usage = [zeros(Int, 256) for _ in 1:n_wc]
    job_ops = Vector{Int}[]
    op_job, op_wc, proc, plant_start = Int[], Int[], Int[], Int[]
    release, completion = Int[], Int[]
    clock = 1.0
    base = 0                                  # Σ_ops (planted wait + 1)
    n_target_ops = max(6, round(Int, target / mean_window))
    # At least n_wc jobs: with >= 2 distinct operations each, some work center
    # is visited twice (needed by the expedited batch below).
    while (length(proc) < n_target_ops && base < 0.85 * target) || length(job_ops) < max(3, n_wc)
        clock += rand(rng, Exponential(1 / rate))
        r = floor(Int, clock)
        k = rand(rng, ops_lo:ops_hi)
        route = _job_shop_route(rng, n_wc, k, rank)
        ops = Int[]
        t = r
        for w in route
            p = clamp(round(Int, rand(rng, LogNormal(log(mean_proc[w]), 0.4))), 1, 12)
            s = _job_shop_place!(usage, w, capacity[w], t, p)
            push!(op_job, length(job_ops) + 1)
            push!(op_wc, w)
            push!(proc, p)
            push!(plant_start, s)
            push!(ops, length(proc))
            t = s + p
        end
        push!(job_ops, ops)
        push!(release, r)
        push!(completion, t - 1)
        total_p = sum(proc[o] for o in ops)
        base += k * (t - r - total_p + 1)
    end
    n_jobs = length(job_ops)
    n_ops = length(proc)
    total_proc = [sum(proc[o] for o in job_ops[j]) for j in 1:n_jobs]

    # Due dates (total-work-content rule) and priority weights.
    due = [release[j] + round(Int, total_proc[j] * (1.3 + 2.2 * rand(rng))) for j in 1:n_jobs]
    weight = [rand(rng) < 0.15 ? 4.0 : (rand(rng) < 0.35 ? 2.0 : 1.0) for _ in 1:n_jobs]
    weight .*= 1.0 .+ 0.1 .* rand(rng, n_jobs)
    wip_cost = 0.02 + 0.03 * rand(rng)

    # Deadlines start at the planted completions (float = planted waiting).
    deadline = copy(completion)

    # --- Expedited batch (infeasible / unknown) ------------------------------
    expedited = Int[]
    certificate = nothing
    cert_parts = nothing
    if feasibility_status != feasible
        ratio = if feasibility_status == infeasible
            1.1 + 0.25 * rand(rng)
        else
            m = 0.03 + 0.27 * rand(rng)
            rand(rng) < 0.5 ? 1.0 - m : 1.0 + m
        end
        # Bottleneck: the most loaded work center visited by >= 2 jobs (one
        # exists: there are at least n_wc jobs of >= 2 distinct operations).
        visitors_of(w) = [j for j in 1:n_jobs if any(op_wc[o] == w for o in job_ops[j])]
        util = [sum(proc[o] for o in 1:n_ops if op_wc[o] == w; init=0) / capacity[w] for w in 1:n_wc]
        wstar = argmax(w -> length(visitors_of(w)) >= 2 ? util[w] : -1.0, 1:n_wc)
        visitors = visitors_of(wstar)
        # Batch size: 3-8 orders per machine, fewer on tiny instances so the
        # batch's windows do not swamp the column budget.
        k_batch = min(rand(rng, 3:8) * capacity[wstar], length(visitors), max(2, target ÷ 150))
        first_idx = rand(rng, 1:(length(visitors) - k_batch + 1))
        # A contiguous run of arrivals: expedited orders arrive together. The
        # batch grows while the interval cannot reach the drawn ratio (a long
        # operation must still fit), and a multi-machine center that still
        # cannot be over-booked loses machines to a breakdown.
        bottleneck_op(j) = first(o for o in job_ops[j] if op_wc[o] == wstar)
        batch_len(batch) = max(
            floor(Int, sum(proc[bottleneck_op(j)] for j in batch) / (capacity[wstar] * ratio)),
            maximum(proc[bottleneck_op(j)] for j in batch),
        )
        batch_ratio(batch) =
            sum(proc[bottleneck_op(j)] for j in batch) / (capacity[wstar] * batch_len(batch))
        expedited = visitors[first_idx:(first_idx + k_batch - 1)]
        target_ratio = feasibility_status == infeasible ? 1.08 : 0.0
        while batch_ratio(expedited) < target_ratio
            rest = setdiff(visitors, expedited)
            if !isempty(rest)
                push!(expedited, rest[1])
            elseif capacity[wstar] > 1
                capacity[wstar] -= 1
            else
                break  # unreachable: two unit-capacity visitors always over-book
            end
        end
        sort!(expedited)
        batch_ops = [bottleneck_op(j) for j in expedited]
        load = sum(proc[o] for o in batch_ops)
        len = batch_len(expedited)
        heads = Int[]
        tails = Int[]
        for (j, o) in zip(expedited, batch_ops)
            pos = findfirst(==(o), job_ops[j])
            push!(heads, sum(proc[q] for q in job_ops[j][1:(pos - 1)]; init=0))
            push!(tails, sum(proc[q] for q in job_ops[j][pos:end]))
        end
        # The interval opens when the batch's operations could first reach
        # the bottleneck.
        a = max(release[expedited[1]] + maximum(heads), maximum(heads) + 1)
        if feasibility_status == unknown
            # Size the interval against the bottleneck's *total* planted load
            # in it — the batch plus the ordinary jobs already booked there —
            # so the drawn ratio straddles the real boundary instead of being
            # swamped by background traffic.
            bg = copy(usage[wstar])
            for o in batch_ops, t in plant_start[o]:(plant_start[o] + proc[o] - 1)
                bg[t] -= 1
            end
            need(l) = (load + sum(bg[t] for t in a:(a + l - 1) if t <= length(bg); init=0)) /
                (capacity[wstar] * l)
            # Cap: the batch's windows may use at most half the column budget.
            batch_cols(l) = sum(
                length(job_ops[j]) * (l - proc[o] + 1) for (j, o) in zip(expedited, batch_ops)
            )
            len = maximum(proc[o] for o in batch_ops)
            while need(len) > ratio && batch_cols(len + 1) <= target ÷ 2
                len += 1
            end
        end
        b = a + len - 1
        for (i, (j, o)) in enumerate(zip(expedited, batch_ops))
            release[j] = a - heads[i]
            deadline[j] = b + tails[i] - proc[o]
            due[j] = deadline[j]          # expedited orders are due at their deadline
        end
        cert_parts = (wstar, a, b, batch_ops, load, capacity[wstar] * len)
    end

    # --- Fit the column budget ------------------------------------------------
    # Columns of job j: one per operation and slot of its common start-window
    # width (deadline - release - total processing + 2).
    window(j) = deadline[j] - release[j] - total_proc[j] + 2
    cols_now = sum(length(job_ops[j]) * window(j) for j in 1:n_jobs)
    # Already over budget (queueing delays or the expedited batch's interval):
    # drop the latest-arriving ordinary jobs. Removing a job only frees
    # capacity, so the planted schedule of the others stays valid.
    keep = trues(n_jobs)
    for j in n_jobs:-1:1
        cols_now <= target && break
        (j in expedited || count(keep) <= 3) && continue
        keep[j] = false
        cols_now -= length(job_ops[j]) * window(j)
    end
    if !all(keep)
        jobs = findall(keep)
        jmap = zeros(Int, n_jobs)
        jmap[jobs] = 1:length(jobs)
        old_ops = reduce(vcat, job_ops[jobs])
        omap = zeros(Int, n_ops)
        omap[old_ops] = 1:length(old_ops)
        job_ops = [omap[job_ops[j]] for j in jobs]
        op_job = jmap[op_job[old_ops]]
        op_wc = op_wc[old_ops]
        proc = proc[old_ops]
        plant_start = plant_start[old_ops]
        release = release[jobs]
        completion = completion[jobs]
        deadline = deadline[jobs]
        due = due[jobs]
        weight = weight[jobs]
        total_proc = total_proc[jobs]
        expedited = jmap[expedited]
        if cert_parts !== nothing
            cert_parts = (cert_parts[1:3]..., omap[cert_parts[4]], cert_parts[5:6]...)
        end
        n_jobs = length(jobs)
        n_ops = length(old_ops)
    end
    if feasibility_status == infeasible
        certificate = JobShopEnergyCertificate(cert_parts...)
    end

    # Deadline slack fills the remaining budget.
    remaining = target - cols_now
    free_jobs = [j for j in 1:n_jobs if !(j in expedited)]
    if remaining > 0 && !isempty(free_jobs)
        u = rand(rng, Exponential(1.0), length(free_jobs))
        mass = sum(u[i] * length(job_ops[j]) for (i, j) in enumerate(free_jobs))
        for (i, j) in enumerate(free_jobs)
            extra = floor(Int, remaining * u[i] / mass)
            deadline[j] += extra
            remaining -= extra * length(job_ops[j])
        end
        # Spend the remainder on the jobs that still fit, smallest first.
        for j in sort(free_jobs; by=j -> length(job_ops[j]))
            while remaining >= length(job_ops[j])
                deadline[j] += 1
                remaining -= length(job_ops[j])
            end
        end
    end

    # Start windows from the routing heads and tails.
    earliest = zeros(Int, n_ops)
    latest = zeros(Int, n_ops)
    for j in 1:n_jobs
        head = 0
        tail = total_proc[j]
        for o in job_ops[j]
            earliest[o] = release[j] + head
            latest[o] = deadline[j] - tail + 1
            head += proc[o]
            tail -= proc[o]
        end
    end
    horizon = maximum(deadline)

    witness = if feasibility_status == feasible
        JobShopScheduleWitness(copy(plant_start), copy(completion))
    else
        nothing
    end

    return JobShopSchedulingProblem(
        n_jobs,
        n_wc,
        n_ops,
        capacity,
        job_ops,
        op_job,
        op_wc,
        proc,
        release,
        deadline,
        due,
        weight,
        wip_cost,
        earliest,
        latest,
        horizon,
        expedited,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    _job_shop_columns(prob) -> (op, slot, first_col)

Column order: operations in index order, start slots ascending within each
window. `first_col[o]` is the index of `x[o, earliest[o]]`.
"""
function _job_shop_columns(prob::JobShopSchedulingProblem)
    first_col = zeros(Int, prob.n_ops + 1)
    n = 0
    for o in 1:prob.n_ops
        first_col[o] = n + 1
        n += prob.latest[o] - prob.earliest[o] + 1
    end
    first_col[end] = n + 1
    op = Vector{Int}(undef, n)
    slot = Vector{Int}(undef, n)
    for o in 1:prob.n_ops, (i, s) in enumerate(prob.earliest[o]:prob.latest[o])
        op[first_col[o] + i - 1] = o
        slot[first_col[o] + i - 1] = s
    end
    return op, slot, first_col
end

"""
    build_model(prob::JobShopSchedulingProblem)

Build the time-indexed job shop model (binary start variables). Deterministic.
"""
function build_model(prob::JobShopSchedulingProblem)
    model = Model()
    op, slot, first_col = _job_shop_columns(prob)
    n = length(op)
    @variable(model, x[1:n], Bin)

    # Assignment rows.
    for o in 1:prob.n_ops
        @constraint(model, sum(x[c] for c in first_col[o]:(first_col[o + 1] - 1)) == 1)
    end

    # Routing precedence (aggregated start times).
    for ops in prob.job_ops, i in 1:(length(ops) - 1)
        o, q = ops[i], ops[i + 1]
        @constraint(
            model,
            sum(slot[c] * x[c] for c in first_col[q]:(first_col[q + 1] - 1)) -
            sum(slot[c] * x[c] for c in first_col[o]:(first_col[o + 1] - 1)) >= prob.proc[o]
        )
    end

    # Work-center capacity rows: column c occupies slots slot[c]:(slot[c]+p-1).
    for w in 1:prob.n_work_centers
        ops_w = [o for o in 1:prob.n_ops if prob.op_work_center[o] == w]
        isempty(ops_w) && continue
        lo = minimum(prob.earliest[o] for o in ops_w)
        hi = maximum(prob.latest[o] + prob.proc[o] - 1 for o in ops_w)
        cols = [Int[] for _ in lo:hi]
        nops = zeros(Int, hi - lo + 1)
        for o in ops_w
            for t in prob.earliest[o]:(prob.latest[o] + prob.proc[o] - 1)
                nops[t - lo + 1] += 1
            end
            for c in first_col[o]:(first_col[o + 1] - 1), t in slot[c]:(slot[c] + prob.proc[o] - 1)
                push!(cols[t - lo + 1], c)
            end
        end
        for (i, cs) in enumerate(cols)
            # At most nops[i] operations can run in this slot; skip rows that
            # can never bind.
            nops[i] > prob.capacity[w] || continue
            @constraint(model, sum(x[c] for c in cs) <= prob.capacity[w])
        end
    end

    # Objective: weighted tardiness on each job's last operation plus a small
    # work-in-process cost on every start time.
    obj = zeros(n)
    for j in 1:prob.n_jobs
        ops = prob.job_ops[j]
        last = ops[end]
        for o in ops, c in first_col[o]:(first_col[o + 1] - 1)
            obj[c] += prob.wip_cost * prob.weight[j] * (slot[c] - prob.earliest[o])
        end
        for c in first_col[last]:(first_col[last + 1] - 1)
            finish = slot[c] + prob.proc[last] - 1
            obj[c] += prob.weight[j] * max(0, finish - prob.due[j])
        end
    end
    @objective(model, Min, sum(obj[c] * x[c] for c in 1:n if obj[c] != 0.0))
    return model
end

register_variant(
    :job_shop_scheduling,
    :standard,
    JobShopSchedulingProblem,
    "Time-indexed job shop scheduling with parallel-machine work centers, release dates, hard deadlines, and weighted tardiness; LP relaxation keeps machine contention, with a planted list schedule and an energetic interval-load infeasibility certificate",
)
