# Batch dataset generation for SyntheticLPs.jl
#
# This file provides a library-level API for generating whole *datasets* of LP
# instances (e.g. for training ML models), with optional solve-based quality
# filtering. It is the in-package counterpart to `scripts/generate_lps.jl`,
# which is now a thin command-line wrapper around `generate_dataset`.
#
# Design notes:
# - The package itself stays solver-agnostic. Quality filtering requires the
#   caller to pass an `optimizer` (e.g. `HiGHS.Optimizer`); without one, no
#   solving is performed and every successfully-built instance is kept.
# - All randomness flows from the master `seed`: it fixes the plan (variant,
#   status, and target size of every index) and each index's private RNG stream,
#   so a given `seed` reproduces the exact same dataset, in one process or
#   split across shards.

const MOI = JuMP.MOI

# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

"""
    QualityCriteria(; kwargs...)

Thresholds used by [`check_quality`](@ref) to decide whether a solved LP
instance is a good test/training instance.

# Keyword arguments

  - `solve_timeout::Float64 = 30.0`: per-instance solve time limit (seconds).
  - `min_constraints::Int = 5`: reject instances with fewer constraints.
  - `min_iterations::Int = 3`: reject instances solved in `≤` this many simplex
    iterations (trivially solved / solved in phase 1 only).
  - `max_iteration_ratio::Float64 = 100.0`: reject instances whose simplex
    iteration count exceeds `max_iteration_ratio × constraints` (likely
    degenerate / numerically nasty).
"""
Base.@kwdef struct QualityCriteria
    solve_timeout::Float64 = 30.0
    min_constraints::Int = 5
    min_iterations::Int = 3
    max_iteration_ratio::Float64 = 100.0
end

"""
    QualityResult

Outcome of a [`check_quality`](@ref) call.

# Fields

  - `passed::Bool`: whether the instance qualifies.
  - `reason::String`: `"passed"`, or the rejection reason (e.g. `"timeout"`,
    `"degenerate"`, `"too_few_iterations"`).
  - `iterations::Int`: simplex iterations reported by the solver (`-1` if
    unavailable or the instance was rejected before solving).
  - `solve_time::Float64`: wall-clock solve time in seconds (`0.0` if not solved).
  - `termination_status`: the MOI termination status (`nothing` if not solved).
"""
struct QualityResult
    passed::Bool
    reason::String
    iterations::Int
    solve_time::Float64
    termination_status::Any
end

"""
    check_quality(model, optimizer; criteria=QualityCriteria(),
                  feasible_only=false, optimizer_attributes=())

Solve `model` with `optimizer` and judge whether it is a good test LP instance.

`optimizer` is anything accepted by `JuMP.set_optimizer` (e.g.
`HiGHS.Optimizer`). `optimizer_attributes` is an iterable of `name => value`
pairs applied to the model after the optimizer is attached (e.g.
`("solver" => "simplex",)` for HiGHS).

Instances are rejected when they are:

  - Too small (fewer than `criteria.min_constraints` constraints) — checked
    *before* solving.
  - Infeasible, but only when `feasible_only` is `true`.
  - Unbounded.
  - Timed out or hit numerical / solver errors.
  - Nearly optimal (`ALMOST_OPTIMAL` — indicates poor numerical conditioning).
  - Trivially solved (simplex iterations `≤ criteria.min_iterations`).
  - Degenerate (simplex iterations `> criteria.max_iteration_ratio × constraints`).

Returns a [`QualityResult`](@ref).
"""
function check_quality(
    model::Model,
    optimizer;
    criteria::QualityCriteria=QualityCriteria(),
    feasible_only::Bool=false,
    optimizer_attributes=(),
)
    n_cons = num_constraints(model; count_variable_in_set_constraints=false)

    # Pre-solve: reject problems with too few constraints.
    if n_cons < criteria.min_constraints
        return QualityResult(false, "too_few_constraints", -1, 0.0, nothing)
    end

    set_optimizer(model, optimizer)
    set_silent(model)
    set_time_limit_sec(model, criteria.solve_timeout)
    for (name, value) in optimizer_attributes
        set_attribute(model, name, value)
    end
    optimize!(model)

    ts = termination_status(model)
    iters = try
        Int(MOI.get(model, MOI.SimplexIterations()))
    catch
        -1
    end
    stime = try
        solve_time(model)
    catch
        0.0
    end

    # Always reject: timeout.
    if ts == MOI.TIME_LIMIT
        return QualityResult(false, "timeout", iters, stime, ts)
    end

    # Always reject: numerical / solver errors.
    if ts == MOI.NUMERICAL_ERROR || ts == MOI.OTHER_ERROR
        return QualityResult(false, "numerical_error", iters, stime, ts)
    end

    # Always reject: unbounded (MOI represents unbounded as DUAL_INFEASIBLE).
    if ts == MOI.DUAL_INFEASIBLE
        return QualityResult(false, "unbounded", iters, stime, ts)
    end

    # ALMOST_OPTIMAL suggests poor numerical conditioning.
    if ts == MOI.ALMOST_OPTIMAL
        return QualityResult(false, "almost_optimal", iters, stime, ts)
    end

    is_infeasible = (ts == MOI.INFEASIBLE || ts == MOI.INFEASIBLE_OR_UNBOUNDED)

    # Reject infeasible only when feasible-only mode is active.
    if is_infeasible && feasible_only
        return QualityResult(false, "infeasible", iters, stime, ts)
    end

    # Reject anything that isn't optimal or (infeasible when allowed).
    if !(ts == MOI.OPTIMAL || (is_infeasible && !feasible_only))
        return QualityResult(false, "other_status", iters, stime, ts)
    end

    # Iteration-based quality checks (skip if iterations unavailable).
    if iters >= 0
        # Too few iterations — trivially solved or solved in phase 1 only.
        if iters <= criteria.min_iterations
            return QualityResult(false, "too_few_iterations", iters, stime, ts)
        end

        # Excessive iterations relative to problem size — likely degenerate.
        # Skip when n_cons == 0 (e.g. min_constraints == 0): max_iters would be
        # 0 and reject every nonzero iteration count as degenerate.
        if n_cons > 0
            max_iters = ceil(Int, criteria.max_iteration_ratio * n_cons)
            if iters > max_iters
                return QualityResult(false, "degenerate", iters, stime, ts)
            end
        end
    end

    return QualityResult(true, "passed", iters, stime, ts)
end

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
#
# A dataset is generated in two stages.
#
# 1. *Planning* (cheap; no model is built). From the master seed, every global index
#    `1:num_problems` is assigned a variant, a requested feasibility status, a target
#    size, and a private RNG stream seed. The variant and status mixes are stratified
#    (systematic sampling), and when size matching is on the targets are stratified
#    quantiles of the size distribution, so even small datasets match the requested
#    mix and size distribution closely.
# 2. *Execution* (per index, independently). Each index builds its instance from its
#    own stream: it calibrates the requested size until the actual variable count
#    matches the planned target, and applies the optional quality filter, retrying
#    with fresh seeds on failure.
#
# Because an index depends only on (master seed, index), shards that execute disjoint
# index subsets reproduce exactly the instances the unsharded run would, and their
# union is the unsharded dataset.

"""
    PlannedInstance

One entry of a dataset plan (see [`plan_dataset`](@ref)): the global `index`, the
variant `ref`, the requested `feasibility_status`, the `target_variables` the
instance aims for, the `size_quantile` that target was drawn at, and the
`stream_seed` of the private RNG that drives this index's per-attempt seeds.
"""
struct PlannedInstance
    index::Int
    ref::ProblemVariant
    feasibility_status::FeasibilityStatus
    target_variables::Int
    size_quantile::Float64
    stream_seed::Int
end

"""
    GeneratedInstance

Metadata describing a single instance produced by [`generate_dataset`](@ref). Rebuild
it exactly with `generate_problem(ProblemVariant(inst), inst.requested_variables,
inst.feasibility_status, inst.seed; <transforms>, dualize=inst.dualized)`, where
`<transforms>` are the run's `relax_integer`/`bounds_to_constraints` flags and its
`transforms=ModelTransforms(...)` (both recorded in the manifest `config`).

# Fields

  - `index::Int`: 1-based position in the (unsharded) dataset.
  - `problem_type::Symbol`, `variant::Symbol`: the category and variant.
  - `feasibility_status::FeasibilityStatus`: the requested status.
  - `target_variables::Int`: the planned target size.
  - `requested_variables::Int`: the `target_variables` actually passed to the generator
    (differs from `target_variables` after size calibration).
  - `num_variables`, `num_constraints`, `num_nonzeros::Int`: size of the returned model
    (constraints exclude variable bounds; nonzeros count affine-row coefficients).
  - `num_integer::Int`: integer/binary columns *before* integrality relaxation.
  - `seed::Int`: resolved generator seed (reproduces this exact instance).
  - `dualized::Bool`: whether the returned model is a dual reformulation.
  - `transforms::Vector{String}`: model transforms applied, in order (e.g.
    `["relax_integer", "aggregate_rows", "scale_units", "dualize"]`).
  - `verified_status::Union{FeasibilityStatus,Nothing}`: the status a solve
    established (feasibility verification or the quality filter's solve), or
    `nothing` when no solve ran or it was inconclusive.
  - `solve_status::Union{String,Nothing}`: the MOI termination status of that solve.
  - `iterations::Int`, `solve_time::Float64`: quality-filter simplex iterations and
    solve time (`-1`/`NaN` without the filter).
  - `build_time::Float64`: seconds to construct and build the accepted model.
  - `generation_time::Float64`: wall time for this index, including every retry,
    calibration build, verification and quality solve.
  - `attempts::Int`: builds spent on this index.
  - `filename::Union{String,Nothing}`: file written, or `nothing` without `output_dir`.
"""
Base.@kwdef struct GeneratedInstance
    index::Int
    problem_type::Symbol
    variant::Symbol
    feasibility_status::FeasibilityStatus
    target_variables::Int
    requested_variables::Int
    num_variables::Int
    num_constraints::Int
    num_nonzeros::Int
    num_integer::Int
    seed::Int
    dualized::Bool
    transforms::Vector{String}
    verified_status::Union{FeasibilityStatus, Nothing}
    solve_status::Union{String, Nothing}
    iterations::Int
    solve_time::Float64
    build_time::Float64
    generation_time::Float64
    attempts::Int
    filename::Union{String, Nothing}
end

ProblemVariant(inst::GeneratedInstance) = ProblemVariant(inst.problem_type, inst.variant)

"""
    DatasetFailure

A planned index that produced no instance (only with `on_failure=:skip`): its
`index`, `problem_type`, `variant`, `feasibility_status`, `target_variables`, the
number of `attempts` spent, the final failure `reason` (`"error"`,
`"size_mismatch"`, `"contract_violated"`, or a [`check_quality`](@ref) rejection
reason), and `reasons`, one message per failed attempt.
"""
struct DatasetFailure
    index::Int
    problem_type::Symbol
    variant::Symbol
    feasibility_status::FeasibilityStatus
    target_variables::Int
    attempts::Int
    reason::String
    reasons::Vector{String}
end

"""
    GeneratedDataset <: AbstractVector{GeneratedInstance}

Result of [`generate_dataset`](@ref). Indexes and iterates like the vector of
generated instances (sorted by `index`), and additionally carries `failures`
(`Vector{DatasetFailure}`) and `manifest` (the `Dict` written to `manifest.json`).
"""
struct GeneratedDataset <: AbstractVector{GeneratedInstance}
    instances::Vector{GeneratedInstance}
    failures::Vector{DatasetFailure}
    manifest::Dict{String, Any}
end

Base.size(d::GeneratedDataset) = size(d.instances)
Base.getindex(d::GeneratedDataset, i::Int) = d.instances[i]
Base.IndexStyle(::Type{GeneratedDataset}) = IndexLinear()

# ---------------------------------------------------------------------------
# Size distributions
# ---------------------------------------------------------------------------

struct _SizeDistributionSpec
    source::Any
    description::String
end

function _resolve_size_distribution(
    size_distribution, mean::Real, std::Real, min_val::Int, max_val::Int
)
    if min_val > max_val
        error("var_min must be <= var_max.")
    end

    if size_distribution isa Symbol || size_distribution isa AbstractString
        name = Symbol(lowercase(String(size_distribution)))
        if name === :normal
            size_distribution = nothing
        elseif name === :uniform
            size_distribution = Uniform(min_val, max_val)
        elseif name === :loguniform
            min_val >= 1 || error("size_distribution=:loguniform requires var_min >= 1.")
            size_distribution = LogUniform(min_val, max_val)
        else
            error(
                "size_distribution must be :normal, :uniform, :loguniform, or a " *
                "Distributions.UnivariateDistribution (got :$name).",
            )
        end
    end

    if size_distribution !== nothing
        if !(size_distribution isa UnivariateDistribution)
            error("size_distribution must be a Distributions.UnivariateDistribution.")
        end
        lower_bound = try
            minimum(size_distribution)
        catch
            -Inf
        end
        upper_bound = try
            maximum(size_distribution)
        catch
            Inf
        end
        if upper_bound < 2
            error("size_distribution upper bound ($upper_bound) must be >= 2.")
        end
        # Truncate to lower=2 whenever the support reaches below 2 (including
        # unbounded-below distributions), so sampled sizes are always valid
        # problem sizes rather than rounding toward 0.
        if !isfinite(lower_bound) || lower_bound < 2
            dist = truncated(size_distribution; lower=2)
            desc = "truncated($(string(size_distribution)); lower=2)"
            return _SizeDistributionSpec(dist, desc)
        end
        return _SizeDistributionSpec(size_distribution, string(size_distribution))
    end

    if std <= 0
        value = clamp(round(Int, mean), min_val, max_val)
        return _SizeDistributionSpec(value, "fixed($value)")
    end

    dist = truncated(Normal(float(mean), float(std)), min_val, max_val)
    desc = "truncated(Normal($(float(mean)), $(float(std))), $min_val, $max_val)"
    return _SizeDistributionSpec(dist, desc)
end

# Target size at quantile `p` of the size distribution.
function _size_target(spec::_SizeDistributionSpec, p::Real)
    spec.source isa Integer && return Int(spec.source)
    p_clamped = clamp(float(p), eps(Float64), 1.0 - eps(Float64))
    q = Float64(quantile(spec.source, p_clamped))
    if !isfinite(q) || q <= 0
        error("size_distribution must produce finite positive size quantiles; got $q at p=$p.")
    end
    return max(1, round(Int, q))
end

# Smallest and largest target the distribution can produce (the largest may be Inf).
function _size_support(spec::_SizeDistributionSpec)
    spec.source isa Integer && return (Float64(spec.source), Float64(spec.source))
    lo = try
        Float64(minimum(spec.source))
    catch
        1.0
    end
    hi = try
        Float64(maximum(spec.source))
    catch
        Inf
    end
    return (max(lo, 1.0), hi)
end

# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

_parse_status(s::FeasibilityStatus) = s
function _parse_status(s::Union{Symbol, AbstractString})
    name = Symbol(lowercase(String(s)))
    name === :feasible && return feasible
    name === :infeasible && return infeasible
    name === :unknown && return unknown
    return error("Unknown feasibility status: $s (expected feasible, infeasible, or unknown).")
end

# Normalize a status or status mix into parallel (statuses, weights) vectors in the
# canonical order feasible, infeasible, unknown, dropping zero weights.
function _status_mix(spec)
    spec isa AbstractDict || return [_parse_status(spec)], [1.0]
    weights = Dict{FeasibilityStatus, Float64}()
    for (k, v) in spec
        v >= 0 || error("Feasibility-status weights must be nonnegative (got $k => $v).")
        weights[_parse_status(k)] = Float64(v)
    end
    statuses = [s for s in (feasible, infeasible, unknown) if get(weights, s, 0.0) > 0]
    isempty(statuses) && error("feasibility_status mix has no positive weight.")
    w = [weights[s] for s in statuses]
    return statuses, w ./ sum(w)
end

# Systematic (stratified) sampling: `n` draws from the categorical distribution `w`
# using evenly spaced points with one random offset, so entry k is drawn
# floor(n*w[k]) or ceil(n*w[k]) times — the realized mix matches the weights up to
# rounding, and inclusion probabilities stay exactly proportional to the weights.
function _systematic_sample(rng::AbstractRNG, w::AbstractVector{Float64}, n::Int)
    cum = cumsum(w)
    cum[end] = 1.0
    u = rand(rng)
    out = Vector{Int}(undef, n)
    k = 1
    for j in 0:(n - 1)
        x = (u + j) / n
        while k < length(cum) && x >= cum[k]
            k += 1
        end
        out[j + 1] = k
    end
    return out
end

const _GOLDEN_FRACTION = (sqrt(5) - 1) / 2

# Variant indices in a random order that keeps each category contiguous, so
# systematic sampling over it also stratifies the category totals.
function _grouped_random_order(rng::AbstractRNG, refs::Vector{ProblemVariant})
    categories = shuffle!(rng, unique(r.category for r in refs))
    order = Int[]
    for c in categories
        append!(order, shuffle!(rng, findall(r -> r.category == c, refs)))
    end
    return order
end

function _plan_dataset(;
    num_problems::Int=100,
    var_mean::Real=500.0,
    var_std::Real=200.0,
    var_min::Int=50,
    var_max::Int=2000,
    size_distribution=nothing,
    problem_types=nothing,
    exclude=nothing,
    model_class::Union{Symbol, Nothing}=nothing,
    tags=nothing,
    any_tags=nothing,
    exclude_tags=nothing,
    variant_weighting=:category,
    feasibility_status=unknown,
    feasible_only::Bool=false,
    seed::Int=0,
    match_size_distribution::Bool=true,
    match_size_by_category::Bool=false,
    shard_index::Int=1,
    num_shards::Int=1,
)
    num_problems < 0 && error("num_problems must be non-negative.")
    num_shards >= 1 || error("num_shards must be >= 1.")
    1 <= shard_index <= num_shards ||
        error("shard_index must be in 1:num_shards (got $shard_index of $num_shards).")
    num_shards > 1 &&
        seed == 0 &&
        error("Sharded generation requires a fixed nonzero `seed` shared by every shard.")
    statuses, status_weights = _status_mix(feasibility_status)
    if feasible_only
        statuses in ([unknown], [feasible]) ||
            error("feasible_only=true conflicts with feasibility_status=$feasibility_status.")
        statuses, status_weights = [feasible], [1.0]
    end
    size_spec = _resolve_size_distribution(size_distribution, var_mean, var_std, var_min, var_max)
    master_seed = seed == 0 ? rand(RandomDevice(), 1:typemax(Int32)) : seed

    refs = list_problems(;
        problem_types=problem_types,
        exclude=exclude,
        model_class=model_class,
        tags=tags,
        any_tags=any_tags,
        exclude_tags=exclude_tags,
    )
    # A variant must support every target the size distribution can produce;
    # otherwise its generator would raise above a documented cap.
    lo, hi = _size_support(size_spec)
    in_range(r) =
        supports_target(r, max(1, floor(Int, lo))) && (
            if isfinite(hi)
                supports_target(r, ceil(Int, hi))
            else
                get_variant(r).max_target_variables === nothing
            end
        )
    dropped = filter(!in_range, refs)
    refs = filter(in_range, refs)
    isempty(refs) && error("No registered problem variant matches the dataset selection.")
    weights = variant_weights(refs, variant_weighting)
    keep = weights .> 0
    refs, weights = refs[keep], weights[keep]

    rng = MersenneTwister(master_seed)
    n = num_problems
    entries = PlannedInstance[]
    if n > 0
        # 1. Variants: stratified over a category-grouped random order.
        order = _grouped_random_order(rng, refs)
        picks = _systematic_sample(rng, weights[order], n)
        variant_of = [refs[order[k]] for k in picks]
        # 2. Statuses: a golden-ratio (low-discrepancy) sequence along the
        #    variant-grouped order, so both the global mix and every variant's own
        #    mix track the requested weights closely.
        status_cum = cumsum(status_weights)
        status_cum[end] = 1.0
        u = rand(rng)
        status_of = map(0:(n - 1)) do j
            x = mod(u + j * _GOLDEN_FRACTION, 1.0)
            statuses[something(findfirst(>(x), status_cum), length(statuses))]
        end
        # 3. Scatter to random dataset positions.
        perm = randperm(rng, n)
        variant_of = variant_of[perm]
        status_of = status_of[perm]
        # 4. Target sizes: stratified quantiles (per category when requested), or iid.
        quantiles = Vector{Float64}(undef, n)
        if match_size_distribution
            groups = if match_size_by_category
                [
                    findall(r -> r.category == c, variant_of) for
                    c in sort(unique(r.category for r in variant_of))
                ]
            else
                [collect(1:n)]
            end
            for g in groups
                positions = randperm(rng, length(g))
                for (k, idx) in enumerate(g)
                    quantiles[idx] = (positions[k] - rand(rng)) / length(g)
                end
            end
        else
            quantiles .= rand(rng, n)
        end
        # 5. Private stream seeds.
        streams = rand(rng, UInt32, n)
        for idx in 1:n
            mod(idx - 1, num_shards) == shard_index - 1 || continue
            push!(
                entries,
                PlannedInstance(
                    idx,
                    variant_of[idx],
                    status_of[idx],
                    _size_target(size_spec, quantiles[idx]),
                    quantiles[idx],
                    Int(streams[idx]),
                ),
            )
        end
    end
    return (; entries, refs, weights, dropped, size_spec, master_seed, statuses, status_weights)
end

"""
    plan_dataset(; kwargs...) -> Vector{PlannedInstance}

The plan [`generate_dataset`](@ref) would execute — which variant, feasibility status,
and target size every index gets — without building any model. Accepts
`generate_dataset`'s selection, sampling, and sharding keyword arguments
(`num_problems`, `var_*`, `size_distribution`, `problem_types`, `exclude`,
`model_class`, `tags`, `any_tags`, `exclude_tags`, `variant_weighting`,
`feasibility_status`, `feasible_only`, `seed`, `match_size_distribution`,
`match_size_by_category`, `shard_index`, `num_shards`) and returns only this shard's
entries. Use it to inspect the mix of a large run before paying for it.
"""
plan_dataset(; kwargs...) = _plan_dataset(; kwargs...).entries

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

# The status a solve of the *returned* model establishes for the primal, if any.
# For a dual reformulation only OPTIMAL is conclusive (strong duality); an
# infeasible dual leaves the primal either infeasible or unbounded.
function _verified_status(ts, dualized::Bool)
    ts == MOI.OPTIMAL && return feasible
    !dualized && ts == MOI.INFEASIBLE && return infeasible
    return nothing
end

_failure_category(reason::AbstractString) = String(first(split(reason, ':'; limit=2)))

function _instance_filename(
    category::Symbol,
    variant::Symbol,
    num_vars::Int,
    idx::Int,
    file_extension::AbstractString,
    width::Int=5,
)
    return "$(category)_$(variant)_v$(num_vars)_$(lpad(idx, width, '0')).$(file_extension)"
end

# Build one planned index. Returns a `GeneratedInstance`, or a `DatasetFailure` once
# the attempt budget is spent.
function _generate_entry(entry::PlannedInstance, cfg)
    t_start = time()
    rng = MersenneTwister(entry.stream_seed)
    ref = entry.ref
    spec = get_variant(ref)
    lo = spec.min_target_variables
    hi = something(spec.max_target_variables, typemax(Int))
    status = entry.feasibility_status
    target = entry.target_variables
    request = clamp(target, lo, hi)

    reasons = String[]
    calibrations = 0
    best = nothing
    problem_seed = 0
    dualized = false
    fresh_draw = true
    # Under the quality filter its solve doubles as verification only where it is
    # conclusive: OPTIMAL proves a `feasible` request for the primal or its dual, but
    # no passing outcome proves an `infeasible` one (`check_quality` accepts
    # INFEASIBLE_OR_UNBOUNDED, and an infeasible dual leaves the primal infeasible or
    # unbounded). Those requests are verified on the source primal instead.
    verify_optimizer = cfg.quality_filter && status != infeasible ? nothing : cfg.verify_optimizer
    for attempt in 1:cfg.max_retries
        if fresh_draw
            problem_seed = rand(rng, 1:typemax(Int32))
            dualized = _should_dualize(rng, cfg.dualize, cfg.dualize_probability)
        end
        fresh_draw = true

        info = Dict{Symbol, Any}()
        built = try
            _generate_problem_verified(
                ref,
                request,
                status,
                problem_seed;
                cfg.transform_flags...,
                transforms=cfg.model_transforms,
                dualize=dualized,
                optimizer=verify_optimizer,
                max_feasibility_retries=cfg.max_feasibility_retries,
                feasibility_timeout=cfg.feasibility_timeout,
                info=info,
            )
        catch e
            # Never swallow a user interrupt.
            e isa InterruptException && rethrow()
            msg = first(sprint(showerror, e), 300)
            push!(reasons, "error: $msg")
            cfg.verbose && println("[$(entry.index)] attempt $attempt of $ref failed: $msg")
            continue
        end
        model, _, resolved_seed = built
        nvar = num_variables(model)
        cand = (;
            model, seed=resolved_seed, request, dualized, info, err=log(max(nvar, 1) / target)
        )

        # Size calibration: rescale the request by target/actual (keeping the seed)
        # until the actual size is within tolerance or the calibration budget is
        # spent, then settle for the closest build seen.
        if cfg.match && abs(cand.err) > cfg.tolerance
            (best === nothing || abs(cand.err) < abs(best.err)) && (best = cand)
            next_request = clamp(round(Int, request * target / max(nvar, 1)), lo, hi)
            if calibrations < cfg.size_match_attempts && next_request != request
                calibrations += 1
                request = next_request
                fresh_draw = false
                continue
            end
            cand = best
            if cfg.strict_size_match && abs(cand.err) > cfg.tolerance
                push!(
                    reasons,
                    "size_mismatch: |log(actual/target)| = $(round(abs(cand.err), digits=3))",
                )
                best = nothing
                continue
            end
        end

        iterations = -1
        stime = NaN
        solve_status = nothing
        verified = nothing
        if cfg.quality_filter
            result = check_quality(
                cand.model,
                cfg.optimizer;
                criteria=cfg.quality_criteria,
                feasible_only=(status == feasible),
                optimizer_attributes=cfg.optimizer_attributes,
            )
            reason = result.passed ? nothing : result.reason
            if result.passed &&
                status == infeasible &&
                _verified_status(result.termination_status, cand.dualized) == feasible
                reason = "contract_violated"
            end
            if reason !== nothing
                push!(reasons, reason)
                cfg.verbose &&
                    println("[$(entry.index)] attempt $attempt of $ref filtered: $reason")
                best = nothing
                continue
            end
            iterations = result.iterations
            stime = result.solve_time
            solve_status = string(result.termination_status)
            verified = if haskey(cand.info, :verification_status)
                status
            else
                _verified_status(result.termination_status, cand.dualized)
            end
        elseif haskey(cand.info, :verification_status)
            # The verification solve ran on the primal and its contract held.
            solve_status = string(cand.info[:verification_status])
            verified = status
        end

        stats = model_statistics(cand.model)
        filename = nothing
        if cfg.output_dir !== nothing
            filename = _instance_filename(
                ref.category,
                ref.variant,
                stats.num_variables,
                entry.index,
                cfg.file_extension,
                cfg.index_width,
            )
            write_to_file(cand.model, joinpath(cfg.output_dir, filename))
        end
        transforms = [string(k) for (k, v) in pairs(cfg.transform_flags) if v === true]
        append!(transforms, _transform_names(cfg.model_transforms))
        cand.dualized && push!(transforms, "dualize")
        return GeneratedInstance(;
            index=entry.index,
            problem_type=ref.category,
            variant=ref.variant,
            feasibility_status=status,
            target_variables=target,
            requested_variables=cand.request,
            num_variables=stats.num_variables,
            num_constraints=stats.num_constraints,
            num_nonzeros=stats.num_nonzeros,
            num_integer=get(cand.info, :num_integer, 0),
            seed=cand.seed,
            dualized=cand.dualized,
            transforms=transforms,
            verified_status=verified,
            solve_status=solve_status,
            iterations=iterations,
            solve_time=stime,
            build_time=get(cand.info, :build_time, NaN),
            generation_time=time() - t_start,
            attempts=attempt,
            filename=filename,
        )
    end
    return DatasetFailure(
        entry.index,
        ref.category,
        ref.variant,
        status,
        target,
        cfg.max_retries,
        isempty(reasons) ? "unknown" : _failure_category(last(reasons)),
        reasons,
    )
end

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

const _PROVENANCE = Ref{Union{Nothing, Dict{String, Any}}}(nothing)

# Package version, Julia version, and (when the package lives in a git checkout) the
# commit and whether the tracked tree is dirty. Computed once per session.
function _provenance()
    _PROVENANCE[] === nothing || return _PROVENANCE[]
    root = pkgdir(@__MODULE__)
    git(args...) =
        try
            readchomp(pipeline(`git -C $root $args`; stderr=devnull))
        catch
            nothing
        end
    commit = root === nothing ? nothing : git("rev-parse", "HEAD")
    dirty = if commit === nothing
        nothing
    else
        status = git("status", "--porcelain", "--untracked-files=no")
        status === nothing ? nothing : !isempty(status)
    end
    version = pkgversion(@__MODULE__)
    _PROVENANCE[] = Dict{String, Any}(
        "package" => "SyntheticLPs",
        "package_version" => version === nothing ? nothing : string(version),
        "git_commit" => commit,
        "git_dirty" => dirty,
        "julia_version" => string(VERSION),
    )
    return _PROVENANCE[]
end

function _instance_dict(inst::GeneratedInstance)
    return Dict{String, Any}(
        "index" => inst.index,
        "problem_type" => string(inst.problem_type),
        "variant" => string(inst.variant),
        "ref" => "$(inst.problem_type)/$(inst.variant)",
        "feasibility_status" => string(inst.feasibility_status),
        "target_variables" => inst.target_variables,
        "requested_variables" => inst.requested_variables,
        "num_variables" => inst.num_variables,
        "num_constraints" => inst.num_constraints,
        "num_nonzeros" => inst.num_nonzeros,
        "num_integer" => inst.num_integer,
        "seed" => inst.seed,
        "dualized" => inst.dualized,
        "transforms" => inst.transforms,
        "verified_status" =>
            inst.verified_status === nothing ? nothing : string(inst.verified_status),
        "solve_status" => inst.solve_status,
        "iterations" => inst.iterations < 0 ? nothing : inst.iterations,
        "solve_time" => isnan(inst.solve_time) ? nothing : inst.solve_time,
        "build_time" => isnan(inst.build_time) ? nothing : inst.build_time,
        "generation_time" => inst.generation_time,
        "attempts" => inst.attempts,
        "filename" => inst.filename,
    )
end

function _failure_dict(f::DatasetFailure)
    return Dict{String, Any}(
        "index" => f.index,
        "problem_type" => string(f.problem_type),
        "variant" => string(f.variant),
        "ref" => "$(f.problem_type)/$(f.variant)",
        "feasibility_status" => string(f.feasibility_status),
        "target_variables" => f.target_variables,
        "attempts" => f.attempts,
        "reason" => f.reason,
        "reasons" => f.reasons,
    )
end

function _size_match_report(
    instances, tolerance::Float64, enabled::Bool, by_category::Bool, description
)
    errs = [abs(log(i.num_variables / i.target_variables)) for i in instances]
    return Dict{String, Any}(
        "enabled" => enabled,
        "by_category" => by_category,
        "distribution" => description,
        "tolerance" => tolerance,
        "mean_abs_log_error" => isempty(errs) ? nothing : sum(errs) / length(errs),
        "max_abs_log_error" => isempty(errs) ? nothing : maximum(errs),
        "fraction_within_tolerance" =>
            isempty(errs) ? nothing : count(<=(tolerance), errs) / length(errs),
    )
end

function _failure_counts(instances, failures)
    counts = Dict{String, Int}()
    # Rejected attempts of accepted instances are not recorded individually; only
    # failed indices carry their per-attempt reasons.
    for f in failures, r in f.reasons
        c = _failure_category(r)
        counts[c] = get(counts, c, 0) + 1
    end
    return counts
end

_manifest_filename(shard_index::Int, num_shards::Int) =
    if num_shards == 1
        "manifest.json"
    else
        "manifest_shard_$(lpad(shard_index, 4, '0'))_of_$(lpad(num_shards, 4, '0')).json"
    end

_write_json(path, data) = open(io -> JSON.print(io, data, 2), path, "w")

"""
    merge_manifests(output_dir; write=true) -> Dict

Merge the per-shard manifests (`manifest_shard_KKKK_of_NNNN.json`) that sharded
[`generate_dataset`](@ref) runs wrote into `output_dir` into one manifest with every
instance and failure sorted by index, and (with `write=true`) save it as
`manifest.json`. Errors if a shard is missing or the shards disagree on their
configuration (other than `shard_index`).
"""
function merge_manifests(output_dir::AbstractString; write::Bool=true)
    files = sort(
        filter(f -> occursin(r"^manifest_shard_\d+_of_\d+\.json$", f), readdir(output_dir))
    )
    isempty(files) && error("No shard manifests (manifest_shard_*_of_*.json) in $output_dir.")
    shards = [JSON.parsefile(joinpath(output_dir, f)) for f in files]
    num_shards = shards[1]["shard"]["num_shards"]
    found = sort([s["shard"]["shard_index"] for s in shards])
    found == collect(1:num_shards) || error(
        "Expected shards 1:$num_shards in $output_dir, found $(found) " *
        "(missing: $(setdiff(1:num_shards, found))).",
    )
    strip_shard(cfg) = Dict(k => v for (k, v) in cfg if k != "shard_index")
    base_cfg = strip_shard(shards[1]["config"])
    for s in shards
        strip_shard(s["config"]) == base_cfg || error(
            "Shard $(s["shard"]["shard_index"]) was generated with a different configuration."
        )
    end
    instances = sort(reduce(vcat, [s["instances"] for s in shards]); by=i -> i["index"])
    failures = sort(reduce(vcat, [s["failures"] for s in shards]); by=f -> f["index"])
    reasons = Dict{String, Int}()
    for s in shards, (k, v) in s["stats"]["failure_reasons"]
        reasons[k] = get(reasons, k, 0) + v
    end
    merged = deepcopy(shards[1])
    merged["config"] = base_cfg
    merged["shard"] = Dict{String, Any}("num_shards" => num_shards, "merged" => true)
    merged["instances"] = instances
    merged["failures"] = failures
    merged["num_instances"] = length(instances)
    merged["num_failures"] = length(failures)
    merged["stats"] = Dict{String, Any}(
        "attempts" => sum(s["stats"]["attempts"] for s in shards),
        "failure_reasons" => reasons,
        "generation_time" => sum(s["stats"]["generation_time"] for s in shards),
    )
    errs = [abs(log(i["num_variables"] / i["target_variables"])) for i in instances]
    tol = merged["size_match"]["tolerance"]
    merged["size_match"]["mean_abs_log_error"] = isempty(errs) ? nothing : sum(errs) / length(errs)
    merged["size_match"]["max_abs_log_error"] = isempty(errs) ? nothing : maximum(errs)
    merged["size_match"]["fraction_within_tolerance"] =
        isempty(errs) ? nothing : count(<=(tol), errs) / length(errs)
    write && _write_json(joinpath(output_dir, "manifest.json"), merged)
    return merged
end

# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------

"""
    generate_dataset(; kwargs...) -> GeneratedDataset

Generate a dataset of synthetic LP instances. Every index `1:num_problems` is
*planned* from the master `seed` (variant, requested feasibility status, target
size; see [`plan_dataset`](@ref)) and then built independently from its own RNG
stream, so the result is reproducible from a nonzero `seed`, and shards are
disjoint and union to exactly the unsharded dataset. When `output_dir` is set,
instances are written to disk together with a manifest. Returns a
[`GeneratedDataset`](@ref): a vector of [`GeneratedInstance`](@ref) (sorted by
index) carrying `failures` and the `manifest`.

# Selection

  - `problem_types = nothing`: selectors to sample from — categories (`:tsp`,
    `"tsp"`), `"category/variant"` strings, or `ProblemVariant`s. `nothing` = all.
  - `exclude = nothing`: selectors to remove from the selection.
  - `model_class = nothing`: `:lp` or `:mip` — keep variants whose `build_model` is
    continuous / emits integer columns (see [`model_class`](@ref)).
  - `tags`, `any_tags`, `exclude_tags = nothing`: keep variants with all / any of
    these tags, and drop variants with any of `exclude_tags` (see
    [`VARIANT_TAGS`](@ref)).
  - `variant_weighting = :category`: `:category` (uniform over categories, then over
    each category's variants), `:variant` (uniform over variants), or a `Dict` of
    weights keyed by category or `"category/variant"` (see
    [`variant_weights`](@ref)). The realized mix is stratified, so each variant
    appears `floor` or `ceil` of its expected count.
  - Variants whose registered size range cannot cover the size distribution's
    support (a documented `max_target_variables` cap below its upper end) are
    dropped from the selection and listed in the manifest.

# Sizes

  - `size_distribution = nothing`: a `Distributions.UnivariateDistribution` over
    target variable counts, or a shortcut: `:normal` (the default: truncated normal
    from `var_mean`, `var_std`, `var_min`, `var_max`), `:uniform` (`Uniform(var_min,
    var_max)`), or `:loguniform` (`LogUniform(var_min, var_max)` — recommended for
    datasets spanning orders of magnitude, e.g. 1k–100k). Distributions whose
    support reaches below 2 are truncated at 2.
  - `var_mean = 500.0`, `var_std = 200.0`, `var_min = 50`, `var_max = 2000`.
  - `match_size_distribution::Bool = true`: plan stratified quantile targets and
    calibrate each build (rescaling the request by target/actual, same seed) until
    `|log(actual/target)| ≤ size_match_tolerance` or `size_match_attempts`
    recalibrations are spent (then the closest build is kept). `false` draws iid
    targets and builds once.
  - `match_size_by_category::Bool = false`: stratify target quantiles within each
    category (each category spans the whole distribution) instead of globally.
  - `size_match_tolerance::Float64 = 0.05`, `size_match_attempts::Int = 3`.
  - `strict_size_match::Bool = false`: treat a build still outside the tolerance
    after calibration as a failed attempt (retry with a new seed).

# Feasibility and transforms

  - `feasibility_status = unknown`: requested status for every instance, or a mix
    such as `Dict(feasible => 0.5, infeasible => 0.5)` (keys may also be symbols or
    strings). The mix is spread with a low-discrepancy sequence, so both the overall
    and each variant's realized proportions track the weights closely.
  - `feasible_only::Bool = false`: shortcut for `feasibility_status = feasible`.
  - `relax_integer::Bool = true`, `bounds_to_constraints::Bool = false`: model
    transforms, as in [`generate_problem`](@ref). Converted bounds become rows and
    count toward `num_constraints`.
  - `transforms = ModelTransforms()`: practitioner-style reformulations (unit
    scaling, aggregate rows, elastic rows, permutation; see [`ModelTransforms`](@ref)
    or pass a `NamedTuple` of its keywords), seeded by each instance's seed. Added
    rows/columns count toward recorded sizes and size matching. Recorded in the
    manifest `config["transforms"]`; non-identity steps appear in each instance's
    `transforms` list. Elastic rows are refused when any `infeasible` instance is
    requested.
  - `dualize::Bool = false`, `dualize_probability::Real = 0.0`: force, or
    independently sample per instance, the dual reformulation.

# Sharding and failures

  - `shard_index::Int = 1`, `num_shards::Int = 1`: generate only the indices `i`
    with `(i - 1) % num_shards == shard_index - 1`. Requires a nonzero `seed`. Every
    shard writes `manifest_shard_KKKK_of_NNNN.json` (instead of `manifest.json`);
    filenames embed the global index, so shards can share an `output_dir`. Combine
    the manifests with [`merge_manifests`](@ref).
  - `on_failure::Symbol = :error`: what to do when an index exhausts its
    `max_retries` attempts (generator errors, quality rejections, strict size
    mismatches). `:error` throws; `:skip` records a [`DatasetFailure`](@ref) and
    continues, returning a short dataset.
  - `max_retries::Int = 10`: builds per index (calibration builds included).
  - `max_feasibility_retries::Int = 10`: seed-walk budget of feasibility
    verification within one build (see [`generate_problem`](@ref)).

# Output

  - `output_dir = nothing`: directory for instance files and the manifest (created
    if needed). `nothing` keeps everything in memory.
  - `file_extension = "mps"`: passed through to `JuMP.write_to_file`.
  - `write_manifest::Bool = true`.

# Solver (optional; the package stays solver-agnostic)

  - `optimizer = nothing`: e.g. `HiGHS.Optimizer`, or a vector of optimizers
    forming a verification escalation chain (see [`generate_problem`](@ref)).
    It verifies `feasible`/`infeasible` requests (rebuilding on violation). With
    `quality_filter=true`, the quality solve (on the chain's first entry) doubles
    as verification for `feasible` requests; `infeasible` requests are still
    verified on the source primal, since no passing quality solve proves
    infeasibility.
  - `quality_filter::Bool = false`, `quality_criteria = QualityCriteria()`,
    `optimizer_attributes = ()`: see [`check_quality`](@ref). An `infeasible`
    request that the quality solve shows feasible is rejected as
    `"contract_violated"`.

# Misc

  - `seed::Int = 0`: master seed; `0` draws a random one (recorded in the manifest
    as `master_seed`, so the run can still be reproduced).
  - `verbose::Bool = false`: print per-instance progress.
"""
function generate_dataset(;
    num_problems::Int=100,
    var_mean::Real=500.0,
    var_std::Real=200.0,
    var_min::Int=50,
    var_max::Int=2000,
    size_distribution=nothing,
    problem_types=nothing,
    exclude=nothing,
    model_class::Union{Symbol, Nothing}=nothing,
    tags=nothing,
    any_tags=nothing,
    exclude_tags=nothing,
    variant_weighting=:category,
    feasibility_status=unknown,
    feasible_only::Bool=false,
    relax_integer::Bool=true,
    bounds_to_constraints::Bool=false,
    transforms=ModelTransforms(),
    dualize::Bool=false,
    dualize_probability::Real=0.0,
    seed::Int=0,
    match_size_distribution::Bool=true,
    match_size_by_category::Bool=false,
    size_match_tolerance::Float64=0.05,
    size_match_attempts::Int=3,
    strict_size_match::Bool=false,
    shard_index::Int=1,
    num_shards::Int=1,
    on_failure::Symbol=:error,
    output_dir=nothing,
    file_extension::AbstractString="mps",
    write_manifest::Bool=true,
    optimizer=nothing,
    quality_filter::Bool=false,
    quality_criteria::QualityCriteria=QualityCriteria(),
    optimizer_attributes=(),
    max_retries::Int=10,
    max_feasibility_retries::Int=10,
    verbose::Bool=false,
)
    if quality_filter && optimizer === nothing
        error("quality_filter=true requires an `optimizer` (e.g. HiGHS.Optimizer).")
    end
    max_retries < 1 && error("max_retries must be >= 1.")
    size_match_attempts < 0 && error("size_match_attempts must be >= 0.")
    size_match_tolerance < 0 && error("size_match_tolerance must be >= 0.")
    on_failure in (:error, :skip) || error("on_failure must be :error or :skip (got :$on_failure).")
    match_size_by_category &&
        !match_size_distribution &&
        error("match_size_by_category=true requires match_size_distribution=true.")
    validated_dualize_probability = _validate_dualize_probability(dualize_probability)

    plan = _plan_dataset(;
        num_problems=num_problems,
        var_mean=var_mean,
        var_std=var_std,
        var_min=var_min,
        var_max=var_max,
        size_distribution=size_distribution,
        problem_types=problem_types,
        exclude=exclude,
        model_class=model_class,
        tags=tags,
        any_tags=any_tags,
        exclude_tags=exclude_tags,
        variant_weighting=variant_weighting,
        feasibility_status=feasibility_status,
        feasible_only=feasible_only,
        seed=seed,
        match_size_distribution=match_size_distribution,
        match_size_by_category=match_size_by_category,
        shard_index=shard_index,
        num_shards=num_shards,
    )

    output_dir === nothing || mkpath(output_dir)

    # Model transforms applied to every instance before (optional) dualization. Every
    # flag is forwarded to `generate_problem` and recorded in the manifest; flags that
    # are `true` are listed in each instance's `transforms`. The practitioner-style
    # `ModelTransforms` follow them (then `dualize`), and their non-identity steps are
    # listed too.
    transform_flags = (; relax_integer=relax_integer, bounds_to_constraints=bounds_to_constraints)
    model_transforms = _as_transforms(transforms)
    if model_transforms.elastic_probability > 0 &&
        any(((st, w),) -> st == infeasible && w > 0, zip(plan.statuses, plan.status_weights))
        throw(
            ArgumentError(
                "elastic_probability > 0 cannot be combined with `infeasible` " *
                "instances: softening rows can make them feasible.",
            ),
        )
    end

    cfg = (;
        transform_flags,
        model_transforms,
        dualize,
        dualize_probability=validated_dualize_probability,
        # With the quality filter on, `_generate_entry` verifies only `infeasible`
        # requests: for the rest the quality solve is conclusive.
        verify_optimizer=optimizer,
        # A verification escalation chain's first entry is the quality-filter
        # solver: iteration-count criteria need a single, simplex-like solve.
        optimizer=optimizer isa AbstractVector ? first(optimizer) : optimizer,
        quality_filter,
        quality_criteria,
        optimizer_attributes,
        feasibility_timeout=quality_criteria.solve_timeout,
        max_feasibility_retries,
        max_retries,
        match=match_size_distribution,
        tolerance=size_match_tolerance,
        size_match_attempts,
        strict_size_match,
        output_dir,
        file_extension,
        index_width=max(5, ndigits(num_problems)),
        verbose,
    )

    if verbose
        println(
            "Generating $(length(plan.entries)) of $num_problems LP instances" *
            (num_shards > 1 ? " (shard $shard_index of $num_shards)" : ""),
        )
        println("  Output: $(output_dir === nothing ? "(in-memory only)" : output_dir)")
        println("  Master seed: $(plan.master_seed)")
        println("  Size distribution: $(plan.size_spec.description)")
        println(
            "  Size matching: $(match_size_distribution ? "enabled" : "disabled")" *
            (match_size_by_category ? " (per category)" : ""),
        )
        println(
            "  Variants: $(length(plan.refs)) in " *
            "$(length(unique(r.category for r in plan.refs))) categories " *
            "(weighting: $(variant_weighting isa Symbol ? variant_weighting : "custom"))",
        )
        isempty(plan.dropped) ||
            println("  Dropped (size range): $(join(string.(plan.dropped), ", "))")
        println(
            "  Feasibility: " * join(
                ["$s=$(round(w, digits=3))" for (s, w) in zip(plan.statuses, plan.status_weights)],
                ", ",
            ),
        )
        bounds_to_constraints && println("  Bounds → constraints: enabled")
        is_identity(model_transforms) ||
            println("  Model transforms: $(join(_transform_names(model_transforms), ", "))")
        if dualize
            println("  Dual reformulation: forced for every instance")
        elseif validated_dualize_probability > 0
            println("  Dual reformulation probability: $validated_dualize_probability")
        end
        quality_filter && println(
            "  Quality filter: enabled (timeout=$(quality_criteria.solve_timeout)s, " *
            "min_iters=$(quality_criteria.min_iterations), " *
            "max_iter_ratio=$(quality_criteria.max_iteration_ratio), " *
            "min_cons=$(quality_criteria.min_constraints))",
        )
        println()
    end

    t_run = time()
    instances = GeneratedInstance[]
    failures = DatasetFailure[]
    total_attempts = 0
    for (k, entry) in enumerate(plan.entries)
        result = _generate_entry(entry, cfg)
        total_attempts += result.attempts
        if result isa DatasetFailure
            if on_failure === :error
                error(
                    "Dataset index $(entry.index) ($(entry.ref), target=$(entry.target_variables), " *
                    "status=$(entry.feasibility_status)) failed after $(result.attempts) " *
                    "attempts. Reasons: $(join(result.reasons, "; ")). " *
                    "Pass on_failure=:skip to record failures and continue.",
                )
            end
            push!(failures, result)
            verbose && println(
                "[$k/$(length(plan.entries))] index $(entry.index) $(entry.ref) FAILED: $(result.reason)",
            )
        else
            push!(instances, result)
            if verbose
                msg =
                    "[$k/$(length(plan.entries))] " *
                    "$(something(result.filename, string(entry.ref))) " *
                    "(target=$(result.target_variables), actual=$(result.num_variables), " *
                    "cons=$(result.num_constraints), nnz=$(result.num_nonzeros), " *
                    "status=$(result.feasibility_status), dual=$(result.dualized), " *
                    "build=$(round(result.build_time, digits=2))s"
                result.iterations >= 0 && (msg *= ", $(result.iterations) iters")
                println(msg * ")")
            end
        end
    end

    config = Dict{String, Any}(
        "num_problems" => num_problems,
        "seed" => seed,
        "master_seed" => plan.master_seed,
        "var_mean" => var_mean,
        "var_std" => var_std,
        "var_min" => var_min,
        "var_max" => var_max,
        "size_distribution" => plan.size_spec.description,
        "problem_types" =>
            problem_types === nothing ? nothing : string.(_selector_list(problem_types)),
        "exclude" => exclude === nothing ? nothing : string.(_selector_list(exclude)),
        "model_class" => model_class === nothing ? nothing : string(model_class),
        "tags" => tags === nothing ? nothing : string.(_tag_list(tags)),
        "any_tags" => any_tags === nothing ? nothing : string.(_tag_list(any_tags)),
        "exclude_tags" => exclude_tags === nothing ? nothing : string.(_tag_list(exclude_tags)),
        "variant_weighting" => _jsonable(variant_weighting),
        "feasibility_status" =>
            Dict(string(s) => w for (s, w) in zip(plan.statuses, plan.status_weights)),
        "feasible_only" => feasible_only,
        "dualize" => dualize,
        "dualize_probability" => validated_dualize_probability,
        "match_size_distribution" => match_size_distribution,
        "match_size_by_category" => match_size_by_category,
        "size_match_tolerance" => size_match_tolerance,
        "size_match_attempts" => size_match_attempts,
        "strict_size_match" => strict_size_match,
        "shard_index" => shard_index,
        "num_shards" => num_shards,
        "on_failure" => string(on_failure),
        "file_extension" => file_extension,
        "quality_filter" => quality_filter,
        "quality_criteria" => _jsonable(quality_criteria),
        "verification" => optimizer !== nothing,
        "max_retries" => max_retries,
        "max_feasibility_retries" => max_feasibility_retries,
    )
    for (k, v) in pairs(transform_flags)
        config[string(k)] = _jsonable(v)
    end
    config["transforms"] = _transforms_config(model_transforms)

    manifest = Dict{String, Any}(
        "format_version" => 2,
        "provenance" => _provenance(),
        "config" => config,
        "selection" => Dict{String, Any}(
            "variants" => string.(plan.refs),
            "weights" => Dict(string(r) => w for (r, w) in zip(plan.refs, plan.weights)),
            "dropped_out_of_size_range" => string.(plan.dropped),
        ),
        "shard" => Dict{String, Any}(
            "shard_index" => shard_index,
            "num_shards" => num_shards,
            "num_planned" => length(plan.entries),
        ),
        "num_problems" => num_problems,
        "num_instances" => length(instances),
        "num_failures" => length(failures),
        "stats" => Dict{String, Any}(
            "attempts" => total_attempts,
            "failure_reasons" => _failure_counts(instances, failures),
            "generation_time" => time() - t_run,
        ),
        "size_match" => _size_match_report(
            instances,
            size_match_tolerance,
            match_size_distribution,
            match_size_by_category,
            plan.size_spec.description,
        ),
        "instances" => [_instance_dict(i) for i in instances],
        "failures" => [_failure_dict(f) for f in failures],
    )

    if write_manifest && output_dir !== nothing
        _write_json(joinpath(output_dir, _manifest_filename(shard_index, num_shards)), manifest)
    end

    if verbose
        println()
        println(
            "Done: $(length(instances))/$(length(plan.entries)) instances " *
            "($total_attempts builds, $(length(failures)) failed indices, " *
            "$(round(time() - t_run, digits=1))s)",
        )
        sm = manifest["size_match"]
        sm["mean_abs_log_error"] === nothing || println(
            "Size fit: mean_abs_log_error=$(round(sm["mean_abs_log_error"], digits=4)), " *
            "within tolerance=$(round(100 * sm["fraction_within_tolerance"], digits=1))%",
        )
    end

    return GeneratedDataset(instances, failures, manifest)
end

# Make weights, criteria, and other config values JSON-friendly.
_jsonable(x) = x
_jsonable(x::Symbol) = string(x)
_jsonable(x::ProblemVariant) = string(x)
_jsonable(d::AbstractDict) = Dict(string(k) => _jsonable(v) for (k, v) in d)
_jsonable(c::QualityCriteria) = Dict(
    "solve_timeout" => c.solve_timeout,
    "min_constraints" => c.min_constraints,
    "min_iterations" => c.min_iterations,
    "max_iteration_ratio" => c.max_iteration_ratio,
)
