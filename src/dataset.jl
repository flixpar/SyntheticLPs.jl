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
# - All randomness flows from a single seeded RNG so that a given `seed`
#   reproduces the exact same dataset (same types, sizes, and per-instance
#   seeds).

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
function check_quality(model::Model, optimizer;
                       criteria::QualityCriteria = QualityCriteria(),
                       feasible_only::Bool = false,
                       optimizer_attributes = ())
    n_cons = num_constraints(model; count_variable_in_set_constraints = false)

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
        max_iters = ceil(Int, criteria.max_iteration_ratio * n_cons)
        if iters > max_iters
            return QualityResult(false, "degenerate", iters, stime, ts)
        end
    end

    return QualityResult(true, "passed", iters, stime, ts)
end

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

"""
    GeneratedInstance

Metadata describing a single instance produced by [`generate_dataset`](@ref).

# Fields
- `index::Int`: 1-based position of the instance within the dataset.
- `problem_type::Symbol`: which generator produced it.
- `target_variables::Int`: requested variable count.
- `num_variables::Int`: actual variable count of the built model.
- `num_constraints::Int`: actual constraint count (excludes variable bounds).
- `seed::Int`: per-instance seed (reproduces this exact instance).
- `feasibility_status::FeasibilityStatus`: requested feasibility status.
- `filename::Union{String,Nothing}`: file the instance was written to, or
  `nothing` if `output_dir` was not set.
- `iterations::Int`: simplex iterations if quality-filtered, else `-1`.
- `solve_time::Float64`: solve time in seconds if quality-filtered, else `NaN`.
"""
struct GeneratedInstance
    index::Int
    problem_type::Symbol
    target_variables::Int
    num_variables::Int
    num_constraints::Int
    seed::Int
    feasibility_status::FeasibilityStatus
    filename::Union{String,Nothing}
    iterations::Int
    solve_time::Float64
end

"""
    resolve_problem_types(problem_types)

Normalize a user-supplied `problem_types` selection into a validated vector of
registered problem-type symbols. `nothing` or an empty collection selects all
registered types. Throws if any requested type is not registered.
"""
function resolve_problem_types(problem_types)
    available = list_problem_types()
    if problem_types === nothing || isempty(problem_types)
        return available
    end
    requested = Symbol.(problem_types)
    invalid = setdiff(requested, available)
    if !isempty(invalid)
        error("Unknown problem types: $(join(invalid, ", ")). " *
              "Available: $(join(sort(available), ", "))")
    end
    return requested
end

function _sample_num_variables(rng::AbstractRNG, mean::Real, std::Real,
                               min_val::Int, max_val::Int)
    if std <= 0
        return clamp(round(Int, mean), min_val, max_val)
    end
    dist = truncated(Normal(float(mean), float(std)), min_val, max_val)
    return round(Int, rand(rng, dist))
end

function _instance_filename(problem_type::Symbol, num_vars::Int, idx::Int,
                            file_extension::AbstractString)
    return "$(problem_type)_v$(num_vars)_$(lpad(idx, 5, '0')).$(file_extension)"
end

"""
    generate_dataset(; kwargs...) -> Vector{GeneratedInstance}

Generate a dataset of synthetic LP instances by repeatedly sampling a problem
type and a target variable count, building each model, and (optionally) solving
it to filter out low-quality instances. When `output_dir` is set, each kept
instance is written to disk; a `manifest.json` summarizing the run is written
too unless `write_manifest=false`.

Returns metadata for every kept instance as a `Vector{GeneratedInstance}`.

# Sampling keyword arguments
- `num_problems::Int = 100`: number of instances to produce.
- `var_mean::Real = 500.0`, `var_std::Real = 200.0`: mean/std of a truncated
  normal over the target variable count.
- `var_min::Int = 50`, `var_max::Int = 2000`: truncation bounds.
- `problem_types = nothing`: collection of type symbols to sample from
  (`nothing`/empty = all registered types).
- `feasible_only::Bool = false`: request guaranteed-feasible instances.
- `relax_integer::Bool = true`: relax integrality of generated models.
- `seed::Int = 0`: master seed (`0` = non-deterministic).

# Output keyword arguments
- `output_dir = nothing`: directory to write instances into (created if
  needed). `nothing` disables file output (metadata is still returned).
- `file_extension::AbstractString = "mps"`: output file format / extension,
  passed through to `JuMP.write_to_file` (e.g. `"mps"`, `"lp"`).
- `write_manifest::Bool = true`: write a `manifest.json` alongside instances.

# Quality-filter keyword arguments
- `optimizer = nothing`: solver used for quality filtering (e.g.
  `HiGHS.Optimizer`). Required when `quality_filter=true`.
- `quality_filter::Bool = false`: solve and filter each instance.
- `quality_criteria::QualityCriteria = QualityCriteria()`: filter thresholds.
- `optimizer_attributes = ()`: `name => value` pairs applied to each solve.
- `max_retries::Int = 10`: attempt budget multiplier when filtering; up to
  `num_problems × max_retries` instances are generated to reach the target.

# Misc
- `verbose::Bool = false`: print per-instance progress.
"""
function generate_dataset(;
        num_problems::Int = 100,
        var_mean::Real = 500.0,
        var_std::Real = 200.0,
        var_min::Int = 50,
        var_max::Int = 2000,
        problem_types = nothing,
        feasible_only::Bool = false,
        relax_integer::Bool = true,
        seed::Int = 0,
        output_dir = nothing,
        file_extension::AbstractString = "mps",
        write_manifest::Bool = true,
        optimizer = nothing,
        quality_filter::Bool = false,
        quality_criteria::QualityCriteria = QualityCriteria(),
        optimizer_attributes = (),
        max_retries::Int = 10,
        verbose::Bool = false,
    )

    if quality_filter && optimizer === nothing
        error("quality_filter=true requires an `optimizer` (e.g. HiGHS.Optimizer).")
    end

    types = resolve_problem_types(problem_types)
    feasibility = feasible_only ? feasible : unknown
    rng = seed == 0 ? MersenneTwister() : MersenneTwister(seed)

    if output_dir !== nothing
        mkpath(output_dir)
    end

    if verbose
        println("Generating $num_problems LP instances")
        println("  Output: $(output_dir === nothing ? "(in-memory only)" : output_dir)")
        println("  Variables: mean=$var_mean, std=$var_std, range=[$var_min, $var_max]")
        println("  Feasibility: $(feasible_only ? "feasible only" : "unknown")")
        println("  Problem types: $(length(types))")
        if quality_filter
            println("  Quality filter: enabled (timeout=$(quality_criteria.solve_timeout)s, " *
                    "min_iters=$(quality_criteria.min_iterations), " *
                    "max_iter_ratio=$(quality_criteria.max_iteration_ratio), " *
                    "min_cons=$(quality_criteria.min_constraints), " *
                    "max_retries=$(max_retries)×n)")
        end
        println()
    end

    instances = GeneratedInstance[]
    failed = 0
    filter_counts = Dict{String,Int}()

    max_attempts = quality_filter ? num_problems * max_retries : num_problems
    attempts = 0

    while length(instances) < num_problems && attempts < max_attempts
        attempts += 1
        problem_type = rand(rng, types)
        target_vars = _sample_num_variables(rng, var_mean, var_std, var_min, var_max)
        problem_seed = rand(rng, 1:typemax(Int32))

        try
            model, _ = generate_problem(problem_type, target_vars, feasibility,
                                        problem_seed; relax_integer = relax_integer)

            iterations = -1
            stime = NaN
            if quality_filter
                result = check_quality(model, optimizer;
                                       criteria = quality_criteria,
                                       feasible_only = feasible_only,
                                       optimizer_attributes = optimizer_attributes)
                if !result.passed
                    filter_counts[result.reason] = get(filter_counts, result.reason, 0) + 1
                    if verbose
                        println("[attempt $attempts] filtered $problem_type " *
                                "($target_vars vars): $(result.reason) " *
                                "($(result.iterations) iters, " *
                                "$(round(result.solve_time, digits = 2))s)")
                    end
                    continue
                end
                iterations = result.iterations
                stime = result.solve_time
            end

            idx = length(instances) + 1
            actual_vars = num_variables(model)
            actual_cons = num_constraints(model; count_variable_in_set_constraints = false)

            filename = nothing
            if output_dir !== nothing
                filename = _instance_filename(problem_type, actual_vars, idx, file_extension)
                write_to_file(model, joinpath(output_dir, filename))
            end

            push!(instances, GeneratedInstance(idx, problem_type, target_vars,
                                               actual_vars, actual_cons, problem_seed,
                                               feasibility, filename, iterations, stime))

            if verbose
                msg = "[$(idx)/$num_problems] $(filename === nothing ? problem_type : filename) " *
                      "(target=$target_vars, actual=$actual_vars, cons=$actual_cons"
                msg *= quality_filter ? ", $iterations iters, $(round(stime, digits = 2))s)" : ")"
                println(msg)
            end
        catch e
            failed += 1
            if verbose
                println("[attempt $attempts] failed $problem_type ($target_vars vars): $e")
            else
                @warn "Failed to generate $problem_type with $target_vars vars" exception = (e, catch_backtrace())
            end
        end
    end

    if write_manifest && output_dir !== nothing
        _write_manifest(output_dir, instances, types; seed = seed,
                        var_mean = var_mean, var_std = var_std,
                        var_min = var_min, var_max = var_max,
                        feasible_only = feasible_only, quality_filter = quality_filter,
                        attempts = attempts, failed = failed, filter_counts = filter_counts)
    end

    if verbose
        println()
        total_filtered = sum(values(filter_counts); init = 0)
        println("Done: $(length(instances))/$num_problems instances " *
                "($attempts attempts, $total_filtered filtered, $failed errors)")
        if !isempty(filter_counts)
            println("Filtered by reason:")
            for (reason, count) in sort(collect(filter_counts), by = x -> -x[2])
                println("  $reason: $count")
            end
        end
        if length(instances) < num_problems
            println("WARNING: only generated $(length(instances)) of $num_problems " *
                    "requested instances (exhausted $max_attempts attempts).")
        end
    end

    return instances
end

function _write_manifest(output_dir, instances, types; kwargs...)
    cfg = Dict{String,Any}(string(k) => _jsonable(v) for (k, v) in kwargs)
    cfg["problem_types"] = string.(types)
    manifest = Dict{String,Any}(
        "config" => cfg,
        "num_instances" => length(instances),
        "instances" => [Dict(
            "index" => inst.index,
            "problem_type" => string(inst.problem_type),
            "target_variables" => inst.target_variables,
            "num_variables" => inst.num_variables,
            "num_constraints" => inst.num_constraints,
            "seed" => inst.seed,
            "feasibility_status" => string(inst.feasibility_status),
            "filename" => inst.filename,
            "iterations" => inst.iterations < 0 ? nothing : inst.iterations,
            "solve_time" => isnan(inst.solve_time) ? nothing : inst.solve_time,
        ) for inst in instances],
    )
    open(joinpath(output_dir, "manifest.json"), "w") do io
        JSON.print(io, manifest, 2)
    end
    return nothing
end

# Make filter-count dicts and other values JSON-friendly.
_jsonable(x) = x
_jsonable(d::AbstractDict) = Dict(string(k) => _jsonable(v) for (k, v) in d)
