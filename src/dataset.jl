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
        # Sort so the default "all types" selection has a stable order: the RNG
        # consumes types positionally, so an unsorted Dict key order would make
        # a seeded dataset reproducible only within a single process/Julia
        # version, contradicting the documented seed-reproducibility guarantee.
        return sort(available)
    end
    requested = Symbol.(problem_types)
    invalid = setdiff(requested, available)
    if !isempty(invalid)
        error("Unknown problem types: $(join(invalid, ", ")). " *
              "Available: $(join(sort(available), ", "))")
    end
    return requested
end

struct _SizeDistributionSpec
    source::Any
    description::String
end

struct _DatasetCandidate
    problem_type::Symbol
    target_variables::Int
    num_variables::Int
    num_constraints::Int
    seed::Int
    iterations::Int
    solve_time::Float64
end

Base.@kwdef struct _SizeMatchSummary
    selected_count::Int = 0
    candidate_count::Int = 0
    mean_abs_log_error::Union{Float64,Nothing} = nothing
    max_abs_log_error::Union{Float64,Nothing} = nothing
    tolerance::Float64 = 0.0
    tolerance_met::Bool = true
end

mutable struct _GenerationStats
    attempts::Int
    failed::Int
    filter_counts::Dict{String,Int}
end

function _resolve_size_distribution(size_distribution, mean::Real, std::Real,
                                    min_val::Int, max_val::Int)
    if min_val > max_val
        error("var_min must be <= var_max.")
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
        if !isfinite(lower_bound)
            dist = truncated(size_distribution; lower = 2)
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

function _size_quantile(spec::_SizeDistributionSpec, p::Real)
    p_clamped = clamp(float(p), eps(Float64), 1.0 - eps(Float64))
    if spec.source isa Integer
        return Float64(spec.source)
    end
    return Float64(quantile(spec.source, p_clamped))
end

function _checked_size_quantile(spec::_SizeDistributionSpec, p::Real)
    q = _size_quantile(spec, p)
    if !isfinite(q) || q <= 0
        error("size_distribution must produce finite positive size quantiles; " *
              "got $q at p=$p.")
    end
    return q
end

function _target_quantiles(spec::_SizeDistributionSpec, count::Int)
    count < 0 && error("num_problems must be non-negative.")
    return [_checked_size_quantile(spec, (i - 0.5) / count) for i in 1:count]
end

function _sample_num_variables(rng::AbstractRNG, spec::_SizeDistributionSpec)
    if spec.source isa Integer
        return Int(spec.source)
    end
    value = Float64(rand(rng, spec.source))
    if !isfinite(value) || value <= 0
        error("size_distribution sampled a non-positive or non-finite size: $value.")
    end
    return max(1, round(Int, value))
end

function _candidate_target_variables(rng::AbstractRNG,
                                     spec::_SizeDistributionSpec,
                                     quota::Int,
                                     draw_index::Int)
    position = mod(draw_index, quota) + 1
    jittered_p = (position - 0.5 + rand(rng) - 0.5) / quota
    q = _checked_size_quantile(spec, jittered_p)
    return max(1, round(Int, q))
end

function _size_match_summary(selected::Vector{_DatasetCandidate},
                             target_quantiles::Vector{Float64},
                             candidate_count::Int,
                             tolerance::Float64)
    if isempty(selected)
        return _SizeMatchSummary(candidate_count = candidate_count,
                                 tolerance = tolerance)
    end
    sorted_selected = sort(selected, by = c -> c.num_variables)
    errors = [abs(log(sorted_selected[i].num_variables / target_quantiles[i]))
              for i in eachindex(sorted_selected)]
    mean_error = sum(errors) / length(errors)
    max_error = maximum(errors)
    return _SizeMatchSummary(
        selected_count = length(selected),
        candidate_count = candidate_count,
        mean_abs_log_error = mean_error,
        max_abs_log_error = max_error,
        tolerance = tolerance,
        tolerance_met = mean_error <= tolerance,
    )
end

function _select_size_matched_candidates(candidates::Vector{_DatasetCandidate},
                                         quota::Int,
                                         spec::_SizeDistributionSpec,
                                         tolerance::Float64)
    quota == 0 && return _DatasetCandidate[],
                         _SizeMatchSummary(candidate_count = length(candidates),
                                           tolerance = tolerance)
    length(candidates) < quota && error("Cannot select $quota candidates from " *
                                        "$(length(candidates)) candidates.")

    sorted_candidates = sort(candidates, by = c -> c.num_variables)
    target_quantiles = _target_quantiles(spec, quota)
    n = quota
    m = length(sorted_candidates)

    previous = zeros(Float64, m + 1)
    take = falses(n + 1, m + 1)

    for i in 1:n
        current = fill(Inf, m + 1)
        for j in 1:m
            skip_cost = current[j]
            target = target_quantiles[i]
            actual = sorted_candidates[j].num_variables
            take_cost = previous[j] + log(actual / target)^2
            if take_cost < skip_cost
                current[j + 1] = take_cost
                take[i + 1, j + 1] = true
            else
                current[j + 1] = skip_cost
            end
        end
        previous = current
    end

    selected_indices = Int[]
    i = n + 1
    j = m + 1
    while i > 1 && j > 1
        if take[i, j]
            push!(selected_indices, j - 1)
            i -= 1
            j -= 1
        else
            j -= 1
        end
    end
    reverse!(selected_indices)

    selected = [sorted_candidates[idx] for idx in selected_indices]
    summary = _size_match_summary(selected, target_quantiles, length(candidates),
                                  tolerance)
    return selected, summary
end

function _increment_filter_count!(stats::_GenerationStats, reason::String)
    stats.filter_counts[reason] = get(stats.filter_counts, reason, 0) + 1
    return nothing
end

function _attempt_candidate(rng::AbstractRNG,
                            problem_type::Symbol,
                            target_vars::Int,
                            feasibility::FeasibilityStatus,
                            relax_integer::Bool,
                            quality_filter::Bool,
                            optimizer,
                            quality_criteria::QualityCriteria,
                            optimizer_attributes,
                            feasible_only::Bool,
                            stats::_GenerationStats,
                            verbose::Bool)
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
                _increment_filter_count!(stats, result.reason)
                if verbose
                    println("[attempt $(stats.attempts)] filtered $problem_type " *
                            "($target_vars vars): $(result.reason) " *
                            "($(result.iterations) iters, " *
                            "$(round(result.solve_time, digits = 2))s)")
                end
                return nothing
            end
            iterations = result.iterations
            stime = result.solve_time
        end

        return _DatasetCandidate(
            problem_type,
            target_vars,
            num_variables(model),
            num_constraints(model; count_variable_in_set_constraints = false),
            problem_seed,
            iterations,
            stime,
        )
    catch e
        # Never swallow a user interrupt: let Ctrl-C abort the run instead of
        # being counted as a generation failure and retried.
        e isa InterruptException && rethrow()
        stats.failed += 1
        if verbose
            println("[attempt $(stats.attempts)] failed $problem_type " *
                    "($target_vars vars): $e")
        else
            @warn "Failed to generate $problem_type with $target_vars vars" exception = (e, catch_backtrace())
        end
        return nothing
    end
end

function _fill_candidate_pool!(candidates::Vector{_DatasetCandidate},
                               rng::AbstractRNG,
                               group_types::Vector{Symbol},
                               quota::Int,
                               desired_count::Int,
                               target_index_start::Int,
                               attempt_limit::Int,
                               size_spec::_SizeDistributionSpec,
                               feasibility::FeasibilityStatus,
                               relax_integer::Bool,
                               quality_filter::Bool,
                               optimizer,
                               quality_criteria::QualityCriteria,
                               optimizer_attributes,
                               feasible_only::Bool,
                               stats::_GenerationStats,
                               verbose::Bool)
    local_attempts = 0
    while length(candidates) < desired_count && local_attempts < attempt_limit
        local_attempts += 1
        stats.attempts += 1
        problem_type = length(group_types) == 1 ? group_types[1] : rand(rng, group_types)
        target_vars = _candidate_target_variables(rng, size_spec, quota,
                                                  target_index_start + local_attempts - 1)
        candidate = _attempt_candidate(rng, problem_type, target_vars, feasibility,
                                       relax_integer, quality_filter, optimizer,
                                       quality_criteria, optimizer_attributes,
                                       feasible_only, stats, verbose)
        candidate === nothing || push!(candidates, candidate)
    end
    return local_attempts
end

function _insufficient_candidates_error(group_label::AbstractString,
                                        quota::Int,
                                        candidates::Vector{_DatasetCandidate},
                                        stats::_GenerationStats,
                                        group_attempts::Int)
    total_filtered = sum(values(stats.filter_counts); init = 0)
    error("Could not generate enough valid candidates for $group_label: " *
          "needed $quota, accepted $(length(candidates)) after $group_attempts " *
          "group attempts ($(stats.failed) generation errors, " *
          "$total_filtered filtered).")
end

function _strict_size_match_error(group_label::AbstractString,
                                  summary::_SizeMatchSummary)
    error("Size matching for $group_label missed the requested tolerance: " *
          "mean_abs_log_error=$(summary.mean_abs_log_error), " *
          "tolerance=$(summary.tolerance).")
end

function _generate_matched_group(rng::AbstractRNG,
                                 group_label::AbstractString,
                                 group_types::Vector{Symbol},
                                 quota::Int,
                                 size_spec::_SizeDistributionSpec,
                                 feasibility::FeasibilityStatus,
                                 relax_integer::Bool,
                                 quality_filter::Bool,
                                 optimizer,
                                 quality_criteria::QualityCriteria,
                                 optimizer_attributes,
                                 feasible_only::Bool,
                                 candidate_multiplier::Int,
                                 max_candidate_multiplier::Int,
                                 max_retries::Int,
                                 size_match_tolerance::Float64,
                                 strict_size_match::Bool,
                                 stats::_GenerationStats,
                                 verbose::Bool)
    candidates = _DatasetCandidate[]
    selected = _DatasetCandidate[]
    summary = _SizeMatchSummary(tolerance = size_match_tolerance)

    attempt_limit = max(quota, quota * max_candidate_multiplier * max_retries)
    group_attempts = 0
    for multiplier in candidate_multiplier:max_candidate_multiplier
        desired_count = max(quota, quota * multiplier)
        remaining_attempts = attempt_limit - group_attempts
        if remaining_attempts > 0 && length(candidates) < desired_count
            group_attempts += _fill_candidate_pool!(
                candidates, rng, group_types, quota, desired_count,
                group_attempts, remaining_attempts, size_spec, feasibility,
                relax_integer, quality_filter, optimizer, quality_criteria,
                optimizer_attributes, feasible_only, stats, verbose)
        end

        if length(candidates) < quota
            _insufficient_candidates_error(group_label, quota, candidates, stats,
                                           group_attempts)
        end

        selected, summary = _select_size_matched_candidates(candidates, quota,
                                                           size_spec,
                                                           size_match_tolerance)
        if verbose
            println("Matched $group_label with $(length(candidates)) candidates: " *
                    "mean_abs_log_error=$(summary.mean_abs_log_error), " *
                    "tolerance_met=$(summary.tolerance_met)")
        end
        (summary.tolerance_met || multiplier == max_candidate_multiplier ||
         group_attempts >= attempt_limit) && break
    end

    if strict_size_match && !summary.tolerance_met
        _strict_size_match_error(group_label, summary)
    end

    return selected, summary
end

function _generate_unmatched_candidates(rng::AbstractRNG,
                                        types::Vector{Symbol},
                                        num_problems::Int,
                                        size_spec::_SizeDistributionSpec,
                                        feasibility::FeasibilityStatus,
                                        relax_integer::Bool,
                                        quality_filter::Bool,
                                        optimizer,
                                        quality_criteria::QualityCriteria,
                                        optimizer_attributes,
                                        feasible_only::Bool,
                                        max_retries::Int,
                                        stats::_GenerationStats,
                                        verbose::Bool)
    candidates = _DatasetCandidate[]
    attempt_limit = max(num_problems, num_problems * max_retries)
    local_attempts = 0
    while length(candidates) < num_problems && local_attempts < attempt_limit
        local_attempts += 1
        stats.attempts += 1
        problem_type = rand(rng, types)
        target_vars = _sample_num_variables(rng, size_spec)
        candidate = _attempt_candidate(rng, problem_type, target_vars, feasibility,
                                       relax_integer, quality_filter, optimizer,
                                       quality_criteria, optimizer_attributes,
                                       feasible_only, stats, verbose)
        candidate === nothing || push!(candidates, candidate)
    end

    if length(candidates) < num_problems
        _insufficient_candidates_error("dataset", num_problems, candidates, stats,
                                       attempt_limit)
    end
    return candidates
end

function _type_quotas(rng::AbstractRNG, types::Vector{Symbol}, num_problems::Int)
    if num_problems < length(types)
        error("match_size_by_type=true requires num_problems >= number of " *
              "selected problem types ($(length(types))).")
    end
    base_count = div(num_problems, length(types))
    remainder = rem(num_problems, length(types))
    quotas = Dict(type => base_count for type in types)
    remainder_types = shuffle(rng, copy(types))
    for type in remainder_types[1:remainder]
        quotas[type] += 1
    end
    return quotas
end

function _summary_dict(summary::_SizeMatchSummary)
    return Dict{String,Any}(
        "selected_count" => summary.selected_count,
        "candidate_count" => summary.candidate_count,
        "mean_abs_log_error" => summary.mean_abs_log_error,
        "max_abs_log_error" => summary.max_abs_log_error,
        "tolerance" => summary.tolerance,
        "tolerance_met" => summary.tolerance_met,
    )
end

function _materialize_instances(candidates::Vector{_DatasetCandidate},
                                output_dir,
                                file_extension::AbstractString,
                                feasibility::FeasibilityStatus,
                                relax_integer::Bool,
                                verbose::Bool)
    instances = GeneratedInstance[]
    for (idx, candidate) in enumerate(candidates)
        filename = nothing
        if output_dir !== nothing
            model, _ = generate_problem(candidate.problem_type,
                                        candidate.target_variables,
                                        feasibility,
                                        candidate.seed;
                                        relax_integer = relax_integer)
            actual_vars = num_variables(model)
            actual_cons = num_constraints(model; count_variable_in_set_constraints = false)
            if actual_vars != candidate.num_variables || actual_cons != candidate.num_constraints
                error("Regenerated $(candidate.problem_type) with seed " *
                      "$(candidate.seed) changed size from " *
                      "$(candidate.num_variables)/$(candidate.num_constraints) " *
                      "to $actual_vars/$actual_cons.")
            end
            filename = _instance_filename(candidate.problem_type,
                                          candidate.num_variables,
                                          idx,
                                          file_extension)
            write_to_file(model, joinpath(output_dir, filename))
        end

        push!(instances, GeneratedInstance(
            idx,
            candidate.problem_type,
            candidate.target_variables,
            candidate.num_variables,
            candidate.num_constraints,
            candidate.seed,
            feasibility,
            filename,
            candidate.iterations,
            candidate.solve_time,
        ))

        if verbose
            msg = "[$idx/$(length(candidates))] " *
                  "$(filename === nothing ? candidate.problem_type : filename) " *
                  "(target=$(candidate.target_variables), " *
                  "actual=$(candidate.num_variables), " *
                  "cons=$(candidate.num_constraints)"
            msg *= candidate.iterations >= 0 ? ", $(candidate.iterations) iters, " *
                                              "$(round(candidate.solve_time, digits = 2))s)" : ")"
            println(msg)
        end
    end
    return instances
end

function _instance_filename(problem_type::Symbol, num_vars::Int, idx::Int,
                            file_extension::AbstractString)
    return "$(problem_type)_v$(num_vars)_$(lpad(idx, 5, '0')).$(file_extension)"
end

"""
    generate_dataset(; kwargs...) -> Vector{GeneratedInstance}

Generate a dataset of synthetic LP instances by sampling problem types and
target variable counts, building candidate models, and (optionally) solving them
to filter out low-quality instances. By default, accepted candidates are
post-selected so the actual model variable counts match the requested size
distribution closely. When `output_dir` is set, final selected instances are
written to disk; a `manifest.json` summarizing the run is written too unless
`write_manifest=false`.

Returns metadata for every kept instance as a `Vector{GeneratedInstance}`.

# Sampling keyword arguments
- `num_problems::Int = 100`: number of instances to produce.
- `var_mean::Real = 500.0`, `var_std::Real = 200.0`: mean/std of a truncated
  normal over the target variable count when `size_distribution` is not set.
- `var_min::Int = 50`, `var_max::Int = 2000`: truncation bounds used with the
  legacy `var_*` arguments.
- `size_distribution = nothing`: optional `Distributions.UnivariateDistribution`
  over target sizes, e.g. `Uniform(50, 2000)` or
  `truncated(Normal(500, 200), 50, 2000)`. Distributions without a finite lower
  support are automatically truncated at `lower = 2`.
- `problem_types = nothing`: collection of type symbols to sample from
  (`nothing`/empty = all registered types).
- `feasible_only::Bool = false`: request guaranteed-feasible instances.
- `relax_integer::Bool = true`: relax integrality of generated models.
- `seed::Int = 0`: master seed (`0` = non-deterministic).
- `match_size_distribution::Bool = true`: post-select candidates so actual
  variable counts match target distribution quantiles.
- `match_size_by_type::Bool = false`: when matching, match the target size
  distribution independently within each selected problem type.
- `candidate_multiplier::Int = 2`: minimum accepted candidates per final
  instance before matching.
- `max_candidate_multiplier::Int = 12`: accepted-candidate cap for iterative
  matching.
- `size_match_tolerance::Float64 = 0.05`: acceptable mean absolute log-ratio
  error between selected actual sizes and target quantiles.
- `strict_size_match::Bool = false`: throw if the tolerance is missed after
  reaching the candidate cap.

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
- `max_retries::Int = 10`: raw attempt budget multiplier used to overcome
  generator failures and quality-filter rejections.

# Misc
- `verbose::Bool = false`: print per-instance progress.
"""
function generate_dataset(;
        num_problems::Int = 100,
        var_mean::Real = 500.0,
        var_std::Real = 200.0,
        var_min::Int = 50,
        var_max::Int = 2000,
        size_distribution = nothing,
        problem_types = nothing,
        feasible_only::Bool = false,
        relax_integer::Bool = true,
        seed::Int = 0,
        match_size_distribution::Bool = true,
        match_size_by_type::Bool = false,
        candidate_multiplier::Int = 2,
        max_candidate_multiplier::Int = 12,
        size_match_tolerance::Float64 = 0.05,
        strict_size_match::Bool = false,
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
    num_problems < 0 && error("num_problems must be non-negative.")
    candidate_multiplier < 1 && error("candidate_multiplier must be >= 1.")
    max_candidate_multiplier < candidate_multiplier &&
        error("max_candidate_multiplier must be >= candidate_multiplier.")
    max_retries < 1 && error("max_retries must be >= 1.")
    size_match_tolerance < 0 && error("size_match_tolerance must be >= 0.")
    match_size_by_type && !match_size_distribution &&
        error("match_size_by_type=true requires match_size_distribution=true.")

    types = resolve_problem_types(problem_types)
    feasibility = feasible_only ? feasible : unknown
    rng = seed == 0 ? MersenneTwister() : MersenneTwister(seed)
    size_spec = _resolve_size_distribution(size_distribution, var_mean, var_std,
                                           var_min, var_max)

    if output_dir !== nothing
        mkpath(output_dir)
    end

    if verbose
        println("Generating $num_problems LP instances")
        println("  Output: $(output_dir === nothing ? "(in-memory only)" : output_dir)")
        println("  Size distribution: $(size_spec.description)")
        println("  Size matching: $(match_size_distribution ? "enabled" : "disabled")" *
                (match_size_by_type ? " (per type)" : ""))
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

    stats = _GenerationStats(0, 0, Dict{String,Int}())
    selected_candidates = _DatasetCandidate[]
    group_reports = Vector{Dict{String,Any}}()
    per_type_quotas = nothing

    if num_problems == 0
        selected_candidates = _DatasetCandidate[]
    elseif match_size_distribution && match_size_by_type
        quotas = _type_quotas(rng, types, num_problems)
        per_type_quotas = Dict(string(k) => v for (k, v) in quotas)
        for problem_type in types
            quota = quotas[problem_type]
            selected, summary = _generate_matched_group(
                rng,
                string(problem_type),
                [problem_type],
                quota,
                size_spec,
                feasibility,
                relax_integer,
                quality_filter,
                optimizer,
                quality_criteria,
                optimizer_attributes,
                feasible_only,
                candidate_multiplier,
                max_candidate_multiplier,
                max_retries,
                size_match_tolerance,
                strict_size_match,
                stats,
                verbose,
            )
            append!(selected_candidates, selected)
            report = _summary_dict(summary)
            report["group"] = string(problem_type)
            report["quota"] = quota
            push!(group_reports, report)
        end
    elseif match_size_distribution
        selected, summary = _generate_matched_group(
            rng,
            "dataset",
            types,
            num_problems,
            size_spec,
            feasibility,
            relax_integer,
            quality_filter,
            optimizer,
            quality_criteria,
            optimizer_attributes,
            feasible_only,
            candidate_multiplier,
            max_candidate_multiplier,
            max_retries,
            size_match_tolerance,
            strict_size_match,
            stats,
            verbose,
        )
        selected_candidates = selected
        report = _summary_dict(summary)
        report["group"] = "dataset"
        report["quota"] = num_problems
        push!(group_reports, report)
    else
        selected_candidates = _generate_unmatched_candidates(
            rng,
            types,
            num_problems,
            size_spec,
            feasibility,
            relax_integer,
            quality_filter,
            optimizer,
            quality_criteria,
            optimizer_attributes,
            feasible_only,
            max_retries,
            stats,
            verbose,
        )
    end

    shuffle!(rng, selected_candidates)
    instances = _materialize_instances(selected_candidates, output_dir,
                                       file_extension, feasibility,
                                       relax_integer, verbose)

    size_match_report = Dict{String,Any}(
        "enabled" => match_size_distribution,
        "by_type" => match_size_by_type,
        "distribution" => size_spec.description,
        "candidate_multiplier" => candidate_multiplier,
        "max_candidate_multiplier" => max_candidate_multiplier,
        "size_match_tolerance" => size_match_tolerance,
        "strict_size_match" => strict_size_match,
        "per_type_quotas" => per_type_quotas,
        "groups" => group_reports,
    )

    if write_manifest && output_dir !== nothing
        _write_manifest(output_dir, instances, types; seed = seed,
                        var_mean = var_mean, var_std = var_std,
                        var_min = var_min, var_max = var_max,
                        feasible_only = feasible_only, quality_filter = quality_filter,
                        attempts = stats.attempts, failed = stats.failed,
                        filter_counts = stats.filter_counts,
                        size_match = size_match_report)
    end

    if verbose
        println()
        total_filtered = sum(values(stats.filter_counts); init = 0)
        println("Done: $(length(instances))/$num_problems instances " *
                "($(stats.attempts) attempts, $total_filtered filtered, " *
                "$(stats.failed) errors)")
        if !isempty(stats.filter_counts)
            println("Filtered by reason:")
            for (reason, count) in sort(collect(stats.filter_counts), by = x -> -x[2])
                println("  $reason: $count")
            end
        end
        for report in group_reports
            if haskey(report, "mean_abs_log_error")
                println("Size fit $(report["group"]): " *
                        "mean_abs_log_error=$(report["mean_abs_log_error"]), " *
                        "tolerance_met=$(report["tolerance_met"])")
            end
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
