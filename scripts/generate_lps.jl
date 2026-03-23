#!/usr/bin/env julia

using ArgParse
using SyntheticLPs
using JuMP
using Random
using Distributions
using HiGHS
const MOI = JuMP.MOI

function parse_commandline()
    s = ArgParseSettings(
        description = "Generate synthetic LP instances using SyntheticLPs.jl",
        prog = "generate_lps.jl"
    )

    @add_arg_table! s begin
        "--output-dir", "-o"
            help = "Directory to save generated .mps files"
            default = "output"
        "--num-problems", "-n"
            help = "Number of LP instances to generate"
            arg_type = Int
            default = 100
        "--var-mean"
            help = "Mean number of variables"
            arg_type = Float64
            default = 500.0
        "--var-std"
            help = "Standard deviation of number of variables"
            arg_type = Float64
            default = 200.0
        "--var-min"
            help = "Minimum number of variables"
            arg_type = Int
            default = 50
        "--var-max"
            help = "Maximum number of variables"
            arg_type = Int
            default = 2000
        "--feasible-only"
            help = "Only generate problems guaranteed to be feasible"
            action = :store_true
        "--problem-types"
            help = "Comma-separated list of problem types to sample from (default: all)"
            default = ""
        "--seed"
            help = "Random seed for reproducibility (0 for non-deterministic)"
            arg_type = Int
            default = 0
        "--verbose", "-v"
            help = "Print progress information"
            action = :store_true
        "--quality-filter", "-q"
            help = "Solve each instance with HiGHS and filter out poor-quality test instances"
            action = :store_true
        "--solve-timeout"
            help = "Per-instance solve time limit in seconds (used with --quality-filter)"
            arg_type = Float64
            default = 30.0
        "--min-iterations"
            help = "Minimum simplex iterations to keep an instance (default: 3)"
            arg_type = Int
            default = 3
        "--max-iteration-ratio"
            help = "Maximum simplex iterations as multiple of constraint count before flagging as degenerate (default: 100)"
            arg_type = Float64
            default = 100.0
        "--min-constraints"
            help = "Minimum number of constraints for a valid instance (default: 5)"
            arg_type = Int
            default = 5
        "--max-retries"
            help = "Maximum retry multiplier when quality-filtering (total attempts = n * max-retries)"
            arg_type = Int
            default = 10
    end

    return parse_args(s)
end

function sample_num_variables(rng::AbstractRNG, mean::Float64, std::Float64, min_val::Int, max_val::Int)
    dist = truncated(Normal(mean, std), min_val, max_val)
    return round(Int, rand(rng, dist))
end

function get_problem_types(types_str::String)
    available = list_problem_types()
    if isempty(types_str)
        return available
    end

    requested = Symbol.(strip.(split(types_str, ",")))
    invalid = setdiff(requested, available)
    if !isempty(invalid)
        error("Unknown problem types: $(join(invalid, ", ")). Available: $(join(available, ", "))")
    end
    return requested
end

function generate_filename(problem_type::Symbol, num_vars::Int, idx::Int)
    return "$(problem_type)_v$(num_vars)_$(lpad(idx, 5, '0')).mps"
end

struct QualityResult
    passed::Bool
    reason::String
    iterations::Int
    solve_time::Float64
end

"""
    check_quality(model, solve_timeout, min_iterations, max_iteration_ratio, min_constraints, feasible_only)

Solve the model with HiGHS simplex and check whether it qualifies as a good test LP instance.

Rejects instances that are:
- Too small (fewer than `min_constraints` constraints)
- Infeasible (when `feasible_only` is set)
- Unbounded
- Timed out or hit numerical errors
- Nearly optimal (ALMOST_OPTIMAL — indicates numerical conditioning issues)
- Trivially solved (simplex iterations ≤ `min_iterations`)
- Degenerate (simplex iterations > `max_iteration_ratio` × constraint count)
"""
function check_quality(model::Model, solve_timeout::Float64, min_iterations::Int,
                       max_iteration_ratio::Float64, min_constraints::Int, feasible_only::Bool)
    n_cons = num_constraints(model; count_variable_in_set_constraints=false)

    # Pre-solve: reject problems with too few constraints
    if n_cons < min_constraints
        return QualityResult(false, "too_few_constraints", 0, 0.0)
    end

    # Solve with HiGHS simplex
    set_optimizer(model, HiGHS.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, solve_timeout)
    set_attribute(model, "solver", "simplex")
    optimize!(model)

    ts = termination_status(model)
    iters = try
        MOI.get(model, MOI.SimplexIterations())
    catch
        -1
    end
    stime = solve_time(model)

    # Always reject: timeout
    if ts == MOI.TIME_LIMIT
        return QualityResult(false, "timeout", iters, stime)
    end

    # Always reject: numerical / solver errors
    if ts == MOI.NUMERICAL_ERROR || ts == MOI.OTHER_ERROR
        return QualityResult(false, "numerical_error", iters, stime)
    end

    # Always reject: unbounded (MOI represents unbounded as DUAL_INFEASIBLE)
    if ts == MOI.DUAL_INFEASIBLE
        return QualityResult(false, "unbounded", iters, stime)
    end

    # Reject infeasible only when feasible-only mode is active
    if ts == MOI.INFEASIBLE || ts == MOI.INFEASIBLE_OR_UNBOUNDED
        if feasible_only
            return QualityResult(false, "infeasible", iters, stime)
        end
        # Infeasible problems still get iteration checks below
    end

    # ALMOST_OPTIMAL suggests poor numerical conditioning
    if ts == MOI.ALMOST_OPTIMAL
        return QualityResult(false, "almost_optimal", iters, stime)
    end

    # Reject anything that isn't optimal or (infeasible when allowed)
    is_optimal = (ts == MOI.OPTIMAL)
    is_infeasible_allowed = !feasible_only && (ts == MOI.INFEASIBLE || ts == MOI.INFEASIBLE_OR_UNBOUNDED)
    if !is_optimal && !is_infeasible_allowed
        return QualityResult(false, "other_status", iters, stime)
    end

    # Iteration-based quality checks (skip if iterations unavailable)
    if iters >= 0
        # Too few iterations — trivially solved or solved in phase 1 only
        if iters <= min_iterations
            return QualityResult(false, "too_few_iterations", iters, stime)
        end

        # Excessive iterations relative to problem size — likely degenerate
        if n_cons > 0
            max_iters = ceil(Int, max_iteration_ratio * n_cons)
            if iters > max_iters
                return QualityResult(false, "degenerate", iters, stime)
            end
        end
    end

    return QualityResult(true, "passed", iters, stime)
end

function main()
    args = parse_commandline()

    output_dir = args["output-dir"]
    num_problems = args["num-problems"]
    var_mean = args["var-mean"]
    var_std = args["var-std"]
    var_min = args["var-min"]
    var_max = args["var-max"]
    feasible_only = args["feasible-only"]
    seed = args["seed"]
    verbose = args["verbose"]
    quality_filter = args["quality-filter"]
    solve_timeout = args["solve-timeout"]
    min_iterations = args["min-iterations"]
    max_iteration_ratio = args["max-iteration-ratio"]
    min_constraints = args["min-constraints"]
    max_retries = args["max-retries"]

    feasibility = feasible_only ? SyntheticLPs.feasible : SyntheticLPs.unknown
    problem_types = get_problem_types(args["problem-types"])

    rng = seed == 0 ? MersenneTwister() : MersenneTwister(seed)

    mkpath(output_dir)

    if verbose
        println("Generating $num_problems LP instances")
        println("  Output directory: $output_dir")
        println("  Variables: mean=$var_mean, std=$var_std, range=[$var_min, $var_max]")
        println("  Feasibility: $(feasible_only ? "feasible only" : "unknown")")
        println("  Problem types: $(length(problem_types)) types")
        if quality_filter
            println("  Quality filter: enabled")
            println("    Solve timeout: $(solve_timeout)s")
            println("    Min iterations: $min_iterations")
            println("    Max iteration ratio: $max_iteration_ratio × constraints")
            println("    Min constraints: $min_constraints")
            println("    Max retries: $max_retries × n")
        end
        println()
    end

    generated = 0
    failed = 0
    filter_counts = Dict{String, Int}()

    if quality_filter
        # With quality filtering: keep retrying until we reach num_problems good
        # instances or exhaust the attempt budget.
        total_attempts = 0
        max_total_attempts = num_problems * max_retries

        while generated < num_problems && total_attempts < max_total_attempts
            total_attempts += 1
            problem_type = rand(rng, problem_types)
            target_vars = sample_num_variables(rng, var_mean, var_std, var_min, var_max)
            problem_seed = rand(rng, 1:typemax(Int32))

            try
                model, _ = generate_problem(problem_type, target_vars, feasibility, problem_seed)

                result = check_quality(model, solve_timeout, min_iterations,
                                       max_iteration_ratio, min_constraints, feasible_only)
                if !result.passed
                    filter_counts[result.reason] = get(filter_counts, result.reason, 0) + 1
                    if verbose
                        println("[attempt $total_attempts] Filtered $problem_type ($(target_vars) vars): $(result.reason) ($(result.iterations) iters, $(round(result.solve_time, digits=2))s)")
                    end
                    continue
                end

                generated += 1
                actual_vars = num_variables(model)
                filename = generate_filename(problem_type, actual_vars, generated)
                filepath = joinpath(output_dir, filename)
                write_to_file(model, filepath)

                if verbose
                    println("[$(generated)/$num_problems] Generated $filename (target=$target_vars, actual=$actual_vars, $(result.iterations) iters, $(round(result.solve_time, digits=2))s)")
                end
            catch e
                failed += 1
                if verbose
                    println("[attempt $total_attempts] Failed to generate $problem_type with $target_vars vars: $e")
                else
                    @warn "Failed to generate $problem_type with $target_vars vars" exception=(e, catch_backtrace())
                end
            end
        end

        println()
        total_filtered = sum(values(filter_counts); init=0)
        println("Generation complete: $generated/$num_problems successful ($total_attempts attempts, $total_filtered filtered, $failed errors)")
        if !isempty(filter_counts)
            println("Filtered by reason:")
            for (reason, count) in sort(collect(filter_counts), by=x -> -x[2])
                println("  $reason: $count")
            end
        end
        if generated < num_problems
            println("WARNING: Only generated $generated of $num_problems requested instances (exhausted $max_total_attempts attempts).")
        end
    else
        # Without quality filtering: original behavior
        for i in 1:num_problems
            problem_type = rand(rng, problem_types)
            target_vars = sample_num_variables(rng, var_mean, var_std, var_min, var_max)
            problem_seed = rand(rng, 1:typemax(Int32))

            try
                model, _ = generate_problem(problem_type, target_vars, feasibility, problem_seed)

                actual_vars = num_variables(model)
                filename = generate_filename(problem_type, actual_vars, i)
                filepath = joinpath(output_dir, filename)

                write_to_file(model, filepath)
                generated += 1

                if verbose
                    println("[$i/$num_problems] Generated $filename (target=$target_vars, actual=$actual_vars)")
                end
            catch e
                failed += 1
                if verbose
                    println("[$i/$num_problems] Failed to generate $problem_type with $target_vars vars: $e")
                else
                    @warn "Failed to generate $problem_type with $target_vars vars" exception=(e, catch_backtrace())
                end
            end
        end

        println()
        println("Generation complete: $generated successful, $failed failed")
        if failed > 0
            println("WARNING: $failed problems failed to generate. Use --verbose for details.")
        end
    end

    println("Output directory: $(abspath(output_dir))")
end

main()
