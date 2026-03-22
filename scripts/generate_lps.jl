#!/usr/bin/env julia

using ArgParse
using SyntheticLPs
using JuMP
using Random
using Distributions

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
        println()
    end

    generated = 0
    failed = 0

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
    println("Output directory: $(abspath(output_dir))")
end

main()
