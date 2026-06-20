#!/usr/bin/env julia
#
# Command-line wrapper around `SyntheticLPs.generate_dataset`. The actual
# dataset-generation logic lives in the package (`src/dataset.jl`); this script
# only parses CLI arguments and supplies HiGHS as the quality-filter solver.
#
# Examples:
#   julia --project=@. scripts/generate_lps.jl -o output -n 100
#   julia --project=@. scripts/generate_lps.jl -o output -n 50 --feasible-only -q -v
#   julia --project=@. scripts/generate_lps.jl --problem-types transportation,knapsack -n 20

using ArgParse
using SyntheticLPs
using Distributions
using HiGHS

function parse_commandline()
    s = ArgParseSettings(
        description = "Generate synthetic LP datasets using SyntheticLPs.jl",
        prog = "generate_lps.jl",
    )

    @add_arg_table! s begin
        "--output-dir", "-o"
            help = "Directory to save generated instance files"
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
        "--size-distribution"
            help = "Target size distribution: normal (truncated by --var-min/max) or uniform"
            default = "normal"
        "--no-size-matching"
            help = "Disable post-selection that matches actual model sizes to the target distribution"
            action = :store_true
        "--match-size-by-type"
            help = "Match the target size distribution independently within each selected problem type"
            action = :store_true
        "--candidate-multiplier"
            help = "Minimum accepted candidates per final instance before size matching"
            arg_type = Int
            default = 2
        "--max-candidate-multiplier"
            help = "Accepted-candidate cap per final instance for iterative size matching"
            arg_type = Int
            default = 12
        "--size-match-tolerance"
            help = "Mean absolute log-ratio size error tolerance for matched actual sizes"
            arg_type = Float64
            default = 0.05
        "--strict-size-match"
            help = "Fail if size matching cannot meet --size-match-tolerance"
            action = :store_true
        "--feasible-only"
            help = "Only generate problems guaranteed to be feasible"
            action = :store_true
        "--problem-types"
            help = "Comma-separated list of categories (e.g. transportation) or " *
                   "category/variant references (e.g. portfolio/cvar) to sample " *
                   "from. A category expands to all its variants. (default: all)"
            default = ""
        "--file-format"
            help = "Output file format / extension (e.g. mps, lp)"
            default = "mps"
        "--no-manifest"
            help = "Do not write a manifest.json describing the dataset"
            action = :store_true
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
            help = "Minimum simplex iterations to keep an instance"
            arg_type = Int
            default = 3
        "--max-iteration-ratio"
            help = "Maximum simplex iterations as multiple of constraint count before flagging as degenerate"
            arg_type = Float64
            default = 100.0
        "--min-constraints"
            help = "Minimum number of constraints for a valid instance"
            arg_type = Int
            default = 5
        "--max-retries"
            help = "Raw retry multiplier for generator failures and quality-filter rejections"
            arg_type = Int
            default = 10
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    types_str = args["problem-types"]
    # Pass selectors through as strings so `resolve_problem_types` can parse both
    # bare categories and `category/variant` references.
    problem_types = isempty(types_str) ? nothing : String.(strip.(split(types_str, ",")))

    distribution_name = lowercase(args["size-distribution"])
    size_distribution = if distribution_name == "normal"
        nothing
    elseif distribution_name == "uniform"
        Uniform(args["var-min"], args["var-max"])
    else
        error("--size-distribution must be either normal or uniform, got $(args["size-distribution"])")
    end

    criteria = QualityCriteria(
        solve_timeout = args["solve-timeout"],
        min_constraints = args["min-constraints"],
        min_iterations = args["min-iterations"],
        max_iteration_ratio = args["max-iteration-ratio"],
    )

    instances = generate_dataset(
        output_dir = args["output-dir"],
        num_problems = args["num-problems"],
        var_mean = args["var-mean"],
        var_std = args["var-std"],
        var_min = args["var-min"],
        var_max = args["var-max"],
        size_distribution = size_distribution,
        problem_types = problem_types,
        feasible_only = args["feasible-only"],
        seed = args["seed"],
        match_size_distribution = !args["no-size-matching"],
        match_size_by_type = args["match-size-by-type"],
        candidate_multiplier = args["candidate-multiplier"],
        max_candidate_multiplier = args["max-candidate-multiplier"],
        size_match_tolerance = args["size-match-tolerance"],
        strict_size_match = args["strict-size-match"],
        file_extension = args["file-format"],
        write_manifest = !args["no-manifest"],
        quality_filter = args["quality-filter"],
        quality_criteria = criteria,
        optimizer = HiGHS.Optimizer,
        optimizer_attributes = ("solver" => "simplex",),
        max_retries = args["max-retries"],
        verbose = args["verbose"],
    )

    println("Generated $(length(instances)) instances in $(abspath(args["output-dir"]))")
end

main()
