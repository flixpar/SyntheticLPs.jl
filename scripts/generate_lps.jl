#!/usr/bin/env julia
#
# Command-line wrapper around `SyntheticLPs.generate_dataset`. The actual
# dataset-generation logic lives in the package (`src/dataset.jl`); this script
# only parses CLI arguments and supplies HiGHS as the quality-filter solver.
#
# Examples:
#   julia --project=scripts scripts/generate_lps.jl -o output -n 100
#   julia --project=scripts scripts/generate_lps.jl -o output -n 50 --feasible-only -q -v
#   julia --project=scripts scripts/generate_lps.jl --problem-types transportation,knapsack -n 20
#   julia --project=scripts scripts/generate_lps.jl -o big -n 1000 --size-distribution loguniform \
#       --var-min 1000 --var-max 100000 --model-class lp --exclude knapsack --dry-run
#   # Four parallel shards of one dataset, then one merged manifest:
#   for k in 1 2 3 4; do julia --project=scripts scripts/generate_lps.jl -o big -n 1000 \
#       --seed 7 --num-shards 4 --shard-index $k --on-failure skip & done; wait
#   julia --project=scripts scripts/generate_lps.jl -o big --merge-manifests

# HiGHS and ArgParse live in the `scripts` environment rather than the package,
# so activate it here and re-resolve the local package into it. This keeps the
# script runnable straight from a clone, and repairs `scripts/Manifest.toml`
# whenever the package gains a dependency (the manifest is untracked, so it goes
# stale on its own). `scripts/analyze_problem_statuses.jl` does the same.
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=dirname(@__DIR__))
Pkg.instantiate()

using ArgParse
using SyntheticLPs
using HiGHS

function parse_commandline()
    s = ArgParseSettings(;
        description="Generate synthetic LP datasets using SyntheticLPs.jl", prog="generate_lps.jl"
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
        help = "Target size distribution: normal (truncated by --var-min/max), uniform, or loguniform (recommended for ranges spanning orders of magnitude)"
        default = "normal"
        "--no-size-matching"
        help = "Disable stratified size targets and per-instance size calibration (draw iid targets, build once)"
        action = :store_true
        "--match-size-by-category"
        help = "Stratify target sizes within each category instead of across the whole dataset"
        action = :store_true
        "--size-match-tolerance"
        help = "Per-instance tolerance on |log(actual/target)| for size calibration"
        arg_type = Float64
        default = 0.05
        "--size-match-attempts"
        help = "Maximum recalibration builds per instance before keeping the closest"
        arg_type = Int
        default = 3
        "--strict-size-match"
        help = "Treat builds still outside --size-match-tolerance after calibration as failed attempts"
        action = :store_true
        "--feasible-only"
        help = "Only generate problems guaranteed to be feasible"
        action = :store_true
        "--bounds-to-constraints"
        help = "Reformulate variable bounds (other than x >= 0) as explicit affine constraints"
        action = :store_true
        "--dualize"
        help = "Force every generated continuous model to use its dual formulation"
        action = :store_true
        "--dualize-probability"
        help = "Probability of dualizing each generated model (default: 0, disabled)"
        arg_type = Float64
        default = 0.0
        "--problem-types"
        help =
            "Comma-separated list of categories (e.g. transportation) or " *
            "category/variant references (e.g. portfolio/cvar) to sample " *
            "from. A category expands to all its variants. (default: all)"
        default = ""
        "--exclude"
        help = "Comma-separated categories or category/variant references to leave out"
        default = ""
        "--model-class"
        help = "Only sample variants of this model class: lp (continuous build_model) or mip"
        default = ""
        "--tags"
        help = "Comma-separated tags a variant must all carry (see list_tags())"
        default = ""
        "--any-tags"
        help = "Comma-separated tags; a variant must carry at least one"
        default = ""
        "--exclude-tags"
        help = "Comma-separated tags; variants carrying any of them are left out"
        default = ""
        "--variant-weighting"
        help =
            "category (uniform over categories, then variants), variant (uniform over " *
            "variants), or explicit weights such as 'tsp=2,knapsack/standard=0.5' " *
            "(unlisted variants get weight 0)"
        default = "category"
        "--feasibility"
        help =
            "Requested feasibility status: feasible, infeasible, unknown, or a mix " *
            "such as 'feasible=0.5,infeasible=0.5' (overridden by --feasible-only)"
        default = "unknown"
        "--shard-index"
        help = "1-based index of the shard to generate (requires --seed != 0)"
        arg_type = Int
        default = 1
        "--num-shards"
        help = "Split the dataset into this many disjoint shards; merge with --merge-manifests"
        arg_type = Int
        default = 1
        "--merge-manifests"
        help = "Only merge the shard manifests in --output-dir into manifest.json, then exit"
        action = :store_true
        "--on-failure"
        help = "error (abort on an index that exhausts its retries) or skip (record it and continue)"
        default = "error"
        "--dry-run"
        help = "Print the planned variant/status/size mix without building any model"
        action = :store_true
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
        help = "Builds per instance (calibration builds, generator errors, quality rejections)"
        arg_type = Int
        default = 10
    end

    return parse_args(s)
end

_csv(s) = String[strip(t) for t in split(s, ",") if !isempty(strip(t))]
_selectors(s) = isempty(s) ? nothing : _csv(s)

# Parse "key=weight,key=weight" into a Dict{String,Float64}.
function _weights(s)
    d = Dict{String, Float64}()
    for item in _csv(s)
        occursin('=', item) || error("Expected key=weight, got '$item'.")
        k, v = split(item, '='; limit=2)
        d[strip(k)] = parse(Float64, v)
    end
    return d
end

function main()
    args = parse_commandline()

    if args["merge-manifests"]
        merged = merge_manifests(args["output-dir"])
        println(
            "Merged $(merged["shard"]["num_shards"]) shard manifests: " *
            "$(merged["num_instances"]) instances, $(merged["num_failures"]) failures " *
            "→ $(joinpath(args["output-dir"], "manifest.json"))",
        )
        return nothing
    end

    weighting_str = args["variant-weighting"]
    variant_weighting = if weighting_str in ("category", "variant")
        Symbol(weighting_str)
    else
        _weights(weighting_str)
    end
    feasibility_str = args["feasibility"]
    feasibility_status =
        occursin('=', feasibility_str) ? _weights(feasibility_str) : feasibility_str
    model_class = isempty(args["model-class"]) ? nothing : Symbol(args["model-class"])

    # Selection, sampling, and sharding options shared by the plan and the run.
    plan_kwargs = (;
        num_problems=args["num-problems"],
        var_mean=args["var-mean"],
        var_std=args["var-std"],
        var_min=args["var-min"],
        var_max=args["var-max"],
        size_distribution=Symbol(lowercase(args["size-distribution"])),
        problem_types=_selectors(args["problem-types"]),
        exclude=_selectors(args["exclude"]),
        model_class=model_class,
        tags=_selectors(args["tags"]),
        any_tags=_selectors(args["any-tags"]),
        exclude_tags=_selectors(args["exclude-tags"]),
        variant_weighting=variant_weighting,
        feasibility_status=feasibility_status,
        feasible_only=args["feasible-only"],
        seed=args["seed"],
        match_size_distribution=(!args["no-size-matching"]),
        match_size_by_category=args["match-size-by-category"],
        shard_index=args["shard-index"],
        num_shards=args["num-shards"],
    )

    if args["dry-run"]
        plan = plan_dataset(; plan_kwargs...)
        counts = Dict{String, Int}()
        for p in plan
            key = "$(p.ref) [$(p.feasibility_status)]"
            counts[key] = get(counts, key, 0) + 1
        end
        for (key, n) in sort(collect(counts))
            println(rpad(key, 64), n)
        end
        targets = sort([p.target_variables for p in plan])
        isempty(targets) || println(
            "\n$(length(plan)) planned instances; target sizes min=$(first(targets)) " *
            "median=$(targets[cld(length(targets), 2)]) max=$(last(targets))",
        )
        return nothing
    end

    criteria = QualityCriteria(;
        solve_timeout=args["solve-timeout"],
        min_constraints=args["min-constraints"],
        min_iterations=args["min-iterations"],
        max_iteration_ratio=args["max-iteration-ratio"],
    )

    instances = generate_dataset(;
        plan_kwargs...,
        output_dir=args["output-dir"],
        bounds_to_constraints=args["bounds-to-constraints"],
        dualize=args["dualize"],
        dualize_probability=args["dualize-probability"],
        size_match_tolerance=args["size-match-tolerance"],
        size_match_attempts=args["size-match-attempts"],
        strict_size_match=args["strict-size-match"],
        on_failure=Symbol(args["on-failure"]),
        file_extension=args["file-format"],
        write_manifest=(!args["no-manifest"]),
        quality_filter=args["quality-filter"],
        quality_criteria=criteria,
        optimizer=HiGHS.Optimizer,
        optimizer_attributes=("solver" => "simplex",),
        max_retries=args["max-retries"],
        verbose=args["verbose"],
    )

    println(
        "Generated $(length(instances)) instances in $(abspath(args["output-dir"]))" *
        (isempty(instances.failures) ? "" : " ($(length(instances.failures)) failed indices)"),
    )
end

main()
