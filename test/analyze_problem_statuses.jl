# Analyze distribution of solver statuses across all LPGeneration problem types.
#
# This script generates multiple instances per problem type, solves them with
# Gurobi, and reports the distribution across {feasible, infeasible, unbounded, unknown}.
#
# Usage examples:
#   julia analyze_problem_statuses.jl                      # defaults
#   julia analyze_problem_statuses.jl --samples=100 --target=200
#   julia analyze_problem_statuses.jl --samples 50 --target-min=100 --target-max=400
#   julia analyze_problem_statuses.jl --samples 50 --size medium
#   julia analyze_problem_statuses.jl --timeout=5 --csv results.csv
#   julia analyze_problem_statuses.jl --samples 50 --json results.json
#   julia analyze_problem_statuses.jl --csv results.csv --json results.json
#   julia analyze_problem_statuses.jl --types transportation,diet_problem --samples=200
#
# Notes:
# - Requires Gurobi.jl and a working Gurobi license.
# - By default, DualReductions is disabled to reduce INFEASIBLE_OR_UNBOUNDED cases.

using LPGeneration

using JuMP
import MathOptInterface
const MOI = MathOptInterface

using Gurobi

using ArgParse
using Dates
using Random
using DelimitedFiles
using Printf
using JSON

const DEFAULT_NUM_SAMPLES = 50
const DEFAULT_TARGET_VARS = 200
const DEFAULT_TIMEOUT_SEC = 2.0
const DEFAULT_SEED = 0

function parse_commandline()
    s = ArgParseSettings(; autofix_names=true)
    @add_arg_table! s begin
        "--samples"
            help = "Number of samples per problem type"
            arg_type = Int
            default = DEFAULT_NUM_SAMPLES
        "--target"
            help = "Target number of variables per instance (mutually exclusive with --size)"
            arg_type = Int
        "--target-min"
            help = "Minimum target number of variables per instance (use with --target-max; mutually exclusive with --size and --target)"
            arg_type = Int
        "--target-max"
            help = "Maximum target number of variables per instance (use with --target-min; mutually exclusive with --size and --target)"
            arg_type = Int
        "--size"
            help = "Legacy size bucket: small, medium, or large (mutually exclusive with --target/--target-min/--target-max)"
            arg_type = String
        "--timeout"
            help = "Per-solve time limit in seconds"
            arg_type = Float64
            default = DEFAULT_TIMEOUT_SEC
        "--seed"
            help = "Base random seed"
            arg_type = Int
            default = DEFAULT_SEED
        "--csv"
            help = "Path to write CSV summary"
            arg_type = String
        "--json"
            help = "Path to write JSON summary"
            arg_type = String
        "--types"
            help = "Filter problem types (space-separated or comma-separated)"
            arg_type = String
            nargs = '*'
        "--solution-status"
            help = "Desired feasibility status for generated problems: feasible | infeasible | all"
            arg_type = String
    end
    return parse_args(s; as_symbols=true)
end

function classify_status(model::Model)::Symbol
    ts = termination_status(model)
    ps = primal_status(model)
    ds = dual_status(model)

    # Classify primarily by termination status; use primal/dual evidence when definitive
    if ts == MOI.OPTIMAL || ts == MOI.ALMOST_OPTIMAL
        return :feasible
    elseif ts == MOI.INFEASIBLE
        return :infeasible
    elseif ts == MOI.UNBOUNDED || ts == MOI.DUAL_INFEASIBLE
        return :unbounded
    elseif ts == MOI.INFEASIBLE_OR_UNBOUNDED
        return :unknown
    end

    # Secondary evidence from primal/dual rays/points
    if ds == MOI.INFEASIBLE_POINT
        return :unbounded
    elseif ps == MOI.INFEASIBLE_POINT
        return :infeasible
    end

    return :unknown
end

function print_header()
    println("Analyzing LPGeneration problem status distribution with Gurobi")
    println("Started at: $(Dates.format(now(), DateFormat("yyyymmdd-HH:MM:SS")))")
end

function format_percentage(n::Int, total::Int)
    if total == 0
        return "0.0%"
    end
    return @sprintf("%.1f%%", 100 * n / total)
end

function write_csv(csv_path::String, rows::Vector{Tuple{Symbol,Int,Int,Int,Int,Int}})
    header = ["problem_type", "feasible", "infeasible", "unbounded", "unknown", "total"]
    open(csv_path, "w") do io
        writedlm(io, permutedims(header), ',')
        for (ptype, n_feas, n_infeas, n_unbdd, n_unk, total) in rows
            writedlm(io, [String(ptype) n_feas n_infeas n_unbdd n_unk total], ',')
        end
    end
end

function build_json_rows(selected_types::Vector{Symbol}, counts_by_type::Dict{Symbol, Dict{Symbol, Int}})
    rows = Vector{Dict{String, Any}}()
    for ptype in selected_types
        totals = counts_by_type[ptype]
        total_n = sum(values(totals))
        push!(rows, Dict(
            "problem_type" => String(ptype),
            "feasible" => totals[:feasible],
            "infeasible" => totals[:infeasible],
            "unbounded" => totals[:unbounded],
            "unknown" => totals[:unknown],
            "total" => total_n,
        ))
    end
    return rows
end

function write_json(json_path::String, selected_types::Vector{Symbol}, counts_by_type::Dict{Symbol, Dict{Symbol, Int}}, meta::Dict{String, Any})
    payload = Dict(
        "meta" => meta,
        "counts" => build_json_rows(selected_types, counts_by_type),
    )
    open(json_path, "w") do io
        JSON.print(io, payload, 2)
    end
end

function main()
    options = parse_commandline()

    num_samples = get(options, :samples, DEFAULT_NUM_SAMPLES)
    timeout_sec = get(options, :timeout, DEFAULT_TIMEOUT_SEC)
    base_seed = get(options, :seed, DEFAULT_SEED)
    csv_path = get(options, :csv, nothing)
    json_path = get(options, :json, nothing)
    solution_status_opt = get(options, :solution_status, nothing)

    # Mutually exclusive handling for target vs size
    target_val = get(options, :target, nothing)
    target_min_val = get(options, :target_min, nothing)
    target_max_val = get(options, :target_max, nothing)
    size_val = get(options, :size, nothing)

    has_target = target_val !== nothing
    has_target_min = target_min_val !== nothing
    has_target_max = target_max_val !== nothing
    has_size = size_val !== nothing

    if has_size && (has_target || has_target_min || has_target_max)
        error("Options --size and any of --target, --target-min, --target-max are mutually exclusive. Use only one.")
    end
    if has_target && (has_target_min || has_target_max)
        error("Options --target and --target-min/--target-max are mutually exclusive. Use only one.")
    end
    if has_target_min != has_target_max
        error("Options --target-min and --target-max must be specified together.")
    end

    use_target_range = has_target_min && has_target_max
    target_min = use_target_range ? Int(target_min_val) : nothing
    target_max = use_target_range ? Int(target_max_val) : nothing
    if use_target_range && target_min > target_max
        error("--target-min must be <= --target-max")
    end

    target_vars = has_target ? Int(target_val) : DEFAULT_TARGET_VARS
    size_choice = has_size ? Symbol(String(size_val)) : nothing

    # Types filter: accept either space-separated or comma-separated lists
    types_filter = nothing
    types_val = get(options, :types, nothing)
    if types_val !== nothing
        # types_val is Vector{String} (possibly empty)
        if !isempty(types_val)
            raw_parts = String.(types_val)
            all_parts = String[]
            for p in raw_parts
                append!(all_parts, split(strip(p), ','))
            end
            types_filter = Symbol.(filter(!isempty, strip.(all_parts)))
            if isempty(types_filter)
                types_filter = nothing
            end
        end
    end

    print_header()

    all_types = sort(list_problem_types())
    selected_types = types_filter === nothing ? all_types : [t for t in all_types if t in types_filter]
    if isempty(selected_types)
        println("No problem types selected. Exiting.")
        return
    end

    println("Problem types: $(join(string.(selected_types), ", "))")
    println("Samples per type: $num_samples")
    if size_choice === nothing
        if use_target_range
            println("Target variables range: ~$(target_min)..$(target_max)")
        else
            println("Target variables: ~$(target_vars)")
        end
    else
        println("Size: $(size_choice)")
    end
    println("Time limit per solve: $(timeout_sec)s")
    if solution_status_opt !== nothing
        println("Requested solution status: $(solution_status_opt)")
    end
    println()

    counts_by_type = Dict{Symbol, Dict{Symbol, Int}}()
    for ptype in selected_types
        counts_by_type[ptype] = Dict(:feasible => 0, :infeasible => 0, :unbounded => 0, :unknown => 0)
    end

    for ptype in selected_types
        println("==> $ptype")
        for i in 1:num_samples
            seed_i = base_seed + i
            # Sample parameters
            if size_choice === nothing
                target_vars_i = target_vars
                if use_target_range
                    rng = Random.MersenneTwister(seed_i)
                    target_vars_i = rand(rng, target_min:target_max)
                end
                params = sample_parameters(ptype, target_vars_i; seed=seed_i)
            else
                params = sample_parameters(ptype, size_choice; seed=seed_i)
            end

            # Pass desired solution status if requested
            if solution_status_opt !== nothing
                ss_str = String(solution_status_opt)
                ss_sym = Symbol(lowercase(ss_str))
                if !(ss_sym in [:feasible, :infeasible, :all])
                    error("Invalid --solution-status: $(ss_str). Use feasible | infeasible | all")
                end
                params[:solution_status] = ss_sym
            end

            # Generate model (relax integrality by default)
            model, _ = generate_problem(ptype, params; seed=seed_i)

            # Configure Gurobi
            set_optimizer(model, Gurobi.Optimizer)
            set_optimizer_attribute(model, "OutputFlag", 0)
            set_optimizer_attribute(model, "TimeLimit", timeout_sec)
            set_optimizer_attribute(model, "Threads", 1)
            # Helps distinguish infeasible vs unbounded in some cases
            set_optimizer_attribute(model, "DualReductions", 0)

            # Optimize and classify
            status_sym = :unknown
            try
                optimize!(model)
                status_sym = classify_status(model)
            catch err
                # On any error, mark as unknown
                status_sym = :unknown
            end

            counts_by_type[ptype][status_sym] += 1
        end

        totals = counts_by_type[ptype]
        total_n = sum(values(totals))
        println("  feasible:  $(totals[:feasible])  (" * format_percentage(totals[:feasible], total_n) * ")")
        println("  infeasible: $(totals[:infeasible]) (" * format_percentage(totals[:infeasible], total_n) * ")")
        println("  unbounded:  $(totals[:unbounded])  (" * format_percentage(totals[:unbounded], total_n) * ")")
        println("  unknown:    $(totals[:unknown])    (" * format_percentage(totals[:unknown], total_n) * ")")
        println()
    end

    # Optional CSV output
    if csv_path !== nothing
        rows = Tuple{Symbol,Int,Int,Int,Int,Int}[]
        for ptype in selected_types
            totals = counts_by_type[ptype]
            total_n = sum(values(totals))
            push!(rows, (ptype, totals[:feasible], totals[:infeasible], totals[:unbounded], totals[:unknown], total_n))
        end
        write_csv(csv_path, rows)
        println("Wrote CSV to: $(csv_path)")
    end

    # Optional JSON output
    if json_path !== nothing
        meta = Dict{String, Any}(
            "started_at" => Dates.format(now(), DateFormat("yyyymmdd-HH:MM:SS")),
            "num_samples" => num_samples,
            "timeout_sec" => timeout_sec,
            "seed" => base_seed,
            "types" => String.(selected_types),
        )
        if size_choice === nothing
            if use_target_range
                meta["target_min"] = target_min
                meta["target_max"] = target_max
            else
                meta["target"] = target_vars
            end
        else
            meta["size"] = String(size_choice)
        end
        write_json(json_path, selected_types, counts_by_type, meta)
        println("Wrote JSON to: $(json_path)")
    end

    println("Done.")
end

main()
