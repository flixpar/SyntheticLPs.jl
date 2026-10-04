# Audit generators at scale: size fidelity, build time, presolve survival, and solve
# behaviour, one JSON line per (variant, target, status, seed) — plus a report mode
# that turns those JSON lines into a per-variant markdown table of red flags.
#
# The question this answers is "what does a solver actually see?" — a generator
# that hits its variable target but presolves to a tenth of its size, or whose
# build time explodes past 10k variables, is not delivering the instance its
# size statistics advertise.
#
# ## Measuring
#
#   julia --project=scripts scripts/audit_generators.jl -o audit.jsonl
#   julia --project=scripts scripts/audit_generators.jl -o audit.jsonl \
#       --variants tsp/standard,energy --targets 1000,10000,50000 --statuses feasible,infeasible
#   julia --project=scripts scripts/audit_generators.jl -o audit.jsonl --model-class lp \
#       --exclude knapsack --tags network
#
# `--variants` takes categories or `category/variant` refs (default: every registered
# variant), narrowed further by the registry filters `--exclude`, `--model-class`,
# `--tags` (all required), and `--exclude-tags`.
#
# To audit a development checkout (e.g. a git worktree) with the `scripts`
# environment's HiGHS/ArgParse, stack the environments instead:
#
#   JULIA_LOAD_PATH="@:/path/to/SyntheticLPs.jl/scripts:@stdlib" \
#       julia --project=. scripts/audit_generators.jl -o audit.jsonl --variants energy
#
# Every record carries the original model's size (`cols`, `rows`, `nnz`, `nint` =
# integer columns before relaxation, `size_ratio` = cols/target), `build_time`, the
# HiGHS presolved size (`presolved_cols/rows/nnz`, `presolve_col_ratio`,
# `presolve_row_ratio`, `presolve_model_status`), and the status the presolve-on
# simplex solve reached (`solve_status`, `simplex_iterations`, `solve_time`). Failed
# builds carry `error`. Models with more than `--max-nnz` (default 20M) nonzeros are recorded but
# not written/presolved/solved (`skipped`), so a pathologically dense generator cannot
# fill the disk with a multi-gigabyte MPS file. Records are appended and flushed one
# at a time, so a run killed by an external `timeout` keeps every completed record;
# temporary MPS files are removed even when a step fails.
#
# ## Reporting
#
#   julia --project=scripts scripts/audit_generators.jl --report audit.jsonl
#   julia --project=scripts scripts/audit_generators.jl --report out_dir/,more.jsonl \
#       --report-out AUDIT.md --flagged-only
#
# `--report` reads JSONL files (or every `*.jsonl` in a directory) and prints one
# markdown row per variant, flagging:
#
#   - `size`: some `size_ratio` outside `[1 - size_tol, 1 + size_tol]` (default 0.1);
#   - `build`: build time above `--max-build-time` (default 30s) at the variant's
#     largest audited target;
#   - `presolve`: a feasible/unknown instance whose presolved column or row ratio is
#     below `--min-presolve-ratio` (default 0.6), among instances presolve did not
#     eliminate entirely;
#   - `presolve-solved`: an instance HiGHS presolve reduced to nothing (it solved or
#     disproved the instance without simplex work) — counted separately for
#     feasible/unknown and for infeasible requests;
#   - `contract`: a feasibility-contract violation (feasible → INFEASIBLE/UNBOUNDED/
#     UNBOUNDED_OR_INFEASIBLE, infeasible → OPTIMAL/UNBOUNDED);
#   - `error` / `timeout` / `skipped`: failed builds, solves at the time limit, and
#     models skipped for exceeding `--max-nnz`.
#
# Report mode does not load SyntheticLPs, JuMP, or HiGHS.

using ArgParse
using JSON
using Printf

function parse_commandline()
    s = ArgParseSettings(;
        description="Audit SyntheticLPs generators at scale, or report on an audit.",
        prog="audit_generators.jl",
    )
    @add_arg_table! s begin
        "--output", "-o"
        help = "JSONL file to append records to (required unless --report)"
        default = ""
        "--variants"
        help = "Comma-separated categories or category/variant refs (default: every registered variant)"
        default = ""
        "--exclude"
        help = "Comma-separated categories or category/variant refs to skip"
        default = ""
        "--model-class"
        help = "Only audit variants of this model class: lp or mip"
        default = ""
        "--tags"
        help = "Comma-separated tags a variant must all carry"
        default = ""
        "--exclude-tags"
        help = "Comma-separated tags; variants carrying any are skipped"
        default = ""
        "--targets"
        help = "Comma-separated target variable counts"
        default = "1000,10000,50000"
        "--statuses"
        help = "Comma-separated feasibility statuses"
        default = "feasible,infeasible,unknown"
        "--seeds"
        help = "Comma-separated seeds"
        default = "0"
        "--solve-time-limit"
        help = "Time limit (s) for the presolve-on simplex solve; 0 skips solving"
        arg_type = Float64
        default = 60.0
        "--skip-presolve"
        help = "Skip the standalone presolve measurement"
        action = :store_true
        "--max-nnz"
        help = "Skip presolve/solve (and the MPS write) for models with more nonzeros than this"
        arg_type = Int
        default = 20_000_000
        "--scratch"
        help = "Directory for temporary MPS files"
        default = tempdir()
        "--report"
        help = "Comma-separated JSONL files or directories to summarize as markdown (no auditing)"
        default = ""
        "--report-out"
        help = "Write the report to this file instead of stdout"
        default = ""
        "--flagged-only"
        help = "Report only variants with at least one flag"
        action = :store_true
        "--size-tol"
        help = "Report: flag size ratios outside [1 - tol, 1 + tol]"
        arg_type = Float64
        default = 0.1
        "--max-build-time"
        help = "Report: flag build times (s) above this at the largest target"
        arg_type = Float64
        default = 30.0
        "--min-presolve-ratio"
        help = "Report: flag feasible/unknown presolved column or row ratios below this"
        arg_type = Float64
        default = 0.6
    end
    return parse_args(s)
end

_csv(s::AbstractString) = String[strip(t) for t in split(s, ',') if !isempty(strip(t))]

# ---------------------------------------------------------------------------
# Audit mode (needs SyntheticLPs, JuMP, HiGHS; loaded on demand)
# ---------------------------------------------------------------------------

function resolve_variants(args)
    sel(s) = isempty(s) ? nothing : _csv(s)
    model_class = isempty(args["model-class"]) ? nothing : Symbol(args["model-class"])
    return list_problems(;
        problem_types=sel(args["variants"]),
        exclude=sel(args["exclude"]),
        model_class=model_class,
        tags=sel(args["tags"]),
        exclude_tags=sel(args["exclude-tags"]),
    )
end

parse_status(s) = Dict("feasible" => feasible, "infeasible" => infeasible, "unknown" => unknown)[s]

function highs_from_mps(path)
    h = Highs_create()
    Highs_setBoolOptionValue(h, "output_flag", false)
    Highs_setIntOptionValue(h, "threads", 1)
    st = Highs_readModel(h, path)
    return h, st
end

const HIGHS_MODEL_STATUS = Dict(
    0 => "NOTSET",
    1 => "LOAD_ERROR",
    2 => "MODEL_ERROR",
    3 => "PRESOLVE_ERROR",
    4 => "SOLVE_ERROR",
    5 => "POSTSOLVE_ERROR",
    6 => "MODEL_EMPTY",
    7 => "OPTIMAL",
    8 => "INFEASIBLE",
    9 => "UNBOUNDED_OR_INFEASIBLE",
    10 => "UNBOUNDED",
    11 => "OBJECTIVE_BOUND",
    12 => "OBJECTIVE_TARGET",
    13 => "TIME_LIMIT",
    14 => "ITERATION_LIMIT",
    15 => "UNKNOWN",
    16 => "SOLUTION_LIMIT",
    17 => "INTERRUPT",
    18 => "MEMORY_LIMIT",
)

model_status_string(h) =
    get(HIGHS_MODEL_STATUS, Int(Highs_getModelStatus(h)), "STATUS_$(Highs_getModelStatus(h))")

function presolve_record(path)
    h, st = highs_from_mps(path)
    t = @elapsed Highs_presolve(h)
    rec = Dict{String, Any}(
        "presolve_time" => t,
        "presolve_model_status" => model_status_string(h),
        "presolved_cols" => Int(Highs_getPresolvedNumCol(h)),
        "presolved_rows" => Int(Highs_getPresolvedNumRow(h)),
        "presolved_nnz" => Int(Highs_getPresolvedNumNz(h)),
    )
    Highs_destroy(h)
    return rec
end

function solve_record(path, time_limit)
    h, st = highs_from_mps(path)
    Highs_setDoubleOptionValue(h, "time_limit", time_limit)
    Highs_setStringOptionValue(h, "solver", "simplex")
    t = @elapsed Highs_run(h)
    iters = Ref{Cint}(0)
    Highs_getIntInfoValue(h, "simplex_iteration_count", iters)
    rec = Dict{String, Any}(
        "solve_time" => t,
        "solve_status" => model_status_string(h),
        "simplex_iterations" => Int(iters[]),
    )
    if model_status_string(h) == "OPTIMAL"
        rec["objective"] = Highs_getObjectiveValue(h)
    end
    Highs_destroy(h)
    return rec
end

function audit_one(ref, target, status, seed, args, out)
    rec = Dict{String, Any}(
        "variant" => string(ref), "target" => target, "status" => string(status), "seed" => seed
    )
    # One reused path per process, removed in `finally`, so neither an error nor
    # many instances can accumulate files on disk.
    path = joinpath(args["scratch"], "audit_$(getpid()).mps")
    try
        local model
        rec["build_time"] = @elapsed begin
            model, _ = generate_problem(ref, target, status, seed; relax_integer=false)
        end
        raw = model_statistics(model)
        rec["nint"] = raw.num_integer
        raw.num_integer > 0 && relax_integrality(model)
        rec["cols"] = raw.num_variables
        rec["rows"] = raw.num_constraints
        rec["nnz"] = raw.num_nonzeros
        rec["size_ratio"] = raw.num_variables / target
        if raw.num_nonzeros > args["max-nnz"]
            # A model this dense is itself the finding; writing it can take tens of
            # GB of disk (an 18.7 GB MPS was observed).
            rec["skipped"] = "nnz $(raw.num_nonzeros) exceeds --max-nnz $(args["max-nnz"])"
        else
            rec["write_time"] = @elapsed write_to_file(model, path)
            model = nothing
            if !args["skip-presolve"]
                merge!(rec, presolve_record(path))
                rec["presolve_col_ratio"] = rec["presolved_cols"] / max(rec["cols"], 1)
                rec["presolve_row_ratio"] = rec["presolved_rows"] / max(rec["rows"], 1)
            end
            if args["solve-time-limit"] > 0
                merge!(rec, solve_record(path, args["solve-time-limit"]))
            end
        end
    catch err
        err isa InterruptException && rethrow()
        rec["error"] = sprint(showerror, err)[1:min(end, 500)]
    finally
        rm(path; force=true)
    end
    println(out, JSON.json(rec))
    flush(out)
    @printf(
        "%-50s %7d %-10s cols=%s rows=%s build=%.2fs pres_cols=%s status=%s it=%s %s\n",
        string(ref),
        target,
        string(status),
        get(rec, "cols", "-"),
        get(rec, "rows", "-"),
        get(rec, "build_time", NaN),
        get(rec, "presolved_cols", "-"),
        get(rec, "solve_status", "-"),
        get(rec, "simplex_iterations", "-"),
        if haskey(rec, "error")
            "ERROR: " * first(rec["error"], 120)
        elseif haskey(rec, "skipped")
            "SKIPPED: " * rec["skipped"]
        else
            ""
        end
    )
    flush(stdout)
end

function audit_main(args)
    isempty(args["output"]) && error("--output is required unless --report is given.")
    refs = resolve_variants(args)
    targets = parse.(Int, _csv(args["targets"]))
    statuses = parse_status.(_csv(args["statuses"]))
    seeds = parse.(Int, _csv(args["seeds"]))
    open(args["output"], "a") do out
        # Ascending targets outermost so an external timeout loses only the largest sizes.
        for target in sort(targets), ref in refs, status in statuses, seed in seeds
            audit_one(ref, target, status, seed, args, out)
        end
    end
end

# ---------------------------------------------------------------------------
# Report mode (pure JSON processing)
# ---------------------------------------------------------------------------

function read_records(spec::AbstractString)
    files = String[]
    for p in _csv(spec)
        if isdir(p)
            append!(files, sort(filter(f -> endswith(f, ".jsonl"), readdir(p; join=true))))
        else
            push!(files, p)
        end
    end
    recs = Dict{String, Any}[]
    for f in files, line in eachline(f)
        isempty(strip(line)) && continue
        push!(recs, JSON.parse(line))
    end
    return recs
end

presolve_solved(r) = haskey(r, "presolved_cols") && r["presolved_cols"] == 0

function contract_violated(r)
    st = get(r, "solve_status", nothing)
    st === nothing && return false
    # A feasible request must solve to OPTIMAL, so HiGHS's undecided
    # UNBOUNDED_OR_INFEASIBLE already breaks it; for an infeasible request only
    # a proven feasible point (OPTIMAL / UNBOUNDED) does.
    r["status"] == "feasible" && return st in ("INFEASIBLE", "UNBOUNDED", "UNBOUNDED_OR_INFEASIBLE")
    r["status"] == "infeasible" && return st in ("OPTIMAL", "UNBOUNDED")
    return false
end

_fmt(x::Nothing) = "–"
_fmt(x::Integer) = string(x)
_fmt(x::Real) = @sprintf("%.2f", x)

function variant_summary(recs, args)
    built = filter(r -> haskey(r, "cols"), recs)
    ratios = [r["size_ratio"] for r in built]
    max_target = maximum(r["target"] for r in recs)
    build_at_max = [r["build_time"] for r in built if r["target"] == max_target]
    # Presolve-solved instances are flagged separately; keep them out of the ratios.
    feas = filter(
        r -> r["status"] != "infeasible" && haskey(r, "presolve_col_ratio") && !presolve_solved(r),
        built,
    )
    col_ratio = isempty(feas) ? nothing : minimum(r["presolve_col_ratio"] for r in feas)
    row_ratio = isempty(feas) ? nothing : minimum(r["presolve_row_ratio"] for r in feas)
    ps_feas = count(r -> r["status"] != "infeasible" && presolve_solved(r), built)
    ps_inf = count(r -> r["status"] == "infeasible" && presolve_solved(r), built)
    violations = filter(contract_violated, built)
    errors = count(r -> haskey(r, "error"), recs)
    timeouts = count(r -> get(r, "solve_status", "") == "TIME_LIMIT", recs)
    skipped = count(r -> haskey(r, "skipped"), recs)

    tol = args["size-tol"]
    flags = String[]
    any(x -> x < 1 - tol || x > 1 + tol, ratios) && push!(flags, "size")
    any(>(args["max-build-time"]), build_at_max) && push!(flags, "build")
    minratio = args["min-presolve-ratio"]
    (
        (col_ratio !== nothing && col_ratio < minratio) ||
        (row_ratio !== nothing && row_ratio < minratio)
    ) && push!(flags, "presolve")
    ps_feas + ps_inf > 0 && push!(flags, "presolve-solved")
    isempty(violations) || push!(flags, "contract")
    errors > 0 && push!(flags, "error")
    timeouts > 0 && push!(flags, "timeout")
    skipped > 0 && push!(flags, "skipped")

    return (;
        n=length(recs),
        targets=sort(unique(r["target"] for r in recs)),
        size_min=isempty(ratios) ? nothing : minimum(ratios),
        size_max=isempty(ratios) ? nothing : maximum(ratios),
        max_target,
        build_max=isempty(build_at_max) ? nothing : maximum(build_at_max),
        col_ratio,
        row_ratio,
        ps_feas,
        ps_inf,
        violations=["$(r["status"])@$(r["target"])→$(r["solve_status"])" for r in violations],
        errors,
        timeouts,
        skipped,
        flags,
    )
end

function report_main(args)
    recs = read_records(args["report"])
    isempty(recs) && error("No audit records found in $(args["report"]).")
    by_variant = Dict{String, Vector{Dict{String, Any}}}()
    for r in recs
        push!(get!(by_variant, r["variant"], Dict{String, Any}[]), r)
    end
    summaries = [(v, variant_summary(by_variant[v], args)) for v in sort(collect(keys(by_variant)))]
    flagged = count(s -> !isempty(s[2].flags), summaries)

    io = IOBuffer()
    println(io, "# Generator audit report\n")
    println(
        io,
        "$(length(recs)) records, $(length(summaries)) variants, $flagged flagged. " *
        "Thresholds: size ratio in [$(1 - args["size-tol"]), $(1 + args["size-tol"])], " *
        "build ≤ $(args["max-build-time"])s at the largest target, " *
        "presolved col/row ratio ≥ $(args["min-presolve-ratio"]) (feasible/unknown).\n",
    )
    println(
        io,
        "| variant | n | size ratio (min–max) | build s @ max target | presolve col / row ratio (min) | " *
        "presolve-solved (feas+unk / inf) | contract violations | err | timeout | flags |",
    )
    println(io, "|---|---|---|---|---|---|---|---|---|---|")
    for (v, s) in summaries
        args["flagged-only"] && isempty(s.flags) && continue
        println(
            io,
            "| `$v` | $(s.n) | $(_fmt(s.size_min))–$(_fmt(s.size_max)) | " *
            "$(_fmt(s.build_max)) @ $(s.max_target) | $(_fmt(s.col_ratio)) / $(_fmt(s.row_ratio)) | " *
            "$(s.ps_feas) / $(s.ps_inf) | $(isempty(s.violations) ? "–" : join(s.violations, ", ")) | " *
            "$(s.errors) | $(s.timeouts) | $(isempty(s.flags) ? "ok" : "**" * join(s.flags, ", ") * "**") |",
        )
    end
    text = String(take!(io))
    if isempty(args["report-out"])
        print(text)
    else
        write(args["report-out"], text)
        println("Wrote $(args["report-out"]) ($(length(summaries)) variants, $flagged flagged)")
    end
end

function main()
    args = parse_commandline()
    if !isempty(args["report"])
        report_main(args)
    else
        @eval using SyntheticLPs, JuMP, HiGHS
        Base.invokelatest(audit_main, args)
    end
end

main()
