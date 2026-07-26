# Quality-assurance sweep over every registered SyntheticLPs variant.
#
# For each variant we build models across a matrix of (target size, feasibility
# status, seed) [+ optional relax_integer / bounds_to_constraints axes], solve
# each with HiGHS simplex, and record everything needed to detect:
#   - build errors / exceptions
#   - feasibility-contract violations (requested feasible but not OPTIMAL, etc.)
#   - unexpected unboundedness
#   - numerical errors / ALMOST_OPTIMAL
#   - trivial instances (≈0 simplex iterations, all-zero objective, etc.)
#   - degenerate instances (huge iteration/constraint ratio)
#   - size-matching drift (actual vars far from target)
#   - extreme coefficient scaling
#
# Results are written to a JSON file and a compact anomaly summary is printed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "scripts"))
Pkg.instantiate()

using SyntheticLPs
import SyntheticLPs: FeasibilityStatus, feasible, infeasible, unknown, ProblemVariant
using JuMP
import MathOptInterface
const MOI = MathOptInterface
using HiGHS
using JSON
using Dates
using Printf
using LinearAlgebra
using SparseArrays

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

const TIMEOUT_SEC   = 15.0
const TARGETS       = [50, 300]            # small + medium-large
const STATUSES      = [feasible, infeasible, unknown]
const SEEDS         = [1, 2]
const RELAX_INTEGER = true                 # package default (LP relaxation of MIPs)
const B2C           = false                # package default

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function simplex_iterations(model)
    try
        return Int(MOI.get(model, MOI.SimplexIterations()))
    catch
        return -1
    end
end

# Stats over the finite, nonzero LP matrix coefficients + objective + RHS.
struct CoefStats
    obj_nnz::Int
    obj_max_abs::Float64
    obj_min_abs::Float64          # smallest nonzero abs objective coeff
    mat_nnz::Int
    mat_max_abs::Float64
    mat_min_abs::Float64
    rhs_max_abs::Float64
    ratio::Float64                # max_abs over min_abs across everything nonzero
end

function coef_stats(model)
    # Objective
    obj_aff = try
        objective_function(model)
    catch
        nothing
    end
    obj_vec = Float64[]
    if obj_aff !== nothing
        try
            for (_, coef) in objective_function(model).terms
                push!(obj_vec, Float64(coef))
            end
            if objective_function(model).constant != 0
                push!(obj_vec, Float64(objective_function(model).constant))
            end
        catch
            # fallback for non-affine objectives (shouldn't happen — all LP)
            for v in all_variables(model)
                push!(obj_vec, Float64(coefficient(objective_function(model), v)))
            end
        end
    end
    obj_nz = filter(x -> x != 0.0, obj_vec)
    obj_max = isempty(obj_nz) ? 0.0 : maximum(abs.(obj_nz))
    obj_min = isempty(obj_nz) ? 0.0 : minimum(abs.(obj_nz))

    # Constraint matrix + RHS via the MOI backend.
    mat_max = 0.0; mat_min = Inf; mat_nnz = 0; rhs_max = 0.0
    all_nz = Float64[]
    try
        crefs = all_constraints(model; include_variable_in_set_constraints=false)
        for cr in crefs
            con = constraint_object(cr)
            f = con.func
            s = con.set
            # RHS magnitude
            rhs = try
                if s isa MOI.GreaterThan
                    MOI.constant(s)
                elseif s isa MOI.LessThan
                    MOI.constant(s)
                elseif s isa MOI.EqualTo
                    MOI.constant(s)
                else
                    NaN
                end
            catch
                NaN
            end
            isfinite(rhs) && (rhs_max = max(rhs_max, abs(rhs)))
            # coefficients
            terms = try
                f.terms
            catch
                nothing
            end
            if terms !== nothing
                for (_, c) in terms
                    cv = Float64(c)
                    push!(all_nz, cv)
                end
            end
        end
    catch
        # ignore matrix-extraction failures
    end
    mat_nz = filter(x -> x != 0.0, all_nz)
    mat_nnz = length(mat_nz)
    mat_max = isempty(mat_nz) ? 0.0 : maximum(abs.(mat_nz))
    mat_min = isempty(mat_nz) ? Inf : minimum(abs.(mat_nz))

    nz = filter(isfinite, vcat(obj_nz, mat_nz))
    nz = filter(x -> x != 0.0, nz)
    ratio = (isempty(nz) || minimum(abs.(nz)) == 0) ? 1.0 :
            maximum(abs.(nz)) / minimum(abs.(nz))
    return CoefStats(length(obj_nz), obj_max, obj_min,
                     mat_nnz, mat_max, isfinite(mat_min) ? mat_min : 0.0, rhs_max, ratio)
end

function classify_status(model)
    ts = termination_status(model)
    ps = primal_status(model)
    ds = dual_status(model)
    if ts == MOI.OPTIMAL
        return :optimal
    elseif ts == MOI.ALMOST_OPTIMAL
        return :almost_optimal
    elseif ts == MOI.INFEASIBLE
        return :infeasible
    elseif ts == MOI.DUAL_INFEASIBLE
        return :unbounded
    elseif ts == MOI.INFEASIBLE_OR_UNBOUNDED
        return :inf_or_unb
    elseif ts == MOI.TIME_LIMIT
        return :timeout
    elseif ts in (MOI.NUMERICAL_ERROR, MOI.OTHER_ERROR)
        return :error
    else
        return :other
    end
end

# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

function evaluate(ref::ProblemVariant, target::Int, fstat::FeasibilityStatus,
                  seed::Int; relax_integer::Bool=RELAX_INTEGER,
                  bounds_to_constraints::Bool=B2C)
    rec = Dict{String,Any}(
        "category" => string(ref.category),
        "variant"  => string(ref.variant),
        "ref"      => string(ref),
        "target"   => target,
        "feas_req" => string(fstat),
        "seed"     => seed,
        "relax_integer" => relax_integer,
        "bounds_to_constraints" => bounds_to_constraints,
    )

    model = nothing
    build_err = nothing
    t_build = @elapsed try
        model, _ = generate_problem(ref, target, fstat, seed;
                                    relax_integer=relax_integer,
                                    bounds_to_constraints=bounds_to_constraints)
    catch e
        build_err = string(typeof(e)) * ": " * sprint(showerror, e)
    end
    if model === nothing
        rec["outcome"] = "build_error"
        rec["build_error"] = build_err
        rec["build_time"] = t_build
        return rec
    end

    nv = num_variables(model)
    nc = num_constraints(model; count_variable_in_set_constraints=false)
    sense = try string(objective_sense(model)) catch; "unknown" end
    cs = coef_stats(model)

    rec["num_variables"] = nv
    rec["num_constraints"] = nc
    rec["objective_sense"] = sense
    rec["obj_nnz"] = cs.obj_nnz
    rec["obj_max_abs"] = cs.obj_max_abs
    rec["mat_nnz"] = cs.mat_nnz
    rec["coef_ratio"] = cs.ratio
    rec["rhs_max_abs"] = cs.rhs_max_abs
    rec["build_time"] = t_build

    # Determinism check: rebuild with same params, compare var/cons counts.
    det_ok = true
    try
        m2, _ = generate_problem(ref, target, fstat, seed;
                                 relax_integer=relax_integer,
                                 bounds_to_constraints=bounds_to_constraints)
        det_ok = (num_variables(m2) == nv) &&
                 (num_constraints(m2; count_variable_in_set_constraints=false) == nc)
    catch
        det_ok = false
    end
    rec["deterministic"] = det_ok

    # Solve
    set_optimizer(model, HiGHS.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, TIMEOUT_SEC)
    try set_attribute(model, "solver", "simplex") catch end
    t_solve = @elapsed optimize!(model)

    cls = classify_status(model)
    iters = simplex_iterations(model)
    obj_val = try
        objective_value(model)
    catch
        NaN
    end
    has_ray = try
        primal_status(model) == MOI.INFEASIBILITY_CERTIFICATE
    catch
        false
    end

    rec["outcome"] = "solved"
    rec["status"] = string(cls)
    rec["termination"] = string(termination_status(model))
    rec["primal_status"] = string(primal_status(model))
    rec["dual_status"] = string(dual_status(model))
    rec["iterations"] = iters
    rec["solve_time"] = t_solve
    rec["objective_value"] = obj_val

    return rec
end

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

function main()
    println("QA sweep started: ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
    println("Timeout: $(TIMEOUT_SEC)s | targets=$TARGETS | statuses=$(string.(STATUSES)) | seeds=$SEEDS")
    println("relax_integer=$RELAX_INTEGER | bounds_to_constraints=$B2C")
    println()

    problems = list_problems()
    println("Testing $(length(problems)) variants")
    println()

    records = Dict{String,Any}[]
    t0 = time()
    for (i, ref) in enumerate(problems)
        for target in TARGETS, fstat in STATUSES, seed in SEEDS
            rec = evaluate(ref, target, fstat, seed)
            push!(records, rec)
        end
        if i % 5 == 0 || i == length(problems)
            elapsed = time() - t0
            @printf("[%d/%d] %s  (%.0fs elapsed)\n", i, length(problems), ref, elapsed)
        end
    end

    out = joinpath(@__DIR__, "qa_results.json")
    open(out, "w") do io
        JSON.print(io, Dict("records" => records,
                            "meta" => Dict("timeout" => TIMEOUT_SEC,
                                           "targets" => TARGETS,
                                           "statuses" => string.(STATUSES),
                                           "seeds" => SEEDS,
                                           "relax_integer" => RELAX_INTEGER,
                                           "bounds_to_constraints" => B2C,
                                           "generated_at" => Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))), 2)
    end
    println("\nWrote $(length(records)) records to $out")

    summarize(records)
    return records
end

# ---------------------------------------------------------------------------
# Anomaly detection / summary
# ---------------------------------------------------------------------------

function summarize(records)
    println("\n" * "="^100)
    println("ANOMALIES")
    println("="^100)

    # 1. Build errors
    println("\n--- BUILD ERRORS ---")
    bes = [r for r in records if r["outcome"] == "build_error"]
    if isempty(bes)
        println("  (none)")
    else
        for r in bes
            @printf("  %-40s target=%-5s feas=%-10s seed=%-3s\n          %s\n",
                    r["ref"], r["target"], r["feas_req"], r["seed"],
                    get(r, "build_error", "?")[1:min(160, end)])
        end
    end

    # 2. Feasibility contract violations
    println("\n--- FEASIBILITY CONTRACT VIOLATIONS ---")
    println("  (requested=feasible but status!=optimal; requested=infeasible but status!=infeasible)")
    vio = []
    for r in records
        r["outcome"] == "solved" || continue
        st = r["status"]
        if r["feas_req"] == "feasible" && st != "optimal"
            push!(vio, r)
        elseif r["feas_req"] == "infeasible" && st != "infeasible"
            push!(vio, r)
        end
    end
    if isempty(vio)
        println("  (none)")
    else
        for r in vio
            @printf("  %-40s target=%-5s feas=%-11s seed=%-3s -> status=%-13s iters=%-5s obj=%.4g\n",
                    r["ref"], r["target"], r["feas_req"], r["seed"],
                    r["status"], r["iterations"], r["objective_value"])
        end
    end

    # 3. Unexpected unbounded / errors / timeouts (for unknown requests too)
    println("\n--- UNEXPECTED STATUSES (excl. requested infeasible→infeasible) ---")
    bad = []
    for r in records
        r["outcome"] == "solved" || continue
        st = r["status"]
        if st in ("unbounded", "inf_or_unb", "error", "almost_optimal", "timeout", "other")
            push!(bad, r)
        end
    end
    if isempty(bad)
        println("  (none)")
    else
        for r in bad
            @printf("  %-40s target=%-5s feas=%-11s seed=%-3s -> status=%-13s term=%s\n",
                    r["ref"], r["target"], r["feas_req"], r["seed"],
                    r["status"], r["termination"])
        end
    end

    # 4. Trivial instances (very few iterations, or empty objective, or tiny)
    println("\n--- TRIVIAL INSTANCES (iterations<=2 on feasible/unknown) ---")
    triv = []
    for r in records
        r["outcome"] == "solved" || continue
        st = r["status"]
        ((st == "optimal") && (r["feas_req"] in ("feasible", "unknown")) &&
         (r["iterations"] in (0, 1, 2))) && push!(triv, r)
    end
    if isempty(triv)
        println("  (none)")
    else
        for r in triv
            @printf("  %-40s target=%-5s feas=%-11s seed=%-3s iters=%-3s nv=%-5s nc=%-5s objnnz=%-4s\n",
                    r["ref"], r["target"], r["feas_req"], r["seed"],
                    r["iterations"], r["num_variables"], r["num_constraints"], r["obj_nnz"])
        end
    end

    # 5. Empty / all-zero objective rows
    println("\n--- ALL-ZERO OBJECTIVE (obj_nnz==0) ---")
    zobj = [r for r in records if haskey(r, "obj_nnz") && r["obj_nnz"] == 0]
    if isempty(zobj)
        println("  (none)")
    else
        refs = unique([r["ref"] for r in zobj])
        @printf("  %d instances across %d variant(s): %s\n",
                length(zobj), length(refs), join(refs[1:min(15, end)], ", "))
    end

    # 6. Size-matching drift
    println("\n--- SIZE-MATCHING DRIFT (|actual-target|/target > 0.5) ---")
    drift = [r for r in records if haskey(r, "num_variables") &&
             r["num_variables"] > 0 &&
             abs(r["num_variables"] - r["target"]) / r["target"] > 0.5]
    if isempty(drift)
        println("  (none)")
    else
        # collapse to per-variant average ratio
        by_ref = Dict{String,Vector{Float64}}()
        for r in drift
            push!(get!(by_ref, r["ref"], Float64[]),
                  r["num_variables"] / r["target"])
        end
        for ref in sort(collect(keys(by_ref)))
            rs = by_ref[ref]
            @printf("  %-40s actual/target ratios: min=%.2f max=%.2f mean=%.2f (n=%d)\n",
                    ref, minimum(rs), maximum(rs), sum(rs)/length(rs), length(rs))
        end
    end

    # 7. Coefficient scaling (ratio > 1e8)
    println("\n--- EXTREME COEFFICIENT SCALING (max/min abs > 1e8) ---")
    scaled = [r for r in records if haskey(r, "coef_ratio") && r["coef_ratio"] > 1e8]
    if isempty(scaled)
        println("  (none)")
    else
        for r in scaled
            @printf("  %-40s target=%-5s feas=%-11s seed=%-3s ratio=%.2e objmax=%.3g\n",
                    r["ref"], r["target"], r["feas_req"], r["seed"],
                    r["coef_ratio"], r["obj_max_abs"])
        end
    end

    # 8. Degenerate (huge iterations relative to constraints)
    println("\n--- DEGENERATE-LIKE (iterations/constraints > 200) ---")
    degen = []
    for r in records
        r["outcome"] == "solved" || continue
        r["iterations"] >= 0 || continue
        r["num_constraints"] > 0 || continue
        if r["iterations"] / r["num_constraints"] > 200
            push!(degen, r)
        end
    end
    if isempty(degen)
        println("  (none)")
    else
        for r in degen
            @printf("  %-40s target=%-5s feas=%-11s seed=%-3s iters=%-6s nc=%-5s ratio=%.1f\n",
                    r["ref"], r["target"], r["feas_req"], r["seed"],
                    r["iterations"], r["num_constraints"],
                    r["iterations"]/r["num_constraints"])
        end
    end

    # 9. Non-deterministic generation
    println("\n--- NON-DETERMINISTIC (same seed rebuilt to different size) ---")
    nondet = [r for r in records if haskey(r, "deterministic") && !r["deterministic"]]
    if isempty(nondet)
        println("  (none)")
    else
        for r in nondet
            @printf("  %-40s target=%-5s feas=%-11s seed=%-3s\n",
                    r["ref"], r["target"], r["feas_req"], r["seed"])
        end
    end

    # Aggregate per-variant status table
    println("\n" * "="^100)
    println("PER-VARIANT STATUS TALLY (status counts across all size/status/seed combos)")
    println("="^100)
    by_ref = Dict{String,Dict{String,Int}}()
    for r in records
        d = get!(by_ref, r["ref"], Dict{String,Int}())
        key = r["outcome"] == "build_error" ? "BUILD_ERR" : r["status"]
        d[key] = get(d, key, 0) + 1
    end
    @printf("  %-40s %s\n", "variant", "  optimal infeasible unbounded almost_opt error timeout other builderr")
    for ref in sort(collect(keys(by_ref)))
        d = by_ref[ref]
        @printf("  %-40s   %7d %9d %9d %11d %5d %7d %5d %8d\n",
                ref,
                get(d, "optimal", 0), get(d, "infeasible", 0),
                get(d, "unbounded", 0), get(d, "almost_optimal", 0),
                get(d, "error", 0), get(d, "timeout", 0), get(d, "other", 0),
                get(d, "BUILD_ERR", 0))
    end
end

main()
