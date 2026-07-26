# Edge-case + true-MIP + bounds_to_constraints audit.
#
# 1. TINY target (n=3) for every variant — catch sizing crashes / degenerate dims.
# 2. LARGE target (n=2000) — catch runaway sizing / timeouts.
# 3. relax_integer=false on the documented MIP variants — verify they build and
#    solve as genuine MIPs (HiGHS MIP), and report integrality gap.
# 4. bounds_to_constraints=true spot check on a few variants — verify it doesn't
#    change feasibility/sense.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "scripts"))
Pkg.instantiate()

using SyntheticLPs
import SyntheticLPs: feasible, infeasible, unknown, ProblemVariant
using JuMP
import MathOptInterface
const MOI = MathOptInterface
using HiGHS
using Printf

function solve!(m; mip=false)
    set_optimizer(m, HiGHS.Optimizer); set_silent(m)
    set_time_limit_sec(m, 30.0)
    mip || try set_attribute(m, "solver", "simplex") catch end
    optimize!(m)
    termination_status(m), objective_value(m)
end

# Documented MIP variants (from CLAUDE.md "Model classes").
const MIP_VARIANTS = String[
    "facility_location/standard", "facility_location/two_echelon", "facility_location/p_median",
    "cutting_stock/setup_cost",
    "inventory/lot_sizing",
    "bin_packing/standard",
    "job_shop_scheduling/standard",
    "supply_chain/single_source",
    "knapsack/multidimensional", "knapsack/bounded",
    "assignment/workload_balance",
    "vehicle_routing/cvrp",
]

println("="^70)
println("1. EDGE SIZES: tiny (n=3) and large (n=2000) — build only")
println("="^70)
problems = list_problems()
tiny_bad = String[]
large_bad = String[]
for ref in problems
    for (target, bucket) in ((3, tiny_bad), (2000, large_bad))
        try
            m, _ = generate_problem(ref, target, unknown, 1)
            nv = num_variables(m)
            (nv <= 0) && push!(bucket, "$(ref)@$(target): 0 vars")
        catch e
            push!(bucket, "$(ref)@$(target): $(typeof(e)): $(sprint(showerror, e)[1:min(120,end)])")
        end
    end
end
println("TINY (n=3) issues: "); foreach(x->println("  ", x), isempty(tiny_bad) ? ["(none)"] : tiny_bad)
println("LARGE (n=2000) issues: "); foreach(x->println("  ", x), isempty(large_bad) ? ["(none)"] : large_bad)

println()
println("="^70)
println("2. TRUE MIPs (relax_integer=false) — build + solve with HiGHS MIP")
println("="^70)
@printf("  %-36s %6s %9s %12s %12s %10s\n", "variant", "nvars", "nintvars", "LPrelax", "MIPobj", "gap%")
for ref in MIP_VARIANTS
    pv = ProblemVariant(ref)
    ok = true
    try
        # MIP
        mm, _ = generate_problem(pv, 80, unknown, 1; relax_integer=false)
        nint = try length([v for v in all_variables(mm) if is_integer(mm, v)]) catch; -1 end
        nv = num_variables(mm)
        ts_m, obj_m = solve!(mm; mip=true)
        # LP relaxation
        ml, _ = generate_problem(pv, 80, unknown, 1; relax_integer=true)
        ts_l, obj_l = solve!(ml)
        gap = (isfinite(obj_l) && isfinite(obj_m) && abs(obj_m) > 1e-6) ?
              100*abs(obj_l - obj_m)/abs(obj_m) : NaN
        sense = objective_sense(mm)
        @printf("  %-36s %6d %9d %12.4g %12.4g %9.2f  [%s]\n",
                ref, nv, nint, obj_l, obj_m, gap, ts_m)
    catch e
        @printf("  %-36s BUILD/SOLVE ERROR: %s\n", ref, typeof(e))
    end
end

println()
println("="^70)
println("3. bounds_to_constraints=true — feasibility/sense preserved?")
println("="^70)
spot = ["transportation/standard", "knapsack/bounded", "portfolio/cvar", "diet_problem/standard", "unit_commitment/standard"]
for ref in spot
    pv = ProblemVariant(ref)
    for fstat in (feasible, infeasible)
        m1, _ = generate_problem(pv, 120, fstat, 3; bounds_to_constraints=false)
        m2, _ = generate_problem(pv, 120, fstat, 3; bounds_to_constraints=true)
        t1, o1 = solve!(m1)
        t2, o2 = solve!(m2)
        same = (t1 == t2)
        # also check vars unchanged
        nv1, nv2 = num_variables(m1), num_variables(m2)
        nc1 = num_constraints(m1; count_variable_in_set_constraints=false)
        nc2 = num_constraints(m2; count_variable_in_set_constraints=false)
        mark = same ? "" : "  <-- STATUS CHANGED"
        @printf("  %-26s %-10s plain:%-12s b2c:%-12s vars=%d/%d cons=%d/%d%s\n",
                ref, fstat, t1, t2, nv1, nv2, nc1, nc2, mark)
    end
end
