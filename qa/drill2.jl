# Drill v2: reproduce the size-dependent energy/blending infeasibility violations
# (they appear at target=300, not 120) and add feed_blending + crop_planning@300.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "scripts"))
Pkg.instantiate()

using SyntheticLPs
import SyntheticLPs: feasible, infeasible, ProblemVariant
using JuMP
import MathOptInterface
const MOI = MathOptInterface
using HiGHS
using Printf

function solve!(m)
    set_optimizer(m, HiGHS.Optimizer); set_silent(m)
    set_time_limit_sec(m, 30.0)
    try set_attribute(m, "solver", "simplex") catch end
    optimize!(m)
    termination_status(m), objective_value(m)
end

for ref in ("energy/standard", "blending/standard", "feed_blending/standard", "crop_planning/standard")
    println("="^70)
    println("INFEASIBLE-request @ target=300 : ", ref)
    println("="^70)
    n_viol = 0
    for seed in 1:12
        m, _ = generate_problem(ProblemVariant(ref), 300, infeasible, seed)
        ts, obj = solve!(m)
        flag = ts == MOI.OPTIMAL ? " [VIOLATION]" : ""
        if ts == MOI.OPTIMAL
            n_viol += 1
        end
        println(@sprintf("  seed=%-2d -> %-22s obj=%.6g%s", seed, ts, obj, flag))
    end
    @printf("  -> %d/12 infeasible-request violations\n\n", n_viol)
end
