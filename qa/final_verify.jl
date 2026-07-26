# Final end-to-end validation: every variant, feasible + infeasible requests,
# verified via the new optimizer kwarg. Must be 0 contract violations everywhere.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "scripts"))
using SyntheticLPs
import SyntheticLPs: feasible, infeasible, unknown, ProblemVariant
using JuMP, HiGHS
import MathOptInterface
const MOI = MathOptInterface
using Printf

const TARGET = 150
const SEEDS  = 1:6

function verified_status(ref, fstat, seed)
    m, _ = generate_problem(ref, TARGET, fstat, seed; optimizer=HiGHS.Optimizer)
    set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
    return termination_status(m)
end

problems = list_problems()
fea_bad = String[]; inf_bad = String[]
for (i, ref) in enumerate(problems)
    for s in SEEDS
        ts = verified_status(ref, feasible, s)
        (ts == MOI.OPTIMAL || ts == MOI.ALMOST_OPTIMAL) || push!(fea_bad, "$ref s=$s -> $ts")
        ts2 = verified_status(ref, infeasible, s)
        (ts2 == MOI.INFEASIBLE || ts2 == MOI.INFEASIBLE_OR_UNBOUNDED) || push!(inf_bad, "$ref s=$s -> $ts2")
    end
    @printf("[%2d/%d] %-34s  fea_bad=%d inf_bad=%d\n", i, length(problems), ref,
            count(x->startswith(x, string(ref)), fea_bad),
            count(x->startswith(x, string(ref)), inf_bad))
end

println("\n=== FEASIBLE-request violations: ", length(fea_bad))
foreach(println, fea_bad[1:min(20,end)])
println("\n=== INFEASIBLE-request violations: ", length(inf_bad))
foreach(println, inf_bad[1:min(20,end)])
println("\nTOTAL: fea=$(length(fea_bad)) inf=$(length(inf_bad)) violations across ",
        length(problems), " variants × ", length(SEEDS), " seeds × 2 statuses")
