using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "scripts"))
using SyntheticLPs
import SyntheticLPs: infeasible, ProblemVariant
using JuMP
using HiGHS
import MathOptInterface
const MOI = MathOptInterface
using Printf

for seed in (1, 10, 2, 3)   # 1,10 violated; 2,3 were correctly infeasible
    println("="^64)
    m, p = generate_problem(ProblemVariant("energy/standard"), 300, infeasible, seed)
    total_cap = sum(values(p.capacities))
    max_demand = maximum(p.demands)
    println(@sprintf("seed=%d  total_capacity(Σcapacities)=%.2f  max_demand=%.2f  max_demand/cap=%.3f",
                     seed, total_cap, max_demand, max_demand / total_cap))
    set_optimizer(m, HiGHS.Optimizer); set_silent(m); set_time_limit_sec(m, 30.0)
    optimize!(m)
    println(@sprintf("  -> status=%s  (model only enforces Σx>=demand & x<=capacities[s])",
                     termination_status(m)))
end
