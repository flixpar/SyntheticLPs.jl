# Drill down on the root-caused feasibility violations: reproduce each and
# print the structural evidence that explains WHY the infeasible-request
# produced a feasible model.

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
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    set_time_limit_sec(m, 30.0)
    try set_attribute(m, "solver", "simplex") catch end
    optimize!(m)
    termination_status(m), objective_value(m)
end

println("="^80)
println("CROP PLANNING — infeasible-request root cause (fallow-land hole)")
println("="^80)
# Find a seed where infeasible-request comes back feasible.
for seed in 1:12
    global crop_seed, crop_problem, crop_model
    m, p = generate_problem(ProblemVariant("crop_planning/standard"), 120, infeasible, seed)
    ts, obj = solve!(m)
    if ts == MOI.OPTIMAL
        println("seed=$seed  ->  OPTIMAL (obj=$(round(obj, digits=2)))  [VIOLATION]")
        total_land = p.total_land
        min_area = p.min_area_per_crop
        true_min_water = sum(p.water_requirements .* min_area)     # plant mandatory only
        true_min_labor = sum(p.labor_requirements .* min_area)
        println(@sprintf("  total_land               = %.2f", total_land))
        println(@sprintf("  sum(min_area_per_crop)   = %.2f  (= mandatory planting)", sum(min_area)))
        println(@sprintf("  land constraint sense    = sum(x) <= total_land   (UPPER bound only!)"))
        println(@sprintf("  water_capacity (set)     = %.2f", p.water_capacity))
        println(@sprintf("  TRUE min water usage     = %.2f  (plant only mandatory, leave rest fallow)",
                         true_min_water))
        println(@sprintf("  -> water_capacity > true_min? %s  => water constraint satisfiable",
                         p.water_capacity > true_min_water + 1e-6))
        println(@sprintf("  labor_capacity (set)     = %.2f", p.labor_capacity))
        println(@sprintf("  TRUE min labor usage     = %.2f", true_min_labor))
        global crop_seed = seed
        break
    end
end

println()
println("="^80)
println("ENERGY — emissions constraint is an algebraic tautology")
println("="^80)
m, p = generate_problem(ProblemVariant("energy/standard"), 120, feasible, 1)
max_em = maximum(values(p.emission_limits))
println("emission_limits per source:")
for s in p.sources
    println(@sprintf("    %-10s emission_limit = %.4f", s, p.emission_limits[s]))
end
println(@sprintf("  max_emission = %.4f", max_em))
println("  Emissions row built as:  Σ em_s·x_s,t  <=  max_emission · Σ x_s,t")
println("  Since em_s <= max_emission for every s, the LHS (weighted avg × total)")
println("  is ALWAYS <= RHS — the constraint can never bind. It is dead weight.")
# count constraints
nc = num_constraints(m; count_variable_in_set_constraints=false)
println(@sprintf("  -> %d affine constraints of which %d are these no-op emissions rows.",
                 nc, length(p.time_periods)))

println()
println("Reproduce energy infeasible-request violation:")
for seed in 1:12
    m, p = generate_problem(ProblemVariant("energy/standard"), 120, infeasible, seed)
    ts, obj = solve!(m)
    flag = ts == MOI.OPTIMAL ? " [VIOLATION]" : ""
    println(@sprintf("  seed=%-2d -> %-22s obj=%.4g%s", seed, ts, obj, flag))
end

println()
println("="^80)
println("BLENDING — infeasible-request violations across seeds")
println("="^80)
for seed in 1:12
    m, p = generate_problem(ProblemVariant("blending/standard"), 120, infeasible, seed)
    ts, obj = solve!(m)
    flag = (ts == MOI.OPTIMAL) ? " [VIOLATION]" : ""
    println(@sprintf("  seed=%-2d -> %-22s obj=%.4g%s", seed, ts, obj, flag))
end
