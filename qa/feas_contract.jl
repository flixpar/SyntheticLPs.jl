# Focused feasibility-contract audit.
#
# For every variant, request `feasible` and `infeasible` across many seeds and
# report the rate at which the solver disagrees (feasible-request not OPTIMAL,
# infeasible-request not INFEASIBLE). The broad sweep only used 2 seeds; this
# quantifies per-variant violation rates and confirms the root-caused cases.

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

const TARGET = 120
const SEEDS  = 1:12
const TIMEOUT = 20.0

function solve_status(ref, fstat, seed)
    m, _ = generate_problem(ref, TARGET, fstat, seed)
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    set_time_limit_sec(m, TIMEOUT)
    try set_attribute(m, "solver", "simplex") catch end
    optimize!(m)
    return termination_status(m)
end

is_feas_ok(ts) = ts == MOI.OPTIMAL
is_infeas_ok(ts) = ts == MOI.INFEASIBLE || ts == MOI.INFEASIBLE_OR_UNBOUNDED

function main()
    problems = list_problems()
    println("Feasibility-contract audit: target=$TARGET, seeds=$(first(SEEDS))-$(last(SEEDS)) ($(length(SEEDS)) each)")
    println()

    fea_bad = Dict{String,Vector{Int}}()   # ref -> seeds where feasible-request failed
    inf_bad = Dict{String,Vector{Int}}()   # ref -> seeds where infeasible-request failed
    inf_other = Dict{String,Vector{Tuple{Int,Any}}}()

    for (i, ref) in enumerate(problems)
        fb = Int[]; ib = Int[]; io = Tuple{Int,Any}[]
        for s in SEEDS
            # feasible request
            ts = solve_status(ref, feasible, s)
            is_feas_ok(ts) || push!(fb, s)
            # infeasible request
            ts2 = solve_status(ref, infeasible, s)
            if !is_infeas_ok(ts2)
                push!(ib, s)
                push!(io, (s, ts2))
            end
        end
        isempty(fb) || (fea_bad[string(ref)] = fb)
        isempty(ib) || (inf_bad[string(ref)] = ib)
        isempty(io) || (inf_other[string(ref)] = io)
        @printf("[%2d/%d] %-40s fea_viol=%2d/%d  inf_viol=%2d/%d\n",
                i, length(problems), ref, length(fb), length(SEEDS), length(ib), length(SEEDS))
    end

    println("\n" * "="^80)
    println("FEASIBLE-request violations (requested feasible, solver did NOT return OPTIMAL)")
    println("="^80)
    if isempty(fea_bad)
        println("  (none)")
    else
        for ref in sort(collect(keys(fea_bad)))
            @printf("  %-40s %d/%d seeds: %s\n", ref, length(fea_bad[ref]), length(SEEDS), join(string.(fea_bad[ref]), ","))
        end
    end

    println("\n" * "="^80)
    println("INFEASIBLE-request violations (requested infeasible, solver did NOT return INFEASIBLE)")
    println("="^80)
    if isempty(inf_bad)
        println("  (none)")
    else
        for ref in sort(collect(keys(inf_bad)))
            statuses = join(["s$s=$(io2)" for (s,io2) in inf_other[ref]], " ")
            @printf("  %-40s %2d/%d seeds (%.0f%%)  [%s]\n", ref, length(inf_bad[ref]),
                    length(SEEDS), 100*length(inf_bad[ref])/length(SEEDS), statuses)
        end
    end
end

main()
