using JuMP
using Random
using Distributions

"""
    WorkforceShiftCoveringProblem <: ProblemGenerator

Multi-skill shift-covering LP for contact-center or retail staffing. Shift-pool
variables buy labor patterns, while undercoverage variables keep realistic
unknown cases solvable with expensive service-level penalties.
"""
struct WorkforceShiftCoveringProblem <: ProblemGenerator
    n_periods::Int
    n_skills::Int
    n_pools::Int
    n_patterns::Int
    coverage::Array{Float64,3}
    demand::Matrix{Float64}
    wage_cost::Matrix{Float64}
    max_workers::Matrix{Float64}
    under_penalty::Matrix{Float64}
    min_total_hours::Float64
end

function _choose_workforce_dims(target_variables::Int)
    best = (Inf, 12, 2, 3, 8)
    for periods in (12, 16, 24, 32, 48), skills in 1:5, pools in 2:8, patterns in 4:40
        vars = pools * patterns + periods * skills
        err = abs(vars - target_variables)
        if err < best[1]
            best = (err, periods, skills, pools, patterns)
        end
    end
    return best[2], best[3], best[4], best[5]
end

function WorkforceShiftCoveringProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)
    n_periods, n_skills, n_pools, n_patterns = _choose_workforce_dims(target_variables)

    coverage = zeros(Float64, n_periods, n_skills, n_patterns)
    starts = collect(1:n_periods)
    for r in 1:n_patterns
        start = rand(starts)
        len = rand(DiscreteUniform(max(2, n_periods ÷ 8), max(3, n_periods ÷ 3)))
        break_at = start + rand(DiscreteUniform(1, max(1, len - 1)))
        primary_skill = rand(1:n_skills)
        for h in 0:(len-1)
            tt = mod(start + h - 1, n_periods) + 1
            if start + h != break_at
                coverage[tt, primary_skill, r] = 1.0
                for s in 1:n_skills
                    if s != primary_skill && rand() < 0.25
                        coverage[tt, s, r] = rand(Uniform(0.35, 0.8))
                    end
                end
            end
        end
    end

    demand = zeros(Float64, n_periods, n_skills)
    for t in 1:n_periods, s in 1:n_skills
        peak1 = exp(-((t - 0.35 * n_periods) / (0.18 * n_periods))^2)
        peak2 = exp(-((t - 0.72 * n_periods) / (0.16 * n_periods))^2)
        demand[t,s] = rand(Uniform(4.0, 16.0)) * (0.65 + peak1 + 0.8 * peak2) * rand(Uniform(0.85, 1.2))
    end

    wage_cost = zeros(Float64, n_pools, n_patterns)
    max_workers = zeros(Float64, n_pools, n_patterns)
    for q in 1:n_pools, r in 1:n_patterns
        active_hours = sum(maximum(coverage[t, :, r]) for t in 1:n_periods)
        wage_cost[q,r] = active_hours * rand(Uniform(18.0, 55.0)) * (1.0 + 0.08 * q)
        max_workers[q,r] = rand(Uniform(6.0, 35.0))
    end
    under_penalty = rand(Uniform(150.0, 450.0), n_periods, n_skills)
    min_total_hours = feasibility_status == infeasible ? 1.35 * sum(max_workers) : 0.0

    return WorkforceShiftCoveringProblem(n_periods, n_skills, n_pools, n_patterns,
        coverage, demand, wage_cost, max_workers, under_penalty, min_total_hours)
end

function build_model(prob::WorkforceShiftCoveringProblem)
    model = Model()
    @variable(model, 0 <= workers[q=1:prob.n_pools, r=1:prob.n_patterns] <= prob.max_workers[q,r])
    @variable(model, under[1:prob.n_periods, 1:prob.n_skills] >= 0)
    @objective(model, Min,
        sum(prob.wage_cost[q,r] * workers[q,r] for q in 1:prob.n_pools, r in 1:prob.n_patterns) +
        sum(prob.under_penalty[t,s] * under[t,s] for t in 1:prob.n_periods, s in 1:prob.n_skills))
    for t in 1:prob.n_periods, s in 1:prob.n_skills
        @constraint(model, sum(prob.coverage[t,s,r] * workers[q,r] for q in 1:prob.n_pools, r in 1:prob.n_patterns) + under[t,s] >= prob.demand[t,s])
    end
    @constraint(model, sum(workers) >= prob.min_total_hours)
    return model
end

register_variant(:workforce_shift_scheduling, :covering, WorkforceShiftCoveringProblem,
    "Generic multi-skill shift-covering/contact-center staffing LP with patterns, labor pools, demand curves, and undercoverage penalties.")
