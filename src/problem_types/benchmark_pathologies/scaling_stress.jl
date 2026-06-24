using JuMP
using Random
using Distributions

"""
    ScalingStressProblem <: ProblemGenerator

A feasible LP with coefficients and bounds spanning several orders of magnitude,
plus nearly redundant rows. Useful for numerical-scaling stress tests.
"""
struct ScalingStressProblem <: ProblemGenerator
    n_vars::Int
    n_constraints::Int
    A::Matrix{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
    upper::Vector{Float64}
end

function ScalingStressProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)
    n = max(5, target_variables)
    m = max(5, round(Int, 0.8 * n))
    upper = 10.0 .^ rand(Uniform(-2.0, 5.0), n)
    x0 = rand(Uniform(0.05, 0.8), n) .* upper
    density = min(0.35, max(0.04, 12 / n))
    A = zeros(Float64, m, n)
    for i in 1:m, j in 1:n
        if rand() < density
            A[i,j] = rand([-1.0, 1.0]) * 10.0^rand(Uniform(-4.0, 4.0))
        end
    end
    for i in 1:m
        if all(A[i,j] == 0.0 for j in 1:n)
            A[i, rand(1:n)] = 10.0^rand(Uniform(-3.0, 3.0))
        end
    end
    slack = 10.0 .^ rand(Uniform(-3.0, 3.0), m)
    b = A * x0 + slack
    if feasibility_status == infeasible
        row = rand(1:m)
        b[row] = minimum(A[row,j] >= 0 ? 0.0 : A[row,j] * upper[j] for j in 1:n) - abs(slack[row]) - 1.0
    end
    c = rand([-1.0, 1.0], n) .* (10.0 .^ rand(Uniform(-3.0, 3.0), n))
    return ScalingStressProblem(n, m, A, b, c, upper)
end

function build_model(prob::ScalingStressProblem)
    model = Model()
    @variable(model, 0 <= x[j=1:prob.n_vars] <= prob.upper[j])
    @objective(model, Min, sum(prob.c[j] * x[j] for j in 1:prob.n_vars))
    for i in 1:prob.n_constraints
        @constraint(model, sum(prob.A[i,j] * x[j] for j in 1:prob.n_vars) <= prob.b[i])
    end
    return model
end

register_variant(:benchmark_pathologies, :scaling_stress, ScalingStressProblem,
    "Numerically stressed bounded LP with sparse coefficients and bounds spanning many orders of magnitude.")
