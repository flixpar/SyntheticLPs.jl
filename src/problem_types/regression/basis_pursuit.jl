using JuMP
using Random
using Distributions

"""
    BasisPursuitProblem <: ProblemGenerator

Dense sparse-recovery LP: minimize the L1 norm of coefficients subject to exact
linear measurements, represented with positive/negative variable splits.
"""
struct BasisPursuitProblem <: ProblemGenerator
    n_features::Int
    n_measurements::Int
    A::Matrix{Float64}
    b::Vector{Float64}
    weights::Vector{Float64}
end

function BasisPursuitProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)
    n_features = max(4, target_variables ÷ 2)
    n_measurements = max(2, round(Int, rand(Uniform(0.25, 0.65)) * n_features))
    A = rand(Normal(0.0, 1.0), n_measurements, n_features)
    scale = rand(LogNormal(0.0, 0.6), n_features)
    A = A .* reshape(scale, 1, n_features)
    x_true = zeros(Float64, n_features)
    sparsity = max(1, round(Int, rand(Uniform(0.05, 0.18)) * n_features))
    for j in randperm(n_features)[1:sparsity]
        x_true[j] = rand(Normal(0.0, 2.0))
    end
    b = A * x_true
    if feasibility_status == infeasible
        # Duplicate one measurement row with an inconsistent right-hand side.
        if n_measurements >= 2
            A[end, :] = A[1, :]
            b[end] = b[1] + rand(Uniform(1.0, 5.0))
        end
    end
    weights = rand(Uniform(0.5, 2.0), n_features)
    return BasisPursuitProblem(n_features, n_measurements, A, b, weights)
end

function build_model(prob::BasisPursuitProblem)
    model = Model()
    @variable(model, x_pos[1:prob.n_features] >= 0)
    @variable(model, x_neg[1:prob.n_features] >= 0)
    @objective(model, Min, sum(prob.weights[j] * (x_pos[j] + x_neg[j]) for j in 1:prob.n_features))
    for i in 1:prob.n_measurements
        @constraint(model, sum(prob.A[i,j] * (x_pos[j] - x_neg[j]) for j in 1:prob.n_features) == prob.b[i])
    end
    return model
end

register_variant(:regression, :basis_pursuit, BasisPursuitProblem,
    "Dense sparse-recovery/basis-pursuit LP with split variables, equality measurements, and optional inconsistent measurements.")
