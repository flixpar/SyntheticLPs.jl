"""Exact primal/dual certificate stored for a classical inverse LP."""
struct ClassicalInverseWitness
    cost::Vector{Float64}
    dual_prices::Vector{Float64}
end

"""
    ClassicalInverseLPProblem <: ProblemGenerator

Classical inverse linear optimization under a weighted L1 norm. The forward
problem is a nonnegative packing LP. Given one observed optimal activity plan,
the inverse model minimally adjusts a prior normalized profit vector while
enforcing dual feasibility and zero duality gap.

The generator plants a strictly positive plan, exhausts every resource at that
plan, and constructs the true profit vector from positive resource shadow
prices. This gives an inspectable exact optimality witness without solving a
forward problem in the constructor.
"""
struct ClassicalInverseLPProblem <: ProblemGenerator
    n_activities::Int
    n_resources::Int
    data::InversePackingData
    observed_decision::Vector{Float64}
    capacity::Vector{Float64}
    resolved_status::FeasibilityStatus
    feasible_witness::Union{Nothing,ClassicalInverseWitness}
    infeasibility_certificate::Union{Nothing,InverseCostSetCertificate}
end

function _classical_inverse_dimensions(target_variables::Int, ratio::Float64)
    target = max(target_variables, 1)
    best = (error=typemax(Int), activities=3, resources=2, count=11)
    max_activities = max(3, cld(target, 3) + 3)
    for n in 3:max_activities
        preferred = clamp(round(Int, n / ratio), 2, n - 1)
        for m in max(2, preferred - 2):min(n - 1, preferred + 2)
            count = 3 * n + m
            candidate = (error=abs(count - target), activities=n,
                         resources=m, count=count)
            (candidate.error, abs(m - preferred), count) <
                (best.error, abs(best.resources - preferred), best.count) &&
                (best = candidate)
        end
    end
    return best.activities, best.resources
end

function ClassicalInverseLPProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = MersenneTwister(seed)
    ratio = rand(rng, Uniform(2.5, 6.5))
    n_activities, n_resources =
        _classical_inverse_dimensions(target_variables, ratio)
    data = _inverse_packing_data(rng, n_resources, n_activities)

    observed = rand(rng, LogNormal(log(18.0), 0.48), n_activities)
    capacity = Vector(data.consumption * observed)
    resolved_status = _inverse_resolved_status(rng, feasibility_status)
    witness = resolved_status == feasible ?
        ClassicalInverseWitness(copy(data.true_cost), copy(data.true_dual)) : nothing
    certificate = resolved_status == infeasible ?
        _inverse_cost_certificate(true, data.true_cost) : nothing
    return ClassicalInverseLPProblem(
        n_activities,
        n_resources,
        data,
        observed,
        capacity,
        resolved_status,
        witness,
        certificate,
    )
end

function build_model(prob::ClassicalInverseLPProblem)
    model = Model()
    data = prob.data
    n, m = prob.n_activities, prob.n_resources

    @variable(model, data.cost_lower[j] <= inferred_cost[j in 1:n] <= data.cost_upper[j])
    @variable(model, shadow_price[1:m] >= 0)
    @variable(model, deviation_positive[1:n] >= 0)
    @variable(model, deviation_negative[1:n] >= 0)

    @objective(
        model,
        Min,
        sum(data.deviation_weight[j] *
            (deviation_positive[j] + deviation_negative[j]) for j in 1:n),
    )
    @constraint(model, cost_normalization, sum(inferred_cost) == 1.0)
    @constraint(
        model,
        stationarity[j in 1:n],
        _inverse_column_expression(data.consumption, shadow_price, j) ==
            inferred_cost[j],
    )
    @constraint(
        model,
        prior_deviation[j in 1:n],
        inferred_cost[j] - data.prior_cost[j] ==
            deviation_positive[j] - deviation_negative[j],
    )
    @constraint(
        model,
        strong_duality,
        sum(prob.observed_decision[j] * inferred_cost[j] for j in 1:n) ==
            sum(prob.capacity[i] * shadow_price[i] for i in 1:m),
    )
    if prob.infeasibility_certificate !== nothing
        certificate = prob.infeasibility_certificate
        @constraint(model, inadmissible_cost_mass,
                    sum(inferred_cost) <= certificate.total_upper)
    end
    return model
end

function _classical_inverse_witness_is_valid(prob::ClassicalInverseLPProblem)
    witness = prob.feasible_witness
    witness isa ClassicalInverseWitness || return false
    data = prob.data
    return all(witness.cost .>= data.cost_lower .- 1.0e-10) &&
           all(witness.cost .<= data.cost_upper .+ 1.0e-10) &&
           all(witness.dual_prices .>= 0.0) &&
           isapprox(sum(witness.cost), 1.0; atol=1.0e-10) &&
           isapprox(transpose(data.consumption) * witness.dual_prices,
                    witness.cost; atol=1.0e-10) &&
           isapprox(dot(prob.observed_decision, witness.cost),
                    dot(prob.capacity, witness.dual_prices); atol=1.0e-9)
end

register_variant(
    :inverse_optimization,
    :classical,
    ClassicalInverseLPProblem,
    "Classical weighted-L1 inverse LP with a planted exact decision and strong-duality certificate";
    default=true,
)
