"""Feasible parameter point for a multi-observation inverse LP."""
struct NoisyInverseWitness
    cost::Vector{Float64}
    dual_prices::Matrix{Float64}
    gaps::Vector{Float64}
end

"""
    NoisyInverseLPProblem <: ProblemGenerator

Data-driven inverse LP using average absolute suboptimality (duality gap) over
multiple feasible observations. Each observation has context-specific resource
capacities. Routine under-utilization, heterogeneous behavior, or a contaminated
panel creates nonzero but controlled suboptimality, while all observations
remain feasible for the forward packing problem.
"""
struct NoisyInverseLPProblem <: ProblemGenerator
    n_activities::Int
    n_resources::Int
    n_observations::Int
    profile::Symbol
    data::InversePackingData
    observed_decisions::Matrix{Float64}
    optimal_decisions::Matrix{Float64}
    capacities::Matrix{Float64}
    regularization::Float64
    gap_scale::Float64
    resolved_status::FeasibilityStatus
    feasible_witness::Union{Nothing,NoisyInverseWitness}
    infeasibility_certificate::Union{Nothing,InverseCostSetCertificate}
end

function _noisy_inverse_dimensions(target_variables::Int, preferred_k::Int)
    target = max(target_variables, 1)
    best = (error=typemax(Int), activities=3, resources=2,
            observations=2, count=15)
    max_n = max(3, cld(target, 3) + 3)
    for n in 3:max_n
        for m in 2:min(n - 1, max(2, round(Int, n / 2)))
            for k in max(2, preferred_k - 2):(preferred_k + 2)
                count = 3 * n + k * m + k
                candidate = (error=abs(count - target), activities=n,
                             resources=m, observations=k, count=count)
                (candidate.error, abs(k - preferred_k), count) <
                    (best.error, abs(best.observations - preferred_k), best.count) &&
                    (best = candidate)
            end
        end
    end
    return best.activities, best.resources, best.observations
end

function _noisy_utilization(
    rng::AbstractRNG,
    profile::Symbol,
    n_observations::Int,
    n_activities::Int,
)
    utilization = Matrix{Float64}(undef, n_observations, n_activities)
    for k in 1:n_observations, j in 1:n_activities
        loss = if profile == :routine
            0.01 + 0.11 * rand(rng, Beta(2.0, 7.0))
        elseif profile == :heterogeneous
            0.02 + 0.20 * rand(rng, Beta(1.8, 3.8))
        else
            0.01 + 0.13 * rand(rng, Beta(2.0, 6.0))
        end
        utilization[k, j] = 1.0 - loss
    end
    if profile == :outlier_contaminated
        outlier = rand(rng, 1:n_observations)
        utilization[outlier, :] .*= 0.62 + 0.15 * rand(rng)
    end
    return utilization
end

function NoisyInverseLPProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = MersenneTwister(seed)
    profiles = (:routine, :heterogeneous, :outlier_contaminated)
    profile = profiles[rand(rng, eachindex(profiles))]
    preferred_k = target_variables < 150 ? 3 : target_variables < 800 ? 6 : 9
    n, m, K = _noisy_inverse_dimensions(target_variables, preferred_k)
    data = _inverse_packing_data(rng, m, n)

    base_plan = rand(rng, LogNormal(log(16.0), 0.42), n)
    optimal = Matrix{Float64}(undef, K, n)
    for k in 1:K, j in 1:n
        optimal[k, j] = base_plan[j] * rand(rng, LogNormal(0.0, 0.20))
    end
    utilization = _noisy_utilization(rng, profile, K, n)
    observed = optimal .* utilization
    capacities = Matrix{Float64}(undef, K, m)
    for k in 1:K
        capacities[k, :] = data.consumption * @view(optimal[k, :])
    end

    true_gaps = [
        dot(@view(capacities[k, :]), data.true_dual) -
        dot(@view(observed[k, :]), data.true_cost)
        for k in 1:K
    ]
    gap_scale = max(1.0, mean(
        dot(@view(observed[k, :]), data.true_cost) for k in 1:K
    ))
    regularization = rand(rng, Uniform(0.03, 0.18))
    resolved_status = _inverse_resolved_status(rng, feasibility_status)
    dual_matrix = repeat(transpose(data.true_dual), K, 1)
    witness = resolved_status == feasible ?
        NoisyInverseWitness(copy(data.true_cost), dual_matrix, true_gaps) : nothing
    certificate = resolved_status == infeasible ?
        _inverse_cost_certificate(true, data.true_cost) : nothing
    return NoisyInverseLPProblem(
        n, m, K, profile, data, observed, optimal, capacities,
        regularization, gap_scale, resolved_status, witness, certificate,
    )
end

function build_model(prob::NoisyInverseLPProblem)
    model = Model()
    data = prob.data
    n, m, K = prob.n_activities, prob.n_resources, prob.n_observations

    @variable(model, data.cost_lower[j] <= inferred_cost[j in 1:n] <= data.cost_upper[j])
    @variable(model, shadow_price[1:K, 1:m] >= 0)
    @variable(model, suboptimality_gap[1:K] >= 0)
    @variable(model, deviation_positive[1:n] >= 0)
    @variable(model, deviation_negative[1:n] >= 0)
    @objective(
        model,
        Min,
        sum(suboptimality_gap) / (K * prob.gap_scale) +
        prob.regularization *
        sum(data.deviation_weight[j] *
            (deviation_positive[j] + deviation_negative[j]) for j in 1:n),
    )
    @constraint(model, cost_normalization, sum(inferred_cost) == 1.0)
    @constraint(
        model,
        dual_feasibility[k in 1:K, j in 1:n],
        _inverse_column_expression(data.consumption, @view(shadow_price[k, :]), j) >=
            inferred_cost[j],
    )
    @constraint(
        model,
        gap_definition[k in 1:K],
        suboptimality_gap[k] ==
            sum(prob.capacities[k, i] * shadow_price[k, i] for i in 1:m) -
            sum(prob.observed_decisions[k, j] * inferred_cost[j] for j in 1:n),
    )
    @constraint(
        model,
        prior_deviation[j in 1:n],
        inferred_cost[j] - data.prior_cost[j] ==
            deviation_positive[j] - deviation_negative[j],
    )
    if prob.infeasibility_certificate !== nothing
        certificate = prob.infeasibility_certificate
        @constraint(model, inadmissible_cost_mass,
                    sum(inferred_cost) <= certificate.total_upper)
    end
    return model
end

function _noisy_inverse_witness_is_valid(prob::NoisyInverseLPProblem)
    witness = prob.feasible_witness
    witness isa NoisyInverseWitness || return false
    data = prob.data
    dual_feasibility = transpose(data.consumption) * transpose(witness.dual_prices)
    observed_feasible = all(
        k -> all(
            data.consumption * @view(prob.observed_decisions[k, :]) .<=
                @view(prob.capacities[k, :]) .+ 1.0e-9,
        ),
        1:prob.n_observations,
    )
    gaps = [
        dot(@view(prob.capacities[k, :]), @view(witness.dual_prices[k, :])) -
        dot(@view(prob.observed_decisions[k, :]), witness.cost)
        for k in 1:prob.n_observations
    ]
    return observed_feasible && all(witness.cost .>= data.cost_lower .- 1.0e-10) &&
           all(witness.cost .<= data.cost_upper .+ 1.0e-10) &&
           isapprox(sum(witness.cost), 1.0; atol=1.0e-10) &&
           all(dual_feasibility .>= reshape(witness.cost, :, 1) .- 1.0e-10) &&
           all(gaps .>= -1.0e-9) && isapprox(gaps, witness.gaps; atol=1.0e-9)
end

register_variant(
    :inverse_optimization,
    :noisy_observations,
    NoisyInverseLPProblem,
    "Multi-observation inverse LP minimizing normalized absolute suboptimality with realistic behavioral noise",
)
