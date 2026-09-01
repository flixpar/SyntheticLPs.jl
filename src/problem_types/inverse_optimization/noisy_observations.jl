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
remain feasible for the forward packing problem. Infeasibility is expressed as
a maximum mean-gap tolerance below a data-derived lower bound, rather than an
unrelated contradiction in the admissible cost set.
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
    gap_tolerance::Union{Nothing,Float64}
    resolved_status::FeasibilityStatus
    feasible_witness::Union{Nothing,NoisyInverseWitness}
    infeasibility_certificate::Union{Nothing,GapToleranceCertificate}
end

function _noisy_inverse_dimensions(target_variables::Int, preferred_k::Int)
    target = max(target_variables, 1)
    best = (error=typemax(Int), shape_error=Inf, activities=3, resources=2,
            observations=2, count=15)
    # Search resource counts and solve for the nearest activity count from
    # 3n + k*m + k = target. This is linear in the requested size; the former
    # nested n-by-m scan was quadratic and unusable near the documented cap.
    for k in max(2, preferred_k - 2):(preferred_k + 2)
        max_m = max(2, target ÷ (k + 6) + 3)
        for m in 2:max_m
            raw_n = (target - k * m - k) / 3
            for n in unique((max(3, floor(Int, raw_n)),
                             max(3, ceil(Int, raw_n))))
                m < n || continue
                m <= max(2, round(Int, n / 2)) || continue
                count = 3 * n + k * m + k
                candidate = (error=abs(count - target),
                             shape_error=abs(m / n - 0.30), activities=n,
                             resources=m, observations=k, count=count)
                (candidate.error, abs(k - preferred_k), candidate.shape_error, count) <
                    (best.error, abs(best.observations - preferred_k),
                     best.shape_error, best.count) &&
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
    _check_inverse_target(target_variables)
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
    dual_matrix = repeat(transpose(data.true_dual), K, 1)
    witness = feasibility_status == feasible ?
        NoisyInverseWitness(copy(data.true_cost), dual_matrix, true_gaps) : nothing
    certificate = nothing

    # Every admissible cost has c >= cost_lower, and each latent optimum is
    # feasible. Weak duality therefore gives a data-derived lower bound on the
    # mean gap at the observed decisions.
    gap_lower_bound = mean(
        sum(data.cost_lower[j] * (optimal[k, j] - observed[k, j]) for j in 1:n)
        for k in 1:K
    )
    gap_tolerance = nothing
    if feasibility_status == infeasible
        gap_tolerance = gap_lower_bound * rand(rng, Uniform(0.55, 0.90))
        certificate = GapToleranceCertificate(gap_lower_bound, gap_tolerance)
    elseif feasibility_status == unknown
        channel = rand(rng)
        if channel < 0.30
            # No fit threshold: always feasible, but no oracle metadata.
        elseif channel < 0.60
            gap_tolerance = gap_lower_bound * rand(rng, Uniform(0.65, 0.98))
        else
            planted_mean = mean(true_gaps)
            gap_tolerance = rand(rng, Uniform(0.95 * gap_lower_bound,
                                               1.08 * planted_mean))
        end
    end
    return NoisyInverseLPProblem(
        n, m, K, profile, data, observed, optimal, capacities,
        regularization, gap_scale, gap_tolerance, feasibility_status,
        witness, certificate,
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
    if prob.gap_tolerance !== nothing
        @constraint(model, fit_tolerance,
                    sum(suboptimality_gap) / K <= prob.gap_tolerance)
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
