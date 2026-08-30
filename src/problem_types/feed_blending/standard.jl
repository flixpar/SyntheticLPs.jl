using JuMP
using Random
using Distributions

"""Broad economic role of an ingredient in a generated feed recipe."""
@enum FeedIngredientKind begin
    feed_energy_source
    feed_protein_source
    feed_mineral_supplement
    feed_specialty_additive
end

"""Nutritional role and scale of a generated quality metric."""
@enum FeedNutrientKind begin
    feed_major_nutrient
    feed_mineral
    feed_trace_nutrient
    feed_restricted_compound
end

"""Direction of an average-content constraint."""
@enum FeedRatioSense begin
    feed_ratio_minimum
    feed_ratio_maximum
end

"""
    FeedRatioConstraint

A typed average-content bound. `target` has the same concentration unit as row
`nutrient` of `nutrient_content`; `sense` determines whether it is a lower or
upper bound. Keeping the direction separate from diagnostic labels prevents a
maximum certificate such as "below achievable minimum" from being parsed as a
minimum constraint.
"""
struct FeedRatioConstraint
    nutrient::Int
    target::Float64
    sense::FeedRatioSense
end

"""Structural reason that a requested-infeasible blend has no feasible recipe."""
@enum FeedInfeasibilityKind begin
    feed_minimum_ratio_above_achievable_maximum
    feed_maximum_ratio_below_achievable_minimum
    feed_minimum_nutrient_above_achievable_maximum
    feed_insufficient_ingredient_capacity
end

"""
    FeedInfeasibilityCertificate

Solver-independent certificate stored on requested-infeasible instances.
`achievable_bound` and `required_bound` are average concentrations for ratio
certificates, total nutrient amounts for a nutrient-minimum certificate, and
ingredient mass for an availability certificate. `nutrient == 0` and
`ratio_constraint == 0` mean that the corresponding index is not applicable.
"""
struct FeedInfeasibilityCertificate
    kind::FeedInfeasibilityKind
    nutrient::Int
    ratio_constraint::Int
    achievable_bound::Float64
    required_bound::Float64
end

const _FEED_INGREDIENT_KINDS = (
    feed_energy_source,
    feed_protein_source,
    feed_mineral_supplement,
    feed_specialty_additive,
)

const _FEED_NUTRIENT_KINDS = (
    feed_major_nutrient,
    feed_mineral,
    feed_trace_nutrient,
    feed_restricted_compound,
)

# Rows are nutrient kinds and columns are ingredient kinds. The values describe
# typical concentration medians and occurrence probabilities. This creates the
# expected economic/nutritional correlations: protein ingredients are rich in
# major nutrients, mineral supplements carry concentrated minerals, and specialty
# additives are more likely to contain trace or restricted compounds.
const _FEED_CONTENT_MEDIAN = [
    14.0 32.0 5.0 2.0
    0.6 1.1 7.0 1.8
    0.03 0.08 0.8 0.35
    1.5 2.5 0.8 4.5
]

const _FEED_CONTENT_PROBABILITY = [
    1.00 1.00 1.00 1.00
    0.75 0.85 0.98 0.90
    0.25 0.45 0.98 0.85
    0.55 0.65 0.40 0.90
]

const _FEED_CONTENT_MAXIMUM = (55.0, 15.0, 3.0, 20.0)
const _FEED_COST_MEDIAN = (0.28, 0.48, 0.95, 2.40)
const _FEED_COST_LOG_SIGMA = (0.28, 0.32, 0.38, 0.50)

"""
    FeedBlendingProblem <: ProblemGenerator

Continuous least-cost feed formulation with a fixed batch mass, ingredient
availability, nutrient floors/caps, and typed average-content bounds.

The generator distinguishes four realistic ingredient roles and four nutrient
roles. Requested-feasible instances store a recipe that satisfies every row.
Requested-infeasible instances start from the same feasible baseline and then
store one checkable structural certificate. Unknown instances draw individually
reasonable requirements without claiming a joint feasibility outcome.
"""
struct FeedBlendingProblem <: ProblemGenerator
    num_ingredients::Int
    num_nutrients::Int
    batch_size::Float64
    ingredient_types::Vector{FeedIngredientKind}
    costs::Vector{Float64}
    nutrient_content::Matrix{Float64}
    nutrient_types::Vector{FeedNutrientKind}
    min_requirements::Vector{Float64}
    max_limits::Vector{Float64}
    availabilities::Vector{Float64}
    ratio_constraints::Vector{FeedRatioConstraint}
    feasible_witness::Union{Nothing,Vector{Float64}}
    infeasibility_certificate::Union{Nothing,FeedInfeasibilityCertificate}
    requested_status::FeasibilityStatus
end

function _feed_sample_kinds(rng::AbstractRNG, kinds::Tuple, count::Int)
    result = Vector{typeof(first(kinds))}(undef, count)
    guaranteed = min(count, length(kinds))
    for i in 1:guaranteed
        result[i] = kinds[i]
    end
    for i in (guaranteed + 1):count
        result[i] = kinds[rand(rng, 1:length(kinds))]
    end
    shuffle!(rng, result)
    return result
end

function _feed_sample_content(
    rng::AbstractRNG,
    nutrient_kind::FeedNutrientKind,
    ingredient_kind::FeedIngredientKind,
)
    j = Int(nutrient_kind) + 1
    i = Int(ingredient_kind) + 1
    rand(rng) <= _FEED_CONTENT_PROBABILITY[j, i] || return 0.0
    median = _FEED_CONTENT_MEDIAN[j, i]
    concentration = rand(rng, LogNormal(log(median), 0.35))
    return min(concentration, _FEED_CONTENT_MAXIMUM[j])
end

function _feed_effective_capacity_sum(
    availabilities::AbstractVector{<:Real},
    batch_size::Real,
)
    return sum(
        isfinite(capacity) ? clamp(Float64(capacity), 0.0, Float64(batch_size)) :
        Float64(batch_size)
        for capacity in availabilities
    )
end

"""
Exact minimum or maximum attainable average of one nutrient under only the
batch equality and ingredient availability bounds. Sorting coefficients and
filling capacity greedily solves this one-row continuous knapsack exactly.
"""
function _feed_achievable_average(
    nutrient_content::AbstractMatrix{<:Real},
    availabilities::AbstractVector{<:Real},
    batch_size::Real,
    nutrient::Int;
    maximize::Bool,
)
    batch = Float64(batch_size)
    order = sortperm(view(nutrient_content, nutrient, :); rev=maximize)
    remaining = batch
    total = 0.0
    for ingredient in order
        remaining <= 1e-10 * max(1.0, batch) && break
        raw_capacity = availabilities[ingredient]
        capacity = isfinite(raw_capacity) ?
                   clamp(Float64(raw_capacity), 0.0, batch) : batch
        amount = min(capacity, remaining)
        total += nutrient_content[nutrient, ingredient] * amount
        remaining -= amount
    end
    remaining <= 1e-8 * max(1.0, batch) ||
        throw(ArgumentError("ingredient availability cannot fill the requested batch"))
    return total / batch
end

function _feed_reference_recipe(
    rng::AbstractRNG,
    costs::Vector{Float64},
    availabilities::Vector{Float64},
    batch_size::Float64,
)
    n = length(costs)
    recipe = batch_size .* rand(rng, Dirichlet(fill(1.0, n)))
    for i in 1:n
        isfinite(availabilities[i]) &&
            (recipe[i] = min(recipe[i], max(availabilities[i], 0.0)))
    end

    remaining = batch_size - sum(recipe)
    # Draw one jitter per ingredient up front. `sortperm(...; by=f)` evaluates
    # `f` inside the comparator, so sampling there would redraw an ingredient's
    # key on every comparison (an inconsistent ordering) and make the number of
    # RNG draws depend on Base's sorting algorithm, breaking seed reproducibility
    # across Julia versions.
    jittered_costs = [costs[i] * rand(rng, Uniform(0.85, 1.15)) for i in 1:n]
    priority = sortperm(jittered_costs)
    for i in priority
        remaining <= 1e-10 * max(1.0, batch_size) && break
        capacity = isfinite(availabilities[i]) ?
                   max(availabilities[i] - recipe[i], 0.0) : remaining
        addition = min(capacity, remaining)
        recipe[i] += addition
        remaining -= addition
    end
    remaining <= 1e-8 * max(1.0, batch_size) ||
        throw(ArgumentError("failed to construct a full feed recipe"))
    return recipe
end

function _feed_ratio_average(
    nutrient_content::AbstractMatrix{<:Real},
    recipe::AbstractVector{<:Real},
    batch_size::Real,
    nutrient::Int,
)
    return sum(
        nutrient_content[nutrient, i] * recipe[i] for i in eachindex(recipe)
    ) / batch_size
end

"""
    feed_recipe_satisfies(prob, recipe=prob.feasible_witness; atol=1e-8)

Check a recipe directly against all generated data and formulation rows. This is
solver-independent and is primarily useful for validating `feasible_witness`.
"""
function feed_recipe_satisfies(
    prob::FeedBlendingProblem,
    recipe::Union{Nothing,AbstractVector{<:Real}}=prob.feasible_witness;
    atol::Float64=1e-8,
)
    recipe === nothing && return false
    length(recipe) == prob.num_ingredients || return false
    mass_tolerance = atol * max(1.0, prob.batch_size)
    all(amount -> amount >= -mass_tolerance, recipe) || return false
    abs(sum(recipe) - prob.batch_size) <= mass_tolerance || return false

    for i in 1:prob.num_ingredients
        if isfinite(prob.availabilities[i]) &&
           recipe[i] > prob.availabilities[i] + mass_tolerance
            return false
        end
    end

    for j in 1:prob.num_nutrients
        total = sum(
            prob.nutrient_content[j, i] * recipe[i]
            for i in 1:prob.num_ingredients
        )
        row_tolerance = atol * max(1.0, abs(total), prob.batch_size)
        total + row_tolerance >= prob.min_requirements[j] || return false
        if isfinite(prob.max_limits[j])
            total <= prob.max_limits[j] + row_tolerance || return false
        end
    end

    for constraint in prob.ratio_constraints
        average = _feed_ratio_average(
            prob.nutrient_content,
            recipe,
            prob.batch_size,
            constraint.nutrient,
        )
        ratio_tolerance = atol * max(1.0, abs(average), abs(constraint.target))
        if constraint.sense == feed_ratio_minimum
            average + ratio_tolerance >= constraint.target || return false
        elseif constraint.sense == feed_ratio_maximum
            average <= constraint.target + ratio_tolerance || return false
        else
            return false
        end
    end
    return true
end

"""
    feed_infeasibility_certificate_holds(prob; atol=1e-8)

Recompute and validate the structural contradiction recorded on a requested-
infeasible feed blend. No optimization solver is used.
"""
function feed_infeasibility_certificate_holds(
    prob::FeedBlendingProblem;
    atol::Float64=1e-8,
)
    certificate = prob.infeasibility_certificate
    certificate === nothing && return false

    if certificate.kind == feed_insufficient_ingredient_capacity
        certificate.nutrient == 0 || return false
        certificate.ratio_constraint == 0 || return false
        achievable = _feed_effective_capacity_sum(
            prob.availabilities, prob.batch_size
        )
        required = prob.batch_size
    elseif certificate.kind == feed_minimum_nutrient_above_achievable_maximum
        1 <= certificate.nutrient <= prob.num_nutrients || return false
        certificate.ratio_constraint == 0 || return false
        achievable = prob.batch_size * _feed_achievable_average(
            prob.nutrient_content,
            prob.availabilities,
            prob.batch_size,
            certificate.nutrient;
            maximize=true,
        )
        required = prob.min_requirements[certificate.nutrient]
    else
        1 <= certificate.ratio_constraint <= length(prob.ratio_constraints) ||
            return false
        constraint = prob.ratio_constraints[certificate.ratio_constraint]
        constraint.nutrient == certificate.nutrient || return false
        if certificate.kind == feed_minimum_ratio_above_achievable_maximum
            constraint.sense == feed_ratio_minimum || return false
            achievable = _feed_achievable_average(
                prob.nutrient_content,
                prob.availabilities,
                prob.batch_size,
                certificate.nutrient;
                maximize=true,
            )
        elseif certificate.kind == feed_maximum_ratio_below_achievable_minimum
            constraint.sense == feed_ratio_maximum || return false
            achievable = _feed_achievable_average(
                prob.nutrient_content,
                prob.availabilities,
                prob.batch_size,
                certificate.nutrient;
                maximize=false,
            )
        else
            return false
        end
        required = constraint.target
    end

    comparison_scale = max(1.0, abs(achievable), abs(required))
    metadata_matches =
        isapprox(certificate.achievable_bound, achievable; atol=atol, rtol=1e-10) &&
        isapprox(certificate.required_bound, required; atol=atol, rtol=1e-10)
    metadata_matches || return false

    if certificate.kind == feed_maximum_ratio_below_achievable_minimum
        return required + atol * comparison_scale < achievable
    end
    return achievable + atol * comparison_scale < required
end

function _feed_set_witness_constraints!(
    rng::AbstractRNG,
    min_requirements::Vector{Float64},
    max_limits::Vector{Float64},
    nutrient_content::Matrix{Float64},
    nutrient_types::Vector{FeedNutrientKind},
    availabilities::Vector{Float64},
    batch_size::Float64,
    recipe::Vector{Float64},
)
    for j in eachindex(nutrient_types)
        kind = nutrient_types[j]
        witness_average = _feed_ratio_average(
            nutrient_content, recipe, batch_size, j
        )
        minimum_average = _feed_achievable_average(
            nutrient_content, availabilities, batch_size, j; maximize=false
        )
        maximum_average = _feed_achievable_average(
            nutrient_content, availabilities, batch_size, j; maximize=true
        )

        minimum_probability = kind == feed_major_nutrient ? 0.95 :
                              kind == feed_mineral ? 0.78 :
                              kind == feed_trace_nutrient ? 0.55 : 0.10
        maximum_probability = kind == feed_restricted_compound ? 0.92 :
                              kind == feed_major_nutrient ? 0.22 : 0.18

        if rand(rng) < minimum_probability && witness_average > 0.0
            q = rand(rng, Uniform(0.50, 0.88))
            target = minimum_average + q * (witness_average - minimum_average)
            target = min(target, witness_average * rand(rng, Uniform(0.94, 0.99)))
            min_requirements[j] = max(0.0, target) * batch_size
        end
        if rand(rng) < maximum_probability
            q = rand(rng, Uniform(0.18, 0.55))
            target = witness_average + q * (maximum_average - witness_average)
            target = max(target, witness_average * rand(rng, Uniform(1.01, 1.08)))
            max_limits[j] = target * batch_size
        end
    end
    return nothing
end

function _feed_set_unknown_constraints!(
    rng::AbstractRNG,
    min_requirements::Vector{Float64},
    max_limits::Vector{Float64},
    nutrient_content::Matrix{Float64},
    nutrient_types::Vector{FeedNutrientKind},
    availabilities::Vector{Float64},
    batch_size::Float64,
)
    for j in eachindex(nutrient_types)
        kind = nutrient_types[j]
        minimum_average = _feed_achievable_average(
            nutrient_content, availabilities, batch_size, j; maximize=false
        )
        maximum_average = _feed_achievable_average(
            nutrient_content, availabilities, batch_size, j; maximize=true
        )
        width = maximum_average - minimum_average
        minimum_probability = kind == feed_major_nutrient ? 0.90 :
                              kind == feed_mineral ? 0.65 :
                              kind == feed_trace_nutrient ? 0.42 : 0.08
        maximum_probability = kind == feed_restricted_compound ? 0.85 : 0.18
        if rand(rng) < minimum_probability
            min_requirements[j] =
                (minimum_average + rand(rng, Uniform(0.15, 0.58)) * width) *
                batch_size
        end
        if rand(rng) < maximum_probability
            max_limits[j] =
                (minimum_average + rand(rng, Uniform(0.65, 0.95)) * width) *
                batch_size
        end
        if isfinite(max_limits[j]) && min_requirements[j] > max_limits[j]
            min_requirements[j], max_limits[j] =
                0.95 * max_limits[j], 1.05 * min_requirements[j]
        end
    end
    return nothing
end

function _feed_add_ratio_constraints!(
    rng::AbstractRNG,
    constraints::Vector{FeedRatioConstraint},
    nutrient_content::Matrix{Float64},
    nutrient_types::Vector{FeedNutrientKind},
    availabilities::Vector{Float64},
    batch_size::Float64,
    reference_recipe::Union{Nothing,Vector{Float64}},
)
    rand(rng) < 0.68 || return nothing
    n_nutrients = length(nutrient_types)
    count = rand(rng, 1:min(4, max(1, ceil(Int, 0.3 * n_nutrients))))
    selected = randperm(rng, n_nutrients)[1:count]
    for j in selected
        kind = nutrient_types[j]
        minimum_average = _feed_achievable_average(
            nutrient_content, availabilities, batch_size, j; maximize=false
        )
        maximum_average = _feed_achievable_average(
            nutrient_content, availabilities, batch_size, j; maximize=true
        )
        use_minimum = kind == feed_restricted_compound ? rand(rng) < 0.20 :
                      kind == feed_major_nutrient ? rand(rng) < 0.78 :
                      rand(rng) < 0.58

        if reference_recipe === nothing
            q = use_minimum ? rand(rng, Uniform(0.20, 0.62)) :
                              rand(rng, Uniform(0.38, 0.82))
            target = minimum_average + q * (maximum_average - minimum_average)
        else
            witness_average = _feed_ratio_average(
                nutrient_content, reference_recipe, batch_size, j
            )
            if use_minimum
                q = rand(rng, Uniform(0.18, 0.55))
                target = witness_average - q * (witness_average - minimum_average)
                target = min(target, witness_average * rand(rng, Uniform(0.94, 0.99)))
                target = max(0.0, target)
            else
                q = rand(rng, Uniform(0.18, 0.55))
                target = witness_average + q * (maximum_average - witness_average)
                target = max(target, witness_average * rand(rng, Uniform(1.01, 1.08)))
            end
        end
        push!(
            constraints,
            FeedRatioConstraint(
                j,
                target,
                use_minimum ? feed_ratio_minimum : feed_ratio_maximum,
            ),
        )
    end
    return nothing
end

"""
    FeedBlendingProblem(target_variables, feasibility_status, seed)

Construct a deterministic feed-blending instance. The model has exactly
`max(3, target_variables)` ingredient variables. All randomness is drawn from a
constructor-local RNG.
"""
function FeedBlendingProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    rng = Random.MersenneTwister(seed)
    num_ingredients = max(3, target_variables)

    if target_variables <= 250
        num_nutrients = rand(rng, 4:8)
        batch_size = rand(
            rng, truncated(Normal(500.0, 200.0), 100.0, 2_000.0)
        )
    elseif target_variables <= 1_000
        num_nutrients = rand(rng, 6:12)
        batch_size = rand(
            rng, truncated(Normal(2_000.0, 800.0), 500.0, 10_000.0)
        )
    else
        num_nutrients = rand(rng, 8:20)
        batch_size = rand(
            rng, truncated(Normal(10_000.0, 5_000.0), 2_000.0, 50_000.0)
        )
    end

    ingredient_types = _feed_sample_kinds(
        rng, _FEED_INGREDIENT_KINDS, num_ingredients
    )
    nutrient_types = _feed_sample_kinds(
        rng, _FEED_NUTRIENT_KINDS, num_nutrients
    )

    costs = Vector{Float64}(undef, num_ingredients)
    for i in 1:num_ingredients
        kind_index = Int(ingredient_types[i]) + 1
        costs[i] = rand(
            rng,
            LogNormal(
                log(_FEED_COST_MEDIAN[kind_index]),
                _FEED_COST_LOG_SIGMA[kind_index],
            ),
        )
    end

    nutrient_content = zeros(Float64, num_nutrients, num_ingredients)
    for j in 1:num_nutrients, i in 1:num_ingredients
        nutrient_content[j, i] = _feed_sample_content(
            rng, nutrient_types[j], ingredient_types[i]
        )
    end

    # Intentional sparsity is realistic for minor/trace metrics, but empty rows
    # or columns are not useful benchmark data. Repair them using role-aware data.
    for j in 1:num_nutrients
        if all(iszero, view(nutrient_content, j, :))
            i = rand(rng, 1:num_ingredients)
            nutrient_content[j, i] = max(
                _feed_sample_content(rng, nutrient_types[j], ingredient_types[i]),
                0.1 * _FEED_CONTENT_MEDIAN[Int(nutrient_types[j]) + 1,
                                            Int(ingredient_types[i]) + 1],
            )
        end
    end
    for i in 1:num_ingredients
        if all(iszero, view(nutrient_content, :, i))
            j = findfirst(==(feed_major_nutrient), nutrient_types)
            j === nothing && (j = rand(rng, 1:num_nutrients))
            nutrient_content[j, i] = max(
                _feed_sample_content(rng, nutrient_types[j], ingredient_types[i]),
                0.1 * _FEED_CONTENT_MEDIAN[Int(nutrient_types[j]) + 1,
                                            Int(ingredient_types[i]) + 1],
            )
        end
    end

    availabilities = fill(Inf, num_ingredients)
    for i in 1:num_ingredients
        kind = ingredient_types[i]
        finite_probability = kind in (feed_energy_source, feed_protein_source) ?
                             0.55 : 0.85
        if rand(rng) < finite_probability
            fraction = if kind == feed_energy_source
                rand(rng, Uniform(0.15, 0.80))
            elseif kind == feed_protein_source
                rand(rng, Uniform(0.10, 0.60))
            elseif kind == feed_mineral_supplement
                rand(rng, Uniform(0.01, 0.15))
            else
                rand(rng, Uniform(0.005, 0.06))
            end
            availabilities[i] = fraction * batch_size
        end
    end

    # A feasible baseline is valuable for requested-feasible instances and makes
    # every requested-infeasible instance a controlled one-certificate mutation.
    if _feed_effective_capacity_sum(availabilities, batch_size) < batch_size
        cheapest = argmin(costs)
        availabilities[cheapest] = batch_size
    end
    baseline_recipe = _feed_reference_recipe(
        rng, costs, availabilities, batch_size
    )

    min_requirements = zeros(Float64, num_nutrients)
    max_limits = fill(Inf, num_nutrients)
    ratio_constraints = FeedRatioConstraint[]

    if feasibility_status == unknown
        _feed_set_unknown_constraints!(
            rng,
            min_requirements,
            max_limits,
            nutrient_content,
            nutrient_types,
            availabilities,
            batch_size,
        )
        _feed_add_ratio_constraints!(
            rng,
            ratio_constraints,
            nutrient_content,
            nutrient_types,
            availabilities,
            batch_size,
            nothing,
        )
    else
        _feed_set_witness_constraints!(
            rng,
            min_requirements,
            max_limits,
            nutrient_content,
            nutrient_types,
            availabilities,
            batch_size,
            baseline_recipe,
        )
        _feed_add_ratio_constraints!(
            rng,
            ratio_constraints,
            nutrient_content,
            nutrient_types,
            availabilities,
            batch_size,
            baseline_recipe,
        )
    end

    certificate = nothing
    if feasibility_status == infeasible
        mode = rand(rng, 1:4)
        if mode == 1
            nutrient = rand(rng, 1:num_nutrients)
            achievable = _feed_achievable_average(
                nutrient_content,
                availabilities,
                batch_size,
                nutrient;
                maximize=true,
            )
            required = achievable +
                       rand(rng, Uniform(0.03, 0.12)) * max(achievable, 1.0)
            push!(
                ratio_constraints,
                FeedRatioConstraint(nutrient, required, feed_ratio_minimum),
            )
            certificate = FeedInfeasibilityCertificate(
                feed_minimum_ratio_above_achievable_maximum,
                nutrient,
                length(ratio_constraints),
                achievable,
                required,
            )
        elseif mode == 2
            nutrient = rand(rng, 1:num_nutrients)
            achievable = batch_size * _feed_achievable_average(
                nutrient_content,
                availabilities,
                batch_size,
                nutrient;
                maximize=true,
            )
            required = achievable +
                       rand(rng, Uniform(0.03, 0.12)) * max(achievable, batch_size)
            min_requirements[nutrient] = required
            certificate = FeedInfeasibilityCertificate(
                feed_minimum_nutrient_above_achievable_maximum,
                nutrient,
                0,
                achievable,
                required,
            )
        elseif mode == 3
            candidates = [
                j for j in 1:num_nutrients
                if _feed_achievable_average(
                    nutrient_content,
                    availabilities,
                    batch_size,
                    j;
                    maximize=false,
                ) > 1e-9
            ]
            nutrient = rand(rng, candidates)
            achievable = _feed_achievable_average(
                nutrient_content,
                availabilities,
                batch_size,
                nutrient;
                maximize=false,
            )
            required = achievable * rand(rng, Uniform(0.72, 0.94))
            push!(
                ratio_constraints,
                FeedRatioConstraint(nutrient, required, feed_ratio_maximum),
            )
            certificate = FeedInfeasibilityCertificate(
                feed_maximum_ratio_below_achievable_minimum,
                nutrient,
                length(ratio_constraints),
                achievable,
                required,
            )
        else
            total_capacity = batch_size * rand(rng, Uniform(0.65, 0.90))
            capacity_weights = rand(
                rng, Dirichlet(fill(1.0, num_ingredients))
            )
            availabilities .= total_capacity .* capacity_weights
            achievable = _feed_effective_capacity_sum(
                availabilities, batch_size
            )
            certificate = FeedInfeasibilityCertificate(
                feed_insufficient_ingredient_capacity,
                0,
                0,
                achievable,
                batch_size,
            )
        end
    end

    problem = FeedBlendingProblem(
        num_ingredients,
        num_nutrients,
        batch_size,
        ingredient_types,
        costs,
        nutrient_content,
        nutrient_types,
        min_requirements,
        max_limits,
        availabilities,
        ratio_constraints,
        feasibility_status == feasible ? baseline_recipe : nothing,
        certificate,
        feasibility_status,
    )

    if feasibility_status == feasible
        @assert feed_recipe_satisfies(problem)
    elseif feasibility_status == infeasible
        @assert feed_infeasibility_certificate_holds(problem)
    end
    return problem
end

"""
    build_model(prob::FeedBlendingProblem)

Build the deterministic continuous feed-blending LP. Ratio-row direction is
selected by the typed `FeedRatioSense`, never by parsing diagnostic text.
"""
function build_model(prob::FeedBlendingProblem)
    model = Model()
    @variable(model, x[1:prob.num_ingredients] >= 0)
    @objective(
        model,
        Min,
        sum(prob.costs[i] * x[i] for i in 1:prob.num_ingredients),
    )
    @constraint(
        model,
        batch_balance,
        sum(x[i] for i in 1:prob.num_ingredients) == prob.batch_size,
    )

    minimum_nutrients = findall(>(0.0), prob.min_requirements)
    maximum_nutrients = findall(isfinite, prob.max_limits)
    finite_ingredients = findall(isfinite, prob.availabilities)
    minimum_ratios = findall(
        constraint -> constraint.sense == feed_ratio_minimum,
        prob.ratio_constraints,
    )
    maximum_ratios = findall(
        constraint -> constraint.sense == feed_ratio_maximum,
        prob.ratio_constraints,
    )

    @constraint(
        model,
        nutrient_min[j in minimum_nutrients],
        sum(
            prob.nutrient_content[j, i] * x[i]
            for i in 1:prob.num_ingredients
        ) >= prob.min_requirements[j],
    )
    @constraint(
        model,
        nutrient_max[j in maximum_nutrients],
        sum(
            prob.nutrient_content[j, i] * x[i]
            for i in 1:prob.num_ingredients
        ) <= prob.max_limits[j],
    )
    @constraint(
        model,
        ingredient_availability[i in finite_ingredients],
        x[i] <= prob.availabilities[i],
    )
    @constraint(
        model,
        ratio_min[r in minimum_ratios],
        sum(
            (prob.nutrient_content[prob.ratio_constraints[r].nutrient, i] -
             prob.ratio_constraints[r].target) * x[i]
            for i in 1:prob.num_ingredients
        ) >= 0,
    )
    @constraint(
        model,
        ratio_max[r in maximum_ratios],
        sum(
            (prob.nutrient_content[prob.ratio_constraints[r].nutrient, i] -
             prob.ratio_constraints[r].target) * x[i]
            for i in 1:prob.num_ingredients
        ) <= 0,
    )
    return model
end

register_variant(
    :feed_blending,
    :standard,
    FeedBlendingProblem,
    "Feed formulation with role-correlated ingredient data, typed nutrient-ratio " *
    "bounds, and checkable feasibility metadata",
)
