"""
    MeanTailDoseIMRTProblem <: ProblemGenerator

Convex IMRT fluence-map planning with linear mean-tail-dose constraints. The
target constrains the mean of its coldest fractional volume; every organ and
normal-tissue structure constrains the mean of its hottest fractional volume.
This is the radiotherapy form of CVaR and is a tractable, globally solvable
surrogate for nonconvex dose-volume-histogram constraints.

An auxiliary threshold is introduced per structure and one positive-part
variable per voxel. The objective minimizes structure-weighted mean normal
tissue dose, monitor-unit-like total fluence, and adjacent-beamlet total
variation, subject to hard safety floors and ceilings as well as the tail-dose
goals.
"""
struct MeanTailDoseIMRTProblem <: ProblemGenerator
    case_data::RadiotherapyCaseData
    resolved_status::FeasibilityStatus
    target_floor::Float64
    target_ceiling::Float64
    structure_max::Dict{Symbol, Float64}
    tail_fraction::Dict{Symbol, Float64}
    tail_bound::Dict{Symbol, Float64}
    mean_dose_weight::Dict{Symbol, Float64}
    fluence_penalty::Float64
    smoothness_penalty::Float64
    feasible_witness::Union{Nothing, RadiotherapyFluenceWitness}
    infeasibility_certificate::Union{Nothing, RadiotherapyDoseConflictCertificate}
end

function _rt_upper_tail_mean(values::AbstractVector{<:Real}, fraction::Float64)
    return _rt_upper_tail_mean(values, ones(length(values)), fraction)
end

function _rt_upper_tail_mean(
    values::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, fraction::Float64
)
    length(values) == length(weights) || throw(DimensionMismatch())
    order = sortperm(values; rev=true)
    mass = fraction * sum(weights)
    remaining = mass
    total = 0.0
    for index in order
        used = min(Float64(weights[index]), remaining)
        total += used * Float64(values[index])
        remaining -= used
        remaining <= 1.0e-12 * mass && break
    end
    return total / mass
end

function _rt_lower_tail_mean(values::AbstractVector{<:Real}, fraction::Float64)
    return -_rt_upper_tail_mean(-Float64.(values), fraction)
end

function _rt_lower_tail_mean(
    values::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, fraction::Float64
)
    return -_rt_upper_tail_mean(-Float64.(values), weights, fraction)
end

function MeanTailDoseIMRTProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    case, spec, conflict_indices, rng = _rt_build_case(
        target_variables, :mean_tail_dose, feasibility_status, seed
    )
    target_floor, target_ceiling, structure_max, witness, certificate = _rt_hard_limits(
        case, spec, feasibility_status, conflict_indices, rng
    )

    tail_fraction = Dict{Symbol, Float64}()
    tail_bound = Dict{Symbol, Float64}()
    mean_dose_weight = Dict{Symbol, Float64}()
    kind_by_structure = Dict(zip(spec.structures, spec.kinds))
    unknown_tail_severity =
        feasibility_status == unknown ? exp(log(0.85) + rand(rng) * (log(2.80) - log(0.85))) : 1.0
    for structure in spec.structures
        indices = case.structure_voxels[structure]
        minimum_fraction = 1.0 / length(indices)
        fraction = clamp(
            spec.tail_fractions[structure] * (0.92 + 0.16 * rand(rng)), minimum_fraction, 1.0
        )
        tail_fraction[structure] = fraction
        reference = case.reference_dose[indices]
        volume = case.voxel_volume_cc[indices]
        if structure == :ptv
            tail_bound[structure] = if feasibility_status == unknown
                0.87 + 0.10 * rand(rng)
            else
                0.97 * _rt_lower_tail_mean(reference, volume, fraction)
            end
            mean_dose_weight[structure] = 0.0
        else
            clinical = spec.clinical_caps[structure] * (0.94 + 0.12 * rand(rng))
            tail_bound[structure] = if feasibility_status == unknown
                clinical * unknown_tail_severity * (0.95 + 0.10 * rand(rng))
            else
                max(clinical, 1.04 * _rt_upper_tail_mean(reference, volume, fraction))
            end
            kind = kind_by_structure[structure]
            base_weight = if kind == :serial_oar
                3.5
            elseif kind == :parallel_oar
                2.0
            else
                0.5
            end
            mean_dose_weight[structure] = base_weight * (0.75 + 0.50 * rand(rng))
        end
    end
    fluence_penalty = (0.001 + 0.0015 * rand(rng)) / length(case.reference_fluence)
    smoothness_penalty =
        isempty(case.beamlet_edges) ? 0.0 : (0.012 + 0.022 * rand(rng)) / length(case.beamlet_edges)

    return MeanTailDoseIMRTProblem(
        case,
        feasibility_status,
        target_floor,
        target_ceiling,
        structure_max,
        tail_fraction,
        tail_bound,
        mean_dose_weight,
        fluence_penalty,
        smoothness_penalty,
        witness,
        certificate,
    )
end

function _rt_mean_tail_witness_is_valid(problem::MeanTailDoseIMRTProblem; atol::Float64=1.0e-9)
    _rt_witness_is_valid(problem; atol=atol) || return false
    dose = problem.case_data.dose_matrix * problem.feasible_witness.fluence
    for structure in problem.case_data.structure_names
        values = dose[problem.case_data.structure_voxels[structure]]
        volume = problem.case_data.voxel_volume_cc[problem.case_data.structure_voxels[structure]]
        fraction = problem.tail_fraction[structure]
        if structure == :ptv
            _rt_lower_tail_mean(values, volume, fraction) >= problem.tail_bound[structure] - atol ||
                return false
        else
            _rt_upper_tail_mean(values, volume, fraction) <= problem.tail_bound[structure] + atol ||
                return false
        end
    end
    return true
end

function build_model(problem::MeanTailDoseIMRTProblem)
    model = Model()
    case = problem.case_data
    n_beamlets = length(case.reference_fluence)
    n_voxels = size(case.voxel_locations_cm, 1)
    n_structures = length(case.structure_names)
    n_edges = length(case.beamlet_edges)

    @variable(model, fluence[1:n_beamlets] >= 0)
    @variable(model, tail_threshold[1:n_structures])
    @variable(model, tail_excess[1:n_voxels] >= 0)
    @variable(model, variation[1:n_edges] >= 0)
    dose = _rt_dose_expressions(model, fluence, case.dose_matrix)

    tail_rows = Dict{Symbol, Any}()
    excess_rows = Dict{Symbol, Any}()
    for (s, structure) in enumerate(case.structure_names)
        indices = case.structure_voxels[structure]
        fraction = problem.tail_fraction[structure]
        volume = case.voxel_volume_cc[indices]
        tail_volume = fraction * sum(volume)
        if structure == :ptv
            excess_rows[structure] = @constraint(
                model,
                [i in indices],
                tail_excess[i] >= tail_threshold[s] - dose[i],
                base_name="$(structure)_cold_tail_excess",
            )
            tail_rows[structure] = @constraint(
                model,
                tail_threshold[s] -
                sum(case.voxel_volume_cc[i] * tail_excess[i] for i in indices) / tail_volume >=
                    problem.tail_bound[structure],
                base_name="$(structure)_cold_tail_goal",
            )
        else
            excess_rows[structure] = @constraint(
                model,
                [i in indices],
                tail_excess[i] >= dose[i] - tail_threshold[s],
                base_name="$(structure)_hot_tail_excess",
            )
            tail_rows[structure] = @constraint(
                model,
                tail_threshold[s] +
                sum(case.voxel_volume_cc[i] * tail_excess[i] for i in indices) / tail_volume <=
                    problem.tail_bound[structure],
                base_name="$(structure)_hot_tail_goal",
            )
        end
    end
    @constraint(
        model,
        variation_positive[e in 1:n_edges],
        variation[e] >= fluence[case.beamlet_edges[e][1]] - fluence[case.beamlet_edges[e][2]]
    )
    @constraint(
        model,
        variation_negative[e in 1:n_edges],
        variation[e] >= fluence[case.beamlet_edges[e][2]] - fluence[case.beamlet_edges[e][1]]
    )
    _rt_add_hard_constraints!(model, problem, dose)

    @objective(
        model,
        Min,
        sum(
                problem.mean_dose_weight[structure] /
                sum(case.voxel_volume_cc[i] for i in case.structure_voxels[structure]) *
                sum(case.voxel_volume_cc[i] * dose[i] for i in case.structure_voxels[structure]) for
                structure in case.structure_names[2:end]
            ) +
            problem.fluence_penalty * sum(fluence) +
            problem.smoothness_penalty * sum(variation),
    )
    model[:dose] = dose
    model[:tail_goal] = tail_rows
    model[:tail_excess_rows] = excess_rows

    for j in 1:n_beamlets
        set_start_value(fluence[j], case.reference_fluence[j])
    end
    for e in 1:n_edges
        a, b = case.beamlet_edges[e]
        set_start_value(variation[e], abs(case.reference_fluence[a] - case.reference_fluence[b]))
    end
    return model
end

register_variant(
    :radiotherapy,
    :mean_tail_dose,
    MeanTailDoseIMRTProblem,
    "Convex IMRT fluence-map LP with target cold-tail and organ hot-tail dose constraints",
)
