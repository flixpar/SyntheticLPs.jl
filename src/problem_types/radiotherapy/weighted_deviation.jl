"""
    WeightedDeviationIMRTProblem <: ProblemGenerator

Beamlet fluence-map optimization with the classical linear dose relation
`dose = D * fluence`. Target underdose and voxel overdose are represented by
hinge-loss variables, and absolute differences between adjacent beamlets form
an anisotropic total-variation delivery penalty. Hard target and organ limits
keep feasibility clinically meaningful instead of allowing slacks to make
every instance feasible.

The anatomy profile is one of `:prostate`, `:head_neck`, `:c_shape`, `:liver`,
`:lung`, or `:breast`. The shared case stores 3-D sampled voxel coordinates,
sampled-volume weights, clinical beam
angles, a 2-D beamlet grid for every field, grid adjacencies, and a sparse
spatial pencil-beam dose-influence matrix.
"""
struct WeightedDeviationIMRTProblem <: ProblemGenerator
    case_data::RadiotherapyCaseData
    resolved_status::FeasibilityStatus
    target_floor::Float64
    target_ceiling::Float64
    structure_max::Dict{Symbol, Float64}
    desired_dose::Vector{Float64}
    underdose_weight::Vector{Float64}
    overdose_weight::Vector{Float64}
    fluence_penalty::Float64
    smoothness_penalty::Float64
    feasible_witness::Union{Nothing, RadiotherapyFluenceWitness}
    infeasibility_certificate::Union{Nothing, RadiotherapyDoseConflictCertificate}
end

function WeightedDeviationIMRTProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    case, spec, conflict_indices, rng = _rt_build_case(
        target_variables, :weighted_deviation, feasibility_status, seed
    )
    target_floor, target_ceiling, structure_max, witness, certificate = _rt_hard_limits(
        case, spec, feasibility_status, conflict_indices, rng
    )

    n_voxels = size(case.voxel_locations_cm, 1)
    desired_dose = zeros(Float64, n_voxels)
    underdose_weight = zeros(Float64, n_voxels)
    overdose_weight = zeros(Float64, n_voxels)
    kind_by_structure = Dict(zip(spec.structures, spec.kinds))
    for structure in spec.structures
        indices = case.structure_voxels[structure]
        volume_share = case.voxel_volume_cc[indices] ./ sum(case.voxel_volume_cc[indices])
        kind = kind_by_structure[structure]
        if kind == :target
            desired_dose[indices] .= 1.0
            underdose_weight[indices] .= (28.0 + 24.0 * rand(rng)) .* volume_share
            overdose_weight[indices] .= (8.0 + 8.0 * rand(rng)) .* volume_share
        else
            desired_dose[indices] .= 0.72 * spec.clinical_caps[structure]
            base_weight = if kind == :serial_oar
                18.0
            elseif kind == :parallel_oar
                8.0
            else
                1.2
            end
            overdose_weight[indices] .= base_weight * (0.75 + 0.50 * rand(rng)) .* volume_share
        end
    end
    fluence_penalty = (0.0015 + 0.0020 * rand(rng)) / length(case.reference_fluence)
    smoothness_penalty =
        isempty(case.beamlet_edges) ? 0.0 : (0.018 + 0.025 * rand(rng)) / length(case.beamlet_edges)

    return WeightedDeviationIMRTProblem(
        case,
        feasibility_status,
        target_floor,
        target_ceiling,
        structure_max,
        desired_dose,
        underdose_weight,
        overdose_weight,
        fluence_penalty,
        smoothness_penalty,
        witness,
        certificate,
    )
end

function build_model(problem::WeightedDeviationIMRTProblem)
    model = Model()
    case = problem.case_data
    n_beamlets = length(case.reference_fluence)
    n_voxels = size(case.voxel_locations_cm, 1)
    target = case.structure_voxels[:ptv]
    n_edges = length(case.beamlet_edges)

    @variable(model, fluence[1:n_beamlets] >= 0)
    @variable(model, underdose[target] >= 0)
    @variable(model, overdose[1:n_voxels] >= 0)
    @variable(model, variation[1:n_edges] >= 0)
    dose = _rt_dose_expressions(model, fluence, case.dose_matrix)

    @constraint(
        model, underdose_hinge[i in target], dose[i] + underdose[i] >= problem.desired_dose[i]
    )
    @constraint(
        model, overdose_hinge[i in 1:n_voxels], dose[i] - overdose[i] <= problem.desired_dose[i]
    )
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
        sum(problem.underdose_weight[i] * underdose[i] for i in target) +
            sum(problem.overdose_weight[i] * overdose[i] for i in 1:n_voxels) +
            problem.fluence_penalty * sum(fluence) +
            problem.smoothness_penalty * sum(variation),
    )

    for j in 1:n_beamlets
        set_start_value(fluence[j], case.reference_fluence[j])
    end
    reference_dose = case.reference_dose
    for i in target
        set_start_value(underdose[i], max(0.0, problem.desired_dose[i] - reference_dose[i]))
    end
    for i in 1:n_voxels
        set_start_value(overdose[i], max(0.0, reference_dose[i] - problem.desired_dose[i]))
    end
    for e in 1:n_edges
        a, b = case.beamlet_edges[e]
        set_start_value(variation[e], abs(case.reference_fluence[a] - case.reference_fluence[b]))
    end
    model[:dose] = dose
    return model
end

register_variant(
    :radiotherapy,
    :weighted_deviation,
    WeightedDeviationIMRTProblem,
    "IMRT fluence-map LP with voxelwise underdose/overdose hinge penalties and total-variation smoothing";
    default=true,
)
