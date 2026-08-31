"""
    MinMaxDeviationIMRTProblem <: ProblemGenerator

IMRT fluence-map planning that minimizes the worst weighted voxel deviation.
Unlike the default sum-of-hinges model, its epigraph objective prevents a large
anatomical structure from hiding a small but badly served region. Hard
pointwise safety limits and two-dimensional fluence-map total variation are
retained, so the formulation remains a pure LP.
"""
struct MinMaxDeviationIMRTProblem <: ProblemGenerator
    case_data::RadiotherapyCaseData
    resolved_status::FeasibilityStatus
    target_floor::Float64
    target_ceiling::Float64
    structure_max::Dict{Symbol,Float64}
    desired_dose::Vector{Float64}
    underdose_importance::Vector{Float64}
    overdose_importance::Vector{Float64}
    fluence_penalty::Float64
    smoothness_penalty::Float64
    feasible_witness::Union{Nothing,RadiotherapyFluenceWitness}
    infeasibility_certificate::Union{Nothing,RadiotherapyDoseConflictCertificate}
end

function MinMaxDeviationIMRTProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    case, spec, conflict_indices, rng = _rt_build_case(
        target_variables, :minmax_deviation, feasibility_status, seed,
    )
    target_floor, target_ceiling, structure_max, witness, certificate =
        _rt_hard_limits(case, spec, feasibility_status, conflict_indices, rng)

    n_voxels = size(case.voxel_locations_cm, 1)
    desired_dose = zeros(n_voxels)
    underdose_importance = zeros(n_voxels)
    overdose_importance = zeros(n_voxels)
    kind_by_structure = Dict(zip(spec.structures, spec.kinds))
    for structure in spec.structures
        indices = case.structure_voxels[structure]
        kind = kind_by_structure[structure]
        if kind == :target
            desired_dose[indices] .= 1.0
            underdose_importance[indices] .= 1.0
            overdose_importance[indices] .= 0.65
        else
            desired_dose[indices] .= 0.72 * spec.clinical_caps[structure]
            importance = kind == :serial_oar ? 0.90 :
                         kind == :parallel_oar ? 0.55 : 0.16
            overdose_importance[indices] .= importance * (0.9 + 0.2rand(rng))
        end
    end
    fluence_penalty = (0.0008 + 0.0012rand(rng)) /
                      length(case.reference_fluence)
    smoothness_penalty = isempty(case.beamlet_edges) ? 0.0 :
        (0.010 + 0.018rand(rng)) / length(case.beamlet_edges)
    return MinMaxDeviationIMRTProblem(
        case, feasibility_status, target_floor, target_ceiling, structure_max,
        desired_dose, underdose_importance, overdose_importance,
        fluence_penalty, smoothness_penalty, witness, certificate,
    )
end

function build_model(problem::MinMaxDeviationIMRTProblem)
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
    @variable(model, worst_deviation >= 0)
    dose = _rt_dose_expressions(model, fluence, case.dose_matrix)

    @constraint(model, underdose_hinge[i in target],
                dose[i] + underdose[i] >= problem.desired_dose[i])
    @constraint(model, overdose_hinge[i in 1:n_voxels],
                dose[i] - overdose[i] <= problem.desired_dose[i])
    @constraint(model, worst_underdose[i in target],
                worst_deviation >= problem.underdose_importance[i] * underdose[i])
    @constraint(model, worst_overdose[i in 1:n_voxels],
                worst_deviation >= problem.overdose_importance[i] * overdose[i])
    @constraint(model, variation_positive[e in 1:n_edges],
                variation[e] >= fluence[case.beamlet_edges[e][1]] -
                                     fluence[case.beamlet_edges[e][2]])
    @constraint(model, variation_negative[e in 1:n_edges],
                variation[e] >= fluence[case.beamlet_edges[e][2]] -
                                     fluence[case.beamlet_edges[e][1]])
    _rt_add_hard_constraints!(model, problem, dose)

    @objective(model, Min,
               worst_deviation + problem.fluence_penalty * sum(fluence) +
               problem.smoothness_penalty * sum(variation))
    model[:dose] = dose
    return model
end

register_variant(
    :radiotherapy,
    :minmax_deviation,
    MinMaxDeviationIMRTProblem,
    "IMRT fluence-map LP minimizing the worst weighted voxel dose deviation",
)
