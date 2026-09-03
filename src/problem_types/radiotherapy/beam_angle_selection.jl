"""
A complete fluence/open-field point for a beam-angle-selection instance.
"""
struct RadiotherapyBeamSelectionWitness
    fluence::Vector{Float64}
    open_beams::Vector{Int}
end

"""
    BeamAngleSelectionIMRTProblem <: ProblemGenerator

Joint beam-angle selection and fluence-map planning over twelve candidate
coplanar fields. Binary field-open variables gate every beamlet. The number of
open fields lies in a range, rather than being fixed exactly, so the field-open
term in the objective is meaningful. The natural formulation is a MILP (and is
relaxed by the package's usual default unless requested otherwise).
"""
struct BeamAngleSelectionIMRTProblem <: ProblemGenerator
    case_data::RadiotherapyCaseData
    resolved_status::FeasibilityStatus
    minimum_open_beams::Int
    maximum_open_beams::Int
    reference_open_beams::Vector{Int}
    beamlet_fluence_max::Vector{Float64}
    target_floor::Float64
    target_ceiling::Float64
    structure_max::Dict{Symbol, Float64}
    desired_dose::Vector{Float64}
    underdose_weight::Vector{Float64}
    overdose_weight::Vector{Float64}
    beam_open_penalty::Float64
    fluence_penalty::Float64
    smoothness_penalty::Float64
    feasible_witness::Union{Nothing, RadiotherapyBeamSelectionWitness}
    infeasibility_certificate::Union{Nothing, RadiotherapyDoseConflictCertificate}
end

function BeamAngleSelectionIMRTProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    profile = _RT_PROFILES[mod(seed, length(_RT_PROFILES)) + 1]
    base_spec = _rt_profile_spec(profile)
    candidate_angles = profile == :breast ? collect(15.0:30.0:345.0) : collect(0.0:30.0:330.0)
    maximum_open = clamp(ceil(Int, 0.75 * length(base_spec.angles)), 4, 7)
    minimum_open = max(3, maximum_open - 2)
    reference_open = unique(round.(Int, range(1, length(candidate_angles); length=maximum_open)))
    case, spec, conflict_indices, rng = _rt_build_case(
        target_variables,
        :beam_angle_selection,
        feasibility_status,
        seed;
        profile_override=profile,
        angles=candidate_angles,
        active_beams=reference_open,
    )
    target_floor, target_ceiling, structure_max, base_witness, certificate = _rt_hard_limits(
        case, spec, feasibility_status, conflict_indices, rng
    )
    witness = if base_witness === nothing
        nothing
    else
        RadiotherapyBeamSelectionWitness(copy(base_witness.fluence), copy(reference_open))
    end

    target = case.structure_voxels[:ptv]
    peak_reference = maximum(case.reference_fluence)
    beamlet_fluence_max = fill(max(0.5, 2.5 * peak_reference), length(case.reference_fluence))
    # Ensure the planted point remains strictly inside every linking row.
    beamlet_fluence_max = max.(beamlet_fluence_max, 1.05 .* case.reference_fluence)

    n_voxels = size(case.voxel_locations_cm, 1)
    desired_dose = zeros(n_voxels)
    underdose_weight = zeros(n_voxels)
    overdose_weight = zeros(n_voxels)
    kind_by_structure = Dict(zip(spec.structures, spec.kinds))
    for structure in spec.structures
        indices = case.structure_voxels[structure]
        volume_share = case.voxel_volume_cc[indices] ./ sum(case.voxel_volume_cc[indices])
        kind = kind_by_structure[structure]
        if kind == :target
            desired_dose[indices] .= 1.0
            underdose_weight[indices] .= (28.0 + 22rand(rng)) .* volume_share
            overdose_weight[indices] .= (8.0 + 7rand(rng)) .* volume_share
        else
            desired_dose[indices] .= 0.72 * spec.clinical_caps[structure]
            base = if kind == :serial_oar
                17.0
            elseif kind == :parallel_oar
                7.0
            else
                1.0
            end
            overdose_weight[indices] .= base * (0.8 + 0.4rand(rng)) .* volume_share
        end
    end
    beam_open_penalty = 0.006 + 0.008rand(rng)
    fluence_penalty = (0.0012 + 0.0018rand(rng)) / length(case.reference_fluence)
    smoothness_penalty =
        isempty(case.beamlet_edges) ? 0.0 : (0.014 + 0.020rand(rng)) / length(case.beamlet_edges)
    return BeamAngleSelectionIMRTProblem(
        case,
        feasibility_status,
        minimum_open,
        maximum_open,
        reference_open,
        beamlet_fluence_max,
        target_floor,
        target_ceiling,
        structure_max,
        desired_dose,
        underdose_weight,
        overdose_weight,
        beam_open_penalty,
        fluence_penalty,
        smoothness_penalty,
        witness,
        certificate,
    )
end

function _rt_beam_selection_witness_is_valid(
    problem::BeamAngleSelectionIMRTProblem; atol::Float64=1.0e-9
)
    witness = problem.feasible_witness
    witness === nothing && return false
    problem.minimum_open_beams <= length(witness.open_beams) <= problem.maximum_open_beams ||
        return false
    open_set = Set(witness.open_beams)
    for j in eachindex(witness.fluence)
        beam = problem.case_data.beam_of_beamlet[j]
        witness.fluence[j] <= problem.beamlet_fluence_max[j] + atol || return false
        beam in open_set || witness.fluence[j] <= atol || return false
    end
    return _rt_witness_is_valid(problem; atol=atol)
end

function build_model(problem::BeamAngleSelectionIMRTProblem)
    model = Model()
    case = problem.case_data
    n_beamlets = length(case.reference_fluence)
    n_beams = length(case.beam_angles_deg)
    n_voxels = size(case.voxel_locations_cm, 1)
    target = case.structure_voxels[:ptv]
    n_edges = length(case.beamlet_edges)

    @variable(model, fluence[1:n_beamlets] >= 0)
    @variable(model, beam_open[1:n_beams], Bin)
    @variable(model, underdose[target] >= 0)
    @variable(model, overdose[1:n_voxels] >= 0)
    @variable(model, variation[1:n_edges] >= 0)
    dose = _rt_dose_expressions(model, fluence, case.dose_matrix)

    @constraint(
        model,
        beamlet_link[j in 1:n_beamlets],
        fluence[j] <= problem.beamlet_fluence_max[j] * beam_open[case.beam_of_beamlet[j]]
    )
    @constraint(model, minimum_fields, sum(beam_open) >= problem.minimum_open_beams)
    @constraint(model, maximum_fields, sum(beam_open) <= problem.maximum_open_beams)
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
            problem.beam_open_penalty * sum(beam_open) +
            problem.fluence_penalty * sum(fluence) +
            problem.smoothness_penalty * sum(variation),
    )
    for beam in 1:n_beams
        set_start_value(beam_open[beam], beam in problem.reference_open_beams ? 1.0 : 0.0)
    end
    model[:dose] = dose
    return model
end

register_variant(
    :radiotherapy,
    :beam_angle_selection,
    BeamAngleSelectionIMRTProblem,
    "Joint beam-angle selection and fluence-map MILP over candidate coplanar fields",
)
