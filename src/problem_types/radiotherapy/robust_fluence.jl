"""
    RobustFluenceIMRTProblem <: ProblemGenerator

Scenario-based robust IMRT fluence-map planning. One fluence map must satisfy
the pointwise safety rows and dose-deviation objective under a nominal setup
and two coherent rigid patient shifts. Scenario influence matrices are rebuilt
from the same anatomy and beam geometry; coefficients are not perturbed
independently. The formulation is a deterministic-equivalent LP.
"""
struct RobustFluenceIMRTProblem <: ProblemGenerator
    case_data::RadiotherapyCaseData
    resolved_status::FeasibilityStatus
    scenario_shifts_cm::Vector{NTuple{3,Float64}}
    scenario_dose_matrices::Vector{SparseMatrixCSC{Float64,Int}}
    target_floor::Float64
    target_ceiling::Float64
    structure_max::Dict{Symbol,Float64}
    desired_dose::Vector{Float64}
    underdose_weight::Vector{Float64}
    overdose_weight::Vector{Float64}
    fluence_penalty::Float64
    smoothness_penalty::Float64
    feasible_witness::Union{Nothing,RadiotherapyFluenceWitness}
    infeasibility_certificate::Union{Nothing,RadiotherapyDoseConflictCertificate}
end

function RobustFluenceIMRTProblem(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int,
)
    n_scenarios = 3
    case, spec, conflict_indices, rng = _rt_build_case(
        target_variables, :robust_fluence, feasibility_status, seed;
        n_scenarios=n_scenarios,
    )
    target_floor, target_ceiling, structure_max, witness, certificate =
        _rt_hard_limits(case, spec, feasibility_status, conflict_indices, rng)

    shifts = NTuple{3,Float64}[(0.0, 0.0, 0.0)]
    phase = 2pi * rand(rng)
    for scenario in 2:n_scenarios
        angle = phase + 2pi * (scenario - 2) / (n_scenarios - 1)
        radius = 0.25 + 0.20rand(rng)
        push!(shifts, (radius * cos(angle), radius * sin(angle),
                       0.20 * (2rand(rng) - 1)))
    end
    matrices = SparseMatrixCSC{Float64,Int}[case.dose_matrix]
    for shift in shifts[2:end]
        matrix = _rt_dose_matrix(
            spec, case.voxel_locations_cm, case.voxel_structure,
            case.beam_of_beamlet, case.beamlet_u_cm, case.beamlet_z_cm,
            case.beamlet_width_cm, case.beamlet_height_cm, case.beam_energy_mv;
            setup_shift_cm=shift,
        )
        matrix .*= case.dose_normalization
        push!(matrices, matrix)
    end

    # The feasible artifact must satisfy every scenario, not just nominal dose.
    if feasibility_status == feasible
        target = case.structure_voxels[:ptv]
        scenario_doses = [matrix * case.reference_fluence for matrix in matrices]
        target_floor = min(target_floor,
                           0.99 * minimum(minimum(dose[target])
                                          for dose in scenario_doses))
        target_ceiling = max(target_ceiling,
                             1.01 * maximum(maximum(dose[target])
                                            for dose in scenario_doses))
        for structure in case.structure_names[2:end]
            indices = case.structure_voxels[structure]
            achieved = maximum(maximum(dose[indices])
                               for dose in scenario_doses)
            structure_max[structure] = max(structure_max[structure],
                                            1.04 * achieved)
        end
    end

    n_voxels = size(case.voxel_locations_cm, 1)
    desired_dose = zeros(n_voxels)
    underdose_weight = zeros(n_voxels)
    overdose_weight = zeros(n_voxels)
    kind_by_structure = Dict(zip(spec.structures, spec.kinds))
    for structure in spec.structures
        indices = case.structure_voxels[structure]
        volume_share = case.voxel_volume_cc[indices] ./
                       sum(case.voxel_volume_cc[indices])
        kind = kind_by_structure[structure]
        if kind == :target
            desired_dose[indices] .= 1.0
            underdose_weight[indices] .= (30.0 + 20rand(rng)) .* volume_share
            overdose_weight[indices] .= (8.0 + 7rand(rng)) .* volume_share
        else
            desired_dose[indices] .= 0.72 * spec.clinical_caps[structure]
            base = kind == :serial_oar ? 17.0 :
                   kind == :parallel_oar ? 7.0 : 1.0
            overdose_weight[indices] .= base * (0.8 + 0.4rand(rng)) .* volume_share
        end
    end
    fluence_penalty = (0.0012 + 0.0015rand(rng)) /
                      length(case.reference_fluence)
    smoothness_penalty = isempty(case.beamlet_edges) ? 0.0 :
        (0.015 + 0.020rand(rng)) / length(case.beamlet_edges)
    return RobustFluenceIMRTProblem(
        case, feasibility_status, shifts, matrices, target_floor,
        target_ceiling, structure_max, desired_dose, underdose_weight,
        overdose_weight, fluence_penalty, smoothness_penalty, witness,
        certificate,
    )
end

function _rt_robust_witness_is_valid(problem::RobustFluenceIMRTProblem;
                                     atol::Float64=1.0e-9)
    problem.feasible_witness === nothing && return false
    case = problem.case_data
    target = case.structure_voxels[:ptv]
    for matrix in problem.scenario_dose_matrices
        dose = matrix * problem.feasible_witness.fluence
        all(dose[target] .>= problem.target_floor - atol) || return false
        all(dose[target] .<= problem.target_ceiling + atol) || return false
        for (structure, upper) in problem.structure_max
            all(dose[case.structure_voxels[structure]] .<= upper + atol) ||
                return false
        end
    end
    return true
end

function build_model(problem::RobustFluenceIMRTProblem)
    model = Model()
    case = problem.case_data
    n_beamlets = length(case.reference_fluence)
    n_voxels = size(case.voxel_locations_cm, 1)
    target = case.structure_voxels[:ptv]
    n_edges = length(case.beamlet_edges)
    n_scenarios = length(problem.scenario_dose_matrices)

    @variable(model, fluence[1:n_beamlets] >= 0)
    @variable(model, underdose[target, 1:n_scenarios] >= 0)
    @variable(model, overdose[1:n_voxels, 1:n_scenarios] >= 0)
    @variable(model, variation[1:n_edges] >= 0)
    scenario_dose = Vector{Vector{AffExpr}}(undef, n_scenarios)
    for scenario in 1:n_scenarios
        dose = _rt_dose_expressions(
            model, fluence, problem.scenario_dose_matrices[scenario],
        )
        scenario_dose[scenario] = dose
        @constraint(model, [i in target],
                    dose[i] + underdose[i, scenario] >= problem.desired_dose[i],
                    base_name="scenario_$(scenario)_underdose_hinge")
        @constraint(model, [i in 1:n_voxels],
                    dose[i] - overdose[i, scenario] <= problem.desired_dose[i],
                    base_name="scenario_$(scenario)_overdose_hinge")
        @constraint(model, [i in target], dose[i] >= problem.target_floor,
                    base_name="scenario_$(scenario)_target_floor")
        @constraint(model, [i in target], dose[i] <= problem.target_ceiling,
                    base_name="scenario_$(scenario)_target_ceiling")
        for structure in case.structure_names[2:end]
            indices = case.structure_voxels[structure]
            @constraint(model, [i in indices],
                        dose[i] <= problem.structure_max[structure],
                        base_name="scenario_$(scenario)_$(structure)_maximum")
        end
    end
    @constraint(model, variation_positive[e in 1:n_edges],
                variation[e] >= fluence[case.beamlet_edges[e][1]] -
                                     fluence[case.beamlet_edges[e][2]])
    @constraint(model, variation_negative[e in 1:n_edges],
                variation[e] >= fluence[case.beamlet_edges[e][2]] -
                                     fluence[case.beamlet_edges[e][1]])
    @objective(
        model,
        Min,
        sum(problem.underdose_weight[i] * underdose[i, scenario]
            for i in target, scenario in 1:n_scenarios) / n_scenarios +
        sum(problem.overdose_weight[i] * overdose[i, scenario]
            for i in 1:n_voxels, scenario in 1:n_scenarios) / n_scenarios +
        problem.fluence_penalty * sum(fluence) +
        problem.smoothness_penalty * sum(variation),
    )
    model[:scenario_dose] = scenario_dose
    return model
end

register_variant(
    :radiotherapy,
    :robust_fluence,
    RobustFluenceIMRTProblem,
    "Scenario-based robust IMRT fluence-map LP with coherent rigid setup shifts",
)
