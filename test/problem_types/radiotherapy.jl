using SparseArrays
using Statistics

const RT_WEIGHTED = "radiotherapy/weighted_deviation"
const RT_MEAN_TAIL = "radiotherapy/mean_tail_dose"
const RT_MINMAX = "radiotherapy/minmax_deviation"
const RT_ROBUST = "radiotherapy/robust_fluence"
const RT_BEAM_ANGLE = "radiotherapy/beam_angle_selection"
const RT_VARIANTS = (RT_WEIGHTED, RT_MEAN_TAIL, RT_MINMAX, RT_ROBUST, RT_BEAM_ANGLE)

@testset "Radiotherapy category" begin
    @test :radiotherapy in list_categories()
    @test list_variants(:radiotherapy) == [
        :beam_angle_selection,
        :mean_tail_dose,
        :minmax_deviation,
        :robust_fluence,
        :weighted_deviation,
    ]
    @test problem_info(:radiotherapy)[:default_variant] == :weighted_deviation
    @test ProblemVariant(:radiotherapy) == ProblemVariant(:radiotherapy, :weighted_deviation)

    @testset "exact sizing and formulation blocks" begin
        for ref in RT_VARIANTS, target in (50, 100, 500, 2_000), seed in 0:3
            model, problem = generate_problem(ref, target, feasible, seed)
            case = problem.case_data
            n_beamlets = length(case.reference_fluence)
            n_voxels = size(case.voxel_locations_cm, 1)
            n_edges = length(case.beamlet_edges)
            expected = if ref == RT_MEAN_TAIL
                n_beamlets + 2 * n_voxels + length(case.structure_names) + n_edges
            elseif ref == RT_MINMAX
                n_beamlets + n_voxels + length(case.structure_voxels[:ptv]) + n_edges + 1
            elseif ref == RT_ROBUST
                n_beamlets +
                n_edges +
                length(problem.scenario_dose_matrices) *
                (n_voxels + length(case.structure_voxels[:ptv]))
            elseif ref == RT_BEAM_ANGLE
                n_beamlets +
                n_voxels +
                length(case.structure_voxels[:ptv]) +
                n_edges +
                length(case.beam_angles_deg)
            else
                n_beamlets + n_voxels + length(case.structure_voxels[:ptv]) + n_edges
            end
            @test num_variables(model) == expected
            if ref == RT_ROBUST && target == 50
                # Three complete scenario blocks cannot fit below the
                # two-voxels-per-structure anatomical floor for every profile.
                @test 45 <= expected <= 60
            else
                @test abs(expected - target) <= 1
            end
        end

        # Structural floors retain every profile structure even for tiny asks.
        for ref in (RT_WEIGHTED, RT_MEAN_TAIL, RT_MINMAX), seed in 0:3
            model, problem = generate_problem(ref, 1, feasible, seed)
            # Mean-tail carries two columns (dose, tail excess) per voxel.
            @test num_variables(model) <= (ref == RT_MEAN_TAIL ? 45 : 30)
            @test all(
                length(indices) >= 2 for indices in values(problem.case_data.structure_voxels)
            )
        end
    end

    @testset "clinical profiles and spatial dose data" begin
        expected_profiles = [:prostate, :head_neck, :c_shape, :liver, :lung, :breast]
        expected_beams = [7, 9, 9, 7, 9, 5]
        expected_prescriptions = [
            (74.0, 80.0), (50.0, 70.0), (48.0, 54.0), (45.0, 60.0), (50.0, 66.0), (40.0, 52.0)
        ]
        for seed in 0:5
            _, problem = generate_problem(RT_WEIGHTED, 500, feasible, seed)
            case = problem.case_data
            @test case.profile == expected_profiles[seed + 1]
            @test length(case.beam_angles_deg) == expected_beams[seed + 1]
            @test expected_prescriptions[seed + 1][1] <=
                case.prescription_gy <=
                expected_prescriptions[seed + 1][2]
            @test 10 <= case.n_fractions <= 44
            @test case.beam_energy_mv in (6, 10)
            @test case.dose_normalization > 0
            @test case.structure_names[1] == :ptv
            @test case.structure_kinds[1] == :target
            @test length(case.structure_names) == length(case.structure_kinds)
            @test sort(vcat(values(case.structure_voxels)...)) ==
                collect(1:size(case.voxel_locations_cm, 1))
            @test all(case.voxel_volume_cc .> 0)
            @test length(case.voxel_volume_cc) == length(case.voxel_structure)
            spec = SyntheticLPs._rt_profile_spec(case.profile)
            for structure in case.structure_names
                sampled_volume = sum(case.voxel_volume_cc[case.structure_voxels[structure]])
                lower, upper = spec.volumes_cc[structure]
                @test lower <= sampled_volume <= upper
            end

            matrix = case.dose_matrix
            @test matrix isa SparseMatrixCSC{Float64, Int}
            @test size(matrix) == (length(case.voxel_structure), length(case.reference_fluence))
            @test all(nonzeros(matrix) .> 0)
            @test all(diff(matrix.colptr) .> 0)  # every beamlet deposits dose
            @test all(vec(sum(matrix .> 0; dims=2)) .> 0)
            density = nnz(matrix) / length(matrix)
            @test 0.08 <= density <= 0.75
            @test size(matrix, 1) >= 2 * size(matrix, 2)

            # Field grids are disjoint, and TV edges join only adjacent
            # beamlets from the same field.
            @test all(1 <= b <= expected_beams[seed + 1] for b in case.beam_of_beamlet)
            @test all(
                case.beam_of_beamlet[a] == case.beam_of_beamlet[b] for (a, b) in case.beamlet_edges
            )
            @test all(a < b for (a, b) in case.beamlet_edges)
            @test length(unique(case.beamlet_edges)) == length(case.beamlet_edges)
            @test all(case.beamlet_width_cm .>= 0.45)
            @test all(case.beamlet_height_cm .>= 0.45)

            target_dose = case.reference_dose[case.structure_voxels[:ptv]]
            target_volume = case.voxel_volume_cc[case.structure_voxels[:ptv]]
            @test SyntheticLPs._rt_weighted_quantile(target_dose, target_volume, 0.5) ≈ 1.0 atol=1e-10
            @test SyntheticLPs._rt_weighted_quantile(target_dose, target_volume, 0.05) >= 0.90
            @test SyntheticLPs._rt_weighted_quantile(target_dose, target_volume, 0.98) <= 1.10
            @test maximum(target_dose) <= 1.16
        end

        # TG-119 field conventions are preserved exactly.
        _, prostate = generate_problem(RT_WEIGHTED, 500, feasible, 0)
        _, head_neck = generate_problem(RT_WEIGHTED, 500, feasible, 1)
        _, c_shape = generate_problem(RT_WEIGHTED, 500, feasible, 2)
        @test prostate.case_data.beam_angles_deg == [0.0, 50.0, 100.0, 150.0, 210.0, 260.0, 310.0]
        @test head_neck.case_data.beam_angles_deg == collect(0.0:40.0:320.0)
        @test c_shape.case_data.beam_angles_deg == collect(0.0:40.0:320.0)

        # As the reduced grid grows toward clinical resolution, local pencil
        # support produces a genuinely sparse CORT-like matrix.
        _, scaled = generate_problem(RT_MEAN_TAIL, 2_000, feasible, 0)
        scaled_matrix = scaled.case_data.dose_matrix
        @test nnz(scaled_matrix) / length(scaled_matrix) <= 0.20
        @test size(scaled_matrix, 1) >= 2.5 * size(scaled_matrix, 2)

        # The C-shape target is an annulus around the avoidance core.
        locations = c_shape.case_data.voxel_locations_cm
        target = c_shape.case_data.structure_voxels[:ptv]
        core = c_shape.case_data.structure_voxels[:core]
        @test all(1.5 <= hypot(locations[i, 1], locations[i, 2]) <= 3.7 for i in target)
        @test all(hypot(locations[i, 1], locations[i, 2]) <= 1.0 for i in core)
    end

    @testset "bounded nonzeros at clinical scale" begin
        # The per-field beamlet grid is capped at a clinical pitch, so large
        # requests add voxels rather than ever-finer beamlets: nonzeros stay
        # linear (the old generator reached ~30M at 100k variables).
        model, problem = generate_problem(RT_WEIGHTED, 100_000, feasible, 0)
        case = problem.case_data
        @test abs(num_variables(model) - 100_000) <= 1
        per_beam = length(case.reference_fluence) / length(case.beam_angles_deg)
        @test per_beam <= SyntheticLPs._RT_MAX_BEAMLETS_PER_BEAM
        @test nnz(case.dose_matrix) <= 5_000_000
        @test nnz(case.dose_matrix) / size(case.dose_matrix, 1) <= 80
        @test size(case.dose_matrix, 1) >= 20 * size(case.dose_matrix, 2)
    end

    @testset "setup shifts translate the full dose geometry" begin
        spec = (angles=[0.0], body=(10.0, 8.0, 10.0))
        locations = [0.0 0.0 0.0]
        shifted_locations = [1.0 0.0 0.0]
        dose_arguments = ([:ptv], [1], [0.0], [0.0], [1.0], [1.0], 6)

        nominal = SyntheticLPs._rt_dose_matrix(spec, locations, dose_arguments...)
        shifted_scenario = SyntheticLPs._rt_dose_matrix(
            spec, locations, dose_arguments...; setup_shift_cm=(1.0, 0.0, 0.0)
        )
        translated_anatomy = SyntheticLPs._rt_dose_matrix(
            spec, shifted_locations, dose_arguments...
        )

        @test shifted_scenario ≈ translated_anatomy
        @test shifted_scenario[1, 1] < nominal[1, 1]
    end

    @testset "constructive status artifacts" begin
        for ref in RT_VARIANTS, target in (50, 220, 500), seed in 0:7
            _, feasible_problem = generate_problem(ref, target, feasible, seed)
            @test feasible_problem.resolved_status == feasible
            @test feasible_problem.feasible_witness !== nothing
            @test feasible_problem.infeasibility_certificate === nothing
            @test SyntheticLPs._rt_witness_is_valid(feasible_problem)
            @test all(
                0 .<=
                feasible_problem.feasible_witness.fluence .<=
                feasible_problem.case_data.fluence_max,
            )
            if ref == RT_MEAN_TAIL
                @test SyntheticLPs._rt_mean_tail_witness_is_valid(feasible_problem)
            elseif ref == RT_ROBUST
                @test SyntheticLPs._rt_robust_witness_is_valid(feasible_problem)
            elseif ref == RT_BEAM_ANGLE
                @test SyntheticLPs._rt_beam_selection_witness_is_valid(feasible_problem)
            end

            _, infeasible_problem = generate_problem(ref, target, infeasible, seed)
            @test infeasible_problem.resolved_status == infeasible
            @test infeasible_problem.feasible_witness === nothing
            certificate = infeasible_problem.infeasibility_certificate
            @test certificate isa SyntheticLPs.RadiotherapyDoseConflictCertificate
            # Checked in every setup scenario for the robust variant.
            @test SyntheticLPs._rt_certificate_is_valid(infeasible_problem)
            @test certificate.organ in infeasible_problem.case_data.structure_names[2:end]
            n_target = length(infeasible_problem.case_data.structure_voxels[:ptv])
            @test min(3, n_target) <= length(certificate.target_voxels) <= 5
            @test allunique(certificate.target_voxels)
            @test certificate.organ_upper_bound <=
                0.86 * certificate.multiplier * certificate.target_lower_bound + 1e-12
            # Farkas arithmetic without a solver: any x >= 0 meeting the target
            # floor on the cited rows overdoses the organ voxel.
            for matrix in SyntheticLPs._rt_scenario_matrices(infeasible_problem)
                combination = zeros(size(matrix, 2))
                for (k, voxel) in enumerate(certificate.target_voxels)
                    combination .+= certificate.interpolation_weights[k] .* Vector(matrix[voxel, :])
                end
                organ_row = Vector(matrix[certificate.organ_voxel, :])
                @test organ_row ≈ certificate.multiplier .* combination
                # No single target row is parallel to the organ row, so the
                # contradiction is not a two-row (presolve-detectable) one.
                for voxel in certificate.target_voxels
                    row = Vector(matrix[voxel, :])
                    ratio = organ_row ./ max.(row, eps())
                    support = findall(!iszero, row)
                    @test findall(!iszero, organ_row) != support ||
                        maximum(ratio[support]) - minimum(ratio[support]) > 1e-6
                end
            end
            # The organ voxel sits inside the target cluster it interpolates.
            location = infeasible_problem.case_data.voxel_locations_cm
            @test vec(location[certificate.organ_voxel, :]) ≈ sum(
                certificate.interpolation_weights[k] .* vec(location[voxel, :]) for
                (k, voxel) in enumerate(certificate.target_voxels)
            )
        end

        for ref in RT_VARIANTS, seed in 0:15
            _, problem = generate_problem(ref, 220, unknown, seed)
            @test problem.resolved_status == unknown
            @test problem.feasible_witness === nothing
            @test problem.infeasibility_certificate === nothing
        end
    end

    @testset "model algebra" begin
        model, problem = generate_problem(RT_WEIGHTED, 100, feasible, 0)
        case = problem.case_data
        target_voxel = first(case.structure_voxels[:ptv])
        positive = findfirst(!iszero, Array(case.dose_matrix[target_voxel, :]))
        @test positive !== nothing
        j = something(positive)
        row = model[:target_dose_deviation][target_voxel]
        @test normalized_coefficient(row, model[:fluence][j]) == case.dose_matrix[target_voxel, j]
        @test normalized_coefficient(row, model[:underdose][target_voxel]) == 1.0
        @test normalized_coefficient(row, model[:overdose][target_voxel]) == -1.0
        @test normalized_rhs(row) == problem.desired_dose[target_voxel] == 1.0
        # Hard limits are deviation bounds: floor <= dose <= ceiling.
        @test upper_bound(model[:underdose][target_voxel]) ≈ 1.0 - problem.target_floor
        @test upper_bound(model[:overdose][target_voxel]) ≈ problem.target_ceiling - 1.0
        for structure in case.structure_names[2:end]
            voxel = first(case.structure_voxels[structure])
            organ_row = model[:organ_overdose][structure][voxel]
            @test normalized_coefficient(organ_row, model[:overdose][voxel]) == -1.0
            @test normalized_rhs(organ_row) == problem.desired_dose[voxel]
            @test 0 < problem.desired_dose[voxel] < problem.structure_max[structure]
            @test upper_bound(model[:overdose][voxel]) ≈
                problem.structure_max[structure] - problem.desired_dose[voxel]
        end
        # Each voxel's dose-influence row appears exactly once.
        dose_rows =
            length(model[:target_dose_deviation]) +
            sum(length(rows) for rows in values(model[:organ_overdose]))
        @test dose_rows == size(case.voxel_locations_cm, 1)
        @test num_constraints(model; count_variable_in_set_constraints=false) ==
            dose_rows + 2 * length(case.beamlet_edges)
        @test length(model[:variation_positive]) == length(case.beamlet_edges)
        @test length(model[:variation_negative]) == length(case.beamlet_edges)
        @test objective_sense(model) == MOI.MIN_SENSE
        @test all(upper_bound(x) == case.fluence_max for x in model[:fluence])
        @test case.fluence_max ≈ max(0.5, 3 * maximum(case.reference_fluence))

        model, problem = generate_problem(RT_MEAN_TAIL, 100, feasible, 1)
        case = problem.case_data
        n_voxels = size(case.voxel_locations_cm, 1)
        @test length(model[:tail_threshold]) == length(case.structure_names)
        @test length(model[:tail_excess]) == n_voxels
        @test length(model[:dose]) == n_voxels
        @test length(model[:dose_definition]) == n_voxels
        @test Set(keys(model[:tail_goal])) == Set(case.structure_names)
        @test Set(keys(model[:tail_excess_rows])) == Set(case.structure_names)
        target_index = first(case.structure_voxels[:ptv])
        @test lower_bound(model[:dose][target_index]) == problem.target_floor
        @test upper_bound(model[:dose][target_index]) == problem.target_ceiling
        for structure in case.structure_names[2:end]
            voxel = first(case.structure_voxels[structure])
            @test upper_bound(model[:dose][voxel]) == problem.structure_max[structure]
        end
        definition = model[:dose_definition][target_index]
        @test normalized_coefficient(definition, model[:dose][target_index]) == -1.0
        j = findfirst(!iszero, Array(case.dose_matrix[target_index, :]))
        @test normalized_coefficient(definition, model[:fluence][j]) ==
            case.dose_matrix[target_index, j]
        @test SyntheticLPs._rt_upper_tail_mean([1.0, 2.0, 3.0, 4.0], 0.5) == 3.5
        @test SyntheticLPs._rt_lower_tail_mean([1.0, 2.0, 3.0, 4.0], 0.5) == 1.5
        @test SyntheticLPs._rt_upper_tail_mean([1.0, 2.0, 3.0], [1.0, 1.0, 2.0], 0.5) == 3.0
        @test SyntheticLPs._rt_lower_tail_mean([1.0, 2.0, 3.0], [1.0, 1.0, 2.0], 0.5) == 1.5
        @test SyntheticLPs._rt_weighted_quantile([1.0, 2.0, 3.0], [1.0, 1.0, 2.0], 0.5) == 2.0
        for structure in case.structure_names
            @test 1 / length(case.structure_voxels[structure]) <=
                problem.tail_fraction[structure] <=
                1.0
        end
        target_volume = sum(case.voxel_volume_cc[case.structure_voxels[:ptv]])
        @test normalized_coefficient(model[:tail_goal][:ptv], model[:tail_excess][target_index]) ≈
            -case.voxel_volume_cc[target_index] / (problem.tail_fraction[:ptv] * target_volume)
        @test coefficient(objective_function(model), model[:fluence][1]) ≈ problem.fluence_penalty
        for structure in case.structure_names[2:end]
            indices = case.structure_voxels[structure]
            structure_volume = sum(case.voxel_volume_cc[indices])
            voxel = first(indices)
            @test coefficient(objective_function(model), model[:dose][voxel]) ≈
                problem.mean_dose_weight[structure] / structure_volume *
                  case.voxel_volume_cc[voxel]
        end

        model, problem = generate_problem(RT_MINMAX, 100, feasible, 2)
        @test haskey(model, :worst_deviation)
        @test length(model[:worst_underdose]) == length(problem.case_data.structure_voxels[:ptv])
        @test length(model[:worst_overdose]) == size(problem.case_data.voxel_locations_cm, 1)

        model, problem = generate_problem(RT_ROBUST, 220, feasible, 3)
        @test length(problem.scenario_shifts_cm) == 3
        @test problem.scenario_shifts_cm[1] == (0.0, 0.0, 0.0)
        @test problem.scenario_dose_matrices[1] === problem.case_data.dose_matrix
        @test all(
            size(matrix) == size(problem.case_data.dose_matrix) for
            matrix in problem.scenario_dose_matrices
        )
        @test all(all(nonzeros(matrix) .> 0) for matrix in problem.scenario_dose_matrices)
        @test any(
            problem.scenario_dose_matrices[s] != problem.scenario_dose_matrices[1] for s in 2:3
        )
        @test length(model[:scenario_dose]) == 3

        model, problem = generate_problem(RT_BEAM_ANGLE, 220, feasible, 4; relax_integer=false)
        case = problem.case_data
        @test length(case.beam_angles_deg) == 12
        @test problem.minimum_open_beams < problem.maximum_open_beams
        @test problem.minimum_open_beams <=
            length(problem.reference_open_beams) <=
            problem.maximum_open_beams
        @test all(is_binary(model[:beam_open][b]) for b in eachindex(case.beam_angles_deg))
        @test coefficient(objective_function(model), model[:beam_open][1]) ==
            problem.beam_open_penalty
        @test length(model[:beamlet_link]) == length(case.reference_fluence)
        for j in eachindex(case.reference_fluence)
            row = model[:beamlet_link][j]
            beam = case.beam_of_beamlet[j]
            @test normalized_coefficient(row, model[:fluence][j]) == 1.0
            @test normalized_coefficient(row, model[:beam_open][beam]) ==
                -problem.beamlet_fluence_max[j]
        end
    end

    @testset "local RNG and reproducibility" begin
        Random.seed!(85_201)
        expected_draw = rand()
        Random.seed!(85_201)
        generate_problem(RT_WEIGHTED, 220, feasible, 91)
        @test rand() == expected_draw

        for ref in RT_VARIANTS, status in (feasible, infeasible, unknown)
            model_a, a = generate_problem(ref, 500, status, 12_345)
            model_b, b = generate_problem(ref, 500, status, 12_345)
            @test a.case_data.profile == b.case_data.profile
            @test a.case_data.voxel_locations_cm == b.case_data.voxel_locations_cm
            @test a.case_data.beam_angles_deg == b.case_data.beam_angles_deg
            @test a.case_data.beamlet_edges == b.case_data.beamlet_edges
            @test a.case_data.dose_matrix == b.case_data.dose_matrix
            @test a.case_data.reference_fluence == b.case_data.reference_fluence
            @test a.structure_max == b.structure_max
            @test num_variables(model_a) == num_variables(model_b)
            @test num_constraints(model_a; count_variable_in_set_constraints=true) ==
                num_constraints(model_b; count_variable_in_set_constraints=true)
        end
    end

    if HAS_HIGHS
        @testset "solver-verified feasibility contracts" begin
            for ref in RT_VARIANTS, status in (feasible, infeasible), seed in 0:7
                model, _ = generate_problem(ref, 220, status, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
                @test termination_status(model) == expected
            end

            # Presolve keeps the model, and the overlap conflict needs real
            # simplex work (the old coincident-voxel pair was refuted by
            # presolve with zero iterations).
            for ref in RT_VARIANTS
                model, _ = generate_problem(ref, 1_500, infeasible, 3)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                MOI.Utilities.attach_optimizer(model)
                highs = unsafe_backend(model)
                HiGHS.Highs_presolve(highs)
                @test HiGHS.Highs_getPresolvedNumCol(highs) >= 0.8 * num_variables(model)
                optimize!(model)
                @test termination_status(model) == MOI.INFEASIBLE
                @test MOI.get(model, MOI.SimplexIterations()) >= 20
            end

            # Unknown uses independent clinical tightness: the deterministic
            # sample contains both usable and overconstrained plans.
            for ref in RT_VARIANTS
                statuses = Set{MOI.TerminationStatusCode}()
                for seed in 0:31
                    model, _ = generate_problem(ref, 220, unknown, seed)
                    set_optimizer(model, HiGHS.Optimizer)
                    set_silent(model)
                    optimize!(model)
                    push!(statuses, termination_status(model))
                end
                @test MOI.OPTIMAL in statuses
                @test MOI.INFEASIBLE in statuses
            end

            # The natural BAO MILP also accepts the integral planted field set.
            model, _ = generate_problem(RT_BEAM_ANGLE, 100, feasible, 5; relax_integer=false)
            set_optimizer(model, HiGHS.Optimizer)
            set_silent(model)
            optimize!(model)
            @test termination_status(model) == MOI.OPTIMAL
        end
    end
end
