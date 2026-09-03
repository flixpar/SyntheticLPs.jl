using Test
using Random
using JuMP
using SyntheticLPs

const BIN_PACKING_MOI = JuMP.MOI
const BIN_PACKING_STANDARD = "bin_packing/standard"
const BIN_PACKING_HETEROGENEOUS = "bin_packing/heterogeneous"
const BIN_PACKING_REFS = (BIN_PACKING_STANDARD, BIN_PACKING_HETEROGENEOUS)
const HAS_BIN_PACKING_HIGHS = try
    @eval using HiGHS
    true
catch
    false
end

function assert_common_bin_packing_data(problem)
    @test problem.n_items >= 3
    @test problem.n_bins >= 2
    @test 2 <= problem.n_categories <= 8
    @test problem.actual_variables == problem.n_bins * (problem.n_items + problem.n_categories + 1)
    @test length(problem.item_sizes) == problem.n_items
    @test all(>(0.0), problem.item_sizes)
    @test length(problem.item_categories) == problem.n_items
    @test all(category -> 1 <= category <= problem.n_categories, problem.item_categories)
    @test all(category -> count(==(category), problem.item_categories) >= 1, 1:problem.n_categories)
    @test length(problem.category_names) == problem.n_categories
    @test length(unique(problem.category_names)) == problem.n_categories
    @test problem.load_profile in
        (:guaranteed_feasible, :aggregate_overload, :light, :nominal, :surge)
    @test problem.incompatible_pairs == sort(unique(problem.incompatible_pairs))
    @test all(first(pair) < last(pair) for pair in problem.incompatible_pairs)
    @test all(
        pair -> 1 <= first(pair) < last(pair) <= problem.n_categories, problem.incompatible_pairs
    )
end

function assert_standard_witness(problem)
    witness = problem.feasible_witness
    @test witness !== nothing
    witness === nothing && return nothing
    @test problem.infeasibility_certificate === nothing
    @test SyntheticLPs.validate_bin_packing_witness(problem)
    @test length(witness) == problem.n_items
    @test all(item -> 1 <= witness[item] <= min(item, problem.n_bins), 1:problem.n_items)

    used_bins = sort(unique(witness))
    @test used_bins == collect(1:maximum(used_bins))
    for bin in used_bins
        items = findall(==(bin), witness)
        @test sum(problem.item_sizes[items]) <= problem.bin_capacity + 1e-8
        categories = Set(problem.item_categories[items])
        for (first_category, second_category) in problem.incompatible_pairs
            @test !(first_category in categories && second_category in categories)
        end
    end
end

function assert_heterogeneous_witness(problem)
    witness = problem.feasible_witness
    @test witness !== nothing
    witness === nothing && return nothing
    @test problem.infeasibility_certificate === nothing
    @test SyntheticLPs.validate_bin_packing_witness(problem)
    @test length(witness) == problem.n_items

    for item in 1:problem.n_items
        bin = witness[item]
        bin_type = problem.bin_types[bin]
        @test problem.type_category_compatibility[bin_type, problem.item_categories[item]]
    end
    for bin in sort(unique(witness))
        items = findall(==(bin), witness)
        bin_type = problem.bin_types[bin]
        @test sum(problem.item_sizes[items]) <= problem.type_capacities[bin_type] + 1e-8
        categories = Set(problem.item_categories[items])
        for (first_category, second_category) in problem.incompatible_pairs
            @test !(first_category in categories && second_category in categories)
        end
    end
    for bin_type in 1:problem.n_bin_types
        slots = findall(==(bin_type), problem.bin_types)
        used_slots = [slot for slot in slots if slot in witness]
        @test used_slots == slots[1:length(used_slots)]
    end
end

function assert_capacity_certificate(problem)
    certificate = problem.infeasibility_certificate
    @test problem.feasible_witness === nothing
    @test certificate !== nothing
    certificate === nothing && return nothing
    @test SyntheticLPs.validate_bin_packing_certificate(problem)

    aggregate_capacity = if problem isa SyntheticLPs.BinPackingProblem
        problem.n_bins * problem.bin_capacity
    else
        sum(problem.type_capacities[bin_type] for bin_type in problem.bin_types)
    end
    @test certificate.total_item_size ≈ sum(problem.item_sizes)
    @test certificate.total_available_capacity ≈ aggregate_capacity
    @test certificate.excess ≈ certificate.total_item_size - certificate.total_available_capacity
    @test certificate.excess > 0.0
end

function assert_complete_witness_start(model, problem)
    witness = something(problem.feasible_witness)
    used = falses(problem.n_bins)
    present = falses(problem.n_categories, problem.n_bins)
    for item in 1:problem.n_items
        bin = witness[item]
        used[bin] = true
        present[problem.item_categories[item], bin] = true
    end

    for item in 1:problem.n_items, bin in 1:problem.n_bins
        @test start_value(model[:x][item, bin]) == (witness[item] == bin ? 1.0 : 0.0)
    end
    for bin in 1:problem.n_bins
        @test start_value(model[:y][bin]) == (used[bin] ? 1.0 : 0.0)
        for category in 1:problem.n_categories
            @test start_value(model[:category_present][category, bin]) ==
                (present[category, bin] ? 1.0 : 0.0)
        end
    end
end

@testset "Bin-packing generator quality" begin
    @testset "Registry and target sizing" begin
        @test list_variants(:bin_packing) == [:heterogeneous, :standard]
        @test ProblemVariant(:bin_packing) == ProblemVariant(:bin_packing, :standard)
        @test problem_info(:bin_packing)[:default_variant] == :standard

        expected_sizes = Dict(
            2 => 12,
            11 => 12,
            12 => 12,
            49 => 48,
            50 => 51,
            99 => 100,
            100 => 100,
            101 => 100,
            249 => 248,
            250 => 250,
            251 => 252,
            999 => 1000,
            1000 => 1000,
            1001 => 1001,
            5000 => 5000,
            10_000 => 10_000,
        )
        for ref in BIN_PACKING_REFS, (target, expected) in expected_sizes
            model, problem = generate_problem(ref, target, unknown, 17)
            @test problem.target_variables == target
            @test problem.actual_variables == expected
            @test num_variables(model) == expected
            assert_common_bin_packing_data(problem)
        end
    end

    @testset "Local RNG and deterministic fields" begin
        for ref in BIN_PACKING_REFS
            Random.seed!(73_109)
            first_draw = rand()
            expected_next_draw = rand()
            Random.seed!(73_109)
            @test rand() == first_draw
            generate_problem(ref, 250, feasible, 12_345)
            @test rand() == expected_next_draw

            for status in (feasible, infeasible, unknown)
                _, first_problem = generate_problem(ref, 250, status, 1_234)
                _, second_problem = generate_problem(ref, 250, status, 1_234)
                for field in fieldnames(typeof(first_problem))
                    first_value = getfield(first_problem, field)
                    second_value = getfield(second_problem, field)
                    if field == :infeasibility_certificate && first_value !== nothing
                        @test second_value !== nothing
                        @test all(
                            isequal(
                                getfield(first_value, certificate_field),
                                getfield(second_value, certificate_field),
                            ) for certificate_field in fieldnames(typeof(first_value))
                        )
                    else
                        @test isequal(first_value, second_value)
                    end
                end
            end
        end
    end

    @testset "Identical-bin evidence and formulation" begin
        for target in (12, 50, 250, 1001), seed in 0:12
            problem = SyntheticLPs.BinPackingProblem(target, feasible, seed)
            assert_common_bin_packing_data(problem)
            @test problem.load_profile == :guaranteed_feasible
            assert_standard_witness(problem)
            @test !SyntheticLPs.validate_bin_packing_certificate(problem)
        end

        for target in (12, 50, 250, 1001), seed in 0:12
            problem = SyntheticLPs.BinPackingProblem(target, infeasible, seed)
            assert_common_bin_packing_data(problem)
            @test problem.load_profile == :aggregate_overload
            assert_capacity_certificate(problem)
            @test !SyntheticLPs.validate_bin_packing_witness(problem)
        end

        problem = SyntheticLPs.BinPackingProblem(120, feasible, 9)
        model = SyntheticLPs.build_model(problem)
        @test length(model[:item_assignment]) == problem.n_items
        @test length(model[:bin_capacity]) == problem.n_bins
        @test length(model[:presence_lower]) == problem.n_items * problem.n_bins
        @test length(model[:presence_upper]) == problem.n_categories * problem.n_bins
        @test length(model[:presence_used]) == problem.n_categories * problem.n_bins
        @test length(model[:category_conflict]) ==
            length(problem.incompatible_pairs) * problem.n_bins
        @test length(model[:used_prefix]) == problem.n_bins - 1
        expected_canonical_rows = sum(max(problem.n_bins - item, 0) for item in 1:problem.n_items)
        @test length(model[:canonical_label]) == expected_canonical_rows
        if expected_canonical_rows > 0
            row = model[:canonical_label][1, 2]
            @test constraint_object(row).set == BIN_PACKING_MOI.EqualTo(0.0)
            @test normalized_coefficient(row, model[:x][1, 2]) == 1.0
        end
        assert_complete_witness_start(model, problem)
    end

    @testset "Typed-fleet evidence and formulation" begin
        for target in (12, 50, 250, 1001), seed in 0:12
            problem = SyntheticLPs.HeterogeneousBinPackingProblem(target, feasible, seed)
            assert_common_bin_packing_data(problem)
            @test problem.load_profile == :guaranteed_feasible
            @test 2 <= problem.n_bin_types <= 4
            @test length(problem.bin_types) == problem.n_bins
            @test length(problem.bin_type_names) == problem.n_bin_types
            @test length(unique(problem.bin_type_names)) == problem.n_bin_types
            @test [count(==(bin_type), problem.bin_types) for bin_type in 1:problem.n_bin_types] == problem.type_availability
            @test sum(problem.type_availability) == problem.n_bins
            @test all(>(0), problem.type_availability)
            @test length(unique(problem.type_capacities)) == problem.n_bin_types
            @test length(unique(problem.type_costs)) == problem.n_bin_types
            @test size(problem.type_category_compatibility) ==
                (problem.n_bin_types, problem.n_categories)
            @test all(
                any(view(problem.type_category_compatibility, :, category)) for
                category in 1:problem.n_categories
            )
            @test any(!, problem.type_category_compatibility)
            assert_heterogeneous_witness(problem)
            @test !SyntheticLPs.validate_bin_packing_certificate(problem)
        end

        for target in (12, 50, 250, 1001), seed in 0:12
            problem = SyntheticLPs.HeterogeneousBinPackingProblem(target, infeasible, seed)
            assert_common_bin_packing_data(problem)
            @test problem.load_profile == :aggregate_overload
            assert_capacity_certificate(problem)
            @test !SyntheticLPs.validate_bin_packing_witness(problem)
        end

        problem = SyntheticLPs.HeterogeneousBinPackingProblem(120, feasible, 9)
        model = SyntheticLPs.build_model(problem)
        @test objective_function(model) isa JuMP.AffExpr
        @test length(model[:item_assignment]) == problem.n_items
        @test length(model[:bin_capacity]) == problem.n_bins
        @test length(model[:presence_lower]) == problem.n_items * problem.n_bins
        @test length(model[:presence_upper]) == problem.n_categories * problem.n_bins
        @test length(model[:presence_used]) == problem.n_categories * problem.n_bins
        @test length(model[:category_conflict]) ==
            length(problem.incompatible_pairs) * problem.n_bins
        expected_eligibility_rows = count(
            !problem.type_category_compatibility[
                problem.bin_types[bin], problem.item_categories[item]
            ] for item in 1:problem.n_items, bin in 1:problem.n_bins
        )
        @test length(model[:category_eligibility]) == expected_eligibility_rows
        if expected_eligibility_rows > 0
            row_index = first(eachindex(model[:category_eligibility]))
            row = model[:category_eligibility][row_index]
            item, bin = Tuple(row_index)
            @test constraint_object(row).set == BIN_PACKING_MOI.EqualTo(0.0)
            @test normalized_coefficient(row, model[:x][item, bin]) == 1.0
        end

        @test length(model[:used_type_prefix]) == problem.n_bins - problem.n_bin_types
        if !isempty(model[:used_type_prefix])
            row = first(model[:used_type_prefix])
            coefficients = [
                normalized_coefficient(row, model[:y][bin]) for
                bin in 1:problem.n_bins if normalized_coefficient(row, model[:y][bin]) != 0.0
            ]
            @test sort(coefficients) == [-1.0, 1.0]
        end
        assert_complete_witness_start(model, problem)

        # The certificate recomputes capacity from concrete fleet slots and
        # separately audits the advertised availability counts.
        certificate_problem = SyntheticLPs.HeterogeneousBinPackingProblem(250, infeasible, 31)
        @test SyntheticLPs.validate_bin_packing_certificate(certificate_problem)
        original_type = certificate_problem.bin_types[1]
        replacement_type = original_type == certificate_problem.n_bin_types ? 1 : original_type + 1
        certificate_problem.bin_types[1] = replacement_type
        @test !SyntheticLPs.validate_bin_packing_certificate(certificate_problem)
        certificate_problem.type_availability[original_type] -= 1
        certificate_problem.type_availability[replacement_type] += 1
        @test !SyntheticLPs.validate_bin_packing_certificate(certificate_problem)
        certificate_problem.bin_types[1] = original_type
        @test !SyntheticLPs.validate_bin_packing_certificate(certificate_problem)
        certificate_problem.type_availability[original_type] += 1
        certificate_problem.type_availability[replacement_type] -= 1
        @test SyntheticLPs.validate_bin_packing_certificate(certificate_problem)
    end

    @testset "Unknown pressure profiles and raw native status mix" begin
        if HAS_BIN_PACKING_HIGHS
            for ref in BIN_PACKING_REFS, target in (40, 120, 400)
                observed_profiles = Set{Symbol}()
                observed_statuses = Set{FeasibilityStatus}()
                for seed in 0:9
                    model, problem = generate_problem(
                        ref, target, unknown, seed; relax_integer=false
                    )
                    @test problem.feasibility_status == unknown
                    @test problem.feasible_witness === nothing
                    @test problem.infeasibility_certificate === nothing
                    @test !SyntheticLPs.validate_bin_packing_witness(problem)
                    @test !SyntheticLPs.validate_bin_packing_certificate(problem)
                    @test problem.load_profile in (:light, :nominal, :surge)
                    push!(observed_profiles, problem.load_profile)
                    @test all(isnothing(start_value(variable)) for variable in model[:x])
                    @test all(isnothing(start_value(variable)) for variable in model[:y])
                    @test all(
                        isnothing(start_value(variable)) for variable in model[:category_present]
                    )

                    set_optimizer(model, HiGHS.Optimizer)
                    set_silent(model)
                    set_time_limit_sec(model, 15.0)
                    optimize!(model)
                    if termination_status(model) == BIN_PACKING_MOI.INFEASIBLE
                        push!(observed_statuses, infeasible)
                        @test problem.load_profile == :surge
                    else
                        @test primal_status(model) == BIN_PACKING_MOI.FEASIBLE_POINT
                        push!(observed_statuses, feasible)
                        @test problem.load_profile in (:light, :nominal)
                    end
                end
                @test observed_profiles == Set((:light, :nominal, :surge))
                @test observed_statuses == Set((feasible, infeasible))
            end
        else
            @info "HiGHS unavailable; skipping unknown-profile solve checks"
        end
    end

    @testset "Relaxed and native status contracts" begin
        if HAS_BIN_PACKING_HIGHS
            for ref in BIN_PACKING_REFS,
                relax_integer in (true, false),
                target in (12, 40, 80, 120),
                status in (feasible, infeasible),
                seed in 0:3

                model, problem = generate_problem(
                    ref, target, status, seed; relax_integer=relax_integer
                )
                @test num_variables(model) == problem.actual_variables
                @test is_binary(model[:x][1, 1]) == !relax_integer
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                set_time_limit_sec(model, 15.0)
                optimize!(model)
                expected = status == feasible ? BIN_PACKING_MOI.OPTIMAL : BIN_PACKING_MOI.INFEASIBLE
                @test termination_status(model) == expected
            end
        else
            @info "HiGHS unavailable; skipping bin-packing solve checks"
        end
    end

    @testset "Large native starts produce an incumbent" begin
        if HAS_BIN_PACKING_HIGHS
            for ref in BIN_PACKING_REFS
                model, problem = generate_problem(ref, 2000, feasible, 71; relax_integer=false)
                assert_complete_witness_start(model, problem)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                set_time_limit_sec(model, 8.0)
                optimize!(model)
                @test termination_status(model) in
                    (BIN_PACKING_MOI.OPTIMAL, BIN_PACKING_MOI.TIME_LIMIT)
                @test primal_status(model) == BIN_PACKING_MOI.FEASIBLE_POINT
            end
        else
            @info "HiGHS unavailable; skipping large native-start checks"
        end
    end
end
