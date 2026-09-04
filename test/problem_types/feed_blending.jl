using Test
using Random
using JuMP
using SyntheticLPs

@testset "Feed blending / standard" begin
    @testset "typed data and exact sizing" begin
        for target in (1, 3, 50, 250, 251, 1_000, 1_001), seed in 0:3
            model, problem = generate_problem("feed_blending/standard", target, unknown, seed)
            @test num_variables(model) == max(3, target)
            @test problem.num_ingredients == max(3, target)
            @test problem.requested_status == unknown
            @test problem.feasible_witness === nothing
            @test problem.infeasibility_certificate === nothing
            @test size(problem.nutrient_content) == (problem.num_nutrients, problem.num_ingredients)
            @test length(problem.ingredient_types) == problem.num_ingredients
            @test length(problem.nutrient_types) == problem.num_nutrients
            @test all(>(0.0), problem.costs)
            @test all(>=(0.0), problem.nutrient_content)
            @test all(sum(problem.nutrient_content[j, :]) > 0.0 for j in 1:problem.num_nutrients)
            @test all(sum(problem.nutrient_content[:, i]) > 0.0 for i in 1:problem.num_ingredients)
            @test Set(problem.nutrient_types) == Set((
                SyntheticLPs.feed_major_nutrient,
                SyntheticLPs.feed_mineral,
                SyntheticLPs.feed_trace_nutrient,
                SyntheticLPs.feed_restricted_compound,
            ))
            if problem.num_ingredients >= 4
                @test Set(problem.ingredient_types) == Set((
                    SyntheticLPs.feed_energy_source,
                    SyntheticLPs.feed_protein_source,
                    SyntheticLPs.feed_mineral_supplement,
                    SyntheticLPs.feed_specialty_additive,
                ))
            end
            @test problem.ratio_constraints isa Vector{SyntheticLPs.FeedRatioConstraint}
            @test all(
                1 <= constraint.nutrient <= problem.num_nutrients &&
                    isfinite(constraint.target) &&
                    constraint.target >= 0.0 &&
                    constraint.sense in
                    (SyntheticLPs.feed_ratio_minimum, SyntheticLPs.feed_ratio_maximum) for
                constraint in problem.ratio_constraints
            )
        end
    end

    @testset "constructor-local RNG and reproducibility" begin
        Random.seed!(91_733)
        expected_first = rand()
        expected_second = rand()
        Random.seed!(91_733)
        actual_first = rand()
        generate_problem("feed_blending/standard", 80, feasible, 17)
        actual_second = rand()
        @test actual_first == expected_first
        @test actual_second == expected_second

        _, first = generate_problem("feed_blending/standard", 80, infeasible, 1_234)
        _, second = generate_problem("feed_blending/standard", 80, infeasible, 1_234)
        @test first.batch_size == second.batch_size
        @test first.ingredient_types == second.ingredient_types
        @test first.costs == second.costs
        @test first.nutrient_content == second.nutrient_content
        @test first.nutrient_types == second.nutrient_types
        @test first.min_requirements == second.min_requirements
        @test first.max_limits == second.max_limits
        @test first.availabilities == second.availabilities
        @test first.ratio_constraints == second.ratio_constraints
        @test first.infeasibility_certificate == second.infeasibility_certificate
    end

    @testset "feasible recipe witness" begin
        for target in (3, 25, 120, 500, 1_200), seed in 0:12
            _, problem = generate_problem("feed_blending/standard", target, feasible, seed)
            @test problem.requested_status == feasible
            @test problem.feasible_witness !== nothing
            @test problem.infeasibility_certificate === nothing
            @test SyntheticLPs.feed_recipe_satisfies(problem)

            recipe = problem.feasible_witness
            @test sum(recipe) ≈ problem.batch_size
            @test all(>=(0.0), recipe)
            @test all(
                !isfinite(problem.availabilities[i]) ||
                    recipe[i] <= problem.availabilities[i] + 1e-8 * problem.batch_size for
                i in 1:problem.num_ingredients
            )
        end
    end

    @testset "checkable infeasibility certificates" begin
        observed_kinds = Set{SyntheticLPs.FeedInfeasibilityKind}()
        maximum_below_minimum_case = nothing

        for target in (25, 120, 500), seed in 0:79
            model, problem = generate_problem("feed_blending/standard", target, infeasible, seed)
            certificate = problem.infeasibility_certificate
            @test problem.requested_status == infeasible
            @test problem.feasible_witness === nothing
            @test certificate !== nothing
            @test SyntheticLPs.feed_infeasibility_certificate_holds(problem)
            push!(observed_kinds, certificate.kind)

            if certificate.kind == SyntheticLPs.feed_maximum_ratio_below_achievable_minimum
                ratio = problem.ratio_constraints[certificate.ratio_constraint]
                @test ratio.sense == SyntheticLPs.feed_ratio_maximum
                @test ratio.target == certificate.required_bound < certificate.achievable_bound

                # Regression for the old string parser: the phrase "below
                # achievable minimum" must create a <= row, not a >= row.
                row = model[:ratio_max][certificate.ratio_constraint]
                row_object = constraint_object(row)
                @test row_object.set isa JuMP.MOI.LessThan{Float64}
                @test row_object.set.upper == 0.0
                maximum_below_minimum_case = certificate.kind
            end
        end

        @test observed_kinds == Set((
            SyntheticLPs.feed_minimum_ratio_above_achievable_maximum,
            SyntheticLPs.feed_maximum_ratio_below_achievable_minimum,
            SyntheticLPs.feed_minimum_nutrient_above_achievable_maximum,
            SyntheticLPs.feed_insufficient_ingredient_capacity,
        ))
        @test maximum_below_minimum_case !== nothing
    end

    @testset "ratio formulation directions" begin
        for seed in 0:40
            model, problem = generate_problem("feed_blending/standard", 80, feasible, seed)
            for (index, ratio) in enumerate(problem.ratio_constraints)
                if ratio.sense == SyntheticLPs.feed_ratio_minimum
                    object = constraint_object(model[:ratio_min][index])
                    @test object.set isa JuMP.MOI.GreaterThan{Float64}
                    @test object.set.lower == 0.0
                    row = model[:ratio_min][index]
                else
                    object = constraint_object(model[:ratio_max][index])
                    @test object.set isa JuMP.MOI.LessThan{Float64}
                    @test object.set.upper == 0.0
                    row = model[:ratio_max][index]
                end
                for ingredient in 1:problem.num_ingredients
                    @test normalized_coefficient(row, model[:x][ingredient]) ≈
                        problem.nutrient_content[ratio.nutrient, ingredient] - ratio.target
                end
            end
        end
    end
end

# HiGHS is a test-only dependency of the package. Keep this file runnable in a
# plain development environment while exercising the status contract under
# `Pkg.test()`, where HiGHS is available.
const FEED_BLENDING_TEST_HAS_HIGHS = try
    @eval using HiGHS
    true
catch
    false
end

if FEED_BLENDING_TEST_HAS_HIGHS
    @testset "Feed blending / standard solver status" begin
        for target in (25, 120, 500), status in (feasible, infeasible), seed in 0:9
            model, _ = generate_problem("feed_blending/standard", target, status, seed)
            set_optimizer(model, HiGHS.Optimizer)
            set_silent(model)
            optimize!(model)
            expected = status == feasible ? JuMP.MOI.OPTIMAL : JuMP.MOI.INFEASIBLE
            @test termination_status(model) == expected
        end
    end
end
