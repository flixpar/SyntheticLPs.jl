using Test
using JuMP
using Random
using Distributions
using JSON

using SyntheticLPs

"""
    test_problem_generator(problem_type::Symbol)

Test the problem generator for the given problem type.
"""
function test_problem_generator(problem_type::Symbol)
    @testset "$(problem_type) Problem Generator" begin
        # Test with different target variable counts
        for target_vars in [50, 100, 500]
            @test_nowarn begin
                model, problem = generate_problem(problem_type, target_vars, unknown, 0)
                @test model isa JuMP.Model
                @test problem isa ProblemGenerator

                # Check that the model has variables, constraints, and an objective
                actual_var_count = num_variables(model)
                @test actual_var_count > 0
                @test num_constraints(model, count_variable_in_set_constraints=true) > 0
                @test objective_function(model) !== nothing

                # Check that variable count is within ±20% of target for most cases
                # Some problem types may have additional variables (e.g., portfolio has n_options + 1)
                error_percentage = abs(actual_var_count - target_vars) / target_vars * 100
                @test error_percentage <= 25.0 || actual_var_count <= 50  # Allow higher error for small problems
            end
        end

        # Test with different feasibility statuses
        for feas_status in [feasible, infeasible, unknown]
            @test_nowarn begin
                model, problem = generate_problem(problem_type, 100, feas_status, 0)
                @test model isa JuMP.Model
                @test problem isa ProblemGenerator
                @test num_variables(model) > 0
            end
        end

        # Test with a fixed seed for reproducibility
        seed = 12345
        @test_nowarn begin
            # Generate the same problem twice with the same seed
            model1, problem1 = generate_problem(problem_type, 150, unknown, seed)
            model2, problem2 = generate_problem(problem_type, 150, unknown, seed)

            # Verify that the models are identical (same number of vars and constraints)
            @test num_variables(model1) == num_variables(model2)
            @test num_constraints(model1, count_variable_in_set_constraints=true) == num_constraints(model2, count_variable_in_set_constraints=true)

            # Verify that problem instances are identical (same struct type and data)
            @test typeof(problem1) == typeof(problem2)
        end
    end
end

# Run tests for all registered problem types
@testset "SyntheticLPs" begin
    # Test core functionality
    @testset "Core Functionality" begin
        # Test listing problem types
        problem_types = list_problem_types()
        @test problem_types isa Vector{Symbol}
        @test !isempty(problem_types)

        # Test getting problem info
        for problem_type in problem_types
            info = problem_info(problem_type)
            @test info isa Dict
            @test haskey(info, :description)
            @test info[:description] isa String
        end

        # Test random problem generation
        @test_nowarn begin
            # Test with target variables
            model, type, problem = generate_random_problem(100)
            @test model isa JuMP.Model
            @test type isa Symbol
            @test problem isa ProblemGenerator
            @test num_variables(model) > 0

            # Test with feasibility status
            model2, type2, problem2 = generate_random_problem(100; feasibility_status=feasible)
            @test model2 isa JuMP.Model
            @test type2 isa Symbol
            @test problem2 isa ProblemGenerator
            @test num_variables(model2) > 0
        end

        # Test FeasibilityStatus enum
        @test feasible isa FeasibilityStatus
        @test infeasible isa FeasibilityStatus
        @test unknown isa FeasibilityStatus
    end

    # Test individual problem generators
    for problem_type in list_problem_types()
        test_problem_generator(problem_type)
    end

    # Test batch dataset generation
    @testset "Dataset Generation" begin
        # Basic in-memory generation (no solver required)
        instances = generate_dataset(num_problems = 6, var_mean = 80, var_std = 20,
                                     var_min = 30, var_max = 150, seed = 123,
                                     problem_types = [:transportation, :knapsack],
                                     max_candidate_multiplier = 2)
        @test instances isa Vector{GeneratedInstance}
        @test length(instances) == 6
        @test all(inst -> inst.num_variables > 0, instances)
        @test all(inst -> inst.num_constraints >= 0, instances)
        @test [inst.index for inst in instances] == collect(1:6)
        @test all(inst -> inst.filename === nothing, instances)  # no output_dir

        # Reproducibility: same seed → identical dataset
        instances2 = generate_dataset(num_problems = 6, var_mean = 80, var_std = 20,
                                      var_min = 30, var_max = 150, seed = 123,
                                      problem_types = [:transportation, :knapsack],
                                      max_candidate_multiplier = 2)
        @test [i.problem_type for i in instances] == [i.problem_type for i in instances2]
        @test [i.num_variables for i in instances] == [i.num_variables for i in instances2]
        @test [i.seed for i in instances] == [i.seed for i in instances2]

        # Restricting problem types is respected
        subset = generate_dataset(num_problems = 5, var_mean = 80, var_std = 20,
                                  var_min = 30, var_max = 150, seed = 1,
                                  problem_types = [:transportation, :knapsack],
                                  max_candidate_multiplier = 2)
        @test all(inst -> inst.problem_type in (:transportation, :knapsack), subset)

        # Direct Distributions.jl size distributions are accepted.
        uniform_subset = generate_dataset(num_problems = 6,
                                          size_distribution = Uniform(30, 150),
                                          problem_types = [:transportation, :knapsack],
                                          seed = 2,
                                          max_candidate_multiplier = 2)
        @test length(uniform_subset) == 6
        @test all(inst -> inst.num_variables > 0, uniform_subset)

        # Distributions without a finite lower support are truncated at n = 2.
        normal_subset = generate_dataset(num_problems = 100,
                                         size_distribution = Normal(500, 200),
                                         problem_types = [:knapsack],
                                         seed = 5,
                                         candidate_multiplier = 1,
                                         max_candidate_multiplier = 1)
        @test length(normal_subset) == 100
        @test minimum(inst -> inst.target_variables, normal_subset) >= 2

        # Per-type matching allocates an even quota to each selected type.
        by_type = generate_dataset(num_problems = 6,
                                   size_distribution = Uniform(30, 150),
                                   problem_types = [:transportation, :knapsack],
                                   match_size_by_type = true,
                                   seed = 3,
                                   max_candidate_multiplier = 2)
        @test count(inst -> inst.problem_type == :transportation, by_type) == 3
        @test count(inst -> inst.problem_type == :knapsack, by_type) == 3

        @test_throws ErrorException generate_dataset(
            num_problems = 1,
            problem_types = [:transportation, :knapsack],
            match_size_by_type = true,
        )

        @test_throws ErrorException generate_dataset(
            num_problems = 2,
            size_distribution = Uniform(-10, -1),
            problem_types = [:knapsack],
        )

        # Matching can be disabled for independent sampling.
        unmatched = generate_dataset(num_problems = 4, var_mean = 80, var_std = 20,
                                     var_min = 30, var_max = 150, seed = 4,
                                     problem_types = [:transportation, :knapsack],
                                     match_size_distribution = false)
        @test length(unmatched) == 4

        # Unknown problem types are rejected
        @test_throws ErrorException generate_dataset(num_problems = 1,
                                                     problem_types = [:not_a_real_type])

        # quality_filter without an optimizer is an error
        @test_throws ErrorException generate_dataset(num_problems = 1, quality_filter = true)

        # File output and manifest
        tmp = mktempdir()
        written = generate_dataset(num_problems = 4, var_mean = 80, var_std = 20,
                                   var_min = 30, var_max = 150, seed = 7,
                                   problem_types = [:transportation, :knapsack],
                                   max_candidate_multiplier = 2,
                                   output_dir = tmp)
        @test length(written) == 4
        @test all(inst -> inst.filename !== nothing, written)
        @test all(inst -> isfile(joinpath(tmp, inst.filename)), written)
        @test isfile(joinpath(tmp, "manifest.json"))
        manifest = JSON.parsefile(joinpath(tmp, "manifest.json"))
        @test manifest["config"]["size_match"]["enabled"] == true
        @test manifest["config"]["size_match"]["candidate_multiplier"] == 2
        @test length(manifest["config"]["size_match"]["groups"]) == 1

        # Manifest can be disabled
        tmp2 = mktempdir()
        generate_dataset(num_problems = 2, var_mean = 80, var_std = 20,
                         var_min = 30, var_max = 150, seed = 7,
                         problem_types = [:transportation, :knapsack],
                         max_candidate_multiplier = 2,
                         output_dir = tmp2, write_manifest = false)
        @test !isfile(joinpath(tmp2, "manifest.json"))

        # QualityCriteria carries through configured thresholds
        crit = QualityCriteria(min_constraints = 10, min_iterations = 5)
        @test crit.min_constraints == 10
        @test crit.min_iterations == 5
    end
end
