using Test
using JuMP
using Random
using Distributions
using JSON

using SyntheticLPs

"""
    test_problem_generator(ref)

Test the problem generator for the given problem reference (a `ProblemVariant`,
or anything else accepted by `generate_problem`).
"""
function test_problem_generator(ref)
    @testset "$(ref) Problem Generator" begin
        # Test with different target variable counts
        for target_vars in [50, 100, 500]
            @test_nowarn begin
                model, problem = generate_problem(ref, target_vars, unknown, 0)
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
                model, problem = generate_problem(ref, 100, feas_status, 0)
                @test model isa JuMP.Model
                @test problem isa ProblemGenerator
                @test num_variables(model) > 0
            end
        end

        # Test with a fixed seed for reproducibility
        seed = 12345
        @test_nowarn begin
            # Generate the same problem twice with the same seed
            model1, problem1 = generate_problem(ref, 150, unknown, seed)
            model2, problem2 = generate_problem(ref, 150, unknown, seed)

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
            model, ref, problem = generate_random_problem(100)
            @test model isa JuMP.Model
            @test ref isa ProblemVariant
            @test problem isa ProblemGenerator
            @test num_variables(model) > 0

            # Test with feasibility status
            model2, ref2, problem2 = generate_random_problem(100; feasibility_status=feasible)
            @test model2 isa JuMP.Model
            @test ref2 isa ProblemVariant
            @test problem2 isa ProblemGenerator
            @test num_variables(model2) > 0
        end

        # Test FeasibilityStatus enum
        @test feasible isa FeasibilityStatus
        @test infeasible isa FeasibilityStatus
        @test unknown isa FeasibilityStatus
    end

    # Test the category/variant interface
    @testset "Variant Interface" begin
        cats = list_categories()
        @test cats isa Vector{Symbol}
        @test Set(cats) == Set(list_problem_types())

        problems = list_problems()
        @test problems isa Vector{ProblemVariant}
        @test !isempty(problems)
        # Every category contributes at least one variant.
        @test Set(p.category for p in problems) == Set(cats)

        # Listing variants of a category (returned sorted by variant name).
        @test issubset(Set([:standard, :balanced, :capacitated, :transshipment,
                            :emission_constrained]), Set(list_variants(:transportation)))
        @test list_variants(:portfolio) == [:cvar, :tracking_error]

        # ProblemVariant construction, parsing, and printing.
        @test ProblemVariant("transportation") == ProblemVariant(:transportation, :standard)
        @test ProblemVariant("transportation/standard") == ProblemVariant(:transportation, :standard)
        @test string(ProblemVariant("transportation/standard")) == "transportation/standard"
        @test_throws ErrorException ProblemVariant("a/b/c")

        # Variant-level info.
        vinfo = problem_info(:transportation, :standard)
        @test vinfo isa Dict
        @test vinfo[:description] isa String

        # Generating via every selector form yields the same model size.
        m_cat, _ = generate_problem(:transportation, 100, unknown, 0)
        m_kw, _ = generate_problem(:transportation, 100, unknown, 0; variant=:standard)
        m_ref, _ = generate_problem(ProblemVariant("transportation/standard"), 100, unknown, 0)
        m_str, _ = generate_problem("transportation/standard", 100, unknown, 0)
        @test num_variables(m_cat) == num_variables(m_kw) == num_variables(m_ref) == num_variables(m_str)

        # Unknown category / variant are rejected.
        @test_throws ErrorException generate_problem(:not_a_category, 50, unknown, 0)
        @test_throws ErrorException generate_problem(:transportation, 50, unknown, 0; variant=:nope)
    end

    # Test individual problem generators (every registered variant)
    for ref in list_problems()
        test_problem_generator(ref)
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
        # A bare category selector samples across all its registered variants.
        @test all(inst -> inst.variant in Set(list_variants(inst.problem_type)), instances)

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
        @test all(inst -> occursin("_$(inst.variant)_", inst.filename), written)  # variant in filename
        @test isfile(joinpath(tmp, "manifest.json"))
        manifest = JSON.parsefile(joinpath(tmp, "manifest.json"))
        @test manifest["config"]["size_match"]["enabled"] == true
        @test manifest["config"]["size_match"]["candidate_multiplier"] == 2
        @test length(manifest["config"]["size_match"]["groups"]) == 1
        @test all(inst -> haskey(inst, "variant"), manifest["instances"])

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

    # Test the bounds-to-constraints reformulation
    @testset "Bounds to Constraints" begin
        # Direct transform on a hand-built model exercising every bound kind.
        m = Model()
        @variable(m, x >= 0)        # plain nonnegativity — preserved
        @variable(m, 2 <= y <= 5)   # nonzero lower + upper — both become rows
        @variable(m, z == 3)        # fixed — becomes an equality row
        @variable(m, w <= 7)        # upper only — becomes a row
        @objective(m, Max, x + y + z + w)
        @constraint(m, x + y + z + w <= 100)

        aff_before = num_constraints(m; count_variable_in_set_constraints = false)
        result = bounds_to_constraints!(m)
        @test result === m  # mutates and returns the same model
        aff_after = num_constraints(m; count_variable_in_set_constraints = false)

        # +4 rows: lower(y), upper(y), fixed(z), upper(w). x ≥ 0 is left alone.
        @test aff_after == aff_before + 4

        # Nonnegativity is preserved; all other bounds are stripped.
        @test has_lower_bound(x)
        @test !has_lower_bound(y)
        @test !has_upper_bound(y)
        @test !is_fixed(z)
        @test !has_upper_bound(w)

        # The variable count is unchanged by the reformulation.
        @test num_variables(m) == 4

        # Via generate_problem: every item in knapsack/bounded carries an upper
        # bound (0 ≤ x ≤ uᵢ), so converting adds affine rows without changing the
        # variable count. (Integrality is relaxed by default before conversion.)
        ref = ProblemVariant("knapsack/bounded")
        m_plain, _ = generate_problem(ref, 100, unknown, 0)
        m_conv, _  = generate_problem(ref, 100, unknown, 0; bounds_to_constraints = true)
        @test num_variables(m_conv) == num_variables(m_plain)
        @test num_constraints(m_conv; count_variable_in_set_constraints = false) >
              num_constraints(m_plain; count_variable_in_set_constraints = false)

        # generate_dataset threads the option through: converted bounds raise the
        # recorded constraint counts, and the choice is recorded in the manifest.
        tmp = mktempdir()
        plain = generate_dataset(num_problems = 4, var_mean = 80, var_std = 20,
                                 var_min = 30, var_max = 150, seed = 21,
                                 problem_types = ["knapsack/bounded"],
                                 max_candidate_multiplier = 2)
        converted = generate_dataset(num_problems = 4, var_mean = 80, var_std = 20,
                                     var_min = 30, var_max = 150, seed = 21,
                                     problem_types = ["knapsack/bounded"],
                                     max_candidate_multiplier = 2,
                                     bounds_to_constraints = true,
                                     output_dir = tmp)
        @test sum(i -> i.num_constraints, converted) > sum(i -> i.num_constraints, plain)
        manifest = JSON.parsefile(joinpath(tmp, "manifest.json"))
        @test manifest["config"]["bounds_to_constraints"] == true
    end
end
