using Test
using JuMP
const MOI = JuMP.MOI
using Random
using Distributions
using JSON
using LinearAlgebra

using SyntheticLPs

# HiGHS is a test-only dependency ([extras]/[targets]); it resolves inside
# `Pkg.test()` but not when running this file directly with `julia --project=.`.
# Load it lazily so the direct command still runs the solver-free testsets and
# only skips the solver-based ones.
const HAS_HIGHS = try
    @eval using HiGHS
    true
catch
    false
end

# Optional focus filter, taken from the command line. Naming one or more
# categories limits the per-variant sweeps and the per-category include loop to
# them, for iterating on one generator without paying for all 127:
#
#     julia --project=@. -O1 test/runtests.jl transportation
#     julia --project=@. -O1 test/runtests.jl tsp,knapsack
#     Pkg.test(; test_args=["tsp"], julia_args=["-O1"])
#
# The framework-level testsets always run: they are cheap and guard the shared
# machinery. No arguments — the default, and what CI uses — runs everything.
const TEST_CATEGORIES = Symbol[
    Symbol(strip(t)) for a in ARGS for t in split(a, ',') if !isempty(strip(t))
]

in_scope(ref) = isempty(TEST_CATEGORIES) || ref.category in TEST_CATEGORIES

if !isempty(TEST_CATEGORIES)
    # A typo would otherwise run only the framework testsets and pass, which
    # looks exactly like a successful focused run.
    unregistered = filter(!in(Set(list_categories())), TEST_CATEGORIES)
    isempty(unregistered) || error(
        "Unknown test categories: $(join(unregistered, ", ")). " *
        "Known categories: $(join(sort(list_categories()), ", "))",
    )
    @info "Focused test run; pass no arguments to run everything." categories = TEST_CATEGORIES
end

# A deliberately always-feasible generator used to exercise retry exhaustion.
struct ContractViolationTestProblem <: ProblemGenerator
    seed::Int
end

const CONTRACT_TEST_SEEDS = Int[]

function ContractViolationTestProblem(::Int, ::FeasibilityStatus, seed::Int)
    push!(CONTRACT_TEST_SEEDS, seed)
    return ContractViolationTestProblem(seed)
end

function SyntheticLPs.build_model(::ContractViolationTestProblem)
    model = Model()
    @variable(model, x >= 0)
    @objective(model, Min, x)
    return model
end

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
            @test num_constraints(model1, count_variable_in_set_constraints=true) ==
                num_constraints(model2, count_variable_in_set_constraints=true)

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

    # Every generator must own a local RNG: generation may neither read nor
    # advance the caller's global stream, and must stay reproducible per seed.
    @testset "Global RNG Isolation" begin
        for ref in list_problems()
            in_scope(ref) || continue
            Random.seed!(5171)
            expected = rand(3)

            Random.seed!(5171)
            m1, _ = generate_problem(ref, 60, feasible, 11)
            @test rand(3) == expected

            Random.seed!(5171)
            m2, _ = generate_problem(ref, 60, feasible, 11)
            @test rand(3) == expected

            # Same seed, different global state before the call => same model.
            @test sprint(print, m1) == sprint(print, m2)
        end
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
        @test issubset(
            Set([:standard, :balanced, :capacitated, :transshipment, :emission_constrained]),
            Set(list_variants(:transportation)),
        )
        @test list_variants(:portfolio) == [:cvar, :tracking_error]

        # ProblemVariant construction, parsing, and printing.
        @test ProblemVariant("transportation") == ProblemVariant(:transportation, :standard)
        @test ProblemVariant("transportation/standard") ==
            ProblemVariant(:transportation, :standard)
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
        @test num_variables(m_cat) ==
            num_variables(m_kw) ==
            num_variables(m_ref) ==
            num_variables(m_str)

        # Unknown category / variant are rejected.
        @test_throws ErrorException generate_problem(:not_a_category, 50, unknown, 0)
        @test_throws ErrorException generate_problem(:transportation, 50, unknown, 0; variant=:nope)
    end

    # Regression guard for the CLI argument tables under `scripts/`. Formatting
    # `@add_arg_table!` blocks with JuliaFormatter's `format_docstrings` option
    # rewrites each bare option-name literal into a triple-quoted docstring,
    # which silently appends a newline and makes ArgParse register `"--seed\n"`
    # instead of `"--seed"`. The scripts are not loadable from the test
    # environment (they need ArgParse and HiGHS), so check the sources textually.
    @testset "Script CLI option names" begin
        script_dir = joinpath(dirname(@__DIR__), "scripts")
        for script in sort(readdir(script_dir; join=true))
            endswith(script, ".jl") || continue
            src = read(script, String)
            occursin("@add_arg_table!", src) || continue
            # Every option name is a plain single-line literal, never a
            # triple-quoted block that would carry a trailing newline.
            @test !occursin("\"\"\"", src)
            for m in eachmatch(r"^[ \t]*\"(-[^\"\n]*)\""m, src)
                @test !occursin(r"\s", m.captures[1])
            end
        end
    end

    # Focused per-category quality contracts live in separate files so a
    # generator's source, documentation, and regression coverage can evolve as
    # one reviewable unit.
    problem_type_test_dir = joinpath(@__DIR__, "problem_types")
    if isdir(problem_type_test_dir)
        for test_file in sort(readdir(problem_type_test_dir; join=true))
            endswith(test_file, ".jl") || continue
            category = Symbol(basename(test_file)[1:(end - 3)])
            (isempty(TEST_CATEGORIES) || category in TEST_CATEGORIES) || continue
            include(test_file)
        end
    end

    # Test individual problem generators (every registered variant)
    for ref in list_problems()
        in_scope(ref) || continue
        test_problem_generator(ref)
    end

    # Test batch dataset generation
    @testset "Dataset Generation" begin
        # Basic in-memory generation (no solver required)
        instances = generate_dataset(
            num_problems=6,
            var_mean=80,
            var_std=20,
            var_min=30,
            var_max=150,
            seed=123,
            problem_types=[:transportation, :knapsack],
            max_candidate_multiplier=2,
        )
        @test instances isa Vector{GeneratedInstance}
        @test length(instances) == 6
        @test all(inst -> inst.num_variables > 0, instances)
        @test all(inst -> inst.num_constraints >= 0, instances)
        @test [inst.index for inst in instances] == collect(1:6)
        @test all(inst -> inst.filename === nothing, instances)  # no output_dir
        # A bare category selector samples across all its registered variants.
        @test all(inst -> inst.variant in Set(list_variants(inst.problem_type)), instances)

        # Reproducibility: same seed → identical dataset
        instances2 = generate_dataset(
            num_problems=6,
            var_mean=80,
            var_std=20,
            var_min=30,
            var_max=150,
            seed=123,
            problem_types=[:transportation, :knapsack],
            max_candidate_multiplier=2,
        )
        @test [i.problem_type for i in instances] == [i.problem_type for i in instances2]
        @test [i.num_variables for i in instances] == [i.num_variables for i in instances2]
        @test [i.seed for i in instances] == [i.seed for i in instances2]

        # Restricting problem types is respected
        subset = generate_dataset(
            num_problems=5,
            var_mean=80,
            var_std=20,
            var_min=30,
            var_max=150,
            seed=1,
            problem_types=[:transportation, :knapsack],
            max_candidate_multiplier=2,
        )
        @test all(inst -> inst.problem_type in (:transportation, :knapsack), subset)

        # Direct Distributions.jl size distributions are accepted.
        uniform_subset = generate_dataset(
            num_problems=6,
            size_distribution=Uniform(30, 150),
            problem_types=[:transportation, :knapsack],
            seed=2,
            max_candidate_multiplier=2,
        )
        @test length(uniform_subset) == 6
        @test all(inst -> inst.num_variables > 0, uniform_subset)

        # Distributions without a finite lower support are truncated at n = 2.
        normal_subset = generate_dataset(
            num_problems=100,
            size_distribution=Normal(500, 200),
            problem_types=[:knapsack],
            seed=5,
            candidate_multiplier=1,
            max_candidate_multiplier=1,
        )
        @test length(normal_subset) == 100
        @test minimum(inst -> inst.target_variables, normal_subset) >= 2

        # Per-type matching allocates an even quota to each selected type.
        by_type = generate_dataset(
            num_problems=6,
            size_distribution=Uniform(30, 150),
            problem_types=[:transportation, :knapsack],
            match_size_by_type=true,
            seed=3,
            max_candidate_multiplier=2,
        )
        @test count(inst -> inst.problem_type == :transportation, by_type) == 3
        @test count(inst -> inst.problem_type == :knapsack, by_type) == 3

        @test_throws ErrorException generate_dataset(
            num_problems=1, problem_types=[:transportation, :knapsack], match_size_by_type=true
        )

        @test_throws ErrorException generate_dataset(
            num_problems=2, size_distribution=Uniform(-10, -1), problem_types=[:knapsack]
        )

        # Matching can be disabled for independent sampling.
        unmatched = generate_dataset(
            num_problems=4,
            var_mean=80,
            var_std=20,
            var_min=30,
            var_max=150,
            seed=4,
            problem_types=[:transportation, :knapsack],
            match_size_distribution=false,
        )
        @test length(unmatched) == 4

        # Unknown problem types are rejected
        @test_throws ErrorException generate_dataset(
            num_problems=1, problem_types=[:not_a_real_type]
        )

        # quality_filter without an optimizer is an error
        @test_throws ErrorException generate_dataset(num_problems=1, quality_filter=true)

        # File output and manifest
        tmp = mktempdir()
        written = generate_dataset(
            num_problems=4,
            var_mean=80,
            var_std=20,
            var_min=30,
            var_max=150,
            seed=7,
            problem_types=[:transportation, :knapsack],
            max_candidate_multiplier=2,
            output_dir=tmp,
        )
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
        generate_dataset(
            num_problems=2,
            var_mean=80,
            var_std=20,
            var_min=30,
            var_max=150,
            seed=7,
            problem_types=[:transportation, :knapsack],
            max_candidate_multiplier=2,
            output_dir=tmp2,
            write_manifest=false,
        )
        @test !isfile(joinpath(tmp2, "manifest.json"))

        # QualityCriteria carries through configured thresholds
        crit = QualityCriteria(min_constraints=10, min_iterations=5)
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

        aff_before = num_constraints(m; count_variable_in_set_constraints=false)
        result = bounds_to_constraints!(m)
        @test result === m  # mutates and returns the same model
        aff_after = num_constraints(m; count_variable_in_set_constraints=false)

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
        m_conv, _ = generate_problem(ref, 100, unknown, 0; bounds_to_constraints=true)
        @test num_variables(m_conv) == num_variables(m_plain)
        @test num_constraints(m_conv; count_variable_in_set_constraints=false) >
            num_constraints(m_plain; count_variable_in_set_constraints=false)

        # generate_dataset threads the option through: converted bounds raise the
        # recorded constraint counts, and the choice is recorded in the manifest.
        tmp = mktempdir()
        plain = generate_dataset(
            num_problems=4,
            var_mean=80,
            var_std=20,
            var_min=30,
            var_max=150,
            seed=21,
            problem_types=["knapsack/bounded"],
            max_candidate_multiplier=2,
        )
        converted = generate_dataset(
            num_problems=4,
            var_mean=80,
            var_std=20,
            var_min=30,
            var_max=150,
            seed=21,
            problem_types=["knapsack/bounded"],
            max_candidate_multiplier=2,
            bounds_to_constraints=true,
            output_dir=tmp,
        )
        @test sum(i -> i.num_constraints, converted) > sum(i -> i.num_constraints, plain)
        manifest = JSON.parsefile(joinpath(tmp, "manifest.json"))
        @test manifest["config"]["bounds_to_constraints"] == true
    end

    @testset "Dual Reformulation" begin
        primal = Model()
        @variable(primal, x >= 0)
        @variable(primal, y >= 0)
        @constraint(primal, capacity_x, 2x + y <= 8)
        @constraint(primal, capacity_y, x + 2y <= 8)
        @objective(primal, Max, 3x + 2y + 5)

        dual = dualize_model(primal)
        @test dual isa Model
        @test dual !== primal
        @test !is_dual_reformulation(primal)
        @test is_dual_reformulation(dual)
        @test objective_sense(primal) == MOI.MAX_SENSE
        @test objective_sense(dual) == MOI.MIN_SENSE
        @test num_variables(primal) == 2
        @test num_variables(dual) == 2
        @test all(startswith(name(v), "dual_var_") for v in all_variables(dual))

        # The descriptive alias has identical structural behavior.
        dual_alias = dual_reformulation(primal)
        @test num_variables(dual_alias) == num_variables(dual)
        @test objective_sense(dual_alias) == objective_sense(dual)

        # A discrete model has no LP/conic dual. Generation normally avoids this
        # through its default integrality relaxation.
        mip = Model()
        @variable(mip, z, Bin)
        @objective(mip, Max, z)
        @test_throws ArgumentError dualize_model(mip)

        # Ranged affine rows are normalized on an internal copy because
        # Dualization does not bridge Interval rows.
        ranged = Model()
        @variable(ranged, a >= 0)
        @variable(ranged, b >= 0)
        @constraint(ranged, band, 1 <= a + b <= 3)
        @objective(ranged, Min, a + 2b)
        ranged_dual = dualize_model(ranged)
        @test ranged_dual isa Model
        @test num_constraints(ranged, AffExpr, MOI.Interval{Float64}) == 1
        @test num_variables(ranged_dual) == 2

        # The option is available throughout model and dataset generation. The
        # selected dimensions and manifest describe the returned dual models.
        generated_primal, _ = generate_problem("product_mix", 60, feasible, 4)
        generated_dual, _ = generate_problem("product_mix", 60, feasible, 4; dualize=true)
        @test objective_sense(generated_dual) != objective_sense(generated_primal)
        @test num_variables(generated_dual) != num_variables(generated_primal)

        # Random generation leaves the transformation off by default, accepts a
        # probability for diversity, and keeps `dualize=true` as an explicit
        # force-all override.
        random_plain, plain_ref, _ = generate_random_problem(40; seed=11)
        random_sampled, sampled_ref, _ = generate_random_problem(
            40; seed=11, dualize_probability=1.0
        )
        random_forced, forced_ref, _ = generate_random_problem(40; seed=11, dualize=true)
        @test plain_ref == sampled_ref == forced_ref
        @test !is_dual_reformulation(random_plain)
        @test is_dual_reformulation(random_sampled)
        @test is_dual_reformulation(random_forced)
        @test objective_sense(random_plain) != objective_sense(random_sampled)
        @test num_variables(random_sampled) == num_variables(random_forced)
        @test num_constraints(random_sampled; count_variable_in_set_constraints=false) ==
            num_constraints(random_forced; count_variable_in_set_constraints=false)
        @test_throws ArgumentError generate_random_problem(40; dualize_probability=-0.1)
        @test_throws ArgumentError generate_random_problem(40; dualize_probability=1.1)

        tmp = mktempdir()
        instances = generate_dataset(
            num_problems=2,
            var_mean=40,
            var_std=5,
            var_min=30,
            var_max=50,
            seed=9,
            problem_types=["product_mix"],
            match_size_distribution=false,
            dualize=true,
            output_dir=tmp,
        )
        @test all(inst -> inst.num_variables > 0 && inst.filename !== nothing, instances)
        @test all(inst -> inst.dualized, instances)
        manifest = JSON.parsefile(joinpath(tmp, "manifest.json"))
        @test manifest["config"]["dualize"] == true
        @test manifest["config"]["dualize_probability"] == 0.0
        @test all(inst -> inst["dualized"], manifest["instances"])

        # A nontrivial probability produces a reproducible primal/dual mixture.
        mixture = generate_dataset(
            num_problems=12,
            var_mean=40,
            var_std=5,
            var_min=30,
            var_max=50,
            seed=19,
            problem_types=["product_mix"],
            match_size_distribution=false,
            dualize_probability=0.5,
        )
        repeated = generate_dataset(
            num_problems=12,
            var_mean=40,
            var_std=5,
            var_min=30,
            var_max=50,
            seed=19,
            problem_types=["product_mix"],
            match_size_distribution=false,
            dualize_probability=0.5,
        )
        @test any(inst -> inst.dualized, mixture)
        @test any(inst -> !inst.dualized, mixture)
        @test [inst.dualized for inst in mixture] == [inst.dualized for inst in repeated]
        @test [inst.seed for inst in mixture] == [inst.seed for inst in repeated]

        default_dataset = generate_dataset(
            num_problems=3,
            var_mean=40,
            var_std=5,
            var_min=30,
            var_max=50,
            seed=19,
            problem_types=["product_mix"],
            match_size_distribution=false,
        )
        @test all(inst -> !inst.dualized, default_dataset)
        @test_throws ArgumentError generate_dataset(num_problems=0, dualize_probability=1.1)

        if HAS_HIGHS
            set_optimizer(primal, HiGHS.Optimizer)
            set_optimizer(dual, HiGHS.Optimizer)
            set_silent(primal)
            set_silent(dual)
            optimize!(primal)
            optimize!(dual)
            @test termination_status(primal) == MOI.OPTIMAL
            @test termination_status(dual) == MOI.OPTIMAL
            @test objective_value(dual) ≈ objective_value(primal) atol = 1e-7
        end
    end

    # Termination-status classification is pure, so the whole table is testable
    # without a solver. The distinction it encodes — disproved vs. uncertifiable —
    # is what keeps a slow solve from being misreported as a contract violation.
    @testset "Termination Status Classification" begin
        classify = SyntheticLPs._classify_termination

        # Proofs.
        @test classify(MOI.OPTIMAL, feasible) === :holds
        @test classify(MOI.INFEASIBLE, infeasible) === :holds

        # Disproofs: each exhibits a certificate contradicting the request.
        @test classify(MOI.INFEASIBLE, feasible) === :violated
        @test classify(MOI.OPTIMAL, infeasible) === :violated
        # Unbounded (MOI: DUAL_INFEASIBLE) implies a nonempty feasible region, so it
        # disproves `infeasible`; it also fails `feasible`, which requires an optimum.
        @test classify(MOI.DUAL_INFEASIBLE, infeasible) === :violated
        @test classify(MOI.DUAL_INFEASIBLE, feasible) === :violated

        # Uncertifiable: must never be reported as a violation or consume a retry.
        for status in (
            MOI.TIME_LIMIT,
            MOI.INFEASIBLE_OR_UNBOUNDED,
            MOI.ALMOST_OPTIMAL,
            MOI.NUMERICAL_ERROR,
            MOI.ITERATION_LIMIT,
            MOI.OTHER_ERROR,
        )
            @test classify(status, feasible) === :inconclusive
            @test classify(status, infeasible) === :inconclusive
        end

        # `unknown` requests are never verified, so every status passes.
        for status in (MOI.OPTIMAL, MOI.INFEASIBLE, MOI.TIME_LIMIT)
            @test classify(status, unknown) === :holds
        end
    end

    # Solver-based testsets (require HiGHS, a test-only dep). Skipped when HiGHS is
    # not resolvable, e.g. running this file directly with `julia --project=.`
    # rather than via `Pkg.test()`.
    if HAS_HIGHS

        # Project-level feasibility-contract verification via the `optimizer` kwarg.
        @testset "Feasibility Contract Verification" begin
            # Without an optimizer, behavior is unchanged (deterministic, no solving).
            m1, _ = generate_problem("transportation/standard", 80, unknown, 5)
            @test num_variables(m1) > 0

            # max_feasibility_retries must be >= 1.
            @test_throws ErrorException generate_problem(
                "transportation/standard", 80, unknown, 5; max_feasibility_retries=0
            )

            # Exhausting the retry budget is an error: never return a model known to
            # violate the requested contract or a seed that does not reproduce it.
            empty!(CONTRACT_TEST_SEEDS)
            exhaustion_error = try
                SyntheticLPs._generate_problem_verified(
                    ContractViolationTestProblem,
                    1,
                    infeasible,
                    41;
                    optimizer=HiGHS.Optimizer,
                    max_feasibility_retries=3,
                )
                nothing
            catch err
                err
            end
            @test exhaustion_error isa ErrorException
            @test CONTRACT_TEST_SEEDS == [41, 42, 43]
            @test occursin("after 3 attempts", sprint(showerror, exhaustion_error))
            @test occursin("seeds 41 through 43", sprint(showerror, exhaustion_error))

            # The returned model is left pristine (no optimizer attached, not solved).
            m2, _ = generate_problem(
                "transportation/standard", 80, feasible, 5; optimizer=HiGHS.Optimizer
            )
            @test JuMP.mode(m2) == JuMP.AUTOMATIC

            # An unbounded model has a nonempty feasible region, so it must never satisfy
            # an `infeasible` request, and it fails a `feasible` request too (the contract
            # requires OPTIMAL). End-to-end through the solve path.
            let unbounded = Model()
                @variable(unbounded, z >= 0)
                @objective(unbounded, Min, -z)
                @test SyntheticLPs._check_feasibility_contract(
                    unbounded, HiGHS.Optimizer, infeasible
                )[1] === :violated
                @test SyntheticLPs._check_feasibility_contract(
                    unbounded, HiGHS.Optimizer, feasible
                )[1] === :violated
            end
            let bounded = Model()
                @variable(bounded, 0 <= z <= 1)
                @objective(bounded, Min, z)
                @test SyntheticLPs._check_feasibility_contract(
                    bounded, HiGHS.Optimizer, feasible
                )[1] === :holds
                @test SyntheticLPs._check_feasibility_contract(
                    bounded, HiGHS.Optimizer, infeasible
                )[1] === :violated
            end
        end

        # Dataset generation honors the contract when an optimizer is supplied.
        @testset "Dataset Feasibility Verification" begin
            # feasible_only + optimizer: every emitted instance must actually be feasible.
            insts = generate_dataset(
                num_problems=8,
                var_mean=120,
                var_std=20,
                var_min=80,
                var_max=200,
                seed=31,
                problem_types=[:unit_commitment, :crop_planning],
                feasible_only=true,
                quality_filter=false,
                optimizer=HiGHS.Optimizer,
                max_candidate_multiplier=3,
            )
            @test length(insts) == 8
            for inst in insts
                # Rebuild with the recorded (resolved) seed and confirm feasibility.
                m, _ = generate_problem(
                    ProblemVariant(inst.problem_type, inst.variant),
                    inst.target_variables,
                    feasible,
                    inst.seed,
                )
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == MOI.OPTIMAL
            end
        end

    else
        @info "HiGHS not available; skipping solver-based feasibility testsets (run via Pkg.test() to include them)."
    end
end
