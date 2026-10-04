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

# Framework-test generators. They are registered under `:__framework_test` only
# inside `with_framework_variants` and removed afterwards, so the per-variant
# sweeps and the global listings never see them.
const FW = :__framework_test

# Undersizes by 20%, so dataset size calibration has real work to do.
struct FrameworkLP <: ProblemGenerator
    n::Int
end
FrameworkLP(target::Int, ::FeasibilityStatus, ::Int) = FrameworkLP(max(1, round(Int, 0.8 * target)))
function SyntheticLPs.build_model(p::FrameworkLP)
    model = Model()
    @variable(model, 0 <= x[1:(p.n)] <= 1)
    @constraint(model, sum(x) >= 1)
    @objective(model, Min, sum(x))
    return model
end

struct FrameworkMIP <: ProblemGenerator
    n::Int
end
FrameworkMIP(target::Int, ::FeasibilityStatus, ::Int) = FrameworkMIP(target)
function SyntheticLPs.build_model(p::FrameworkMIP)
    model = Model()
    @variable(model, x[1:(p.n)], Bin)
    @constraint(model, sum(x) >= 1)
    @objective(model, Min, sum(x))
    return model
end

# Always throws: exercises `on_failure`.
struct FrameworkFlaky <: ProblemGenerator end
FrameworkFlaky(::Int, ::FeasibilityStatus, ::Int) = error("planted generator failure")
SyntheticLPs.build_model(::FrameworkFlaky) = Model()

# Throws on even seeds: exercises per-index retries with fresh seeds.
struct FrameworkHalfFlaky <: ProblemGenerator
    n::Int
end
function FrameworkHalfFlaky(target::Int, ::FeasibilityStatus, seed::Int)
    iseven(seed) && error("planted even-seed failure")
    return FrameworkHalfFlaky(target)
end
SyntheticLPs.build_model(p::FrameworkHalfFlaky) = SyntheticLPs.build_model(FrameworkMIP(p.n))

function with_framework_variants(f)
    register_variant(
        FW, :lp, FrameworkLP, "Framework test LP"; tags=[:network, :dense], max_target_variables=500
    )
    register_variant(FW, :mip, FrameworkMIP, "Framework test MIP"; tags=[:network])
    register_variant(FW, :flaky, FrameworkFlaky, "Always fails"; tags=[:big_m], model_class=:lp)
    register_variant(FW, :half_flaky, FrameworkHalfFlaky, "Fails on even seeds"; model_class=:mip)
    try
        f()
    finally
        delete!(SyntheticLPs.LP_REGISTRY, FW)
        filter!(p -> first(p).category != FW, SyntheticLPs._MODEL_CLASS_CACHE)
    end
end

"""
    test_problem_generator(ref)

Test the problem generator for the given problem reference (a `ProblemVariant`,
or anything else accepted by `generate_problem`).
"""
function test_problem_generator(ref)
    @testset "$(ref) Problem Generator" begin
        # Test with different target variable counts
        for target_vars in [50, 100, 500], seed in (0, 1)
            @test_nowarn begin
                model, problem = generate_problem(ref, target_vars, unknown, seed)
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
            Set([:standard, :transshipment, :emission_constrained, :fixed_charge]),
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

    # Practitioner-style model transforms (framework-level: always runs).
    include("transforms.jl")

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
        )
        @test instances isa GeneratedDataset
        @test instances isa AbstractVector{GeneratedInstance}
        @test isempty(instances.failures)
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
        )
        @test all(inst -> inst.problem_type in (:transportation, :knapsack), subset)

        # Direct Distributions.jl size distributions are accepted.
        uniform_subset = generate_dataset(
            num_problems=6,
            size_distribution=Uniform(30, 150),
            problem_types=[:transportation, :knapsack],
            seed=2,
        )
        @test length(uniform_subset) == 6
        @test all(inst -> inst.num_variables > 0, uniform_subset)

        # Distributions without a finite lower support are truncated at n = 2.
        normal_subset = generate_dataset(
            num_problems=100, size_distribution=Normal(500, 200), problem_types=[:knapsack], seed=5
        )
        @test length(normal_subset) == 100
        @test minimum(inst -> inst.target_variables, normal_subset) >= 2

        # Category weighting (the default) splits the dataset evenly across the
        # selected categories, whatever their variant counts; per-category matching
        # stratifies sizes within each of them.
        by_category = generate_dataset(
            num_problems=6,
            size_distribution=Uniform(30, 150),
            problem_types=[:transportation, :knapsack],
            match_size_by_category=true,
            seed=3,
        )
        @test count(inst -> inst.problem_type == :transportation, by_category) == 3
        @test count(inst -> inst.problem_type == :knapsack, by_category) == 3
        @test_throws ErrorException generate_dataset(
            num_problems=1, match_size_by_category=true, match_size_distribution=false
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
            output_dir=tmp,
        )
        @test length(written) == 4
        @test all(inst -> inst.filename !== nothing, written)
        @test all(inst -> isfile(joinpath(tmp, inst.filename)), written)
        @test all(inst -> occursin("_$(inst.variant)_", inst.filename), written)  # variant in filename
        @test isfile(joinpath(tmp, "manifest.json"))
        manifest = JSON.parsefile(joinpath(tmp, "manifest.json"))
        @test manifest["size_match"]["enabled"] == true
        @test manifest["config"]["seed"] == 7
        @test manifest["num_instances"] == 4
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
            output_dir=tmp2,
            write_manifest=false,
        )
        @test !isfile(joinpath(tmp2, "manifest.json"))

        # QualityCriteria carries through configured thresholds
        crit = QualityCriteria(min_constraints=10, min_iterations=5)
        @test crit.min_constraints == 10
        @test crit.min_iterations == 5
    end

    # Registry metadata: tags, documented size ranges, the derived model class, and
    # the filters/weights built on them.
    @testset "Registry Metadata" begin
        @test !isempty(list_tags())
        @test all(p -> p isa Pair{Symbol, String}, list_tags())

        with_framework_variants() do
            lp = ProblemVariant(FW, :lp)
            mip = ProblemVariant(FW, :mip)
            flaky = ProblemVariant(FW, :flaky)
            half = ProblemVariant(FW, :half_flaky)

            info = problem_info(FW, :lp)
            @test info[:tags] == [:dense, :network]
            @test info[:min_target_variables] == 1
            @test info[:max_target_variables] == 500
            @test info[:model_class] === :lp
            @test info[:default] == true
            @test info[:ref] == lp
            @test problem_info(mip)[:model_class] === :mip
            @test problem_info(mip)[:max_target_variables] === nothing
            @test haskey(SyntheticLPs._MODEL_CLASS_CACHE, mip)  # derived once, cached
            @test !haskey(SyntheticLPs._MODEL_CLASS_CACHE, flaky)  # declared, never probed
            @test model_class(flaky) === :lp
            @test problem_info(FW)[:tags] == [:big_m, :dense, :network]
            @test problem_info(FW)[:num_variants] == 4
            @test variant_tags("__framework_test/lp") == [:dense, :network]
            @test supports_target(lp, 500) && !supports_target(lp, 501)

            @test list_problems(; problem_types=FW) == [flaky, half, lp, mip]
            @test list_problems(; problem_types=FW, model_class=:mip) == [half, mip]
            @test list_problems(; problem_types=FW, tags=[:network, :dense]) == [lp]
            @test list_problems(; problem_types=FW, any_tags=[:dense, :big_m]) == [flaky, lp]
            @test list_problems(; problem_types=FW, exclude_tags=:network) == [flaky, half]
            @test list_problems(;
                problem_types=FW, exclude=["__framework_test/flaky", half, mip]
            ) == [lp]
            @test list_problems(; problem_types=FW, target_variables=600) == [flaky, half, mip]
            @test !(lp in list_problems(; problem_types=FW, target_variables=(100, 1000)))
            @test lp in list_problems(; problem_types=FW, target_variables=(100, 500))
            @test_throws ErrorException list_problems(; tags=:not_a_tag)
            @test_throws ErrorException list_problems(; model_class=:qp)

            @test_throws ErrorException register_variant(
                FW, :bad, FrameworkLP, "x"; tags=[:not_a_tag]
            )
            @test_throws ErrorException register_variant(
                FW, :bad, FrameworkLP, "x"; model_class=:qp
            )
            @test_throws ErrorException register_variant(
                FW, :bad, FrameworkLP, "x"; min_target_variables=10, max_target_variables=5
            )
            @test !haskey(SyntheticLPs.LP_REGISTRY[FW].variants, :bad)
        end
        @test !(FW in list_categories())  # cleaned up

        # The derived class agrees with what an unrelaxed build actually contains.
        for ref in ("tsp/standard", "transportation/standard")
            m, _ = generate_problem(ref, 200, unknown, 1; relax_integer=false)
            has_int = any(x -> is_integer(x) || is_binary(x), all_variables(m))
            @test model_class(ref) === (has_int ? :mip : :lp)
        end

        # Weighting schemes.
        refs = list_problems(; problem_types=[:tsp, :transportation])
        share(w, cat) = sum(wi for (r, wi) in zip(refs, w) if r.category == cat)
        w_cat = SyntheticLPs.variant_weights(refs, :category)
        @test share(w_cat, :tsp) ≈ 0.5
        @test sum(w_cat) ≈ 1
        @test SyntheticLPs.variant_weights(refs, :variant) ≈ fill(1 / length(refs), length(refs))
        w_dict = SyntheticLPs.variant_weights(
            refs, Dict(:tsp => 3.0, "transportation/standard" => 1.0)
        )
        @test share(w_dict, :tsp) ≈ 0.75
        @test w_dict[findfirst(==(ProblemVariant("transportation/standard")), refs)] ≈ 0.25
        @test_throws ErrorException SyntheticLPs.variant_weights(refs, Dict(:tsp => -1.0))
        @test_throws ErrorException SyntheticLPs.variant_weights(refs, :bogus)

        # Random generation honours the selection.
        _, ref_k, _ = generate_random_problem(60; problem_types=[:knapsack], seed=3)
        @test ref_k.category == :knapsack
    end

    # Every registered variant declares exactly one domain tag (plus any structure
    # tags), so tag filters and per-domain summaries cover the whole corpus.
    @testset "Registry Tag Coverage" begin
        @test DOMAIN_TAGS ⊆ keys(SyntheticLPs.VARIANT_TAGS)
        for ref in list_problems()
            tags = variant_tags(ref)
            n_domain = count(in(DOMAIN_TAGS), tags)
            n_domain == 1 || @info "$ref carries $n_domain domain tags" tags
            @test !isempty(tags)
            @test n_domain == 1
        end
    end

    # Dataset plans are cheap (no model is built), so the mix guarantees are tested
    # exactly here rather than statistically through generation.
    @testset "Dataset Planning" begin
        cats = list_categories()
        plan = plan_dataset(; num_problems=10 * length(cats), seed=11)
        @test plan isa Vector{PlannedInstance}
        @test [p.index for p in plan] == 1:(10 * length(cats))
        # Category weighting (the default): every category exactly 10 times.
        @test all(c -> count(p -> p.ref.category == c, plan) == 10, cats)
        @test plan == plan_dataset(; num_problems=10 * length(cats), seed=11)

        # Variant weighting: each variant floor or ceil of its expected count.
        nv = length(list_problems())
        plan_v = plan_dataset(; num_problems=500, seed=11, variant_weighting=:variant)
        counts_v = [count(p -> p.ref == r, plan_v) for r in list_problems()]
        @test all(c -> fld(500, nv) <= c <= cld(500, nv), counts_v)

        # Explicit weights.
        plan_d = plan_dataset(;
            num_problems=400,
            seed=2,
            problem_types=[:tsp, :knapsack],
            variant_weighting=Dict(:tsp => 3, :knapsack => 1),
        )
        @test count(p -> p.ref.category == :tsp, plan_d) == 300
        @test_throws ErrorException plan_dataset(;
            problem_types=:tsp, variant_weighting=Dict(:knapsack => 1)
        )

        # Exclusion.
        plan_x = plan_dataset(; num_problems=50, seed=2, exclude=[:tsp, "knapsack/bounded"])
        @test !any(
            p -> p.ref.category == :tsp || p.ref == ProblemVariant("knapsack/bounded"), plan_x
        )

        # Feasibility mixes are stratified within each variant.
        nt = length(list_variants(:transportation))
        plan_s = plan_dataset(;
            num_problems=200,
            seed=5,
            problem_types=:transportation,
            feasibility_status=Dict(feasible => 0.5, :infeasible => 0.5),
        )
        @test abs(count(p -> p.feasibility_status == feasible, plan_s) - 100) <= nt
        @test !any(p -> p.feasibility_status == unknown, plan_s)
        @test all(
            p -> p.feasibility_status == feasible,
            plan_dataset(; num_problems=5, seed=1, feasible_only=true),
        )
        @test_throws ErrorException plan_dataset(;
            feasible_only=true, feasibility_status=infeasible
        )

        # Size targets are stratified quantiles: the k-th smallest lies in stratum k.
        n = 50
        plan_q = plan_dataset(; num_problems=n, seed=4, size_distribution=Uniform(100, 1100))
        q = sort([p.size_quantile for p in plan_q])
        @test all(k -> (k - 1) / n < q[k] <= k / n, 1:n)
        @test all(p -> 100 <= p.target_variables <= 1100, plan_q)
        # ... and per category when requested.
        plan_c = plan_dataset(;
            num_problems=40,
            seed=4,
            problem_types=[:tsp, :knapsack],
            size_distribution=Uniform(100, 1100),
            match_size_by_category=true,
        )
        for c in (:tsp, :knapsack)
            qc = sort([p.size_quantile for p in plan_c if p.ref.category == c])
            @test all(k -> (k - 1) / length(qc) < qc[k] <= k / length(qc), eachindex(qc))
        end

        # Log-uniform shortcut for datasets spanning orders of magnitude.
        plan_l = plan_dataset(;
            num_problems=40, seed=3, size_distribution=:loguniform, var_min=1000, var_max=100_000
        )
        t = sort([p.target_variables for p in plan_l])
        @test 1000 <= t[1] && t[end] <= 100_000
        @test 6000 <= t[20] <= 16_000  # median near the geometric mean, 10k
        @test_throws ErrorException plan_dataset(; size_distribution=:bogus)

        # Shards are disjoint and union to exactly the unsharded plan.
        full = plan_dataset(; num_problems=37, seed=9)
        shards = [plan_dataset(; num_problems=37, seed=9, shard_index=k, num_shards=4) for k in 1:4]
        @test sum(length, shards) == 37
        @test sort(reduce(vcat, shards); by=p -> p.index) == full
        @test_throws ErrorException plan_dataset(; num_shards=2)  # needs a fixed seed
        @test_throws ErrorException plan_dataset(; seed=1, num_shards=2, shard_index=3)

        # Variants whose documented cap cannot cover the size distribution are dropped.
        with_framework_variants() do
            sel = ["__framework_test/lp", "__framework_test/mip"]
            capped = plan_dataset(; num_problems=10, seed=1, problem_types=sel, var_max=2000)
            @test all(p -> p.ref.variant == :mip, capped)
            fits = plan_dataset(; num_problems=10, seed=1, problem_types=sel, var_max=400)
            @test count(p -> p.ref.variant == :lp, fits) == 5
        end
    end

    # End-to-end dataset controls on the cheap framework generators.
    @testset "Dataset Generation Controls" begin
        with_framework_variants() do
            lp_sel = "__framework_test/lp"
            sizes = Uniform(50, 400)

            # Size calibration rescales the request until the actual size matches.
            ds = generate_dataset(;
                num_problems=8, seed=3, problem_types=lp_sel, size_distribution=sizes
            )
            @test length(ds) == 8
            @test all(i -> abs(log(i.num_variables / i.target_variables)) <= 0.05, ds)
            @test all(i -> i.requested_variables > i.target_variables && i.attempts >= 2, ds)
            @test ds.manifest["size_match"]["fraction_within_tolerance"] == 1.0
            # Without matching every index is built once, as requested.
            raw = generate_dataset(;
                num_problems=8,
                seed=3,
                problem_types=lp_sel,
                size_distribution=sizes,
                match_size_distribution=false,
            )
            @test all(i -> i.attempts == 1 && i.requested_variables == i.target_variables, raw)

            # Per-instance metadata.
            mip_ds = generate_dataset(;
                num_problems=3,
                seed=2,
                problem_types="__framework_test/mip",
                size_distribution=sizes,
            )
            @test all(i -> i.num_integer == i.num_variables, mip_ds)  # counted before relaxation
            @test all(i -> i.num_nonzeros == i.num_variables, mip_ds)  # one row: sum(x) >= 1
            @test all(i -> i.transforms == ["relax_integer"], mip_ds)
            @test all(i -> 0 <= i.build_time <= i.generation_time, mip_ds)
            @test all(i -> i.verified_status === nothing && i.solve_status === nothing, mip_ds)
            unrelaxed = generate_dataset(;
                num_problems=2,
                seed=2,
                problem_types="__framework_test/mip",
                size_distribution=sizes,
                relax_integer=false,
                bounds_to_constraints=true,
            )
            @test all(i -> i.transforms == ["bounds_to_constraints"], unrelaxed)

            # Registry filters reach dataset selection.
            mc = generate_dataset(;
                num_problems=4,
                seed=1,
                problem_types=FW,
                exclude="__framework_test/flaky",
                model_class=:mip,
                size_distribution=sizes,
            )
            @test all(i -> i.variant in (:mip, :half_flaky), mc)
            tagged = generate_dataset(;
                num_problems=4, seed=1, problem_types=FW, tags=:dense, size_distribution=sizes
            )
            @test all(i -> i.variant == :lp, tagged)

            # Retries walk fresh seeds, so a seed-dependent failure is recovered.
            half = generate_dataset(;
                num_problems=6,
                seed=8,
                problem_types="__framework_test/half_flaky",
                size_distribution=sizes,
            )
            @test length(half) == 6
            @test all(i -> isodd(i.seed), half)

            # on_failure=:error aborts; :skip yields a short batch with recorded reasons.
            sel = [lp_sel, "__framework_test/flaky"]
            @test_throws ErrorException generate_dataset(;
                num_problems=2,
                seed=1,
                problem_types="__framework_test/flaky",
                size_distribution=sizes,
                max_retries=2,
            )
            tmp = mktempdir()
            skipped = generate_dataset(;
                num_problems=6,
                seed=5,
                problem_types=sel,
                variant_weighting=:variant,
                size_distribution=sizes,
                on_failure=:skip,
                max_retries=2,
                output_dir=tmp,
            )
            @test length(skipped) == 3
            @test length(skipped.failures) == 3
            @test all(
                f -> f isa DatasetFailure && f.variant == :flaky && f.reason == "error",
                skipped.failures,
            )
            @test all(
                f -> length(f.reasons) == 2 && occursin("planted", f.reasons[1]), skipped.failures
            )
            @test sort([[i.index for i in skipped]; [f.index for f in skipped.failures]]) == 1:6
            manifest = JSON.parsefile(joinpath(tmp, "manifest.json"))
            @test manifest["num_failures"] == 3
            @test manifest["stats"]["failure_reasons"]["error"] == 6
            @test [f["index"] for f in manifest["failures"]] == [f.index for f in skipped.failures]
            @test count(endswith(".mps"), readdir(tmp)) == 3

            # Sharding: disjoint shards reproduce the unsharded dataset, share an
            # output directory without filename collisions, and merge back.
            kw = (;
                num_problems=7,
                seed=21,
                problem_types=[lp_sel, "__framework_test/mip"],
                size_distribution=sizes,
            )
            full = generate_dataset(; kw...)
            tmp2 = mktempdir()
            shards = [
                generate_dataset(; kw..., shard_index=k, num_shards=3, output_dir=tmp2) for k in 1:3
            ]
            merged_insts = sort(reduce(vcat, collect.(shards)); by=i -> i.index)
            key(i) = (
                i.index,
                i.variant,
                i.seed,
                i.requested_variables,
                i.num_variables,
                i.num_constraints,
            )
            @test key.(merged_insts) == key.(collect(full))
            @test count(endswith(".mps"), readdir(tmp2)) == 7
            @test !isfile(joinpath(tmp2, "manifest.json"))
            @test isfile(joinpath(tmp2, "manifest_shard_0002_of_0003.json"))
            merged = merge_manifests(tmp2)
            @test isfile(joinpath(tmp2, "manifest.json"))
            @test merged["num_instances"] == 7
            @test [i["index"] for i in merged["instances"]] == 1:7
            @test [i["seed"] for i in merged["instances"]] == [i.seed for i in full]
            rm(joinpath(tmp2, "manifest_shard_0003_of_0003.json"))
            @test_throws ErrorException merge_manifests(tmp2)

            # Manifest contents (also returned in memory without an output_dir).
            m = full.manifest
            @test m["format_version"] == 2
            @test m["config"]["master_seed"] == 21
            @test m["config"]["variant_weighting"] == "category"
            @test m["provenance"]["julia_version"] == string(VERSION)
            @test haskey(m["provenance"], "git_commit")
            @test m["selection"]["variants"] == [lp_sel, "__framework_test/mip"]
            @test sum(values(m["selection"]["weights"])) ≈ 1
            for field in (
                "num_nonzeros",
                "num_integer",
                "requested_variables",
                "build_time",
                "generation_time",
                "transforms",
                "verified_status",
                "solve_status",
                "attempts",
                "seed",
            )
                @test all(i -> haskey(i, field), m["instances"])
            end

            # seed=0 still records the master seed it drew, so the run is reproducible.
            r0 = generate_dataset(; num_problems=2, problem_types=lp_sel, size_distribution=sizes)
            again = generate_dataset(;
                num_problems=2,
                seed=r0.manifest["config"]["master_seed"],
                problem_types=lp_sel,
                size_distribution=sizes,
            )
            @test [i.seed for i in r0] == [i.seed for i in again]
            @test_throws ErrorException generate_dataset(; num_problems=1, on_failure=:ignore)
        end
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
        )
        converted = generate_dataset(
            num_problems=4,
            var_mean=80,
            var_std=20,
            var_min=30,
            var_max=150,
            seed=21,
            problem_types=["knapsack/bounded"],
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

            # An optimizer vector is an escalation chain: an inconclusive solve (here an
            # iteration limit of 0) falls through to the next optimizer, a conclusive one
            # stops the chain, and an all-inconclusive chain still raises.
            stalled = optimizer_with_attributes(
                HiGHS.Optimizer, "simplex_iteration_limit" => 0, "presolve" => "off"
            )
            let lp = first(generate_problem("transportation/standard", 80, feasible, 5))
                @test SyntheticLPs._check_feasibility_contract(lp, stalled, feasible)[1] ===
                    :inconclusive
                verdict, ts = SyntheticLPs._check_feasibility_contract(
                    lp, [stalled, HiGHS.Optimizer], feasible
                )
                @test verdict === :holds && ts == MOI.OPTIMAL
                @test_throws ErrorException SyntheticLPs._check_feasibility_contract(
                    lp, [], feasible
                )
            end
            m3, _ = generate_problem(
                "transportation/standard", 80, feasible, 5; optimizer=[stalled, HiGHS.Optimizer]
            )
            @test num_variables(m3) == num_variables(m2)
            @test_throws ErrorException generate_problem(
                "transportation/standard", 80, feasible, 5; optimizer=[stalled]
            )
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
            )
            @test length(insts) == 8
            for inst in insts
                # Rebuild with the recorded (resolved) seed and confirm feasibility.
                @test inst.verified_status == feasible
                @test inst.solve_status == "OPTIMAL"
                m, _ = generate_problem(
                    ProblemVariant(inst), inst.requested_variables, feasible, inst.seed
                )
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == MOI.OPTIMAL
            end

            # With the quality filter, its solve doubles as verification: a status
            # mix comes back labelled, and infeasible requests are kept only when
            # the solve proves them infeasible.
            mixed = generate_dataset(;
                num_problems=6,
                size_distribution=Uniform(80, 200),
                seed=13,
                problem_types=:transportation,
                feasibility_status=Dict(feasible => 0.5, infeasible => 0.5),
                quality_filter=true,
                quality_criteria=QualityCriteria(; min_iterations=0, min_constraints=1),
                optimizer=HiGHS.Optimizer,
                on_failure=:skip,
            )
            @test !isempty(mixed)
            @test all(i -> i.verified_status == i.feasibility_status, mixed)
            @test all(i -> i.solve_status in ("OPTIMAL", "INFEASIBLE") && i.iterations >= 0, mixed)
        end

    else
        @info "HiGHS not available; skipping solver-based feasibility testsets (run via Pkg.test() to include them)."
    end
end
