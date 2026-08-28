using Test
using JuMP
const MOI = JuMP.MOI
using Random
using Distributions
using JSON

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

    # TSP family: registry wiring, data contracts, variable-count formulas, and
    # the Hall-deficit structure behind every infeasible branch.
    @testset "TSP Variants" begin
        @test list_variants(:tsp) ==
              [:assignment_relaxation, :asymmetric, :flow, :multiple_salespersons,
               :precedence, :prize_collecting, :standard, :time_windows]
        @test problem_info(:tsp)[:default_variant] == :standard
        @test ProblemVariant("tsp") == ProblemVariant(:tsp, :standard)

        # Symmetric road metrics for the symmetric-data variants (the
        # time-window variant stores it as travel_time); genuinely asymmetric
        # travel times for the ATSP variant.
        for v in (:standard, :flow, :multiple_salespersons, :precedence,
                  :prize_collecting, :time_windows, :assignment_relaxation)
            _, p = generate_problem(ProblemVariant(:tsp, v), 100, unknown, 0)
            mat = hasproperty(p, :dist) ? p.dist : p.travel_time
            @test mat == mat'
            @test all(iszero, mat[i, i] for i in axes(mat, 1))
        end
        _, p = generate_problem("tsp/asymmetric", 100, unknown, 0)
        @test count(p.dist[i, j] != p.dist[j, i]
                    for i in 1:p.n_stops, j in 1:p.n_stops if i != j) > 0
        @test length(p.row_weight) == length(p.col_weight) == p.grid_side
        @test all(p.dist[i, j] <= p.dist[i, k] + p.dist[k, j]
                  for i in 1:p.n_stops, j in 1:p.n_stops, k in 1:p.n_stops)

        # Variable-count formulas, straight from each struct's n_stops.
        count_formulas = [
            (:standard => (p -> p.n_stops^2 - 1)),
            (:asymmetric => (p -> p.n_stops^2 - 1)),
            (:flow => (p -> 2 * p.n_stops * (p.n_stops - 1))),
            (:time_windows => (p -> p.n_stops^2)),
            (:assignment_relaxation => (p -> p.n_stops * (p.n_stops - 1))),
            (:multiple_salespersons => (p -> p.n_stops^2 - 1)),
            (:precedence => (p -> p.n_stops^2 - 1)),
            (:prize_collecting =>
                (p -> 2 * p.n_stops * (p.n_stops - 1) + p.n_stops - 1)),
        ]
        for (v, f) in count_formulas
            m, p = generate_problem(ProblemVariant(:tsp, v), 100, unknown, 0)
            @test num_variables(m) == f(p)
        end

        # The infeasible branch sizes n against the *delivered* count after the
        # Hall block deletes k*(n-k) arcs, via a per-variant delivered() lambda.
        # These assertions tie those lambdas to the models actually built.
        delivered_formulas = [
            (:standard => ((n, k) -> n^2 - 1 - k * (n - k))),
            (:asymmetric => ((n, k) -> n^2 - 1 - k * (n - k))),
            (:flow => ((n, k) -> 2 * (n^2 - n) - 2 * k * (n - k))),
            (:assignment_relaxation => ((n, k) -> n^2 - n - k * (n - k))),
        ]
        for (v, f) in delivered_formulas, s in 1:3
            m, p = generate_problem(ProblemVariant(:tsp, v), 120, infeasible, s)
            @test num_variables(m) == f(p.n_stops, length(p.blocked_set))
        end

        # Hall block: every in-arc to the blocked set S originates in the gate
        # set T, T is disjoint from S and one node short of it (the degree-row
        # deficit that makes these instances infeasible even in the LP
        # relaxation), and blocked stops keep at least one allowed in-arc.
        for v in (:standard, :asymmetric, :flow, :assignment_relaxation)
            for s in 1:3
                _, p = generate_problem(ProblemVariant(:tsp, v), 120, infeasible, s)
                @test length(p.gate_set) == length(p.blocked_set) - 1
                @test isempty(intersect(p.blocked_set, p.gate_set))
                for j in p.blocked_set, i in 1:p.n_stops
                    (i in p.gate_set || i == j) && continue
                    @test !p.arc_ok[i, j]
                end
                for j in p.blocked_set
                    @test any(p.arc_ok[i, j] for i in 1:p.n_stops)
                end
            end
        end

        # Time-window data contract: nonempty windows, and the planted tour's
        # travel time fits the route budget of a feasible instance.
        _, p = generate_problem("tsp/time_windows", 100, feasible, 0)
        @test all(p.window_start[j] <= p.window_end[j] for j in 2:p.n_stops)
        tour_time = sum(p.travel_time[p.planted_tour[i-1], p.planted_tour[i]]
                        for i in 2:length(p.planted_tour))
        @test tour_time <= p.route_budget

        # Application-variant data contracts and relaxation-proof
        # infeasibility certificates.
        _, p = generate_problem("tsp/prize_collecting", 100, feasible, 4)
        @test 0 < p.prize_quota <= sum(p.prizes)
        _, p = generate_problem("tsp/prize_collecting", 100, infeasible, 4)
        @test p.prize_quota > sum(p.prizes)

        _, p = generate_problem("tsp/multiple_salespersons", 100, feasible, 4)
        @test p.n_salespersons * p.min_stops <= p.n_stops - 1 <=
              p.n_salespersons * p.max_stops
        _, p = generate_problem("tsp/multiple_salespersons", 100, infeasible, 4)
        @test p.n_salespersons * p.max_stops < p.n_stops - 1

        _, p = generate_problem("tsp/precedence", 100, infeasible, 4)
        @test length(p.precedence_pairs) == 3
        @test p.precedence_pairs[1][2] == p.precedence_pairs[2][1]
        @test p.precedence_pairs[2][2] == p.precedence_pairs[3][1]
        @test p.precedence_pairs[3][2] == p.precedence_pairs[1][1]
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

    # Generator robustness fixes (P1): edge sizes that used to crash during build.
    @testset "Generator Robustness Fixes" begin
        # portfolio/cvar used to crash with ArgumentError (Uniform a < b) for
        # n_assets > 250 (target > ~1250) and for very small n_assets.
        @test_nowarn generate_problem("portfolio/cvar", 2000, unknown, 1)
        @test_nowarn generate_problem("portfolio/cvar", 1300, unknown, 1)
        @test_nowarn generate_problem("portfolio/cvar", 3, unknown, 1)
        # land_use used to crash with an empty-range rand when n_parcels == 2.
        @test_nowarn generate_problem("land_use/standard", 3, unknown, 1)
        @test_nowarn generate_problem("land_use/standard", 4, unknown, 1)
        # tsp variants clamp tiny targets to n = 5, where the Hall-block size
        # must also fall back to k = 2.
        @test_nowarn generate_problem("tsp/standard", 3, infeasible, 1)
        @test_nowarn generate_problem("tsp/asymmetric", 3, infeasible, 1)
        @test_nowarn generate_problem("tsp/flow", 3, infeasible, 1)
        @test_nowarn generate_problem("tsp/time_windows", 3, unknown, 1)
        @test_nowarn generate_problem("tsp/multiple_salespersons", 3, infeasible, 1)
        @test_nowarn generate_problem("tsp/precedence", 3, infeasible, 1)
        @test_nowarn generate_problem("tsp/prize_collecting", 3, infeasible, 1)
        @test_nowarn generate_problem("tsp/assignment_relaxation", 3, infeasible, 1)
        # discrete MCF variants sample extra arcs by rejection instead of
        # materializing every ordered node pair.
        for ref in ("multi_commodity_flow/binary_capacity",
                    "multi_commodity_flow/integer_flow")
            @test_nowarn generate_problem(ref, 20, unknown, 1)
            _, mcf = generate_problem(ref, 200, unknown, 1)
            @test length(mcf.arcs) == mcf.n_arcs
            @test length(unique(mcf.arcs)) == mcf.n_arcs
            @test all(a[1] != a[2] for a in mcf.arcs)
        end
        # energy now stores an emissions intensity target (the previous per-period
        # emissions row was an algebraic tautology).
        _, eprob = generate_problem("energy/standard", 120, unknown, 1)
        @test hasproperty(eprob, :emission_intensity_target)
        @test eprob.emission_intensity_target > 0

        # Feasible energy instances choose an emissions cap that is attainable at
        # peak demand and provide enough zero-emission capacity for the renewable
        # floor, even when no optimizer-backed retry guard is requested.
        for target in (120, 300, 1200), seed in 1:10
            _, prob = generate_problem("energy/standard", target, feasible, seed)
            peak_demand = maximum(prob.demands)
            clean_sources = [s for s in prob.sources if iszero(prob.emission_limits[s])]
            @test sum(prob.capacities[s] for s in clean_sources) + 1e-8 >=
                  prob.renewable_fraction * peak_demand

            remaining_demand = peak_demand
            minimum_emissions = 0.0
            for source in sort(prob.sources; by=s -> prob.emission_limits[s])
                generation = min(prob.capacities[source], remaining_demand)
                minimum_emissions += prob.emission_limits[source] * generation
                remaining_demand -= generation
                remaining_demand <= 0 && break
            end
            @test remaining_demand <= 1e-8
            @test minimum_emissions / peak_demand <=
                  prob.emission_intensity_target + 1e-12
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
        for status in (MOI.TIME_LIMIT, MOI.INFEASIBLE_OR_UNBOUNDED, MOI.ALMOST_OPTIMAL,
                       MOI.NUMERICAL_ERROR, MOI.ITERATION_LIMIT, MOI.OTHER_ERROR)
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
        @test_throws ErrorException generate_problem("transportation/standard", 80,
                                                     unknown, 5; max_feasibility_retries = 0)

        # Exhausting the retry budget is an error: never return a model known to
        # violate the requested contract or a seed that does not reproduce it.
        empty!(CONTRACT_TEST_SEEDS)
        exhaustion_error = try
            SyntheticLPs._generate_problem_verified(
                ContractViolationTestProblem, 1, infeasible, 41;
                optimizer = HiGHS.Optimizer, max_feasibility_retries = 3,
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
        m2, _ = generate_problem("transportation/standard", 80, feasible, 5;
                                 optimizer = HiGHS.Optimizer)
        @test JuMP.mode(m2) == JuMP.AUTOMATIC

        # An unbounded model has a nonempty feasible region, so it must never satisfy
        # an `infeasible` request, and it fails a `feasible` request too (the contract
        # requires OPTIMAL). End-to-end through the solve path.
        let unbounded = Model()
            @variable(unbounded, z >= 0)
            @objective(unbounded, Min, -z)
            @test SyntheticLPs._check_feasibility_contract(unbounded, HiGHS.Optimizer,
                                                           infeasible)[1] === :violated
            @test SyntheticLPs._check_feasibility_contract(unbounded, HiGHS.Optimizer,
                                                           feasible)[1] === :violated
        end
        let bounded = Model()
            @variable(bounded, 0 <= z <= 1)
            @objective(bounded, Min, z)
            @test SyntheticLPs._check_feasibility_contract(bounded, HiGHS.Optimizer,
                                                           feasible)[1] === :holds
            @test SyntheticLPs._check_feasibility_contract(bounded, HiGHS.Optimizer,
                                                           infeasible)[1] === :violated
        end

        # crop_planning/standard infeasible-request: previously ~17% came back
        # feasible (the "fallow-land" hole). With the optimizer guard every seed
        # must now solve INFEASIBLE.
        for s in 1:8
            m, _ = generate_problem("crop_planning/standard", 120, infeasible, s;
                                    optimizer = HiGHS.Optimizer)
            set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
            @test termination_status(m) in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
        end

        # energy/standard infeasible-request: previously failed at larger sizes
        # because the infeasibility logic targeted a reserve constraint that is not
        # in the model.
        for s in 1:8
            m, _ = generate_problem("energy/standard", 300, infeasible, s;
                                    optimizer = HiGHS.Optimizer)
            set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
            @test termination_status(m) in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
        end

        # Feasible energy requests also honor their label by construction when the
        # initial generation call does not use the optimizer-backed retry guard.
        for s in 1:10
            m, _ = generate_problem("energy/standard", 300, feasible, s)
            set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
            @test termination_status(m) == MOI.OPTIMAL
        end

        # supply_chain/standard reserves fallback routes within its variable budget.
        # Capacity smoothing must use that same capped coverage count; otherwise
        # tiny requested-feasible instances can remain infeasible.
        for s in 1:10
            m, _ = generate_problem("supply_chain/standard", 50, feasible, s)
            @test num_variables(m) == 50
            set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
            @test termination_status(m) == MOI.OPTIMAL
        end

        # unit_commitment/standard feasible-request: previously ~8% came back
        # infeasible (documented heuristic). The optimizer guard rejects those.
        for s in 1:10
            m, _ = generate_problem("unit_commitment/standard", 120, feasible, s;
                                    optimizer = HiGHS.Optimizer)
            set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
            @test termination_status(m) == MOI.OPTIMAL
        end

        # blending/feed_blending infeasible-request was heuristic (~8-17%);
        # the guard now guarantees the contract.
        for ref in ("blending/standard", "feed_blending/standard")
            for s in 1:6
                m, _ = generate_problem(ref, 300, infeasible, s;
                                        optimizer = HiGHS.Optimizer)
                set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
                @test termination_status(m) in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
            end
        end

        # tsp variants: feasible requests deliver a relaxed-feasible model and
        # infeasible requests a relaxed-infeasible one (Hall-deficit arc block
        # / route-budget shortfall), by construction rather than heuristic repair.
        for variant in list_variants(:tsp)
            ref = "tsp/$variant"
            for s in 1:5
                m, _ = generate_problem(ref, 120, feasible, s;
                                        optimizer = HiGHS.Optimizer)
                set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
                @test termination_status(m) == MOI.OPTIMAL
                m, _ = generate_problem(ref, 120, infeasible, s;
                                        optimizer = HiGHS.Optimizer)
                set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
                @test termination_status(m) in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
            end
        end

        # The newly integrated natural MIPs and lifted-MTZ variants also honor
        # the contract without relaxing integrality.
        for ref in ("tsp/standard", "tsp/asymmetric", "tsp/multiple_salespersons",
                    "tsp/precedence", "tsp/prize_collecting")
            for status in (feasible, infeasible), s in 1:2
                m, _ = generate_problem(ref, 80, status, s;
                                        relax_integer=false,
                                        optimizer=HiGHS.Optimizer,
                                        feasibility_timeout=30.0)
                set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
                expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
                @test termination_status(m) == expected
            end
        end

        # Reconstruct every route of one m-TSP solution and verify that the
        # modeled stop-count bounds hold route by route, not just in aggregate.
        m, p = generate_problem("tsp/multiple_salespersons", 100, feasible, 17;
                                relax_integer=false)
        set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
        @test termination_status(m) == MOI.OPTIMAL
        x = m[:x]
        route_lengths = Int[]
        for first_stop in 2:p.n_stops
            value(x[1, first_stop]) > 0.5 || continue
            current = first_stop
            route_length = 1
            while value(x[current, 1]) <= 0.5
                successors = [j for j in 2:p.n_stops
                              if j != current && value(x[current, j]) > 0.5]
                @test length(successors) == 1
                current = only(successors)
                route_length += 1
                @test route_length <= p.n_stops - 1
            end
            push!(route_lengths, route_length)
        end
        @test length(route_lengths) == p.n_salespersons
        @test all(p.min_stops <= len <= p.max_stops for len in route_lengths)
    end

    # Dataset generation honors the contract when an optimizer is supplied.
    @testset "Dataset Feasibility Verification" begin
        # feasible_only + optimizer: every emitted instance must actually be feasible.
        insts = generate_dataset(num_problems = 8, var_mean = 120, var_std = 20,
                                 var_min = 80, var_max = 200, seed = 31,
                                 problem_types = [:unit_commitment, :crop_planning],
                                 feasible_only = true, quality_filter = false,
                                 optimizer = HiGHS.Optimizer,
                                 max_candidate_multiplier = 3)
        @test length(insts) == 8
        for inst in insts
            # Rebuild with the recorded (resolved) seed and confirm feasibility.
            m, _ = generate_problem(ProblemVariant(inst.problem_type, inst.variant),
                                    inst.target_variables, feasible, inst.seed)
            set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
            @test termination_status(m) == MOI.OPTIMAL
        end
    end

    else
        @info "HiGHS not available; skipping solver-based feasibility testsets (run via Pkg.test() to include them)."
    end
end
