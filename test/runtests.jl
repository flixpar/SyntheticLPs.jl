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

    @testset "Supply-chain network planning" begin
        @test :network_planning in list_variants(:supply_chain)
        info = problem_info(:supply_chain, :network_planning)
        @test info[:variant] == :network_planning
        @test occursin("Multi-period", info[:description])

        # Truly tiny requests resolve to the smallest meaningful two-plant,
        # two-product profile instead of dropping either block.
        for seed in 0:2
            model, p = generate_problem(
                "supply_chain/network_planning", 1, unknown, seed
            )
            @test p.n_plants == p.n_customers == p.n_products == 2
            @test length(p.shipment_arcs) ==
                  p.n_customers * p.n_products * p.n_periods
            @test num_variables(model) ==
                  2 * p.n_plants * p.n_products * p.n_periods +
                  length(p.shipment_arcs)
        end

        # A committed multi-target/multi-seed matrix guards sizing and
        # certificate arithmetic across every profile and status.
        for target in (50, 500, 5000), seed in 0:5,
            status in (feasible, infeasible, unknown)
            model, p = generate_problem(
                "supply_chain/network_planning", target, status, seed
            )
            expected =
                2 * p.n_plants * p.n_products * p.n_periods +
                length(p.shipment_arcs)
            @test num_variables(model) == expected
            @test abs(num_variables(model) - target) <= 0.25 * target
            @test p.n_products >= 2

            if status == infeasible
                cert = p.infeasibility_certificate
                k, tau = cert.product, cert.period
                demand = sum(p.demand[:, k, 1:tau])
                supply = sum(p.initial_inventory[:, k]) + sum(
                    min(
                        p.production_capacity[plant, k, period],
                        p.plant_capacity[plant, period] /
                        p.resource_use[plant, k],
                    )
                    for plant in 1:p.n_plants, period in 1:tau
                )
                lanes = sum(
                    p.lane_capacity[a] for a in p.shipment_arcs
                    if a[3] == k && a[4] <= tau
                )
                @test cert.demand == demand
                @test cert.supply_bound == supply
                @test cert.lane_bound == lanes
                @test cert.upper_bound == min(supply, lanes)
                @test cert.margin == demand - cert.upper_bound > 0
            end
        end

        # The analytical search reaches the documented maximum exactly without
        # allocating its million coordinates; larger requests fail explicitly.
        maximum_target = SyntheticLPs.MAX_NETWORK_PLANNING_VARIABLES
        for profile in (:regional_stable, :seasonal_prebuild, :disruption)
            P, C, K, T, A =
                SyntheticLPs._choose_network_planning_dimensions(
                    maximum_target, profile
                )
            @test 2 * P * K * T + A == maximum_target
            @test C > 0
        end
        large_problem = SyntheticLPs.SupplyChainNetworkPlanningProblem(
            100_000, unknown, 0
        )
        @test 2 * large_problem.n_plants * large_problem.n_products *
              large_problem.n_periods +
              length(large_problem.shipment_arcs) == 100_000
        large_error = try
            generate_problem(
                "supply_chain/network_planning", maximum_target + 1, unknown, 0
            )
            nothing
        catch err
            err
        end
        @test large_error isa ArgumentError
        @test occursin("supports target_variables <= 1000000",
                       sprint(showerror, large_error))

        # Validate the stored constructive witness without a solver.
        for seed in 0:8
            _, p = generate_problem(
                "supply_chain/network_planning", 240, feasible, seed
            )
            witness = p.feasible_witness
            @test witness !== nothing
            @test p.infeasibility_certificate === nothing
            @test p.nominal_scenario === nothing
            for plant in 1:p.n_plants, product in 1:p.n_products,
                period in 1:p.n_periods
                outbound = sum(
                    (witness.shipment[a] for a in p.shipment_arcs
                     if a[1] == plant && a[3] == product &&
                        a[4] == period);
                    init=0.0,
                )
                previous = period == 1 ? p.initial_inventory[plant, product] :
                           witness.inventory[plant, product, period - 1]
                @test isapprox(
                    previous + witness.production[plant, product, period] -
                    outbound,
                    witness.inventory[plant, product, period];
                    atol=1e-8,
                )
                @test witness.production[plant, product, period] <=
                      p.production_capacity[plant, product, period] + 1e-8
                @test witness.inventory[plant, product, period] <=
                      p.inventory_capacity[plant, product] + 1e-8
            end
            for customer in 1:p.n_customers, product in 1:p.n_products,
                period in 1:p.n_periods
                delivered = sum(
                    witness.shipment[a] for a in p.shipment_arcs
                    if a[2] == customer && a[3] == product &&
                       a[4] == period
                )
                @test isapprox(
                    delivered, p.demand[customer, product, period]; atol=1e-8
                )
            end
            for plant in 1:p.n_plants, period in 1:p.n_periods
                used = sum(
                    p.resource_use[plant, product] *
                    witness.production[plant, product, period]
                    for product in 1:p.n_products
                )
                @test used <= p.plant_capacity[plant, period] + 1e-8
            end
            @test all(witness.shipment[a] <= p.lane_capacity[a] + 1e-8
                      for a in p.shipment_arcs)
            @test all(>=(0), witness.production)
            @test all(>=(0), witness.inventory)
            @test all(value >= -1e-12 for value in values(witness.shipment))
            @test Set(keys(witness.shipment)) == Set(p.shipment_arcs)
        end

        # Status-aware metadata has no ambiguous zero-valued witness,
        # certificate, or absence sentinels.
        for seed in 0:5
            _, pf = generate_problem(
                "supply_chain/network_planning", 200, feasible, seed
            )
            _, pi = generate_problem(
                "supply_chain/network_planning", 200, infeasible, seed
            )
            _, pu = generate_problem(
                "supply_chain/network_planning", 200, unknown, seed
            )
            @test pf.feasible_witness !== nothing
            @test pf.infeasibility_certificate === nothing
            @test pf.nominal_scenario === nothing
            @test pi.feasible_witness === nothing
            @test pi.infeasibility_certificate !== nothing
            @test pi.nominal_scenario === nothing
            @test pu.feasible_witness === nothing
            @test pu.infeasibility_certificate === nothing
            @test pu.nominal_scenario !== nothing
            @test (pf.disruption !== nothing) ==
                  (pi.disruption !== nothing) ==
                  (pu.disruption !== nothing) ==
                  (pf.profile == :disruption)
        end

        # Sparse coordinate, degree, and JuMP-axis invariants.
        profiles = Set{Symbol}()
        for seed in 0:2
            model, p = generate_problem(
                "supply_chain/network_planning", 500, feasible, seed
            )
            push!(profiles, p.profile)
            @test length(p.shipment_arcs) == length(unique(p.shipment_arcs))
            @test Set(keys(p.shipment_cost)) == Set(p.shipment_arcs)
            @test Set(keys(p.lane_capacity)) == Set(p.shipment_arcs)
            @test collect(only(axes(model[:ship]))) == p.shipment_arcs
            @test length(p.shipment_arcs) <
                  p.n_plants * p.n_customers * p.n_products * p.n_periods
            @test all(1 <= a[1] <= p.n_plants &&
                      1 <= a[2] <= p.n_customers &&
                      1 <= a[3] <= p.n_products &&
                      1 <= a[4] <= p.n_periods for a in p.shipment_arcs)
            _, density_hi = SyntheticLPs._network_density(p.profile)
            max_degree =
                min(p.n_plants, max(2, ceil(Int, density_hi * p.n_plants)))
            for customer in 1:p.n_customers, product in 1:p.n_products,
                period in 1:p.n_periods
                degree = count(
                    a -> a[2] == customer && a[3] == product &&
                         a[4] == period,
                    p.shipment_arcs,
                )
                period_max =
                    p.profile == :disruption &&
                    period == p.disruption.period ?
                    min(max_degree, p.n_plants - 1) : max_degree
                @test 1 <= degree <= period_max
            end
            @test all(maximum(p.specialization[:, k]) > 1.2
                      for k in 1:p.n_products)
        end
        @test profiles == Set([:regional_stable, :seasonal_prebuild, :disruption])

        # Profile labels correspond to materially different coefficients and
        # structure. Apply the contracts across several seeds of each profile.
        function check_regional_profile(p)
            @test p.profile == :regional_stable
            totals = [sum(p.demand[:, :, t]) for t in 1:p.n_periods]
            share = count(
                a -> p.plant_regions[a[1]] == p.customer_regions[a[2]],
                p.shipment_arcs,
            ) / length(p.shipment_arcs)
            @test maximum(totals) < 1.30 * minimum(totals)
            @test share > 0.55
        end
        function check_seasonal_profile(p)
            @test p.profile == :seasonal_prebuild
            totals = [sum(p.demand[:, :, t]) for t in 1:p.n_periods]
            @test maximum(totals) > 1.6 * minimum(totals)
            @test sum(p.production_cost[:, :, 1]) <
                  sum(p.production_cost[:, :, end])
            prepeak = max(1, argmax(totals) - 1)
            @test sum(p.feasible_witness.inventory[:, :, prepeak]) >
                  sum(p.initial_inventory)
        end
        function check_disruption_profile(p)
            @test p.profile == :disruption
            event = p.disruption
            @test event.production_factor == 0.35
            @test event.shipment_surcharge == 1.55
            @test all(!(a[1] == event.plant && a[4] == event.period)
                      for a in p.shipment_arcs)
            @test p.plant_capacity[event.plant, event.period] <
                  minimum(p.plant_capacity[event.plant,
                          setdiff(1:p.n_periods, [event.period])])
            disruption_arcs =
                [a for a in p.shipment_arcs if a[4] == event.period]
            ordinary_arcs =
                [a for a in p.shipment_arcs if a[4] != event.period]
            @test sum(p.shipment_cost[a] for a in disruption_arcs) /
                  length(disruption_arcs) >
                  1.15 * sum(p.shipment_cost[a] for a in ordinary_arcs) /
                  length(ordinary_arcs)
            @test all(any(a[2] == c && a[3] == k && a[4] == event.period
                          for a in p.shipment_arcs)
                      for c in 1:p.n_customers, k in 1:p.n_products)
        end
        for seed in 0:3:9
            _, p = generate_problem(
                "supply_chain/network_planning", 500, feasible, seed
            )
            check_regional_profile(p)
        end
        for seed in 1:3:10
            _, p = generate_problem(
                "supply_chain/network_planning", 500, feasible, seed
            )
            check_seasonal_profile(p)
        end
        for seed in 2:3:11
            _, p = generate_problem(
                "supply_chain/network_planning", 500, feasible, seed
            )
            check_disruption_profile(p)
        end

        # Exact JuMP algebra, domains, bounds, sparse shipment axes, and
        # objective coefficients. Every named constraint family is checked in
        # full, including absent coefficients.
        algebra_model, algebra = generate_problem(
            "supply_chain/network_planning", 120, feasible, 2
        )
        produce = algebra_model[:produce]
        inventory = algebra_model[:inventory]
        ship = algebra_model[:ship]
        balances = algebra_model[:inventory_balance]
        demands = algebra_model[:demand_balance]
        resources = algebra_model[:resource_capacity]
        @test length(balances) ==
              algebra.n_plants * algebra.n_products * algebra.n_periods
        @test length(demands) ==
              algebra.n_customers * algebra.n_products * algebra.n_periods
        @test length(resources) == algebra.n_plants * algebra.n_periods

        function expected_inventory_coefficient(var, plant, product, period)
            var == produce[plant, product, period] && return 1.0
            var == inventory[plant, product, period] && return -1.0
            if period > 1 && var == inventory[plant, product, period - 1]
                return 1.0
            end
            for arc in algebra.shipment_arcs
                arc[1] == plant && arc[3] == product &&
                    arc[4] == period || continue
                var == ship[arc] && return -1.0
            end
            return 0.0
        end
        for plant in 1:algebra.n_plants, product in 1:algebra.n_products,
            period in 1:algebra.n_periods
            row = balances[plant, product, period]
            @test normalized_rhs(row) ==
                  (period == 1 ? -algebra.initial_inventory[plant, product] : 0.0)
            for var in all_variables(algebra_model)
                @test normalized_coefficient(row, var) ==
                      expected_inventory_coefficient(var, plant, product, period)
            end
        end
        for customer in 1:algebra.n_customers, product in 1:algebra.n_products,
            period in 1:algebra.n_periods
            row = demands[customer, product, period]
            @test normalized_rhs(row) == algebra.demand[customer, product, period]
            for var in all_variables(algebra_model)
                expected = 0.0
                for arc in algebra.shipment_arcs
                    arc[2] == customer && arc[3] == product &&
                        arc[4] == period || continue
                    if var == ship[arc]
                        expected = 1.0
                        break
                    end
                end
                @test normalized_coefficient(row, var) == expected
            end
        end
        for plant in 1:algebra.n_plants, period in 1:algebra.n_periods
            row = resources[plant, period]
            @test normalized_rhs(row) == algebra.plant_capacity[plant, period]
            for var in all_variables(algebra_model)
                expected = 0.0
                for product in 1:algebra.n_products
                    if var == produce[plant, product, period]
                        expected = algebra.resource_use[plant, product]
                        break
                    end
                end
                @test normalized_coefficient(row, var) == expected
            end
        end

        @test objective_sense(algebra_model) == MOI.MIN_SENSE
        @test collect(only(axes(ship))) == algebra.shipment_arcs
        for p in 1:algebra.n_plants, k in 1:algebra.n_products,
            t in 1:algebra.n_periods
            x = produce[p, k, t]
            inv = inventory[p, k, t]
            @test !is_binary(x) && !is_integer(x)
            @test !is_binary(inv) && !is_integer(inv)
            @test lower_bound(x) == 0
            @test upper_bound(x) == algebra.production_capacity[p, k, t]
            @test lower_bound(inv) == 0
            @test upper_bound(inv) == algebra.inventory_capacity[p, k]
            @test coefficient(objective_function(algebra_model), x) ==
                  algebra.production_cost[p, k, t]
            @test coefficient(objective_function(algebra_model), inv) ==
                  algebra.holding_cost[p, k, t]
        end
        for arc in algebra.shipment_arcs
            x = ship[arc]
            @test !is_binary(x) && !is_integer(x)
            @test lower_bound(x) == 0
            @test upper_bound(x) == algebra.lane_capacity[arc]
            @test coefficient(objective_function(algebra_model), x) ==
                  algebra.shipment_cost[arc]
        end

        # Unknown samples use correlated network conditions and retain local
        # lane service; they do not expose a baseline plan as a feasible witness.
        unknown_supply_factors = Float64[]
        for target in (200, 5000), seed in 0:11
            _, p = generate_problem(
                "supply_chain/network_planning", target, unknown, seed
            )
            scenario = p.nominal_scenario
            push!(unknown_supply_factors, scenario.supply_factor)
            @test 0.65 <= scenario.supply_factor <= 1.20
            @test 0.92 <= scenario.lane_factor <= 1.14
            @test scenario.minimum_local_service >= 1.03 - 1e-12
            @test all(
                sum(p.lane_capacity[a] for a in p.shipment_arcs
                    if a[2] == c && a[3] == k && a[4] == t) >=
                p.demand[c, k, t] * (1.03 - 1e-12)
                for c in 1:p.n_customers, k in 1:p.n_products,
                    t in 1:p.n_periods
            )
        end
        @test minimum(unknown_supply_factors) < 0.80
        @test maximum(unknown_supply_factors) > 1.05

        # Local-RNG reproducibility compares every stored field for each profile
        # and status, plus repeated byte-identical MPS output.
        function stored_equal(a, b)
            typeof(a) == typeof(b) || return false
            if a === nothing || a isa Number || a isa Symbol ||
               a isa AbstractString || a isa Tuple ||
               a isa AbstractArray || a isa AbstractDict
                return isequal(a, b)
            end
            return all(
                stored_equal(getfield(a, name), getfield(b, name))
                for name in fieldnames(typeof(a))
            )
        end
        Random.seed!(9182)
        expected_global_draw = rand()
        Random.seed!(9182)
        generate_problem("supply_chain/network_planning", 120, feasible, 7)
        @test rand() == expected_global_draw

        mktempdir() do dir
            for seed in 0:2, status in (feasible, infeasible, unknown)
                m1, p1 = generate_problem(
                    "supply_chain/network_planning", 360, status, seed
                )
                m2, p2 = generate_problem(
                    "supply_chain/network_planning", 360, status, seed
                )
                @test all(
                    stored_equal(getfield(p1, name), getfield(p2, name))
                    for name in fieldnames(typeof(p1))
                )
                repeated = SyntheticLPs.build_model(p1)
                @test num_variables(m1) == num_variables(repeated)
                @test num_constraints(
                    m1, count_variable_in_set_constraints=true
                ) == num_constraints(
                    repeated, count_variable_in_set_constraints=true
                )
                first_mps = joinpath(dir, "first-$seed-$status.mps")
                second_mps = joinpath(dir, "second-$seed-$status.mps")
                write_to_file(m1, first_mps)
                write_to_file(m2, second_mps)
                @test filesize(first_mps) > 0
                @test read(first_mps, String) == read(second_mps, String)
            end

            # Seeds three apart select the same profile but change topology and
            # the resulting serialized model.
            for seed in 0:2
                m1, p1 = generate_problem(
                    "supply_chain/network_planning", 360, feasible, seed
                )
                m2, p2 = generate_problem(
                    "supply_chain/network_planning", 360, feasible, seed + 3
                )
                @test p1.profile == p2.profile
                @test p1.shipment_arcs != p2.shipment_arcs
                @test p1.demand != p2.demand
                first_mps = joinpath(dir, "different-$seed-a.mps")
                second_mps = joinpath(dir, "different-$seed-b.mps")
                write_to_file(m1, first_mps)
                write_to_file(m2, second_mps)
                @test read(first_mps, String) != read(second_mps, String)
            end
        end
    end

    @testset "Workforce Shift Covering" begin
        ref = ProblemVariant(:workforce_shift_scheduling, :covering)
        @test :workforce_shift_scheduling in list_categories()
        @test list_variants(:workforce_shift_scheduling) == [:covering]
        @test problem_info(:workforce_shift_scheduling)[:default_variant] == :covering
        @test problem_info(:workforce_shift_scheduling, :covering)[:type] <:
              ProblemGenerator

        # There is exactly one decision-variable block. Sizing is exact from
        # small instances through scales above the source implementation's cap.
        for target in (10, 50, 200, 1500, 5000)
            model, problem = generate_problem(ref, target, feasible, 19)
            @test num_variables(model) == target ==
                  length(problem.column_pools)
        end
        # A target of one is below the structural floor needed to retain
        # skill-period coverage and representative generated labor pools.
        expected_minimums = Dict(1 => 8, 2 => 9, 4 => 6)
        for (seed, expected) in expected_minimums
            model, problem = generate_problem(ref, 1, feasible, seed)
            @test num_variables(model) == expected ==
                  length(problem.column_pools)
            @test num_variables(model) > 1
        end

        # Every profile is also exercised above 1,000 variables. Contact-center
        # and continuous-operations instances take their four-skill branches;
        # retail intentionally has three profile-defined skills.
        large_profiles = Dict{Symbol,Any}()
        for seed in (1, 2, 4)
            model, problem = generate_problem(ref, 1500, feasible, seed)
            @test num_variables(model) == 1500
            large_profiles[problem.profile] = problem
        end
        @test Set(keys(large_profiles)) ==
              Set((:contact_center, :retail, :continuous_operations))
        @test length(large_profiles[:contact_center].skill_names) == 4
        @test length(large_profiles[:continuous_operations].skill_names) == 4
        @test length(large_profiles[:retail].skill_names) == 3

        # Validate one large four-skill instance end to end: the stored
        # witness respects pool capacities, covers every row (including skill
        # 4), and the named model row block has exactly the expected shape.
        large_model, large_problem =
            generate_problem(ref, 1500, feasible, 2)
        @test large_problem.feasibility_status == feasible
        @test length(large_problem.skill_names) == 4
        large_witness = something(large_problem.feasible_staffing)
        for pool in eachindex(large_problem.pool_names)
            usage = sum(
                large_witness[column]
                for column in eachindex(large_problem.column_pools)
                if large_problem.column_pools[column] == pool
            )
            @test usage <= large_problem.pool_capacities[pool] + 1e-8
        end
        large_coverage_rows = large_model[:skill_coverage]
        @test size(large_coverage_rows) == (large_problem.n_periods, 4)
        @test length(large_coverage_rows) == 4 * large_problem.n_periods
        @test all(is_valid(large_model, large_coverage_rows[period, 4])
                  for period in 1:large_problem.n_periods)
        for period in 1:large_problem.n_periods, skill in 1:4
            row = constraint_object(large_coverage_rows[period, skill])
            @test row.set.lower == large_problem.demand[period, skill]
            supplied = sum(
                large_problem.pool_productivity[
                    large_problem.column_pools[column], skill,
                ] * large_witness[column]
                for column in eachindex(large_problem.column_pools)
                if large_problem.column_skills[column] == skill &&
                   large_problem.pattern_coverage[
                       period, large_problem.column_patterns[column],
                   ]
            )
            @test supplied + 1e-8 >= large_problem.demand[period, skill]
        end

        # Exact field and model reproducibility, including repeated builds and
        # deterministic MPS export.
        model1, problem1 = generate_problem(ref, 320, unknown, 12345)
        model2, problem2 = generate_problem(ref, 320, unknown, 12345)
        @test all(
            isequal(getfield(problem1, field), getfield(problem2, field))
            for field in fieldnames(typeof(problem1))
        )
        rebuilt1 = SyntheticLPs.build_model(problem1)
        rebuilt2 = SyntheticLPs.build_model(problem1)
        @test num_variables(model1) == num_variables(model2) ==
              num_variables(rebuilt1) == num_variables(rebuilt2)
        @test num_constraints(model1; count_variable_in_set_constraints=true) ==
              num_constraints(model2; count_variable_in_set_constraints=true) ==
              num_constraints(rebuilt1; count_variable_in_set_constraints=true) ==
              num_constraints(rebuilt2; count_variable_in_set_constraints=true)

        # Exact model contract: one continuous nonnegative staffing block,
        # minimization, and objective coefficients sourced without alteration
        # from the stored data.
        assigned_workers = model1[:assigned_workers]
        variables = all_variables(model1)
        @test length(assigned_workers) == length(problem1.staffing_costs)
        @test Set(variables) == Set(assigned_workers)
        @test all(!is_binary(variable) && !is_integer(variable)
                  for variable in variables)
        @test all(has_lower_bound(variable) && lower_bound(variable) == 0.0
                  for variable in variables)
        @test all(!has_upper_bound(variable) for variable in variables)
        @test objective_sense(model1) == MOI.MIN_SENSE
        objective = objective_function(model1)
        @test objective isa JuMP.AffExpr
        @test objective.constant == 0.0
        @test all(
            coefficient(objective, assigned_workers[column]) ==
            problem1.staffing_costs[column]
            for column in eachindex(problem1.staffing_costs)
        )

        export_dir = mktempdir()
        path1 = joinpath(export_dir, "workforce_1.mps")
        path2 = joinpath(export_dir, "workforce_2.mps")
        write_to_file(rebuilt1, path1)
        write_to_file(rebuilt2, path2)
        @test filesize(path1) > 0
        @test read(path1, String) == read(path2, String)

        # Fixed seeds exercise all structural profiles and their distinct
        # horizons, shift rules, demand curves, and availability regimes.
        profiles = Dict{Symbol,Any}()
        for seed in (1, 2, 4)
            _, problem = generate_problem(ref, 240, unknown, seed)
            profiles[problem.profile] = problem
        end
        @test Set(keys(profiles)) ==
              Set((:contact_center, :retail, :continuous_operations))
        @test (profiles[:contact_center].period_minutes,
               profiles[:contact_center].n_periods) == (30, 24)
        @test (profiles[:retail].period_minutes,
               profiles[:retail].n_periods) == (60, 14)
        @test (profiles[:continuous_operations].period_minutes,
               profiles[:continuous_operations].n_periods) == (60, 24)
        @test !any(profiles[:contact_center].pattern_wraps)
        @test !any(profiles[:retail].pattern_wraps)
        @test any(profiles[:continuous_operations].pattern_wraps)
        @test all(profile -> any(profile.pattern_break_periods .> 0),
                  values(profiles))
        @test all(profile -> length(unique(profile.pattern_span_periods)) >= 3,
                  values(profiles))

        for problem in values(profiles)
            n_pools = length(problem.pool_names)
            n_skills = length(problem.skill_names)
            n_patterns = size(problem.pattern_coverage, 2)
            @test n_pools >= 4
            @test n_skills >= 2
            @test n_patterns > 0
            @test all(sum(problem.pattern_coverage; dims=1) .> 0)
            @test length(unique(Tuple(problem.pool_qualifications[q, :])
                                for q in 1:n_pools)) > 1
            @test length(unique(Tuple(problem.pool_availability[q, :])
                                for q in 1:n_pools)) > 1
            @test all(problem.pool_productivity[problem.pool_qualifications] .> 0)
            @test all(problem.pool_productivity[.!problem.pool_qualifications] .== 0)
            @test all(problem.hourly_wages .> 0)
            @test all(problem.pool_capacities .> 0)
            @test all(problem.staffing_costs .> 0)
            @test all(problem.demand .> 0)
            @test any(maximum(problem.demand[:, skill]) >
                      1.10 * minimum(problem.demand[:, skill])
                      for skill in 1:n_skills)

            # Pattern metadata reconstructs each contiguous (possibly
            # wraparound) start/span window exactly. A stored break is inside
            # that window and is the sole excluded period.
            supports = Tuple[]
            for pattern in 1:n_patterns
                start = problem.pattern_starts[pattern]
                span = problem.pattern_span_periods[pattern]
                break_period = problem.pattern_break_periods[pattern]
                window = if problem.profile == :continuous_operations
                    [mod1(start + offset, problem.n_periods)
                     for offset in 0:(span - 1)]
                else
                    [start + offset for offset in 0:(span - 1)]
                end
                @test length(unique(window)) == span
                @test all(period -> 1 <= period <= problem.n_periods, window)
                expected_support = if break_period == 0
                    copy(window)
                else
                    @test break_period in window
                    @test !problem.pattern_coverage[break_period, pattern]
                    [period for period in window if period != break_period]
                end
                actual_support = findall(problem.pattern_coverage[:, pattern])
                @test sort(actual_support) == sort(expected_support)
                @test length(actual_support) ==
                      span - (break_period == 0 ? 0 : 1)
                @test problem.pattern_wraps[pattern] ==
                      (start + span - 1 > problem.n_periods)
                push!(supports, Tuple(actual_support))
            end
            # `_workforce_patterns` drops duplicate supports even when
            # different start/span/break samples would produce the same set.
            @test length(unique(supports)) == n_patterns

            # Every selected column obeys qualification, availability, and
            # pattern eligibility. Every skill-period has nonempty row support.
            for column in eachindex(problem.column_pools)
                pool = problem.column_pools[column]
                pattern = problem.column_patterns[column]
                skill = problem.column_skills[column]
                @test problem.pool_qualifications[pool, skill]
                @test problem.pattern_eligibility[pool, pattern]
                @test all(
                    !problem.pattern_coverage[period, pattern] ||
                    problem.pool_availability[pool, period]
                    for period in 1:problem.n_periods
                )
            end
            @test all(
                any(problem.column_skills[column] == skill &&
                    problem.pattern_coverage[period,
                                             problem.column_patterns[column]]
                    for column in eachindex(problem.column_pools))
                for period in 1:problem.n_periods, skill in 1:n_skills
            )

            # Coverage + pool-row signatures are unique; costs are not being
            # used to disguise duplicate staffing columns.
            signatures = [
                (
                    problem.column_pools[column],
                    problem.column_skills[column],
                    Tuple(findall(problem.pattern_coverage[:,
                                      problem.column_patterns[column]])),
                )
                for column in eachindex(problem.column_pools)
            ]
            @test length(unique(signatures)) == length(signatures)
        end

        # Different seeds alter profile and numerical/structural data.
        _, seed1 = generate_problem(ref, 240, unknown, 1)
        _, seed2 = generate_problem(ref, 240, unknown, 2)
        @test seed1.profile != seed2.profile
        @test seed1.demand != seed2.demand
        @test seed1.skill_names != seed2.skill_names
        # Diversity also holds within each profile, rather than relying on
        # profile selection alone.
        for (first_seed, second_seed) in ((1, 3), (2, 5), (4, 7))
            _, first_problem =
                generate_problem(ref, 240, unknown, first_seed)
            _, second_problem =
                generate_problem(ref, 240, unknown, second_seed)
            @test first_problem.profile == second_problem.profile
            @test first_problem.demand != second_problem.demand
            @test first_problem.pattern_coverage !=
                  second_problem.pattern_coverage
            @test first_problem.pool_qualifications !=
                  second_problem.pool_qualifications
        end

        # The planted staffing vector proves feasible requests directly.
        for seed in 1:6
            _, problem = generate_problem(ref, 260, feasible, seed)
            @test problem.feasibility_status == feasible
            @test problem.feasible_staffing !== nothing
            @test problem.infeasible_skill === nothing
            @test problem.infeasibility_capacity_bound === nothing
            witness = something(problem.feasible_staffing)
            for pool in eachindex(problem.pool_names)
                usage = sum(
                    witness[column]
                    for column in eachindex(problem.column_pools)
                    if problem.column_pools[column] == pool
                )
                @test usage <= problem.pool_capacities[pool] + 1e-8
            end
            for period in 1:problem.n_periods,
                skill in eachindex(problem.skill_names)
                supplied = sum(
                    problem.pool_productivity[
                        problem.column_pools[column], skill,
                    ] * witness[column]
                    for column in eachindex(problem.column_pools)
                    if problem.column_skills[column] == skill &&
                       problem.pattern_coverage[
                           period, problem.column_patterns[column],
                       ]
                )
                @test supplied + 1e-8 >= problem.demand[period, skill]
            end
        end

        # At least one skill violates a valid aggregate capacity upper bound in
        # every requested-infeasible instance.
        for seed in 1:6
            _, problem = generate_problem(ref, 260, infeasible, seed)
            @test problem.feasibility_status == infeasible
            @test problem.feasible_staffing === nothing
            @test problem.infeasible_skill !== nothing
            @test problem.infeasibility_capacity_bound !== nothing
            certified = false
            for skill in eachindex(problem.skill_names)
                upper = 0.0
                for pool in eachindex(problem.pool_names)
                    paid = [
                        count(problem.pattern_coverage[:,
                              problem.column_patterns[column]])
                        for column in eachindex(problem.column_pools)
                        if problem.column_pools[column] == pool &&
                           problem.column_skills[column] == skill
                    ]
                    max_paid = isempty(paid) ? 0 : maximum(paid)
                    upper += problem.pool_capacities[pool] *
                             problem.pool_productivity[pool, skill] * max_paid
                end
                certified |= sum(problem.demand[:, skill]) > upper + 1e-6
            end
            @test certified
            certificate_skill = something(problem.infeasible_skill)
            expected_bound = 0.0
            for pool in eachindex(problem.pool_names)
                paid = [
                    count(problem.pattern_coverage[:,
                          problem.column_patterns[column]])
                    for column in eachindex(problem.column_pools)
                    if problem.column_pools[column] == pool &&
                       problem.column_skills[column] == certificate_skill
                ]
                max_paid = isempty(paid) ? 0 : maximum(paid)
                expected_bound += problem.pool_capacities[pool] *
                                  problem.pool_productivity[
                                      pool, certificate_skill,
                                  ] * max_paid
            end
            @test something(problem.infeasibility_capacity_bound) ≈
                  expected_bound
            @test sum(problem.demand[:, certificate_skill]) >
                  something(problem.infeasibility_capacity_bound)
        end

        # Unknown mode starts from the same sampled structure as feasible mode
        # but applies genuine labor and workload shocks. It exposes no witness
        # or infeasibility certificate and makes no solver-status promise.
        feasible_model, feasible_problem =
            generate_problem(ref, 260, feasible, 23)
        unknown_model, unknown_problem =
            generate_problem(ref, 260, unknown, 23)
        @test feasible_problem.feasibility_status == feasible
        @test unknown_problem.feasibility_status == unknown
        @test unknown_problem.profile == feasible_problem.profile
        @test unknown_problem.pattern_coverage ==
              feasible_problem.pattern_coverage
        @test unknown_problem.column_pools == feasible_problem.column_pools
        @test unknown_problem.column_patterns ==
              feasible_problem.column_patterns
        @test unknown_problem.column_skills == feasible_problem.column_skills
        @test unknown_problem.pool_capacities !=
              feasible_problem.pool_capacities
        @test unknown_problem.demand != feasible_problem.demand
        @test unknown_problem.feasible_staffing === nothing
        @test unknown_problem.infeasible_skill === nothing
        @test unknown_problem.infeasibility_capacity_bound === nothing

        if HAS_HIGHS
            for seed in 1:6, status in (feasible, infeasible)
                model, _ = generate_problem(ref, 260, status, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
                @test termination_status(model) == expected
            end
            set_optimizer(unknown_model, HiGHS.Optimizer)
            set_silent(unknown_model)
            optimize!(unknown_model)
            @test termination_status(unknown_model) in
                  (MOI.OPTIMAL, MOI.INFEASIBLE)
            set_optimizer(large_model, HiGHS.Optimizer)
            set_silent(large_model)
            optimize!(large_model)
            @test termination_status(large_model) == MOI.OPTIMAL
        end
    end

    # Operating room scheduling: registry wiring, data contracts,
    # variable-count formulas, witness validity, and certificate structure.
    @testset "Operating Room Scheduling" begin
        @test :operating_room_scheduling in list_categories()
        @test list_variants(:operating_room_scheduling) ==
              [:case_sequencing, :elective_assignment, :weekly_planning]
        @test problem_info(:operating_room_scheduling)[:default_variant] ==
              :elective_assignment
        @test ProblemVariant("operating_room_scheduling") ==
              ProblemVariant(:operating_room_scheduling, :elective_assignment)

        elective_ref = ProblemVariant(:operating_room_scheduling, :elective_assignment)
        sequencing_ref = ProblemVariant(:operating_room_scheduling, :case_sequencing)
        weekly_ref = ProblemVariant(:operating_room_scheduling, :weekly_planning)

        # --- elective_assignment data contracts and exact variable count ---
        for seed in 0:3, target in (100, 500)
            model, p = generate_problem(elective_ref, target, feasible, seed)
            @test num_variables(model) ==
                  length(p.admissible) + p.n_surgeries + length(p.open_blocks)
            @test all(20.0 <= d <= 480.0 for d in p.surgery_duration)
            @test all(1 <= p.surgery_deadline[i] <= p.n_days for i in 1:p.n_surgeries)
            @test all((p.mss[r, d] == 0) == (p.session_length[r, d] == 0)
                      for r in 1:p.n_rooms, d in 1:p.n_days)
            @test Set(p.open_blocks) ==
                  Set((r, d) for r in 1:p.n_rooms, d in 1:p.n_days if p.session_length[r, d] > 0)
            # Every specialty has at least one block per (partial) week.
            for k in 1:p.n_specialties
                @test count(==(k), p.mss) >= max(1, p.n_days ÷ 5)
            end
            # Admissible triples: specialty match, surgeon works, within deadline.
            for (i, r, d) in p.admissible
                @test p.mss[r, d] == p.surgery_specialty[i]
                @test p.surgeon_budget[p.surgery_surgeon[i], d] > 0
                @test d <= p.surgery_deadline[i]
            end
            # Mandatory exactly the urgent cases; each has an admissible block.
            for i in 1:p.n_surgeries
                @test p.mandatory[i] == (p.surgery_urgency[i] == :urgent)
                p.mandatory[i] &&
                    @test any(t[1] == i for t in p.admissible)
            end
            # The planted witness is a feasible point: at most one block per
            # surgery, block and surgeon capacities respected, all mandatory
            # cases scheduled.
            w = something(p.feasible_witness)
            rem_room = copy(p.session_length)
            rem_surg = copy(p.surgeon_budget)
            assigned = falses(p.n_surgeries)
            for a in w
                (i, r, d) = p.admissible[a]
                @test !assigned[i]
                assigned[i] = true
                rem_room[r, d] -= p.surgery_duration[i] + p.turnover
                rem_surg[p.surgery_surgeon[i], d] -= p.surgery_duration[i]
            end
            @test all(rem_room .>= -1e-9)
            @test all(rem_surg .>= -1e-9)
            @test all(!p.mandatory[i] || assigned[i] for i in 1:p.n_surgeries)
        end

        # elective_assignment infeasibility certificate: the victim is
        # mandatory and its surgeon's budgets over its admissible days sum to
        # less than its duration.
        for seed in 0:3
            _, p = generate_problem(elective_ref, 200, infeasible, seed)
            victim = something(p.infeasible_surgery)
            @test p.feasible_witness === nothing
            @test p.mandatory[victim]
            surgeon = p.surgery_surgeon[victim]
            days = unique(t[3] for t in p.admissible if t[1] == victim)
            @test sum(p.surgeon_budget[surgeon, d] for d in days; init=0.0) <
                  p.surgery_duration[victim]
        end
        _, pu = generate_problem(elective_ref, 200, unknown, 0)
        @test pu.feasible_witness === nothing
        @test pu.infeasible_surgery === nothing
        @test pu.feasibility_status == unknown

        # --- case_sequencing data contracts and exact variable count ---
        for seed in 0:3, target in (100, 500)
            model, p = generate_problem(sequencing_ref, target, feasible, seed)
            @test num_variables(model) ==
                  sum(length, p.eligible_rooms) +
                  sum(length, p.eligible_surgeons) +
                  length(p.room_pairs) + length(p.surgeon_pairs) +
                  2 * p.n_surgeries + 1
            @test all(!isempty(p.eligible_rooms[o]) for o in 1:p.n_surgeries)
            @test all(!isempty(p.eligible_surgeons[o]) for o in 1:p.n_surgeries)
            # Every case individually fits at least one eligible surgeon's window.
            for o in 1:p.n_surgeries
                @test any(p.surgeon_window_end[s] - p.surgeon_window_start[s] >=
                          p.surgery_duration[o] for s in p.eligible_surgeons[o])
            end
            # Pair lists match eligibility intersections exactly.
            expected_room_pairs = Tuple{Int,Int,Int}[]
            expected_surgeon_pairs = Tuple{Int,Int,Int}[]
            for o in 1:p.n_surgeries, q in (o + 1):p.n_surgeries
                for r in intersect(p.eligible_rooms[o], p.eligible_rooms[q])
                    push!(expected_room_pairs, (o, q, r))
                end
                for s in intersect(p.eligible_surgeons[o], p.eligible_surgeons[q])
                    push!(expected_surgeon_pairs, (o, q, s))
                end
            end
            @test p.room_pairs == expected_room_pairs
            @test p.surgeon_pairs == expected_surgeon_pairs
            # Big-M dominates any feasible start plus one duration and turnover.
            @test p.big_m >= maximum(p.surgeon_window_end) - 1e-9
            # The witness satisfies eligibility, windows, and both no-overlap
            # families with their turnovers.
            w = something(p.feasible_witness)
            room_seen = Dict{Int,Vector{Tuple{Float64,Float64}}}()
            surg_seen = Dict{Int,Vector{Tuple{Float64,Float64}}}()
            for o in 1:p.n_surgeries
                (r, s, t) = w[o]
                dur = p.surgery_duration[o]
                @test r in p.eligible_rooms[o]
                @test s in p.eligible_surgeons[o]
                @test t >= p.surgeon_window_start[s] - 1e-9
                @test t + dur <= p.surgeon_window_end[s] + 1e-9
                for (a, b) in get(room_seen, r, [])
                    @test t + dur + p.room_turnover <= a + 1e-9 ||
                          b + p.room_turnover <= t + 1e-9
                end
                for (a, b) in get(surg_seen, s, [])
                    @test t + dur + p.surgeon_turnover <= a + 1e-9 ||
                          b + p.surgeon_turnover <= t + 1e-9
                end
                push!(get!(room_seen, r, []), (t, t + dur))
                push!(get!(surg_seen, s, []), (t, t + dur))
            end
        end

        # case_sequencing infeasibility certificate: hard deadline below the
        # victim's own duration (completion >= duration since start >= 0).
        for seed in 0:3
            model, p = generate_problem(sequencing_ref, 200, infeasible, seed)
            victim = something(p.infeasible_surgery)
            @test p.feasible_witness === nothing
            @test something(p.hard_deadline) < p.surgery_duration[victim]
            start_vars = model[:start]
            @test length(start_vars) == p.n_surgeries
        end
        _, ps = generate_problem(sequencing_ref, 200, unknown, 0)
        @test ps.feasible_witness === nothing
        @test ps.infeasible_surgery === nothing

        # --- weekly_planning data contracts and exact variable count ---
        for seed in 0:3, target in (100, 500)
            model, p = generate_problem(weekly_ref, target, feasible, seed)
            @test num_variables(model) ==
                  sum(length, p.admissible_days) + p.n_surgeries
            @test all(p.ward_los[i] >= p.icu_los[i] for i in 1:p.n_surgeries)
            @test all(c > 0 for c in p.ward_capacity)
            for i in 1:p.n_surgeries, d in p.admissible_days[i]
                @test p.specialty_capacity[p.surgery_specialty[i], d] > 0
                @test p.surgeon_budget[p.surgery_surgeon[i], d] > 0
                @test d <= p.surgery_deadline[i]
            end
            for i in 1:p.n_surgeries
                @test p.mandatory[i] == (p.surgery_urgency[i] == :urgent)
                p.mandatory[i] && @test !isempty(p.admissible_days[i])
            end
            # The witness respects specialty-day OR capacity, surgeon budgets,
            # and day-by-day ward/ICU occupancy; all mandatory cases scheduled.
            w = something(p.feasible_witness)
            @test length(w) == p.n_surgeries
            rem_spec = copy(p.specialty_capacity)
            rem_surg = copy(p.surgeon_budget)
            occ_ward = zeros(p.n_days)
            occ_icu = zeros(p.n_days)
            for i in 1:p.n_surgeries
                d = w[i]
                d == 0 && continue
                @test d in p.admissible_days[i]
                rem_spec[p.surgery_specialty[i], d] -=
                    p.surgery_duration[i] + p.turnover
                rem_surg[p.surgery_surgeon[i], d] -= p.surgery_duration[i]
                p.ward_los[i] > 0 || continue
                for t in d:min(p.n_days, d + p.ward_los[i] - 1)
                    occ_ward[t] += 1
                end
                for t in d:min(p.n_days, d + p.icu_los[i] - 1)
                    p.icu_los[i] > 0 && (occ_icu[t] += 1)
                end
            end
            @test all(rem_spec .>= -1e-9)
            @test all(rem_surg .>= -1e-9)
            @test all(occ_ward .<= p.ward_capacity .+ 1e-9)
            @test all(occ_icu .<= p.icu_capacity .+ 1e-9)
            @test all(!p.mandatory[i] || w[i] > 0 for i in 1:p.n_surgeries)
        end

        # weekly_planning infeasibility certificate (same surgeon-shortage
        # structure as elective_assignment).
        for seed in 0:3
            _, p = generate_problem(weekly_ref, 200, infeasible, seed)
            victim = something(p.infeasible_surgery)
            @test p.feasible_witness === nothing
            @test p.mandatory[victim]
            surgeon = p.surgery_surgeon[victim]
            @test sum(p.surgeon_budget[surgeon, d]
                      for d in p.admissible_days[victim]; init=0.0) <
                  p.surgery_duration[victim]
        end

        # Exact field reproducibility for one variant of each kind.
        for ref in (elective_ref, sequencing_ref, weekly_ref)
            _, p1 = generate_problem(ref, 240, unknown, 12345)
            _, p2 = generate_problem(ref, 240, unknown, 12345)
            @test all(isequal(getfield(p1, f), getfield(p2, f))
                      for f in fieldnames(typeof(p1)))
        end

        # Solver-based feasibility contracts on the (relaxed) models.
        if HAS_HIGHS
            for ref in (elective_ref, sequencing_ref, weekly_ref)
                for seed in 1:4, status in (feasible, infeasible)
                    model, _ = generate_problem(ref, 220, status, seed)
                    set_optimizer(model, HiGHS.Optimizer)
                    set_silent(model)
                    optimize!(model)
                    expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
                    @test termination_status(model) == expected
                end
            end
        end
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

    @testset "Regression Basis Pursuit" begin
        @test :basis_pursuit in list_variants(:regression)
        info = problem_info(:regression, :basis_pursuit)
        @test occursin("basis-pursuit", lowercase(info[:description]))
        @test ProblemVariant("regression/basis_pursuit") ==
              ProblemVariant(:regression, :basis_pursuit)

        profiles = (
            :gaussian_well_conditioned,
            :correlated_columns,
            :sparse_measurements,
        )
        profile_seeds = Dict(profile => Int[] for profile in profiles)
        for seed in 1:100
            _, prob = generate_problem("regression/basis_pursuit", 150, feasible, seed)
            length(profile_seeds[prob.profile]) < 3 &&
                push!(profile_seeds[prob.profile], seed)
        end
        @test all(length(profile_seeds[profile]) == 3 for profile in profiles)

        check_status_data = function(prob)
            @test (prob.certificate !== nothing) ==
                  (prob.resolved_status == infeasible)
            @test all(any(!iszero, @view prob.A[i, :])
                      for i in 1:prob.n_measurements)
            @test all(any(!iszero, @view prob.A[:, j])
                      for j in 1:prob.n_features)
            if prob.resolved_status == feasible
                @test prob.certificate === nothing
                @test prob.A * prob.source_signal ≈ prob.b
            else
                certificate = prob.certificate
                @test certificate isa SyntheticLPs.BasisPursuitCertificate
                if certificate !== nothing
                    r1, r2 = certificate.rows
                    @test 1 <= r1 <= prob.n_measurements
                    @test 1 <= r2 <= prob.n_measurements
                    @test r1 != r2
                    @test prob.A[r2, :] ==
                          certificate.multiplier .* prob.A[r1, :]
                    @test prob.b[r2] ≈
                          certificate.multiplier * prob.b[r1] +
                          certificate.rhs_gap
                    @test !iszero(certificate.rhs_gap)
                    @test !(prob.A * prob.source_signal ≈ prob.b)
                end
            end
        end

        # Positive/negative splitting makes the count intrinsically even: even
        # targets are exact, odd targets round up one, and two is the minimum.
        for target in (1, 2, 3, 4, 5, 50, 501, 2000)
            model, prob = generate_problem(
                "regression/basis_pursuit",
                target,
                feasible,
                17,
            )
            expected = 2 * max(1, cld(max(target, 1), 2))
            @test num_variables(model) == expected == 2 * prob.n_features
            @test size(prob.A) == (prob.n_measurements, prob.n_features)
            @test length(prob.b) == prob.n_measurements
            @test length(prob.weights) == prob.n_features
            @test length(prob.source_signal) == prob.n_features
        end

        # Deterministic data, repeated builds, and MPS export for every profile
        # under both resolved statuses.
        mktempdir() do tmp
            for profile in profiles, status in (feasible, infeasible)
                seed = first(profile_seeds[profile])
                model1, prob1 =
                    generate_problem("regression/basis_pursuit", 150, status, seed)
                model2, prob2 =
                    generate_problem("regression/basis_pursuit", 150, status, seed)
                @test prob1.profile == prob2.profile == profile
                for field in fieldnames(typeof(prob1))
                    @test getfield(prob1, field) == getfield(prob2, field)
                end
                rebuilt = SyntheticLPs.build_model(prob1)
                @test num_variables(model1) == num_variables(model2) ==
                      num_variables(rebuilt)
                @test num_constraints(model1; count_variable_in_set_constraints=true) ==
                      num_constraints(model2; count_variable_in_set_constraints=true) ==
                      num_constraints(rebuilt; count_variable_in_set_constraints=true)

                prefix = "$(profile)_$(status)"
                paths = [joinpath(tmp, "$(prefix)_$copy.mps") for copy in 1:3]
                write_to_file(model1, paths[1])
                write_to_file(model2, paths[2])
                write_to_file(rebuilt, paths[3])
                @test read(paths[1], String) == read(paths[2], String) ==
                      read(paths[3], String)
            end
        end

        # The constructor owns a local RNG and does not perturb Random.default_rng().
        Random.seed!(8801)
        expected_draws = rand(4)
        Random.seed!(8801)
        SyntheticLPs.BasisPursuitProblem(100, feasible, 9)
        @test rand(4) == expected_draws

        # A deterministic seed sample covers both natural unknown outcomes, and
        # each outcome carries exactly its matching witness/certificate data.
        unknown_statuses = Set{FeasibilityStatus}()
        varied = Any[]
        for seed in 1:60
            _, feasible_prob =
                generate_problem("regression/basis_pursuit", 120, feasible, seed)
            _, unknown_prob =
                generate_problem("regression/basis_pursuit", 120, unknown, seed)
            push!(unknown_statuses, unknown_prob.resolved_status)
            check_status_data(unknown_prob)
            seed <= 12 && push!(varied, feasible_prob)
        end
        @test unknown_statuses == Set((feasible, infeasible))
        @test length(unique(p.profile for p in varied)) > 1
        @test length(unique(Tuple(p.support) for p in varied)) > 1
        @test any(p.A != varied[1].A for p in varied[2:end])

        # Profile statistics are checked on multiple seeds, including a
        # large-instance regression against coherence decay.
        for profile in profiles, seed in profile_seeds[profile]
            _, prob = generate_problem("regression/basis_pursuit", 150, feasible, seed)
            @test prob.resolved_status == feasible
            @test issorted(prob.support)
            @test allunique(prob.support)
            @test all(1 <= j <= prob.n_features for j in prob.support)
            @test findall(!iszero, prob.source_signal) == prob.support
            @test all(>(0.0), prob.weights)
            @test norm(prob.A * prob.source_signal - prob.b, Inf) <= 1.0e-10
            @test norm(prob.b, Inf) > 1.0e-8
            @test prob.certificate === nothing

            if profile == :gaussian_well_conditioned
                identity_rows = Matrix{Float64}(
                    I,
                    prob.n_measurements,
                    prob.n_measurements,
                )
                @test norm(prob.A * transpose(prob.A) - identity_rows, Inf) <=
                      1.0e-10
            elseif profile == :correlated_columns
                normalized = prob.A ./ sqrt.(sum(abs2, prob.A; dims=1))
                gram = transpose(normalized) * normalized
                identity_columns = Matrix{Float64}(
                    I,
                    prob.n_features,
                    prob.n_features,
                )
                @test maximum(abs.(gram - identity_columns)) >= 0.985
            else
                density = count(!iszero, prob.A) / length(prob.A)
                @test density <= 0.2
                @test all(any(!iszero, @view prob.A[i, :])
                          for i in 1:prob.n_measurements)
                @test all(any(!iszero, @view prob.A[:, j])
                          for j in 1:prob.n_features)
            end
        end

        for seed in profile_seeds[:correlated_columns]
            _, prob =
                generate_problem("regression/basis_pursuit", 2000, feasible, seed)
            normalized = prob.A ./ sqrt.(sum(abs2, prob.A; dims=1))
            sample_width = min(200, prob.n_features)
            sample = @view normalized[:, 1:sample_width]
            gram = transpose(sample) * sample
            identity_columns = Matrix{Float64}(I, sample_width, sample_width)
            @test maximum(abs.(gram - identity_columns)) >= 0.985
        end

        # Feasible instances retain their source signal as an exact witness;
        # infeasible instances carry only the inspectable algebraic certificate.
        for seed in 1:12
            _, feasible_prob =
                generate_problem("regression/basis_pursuit", 100, feasible, seed)
            @test feasible_prob.resolved_status == feasible
            check_status_data(feasible_prob)

            _, infeasible_prob =
                generate_problem("regression/basis_pursuit", 100, infeasible, seed)
            @test infeasible_prob.resolved_status == infeasible
            check_status_data(infeasible_prob)
        end

        # Certificate injection must not erase sparse columns whose only
        # nonzero sat in the replaced row. Target 20 has measurement width 1.
        sparse_infeasible = 0
        for seed in 0:199
            _, prob = generate_problem(
                "regression/basis_pursuit", 20, infeasible, seed
            )
            prob.profile == :sparse_measurements || continue
            sparse_infeasible += 1
            check_status_data(prob)
        end
        @test sparse_infeasible >= 20

        # Every profile also constructs correctly at the one-feature minimum,
        # under both statuses. Gaussian rows cannot both be orthonormal in this
        # 2×1 geometry, so its feasible matrix is normalized as one column.
        tiny_profile_seeds = Dict{Symbol,Int}()
        for seed in 1:60
            _, prob = generate_problem("regression/basis_pursuit", 1, feasible, seed)
            get!(tiny_profile_seeds, prob.profile, seed)
        end
        @test Set(keys(tiny_profile_seeds)) == Set(profiles)
        for profile in profiles, target in (1, 2, 3), status in (feasible, infeasible)
            model, prob = generate_problem(
                "regression/basis_pursuit",
                target,
                status,
                tiny_profile_seeds[profile],
            )
            @test prob.profile == profile
            @test num_variables(model) == (target <= 2 ? 2 : 4)
            @test prob.n_measurements == 2
            check_status_data(prob)
        end
        _, tiny_gaussian = generate_problem(
            "regression/basis_pursuit",
            1,
            feasible,
            tiny_profile_seeds[:gaussian_well_conditioned],
        )
        @test size(tiny_gaussian.A) == (2, 1)
        @test norm(tiny_gaussian.A) ≈ 1.0

        # Coherent and sparse profiles vary numerically between same-profile
        # seeds, not merely through their profile labels.
        for profile in (:correlated_columns, :sparse_measurements)
            matrices = [
                last(generate_problem("regression/basis_pursuit", 150, feasible, seed)).A
                for seed in profile_seeds[profile]
            ]
            @test all(matrices[i] != matrices[j]
                      for (i, j) in ((1, 2), (1, 3), (2, 3)))
        end

        # Assert the complete JuMP formulation, not only variable domains/counts.
        domain_model, domain_prob =
            generate_problem("regression/basis_pursuit", 80, feasible, 4)
        @test objective_sense(domain_model) == MOI.MIN_SENSE
        @test num_constraints(
            domain_model,
            AffExpr,
            MOI.EqualTo{Float64},
        ) == domain_prob.n_measurements
        for variable in all_variables(domain_model)
            @test !is_binary(variable)
            @test !is_integer(variable)
            @test has_lower_bound(variable)
            @test lower_bound(variable) == 0.0
            @test !has_upper_bound(variable)
        end
        objective = objective_function(domain_model)
        for j in 1:domain_prob.n_features
            @test coefficient(objective, domain_model[:x_pos][j]) ==
                  domain_prob.weights[j]
            @test coefficient(objective, domain_model[:x_neg][j]) ==
                  domain_prob.weights[j]
        end
        for i in 1:domain_prob.n_measurements
            row = domain_model[:measurements][i]
            @test normalized_rhs(row) == domain_prob.b[i]
            for j in 1:domain_prob.n_features
                @test normalized_coefficient(row, domain_model[:x_pos][j]) ==
                      domain_prob.A[i, j]
                @test normalized_coefficient(row, domain_model[:x_neg][j]) ==
                      -domain_prob.A[i, j]
            end
        end
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
        # set_system variants used to reject targets below 4; they now size
        # a feasible planted partition down to 2 columns/bids.
        for variant in ("set_cover", "set_packing", "set_partitioning",
                        "combinatorial_auction")
            ref = "set_system/$variant"
            @test_nowarn generate_problem(ref, 2, unknown, 1)
            @test_nowarn generate_problem(ref, 3, infeasible, 1)
        end
        tiny = generate_dataset(num_problems = 4,
                                size_distribution = Uniform(2, 3),
                                problem_types = [:set_system],
                                seed = 1,
                                max_candidate_multiplier = 3)
        @test length(tiny) == 4
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
        # graph_optimization/generalized_independent_set used to request more
        # hard edges than leave n_soft unused pairs, throwing for targets 6–10.
        for target in 6:10, status in (feasible, unknown, infeasible)
            @test_nowarn generate_problem("graph_optimization/generalized_independent_set",
                                          target, status, 1)
        end
        _, gis = generate_problem("graph_optimization/generalized_independent_set", 6, feasible, 1)
        @test length(gis.soft_edges) == 6 - gis.n_vertices
        @test length(gis.hard_edges) + length(gis.soft_edges) <=
              gis.n_vertices * (gis.n_vertices - 1) ÷ 2
        # knapsack/mixed_integer_set stores sparse row supports instead of a
        # dense n_rows × n_variables coefficient matrix.
        @test_nowarn generate_problem("knapsack/mixed_integer_set", 1, unknown, 1)
        _, mik = generate_problem("knapsack/mixed_integer_set", 80, unknown, 1)
        @test length(mik.row_indices) == mik.n_rows
        @test all(length(mik.row_indices[r]) == length(mik.row_coefficients[r])
                  for r in 1:mik.n_rows)
        @test all(allunique(mik.row_indices[r]) &&
                  all(1 <= i <= mik.n_integer + mik.n_continuous for i in mik.row_indices[r])
                  for r in 1:mik.n_rows)
        # generic_milp samples each row support in O(width) rather than
        # permuting all n columns; keep a moderately large constructor cheap.
        @test_nowarn generate_problem("generic_milp/standard", 3, unknown, 1)
        _, gmilp = generate_problem("generic_milp/standard", 200, unknown, 1)
        @test all(issorted(row.indices) && allunique(row.indices) for row in gmilp.rows)
        @test all(1 <= length(row.indices) <= gmilp.n_variables for row in gmilp.rows)
        # container_loading used to reject targets below 12 (standard) / 30
        # (2-D packing); both now clamp to their smallest formulation.
        @test_nowarn generate_problem("container_loading/standard", 2, unknown, 1)
        @test_nowarn generate_problem("container_loading/standard", 11, feasible, 1)
        @test_nowarn generate_problem("container_loading/two_dimensional_bin_packing", 2, unknown, 1)
        @test_nowarn generate_problem("container_loading/two_dimensional_bin_packing", 29, infeasible, 1)
        # energy/optimal_transmission_switching samples extra lines by rejection
        # instead of materializing the complete undirected edge set.
        @test_nowarn generate_problem("energy/optimal_transmission_switching", 3, unknown, 1)
        _, ots = generate_problem("energy/optimal_transmission_switching", 500, unknown, 1)
        @test ots.n_lines >= ots.n_buses - 1
        @test length(unique(ots.line_from[k] < ots.line_to[k] ?
                            (ots.line_from[k], ots.line_to[k]) :
                            (ots.line_to[k], ots.line_from[k])
                            for k in 1:ots.n_lines)) == ots.n_lines
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

    @testset "Basis Pursuit Feasibility Contracts" begin
        # Exercise three seeds per profile under both labels. Passing the
        # optimizer invokes the package-level contract check before returning
        # the pristine model.
        profiles = (
            :gaussian_well_conditioned,
            :correlated_columns,
            :sparse_measurements,
        )
        profile_seeds = Dict(profile => Int[] for profile in profiles)
        for seed in 1:100
            _, prob = generate_problem("regression/basis_pursuit", 120, feasible, seed)
            length(profile_seeds[prob.profile]) < 3 &&
                push!(profile_seeds[prob.profile], seed)
        end
        @test all(length(profile_seeds[profile]) == 3 for profile in profiles)

        for profile in profiles, seed in profile_seeds[profile]
            feasible_model, feasible_prob = generate_problem(
                "regression/basis_pursuit",
                120,
                feasible,
                seed;
                optimizer=HiGHS.Optimizer,
            )
            set_optimizer(feasible_model, HiGHS.Optimizer)
            set_silent(feasible_model)
            optimize!(feasible_model)
            @test termination_status(feasible_model) == MOI.OPTIMAL
            @test objective_value(feasible_model) > 1.0e-8
            @test feasible_prob.profile == profile
            @test feasible_prob.certificate === nothing
            @test norm(
                feasible_prob.A * feasible_prob.source_signal - feasible_prob.b,
                Inf,
            ) <= 1.0e-10

            infeasible_model, infeasible_prob = generate_problem(
                "regression/basis_pursuit",
                120,
                infeasible,
                seed;
                optimizer=HiGHS.Optimizer,
            )
            set_optimizer(infeasible_model, HiGHS.Optimizer)
            set_silent(infeasible_model)
            optimize!(infeasible_model)
            @test termination_status(infeasible_model) in
                  (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
            @test infeasible_prob.profile == profile
            @test infeasible_prob.resolved_status == infeasible
            @test infeasible_prob.certificate !== nothing
        end

        # Unknown requests skip package-level verification, so solve both
        # resolved labels directly and require metadata and solver status to
        # agree. Two seeds per label avoid a single representative special case.
        unknown_seeds = Dict(feasible => Int[], infeasible => Int[])
        for seed in 1:100
            _, prob = generate_problem("regression/basis_pursuit", 120, unknown, seed)
            seeds = unknown_seeds[prob.resolved_status]
            length(seeds) < 2 && push!(seeds, seed)
        end
        @test all(length(unknown_seeds[status]) == 2
                  for status in (feasible, infeasible))
        for status in (feasible, infeasible), seed in unknown_seeds[status]
            model, prob =
                generate_problem("regression/basis_pursuit", 120, unknown, seed)
            @test prob.resolved_status == status
            @test (prob.certificate !== nothing) == (status == infeasible)
            if status == feasible
                @test prob.A * prob.source_signal ≈ prob.b
            else
                certificate = prob.certificate
                @test certificate !== nothing
                if certificate !== nothing
                    r1, r2 = certificate.rows
                    @test prob.A[r2, :] ==
                          certificate.multiplier .* prob.A[r1, :]
                    @test prob.b[r2] ≈
                          certificate.multiplier * prob.b[r1] +
                          certificate.rhs_gap
                end
            end
            set_optimizer(model, HiGHS.Optimizer)
            set_silent(model)
            optimize!(model)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(model) == expected
        end
    end

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

        # The network-planning variant has a solver-independent planted plan for
        # feasible requests and a cumulative product cut for infeasible requests.
        for status in (feasible, infeasible), s in 0:5
            m, _ = generate_problem(
                "supply_chain/network_planning", 240, status, s
            )
            set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end

        # Unknown is a mixed nominal distribution, not an implicit
        # almost-always-infeasible branch. Local lane cuts are excluded by
        # construction; correlated aggregate supply conditions produce both
        # outcomes over this deterministic representative sample.
        unknown_optimal = 0
        unknown_infeasible = 0
        unknown_singleton_cuts = 0
        for target in (200, 500, 5000), s in 0:11
            m, p = generate_problem(
                "supply_chain/network_planning", target, unknown, s
            )
            for c in 1:p.n_customers, k in 1:p.n_products,
                t in 1:p.n_periods
                incoming_capacity = sum(
                    p.lane_capacity[a] for a in p.shipment_arcs
                    if a[2] == c && a[3] == k && a[4] == t
                )
                unknown_singleton_cuts +=
                    incoming_capacity + 1e-10 < p.demand[c, k, t]
            end
            set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
            ts = termination_status(m)
            unknown_optimal += ts == MOI.OPTIMAL
            unknown_infeasible +=
                ts in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
        end
        @test unknown_singleton_cuts == 0
        @test unknown_optimal >= 6
        @test unknown_infeasible >= 3
        @test unknown_optimal + unknown_infeasible == 36

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
