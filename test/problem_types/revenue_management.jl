using Test
using JuMP
using Random
using SyntheticLPs

const REVENUE_MOI = JuMP.MOI
const REVENUE_STANDARD = "revenue_management/standard"
const REVENUE_OVERBOOKING = "revenue_management/stochastic_overbooking"

const HAS_REVENUE_HIGHS = try
    @eval using HiGHS
    true
catch
    false
end

function revenue_product_signature(product)
    return (product.id, product.origin, product.destination, product.fare_class, product.resources)
end

function check_revenue_network(problem)
    @test problem.n_nodes >= 2
    @test problem.n_resources >= 2
    @test length(problem.products) == problem.n_products
    @test length(problem.product_resources) == problem.n_products
    @test length(problem.resource_products) == problem.n_resources
    @test length(problem.resource_names) == problem.n_resources
    @test length(problem.resource_origin) == problem.n_resources
    @test length(problem.resource_destination) == problem.n_resources
    @test all(!isempty, problem.resource_products)

    for (j, product) in enumerate(problem.products)
        @test product isa SyntheticLPs.RevenueManagementProduct
        @test product.id == j
        @test product.resources == problem.product_resources[j]
        @test length(product.resources) in (1, 2)
        @test all(r -> 1 <= r <= problem.n_resources, product.resources)
        @test product.origin == problem.resource_origin[first(product.resources)]
        @test product.destination == problem.resource_destination[last(product.resources)]
        @test product.fare_class in (:economy, :premium, :business)
        @test all(j in problem.resource_products[r] for r in product.resources)

        if length(product.resources) == 2
            first_leg, second_leg = product.resources
            @test problem.resource_destination[first_leg] == 1
            @test problem.resource_origin[second_leg] == 1
            @test product.origin != product.destination
        end
    end
    for r in 1:problem.n_resources, j in problem.resource_products[r]
        @test r in problem.product_resources[j]
    end
end

@testset "Revenue management category" begin
    @test list_variants(:revenue_management) == [:standard, :stochastic_overbooking]
    @test ProblemVariant(:revenue_management) == ProblemVariant(:revenue_management, :standard)
    @test problem_info(:revenue_management)[:default_variant] == :standard

    @testset "standard sizing, network data, and deterministic construction" begin
        for target in (-5, 0, 1, 2, 3, 50, 120, 500, 2_000)
            model, problem = generate_problem(REVENUE_STANDARD, target, feasible, 17)
            @test num_variables(model) == max(2, target)
            @test problem.n_products == max(2, target)
            check_revenue_network(problem)
            @test all(problem.fare .> 0)
            @test all(problem.demand .>= 2)
            @test all(0 .<= problem.commitment .<= problem.demand)
            @test all(problem.capacity .> 0)
        end

        _, first = generate_problem(REVENUE_STANDARD, 240, infeasible, 12_345)
        _, second = generate_problem(REVENUE_STANDARD, 240, infeasible, 12_345)
        @test first.n_resources == second.n_resources
        @test first.n_nodes == second.n_nodes
        @test revenue_product_signature.(first.products) ==
            revenue_product_signature.(second.products)
        @test first.product_resources == second.product_resources
        @test first.resource_products == second.resource_products
        @test first.resource_names == second.resource_names
        @test first.resource_origin == second.resource_origin
        @test first.resource_destination == second.resource_destination
        @test first.fare == second.fare
        @test first.demand == second.demand
        @test first.commitment == second.commitment
        @test first.capacity == second.capacity
        @test first.market_profile == second.market_profile
        @test first.resolved_status == second.resolved_status
        first_certificate = something(first.infeasibility_certificate)
        second_certificate = something(second.infeasibility_certificate)
        @test first_certificate.resource == second_certificate.resource
        @test first_certificate.committed_load == second_certificate.committed_load
        @test first_certificate.capacity == second_certificate.capacity
        @test first_certificate.excess == second_certificate.excess

        Random.seed!(68_731)
        expected_first = rand()
        expected_second = rand()
        Random.seed!(68_731)
        @test rand() == expected_first
        generate_problem(REVENUE_STANDARD, 120, feasible, 99)
        @test rand() == expected_second

        markets = Set{Symbol}()
        fare_classes = Set{Symbol}()
        found_connection = false
        for seed in 0:31
            _, problem = generate_problem(REVENUE_STANDARD, 500, feasible, seed)
            push!(markets, problem.market_profile)
            union!(fare_classes, (product.fare_class for product in problem.products))
            found_connection |= any(length(product.resources) == 2 for product in problem.products)
        end
        @test markets == Set((:regional_airline, :network_airline, :intercity_rail))
        @test fare_classes == Set((:economy, :premium, :business))
        @test found_connection
    end

    @testset "standard constructive status guarantees" begin
        for target in (2, 50, 120, 500, 2_000), seed in 0:11
            model, problem = generate_problem(REVENUE_STANDARD, target, feasible, seed)
            @test problem.resolved_status == feasible
            @test problem.feasible_witness !== nothing
            @test problem.infeasibility_certificate === nothing
            @test SyntheticLPs._revenue_management_witness_is_valid(problem)
            @test model[:x] === model[:acceptance]

            witness = something(problem.feasible_witness)
            @test witness.acceptance == problem.commitment
            @test start_value(model[:acceptance][1]) == witness.acceptance[1]
            @test length(model[:resource_capacity]) == problem.n_resources
            for r in 1:problem.n_resources
                row = model[:resource_capacity][r]
                object = constraint_object(row)
                @test object.set isa REVENUE_MOI.LessThan{Float64}
                @test object.set.upper == problem.capacity[r]
                @test all(
                    normalized_coefficient(row, model[:acceptance][j]) == 1.0 for
                    j in problem.resource_products[r]
                )
            end
        end

        for target in (2, 50, 120, 500, 2_000), seed in 0:11
            _, problem = generate_problem(REVENUE_STANDARD, target, infeasible, seed)
            @test problem.resolved_status == infeasible
            @test problem.feasible_witness === nothing
            @test problem.infeasibility_certificate !== nothing
            @test SyntheticLPs._revenue_management_certificate_is_valid(problem)

            certificate = something(problem.infeasibility_certificate)
            mandatory = sum(
                problem.commitment[j] for j in problem.resource_products[certificate.resource]
            )
            @test certificate.committed_load ≈ mandatory
            @test certificate.capacity == problem.capacity[certificate.resource]
            @test certificate.excess ≈ mandatory - certificate.capacity
            @test certificate.excess > 0
        end

        statuses = Set{FeasibilityStatus}()
        for seed in 0:63
            _, problem = generate_problem(REVENUE_STANDARD, 120, unknown, seed)
            push!(statuses, problem.resolved_status)
            @test if problem.resolved_status == feasible
                SyntheticLPs._revenue_management_witness_is_valid(problem)
            else
                SyntheticLPs._revenue_management_certificate_is_valid(problem)
            end
        end
        @test statuses == Set((feasible, infeasible))
    end

    @testset "stochastic overbooking sizing and scenario data" begin
        for target in (-5, 0, 2, 14, 15, 50, 149, 150, 500, 1_199, 1_200, 5_000)
            model, problem = generate_problem(REVENUE_OVERBOOKING, target, feasible, 29)
            actual = num_variables(model)
            @test actual == problem.n_products * (1 + 2 * problem.n_scenarios)
            @test actual >= 14

            adjusted_target = max(target, 14)
            scenarios = if adjusted_target < 150
                (3:5)
            elseif adjusted_target < 1_200
                (4:8)
            else
                (6:12)
            end
            best_error = minimum(
                abs(
                    max(2, round(Int, adjusted_target / (1 + 2 * s))) * (1 + 2 * s) -
                    adjusted_target,
                ) for s in scenarios
            )
            @test abs(actual - adjusted_target) == best_error
            check_revenue_network(problem)

            @test length(problem.scenario_probability) == problem.n_scenarios
            @test all(problem.scenario_probability .> 0)
            @test sum(problem.scenario_probability) ≈ 1.0
            @test size(problem.show_rate) == (problem.n_products, problem.n_scenarios)
            @test all(0.55 .<= problem.show_rate .<= 0.995)
            @test all(problem.denied_service_cost .> problem.fare)
            @test all(0 .< problem.max_denied_fraction .< 0.08)
            @test all(problem.scenario_denied_cap .> 0)
        end

        _, first = generate_problem(REVENUE_OVERBOOKING, 500, infeasible, 7_771)
        _, second = generate_problem(REVENUE_OVERBOOKING, 500, infeasible, 7_771)
        @test revenue_product_signature.(first.products) ==
            revenue_product_signature.(second.products)
        @test first.product_resources == second.product_resources
        @test first.resource_products == second.resource_products
        @test first.resource_names == second.resource_names
        @test first.resource_origin == second.resource_origin
        @test first.resource_destination == second.resource_destination
        @test first.fare == second.fare
        @test first.demand == second.demand
        @test first.commitment == second.commitment
        @test first.capacity == second.capacity
        @test first.scenario_probability == second.scenario_probability
        @test first.show_rate == second.show_rate
        @test first.denied_service_cost == second.denied_service_cost
        @test first.max_denied_fraction == second.max_denied_fraction
        @test first.scenario_denied_cap == second.scenario_denied_cap
        @test first.market_profile == second.market_profile
        @test first.show_profile == second.show_profile

        Random.seed!(91_337)
        expected_first = rand()
        expected_second = rand()
        Random.seed!(91_337)
        @test rand() == expected_first
        generate_problem(REVENUE_OVERBOOKING, 500, feasible, 6)
        @test rand() == expected_second

        show_profiles = Set{Symbol}()
        for seed in 0:47
            _, problem = generate_problem(REVENUE_OVERBOOKING, 500, feasible, seed)
            push!(show_profiles, problem.show_profile)
        end
        @test show_profiles == Set((:stable_business, :mixed_leisure, :disruption_prone))
    end

    @testset "stochastic recourse formulation and status guarantees" begin
        for target in (14, 50, 150, 500, 1_200), seed in 0:9
            model, problem = generate_problem(REVENUE_OVERBOOKING, target, feasible, seed)
            @test problem.resolved_status == feasible
            @test problem.feasible_witness !== nothing
            @test problem.infeasibility_certificate === nothing
            @test SyntheticLPs._stochastic_overbooking_witness_is_valid(problem)

            witness = something(problem.feasible_witness)
            @test witness.bookings == problem.commitment
            @test all(iszero, witness.denied)
            @test witness.served ≈
                problem.show_rate .* reshape(problem.commitment, problem.n_products, 1)
            @test start_value(model[:bookings][1]) == witness.bookings[1]
            @test start_value(model[:served][1, 1]) == witness.served[1, 1]
            @test start_value(model[:denied][1, 1]) == 0.0

            balance = model[:show_balance][1, 1]
            balance_object = constraint_object(balance)
            @test balance_object.set isa REVENUE_MOI.EqualTo{Float64}
            @test balance_object.set.value == 0.0
            @test normalized_coefficient(balance, model[:served][1, 1]) == 1.0
            @test normalized_coefficient(balance, model[:denied][1, 1]) == 1.0
            @test normalized_coefficient(balance, model[:bookings][1]) ≈ -problem.show_rate[1, 1]

            denial_row = model[:product_denial_cap][1, 1]
            denial_object = constraint_object(denial_row)
            @test denial_object.set isa REVENUE_MOI.LessThan{Float64}
            @test denial_object.set.upper == 0.0
            @test normalized_coefficient(denial_row, model[:denied][1, 1]) == 1.0
            @test normalized_coefficient(denial_row, model[:bookings][1]) ≈
                -problem.max_denied_fraction[1] * problem.show_rate[1, 1]

            @test size(model[:scenario_capacity]) == (problem.n_resources, problem.n_scenarios)
            @test length(model[:scenario_denial_cap]) == problem.n_scenarios
        end

        for target in (14, 50, 150, 500, 1_200), seed in 0:9
            _, problem = generate_problem(REVENUE_OVERBOOKING, target, infeasible, seed)
            @test problem.resolved_status == infeasible
            @test problem.feasible_witness === nothing
            @test problem.infeasibility_certificate !== nothing
            @test SyntheticLPs._stochastic_overbooking_certificate_is_valid(problem)

            certificate = something(problem.infeasibility_certificate)
            mandatory = sum(
                (1 - problem.max_denied_fraction[j]) *
                problem.show_rate[j, certificate.scenario] *
                problem.commitment[j] for j in problem.resource_products[certificate.resource]
            )
            @test certificate.mandatory_service_load ≈ mandatory
            @test certificate.capacity == problem.capacity[certificate.resource]
            @test certificate.excess ≈ mandatory - certificate.capacity
            @test certificate.excess > 0
        end

        statuses = Set{FeasibilityStatus}()
        for seed in 0:63
            _, problem = generate_problem(REVENUE_OVERBOOKING, 500, unknown, seed)
            push!(statuses, problem.resolved_status)
            @test if problem.resolved_status == feasible
                SyntheticLPs._stochastic_overbooking_witness_is_valid(problem)
            else
                SyntheticLPs._stochastic_overbooking_certificate_is_valid(problem)
            end
        end
        @test statuses == Set((feasible, infeasible))
    end

    @testset "build_model is deterministic" begin
        for reference in (REVENUE_STANDARD, REVENUE_OVERBOOKING)
            _, problem = generate_problem(reference, 500, feasible, 42)
            Random.seed!(31_415)
            expected = rand()
            Random.seed!(31_415)
            first_model = SyntheticLPs.build_model(problem)
            @test rand() == expected
            second_model = SyntheticLPs.build_model(problem)
            @test num_variables(first_model) == num_variables(second_model)
            @test num_constraints(first_model; count_variable_in_set_constraints=true) ==
                num_constraints(second_model; count_variable_in_set_constraints=true)
            first_variables = all_variables(first_model)
            second_variables = all_variables(second_model)
            @test name.(first_variables) == name.(second_variables)
            first_objective = objective_function(first_model)
            second_objective = objective_function(second_model)
            @test [coefficient(first_objective, variable) for variable in first_variables] == [coefficient(second_objective, variable) for variable in second_variables]
        end
    end

    if HAS_REVENUE_HIGHS
        @testset "direct HiGHS status contracts (no retries)" begin
            for reference in (REVENUE_STANDARD, REVENUE_OVERBOOKING),
                status in (feasible, infeasible), target in (50, 150, 500),
                seed in 0:5

                model, _ = generate_problem(reference, target, status, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                expected = status == feasible ? REVENUE_MOI.OPTIMAL : REVENUE_MOI.INFEASIBLE
                @test termination_status(model) == expected
            end
        end
    else
        @info "HiGHS unavailable; skipping revenue-management solve checks"
    end
end
