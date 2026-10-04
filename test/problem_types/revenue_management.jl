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

    @testset "standard choice-based network: sizing and schedule data" begin
        for target in (-5, 0, 1, 2, 3, 10, 50, 120, 500, 2_000, 20_000)
            model, problem = generate_problem(REVENUE_STANDARD, target, feasible, 17)
            n_products, n_markets = length(problem.products), length(problem.markets)
            @test num_variables(model) == n_products + n_markets
            @test abs(num_variables(model) - max(2, target)) <= 1
            @test num_constraints(model; count_variable_in_set_constraints=false) ==
                n_products +
                  n_markets +
                  length(problem.flights) +
                  length(problem.contracted_flights)
            @test all(>(0), problem.fare)
            @test all(>(0), problem.attraction)
            @test all(>(0), problem.no_purchase_attraction)
            @test all(>(0), problem.market_size)
            @test all(>(0), problem.capacity)
            # Every flight carries at least one product; itineraries are 1-2
            # flights on the market's day, connecting at a hub in one bank.
            used = falses(length(problem.flights))
            for product in problem.products
                market = problem.markets[product.market]
                legs = [problem.flights[f] for f in product.flights]
                @test length(legs) in (1, 2)
                @test legs[1].origin == market.origin && legs[end].destination == market.destination
                @test all(leg.day == market.day for leg in legs)
                if length(legs) == 2
                    @test legs[1].destination == legs[2].origin <= problem.n_hubs
                    @test legs[1].bank == legs[2].bank
                end
                used[product.flights] .= true
            end
            @test all(used)
            @test issorted(problem.contracted_flights) && allunique(problem.contracted_flights)
            @test all(problem.min_load[f] > 0 for f in problem.contracted_flights)
        end

        # Product columns are not parallel: each has its own scale row.
        model, problem = generate_problem(REVENUE_STANDARD, 800, feasible, 3)
        @test length(model[:scale]) == length(problem.products)
        for j in (1, length(problem.products))
            m = problem.products[j].market
            row = model[:scale][j]
            @test normalized_coefficient(row, model[:sales][j]) == 1.0
            @test normalized_coefficient(row, model[:no_purchase][m]) ≈
                -problem.attraction[j] / problem.no_purchase_attraction[m]
        end

        # Determinism and local RNG.
        _, first = generate_problem(REVENUE_STANDARD, 1500, infeasible, 12_345)
        _, second = generate_problem(REVENUE_STANDARD, 1500, infeasible, 12_345)
        function rm_stored_equal(a, b)
            typeof(a) == typeof(b) || return false
            if a === nothing || a isa Number || a isa Symbol || a isa Tuple || a isa AbstractString
                return isequal(a, b)
            elseif a isa AbstractArray
                return size(a) == size(b) && all(rm_stored_equal(x, y) for (x, y) in zip(a, b))
            end
            return all(
                rm_stored_equal(getfield(a, n), getfield(b, n)) for n in fieldnames(typeof(a))
            )
        end
        @test rm_stored_equal(first, second)
        Random.seed!(68_731)
        expected_first = rand()
        expected_second = rand()
        Random.seed!(68_731)
        @test rand() == expected_first
        generate_problem(REVENUE_STANDARD, 120, feasible, 99)
        @test rand() == expected_second

        # Schedules grow with the target: more days, fare classes, hubs.
        _, small = generate_problem(REVENUE_STANDARD, 500, feasible, 1)
        _, large = generate_problem(REVENUE_STANDARD, 50_000, feasible, 1)
        @test large.n_days > small.n_days
        @test large.n_hubs > small.n_hubs
        @test length(unique(p.fare_class for p in large.products)) == 5
        @test any(length(p.flights) == 2 for p in large.products)
    end

    @testset "standard witness and certificate arithmetic" begin
        for target in (50, 500, 4_000), seed in 0:5
            _, problem = generate_problem(REVENUE_STANDARD, target, feasible, seed)
            w = problem.feasible_witness
            @test w !== nothing && problem.infeasibility_certificate === nothing
            @test all(0 .< w.offer_fraction .<= 1)
            n_markets = length(problem.markets)
            sold = zeros(n_markets)
            for (j, product) in enumerate(problem.products)
                m = product.market
                sold[m] += w.sales[j]
                @test w.sales[j] >= 0
                @test w.sales[j] <=
                    problem.attraction[j] / problem.no_purchase_attraction[m] * w.no_purchase[m] +
                      1e-9
            end
            @test all(isapprox.(sold .+ w.no_purchase, problem.market_size; atol=1e-8))
            load = zeros(length(problem.flights))
            for (j, product) in enumerate(problem.products), f in product.flights
                load[f] += w.sales[j]
            end
            @test all(load .<= problem.capacity .+ 1e-8)
            @test all(load[f] >= problem.min_load[f] - 1e-8 for f in problem.contracted_flights)
        end

        for target in (50, 500, 4_000), seed in 0:5
            _, problem = generate_problem(REVENUE_STANDARD, target, infeasible, seed)
            c = problem.infeasibility_certificate
            @test c !== nothing && problem.feasible_witness === nothing
            @test c.flight in problem.contracted_flights
            @test c.min_load == problem.min_load[c.flight]
            by_market = Dict{Int, Float64}()
            for (j, product) in enumerate(problem.products)
                c.flight in product.flights || continue
                by_market[product.market] =
                    get(by_market, product.market, 0.0) + problem.attraction[j]
            end
            @test c.markets == sort!(collect(keys(by_market)))
            for (k, m) in enumerate(c.markets)
                V = by_market[m]
                @test c.market_bounds[k] ≈
                    problem.market_size[m] * V / (V + problem.no_purchase_attraction[m])
            end
            @test c.sellable_bound ≈ sum(c.market_bounds)
            @test c.margin ≈ c.min_load - c.sellable_bound
            @test c.margin >= 0.05 * c.sellable_bound - 1e-9
            # The contract fits the cabin, so no single row is contradictory.
            @test c.min_load <= problem.capacity[c.flight]
        end

        for seed in 0:5
            _, problem = generate_problem(REVENUE_STANDARD, 800, unknown, seed)
            @test problem.feasible_witness === nothing
            @test problem.infeasibility_certificate === nothing
            @test all(
                problem.min_load[f] <= problem.capacity[f] for f in problem.contracted_flights
            )
        end
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

        # `unknown` is a natural instance: nothing planted, nothing recorded.
        for seed in 0:15
            _, problem = generate_problem(REVENUE_OVERBOOKING, 500, unknown, seed)
            @test problem.resolved_status == unknown
            @test problem.feasible_witness === nothing
            @test problem.infeasibility_certificate === nothing
        end
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
                status in (feasible, infeasible), target in (50, 150, 500, 3000),
                seed in 0:5

                model, _ = generate_problem(reference, target, status, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                expected = status == feasible ? REVENUE_MOI.OPTIMAL : REVENUE_MOI.INFEASIBLE
                @test termination_status(model) == expected
            end
        end
        @testset "overbooking unknown is two-sided" begin
            outcomes = map(0:15) do seed
                model, _ = generate_problem(REVENUE_OVERBOOKING, 1000, unknown, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                termination_status(model)
            end
            @test count(==(REVENUE_MOI.OPTIMAL), outcomes) >= 2
            @test count(==(REVENUE_MOI.INFEASIBLE), outcomes) >= 2
        end
        @testset "standard unknown is two-sided" begin
            outcomes = map(0:11) do seed
                model, _ = generate_problem(REVENUE_STANDARD, 2000, unknown, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                termination_status(model)
            end
            @test count(==(REVENUE_MOI.OPTIMAL), outcomes) >= 2
            @test count(==(REVENUE_MOI.INFEASIBLE), outcomes) >= 2
            @test all(in((REVENUE_MOI.OPTIMAL, REVENUE_MOI.INFEASIBLE)), outcomes)
        end
    else
        @info "HiGHS unavailable; skipping revenue-management solve checks"
    end
end
