# Focused quality contracts for the supply_chain category: the network_planning
# variant's registry wiring, sparse sizing cap, planted plan / cumulative
# product cut, and disruption metadata, plus the HiGHS feasibility contracts for
# the standard and network_planning variants.
@testset "Supply-chain network planning" begin
    @test :network_planning in list_variants(:supply_chain)
    info = problem_info(:supply_chain, :network_planning)
    @test info[:variant] == :network_planning
    @test occursin("Multi-period", info[:description])

    # Truly tiny requests resolve to the smallest meaningful two-plant,
    # two-product profile instead of dropping either block.
    for seed in 0:2
        model, p = generate_problem("supply_chain/network_planning", 1, unknown, seed)
        @test p.n_plants == p.n_customers == p.n_products == 2
        @test length(p.shipment_arcs) == p.n_customers * p.n_products * p.n_periods
        @test num_variables(model) ==
            2 * p.n_plants * p.n_products * p.n_periods + length(p.shipment_arcs)
    end

    # A committed multi-target/multi-seed matrix guards sizing and
    # certificate arithmetic across every profile and status.
    for target in (50, 500, 5000), seed in 0:5, status in (feasible, infeasible, unknown)
        model, p = generate_problem("supply_chain/network_planning", target, status, seed)
        expected = 2 * p.n_plants * p.n_products * p.n_periods + length(p.shipment_arcs)
        @test num_variables(model) == expected
        @test abs(num_variables(model) - target) <= 0.25 * target
        @test p.n_products >= 2

        if status == infeasible
            cert = p.infeasibility_certificate
            k, tau = cert.product, cert.period
            demand = sum(p.demand[:, k, 1:tau])
            supply =
                sum(p.initial_inventory[:, k]) + sum(
                    min(
                        p.production_capacity[plant, k, period],
                        p.plant_capacity[plant, period] / p.resource_use[plant, k],
                    ) for plant in 1:p.n_plants, period in 1:tau
                )
            lanes = sum(p.lane_capacity[a] for a in p.shipment_arcs if a[3] == k && a[4] <= tau)
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
        P, C, K, T, A = SyntheticLPs._choose_network_planning_dimensions(maximum_target, profile)
        @test 2 * P * K * T + A == maximum_target
        @test C > 0
    end
    large_problem = SyntheticLPs.SupplyChainNetworkPlanningProblem(100_000, unknown, 0)
    @test 2 * large_problem.n_plants * large_problem.n_products * large_problem.n_periods +
          length(large_problem.shipment_arcs) == 100_000
    large_error = try
        generate_problem("supply_chain/network_planning", maximum_target + 1, unknown, 0)
        nothing
    catch err
        err
    end
    @test large_error isa ArgumentError
    @test occursin("supports target_variables <= 1000000", sprint(showerror, large_error))

    # Validate the stored constructive witness without a solver.
    for seed in 0:8
        _, p = generate_problem("supply_chain/network_planning", 240, feasible, seed)
        witness = p.feasible_witness
        @test witness !== nothing
        @test p.infeasibility_certificate === nothing
        @test p.nominal_scenario === nothing
        for plant in 1:p.n_plants, product in 1:p.n_products, period in 1:p.n_periods
            outbound = sum(
                (
                    witness.shipment[a] for
                    a in p.shipment_arcs if a[1] == plant && a[3] == product && a[4] == period
                );
                init=0.0,
            )
            previous = if period == 1
                p.initial_inventory[plant, product]
            else
                witness.inventory[plant, product, period - 1]
            end
            @test isapprox(
                previous + witness.production[plant, product, period] - outbound,
                witness.inventory[plant, product, period];
                atol=1e-8,
            )
            @test witness.production[plant, product, period] <=
                p.production_capacity[plant, product, period] + 1e-8
            @test witness.inventory[plant, product, period] <=
                p.inventory_capacity[plant, product] + 1e-8
        end
        for customer in 1:p.n_customers, product in 1:p.n_products, period in 1:p.n_periods
            delivered = sum(
                witness.shipment[a] for
                a in p.shipment_arcs if a[2] == customer && a[3] == product && a[4] == period
            )
            @test isapprox(delivered, p.demand[customer, product, period]; atol=1e-8)
        end
        for plant in 1:p.n_plants, period in 1:p.n_periods
            used = sum(
                p.resource_use[plant, product] * witness.production[plant, product, period] for
                product in 1:p.n_products
            )
            @test used <= p.plant_capacity[plant, period] + 1e-8
        end
        @test all(witness.shipment[a] <= p.lane_capacity[a] + 1e-8 for a in p.shipment_arcs)
        @test all(>=(0), witness.production)
        @test all(>=(0), witness.inventory)
        @test all(value >= -1e-12 for value in values(witness.shipment))
        @test Set(keys(witness.shipment)) == Set(p.shipment_arcs)
    end

    # Status-aware metadata has no ambiguous zero-valued witness,
    # certificate, or absence sentinels.
    for seed in 0:5
        _, pf = generate_problem("supply_chain/network_planning", 200, feasible, seed)
        _, pi = generate_problem("supply_chain/network_planning", 200, infeasible, seed)
        _, pu = generate_problem("supply_chain/network_planning", 200, unknown, seed)
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
        model, p = generate_problem("supply_chain/network_planning", 500, feasible, seed)
        push!(profiles, p.profile)
        @test length(p.shipment_arcs) == length(unique(p.shipment_arcs))
        @test Set(keys(p.shipment_cost)) == Set(p.shipment_arcs)
        @test Set(keys(p.lane_capacity)) == Set(p.shipment_arcs)
        @test collect(only(axes(model[:ship]))) == p.shipment_arcs
        @test length(p.shipment_arcs) < p.n_plants * p.n_customers * p.n_products * p.n_periods
        @test all(
            1 <= a[1] <= p.n_plants &&
                1 <= a[2] <= p.n_customers &&
                1 <= a[3] <= p.n_products &&
                1 <= a[4] <= p.n_periods for a in p.shipment_arcs
        )
        _, density_hi = SyntheticLPs._network_density(p.profile)
        max_degree = min(p.n_plants, max(2, ceil(Int, density_hi * p.n_plants)))
        for customer in 1:p.n_customers, product in 1:p.n_products, period in 1:p.n_periods
            degree = count(
                a -> a[2] == customer && a[3] == product && a[4] == period, p.shipment_arcs
            )
            period_max = if p.profile == :disruption && period == p.disruption.period
                min(max_degree, p.n_plants - 1)
            else
                max_degree
            end
            @test 1 <= degree <= period_max
        end
        @test all(maximum(p.specialization[:, k]) > 1.2 for k in 1:p.n_products)
    end
    @test profiles == Set([:regional_stable, :seasonal_prebuild, :disruption])

    # Profile labels correspond to materially different coefficients and
    # structure. Apply the contracts across several seeds of each profile.
    function check_regional_profile(p)
        @test p.profile == :regional_stable
        totals = [sum(p.demand[:, :, t]) for t in 1:p.n_periods]
        share =
            count(a -> p.plant_regions[a[1]] == p.customer_regions[a[2]], p.shipment_arcs) /
            length(p.shipment_arcs)
        @test maximum(totals) < 1.30 * minimum(totals)
        @test share > 0.55
    end
    function check_seasonal_profile(p)
        @test p.profile == :seasonal_prebuild
        totals = [sum(p.demand[:, :, t]) for t in 1:p.n_periods]
        @test maximum(totals) > 1.6 * minimum(totals)
        @test sum(p.production_cost[:, :, 1]) < sum(p.production_cost[:, :, end])
        prepeak = max(1, argmax(totals) - 1)
        @test sum(p.feasible_witness.inventory[:, :, prepeak]) > sum(p.initial_inventory)
    end
    function check_disruption_profile(p)
        @test p.profile == :disruption
        event = p.disruption
        @test event.production_factor == 0.35
        @test event.shipment_surcharge == 1.55
        @test all(!(a[1] == event.plant && a[4] == event.period) for a in p.shipment_arcs)
        @test p.plant_capacity[event.plant, event.period] <
            minimum(p.plant_capacity[event.plant, setdiff(1:p.n_periods, [event.period])])
        disruption_arcs = [a for a in p.shipment_arcs if a[4] == event.period]
        ordinary_arcs = [a for a in p.shipment_arcs if a[4] != event.period]
        @test sum(p.shipment_cost[a] for a in disruption_arcs) / length(disruption_arcs) >
            1.15 * sum(p.shipment_cost[a] for a in ordinary_arcs) / length(ordinary_arcs)
        @test all(
            any(a[2] == c && a[3] == k && a[4] == event.period for a in p.shipment_arcs) for
            c in 1:p.n_customers, k in 1:p.n_products
        )
    end
    for seed in 0:3:9
        _, p = generate_problem("supply_chain/network_planning", 500, feasible, seed)
        check_regional_profile(p)
    end
    for seed in 1:3:10
        _, p = generate_problem("supply_chain/network_planning", 500, feasible, seed)
        check_seasonal_profile(p)
    end
    for seed in 2:3:11
        _, p = generate_problem("supply_chain/network_planning", 500, feasible, seed)
        check_disruption_profile(p)
    end

    # Exact JuMP algebra, domains, bounds, sparse shipment axes, and
    # objective coefficients. Every named constraint family is checked in
    # full, including absent coefficients.
    algebra_model, algebra = generate_problem("supply_chain/network_planning", 120, feasible, 2)
    produce = algebra_model[:produce]
    inventory = algebra_model[:inventory]
    ship = algebra_model[:ship]
    balances = algebra_model[:inventory_balance]
    demands = algebra_model[:demand_balance]
    resources = algebra_model[:resource_capacity]
    @test length(balances) == algebra.n_plants * algebra.n_products * algebra.n_periods
    @test length(demands) == algebra.n_customers * algebra.n_products * algebra.n_periods
    @test length(resources) == algebra.n_plants * algebra.n_periods

    function expected_inventory_coefficient(var, plant, product, period)
        var == produce[plant, product, period] && return 1.0
        var == inventory[plant, product, period] && return -1.0
        if period > 1 && var == inventory[plant, product, period - 1]
            return 1.0
        end
        for arc in algebra.shipment_arcs
            arc[1] == plant && arc[3] == product && arc[4] == period || continue
            var == ship[arc] && return -1.0
        end
        return 0.0
    end
    for plant in 1:algebra.n_plants, product in 1:algebra.n_products, period in 1:algebra.n_periods
        row = balances[plant, product, period]
        @test normalized_rhs(row) ==
            (period == 1 ? -algebra.initial_inventory[plant, product] : 0.0)
        for var in all_variables(algebra_model)
            @test normalized_coefficient(row, var) ==
                expected_inventory_coefficient(var, plant, product, period)
        end
    end
    for customer in 1:algebra.n_customers,
        product in 1:algebra.n_products,
        period in 1:algebra.n_periods

        row = demands[customer, product, period]
        @test normalized_rhs(row) == algebra.demand[customer, product, period]
        for var in all_variables(algebra_model)
            expected = 0.0
            for arc in algebra.shipment_arcs
                arc[2] == customer && arc[3] == product && arc[4] == period || continue
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
    for p in 1:algebra.n_plants, k in 1:algebra.n_products, t in 1:algebra.n_periods
        x = produce[p, k, t]
        inv = inventory[p, k, t]
        @test !is_binary(x) && !is_integer(x)
        @test !is_binary(inv) && !is_integer(inv)
        @test lower_bound(x) == 0
        @test upper_bound(x) == algebra.production_capacity[p, k, t]
        @test lower_bound(inv) == 0
        @test upper_bound(inv) == algebra.inventory_capacity[p, k]
        @test coefficient(objective_function(algebra_model), x) == algebra.production_cost[p, k, t]
        @test coefficient(objective_function(algebra_model), inv) == algebra.holding_cost[p, k, t]
    end
    for arc in algebra.shipment_arcs
        x = ship[arc]
        @test !is_binary(x) && !is_integer(x)
        @test lower_bound(x) == 0
        @test upper_bound(x) == algebra.lane_capacity[arc]
        @test coefficient(objective_function(algebra_model), x) == algebra.shipment_cost[arc]
    end

    # Unknown samples use correlated network conditions and retain local
    # lane service; they do not expose a baseline plan as a feasible witness.
    unknown_supply_factors = Float64[]
    for target in (200, 5000), seed in 0:11
        _, p = generate_problem("supply_chain/network_planning", target, unknown, seed)
        scenario = p.nominal_scenario
        push!(unknown_supply_factors, scenario.supply_factor)
        @test 0.65 <= scenario.supply_factor <= 1.20
        @test 0.92 <= scenario.lane_factor <= 1.14
        @test scenario.minimum_local_service >= 1.03 - 1e-12
        @test all(
            sum(
                p.lane_capacity[a] for a in p.shipment_arcs if a[2] == c && a[3] == k && a[4] == t
            ) >= p.demand[c, k, t] * (1.03 - 1e-12) for
            c in 1:p.n_customers, k in 1:p.n_products, t in 1:p.n_periods
        )
    end
    @test minimum(unknown_supply_factors) < 0.80
    @test maximum(unknown_supply_factors) > 1.05

    # Local-RNG reproducibility compares every stored field for each profile
    # and status, plus repeated byte-identical MPS output.
    function stored_equal(a, b)
        typeof(a) == typeof(b) || return false
        if a === nothing ||
            a isa Number ||
            a isa Symbol ||
            a isa AbstractString ||
            a isa Tuple ||
            a isa AbstractArray ||
            a isa AbstractDict
            return isequal(a, b)
        end
        return all(
            stored_equal(getfield(a, name), getfield(b, name)) for name in fieldnames(typeof(a))
        )
    end
    Random.seed!(9182)
    expected_global_draw = rand()
    Random.seed!(9182)
    generate_problem("supply_chain/network_planning", 120, feasible, 7)
    @test rand() == expected_global_draw

    mktempdir() do dir
        for seed in 0:2, status in (feasible, infeasible, unknown)
            m1, p1 = generate_problem("supply_chain/network_planning", 360, status, seed)
            m2, p2 = generate_problem("supply_chain/network_planning", 360, status, seed)
            @test all(
                stored_equal(getfield(p1, name), getfield(p2, name)) for
                name in fieldnames(typeof(p1))
            )
            repeated = SyntheticLPs.build_model(p1)
            @test num_variables(m1) == num_variables(repeated)
            @test num_constraints(m1, count_variable_in_set_constraints=true) ==
                num_constraints(repeated, count_variable_in_set_constraints=true)
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
            m1, p1 = generate_problem("supply_chain/network_planning", 360, feasible, seed)
            m2, p2 = generate_problem("supply_chain/network_planning", 360, feasible, seed + 3)
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

@testset "Supply Chain Feasibility Contracts" begin
    if HAS_HIGHS
        # supply_chain/standard reserves fallback routes within its variable budget.
        # Capacity smoothing must use that same capped coverage count; otherwise
        # tiny requested-feasible instances can remain infeasible.
        for s in 1:10
            m, _ = generate_problem("supply_chain/standard", 50, feasible, s)
            @test num_variables(m) == 50
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            @test termination_status(m) == MOI.OPTIMAL
        end

        # The network-planning variant has a solver-independent planted plan for
        # feasible requests and a cumulative product cut for infeasible requests.
        for status in (feasible, infeasible), s in 0:5
            m, _ = generate_problem("supply_chain/network_planning", 240, status, s)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
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
            m, p = generate_problem("supply_chain/network_planning", target, unknown, s)
            for c in 1:p.n_customers, k in 1:p.n_products, t in 1:p.n_periods
                incoming_capacity = sum(
                    p.lane_capacity[a] for
                    a in p.shipment_arcs if a[2] == c && a[3] == k && a[4] == t
                )
                unknown_singleton_cuts += incoming_capacity + 1e-10 < p.demand[c, k, t]
            end
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            ts = termination_status(m)
            unknown_optimal += ts == MOI.OPTIMAL
            unknown_infeasible += ts in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
        end
        @test unknown_singleton_cuts == 0
        @test unknown_optimal >= 6
        @test unknown_infeasible >= 3
        @test unknown_optimal + unknown_infeasible == 36
    end
end
