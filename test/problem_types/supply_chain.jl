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

        if status == infeasible && p.infeasibility_certificate isa SyntheticLPs.NetworkPlanningResourceCertificate
            cert = p.infeasibility_certificate
            tau = cert.period
            @test cert.cumulative_demand ≈ [sum(p.demand[:, k, 1:tau]) for k in 1:p.n_products]
            @test cert.initial_stock ≈ vec(sum(p.initial_inventory; dims=1))
            @test cert.min_resource_use == [minimum(p.resource_use[:, k]) for k in 1:p.n_products]
            required = sum(
                cert.min_resource_use[k] * max(0.0, cert.cumulative_demand[k] - cert.initial_stock[k]) for
                k in 1:p.n_products
            )
            @test cert.required_resource ≈ required
            @test cert.available_resource ≈ sum(p.plant_capacity[:, 1:tau])
            @test cert.margin ≈ required - cert.available_resource
            @test cert.margin > 0.05 * required
        elseif status == infeasible
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
    # Regression guard for the 50k presolve collapse: the degree target is
    # absolute, so large instances keep several lanes per demand node instead
    # of a handful of plants feeding thousands of singleton demand rows.
    for target in (50_000, 100_000), profile in (:regional_stable, :seasonal_prebuild, :disruption)
        P, C, K, T, A = SyntheticLPs._choose_network_planning_dimensions(target, profile)
        @test 2 * P * K * T + A == target
        @test A / (C * K * T) >= first(SyntheticLPs._network_degree_range(profile)) - 0.05
        @test P >= 20
    end
    large_problem = SyntheticLPs.SupplyChainNetworkPlanningProblem(100_000, unknown, 0)
    node_degree = Dict{NTuple{3, Int}, Int}()
    for (_, c, k, t) in large_problem.shipment_arcs
        node_degree[(c, k, t)] = get(node_degree, (c, k, t), 0) + 1
    end
    @test length(node_degree) ==
        large_problem.n_customers * large_problem.n_products * large_problem.n_periods
    @test minimum(values(node_degree)) >= 2
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
        max_degree = SyntheticLPs._network_max_degree(p.profile, p.n_plants)
        min_degree = SyntheticLPs._network_min_degree(
            length(p.shipment_arcs), p.n_customers * p.n_products * p.n_periods, p.n_plants
        )
        for customer in 1:p.n_customers, product in 1:p.n_products, period in 1:p.n_periods
            degree = count(
                a -> a[2] == customer && a[3] == product && a[4] == period, p.shipment_arcs
            )
            period_max = if p.profile == :disruption && period == p.disruption.period
                min(max_degree, p.n_plants - 1)
            else
                max_degree
            end
            @test min(min_degree, period_max) <= degree <= period_max
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
        @test 0.66 <= scenario.supply_factor <= 1.14
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


# Shared helpers for the multi-echelon network variants (standard, carbon,
# multi_product): exact row formula and planted-plan evaluation.
function scn_expected_rows(net)
    T = net.n_periods
    ship_keys = SyntheticLPs._scn_ship_keys(net)
    line_rows = length(
        unique(
            (net.lanes[l][1], k, t) for
            (l, k, t) in ship_keys if isfinite(net.line_capacity[net.lanes[l][1], k])
        ),
    )
    lane_rows = count(isfinite, net.lane_capacity) * T
    mode_rows = count(mi -> all(isfinite, net.mode_capacity[mi, :]), eachindex(net.modes)) * T
    demand_rows = sum(length(net.customer_products[c]) for c in 1:net.n_customers) * T
    return net.n_plants * T + line_rows + lane_rows + mode_rows + net.n_dcs * net.n_products * T +
           2 * net.n_dcs * T + demand_rows + (net.design ? length(net.arcs) * T : 0)
end

function scn_witness_point(model, net, w)
    point = Dict{VariableRef, Float64}()
    if net.design
        for d in 1:net.n_dcs
            point[model[:open][d]] = w.open[d]
        end
    end
    for (i, v) in enumerate(model[:ship])
        point[v] = w.ship[i]
    end
    for (i, v) in enumerate(model[:deliver])
        point[v] = w.deliver[i]
    end
    for I in CartesianIndices(w.stock)
        point[model[:stock][I]] = w.stock[I]
    end
    return point
end

@testset "Supply-chain multi-echelon network variants" begin
    @test Set(list_variants(:supply_chain)) ==
        Set([:standard, :carbon, :multi_product, :network_planning, :single_source])
    @test problem_info(:supply_chain)[:default_variant] == :standard

    refs = ("supply_chain/standard", "supply_chain/carbon", "supply_chain/multi_product")

    # Exact variable and row formulas, size fidelity, and design structure.
    for ref in refs, target in (100, 1000, 6000), status in (feasible, infeasible, unknown), seed in 0:1
        model, p = generate_problem(ref, target, status, seed; relax_integer=false)
        net = p.network
        @test num_variables(model) == SyntheticLPs._scn_num_variables(net)
        extra_rows = ref == "supply_chain/carbon" ? net.n_periods : 0
        @test num_constraints(model; count_variable_in_set_constraints=false) ==
            scn_expected_rows(net) + extra_rows
        tolerance = target >= 1000 ? max(0.02 * target, net.n_products * net.n_periods) : 0.3 * target
        @test abs(num_variables(model) - target) <= tolerance
        @test net.design == (ref != "supply_chain/multi_product")
        @test count(is_binary, all_variables(model)) == (net.design ? net.n_dcs : 0)
        # Every customer reaches at least two DCs, every DC is supplied with
        # every product, and rows grow with the instance (no wide-thin LPs).
        @test all(count(a -> a[2] == c, net.arcs) >= min(2, net.n_dcs) for c in 1:net.n_customers)
        for d in 1:net.n_dcs, k in 1:net.n_products
            @test any(l[2] == d && k in net.plant_products[l[1]] for l in net.lanes)
        end
        if target >= 1000
            @test num_constraints(model; count_variable_in_set_constraints=false) >=
                0.3 * num_variables(model)
        end
        @test (p.feasible_witness !== nothing) == (status == feasible)
        @test (p.infeasibility_certificate !== nothing) == (status == infeasible)
    end

    # Large targets build quickly and land on the target.
    for ref in refs
        _, p = generate_problem(ref, 100_000, feasible, 0)
        @test abs(SyntheticLPs._scn_num_variables(p.network) - 100_000) <= 0.01 * 100_000
    end

    # Planted plans satisfy every row of the unrelaxed model (binary opens).
    for ref in refs, target in (60, 800, 4000), seed in 0:2
        model, p = generate_problem(ref, target, feasible, seed; relax_integer=false)
        w = p.feasible_witness
        @test all(x -> x == 0.0 || x == 1.0, w.open)
        point = scn_witness_point(model, p.network, w)
        @test isempty(primal_feasibility_report(model, point; atol=1e-7))
    end

    # Regional throughput certificate (standard).
    for seed in 0:5, target in (300, 3000)
        _, p = generate_problem("supply_chain/standard", target, infeasible, seed)
        net, cert = p.network, p.infeasibility_certificate
        @test cert.customers == findall(==(cert.region), net.customer_region)
        @test cert.dcs ==
            sort(unique(d for (d, c) in net.arcs if net.customer_region[c] == cert.region))
        demand = sum(
            net.demand[c, k, cert.period] for c in cert.customers for k in net.customer_products[c]
        )
        @test cert.demand ≈ demand
        @test cert.throughput ≈ sum(net.dc_throughput[d] for d in cert.dcs)
        @test cert.margin ≈ demand - cert.throughput
        @test cert.margin > 0.05 * demand
    end

    # Carbon lower bound recomputed independently.
    for seed in 0:5, target in (300, 3000)
        _, p = generate_problem("supply_chain/carbon", target, infeasible, seed)
        net, cert = p.network, p.infeasibility_certificate
        inbound = [
            minimum(p.lane_emission[l] for l in eachindex(net.lanes) if net.lanes[l][2] == d) for
            d in 1:net.n_dcs
        ]
        unit = [
            minimum(
                p.arc_emission[a] + inbound[net.arcs[a][1]] for
                a in eachindex(net.arcs) if net.arcs[a][2] == c
            ) for c in 1:net.n_customers
        ]
        bound =
            sum(
                net.demand[c, k, t] * unit[c] for c in 1:net.n_customers for
                k in net.customer_products[c] for t in 1:net.n_periods
            ) - sum(inbound[d] * net.initial_stock[d, k] for d in 1:net.n_dcs, k in 1:net.n_products)
        @test cert.lower_bound ≈ bound
        @test cert.budget ≈ sum(p.period_budget)
        @test p.carbon_budget ≈ cert.budget
        @test cert.margin ≈ bound - cert.budget
        @test cert.margin > 0.05 * bound
        @test all(>(0), p.period_budget)
    end
    # Feasible carbon budgets sit just above the planted plan's emissions.
    for seed in 0:3
        model, p = generate_problem("supply_chain/carbon", 1500, feasible, seed)
        @test p.plan_emissions <= p.carbon_budget <= 1.05 * p.plan_emissions + 1e-6
        @test length(model[:carbon_cap]) == p.network.n_periods
    end

    # Cumulative product-supply certificate (multi_product).
    # (1, 12): a tiny network where the first-drawn product had no orders.
    for (seed, target) in vcat(vec(collect(Iterators.product(0:5, (300, 3000)))), [(12, 1)])
        _, p = generate_problem("supply_chain/multi_product", target, infeasible, seed)
        net, cert = p.network, p.infeasibility_certificate
        k, tau = cert.product, cert.period
        @test cert.plants == [q for q in 1:net.n_plants if k in net.plant_products[q]]
        @test cert.demand ≈ sum(net.demand[:, k, 1:tau])
        @test cert.initial_stock ≈ sum(net.initial_stock[:, k])
        bound =
            cert.initial_stock + sum(
                min(net.line_capacity[q, k], net.plant_capacity[q, t] / net.resource_use[q, k]) for
                q in cert.plants, t in 1:tau
            )
        @test cert.supply_bound ≈ bound
        @test cert.margin ≈ cert.demand - bound
        @test cert.demand > 0
        @test cert.margin > 0.05 * cert.demand
    end

    # Unknown requests scale capacities by one recorded network-wide factor.
    for ref in refs, seed in 0:5
        _, p = generate_problem(ref, 800, unknown, seed)
        @test 0.60 <= p.capacity_factor <= 1.05
        @test p.feasible_witness === nothing && p.infeasibility_certificate === nothing
    end

    # Reproducibility of every stored datum.
    for ref in refs
        _, a = generate_problem(ref, 900, unknown, 11)
        _, b = generate_problem(ref, 900, unknown, 11)
        for name in fieldnames(typeof(a.network))
            @test isequal(getfield(a.network, name), getfield(b.network, name))
        end
    end
end

@testset "Supply Chain Feasibility Contracts" begin
    if HAS_HIGHS
        function sc_status(m)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            return termination_status(m)
        end
        for ref in ("supply_chain/standard", "supply_chain/carbon", "supply_chain/multi_product")
            for target in (300, 2000), seed in 0:2
                m, _ = generate_problem(ref, target, feasible, seed)
                @test sc_status(m) == MOI.OPTIMAL
                m, _ = generate_problem(ref, target, infeasible, seed)
                @test sc_status(m) == MOI.INFEASIBLE
            end
            # Unknown is two-sided over a seed block.
            outcomes = [sc_status(first(generate_problem(ref, 1000, unknown, s))) for s in 0:11]
            @test count(==(MOI.OPTIMAL), outcomes) >= 2
            @test count(==(MOI.INFEASIBLE), outcomes) >= 1
            @test all(in((MOI.OPTIMAL, MOI.INFEASIBLE)), outcomes)
        end

        # The network-planning variant has a solver-independent planted plan for
        # feasible requests and a resource or product cut for infeasible requests.
        for status in (feasible, infeasible), s in 0:5
            m, _ = generate_problem("supply_chain/network_planning", 240, status, s)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test sc_status(m) == expected
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
            ts = sc_status(m)
            unknown_optimal += ts == MOI.OPTIMAL
            unknown_infeasible += ts in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
        end
        @test unknown_singleton_cuts == 0
        @test unknown_optimal >= 6
        @test unknown_infeasible >= 3
        @test unknown_optimal + unknown_infeasible == 36
    end
end
