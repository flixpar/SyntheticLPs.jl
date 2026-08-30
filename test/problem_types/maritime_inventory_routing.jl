# Focused quality contracts for the maritime_inventory_routing category:
# registry shape, the exact closed-form variable count, sizing accuracy, the
# sailing-network conventions, witness and certificate arithmetic,
# reproducibility, and HiGHS feasibility contracts on both the LP relaxation
# and the unrelaxed integer model.
@testset "Maritime Inventory Routing" begin
    @test :maritime_inventory_routing in list_categories()
    @test Set(list_variants(:maritime_inventory_routing)) == Set([:standard])
    info = problem_info(:maritime_inventory_routing)
    @test info[:default_variant] == :standard
    @test occursin("maritime", lowercase(info[:description]))

    # Exact variable-count formula, straight from the struct's fields:
    # vessel positions, sailing legs, deliveries, pickups, onboard loads and
    # customer tank levels.
    mirp_variables(p) =
        p.n_vessels * p.n_ports * (p.n_periods + 1) +
        p.n_vessels * length(p.arcs) * p.n_periods +
        p.n_vessels * p.n_customers * p.n_periods +
        p.n_vessels * p.n_periods +
        p.n_vessels * (p.n_periods + 1) +
        p.n_customers * (p.n_periods + 1)

    for target in (60, 240, 1500, 9000), status in (feasible, infeasible, unknown)
        m, p = generate_problem(:maritime_inventory_routing, target, status, 3)
        @test num_variables(m) == mirp_variables(p)
        @test p.n_customers == p.n_ports - 1
        @test p.feasibility_status == status
    end

    # The binary block is exactly the routing decisions.
    m, p = generate_problem(:maritime_inventory_routing, 600, unknown, 1;
                            relax_integer=false)
    @test num_variables(m) == mirp_variables(p)
    @test num_constraints(m, VariableRef, MOI.ZeroOne) ==
          p.n_vessels * p.n_ports * (p.n_periods + 1) +
          p.n_vessels * length(p.arcs) * p.n_periods

    # Sizing: the arc count is solved from the target, so every request lands
    # within 10% across four decades (the old dimension search saturated at
    # ~15.6k variables and missed 20000 by 22%, and missed 50 by 20%).
    for target in (50, 200, 1000, 4000, 20000),
        status in (feasible, infeasible, unknown), seed in 0:3
        m, _ = generate_problem(:maritime_inventory_routing, target, status, seed)
        @test abs(num_variables(m) - target) <= 0.10 * target
    end
    # Realised sizes are monotone in the target.
    sizes = [num_variables(generate_problem(:maritime_inventory_routing,
                                            target, unknown, 7)[1])
             for target in (50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 20000)]
    @test issorted(sizes)

    # Sailing-network conventions: waiting arcs everywhere, both depot shuttle
    # legs per customer, no duplicates, and every leg sailable within a period.
    for target in (120, 900, 5000), status in (feasible, infeasible, unknown)
        _, p = generate_problem(:maritime_inventory_routing, target, status, 5)
        P = p.n_ports
        @test length(p.arcs) == length(unique(p.arcs))
        @test 3P - 2 <= length(p.arcs) <= P^2
        @test all((i, i) in p.arcs for i in 1:P)
        @test all((1, c) in p.arcs && (c, 1) in p.arcs for c in 2:P)
        @test p.travel_time == p.travel_time'
        @test all(p.travel_time[i, i] == 0.0 for i in 1:P)
        @test p.period_length == maximum(p.travel_time[i, j] for (i, j) in p.arcs)
        @test all(p.travel_time[i, j] <= p.period_length for (i, j) in p.arcs)
        @test all(p.initial_inventory[c] <= p.inventory_capacity[c]
                  for c in 1:p.n_customers)
        @test all(p.initial_load[v] <= p.vessel_capacity[v]
                  for v in 1:p.n_vessels)
        # Every customer can be reached inside the horizon by the rotation.
        @test p.n_customers <= p.n_vessels * (p.n_periods ÷ 2)
    end

    # Feasible witness: re-verify the planted schedule by direct arithmetic
    # against every structural requirement of the model.
    for target in (120, 800, 4000), seed in 0:2
        _, p = generate_problem(:maritime_inventory_routing, target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing
        @test p.infeasibility_certificate === nothing
        V, C, T = p.n_vessels, p.n_customers, p.n_periods
        @test size(w.position) == (V, T + 1)
        @test all(w.position[v, 1] == 1 for v in 1:V)

        for v in 1:V
            # Routes only use legs of the sailing network, so every move fits
            # inside one period.
            for t in 1:T
                leg = (w.position[v, t], w.position[v, t + 1])
                @test leg in p.arcs
                @test p.travel_time[leg[1], leg[2]] <= p.period_length
            end
            # Cargo only moves where the vessel actually is, and never more
            # than the vessel can hold.
            for t in 1:T
                @test w.pickup[v, t] == 0.0 || w.position[v, t + 1] == 1
                @test w.pickup[v, t] <= p.vessel_capacity[v]
                for c in 1:C
                    @test w.delivery[v, c, t] == 0.0 ||
                          w.position[v, t + 1] == c + 1
                    @test w.delivery[v, c, t] <= p.vessel_capacity[v]
                end
            end
            # Onboard load balance and bounds.
            @test w.load[v, 1] == p.initial_load[v]
            for t in 1:T
                @test isapprox(w.load[v, t + 1],
                               w.load[v, t] + w.pickup[v, t] -
                               sum(w.delivery[v, c, t] for c in 1:C); atol=1e-6)
            end
            @test all(-1e-9 <= w.load[v, k] <= p.vessel_capacity[v] + 1e-9
                      for k in 1:(T + 1))
        end

        # The depot never releases more than it has.
        for t in 1:T
            @test sum(w.pickup[v, t] for v in 1:V) <= p.depot_supply[t] + 1e-9
        end

        # Tank balance and tank bounds at every port and period.
        for c in 1:C
            @test w.inventory[c, 1] == p.initial_inventory[c]
            for t in 1:T
                @test isapprox(w.inventory[c, t + 1],
                               w.inventory[c, t] +
                               sum(w.delivery[v, c, t] for v in 1:V) -
                               p.consumption[c, t]; atol=1e-6)
            end
            @test all(-1e-9 <= w.inventory[c, k] <= p.inventory_capacity[c] + 1e-9
                      for k in 1:(T + 1))
        end
    end

    # The witness is a primal point of the unrelaxed model: map it onto the
    # variables and let JuMP re-check every row, bound and integrality set.
    for target in (150, 900), seed in 0:1
        model, p = generate_problem(:maritime_inventory_routing, target,
                                    feasible, seed; relax_integer=false)
        w = p.feasible_witness
        V, C, T, P = p.n_vessels, p.n_customers, p.n_periods, p.n_ports
        arc_index = Dict(a => k for (k, a) in enumerate(p.arcs))
        point = Dict{VariableRef,Float64}()
        for v in 1:V, t in 0:T, q in 1:P
            point[model[:location][v, q, t]] = w.position[v, t + 1] == q ? 1.0 : 0.0
        end
        for v in 1:V, t in 1:T
            used = arc_index[(w.position[v, t], w.position[v, t + 1])]
            for a in 1:length(p.arcs)
                point[model[:move][v, a, t]] = a == used ? 1.0 : 0.0
            end
            point[model[:pickup][v, t]] = w.pickup[v, t]
            for c in 1:C
                point[model[:delivery][v, c, t]] = w.delivery[v, c, t]
            end
        end
        for v in 1:V, t in 0:T
            point[model[:load][v, t]] = w.load[v, t + 1]
        end
        for c in 1:C, t in 0:T
            point[model[:inventory][c, t]] = w.inventory[c, t + 1]
        end
        @test isempty(primal_feasibility_report(model, point; atol=1e-7))
    end

    # Aggregate material certificate: recompute both bounds exactly and check
    # the shortage. Both bounds use only linear rows, so the refutation also
    # applies to the LP relaxation.
    for target in (120, 800, 4000), seed in 0:2
        _, q = generate_problem(:maritime_inventory_routing, target,
                                infeasible, seed)
        cert = q.infeasibility_certificate
        @test cert !== nothing
        @test q.feasible_witness === nothing
        H = cert.horizon
        @test 1 <= H <= q.n_periods
        @test isapprox(cert.consumption, sum(q.consumption[:, 1:H]); atol=1e-6)
        @test isapprox(cert.initial_inventory, sum(q.initial_inventory); atol=1e-6)
        @test isapprox(cert.initial_load, sum(q.initial_load); atol=1e-6)
        @test isapprox(cert.depot_supply, sum(q.depot_supply[1:H]); atol=1e-6)
        @test isapprox(cert.supply_bound,
                       cert.initial_inventory + cert.initial_load +
                       cert.depot_supply; atol=1e-6)
        @test isapprox(cert.throughput_bound,
                       cert.initial_inventory +
                       sum((q.initial_load[v] + H * q.vessel_capacity[v]) / 2
                           for v in 1:q.n_vessels); atol=1e-6)
        @test cert.deliverable == min(cert.supply_bound, cert.throughput_bound)
        @test cert.deliverable < cert.consumption
        # The starving is structural: the whole horizon is short too.
        @test sum(q.initial_inventory) + sum(q.initial_load) +
              sum(q.depot_supply) < sum(q.consumption)
    end

    # Reproducibility and global-RNG isolation.
    Random.seed!(987)
    _, p1 = generate_problem(:maritime_inventory_routing, 220, unknown, 42)
    Random.seed!(12345)
    _, p2 = generate_problem(:maritime_inventory_routing, 220, unknown, 42)
    @test all(isequal(getfield(p1, f), getfield(p2, f))
              for f in fieldnames(typeof(p1)))

    if HAS_HIGHS
        # The feasibility contract holds end-to-end on the LP relaxation.
        for target in (150, 700, 2500), status in (feasible, infeasible), s in 0:4
            m, _ = generate_problem(:maritime_inventory_routing, target, status, s)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end

        # ... and on the unrelaxed integer model, which is what catches a
        # witness that only works fractionally.
        for target in (150, 700), status in (feasible, infeasible), s in 0:2
            m, _ = generate_problem(:maritime_inventory_routing, target, status,
                                    s; relax_integer=false)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            set_time_limit_sec(m, 60.0)
            optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end

        # Unknown is a genuine mix at every scale, not an implicit branch.
        optimal = 0
        infeasible_count = 0
        for target in (150, 700, 2500), s in 0:9
            m, _ = generate_problem(:maritime_inventory_routing, target, unknown, s)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            if termination_status(m) == MOI.OPTIMAL
                optimal += 1
            elseif termination_status(m) == MOI.INFEASIBLE
                infeasible_count += 1
            end
        end
        @test optimal > 0
        @test infeasible_count > 0
    end
end
