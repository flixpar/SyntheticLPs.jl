# Focused quality contracts for the telecom_network_design category: registry
# shape, the exact variable-count formula and cliff-free sizing, the joint
# calibration of topology/demand/capacity/budget, exact witness and certificate
# arithmetic, reproducibility, and HiGHS feasibility contracts on both the LP
# relaxation and the unrelaxed integer model.
@testset "Telecom Network Design" begin
    TND = "telecom_network_design/standard"
    link(i, j) = i < j ? (i, j) : (j, i)

    @test :telecom_network_design in list_categories()
    @test list_variants(:telecom_network_design) == [:standard]
    info = problem_info(:telecom_network_design)
    @test info[:default_variant] == :standard
    @test occursin("network", lowercase(info[:description]))

    # Exact variable count: one binary per physical link plus one flow variable
    # per (commodity, directed arc) pair.
    for status in (feasible, infeasible, unknown), seed in 0:2
        m, p = generate_problem(TND, 600, status, seed)
        @test num_variables(m) == p.n_arcs * (2 * p.n_commodities + 1)
        @test length(p.arcs) == p.n_arcs
        @test length(p.directed_arcs) == 2 * p.n_arcs
    end

    # Sizing: a fine logarithmic sweep must track the target closely, with no
    # plateaus. Before the joint recalibration whole bands collapsed onto a
    # single realised size (every target in 101:290 produced 315 variables,
    # a 57% error at target 200), so the 5% bound below would have failed.
    sweep = unique(round.(Int, exp.(range(log(50), log(20000), length=25))))
    for target in sweep, seed in 0:1
        m, _ = generate_problem(TND, target, unknown, seed)
        @test abs(num_variables(m) - target) <= 0.05 * target
    end
    for target in (50, 200, 1000, 4000, 20000),
        status in (feasible, infeasible, unknown),
        seed in 0:3

        m, _ = generate_problem(TND, target, status, seed)
        @test abs(num_variables(m) - target) <= 0.05 * target
    end

    # Documented size cap (same explicit-error convention as
    # supply_chain/network_planning) rather than silently undersizing.
    @test_throws ArgumentError generate_problem(
        TND, SyntheticLPs.TELECOM_MAX_VARIABLES + 1, unknown, 0
    )

    # Topology and traffic data contracts.
    for status in (feasible, infeasible, unknown)
        _, p = generate_problem(TND, 800, status, 5)
        @test p.arcs == sort(unique(p.arcs))
        @test all(i < j for (i, j) in p.arcs)
        @test all(1 <= i && j <= p.n_nodes for (i, j) in p.arcs)
        @test Set(p.directed_arcs) ==
            Set(vcat([(i, j) for (i, j) in p.arcs], [(j, i) for (i, j) in p.arcs]))
        @test Set(keys(p.link_capacities)) == Set(p.arcs)
        @test all(p.distances[(i, j)] == p.distances[(j, i)] for (i, j) in p.arcs)
        @test all(p.flow_costs[(i, j)] == p.flow_costs[(j, i)] for (i, j) in p.arcs)
        @test all(
            cap in (155.0, 622.0, 2488.0, 9953.0, 39813.0) for cap in values(p.link_capacities)
        )  # SONET/OTN ladder
        @test all(c[:source] != c[:sink] for c in p.commodities)
        @test all(c[:demand] > 0 for c in p.commodities)
        @test length(p.commodities) == p.n_commodities
        @test p.total_demand ≈ sum(c[:demand] for c in p.commodities)
        # Adjacency bookkeeping used by the flow-conservation rows.
        @test sum(length(v) for v in values(p.outgoing_arcs)) == 2 * p.n_arcs
        @test all(all(a[1] == node for a in p.outgoing_arcs[node]) for node in 1:p.n_nodes)
        @test all(all(a[2] == node for a in p.incoming_arcs[node]) for node in 1:p.n_nodes)
        # The topology is connected (an MST seeds it), so every node is usable.
        seen = Set([1])
        frontier = [1]
        neighbours = Dict(v => Int[] for v in 1:p.n_nodes)
        for (i, j) in p.arcs
            push!(neighbours[i], j)
            push!(neighbours[j], i)
        end
        while !isempty(frontier)
            v = pop!(frontier)
            for w in neighbours[v]
                if !(w in seen)
                    push!(seen, w)
                    push!(frontier, w)
                end
            end
        end
        @test length(seen) == p.n_nodes
        # The planted routing lower-bounds the cut bound by construction.
        @test p.routable_scale <= p.cut_bound_scale + 1e-9
        @test p.nominal_cost > 0
    end

    # Feasible witness: an exact feasible point of the model. Every commodity is
    # fully routed over installed links, no link load exceeds its capacity, and
    # the design's spend is inside the budget.
    for target in (60, 500, 4000), seed in 0:3
        _, p = generate_problem(TND, target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing
        @test p.infeasibility_certificate === nothing
        # Demand is calibrated against the planted routing's capacity.
        @test p.total_demand <= 0.91 * p.routable_scale
        installed = Set(w.installed_links)
        @test issubset(installed, Set(p.arcs))
        loads = Dict{Tuple{Int, Int}, Float64}()
        for k in 1:p.n_commodities
            c = p.commodities[k]
            routed = 0.0
            for (nodes, flow) in w.routes[k]
                @test nodes[1] == c[:source]
                @test nodes[end] == c[:sink]
                @test flow > 0
                routed += flow
                for t in 1:(length(nodes) - 1)
                    a = link(nodes[t], nodes[t + 1])
                    @test a in installed
                    loads[a] = get(loads, a, 0.0) + flow
                end
            end
            @test routed ≈ c[:demand]
        end
        @test all(loads[a] <= p.link_capacities[a] + 1e-9 for a in keys(loads))
        @test all(loads[a] ≈ w.link_loads[a] for a in keys(loads))
        @test w.installation_cost ≈ sum(p.installation_costs[a] for a in w.installed_links)
        @test w.installation_cost <= p.budget
    end

    # Infeasibility certificates: both modes must appear, and both must
    # recompute exactly from the struct fields. Each argument only uses
    # 0 <= y <= 1, so it also refutes the LP relaxation.
    capacity_mode = 0
    budget_mode = 0
    for target in (60, 500, 4000), seed in 0:5
        _, p = generate_problem(TND, target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing
        @test p.feasible_witness === nothing
        side = Set(cert.side)
        @test !isempty(side) && length(side) < p.n_nodes
        crossing = [a for a in p.arcs if (a[1] in side) != (a[2] in side)]
        @test Set(cert.crossing_links) == Set(crossing)
        @test cert.crossing_demand ≈ sum(
            c[:demand] for c in p.commodities if (c[:source] in side) != (c[:sink] in side);
            init=0.0,
        )
        if cert isa SyntheticLPs.TelecomCapacityCutCertificate
            capacity_mode += 1
            @test cert.crossing_capacity ≈ sum(p.link_capacities[a] for a in crossing)
            @test cert.crossing_capacity < cert.crossing_demand
        else
            budget_mode += 1
            @test cert.cost_per_capacity ≈
                minimum(p.installation_costs[a] / p.link_capacities[a] for a in crossing)
            @test cert.implied_minimum ≈ cert.crossing_demand * cert.cost_per_capacity
            @test cert.budget == p.budget
            @test cert.budget < cert.implied_minimum
        end
    end
    @test capacity_mode > 0
    @test budget_mode > 0

    # Reproducibility and global-RNG isolation: identical seeds produce
    # field-identical structs even with a seeded/dirty global RNG.
    Random.seed!(987)
    _, p1 = generate_problem(TND, 777, unknown, 42)
    Random.seed!(12345)
    _, p2 = generate_problem(TND, 777, unknown, 42)
    @test all(isequal(getfield(p1, f), getfield(p2, f)) for f in fieldnames(typeof(p1)))

    if HAS_HIGHS
        # The feasibility contract holds end-to-end on the LP relaxation.
        for target in (120, 600, 3000), status in (feasible, infeasible), seed in 0:3
            m, _ = generate_problem(TND, target, status, seed)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            @test termination_status(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
        end

        # ... and on the unrelaxed integer model: the planted design is an
        # integral point, and both certificates are relaxation-proof.
        for status in (feasible, infeasible), seed in 0:1
            m, _ = generate_problem(TND, 300, status, seed; relax_integer=false)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            set_time_limit_sec(m, 120.0)
            optimize!(m)
            @test termination_status(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
        end

        # `unknown` is a genuine mix at every scale - the drift that made it
        # almost always feasible when small and almost always infeasible when
        # large is what the joint calibration removes.
        for target in (100, 1000, 5000)
            optimal = 0
            infeasible_count = 0
            for seed in 0:9
                m, _ = generate_problem(TND, target, unknown, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                if termination_status(m) == MOI.OPTIMAL
                    optimal += 1
                elseif termination_status(m) == MOI.INFEASIBLE
                    infeasible_count += 1
                end
            end
            @test optimal >= 3
            @test infeasible_count >= 3
        end
    end
end
