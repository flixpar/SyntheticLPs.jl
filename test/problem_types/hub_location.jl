# Focused quality contracts for the hub_location category: registry shape,
# exact variable-count formulas, sizing, benchmark data conventions, witness
# and certificate arithmetic, reproducibility, and HiGHS feasibility contracts
# on both the LP relaxation and the unrelaxed integer models.
@testset "Hub Location" begin
    @test :hub_location in list_categories()
    @test Set(list_variants(:hub_location)) == Set([
        :p_hub_median,
        :compact_single_allocation,
        :r_allocation,
        :multiple_allocation,
        :capacitated,
        :hub_covering,
        :hub_network,
        :budgeted_backbone,
    ])
    info = problem_info(:hub_location)
    @test info[:default_variant] == :p_hub_median
    @test occursin("hub", lowercase(info[:description]))

    # Exact variable-count formulas, straight from each struct's fields.
    path_count(A) =
        sum(length(A[i]) * length(A[j]) for i in 1:length(A) for j in (i + 1):length(A)) +
        sum(length(a) for a in A) +
        length(union(A...))
    for v in (:p_hub_median, :r_allocation), status in (feasible, infeasible)
        m, p = generate_problem(ProblemVariant(:hub_location, v), 120, status, 3)
        @test num_variables(m) == path_count(p.admissible)
    end
    for status in (feasible, infeasible)
        m, p = generate_problem("hub_location/compact_single_allocation", 120, status, 3)
        @test num_variables(m) == p.n_nodes^3
        m, p = generate_problem("hub_location/multiple_allocation", 120, status, 3)
        n, h, A = p.n_nodes, length(p.hubs), p.admissible
        @test num_variables(m) ==
            sum(
            sum(length(A[i]) for i in 1:n if i != j) + h * (h - 1) + length(A[j]) for j in 1:n
        ) + h
        m, p = generate_problem("hub_location/capacitated", 120, status, 3)
        n, h = p.n_nodes, length(p.hubs)
        @test num_variables(m) == n * h * (n + h - 1) + h * (n + 1)
        m, p = generate_problem("hub_location/hub_network", 120, status, 3)
        n, h, A, L = p.n_nodes, length(p.hubs), p.admissible, length(p.links)
        @test num_variables(m) ==
            sum(sum(length(A[i]) for i in 1:n if i != j) + 2L + length(A[j]) for j in 1:n) +
              sum(length(A[i]) for i in 1:n) +
              h +
              L
        m, p = generate_problem("hub_location/hub_covering", 120, status, 3)
        @test num_variables(m) ==
            p.n_nodes + sum(length(paths) for paths in values(p.covering_sets))
        m, p = generate_problem("hub_location/budgeted_backbone", 120, status, 3)
        n, h = p.n_nodes, length(p.hubs)
        @test num_variables(m) == n * h^2 + h + h * (h - 1) ÷ 2
    end

    # Sizing: every variant lands within 25% of the target (or <= 50 vars)
    # across a committed target/status matrix.
    for v in list_variants(:hub_location),
        target in (50, 200, 1000, 4000), status in (feasible, infeasible, unknown),
        seed in 0:3

        m, _ = generate_problem(ProblemVariant(:hub_location, v), target, status, seed)
        nv = num_variables(m)
        @test abs(nv - target) <= 0.25 * target || nv <= 50
    end

    # Reach-window data contracts for the path-flow variants: every node
    # can reach itself, admissible lists respect the reach window, and
    # airline conventions hold (symmetric CAB-style data, chi = delta = 1).
    for v in (:p_hub_median, :r_allocation)
        _, p = generate_problem(ProblemVariant(:hub_location, v), 150, unknown, 2)
        @test all(i in p.admissible[i] for i in 1:p.n_nodes)
        @test all(all(p.dist[i, k] <= p.reach + 1e-9 for k in p.admissible[i]) for i in 1:p.n_nodes)
        @test p.flow == p.flow'
        @test p.cost == p.cost'
        @test p.chi == p.delta == 1.0
        @test 0.2 <= p.alpha <= 0.8
        @test 2 <= p.p <= 8
    end

    # p-hub median witness: the planted assignment is admissible.
    _, p = generate_problem("hub_location/p_hub_median", 150, feasible, 0)
    w = p.feasible_witness
    @test w !== nothing
    @test length(w.hubs) == p.p
    @test all(w.assignment[i] in w.hubs for i in 1:p.n_nodes)
    @test all(p.dist[i, w.assignment[i]] <= p.reach for i in 1:p.n_nodes)
    @test all(w.assignment[k] == k for k in w.hubs)

    # Compact single allocation retains the complete directed OD matrix
    # in an origin-indexed flow formulation, and uses the classical
    # diagonal assignment/opening convention.
    for s in 0:2
        _, p = generate_problem("hub_location/compact_single_allocation", 150, feasible, s)
        w = p.feasible_witness
        @test w !== nothing
        @test p.profile in (:passenger, :freight, :telecom)
        @test length(w.hubs) == p.p
        @test all(w.assignment[k] == k for k in w.hubs)
        @test all(w.assignment[i] in w.hubs for i in 1:p.n_nodes)
        @test p.outvolume == vec(sum(p.flow; dims=2))
        @test p.involume == vec(sum(p.flow; dims=1))

        # The reach window admits every witness allocation.
        @test all(p.dist[i, w.assignment[i]] <= p.reach for i in 1:p.n_nodes)

        # Infeasible: p + 1 island regions; every node's reach window stays in
        # its own region and holds at least two candidates (no single-candidate
        # window that presolve could force open).
        _, q = generate_problem("hub_location/compact_single_allocation", 1000, infeasible, s)
        cert = q.infeasibility_certificate
        @test cert !== nothing
        @test q.feasible_witness === nothing
        @test length(cert.groups) == q.p + 1
        @test sort(vcat(cert.groups...)) == collect(1:q.n_nodes)
        region = Dict(v => g for (g, members) in enumerate(cert.groups) for v in members)
        for i in 1:q.n_nodes
            window = [k for k in 1:q.n_nodes if q.dist[i, k] <= q.reach]
            @test all(region[k] == region[i] for k in window)
            @test length(window) >= 2
        end
    end

    # Disjoint-region certificates: pairwise disjoint admissible sets, every
    # window nonempty. p_hub_median needs one hub per group (p + 1 groups);
    # r_allocation needs r hubs per group (floor(p/r) + 1 groups), and every
    # window holds more than r candidates so no hub is forced by bounds.
    group_admissibles(q, g) = union(q.admissible[i] for i in g)
    for v in (:p_hub_median, :r_allocation), s in 0:2
        target = v == :r_allocation ? 1000 : 150
        _, p = generate_problem(ProblemVariant(:hub_location, v), target, infeasible, s)
        cert = p.infeasibility_certificate
        @test cert !== nothing
        @test p.feasible_witness === nothing
        if v == :p_hub_median
            @test length(cert.groups) == p.p + 1
        else
            @test cert.r == p.r && cert.p == p.p
            @test length(cert.groups) == fld(p.p, p.r) + 1
            @test length(cert.groups) * p.r > p.p
            @test all(length(p.admissible[i]) > p.r for i in 1:p.n_nodes)
        end
        @test all(!isempty(g) for g in cert.groups)
        @test sort(vcat(cert.groups...)) == collect(1:p.n_nodes)
        for a in 1:length(cert.groups), b in (a + 1):length(cert.groups)
            @test isempty(
                intersect(
                    group_admissibles(p, cert.groups[a]), group_admissibles(p, cert.groups[b])
                ),
            )
        end
        @test all(!isempty(p.admissible[i]) for i in 1:p.n_nodes)
    end

    # r-allocation witness: r distinct admissible hubs per node, p distinct
    # open hubs. Cover both placement branches: the small target goes
    # through the exhaustive subset search, the large one through
    # farthest-first (which must never re-pick a selected hub).
    for target in (150, 20000)
        _, p = generate_problem("hub_location/r_allocation", target, feasible, 1)
        w = p.feasible_witness
        @test w !== nothing
        @test length(unique(w.hubs)) == p.p
        @test all(length(unique(a)) == p.r for a in w.assignments)
        @test all(all(k in p.admissible[i] for k in w.assignments[i]) for i in 1:p.n_nodes)
        @test all(all(k in w.hubs for k in w.assignments[i]) for i in 1:p.n_nodes)
        @test all(k in w.assignments[k] for k in w.hubs)
    end

    # AP postal conventions for the flow variants.
    for v in (:multiple_allocation, :capacitated, :hub_network)
        _, p = generate_problem(ProblemVariant(:hub_location, v), 150, unknown, 4)
        @test p.hubs == sort(p.hubs)
        @test issetequal(p.hubs, collect(p.hubs))
        @test p.flow != p.flow'          # asymmetric volumes
        @test all(p.flow[i, i] == 0 for i in 1:p.n_nodes)
        if v != :hub_network
            @test 2.7 <= p.chi <= 3.3
            @test 1.8 <= p.delta <= 2.2
            @test 0.7 <= p.alpha <= 0.8
        else
            @test 1.0 <= p.chi <= 2.5
            @test 0.05 <= p.alpha <= 0.4
        end
    end
    _, p = generate_problem("hub_location/capacitated", 150, unknown, 4)
    @test p.profile in (:loose, :tight)

    # Multiple allocation: feasible cover witness and budget arithmetic.
    _, p = generate_problem("hub_location/multiple_allocation", 150, feasible, 0)
    w = p.feasible_witness
    @test w !== nothing
    position = Dict(k => t for (t, k) in enumerate(p.hubs))
    @test all(any(k in p.admissible[i] for k in w.open_hubs) for i in 1:p.n_nodes)
    @test p.budget >= sum(p.fixed_cost[position[k]] for k in w.open_hubs)
    for s in 0:2
        _, q = generate_problem("hub_location/multiple_allocation", 150, infeasible, s)
        cert = q.infeasibility_certificate
        @test cert !== nothing
        @test q.feasible_witness === nothing
        @test cert.budget < cert.minimum_fixed_cost
        @test cert.minimum_fixed_cost == length(cert.groups) * minimum(q.fixed_cost)
        # Disjoint admissible sets across groups.
        for a in 1:length(cert.groups), b in (a + 1):length(cert.groups)
            @test isempty(
                intersect(
                    group_admissibles(q, cert.groups[a]), group_admissibles(q, cert.groups[b])
                ),
            )
        end
    end

    # Capacitated: witness respects hub capacities; the shortfall
    # certificate refutes even the LP relaxation.
    _, p = generate_problem("hub_location/capacitated", 150, feasible, 0)
    w = p.feasible_witness
    @test w !== nothing
    position = Dict(k => t for (t, k) in enumerate(p.hubs))
    for (t, k) in enumerate(p.hubs)
        load = sum(p.outvolume[i] for i in 1:p.n_nodes if w.assignment[i] == k)
        @test load <= p.capacity[t] + 1e-9
        @test w.assignment[k] == k
    end
    for s in 0:2
        _, q = generate_problem("hub_location/capacitated", 150, infeasible, s)
        cert = q.infeasibility_certificate
        @test cert !== nothing
        @test cert.total_flow == sum(q.outvolume)
        @test cert.total_capacity == sum(q.capacity)
        @test cert.total_capacity < cert.total_flow
        # Each hub has room for its own city's volume (no hub closed by its
        # own capacity row) and the roomiest fits the largest origin.
        @test all(q.capacity[t] >= q.outvolume[k] for (t, k) in enumerate(q.hubs))
        @test maximum(q.capacity) >= maximum(q.outvolume)
    end

    # Hub network: backbone witness carries the planted routing, and the
    # gateway-cut certificate arithmetic recomputes exactly.
    _, p = generate_problem("hub_location/hub_network", 150, feasible, 0)
    w = p.feasible_witness
    @test w !== nothing
    @test all(w.assignment[i] in w.open_hubs for i in 1:p.n_nodes)
    @test all(p.dist[i, w.assignment[i]] <= p.reach for i in 1:p.n_nodes)
    @test all(w.assignment[k] == k for k in w.open_hubs)
    loads = SyntheticLPs._hub_tree_link_loads(p.n_nodes, w.assignment, p.flow, w.backbone)
    link_pos = Dict(l => t for (t, l) in enumerate(p.links))
    for (l, load) in loads
        @test load <= p.link_capacity[link_pos[l]] + 1e-9
    end
    for s in 0:2
        _, q = generate_problem("hub_location/hub_network", 150, infeasible, s)
        cert = q.infeasibility_certificate
        @test cert !== nothing
        recomputed = sum(
            q.flow[i, j] + q.flow[j, i] for
            i in cert.side_a, j in 1:q.n_nodes if !(j in cert.side_a)
        )
        @test cert.crossing_flow ≈ recomputed
        crossing = [
            t for (t, (k, m)) in enumerate(q.links) if ((k in cert.side_a) != (m in cert.side_a))
        ]
        @test cert.crossing_capacity == sum(q.link_capacity[t] for t in crossing)
        @test cert.crossing_capacity < cert.crossing_flow
    end

    # Hub covering: the pruned witness keeps an open path for every OD pair
    # within budget; the infeasible budget certificate's in-region pairs only
    # use their own (disjoint) hub sets, and the budget is below the sum of
    # those sets' cheapest hubs. No covering row is empty.
    for s in 0:2
        _, p = generate_problem("hub_location/hub_covering", 150, feasible, s)
        w = p.feasible_witness
        @test w !== nothing
        open_set = Set(w.open_hubs)
        @test all(
            any(k in open_set && m in open_set for (k, m) in paths) for
            paths in values(p.covering_sets)
        )
        @test w.opening_cost ≈ sum(p.fixed_cost[w.open_hubs])
        @test p.budget >= w.opening_cost
        @test p.profile in (:passenger, :freight, :express)

        _, q = generate_problem("hub_location/hub_covering", 1000, infeasible, s)
        cert = q.infeasibility_certificate
        @test cert isa SyntheticLPs.HubCoveringBudgetCertificate
        @test all(!isempty(paths) for paths in values(q.covering_sets))
        for (od, hs) in zip(cert.pairs, cert.hub_sets)
            @test issubset(vcat(first.(q.covering_sets[od]), last.(q.covering_sets[od])), hs)
        end
        for a in 1:length(cert.hub_sets), b in (a + 1):length(cert.hub_sets)
            @test isempty(intersect(cert.hub_sets[a], cert.hub_sets[b]))
        end
        @test cert.minimum_cost ≈ sum(minimum(q.fixed_cost[hs]) for hs in cert.hub_sets)
        @test cert.budget == q.budget < cert.minimum_cost
    end

    # Budgeted backbone: the planted hubs form a complete selected-hub
    # subgraph with self-allocated candidate nodes and sufficient direct
    # link capacities. The infeasible budget violates the degree lower
    # bound even in the LP relaxation.
    for s in 0:2
        _, p = generate_problem("hub_location/budgeted_backbone", 150, feasible, s)
        w = p.feasible_witness
        @test w !== nothing
        @test length(w.open_hubs) == p.p
        @test length(w.links) == p.p * (p.p - 1) ÷ 2
        @test all(w.assignment[p.hubs[k]] == k for k in w.open_hubs)
        @test sum(p.link_cost[k, m] for (k, m) in w.links) <= p.link_budget + 1e-9
        for (k, m) in w.links
            direct_load = sum(
                p.flow[i, j] for i in 1:p.n_nodes, j in 1:p.n_nodes if
                (w.assignment[i] == k && w.assignment[j] == m) ||
                    (w.assignment[i] == m && w.assignment[j] == k)
            )
            @test direct_load <= p.link_capacity[k, m] + 1e-9
        end

        _, q = generate_problem("hub_location/budgeted_backbone", 150, infeasible, s)
        cert = q.infeasibility_certificate
        @test cert !== nothing
        @test cert.implied_minimum > cert.budget
        @test cert.budget == q.link_budget
    end

    # Window floors for feasible/unknown requests: no single-candidate window
    # (forces a hub open) and, where the variant allows it, no two-candidate
    # window (turns every supply row into a doubleton equation) - both let
    # presolve strip a large share of the rows on small instances.
    for status in (feasible, unknown), s in 0:3
        _, p = generate_problem("hub_location/p_hub_median", 1000, status, s)
        @test all(length(a) >= 2 for a in p.admissible)
        _, p = generate_problem("hub_location/r_allocation", 1000, status, s)
        @test all(length(a) >= p.r + 1 for a in p.admissible)
        _, p = generate_problem("hub_location/multiple_allocation", 1000, status, s)
        @test all(length(a) >= 2 for a in p.admissible)
        _, p = generate_problem("hub_location/hub_network", 1000, status, s)
        @test all(length(a) >= 2 for a in p.admissible)
        @test count(a -> length(a) >= 3, p.admissible) >= 0.8 * p.n_nodes
    end
    # r_allocation lands near the target from 500 variables up.
    for target in (500, 1_000, 5_000), status in (feasible, infeasible, unknown), s in 0:3
        m, _ = generate_problem("hub_location/r_allocation", target, status, s)
        @test abs(num_variables(m) - target) <= 0.07 * target
    end

    # Reproducibility and global-RNG isolation: identical seeds produce
    # field-identical structs even with a seeded/dirty global RNG.
    for v in list_variants(:hub_location)
        ref = ProblemVariant(:hub_location, v)
        Random.seed!(987)
        _, p1 = generate_problem(ref, 220, unknown, 42)
        Random.seed!(12345)
        _, p2 = generate_problem(ref, 220, unknown, 42)
        @test all(isequal(getfield(p1, f), getfield(p2, f)) for f in fieldnames(typeof(p1)))
    end

    if HAS_HIGHS
        # The feasibility contract holds end-to-end on the LP relaxation.
        for v in list_variants(:hub_location), status in (feasible, infeasible), s in 0:4
            m, _ = generate_problem(ProblemVariant(:hub_location, v), 220, status, s)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end

        # The same contract also holds for the unrelaxed integer models;
        # this catches witnesses that only work fractionally (especially
        # diagonal hub self-allocation and physical-link decisions).
        for v in list_variants(:hub_location), status in (feasible, infeasible), s in 0:1
            m, _ = generate_problem(
                ProblemVariant(:hub_location, v), 120, status, s; relax_integer=false
            )
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end

        # Unknown is a genuine mix, not an implicit always-one-way branch.
        optimal = 0
        infeasible_count = 0
        for v in list_variants(:hub_location), s in 0:9
            m, _ = generate_problem(ProblemVariant(:hub_location, v), 200, unknown, s)
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

        # Presolve keeps most of the small (~1k) instances of the variants
        # whose windows used to collapse under it (0.37-0.56 of the rows).
        for v in (:p_hub_median, :multiple_allocation, :hub_network),
            status in (feasible, unknown),
            s in 0:1

            m, _ = generate_problem(ProblemVariant(:hub_location, v), 1000, status, s)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            MOI.Utilities.attach_optimizer(m)
            highs = unsafe_backend(m)
            HiGHS.Highs_presolve(highs)
            @test HiGHS.Highs_getPresolvedNumCol(highs) >= 0.6 * num_variables(m)
            @test HiGHS.Highs_getPresolvedNumRow(highs) >=
                0.6 * num_constraints(m; count_variable_in_set_constraints=false)
        end

        # The default infeasible modes need simplex work: HiGHS presolve alone
        # does not refute them (it used to for these four variants, through
        # an impossible single row or bound-propagated hub closures).
        for v in (:capacitated, :compact_single_allocation, :r_allocation, :hub_covering), s in 0:1
            m, _ = generate_problem(ProblemVariant(:hub_location, v), 2000, infeasible, s)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            @test termination_status(m) == MOI.INFEASIBLE
            @test MOI.get(m, MOI.SimplexIterations()) > 0
        end
    end
end
