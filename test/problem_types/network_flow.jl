# Focused quality contracts for the network_flow category: registry shape,
# exact arc-count sizing and row formulas, geography-grounded data invariants,
# max-flow witness and min-cut certificate arithmetic recomputed from the
# struct fields, the exact feasibility boundary that keeps the `unknown`
# profile genuinely mixed at every scale, reproducibility, and HiGHS
# feasibility contracts (including Dinic's stored max flow against the solver).
@testset "Network Flow" begin
    @test :network_flow in list_categories()
    @test Set(list_variants(:network_flow)) == Set([:standard, :generalized_flow])
    info = problem_info(:network_flow)
    @test info[:default_variant] == :standard
    @test occursin("flow", lowercase(info[:description]))

    # Sizing: variables are exactly the arcs, i.e. exactly the target (the DAG
    # is sized so n*(n-1)/2 >= target candidates exist and filled to the cap),
    # and rows are one capacity row per arc plus one conservation row per
    # intermediate node plus the source-outflow equality on :min_cost.
    for target in (50, 200, 1000, 5000), status in (feasible, infeasible, unknown), seed in 0:2
        m, p = generate_problem(:network_flow, target, status, seed)
        @test num_variables(m) == length(p.arcs) == target
        @test p.n_nodes == max(4, ceil(Int, (1 + sqrt(1 + 8 * target)) / 2))
        expected_rows = length(p.arcs) + (p.n_nodes - 2) + (p.flow_objective == :min_cost ? 1 : 0)
        @test num_constraints(m; count_variable_in_set_constraints=false) == expected_rows
        @test abs(num_variables(m) - target) <= 0.25 * target || num_variables(m) <= 50
    end

    # Sizing cap: above the documented limit the request is rejected rather
    # than silently undersized; at the limit the constructor still delivers
    # exactly the requested arc count (constructor only -- no model build).
    cap = SyntheticLPs.NETWORK_FLOW_MAX_ARCS
    @test cap == 1_000_000
    @test_throws ArgumentError SyntheticLPs.NetworkFlowProblem(cap + 1, unknown, 0)
    @test_throws ArgumentError generate_problem(:network_flow, cap + 1, unknown, 0)
    big = SyntheticLPs.NetworkFlowProblem(cap, unknown, 0)
    @test length(big.arcs) == cap
    @test big.n_nodes == 1415

    # Structural data contracts shared by all three profiles: a sorted
    # forward-arc DAG on the full backbone with aligned positive data and a
    # strictly positive exact max flow.
    for target in (60, 400, 3000), status in (feasible, infeasible, unknown)
        _, p = generate_problem(:network_flow, target, status, 7)
        @test p.source_node == 1
        @test p.sink_node == p.n_nodes
        @test issorted(p.arcs)
        @test allunique(p.arcs)
        @test all(a[1] < a[2] for a in p.arcs)                    # forward only: a DAG
        @test all(1 <= a[1] < a[2] <= p.n_nodes for a in p.arcs)  # in range, no self-loops
        @test all(((i, i + 1) in p.arcs) for i in 1:(p.n_nodes - 1))  # backbone present
        @test length(p.capacities) == length(p.costs) == length(p.arcs)
        @test all(>(0.0), p.capacities)
        @test all(>(0.0), p.costs)
        @test p.geography in (:corridor, :clustered, :uniform)
        @test length(p.positions) == p.n_nodes
        @test all(0.0 <= c <= 100.0 for pos in p.positions for c in pos)
        @test p.flow_objective in (:max_flow, :min_cost)
        @test p.max_flow_value > 0
        @test p.feasibility_status == status
    end

    # Geography grounding: per-unit cost is distance-proportional with
    # lognormal route noise, so cost and endpoint distance must be strongly
    # positively correlated (pure uniform noise would sit near zero).
    function nf_pearson(x::Vector{Float64}, y::Vector{Float64})
        n = length(x)
        mx, my = sum(x) / n, sum(y) / n
        sxy = sum((x[k] - mx) * (y[k] - my) for k in 1:n)
        sxx = sum((x[k] - mx)^2 for k in 1:n)
        syy = sum((y[k] - my)^2 for k in 1:n)
        return sxy / sqrt(sxx * syy)
    end
    for target in (60, 400, 3000), seed in 0:4
        _, p = generate_problem(:network_flow, target, unknown, seed)
        d = [
            hypot(
                p.positions[a[2]][1] - p.positions[a[1]][1],
                p.positions[a[2]][2] - p.positions[a[1]][2],
            ) for a in p.arcs
        ]
        @test nf_pearson(p.costs, d) > 0.5
    end

    # Max-flow witness: the stored plan is a genuine feasible point. Checked by
    # arithmetic on the struct fields and, on smaller instances, against the
    # model itself via JuMP's primal feasibility report.
    for target in (50, 300, 2000), seed in 0:2
        m, p = generate_problem(:network_flow, target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing
        @test p.infeasibility_certificate === nothing
        @test length(w.arc_flows) == length(p.arcs)
        @test all(w.arc_flows .>= -1e-12)
        @test all(w.arc_flows[k] <= p.capacities[k] + 1e-9 for k in eachindex(p.arcs))
        # Net accumulation per node: zero at every intermediate node (flow
        # conservation), all of the plan's outflow absorbed by the sink.
        net = zeros(p.n_nodes)
        for (k, (u, v)) in enumerate(p.arcs)
            net[u] -= w.arc_flows[k]
            net[v] += w.arc_flows[k]
        end
        for v in 1:p.n_nodes
            (v == p.source_node || v == p.sink_node) && continue
            @test abs(net[v]) < 1e-6 * p.max_flow_value
        end
        @test net[p.sink_node] ≈ w.source_outflow atol = 1e-6 * p.max_flow_value
        src_out = sum(w.arc_flows[k] for (k, (u, v)) in enumerate(p.arcs) if u == p.source_node)
        @test w.source_outflow ≈ src_out rtol = 1e-9
        if p.flow_objective == :min_cost
            # Contracted volume: 25%-85% of the exact max flow, and the plan
            # ships exactly the contract out of the source.
            @test isapprox(p.target_flow, src_out; rtol=1e-9)
            @test 0.25 - 1e-12 <= p.target_flow / p.max_flow_value <= 0.85 + 1e-9
        else
            @test p.target_flow === nothing
            @test src_out ≈ p.max_flow_value rtol = 1e-9
        end
        # End-to-end: the plan satisfies every row of the built model.
        report = primal_feasibility_report(
            m, Dict(m[:flow][k] => w.arc_flows[k] for k in 1:length(p.arcs)); atol=1e-6
        )
        @test isempty(report)
    end

    # Min-cut certificate: the listed cut arcs are exactly the arcs crossing
    # the stored source side, their capacity is the max flow (a minimum cut),
    # and it sits strictly below the contracted volume -- which refutes the
    # source-outflow equality, since any feasible flow must push at least the
    # contract across the cut (nothing re-enters the source in a forward DAG).
    for target in (50, 300, 2000), seed in 0:3
        _, p = generate_problem(:network_flow, target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing
        @test p.feasible_witness === nothing
        # A max-flow objective is always feasible (zero flow), so an
        # infeasible request is a min-cost contract that cannot be routed.
        @test p.flow_objective == :min_cost
        @test p.target_flow > p.max_flow_value
        @test 1.15 - 1e-12 <= p.target_flow / p.max_flow_value <= 1.6 + 1e-9
        @test p.source_node in cert.source_side
        @test !(p.sink_node in cert.source_side)
        side = Set(cert.source_side)
        @test cert.cut_arcs == [k for (k, (u, v)) in enumerate(p.arcs) if u in side && !(v in side)]
        @test !isempty(cert.cut_arcs)
        @test cert.cut_capacity ≈ sum(p.capacities[k] for k in cert.cut_arcs) rtol = 1e-12
        @test cert.cut_capacity ≈ p.max_flow_value rtol = 1e-9  # min cut = max flow
        @test cert.cut_capacity < p.target_flow                 # the refutation
    end

    # The `unknown` profile keeps the sampled objective, stores neither a
    # witness nor a certificate, and -- on :min_cost -- draws the contract as
    # 60%-140% of the exact max flow, a genuine coin flip on either side of
    # the feasibility boundary at every problem size.
    for target in (100, 300)
        objectives = Set{Symbol}()
        below = above = 0
        for seed in 0:29
            _, p = generate_problem(:network_flow, target, unknown, seed)
            push!(objectives, p.flow_objective)
            @test p.feasible_witness === nothing
            @test p.infeasibility_certificate === nothing
            if p.flow_objective == :max_flow
                @test p.target_flow === nothing
            else
                ratio = p.target_flow / p.max_flow_value
                @test 0.6 - 1e-12 <= ratio <= 1.4 + 1e-9
                p.target_flow <= p.max_flow_value ? (below += 1) : (above += 1)
            end
        end
        @test :max_flow in objectives
        @test :min_cost in objectives
        @test below > 0
        @test above > 0
    end

    # Reproducibility, including isolation from a seeded/dirty global RNG.
    # Witness/certificate structs hold Vectors, so `isequal` on them is
    # identity-based; their own fields are compared element-wise instead.
    for status in (feasible, infeasible, unknown)
        Random.seed!(987)
        _, p1 = generate_problem(:network_flow, 220, status, 42)
        Random.seed!(12345)
        _, p2 = generate_problem(:network_flow, 220, status, 42)
        plain = (
            :n_nodes,
            :source_node,
            :sink_node,
            :arcs,
            :capacities,
            :costs,
            :positions,
            :geography,
            :flow_objective,
            :target_flow,
            :max_flow_value,
            :feasibility_status,
        )
        @test all(isequal(getfield(p1, f), getfield(p2, f)) for f in plain)
        if p1.feasible_witness !== nothing
            @test p1.feasible_witness.arc_flows == p2.feasible_witness.arc_flows
            @test p1.feasible_witness.source_outflow == p2.feasible_witness.source_outflow
        end
        if p1.infeasibility_certificate !== nothing
            c1, c2 = p1.infeasibility_certificate, p2.infeasibility_certificate
            @test c1.source_side == c2.source_side
            @test c1.cut_arcs == c2.cut_arcs
            @test c1.cut_capacity == c2.cut_capacity
        end
    end

    if HAS_HIGHS
        # End-to-end feasibility contract across scales and seeds.
        for target in (50, 200, 1000), status in (feasible, infeasible), seed in 0:4
            m, _ = generate_problem(:network_flow, target, status, seed)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end

        # The same contract through the framework's verify-and-retry path.
        for status in (feasible, infeasible), seed in 0:2
            m, _ = generate_problem(:network_flow, 300, status, seed; optimizer=HiGHS.Optimizer)
            @test num_variables(m) > 0
        end

        # The stored Dinic max flow agrees with the solver's objective on
        # :max_flow instances, and `unknown` is decided exactly by the
        # max-flow boundary: the contract is routable iff it does not exceed
        # the max flow. Both outcomes occur at both scales.
        for target in (200, 1000)
            optimal = infeasible_count = 0
            for seed in 0:19
                m, p = generate_problem(:network_flow, target, unknown, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                ts = termination_status(m)
                if p.flow_objective == :max_flow
                    @test ts == MOI.OPTIMAL
                    @test objective_value(m) ≈ p.max_flow_value rtol = 1e-6
                    optimal += 1
                else
                    @test ts == (p.target_flow <= p.max_flow_value ? MOI.OPTIMAL : MOI.INFEASIBLE)
                    ts == MOI.OPTIMAL ? (optimal += 1) : (infeasible_count += 1)
                end
            end
            @test optimal > 0
            @test infeasible_count > 0
        end
    end
end
