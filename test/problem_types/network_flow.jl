# Focused quality contracts for the network_flow category: registry shape,
# exact arc-count sizing and row formulas, the shared geographic-network
# invariants (strong connectivity, exact arc budget, trunk tree), witness and
# certificate arithmetic recomputed from the struct fields (max-flow plan and
# Gale cut for `standard`; lossy tree routing and efficiency-potential Farkas
# certificate for `generalized_flow`), reproducibility, and HiGHS feasibility
# contracts — including that the infeasible profiles need real simplex work
# rather than being disproved by presolve.

# Reachability from `root` along arcs (or reversed arcs).
function nf_reaches_all(n, arcs, root; reversed=false)
    adj = [Int[] for _ in 1:n]
    for (u, v) in arcs
        reversed ? push!(adj[v], u) : push!(adj[u], v)
    end
    seen = falses(n)
    seen[root] = true
    stack = [root]
    while !isempty(stack)
        u = pop!(stack)
        for v in adj[u]
            seen[v] || (seen[v] = true; push!(stack, v))
        end
    end
    return all(seen)
end

function nf_pearson(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((x[k] - mx) * (y[k] - my) for k in 1:n)
    sxx = sum((x[k] - mx)^2 for k in 1:n)
    syy = sum((y[k] - my)^2 for k in 1:n)
    return sxy / sqrt(sxx * syy)
end

nf_arc_length(p, k) = hypot(
    p.positions[p.arcs[k][2]][1] - p.positions[p.arcs[k][1]][1],
    p.positions[p.arcs[k][2]][2] - p.positions[p.arcs[k][1]][2],
)

@testset "Network Flow" begin
    @test :network_flow in list_categories()
    @test Set(list_variants(:network_flow)) == Set([:standard, :generalized_flow, :time_expanded])
    info = problem_info(:network_flow)
    @test info[:default_variant] == :standard
    @test occursin("flow", lowercase(info[:description]))

    variants = ("network_flow/standard", "network_flow/generalized_flow")

    @testset "sizing" begin
        # Variables are exactly the arcs and equal the target (tiny targets
        # round up to the spanning tree in both directions); rows are one
        # balance row per node.
        for ref in variants, target in (2, 3, 6, 10, 50, 200, 1000, 5000), status in
                                                                            (feasible, infeasible, unknown)
            m, p = generate_problem(ref, target, status, 3)
            @test num_variables(m) == length(p.arcs)
            @test length(p.arcs) == max(target, 2 * (p.n_nodes - 1))
            target >= 6 && @test length(p.arcs) == target
            @test num_constraints(m; count_variable_in_set_constraints=false) == p.n_nodes
            if target >= 1000
                # Sparse, scalable networks: 3.2-4.6 arcs per node, so rows
                # grow with the target instead of saturating.
                @test 3.0 <= length(p.arcs) / p.n_nodes <= 4.8
            end
        end
        @test length(generate_problem("network_flow/standard", 1, feasible, 0)[2].arcs) == 2
        @test length(generate_problem("network_flow/generalized_flow", 3, feasible, 0)[2].arcs) == 4

        cap = SyntheticLPs.NETWORK_FLOW_MAX_ARCS
        @test cap == 1_000_000
        @test_throws ArgumentError SyntheticLPs.NetworkFlowProblem(cap + 1, unknown, 0)
        @test_throws ArgumentError SyntheticLPs.GeneralizedFlowProblem(cap + 1, unknown, 0)
        @test_throws ArgumentError generate_problem("network_flow/standard", 0, unknown, 0)

        # Large-target sizing (constructor only for the generalized variant,
        # which builds in about a second at this scale).
        big = SyntheticLPs.GeneralizedFlowProblem(100_000, unknown, 0)
        @test length(big.arcs) == 100_000
        @test 100_000 / 4.8 <= big.n_nodes <= 100_000 / 3.0
    end

    @testset "geographic network invariants" begin
        for ref in variants, target in (40, 400, 3000), status in (feasible, infeasible, unknown)
            _, p = generate_problem(ref, target, status, 7)
            n = p.n_nodes
            @test issorted(p.arcs)
            @test allunique(p.arcs)
            @test all(1 <= u <= n && 1 <= v <= n && u != v for (u, v) in p.arcs)
            @test nf_reaches_all(n, p.arcs, 1)                  # strongly connected
            @test nf_reaches_all(n, p.arcs, 1; reversed=true)
            @test count(p.trunk) == 2 * (n - 1)                 # tree, both directions
            @test all(((v, u) in p.arcs) for (k, (u, v)) in enumerate(p.arcs) if p.trunk[k])
            @test length(p.capacities) == length(p.costs) == length(p.arcs)
            @test all(>(0.0), p.capacities)
            @test all(>(0.0), p.costs)
            @test p.geography in (:uniform, :clustered, :corridor)
            @test length(p.positions) == n
            @test !isempty(p.supply_nodes) && !isempty(p.demand_nodes)
            @test isempty(intersect(p.supply_nodes, p.demand_nodes))
            @test issorted(p.supply_nodes) && issorted(p.demand_nodes)
            @test all(p.supplies[v] > 0 for v in p.supply_nodes)
            @test all(p.demands[v] > 0 for v in p.demand_nodes)
            @test count(>(0.0), p.supplies) == length(p.supply_nodes)
            @test count(>(0.0), p.demands) == length(p.demand_nodes)
            @test p.feasibility_status == status
        end
        # Distance-grounded costs: strongly correlated with arc length.
        for ref in variants, seed in 0:3
            _, p = generate_problem(ref, 1500, unknown, seed)
            d = [nf_arc_length(p, k) for k in eachindex(p.arcs)]
            @test nf_pearson(p.costs, d) > 0.5
        end
        # Gains: lossy, never amplifying, never exactly lossless, and longer
        # arcs lose more.
        for seed in 0:3
            _, p = generate_problem("network_flow/generalized_flow", 1500, unknown, seed)
            @test all(0.5 <= g <= 0.9995 for g in p.gains)
            d = [nf_arc_length(p, k) for k in eachindex(p.arcs)]
            @test nf_pearson(-log.(p.gains), d) > 0.5
        end
    end

    @testset "standard: max-flow witness and Gale cut certificate" begin
        for target in (30, 300, 3000), seed in 0:3
            m, p = generate_problem("network_flow/standard", target, feasible, seed)
            w = p.feasible_witness
            @test w !== nothing && p.infeasibility_certificate === nothing
            @test 0.55 - 1e-12 <= p.load_factor <= 0.9 + 1e-12
            @test p.max_flow_value >= p.total_demand * (1 - 1e-9)
            @test p.total_demand ≈ sum(p.demands)
            f = w.arc_flows
            @test length(f) == length(p.arcs)
            @test all(-1e-9 .<= f .<= p.capacities .+ 1e-9)
            net_in = zeros(p.n_nodes)
            for (k, (u, v)) in enumerate(p.arcs)
                net_in[v] += f[k]
                net_in[u] -= f[k]
            end
            tol = 1e-7 * max(1.0, p.total_demand)
            for v in 1:(p.n_nodes)
                if v in p.supply_nodes
                    @test -net_in[v] <= p.supplies[v] + tol
                else
                    @test abs(net_in[v] - p.demands[v]) <= tol
                end
            end
            report = primal_feasibility_report(
                m, Dict(m[:flow][k] => f[k] for k in eachindex(f)); atol=1e-6
            )
            @test isempty(report)
        end

        nontrivial = 0
        for target in (30, 300, 3000), seed in 0:3
            _, p = generate_problem("network_flow/standard", target, infeasible, seed)
            c = p.infeasibility_certificate
            @test c !== nothing && p.feasible_witness === nothing
            @test 1.08 - 1e-12 <= p.load_factor <= 1.3 + 1e-12
            @test p.max_flow_value < p.total_demand
            T = falses(p.n_nodes)
            T[c.region] .= true
            @test !isempty(c.region)
            @test c.inbound_arcs == [k for (k, (u, v)) in enumerate(p.arcs) if !T[u] && T[v]]
            @test c.inbound_capacity ≈ sum(p.capacities[c.inbound_arcs]; init=0.0)
            @test c.region_demand ≈ sum(p.demands[c.region])
            @test c.region_supply ≈ sum(p.supplies[c.region])
            # The refutation, with the deficit equal to the max-flow shortfall.
            @test c.region_demand > c.region_supply + c.inbound_capacity
            @test c.region_demand - c.region_supply - c.inbound_capacity ≈
                p.total_demand - p.max_flow_value rtol = 1e-6
            nontrivial += length(c.region) > 1
        end
        @test nontrivial >= 8  # regions, not single starved nodes

        below = above = 0
        for seed in 0:19
            _, p = generate_problem("network_flow/standard", 300, unknown, seed)
            @test p.feasible_witness === nothing && p.infeasibility_certificate === nothing
            @test 0.85 - 1e-12 <= p.load_factor <= 1.15 + 1e-12
            p.max_flow_value >= p.total_demand * (1 - 1e-9) ? (below += 1) : (above += 1)
        end
        @test below > 0 && above > 0
    end

    @testset "generalized_flow: lossy witness and efficiency certificate" begin
        for target in (30, 300, 3000), seed in 0:3
            m, p = generate_problem("network_flow/generalized_flow", target, feasible, seed)
            w = p.feasible_witness
            @test w !== nothing && p.infeasibility_certificate === nothing
            f = w.arc_flows
            @test all(-1e-9 .<= f .<= p.capacities .+ 1e-9)
            net_in = zeros(p.n_nodes)
            for (k, (u, v)) in enumerate(p.arcs)
                net_in[v] += p.gains[k] * f[k]
                net_in[u] -= f[k]
            end
            tol = 1e-7 * max(1.0, sum(p.demands))
            for (i, s) in enumerate(p.supply_nodes)
                @test -net_in[s] ≈ w.source_outflow[i] atol = tol
                @test w.source_outflow[i] <= p.supplies[s] + tol
            end
            for v in 1:(p.n_nodes)
                v in p.supply_nodes && continue
                @test abs(net_in[v] - p.demands[v]) <= tol
            end
            report = primal_feasibility_report(
                m, Dict(m[:flow][k] => f[k] for k in eachindex(f)); atol=1e-6
            )
            @test isempty(report)
        end

        for target in (30, 300, 3000), seed in 0:3
            _, p = generate_problem("network_flow/generalized_flow", target, infeasible, seed)
            c = p.infeasibility_certificate
            @test c !== nothing && p.feasible_witness === nothing
            E = c.efficiency
            # Potentials: 1 at supply sites, monotone along every arc, so every
            # column of the 1/E-weighted row sum is nonpositive.
            @test all(E[s] == 1.0 for s in p.supply_nodes)
            @test all(0.0 < e <= 1.0 for e in E)
            @test all(
                E[v] >= E[u] * p.gains[k] * (1 - 1e-12) for (k, (u, v)) in enumerate(p.arcs)
            )
            # Tightness: every non-supply node attains its potential via some arc.
            for v in 1:(p.n_nodes)
                v in p.supply_nodes && continue
                @test any(
                    isapprox(E[v], E[u] * p.gains[k]; rtol=1e-9) for
                    (k, (u, x)) in enumerate(p.arcs) if x == v
                )
            end
            @test c.required_supply ≈ sum(p.demands[v] / E[v] for v in p.demand_nodes)
            @test c.total_supply ≈ sum(p.supplies)
            @test c.total_supply < c.required_supply
            # Not an aggregate shortage: lossless supply exceeds demand.
            @test c.total_supply > sum(p.demands)
        end
    end

    @testset "reproducibility" begin
        for ref in variants, status in (feasible, infeasible, unknown)
            Random.seed!(987)
            _, p1 = generate_problem(ref, 400, status, 42)
            Random.seed!(12345)
            rand(1000)
            _, p2 = generate_problem(ref, 400, status, 42)
            for f in fieldnames(typeof(p1))
                a, b = getfield(p1, f), getfield(p2, f)
                if a === nothing || a isa Union{Number, Symbol, Vector, FeasibilityStatus}
                    @test isequal(a, b)
                else
                    @test all(isequal(getfield(a, g), getfield(b, g)) for g in fieldnames(typeof(a)))
                end
            end
        end
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            function nf_solve(m)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                return termination_status(m), MOI.get(m, MOI.SimplexIterations())
            end
            for ref in variants, target in (50, 400, 2500), seed in 0:2
                ts, _ = nf_solve(generate_problem(ref, target, feasible, seed)[1])
                @test ts == MOI.OPTIMAL
                ts, iters = nf_solve(generate_problem(ref, target, infeasible, seed)[1])
                @test ts == MOI.INFEASIBLE
                # Not refuted by presolve: simplex had to work for it.
                target >= 400 && @test iters > 0
            end
            # standard/unknown is decided exactly by the stored max flow.
            for seed in 0:9
                m, p = generate_problem("network_flow/standard", 400, unknown, seed)
                ts, _ = nf_solve(m)
                @test ts ==
                    (p.max_flow_value >= p.total_demand * (1 - 1e-9) ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end
            # generalized_flow/unknown is genuinely two-sided.
            outcomes = Set{Any}()
            for seed in 0:11
                ts, _ = nf_solve(generate_problem("network_flow/generalized_flow", 1000, unknown, seed)[1])
                push!(outcomes, ts)
            end
            @test MOI.OPTIMAL in outcomes && MOI.INFEASIBLE in outcomes
            # The framework's verify-and-retry path.
            for ref in variants, status in (feasible, infeasible)
                m, _ = generate_problem(ref, 300, status, 1; optimizer=HiGHS.Optimizer)
                @test num_variables(m) == 300
            end
        end
    end

    @testset "time_expanded evacuation" begin
        ref = "network_flow/time_expanded"
        for target in (1000, 5000, 20_000), seed in 0:2
            m, p = generate_problem(ref, target, unknown, seed)
            nv = length(p.moves) + length(p.waits) + length(p.intakes)
            @test num_variables(m) == nv
            @test 0.65 * target <= nv <= 1.35 * target
            @test num_constraints(m; count_variable_in_set_constraints=false) == length(p.node_copies)
        end
        big = SyntheticLPs.TimeExpandedEvacuationProblem(100_000, unknown, 0)
        @test abs(length(big.moves) + length(big.waits) + length(big.intakes) - 100_000) <= 10_000

        for seed in 0:2
            _, p = generate_problem(ref, 4000, feasible, seed)
            copies = Set(p.node_copies)
            @test all(1 <= t <= 4 for t in p.travel_time)
            @test all(t + p.travel_time[a] <= p.horizon for (a, t) in p.moves)
            @test all((p.arcs[a][1], t) in copies && (p.arcs[a][2], t + p.travel_time[a]) in copies for (a, t) in p.moves)
            @test all((v, t) in copies && (v, t + 1) in copies && p.hold_capacity[v] > 0 for (v, t) in p.waits)
            @test all(p.intake_rate[v] > 0 && (v, t) in copies for (v, t) in p.intakes)
            zones = findall(>(0.0), p.supply)
            @test all((v, 0) in copies for v in zones)
            @test isempty(intersect(zones, findall(>(0.0), p.intake_rate)))
        end

        for target in (800, 4000), seed in 0:2
            m, p = generate_problem(ref, target, feasible, seed)
            w = p.feasible_witness
            vals = Dict{VariableRef, Float64}()
            for i in eachindex(p.moves)
                vals[m[:move][i]] = w.moves[i]
            end
            for i in eachindex(p.waits)
                vals[m[:wait][i]] = w.waits[i]
            end
            for i in eachindex(p.intakes)
                vals[m[:intake][i]] = w.intakes[i]
            end
            @test isempty(primal_feasibility_report(m, vals; atol=1e-6))
            @test sum(w.intakes) ≈ p.total_supply rtol = 1e-9
        end

        for target in (800, 4000), seed in 0:3
            _, p = generate_problem(ref, target, infeasible, seed)
            c = p.infeasibility_certificate
            X = Set(p.node_copies[c.region])
            @test c.exit_moves == [
                i for (i, (a, t)) in enumerate(p.moves) if (p.arcs[a][1], t) in X && !((p.arcs[a][2], t + p.travel_time[a]) in X)
            ]
            @test c.exit_waits == [i for (i, (v, t)) in enumerate(p.waits) if (v, t) in X && !((v, t + 1) in X)]
            @test c.exit_intakes == [i for (i, (v, t)) in enumerate(p.intakes) if (v, t) in X]
            cap = sum(p.road_capacity[p.moves[i][1]] for i in c.exit_moves; init=0.0) +
                sum(p.hold_capacity[p.waits[i][1]] for i in c.exit_waits; init=0.0) +
                sum(p.intake_rate[p.intakes[i][1]] for i in c.exit_intakes; init=0.0)
            @test c.exit_capacity ≈ cap
            @test c.trapped_supply ≈ sum(p.supply[v] for (v, t) in X if t == 0; init=0.0)
            @test c.trapped_supply > c.exit_capacity
        end

        for status in (feasible, infeasible, unknown)
            Random.seed!(31)
            _, p1 = generate_problem(ref, 900, status, 5)
            Random.seed!(32)
            rand(5)
            _, p2 = generate_problem(ref, 900, status, 5)
            for f in fieldnames(typeof(p1))
                a, b = getfield(p1, f), getfield(p2, f)
                if a === nothing || a isa Union{Number, Symbol, Vector, FeasibilityStatus}
                    @test isequal(a, b)
                else
                    @test all(isequal(getfield(a, g), getfield(b, g)) for g in fieldnames(typeof(a)))
                end
            end
        end

        if HAS_HIGHS
            for target in (800, 4000), seed in 0:2
                m, _ = generate_problem(ref, target, feasible, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == MOI.OPTIMAL
                m, _ = generate_problem(ref, target, infeasible, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == MOI.INFEASIBLE
                target >= 4000 && @test MOI.get(m, MOI.SimplexIterations()) > 0
            end
            for seed in 0:7
                m, p = generate_problem(ref, 1500, unknown, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) ==
                    (p.max_flow_value >= p.total_supply * (1 - 1e-9) ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end
        end
    end
end
