# Focused quality contracts for resilient_network_design: exact sizing, the
# geometric candidate topology (spanning tree first), spatially correlated
# hazard scenarios, the planted build+harden+route witness, the hardening-budget
# certificate, reproducibility, and HiGHS status contracts.
@testset "Resilient network design" begin
    @test list_variants(:resilient_network_design) == [:standard]
    ref = "resilient_network_design/standard"

    for target in (20, 200, 2000, 20_000), status in (feasible, infeasible, unknown), seed in 0:1
        model, p = generate_problem(ref, target, status, seed; relax_integer=false)
        E, S, N = p.n_edges, p.n_scenarios, p.n_nodes
        S_target, E_target, N_target = SyntheticLPs._resilient_dimensions(target)
        @test (S, N) == (S_target, N_target)
        @test 0.9 * E_target <= E <= E_target
        @test num_variables(model) == 2E * (1 + S)
        @test num_constraints(model; count_variable_in_set_constraints=false) ==
            1 + E * (1 + S) + N * S
        @test count(is_binary, all_variables(model)) == 2E
        target >= 200 && @test abs(num_variables(model) - target) <= 0.1 * target
        # Topology: unique undirected links, the first N-1 a spanning tree.
        @test allunique(p.edges) && all(i < j for (i, j) in p.edges)
        parent = collect(1:N)
        find(x) = parent[x] == x ? x : (parent[x] = find(parent[x]))
        for e in 1:(N - 1)
            a, b = find(p.edges[e][1]), find(p.edges[e][2])
            @test a != b
            parent[a] = b
        end
        @test length(unique(find(v) for v in 1:N)) == 1
        @test all(p.sources .!= p.sinks)
        @test all(>(0), p.demands) && all(>(0), p.capacities)
        @test all(any(@view p.failed[:, s]) for s in 1:S)
        @test (p.feasible_witness !== nothing) == (status == feasible)
        @test (p.infeasibility_certificate !== nothing) == (status == infeasible)
    end

    # Hazards are spatial: links near the hazard center fail far more often.
    near, far = Int[0, 0], Int[0, 0]
    for seed in 0:3
        _, p = generate_problem(ref, 5000, unknown, seed)
        for s in 1:p.n_scenarios, (e, (i, j)) in enumerate(p.edges)
            mid = (
                (p.positions[i][1] + p.positions[j][1]) / 2,
                (p.positions[i][2] + p.positions[j][2]) / 2,
            )
            r = hypot(mid[1] - p.hazard_center[s][1], mid[2] - p.hazard_center[s][2])
            bucket =
                r < 0.5 * p.hazard_radius[s] ? near : (r > 2 * p.hazard_radius[s] ? far : nothing)
            bucket === nothing && continue
            bucket[1] += p.failed[e, s]
            bucket[2] += 1
        end
    end
    @test near[1] / near[2] > 3 * far[1] / far[2]

    # Planted witness satisfies every row of the unrelaxed model.
    for target in (100, 1500), seed in 0:3
        model, p = generate_problem(ref, target, feasible, seed; relax_integer=false)
        w = p.feasible_witness
        point = Dict{VariableRef, Float64}()
        for e in 1:p.n_edges
            point[model[:build][e]] = w.build[e]
            point[model[:harden][e]] = w.harden[e]
            for s in 1:p.n_scenarios
                point[model[:forward][e, s]] = w.forward[e, s]
                point[model[:reverse][e, s]] = w.reverse[e, s]
            end
        end
        @test isempty(primal_feasibility_report(model, point; atol=1e-7))
        tree = p.n_nodes - 1
        @test all(w.build[1:tree] .== 1.0) && all(w.build[(tree + 1):end] .== 0.0)
        @test p.design_budget ≈
            1.05 * sum(p.build_cost[e] + p.hardening_cost[e] for e in 1:tree) + 1.0
    end

    # Hardening-budget certificate arithmetic, recomputed from the data.
    function components_without(p, skip)
        parent = collect(1:p.n_nodes)
        find(x) = parent[x] == x ? x : (parent[x] = find(parent[x]))
        for (e, (i, j)) in enumerate(p.edges)
            e == skip && continue
            parent[find(i)] = find(j)
        end
        return [find(v) for v in 1:p.n_nodes]
    end
    for target in (200, 2000, 20_000), seed in 0:3
        _, p = generate_problem(ref, target, infeasible, seed)
        c = p.infeasibility_certificate
        s = c.scenario
        region = Set(c.region)
        @test p.sinks[s] in region && !(p.sources[s] in region)
        # No other scenario has to reach into the district.
        for t in 1:p.n_scenarios
            t == s && continue
            @test !(p.sources[t] in region) || p.sources[t] == p.sinks[s]
            @test !(p.sinks[t] in region) || p.sinks[t] == p.sinks[s]
        end
        @test c.cut_edges ==
            [e for (e, (i, j)) in enumerate(p.edges) if (i in region) != (j in region)]
        @test all(p.failed[e, s] for e in c.cut_edges)   # the hazard takes out every access link
        @test c.demand == p.demands[s]
        @test c.cut_capacity ≈ sum(p.capacities[e] for e in c.cut_edges)
        @test c.cut_capacity >= 1.24 * c.demand          # capacity alone is not the obstruction
        # With everything built and hardened, every scenario routes with headroom.
        for t in 1:p.n_scenarios
            value, _ = SyntheticLPs._resilient_max_flow(
                p.n_nodes, p.edges, p.capacities, p.sources[t], p.sinks[t]
            )
            @test value >= 1.24 * p.demands[t]
        end
        # Bridge-forced design levels.
        for (k, b) in enumerate(c.bridges)
            comp = components_without(p, b)
            @test length(unique(comp)) == 2
            build, harden = 0.0, 0.0
            for t in 1:p.n_scenarios
                comp[p.sources[t]] == comp[p.sinks[t]] && continue
                level = p.demands[t] / p.capacities[b]
                build = max(build, level)
                p.failed[b, t] && (harden = max(harden, level))
            end
            @test c.bridge_build[k] ≈ build
            @test c.bridge_harden[k] ≈ harden
            @test build <= 0.81
        end
        forced = sum(
            p.build_cost[b] * c.bridge_build[k] + p.hardening_cost[b] * c.bridge_harden[k] for
            (k, b) in enumerate(c.bridges);
            init=0.0,
        )
        @test c.forced_spend ≈ forced
        # District knapsack: harden the boundary beyond its forced levels.
        level_b = Dict(b => c.bridge_build[k] for (k, b) in enumerate(c.bridges))
        level_h = Dict(b => c.bridge_harden[k] for (k, b) in enumerate(c.bridges))
        pieces = Tuple{Float64, Float64}[]   # (cost per capacity, capacity)
        required = c.demand
        for e in c.cut_edges
            fb, fh = get(level_b, e, 0.0), get(level_h, e, 0.0)
            required -= p.capacities[e] * fh
            fb > fh &&
                push!(pieces, (p.hardening_cost[e] / p.capacities[e], p.capacities[e] * (fb - fh)))
            push!(
                pieces,
                (
                    (p.build_cost[e] + p.hardening_cost[e]) / p.capacities[e],
                    p.capacities[e] * (1 - max(fb, fh)),
                ),
            )
        end
        spend = 0.0
        for (ratio, cap) in sort(pieces)
            required <= 0 && break
            take = min(cap, required)
            spend += ratio * take
            required -= take
        end
        @test c.cut_spend ≈ spend rtol = 1e-9
        @test c.cut_spend > 0
        @test c.implied_minimum ≈ c.forced_spend + c.cut_spend
        @test c.budget == p.design_budget
        @test c.margin ≈ c.implied_minimum - c.budget
        @test c.margin >= 0.06 * c.cut_spend
    end

    # Reproducibility.
    _, a = generate_problem(ref, 3000, unknown, 5)
    _, b = generate_problem(ref, 3000, unknown, 5)
    for name in fieldnames(typeof(a))
        @test isequal(getfield(a, name), getfield(b, name))
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            function rnd_status(m)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                return termination_status(m)
            end
            for target in (200, 2000), seed in 0:3
                @test rnd_status(first(generate_problem(ref, target, feasible, seed))) ==
                    MOI.OPTIMAL
                @test rnd_status(first(generate_problem(ref, target, infeasible, seed))) ==
                    MOI.INFEASIBLE
            end
            outcomes = [rnd_status(first(generate_problem(ref, 1000, unknown, s))) for s in 0:11]
            @test count(==(MOI.OPTIMAL), outcomes) >= 2
            @test count(==(MOI.INFEASIBLE), outcomes) >= 1
        end
    end
end
