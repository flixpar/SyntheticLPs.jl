# Focused quality contracts for the load_balancing category: registry shape,
# exact path-count sizing, candidate-path and port invariants, the planted TE
# routing and placement witnesses (checked against the built models), the
# latency-metric and aggregate-workload certificates recomputed from the struct
# fields, the local repair that keeps single rows unrefutable, reproducibility,
# and HiGHS contracts.

@testset "Load Balancing" begin
    @test Set(list_variants(:load_balancing)) == Set([:standard, :discrete_placement])

    @testset "standard sizing and structure" begin
        for target in (30, 300, 3000, 30_000), status in (feasible, infeasible, unknown)
            m, p = generate_problem("load_balancing/standard", target, status, 1)
            @test num_variables(m) == length(p.paths) + 1
            target >= 300 && @test abs(num_variables(m) - target) <= 2
            used = falses(length(p.links))
            for path in p.paths, a in path
                used[a] = true
            end
            n_link_rows = count(a -> used[a] || p.background[a] > 0, eachindex(p.links))
            @test num_constraints(m; count_variable_in_set_constraints=false) ==
                length(p.od_pairs) + n_link_rows
        end
        @test_throws ArgumentError SyntheticLPs.LoadBalancingProblem(
            SyntheticLPs.LOAD_BALANCING_MAX_VARIABLES + 1, unknown, 0
        )
        for seed in 0:2
            _, p = generate_problem("load_balancing/standard", 4000, unknown, seed)
            @test issorted(p.od_pairs) && allunique(p.od_pairs)
            @test p.max_utilization in (0.8, 0.9, 1.0)
            @test all(c -> c in SyntheticLPs._LB_PORT_SIZES || (c % 400 == 0), p.capacities)
            @test all(>=(0.0), p.background)
            counts = zeros(Int, length(p.od_pairs))
            for (pi, k) in enumerate(p.path_od)
                counts[k] += 1
                o, d = p.od_pairs[k]
                path = p.paths[pi]
                # A path is a contiguous walk from the OD's origin to its destination.
                @test p.links[path[1]][1] == o && p.links[path[end]][2] == d
                @test all(p.links[path[i]][2] == p.links[path[i + 1]][1] for i in 1:(length(path) - 1))
            end
            @test all(>=(3), counts)          # every TE pair has real choices
            @test all(allunique(p.paths[q] for q in eachindex(p.paths) if p.path_od[q] == k) for k in eachindex(p.od_pairs)[1:20])
        end
    end

    @testset "standard witness and certificate" begin
        for target in (200, 3000), seed in 0:2
            m, p = generate_problem("load_balancing/standard", target, feasible, seed)
            w = p.feasible_witness
            load = copy(p.background)
            for (q, path) in enumerate(p.paths), a in path
                load[a] += w.path_flows[q]
            end
            @test w.utilization ≈ maximum(load ./ p.capacities)
            @test w.utilization <= p.max_utilization
            vals = Dict{VariableRef, Float64}(m[:x][q] => w.path_flows[q] for q in eachindex(p.paths))
            vals[m[:U]] = w.utilization
            @test isempty(primal_feasibility_report(m, vals; atol=1e-6))
        end
        for target in (200, 3000), seed in 0:3
            _, p = generate_problem("load_balancing/standard", target, infeasible, seed)
            c = p.infeasibility_certificate
            @test all(l -> l == 0 || l in p.link_latency, c.lengths)
            min_len = [
                minimum(sum(c.lengths[a] for a in p.paths[q]) for q in eachindex(p.paths) if p.path_od[q] == k)
                for k in eachindex(p.od_pairs)
            ]
            @test c.required ≈ sum(p.demands .* min_len) + sum(p.background .* c.lengths)
            @test c.capacity_length ≈ p.max_utilization * sum(p.capacities .* c.lengths)
            @test c.capacity_length < c.required
        end
        # Local repair at medium sizes: no demand row or link row refutable alone.
        for seed in 0:2
            _, p = generate_problem("load_balancing/standard", 3000, infeasible, seed)
            K = length(p.od_pairs)
            paths_of = [[q for q in eachindex(p.paths) if p.path_od[q] == k] for k in 1:K]
            for k in 1:K
                bott = sum(minimum(p.capacities[a] for a in p.paths[q]) for q in paths_of[k])
                @test bott * p.max_utilization >= p.demands[k]
            end
            forced = copy(p.background)
            for k in 1:K
                common = reduce(intersect, (Set(p.paths[q]) for q in paths_of[k]))
                for a in common
                    forced[a] += p.demands[k]
                end
            end
            @test all(forced .<= p.capacities .* p.max_utilization .+ 1e-9)
        end
    end

    @testset "discrete_placement" begin
        for target in (100, 2000), status in (feasible, infeasible, unknown)
            m, p = generate_problem("load_balancing/discrete_placement", target, status, 2)
            S, M, K = p.n_services, p.n_machines, p.n_classes
            @test num_variables(m) == S * M + K * S * M + M + 1
            @test abs(num_variables(m) - target) <= 0.25 * target
            @test p.feasibility_status == status
        end
        for seed in 0:2
            m, p = generate_problem("load_balancing/discrete_placement", 800, feasible, seed)
            w = p.feasible_witness
            S, M, K = p.n_services, p.n_machines, p.n_classes
            vals = Dict{VariableRef, Float64}()
            for s in 1:S, mm in 1:M
                vals[m[:placement][s, mm]] = mm == w.machine[s] ? 1.0 : 0.0
                for k in 1:K
                    vals[m[:workload][k, s, mm]] = mm == w.machine[s] ? p.demand[k, s] : 0.0
                end
            end
            for mm in 1:M
                vals[m[:machine_load][mm]] = w.machine_load[mm]
            end
            vals[m[:makespan]] = w.makespan
            @test isempty(primal_feasibility_report(m, vals; atol=1e-6))
            _, q = generate_problem("load_balancing/discrete_placement", 800, infeasible, seed)
            c = q.infeasibility_certificate
            lb = sum(q.demand[k, s] * minimum(q.processing_time[s, :]) for k in 1:(q.n_classes), s in 1:(q.n_services))
            @test c.workload_lower_bound ≈ lb
            @test c.total_capacity ≈ sum(q.machine_capacity)
            @test c.total_capacity < c.workload_lower_bound
        end
    end

    @testset "reproducibility" begin
        for ref in ("load_balancing/standard", "load_balancing/discrete_placement"), status in
                                                                                   (feasible, infeasible, unknown)
            Random.seed!(5)
            _, p1 = generate_problem(ref, 700, status, 21)
            Random.seed!(6)
            rand(7)
            _, p2 = generate_problem(ref, 700, status, 21)
            for f in fieldnames(typeof(p1))
                a, b = getfield(p1, f), getfield(p2, f)
                if a === nothing || a isa Union{Number, Symbol, AbstractArray, FeasibilityStatus}
                    @test isequal(a, b)
                else
                    @test all(isequal(getfield(a, g), getfield(b, g)) for g in fieldnames(typeof(a)))
                end
            end
        end
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            function lb_solve(m)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                return termination_status(m), MOI.get(m, MOI.SimplexIterations())
            end
            for ref in ("load_balancing/standard", "load_balancing/discrete_placement"), target in (150, 3000),
                seed in 0:2

                ts, _ = lb_solve(generate_problem(ref, target, feasible, seed)[1])
                @test ts == MOI.OPTIMAL
                ts, iters = lb_solve(generate_problem(ref, target, infeasible, seed)[1])
                @test ts == MOI.INFEASIBLE
                target >= 3000 && @test iters > 0
            end
            for ref in ("load_balancing/standard", "load_balancing/discrete_placement")
                outcomes = Set{Any}()
                for seed in 0:15
                    push!(outcomes, lb_solve(generate_problem(ref, 5000, unknown, seed)[1])[1])
                end
                @test outcomes == Set([MOI.OPTIMAL, MOI.INFEASIBLE])
            end
        end
    end
end
