# Focused quality contracts for the multi_commodity_flow category: registry
# shape, sizing formulas, network/commodity invariants, the planted-routing
# witnesses (checked arithmetically and against the built models), the
# metric-inequality certificates recomputed from scratch with an independent
# Bellman-Ford, the local-repair guarantee that keeps infeasibility out of
# presolve's reach, reproducibility, and HiGHS contracts.

# Independent shortest paths (Bellman-Ford) for certificate checks.
function mcf_bf_dist(n, arcs, lengths, src)
    d = fill(Inf, n)
    d[src] = 0.0
    for _ in 1:n
        changed = false
        for (k, (u, v)) in enumerate(arcs)
            if d[u] + lengths[k] < d[v] - 1e-12
                d[v] = d[u] + lengths[k]
                changed = true
            end
        end
        changed || break
    end
    return d
end

function mcf_requirement(p, lengths, origins, destinations, demands)
    total = 0.0
    for k in eachindex(origins)
        d = mcf_bf_dist(p.n_nodes, p.arcs, lengths, origins[k])
        total += sum(demands[k][i] * d[v] for (i, v) in enumerate(destinations[k]))
    end
    return total
end

@testset "Multi-Commodity Flow" begin
    @test Set(list_variants(:multi_commodity_flow)) == Set([:standard, :binary_capacity])
    @test problem_info(:multi_commodity_flow)[:default_variant] == :standard

    @testset "sizing" begin
        for target in (20, 200, 2000, 20_000), status in (feasible, infeasible, unknown)
            m, p = generate_problem("multi_commodity_flow/standard", target, status, 2)
            K, A = length(p.origins), length(p.arcs)
            @test K == clamp(round(Int, 0.4 * target^0.35), 2, 60)
            @test num_variables(m) == A * K
            target >= 200 && @test abs(num_variables(m) - target) <= K
            @test num_constraints(m; count_variable_in_set_constraints=false) == p.n_nodes * K + A

            m, p = generate_problem("multi_commodity_flow/binary_capacity", target, status, 2)
            K, A = length(p.origins), length(p.arcs)
            @test K == clamp(round(Int, 0.5 * target^0.3), 3, 40)
            @test num_variables(m) == A * (K + 3)
            target >= 200 && @test abs(num_variables(m) - target) <= K + 3
            @test num_constraints(m; count_variable_in_set_constraints=false) ==
                p.n_nodes * K + 2A + A * K
        end
        @test_throws ArgumentError SyntheticLPs.MultiCommodityFlow(SyntheticLPs.MCF_MAX_VARIABLES + 1, unknown, 0)
        @test_throws ArgumentError SyntheticLPs.BinaryCapacityMultiCommodityFlowProblem(0, unknown, 0)
    end

    @testset "network and commodity invariants" begin
        for seed in 0:2
            _, p = generate_problem("multi_commodity_flow/standard", 3000, unknown, seed)
            @test issorted(p.arcs) && allunique(p.arcs)
            @test allunique(p.origins)
            for k in eachindex(p.origins)
                @test issorted(p.destinations[k]) && allunique(p.destinations[k])
                @test !(p.origins[k] in p.destinations[k])
                @test length(p.demands[k]) == length(p.destinations[k])
                @test all(>(0.0), p.demands[k])
            end
            @test all(>(0.0), p.capacities)
            @test all(>(0.0), p.costs)
            # Commodities differ in value: per-commodity cost columns differ.
            @test size(p.costs, 2) == length(p.origins)
            @test any(p.costs[:, 1] .!= p.costs[:, 2])
            _, q = generate_problem("multi_commodity_flow/binary_capacity", 3000, unknown, seed)
            @test all(q.origins .!= q.destinations)
            @test all(issorted(q.module_capacity[a, :]) for a in eachindex(q.arcs))
            @test all(>(0.0), q.module_cost)
        end
    end

    @testset "planted witnesses" begin
        for target in (100, 1500), seed in 0:2
            m, p = generate_problem("multi_commodity_flow/standard", target, feasible, seed)
            F = p.feasible_witness.flows
            @test all(F .>= 0)
            @test all(vec(sum(F; dims=2)) .<= p.capacities .+ 1e-9)
            b = SyntheticLPs._mcf_supply_matrix(p.n_nodes, p.origins, p.destinations, p.demands)
            for k in eachindex(p.origins)
                net = zeros(p.n_nodes)
                for (a, (u, v)) in enumerate(p.arcs)
                    net[u] += F[a, k]
                    net[v] -= F[a, k]
                end
                @test maximum(abs.(net .- b[:, k])) < 1e-6
            end
            vals = Dict(m[:x][a, k] => F[a, k] for a in axes(F, 1), k in axes(F, 2))
            @test isempty(primal_feasibility_report(m, vals; atol=1e-6))

            m, q = generate_problem("multi_commodity_flow/binary_capacity", target, feasible, seed)
            w = q.feasible_witness
            @test all(in((0.0, 1.0)), w.install)
            vals = Dict{VariableRef, Float64}()
            for a in axes(w.flows, 1), k in axes(w.flows, 2)
                vals[m[:x][a, k]] = w.flows[a, k]
            end
            for a in axes(w.install, 1), mm in axes(w.install, 2)
                vals[m[:y][a, mm]] = w.install[a, mm]
            end
            @test isempty(primal_feasibility_report(m, vals; atol=1e-6))
        end
    end

    @testset "metric certificates" begin
        modes = Set{Symbol}()
        for target in (100, 1500), seed in 0:5
            _, p = generate_problem("multi_commodity_flow/standard", target, infeasible, seed)
            c = p.infeasibility_certificate
            push!(modes, c.mode)
            @test all(>=(0.0), c.lengths)
            @test c.capacity_length ≈ sum(p.capacities .* c.lengths)
            req = mcf_requirement(p, c.lengths, p.origins, p.destinations, p.demands)
            @test c.required ≈ req rtol = 1e-9
            @test c.capacity_length < c.required
            if c.mode == :regional_cut
                S = Set(c.region)
                @test c.lengths == [(u in S && !(v in S)) ? 1.0 : 0.0 for (u, v) in p.arcs]
            end
            # Local repair: no node is starved behind its own arcs.
            need_in = zeros(p.n_nodes)
            need_out = zeros(p.n_nodes)
            for k in eachindex(p.origins)
                need_out[p.origins[k]] += sum(p.demands[k])
                need_in[p.destinations[k]] .+= p.demands[k]
            end
            cap_in = zeros(p.n_nodes)
            cap_out = zeros(p.n_nodes)
            for (a, (u, v)) in enumerate(p.arcs)
                cap_out[u] += p.capacities[a]
                cap_in[v] += p.capacities[a]
            end
            # (On tiny networks the repair may be relaxed so the certificate
            # can separate at all; from medium sizes on it always holds.)
            if target >= 1500
                @test all(cap_in .>= need_in)
                @test all(cap_out .>= need_out)
            end
        end
        @test modes == Set([:length, :regional_cut])
        for target in (100, 1500), seed in 0:3
            _, q = generate_problem("multi_commodity_flow/binary_capacity", target, infeasible, seed)
            c = q.infeasibility_certificate
            maxcap = q.module_capacity[:, end]
            @test c.capacity_length ≈ sum(maxcap .* c.lengths)
            req = mcf_requirement(q, c.lengths, q.origins, [[d] for d in q.destinations], [[d] for d in q.demands])
            @test c.required ≈ req rtol = 1e-9
            @test c.capacity_length < c.required
        end
    end

    @testset "reproducibility" begin
        for ref in ("multi_commodity_flow/standard", "multi_commodity_flow/binary_capacity"), status in
                                                                                             (feasible, infeasible, unknown)
            Random.seed!(3)
            _, p1 = generate_problem(ref, 600, status, 11)
            Random.seed!(4)
            rand(10)
            _, p2 = generate_problem(ref, 600, status, 11)
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
            function mcf_solve(m)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                return termination_status(m), MOI.get(m, MOI.SimplexIterations())
            end
            for ref in ("multi_commodity_flow/standard", "multi_commodity_flow/binary_capacity"), target in
                                                                                                  (150, 3000),
                seed in 0:2

                ts, _ = mcf_solve(generate_problem(ref, target, feasible, seed)[1])
                @test ts == MOI.OPTIMAL
                ts, iters = mcf_solve(generate_problem(ref, target, infeasible, seed)[1])
                @test ts == MOI.INFEASIBLE
                target >= 3000 && @test iters > 0   # not refuted by presolve
            end
            for ref in ("multi_commodity_flow/standard", "multi_commodity_flow/binary_capacity")
                outcomes = Set{Any}()
                for seed in 0:15
                    push!(outcomes, mcf_solve(generate_problem(ref, 1000, unknown, seed)[1])[1])
                end
                @test outcomes == Set([MOI.OPTIMAL, MOI.INFEASIBLE])
            end
        end
    end
end
