# Focused quality contracts for the transportation category: registry shape,
# exact lane-count sizing and row formulas, sparse geographic lane invariants,
# witness arithmetic recomputed from the struct fields (and checked against
# the built model), certificate arithmetic (Hall/Gale regions, DC-split cuts,
# emission lower bound), reproducibility, and HiGHS contracts — including that
# infeasible instances need simplex work rather than being refuted by presolve.

function tp_row_count(m)
    return num_constraints(m; count_variable_in_set_constraints=false)
end

function tp_cor(x, y)
    mx, my = sum(x) / length(x), sum(y) / length(y)
    return sum((x .- mx) .* (y .- my)) / sqrt(sum((x .- mx) .^ 2) * sum((y .- my) .^ 2))
end

@testset "Transportation" begin
    @test Set(list_variants(:transportation)) ==
        Set([:standard, :transshipment, :emission_constrained, :fixed_charge])
    @test problem_info(:transportation)[:default_variant] == :standard
    @test occursin("lane", lowercase(problem_info(:transportation)[:description]))

    @testset "sizing" begin
        for target in (2, 7, 40, 300, 3000), status in (feasible, infeasible, unknown)
            m, p = generate_problem("transportation/standard", target, status, 1)
            @test num_variables(m) == length(p.lanes) == max(target, 2)
            @test tp_row_count(m) == p.n_sources + p.n_customers

            m, p = generate_problem("transportation/fixed_charge", target, status, 1)
            L = length(p.lanes)
            @test L == max(round(Int, target / 2), 2)
            @test num_variables(m) == 2L
            @test tp_row_count(m) == 2 * p.n_sources + p.n_customers + L

            m, p = generate_problem("transportation/emission_constrained", target, status, 1)
            @test num_variables(m) == length(p.options) == max(target, 2)
            n_rail = length(unique(p.lanes[l][1] for (l, mo) in p.options if mo == 2))
            @test tp_row_count(m) ==
                p.n_sources + p.n_customers + n_rail + length(p.region_cap) + 1

            m, p = generate_problem("transportation/transshipment", target, status, 1)
            @test num_variables(m) ==
                length(p.inbound) + length(p.outbound) + length(p.direct) == max(target, 6)
            @test tp_row_count(m) == p.n_plants + 2 * p.n_dcs + p.n_customers
        end
        # Rows scale with the instance (sparse lanes, many nodes).
        for ref in ("transportation/standard", "transportation/transshipment")
            m, _ = generate_problem(ref, 20_000, unknown, 0)
            @test tp_row_count(m) >= 0.1 * num_variables(m)
        end
        cap = SyntheticLPs.TRANSPORTATION_MAX_VARIABLES
        for T in (
            SyntheticLPs.TransportationProblem,
            SyntheticLPs.FixedChargeTransportationProblem,
            SyntheticLPs.EmissionConstrainedTransportationProblem,
            SyntheticLPs.TransshipmentProblem,
        )
            @test_throws ArgumentError T(cap + 1, unknown, 0)
            @test_throws ArgumentError T(0, unknown, 0)
        end
    end

    @testset "lane invariants" begin
        for ref in ("transportation/standard", "transportation/fixed_charge"), seed in 0:2
            _, p = generate_problem(ref, 2000, unknown, seed)
            @test issorted(p.lanes) && allunique(p.lanes)
            @test all(1 <= i <= p.n_sources && 1 <= j <= p.n_customers for (i, j) in p.lanes)
            deg = zeros(Int, p.n_customers)
            for (_, j) in p.lanes
                deg[j] += 1
            end
            @test all(>=(2), deg)
            @test all(>(0.0), p.supplies) && all(>(0.0), p.demands)
            # Every customer keeps an uncapped (primary) lane.
            uncapped = falses(p.n_customers)
            for (l, (_, j)) in enumerate(p.lanes)
                isinf(p.lane_capacity[l]) && (uncapped[j] = true)
            end
            @test all(uncapped)
            # Landed cost grows with distance.
            d = [
                hypot(
                    p.source_positions[i][1] - p.customer_positions[j][1],
                    p.source_positions[i][2] - p.customer_positions[j][2],
                ) for (i, j) in p.lanes
            ]
            c = ref == "transportation/standard" ? p.costs : p.unit_cost
            @test tp_cor(c, d) > 0.3
        end
        _, p = generate_problem("transportation/fixed_charge", 2000, unknown, 3)
        @test p.link_bound ≈
            [min(p.supplies[i], p.demands[j], p.lane_capacity[l]) for (l, (i, j)) in enumerate(p.lanes)]
        @test all(>=(1), p.max_lanes)
        _, p = generate_problem("transportation/emission_constrained", 2000, unknown, 3)
        @test issorted(p.options) && allunique(p.options)
        @test all(any(o == (l, 1) for o in p.options) for l in eachindex(p.lanes))  # truck always
        # Rail is cleaner than truck on the same lane once the haul is long
        # enough to amortise its drayage.
        e = Dict(p.options[k] => p.emission[k] for k in eachindex(p.options))
        lane_len(l) = hypot(
            p.source_positions[p.lanes[l][1]][1] - p.customer_positions[p.lanes[l][2]][1],
            p.source_positions[p.lanes[l][1]][2] - p.customer_positions[p.lanes[l][2]][2],
        )
        long_rail = [l for (l, mo) in p.options if mo == 2 && lane_len(l) >= 20]
        @test !isempty(long_rail)
        @test all(e[(l, 2)] < e[(l, 1)] for l in long_rail)
    end

    @testset "standard witness and region certificate" begin
        for target in (50, 600, 4000), seed in 0:2
            m, p = generate_problem("transportation/standard", target, feasible, seed)
            f = p.feasible_witness.flows
            @test all(-1e-9 .<= f .<= p.lane_capacity .+ 1e-9)
            out = zeros(p.n_sources)
            inn = zeros(p.n_customers)
            for (l, (i, j)) in enumerate(p.lanes)
                out[i] += f[l]
                inn[j] += f[l]
            end
            @test all(out .<= p.supplies .+ 1e-6)
            @test all(inn .>= p.demands .- 1e-6)
            @test isempty(primal_feasibility_report(m, Dict(m[:x][l] => f[l] for l in eachindex(f)); atol=1e-6))
        end
        multi = 0
        for target in (50, 600, 4000), seed in 0:3
            _, p = generate_problem("transportation/standard", target, infeasible, seed)
            c = p.infeasibility_certificate
            S = Set(c.sources)
            J = Set(c.customers)
            @test c.inbound_lanes == [l for (l, (i, j)) in enumerate(p.lanes) if !(i in S) && j in J]
            @test all(isfinite, p.lane_capacity[c.inbound_lanes])
            @test c.inbound_capacity ≈ sum(p.lane_capacity[c.inbound_lanes]; init=0.0)
            @test c.region_demand ≈ sum(p.demands[c.customers])
            @test c.region_supply ≈ sum(p.supplies[c.sources]; init=0.0)
            @test c.region_demand > c.region_supply + c.inbound_capacity
            @test p.max_flow_value < p.total_demand
            multi += length(c.customers) > 1
        end
        @test multi >= 10
    end

    @testset "fixed_charge witness and certificate" begin
        for target in (60, 800, 5000), seed in 0:2
            m, p = generate_problem("transportation/fixed_charge", target, feasible, seed)
            w = p.feasible_witness
            @test all(o in (0.0, 1.0) for o in w.open)  # an integer point of the MIP
            vals = Dict{VariableRef, Float64}()
            for l in eachindex(p.lanes)
                vals[m[:x][l]] = w.flows[l]
                vals[m[:y][l]] = w.open[l]
            end
            @test isempty(primal_feasibility_report(m, vals; atol=1e-6))
            @test all(w.flows .<= p.link_bound .* w.open .+ 1e-9)
        end
        for target in (60, 800, 5000), seed in 0:2
            _, p = generate_problem("transportation/fixed_charge", target, infeasible, seed)
            c = p.infeasibility_certificate
            S, J = Set(c.sources), Set(c.customers)
            @test c.inbound_lanes == [l for (l, (i, j)) in enumerate(p.lanes) if !(i in S) && j in J]
            @test c.region_demand > c.region_supply + c.inbound_capacity
        end
    end

    @testset "emission witness and lower-bound certificate" begin
        for target in (60, 800, 5000), seed in 0:2
            m, p = generate_problem("transportation/emission_constrained", target, feasible, seed)
            f = p.feasible_witness.flows
            @test isempty(primal_feasibility_report(m, Dict(m[:x][k] => f[k] for k in eachindex(f)); atol=1e-6))
            @test sum(p.emission .* f) <= p.global_cap + 1e-6
        end
        for target in (60, 800, 5000), seed in 0:2
            _, p = generate_problem("transportation/emission_constrained", target, infeasible, seed)
            c = p.infeasibility_certificate
            into = [Int[] for _ in 1:(p.n_customers)]
            for (k, (l, _)) in enumerate(p.options)
                push!(into[p.lanes[l][2]], k)
            end
            @test c.min_rate ≈ [minimum(p.emission[into[j]]) for j in 1:(p.n_customers)]
            @test c.lower_bound ≈ sum(p.demands .* c.min_rate)
            @test c.global_cap == p.global_cap
            @test p.global_cap < c.lower_bound
        end
    end

    @testset "transshipment witness and DC-split certificate" begin
        for target in (60, 800, 5000), seed in 0:2
            m, p = generate_problem("transportation/transshipment", target, feasible, seed)
            w = p.feasible_witness
            vals = Dict{VariableRef, Float64}()
            for l in eachindex(p.inbound)
                vals[m[:x_in][l]] = w.inbound[l]
            end
            for l in eachindex(p.outbound)
                vals[m[:x_out][l]] = w.outbound[l]
            end
            for l in eachindex(p.direct)
                vals[m[:x_dir][l]] = w.direct[l]
            end
            @test isempty(primal_feasibility_report(m, vals; atol=1e-6))
        end
        for target in (60, 800, 5000), seed in 0:2
            _, p = generate_problem("transportation/transshipment", target, infeasible, seed)
            c = p.infeasibility_certificate
            Pl, Hi, Ho, Cu = Set(c.plants), Set(c.dcs_in), Set(c.dcs_out), Set(c.customers)
            @test c.inbound_lanes == [l for (l, (q, h)) in enumerate(p.inbound) if !(q in Pl) && h in Hi]
            @test c.direct_lanes == [l for (l, (q, k)) in enumerate(p.direct) if !(q in Pl) && k in Cu]
            @test c.throughput_dcs == [h for h in 1:(p.n_dcs) if !(h in Hi) && h in Ho]
            # Uncapped outbound lanes never enter the region.
            @test !any(!(h in Ho) && k in Cu for (h, k) in p.outbound)
            entry = sum(p.inbound_capacity[c.inbound_lanes]; init=0.0) +
                sum(p.direct_capacity[c.direct_lanes]; init=0.0) +
                sum(p.throughput[c.throughput_dcs]; init=0.0)
            @test c.entry_capacity ≈ entry
            @test c.region_demand ≈ sum(p.demands[c.customers])
            @test c.region_demand > c.region_supply + c.entry_capacity
        end
    end

    @testset "reproducibility" begin
        for v in (:standard, :transshipment, :emission_constrained, :fixed_charge), status in
                                                                                     (feasible, infeasible, unknown)
            Random.seed!(1)
            _, p1 = generate_problem(ProblemVariant(:transportation, v), 500, status, 9)
            Random.seed!(2)
            rand(100)
            _, p2 = generate_problem(ProblemVariant(:transportation, v), 500, status, 9)
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
            function tp_solve(m)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                return termination_status(m), MOI.get(m, MOI.SimplexIterations())
            end
            for v in (:standard, :transshipment, :emission_constrained, :fixed_charge), target in
                                                                                       (80, 1500),
                seed in 0:2

                ref = ProblemVariant(:transportation, v)
                ts, _ = tp_solve(generate_problem(ref, target, feasible, seed)[1])
                @test ts == MOI.OPTIMAL
                ts, iters = tp_solve(generate_problem(ref, target, infeasible, seed)[1])
                @test ts == MOI.INFEASIBLE
                target >= 1500 && @test iters > 0
            end
            # standard/unknown is decided by the stored exact max flow.
            for seed in 0:7
                m, p = generate_problem("transportation/standard", 600, unknown, seed)
                ts, _ = tp_solve(m)
                @test ts ==
                    (p.max_flow_value >= p.total_demand * (1 - 1e-9) ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end
            # Every variant's `unknown` profile is genuinely two-sided.
            for v in (:standard, :transshipment, :emission_constrained, :fixed_charge)
                outcomes = Set{Any}()
                for seed in 0:15
                    m, _ = generate_problem(ProblemVariant(:transportation, v), 2000, unknown, seed)
                    push!(outcomes, tp_solve(m)[1])
                end
                @test outcomes == Set([MOI.OPTIMAL, MOI.INFEASIBLE])
            end
            # The unrelaxed fixed-charge MIP is feasible too (planted integer plan).
            m, _ = generate_problem("transportation/fixed_charge", 200, feasible, 4; relax_integer=false)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            set_time_limit_sec(m, 20.0)
            optimize!(m)
            @test primal_status(m) == MOI.FEASIBLE_POINT
        end
    end
end
