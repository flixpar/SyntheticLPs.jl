# Focused quality contracts for the cutting_stock category: registry shape,
# exact sizing and row formulas (rows grow with the instance), the shared
# sparse pattern enumerator (fit, distinctness, singles, near-linear scale),
# witness and certificate arithmetic recomputed from the struct fields,
# reproducibility, and HiGHS contracts with a genuinely mixed `unknown`.

cs_rows(m) = num_constraints(m; count_variable_in_set_constraints=false)

function cs_status(m)
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    optimize!(m)
    return termination_status(m)
end

function cs_check_patterns(pats, stock_lengths, piece_lengths)
    @test all(issorted(its) && allunique(its) for its in pats.items)
    @test all(all(>(0), c) for c in pats.counts)
    @test all(
        SyntheticLPs.cs_pattern_length(pats, j, piece_lengths) <= stock_lengths[pats.stock[j]] for
        j in 1:length(pats)
    )
    @test allunique(zip(pats.stock, pats.items, pats.counts))
    # Every (item, stock) pair that fits has its maximal single-item pattern.
    single = SyntheticLPs.cs_single_index(pats, stock_lengths, piece_lengths)
    for i in eachindex(piece_lengths), k in eachindex(stock_lengths)
        @test (single[i, k] > 0) == (piece_lengths[i] <= stock_lengths[k])
    end
end

@testset "Cutting Stock" begin
    @test :cutting_stock in list_categories()
    @test Set(list_variants(:cutting_stock)) == Set([:standard, :due_dates, :setup_cost, :arc_flow])
    @test problem_info(:cutting_stock)[:default_variant] == :standard

    @testset "standard" begin
        for target in (1, 2, 5, 40, 300, 2500), status in (feasible, infeasible, unknown)
            m, p = generate_problem("cutting_stock/standard", target, status, 1)
            @test num_variables(m) == length(p.patterns) == target
            n_stock, n_types = SyntheticLPs.cs_standard_dimensions(target)
            @test length(p.stock_lengths) == n_stock && length(p.piece_lengths) == n_types
            @test cs_rows(m) == n_types + n_stock
            cs_check_patterns(p.patterns, p.stock_lengths, p.piece_lengths)
            @test all(>(0), p.demands)
            @test all(>=(0), p.availability)
        end
        # Rows scale (~5% of columns) and the enumerator reaches 100k patterns
        # (the old one failed at 50k and had ~100 rows).
        _, big = generate_problem("cutting_stock/standard", 100_000, unknown, 0)
        @test length(big.patterns) == 100_000
        @test length(big.piece_lengths) + length(big.stock_lengths) >= 0.04 * 100_000
        @test sum(length, big.patterns.items) <= 7 * 100_000

        for target in (50, 600, 4000), seed in 0:2
            m, p = generate_problem("cutting_stock/standard", target, feasible, seed)
            w = p.feasible_witness
            @test w !== nothing && p.infeasibility_certificate === nothing
            per_stock = zeros(Int, length(p.stock_lengths))
            for i in eachindex(p.piece_lengths)
                j = w.pattern[i]
                @test p.patterns.items[j] == [i]
                @test p.patterns.counts[j][1] * w.usage[i] >= p.demands[i]
                per_stock[p.patterns.stock[j]] += w.usage[i]
            end
            @test all(per_stock .<= p.availability)
            point = Dict(v => 0.0 for v in all_variables(m))
            for i in eachindex(w.pattern)
                point[m[:x][w.pattern[i]]] += w.usage[i]
            end
            @test isempty(primal_feasibility_report(m, point; atol=1e-6))
        end
        for target in (50, 600, 4000), seed in 0:2
            _, p = generate_problem("cutting_stock/standard", target, infeasible, seed)
            c = p.infeasibility_certificate
            @test c !== nothing && p.feasible_witness === nothing
            @test c.demand_length ≈ sum(p.piece_lengths .* p.demands)
            @test c.supply_length ≈ sum(p.stock_lengths .* p.availability)
            @test c.demand_length >= 1.04 * c.supply_length
        end
    end

    @testset "due_dates" begin
        for target in (1, 3, 40, 500, 4000), status in (feasible, infeasible, unknown)
            m, p = generate_problem("cutting_stock/due_dates", target, status, 2)
            T, n_types, n_patterns, n_stock = SyntheticLPs.cs_due_dates_dimensions(target)
            @test p.n_periods == T && length(p.patterns) == n_patterns
            @test num_variables(m) == T * (n_patterns + n_types)
            target >= 40 && @test abs(num_variables(m) - target) < T
            used_stock = length(unique(p.patterns.stock))
            @test cs_rows(m) == T * (n_types + used_stock)
            cs_check_patterns(p.patterns, p.stock_lengths, p.piece_lengths)
            @test size(p.demands) == (n_types, T) && all(>=(0), p.demands)
            @test all(any(>(0), p.demands[i, :]) for i in 1:n_types)
        end
        _, big = generate_problem("cutting_stock/due_dates", 100_000, unknown, 0)
        @test abs(big.n_periods * (length(big.patterns) + length(big.piece_lengths)) - 100_000) < big.n_periods
        @test big.n_periods * length(big.piece_lengths) >= 0.05 * 100_000

        for target in (60, 800, 4000), seed in 0:2
            m, p = generate_problem("cutting_stock/due_dates", target, feasible, seed)
            w = p.feasible_witness
            @test w !== nothing
            T = p.n_periods
            point = Dict(v => 0.0 for v in all_variables(m))
            used = zeros(Int, length(p.stock_lengths), T)
            for i in eachindex(p.piece_lengths)
                j = w.pattern[i]
                @test p.patterns.items[j] == [i]
                inv = 0
                for t in 1:T
                    inv += p.patterns.counts[j][1] * w.usage[i, t] - p.demands[i, t]
                    @test inv >= 0
                    point[m[:x][j, t]] += w.usage[i, t]
                    point[m[:inventory][i, t]] = inv
                    used[p.patterns.stock[j], t] += w.usage[i, t]
                end
            end
            @test all(used .<= p.availability)
            @test isempty(primal_feasibility_report(m, point; atol=1e-6))
        end
        for target in (60, 800, 4000), seed in 0:2
            _, p = generate_problem("cutting_stock/due_dates", target, infeasible, seed)
            c = p.infeasibility_certificate
            t = c.period
            @test t == 1
            @test c.demand_length ≈ sum(p.piece_lengths .* vec(sum(p.demands[:, 1:t]; dims=2)))
            @test c.supply_length ≈ sum(p.stock_lengths .* vec(sum(p.availability[:, 1:t]; dims=2)))
            @test c.demand_length >= 1.04 * c.supply_length
        end
    end

    @testset "setup_cost" begin
        for target in (2, 9, 60, 700, 5000), status in (feasible, infeasible, unknown)
            m, p = generate_problem("cutting_stock/setup_cost", target, status, 3)
            Q, P, n_machines, n_stock, n_types = SyntheticLPs.cs_setup_dimensions(target)
            @test num_variables(m) == 2 * Q == 2 * length(p.pair_pattern)
            @test abs(num_variables(m) - target) <= 1
            @test length(p.patterns) == P && p.n_machines == n_machines
            @test sort(unique(p.pair_pattern)) == collect(1:P)      # every pattern runs somewhere
            @test allunique(zip(p.pair_pattern, p.pair_machine))
            @test cs_rows(m) ==
                Q + n_types + length(unique(p.patterns.stock)) + length(unique(p.pair_machine))
            @test all(p.link_bound .>= 1)
            cs_check_patterns(p.patterns, p.stock_lengths, p.piece_lengths)
        end
        for target in (60, 900, 5000), seed in 0:2
            m, p = generate_problem("cutting_stock/setup_cost", target, feasible, seed)
            w = p.feasible_witness
            point = Dict(v => 0.0 for v in all_variables(m))
            for i in eachindex(w.pair)
                q = w.pair[i]
                j = p.pair_pattern[q]
                @test p.patterns.items[j] == [i]
                @test p.patterns.counts[j][1] * w.runs[i] >= p.demands[i]
                @test w.runs[i] <= p.link_bound[q]
                point[m[:x][q]] += w.runs[i]
                point[m[:y][q]] = 1.0
            end
            @test isempty(primal_feasibility_report(m, point; atol=1e-6))
        end
        for target in (60, 900, 5000), seed in 0:2
            _, p = generate_problem("cutting_stock/setup_cost", target, infeasible, seed)
            c = p.infeasibility_certificate
            @test c.demand_length ≈ sum(p.piece_lengths .* p.demands)
            @test c.supply_length ≈ sum(p.stock_lengths .* p.availability)
            @test c.demand_length >= 1.04 * c.supply_length
        end
    end

    @testset "arc_flow" begin
        for target in (1, 50, 400, 3000, 20_000), status in (feasible, infeasible, unknown)
            m, p = generate_problem("cutting_stock/arc_flow", target, status, 4)
            A = length(p.arc_tail)
            @test num_variables(m) == A
            target >= 400 && @test abs(A - target) <= 0.1 * target
            @test issorted(p.piece_lengths; rev=true) && allunique(p.piece_lengths)
            L = p.stock_length
            # Item arcs have the item's length; loss arcs join consecutive nodes.
            @test all(
                p.arc_item[a] > 0 ? p.arc_head[a] - p.arc_tail[a] == p.piece_lengths[p.arc_item[a]] :
                p.arc_tail[a] > 0 for a in 1:A
            )
            @test all(0 <= p.arc_tail[a] < p.arc_head[a] <= L for a in 1:A)
            nodes = setdiff(union(Set(p.arc_tail), Set(p.arc_head)), Set([0, L]))
            @test cs_rows(m) == length(nodes) + length(p.piece_lengths) + 1
        end
        _, big = generate_problem("cutting_stock/arc_flow", 100_000, unknown, 0)
        @test abs(length(big.arc_tail) - 100_000) <= 10_000
        @test_throws ArgumentError SyntheticLPs.ArcFlowCuttingStockProblem(1_000_001, unknown, 0)

        for target in (60, 800, 5000), seed in 0:2
            m, p = generate_problem("cutting_stock/arc_flow", target, feasible, seed)
            w = p.feasible_witness
            f = w.flow
            L = p.stock_length
            net = Dict{Int, Int}()
            for a in eachindex(f)
                net[p.arc_tail[a]] = get(net, p.arc_tail[a], 0) - f[a]
                net[p.arc_head[a]] = get(net, p.arc_head[a], 0) + f[a]
            end
            @test all(v == 0 for (u, v) in net if u != 0 && u != L)
            @test -net[0] == sum(w.rolls) <= p.stock_limit
            for i in eachindex(p.piece_lengths)
                @test sum(f[a] for a in eachindex(f) if p.arc_item[a] == i) >= p.demands[i]
            end
            point = Dict(m[:flow][a] => Float64(f[a]) for a in eachindex(f))
            @test isempty(primal_feasibility_report(m, point; atol=1e-6))
        end
        for target in (60, 800, 5000), seed in 0:2
            _, p = generate_problem("cutting_stock/arc_flow", target, infeasible, seed)
            c = p.infeasibility_certificate
            @test c.demand_length ≈ sum(p.piece_lengths .* p.demands)
            @test c.supply_length ≈ p.stock_length * p.stock_limit
            @test c.demand_length >= 1.04 * c.supply_length
        end
    end

    # Reproducibility, independent of the global RNG.
    for v in ("standard", "due_dates", "setup_cost", "arc_flow"), status in (feasible, infeasible, unknown)
        Random.seed!(3)
        _, p1 = generate_problem("cutting_stock/$v", 900, status, 21)
        Random.seed!(77)
        _, p2 = generate_problem("cutting_stock/$v", 900, status, 21)
        for f in fieldnames(typeof(p1))
            a, b = getfield(p1, f), getfield(p2, f)
            if a isa SyntheticLPs.CSPatterns
                @test a.stock == b.stock && a.items == b.items && a.counts == b.counts
            elseif a isa Union{Number, AbstractArray, FeasibilityStatus, Nothing}
                @test isequal(a, b)
            end
        end
    end

    @testset "Cutting Stock HiGHS contracts" begin
        if HAS_HIGHS
            for v in ("standard", "due_dates", "setup_cost", "arc_flow")
                for target in (100, 1500), status in (feasible, infeasible), seed in 0:1
                    m, _ = generate_problem("cutting_stock/$v", target, status, seed)
                    @test cs_status(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
                end
                outcomes = Set{MOI.TerminationStatusCode}()
                for target in (300, 2000), seed in 0:9
                    m, _ = generate_problem("cutting_stock/$v", target, unknown, seed)
                    push!(outcomes, cs_status(m))
                end
                @test MOI.OPTIMAL in outcomes
                @test MOI.INFEASIBLE in outcomes
            end
        end
    end
end
