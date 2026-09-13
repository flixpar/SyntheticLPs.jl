# Focused quality contracts for the cutting_stock/standard variant: registry
# shape, exact pattern-count sizing, pattern fitness and single-piece-pattern
# structure, integer witness and stock-shortage certificate arithmetic
# recomputed from the struct fields, no uncoverable demand rows,
# reproducibility, and HiGHS feasibility contracts.
@testset "Cutting Stock Standard" begin
    @test :cutting_stock in list_categories()
    @test Set(list_variants(:cutting_stock)) ==
        Set([:standard, :due_dates, :integer_patterns, :setup_cost])
    info = problem_info(:cutting_stock)
    @test info[:default_variant] == :standard
    @test occursin("cutting", lowercase(info[:description]))

    # Exact sizing: variables are the generated patterns, one per request, and
    # the model carries one demand row per piece type plus the stock-limit row.
    for target in (50, 100, 200, 500, 1000), status in (feasible, infeasible, unknown), seed in 0:2
        m, p = generate_problem("cutting_stock/standard", target, status, seed)
        @test num_variables(m) == length(p.patterns) == target
        @test num_constraints(m; count_variable_in_set_constraints=false) ==
            length(p.piece_lengths) + 1
        @test abs(num_variables(m) - target) <= 0.25 * target || num_variables(m) <= 50
    end

    # Structural data contracts shared by all three profiles: patterns fit the
    # stock, demands are positive, every piece type owns a leading single-piece
    # pattern (so no demand row can be all-zero), and the stock limit is a
    # finite positive budget.
    for target in (60, 400, 1500), status in (feasible, infeasible, unknown), seed in 0:2
        m, p = generate_problem("cutting_stock/standard", target, status, seed)
        n_pieces = length(p.piece_lengths)
        n_patterns = length(p.patterns)
        @test length(p.demands) == n_pieces
        @test all(length.(p.patterns) .== n_pieces)
        @test all(>(0), p.demands)
        @test all(>=(1), p.stock_limit)
        for pattern in p.patterns
            @test dot(pattern, p.piece_lengths) <= p.stock_length + 1e-9
            @test sum(pattern) > 0
            @test all(>=(0), pattern)
        end
        # The first n_pieces patterns are the direct single-piece patterns, in
        # piece order -- the anchor both the witness mapping and the yield
        # bounds rely on.
        for i in 1:n_pieces
            @test p.patterns[i][i] == floor(Int, p.stock_length / p.piece_lengths[i])
            @test sum(p.patterns[i]) == p.patterns[i][i]
        end
        # No degenerate demand rows: every piece type has a production route,
        # and the stock row has all-unit coefficients.
        @test all(any(p.patterns[j][i] > 0 for j in 1:n_patterns) for i in 1:n_pieces)
        @test p.stock_length > 0
        @test all(0.1 .<= p.piece_lengths .<= p.stock_length)
        @test allunique(p.piece_lengths)
    end

    # Scenario flavor: the feasible profile narrates a steady market, the
    # infeasible one a demand-shock regime, and unknown a capacity review.
    for status in (feasible, infeasible, unknown)
        _, p = generate_problem("cutting_stock/standard", 300, status, 5)
        if status == feasible
            @test p.scenario == :steady_demand
        elseif status == infeasible
            @test p.scenario in (:rush_order, :seasonal_spike, :backlog_clearing, :mixed)
        else
            @test p.scenario == :capacity_review
        end
    end

    # Plan witness: the trivial per-piece plan meets every demand row in exact
    # integer arithmetic and lives under the stock limit; the same point is
    # primal-feasible for the built model.
    for target in (50, 300, 2000), seed in 0:2
        m, p = generate_problem("cutting_stock/standard", target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing
        @test p.infeasibility_certificate === nothing
        n_pieces = length(p.piece_lengths)
        @test length(w.usage) == n_pieces
        for i in 1:n_pieces
            s_i = p.patterns[i][i]
            @test w.usage[i] == cld(p.demands[i], s_i)
            @test s_i * w.usage[i] >= p.demands[i]          # exact Int arithmetic
        end
        @test w.total_stock == sum(w.usage)
        @test w.total_stock <= p.stock_limit
        # The budget is generous but finite: 1.3-1.8x the direct plan's usage
        # (the small slack window accounts for integer rounding).
        ratio = p.stock_limit / w.total_stock
        @test 1.25 <= ratio <= 1.85

        # The same plan, expanded over pattern variables, is primal-feasible.
        point = Dict(m[:x][j] => 0.0 for j in 1:length(p.patterns))
        for i in 1:n_pieces
            point[m[:x][i]] = w.usage[i]
        end
        atol = 1e-6 * max(p.stock_limit, maximum(p.demands))
        @test isempty(primal_feasibility_report(m, point; atol=atol))
    end

    # Stock-shortage certificate: the bottleneck piece's demand exceeds what
    # the entire stock budget could produce of it even under the best pattern.
    # Recomputed from the raw fields, including the yield bound itself.
    for target in (50, 300, 2000), seed in 0:3
        _, p = generate_problem("cutting_stock/standard", target, infeasible, seed)
        c = p.infeasibility_certificate
        @test c !== nothing
        @test p.feasible_witness === nothing
        @test 1 <= c.piece_index <= length(p.piece_lengths)
        @test c.max_yield_per_stock ==
            maximum(p.patterns[j][c.piece_index] for j in 1:length(p.patterns))
        @test c.max_yield_per_stock >= p.patterns[c.piece_index][c.piece_index] >= 1
        @test c.stock_limit == p.stock_limit
        @test c.demand == p.demands[c.piece_index]
        # The refutation: e_i * sum(x) <= e_i * S < d_i for any x >= 0. The
        # margin (>= 1.19 in practice) keeps it robust to solver tolerances.
        @test c.demand > c.stock_limit * c.max_yield_per_stock
        @test c.demand >= 1.19 * c.stock_limit * c.max_yield_per_stock
        # The contradiction involves the aggregate stock row, never a missing
        # production route: the certified piece still has usable patterns.
        @test any(p.patterns[j][c.piece_index] > 0 for j in 1:length(p.patterns))
    end

    # The `unknown` profile asserts nothing: no witness, no certificate, but
    # the same structural guarantees as the committed profiles.
    for seed in 0:2
        _, p = generate_problem("cutting_stock/standard", 250, unknown, seed)
        @test p.feasible_witness === nothing
        @test p.infeasibility_certificate === nothing
        @test p.stock_limit >= 1
    end

    # Reproducibility and global-RNG isolation: identical seeds produce
    # field-identical structs even with a seeded/dirty global RNG. The witness
    # and certificate are plain structs (identity `==`), so their contents are
    # compared field-wise, as with the product_mix witness.
    for status in (feasible, infeasible, unknown)
        Random.seed!(987)
        _, p1 = generate_problem("cutting_stock/standard", 220, status, 42)
        Random.seed!(12345)
        _, p2 = generate_problem("cutting_stock/standard", 220, status, 42)
        for f in (
            :stock_length,
            :piece_lengths,
            :demands,
            :patterns,
            :stock_limit,
            :scenario,
            :feasibility_status,
        )
            @test isequal(getfield(p1, f), getfield(p2, f))
        end
        if p1.feasible_witness !== nothing
            @test p1.feasible_witness.usage == p2.feasible_witness.usage
            @test p1.feasible_witness.total_stock == p2.feasible_witness.total_stock
        end
        if p1.infeasibility_certificate !== nothing
            c1, c2 = p1.infeasibility_certificate, p2.infeasibility_certificate
            @test c1.piece_index == c2.piece_index
            @test c1.max_yield_per_stock == c2.max_yield_per_stock
            @test c1.stock_limit == c2.stock_limit
            @test c1.demand == c2.demand
        end
    end

    if HAS_HIGHS
        # End-to-end feasibility contract across scales and seeds, both on the
        # directly returned models and through the `optimizer` backstop (which
        # raises on any contract violation).
        for target in (50, 200, 1000), status in (feasible, infeasible), seed in 0:4
            m, _ = generate_problem("cutting_stock/standard", target, status, seed)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end
        for target in (100, 700), status in (feasible, infeasible), seed in 0:2
            generate_problem(
                "cutting_stock/standard", target, status, seed; optimizer=HiGHS.Optimizer
            )
        end

        # `unknown` is a genuine mix, not an implicit always-one-way branch.
        for target in (100, 500, 2000)
            outcomes = Set{MOI.TerminationStatusCode}()
            for seed in 0:9
                m, _ = generate_problem("cutting_stock/standard", target, unknown, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                push!(outcomes, termination_status(m))
            end
            @test MOI.OPTIMAL in outcomes
            @test MOI.INFEASIBLE in outcomes
        end
    end
end
