# Focused quality contracts for feed_blending/standard (a network of feed mills
# with formula books): exact sizing, species exclusions and inclusion caps,
# the planted formulation checked row by row without a solver, the
# fractional-knapsack and mill-stock certificates recomputed from the data,
# reproducibility / global-RNG isolation, and HiGHS-backed status contracts.
@testset "Feed Blending" begin
    @test Set(list_variants(:feed_blending)) == Set([:standard])
    S = SyntheticLPs
    ref = "feed_blending/standard"

    @testset "knapsack bound" begin
        @test S._feed_knapsack_max([3.0, 1.0, 2.0], [1.0, 5.0, 1.0], 2.5) ≈ 3.0 + 2.0 + 0.5
        @test S._feed_knapsack_max([1.0], [1.0], 2.0) == -Inf
    end

    @testset "sizing, data, witness, certificates" begin
        for target in (10, 50, 300, 2_000, 12_000), status in (feasible, infeasible, unknown), seed in 0:2
            model, p = generate_problem(ref, target, status, seed)
            @test num_variables(model) == length(p.pairs)
            @test abs(length(p.pairs) - target) <= max(0.05 * target, 8) || length(p.pairs) <= 30
            # Species rules: no animal protein for ruminants, urea only for them.
            class(l) = S._FEED_INGREDIENTS[p.lot_ingredient[l]].class
            group(f) = S._FEED_FORMULAS[p.formula_type[f]].group
            @test all(!(group(f) == :ruminant && class(l) == :animal) for (l, f) in p.pairs)
            @test all(class(l) != :npn || group(f) == :ruminant for (l, f) in p.pairs)
            @test all(p.lot_mill[l] == p.formula_mill[f] for (l, f) in p.pairs)
            @test all(0 <= p.lower[k] <= p.upper[k] <= p.batch[f] + 1e-9 for (k, (_, f)) in enumerate(p.pairs))
            @test all(p.ratio_band[1, :] .< p.ratio_band[2, :])
            # Contracts left in the model couple at least two lots.
            for s in eachindex(p.contract)
                isfinite(p.contract[s]) && @test count(==(s), p.lot_supplier) >= 2
            end
            if status == feasible
                @test S.feed_formulation_satisfies(p)
            elseif status == infeasible
                @test S.feed_certificate_holds(p)
                c = p.infeasibility_certificate
                @test c.achievable <= c.required / 1.05
            else
                @test p.feasible_witness === nothing && p.infeasibility_certificate === nothing
            end
        end
        _, p = generate_problem(ref, 700, feasible, 5)
        bad = copy(p.feasible_witness)
        bad[first(p.formula_pairs[1])] += 1.0
        @test !S.feed_formulation_satisfies(p, bad)
        big = S.FeedBlendingProblem(100_000, unknown, 0)
        @test abs(length(big.pairs) - 100_000) <= 10
    end

    @testset "reproducibility and global-RNG isolation" begin
        _, p1 = generate_problem(ref, 500, infeasible, 1234)
        _, p2 = generate_problem(ref, 500, infeasible, 1234)
        for f in fieldnames(typeof(p1))
            a, b = getfield(p1, f), getfield(p2, f)
            if a isa S.FeedInfeasibilityCertificate
                @test all(isequal(getfield(a, g), getfield(b, g)) for g in fieldnames(typeof(a)))
            else
                @test isequal(a, b)
            end
        end
        Random.seed!(91_733)
        expected = rand()
        Random.seed!(91_733)
        generate_problem(ref, 80, feasible, 17)
        @test rand() == expected
    end

    @testset "HiGHS feasibility contracts" begin
        if HAS_HIGHS
            for target in (60, 800), seed in 0:4, status in (feasible, infeasible)
                model, _ = generate_problem(ref, target, status, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                @test termination_status(model) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end
            outcomes = Set{MOI.TerminationStatusCode}()
            for seed in 0:19
                model, _ = generate_problem(ref, 400, unknown, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                push!(outcomes, termination_status(model))
            end
            @test MOI.OPTIMAL in outcomes && MOI.INFEASIBLE in outcomes
        end
    end
end
