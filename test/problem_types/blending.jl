# Focused quality contracts for the blending category (secondary-aluminium
# alloy blending): registry shape, exact sizing, catalog/assay invariants, the
# planted charge recipes checked row by row without a solver, the
# family/element shortage and cumulative melt-capacity certificates recomputed
# from the data, the exact budgeted-robust protection, reproducibility /
# global-RNG isolation, and HiGHS-backed status contracts.
@testset "Blending" begin
    @test :blending in list_categories()
    @test Set(list_variants(:blending)) == Set([:standard, :multi_period, :robust])
    @test problem_info(:blending)[:default_variant] == :standard
    S = SyntheticLPs

    @testset "catalog and assays" begin
        # Registered windows are windows; every grade can take primary metal.
        for g in S._BLEND_GRADES
            @test all(g.lo[e] <= g.hi[e] for e in eachindex(S.BLEND_ELEMENTS))
        end
        @test S._blend_assay(0.004) == 0.0
        @test S._blend_assay(0.123456) == 0.123
        comp, cost = S._blend_lot(MersenneTwister(1), 3)
        @test all(c -> c == 0 || c >= 0.005, comp) && cost > 0
        # The covering-knapsack bound is exact on a tiny case.
        @test S._blend_max_excess([1.0, -1.0], [1.0, 1.0], [2.0, 10.0], 5.0) ≈ 2.0 - 3.0
        @test S._blend_max_excess([1.0], [1.0], [1.0], 5.0) == -Inf
    end

    @testset "standard: sizing, recipes, certificates" begin
        for target in (10, 50, 300, 2_000, 12_000),
            status in (feasible, infeasible, unknown),
            seed in 0:2

            model, p = generate_problem("blending/standard", target, status, seed)
            n = length(p.pairs) + length(p.order_grade)
            @test num_variables(model) == n
            @test n >= target && n <= target + 60
            # Pairs respect plant yards and compatibility rules.
            @test all(p.materials.site[i] in (0, p.order_plant[o]) for (i, o) in p.pairs)
            @test all(S.blend_compatible(p.materials, i, p.order_grade[o]) for (i, o) in p.pairs)
            @test all(p.demand_min .< p.demand_max)
            if status == feasible
                @test S.blend_charge_satisfies(p)
            elseif status == infeasible
                @test S.blend_certificate_holds(p)
                c = p.infeasibility_certificate
                if c.kind == S.blend_family_shortage
                    @test c.achievable <= c.required / 1.08 + 1e-9
                else
                    @test c.achievable < 0
                end
            end
        end
        _, p = generate_problem("blending/standard", 800, feasible, 3)
        @test !S.blend_charge_satisfies(p, p.feasible_witness .* 3.0)
        big = S.BlendingProblem(100_000, unknown, 0)
        @test 100_000 <= length(big.pairs) + length(big.order_grade) <= 100_100
    end

    @testset "multi_period: sizing, plan, certificate" begin
        for target in (30, 50, 300, 2_000, 12_000),
            status in (feasible, infeasible, unknown),
            seed in 0:2

            model, p = generate_problem("blending/multi_period", target, status, seed)
            nI = length(p.materials.kind)
            grade_slots = sum(length(p.portfolio[k]) for k in 1:p.n_plants) * p.n_periods
            n = length(p.blend_vars) + 2 * grade_slots + 2 * nI * p.n_plants * p.n_periods
            @test num_variables(model) == n
            @test abs(n - target) <= max(0.15 * target, 30)
            if status == feasible
                @test S.mp_plan_satisfies(p)
            elseif status == infeasible
                @test S.mp_certificate_holds(p)
                c = p.infeasibility_certificate
                @test c.achievable <= c.required / 1.08 + 1e-9
            end
        end
        _, p = generate_problem("blending/multi_period", 900, feasible, 1)
        w = p.feasible_witness
        @test !S.mp_plan_satisfies(
            p, S.MultiPeriodBlendingPlan(w.blend, w.charge, w.buy .* 0.5, w.stock, w.finished)
        )
    end

    @testset "robust: sizing, protection, plan" begin
        # Budgeted protection: top-Γ deviations with a fractional remainder.
        beta, z, pr = S._robust_protection([5.0, 1.0, 3.0], 1.5)
        @test beta ≈ 5.0 + 0.5 * 3.0
        @test z ≈ 3.0 && pr ≈ [2.0, 0.0, 0.0]
        @test 1.5 * z + sum(pr) ≈ beta
        for target in (30, 50, 300, 2_000, 12_000),
            status in (feasible, infeasible, unknown),
            seed in 0:2

            model, p = generate_problem("blending/robust", target, status, seed)
            n =
                length(p.pairs) +
                length(p.order_grade) +
                length(p.robust_rows) +
                length(p.robust_terms)
            @test num_variables(model) == n
            @test n >= target
            @test all(p.materials.kind[p.pairs[k][1]] == :scrap for (_, k) in p.robust_terms)
            @test all(e in S.BLEND_UNCERTAIN_ELEMENTS for (_, e) in p.robust_rows)
            if status == feasible
                @test S.robust_plan_satisfies(p)
            elseif status == infeasible
                @test S.robust_certificate_holds(p)
            end
        end
        _, p = generate_problem("blending/robust", 900, feasible, 2)
        w = p.feasible_witness
        @test !S.robust_plan_satisfies(
            p, S.RobustBlendPlan(w.charge_plan, zero(w.protection), zero(w.excess))
        )
    end

    @testset "reproducibility and global-RNG isolation" begin
        for ref in ("blending/standard", "blending/multi_period", "blending/robust")
            _, p1 = generate_problem(ref, 600, feasible, 31)
            _, p2 = generate_problem(ref, 600, feasible, 31)
            for f in fieldnames(typeof(p1))
                a, b = getfield(p1, f), getfield(p2, f)
                if a isa S.BlendMaterials ||
                    a isa S.RobustBlendPlan ||
                    a isa S.MultiPeriodBlendingPlan
                    @test all(
                        isequal(getfield(a, g), getfield(b, g)) for g in fieldnames(typeof(a))
                    )
                else
                    @test isequal(a, b)
                end
            end
            Random.seed!(777)
            expected = rand()
            Random.seed!(777)
            generate_problem(ref, 400, infeasible, 5)
            @test rand() == expected
        end
    end

    @testset "HiGHS feasibility contracts" begin
        if HAS_HIGHS
            for ref in ("blending/standard", "blending/multi_period", "blending/robust"),
                target in (60, 600),
                seed in 0:3

                model, _ = generate_problem(ref, target, feasible, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                @test termination_status(model) == MOI.OPTIMAL
                # HiGHS's dual simplex occasionally ends infeasible blending LPs
                # with OTHER_ERROR (cleanup after cost perturbation); a proven
                # infeasibility is required, any optimal answer is a failure.
                model, _ = generate_problem(ref, target, infeasible, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                @test termination_status(model) in
                    (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED, MOI.OTHER_ERROR)
                @test termination_status(model) != MOI.OPTIMAL
            end
            # The framework backstop agrees with the planted contract.
            for ref in ("blending/standard", "feed_blending/standard"), s in 1:3
                m, _ = generate_problem(ref, 300, infeasible, s; optimizer=HiGHS.Optimizer)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
            end
        end
    end
end
