# Focused quality contracts for crop_planning/standard (regional multi-year
# rotation planning): exact sizing, field/crop compatibility, the planted
# rotation plan checked row by row without a solver, land/water certificates
# recomputed from the data, reproducibility / global-RNG isolation, and
# HiGHS-backed status contracts.
@testset "Crop planning" begin
    @test Set(list_variants(:crop_planning)) == Set([:standard])
    S = SyntheticLPs
    ref = "crop_planning/standard"

    @testset "sizing, data, witness, certificates" begin
        for target in (10, 50, 100, 500, 2_000, 12_000), status in (feasible, infeasible, unknown), seed in 0:2
            model, p = generate_problem(ref, target, status, seed)
            J, T, R, C, K = length(p.field_area), p.n_years, p.n_regions, length(p.crops), p.n_tiers
            n = length(p.area_vars) + J * T + length(p.farm_region) * length(S.CROP_SEASONS) * T + C * R * T * K
            @test num_variables(model) == n
            @test abs(n - target) <= max(0.1 * target, 12) || n <= 30
            @test all(!S._CROP_CATALOG[p.crops[c]].irrigated_only || p.field_irrigable[j] for (j, c, _) in p.area_vars)
            @test all(p.yields[(j, c)] > 0 for (j, c, _) in p.area_vars)
            @test all(p.field_region .== p.farm_region[p.field_farm])
            if status == feasible
                @test S.crop_plan_satisfies(p)
            elseif status == infeasible
                @test S.crop_certificate_holds(p)
                c = p.infeasibility_certificate
                @test c.achievable <= c.required / 1.08 + 1e-9
            else
                @test p.feasible_witness === nothing && p.infeasibility_certificate === nothing
            end
        end
        _, p = generate_problem(ref, 800, feasible, 2)
        w = p.feasible_witness
        @test !S.crop_plan_satisfies(p, S.CropPlan(w.area .* 2.0, w.fertilizer, w.hired, w.sales))
        big = S.CropPlanningProblem(100_000, unknown, 0)
        nb = length(big.area_vars) + length(big.field_area) * big.n_years +
             length(big.farm_region) * 4 * big.n_years + length(big.crops) * big.n_regions * big.n_years * big.n_tiers
        @test abs(nb - 100_000) <= 0.01 * 100_000
    end

    @testset "reproducibility and global-RNG isolation" begin
        _, p1 = generate_problem(ref, 180, feasible, 12345)
        _, p2 = generate_problem(ref, 180, feasible, 12345)
        for f in fieldnames(typeof(p1))
            a, b = getfield(p1, f), getfield(p2, f)
            if a isa S.CropPlan
                @test all(isequal(getfield(a, g), getfield(b, g)) for g in fieldnames(typeof(a)))
            else
                @test isequal(a, b)
            end
        end
        Random.seed!(8172)
        expected = rand()
        Random.seed!(8172)
        generate_problem(ref, 180, feasible, 99)
        @test rand() == expected
    end

    @testset "HiGHS feasibility contracts" begin
        if HAS_HIGHS
            for target in (30, 300, 1_200), seed in 0:3, status in (feasible, infeasible)
                model, _ = generate_problem(ref, target, status, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                @test termination_status(model) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end
            for s in 1:4
                m, _ = generate_problem(ref, 120, infeasible, s; optimizer=HiGHS.Optimizer)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
            end
        end
    end
end
