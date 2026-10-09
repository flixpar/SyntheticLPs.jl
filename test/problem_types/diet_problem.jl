# Focused quality contracts for the diet_problem category: registry shape,
# exact sizing formulas of the three structurally different variants (cohort
# diet, menu planning with perishable inventory, food-aid sourcing network),
# role-correlated food data, planted witnesses checked row by row without a
# solver, LP-row infeasibility certificates recomputed from the data,
# reproducibility / global-RNG isolation, and HiGHS-backed status contracts.
@testset "Diet Problem" begin
    @test :diet_problem in list_categories()
    @test Set(list_variants(:diet_problem)) == Set([:standard, :food_groups, :food_aid])
    @test problem_info(:diet_problem)[:default_variant] == :standard

    S = SyntheticLPs

    @testset "food table is role-correlated" begin
        table = S._diet_sample_food_table(MersenneTwister(3), 400)
        C = table.content
        # Energy follows the macronutrients (Atwater 4/9/4, ±4% noise).
        atwater = 4 .* C[S.DIET_PROTEIN, :] .+ 9 .* C[S.DIET_FAT, :] .+ 4 .* C[S.DIET_CARB, :]
        @test all(abs.(C[S.DIET_ENERGY, :] .- atwater) .<= 0.05 .* atwater .+ 5.0)
        @test all(C[S.DIET_SATFAT, :] .<= C[S.DIET_FAT, :] .+ 1e-12)
        @test all(C[S.DIET_SUGAR, :] .<= C[S.DIET_CARB, :] .+ 1e-12)
        # Profiles cluster by category: meat is protein-dense, fruit is not.
        meat = findall(==(5), table.category)
        fruit = findall(==(3), table.category)
        @test sum(C[S.DIET_PROTEIN, meat]) / length(meat) >
            5 * sum(C[S.DIET_PROTEIN, fruit]) / length(fruit)
        # Vitamin D is carried by few categories (fish, dairy, eggs, fortified).
        @test count(>(0), C[16, :]) < 0.5 * size(C, 2)
        @test all(>(0), table.cost) && all(>(0), table.max_servings)
    end

    @testset "standard: sizing, witness, certificates" begin
        for target in (8, 50, 300, 2_000, 12_000),
            status in (feasible, infeasible, unknown),
            seed in 0:2

            model, p = generate_problem("diet_problem/standard", target, status, seed)
            @test num_variables(model) == p.n_foods * p.n_cohorts
            @test abs(num_variables(model) - target) <= max(p.n_cohorts / 2, 8)
            rows =
                p.n_cohorts *
                (3 + length(p.min_nutrients) + p.has_sugar_limit + 2 * p.has_fat_band) +
                count(isfinite, p.supply)
            @test num_constraints(model; count_variable_in_set_constraints=false) == rows
            @test p.min_nutrients[1:2] == [S.DIET_PROTEIN, S.DIET_FIBER]
            @test all(p.energy_band[1, :] .< p.energy_band[2, :])
            @test p.feasibility_status == status
            if status == feasible
                @test S.diet_plan_satisfies(p)
                @test p.infeasibility_certificate === nothing
                # Requirements are DRIs lowered only where the planted diet
                # falls short, so most rows sit at the reference values.
                @test all(p.min_requirement .> 0)
            elseif status == infeasible
                @test p.feasible_witness === nothing
                @test S.diet_certificate_holds(p)
                c = p.infeasibility_certificate
                @test c.achievable <= c.required / 1.05
            else
                @test p.feasible_witness === nothing && p.infeasibility_certificate === nothing
            end
        end
        # A perturbed witness is rejected (the checker is not vacuous).
        _, p = generate_problem("diet_problem/standard", 400, feasible, 4)
        bad = copy(p.feasible_witness)
        bad .*= 1.6
        @test !S.diet_plan_satisfies(p, bad)
        # Large target: sizing only (constructor), bounded nonzeros per column.
        big = S.DietProblem(100_000, feasible, 1)
        @test abs(big.n_foods * big.n_cohorts - 100_000) <= big.n_cohorts / 2
        @test big.n_foods <= 200
        @test count(>(0), big.content) / big.n_foods <= length(S.DIET_NUTRIENTS)
    end

    @testset "food_groups: sizing, witness, certificates" begin
        for target in (20, 50, 300, 2_000, 12_000),
            status in (feasible, infeasible, unknown),
            seed in 0:2

            model, p = generate_problem("diet_problem/food_groups", target, status, seed)
            @test num_variables(model) == 3 * p.n_foods * p.n_days
            @test p.n_weeks == cld(p.n_days, 7)
            @test abs(num_variables(model) - target) <= max(0.1 * target, 39)
            @test all(0 .< p.decay .< 0.2)
            @test all(p.spot_markup .>= 1.3)
            if status == feasible
                @test S.menu_plan_satisfies(p)
                w = p.feasible_witness
                @test all(iszero, w.spot)
            elseif status == infeasible
                @test S.menu_certificate_holds(p)
                c = p.infeasibility_certificate
                @test c.achievable <= c.required / 1.05
            end
        end
        _, p = generate_problem("diet_problem/food_groups", 600, feasible, 2)
        w = p.feasible_witness
        broken = S.MenuPlan(w.servings, w.purchases .* 0.5, w.spot, w.inventory)
        @test !S.menu_plan_satisfies(p, broken)
        big = S.FoodGroupsDietProblem(100_000, unknown, 0)
        @test abs(3 * big.n_foods * big.n_days - 100_000) <= 0.03 * 100_000
    end

    @testset "food_aid: sizing, witness, certificates" begin
        for target in (30, 50, 300, 2_000, 12_000),
            status in (feasible, infeasible, unknown),
            seed in 0:2

            model, p = generate_problem("diet_problem/food_aid", target, status, seed)
            n = length(p.ration_pairs) + length(p.delivery_arcs) + length(p.procurement_arcs)
            @test num_variables(model) == n
            @test n >= target && n <= target + 60
            # Delivery arcs exist only for sites served by two hubs.
            @test all(length(p.site_hubs[j]) == 2 for (_, _, j) in p.delivery_arcs)
            @test all(1 .<= p.site_programme .<= length(S._AID_PROGRAMMES))
            if status == feasible
                @test S.aid_plan_satisfies(p)
            elseif status == infeasible
                @test S.aid_certificate_holds(p)
                c = p.infeasibility_certificate
                @test c.achievable <= c.required / 1.07
            end
        end
        _, p = generate_problem("diet_problem/food_aid", 500, feasible, 1)
        w = p.feasible_witness
        @test !S.aid_plan_satisfies(p, S.AidPlan(w.ration .* 1.5, w.delivery, w.procurement))
        big = S.FoodAidDietProblem(100_000, feasible, 0)
        @test 100_000 <=
            length(big.ration_pairs) + length(big.delivery_arcs) + length(big.procurement_arcs) <=
            100_060
    end

    @testset "reproducibility and global-RNG isolation" begin
        for ref in ("diet_problem/standard", "diet_problem/food_groups", "diet_problem/food_aid")
            _, p1 = generate_problem(ref, 700, infeasible, 77)
            _, p2 = generate_problem(ref, 700, infeasible, 77)
            for f in fieldnames(typeof(p1))
                a, b = getfield(p1, f), getfield(p2, f)
                if a isa Union{
                    S.DietInfeasibilityCertificate,
                    S.MenuInfeasibilityCertificate,
                    S.AidInfeasibilityCertificate,
                }
                    @test all(
                        isequal(getfield(a, g), getfield(b, g)) for g in fieldnames(typeof(a))
                    )
                else
                    @test isequal(a, b)
                end
            end
            Random.seed!(4242)
            expected = rand()
            Random.seed!(4242)
            generate_problem(ref, 500, feasible, 3)
            @test rand() == expected
        end
    end

    @testset "HiGHS feasibility contracts" begin
        if HAS_HIGHS
            for ref in
                ("diet_problem/standard", "diet_problem/food_groups", "diet_problem/food_aid"),
                target in (60, 800), seed in 0:3,
                status in (feasible, infeasible)

                model, _ = generate_problem(ref, target, status, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                @test termination_status(model) ==
                    (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end
            # `unknown` is genuinely two-sided across seeds.
            outcomes = Set{MOI.TerminationStatusCode}()
            for seed in 0:19
                model, _ = generate_problem("diet_problem/food_aid", 400, unknown, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                push!(outcomes, termination_status(model))
            end
            @test MOI.OPTIMAL in outcomes && MOI.INFEASIBLE in outcomes
        end
    end
end
