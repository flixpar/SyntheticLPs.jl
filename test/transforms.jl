# Practitioner-style model transforms (`src/transforms.jl`): unit scaling,
# redundant aggregate rows, elastic rows, and row/column permutation. Framework-
# level coverage: included unconditionally from `runtests.jl`, so it runs on
# focused runs too. Solver-backed equivalence checks are guarded by HAS_HIGHS.

# A small hand-built LP touching every row sense, a named and an anonymous
# family, bounds of every kind, and an integer column.
function _transform_test_model()
    m = Model()
    @variable(m, 0 <= x[1:4] <= 10)
    @variable(m, y[1:2] >= 1)
    @variable(m, z == 2)
    @variable(m, 0 <= k <= 5, Int)
    @constraint(m, cap[i = 1:4], x[i] + y[1 + (i % 2)] <= 8 + i)
    @constraint(m, x[1] + x[2] + x[3] + x[4] >= 3)
    @constraint(m, x[1] + x[2] - y[1] + z == 4)
    @constraint(m, x[3] + x[4] - y[2] + z == 4)
    @constraint(m, 1 <= y[1] + y[2] + k <= 9)
    @objective(m, Min, 3x[1] + 2x[2] + x[3] + 4x[4] + 5y[1] + 6y[2] + z + 2k + 7)
    return m
end

_row_count(m) = num_constraints(m; count_variable_in_set_constraints=false)

_set_values(s::MOI.Interval) = (s.lower, s.upper)
_set_values(s) = (MOI.constant(s),)

@testset "Model Transforms" begin
    @testset "Configuration" begin
        t = ModelTransforms()
        @test SyntheticLPs.is_identity(t)
        @test !SyntheticLPs.is_identity(ModelTransforms(; permute=true))
        @test !SyntheticLPs.is_identity(ModelTransforms(; unit_scale_decades=1))
        @test SyntheticLPs._as_transforms(nothing) == t
        @test SyntheticLPs._as_transforms((unit_scale_decades=2, permute=true)) ==
            ModelTransforms(; unit_scale_decades=2, permute=true)
        @test_throws ArgumentError ModelTransforms(; unit_scale_decades=-1)
        @test_throws ArgumentError ModelTransforms(; unit_scale_decades=7)
        @test_throws ArgumentError ModelTransforms(; aggregate_probability=1.5)
        @test_throws ArgumentError ModelTransforms(; aggregate_max_block=1)
        @test_throws ArgumentError ModelTransforms(; elastic_probability=-0.1)
        @test_throws ArgumentError ModelTransforms(; elastic_penalty=0)
        cfg = SyntheticLPs._transforms_config(ModelTransforms(; aggregate_probability=0.3))
        @test cfg["aggregate_probability"] == 0.3 && cfg["unit_scale_decades"] == 0

        # The identity configuration returns the very same model object.
        m = _transform_test_model()
        @test apply_transforms(m, t, 1) === m

        # Default keyword is a no-op through generate_problem.
        a, _ = generate_problem("transportation/standard", 200, feasible, 3)
        b, _ = generate_problem("transportation/standard", 200, feasible, 3; transforms=t)
        @test sprint(print, a) == sprint(print, b)
    end

    @testset "Unit scaling" begin
        m = _transform_test_model()
        ref = _transform_test_model()
        rng = MersenneTwister(5)
        sc = scale_units!(m, rng; decades=3)
        @test m.ext[:SyntheticLPs_unit_scaling] === sc
        # One exponent per family, all within range.
        @test Set(keys(sc.column_exponents)) == Set(["x", "y", "z", "k"])
        @test all(e -> -3 <= e <= 3, values(sc.column_exponents))
        @test all(e -> -3 <= e <= 3, values(sc.row_exponents))
        xs = all_variables(m)
        xr = all_variables(ref)
        s = [sc.column_scale[v] for v in xs]
        # Columns of one family share a factor; the integer column is untouched.
        @test s[1] == s[2] == s[3] == s[4] == 10.0^sc.column_exponents["x"]
        @test s[5] == s[6]
        @test sc.column_scale[m[:k]] == 1.0
        # Bounds scale with the column.
        @test upper_bound(m[:x][2]) ≈ 10 * s[2]
        @test lower_bound(m[:y][1]) ≈ 1 * s[5]
        @test fix_value(m[:z]) ≈ 2 * sc.column_scale[m[:z]]
        @test upper_bound(m[:k]) == 5
        # Named family `cap` shares one row factor.
        caps = m[:cap]
        @test length(unique(sc.row_scale[c] for c in caps)) == 1
        # Every coefficient and rhs follows a_ij -> r_i a_ij / s_j.
        rows_m = SyntheticLPs._linear_rows(m)
        rows_r = SyntheticLPs._linear_rows(ref)
        for (cm, cr) in zip(rows_m, rows_r)
            r = sc.row_scale[cm]
            om, or = constraint_object(cm), constraint_object(cr)
            for (j, v) in enumerate(xs)
                @test coefficient(om.func, v) ≈ r * coefficient(or.func, xr[j]) / s[j]
            end
            @test collect(_set_values(om.set)) ≈ r .* collect(_set_values(or.set))
        end
        # The two anonymous balance equalities form one family (same signature).
        eqs = all_constraints(m, AffExpr, MOI.EqualTo{Float64})
        @test sc.row_scale[eqs[1]] == sc.row_scale[eqs[2]]
        # Objective: sigma * c_j / s_j, constant scaled by sigma.
        obj, objr = objective_function(m), objective_function(ref)
        σ = sc.objective_scale
        @test obj.constant ≈ 7σ
        for (j, v) in enumerate(xs)
            @test coefficient(obj, v) ≈ σ * coefficient(objr, xr[j]) / s[j]
        end
        # scale_objective=false keeps the objective's units.
        m2 = _transform_test_model()
        @test scale_units!(m2, MersenneTwister(5); decades=2, scale_objective=false).objective_scale ==
            1.0
        # decades = 0 is an exact no-op on values.
        m3 = _transform_test_model()
        sc3 = scale_units!(m3, MersenneTwister(1); decades=0)
        @test all(==(1.0), values(sc3.column_scale)) && sc3.objective_scale == 1.0
        @test sprint(print, m3) == sprint(print, _transform_test_model())
        @test_throws ArgumentError scale_units!(_transform_test_model(), rng; decades=-1)
        # Magnitude window: whatever the draws, scaled entries and costs stay in
        # [1e-6, 1e6] when the family's span allows, and a generator's own
        # out-of-window magnitudes are never made worse.
        function window_model()
            w = Model()
            @variable(w, a[1:3] >= 0)
            @variable(w, b[1:3] >= 0)
            @variable(w, big >= 0)
            @constraint(w, r[i = 1:3], 1e-2 * a[i] + 1e2 * b[i] + big <= 10)
            @objective(w, Min, 1e4 * sum(a) + 1e5 * sum(b) + 1e8 * big)
            return w
        end
        for seed in 1:30
            w = window_model()
            scale_units!(w, MersenneTwister(seed); decades=3)
            coefs = [abs(c) for cr in w[:r] for (_, c) in constraint_object(cr).func.terms]
            @test all(c -> 1e-6 * (1 - 1e-9) <= c <= 1e6 * (1 + 1e-9), coefs)
            costs = Dict(name(x) => abs(c) for (x, c) in objective_function(w).terms)
            @test costs["big"] <= 1e8 * (1 + 1e-9)
            @test all(c -> c >= 1e-6 * (1 - 1e-9), values(costs))
        end
        # Unicode base names (energy/dc_opf's θ) are handled.
        dc, _ = generate_problem(
            "energy/dc_opf", 300, feasible, 2; transforms=(unit_scale_decades=2,)
        )
        @test haskey(dc.ext[:SyntheticLPs_unit_scaling].column_exponents, "θ")
    end

    @testset "Aggregate rows" begin
        m = _transform_test_model()
        before = _row_count(m)
        added = aggregate_rows!(m, MersenneTwister(1); probability=1.0, max_block=2)
        # Families: cap (4 rows -> 2 blocks of 2), the ≥ row (single, skipped),
        # the two balance equalities (1 block), the range (single, skipped).
        @test added == 3
        @test _row_count(m) == before + 3
        totals = [
            c for c in all_constraints(m, AffExpr, MOI.LessThan{Float64}) if
            startswith(name(c), "cap[total")
        ]
        @test length(totals) == 2
        x, y = m[:x], m[:y]
        t1 = constraint_object(only(c for c in totals if name(c) == "cap[total1]"))
        # cap[1] + cap[2]: x1 + y2 + x2 + y1 <= 9 + 10
        @test coefficient(t1.func, x[1]) == 1 && coefficient(t1.func, x[2]) == 1
        @test coefficient(t1.func, y[1]) == 1 && coefficient(t1.func, y[2]) == 1
        @test t1.set.upper == 19
        # Summed equalities: z enters twice, rhs 8.
        eqs = all_constraints(m, AffExpr, MOI.EqualTo{Float64})
        @test length(eqs) == 3
        agg = constraint_object(eqs[3])
        @test coefficient(agg.func, m[:z]) == 2 && agg.set.value == 8
        # probability = 0 adds nothing.
        @test aggregate_rows!(_transform_test_model(), MersenneTwister(1); probability=0.0) == 0
        # Exactly cancelling coefficients are dropped; an all-cancelling block is skipped.
        c = Model()
        @variable(c, u[1:3] >= 0)
        @constraint(c, u[1] - u[2] == 0)
        @constraint(c, u[2] - u[1] == 0)
        @constraint(c, u[1] + u[3] <= 1)
        @constraint(c, u[3] - u[1] <= 1)
        @test aggregate_rows!(c, MersenneTwister(1); probability=1.0, max_block=2) == 1
        leq = all_constraints(c, AffExpr, MOI.LessThan{Float64})
        @test length(constraint_object(leq[3]).func.terms) == 1  # 2u3 <= 2
        @test_throws ArgumentError aggregate_rows!(c, MersenneTwister(1); probability=2.0)
        @test_throws ArgumentError aggregate_rows!(c, MersenneTwister(1); max_block=1)
    end

    @testset "Elastic rows" begin
        m = _transform_test_model()
        n0 = num_variables(m)
        added = elasticize_rows!(m, MersenneTwister(1); probability=1.0, penalty=100.0)
        # 4 cap (≤: 1 each) + 1 (≥: 1) + 2 (==: 2 each) + 1 range (2) = 11.
        @test added == 11
        @test num_variables(m) == n0 + 11
        @test _row_count(m) == _row_count(_transform_test_model())
        over = variable_by_name(m, "cap_over[2]")
        @test over !== nothing && lower_bound(over) == 0
        @test normalized_coefficient(m[:cap][2], over) == -1
        # Penalty = penalty × max|c| = 100 × 6, charged under Min.
        @test coefficient(objective_function(m), over) == 600
        # Under Max the penalty is subtracted.
        mx = _transform_test_model()
        @objective(mx, Max, sum(mx[:x]))
        elasticize_rows!(mx, MersenneTwister(1); probability=1.0, penalty=10.0)
        @test coefficient(objective_function(mx), variable_by_name(mx, "cap_over[1]")) == -10
        # A feasibility model becomes a violation minimization.
        f = Model()
        @variable(f, w >= 0)
        @constraint(f, w <= -1)
        elasticize_rows!(f, MersenneTwister(1); probability=1.0)
        @test objective_sense(f) == MIN_SENSE
        @test elasticize_rows!(_transform_test_model(), MersenneTwister(1); probability=0.0) == 0
        # One base name used with two senses forms two families; generated
        # column and row names stay unique.
        dup = Model()
        @variable(dup, q[1:4] >= 0)
        @constraint(dup, bal[i = 1:2], q[i] + q[i + 2] <= 5)
        @constraint(dup, bal2[i = 1:2], q[i] - q[i + 2] >= -1)
        for c in bal2
            set_name(c, replace(name(c), "bal2" => "bal"))
        end
        aggregate_rows!(dup, MersenneTwister(2); probability=1.0, max_block=2)
        elasticize_rows!(dup, MersenneTwister(2); probability=1.0)
        totals = filter(startswith("bal[total"), name.(SyntheticLPs._linear_rows(dup)))
        @test length(totals) == 2 && allunique(totals)
        @test allunique(name.(all_variables(dup)))
        @test num_variables(dup) == 4 + 4 + 2  # 2 ≤ + 2 ≥ rows + 2 aggregates
        @test_throws ArgumentError elasticize_rows!(f, MersenneTwister(1); penalty=-1)
        # Elastic rows cannot honor an `infeasible` request.
        @test_throws ArgumentError generate_problem(
            "transportation/standard", 100, infeasible, 1; transforms=(elastic_probability=0.5,)
        )
        # ...but scaling/aggregation/permutation can.
        inf, _ = generate_problem(
            "transportation/standard",
            100,
            infeasible,
            1;
            transforms=(unit_scale_decades=2, aggregate_probability=1.0, permute=true),
        )
        @test num_variables(inf) > 0
    end

    @testset "Permutation" begin
        m = _transform_test_model()
        snapshot = sprint(print, m)
        p = permute_model(m, MersenneTwister(3))
        @test sprint(print, m) == snapshot  # input untouched
        @test num_variables(p) == num_variables(m)
        @test _row_count(p) == _row_count(m)
        @test Set(name.(all_variables(p))) == Set(name.(all_variables(m)))
        @test name.(all_variables(p)) != name.(all_variables(m))
        @test is_integer(variable_by_name(p, "k"))
        @test fix_value(variable_by_name(p, "z")) == 2
        @test upper_bound(variable_by_name(p, "x[3]")) == 10
        @test objective_function(p).constant == 7
        @test coefficient(objective_function(p), variable_by_name(p, "y[2]")) == 6
        q = permute_model(m, MersenneTwister(3))
        @test sprint(print, p) == sprint(print, q)
    end

    @testset "Determinism and RNG isolation" begin
        t = ModelTransforms(;
            unit_scale_decades=2, aggregate_probability=0.7, elastic_probability=0.5, permute=true
        )
        a, _ = generate_problem("process_planning/refinery", 400, feasible, 9; transforms=t)
        b, _ = generate_problem("process_planning/refinery", 400, feasible, 9; transforms=t)
        @test sprint(print, a) == sprint(print, b)
        c, _ = generate_problem("process_planning/refinery", 400, feasible, 10; transforms=t)
        @test sprint(print, a) != sprint(print, c)
        # Transforms use their own seeded streams, never the global RNG.
        Random.seed!(1234)
        expected = rand()
        Random.seed!(1234)
        generate_problem("process_planning/refinery", 400, feasible, 9; transforms=t)
        @test rand() == expected
        # Each transform has its own stream: turning on permutation does not
        # change which unit exponents the families receive.
        s1, _ = generate_problem(
            "energy/dc_opf", 300, unknown, 4; transforms=(unit_scale_decades=3,)
        )
        s2, _ = generate_problem(
            "energy/dc_opf", 300, unknown, 4; transforms=(unit_scale_decades=3, permute=true)
        )
        @test s1.ext[:SyntheticLPs_unit_scaling].column_exponents ==
            s2.ext[:SyntheticLPs_unit_scaling].column_exponents
        # Applied before dualization: the dual of a transformed primal.
        d, _ = generate_problem(
            "transportation/standard",
            100,
            feasible,
            2;
            transforms=(aggregate_probability=1.0,),
            dualize=true,
        )
        d0, _ = generate_problem("transportation/standard", 100, feasible, 2; dualize=true)
        @test is_dual_reformulation(d)
        @test num_variables(d) > num_variables(d0)  # one dual per added aggregate row
        # generate_random_problem forwards the configuration.
        r1, ref1, _ = generate_random_problem(150; seed=8, transforms=(permute=true,))
        r2, ref2, _ = generate_random_problem(150; seed=8, transforms=(permute=true,))
        @test ref1 == ref2 && sprint(print, r1) == sprint(print, r2)
    end

    @testset "Dataset plumbing" begin
        tmp = mktempdir()
        kw = (
            num_problems=3,
            var_mean=150,
            var_std=30,
            var_min=80,
            var_max=250,
            seed=31,
            problem_types=["transportation/standard"],
            max_candidate_multiplier=2,
            match_size_distribution=false,
        )
        plain = generate_dataset(; kw...)
        agg = generate_dataset(;
            kw..., transforms=ModelTransforms(; aggregate_probability=1.0), output_dir=tmp
        )
        @test [i.seed for i in agg] == [i.seed for i in plain]
        @test all(agg[i].num_constraints > plain[i].num_constraints for i in eachindex(agg))
        manifest = JSON.parsefile(joinpath(tmp, "manifest.json"))
        @test manifest["config"]["transforms"]["aggregate_probability"] == 1.0
        @test manifest["config"]["transforms"]["permute"] == false
        default_manifest_dir = mktempdir()
        generate_dataset(; kw..., output_dir=default_manifest_dir)
        dm = JSON.parsefile(joinpath(default_manifest_dir, "manifest.json"))
        @test dm["config"]["transforms"]["unit_scale_decades"] == 0
    end

    @testset "Solver equivalence" begin
        if HAS_HIGHS
            function solve_lp(m)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                return m
            end
            for (ref, n) in (
                ("transportation/standard", 300),
                ("energy/dc_opf", 400),
                ("process_planning/refinery", 400),
                ("inventory/multi_item", 300),
            )
                base = solve_lp(first(generate_problem(ref, n, feasible, 5)))
                @test termination_status(base) == MOI.OPTIMAL
                z = objective_value(base)
                tol = 1e-6 * max(1, abs(z))

                # Unit scaling: same optimum up to σ, and y*/s solves the original.
                m, _ = generate_problem(ref, n, feasible, 5; transforms=(unit_scale_decades=2,))
                sc = m.ext[:SyntheticLPs_unit_scaling]
                solve_lp(m)
                @test termination_status(m) == MOI.OPTIMAL
                @test objective_value(m) / sc.objective_scale ≈ z atol = tol
                point = Dict(
                    variable_by_name(base, name(v)) => value(v) / sc.column_scale[v] for
                    v in all_variables(m)
                )
                @test isempty(primal_feasibility_report(base, point; atol=1e-4))

                # Aggregation and permutation preserve the optimum exactly.
                for t in ((aggregate_probability=1.0,), (permute=true,))
                    m = solve_lp(first(generate_problem(ref, n, feasible, 5; transforms=t)))
                    @test termination_status(m) == MOI.OPTIMAL
                    @test objective_value(m) ≈ z atol = tol
                end

                # Elastic rows relax: never worse, and with the default penalty
                # unchanged on these instances.
                m = solve_lp(
                    first(
                        generate_problem(ref, n, feasible, 5; transforms=(elastic_probability=1.0,))
                    ),
                )
                @test termination_status(m) == MOI.OPTIMAL
                sense = objective_sense(base)
                @test if sense == MIN_SENSE
                    objective_value(m) <= z + tol
                else
                    objective_value(m) >= z - tol
                end
                @test objective_value(m) ≈ z atol = tol
            end

            # Equivalence-preserving transforms keep an infeasible label.
            t = (unit_scale_decades=3, aggregate_probability=1.0, permute=true)
            for ref in ("transportation/standard", "process_planning/refinery")
                m = solve_lp(first(generate_problem(ref, 300, infeasible, 2; transforms=t)))
                @test termination_status(m) == MOI.INFEASIBLE
            end
            # Verification still runs on the source primal and returns a
            # transformed model.
            v, _ = generate_problem(
                "transportation/standard",
                200,
                feasible,
                3;
                transforms=(unit_scale_decades=2,),
                optimizer=HiGHS.Optimizer,
            )
            @test haskey(v.ext, :SyntheticLPs_unit_scaling)
        end
    end
end
