# Focused quality contracts for the workforce_shift_scheduling category: the
# covering variant's registry wiring, sizing and row scaling, column validity,
# planted-roster witness and site-day-skill capacity certificate, and its HiGHS
# feasibility contracts.
@testset "Workforce Shift Covering" begin
    ref = ProblemVariant(:workforce_shift_scheduling, :covering)
    @test :workforce_shift_scheduling in list_categories()
    @test list_variants(:workforce_shift_scheduling) == [:covering]
    @test problem_info(:workforce_shift_scheduling)[:default_variant] == :covering
    @test problem_info(:workforce_shift_scheduling, :covering)[:type] <: ProblemGenerator

    nrows(model) = num_constraints(model; count_variable_in_set_constraints=false)

    # Recompute every row of a problem at a staffing vector `x`.
    function staffing_holds(problem, x; tol=1e-8)
        day_rows, week_rows, single_bounds = SyntheticLPs._workforce_pool_rows(problem)
        for (pool, day, columns) in day_rows
            sum(x[c] for c in columns) <= problem.pool_day_capacity[pool, day] + tol || return false
        end
        for (pool, columns) in week_rows
            sum(x[c] for c in columns) <= problem.pool_week_capacity[pool] + tol || return false
        end
        for (column, bound) in single_bounds
            x[column] <= bound + tol || return false
        end
        supplied = zeros(size(problem.demand))
        for column in eachindex(problem.column_pools)
            site = problem.column_sites[column]
            day = problem.column_days[column]
            skill = problem.column_skills[column]
            rate = problem.pool_productivity[problem.column_pools[column], skill]
            for period in findall(problem.pattern_coverage[:, problem.column_patterns[column]])
                supplied[site, day, period, skill] += rate * x[column]
            end
        end
        return all(supplied .+ tol .>= problem.demand)
    end

    # There is exactly one decision-variable block. Sizing is exact from
    # small instances through 20k; rows are coverage (sites x days x periods
    # x skills) plus pool-day and weekly capacity rows.
    for target in (10, 50, 200, 1500, 5000, 20_000)
        model, problem = generate_problem(ref, target, feasible, 19)
        @test num_variables(model) == target == length(problem.column_pools)
        n_days, n_sites = SyntheticLPs._workforce_horizon(target)
        @test (problem.n_days, problem.n_sites) == (n_days, n_sites)
        day_rows, week_rows, _ = SyntheticLPs._workforce_pool_rows(problem)
        @test nrows(model) ==
            n_sites * n_days * problem.n_periods * length(problem.skill_names) +
              length(day_rows) +
              length(week_rows)
    end
    @test SyntheticLPs._workforce_horizon(1) == (1, 1)
    @test SyntheticLPs._workforce_horizon(2000) == (7, 1)
    @test SyntheticLPs._workforce_horizon(100_000) == (7, 20)

    # A target of one is below the structural floor needed to keep every
    # skill-period covered and every generated labor pool represented.
    for seed in (1, 2, 4)
        model, problem = generate_problem(ref, 1, feasible, seed)
        @test num_variables(model) == length(problem.column_pools) > 1
    end

    # Rows grow with the column count (the old single-day, single-site model
    # had ~350 rows at 10k columns) and 100k builds in seconds.
    for (target, seed) in ((10_000, 1), (100_000, 2))
        elapsed = @elapsed model, problem = generate_problem(ref, target, feasible, seed)
        @test num_variables(model) == target
        @test nrows(model) >= 0.08 * target
        @test elapsed < 60
    end

    # Every profile is exercised above 1,000 variables. Contact-center and
    # continuous-operations instances take their four-skill branches; retail
    # intentionally has three profile-defined skills.
    large_profiles = Dict{Symbol, Any}()
    for seed in (1, 2, 4)
        model, problem = generate_problem(ref, 1500, feasible, seed)
        @test num_variables(model) == 1500
        large_profiles[problem.profile] = problem
    end
    @test Set(keys(large_profiles)) == Set((:contact_center, :retail, :continuous_operations))
    @test length(large_profiles[:contact_center].skill_names) == 4
    @test length(large_profiles[:continuous_operations].skill_names) == 4
    @test length(large_profiles[:retail].skill_names) == 3

    # Validate one large multi-site instance end to end: the stored witness
    # satisfies every row, and the named coverage block has exactly the
    # expected shape and right-hand sides.
    large_model, large_problem = generate_problem(ref, 12_000, feasible, 2)
    @test large_problem.n_sites == 2 && large_problem.n_days == 7
    large_witness = something(large_problem.feasible_staffing)
    @test all(large_witness .>= 0)
    @test staffing_holds(large_problem, large_witness)
    coverage_rows = large_model[:skill_coverage]
    @test size(coverage_rows) == size(large_problem.demand)
    for idx in CartesianIndices(coverage_rows)
        @test constraint_object(coverage_rows[idx]).set.lower == large_problem.demand[idx]
    end
    # Float pools couple the sites.
    @test any(sum(large_problem.pool_sites; dims=2) .> 1)

    # Exact field and model reproducibility, including repeated builds and
    # deterministic MPS export.
    model1, problem1 = generate_problem(ref, 320, unknown, 12345)
    model2, problem2 = generate_problem(ref, 320, unknown, 12345)
    @test all(
        isequal(getfield(problem1, field), getfield(problem2, field)) for
        field in fieldnames(typeof(problem1))
    )
    rebuilt1 = SyntheticLPs.build_model(problem1)
    rebuilt2 = SyntheticLPs.build_model(problem1)
    @test num_variables(model1) ==
        num_variables(model2) ==
        num_variables(rebuilt1) ==
        num_variables(rebuilt2)
    @test num_constraints(model1; count_variable_in_set_constraints=true) ==
        num_constraints(model2; count_variable_in_set_constraints=true) ==
        num_constraints(rebuilt1; count_variable_in_set_constraints=true) ==
        num_constraints(rebuilt2; count_variable_in_set_constraints=true)

    # Exact model contract: one continuous nonnegative staffing block,
    # minimization, objective coefficients sourced from the stored data, and
    # upper bounds only where a pool-day has a single column.
    assigned_workers = model1[:assigned_workers]
    variables = all_variables(model1)
    @test length(assigned_workers) == length(problem1.staffing_costs)
    @test Set(variables) == Set(assigned_workers)
    @test all(!is_binary(variable) && !is_integer(variable) for variable in variables)
    @test all(has_lower_bound(variable) && lower_bound(variable) == 0.0 for variable in variables)
    _, _, single_bounds1 = SyntheticLPs._workforce_pool_rows(problem1)
    @test Set(findall(has_upper_bound, assigned_workers)) == Set(keys(single_bounds1))
    @test objective_sense(model1) == MOI.MIN_SENSE
    objective = objective_function(model1)
    @test objective isa JuMP.AffExpr
    @test objective.constant == 0.0
    @test all(
        coefficient(objective, assigned_workers[column]) == problem1.staffing_costs[column] for
        column in eachindex(problem1.staffing_costs)
    )

    export_dir = mktempdir()
    path1 = joinpath(export_dir, "workforce_1.mps")
    path2 = joinpath(export_dir, "workforce_2.mps")
    write_to_file(rebuilt1, path1)
    write_to_file(rebuilt2, path2)
    @test filesize(path1) > 0
    @test read(path1, String) == read(path2, String)

    # Fixed seeds exercise all structural profiles and their distinct
    # horizons, shift rules, demand curves, and availability regimes.
    profiles = Dict{Symbol, Any}()
    for seed in (1, 2, 4)
        _, problem = generate_problem(ref, 2400, unknown, seed)
        profiles[problem.profile] = problem
    end
    @test Set(keys(profiles)) == Set((:contact_center, :retail, :continuous_operations))
    @test (profiles[:contact_center].period_minutes, profiles[:contact_center].n_periods) ==
        (30, 24)
    @test (profiles[:retail].period_minutes, profiles[:retail].n_periods) == (60, 14)
    @test (
        profiles[:continuous_operations].period_minutes, profiles[:continuous_operations].n_periods
    ) == (60, 24)
    @test !any(profiles[:contact_center].pattern_wraps)
    @test !any(profiles[:retail].pattern_wraps)
    @test any(profiles[:continuous_operations].pattern_wraps)
    @test all(profile -> any(profile.pattern_break_periods .> 0), values(profiles))
    @test all(profile -> length(unique(profile.pattern_span_periods)) >= 3, values(profiles))
    # Weekly demand: contact centers are quiet at the weekend, retail peaks on
    # Saturday.
    cc, rt = profiles[:contact_center], profiles[:retail]
    @test cc.n_days == 7 && rt.n_days == 7
    @test sum(cc.demand[:, 7, :, :]) < 0.6 * sum(cc.demand[:, 1, :, :])
    @test sum(rt.demand[:, 6, :, :]) > sum(rt.demand[:, 2, :, :])

    for problem in values(profiles)
        n_pools = length(problem.pool_names)
        n_skills = length(problem.skill_names)
        n_patterns = size(problem.pattern_coverage, 2)
        @test n_pools >= 4
        @test n_skills >= 2
        @test n_patterns > 0
        @test all(sum(problem.pattern_coverage; dims=1) .> 0)
        @test length(unique(Tuple(problem.pool_qualifications[q, :]) for q in 1:n_pools)) > 1
        @test length(unique(Tuple(problem.pool_availability[q, :]) for q in 1:n_pools)) > 1
        @test all(problem.pool_productivity[problem.pool_qualifications] .> 0)
        @test all(problem.pool_productivity[.!problem.pool_qualifications] .== 0)
        @test all(problem.hourly_wages .> 0)
        @test all(problem.pool_week_capacity .> 0)
        @test all(problem.staffing_costs .> 0)
        @test all(problem.demand .> 0)
        @test all(problem.pool_sites[q, problem.pool_home_site[q]] for q in 1:n_pools)

        # Pattern metadata reconstructs each contiguous (possibly
        # wraparound) start/span window exactly. A stored break is inside
        # that window and is the sole excluded period.
        supports = Tuple[]
        for pattern in 1:n_patterns
            start = problem.pattern_starts[pattern]
            span = problem.pattern_span_periods[pattern]
            break_period = problem.pattern_break_periods[pattern]
            window = if problem.profile == :continuous_operations
                [mod1(start + offset, problem.n_periods) for offset in 0:(span - 1)]
            else
                [start + offset for offset in 0:(span - 1)]
            end
            @test length(unique(window)) == span
            @test all(period -> 1 <= period <= problem.n_periods, window)
            expected_support = if break_period == 0
                copy(window)
            else
                @test break_period in window
                @test !problem.pattern_coverage[break_period, pattern]
                [period for period in window if period != break_period]
            end
            actual_support = findall(problem.pattern_coverage[:, pattern])
            @test sort(actual_support) == sort(expected_support)
            @test problem.pattern_wraps[pattern] == (start + span - 1 > problem.n_periods)
            push!(supports, Tuple(actual_support))
        end
        @test length(unique(supports)) == n_patterns

        # Every selected column obeys qualification, period availability,
        # pattern eligibility, the pool's working days and served sites, and
        # has a positive pool-day capacity. Every coverage row has support.
        supported = falses(size(problem.demand))
        for column in eachindex(problem.column_pools)
            pool = problem.column_pools[column]
            pattern = problem.column_patterns[column]
            skill = problem.column_skills[column]
            site = problem.column_sites[column]
            day = problem.column_days[column]
            @test problem.pool_qualifications[pool, skill]
            @test problem.pattern_eligibility[pool, pattern]
            @test problem.pool_days[pool, day]
            @test problem.pool_sites[pool, site]
            @test problem.pool_day_capacity[pool, day] > 0
            @test all(
                !problem.pattern_coverage[period, pattern] ||
                    problem.pool_availability[pool, period] for period in 1:problem.n_periods
            )
            for period in findall(problem.pattern_coverage[:, pattern])
                supported[site, day, period, skill] = true
            end
        end
        @test all(supported)

        # Signatures are unique: costs never disguise duplicate columns.
        signatures = [
            (
                problem.column_pools[c],
                problem.column_sites[c],
                problem.column_days[c],
                problem.column_skills[c],
                Tuple(findall(problem.pattern_coverage[:, problem.column_patterns[c]])),
            ) for c in eachindex(problem.column_pools)
        ]
        @test length(unique(signatures)) == length(signatures)
    end

    # Different seeds alter profile and numerical/structural data.
    _, seed1 = generate_problem(ref, 240, unknown, 1)
    _, seed2 = generate_problem(ref, 240, unknown, 2)
    @test seed1.profile != seed2.profile
    @test seed1.skill_names != seed2.skill_names
    for (first_seed, second_seed) in ((1, 3), (2, 5), (4, 7))
        _, first_problem = generate_problem(ref, 240, unknown, first_seed)
        _, second_problem = generate_problem(ref, 240, unknown, second_seed)
        @test first_problem.profile == second_problem.profile
        @test first_problem.demand != second_problem.demand
        @test first_problem.pattern_coverage != second_problem.pattern_coverage
        @test first_problem.pool_qualifications != second_problem.pool_qualifications
    end

    # The planted staffing vector proves feasible requests directly.
    for seed in 1:6, target in (260, 3000)
        _, problem = generate_problem(ref, target, feasible, seed)
        @test problem.feasibility_status == feasible
        @test problem.feasible_staffing !== nothing
        @test problem.infeasible_group === nothing
        @test problem.infeasibility_capacity_bound === nothing
        @test staffing_holds(problem, something(problem.feasible_staffing))
    end

    # Infeasible: the summed coverage rows of one (site, day, skill) group
    # exceed a valid upper bound built from the pool-day capacity rows.
    for seed in 1:6, target in (260, 3000)
        _, problem = generate_problem(ref, target, infeasible, seed)
        @test problem.feasibility_status == infeasible
        @test problem.feasible_staffing === nothing
        site, day, skill = something(problem.infeasible_group)
        longest = Dict{Int, Int}()
        for c in eachindex(problem.column_pools)
            (problem.column_sites[c], problem.column_days[c], problem.column_skills[c]) ==
            (site, day, skill) || continue
            pool = problem.column_pools[c]
            paid = count(problem.pattern_coverage[:, problem.column_patterns[c]])
            longest[pool] = max(get(longest, pool, 0), paid)
        end
        bound = sum(
            problem.pool_day_capacity[pool, day] * problem.pool_productivity[pool, skill] * paid for
            (pool, paid) in longest
        )
        @test something(problem.infeasibility_capacity_bound) ≈ bound
        # A margin of at least 3% (or 0.5 worker-periods), not a knife edge.
        @test sum(problem.demand[site, day, :, skill]) >= bound + max(0.5, 0.03 * bound) - 1e-6
        # Every pool-day of the group really is capped by a row or a bound.
        day_rows, _, single_bounds = SyntheticLPs._workforce_pool_rows(problem)
        capped = Set((pool, d) for (pool, d, _) in day_rows)
        for (column, _) in single_bounds
            push!(capped, (problem.column_pools[column], problem.column_days[column]))
        end
        @test all((pool, day) in capped for pool in keys(longest))
    end

    # Unknown mode starts from the same sampled structure as feasible mode
    # but applies genuine labor-market and workload shocks. It exposes no
    # witness or infeasibility certificate.
    _, feasible_problem = generate_problem(ref, 2600, feasible, 23)
    _, unknown_problem = generate_problem(ref, 2600, unknown, 23)
    @test unknown_problem.profile == feasible_problem.profile
    @test unknown_problem.column_pools == feasible_problem.column_pools
    @test unknown_problem.column_patterns == feasible_problem.column_patterns
    @test unknown_problem.column_days == feasible_problem.column_days
    @test unknown_problem.pool_day_capacity != feasible_problem.pool_day_capacity
    @test unknown_problem.demand != feasible_problem.demand
    @test unknown_problem.feasible_staffing === nothing
    @test unknown_problem.infeasible_group === nothing
    @test unknown_problem.infeasibility_capacity_bound === nothing

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            for seed in 1:6, status in (feasible, infeasible), target in (260, 3000)
                model, _ = generate_problem(ref, target, status, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
                @test termination_status(model) == expected
            end
            # The certificate needs simplex work: presolve alone does not
            # refute the aggregated site-day-skill demand curve.
            model, _ = generate_problem(ref, 8000, infeasible, 3)
            set_optimizer(model, HiGHS.Optimizer)
            set_silent(model)
            optimize!(model)
            @test termination_status(model) == MOI.INFEASIBLE
            @test MOI.get(model, MOI.SimplexIterations()) > 0

            # Unknown is two-sided.
            outcomes = Set{MOI.TerminationStatusCode}()
            for seed in 0:9
                model, _ = generate_problem(ref, 2000, unknown, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                push!(outcomes, termination_status(model))
            end
            @test outcomes == Set([MOI.OPTIMAL, MOI.INFEASIBLE])

            set_optimizer(large_model, HiGHS.Optimizer)
            set_silent(large_model)
            optimize!(large_model)
            @test termination_status(large_model) == MOI.OPTIMAL
        end
    end
end
