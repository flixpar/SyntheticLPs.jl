# Focused quality contracts for the workforce_shift_scheduling category: the
# covering variant's registry wiring, sizing, planted-roster witness and
# coverage-shortfall certificate, and its HiGHS feasibility contracts.
@testset "Workforce Shift Covering" begin
    ref = ProblemVariant(:workforce_shift_scheduling, :covering)
    @test :workforce_shift_scheduling in list_categories()
    @test list_variants(:workforce_shift_scheduling) == [:covering]
    @test problem_info(:workforce_shift_scheduling)[:default_variant] == :covering
    @test problem_info(:workforce_shift_scheduling, :covering)[:type] <:
          ProblemGenerator

    # There is exactly one decision-variable block. Sizing is exact from
    # small instances through scales above the source implementation's cap.
    for target in (10, 50, 200, 1500, 5000)
        model, problem = generate_problem(ref, target, feasible, 19)
        @test num_variables(model) == target ==
              length(problem.column_pools)
    end
    # A target of one is below the structural floor needed to retain
    # skill-period coverage and representative generated labor pools.
    expected_minimums = Dict(1 => 8, 2 => 9, 4 => 6)
    for (seed, expected) in expected_minimums
        model, problem = generate_problem(ref, 1, feasible, seed)
        @test num_variables(model) == expected ==
              length(problem.column_pools)
        @test num_variables(model) > 1
    end

    # Every profile is also exercised above 1,000 variables. Contact-center
    # and continuous-operations instances take their four-skill branches;
    # retail intentionally has three profile-defined skills.
    large_profiles = Dict{Symbol,Any}()
    for seed in (1, 2, 4)
        model, problem = generate_problem(ref, 1500, feasible, seed)
        @test num_variables(model) == 1500
        large_profiles[problem.profile] = problem
    end
    @test Set(keys(large_profiles)) ==
          Set((:contact_center, :retail, :continuous_operations))
    @test length(large_profiles[:contact_center].skill_names) == 4
    @test length(large_profiles[:continuous_operations].skill_names) == 4
    @test length(large_profiles[:retail].skill_names) == 3

    # Validate one large four-skill instance end to end: the stored
    # witness respects pool capacities, covers every row (including skill
    # 4), and the named model row block has exactly the expected shape.
    large_model, large_problem =
        generate_problem(ref, 1500, feasible, 2)
    @test large_problem.feasibility_status == feasible
    @test length(large_problem.skill_names) == 4
    large_witness = something(large_problem.feasible_staffing)
    for pool in eachindex(large_problem.pool_names)
        usage = sum(
            large_witness[column]
            for column in eachindex(large_problem.column_pools)
            if large_problem.column_pools[column] == pool
        )
        @test usage <= large_problem.pool_capacities[pool] + 1e-8
    end
    large_coverage_rows = large_model[:skill_coverage]
    @test size(large_coverage_rows) == (large_problem.n_periods, 4)
    @test length(large_coverage_rows) == 4 * large_problem.n_periods
    @test all(is_valid(large_model, large_coverage_rows[period, 4])
              for period in 1:large_problem.n_periods)
    for period in 1:large_problem.n_periods, skill in 1:4
        row = constraint_object(large_coverage_rows[period, skill])
        @test row.set.lower == large_problem.demand[period, skill]
        supplied = sum(
            large_problem.pool_productivity[
                large_problem.column_pools[column], skill,
            ] * large_witness[column]
            for column in eachindex(large_problem.column_pools)
            if large_problem.column_skills[column] == skill &&
               large_problem.pattern_coverage[
                   period, large_problem.column_patterns[column],
               ]
        )
        @test supplied + 1e-8 >= large_problem.demand[period, skill]
    end

    # Exact field and model reproducibility, including repeated builds and
    # deterministic MPS export.
    model1, problem1 = generate_problem(ref, 320, unknown, 12345)
    model2, problem2 = generate_problem(ref, 320, unknown, 12345)
    @test all(
        isequal(getfield(problem1, field), getfield(problem2, field))
        for field in fieldnames(typeof(problem1))
    )
    rebuilt1 = SyntheticLPs.build_model(problem1)
    rebuilt2 = SyntheticLPs.build_model(problem1)
    @test num_variables(model1) == num_variables(model2) ==
          num_variables(rebuilt1) == num_variables(rebuilt2)
    @test num_constraints(model1; count_variable_in_set_constraints=true) ==
          num_constraints(model2; count_variable_in_set_constraints=true) ==
          num_constraints(rebuilt1; count_variable_in_set_constraints=true) ==
          num_constraints(rebuilt2; count_variable_in_set_constraints=true)

    # Exact model contract: one continuous nonnegative staffing block,
    # minimization, and objective coefficients sourced without alteration
    # from the stored data.
    assigned_workers = model1[:assigned_workers]
    variables = all_variables(model1)
    @test length(assigned_workers) == length(problem1.staffing_costs)
    @test Set(variables) == Set(assigned_workers)
    @test all(!is_binary(variable) && !is_integer(variable)
              for variable in variables)
    @test all(has_lower_bound(variable) && lower_bound(variable) == 0.0
              for variable in variables)
    @test all(!has_upper_bound(variable) for variable in variables)
    @test objective_sense(model1) == MOI.MIN_SENSE
    objective = objective_function(model1)
    @test objective isa JuMP.AffExpr
    @test objective.constant == 0.0
    @test all(
        coefficient(objective, assigned_workers[column]) ==
        problem1.staffing_costs[column]
        for column in eachindex(problem1.staffing_costs)
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
    profiles = Dict{Symbol,Any}()
    for seed in (1, 2, 4)
        _, problem = generate_problem(ref, 240, unknown, seed)
        profiles[problem.profile] = problem
    end
    @test Set(keys(profiles)) ==
          Set((:contact_center, :retail, :continuous_operations))
    @test (profiles[:contact_center].period_minutes,
           profiles[:contact_center].n_periods) == (30, 24)
    @test (profiles[:retail].period_minutes,
           profiles[:retail].n_periods) == (60, 14)
    @test (profiles[:continuous_operations].period_minutes,
           profiles[:continuous_operations].n_periods) == (60, 24)
    @test !any(profiles[:contact_center].pattern_wraps)
    @test !any(profiles[:retail].pattern_wraps)
    @test any(profiles[:continuous_operations].pattern_wraps)
    @test all(profile -> any(profile.pattern_break_periods .> 0),
              values(profiles))
    @test all(profile -> length(unique(profile.pattern_span_periods)) >= 3,
              values(profiles))

    for problem in values(profiles)
        n_pools = length(problem.pool_names)
        n_skills = length(problem.skill_names)
        n_patterns = size(problem.pattern_coverage, 2)
        @test n_pools >= 4
        @test n_skills >= 2
        @test n_patterns > 0
        @test all(sum(problem.pattern_coverage; dims=1) .> 0)
        @test length(unique(Tuple(problem.pool_qualifications[q, :])
                            for q in 1:n_pools)) > 1
        @test length(unique(Tuple(problem.pool_availability[q, :])
                            for q in 1:n_pools)) > 1
        @test all(problem.pool_productivity[problem.pool_qualifications] .> 0)
        @test all(problem.pool_productivity[.!problem.pool_qualifications] .== 0)
        @test all(problem.hourly_wages .> 0)
        @test all(problem.pool_capacities .> 0)
        @test all(problem.staffing_costs .> 0)
        @test all(problem.demand .> 0)
        @test any(maximum(problem.demand[:, skill]) >
                  1.10 * minimum(problem.demand[:, skill])
                  for skill in 1:n_skills)

        # Pattern metadata reconstructs each contiguous (possibly
        # wraparound) start/span window exactly. A stored break is inside
        # that window and is the sole excluded period.
        supports = Tuple[]
        for pattern in 1:n_patterns
            start = problem.pattern_starts[pattern]
            span = problem.pattern_span_periods[pattern]
            break_period = problem.pattern_break_periods[pattern]
            window = if problem.profile == :continuous_operations
                [mod1(start + offset, problem.n_periods)
                 for offset in 0:(span - 1)]
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
            @test length(actual_support) ==
                  span - (break_period == 0 ? 0 : 1)
            @test problem.pattern_wraps[pattern] ==
                  (start + span - 1 > problem.n_periods)
            push!(supports, Tuple(actual_support))
        end
        # `_workforce_patterns` drops duplicate supports even when
        # different start/span/break samples would produce the same set.
        @test length(unique(supports)) == n_patterns

        # Every selected column obeys qualification, availability, and
        # pattern eligibility. Every skill-period has nonempty row support.
        for column in eachindex(problem.column_pools)
            pool = problem.column_pools[column]
            pattern = problem.column_patterns[column]
            skill = problem.column_skills[column]
            @test problem.pool_qualifications[pool, skill]
            @test problem.pattern_eligibility[pool, pattern]
            @test all(
                !problem.pattern_coverage[period, pattern] ||
                problem.pool_availability[pool, period]
                for period in 1:problem.n_periods
            )
        end
        @test all(
            any(problem.column_skills[column] == skill &&
                problem.pattern_coverage[period,
                                         problem.column_patterns[column]]
                for column in eachindex(problem.column_pools))
            for period in 1:problem.n_periods, skill in 1:n_skills
        )

        # Coverage + pool-row signatures are unique; costs are not being
        # used to disguise duplicate staffing columns.
        signatures = [
            (
                problem.column_pools[column],
                problem.column_skills[column],
                Tuple(findall(problem.pattern_coverage[:,
                                  problem.column_patterns[column]])),
            )
            for column in eachindex(problem.column_pools)
        ]
        @test length(unique(signatures)) == length(signatures)
    end

    # Different seeds alter profile and numerical/structural data.
    _, seed1 = generate_problem(ref, 240, unknown, 1)
    _, seed2 = generate_problem(ref, 240, unknown, 2)
    @test seed1.profile != seed2.profile
    @test seed1.demand != seed2.demand
    @test seed1.skill_names != seed2.skill_names
    # Diversity also holds within each profile, rather than relying on
    # profile selection alone.
    for (first_seed, second_seed) in ((1, 3), (2, 5), (4, 7))
        _, first_problem =
            generate_problem(ref, 240, unknown, first_seed)
        _, second_problem =
            generate_problem(ref, 240, unknown, second_seed)
        @test first_problem.profile == second_problem.profile
        @test first_problem.demand != second_problem.demand
        @test first_problem.pattern_coverage !=
              second_problem.pattern_coverage
        @test first_problem.pool_qualifications !=
              second_problem.pool_qualifications
    end

    # The planted staffing vector proves feasible requests directly.
    for seed in 1:6
        _, problem = generate_problem(ref, 260, feasible, seed)
        @test problem.feasibility_status == feasible
        @test problem.feasible_staffing !== nothing
        @test problem.infeasible_skill === nothing
        @test problem.infeasibility_capacity_bound === nothing
        witness = something(problem.feasible_staffing)
        for pool in eachindex(problem.pool_names)
            usage = sum(
                witness[column]
                for column in eachindex(problem.column_pools)
                if problem.column_pools[column] == pool
            )
            @test usage <= problem.pool_capacities[pool] + 1e-8
        end
        for period in 1:problem.n_periods,
            skill in eachindex(problem.skill_names)
            supplied = sum(
                problem.pool_productivity[
                    problem.column_pools[column], skill,
                ] * witness[column]
                for column in eachindex(problem.column_pools)
                if problem.column_skills[column] == skill &&
                   problem.pattern_coverage[
                       period, problem.column_patterns[column],
                   ]
            )
            @test supplied + 1e-8 >= problem.demand[period, skill]
        end
    end

    # At least one skill violates a valid aggregate capacity upper bound in
    # every requested-infeasible instance.
    for seed in 1:6
        _, problem = generate_problem(ref, 260, infeasible, seed)
        @test problem.feasibility_status == infeasible
        @test problem.feasible_staffing === nothing
        @test problem.infeasible_skill !== nothing
        @test problem.infeasibility_capacity_bound !== nothing
        certified = false
        for skill in eachindex(problem.skill_names)
            upper = 0.0
            for pool in eachindex(problem.pool_names)
                paid = [
                    count(problem.pattern_coverage[:,
                          problem.column_patterns[column]])
                    for column in eachindex(problem.column_pools)
                    if problem.column_pools[column] == pool &&
                       problem.column_skills[column] == skill
                ]
                max_paid = isempty(paid) ? 0 : maximum(paid)
                upper += problem.pool_capacities[pool] *
                         problem.pool_productivity[pool, skill] * max_paid
            end
            certified |= sum(problem.demand[:, skill]) > upper + 1e-6
        end
        @test certified
        certificate_skill = something(problem.infeasible_skill)
        expected_bound = 0.0
        for pool in eachindex(problem.pool_names)
            paid = [
                count(problem.pattern_coverage[:,
                      problem.column_patterns[column]])
                for column in eachindex(problem.column_pools)
                if problem.column_pools[column] == pool &&
                   problem.column_skills[column] == certificate_skill
            ]
            max_paid = isempty(paid) ? 0 : maximum(paid)
            expected_bound += problem.pool_capacities[pool] *
                              problem.pool_productivity[
                                  pool, certificate_skill,
                              ] * max_paid
        end
        @test something(problem.infeasibility_capacity_bound) ≈
              expected_bound
        @test sum(problem.demand[:, certificate_skill]) >
              something(problem.infeasibility_capacity_bound)
    end

    # Unknown mode starts from the same sampled structure as feasible mode
    # but applies genuine labor and workload shocks. It exposes no witness
    # or infeasibility certificate and makes no solver-status promise.
    feasible_model, feasible_problem =
        generate_problem(ref, 260, feasible, 23)
    unknown_model, unknown_problem =
        generate_problem(ref, 260, unknown, 23)
    @test feasible_problem.feasibility_status == feasible
    @test unknown_problem.feasibility_status == unknown
    @test unknown_problem.profile == feasible_problem.profile
    @test unknown_problem.pattern_coverage ==
          feasible_problem.pattern_coverage
    @test unknown_problem.column_pools == feasible_problem.column_pools
    @test unknown_problem.column_patterns ==
          feasible_problem.column_patterns
    @test unknown_problem.column_skills == feasible_problem.column_skills
    @test unknown_problem.pool_capacities !=
          feasible_problem.pool_capacities
    @test unknown_problem.demand != feasible_problem.demand
    @test unknown_problem.feasible_staffing === nothing
    @test unknown_problem.infeasible_skill === nothing
    @test unknown_problem.infeasibility_capacity_bound === nothing

    if HAS_HIGHS
        for seed in 1:6, status in (feasible, infeasible)
            model, _ = generate_problem(ref, 260, status, seed)
            set_optimizer(model, HiGHS.Optimizer)
            set_silent(model)
            optimize!(model)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(model) == expected
        end
        set_optimizer(unknown_model, HiGHS.Optimizer)
        set_silent(unknown_model)
        optimize!(unknown_model)
        @test termination_status(unknown_model) in
              (MOI.OPTIMAL, MOI.INFEASIBLE)
        set_optimizer(large_model, HiGHS.Optimizer)
        set_silent(large_model)
        optimize!(large_model)
        @test termination_status(large_model) == MOI.OPTIMAL
    end
end
