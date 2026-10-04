# Operating room scheduling: registry, shared-helper properties, exact sparse
# sizing, witnesses, LP-level infeasibility certificates (checked without a
# solver), and HiGHS contracts (statuses, certificates needing simplex work,
# two-sided unknown).
@testset "Operating Room Scheduling" begin
    variants = [
        :benchmark_loading,
        :case_sequencing,
        :elective_assignment,
        :master_surgical_schedule,
        :robust_elective,
        :weekly_planning,
    ]
    @test :operating_room_scheduling in list_categories()
    @test list_variants(:operating_room_scheduling) == variants
    @test problem_info(:operating_room_scheduling)[:default_variant] == :elective_assignment

    refs = Dict(v => ProblemVariant(:operating_room_scheduling, v) for v in variants)
    elective_ref = refs[:elective_assignment]
    sequencing_ref = refs[:case_sequencing]
    weekly_ref = refs[:weekly_planning]
    mss_ref = refs[:master_surgical_schedule]
    robust_ref = refs[:robust_elective]
    benchmark_ref = refs[:benchmark_loading]
    nrows(m) = num_constraints(m; count_variable_in_set_constraints=false)

    # Regression sweeps for both helper repair bugs.
    for n in 3:11, seed in 1:100
        ids = SyntheticLPs._orsched_case_mix(MersenneTwister(seed), n)
        @test length(unique(ids)) == n
        @test any(SyntheticLPs._ORSCHED_SPECIALTIES[k].aggregate_mean <= 90 for k in ids)
        @test any(SyntheticLPs._ORSCHED_SPECIALTIES[k].aggregate_mean >= 160 for k in ids)
    end
    for (rooms, days, specs) in ((2, 5, 3), (5, 10, 7), (8, 10, 11), (60, 10, 9)), seed in 1:40
        rng = MersenneTwister(seed)
        ids = SyntheticLPs._orsched_case_mix(rng, specs)
        mss, session = SyntheticLPs._orsched_master_schedule(rng, rooms, days, ids)
        @test all(count(==(k), mss) >= max(1, days ÷ 5) for k in 1:specs)
        @test all((mss[r, d] == 0) == (session[r, d] == 0) for r in 1:rooms, d in 1:days)
    end
    # The hospital grows with the target instead of saturating at 16 rooms.
    for growth in (:quadratic, :linear)
        small = SyntheticLPs._orsched_hospital_scale(MersenneTwister(1), 10_000; growth=growth)[1]
        large = SyntheticLPs._orsched_hospital_scale(MersenneTwister(1), 100_000; growth=growth)[1]
        @test large > 2 * small >= 16
    end

    # Constructors must not perturb caller/global randomness.
    for ref in values(refs)
        Random.seed!(77123)
        expected_draw = rand()
        Random.seed!(77123)
        generate_problem(ref, 100, unknown, 9)
        @test rand() == expected_draw
    end

    # Surgeon-overload certificate arithmetic, shared by the waiting-list
    # variants: every listed case belongs to the surgeon, is mandatory and is
    # admissible only on the certificate days, whose budgets total at most 90%
    # of the cases' minutes. In the presolve-proof (strict) form every case has
    # at least two of those days and each day is budgeted the longest case.
    function check_overload(p, cert, durations, days_of; strict=false)
        @test all(p.surgery_surgeon[i] == cert.surgeon for i in cert.cases)
        @test all(p.mandatory[i] for i in cert.cases)
        @test all(issubset(days_of(i), cert.days) for i in cert.cases)
        @test cert.case_minutes ≈ sum(durations[cert.cases])
        @test cert.budget_minutes ≈ sum(p.surgeon_budget[cert.surgeon, d] for d in cert.days)
        @test cert.budget_minutes <= 0.9 * cert.case_minutes + 1e-9
        @test all(p.surgeon_budget[cert.surgeon, d] > 0 for d in cert.days)
        if strict
            @test length(cert.cases) >= 3
            @test all(length(days_of(i)) >= 2 for i in cert.cases)
            longest = maximum(durations[cert.cases])
            @test all(p.surgeon_budget[cert.surgeon, d] ≈ longest for d in cert.days)
        end
    end

    # Elective assignment: sparse graph and planted witness.
    for seed in 0:3, target in (100, 500, 4000)
        model, p = generate_problem(elective_ref, target, feasible, seed)
        @test num_variables(model) ==
            length(p.admissible) + count(!, p.mandatory) + length(p.open_blocks)
        @test all(20 <= d <= 480 for d in p.surgery_duration)
        @test all(p.surgery_duration_sd .> 0)
        @test all(
            (p.mss[r, d] == 0) == (p.session_length[r, d] == 0) for
            r in 1:p.n_rooms, d in 1:p.n_days
        )
        for (i, r, d) in p.admissible
            @test p.mss[r, d] == p.surgery_specialty[i]
            @test p.surgeon_budget[p.surgery_surgeon[i], d] > 0
            @test d <= p.surgery_deadline[i]
        end
        @test p.mandatory == BitVector(p.surgery_urgency .== :urgent)
        rem_room, rem_surg = copy(p.session_length), copy(p.surgeon_budget)
        assigned = falses(p.n_surgeries)
        for a in something(p.feasible_witness)
            i, r, d = p.admissible[a]
            @test !assigned[i]
            assigned[i] = true
            rem_room[r, d] -= p.surgery_duration[i] + p.turnover
            rem_surg[p.surgery_surgeon[i], d] -= p.surgery_duration[i]
        end
        @test all(rem_room .>= -1e-9)
        @test all(rem_surg .>= -1e-9)
        @test all(.!p.mandatory .| assigned)
    end
    for seed in 0:3, target in (200, 3000)
        _, p = generate_problem(elective_ref, target, infeasible, seed)
        cert = something(p.infeasibility_certificate)
        check_overload(
            p, cert, p.surgery_duration, i -> unique(t[3] for t in p.admissible if t[1] == i);
            strict=target >= 3000,
        )
    end
    # Unknown: mandatory cases always have an admissible slot.
    for seed in 0:3
        _, p = generate_problem(elective_ref, 3000, unknown, seed)
        has = falses(p.n_surgeries)
        for (i, _, _) in p.admissible
            has[i] = true
        end
        @test all(has[p.mandatory])
        @test p.feasible_witness === nothing && p.infeasibility_certificate === nothing
    end

    # Robust assignment: every dual variable corresponds to a sparse
    # admissible triple, and the witness satisfies the exact Γ-budget load.
    for seed in 0:3, target in (100, 500)
        model, p = generate_problem(robust_ref, target, feasible, seed)
        @test num_variables(model) ==
            2length(p.admissible) + count(!, p.mandatory) + 2length(p.open_blocks)
        by_block = Dict(block => Int[] for block in p.open_blocks)
        rem_surg = copy(p.surgeon_budget)
        for a in something(p.feasible_witness)
            i, r, d = p.admissible[a]
            push!(by_block[(r, d)], i)
            rem_surg[p.surgery_surgeon[i], d] -= p.nominal_duration[i]
        end
        @test all(rem_surg .>= -1e-9)
        for (q, (r, d)) in enumerate(p.open_blocks)
            cases = by_block[(r, d)]
            load = sum(p.nominal_duration[i] + p.turnover for i in cases; init=0.0)
            robust = SyntheticLPs._robust_extra_capacity(
                p.duration_deviation[cases], p.uncertainty_budget[q]
            )
            @test load + robust <= p.session_length[r, d] + p.max_overtime[q] + 1e-9
        end
    end
    for seed in 0:3
        _, p = generate_problem(robust_ref, 600, infeasible, seed)
        cert = something(p.infeasibility_certificate)
        check_overload(
            p, cert, p.nominal_duration, i -> unique(t[3] for t in p.admissible if t[1] == i)
        )
    end

    # Weekly planning: the patient path is ICU followed by ward, and beds
    # are constrained through discharge beyond the final surgery day.
    for seed in 0:3, target in (100, 500, 4000)
        model, p = generate_problem(weekly_ref, target, feasible, seed)
        @test num_variables(model) == sum(length, p.admissible_days) + count(!, p.mandatory)
        @test length(p.ward_capacity) == length(p.icu_capacity) == p.bed_horizon
        @test p.bed_horizon >= p.n_days
        occ_ward, occ_icu = zeros(p.bed_horizon), zeros(p.bed_horizon)
        rem_spec, rem_surg = copy(p.specialty_capacity), copy(p.surgeon_budget)
        w = something(p.feasible_witness)
        for i in 1:p.n_surgeries
            d = w[i]
            d == 0 && continue
            @test d in p.admissible_days[i]
            rem_spec[p.surgery_specialty[i], d] -= p.surgery_duration[i] + p.turnover
            rem_surg[p.surgery_surgeon[i], d] -= p.surgery_duration[i]
            icu_days, ward_days = SyntheticLPs._orsched_postop_days(
                d, p.icu_los[i], p.ward_los[i], p.bed_horizon
            )
            @test isempty(intersect(icu_days, ward_days))
            occ_icu[icu_days] .+= 1
            occ_ward[ward_days] .+= 1
        end
        @test all(rem_spec .>= -1e-9) && all(rem_surg .>= -1e-9)
        @test all(occ_ward .<= p.ward_capacity .+ 1e-9)
        @test all(occ_icu .<= p.icu_capacity .+ 1e-9)
        @test all(.!p.mandatory .| (w .> 0))
    end
    for seed in 0:3, target in (200, 3000)
        _, p = generate_problem(weekly_ref, target, infeasible, seed)
        cert = something(p.infeasibility_certificate)
        check_overload(p, cert, p.surgery_duration, i -> p.admissible_days[i]; strict=target >= 3000)
    end

    # Tactical MSS: periodized ICU/ward profiles keep bed-days per block.
    for specialty in SyntheticLPs._ORSCHED_SPECIALTIES, days in (5, 10)
        components = SyntheticLPs._mss_profile_components(specialty, days)
        cases_per_block = 480.0 / (specialty.aggregate_mean + 25.0)
        los_values = collect(specialty.ward_los[1]:specialty.ward_los[2])
        direct_mean_los = sum(los_values) / length(los_values)
        post_icu_mean_los = sum(max(1, los) for los in los_values) / length(los_values)
        @test isapprox(sum(components.icu), cases_per_block * specialty.icu * 1.5)
        @test isapprox(
            sum(components.direct_ward),
            cases_per_block * (1 - specialty.icu) * (1 - specialty.day_case) * direct_mean_los,
        )
        @test isapprox(
            sum(components.post_icu_ward), cases_per_block * specialty.icu * post_icu_mean_los
        )
    end
    for specialty in SyntheticLPs._ORSCHED_SPECIALTIES
        components = SyntheticLPs._mss_profile_components(specialty, 10)
        cases_per_block = 480.0 / (specialty.aggregate_mean + 25.0)
        half_icu_cohort = 0.5 * cases_per_block * specialty.icu
        @test components.post_icu_ward[1] == 0
        @test isapprox(components.icu[2], half_icu_cohort)
        @test isapprox(components.post_icu_ward[2], half_icu_cohort)
        @test components.icu[3] == 0
    end
    # MSS sizing, block plan witness and ward-shortage certificate.
    for seed in 0:3, target in (100, 500, 5000)
        model, p = generate_problem(mss_ref, target, feasible, seed)
        S, R, D, W = p.n_services, p.n_rooms, p.n_days, p.n_wards
        @test num_variables(model) == length(p.admissible_blocks) + R * D + 2S + W * D + D + W + 1
        @test abs(num_variables(model) - target) <= 0.15 * target
        @test Set(p.admissible_blocks) == Set((g, r, d) for g in 1:S for r in p.service_rooms[g] for d in 1:D)
        @test all(!isempty(rs) for rs in p.service_rooms)
        plan = p.admissible_blocks[something(p.feasible_witness)]
        @test length(unique((r, d) for (_, r, d) in plan)) == length(plan)   # room exclusivity
        counts = zeros(Int, S)
        daily = zeros(Int, S, D)
        ward = zeros(W, D)
        icu = zeros(D)
        for (g, r, dp) in plan
            counts[g] += 1
            daily[g, dp] += 1
            w = p.service_ward[g]
            for d in 1:D
                ward[w, d] += p.ward_profile[w, mod(d - dp, D) + 1]
                icu[d] += p.icu_profile[w, mod(d - dp, D) + 1]
            end
        end
        @test all(p.min_blocks .<= counts .<= p.max_blocks)
        @test all(daily[g, d] <= p.max_daily_rooms[g] for g in 1:S, d in 1:D)
        @test all(ward .<= p.ward_capacity .+ 1e-9)
        @test all(icu .<= p.icu_capacity .+ 1e-9)
    end
    for seed in 0:3, target in (200, 5000)
        _, p = generate_problem(mss_ref, target, infeasible, seed)
        cert = something(p.infeasibility_certificate)
        w = cert.ward
        @test cert.services == [g for g in 1:p.n_services if p.service_ward[g] == w]
        mass = sum(p.ward_profile[w, :])
        @test cert.required_bed_days ≈ sum(mass * p.min_blocks[g] for g in cert.services)
        @test cert.capacity_bed_days ≈ sum(p.ward_capacity[w, :])
        @test cert.capacity_bed_days <= 0.9 * cert.required_bed_days + 1e-9
        @test all(p.ward_capacity[w, :] .>= 0)
    end

    # Benchmark variant: published load grid, empirical parameter identity,
    # sparse specialty-block windows, and aggregate load certificates.
    for seed in 0:3, target in (50, 200, 500, 5000)
        model, p = generate_problem(benchmark_ref, target, feasible, seed)
        @test num_variables(model) == sum(length, p.admissible) + count(!, p.mandatory) + p.n_or_days
        @test p.target_load in collect(0.80:0.05:1.20)
        @test abs(p.achieved_load - p.target_load) <= 0.025 + 1e-9
        @test all(
            isapprox(
                p.expected_duration[i],
                p.duration_gamma[i] + exp(p.duration_mu[i] + p.duration_sigma[i]^2 / 2),
            ) for i in 1:p.n_surgeries
        )
        for i in 1:p.n_surgeries
            @test !isempty(p.admissible[i])
            @test all(p.or_day_specialty[q] == p.specialty_code[i] for q in p.admissible[i])
            @test all(p.case_release[i] <= p.or_day_calendar[q] <= p.case_due[i] for q in p.admissible[i])
        end
        load = zeros(p.n_or_days)
        for i in 1:p.n_surgeries
            q = something(p.feasible_witness)[i]
            @test q in p.admissible[i]
            load[q] += p.expected_duration[i]
        end
        @test all(load .<= p.session_length .+ p.max_overtime .+ 1e-9)
    end
    # Rows now scale with columns (the dense model had cases + OR-days rows).
    model, _ = generate_problem(benchmark_ref, 20_000, feasible, 1)
    @test nrows(model) >= 0.08 * num_variables(model)
    for seed in 0:3
        _, p = generate_problem(benchmark_ref, 400, infeasible, seed)
        @test all(p.mandatory)
        @test all(p.max_overtime .== 0)
        @test something(p.infeasibility_excess) ≈
            sum(p.expected_duration) - p.session_length * p.n_or_days
        @test something(p.infeasibility_excess) > 0
    end

    # Time-indexed sequencing: exact sizing, valid columns, conflict-free
    # planted schedule, and the surgeon-day overbooking certificate.
    for seed in 0:3, target in (500, 3000), status in (feasible, infeasible, unknown)
        model, p = generate_problem(sequencing_ref, target, status, seed)
        @test num_variables(model) == length(p.columns) == target
        @test allunique(p.columns)
        for (o, r, d, t) in p.columns
            s = p.case_surgeon[o]
            @test r in p.case_rooms[o] && d in p.case_days[o]
            @test d in p.surgeon_days[s]
            @test p.surgeon_window_start[s] <= t
            @test t + p.case_slots[o] <= p.surgeon_window_end[s] <= p.horizon_slots
            @test p.room_specialty[r] == p.room_specialty[p.case_rooms[o][1]]
        end
        @test all(p.case_slots .== cld.(Int.(p.case_duration), p.slot_minutes))
        assignment, room_rows, surgeon_rows = SyntheticLPs._sequencing_rows(p)
        @test all(!isempty, assignment)
        @test nrows(model) == p.n_cases + length(room_rows) + length(surgeon_rows)
        if status == feasible
            w = something(p.feasible_witness)
            @test length(w) == p.n_cases
            @test [p.columns[c][1] for c in w] == collect(1:p.n_cases)
            chosen = Set(w)
            @test all(count(in(chosen), cols) <= 1 for (_, cols) in room_rows)
            @test all(count(in(chosen), cols) <= 1 for (_, cols) in surgeon_rows)
        elseif status == infeasible
            cert = something(p.infeasibility_certificate)
            @test all(p.case_surgeon[o] == cert.surgeon && p.case_days[o] == [cert.day] for o in cert.cases)
            @test cert.busy_slots == sum(p.case_slots[o] + p.surgeon_turnover for o in cert.cases)
            s = cert.surgeon
            @test cert.available_slots ==
                p.surgeon_window_end[s] - p.surgeon_window_start[s] + p.surgeon_turnover
            @test cert.busy_slots >= 1.05 * cert.available_slots
            # Each case fits the window on its own.
            @test all(
                p.case_slots[o] <= p.surgeon_window_end[s] - p.surgeon_window_start[s] for o in cert.cases
            )
        end
    end

    # Large targets build quickly and keep their size.
    for ref in (elective_ref, sequencing_ref, mss_ref, benchmark_ref, weekly_ref)
        elapsed = @elapsed model, _ = generate_problem(ref, 60_000, feasible, 2)
        @test abs(num_variables(model) - 60_000) <= 0.12 * 60_000
        @test elapsed < 60
    end

    # Field-level determinism for every formulation.
    for ref in values(refs)
        _, p1 = generate_problem(ref, 240, unknown, 12345)
        _, p2 = generate_problem(ref, 240, unknown, 12345)
        @test all(isequal(getfield(p1, f), getfield(p2, f)) for f in fieldnames(typeof(p1)))
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            for ref in values(refs), seed in 1:2, status in (feasible, infeasible), target in (220, 3000)
                model, _ = generate_problem(ref, target, status, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
                @test termination_status(model) == expected
                # The certificates aggregate many rows: presolve alone does
                # not refute them.
                if status == infeasible && target >= 3000
                    @test MOI.get(model, MOI.SimplexIterations()) > 0
                end
            end
            # Unknown is two-sided (seeds verified when calibrating).
            for (ref, target) in (
                (elective_ref, 30_000),
                (weekly_ref, 3000),
                (mss_ref, 3000),
                (robust_ref, 3000),
                (benchmark_ref, 10_000),
                (sequencing_ref, 3000),
            )
                outcomes = Set{MOI.TerminationStatusCode}()
                for seed in 0:5
                    model, _ = generate_problem(ref, target, unknown, seed)
                    set_optimizer(model, HiGHS.Optimizer)
                    set_silent(model)
                    optimize!(model)
                    push!(outcomes, termination_status(model))
                end
                @test outcomes == Set([MOI.OPTIMAL, MOI.INFEASIBLE])
            end
        end
    end
end
