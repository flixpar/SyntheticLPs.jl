# Operating room scheduling: registry, shared-helper properties, exact
# sparse sizing, witnesses, and LP-level infeasibility certificates.
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

    # Regression sweeps for both helper repair bugs.
    for n in 3:11, seed in 1:100
        ids = SyntheticLPs._orsched_case_mix(MersenneTwister(seed), n)
        @test length(unique(ids)) == n
        @test any(SyntheticLPs._ORSCHED_SPECIALTIES[k].aggregate_mean <= 90 for k in ids)
        @test any(SyntheticLPs._ORSCHED_SPECIALTIES[k].aggregate_mean >= 160 for k in ids)
    end
    for (rooms, days, specs) in ((2, 5, 3), (5, 10, 7), (8, 10, 11)), seed in 1:100
        rng = MersenneTwister(seed)
        ids = SyntheticLPs._orsched_case_mix(rng, specs)
        mss, session = SyntheticLPs._orsched_master_schedule(rng, rooms, days, ids)
        @test all(count(==(k), mss) >= max(1, days ÷ 5) for k in 1:specs)
        @test all((mss[r, d] == 0) == (session[r, d] == 0) for r in 1:rooms, d in 1:days)
    end

    # Constructors must not perturb caller/global randomness.
    for ref in values(refs)
        Random.seed!(77123)
        expected_draw = rand()
        Random.seed!(77123)
        generate_problem(ref, 100, unknown, 9)
        @test rand() == expected_draw
    end

    # Elective assignment: sparse graph and planted witness.
    for seed in 0:3, target in (100, 500)
        model, p = generate_problem(elective_ref, target, feasible, seed)
        @test num_variables(model) == length(p.admissible) + p.n_surgeries + length(p.open_blocks)
        @test all(20 <= d <= 480 for d in p.surgery_duration)
        @test all(p.surgery_duration_sd .> 0)
        @test length(p.surgery_type_id) == p.n_surgeries
        @test all(
            (p.mss[r, d] == 0) == (p.session_length[r, d] == 0) for
            r in 1:p.n_rooms, d in 1:p.n_days
        )
        @test all(count(==(k), p.mss) >= max(1, p.n_days ÷ 5) for k in 1:p.n_specialties)
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
    for seed in 0:3
        _, p = generate_problem(elective_ref, 200, infeasible, seed)
        victim = something(p.infeasible_surgery)
        days = unique(t[3] for t in p.admissible if t[1] == victim)
        @test p.mandatory[victim] && !isempty(days)
        @test sum(p.surgeon_budget[p.surgery_surgeon[victim], days]) < p.surgery_duration[victim]
    end

    # Daily sequencing: exact shared-resource pair graph and no-overlap witness.
    for seed in 0:3, target in (100, 500)
        model, p = generate_problem(sequencing_ref, target, feasible, seed)
        @test num_variables(model) ==
            sum(length, p.eligible_rooms) +
              sum(length, p.eligible_surgeons) +
              length(p.room_pairs) +
              length(p.surgeon_pairs) +
              2p.n_surgeries +
              1
        expected_rooms = Tuple{Int, Int, Int}[]
        expected_surgeons = Tuple{Int, Int, Int}[]
        for o in 1:p.n_surgeries, q in (o + 1):p.n_surgeries
            append!(
                expected_rooms,
                [(o, q, r) for r in intersect(p.eligible_rooms[o], p.eligible_rooms[q])],
            )
            append!(
                expected_surgeons,
                [(o, q, s) for s in intersect(p.eligible_surgeons[o], p.eligible_surgeons[q])],
            )
        end
        @test p.room_pairs == expected_rooms
        @test p.surgeon_pairs == expected_surgeons
        room_seen = Dict{Int, Vector{Tuple{Float64, Float64}}}()
        surg_seen = Dict{Int, Vector{Tuple{Float64, Float64}}}()
        for o in 1:p.n_surgeries
            r, s, t = something(p.feasible_witness)[o]
            finish = t + p.surgery_duration[o]
            @test r in p.eligible_rooms[o] && s in p.eligible_surgeons[o]
            @test t >= p.surgeon_window_start[s] - 1e-9
            @test finish <= p.surgeon_window_end[s] + 1e-9
            @test all(
                finish + p.room_turnover <= a + 1e-9 || b + p.room_turnover <= t + 1e-9 for
                (a, b) in get(room_seen, r, [])
            )
            @test all(
                finish + p.surgeon_turnover <= a + 1e-9 || b + p.surgeon_turnover <= t + 1e-9 for
                (a, b) in get(surg_seen, s, [])
            )
            push!(get!(room_seen, r, []), (t, finish))
            push!(get!(surg_seen, s, []), (t, finish))
        end
    end
    for seed in 0:3
        _, p = generate_problem(sequencing_ref, 200, infeasible, seed)
        @test something(p.hard_deadline) < p.surgery_duration[something(p.infeasible_surgery)]
    end

    # Weekly planning: the patient path is ICU followed by ward, and beds
    # are constrained through discharge beyond the final surgery day.
    for seed in 0:3, target in (100, 500)
        model, p = generate_problem(weekly_ref, target, feasible, seed)
        @test num_variables(model) == sum(length, p.admissible_days) + p.n_surgeries
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
    for seed in 0:3
        _, p = generate_problem(weekly_ref, 200, infeasible, seed)
        victim = something(p.infeasible_surgery)
        @test sum(p.surgeon_budget[p.surgery_surgeon[victim], p.admissible_days[victim]]) <
            p.surgery_duration[victim]
    end

    # Tactical MSS: sparse compatible variables, cyclic ICU/ward profiles,
    # and a quota witness/certificate.
    for specialty in SyntheticLPs._ORSCHED_SPECIALTIES, days in (5, 10)
        components = SyntheticLPs._mss_profile_components(specialty, days)
        cases_per_block = 480.0 / (specialty.aggregate_mean + 25.0)
        los_values = collect(specialty.ward_los[1]:specialty.ward_los[2])
        direct_mean_los = sum(los_values) / length(los_values)
        post_icu_mean_los = sum(max(1, los) for los in los_values) / length(los_values)

        # Periodizing the profile must preserve expected bed-days per
        # block, including the complete tail from the prior cycle.
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
    for seed in 0:3, target in (100, 500)
        model, p = generate_problem(mss_ref, target, feasible, seed)
        @test num_variables(model) ==
            length(p.admissible_blocks) + p.n_rooms * p.n_days + 2p.n_specialties + 2p.n_days + 2
        @test Set(p.admissible_blocks) == Set(
            (s, r, d) for d in 1:p.n_days, r in 1:p.n_rooms, s in 1:p.n_specialties if
            p.room_specialty_compatible[r, s]
        )
        w = something(p.feasible_witness)
        @test all(sum(w[:, r, d]) <= 1 for r in 1:p.n_rooms, d in 1:p.n_days)
        for s in 1:p.n_specialties
            blocks = sum(w[s, :, :])
            @test p.min_blocks[s] <= blocks <= p.max_blocks[s]
            @test all(sum(w[s, :, d]) <= p.max_daily_rooms[s] for d in 1:p.n_days)
            @test all(
                w[s, r, d] == 0 || p.room_specialty_compatible[r, s] for
                r in 1:p.n_rooms, d in 1:p.n_days
            )
        end
        for d in 1:p.n_days
            ward = sum(
                p.ward_profile[s, mod(d - dp, p.n_days) + 1] * w[s, r, dp] for
                s in 1:p.n_specialties, r in 1:p.n_rooms, dp in 1:p.n_days
            )
            icu = sum(
                p.icu_profile[s, mod(d - dp, p.n_days) + 1] * w[s, r, dp] for
                s in 1:p.n_specialties, r in 1:p.n_rooms, dp in 1:p.n_days
            )
            @test ward <= p.ward_capacity[d] + 1e-9
            @test icu <= p.icu_capacity[d] + 1e-9
        end
    end
    for seed in 0:3
        _, p = generate_problem(mss_ref, 200, infeasible, seed)
        s = something(p.infeasible_specialty)
        @test p.min_blocks[s] > sum(p.room_specialty_compatible[:, s]) * p.n_days
    end

    # Robust assignment: every dual variable corresponds to a sparse
    # admissible triple, and the witness satisfies the exact Γ-budget load.
    for seed in 0:3, target in (100, 500)
        model, p = generate_problem(robust_ref, target, feasible, seed)
        @test num_variables(model) == 2length(p.admissible) + p.n_surgeries + 2length(p.open_blocks)
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
        _, p = generate_problem(robust_ref, 200, infeasible, seed)
        victim = something(p.infeasible_surgery)
        days = unique(t[3] for t in p.admissible if t[1] == victim)
        @test sum(p.surgeon_budget[p.surgery_surgeon[victim], days]) < p.nominal_duration[victim]
    end

    # Benchmark variant: published load grid, empirical parameter identity,
    # and aggregate load certificates.
    for seed in 0:3, target in (50, 200, 500)
        model, p = generate_problem(benchmark_ref, target, feasible, seed)
        @test num_variables(model) == p.n_surgeries * p.n_or_days + p.n_surgeries + p.n_or_days
        @test p.target_load in collect(0.80:0.05:1.20)
        @test abs(p.achieved_load - p.target_load) <= 0.025 + 1e-9
        @test all(
            isapprox(
                p.expected_duration[i],
                p.duration_gamma[i] + exp(p.duration_mu[i] + p.duration_sigma[i]^2 / 2),
            ) for i in 1:p.n_surgeries
        )
        load = zeros(p.n_or_days)
        for i in 1:p.n_surgeries
            load[something(p.feasible_witness)[i]] += p.expected_duration[i]
        end
        @test all(load .<= p.session_length .+ p.max_overtime .+ 1e-9)
    end
    for seed in 0:3
        _, p = generate_problem(benchmark_ref, 200, infeasible, seed)
        @test all(p.mandatory)
        @test something(p.infeasibility_excess) ==
            sum(p.expected_duration) - p.session_length * p.n_or_days
        @test something(p.infeasibility_excess) > 0
    end

    # Field-level determinism for every formulation.
    for ref in values(refs)
        _, p1 = generate_problem(ref, 240, unknown, 12345)
        _, p2 = generate_problem(ref, 240, unknown, 12345)
        @test all(isequal(getfield(p1, f), getfield(p2, f)) for f in fieldnames(typeof(p1)))
    end

    if HAS_HIGHS
        for ref in values(refs), seed in 1:3, status in (feasible, infeasible)
            model, _ = generate_problem(ref, 220, status, seed)
            set_optimizer(model, HiGHS.Optimizer)
            set_silent(model)
            optimize!(model)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(model) == expected
        end
    end
end
