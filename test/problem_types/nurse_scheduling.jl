# Focused quality contracts for the nurse_scheduling category: registry shape,
# exact sizing over available assignment slots, ward structure, the
# binary/relaxed split of the natural MIP and its LP relaxation, the per-slot
# availability minimum, the planted-roster witness and night-shortage
# certificate arithmetic, reproducibility, and HiGHS feasibility contracts on
# both model classes.

# Field-wise comparison that looks inside the witness/certificate structs.
nurse_same(x::SyntheticLPs.NurseRosterWitness, y::SyntheticLPs.NurseRosterWitness) =
    all(getfield(x, f) == getfield(y, f) for f in fieldnames(SyntheticLPs.NurseRosterWitness))
nurse_same(x, y) = isequal(x, y)

# Evaluate every row of the model at a 0/1 (or fractional) vector `x`.
function nurse_rows_hold(p, x; tol=1e-9)
    rows = SyntheticLPs.nurse_model_rows(p)
    act(vars) = sum((x[v] for v in vars); init=0.0)
    all(act(v) >= r - tol for (_, v, r) in rows.coverage) || return false
    all(act(v) >= r - tol for (_, v, r) in rows.skill) || return false
    all(act(v) <= 1 + tol for (_, _, v) in rows.one_per_day) || return false
    all(p.min_shifts[n] - tol <= act(v) <= p.max_shifts[n] + tol for (n, v) in rows.totals) ||
        return false
    all(p.weekend_bounds[n][1] - tol <= act(v) <= p.weekend_bounds[n][2] + tol for (n, v) in rows.weekends) ||
        return false
    all(act(v) <= p.night_limits[n] + tol for (n, v) in rows.nights) || return false
    all(act(v) <= p.max_consecutive_days[n] + tol for (n, _, v) in rows.windows) || return false
    all(act(v) <= 1 + tol for (_, _, _, v) in rows.rests) || return false
    return true
end

@testset "Nurse Scheduling" begin
    @test :nurse_scheduling in list_categories()
    @test Set(list_variants(:nurse_scheduling)) == Set([:standard])
    info = problem_info(:nurse_scheduling)
    @test info[:default_variant] == :standard
    @test occursin("nurse", lowercase(info[:description]))

    nrows(m) = num_constraints(m; count_variable_in_set_constraints=false)

    # Exact sizing: one variable per available assignment slot, and the slot
    # list is trimmed to the target exactly. Rows match the row builder.
    for target in (100, 500, 1000, 5000, 20_000), status in (feasible, infeasible, unknown)
        m, p = generate_problem(:nurse_scheduling, target, status, 3)
        @test num_variables(m) == target == length(p.assignment_slots) == length(p.costs)
        rows = SyntheticLPs.nurse_model_rows(p)
        @test nrows(m) == sum(length(getfield(rows, f)) for f in keys(rows))
    end
    # Tiny targets are raised to two variables per (ward, day, shift).
    for seed in 0:2
        m, p = generate_problem(:nurse_scheduling, 5, feasible, seed)
        @test num_variables(m) >= 2 * p.n_wards * p.n_days * p.n_shifts
    end

    # 100k builds quickly; rows scale with columns.
    elapsed = @elapsed m, p = generate_problem(:nurse_scheduling, 100_000, feasible, 1)
    @test num_variables(m) == 100_000
    @test elapsed < 60
    @test nrows(m) > 0.5 * 100_000
    @test p.n_wards >= 30

    # Shift structure: whole weeks (so weekends exist) and always a night shift.
    for target in (50, 100, 500, 1000, 5000)
        _, p = generate_problem(:nurse_scheduling, target, unknown, 1)
        @test p.n_days % 7 == 0
        @test 2 <= p.n_shifts <= 3
        @test :night in p.shift_labels
        @test p.weekend_days == [d for d in 1:p.n_days if mod1(d, 7) in (6, 7)]
    end

    # Slot data invariants: sorted unique slots, wards the nurse serves, no
    # night slot for a nurse who is not night-qualified, the per-slot minimum,
    # and float-pool nurses coupling wards.
    for target in (500, 5000), seed in 0:2
        _, p = generate_problem(:nurse_scheduling, target, unknown, seed)
        @test issorted(p.assignment_slots)
        @test allunique(p.assignment_slots)
        night_idx = findfirst(==(:night), p.shift_labels)
        per_slot = zeros(Int, p.n_wards, p.n_days, p.n_shifts)
        for (n, w, d, s) in p.assignment_slots
            @test w in p.nurse_wards[n]
            s == night_idx && @test p.night_qualified[n]
            per_slot[w, d, s] += 1
        end
        @test all(per_slot .>= p.min_available_per_slot)
        @test p.min_available_per_slot == SyntheticLPs.NURSE_MIN_AVAILABLE_PER_SHIFT
        @test all(p.nurse_skills[:, 1] .== 1)
        @test all(p.costs .> 0)
        target >= 5000 && @test any(length(ws) > 1 for ws in p.nurse_wards)
        @test all(p.nurse_wards[n][1] == mod1(n, p.n_wards) for n in 1:p.n_nurses)
    end

    # The natural model is a genuine MIP; the default relaxation is the [0, 1] box.
    for target in (100, 1000), status in (feasible, infeasible, unknown)
        mi, p = generate_problem(:nurse_scheduling, target, status, 7; relax_integer=false)
        @test count(is_binary, all_variables(mi)) == num_variables(mi) == target
        mr, _ = generate_problem(:nurse_scheduling, target, status, 7)
        @test count(is_binary, all_variables(mr)) == 0
        @test all(
            has_lower_bound(v) && lower_bound(v) == 0.0 && has_upper_bound(v) && upper_bound(v) == 1.0
            for v in all_variables(mr)
        )
    end

    # The planted roster is a genuine integral feasible point: it satisfies every
    # row of the model, and its aggregates recompute from the slots.
    for target in (50, 500, 5000), seed in 0:2
        _, p = generate_problem(:nurse_scheduling, target, feasible, seed)
        @test p.infeasibility_certificate === nothing
        w = p.feasible_witness
        @test w !== nothing
        x = zeros(length(p.assignment_slots))
        x[w.assigned] .= 1.0
        @test allunique(w.assigned)
        @test nurse_rows_hold(p, x)
        night_idx = findfirst(==(:night), p.shift_labels)
        totals = zeros(Int, p.n_nurses)
        nights = zeros(Int, p.n_nurses)
        weekends = zeros(Int, p.n_nurses)
        for v in w.assigned
            n, _, d, s = p.assignment_slots[v]
            totals[n] += 1
            s == night_idx && (nights[n] += 1)
            d in p.weekend_days && (weekends[n] += 1)
        end
        @test w.shift_totals == totals
        @test w.night_counts == nights
        @test w.weekend_counts == weekends
        @test all(w.max_consecutive .<= p.max_consecutive_days)
    end

    # The night-shortage certificate refutes the LP relaxation too: every night
    # variable sits in one night coverage row and one night-limit row.
    for target in (50, 500, 5000), seed in 0:2
        _, q = generate_problem(:nurse_scheduling, target, infeasible, seed)
        @test q.feasible_witness === nothing
        cert = q.infeasibility_certificate
        @test cert !== nothing
        night_idx = findfirst(==(:night), q.shift_labels)
        @test cert.night_demand == sum(q.demand[:, :, night_idx])
        rows = SyntheticLPs.nurse_model_rows(q)
        limited = Set(n for (n, _) in rows.nights)
        @test all(n in limited for (n, _, _, s) in q.assignment_slots if s == night_idx)
        @test cert.night_capacity == sum(q.night_limits[n] for n in limited)
        @test cert.night_capacity <= cert.night_demand - 1
        @test cert.night_capacity <= 0.95 * cert.night_demand
        # Every night coverage row is individually satisfiable (no single-row
        # contradiction for presolve to spot).
        for ((w, d, s), vars, rhs) in rows.coverage
            @test length(vars) >= rhs
        end
    end

    # Unknown carries no metadata.
    for seed in 0:3
        _, p = generate_problem(:nurse_scheduling, 600, unknown, seed)
        @test p.feasible_witness === nothing
        @test p.infeasibility_certificate === nothing
        @test p.feasibility_status == unknown
    end

    # Reproducibility and global-RNG isolation.
    for status in (feasible, infeasible, unknown)
        Random.seed!(987)
        _, p1 = generate_problem(:nurse_scheduling, 400, status, 42)
        Random.seed!(12345)
        _, p2 = generate_problem(:nurse_scheduling, 400, status, 42)
        @test all(nurse_same(getfield(p1, f), getfield(p2, f)) for f in fieldnames(typeof(p1)))
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            # The feasibility contract holds end-to-end on the LP relaxation...
            for target in (100, 600, 3000), status in (feasible, infeasible), s in 0:3
                m, _ = generate_problem(:nurse_scheduling, target, status, s)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
                @test termination_status(m) == expected
                # Infeasibility takes simplex work, not presolve alone.
                status == infeasible && target >= 3000 &&
                    @test MOI.get(m, MOI.SimplexIterations()) > 0
            end

            # ...and on the unrelaxed integer model.
            for target in (100, 600), status in (feasible, infeasible), s in 0:2
                m, _ = generate_problem(:nurse_scheduling, target, status, s; relax_integer=false)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                set_time_limit_sec(m, 120.0)
                optimize!(m)
                expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
                @test termination_status(m) == expected
            end

            # Unknown is two-sided.
            outcomes = Set{MOI.TerminationStatusCode}()
            for s in 0:11
                m, _ = generate_problem(:nurse_scheduling, 1000, unknown, s)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                push!(outcomes, termination_status(m))
            end
            @test outcomes == Set([MOI.OPTIMAL, MOI.INFEASIBLE])
        end
    end
end
