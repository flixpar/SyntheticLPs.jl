# Focused quality contracts for the nurse_scheduling category: registry shape,
# the exact variable-count formula and sizing, the binary/relaxed split of the
# natural MIP and its LP relaxation, the structural availability minimum, the
# planted-roster witness and skill-shortage certificate arithmetic,
# reproducibility, and HiGHS feasibility contracts on both model classes.

# Field-wise comparison that looks inside the witness struct (arrays inside an
# immutable struct compare by identity under `isequal`).
nurse_same(x::SyntheticLPs.NurseRosterWitness, y::SyntheticLPs.NurseRosterWitness) =
    all(getfield(x, f) == getfield(y, f)
        for f in fieldnames(SyntheticLPs.NurseRosterWitness))
nurse_same(x, y) = isequal(x, y)

# Longest run of consecutive working days, recomputed from a 0/1 roster.
function nurse_longest_run(A::Array{Int,3}, n::Int)
    best = 0
    current = 0
    for d in 1:size(A, 2)
        if sum(A[n, d, :]) > 0
            current += 1
            best = max(best, current)
        else
            current = 0
        end
    end
    return best
end

@testset "Nurse Scheduling" begin
    @test :nurse_scheduling in list_categories()
    @test Set(list_variants(:nurse_scheduling)) == Set([:standard])
    info = problem_info(:nurse_scheduling)
    @test info[:default_variant] == :standard
    @test occursin("nurse", lowercase(info[:description]))

    # Exact variable-count formula, straight from the struct's dimensions.
    for target in (50, 100, 500, 1000, 5000), status in (feasible, infeasible)
        m, p = generate_problem(:nurse_scheduling, target, status, 3)
        @test num_variables(m) == p.n_nurses * p.n_days * p.n_shifts
        @test p.total_shifts == p.n_days * p.n_shifts
        @test num_variables(m) == p.n_nurses * p.total_shifts
    end

    # Sizing: every instance lands within 25% of the target (or <= 50 vars).
    for target in (50, 100, 500, 1000, 5000),
        status in (feasible, infeasible, unknown), seed in 0:3
        m, _ = generate_problem(:nurse_scheduling, target, status, seed)
        nv = num_variables(m)
        @test abs(nv - target) <= 0.25 * target || nv <= 50
    end

    # Shift structure: whole weeks (so weekends exist) and always a night shift
    # with its rest rules.
    for target in (50, 100, 500, 1000, 5000)
        _, p = generate_problem(:nurse_scheduling, target, unknown, 1)
        @test p.n_days % 7 == 0
        @test 2 <= p.n_shifts <= 3
        @test :night in p.shift_labels
        @test !isempty(p.weekend_days)
        @test p.weekend_days == [d for d in 1:p.n_days if mod1(d, 7) in (6, 7)]
    end

    # The natural model is a genuine MIP: every assignment variable is binary
    # when integrality is kept, and none survives the central relaxation.
    for target in (100, 1000), status in (feasible, infeasible, unknown)
        mi, p = generate_problem(:nurse_scheduling, target, status, 7;
                                 relax_integer=false)
        expected = p.n_nurses * p.n_days * p.n_shifts
        @test count(is_binary, all_variables(mi)) == expected
        @test num_variables(mi) == expected
        @test !any(is_integer, all_variables(mi))

        mr, _ = generate_problem(:nurse_scheduling, target, status, 7)
        @test count(is_binary, all_variables(mr)) == 0
        @test !any(is_integer, all_variables(mr))
        @test num_variables(mr) == expected
        # The relaxation is exactly the [0, 1] box the binaries came from.
        @test all(has_lower_bound(v) && lower_bound(v) == 0.0 &&
                  has_upper_bound(v) && upper_bound(v) == 1.0
                  for v in all_variables(mr))
    end

    # The availability repair achieves its promised minimum for *every* slot:
    # a pool drawn from all nurses can re-pick an already-available one, so this
    # is checked arithmetically over a committed sweep rather than assumed.
    shortfalls = 0
    slots = 0
    for target in (50, 100, 500, 1000, 5000),
        status in (feasible, infeasible, unknown), seed in 0:5
        _, p = generate_problem(:nurse_scheduling, target, status, seed)
        @test p.min_available_per_shift == SyntheticLPs.NURSE_MIN_AVAILABLE_PER_SHIFT
        for d in 1:p.n_days, s in 1:p.n_shifts
            slots += 1
            sum(p.availability[:, d, s]) < p.min_available_per_shift &&
                (shortfalls += 1)
        end
        @test all(a in (0, 1) for a in p.availability)
    end
    @test slots > 0
    @test shortfalls == 0

    # The planted roster is a genuine integral feasible point: it satisfies every
    # row of the model, recomputed here directly from the struct's own fields.
    for target in (50, 500, 5000), seed in 0:2
        _, p = generate_problem(:nurse_scheduling, target, feasible, seed)
        @test p.infeasibility_certificate === nothing
        w = p.feasible_witness
        @test w !== nothing
        A = w.assignments
        @test size(A) == (p.n_nurses, p.n_days, p.n_shifts)
        @test all(a in (0, 1) for a in A)
        n_skills = size(p.nurse_skills, 2)
        night_idx = findfirst(==(:night), p.shift_labels)
        early = SyntheticLPs.nurse_early_shift_indices(p.n_shifts)

        # Availability and one-shift-per-day.
        @test all(A[n, d, s] <= p.availability[n, d, s]
                  for n in 1:p.n_nurses, d in 1:p.n_days, s in 1:p.n_shifts)
        @test all(sum(A[n, d, :]) <= 1 for n in 1:p.n_nurses, d in 1:p.n_days)

        # Coverage and skill mix.
        @test all(sum(A[:, d, s]) >= p.demand[d, s]
                  for d in 1:p.n_days, s in 1:p.n_shifts)
        @test all(sum(p.nurse_skills[n, k] * A[n, d, s] for n in 1:p.n_nurses) >=
                  p.skill_requirements[d, s, k]
                  for d in 1:p.n_days, s in 1:p.n_shifts, k in 2:n_skills)

        # Per-nurse aggregates match the roster and respect their bounds.
        @test w.shift_totals == [sum(A[n, :, :]) for n in 1:p.n_nurses]
        @test all(p.min_shifts[n] <= w.shift_totals[n] <= p.max_shifts[n]
                  for n in 1:p.n_nurses)
        @test w.weekend_counts ==
              [sum(A[n, d, s] for d in p.weekend_days, s in 1:p.n_shifts)
               for n in 1:p.n_nurses]
        @test all(p.weekend_bounds[n][1] <= w.weekend_counts[n] <=
                  p.weekend_bounds[n][2] for n in 1:p.n_nurses)
        @test w.night_counts ==
              [sum(A[n, d, night_idx] for d in 1:p.n_days) for n in 1:p.n_nurses]
        @test all(w.night_counts[n] <= p.night_limits[n] for n in 1:p.n_nurses)

        # Consecutive-day limits and post-night rest windows.
        @test w.max_consecutive == [nurse_longest_run(A, n) for n in 1:p.n_nurses]
        @test all(w.max_consecutive[n] <= p.max_consecutive_days[n]
                  for n in 1:p.n_nurses)
        @test all(A[n, d, night_idx] + A[n, d + offset, idx] <= 1
                  for n in 1:p.n_nurses
                  for d in 1:p.n_days
                  for offset in 1:p.rest_after_night[n]
                  if d + offset <= p.n_days
                  for idx in early)
    end

    # The skill-shortage certificate refutes the LP relaxation too: the row asks
    # for more qualified nurses than exist, and every variable is capped at 1.
    for target in (50, 500, 5000), seed in 0:2
        _, q = generate_problem(:nurse_scheduling, target, infeasible, seed)
        @test q.feasible_witness === nothing
        cert = q.infeasibility_certificate
        @test cert !== nothing
        @test cert.qualified == sum(q.nurse_skills[:, cert.skill])
        @test cert.required == cert.qualified + 1
        @test cert.required > cert.qualified
        @test q.skill_requirements[cert.day, cert.shift, cert.skill] == cert.required
        @test 2 <= cert.skill <= size(q.nurse_skills, 2)   # never the base skill
        @test 1 <= cert.day <= q.n_days
        @test 1 <= cert.shift <= q.n_shifts
    end

    # `unknown` is a genuine mix of both branches, not an implicit one-way default.
    statuses = [generate_problem(:nurse_scheduling, 200, unknown, s)[2].feasibility_status
                for s in 0:19]
    @test count(==(feasible), statuses) > 0
    @test count(==(infeasible), statuses) > 0

    # Reproducibility and global-RNG isolation.
    for status in (feasible, infeasible, unknown)
        Random.seed!(987)
        _, p1 = generate_problem(:nurse_scheduling, 400, status, 42)
        Random.seed!(12345)
        _, p2 = generate_problem(:nurse_scheduling, 400, status, 42)
        @test all(nurse_same(getfield(p1, f), getfield(p2, f))
                  for f in fieldnames(typeof(p1)))
    end

    if HAS_HIGHS
        # The feasibility contract holds end-to-end on the LP relaxation...
        for target in (100, 600), status in (feasible, infeasible), s in 0:4
            m, _ = generate_problem(:nurse_scheduling, target, status, s)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end

        # ...and on the unrelaxed integer model, which is the point of planting
        # an integral roster rather than a fractional one.
        for target in (100, 600), status in (feasible, infeasible), s in 0:2
            m, _ = generate_problem(:nurse_scheduling, target, status, s;
                                    relax_integer=false)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            set_time_limit_sec(m, 120.0)
            optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end
    end
end
