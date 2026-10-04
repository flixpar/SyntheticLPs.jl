# Focused quality contracts for the airline_crew category: registry shape,
# exact sizing, an independent re-validation of every generated pairing against
# the five operational legality properties, dense flight coverage (the guard
# against the old presolve fixing cascade), schedule data conventions, witness
# and crew-shortage certificate arithmetic, the credit-hour cost formula,
# reproducibility, and HiGHS feasibility contracts on the LP relaxation and the
# integer model.
@testset "Airline Crew" begin
    @test :airline_crew in list_categories()
    @test list_variants(:airline_crew) == [:standard]
    info = problem_info(:airline_crew)
    @test info[:default_variant] == :standard
    @test occursin("pairing", lowercase(info[:description]))

    # ---------------------------------------------------------------------
    # Independent pairing validator. Re-derives duty periods from the leg
    # times alone (a ground time above `max_sit` ends a duty) and re-checks
    # the five properties a flyable pairing must have. Deliberately shares no
    # code with the generator.
    # ---------------------------------------------------------------------
    function pairing_violations(p, idx)
        r = p.rules
        legs = p.flights_in_pairing[idx]
        base = p.pairing_bases[idx]
        bad = Set{Symbol}()

        isempty(legs) && return Set([:base_return])
        # Base return: starts at a crew base and ends back at that same base.
        if !(base in p.bases) ||
            p.flight_origins[legs[1]] != base ||
            p.flight_destinations[legs[end]] != base
            push!(bad, :base_return)
        end
        length(unique(legs)) == length(legs) || push!(bad, :time)
        for f in legs
            p.arrival_times[f] > p.departure_times[f] || push!(bad, :time)
        end

        duties = UnitRange{Int}[]
        duty_start = 1
        for i in 1:(length(legs) - 1)
            f, g = legs[i], legs[i + 1]
            # Airport continuity.
            p.flight_destinations[f] == p.flight_origins[g] || push!(bad, :continuity)
            gap = p.departure_times[g] - p.arrival_times[f]
            # Time feasibility: depart after the previous arrival plus sit time.
            gap >= r.min_connect || push!(bad, :time)
            if gap > r.max_sit
                # A duty break; it must be a legal rest.
                r.min_rest <= gap <= r.max_rest || push!(bad, :rest)
                push!(duties, duty_start:i)
                duty_start = i + 1
            end
        end
        push!(duties, duty_start:length(legs))

        length(duties) <= r.max_duties || push!(bad, :duty)
        for d in duties
            length(d) <= r.max_legs_per_duty || push!(bad, :duty)
            block = sum(p.arrival_times[legs[i]] - p.departure_times[legs[i]] for i in d)
            block <= r.max_block_minutes || push!(bad, :duty)
            elapsed = p.arrival_times[legs[last(d)]] - p.departure_times[legs[first(d)]]
            elapsed <= r.max_duty_minutes || push!(bad, :duty)
        end
        return bad
    end

    function instance_violations(p)
        bad = Set{Symbol}()
        for i in 1:length(p.flights_in_pairing)
            union!(bad, pairing_violations(p, i))
        end
        return bad
    end

    # Every generated pairing is operationally legal, at every size, seed and
    # feasibility status. This is the regression guard for the old generator,
    # which chained flights across disconnected airports and then let a
    # filtering step break the sequences that were connected.
    for target in (50, 120, 600, 2000), status in (feasible, infeasible, unknown), seed in 0:4
        _, p = generate_problem(:airline_crew, target, status, seed)
        @test instance_violations(p) == Set{Symbol}()
        # Columns are whole pairings: distinct, non-empty, no repeated leg.
        @test all(!isempty(legs) for legs in p.flights_in_pairing)
        @test length(unique(p.flights_in_pairing)) == length(p.flights_in_pairing)
    end

    # Sizing: one variable per pairing column, and the generator emits exactly
    # the requested number of columns. Rows: one covering equality per flight,
    # one crew-availability row per emitted (base, day), one block-hour band
    # per base that owns a column.
    for target in (50, 100, 500, 1000, 5000), status in (feasible, infeasible, unknown), seed in 0:2
        m, p = generate_problem(:airline_crew, target, status, seed)
        @test num_variables(m) == target
        @test num_variables(m) ==
            length(p.pairing_costs) ==
            length(p.flights_in_pairing) ==
            length(p.pairing_bases) ==
            length(p.pairing_first_day) ==
            length(p.pairing_block_hours)
        n_band = length(unique(p.pairing_bases))
        @test num_constraints(m; count_variable_in_set_constraints=false) ==
            p.num_flights + length(p.crew_rows) + n_band
        # Flights scale with the column budget (about 0.35 per column).
        target >= 1000 && @test 0.30 * target <= p.num_flights <= 0.45 * target
    end

    # Large targets build quickly and keep their shape.
    for target in (20_000, 100_000)
        t = @elapsed m, p = generate_problem(:airline_crew, target, feasible, 3)
        @test num_variables(m) == target
        @test t < 60
        @test 0.30 * target <= p.num_flights <= 0.45 * target
        @test num_constraints(m; count_variable_in_set_constraints=false) > 0.3 * target
    end

    # Dense coverage: every flight is in several pairings, so no singleton
    # covering row can seed a presolve fixing cascade (the old generator's
    # collapse), and the average flight sits in well over ten pairings.
    for seed in 0:3
        _, p = generate_problem(:airline_crew, 4000, unknown, seed)
        cover = zeros(Int, p.num_flights)
        for legs in p.flights_in_pairing, f in legs
            cover[f] += 1
        end
        @test all(cover .>= 1)
        @test count(>=(3), cover) >= 0.97 * p.num_flights
        @test sum(cover) / p.num_flights >= 10
    end

    # Schedule and rule conventions.
    for seed in 0:3
        _, p = generate_problem(:airline_crew, 300, unknown, seed)
        r = p.rules
        @test p.bases == collect(1:length(p.bases))
        @test 2 <= length(p.bases) < p.num_airports
        @test p.num_flights == length(p.flight_origins) == length(p.arrival_times)
        @test all(p.flight_origins .!= p.flight_destinations)
        @test all(p.arrival_times .> p.departure_times)
        # Flight times come from the block-time matrix (distance based).
        @test all(
            p.arrival_times[f] - p.departure_times[f] ==
            p.block_minutes[p.flight_origins[f], p.flight_destinations[f]] for f in 1:p.num_flights
        )
        @test all(
            45 <= p.block_minutes[i, j] <= 240 for
            i in 1:p.num_airports, j in 1:p.num_airports if i != j
        )
        # Duty segmentation is unambiguous, and a two-leg out-and-back always
        # fits one duty (the schedule builder's fallback relies on it).
        @test r.max_sit < r.min_rest
        @test r.min_rest <= r.max_rest
        @test r.max_block_minutes >= 2 * 240
        @test r.max_duty_minutes >= 2 * 240 + r.min_connect
        @test r.min_connect <= r.max_sit
        @test 1 <= r.max_duties && 1 <= r.max_legs_per_duty
        # Calendar bookkeeping of each column.
        for (i, legs) in enumerate(p.flights_in_pairing)
            @test p.pairing_first_day[i] == fld(p.departure_times[legs[1]], 1440) + 1
            @test p.pairing_last_day[i] == fld(p.arrival_times[legs[end]], 1440) + 1
            @test p.pairing_block_hours[i] ≈
                round(sum(p.arrival_times[f] - p.departure_times[f] for f in legs) / 60; digits=2)
        end
        # Emitted crew rows can bind: more columns than crews.
        for (k, (b, d)) in enumerate(p.crew_rows)
            members = count(
                i ->
                    p.pairing_bases[i] == b && p.pairing_first_day[i] <= d <= p.pairing_last_day[i],
                eachindex(p.pairing_bases),
            )
            @test members > p.crew_capacity[k] >= 0
        end
        @test all(p.base_block_lower .<= p.base_block_upper)
    end

    # Recompute every side row's activity at an integral column selection.
    function side_rows_hold(p, chosen)
        for (k, (b, d)) in enumerate(p.crew_rows)
            used = count(
                i ->
                    p.pairing_bases[i] == b && p.pairing_first_day[i] <= d <= p.pairing_last_day[i],
                chosen,
            )
            used <= p.crew_capacity[k] || return false
        end
        for b in p.bases
            hours = sum(
                (p.pairing_block_hours[i] for i in chosen if p.pairing_bases[i] == b); init=0.0
            )
            p.base_block_lower[b] - 1e-9 <= hours <= p.base_block_upper[b] + 1e-9 || return false
        end
        return true
    end

    # Feasible: the planted witness is an exact cover of the flight set that
    # also respects every crew-availability row and block-hour band, so the
    # model has an integral solution.
    for target in (60, 200, 1500), seed in 0:3
        _, p = generate_problem(:airline_crew, target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing
        @test p.infeasibility_certificate === nothing
        @test length(unique(w.pairings)) == length(w.pairings)
        covered = Int[]
        for i in w.pairings
            append!(covered, p.flights_in_pairing[i])
        end
        @test sort(covered) == collect(1:p.num_flights)   # partition, exactly once
        @test all(isempty(pairing_violations(p, i)) for i in w.pairings)
        @test side_rows_hold(p, w.pairings)
    end

    # Infeasible: the crew-shortage certificate recomputes independently from
    # the schedule, the columns and the crew rows.
    for target in (60, 200, 1500, 6000), seed in 0:3
        _, p = generate_problem(:airline_crew, target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing
        @test p.feasible_witness === nothing
        day = cert.day
        on_day(f) = fld(p.departure_times[f], 1440) + 1 == day
        @test cert.flights_on_day == count(on_day, 1:p.num_flights)
        legs_on_day = [count(on_day, legs) for legs in p.flights_in_pairing]
        @test cert.max_legs_on_day == maximum(legs_on_day)
        # The certificate rows are exactly that day's crew rows ...
        @test cert.rows == [k for (k, (_, d)) in enumerate(p.crew_rows) if d == day]
        @test cert.crew_capacity == sum(p.crew_capacity[cert.rows])
        # ... and they cover every pairing flying that day.
        row_bases = Set(p.crew_rows[k][1] for k in cert.rows)
        @test all(
            p.pairing_bases[i] in row_bases for i in eachindex(legs_on_day) if legs_on_day[i] > 0
        )
        # Farkas arithmetic with a 10% margin.
        @test cert.max_legs_on_day * cert.crew_capacity <= 0.9 * cert.flights_on_day
        # No single row is trivially contradictory: every flight is covered.
        covered = falses(p.num_flights)
        for legs in p.flights_in_pairing, f in legs
            covered[f] = true
        end
        @test all(covered)
    end

    # Unknown promises nothing, so it carries no status metadata.
    for seed in 0:4
        _, p = generate_problem(:airline_crew, 200, unknown, seed)
        @test p.feasible_witness === nothing
        @test p.infeasibility_certificate === nothing
    end

    # Credit-hour cost arithmetic, recomputed from the schedule: crews are paid
    # the largest of block time, the duty guarantee and the daily minimum, plus
    # per-diem over the time away from base and a hotel night per overnight.
    for seed in 0:2
        _, p = generate_problem(:airline_crew, 400, feasible, seed)
        for idx in 1:length(p.flights_in_pairing)
            legs = p.flights_in_pairing[idx]
            duties =
                1 + count(
                    p.departure_times[legs[i + 1]] - p.arrival_times[legs[i]] > p.rules.max_sit for
                    i in 1:(length(legs) - 1);
                    init=0,
                )
            block = sum(p.arrival_times[f] - p.departure_times[f] for f in legs)
            duty_time = 0
            start = 1
            for i in 1:length(legs)
                if i == length(legs) ||
                    p.departure_times[legs[i + 1]] - p.arrival_times[legs[i]] > p.rules.max_sit
                    duty_time += p.arrival_times[legs[i]] - p.departure_times[legs[start]]
                    start = i + 1
                end
            end
            tafb = p.arrival_times[legs[end]] - p.departure_times[legs[1]]
            credit = max(
                float(block), p.duty_guarantee * duty_time, float(p.min_daily_credit * duties)
            )
            expected =
                p.pay_rate * credit / 60 + p.per_diem_rate * tafb / 60 + p.hotel_cost * (duties - 1)
            @test p.pairing_costs[idx] ≈ expected
        end
        @test all(p.pairing_costs .> 0)
    end

    # Reproducibility and global-RNG isolation: identical seeds produce
    # field-identical structs even with a seeded/dirty global RNG. The
    # comparison recurses through the witness/certificate structs, whose
    # vector fields would otherwise only compare by identity.
    function deep_isequal(a, b)
        typeof(a) === typeof(b) || return false
        a isa AbstractArray &&
            return size(a) == size(b) && all(deep_isequal(a[i], b[i]) for i in eachindex(a))
        if isstructtype(typeof(a)) && !isempty(fieldnames(typeof(a)))
            return all(deep_isequal(getfield(a, f), getfield(b, f)) for f in fieldnames(typeof(a)))
        end
        return isequal(a, b)
    end
    for status in (feasible, infeasible, unknown)
        Random.seed!(987)
        _, p1 = generate_problem(:airline_crew, 220, status, 42)
        Random.seed!(12345)
        _, p2 = generate_problem(:airline_crew, 220, status, 42)
        @test all(deep_isequal(getfield(p1, f), getfield(p2, f)) for f in fieldnames(typeof(p1)))
    end

    if HAS_HIGHS
        # The feasibility contract holds on the LP relaxation ...
        for target in (60, 220, 900), status in (feasible, infeasible), seed in 0:4
            m, _ = generate_problem(:airline_crew, target, status, seed)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end

        # ... and on the unrelaxed set-partitioning model, which only the
        # planted exact cover can satisfy integrally.
        for target in (60, 220), status in (feasible, infeasible), seed in 0:2
            m, _ = generate_problem(:airline_crew, target, status, seed; relax_integer=false)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end

        # The default infeasibility is not a presolve one-liner: HiGHS needs
        # simplex iterations to refute the crew-shortage aggregation.
        m, _ = generate_problem(:airline_crew, 2000, infeasible, 0)
        set_optimizer(m, HiGHS.Optimizer)
        set_silent(m)
        optimize!(m)
        @test termination_status(m) == MOI.INFEASIBLE
        @test MOI.get(m, MOI.SimplexIterations()) > 0

        # Unknown is a genuine mix, not an implicit always-one-way branch.
        optimal = 0
        infeasible_count = 0
        for seed in 0:19
            m, _ = generate_problem(:airline_crew, 200, unknown, seed)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            if termination_status(m) == MOI.OPTIMAL
                optimal += 1
            elseif termination_status(m) == MOI.INFEASIBLE
                infeasible_count += 1
            end
        end
        @test optimal > 0
        @test infeasible_count > 0

        # The planted witness is an optimal-model-feasible integral solution:
        # fixing it reproduces its own cost.
        m, p = generate_problem(:airline_crew, 200, feasible, 1; relax_integer=false)
        x = all_variables(m)
        for i in p.feasible_witness.pairings
            fix(x[i], 1.0; force=true)
        end
        set_optimizer(m, HiGHS.Optimizer)
        set_silent(m)
        optimize!(m)
        @test termination_status(m) == MOI.OPTIMAL
        @test objective_value(m) ≈ sum(p.pairing_costs[i] for i in p.feasible_witness.pairings)
    end
end
