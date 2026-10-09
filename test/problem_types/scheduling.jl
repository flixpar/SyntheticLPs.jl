# Focused quality contracts for the scheduling category: registry shape,
# column-budget sizing and the exact row formula of the rostering model,
# worker/contract data invariants, the planted roster checked rule by rule and
# against the built model, the department-week staffing certificate recomputed
# from the struct fields (including that no single coverage row is
# unattainable), reproducibility under a dirty global RNG, and HiGHS contracts.
@testset "Scheduling" begin
    @test :scheduling in list_categories()
    @test list_variants(:scheduling) == [:standard]

    function sched_rows(p)
        cols = SyntheticLPs._scheduling_columns(p)
        D, c = p.n_days, p.max_consecutive
        wd = Dict{Tuple{Int, Int}, Vector{Int}}()
        for (j, (w, d, k, m)) in enumerate(cols)
            push!(get!(wd, (w, d), Int[]), j)
        end
        rows = count(>(0.0), p.requirement)
        for w in 1:p.n_workers
            rows += count(d -> length(get(wd, (w, d), Int[])) > 1, 1:D)
            rows += count(
                wk -> any(haskey(wd, (w, d)) for d in ((wk - 1) * 7 + 1):min(wk * 7, D)),
                1:cld(D, 7),
            )
            rows += count(s -> count(d -> haskey(wd, (w, d)), s:(s + c)) > c, 1:(D - c))
            rows += count(
                d ->
                    any(p.closing[cols[j][3]] for j in get(wd, (w, d), Int[])) &&
                    any(p.opening[cols[j][3]] for j in get(wd, (w, d + 1), Int[])),
                1:(D - 1),
            )
        end
        return rows
    end

    for target in (50, 400, 3000), status in (feasible, infeasible, unknown), seed in 0:1
        m, p = generate_problem(:scheduling, target, status, seed)
        cols = SyntheticLPs._scheduling_columns(p)
        @test num_variables(m) == length(cols)
        @test num_constraints(m; count_variable_in_set_constraints=false) == sched_rows(p)
        @test abs(length(cols) - target) <= max(0.1 * target, 15)
    end
    m, p = generate_problem(:scheduling, 100_000, unknown, 0)
    @test abs(num_variables(m) - 100_000) <= 1_000
    @test num_constraints(m; count_variable_in_set_constraints=false) >= 30_000

    # Data invariants.
    for target in (200, 2000), status in (feasible, infeasible), seed in 0:1
        _, p = generate_problem(:scheduling, target, status, seed)
        @test p.n_days % 7 == 0
        @test all(p.home[w] in p.departments[w] for w in 1:p.n_workers)
        @test all(
            (p.efficiency[w, m] > 0) == (m in p.departments[w]) for
            w in 1:p.n_workers, m in 1:p.n_departments
        )
        @test all(!isempty(ks) && issubset(ks, 1:p.n_templates) for ks in p.templates)
        @test all(0 .<= p.min_hours .<= p.max_hours)
        @test all(>(0.0), p.wage)
        @test all(p.requirement .>= 0)
    end

    # Planted roster: every rule by hand, then the built model.
    for target in (200, 2000), seed in 0:2
        m, p = generate_problem(:scheduling, target, feasible, seed)
        r = p.feasible_witness
        @test r !== nothing && p.infeasibility_certificate === nothing
        supply = zeros(size(p.requirement))
        for w in 1:p.n_workers
            a = r.assignment[w]
            days = [x[1] for x in a]
            @test allunique(days)                                   # one shift per day
            for (d, k, dep) in a
                @test p.available[w, d]
                @test k in p.templates[w]
                @test dep in p.departments[w]
                supply[d, k, dep] += p.efficiency[w, dep]
            end
            for wk in 1:cld(p.n_days, 7)
                h = sum((p.template_length[k] for (d, k, _) in a if cld(d, 7) == wk); init=0.0)
                @test p.min_hours[w] - 1e-9 <= h <= p.max_hours[w] + 1e-9
            end
            worked = falses(p.n_days)
            worked[days] .= true
            for s in 1:(p.n_days - p.max_consecutive)
                @test count(worked[s:(s + p.max_consecutive)]) <= p.max_consecutive
            end
            kd = Dict(d => k for (d, k, _) in a)
            for d in 1:(p.n_days - 1)
                if haskey(kd, d) && haskey(kd, d + 1)
                    @test !(p.closing[kd[d]] && p.opening[kd[d + 1]])
                end
            end
        end
        @test all(supply .>= p.requirement .- 1e-9)
        cols = SyntheticLPs._scheduling_columns(p)
        on = Set((w, d, k, dep) for w in 1:p.n_workers for (d, k, dep) in r.assignment[w])
        point = Dict(m[:x][j] => (c in on ? 1.0 : 0.0) for (j, c) in enumerate(cols))
        @test isempty(primal_feasibility_report(m, point; atol=1e-7))
    end

    # Department-week certificate.
    for target in (200, 2000), seed in 0:3
        _, p = generate_problem(:scheduling, target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing && p.feasible_witness === nothing
        dep, wk = cert.department, cert.week
        @test cert.workers == [w for w in 1:p.n_workers if p.efficiency[w, dep] > 0]
        days = ((wk - 1) * 7 + 1):(wk * 7)
        for (w, cap) in zip(cert.workers, cert.caps)
            shortest = minimum(p.template_length[k] for k in p.templates[w])
            @test cap ≈ min(count(d -> p.available[w, d], days), p.max_hours[w] / shortest)
        end
        @test cert.required ≈ sum(p.requirement[d, k, dep] for d in days, k in 1:p.n_templates)
        @test cert.available ≈
            sum(p.efficiency[w, dep] * c for (w, c) in zip(cert.workers, cert.caps))
        @test cert.required >= 1.08 * cert.available
        # No single coverage row is unattainable on its own.
        for d in days, k in 1:p.n_templates
            reach = sum(
                (
                    p.efficiency[w, dep] for
                    w in cert.workers if p.available[w, d] && k in p.templates[w]
                );
                init=0.0,
            )
            @test p.requirement[d, k, dep] <= reach + 1e-9
        end
    end

    for seed in 0:3
        _, p = generate_problem(:scheduling, 500, unknown, seed)
        @test p.feasible_witness === nothing && p.infeasibility_certificate === nothing
    end

    for status in (feasible, infeasible, unknown)
        Random.seed!(2)
        _, p1 = generate_problem(:scheduling, 900, status, 31)
        Random.seed!(2024)
        _, p2 = generate_problem(:scheduling, 900, status, 31)
        for f in fieldnames(typeof(p1))
            f in (:feasible_witness, :infeasibility_certificate) && continue
            @test isequal(getfield(p1, f), getfield(p2, f))
        end
        if p1.feasible_witness !== nothing
            @test p1.feasible_witness.assignment == p2.feasible_witness.assignment
        end
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            for target in (150, 1500), status in (feasible, infeasible), seed in 0:2
                m, _ = generate_problem(:scheduling, target, status, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
                @test MOI.get(m, MOI.SimplexIterations()) > 0
            end
            outcomes = Set{MOI.TerminationStatusCode}()
            for seed in 0:9
                m, _ = generate_problem(:scheduling, 600, unknown, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                push!(outcomes, termination_status(m))
            end
            @test outcomes == Set([MOI.OPTIMAL, MOI.INFEASIBLE])
        end
    end
end
