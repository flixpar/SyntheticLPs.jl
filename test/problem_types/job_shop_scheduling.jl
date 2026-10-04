# Focused quality contracts for the job_shop_scheduling category: registry
# shape, column-budget sizing and the exact row formula of the time-indexed
# model, routing/window data invariants, planted list-schedule witness checked
# slot by slot and against the built model, energetic interval-load
# certificate arithmetic recomputed from the struct fields, reproducibility
# under a dirty global RNG, and HiGHS contracts — including the check that
# the LP relaxation keeps machine contention (the big-M formulation this
# replaced relaxed to the no-contention bound exactly).
@testset "Job Shop Scheduling" begin
    @test :job_shop_scheduling in list_categories()
    @test list_variants(:job_shop_scheduling) == [:standard]

    function js_expected_rows(p)
        rows = p.n_ops + sum(length(ops) - 1 for ops in p.job_ops)
        for w in 1:p.n_work_centers
            ops_w = [o for o in 1:p.n_ops if p.op_work_center[o] == w]
            isempty(ops_w) && continue
            running = Dict{Int, Int}()
            for o in ops_w, t in p.earliest[o]:(p.latest[o] + p.proc[o] - 1)
                running[t] = get(running, t, 0) + 1
            end
            rows += count(>(p.capacity[w]), values(running))
        end
        return rows
    end

    # Sizing: the start-variable count lands within one job's routing length
    # of the target (deadline slack is spent in whole-job increments), and
    # the row count matches assignment + precedence + every capacity row that
    # can bind.
    for target in (60, 300, 1500, 6000), status in (feasible, infeasible, unknown), seed in 0:2
        m, p = generate_problem(:job_shop_scheduling, target, status, seed)
        @test num_variables(m) == sum(p.latest .- p.earliest .+ 1)
        @test abs(num_variables(m) - target) <= max(8, 0.02 * target)
        @test num_constraints(m; count_variable_in_set_constraints=false) == js_expected_rows(p)
    end
    # Large target: exact up to the routing-length granularity, fast to build.
    m, p = generate_problem(:job_shop_scheduling, 100_000, feasible, 0)
    @test abs(num_variables(m) - 100_000) <= 8
    @test num_constraints(m; count_variable_in_set_constraints=false) > 10_000

    # Data invariants.
    for target in (120, 900, 4000), status in (feasible, infeasible, unknown), seed in 0:1
        _, p = generate_problem(:job_shop_scheduling, target, status, seed)
        @test p.n_ops == length(p.proc) == length(p.op_job) == length(p.op_work_center)
        @test sort(reduce(vcat, p.job_ops)) == collect(1:p.n_ops)
        @test all(1 <= c <= 3 for c in p.capacity)
        @test all(1 <= d <= 12 for d in p.proc)
        for j in 1:p.n_jobs
            ops = p.job_ops[j]
            @test all(p.op_job[o] == j for o in ops)
            @test allunique(p.op_work_center[o] for o in ops)   # distinct centers
            @test length(ops) >= 1
            # Windows from release + routing head and deadline - routing tail.
            head = 0
            tail = sum(p.proc[o] for o in ops)
            for o in ops
                @test p.earliest[o] == p.release[j] + head
                @test p.latest[o] == p.deadline[j] - tail + 1
                @test p.latest[o] >= p.earliest[o]
                head += p.proc[o]
                tail -= p.proc[o]
            end
            @test p.release[j] >= 1
            @test p.deadline[j] <= p.horizon
        end
        @test all(>(0.0), p.weight)
        @test p.wip_cost > 0
        @test isempty(p.expedited_jobs) == (status == feasible)
    end

    # Planted list schedule: inside every window, routing order respected,
    # no work center over capacity in any slot, and a 0/1 feasible point of
    # the built model.
    for target in (100, 1000, 5000), seed in 0:2
        m, p = generate_problem(:job_shop_scheduling, target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing
        @test p.infeasibility_certificate === nothing
        @test all(p.earliest .<= w.start .<= p.latest)
        for (j, ops) in enumerate(p.job_ops)
            for i in 1:(length(ops) - 1)
                @test w.start[ops[i + 1]] >= w.start[ops[i]] + p.proc[ops[i]]
            end
            @test w.completion[j] == w.start[ops[end]] + p.proc[ops[end]] - 1
            @test w.completion[j] <= p.deadline[j]
        end
        load = Dict{Tuple{Int, Int}, Int}()
        for o in 1:p.n_ops, t in w.start[o]:(w.start[o] + p.proc[o] - 1)
            key = (p.op_work_center[o], t)
            load[key] = get(load, key, 0) + 1
        end
        @test all(v <= p.capacity[k[1]] for (k, v) in load)
        op, slot, _ = SyntheticLPs._job_shop_columns(p)
        point = Dict(m[:x][c] => (slot[c] == w.start[op[c]] ? 1.0 : 0.0) for c in eachindex(op))
        @test isempty(primal_feasibility_report(m, point; atol=1e-9))
    end

    # Energetic certificate: every listed operation runs on the certified
    # work center and must lie entirely inside the interval, and their total
    # processing exceeds the interval's machine-slots by the planted margin.
    for target in (100, 1000, 5000), seed in 0:3
        _, p = generate_problem(:job_shop_scheduling, target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing
        @test p.feasible_witness === nothing
        @test length(cert.operations) >= 2
        len = cert.interval_end - cert.interval_start + 1
        for o in cert.operations
            @test p.op_work_center[o] == cert.work_center
            @test p.earliest[o] >= cert.interval_start
            @test p.latest[o] + p.proc[o] - 1 <= cert.interval_end
            @test p.op_job[o] in p.expedited_jobs
        end
        @test cert.required_load == sum(p.proc[o] for o in cert.operations)
        @test cert.available_capacity == p.capacity[cert.work_center] * len
        @test cert.required_load >= 1.08 * cert.available_capacity
    end

    for seed in 0:5
        _, p = generate_problem(:job_shop_scheduling, 400, unknown, seed)
        @test p.feasible_witness === nothing
        @test p.infeasibility_certificate === nothing
    end

    # Reproducibility, isolated from a dirty global RNG.
    for status in (feasible, infeasible, unknown)
        Random.seed!(3)
        _, p1 = generate_problem(:job_shop_scheduling, 800, status, 21)
        Random.seed!(77)
        _, p2 = generate_problem(:job_shop_scheduling, 800, status, 21)
        for f in fieldnames(typeof(p1))
            f in (:feasible_witness, :infeasibility_certificate) && continue
            @test isequal(getfield(p1, f), getfield(p2, f))
        end
        if p1.feasible_witness !== nothing
            @test p1.feasible_witness.start == p2.feasible_witness.start
        end
        if p1.infeasibility_certificate !== nothing
            @test p1.infeasibility_certificate.operations == p2.infeasibility_certificate.operations
        end
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            for target in (150, 800, 3000), status in (feasible, infeasible), seed in 0:2
                m, _ = generate_problem(:job_shop_scheduling, target, status, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
                if status == infeasible
                    # Not disproved by presolve alone.
                    @test MOI.get(m, MOI.SimplexIterations()) > 0
                end
            end

            # `unknown` lands on both sides.
            outcomes = Set{MOI.TerminationStatusCode}()
            for seed in 0:9
                m, _ = generate_problem(:job_shop_scheduling, 500, unknown, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                push!(outcomes, termination_status(m))
            end
            @test outcomes == Set([MOI.OPTIMAL, MOI.INFEASIBLE])

            # Machine contention survives relaxation: dropping the capacity
            # rows (no-contention bound) strictly lowers the relaxed optimum.
            gaps = 0
            for seed in 0:3
                m, p = generate_problem(:job_shop_scheduling, 1500, feasible, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                relaxed = objective_value(m)
                free = deepcopy(p)
                free.capacity .= 10^6
                m0 = SyntheticLPs.build_model(free)
                relax_integrality(m0)
                set_optimizer(m0, HiGHS.Optimizer)
                set_silent(m0)
                optimize!(m0)
                @test objective_value(m0) <= relaxed + 1e-6
                gaps += relaxed > objective_value(m0) * 1.01 + 1e-6
            end
            @test gaps >= 3
        end
    end
end
