# Focused quality contracts for the assignment category: registry shape, exact
# sparse sizing, eligibility invariants (no dead `x == 0` columns), witness
# arithmetic (planted matching / greedy balanced plan, checked against the
# built model), Hall and pigeonhole certificates recomputed from the struct
# fields, reproducibility, and HiGHS contracts.

@testset "Assignment" begin
    @test Set(list_variants(:assignment)) == Set([:standard, :workload_balance])
    @test problem_info(:assignment)[:default_variant] == :standard

    @testset "sizing" begin
        for target in (10, 100, 1000, 20_000), status in (feasible, infeasible, unknown)
            m, p = generate_problem("assignment/standard", target, status, 4)
            @test num_variables(m) == length(p.edges)
            target >= 100 && @test num_variables(m) == target
            n_worker_rows = length(unique(w for (w, _) in p.edges))
            @test num_constraints(m; count_variable_in_set_constraints=false) ==
                p.n_jobs + n_worker_rows

            m, q = generate_problem("assignment/workload_balance", target, status, 4)
            @test num_variables(m) == length(q.edges) + 1
            target >= 100 && @test num_variables(m) == target
            @test num_constraints(m; count_variable_in_set_constraints=false) ==
                q.n_tasks + q.n_workers
        end
        @test_throws ArgumentError SyntheticLPs.AssignmentProblem(
            SyntheticLPs.ASSIGNMENT_MAX_VARIABLES + 1, unknown, 0
        )
        @test_throws ArgumentError SyntheticLPs.WorkloadBalanceAssignmentProblem(0, unknown, 0)
        # Large-target sizing (constructors only; both build in well under a second).
        @test length(SyntheticLPs.AssignmentProblem(100_000, unknown, 0).edges) == 100_000
        @test length(SyntheticLPs.WorkloadBalanceAssignmentProblem(100_000, infeasible, 0).edges) ==
            99_999
    end

    @testset "eligibility invariants" begin
        for seed in 0:2
            _, p = generate_problem("assignment/standard", 3000, feasible, seed)
            @test issorted(p.edges) && allunique(p.edges)
            deg = zeros(Int, p.n_jobs)
            for (_, j) in p.edges
                deg[j] += 1
            end
            @test all(>=(2), deg)
            @test p.qualified == [p.job_skill[j] in p.worker_skills[w] for (w, j) in p.edges]
            @test count(p.qualified) >= 0.8 * length(p.edges)       # mostly qualified
            # Unqualified pairs pay the cross-skill premium.
            @test all(>(0.0), p.costs)
            _, q = generate_problem("assignment/workload_balance", 3000, feasible, seed)
            @test issorted(q.edges) && allunique(q.edges)
            @test all(>(0.0), q.processing_time)
            @test all(a -> a in (0.5, 0.75, 1.0), q.availability)
            @test q.max_makespan > 0 && q.makespan_weight > 0
        end
    end

    @testset "witnesses" begin
        for target in (100, 3000), seed in 0:3
            m, p = generate_problem("assignment/standard", target, feasible, seed)
            w = p.feasible_witness
            @test length(w.edge_of_job) == p.n_jobs
            @test [p.edges[e][2] for e in w.edge_of_job] == collect(1:(p.n_jobs))
            @test allunique(p.edges[e][1] for e in w.edge_of_job)
            vals = Dict(m[:x][e] => 0.0 for e in eachindex(p.edges))
            for e in w.edge_of_job
                vals[m[:x][e]] = 1.0
            end
            @test isempty(primal_feasibility_report(m, vals; atol=1e-9))

            m, q = generate_problem("assignment/workload_balance", target, feasible, seed)
            wq = q.feasible_witness
            load = zeros(q.n_workers)
            for e in wq.edge_of_task
                load[q.edges[e][1]] += q.processing_time[e]
            end
            @test wq.makespan ≈ maximum(load ./ q.availability)
            @test wq.makespan <= q.max_makespan
            vals = Dict{VariableRef, Float64}(m[:x][e] => 0.0 for e in eachindex(q.edges))
            for e in wq.edge_of_task
                vals[m[:x][e]] = 1.0
            end
            vals[m[:L]] = wq.makespan
            @test isempty(primal_feasibility_report(m, vals; atol=1e-6))
        end
    end

    @testset "certificates" begin
        group_sized = 0
        for target in (100, 3000, 20_000), seed in 0:3
            _, p = generate_problem("assignment/standard", target, infeasible, seed)
            c = p.infeasibility_certificate
            J = Set(c.jobs)
            @test c.workers == sort!(unique([w for (w, j) in p.edges if j in J]))
            @test length(c.workers) < length(c.jobs)
            group_sized += length(c.jobs) < p.n_jobs

            _, q = generate_problem("assignment/workload_balance", target, infeasible, seed)
            cq = q.infeasibility_certificate
            Tq = Set(cq.tasks)
            @test cq.workers == sort!(unique([w for (w, t) in q.edges if t in Tq]))
            fastest = sum(
                minimum(q.processing_time[e] for e in eachindex(q.edges) if q.edges[e][2] == t) for
                t in cq.tasks
            )
            @test cq.required ≈ fastest
            @test cq.available ≈ sum(q.availability[cq.workers]) * q.max_makespan
            @test cq.available < cq.required
        end
        @test group_sized >= 8   # skill-group Hall sets, not just a global shortfall
    end

    @testset "reproducibility" begin
        for ref in ("assignment/standard", "assignment/workload_balance"),
            status in (feasible, infeasible, unknown)

            Random.seed!(8)
            _, p1 = generate_problem(ref, 900, status, 13)
            Random.seed!(9)
            rand(3)
            _, p2 = generate_problem(ref, 900, status, 13)
            for f in fieldnames(typeof(p1))
                a, b = getfield(p1, f), getfield(p2, f)
                if a === nothing || a isa Union{Number, Symbol, AbstractArray, FeasibilityStatus}
                    @test isequal(a, b)
                else
                    @test all(
                        isequal(getfield(a, g), getfield(b, g)) for g in fieldnames(typeof(a))
                    )
                end
            end
        end
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            function asg_solve(m)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                return termination_status(m), MOI.get(m, MOI.SimplexIterations())
            end
            for ref in ("assignment/standard", "assignment/workload_balance"),
                target in (150, 4000),
                seed in 0:2

                ts, _ = asg_solve(generate_problem(ref, target, feasible, seed)[1])
                @test ts == MOI.OPTIMAL
                ts, iters = asg_solve(generate_problem(ref, target, infeasible, seed)[1])
                @test ts == MOI.INFEASIBLE
                target >= 4000 && @test iters > 0
            end
            for ref in ("assignment/standard", "assignment/workload_balance")
                outcomes = Set{Any}()
                for seed in 0:15
                    push!(outcomes, asg_solve(generate_problem(ref, 2000, unknown, seed)[1])[1])
                end
                @test outcomes == Set([MOI.OPTIMAL, MOI.INFEASIBLE])
            end
        end
    end
end
