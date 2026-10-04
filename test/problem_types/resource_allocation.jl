# Focused quality contracts for the resource_allocation category: registry
# shape, exact column-count sizing and the row formula, window/eligibility data
# invariants, planted-plan witness feasibility (via JuMP's primal feasibility
# report) and department over-commitment certificate arithmetic recomputed from
# the struct fields, the two-sided `unknown` profile, reproducibility under a
# dirty global RNG, and HiGHS feasibility contracts including a presolve
# survival regression (the previous single-period formulation presolved to an
# empty model on every instance).
@testset "Resource Allocation" begin
    @test :resource_allocation in list_categories()
    @test list_variants(:resource_allocation) == [:standard]
    info = problem_info(:resource_allocation)
    @test info[:default_variant] == :standard
    @test occursin("resource", lowercase(info[:description]))

    ra_rows(p) =
        length(
            Set(
                (q, t) for a in 1:p.n_activities for q in p.eligible_pools[a] for
                t in p.release[a]:p.deadline[a]
            ),
        ) +
        p.n_activities +
        sum(
            length(p.eligible_pools[a]) > 1 ? p.deadline[a] - p.release[a] + 1 : 0 for
            a in 1:p.n_activities
        )

    # Sizing: the column count is exactly the target (targets below 4 round
    # up), rows are one per used (pool, period) pair, one scope row per
    # activity, and one rate row per in-window period of every multi-pool
    # activity (single-pool rate limits become variable bounds).
    for target in (4, 50, 500, 3000), status in (feasible, infeasible, unknown), seed in 0:2
        m, p = generate_problem(:resource_allocation, target, status, seed)
        act, _, _ = SyntheticLPs._resource_allocation_columns(p)
        @test num_variables(m) == length(act) == max(target, 4)
        @test num_constraints(m; count_variable_in_set_constraints=false) == ra_rows(p)
    end
    for target in (1, 2, 3)
        m, _ = generate_problem(:resource_allocation, target, feasible, 0)
        @test num_variables(m) == 4
    end

    # Structural data contracts.
    for target in (80, 900, 4000), status in (feasible, infeasible, unknown)
        _, p = generate_problem(:resource_allocation, target, status, 3)
        @test p.profile in (:engineering_portfolio, :cloud_capacity, :maintenance_crews)
        @test size(p.capacity) == (p.n_pools, p.n_periods)
        @test all(>(0.0), p.capacity)
        @test length(p.pool_department) == p.n_pools
        @test sort(unique(p.pool_department)) == collect(1:p.n_departments)
        @test all(1 .<= p.release .<= p.deadline .<= p.n_periods)
        @test all(issorted(e) && allunique(e) && !isempty(e) for e in p.eligible_pools)
        @test all(length(p.efficiency[a]) == length(p.eligible_pools[a]) for a in 1:p.n_activities)
        @test all(all(>(0.0), e) for e in p.efficiency)
        @test all(0.0 .<= p.floors .< p.workload)
        @test all(>(0.0), p.rate_cap)
        @test 0.97 - 1e-12 <= p.discount < 1.0
        # Most activities are profitable at an average hour: value per unit
        # exceeds the mean pool cost, so dual fixing cannot zero the model.
        @test count(p.value .> sum(p.pool_cost) / p.n_pools) >= 0.7 * p.n_activities
        @test p.feasibility_status == status
    end

    # Planted-plan witness: arithmetic on the struct fields and the built model.
    for target in (60, 700, 3000), seed in 0:2
        m, p = generate_problem(:resource_allocation, target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing
        @test p.infeasibility_certificate === nothing
        act, pool, period = SyntheticLPs._resource_allocation_columns(p)
        @test length(w.allocation) == length(act)
        @test all(>(0.0), w.allocation)
        hours = zeros(p.n_pools, p.n_periods)
        output = zeros(p.n_activities)
        for c in eachindex(act)
            hours[pool[c], period[c]] += w.allocation[c]
            k = findfirst(==(pool[c]), p.eligible_pools[act[c]])
            output[act[c]] += p.efficiency[act[c]][k] * w.allocation[c]
        end
        @test hours ≈ w.pool_hours
        @test output ≈ w.output
        @test all(w.pool_hours .< p.capacity)           # strict slack on pool rows
        @test all(p.floors .<= w.output .+ 1e-9)
        @test all(w.output .< p.workload)
        report = primal_feasibility_report(
            m, Dict(m[:y][c] => w.allocation[c] for c in eachindex(act)); atol=1e-7
        )
        @test isempty(report)
    end

    # Department over-commitment certificate: recomputed from the data.
    for target in (60, 700, 3000), seed in 0:3
        _, p = generate_problem(:resource_allocation, target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing
        @test p.feasible_witness === nothing
        pools = Set(cert.pools)
        if cert.department == 0
            @test pools == Set(1:p.n_pools)
        else
            @test cert.pools == findall(==(cert.department), p.pool_department)
        end
        @test !isempty(cert.activities)
        for (k, a) in enumerate(cert.activities)
            @test all(q -> q in pools, p.eligible_pools[a])   # only department pools
            @test p.floors[a] > 0
            @test cert.max_efficiency[k] == maximum(p.efficiency[a])
            # No single activity's own rows refute it: its floor fits under
            # its workload and its cumulative rate cap.
            len = p.deadline[a] - p.release[a] + 1
            @test p.floors[a] < min(p.workload[a], p.rate_cap[a] * len)
            # ... nor its eligible pools' cut capacities: the floor stays below
            # the most the activity could deliver with every eligible pool to
            # itself (otherwise presolve refutes that one row by propagation).
            standalone = sum(
                min(
                    p.rate_cap[a],
                    sum(
                        p.efficiency[a][k] * p.capacity[q, t] for
                        (k, q) in enumerate(p.eligible_pools[a])
                    ),
                ) for t in p.release[a]:p.deadline[a]
            )
            @test p.floors[a] < standalone
        end
        required = sum(
            p.floors[a] / cert.max_efficiency[k] for (k, a) in enumerate(cert.activities)
        )
        available = sum(p.capacity[q, t] for q in cert.pools, t in 1:p.n_periods)
        @test cert.required_hours ≈ required rtol = 1e-10
        @test cert.available_hours ≈ available rtol = 1e-10
        @test 1.12 - 1e-9 <= required / available <= 1.35 + 1e-9   # planted margin
    end

    # `unknown`: neither witness nor certificate.
    for seed in 0:5
        _, p = generate_problem(:resource_allocation, 400, unknown, seed)
        @test p.feasible_witness === nothing
        @test p.infeasibility_certificate === nothing
    end

    # Large targets realize exactly and build quickly (constructor + model).
    m, p = generate_problem(:resource_allocation, 100_000, feasible, 0)
    @test num_variables(m) == 100_000
    @test num_constraints(m; count_variable_in_set_constraints=false) > 25_000

    # Reproducibility, isolated from a dirty global RNG.
    for status in (feasible, infeasible, unknown)
        Random.seed!(1)
        _, p1 = generate_problem(:resource_allocation, 600, status, 42)
        Random.seed!(999)
        _, p2 = generate_problem(:resource_allocation, 600, status, 42)
        for f in fieldnames(typeof(p1))
            f in (:feasible_witness, :infeasibility_certificate) && continue
            @test isequal(getfield(p1, f), getfield(p2, f))
        end
        if p1.feasible_witness !== nothing
            @test p1.feasible_witness.allocation == p2.feasible_witness.allocation
        end
        if p1.infeasibility_certificate !== nothing
            @test p1.infeasibility_certificate.activities == p2.infeasibility_certificate.activities
            @test p1.infeasibility_certificate.available_hours ==
                p2.infeasibility_certificate.available_hours
        end
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            for target in (100, 800, 3000), status in (feasible, infeasible), seed in 0:2
                m, _ = generate_problem(:resource_allocation, target, status, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end

            # `unknown` lands on both sides of the boundary.
            outcomes = Set{MOI.TerminationStatusCode}()
            for seed in 0:11
                m, _ = generate_problem(:resource_allocation, 600, unknown, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                push!(outcomes, termination_status(m))
            end
            @test MOI.OPTIMAL in outcomes
            @test MOI.INFEASIBLE in outcomes

            # Presolve survival: HiGHS presolve keeps most of the model and
            # simplex has real work to do, on feasible and infeasible requests.
            for status in (feasible, infeasible)
                m, _ = generate_problem(:resource_allocation, 5000, status, 1)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                # Infeasibility needs simplex work too (not disproved by presolve).
                @test MOI.get(m, MOI.SimplexIterations()) > (status == feasible ? 500 : 0)
            end
        end
    end
end
