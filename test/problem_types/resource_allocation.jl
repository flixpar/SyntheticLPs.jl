# Focused quality contracts for the resource_allocation category: registry
# shape, exact sizing against the documented cap, sparsity and data
# invariants, planted-plan witness and floor-overcommit certificate arithmetic
# recomputed directly from the struct fields, the analytic utilization
# characterization that keeps the `unknown` profile genuinely mixed at every
# scale, reproducibility, and HiGHS feasibility contracts.
@testset "Resource Allocation" begin
    @test :resource_allocation in list_categories()
    @test list_variants(:resource_allocation) == [:standard]
    info = problem_info(:resource_allocation)
    @test info[:default_variant] == :standard
    @test occursin("resource", lowercase(info[:description]))

    # Sizing: variables are exactly the activities, i.e. the clamped target,
    # and the old silent 2000-activity cap is gone (5000 realizes in full).
    tier_bounds = Dict(50 => (4, 12), 200 => (4, 12), 1000 => (10, 36), 5000 => (20, 64))
    for target in (50, 200, 1000, 5000), status in (feasible, infeasible, unknown), seed in 0:2
        m, p = generate_problem(:resource_allocation, target, status, seed)
        @test num_variables(m) == p.n_activities
        @test p.n_activities == max(3, min(100_000, target))
        @test abs(num_variables(m) - target) <= 0.25 * target || num_variables(m) <= 50
        @test num_constraints(m; count_variable_in_set_constraints=false) ==
            p.n_resources + count(>(0.0), p.min_levels)
        # Pool counts follow the documented tiers, so resources scale with the
        # portfolio instead of being a size-independent 2:50 draw.
        lo, hi = tier_bounds[target]
        @test lo <= p.n_resources <= hi
    end

    # The documented sizing limit raises instead of silently undersizing,
    # and targets far above the old cap realize exactly.
    @test_throws ArgumentError generate_problem(:resource_allocation, 100_001, feasible, 0)
    m, p = generate_problem(:resource_allocation, 20_000, unknown, 0)
    @test num_variables(m) == 20_000
    @test 32 <= p.n_resources <= 96

    # Structural data contracts shared by all three profiles.
    for target in (60, 400, 1500), status in (feasible, infeasible, unknown)
        _, p = generate_problem(:resource_allocation, target, status, 7)
        @test size(p.usage) == (p.n_activities, p.n_resources)
        @test all(>=(0.0), p.usage)
        # Sparsity: every activity draws on at least one pool (this keeps the
        # profit-max LP bounded), every pool is drawn on by at least one
        # activity (no vacuous capacity rows), and the matrix really is sparse.
        @test all(any(p.usage[i, :] .> 0) for i in 1:p.n_activities)
        @test all(any(p.usage[:, j] .> 0) for j in 1:p.n_resources)
        @test count(>(0.0), p.usage) < 0.75 * p.n_activities * p.n_resources
        @test all(>(0.0), p.profits)
        @test all(>(0.0), p.nominal_plan)
        @test all(>(0.0), p.capacities)
        @test all(>=(0.0), p.min_levels)
        @test p.profile in
            (:manufacturing_capacity, :cloud_compute, :workforce_hours, :advertising_budget)
        @test p.feasibility_status == status
        # The stored utilization scalar recomputes exactly from the data.
        required = [
            sum(p.usage[i, j] * p.min_levels[i] for i in 1:p.n_activities) for j in 1:p.n_resources
        ]
        @test p.floor_utilization ≈ maximum(required[j] / p.capacities[j] for j in 1:p.n_resources)
    end

    # Planted-plan witness: the nominal plan is an actual feasible point of
    # the built model. Checked by arithmetic on the struct fields *and*
    # against the model itself via JuMP's primal feasibility report.
    for target in (50, 300, 2000), seed in 0:2
        m, p = generate_problem(:resource_allocation, target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing
        @test p.infeasibility_certificate === nothing
        @test w.plan == p.nominal_plan
        @test w.consumption ≈
            [sum(p.usage[i, j] * w.plan[i] for i in 1:p.n_activities) for j in 1:p.n_resources]
        @test w.slack ≈ p.capacities .- w.consumption
        # Capacities strictly cover the plan's consumption ...
        @test all(w.slack .> 0.0)
        # ... and commitment floors sit at or below the plan's levels.
        @test all(p.min_levels .<= w.plan)
        # Hence the floors can never over-commit a pool.
        @test p.floor_utilization < 1.0

        atol = 1e-6 * maximum(p.capacities)
        report = primal_feasibility_report(
            m, Dict(m[:x][i] => w.plan[i] for i in 1:p.n_activities); atol=atol
        )
        @test isempty(report)
    end

    # Floor-overcommit certificate: the committed activities' mandatory
    # minimums provably exhaust one pool's capacity. Recomputed from the raw
    # fields, with the refutation relying only on LP rows.
    for target in (50, 300, 2000), seed in 0:3
        _, p = generate_problem(:resource_allocation, target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing
        @test p.feasible_witness === nothing
        @test 1 <= cert.resource <= p.n_resources
        @test !isempty(cert.activities)
        @test allunique(cert.activities)
        @test all(p.min_levels[i] > 0.0 for i in cert.activities)
        @test all(p.usage[i, cert.resource] > 0.0 for i in cert.activities)
        # Every activity left out contributes nothing to that capacity row.
        listed = Set(cert.activities)
        @test all(
            p.usage[i, cert.resource] * p.min_levels[i] == 0.0 for
            i in 1:p.n_activities if !(i in listed)
        )
        recomputed = sum(p.usage[i, cert.resource] * p.min_levels[i] for i in cert.activities)
        @test cert.floor_consumption ≈ recomputed
        @test cert.capacity == p.capacities[cert.resource]
        @test cert.floor_consumption > cert.capacity
        # The violation is a deliberate margin above 1, not a rounding hair.
        @test p.floor_utilization > 1.0
        @test 1.1 - 1e-9 <= p.floor_utilization <= 1.4 + 1e-9
    end

    # The `unknown` profile must stay genuinely mixed at every scale. Because
    # usage is nonnegative and there are no activity ceilings, `x = min_levels`
    # is the pointwise-smallest candidate, so the instance is feasible exactly
    # when `floor_utilization <= 1` -- which makes the mix measurable without
    # a solver (the solver-backed cross-check lives below).
    for target in (50, 100, 500, 1000, 5000)
        feas = count(0:39) do seed
            _, p = generate_problem(:resource_allocation, target, unknown, seed)
            p.floor_utilization <= 1.0
        end
        @test 10 <= feas <= 30      # neither outcome dominates
    end

    # Reproducibility, including isolation from a seeded/dirty global RNG.
    # Witness and certificate are compared subfield by subfield because
    # `isequal` on structs holding heap vectors falls back to object identity.
    for status in (feasible, infeasible, unknown)
        Random.seed!(987)
        _, p1 = generate_problem(:resource_allocation, 220, status, 42)
        Random.seed!(12345)
        _, p2 = generate_problem(:resource_allocation, 220, status, 42)
        plain = filter(!in((:feasible_witness, :infeasibility_certificate)), fieldnames(typeof(p1)))
        @test all(isequal(getfield(p1, f), getfield(p2, f)) for f in plain)
        if p1.feasible_witness !== nothing
            @test p2.feasible_witness !== nothing
            @test p1.feasible_witness.plan == p2.feasible_witness.plan
            @test p1.feasible_witness.consumption == p2.feasible_witness.consumption
            @test p1.feasible_witness.slack == p2.feasible_witness.slack
        else
            @test p2.feasible_witness === nothing
        end
        if p1.infeasibility_certificate !== nothing
            c1, c2 = p1.infeasibility_certificate, p2.infeasibility_certificate
            @test c2 !== nothing
            @test c1.resource == c2.resource
            @test c1.activities == c2.activities
            @test c1.floor_consumption == c2.floor_consumption
            @test c1.capacity == c2.capacity
        else
            @test p2.infeasibility_certificate === nothing
        end
    end

    if HAS_HIGHS
        # End-to-end feasibility contract across scales and seeds.
        for target in (50, 200, 1000, 5000), status in (feasible, infeasible), seed in 0:4
            m, _ = generate_problem(:resource_allocation, target, status, seed)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end

        # The analytic characterization used above agrees with the solver on
        # the `unknown` profile, and both outcomes really occur at scale.
        for target in (500, 5000)
            outcomes = MOI.TerminationStatusCode[]
            for seed in 0:9
                m, p = generate_problem(:resource_allocation, target, unknown, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                ts = termination_status(m)
                push!(outcomes, ts)
                @test ts == (p.floor_utilization <= 1.0 ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end
            @test MOI.OPTIMAL in outcomes
            @test MOI.INFEASIBLE in outcomes
        end
    end
end
