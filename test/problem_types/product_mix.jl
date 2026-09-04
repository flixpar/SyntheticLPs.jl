# Focused quality contracts for the product_mix category: registry shape,
# sizing, planted-plan witness and over-commitment certificate arithmetic
# checked directly against the struct fields, the analytic feasibility
# characterisation that keeps the `unknown` profile genuinely mixed at every
# scale, reproducibility, and HiGHS feasibility contracts.
@testset "Product Mix" begin
    @test :product_mix in list_categories()
    @test list_variants(:product_mix) == [:standard]
    info = problem_info(:product_mix)
    @test info[:default_variant] == :standard
    @test occursin("product", lowercase(info[:description]))

    # Sizing: variables are exactly the products, i.e. the clamped target.
    for target in (50, 200, 1000, 5000), status in (feasible, infeasible, unknown), seed in 0:2
        m, p = generate_problem(:product_mix, target, status, seed)
        @test num_variables(m) == p.num_products
        @test p.num_products == max(2, min(10000, target))
        @test abs(num_variables(m) - target) <= 0.25 * target || num_variables(m) <= 50
        @test num_constraints(m; count_variable_in_set_constraints=false) ==
            p.num_resources + count(>(0.0), p.lower_bounds) + count(isfinite, p.upper_bounds)
    end

    # Structural data contracts shared by all three profiles.
    for target in (60, 400, 1500), status in (feasible, infeasible, unknown)
        _, p = generate_problem(:product_mix, target, status, 7)
        @test size(p.usage_matrix) == (p.num_resources, p.num_products)
        @test all(>=(0.0), p.usage_matrix)
        @test all(any(p.usage_matrix[:, j] .> 0) for j in 1:p.num_products)
        @test all(any(p.usage_matrix[i, :] .> 0) for i in 1:p.num_resources)
        @test all(>(0.0), p.profits)
        @test all(>(0.0), p.nominal_plan)
        @test all(>(0.0), p.availabilities)
        @test all(p.lower_bounds .>= 0.0)
        # Never a trivial bound clash: every ceiling stays above its floor.
        @test all(p.lower_bounds[j] <= p.upper_bounds[j] for j in 1:p.num_products)
        @test p.industry in
            (:manufacturing, :food_processing, :electronics, :furniture, :chemical, :automotive)
        @test p.feasibility_status == status
        # The stored utilization scalar recomputes exactly from the data.
        required = [
            sum(p.usage_matrix[i, j] * p.lower_bounds[j] for j in 1:p.num_products) for
            i in 1:p.num_resources
        ]
        @test p.floor_utilization ≈
            maximum(required[i] / p.availabilities[i] for i in 1:p.num_resources)
    end

    # Planted-plan witness: the nominal plan is an actual feasible point of the
    # built model. Checked by arithmetic on the struct fields *and* against the
    # model itself via JuMP's primal feasibility report.
    for target in (50, 300, 2000), seed in 0:2
        m, p = generate_problem(:product_mix, target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing
        @test p.infeasibility_certificate === nothing
        @test w.plan == p.nominal_plan
        @test w.consumption ≈ p.usage_matrix * w.plan
        @test w.slack ≈ p.availabilities .- w.consumption
        # Capacities strictly cover the plan's consumption ...
        @test all(w.slack .> 0.0)
        @test all(p.availabilities[i] > w.consumption[i] for i in 1:p.num_resources)
        # ... floors sit at or below the plan's output ...
        @test all(p.lower_bounds .<= w.plan)
        # ... and market ceilings at or above it.
        @test all(p.upper_bounds .>= w.plan)
        # Hence the floors can never over-commit a resource.
        @test p.floor_utilization < 1.0

        atol = 1e-6 * maximum(p.availabilities)
        report = primal_feasibility_report(
            m, Dict(m[:x][j] => w.plan[j] for j in 1:p.num_products); atol=atol
        )
        @test isempty(report)
    end

    # Over-commitment certificate: a set of products whose floors provably
    # exhaust one resource's capacity. Recomputed from the raw fields.
    for target in (50, 300, 2000), seed in 0:3
        _, p = generate_problem(:product_mix, target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing
        @test p.feasible_witness === nothing
        @test 1 <= cert.resource <= p.num_resources
        @test !isempty(cert.products)
        @test allunique(cert.products)
        @test all(p.lower_bounds[j] > 0.0 for j in cert.products)
        @test all(p.usage_matrix[cert.resource, j] > 0.0 for j in cert.products)
        # Every product left out contributes nothing to that resource row.
        listed = Set(cert.products)
        @test all(
            p.usage_matrix[cert.resource, j] * p.lower_bounds[j] == 0.0 for
            j in 1:p.num_products if !(j in listed)
        )
        recomputed = sum(
            p.usage_matrix[cert.resource, j] * p.lower_bounds[j] for j in cert.products
        )
        @test cert.required_usage ≈ recomputed
        @test cert.availability == p.availabilities[cert.resource]
        @test cert.required_usage > cert.availability
        # The refutation is a genuine resource over-commitment, not a bound
        # clash: every floor still leaves room under its own ceiling.
        @test all(
            p.lower_bounds[j] < p.upper_bounds[j] for
            j in 1:p.num_products if p.lower_bounds[j] > 0.0
        )
        @test p.floor_utilization > 1.0
        @test 1.15 <= p.floor_utilization <= 1.60 + 1e-9
    end

    # The `unknown` profile must stay genuinely mixed at every scale. Because
    # all usage coefficients are nonnegative, `x = lower_bounds` is the
    # pointwise-smallest candidate, so the instance is feasible exactly when
    # `floor_utilization <= 1` -- which makes the mix measurable without a
    # solver (the solver-backed cross-check lives below).
    for target in (50, 100, 500, 1000, 5000)
        feas = count(0:39) do seed
            _, p = generate_problem(:product_mix, target, unknown, seed)
            p.floor_utilization <= 1.0
        end
        @test 10 <= feas <= 30      # neither outcome dominates
    end

    # Reproducibility, including isolation from a seeded/dirty global RNG.
    for status in (feasible, infeasible, unknown)
        Random.seed!(987)
        _, p1 = generate_problem(:product_mix, 220, status, 42)
        Random.seed!(12345)
        _, p2 = generate_problem(:product_mix, 220, status, 42)
        for f in (
            :num_products,
            :num_resources,
            :profits,
            :usage_matrix,
            :availabilities,
            :lower_bounds,
            :upper_bounds,
            :nominal_plan,
            :floor_utilization,
            :industry,
        )
            @test isequal(getfield(p1, f), getfield(p2, f))
        end
        if p1.feasible_witness !== nothing
            @test p1.feasible_witness.plan == p2.feasible_witness.plan
            @test p1.feasible_witness.slack == p2.feasible_witness.slack
        end
        if p1.infeasibility_certificate !== nothing
            c1, c2 = p1.infeasibility_certificate, p2.infeasibility_certificate
            @test c1.resource == c2.resource
            @test c1.products == c2.products
            @test c1.required_usage == c2.required_usage
        end
    end

    if HAS_HIGHS
        # End-to-end feasibility contract across scales and seeds.
        for target in (50, 200, 1000, 5000), status in (feasible, infeasible), seed in 0:3
            m, _ = generate_problem(:product_mix, target, status, seed)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
            @test termination_status(m) == expected
        end

        # The analytic characterisation used above agrees with the solver on
        # the `unknown` profile, and both outcomes really occur at scale.
        for target in (500, 5000)
            outcomes = MOI.TerminationStatusCode[]
            for seed in 0:9
                m, p = generate_problem(:product_mix, target, unknown, seed)
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
