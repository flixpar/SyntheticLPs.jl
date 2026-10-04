# Focused quality contracts for the product_mix category: registry shape,
# exact routing-column sizing and the row formula (machines, labor pools,
# materials, ranged market rows), shop/routing data invariants, the planted
# plan witness checked against the built model, the department
# over-commitment certificate recomputed from the struct fields,
# reproducibility under a dirty global RNG, and HiGHS contracts including a
# presolve-survival regression (the previous formulation presolved to an empty
# model).
@testset "Product Mix" begin
    @test :product_mix in list_categories()
    @test list_variants(:product_mix) == [:standard]
    info = problem_info(:product_mix)
    @test info[:default_variant] == :standard
    @test occursin("product", lowercase(info[:description]))

    pm_routings(p) = [findall(==(q), p.routing_product) for q in 1:p.n_products]
    function pm_rows(p)
        machines = Set(m for ms in p.routing_machines for m in ms)
        depts = Set(p.machine_department[m] for m in machines)
        materials = Set(k for q in 1:p.n_products for k in p.product_materials[q])
        multi = count(rs -> length(rs) > 1, pm_routings(p))
        return length(machines) + length(depts) + length(materials) + multi
    end

    # Sizing: one column per routing, exactly the target; rows as documented.
    for target in (2, 50, 300, 2000, 8000), status in (feasible, infeasible, unknown), seed in 0:1
        m, p = generate_problem(:product_mix, target, status, seed)
        @test num_variables(m) == p.n_routings == max(target, 2)
        @test num_constraints(m; count_variable_in_set_constraints=false) == pm_rows(p)
    end
    m, p = generate_problem(:product_mix, 100_000, unknown, 0)
    @test num_variables(m) == 100_000
    @test num_constraints(m; count_variable_in_set_constraints=false) >= 30_000
    # Ranged market rows exist (the corpus otherwise has almost none).
    @test num_constraints(m, AffExpr, MOI.Interval{Float64}) > 0

    # Shop and routing invariants.
    for target in (80, 900, 4000), status in (feasible, infeasible, unknown)
        _, p = generate_problem(:product_mix, target, status, 3)
        @test p.industry in SyntheticLPs._PRODUCT_MIX_INDUSTRIES
        rs = pm_routings(p)
        @test all(1 <= length(r) <= 3 for r in rs)
        for r in 1:p.n_routings
            ms = p.routing_machines[r]
            @test allunique(ms)
            @test length(p.routing_times[r]) == length(p.routing_labor[r]) == length(ms)
            @test all(>(0.0), p.routing_times[r])
            @test all(>(0.0), p.routing_labor[r])
            # Every routing starts in its product's primary department.
            @test p.machine_department[ms[1]] == p.primary_department[p.routing_product[r]]
            @test 1.0 <= p.routing_yield[r] <= 1.18
        end
        @test all(p.floor .>= 0)
        @test all(p.floor .< p.ceiling)        # never a bound clash
        @test all(>(0.0), p.machine_capacity)
        @test all(>(0.0), p.labor_capacity)
        @test all(>(0.0), p.material_capacity)
        # Every routing is profitable (no dual-fixable dead columns).
        for r in 1:p.n_routings
            q = p.routing_product[r]
            mat = sum(a * p.material_cost[k] for (k, a) in zip(p.product_materials[q], p.material_qty[q]))
            @test p.price[q] - mat * p.routing_yield[r] - p.routing_cost[r] > 0
        end
    end

    # Planted plan witness.
    for target in (50, 600, 4000), seed in 0:2
        m, p = generate_problem(:product_mix, target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing
        @test p.infeasibility_certificate === nothing
        machine, labor, material = SyntheticLPs._product_mix_usage(
            p.n_machines,
            p.n_departments,
            p.n_materials,
            p.routing_product,
            p.routing_machines,
            p.routing_times,
            p.routing_labor,
            p.routing_yield,
            p.machine_department,
            p.product_materials,
            p.material_qty,
            w.production,
        )
        @test machine ≈ w.machine_hours
        @test labor ≈ w.labor_hours
        @test material ≈ w.material_use
        @test all(w.machine_hours .< p.machine_capacity)
        @test all(w.labor_hours .< p.labor_capacity)
        @test all(w.material_use .< p.material_capacity)
        sales = [sum(w.production[r] for r in rr) for rr in pm_routings(p)]
        @test all(p.floor .<= sales .< p.ceiling)
        report = primal_feasibility_report(
            m, Dict(m[:x][r] => w.production[r] for r in 1:p.n_routings); atol=1e-6
        )
        @test isempty(report)
    end

    # Department over-commitment certificate, recomputed from the data.
    for target in (50, 600, 4000), seed in 0:3
        _, p = generate_problem(:product_mix, target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing
        @test p.feasible_witness === nothing
        @test cert.machines == findall(==(cert.department), p.machine_department)
        rs = pm_routings(p)
        for (k, q) in enumerate(cert.products)
            @test p.floor[q] > 0
            hours = [
                sum(
                    (t for (m, t) in zip(p.routing_machines[r], p.routing_times[r]) if
                     p.machine_department[m] == cert.department);
                    init=0.0,
                ) for r in rs[q]
            ]
            @test cert.min_hours[k] ≈ minimum(hours)
            @test cert.min_hours[k] > 0              # no routing avoids the department
        end
        required = sum(p.floor[q] * h for (q, h) in zip(cert.products, cert.min_hours))
        @test cert.required_hours ≈ required
        @test cert.available_hours ≈ sum(p.machine_capacity[m] for m in cert.machines)
        @test cert.required_hours > 1.05 * cert.available_hours
    end

    for seed in 0:4
        _, p = generate_problem(:product_mix, 500, unknown, seed)
        @test p.feasible_witness === nothing
        @test p.infeasibility_certificate === nothing
    end

    # Reproducibility, isolated from a dirty global RNG.
    for status in (feasible, infeasible, unknown)
        Random.seed!(987)
        _, p1 = generate_problem(:product_mix, 700, status, 42)
        Random.seed!(12345)
        _, p2 = generate_problem(:product_mix, 700, status, 42)
        for f in fieldnames(typeof(p1))
            f in (:feasible_witness, :infeasibility_certificate) && continue
            @test isequal(getfield(p1, f), getfield(p2, f))
        end
        if p1.feasible_witness !== nothing
            @test p1.feasible_witness.production == p2.feasible_witness.production
        end
        if p1.infeasibility_certificate !== nothing
            @test p1.infeasibility_certificate.products == p2.infeasibility_certificate.products
        end
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            for target in (50, 400, 3000), status in (feasible, infeasible), seed in 0:2
                m, _ = generate_problem(:product_mix, target, status, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end
            outcomes = Set{MOI.TerminationStatusCode}()
            for seed in 0:9
                m, _ = generate_problem(:product_mix, 600, unknown, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                push!(outcomes, termination_status(m))
            end
            @test outcomes == Set([MOI.OPTIMAL, MOI.INFEASIBLE])
            # Presolve survival: simplex has real work left on a feasible 5k
            # instance (the old formulation presolved to empty).
            m, _ = generate_problem(:product_mix, 5000, feasible, 1)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            @test MOI.get(m, MOI.SimplexIterations()) > 1000
        end
    end
end
