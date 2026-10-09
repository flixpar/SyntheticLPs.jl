# Focused quality contracts for the production_planning category: registry
# shape, exact column/row formulas of the multi-level MRP LP and size fidelity
# up to 100k, bill-of-materials and routing invariants, the planted lot-for-lot
# MRP witness (balance identities recomputed by hand and checked against the
# built model), the echelon-load certificate recomputed independently from the
# struct fields, reproducibility under a dirty global RNG, and HiGHS contracts.
@testset "Production Planning" begin
    @test :production_planning in list_categories()
    @test list_variants(:production_planning) == [:standard]

    pp_cols(p) =
        sum(p.n_periods - L for L in p.lead_time) +
        p.n_items * p.n_periods +
        p.n_end_items * (p.n_periods - 1) +
        p.n_work_centers * p.n_periods
    function pp_rows(p)
        T = p.n_periods
        sup_rows = 0
        for s in 1:p.n_suppliers, t in 1:T
            sup_rows += any(p.supplier[i] == s && t <= T - p.lead_time[i] for i in 1:p.n_items)
        end
        return p.n_items * T + p.n_work_centers * T + sup_rows
    end

    # Sizing: exact formulas, and the column count tracks the target.
    for target in (50, 100, 500, 2000, 8000), status in (feasible, infeasible, unknown), seed in 0:1
        m, p = generate_problem(:production_planning, target, status, seed)
        @test num_variables(m) == pp_cols(p)
        @test num_constraints(m; count_variable_in_set_constraints=false) == pp_rows(p)
        @test abs(num_variables(m) - target) <= 0.1 * target
    end
    m, p = generate_problem(:production_planning, 100_000, unknown, 0)
    @test abs(num_variables(m) - 100_000) <= 5_000
    @test num_constraints(m; count_variable_in_set_constraints=false) >= 40_000

    # Product structure invariants.
    for target in (300, 3000), status in (feasible, infeasible), seed in 0:2
        _, p = generate_problem(:production_planning, target, status, seed)
        N = p.n_items
        @test issorted(p.level)
        @test p.n_end_items == count(==(1), p.level)
        @test p.purchased == (p.level .== maximum(p.level))
        @test all(p.bom_parent .< p.bom_child)                 # parents precede children
        @test all(p.level[p.bom_parent] .< p.level[p.bom_child])
        @test allunique(zip(p.bom_parent, p.bom_child))
        @test all(>(0.0), p.bom_qty)
        has_parent = falses(N)
        has_parent[p.bom_child] .= true
        @test all(has_parent[i] for i in 1:N if p.level[i] > 1)  # no orphan parts
        @test all(!p.purchased[i] for i in p.bom_parent)        # purchased items are leaves
        @test all((p.work_center[i] > 0) == !p.purchased[i] for i in 1:N)
        @test all((p.supplier[i] > 0) == p.purchased[i] for i in 1:N)
        @test sort(unique(p.work_center[.!p.purchased])) == collect(1:p.n_work_centers)
        @test sort(unique(p.supplier[p.purchased])) == collect(1:p.n_suppliers)
        @test all(0 .<= p.lead_time .<= 3)
        @test all(p.demand .>= 0)
        @test sum(p.demand[1:p.n_end_items, :]) > 0
        @test all(>(0.0), p.regular_capacity)
        @test all(p.max_overtime .>= 0)
        @test all(>(0.0), p.supplier_capacity)
        @test all(p.holding_cost .> 0)
        @test all(p.backlog_cost[e] > 0 for e in 1:p.n_end_items)
    end

    # Planted MRP plan: balance identities by hand, capacity and supplier
    # limits, and a feasible point of the built model.
    for target in (100, 1000, 5000), seed in 0:2
        m, p = generate_problem(:production_planning, target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing
        @test p.infeasibility_certificate === nothing
        N, T = p.n_items, p.n_periods
        @test all(w.production .>= 0)
        @test all(w.inventory .>= -1e-9)
        @test all(w.production[i, t] == 0 for i in 1:N for t in (T - p.lead_time[i] + 1):T)
        for i in 1:N, t in 1:T
            consumption = sum(
                a * w.production[pp, t] for
                (pp, c, a) in zip(p.bom_parent, p.bom_child, p.bom_qty) if c == i;
                init=0.0,
            )
            arrival = t > p.lead_time[i] ? w.production[i, t - p.lead_time[i]] : 0.0
            prev = t == 1 ? p.initial_inventory[i] : w.inventory[i, t - 1]
            @test prev + arrival - consumption - p.demand[i, t] ≈ w.inventory[i, t] atol =
                1e-6 * (1 + prev)
        end
        for wc in 1:p.n_work_centers, t in 1:T
            load = sum(
                p.run_time[i] * w.production[i, t] for i in 1:N if p.work_center[i] == wc; init=0.0
            )
            @test 0 <= w.overtime[wc, t] <= p.max_overtime[wc, t]
            @test load - w.overtime[wc, t] <= p.regular_capacity[wc, t] + 1e-6
        end
        point = Dict{VariableRef, Float64}()
        for i in 1:N, t in 1:(T - p.lead_time[i])
            point[m[:x][i, t]] = w.production[i, t]
        end
        for i in 1:N, t in 1:T
            point[m[:I][i, t]] = w.inventory[i, t]
        end
        for e in 1:p.n_end_items, t in 1:(T - 1)
            point[m[:B][e, t]] = 0.0
        end
        for wc in 1:p.n_work_centers, t in 1:T
            point[m[:O][wc, t]] = w.overtime[wc, t]
        end
        @test isempty(primal_feasibility_report(m, point; atol=1e-6))
    end

    # Echelon-load certificate, recomputed independently.
    for target in (100, 1000, 5000), seed in 0:3
        _, p = generate_problem(:production_planning, target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing
        @test p.feasible_witness === nothing
        lb = zeros(p.n_items)
        for i in 1:p.n_items
            need = sum(p.demand[i, :]) - p.initial_inventory[i]
            for (pp, c, a) in zip(p.bom_parent, p.bom_child, p.bom_qty)
                c == i && (need += a * lb[pp])
            end
            lb[i] = max(0.0, need)
        end
        @test cert.lower_bounds ≈ lb
        @test cert.items == [i for i in 1:p.n_items if p.work_center[i] == cert.work_center]
        @test cert.required_load ≈ sum(p.run_time[i] * lb[i] for i in cert.items)
        @test cert.available_capacity ≈
            sum(p.regular_capacity[cert.work_center, :]) +
              sum(p.max_overtime[cert.work_center, :])
        @test cert.required_load >= 1.08 * cert.available_capacity * (1 - 1e-9)
    end

    for seed in 0:4
        _, p = generate_problem(:production_planning, 600, unknown, seed)
        @test p.feasible_witness === nothing
        @test p.infeasibility_certificate === nothing
    end

    # Reproducibility, isolated from a dirty global RNG.
    for status in (feasible, infeasible, unknown)
        Random.seed!(5)
        _, p1 = generate_problem(:production_planning, 900, status, 13)
        Random.seed!(500)
        _, p2 = generate_problem(:production_planning, 900, status, 13)
        for f in fieldnames(typeof(p1))
            f in (:feasible_witness, :infeasibility_certificate) && continue
            @test isequal(getfield(p1, f), getfield(p2, f))
        end
        if p1.feasible_witness !== nothing
            @test p1.feasible_witness.production == p2.feasible_witness.production
        end
        if p1.infeasibility_certificate !== nothing
            @test p1.infeasibility_certificate.lower_bounds ==
                p2.infeasibility_certificate.lower_bounds
        end
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            for target in (200, 1500), status in (feasible, infeasible), seed in 0:2
                m, _ = generate_problem(:production_planning, target, status, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
                @test MOI.get(m, MOI.SimplexIterations()) > 0   # never presolve-only
            end
            outcomes = Set{MOI.TerminationStatusCode}()
            for seed in 0:9
                m, _ = generate_problem(:production_planning, 500, unknown, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                push!(outcomes, termination_status(m))
            end
            @test outcomes == Set([MOI.OPTIMAL, MOI.INFEASIBLE])
        end
    end
end
