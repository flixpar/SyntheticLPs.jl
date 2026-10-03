# Focused quality contracts for the facility_location category: registry
# wiring, exact variable-count formulas, sparse-lane structure, planted
# witnesses checked row by row without a solver (JuMP's
# primal_feasibility_report), certificate arithmetic, and HiGHS contracts.

# Evaluate a witness point against every row of the (relaxed) model.
_fl_witness_violations(model, point) = primal_feasibility_report(model, point; atol=1e-6)

@testset "Facility Location" begin
    @test list_variants(:facility_location) == [:p_median, :standard, :two_echelon]
    @test problem_info(:facility_location)[:default_variant] == :standard

    @testset "Variable counts" begin
        for target in (50, 500, 5_000, 40_000), status in (feasible, infeasible, unknown)
            m, p = generate_problem("facility_location/standard", target, status, 3)
            @test num_variables(m) == p.n_facilities * (p.n_customers + 1)
            @test abs(num_variables(m) - target) <= max(2, 0.03 * target)
            # strong formulation: one linking row per pair
            @test num_constraints(m; count_variable_in_set_constraints=false) ==
                p.n_facilities * p.n_customers + p.n_customers + p.n_facilities + 1

            m, p = generate_problem("facility_location/p_median", target, status, 3)
            @test num_variables(m) == p.n_facilities * (p.n_customers + 1)
            @test abs(num_variables(m) - target) <= max(3, 0.03 * target)

            m, p = generate_problem("facility_location/two_echelon", target, status, 3)
            K = size(p.size_capacity, 2)
            @test num_variables(m) ==
                p.n_warehouses * (1 + K) + length(p.in_supplier) + length(p.out_customer)
            @test num_variables(m) == target
        end
        # Large targets: exact sizing and no silent cap (the old two_echelon
        # saturated at 20,600 columns).
        m, _ = generate_problem("facility_location/two_echelon", 150_000, unknown, 1)
        @test num_variables(m) == 150_000
        m, _ = generate_problem("facility_location/standard", 100_000, unknown, 1)
        @test abs(num_variables(m) - 100_000) <= 1_000
    end

    @testset "two_echelon lane structure" begin
        _, p = generate_problem("facility_location/two_echelon", 3_000, unknown, 2)
        # Delivery lanes are grouped by customer, each customer's lanes go to
        # its nearest warehouses in increasing distance, and no lane repeats.
        @test issorted(p.out_customer)
        for c in 1:p.n_customers
            lanes = findall(==(c), p.out_customer)
            ws = p.out_warehouse[lanes]
            @test allunique(ws)
            d = [hypot((p.customer_locations[c] .- p.warehouse_locations[w])...) for w in ws]
            @test issorted(d)
            all_d = sort([hypot((p.customer_locations[c] .- loc)...) for loc in p.warehouse_locations])
            @test d ≈ all_d[1:length(d)]
        end
        @test 3 <= length(p.out_customer) / p.n_customers <= 6
        @test all(p.size_capacity[:, k] <= p.size_capacity[:, k + 1] for k in 1:(size(p.size_capacity, 2) - 1))
        @test all(p.customer_demands .> 0)
    end

    @testset "Witnesses satisfy every row" begin
        for target in (60, 800, 6_000), seed in 1:3
            m, p = generate_problem("facility_location/two_echelon", target, feasible, seed)
            w = p.feasible_witness
            @test w isa SyntheticLPs.TwoEchelonWitness
            @test p.infeasibility_certificate === nothing
            K = size(p.size_capacity, 2)
            point = Dict{VariableRef, Float64}()
            for i in 1:p.n_warehouses
                point[m[:y][i]] = i in w.open ? 1.0 : 0.0
                for k in 1:K
                    point[m[:z][i, k]] = w.size_choice[i] == k ? 1.0 : 0.0
                end
            end
            for l in eachindex(w.supply_flow)
                point[m[:f1][l]] = w.supply_flow[l]
            end
            for l in eachindex(w.delivery_flow)
                point[m[:f2][l]] = w.delivery_flow[l]
            end
            @test isempty(_fl_witness_violations(m, point))

            m, p = generate_problem("facility_location/standard", target, feasible, seed)
            w = p.feasible_witness
            @test w isa SyntheticLPs.FacilityLocationWitness
            point = Dict{VariableRef, Float64}(v => 0.0 for v in all_variables(m))
            for i in w.open
                point[m[:y][i]] = 1.0
            end
            for (i, c, q) in w.shipments
                point[m[:x][i, c]] += q
            end
            @test isempty(_fl_witness_violations(m, point))

            m, p = generate_problem("facility_location/p_median", target, feasible, seed)
            w = p.feasible_witness
            @test w isa SyntheticLPs.PMedianWitness
            @test length(w.open) == p.p
            @test all(in(w.open), w.assignment)
            point = Dict{VariableRef, Float64}(v => 0.0 for v in all_variables(m))
            for i in w.open
                point[m[:z][i]] = 1.0
            end
            for (c, i) in enumerate(w.assignment)
                point[m[:y][i, c]] = 1.0
            end
            @test isempty(_fl_witness_violations(m, point))
        end
    end

    @testset "Certificate arithmetic" begin
        for target in (60, 800, 6_000), seed in 1:3
            _, p = generate_problem("facility_location/two_echelon", target, infeasible, seed)
            cert = p.infeasibility_certificate
            @test cert isa SyntheticLPs.TwoEchelonRegionalDeficit
            @test p.feasible_witness === nothing
            region = Set(cert.customers)
            reach = Set(
                p.out_warehouse[l] for l in eachindex(p.out_customer) if p.out_customer[l] in region
            )
            @test reach == Set(cert.warehouses)
            @test cert.region_demand ≈ sum(p.customer_demands[cert.customers])
            @test cert.max_capacity ≈ sum(maximum(p.size_capacity[w, :]) for w in cert.warehouses)
            @test cert.region_demand >= 1.1 * cert.max_capacity - 1e-6
            @test length(cert.customers) < p.n_customers || p.n_customers == 1

            _, p = generate_problem("facility_location/standard", target, infeasible, seed)
            cert = p.infeasibility_certificate
            @test cert isa SyntheticLPs.FacilityBudgetCertificate
            @test cert.total_demand ≈ sum(p.demands)
            @test cert.budget == p.budget
            # Independent fractional-knapsack evaluation (sorted by ratio).
            order = sortperm(p.capacities ./ p.fixed_costs; rev=true)
            left, fundable = p.budget, 0.0
            for i in order
                take = clamp(left / p.fixed_costs[i], 0.0, 1.0)
                fundable += take * p.capacities[i]
                left -= take * p.fixed_costs[i]
            end
            @test fundable ≈ cert.fundable_capacity
            @test cert.fundable_capacity <= 0.96 * cert.total_demand

            _, p = generate_problem("facility_location/p_median", target, infeasible, seed)
            cert = p.infeasibility_certificate
            @test cert isa SyntheticLPs.PMedianCapacityCertificate
            @test cert.top_p_capacity ≈ sum(sort(p.capacities; rev=true)[1:p.p])
            @test cert.total_demand ≈ sum(p.demands)
            @test cert.top_p_capacity * 1.05 <= cert.total_demand + 1e-6
        end
        for v in (:standard, :p_median, :two_echelon)
            _, p = generate_problem(ProblemVariant(:facility_location, v), 500, unknown, 1)
            @test p.feasible_witness === nothing
            @test p.infeasibility_certificate === nothing
        end
    end

    @testset "Reproducibility" begin
        for v in (:standard, :p_median, :two_echelon), status in (feasible, infeasible, unknown)
            _, a = generate_problem(ProblemVariant(:facility_location, v), 700, status, 11)
            _, b = generate_problem(ProblemVariant(:facility_location, v), 700, status, 11)
            for f in fieldnames(typeof(a))
                x, y = getfield(a, f), getfield(b, f)
                if x === nothing || isbits(x) || x isa AbstractArray
                    @test x == y
                else
                    @test all(getfield(x, g) == getfield(y, g) for g in fieldnames(typeof(x)))
                end
            end
        end
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            for v in (:standard, :p_median, :two_echelon), seed in 1:3
                ref = ProblemVariant(:facility_location, v)
                for (status, expected) in ((feasible, MOI.OPTIMAL), (infeasible, MOI.INFEASIBLE))
                    m, _ = generate_problem(ref, 1_500, status, seed)
                    set_optimizer(m, HiGHS.Optimizer)
                    set_silent(m)
                    optimize!(m)
                    @test termination_status(m) == expected
                end
            end
            # `unknown` is genuinely two-sided for two_echelon (regional demand
            # shocks against forecast-planned capacity).
            outcomes = map(1:12) do seed
                m, _ = generate_problem("facility_location/two_echelon", 1_000, unknown, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                termination_status(m)
            end
            @test MOI.OPTIMAL in outcomes
            @test MOI.INFEASIBLE in outcomes
        end
    end
end
