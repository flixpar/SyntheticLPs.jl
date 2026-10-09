# Focused quality contracts for the vehicle_routing category: the CVRP sizing
# formula, degree/flow/coupling row structure, the FFD route witness checked
# row by row without a solver, the fleet-capacity certificate arithmetic,
# reproducibility, and HiGHS contracts.
@testset "Vehicle Routing (CVRP)" begin
    @test list_variants(:vehicle_routing) == [:cvrp]
    @test problem_info(:vehicle_routing)[:default_variant] == :cvrp

    @testset "Sizing and row structure" begin
        for target in (50, 500, 5_000, 100_000)
            m, p = generate_problem("vehicle_routing/cvrp", target, unknown, 1)
            N = p.n_customers
            @test N == max(3, round(Int, (sqrt(1 + 2 * target) - 1) / 2))
            @test num_variables(m) == 2 * (N + 1) * N
            target >= 500 && @test abs(num_variables(m) - target) <= 0.05 * target
            # 2N customer degree + 2 depot degree + (N+1) load balance +
            # (N+1)N upper coupling + N·N lower coupling rows.
            @test num_constraints(m; count_variable_in_set_constraints=false) ==
                2N + 2 + (N + 1) + (N + 1) * N + N * N
        end
        _, p = generate_problem("vehicle_routing/cvrp", 2_000, unknown, 4)
        @test all(p.demands .> 0)
        @test 1 <= p.n_vehicles <= p.n_customers
        @test p.vehicle_capacity >= maximum(p.demands)
        @test all(iszero, p.dist[i, i] for i in axes(p.dist, 1))
        @test all(p.dist[i, j] > 0 for i in axes(p.dist, 1), j in axes(p.dist, 2) if i != j)
    end

    @testset "Route witness satisfies every row" begin
        for target in (60, 700, 4_000), seed in 1:4
            m, p = generate_problem("vehicle_routing/cvrp", target, feasible, seed)
            w = p.feasible_witness
            @test w isa SyntheticLPs.CVRPWitness
            @test p.infeasibility_certificate === nothing
            @test length(w.routes) == p.n_vehicles
            @test all(!isempty, w.routes)
            @test sort(reduce(vcat, w.routes)) == 1:p.n_customers
            @test all(sum(p.demands[r]) <= p.vehicle_capacity for r in w.routes)

            point = Dict{VariableRef, Float64}(v => 0.0 for v in all_variables(m))
            for r in w.routes
                nodes = [1; r .+ 1; 1]
                load = sum(p.demands[r])
                for (a, b) in zip(nodes[1:(end - 1)], nodes[2:end])
                    point[m[:x][a, b]] = 1.0
                    point[m[:f][a, b]] = load
                    b == 1 || (load -= p.demands[b - 1])
                end
            end
            @test isempty(primal_feasibility_report(m, point; atol=1e-6))
        end
    end

    @testset "Fleet-capacity certificate" begin
        for target in (60, 700, 4_000), seed in 1:4
            _, p = generate_problem("vehicle_routing/cvrp", target, infeasible, seed)
            cert = p.infeasibility_certificate
            @test cert isa SyntheticLPs.CVRPFleetCapacityCertificate
            @test p.feasible_witness === nothing
            @test cert.total_demand ≈ sum(p.demands)
            @test cert.fleet_capacity ≈ p.n_vehicles * p.vehicle_capacity
            @test cert.total_demand >= 1.1 * cert.fleet_capacity - 1e-6
        end
        # Above tiny sizes the shortfall comes from a reduced fleet, not from
        # demands that exceed a single vehicle.
        _, p = generate_problem("vehicle_routing/cvrp", 4_000, infeasible, 1)
        @test maximum(p.demands) <= p.vehicle_capacity
        _, p = generate_problem("vehicle_routing/cvrp", 500, unknown, 1)
        @test p.feasible_witness === nothing
        @test p.infeasibility_certificate === nothing
    end

    @testset "Reproducibility" begin
        for status in (feasible, infeasible, unknown)
            _, a = generate_problem("vehicle_routing/cvrp", 900, status, 7)
            _, b = generate_problem("vehicle_routing/cvrp", 900, status, 7)
            @test a.demands == b.demands
            @test a.dist == b.dist
            @test a.vehicle_capacity == b.vehicle_capacity
            @test a.n_vehicles == b.n_vehicles
            if status == feasible
                @test a.feasible_witness.routes == b.feasible_witness.routes
            end
        end
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            for seed in 1:4
                for (status, expected) in ((feasible, MOI.OPTIMAL), (infeasible, MOI.INFEASIBLE))
                    m, _ = generate_problem("vehicle_routing/cvrp", 1_200, status, seed)
                    set_optimizer(m, HiGHS.Optimizer)
                    set_silent(m)
                    optimize!(m)
                    @test termination_status(m) == expected
                end
            end
            # The planted FFD routing is integral: the unrelaxed MIP is feasible.
            m, _ = generate_problem("vehicle_routing/cvrp", 60, feasible, 2; relax_integer=false)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            set_time_limit_sec(m, 30.0)
            optimize!(m)
            @test primal_status(m) == MOI.FEASIBLE_POINT
        end
    end
end
