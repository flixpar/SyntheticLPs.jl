using Test
using JuMP
using Random
using SyntheticLPs

const UNIT_COMMITMENT_MOI = JuMP.MOI

const HAS_UNIT_COMMITMENT_HIGHS = try
    @eval using HiGHS
    true
catch
    false
end

const UNIT_COMMITMENT_REF = "unit_commitment/standard"

@testset "Unit commitment standard" begin
    @test :standard in list_variants(:unit_commitment)

    @testset "Sizing and deterministic data" begin
        for target in (50, 100, 120, 192, 500, 960, 1_200, 3_000, 3_840, 5_000)
            model, problem = generate_problem(UNIT_COMMITMENT_REF, target, feasible, 7)
            actual = num_variables(model)
            @test actual == 4 * problem.n_units * problem.n_periods
            @test abs(actual - target) / target <= 0.11
        end

        # Tiny public targets clamp to the smallest useful formulation.
        for target in (-5, 0, 2)
            tiny_model, tiny_problem = generate_problem(UNIT_COMMITMENT_REF, target, feasible, 7)
            @test num_variables(tiny_model) == 48
            @test tiny_problem.n_units == 2
            @test tiny_problem.n_periods == 6
        end

        # Large requests grow the fleet instead of saturating at the former
        # 32,256-variable cap. Constructor-only keeps this sizing check cheap.
        large_problem = SyntheticLPs.UnitCommitmentProblem(100_000, feasible, 7)
        large_actual = 4 * large_problem.n_units * large_problem.n_periods
        @test abs(large_actual - 100_000) / 100_000 <= 0.11

        _, problem1 = generate_problem(UNIT_COMMITMENT_REF, 500, feasible, 12345)
        _, problem2 = generate_problem(UNIT_COMMITMENT_REF, 500, feasible, 12345)
        for field in fieldnames(typeof(problem1))
            field in (:feasible_witness, :infeasibility_certificate) && continue
            @test isequal(getfield(problem1, field), getfield(problem2, field))
        end
        witness1 = something(problem1.feasible_witness)
        witness2 = something(problem2.feasible_witness)
        @test witness1.generation == witness2.generation
        @test witness1.commitment == witness2.commitment
        @test witness1.startup == witness2.startup
        @test witness1.shutdown == witness2.shutdown

        # Construction must not reset or consume Julia's process-global RNG.
        Random.seed!(8128)
        expected_next = rand()
        Random.seed!(8128)
        generate_problem(UNIT_COMMITMENT_REF, 120, feasible, 99)
        @test rand() == expected_next

        # The sampled fleet retains several materially different operating types.
        sampled_types = Set{Symbol}()
        for seed in 0:15
            _, problem = generate_problem(UNIT_COMMITMENT_REF, 1_200, feasible, seed)
            union!(sampled_types, values(problem.unit_types))
        end
        @test length(sampled_types) >= 5
        @test sampled_types <= Set((:nuclear, :coal, :ccgt, :gas_ct, :hydro, :wind))

        natural_model, _ = generate_problem(
            UNIT_COMMITMENT_REF, 120, feasible, 3; relax_integer=false
        )
        relaxed_model, _ = generate_problem(UNIT_COMMITMENT_REF, 120, feasible, 3)
        @test all(is_binary, natural_model[:on])
        @test all(is_binary, natural_model[:startup])
        @test all(is_binary, natural_model[:shutdown])
        @test all(!is_binary(variable) for variable in relaxed_model[:on])
        @test all(
            has_lower_bound(variable) &&
                lower_bound(variable) == 0.0 &&
                has_upper_bound(variable) &&
                upper_bound(variable) == 1.0 for variable in relaxed_model[:on]
        )
    end

    @testset "Constructive feasible contract" begin
        for target in (50, 120, 500, 1_200, 3_000), seed in 0:9
            model, problem = generate_problem(UNIT_COMMITMENT_REF, target, feasible, seed)
            @test problem.resolved_status == feasible
            @test problem.feasible_witness !== nothing
            @test problem.infeasibility_certificate === nothing
            @test SyntheticLPs._unit_commitment_witness_is_valid(problem)

            witness = something(problem.feasible_witness)
            @test all(problem.demand .> 0)
            @test all(problem.reserve_requirements .> 0)
            @test all(
                isapprox(sum(witness.generation[:, t]), problem.demand[t]; atol=1e-7, rtol=1e-10)
                for t in problem.time_periods
            )

            # The stored witness is also attached as a JuMP start, and demand is
            # represented by an equality rather than a permissive lower bound.
            first_unit = first(problem.units)
            @test start_value(model[:g][first_unit, 1]) == witness.generation[1, 1]
            demand_balance = model[:demand_balance]
            @test length(demand_balance) == problem.n_periods
            for t in problem.time_periods
                object = constraint_object(demand_balance[t])
                @test object.set isa UNIT_COMMITMENT_MOI.EqualTo{Float64}
                @test object.set.value == problem.demand[t]
                @test all(
                    normalized_coefficient(demand_balance[t], model[:g][u, t]) == 1.0 for
                    u in problem.units
                )
            end
        end
    end

    @testset "Explicit infeasibility certificate" begin
        for target in (50, 120, 500, 1_200, 3_000), seed in 0:9
            _, problem = generate_problem(UNIT_COMMITMENT_REF, target, infeasible, seed)
            @test problem.resolved_status == infeasible
            @test problem.feasible_witness === nothing
            @test problem.infeasibility_certificate !== nothing
            @test SyntheticLPs._unit_commitment_certificate_is_valid(problem)

            certificate = something(problem.infeasibility_certificate)
            t = certificate.period
            available = sum(
                problem.max_output[u] * problem.availability_factors[u][t] for u in problem.units
            )
            @test isapprox(certificate.available_capacity, available; atol=1e-9)
            @test isapprox(
                certificate.required_capacity,
                problem.demand[t] + problem.reserve_requirements[t];
                atol=1e-9,
            )
            @test isapprox(certificate.excess, certificate.required_capacity - available; atol=1e-9)
            @test certificate.excess > 0
            # The contradiction is reserve-driven rather than requiring demand
            # alone to exceed available generation.
            @test problem.demand[t] < available || iszero(available)
        end
    end

    @testset "Unknown resolves to a recorded mixed profile" begin
        statuses = Set{FeasibilityStatus}()
        for seed in 0:31
            _, problem = generate_problem(UNIT_COMMITMENT_REF, 500, unknown, seed)
            push!(statuses, problem.resolved_status)
            if problem.resolved_status == feasible
                @test SyntheticLPs._unit_commitment_witness_is_valid(problem)
            else
                @test SyntheticLPs._unit_commitment_certificate_is_valid(problem)
            end
        end
        @test statuses == Set((feasible, infeasible))
    end

    if HAS_UNIT_COMMITMENT_HIGHS
        @testset "Direct HiGHS contracts (no retry guard)" begin
            for status in (feasible, infeasible), target in (120, 500, 1_200), seed in 0:5
                model, _ = generate_problem(UNIT_COMMITMENT_REF, target, status, seed)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                expected = if status == feasible
                    UNIT_COMMITMENT_MOI.OPTIMAL
                else
                    UNIT_COMMITMENT_MOI.INFEASIBLE
                end
                @test termination_status(model) == expected
            end
            for status in (feasible, infeasible), seed in 0:2
                model, _ = generate_problem(
                    UNIT_COMMITMENT_REF, 120, status, seed; relax_integer=false
                )
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                expected = if status == feasible
                    UNIT_COMMITMENT_MOI.OPTIMAL
                else
                    UNIT_COMMITMENT_MOI.INFEASIBLE
                end
                @test termination_status(model) == expected
            end
        end
    else
        @info "HiGHS unavailable; skipping unit-commitment solve checks"
    end
end

@testset "Unit Commitment Feasibility Contracts" begin
    if HAS_HIGHS
        # unit_commitment/standard feasible-request: previously ~8% came back
        # infeasible (documented heuristic). The optimizer guard rejects those.
        for s in 1:10
            m, _ = generate_problem(
                "unit_commitment/standard", 120, feasible, s; optimizer=HiGHS.Optimizer
            )
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            @test termination_status(m) == MOI.OPTIMAL
        end
    end
end
