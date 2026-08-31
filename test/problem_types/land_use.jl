using Test
using Random
using LinearAlgebra
using JuMP
using SyntheticLPs

const LAND_USE_MOI = JuMP.MOI
const HAS_LAND_USE_HIGHS = try
    @eval using HiGHS
    true
catch
    false
end
const LAND_USE_REF = "land_use/standard"

function assert_connected_spatial_graph(problem)
    @test size(problem.parcel_coordinates) == (problem.n_parcels, 2)
    @test all(coordinate -> 0.0 < coordinate < 1.0,
              problem.parcel_coordinates)
    @test size(problem.adjacency_matrix) ==
          (problem.n_parcels, problem.n_parcels)
    @test issymmetric(problem.adjacency_matrix)
    @test all(!problem.adjacency_matrix[i, i] for i in 1:problem.n_parcels)
    @test length(problem.adjacency_edges) ==
          length(unique(problem.adjacency_edges))
    @test issorted(problem.adjacency_edges)
    @test all(first(edge) < last(edge) for edge in problem.adjacency_edges)
    @test count(problem.adjacency_matrix) == 2 * length(problem.adjacency_edges)
    @test all(problem.adjacency_matrix[i, j]
              for (i, j) in problem.adjacency_edges)

    visited = falses(problem.n_parcels)
    frontier = [1]
    visited[1] = true
    while !isempty(frontier)
        parcel = pop!(frontier)
        for neighbor in 1:problem.n_parcels
            if problem.adjacency_matrix[parcel, neighbor] && !visited[neighbor]
                visited[neighbor] = true
                push!(frontier, neighbor)
            end
        end
    end
    @test all(visited)
end

function assert_feasible_witness(problem)
    witness = problem.feasible_witness
    @test witness !== nothing
    witness === nothing && return
    @test problem.infeasibility_certificate === nothing
    @test length(witness) == problem.n_parcels
    @test all(zoning -> 1 <= zoning <= problem.n_zoning_types, witness)
    @test all(!problem.environmental_restrictions[parcel, witness[parcel]]
              for parcel in 1:problem.n_parcels)

    if problem.minimum_zoning_requirements
        for zoning in eachindex(problem.min_counts_by_type)
            @test count(==(zoning), witness) >= problem.min_counts_by_type[zoning]
        end
    end

    if problem.zoning_adjacency_constraints && problem.n_zoning_types >= 3
        for (first_parcel, second_parcel) in problem.adjacency_edges
            pair = (witness[first_parcel], witness[second_parcel])
            @test pair != (1, 3)
            @test pair != (3, 1)
        end
    end

    for resource in 1:problem.n_resources
        usage = sum(
            problem.parcel_sizes[parcel] *
            problem.resource_consumption[witness[parcel], resource]
            for parcel in 1:problem.n_parcels
        )
        @test usage <= problem.resource_capacities[resource] + 1e-8
    end
end

function assert_infeasibility_certificate(problem)
    certificate = problem.infeasibility_certificate
    @test problem.feasible_witness === nothing
    @test certificate !== nothing
    certificate === nothing && return

    resource = certificate.resource_index
    @test 1 <= resource <= problem.n_resources
    @test length(certificate.per_parcel_minimum) == problem.n_parcels
    recomputed = zeros(problem.n_parcels)
    for parcel in 1:problem.n_parcels
        allowed = [zoning for zoning in 1:problem.n_zoning_types
                   if !problem.environmental_restrictions[parcel, zoning]]
        @test !isempty(allowed)
        recomputed[parcel] = problem.parcel_sizes[parcel] * minimum(
            problem.resource_consumption[zoning, resource] for zoning in allowed
        )
    end
    @test certificate.per_parcel_minimum ≈ recomputed
    @test certificate.lower_bound ≈ sum(recomputed)
    @test certificate.capacity == problem.resource_capacities[resource]
    @test certificate.capacity < certificate.lower_bound
end

@testset "Land-use generator quality" begin
    @testset "Complete catalogs and large sizing" begin
        # These seeds exercise both 11- and 12-zone catalog selections, which
        # used to crash while slicing ten-element metadata vectors.
        for target in (1001, 2000, 10_000), status in
            (feasible, infeasible, unknown), seed in (5, 22, 32)
            problem = SyntheticLPs.LandUseProblem(target, status, seed)
            @test 3 <= problem.n_zoning_types <= 12
            @test 3 <= problem.n_resources <= 8
            @test length(problem.zoning_names) == problem.n_zoning_types
            @test length(unique(problem.zoning_names)) == problem.n_zoning_types
            @test length(problem.resource_names) == problem.n_resources
            @test size(problem.development_costs) ==
                  (problem.n_parcels, problem.n_zoning_types)
            @test size(problem.revenues) ==
                  (problem.n_parcels, problem.n_zoning_types)
            @test size(problem.resource_consumption) ==
                  (problem.n_zoning_types, problem.n_resources)
        end
    end

    @testset "Local RNG and field determinism" begin
        Random.seed!(9182)
        first_draw = rand()
        expected_next_draw = rand()
        Random.seed!(9182)
        @test rand() == first_draw
        SyntheticLPs.LandUseProblem(300, feasible, 22)
        @test rand() == expected_next_draw

        for status in (feasible, infeasible, unknown)
            first_problem = SyntheticLPs.LandUseProblem(500, status, 12345)
            second_problem = SyntheticLPs.LandUseProblem(500, status, 12345)
            for field in fieldnames(typeof(first_problem))
                first_value = getfield(first_problem, field)
                second_value = getfield(second_problem, field)
                if field == :infeasibility_certificate && first_value !== nothing
                    @test second_value !== nothing
                    @test all(isequal(getfield(first_value, certificate_field),
                                      getfield(second_value, certificate_field))
                              for certificate_field in
                              fieldnames(typeof(first_value)))
                else
                    @test isequal(first_value, second_value)
                end
            end
        end
    end

    @testset "Spatial graph invariants" begin
        for target in (3, 4, 50, 500, 2000), seed in 0:5
            problem = SyntheticLPs.LandUseProblem(target, unknown, seed)
            assert_connected_spatial_graph(problem)
        end
    end

    @testset "Constructive feasibility evidence" begin
        for target in (3, 4, 50, 300, 1001), seed in 0:12
            problem = SyntheticLPs.LandUseProblem(target, feasible, seed)
            assert_feasible_witness(problem)
        end
    end

    @testset "Resource lower-bound certificates" begin
        for target in (3, 4, 50, 300, 1001), seed in 0:12
            problem = SyntheticLPs.LandUseProblem(target, infeasible, seed)
            assert_infeasibility_certificate(problem)
        end
    end

    @testset "One model row pair per undirected edge" begin
        problem = nothing
        for seed in 0:30
            candidate = SyntheticLPs.LandUseProblem(240, feasible, seed)
            if candidate.zoning_adjacency_constraints
                problem = candidate
                break
            end
        end
        @test problem !== nothing
        if problem !== nothing
            model = SyntheticLPs.build_model(problem)
            less_than_rows = num_constraints(
                model,
                JuMP.AffExpr,
                LAND_USE_MOI.LessThan{Float64},
            )
            @test less_than_rows ==
                  problem.n_resources + 2 * length(problem.adjacency_edges)
            @test length(model[:residential_industrial_forward]) ==
                  length(problem.adjacency_edges)
            @test length(model[:residential_industrial_reverse]) ==
                  length(problem.adjacency_edges)
        end
    end

    @testset "Solver status agrees without retry filtering" begin
        if HAS_LAND_USE_HIGHS
            for relax_integer in (true, false), target in (40, 120), seed in 0:3,
                status in (feasible, infeasible)
                model, problem = generate_problem(
                    LAND_USE_REF,
                    target,
                    status,
                    seed;
                    relax_integer = relax_integer,
                )
                @test num_variables(model) ==
                      problem.n_parcels * problem.n_zoning_types
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                expected = status == feasible ? LAND_USE_MOI.OPTIMAL :
                                                 LAND_USE_MOI.INFEASIBLE
                @test termination_status(model) == expected
            end
        else
            @info "HiGHS unavailable; skipping land-use solve checks"
        end
    end
end

# The smallest formulations used to crash with an empty-range rand when
# n_parcels == 2.
@testset "Land Use Tiny Target Robustness" begin
    @test_nowarn generate_problem("land_use/standard", 3, unknown, 1)
    @test_nowarn generate_problem("land_use/standard", 4, unknown, 1)
end
