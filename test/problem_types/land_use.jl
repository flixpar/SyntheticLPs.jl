# Focused quality contracts for land_use/standard (spatial zoning plan):
# sparse allowed pairs and sizing, spatial graph and district invariants
# (edge list only, no dense adjacency), the planted integer plan checked row
# by row, the district resource certificate recomputed from the data,
# reproducibility / global-RNG isolation, and HiGHS-backed status contracts
# for both the LP relaxation and the MIP.
@testset "Land use" begin
    @test Set(list_variants(:land_use)) == Set([:standard])
    S = SyntheticLPs
    ref = "land_use/standard"

    function connected(p)
        neighbors = S._land_use_neighbors(p.n_parcels, p.adjacency_edges)
        seen = falses(p.n_parcels)
        stack = [1]
        seen[1] = true
        while !isempty(stack)
            i = pop!(stack)
            for j in neighbors[i]
                seen[j] || (seen[j] = true; push!(stack, j))
            end
        end
        return all(seen)
    end

    @testset "sizing, graph, witness, certificates" begin
        for target in (3, 50, 300, 2_000, 12_000), status in (feasible, infeasible, unknown), seed in 0:2
            model, p = generate_problem(ref, target, status, seed)
            @test num_variables(model) == length(p.pairs)
            @test abs(length(p.pairs) - target) <= max(0.1 * target, 12)
            @test allunique(p.pairs) && issorted(p.pairs)
            @test all(1 <= z <= p.n_zoning_types for (_, z) in p.pairs)
            # Every parcel keeps at least one zone; no singleton exclusion rows.
            @test length(unique(first.(p.pairs))) == p.n_parcels
            @test issorted(p.adjacency_edges) && allunique(p.adjacency_edges)
            @test all(i < j for (i, j) in p.adjacency_edges)
            @test connected(p)
            @test all(1 .<= p.parcel_district .<= p.n_districts)
            @test size(p.resource_capacities) == (p.n_districts, p.n_resources)
            @test all(0.0 .< p.parcel_coordinates .< 1.0)
            if status == feasible
                @test S.land_use_plan_satisfies(p)
            elseif status == infeasible
                @test S.land_use_certificate_holds(p)
                c = p.infeasibility_certificate
                @test c.capacity <= c.lower_bound * 0.93 + 1e-9
            else
                @test p.feasible_witness === nothing && p.infeasibility_certificate === nothing
            end
        end
        big = S.LandUseProblem(100_000, feasible, 0)
        @test abs(length(big.pairs) - 100_000) <= 3_000
        @test big.n_districts > 100
    end

    @testset "reproducibility and global-RNG isolation" begin
        for status in (feasible, infeasible, unknown)
            p1 = S.LandUseProblem(500, status, 12345)
            p2 = S.LandUseProblem(500, status, 12345)
            for f in fieldnames(typeof(p1))
                a, b = getfield(p1, f), getfield(p2, f)
                if a isa S.LandUseInfeasibilityCertificate
                    @test all(isequal(getfield(a, g), getfield(b, g)) for g in fieldnames(typeof(a)))
                else
                    @test isequal(a, b)
                end
            end
        end
        Random.seed!(9182)
        expected = rand()
        Random.seed!(9182)
        S.LandUseProblem(300, feasible, 22)
        @test rand() == expected
    end

    @testset "HiGHS feasibility contracts" begin
        if HAS_HIGHS
            for relax in (true, false), target in (40, 400), seed in 0:3, status in (feasible, infeasible)
                model, _ = generate_problem(ref, target, status, seed; relax_integer=relax)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                optimize!(model)
                @test termination_status(model) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end
        end
    end
end
