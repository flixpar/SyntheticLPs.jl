# Focused quality contracts for the multi_commodity_flow category: the discrete
# variants sample extra arcs by rejection rather than materializing every
# ordered node pair, so the arc list stays distinct and loop-free.
@testset "Multi-Commodity Flow Arc Sampling" begin
    for ref in ("multi_commodity_flow/binary_capacity",
                "multi_commodity_flow/integer_flow")
        @test_nowarn generate_problem(ref, 20, unknown, 1)
        _, mcf = generate_problem(ref, 200, unknown, 1)
        @test length(mcf.arcs) == mcf.n_arcs
        @test length(unique(mcf.arcs)) == mcf.n_arcs
        @test all(a[1] != a[2] for a in mcf.arcs)
    end
end
