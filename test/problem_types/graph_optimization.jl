# Focused quality contracts for the graph_optimization category: the
# generalized_independent_set variant's hard/soft edge budget, which used to
# request more hard edges than there were unused pairs for targets 6-10.
@testset "Generalized Independent Set Edge Budget" begin
    for target in 6:10, status in (feasible, unknown, infeasible)
        @test_nowarn generate_problem("graph_optimization/generalized_independent_set",
                                      target, status, 1)
    end
    _, gis = generate_problem("graph_optimization/generalized_independent_set", 6, feasible, 1)
    @test length(gis.soft_edges) == 6 - gis.n_vertices
    @test length(gis.hard_edges) + length(gis.soft_edges) <=
          gis.n_vertices * (gis.n_vertices - 1) ÷ 2
end
