# Focused robustness contracts for the container_loading category: both
# variants clamp to their smallest formulation instead of rejecting targets
# below 12 (standard) / 30 (2-D packing).
@testset "Container Loading Tiny Target Robustness" begin
    @test_nowarn generate_problem("container_loading/standard", 2, unknown, 1)
    @test_nowarn generate_problem("container_loading/standard", 11, feasible, 1)
    @test_nowarn generate_problem("container_loading/two_dimensional_bin_packing", 2, unknown, 1)
    @test_nowarn generate_problem(
        "container_loading/two_dimensional_bin_packing", 29, infeasible, 1
    )
end
