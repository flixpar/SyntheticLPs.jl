# Focused robustness contracts for the set_system category: every variant
# sizes a feasible planted partition down to 2 columns/bids instead of
# rejecting targets below 4, so tiny dataset draws succeed.
@testset "Set System Tiny Target Robustness" begin
    for variant in ("set_cover", "set_packing", "set_partitioning",
                    "combinatorial_auction")
        ref = "set_system/$variant"
        @test_nowarn generate_problem(ref, 2, unknown, 1)
        @test_nowarn generate_problem(ref, 3, infeasible, 1)
    end
    tiny = generate_dataset(num_problems = 4,
                            size_distribution = Uniform(2, 3),
                            problem_types = [:set_system],
                            seed = 1,
                            max_candidate_multiplier = 3)
    @test length(tiny) == 4
end
