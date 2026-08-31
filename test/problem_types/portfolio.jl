# Focused robustness contracts for the portfolio category: extreme asset
# counts that used to throw an ArgumentError (Uniform a < b) during sampling.
@testset "Portfolio Sizing Robustness" begin
    @test_nowarn generate_problem("portfolio/cvar", 2000, unknown, 1)
    @test_nowarn generate_problem("portfolio/cvar", 1300, unknown, 1)
    @test_nowarn generate_problem("portfolio/cvar", 3, unknown, 1)
end
