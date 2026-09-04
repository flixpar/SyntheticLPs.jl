# Focused quality contracts for the knapsack category: the mixed_integer_set
# variant stores sparse row supports instead of a dense n_rows x n_variables
# coefficient matrix, and every stored index is in range.
@testset "Knapsack Mixed-Integer Set Sparsity" begin
    @test_nowarn generate_problem("knapsack/mixed_integer_set", 1, unknown, 1)
    _, mik = generate_problem("knapsack/mixed_integer_set", 80, unknown, 1)
    @test length(mik.row_indices) == mik.n_rows
    @test all(length(mik.row_indices[r]) == length(mik.row_coefficients[r]) for r in 1:mik.n_rows)
    @test all(
        allunique(mik.row_indices[r]) &&
            all(1 <= i <= mik.n_integer + mik.n_continuous for i in mik.row_indices[r]) for
        r in 1:mik.n_rows
    )
end
