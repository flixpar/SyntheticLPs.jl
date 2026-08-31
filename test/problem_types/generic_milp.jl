# Focused quality contracts for the generic_milp category: each row support is
# sampled in O(width) rather than by permuting all n columns, and comes back
# sorted, distinct, and within the column range.
@testset "Generic MILP Row Supports" begin
    @test_nowarn generate_problem("generic_milp/standard", 3, unknown, 1)
    _, gmilp = generate_problem("generic_milp/standard", 200, unknown, 1)
    @test all(issorted(row.indices) && allunique(row.indices) for row in gmilp.rows)
    @test all(1 <= length(row.indices) <= gmilp.n_variables for row in gmilp.rows)
end
