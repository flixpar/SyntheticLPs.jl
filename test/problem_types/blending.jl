# Focused quality contracts for the blending category (and its feed_blending
# cousin, exercised in the same loop): infeasible requests used to be heuristic
# (~8-17% came back feasible); the optimizer guard now enforces the contract.
@testset "Blending Feasibility Contracts" begin
    if HAS_HIGHS
        for ref in ("blending/standard", "feed_blending/standard")
            for s in 1:6
                m, _ = generate_problem(ref, 300, infeasible, s; optimizer=HiGHS.Optimizer)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
            end
        end
    end
end
