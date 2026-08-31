# Focused quality contracts for the energy category: line sampling in the
# optimal_transmission_switching variant, the standard variant's emissions
# intensity target and the attainability of its feasible-profile cap, plus the
# HiGHS feasibility contracts for both labels.
@testset "Energy Generator Contracts" begin
    # energy/optimal_transmission_switching samples extra lines by rejection
    # instead of materializing the complete undirected edge set.
    @test_nowarn generate_problem("energy/optimal_transmission_switching", 3, unknown, 1)
    _, ots = generate_problem("energy/optimal_transmission_switching", 500, unknown, 1)
    @test ots.n_lines >= ots.n_buses - 1
    @test length(unique(ots.line_from[k] < ots.line_to[k] ?
                        (ots.line_from[k], ots.line_to[k]) :
                        (ots.line_to[k], ots.line_from[k])
                        for k in 1:ots.n_lines)) == ots.n_lines
    # energy now stores an emissions intensity target (the previous per-period
    # emissions row was an algebraic tautology).
    _, eprob = generate_problem("energy/standard", 120, unknown, 1)
    @test hasproperty(eprob, :emission_intensity_target)
    @test eprob.emission_intensity_target > 0

    # Feasible energy instances choose an emissions cap that is attainable at
    # peak demand and provide enough zero-emission capacity for the renewable
    # floor, even when no optimizer-backed retry guard is requested.
    for target in (120, 300, 1200), seed in 1:10
        _, prob = generate_problem("energy/standard", target, feasible, seed)
        peak_demand = maximum(prob.demands)
        clean_sources = [s for s in prob.sources if iszero(prob.emission_limits[s])]
        @test sum(prob.capacities[s] for s in clean_sources) + 1e-8 >=
              prob.renewable_fraction * peak_demand

        remaining_demand = peak_demand
        minimum_emissions = 0.0
        for source in sort(prob.sources; by=s -> prob.emission_limits[s])
            generation = min(prob.capacities[source], remaining_demand)
            minimum_emissions += prob.emission_limits[source] * generation
            remaining_demand -= generation
            remaining_demand <= 0 && break
        end
        @test remaining_demand <= 1e-8
        @test minimum_emissions / peak_demand <=
              prob.emission_intensity_target + 1e-12
    end
end

@testset "Energy Feasibility Contracts" begin
    if HAS_HIGHS
        # energy/standard infeasible-request: previously failed at larger sizes
        # because the infeasibility logic targeted a reserve constraint that is not
        # in the model.
        for s in 1:8
            m, _ = generate_problem("energy/standard", 300, infeasible, s;
                                    optimizer = HiGHS.Optimizer)
            set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
            @test termination_status(m) in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
        end

        # Feasible energy requests also honor their label by construction when the
        # initial generation call does not use the optimizer-backed retry guard.
        for s in 1:10
            m, _ = generate_problem("energy/standard", 300, feasible, s)
            set_optimizer(m, HiGHS.Optimizer); set_silent(m); optimize!(m)
            @test termination_status(m) == MOI.OPTIMAL
        end
    end
end
