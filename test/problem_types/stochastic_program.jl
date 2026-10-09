# Focused quality contracts for the stochastic_program category: registry
# shape; the two-stage `standard` variant's exact sizing, sparse-lane data,
# coherent first/second stage (capacity serves most demand), planted witness
# and service-level/capital certificate; the `multistage_alm` variant's
# scenario tree, sizing, fixed-mix witness and path-bound certificate; and
# HiGHS contracts including a genuinely two-sided `unknown`.
@testset "Stochastic program" begin
    @test Set(list_variants(:stochastic_program)) == Set([:standard, :multistage_alm])
    @test problem_info(:stochastic_program)[:default_variant] == :standard

    # ------------------------------------------------------------------ standard
    for target in (50, 500, 5000, 50_000), status in (feasible, infeasible, unknown), seed in 0:1
        model, p = generate_problem("stochastic_program/standard", target, status, seed)
        I, J, S, L = p.n_facilities, p.n_customers, p.n_scenarios, length(p.lanes)
        @test num_variables(model) == I + S * (L + J)
        @test num_constraints(model; count_variable_in_set_constraints=false) == 1 + S * (I + J + 1)
        @test abs(num_variables(model) - target) <= max(0.04 * target, S)
        # Sparse lanes: every customer has 2..6 distinct facilities, sorted by customer.
        @test allunique(p.lanes)
        @test issorted(p.lanes; by=l -> (l[2], l[1]))
        for j in 1:J
            @test min(2, I) <= count(l -> l[2] == j, p.lanes) <= min(6, I)
        end
        @test sum(p.scenario_prob) ≈ 1.0
        @test all(>(0), p.demand)
        @test 0.85 <= p.service_level <= 0.96
        @test all(p.existing_capacity .<= p.capacity_max)
        # Shortfall is dearer than the customer's average lane plus amortized capacity.
        for j in 1:J
            lane_costs = [p.ship_cost[l] for l in eachindex(p.lanes) if p.lanes[l][2] == j]
            @test p.shortfall_cost[j] > sum(lane_costs) / length(lane_costs) + minimum(p.build_cost)
        end
        @test (p.feasible_witness !== nothing) == (status == feasible)
        @test (p.infeasibility_certificate !== nothing) == (status == infeasible)
    end

    # Witness arithmetic without a solver.
    for seed in 0:3
        _, p = generate_problem("stochastic_program/standard", 3000, feasible, seed)
        w = p.feasible_witness
        I, J, S = p.n_facilities, p.n_customers, p.n_scenarios
        @test all(p.existing_capacity .- 1e-9 .<= w.capacity .<= p.capacity_max .+ 1e-9)
        @test sum(p.capital_use .* w.capacity) <= p.capital_budget + 1e-6
        for s in 1:S
            out = zeros(I)
            into = zeros(J)
            for (l, (i, j)) in enumerate(p.lanes)
                @test w.shipment[l, s] >= 0
                out[i] += w.shipment[l, s]
                into[j] += w.shipment[l, s]
            end
            @test all(out .<= w.capacity .+ 1e-7)
            @test all(isapprox.(into .+ w.shortfall[:, s], p.demand[:, s]; atol=1e-7))
            @test sum(w.shortfall[:, s]) <= (1 - p.service_level) * sum(p.demand[:, s]) + 1e-6
        end
    end

    # Certificate arithmetic: greedy minimum capital for the worst scenario.
    for seed in 0:5
        _, p = generate_problem("stochastic_program/standard", 2000, infeasible, seed)
        c = p.infeasibility_certificate
        totals = vec(sum(p.demand; dims=1))
        @test c.scenario == argmax(totals)
        @test c.required_capacity ≈ p.service_level * totals[c.scenario]
        @test c.min_capital ≈ SyntheticLPs._stochastic_program_min_capital(
            p.capital_use, p.existing_capacity, p.capacity_max, c.required_capacity
        )
        @test c.capital_budget == p.capital_budget
        @test c.margin ≈ c.min_capital - p.capital_budget
        @test c.margin >= 0.08 * c.min_capital - 1e-9
        # Not a single-row contradiction: existing capacity alone fits the budget.
        @test sum(p.capital_use .* p.existing_capacity) < p.capital_budget
    end
    # The greedy bound itself, on a tiny hand-checked case.
    @test SyntheticLPs._stochastic_program_min_capital([2.0, 1.0], [1.0, 0.0], [3.0, 2.0], 4.0) ==
        2.0 + 2.0 + 2.0
    @test SyntheticLPs._stochastic_program_min_capital([1.0], [0.0], [1.0], 2.0) == Inf

    # ------------------------------------------------------------ multistage_alm
    for target in (60, 600, 6000, 60_000), status in (feasible, infeasible, unknown), seed in 0:1
        model, p = generate_problem("stochastic_program/multistage_alm", target, status, seed)
        A, N = p.n_assets, length(p.parent)
        leaves = count(==(p.n_stages), p.stage)
        @test num_variables(model) == (N - leaves) * (3A - 1) + leaves * 3A
        equity_rows = N
        @test num_constraints(model; count_variable_in_set_constraints=false) ==
            N * (A - 1) + N + N + equity_rows + leaves
        @test abs(num_variables(model) - target) <= max(0.03 * target, 3A)
        # Breadth-first tree: parents precede children, stages increase by one,
        # every pre-leaf node has a child, probabilities are consistent.
        @test p.parent[1] == 0 && p.stage[1] == 0
        @test all(p.parent[n] < n && p.stage[n] == p.stage[p.parent[n]] + 1 for n in 2:N)
        @test all(any(==(n), p.parent) for n in 1:N if p.stage[n] < p.n_stages)
        @test sum(p.probability[n] for n in 1:N if p.stage[n] == p.n_stages) ≈ 1.0
        @test all(>(0), p.returns)
        @test p.liability_value[1] ≈ 1.0
        @test p.asset_kind[1] == :cash && p.transaction_cost[1] == 0.0
        @test (p.feasible_witness !== nothing) == (status == feasible)
        @test (p.infeasibility_certificate !== nothing) == (status == infeasible)
    end

    # Fixed-mix witness: balances hold exactly, caps and funding floors hold.
    for seed in 0:3, target in (500, 4000)
        _, p = generate_problem("stochastic_program/multistage_alm", target, feasible, seed)
        w = p.feasible_witness
        A, N = p.n_assets, length(p.parent)
        tc = p.transaction_cost
        @test sum(w.weights) ≈ 1.0
        @test all(w.weights .<= p.max_weight .+ 1e-12)
        for n in 1:N
            prev(a) =
                if p.parent[n] == 0
                    p.initial_holdings[a]
                else
                    p.returns[a, n] * w.holdings[a, p.parent[n]]
                end
            for a in 2:A
                @test w.holdings[a, n] ≈ prev(a) + w.buys[a, n] - w.sells[a, n] atol = 1e-9
                if p.max_weight[a] < 1.0
                    @test w.holdings[a, n] <=
                        p.max_weight[a] * p.exposure_scale * p.liability_value[n] + 1e-9
                end
            end
            cash =
                prev(1) +
                sum((1 - tc[a]) * w.sells[a, n] - (1 + tc[a]) * w.buys[a, n] for a in 2:A) +
                p.inflow[n] - p.outflow[n]
            @test w.holdings[1, n] ≈ cash atol = 1e-9
            @test all(>=(-1e-12), w.holdings[:, n])
            wealth = sum(w.holdings[:, n])
            equities = sum(w.holdings[a, n] for a in 1:A if p.asset_kind[a] == :equity)
            @test equities <= p.equity_cap * p.exposure_scale * p.liability_value[n] + 1e-9
            n > 1 && @test wealth >= p.funding_ratio * p.liability_value[n] - 1e-9
        end
        @test p.funding_ratio <= 0.97 * w.min_funding_ratio + 1e-12
    end

    # Path-bound certificate recomputed from the stored data.
    for seed in 0:5
        _, p = generate_problem("stochastic_program/multistage_alm", 3000, infeasible, seed)
        c = p.infeasibility_certificate
        @test c.path[1] == 1
        @test all(p.parent[c.path[k]] == c.path[k - 1] for k in 2:length(c.path))
        bound = sum(p.initial_holdings)
        @test c.wealth_bound[1] ≈ bound
        for k in 2:length(c.path)
            n, parent = c.path[k], c.path[k - 1]
            caps = [
                if p.max_weight[a] < 1.0
                    p.max_weight[a] * p.exposure_scale * p.liability_value[parent]
                else
                    Inf
                end for a in 1:p.n_assets
            ]
            growth = SyntheticLPs._alm_best_growth(p.returns[:, n], caps, c.wealth_bound[k - 1])
            @test c.growth_bound[k] ≈ growth
            @test c.wealth_bound[k] ≈ growth * c.wealth_bound[k - 1] + p.inflow[n] - p.outflow[n]
            @test growth <= maximum(p.returns[:, n]) + 1e-12
        end
        @test c.required_wealth ≈ p.funding_ratio * p.liability_value[c.path[end]]
        @test c.margin ≈ c.required_wealth - c.wealth_bound[end]
        @test c.margin > 0.03 * c.required_wealth
    end
    @test SyntheticLPs._alm_best_growth([1.1, 1.5, 0.9], [Inf, 2.0, Inf], 4.0) ≈
        (1.5 * 2 + 1.1 * 2) / 4

    # Reproducibility of the stored data.
    for ref in ("stochastic_program/standard", "stochastic_program/multistage_alm")
        _, a = generate_problem(ref, 2500, unknown, 9)
        _, b = generate_problem(ref, 2500, unknown, 9)
        for name in fieldnames(typeof(a))
            @test isequal(getfield(a, name), getfield(b, name))
        end
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            function sp_status(m)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                return termination_status(m)
            end
            for ref in ("stochastic_program/standard", "stochastic_program/multistage_alm")
                for target in (300, 3000), seed in 0:3
                    @test sp_status(first(generate_problem(ref, target, feasible, seed))) ==
                        MOI.OPTIMAL
                    @test sp_status(first(generate_problem(ref, target, infeasible, seed))) ==
                        MOI.INFEASIBLE
                end
                outcomes = [sp_status(first(generate_problem(ref, 2000, unknown, s))) for s in 0:15]
                @test count(==(MOI.OPTIMAL), outcomes) >= 3
                @test count(==(MOI.INFEASIBLE), outcomes) >= 2
                @test all(in((MOI.OPTIMAL, MOI.INFEASIBLE)), outcomes)
            end

            # Coherent stages: with the budget slack, the optimal plan serves
            # most of the expected demand (the shortfall penalty is not just
            # absorbing it).
            for seed in 0:3
                m, p = generate_problem("stochastic_program/standard", 3000, feasible, seed)
                @test sp_status(m) == MOI.OPTIMAL
                z = value.(m[:z])
                expected_demand = sum(
                    p.scenario_prob[s] * sum(p.demand[:, s]) for s in 1:p.n_scenarios
                )
                expected_short = sum(p.scenario_prob[s] * sum(z[:, s]) for s in 1:p.n_scenarios)
                @test expected_short <= 0.10 * expected_demand
            end
        end
    end
end
