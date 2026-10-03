# Focused quality contracts for the game_theory category: registry shape,
# exact sizing formulas (and the documented 1,000,000-variable cap),
# sequence-form / layered-DAG / time-expanded structural invariants, witness
# and certificate arithmetic recomputed from the struct fields without a
# solver, reproducibility, and HiGHS-backed contracts (including the known
# values of Kuhn and Leduc poker).
using SparseArrays

const GT = SyntheticLPs

gt_rows(m) = num_constraints(m; count_variable_in_set_constraints=false)

# ---------------------------------------------------------------------------
# Sequence-form helpers (pure arithmetic on the struct fields)
# ---------------------------------------------------------------------------

# Flow-row residuals of a realization plan: x[1] - 1 and, per infoset,
# sum(x[seqs(I)]) - x[parent(I)].
function gt_flow_residual(T, x)
    r = abs(x[1] - 1)
    for I in eachindex(T.infoset_parent)
        f, k = T.infoset_first[I], T.infoset_num_actions[I]
        r = max(r, abs(sum(x[f:(f + k - 1)]) - x[T.infoset_parent[I]]))
    end
    return r
end

# (F' q)[τ] for the opponent treeplex: q[owner(τ)] - Σ q[children(τ)].
function gt_Ftq(T, q)
    out = zeros(T.num_sequences)
    out[1] = q[1]
    for J in eachindex(T.infoset_parent)
        f, k = T.infoset_first[J], T.infoset_num_actions[J]
        out[f:(f + k - 1)] .+= q[J + 1]
        out[T.infoset_parent[J]] -= q[J + 1]
    end
    return out
end

# (E' p - T' μ)[σ] for the seat treeplex with tremble ε.
function gt_Etp_minus_Ttmu(T, p, μ, ε)
    out = gt_Ftq(T, p)  # same incidence structure as F' q
    for I in eachindex(T.infoset_parent)
        f, k = T.infoset_first[I], T.infoset_num_actions[I]
        for s in f:(f + k - 1)
            out[s] -= μ[s]
            out[T.infoset_parent[I]] += ε * μ[s]
        end
    end
    return out
end

@testset "Game Theory" begin
    @test :game_theory in list_categories()
    @test Set(list_variants(:game_theory)) == Set([:poker_sequence_form, :colonel_blotto, :patrol_security])
    info = problem_info(:game_theory)
    @test info[:default_variant] == :poker_sequence_form
    @test occursin("zero-sum", lowercase(info[:description]))
    @test GT.GAME_THEORY_MAX_VARIABLES == 1_000_000

    @testset "poker_sequence_form" begin
        # Sizing: the model has exactly the formula's variables and rows, the
        # formula equals the treeplex dimensions, and the chosen game lands
        # within 3% of the target (targets below the 20-variable Kuhn game
        # round up to it).
        for target in (1, 20, 50, 200, 1000, 5000), status in (feasible, infeasible, unknown), seed in 0:1
            m, p = generate_problem(:game_theory, target, status, seed; variant=:poker_sequence_form)
            shapes = [(length(r.bet_sizes), r.raise_cap) for r in p.rounds]
            v, r, _ = GT._poker_size_formula(p.n_ranks, shapes, p.seat)
            X, Y = p.player_tree, p.opponent_tree
            @test num_variables(m) == v == X.num_sequences + length(Y.infoset_parent) + 1
            n_tremble = p.tremble > 0 ? X.num_sequences - 1 : 0
            @test gt_rows(m) == r + n_tremble == length(X.infoset_parent) + 1 + Y.num_sequences + n_tremble
            # 3% from 200 variables up; small games are coarser (at most ~5%).
            @test abs(v - max(target, 20)) <= (target >= 200 ? 0.03 : 0.05) * max(target, 20)
            @test size(p.payoff) == (X.num_sequences, Y.num_sequences)
            @test p.seat in (1, 2)
            @test length(p.rounds) in (1, 2)
            @test 3 <= p.n_ranks <= 20
            @test length(p.rounds) == 1 || p.n_suits >= 2
            @test p.tremble == 0 || 0.002 <= p.tremble <= 0.02
        end
        # Large targets: constructor-only sizing (fast) and the cap.
        p = GT.PokerSequenceFormProblem(100_000, unknown, 3)
        shapes = [(length(r.bet_sizes), r.raise_cap) for r in p.rounds]
        v, _, _ = GT._poker_size_formula(p.n_ranks, shapes, p.seat)
        @test v == p.player_tree.num_sequences + length(p.opponent_tree.infoset_parent) + 1
        @test abs(v - 100_000) <= 3_000
        @test nnz(p.payoff) <= 10 * 100_000 + 2000
        @test_throws ArgumentError generate_problem(
            :game_theory, 1_000_001, unknown, 0; variant=:poker_sequence_form
        )

        # Round shape counts match the enumerated public tree.
        for nb in 1:3, cap in 1:3
            c = GT._poker_round_counts(nb, cap)
            pub = GT._poker_public_tree([GT.PokerBettingRound(collect(1:nb), cap)])
            dec = [count(h -> !pub.is_terminal[h] && pub.actor[h] == q, eachindex(pub.actor)) for q in 1:2]
            @test Tuple(dec) == c.decisions
            @test count(pub.is_terminal) == c.continuations + c.folds
            @test count(h -> pub.is_terminal[h] && pub.folder[h] > 0, eachindex(pub.actor)) == c.folds
        end

        # Treeplex invariants: sequences 2..n are partitioned into contiguous
        # infoset blocks, parents precede their children, and every infoset
        # offers at least two actions.
        for seed in 0:3
            _, p = generate_problem(:game_theory, 3000, unknown, seed; variant=:poker_sequence_form)
            for T in (p.player_tree, p.opponent_tree)
                covered = falses(T.num_sequences)
                for I in eachindex(T.infoset_parent)
                    @test T.infoset_num_actions[I] >= 2
                    @test T.infoset_parent[I] < T.infoset_first[I]
                    seqs = T.infoset_first[I]:(T.infoset_first[I] + T.infoset_num_actions[I] - 1)
                    @test !any(covered[seqs])
                    covered[seqs] .= true
                end
                @test !covered[1] && all(covered[2:end])
            end
        end

        # Zero-sum consistency: the seat-2 payoff is exactly minus the
        # transpose of player 1's payoff, the seat-1 payoff is player 1's, and
        # the payoff block stores only nonzero chance-weighted entries.
        rounds = [GT.PokerBettingRound([2, 4], 2), GT.PokerBettingRound([4], 1)]
        p1 = GT._poker_assemble(MersenneTwister(0), 4, 3, rounds, 1, 0.0, unknown)
        p2 = GT._poker_assemble(MersenneTwister(0), 4, 3, rounds, 2, 0.0, unknown)
        @test p2.payoff == -copy(transpose(p1.payoff))
        @test p1.player_tree == p2.opponent_tree || (p1.player_tree.infoset_parent == p2.opponent_tree.infoset_parent)
        @test all(!iszero, nonzeros(p1.payoff))
        # Both seats bracket the same game value with opposite signs.
        @test p1.lower_bound <= -p2.lower_bound + 1e-9 * p1.value_scale
        @test -p2.upper_bound <= p1.upper_bound + 1e-9 * p1.value_scale
        @test p1.lower_bound <= -p2.lower_bound || p1.lower_bound <= p1.upper_bound

        # Witness arithmetic: a realization plan satisfying the flow rows and
        # trembles exactly, opponent values satisfying every best-response
        # row, and a guarantee at least the requirement.
        for target in (100, 800, 4000), seed in 0:3
            m, p = generate_problem(:game_theory, target, feasible, seed; variant=:poker_sequence_form)
            w = p.feasible_witness
            @test w !== nothing && p.infeasibility_certificate === nothing
            X, Y, A = p.player_tree, p.opponent_tree, p.payoff
            x, q = w.realization_plan, w.infoset_values
            @test all(>=(-1e-12), x)
            @test gt_flow_residual(X, x) <= 1e-9
            for I in eachindex(X.infoset_parent), s in X.infoset_first[I]:(X.infoset_first[I] + X.infoset_num_actions[I] - 1)
                @test x[s] - p.tremble * x[X.infoset_parent[I]] >= -1e-9
            end
            slack = transpose(A) * x .- gt_Ftq(Y, q)
            @test minimum(slack) >= -1e-7 * p.value_scale
            @test q[1] ≈ w.guaranteed_value
            @test w.guaranteed_value ≈ p.lower_bound
            @test p.lower_bound <= p.upper_bound + 1e-9 * p.value_scale
            margin = p.lower_bound - p.required_value
            @test 0.02 * p.value_scale - 1e-9 <= margin <= 0.1 * p.value_scale + 1e-9
            # End-to-end against the built model.
            vals = Dict{VariableRef, Float64}()
            for s in eachindex(x)
                vals[m[:x][s]] = x[s]
            end
            for j in eachindex(q)
                vals[m[:q][j]] = q[j]
            end
            @test isempty(primal_feasibility_report(m, vals; atol=1e-6 * p.value_scale))
        end

        # Certificate arithmetic: an opponent realization plan, multipliers
        # with E'p - T'μ >= A y componentwise and μ >= 0, and a value bound
        # strictly below the requirement.
        for target in (100, 800, 4000), seed in 0:3
            _, p = generate_problem(:game_theory, target, infeasible, seed; variant=:poker_sequence_form)
            c = p.infeasibility_certificate
            @test c !== nothing && p.feasible_witness === nothing
            X, Y, A = p.player_tree, p.opponent_tree, p.payoff
            y = c.opponent_plan
            @test all(>=(-1e-12), y)
            @test gt_flow_residual(Y, y) <= 1e-9
            @test all(>=(0.0), c.tremble_multipliers)
            lhs = gt_Etp_minus_Ttmu(X, c.infoset_multipliers, c.tremble_multipliers, p.tremble)
            @test minimum(lhs .- A * y) >= -1e-7 * p.value_scale
            @test c.infoset_multipliers[1] ≈ c.value_bound
            @test c.value_bound ≈ p.upper_bound
            @test p.required_value - c.value_bound >= 0.02 * p.value_scale - 1e-9
        end

        # Unknown: no witness or certificate; the requirement brackets the
        # value bounds from both sides across seeds.
        below = above = 0
        for seed in 0:19
            _, p = generate_problem(:game_theory, 300, unknown, seed; variant=:poker_sequence_form)
            @test p.feasible_witness === nothing && p.infeasibility_certificate === nothing
            w = max(0.5 * (p.upper_bound - p.lower_bound), 0.02 * p.value_scale)
            @test p.lower_bound - w - 1e-9 <= p.required_value <= p.upper_bound + w + 1e-9
            p.required_value < p.lower_bound && (below += 1)
            p.required_value > p.upper_bound && (above += 1)
        end
        @test below > 0 && above > 0

        # Reproducibility, isolated from the global RNG.
        for status in (feasible, infeasible, unknown)
            Random.seed!(1)
            _, a = generate_problem(:game_theory, 900, status, 42; variant=:poker_sequence_form)
            Random.seed!(2)
            _, b = generate_problem(:game_theory, 900, status, 42; variant=:poker_sequence_form)
            @test a.payoff == b.payoff
            @test a.required_value == b.required_value
            @test a.player_tree.infoset_parent == b.player_tree.infoset_parent
            @test (a.n_ranks, a.n_suits, a.seat, a.tremble) == (b.n_ranks, b.n_suits, b.seat, b.tremble)
        end

        @testset "HiGHS contracts" begin
            if HAS_HIGHS
                # Classical games: Kuhn poker (value -1/18 for the first
                # player) and Leduc hold'em (about -0.0856), as LP values per
                # hand. The requirement is far below the value here.
                kuhn = GT._poker_assemble(
                    MersenneTwister(0), 3, 1, [GT.PokerBettingRound([1], 1)], 1, 0.0, feasible
                )
                m = GT.build_model(kuhn)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == MOI.OPTIMAL
                @test objective_value(m) / kuhn.value_scale ≈ -1 / 18 atol = 1e-7
                leduc = GT._poker_assemble(
                    MersenneTwister(0),
                    3,
                    2,
                    [GT.PokerBettingRound([2], 2), GT.PokerBettingRound([4], 2)],
                    1,
                    0.0,
                    feasible,
                )
                @test leduc.player_tree.num_sequences == 1 + 3 * 7 + 9 * 5 * 7
                m = GT.build_model(leduc)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == MOI.OPTIMAL
                @test objective_value(m) / leduc.value_scale ≈ -0.0856 atol = 5e-4

                for target in (60, 400, 1500), status in (feasible, infeasible), seed in 0:2
                    m, p = generate_problem(:game_theory, target, status, seed; variant=:poker_sequence_form)
                    set_optimizer(m, HiGHS.Optimizer)
                    set_silent(m)
                    optimize!(m)
                    if status == feasible
                        @test termination_status(m) == MOI.OPTIMAL
                        v = objective_value(m)
                        tol = 1e-6 * p.value_scale
                        @test p.lower_bound - tol <= v <= p.upper_bound + tol
                    else
                        @test termination_status(m) == MOI.INFEASIBLE
                    end
                end
                # Unknown instances resolve both ways.
                outcomes = Set{MOI.TerminationStatusCode}()
                for seed in 0:11
                    m, _ = generate_problem(:game_theory, 300, unknown, seed; variant=:poker_sequence_form)
                    set_optimizer(m, HiGHS.Optimizer)
                    set_silent(m)
                    optimize!(m)
                    push!(outcomes, termination_status(m))
                end
                @test MOI.OPTIMAL in outcomes && MOI.INFEASIBLE in outcomes
            end
        end
    end

    @testset "colonel_blotto" begin
        for target in (1, 50, 200, 1000, 5000), status in (feasible, infeasible, unknown), seed in 0:1
            m, p = generate_problem(:game_theory, target, status, seed; variant=:colonel_blotto)
            K, S, So = p.n_battlefields, p.budget, p.opponent_budget
            v, r = GT._blotto_size_formula(K, S, So)
            @test num_variables(m) == v
            @test gt_rows(m) == r
            target >= 50 && @test abs(v - target) <= 0.02 * target
            @test K >= 3 && S >= 2 && So >= 2
            @test 0.6 * S - 1 <= So <= 1.5 * S + 1
            @test length(p.weights) == length(p.advantage) == K
            @test all(>=(1.0), p.weights)
            @test p.contest in (:majority, :lottery)
            @test p.lower_bound <= p.upper_bound + 1e-9
        end
        p = GT.ColonelBlottoProblem(100_000, unknown, 5)
        @test abs(GT._blotto_size_formula(p.n_battlefields, p.budget, p.opponent_budget)[1] - 100_000) <= 2_000
        @test_throws ArgumentError generate_problem(:game_theory, 1_000_001, unknown, 0; variant=:colonel_blotto)

        # DAG structure: edge count formula, every edge spends within budget,
        # the last layer spends the remainder, and edge indices are consistent.
        for (K, S) in ((3, 4), (5, 7), (8, 2))
            layer, from, step, tail, head = GT._blotto_edges(K, S)
            @test length(layer) == GT._blotto_num_edges(K, S)
            @test all(from .+ step .<= S)
            @test all(from[e] + step[e] == S for e in eachindex(layer) if layer[e] == K)
            @test all(GT._blotto_edge_index(K, S, layer[e], from[e], step[e]) == e for e in eachindex(layer))
            @test all((head[e] == 0) == (layer[e] == K) for e in eachindex(layer))
        end

        # Payoff tables: majority is antisymmetric at equal advantage-free
        # budgets; lottery payoffs stay within the battlefield weight.
        u = GT._blotto_payoffs([3.0, 5.0], [0, 0], :majority, 1.0, 4, 4)
        @test all(u[k, a, b] == -u[k, b, a] for k in 1:2, a in 1:5, b in 1:5)
        u = GT._blotto_payoffs([3.0, 5.0], [0, 1], :lottery, 0.8, 6, 5)
        @test all(abs(u[k, a, b]) <= [3.0, 5.0][k] + 1e-12 for k in 1:2, a in 1:7, b in 1:6)

        # Witness: a unit flow whose marginals and payoffs match, potentials
        # satisfying every opponent-edge row, and a guarantee above the
        # requirement.
        for target in (100, 900, 4000), seed in 0:2
            m, p = generate_problem(:game_theory, target, feasible, seed; variant=:colonel_blotto)
            w = p.feasible_witness
            @test w !== nothing && p.infeasibility_certificate === nothing
            K, S, So = p.n_battlefields, p.budget, p.opponent_budget
            layer, from, step, tail, head = GT._blotto_edges(K, S)
            x = w.allocation_flow
            @test all(>=(0.0), x)
            net = zeros(GT._blotto_num_nodes(K, S))
            for e in eachindex(x)
                net[tail[e]] += x[e]
                head[e] > 0 && (net[head[e]] -= x[e])
            end
            @test net[1] ≈ 1 atol = 1e-9
            @test maximum(abs, net[2:end]) <= 1e-9
            @test w.marginals ≈ GT._blotto_marginals(x, K, S)
            u = GT._blotto_payoffs(p.weights, p.advantage, p.contest, p.lottery_exponent, S, So)
            for k in 1:K
                @test w.expected_payoffs[k, :] ≈ transpose(u[k, :, :]) * w.marginals[k, :]
            end
            ol, _, os, ot, oh = GT._blotto_edges(K, So)
            pot = w.potentials
            viol = maximum(
                pot[ot[e]] - (oh[e] > 0 ? pot[oh[e]] : 0.0) - w.expected_payoffs[ol[e], os[e] + 1] for e in eachindex(ol)
            )
            @test viol <= 1e-9
            @test pot[1] ≈ w.guaranteed_value ≈ p.lower_bound
            @test 0.02 * sum(p.weights) - 1e-9 <= p.lower_bound - p.required_value <= 0.1 * sum(p.weights) + 1e-9
            vals = Dict{VariableRef, Float64}()
            for e in eachindex(x)
                vals[m[:x][e]] = x[e]
            end
            for k in 1:K, a in 0:S
                vals[m[:p][k, a]] = w.marginals[k, a + 1]
            end
            for k in 1:K, b in 0:So
                vals[m[:g][k, b]] = w.expected_payoffs[k, b + 1]
            end
            for j in eachindex(pot)
                vals[m[:pot][j]] = pot[j]
            end
            @test isempty(primal_feasibility_report(m, vals; atol=1e-6))
        end

        # Certificate: an opponent unit flow and seat potentials dominating
        # every seat edge against the opponent's marginals, bound below the
        # requirement.
        for target in (100, 900, 4000), seed in 0:2
            _, p = generate_problem(:game_theory, target, infeasible, seed; variant=:colonel_blotto)
            c = p.infeasibility_certificate
            @test c !== nothing && p.feasible_witness === nothing
            K, S, So = p.n_battlefields, p.budget, p.opponent_budget
            ol, _, os, ot, oh = GT._blotto_edges(K, So)
            z = c.opponent_flow
            @test all(>=(0.0), z)
            net = zeros(GT._blotto_num_nodes(K, So))
            for e in eachindex(z)
                net[ot[e]] += z[e]
                oh[e] > 0 && (net[oh[e]] -= z[e])
            end
            @test net[1] ≈ 1 atol = 1e-9
            @test maximum(abs, net[2:end]) <= 1e-9
            q = GT._blotto_marginals(z, K, So)
            u = GT._blotto_payoffs(p.weights, p.advantage, p.contest, p.lottery_exponent, S, So)
            h = reduce(vcat, [transpose(u[k, :, :] * q[k, :]) for k in 1:K])
            layer, _, step, tail, head = GT._blotto_edges(K, S)
            λ = c.potentials
            @test all(
                λ[tail[e]] >= h[layer[e], step[e] + 1] + (head[e] > 0 ? λ[head[e]] : 0.0) - 1e-9 for
                e in eachindex(layer)
            )
            @test λ[1] ≈ c.value_bound ≈ p.upper_bound
            @test p.required_value - c.value_bound >= 0.02 * sum(p.weights) - 1e-9
        end

        below = above = 0
        for seed in 0:19
            _, p = generate_problem(:game_theory, 300, unknown, seed; variant=:colonel_blotto)
            @test p.feasible_witness === nothing && p.infeasibility_certificate === nothing
            p.required_value < p.lower_bound && (below += 1)
            p.required_value > p.upper_bound && (above += 1)
        end
        @test below > 0 && above > 0

        for status in (feasible, infeasible, unknown)
            Random.seed!(3)
            m1, a = generate_problem(:game_theory, 700, status, 9; variant=:colonel_blotto)
            Random.seed!(4)
            m2, b = generate_problem(:game_theory, 700, status, 9; variant=:colonel_blotto)
            @test (a.n_battlefields, a.budget, a.opponent_budget, a.contest) ==
                (b.n_battlefields, b.budget, b.opponent_budget, b.contest)
            @test a.weights == b.weights && a.advantage == b.advantage
            @test a.required_value == b.required_value
            @test sprint(print, m1) == sprint(print, m2)
        end

        @testset "HiGHS contracts" begin
            if HAS_HIGHS
                for target in (60, 400, 1500), status in (feasible, infeasible), seed in 0:2
                    m, p = generate_problem(:game_theory, target, status, seed; variant=:colonel_blotto)
                    set_optimizer(m, HiGHS.Optimizer)
                    set_silent(m)
                    optimize!(m)
                    if status == feasible
                        @test termination_status(m) == MOI.OPTIMAL
                        @test p.lower_bound - 1e-6 <= objective_value(m) <= p.upper_bound + 1e-6
                    else
                        @test termination_status(m) == MOI.INFEASIBLE
                    end
                end
                outcomes = Set{MOI.TerminationStatusCode}()
                for seed in 0:11
                    m, _ = generate_problem(:game_theory, 300, unknown, seed; variant=:colonel_blotto)
                    set_optimizer(m, HiGHS.Optimizer)
                    set_silent(m)
                    optimize!(m)
                    push!(outcomes, termination_status(m))
                end
                @test MOI.OPTIMAL in outcomes && MOI.INFEASIBLE in outcomes
            end
        end
    end

    @testset "patrol_security" begin
        for target in (1, 50, 200, 1000, 5000), status in (feasible, infeasible, unknown), seed in 0:1
            m, p = generate_problem(:game_theory, target, status, seed; variant=:patrol_security)
            n, H, K = p.n_stations, p.horizon, length(p.type_prior)
            v = GT._patrol_size_formula(n, length(p.edges), H, K)
            @test num_variables(m) == v
            @test gt_rows(m) == 1 + (H - 1) * n + n * H + length(p.option_type) + 1
            target >= 200 && @test abs(v - target) <= 0.05 * target
            @test 2 <= K <= 6
            @test sum(p.type_prior) ≈ 1
            @test 2 <= p.n_units <= 16
            # Opportunistic type 1 threatens every node-time, so every
            # coverage variable sits in at least one attack row.
            @test sort(p.option_target[p.option_type .== 1]) == collect(1:(n * H))
            @test all(p.option_loss .> p.option_gain .> 0)
            @test p.lower_bound <= p.upper_bound + 1e-9
            # Network: connected spanning structure on the grid.
            @test length(p.edges) >= n - 1
            parent = collect(1:n)
            root(x) = parent[x] == x ? x : (parent[x] = root(parent[x]))
            for (a, b) in p.edges
                parent[root(a)] = root(b)
            end
            @test length(unique(root.(1:n))) == 1
        end
        p = GT.PatrolSecurityProblem(100_000, unknown, 2)
        @test abs(GT._patrol_size_formula(p.n_stations, length(p.edges), p.horizon, length(p.type_prior)) - 100_000) <=
            5_000
        @test_throws ArgumentError generate_problem(:game_theory, 1_000_001, unknown, 0; variant=:patrol_security)

        # Witness: the average patrol plan, its capped coverage, and each
        # type's best-response value satisfy every row of the model.
        for target in (150, 900, 4000), seed in 0:2
            m, p = generate_problem(:game_theory, target, feasible, seed; variant=:patrol_security)
            w = p.feasible_witness
            @test w !== nothing && p.infeasibility_certificate === nothing
            @test all(>=(0.0), w.arc_flows)
            @test all(0 .<= w.coverage .<= 1)
            @test sum(w.arc_flows[1:(p.n_stations)]) <= p.n_units + 1e-9
            @test w.expected_loss ≈ sum(p.type_prior .* w.type_values) ≈ p.upper_bound
            @test 0 < p.loss_requirement - w.expected_loss
            vals = Dict{VariableRef, Float64}()
            for a in eachindex(w.arc_flows)
                vals[m[:f][a]] = w.arc_flows[a]
            end
            for j in eachindex(w.coverage)
                vals[m[:c][j]] = w.coverage[j]
            end
            for k in eachindex(w.type_values)
                vals[m[:v][k]] = w.type_values[k]
            end
            @test isempty(primal_feasibility_report(m, vals; atol=1e-6))
        end

        # Certificate: recompute the Lagrangian lower bound from the attack
        # mix, threshold and potentials, and check the potential inequalities
        # on every time-expanded arc.
        for target in (150, 900, 4000), seed in 0:2
            _, p = generate_problem(:game_theory, target, infeasible, seed; variant=:patrol_security)
            c = p.infeasibility_certificate
            @test c !== nothing && p.feasible_witness === nothing
            n, H, K = p.n_stations, p.horizon, length(p.type_prior)
            α = c.attack_mix
            @test all(>=(0.0), α)
            for k in 1:K
                @test sum(α[p.option_type .== k]) ≈ 1
            end
            wb = zeros(n * H)
            base = 0.0
            for j in eachindex(α)
                pk = p.type_prior[p.option_type[j]]
                wb[p.option_target[j]] += pk * α[j] * p.option_loss[j]
                base += pk * α[j] * p.option_gain[j]
            end
            wcap = min.(wb, c.threshold)
            Φ = c.potentials
            @test all(Φ[((H - 1) * n + 1):(H * n)] .== 0)
            @test c.source_potential >= 0
            ptr, nbr = GT._patrol_moves(n, p.edges)
            ok = all(wcap[i] + Φ[i] <= c.source_potential + 1e-9 for i in 1:n)
            for t in 1:(H - 1), i in 1:n, k in ptr[i]:(ptr[i + 1] - 1)
                hd = (t) * n + nbr[k]
                ok &= wcap[hd] + Φ[hd] <= Φ[(t - 1) * n + i] + 1e-9
            end
            @test ok
            bound = base - sum(max.(wb .- c.threshold, 0.0)) - p.n_units * c.source_potential
            @test bound ≈ c.loss_bound rtol = 1e-9 atol = 1e-9
            @test c.loss_bound ≈ p.lower_bound
            @test p.loss_requirement < c.loss_bound
        end

        below = above = 0
        for seed in 0:19
            _, p = generate_problem(:game_theory, 300, unknown, seed; variant=:patrol_security)
            @test p.feasible_witness === nothing && p.infeasibility_certificate === nothing
            p.loss_requirement < p.lower_bound && (below += 1)
            p.loss_requirement > p.upper_bound && (above += 1)
        end
        @test below > 0 && above > 0

        for status in (feasible, infeasible, unknown)
            Random.seed!(5)
            m1, a = generate_problem(:game_theory, 800, status, 13; variant=:patrol_security)
            Random.seed!(6)
            m2, b = generate_problem(:game_theory, 800, status, 13; variant=:patrol_security)
            @test a.edges == b.edges && a.option_gain == b.option_gain
            @test a.loss_requirement == b.loss_requirement
            @test sprint(print, m1) == sprint(print, m2)
        end

        @testset "HiGHS contracts" begin
            if HAS_HIGHS
                for target in (60, 400, 1500), status in (feasible, infeasible), seed in 0:2
                    m, p = generate_problem(:game_theory, target, status, seed; variant=:patrol_security)
                    set_optimizer(m, HiGHS.Optimizer)
                    set_silent(m)
                    optimize!(m)
                    if status == feasible
                        @test termination_status(m) == MOI.OPTIMAL
                        @test p.lower_bound - 1e-6 <= objective_value(m) <= p.upper_bound + 1e-6
                    else
                        @test termination_status(m) == MOI.INFEASIBLE
                    end
                end
                outcomes = Set{MOI.TerminationStatusCode}()
                for seed in 0:11
                    m, _ = generate_problem(:game_theory, 300, unknown, seed; variant=:patrol_security)
                    set_optimizer(m, HiGHS.Optimizer)
                    set_silent(m)
                    optimize!(m)
                    push!(outcomes, termination_status(m))
                end
                @test MOI.OPTIMAL in outcomes && MOI.INFEASIBLE in outcomes
            end
        end
    end
end
