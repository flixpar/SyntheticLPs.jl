# Focused quality contracts for the markov_decision_process category: registry
# shape, exact state-action sizing formulas and row counts, stochastic-kernel
# invariants, domain data invariants, occupation-measure witness arithmetic
# against every LP row and value-function Farkas certificate arithmetic against
# every column (both without a solver), reproducibility, and HiGHS contracts
# (including LP optimum == policy-iteration optimum on unconstrained instances).
@testset "Markov Decision Process" begin
    SL = SyntheticLPs
    variants = [:inventory_control, :queueing_control, :machine_maintenance, :constrained]
    @test :markov_decision_process in list_categories()
    @test Set(list_variants(:markov_decision_process)) == Set(variants)
    info = problem_info(:markov_decision_process)
    @test info[:default_variant] == :inventory_control
    @test occursin("markov", lowercase(info[:description]))

    gen(v, t, st, seed) = generate_problem(:markov_decision_process, t, st, seed; variant=v)
    nrows(m) = num_constraints(m; count_variable_in_set_constraints=false)
    γof(p) = p.criterion == :discounted ? p.discount : 1.0

    # Exact pair count from the variant's own sizing formula.
    function expected_pairs(p)
        if p isa SL.InventoryControlMDP
            return SL._inventory_mdp_pairs(
                p.lead_time,
                p.max_inventory,
                p.max_backlog,
                p.max_order,
                length(p.price_multipliers),
                p.n_phases,
            )
        elseif p isa SL.QueueingControlMDP
            return SL._queueing_mdp_pairs(p.buffer1, p.buffer2, length(p.rates1), length(p.rates2))
        elseif p isa SL.MachineMaintenanceMDP
            return SL._maintenance_mdp_pairs(
                p.n_conditions, p.max_spares, p.max_order, length(p.speeds), p.n_phases
            )
        else
            return expected_pairs(p.base)
        end
    end
    function expected_states(p)
        if p isa SL.InventoryControlMDP
            return p.n_phases *
                   length(SL._inventory_mdp_phase_states(p.lead_time, p.max_inventory, p.max_backlog, p.max_order))
        elseif p isa SL.QueueingControlMDP
            return (p.buffer1 + 1) * (p.buffer2 + 1)
        elseif p isa SL.MachineMaintenanceMDP
            return p.n_phases * (p.n_conditions + 1) * (p.max_spares + 1)
        else
            return expected_states(p.base)
        end
    end

    # Balance / normalization residuals and budget-row values of a per-pair
    # point, computed from the struct fields only (no JuMP, no solver).
    function lp_residuals(p, x)
        m = p.mdp
        γ = γof(p)
        lhs = zeros(m.n_states)
        for s in 1:(m.n_states), k in m.state_ptr[s]:(m.state_ptr[s + 1] - 1)
            lhs[s] += x[k]
            for t in m.trans_ptr[k]:(m.trans_ptr[k + 1] - 1)
                lhs[m.trans_next[t]] -= γ * m.trans_prob[t] * x[k]
            end
        end
        balance = maximum(abs.(lhs .- p.rhs))
        norm_res = p.criterion == :average ? abs(sum(x) - p.normalization) : 0.0
        budget_vals = [sum(m.streams[j] .* x) for j in p.budget_streams]
        return balance, norm_res, budget_vals
    end

    # --- sizing -----------------------------------------------------------------
    for v in variants, target in (60, 1000, 5000), status in (feasible, infeasible, unknown), seed in 0:1
        model, p = gen(v, target, status, seed)
        n = num_variables(model)
        @test n == length(p.mdp.cost) == expected_pairs(p)
        @test p.mdp.n_states == expected_states(p)
        @test nrows(model) ==
              p.mdp.n_states + (p.criterion == :average ? 1 : 0) + length(p.budgets)
        @test abs(n - target) <= 0.1 * target || target < 100 && abs(n - target) <= 0.35 * target
        # Rows grow with the model: at most ~25 columns per balance row.
        target >= 1000 && @test n <= 25 * p.mdp.n_states
        @test p.feasibility_status == status
    end

    # Sizing cap: rejected above MDP_MAX_PAIRS before any data is built.
    @test SL.MDP_MAX_PAIRS == 1_000_000
    for T in (SL.InventoryControlMDP, SL.QueueingControlMDP, SL.MachineMaintenanceMDP, SL.ConstrainedMDP)
        @test_throws ArgumentError T(SL.MDP_MAX_PAIRS + 1, unknown, 0)
        @test_throws ArgumentError T(0, unknown, 0)
    end
    # Large targets still land near the request (constructor only, no model).
    for T in (SL.QueueingControlMDP, SL.MachineMaintenanceMDP)
        p = T(100_000, unknown, 3)
        @test abs(length(p.mdp.cost) - 100_000) <= 0.05 * 100_000
    end

    # --- kernel and data invariants ----------------------------------------------
    for v in variants, status in (feasible, infeasible, unknown), seed in 0:2
        _, p = gen(v, 700, status, seed)
        m = p.mdp
        npairs = length(m.cost)
        @test m.state_ptr[1] == 1 && m.state_ptr[end] == npairs + 1
        @test all(m.state_ptr[s + 1] > m.state_ptr[s] for s in 1:(m.n_states))
        @test length(m.trans_ptr) == npairs + 1
        rowsum_ok = true
        distinct_ok = true
        for k in 1:npairs
            span = m.trans_ptr[k]:(m.trans_ptr[k + 1] - 1)
            rowsum_ok &= abs(sum(m.trans_prob[span]) - 1.0) <= 1e-12
            distinct_ok &= allunique(m.trans_next[span])
        end
        @test rowsum_ok                                   # rows of P sum to 1
        @test distinct_ok
        @test all(>(0.0), m.trans_prob)
        @test all(1 .<= m.trans_next .<= m.n_states)
        @test length(m.streams) == length(m.stream_names) == 3
        @test all(all(>=(0.0), st) for st in m.streams)   # budget metrics are nonnegative
        @test all(m.state_ptr[s] <= m.reference_policy[s] < m.state_ptr[s + 1] for s in 1:(m.n_states))
        @test p.criterion in (:discounted, :average)
        if p.criterion == :discounted
            @test 0.9 < p.discount < 1.0
            @test all(>(0.0), p.rhs)                     # full-support state relevance
            @test sum(p.rhs) ≈ (1 - p.discount) * p.normalization rtol = 1e-9
        else
            @test p.discount == 1.0
            @test all(iszero, p.rhs)
        end
        @test p.normalization == m.n_states
        @test length(p.budgets) == length(p.budget_streams)
        if v == :constrained
            @test p.base_model in SL.CONSTRAINED_MDP_BASES
            @test 1 in p.budget_streams && 2 <= length(p.budget_streams) <= 3
            @test p.mdp === p.base.mdp
            @test isempty(p.base.budget_streams)
        else
            @test p.budget_streams in ([1], Int[])
            status == infeasible && @test p.budget_streams == [1]
        end
    end

    # Domain-specific invariants.
    for seed in 0:3
        _, p = gen(:inventory_control, 3000, unknown, seed)
        @test p.lead_time in 0:2
        @test length(p.price_multipliers) in 1:3
        @test all(p.state_i[s] + sum(p.state_pipeline[s]; init=0) <= p.max_inventory for s in 1:(p.mdp.n_states))
        @test all(p.state_i .>= -p.max_backlog)
        qs = [div(l, 100) for l in p.mdp.action_label]
        @test all(0 .<= qs .<= p.max_order)
        @test p.mdp.stream_names == [:shortage, :on_hand, :orders]

        _, q = gen(:queueing_control, 3000, unknown, seed)
        @test q.arrival_rate / min(maximum(q.rates1), maximum(q.rates2)) >= 1.03 - 1e-12   # overloaded peak
        @test issorted(q.energy1) && issorted(q.energy2)
        # No two pairs of a state share a transition law (no parallel columns).
        m = q.mdp
        laws_distinct = true
        for s in 1:(m.n_states)
            laws = [
                Dict(zip(m.trans_next[m.trans_ptr[k]:(m.trans_ptr[k + 1] - 1)], m.trans_prob[m.trans_ptr[k]:(m.trans_ptr[k + 1] - 1)]))
                for k in m.state_ptr[s]:(m.state_ptr[s + 1] - 1)
            ]
            laws_distinct &= allunique(laws)
        end
        @test laws_distinct

        _, mm = gen(:machine_maintenance, 3000, unknown, seed)
        V = length(mm.speeds)
        C, K = mm.n_conditions, mm.max_spares
        codes = [div(l, 100) for l in mm.mdp.action_label]
        # Failed states offer only corrective / emergency / wait actions.
        for e in 1:(mm.n_phases), k in 0:K
            s = (e - 1) * (C + 1) * (K + 1) + C * (K + 1) + k + 1
            cs = codes[mm.mdp.state_ptr[s]:(mm.mdp.state_ptr[s + 1] - 1)]
            @test all(c -> c > V + 2, cs)
        end
        @test all(0 .<= mm.downtime .<= 1)
    end

    # --- witness arithmetic -------------------------------------------------------
    for v in variants, target in (300, 2000), seed in 0:2
        model, p = gen(v, target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing
        @test p.infeasibility_certificate === nothing
        x = w.occupation
        @test length(x) == length(p.mdp.cost)
        @test all(>=(0.0), x)
        @test sum(w.mix) ≈ 1.0
        @test length(w.policies) == length(w.mix)
        balance, norm_res, vals = lp_residuals(p, x)
        @test balance <= 1e-8 * p.normalization            # every balance row
        @test norm_res <= 1e-8 * p.normalization           # average-cost normalization
        @test sum(x) ≈ p.normalization rtol = 1e-8          # implied by the rows
        @test all(vals .< p.budgets)                        # every budget row, strictly
        if target == 300
            report = primal_feasibility_report(model, Dict(model[:x][k] => x[k] for k in eachindex(x)); atol=1e-7)
            @test isempty(report)
        end
    end

    # --- certificate arithmetic -------------------------------------------------
    for v in variants, target in (300, 2000), seed in 0:2
        _, p = gen(v, target, infeasible, seed)
        c = p.infeasibility_certificate
        @test c !== nothing
        @test p.feasible_witness === nothing
        m = p.mdp
        γ = γof(p)
        @test length(c.weights) == length(p.budget_streams)
        @test all(>=(0.0), c.weights) && sum(c.weights) > 0
        dw = SL._mdp_combined_stream(m, p.budget_streams, c.weights)
        # Dual feasibility of the potential at EVERY state-action pair.
        worst = -Inf
        for s in 1:(m.n_states), k in m.state_ptr[s]:(m.state_ptr[s + 1] - 1)
            ev = sum(m.trans_prob[t] * c.potential[m.trans_next[t]] for t in m.trans_ptr[k]:(m.trans_ptr[k + 1] - 1))
            worst = max(worst, c.gain + c.potential[s] - γ * ev - dw[k])
        end
        @test worst <= 0.0
        lb = p.criterion == :discounted ? sum(p.rhs .* c.potential) : p.normalization * c.gain
        @test lb ≈ c.lower_bound rtol = 1e-9
        @test c.weighted_budget ≈ sum(c.weights .* p.budgets) rtol = 1e-12
        @test c.weighted_budget <= (1 - 0.08) * c.lower_bound + 1e-12   # planted margin
        @test c.lower_bound > 0
        # Algebraic Farkas check on the witness-free side: any point satisfying
        # the balance rows has weighted budget value >= lower bound, so the
        # reference policy's own occupation measure must respect it.
        xref = SL._mdp_occupation(m, m.reference_policy, p.criterion, p.discount, p.rhs, p.normalization)
        @test sum(dw .* xref) >= c.lower_bound * (1 - 1e-9)
    end

    # Joint (not single-row) infeasibility for the constrained variant: in most
    # instances every budget individually exceeds its own stream's optimum.
    joint = 0
    for seed in 0:5
        _, p = gen(:constrained, 800, infeasible, seed)
        m = p.mdp
        own = [
            SL._mdp_lower_bound(m, m.streams[j], p.criterion, p.discount, p.rhs, p.normalization)[3]
            for j in p.budget_streams
        ]
        joint += all(p.budgets .> own)
    end
    @test joint >= 4

    # Unknown: no witness / certificate.
    for v in variants, seed in 0:2
        _, p = gen(v, 400, unknown, seed)
        @test p.feasible_witness === nothing
        @test p.infeasibility_certificate === nothing
    end

    # --- reproducibility (and isolation from the global RNG) ----------------------
    for v in variants, status in (feasible, infeasible, unknown)
        Random.seed!(1)
        _, p1 = gen(v, 500, status, 42)
        Random.seed!(999)
        _, p2 = gen(v, 500, status, 42)
        m1, m2 = p1.mdp, p2.mdp
        @test m1.state_ptr == m2.state_ptr
        @test m1.trans_next == m2.trans_next
        @test m1.trans_prob == m2.trans_prob
        @test m1.cost == m2.cost
        @test m1.streams == m2.streams
        @test m1.reference_policy == m2.reference_policy
        @test p1.rhs == p2.rhs && p1.budgets == p2.budgets && p1.budget_streams == p2.budget_streams
        @test p1.criterion == p2.criterion && p1.discount == p2.discount
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            # HiGHS's default dual simplex sometimes cannot finish the
            # infeasibility proof on these ill-conditioned MDP bases (it reports
            # "possibly dual unbounded" and then OTHER_ERROR), so infeasibility
            # is confirmed with the interior-point solver; feasible instances
            # use the default.
            ipm = optimizer_with_attributes(HiGHS.Optimizer, "solver" => "ipm")
            function solve_status(model, opt)
                set_optimizer(model, opt)
                set_silent(model)
                optimize!(model)
                return termination_status(model)
            end
            for v in variants, target in (200, 1000), seed in 0:2
                mf, _ = gen(v, target, feasible, seed)
                @test solve_status(mf, HiGHS.Optimizer) == MOI.OPTIMAL
                mi, _ = gen(v, target, infeasible, seed)
                @test solve_status(mi, ipm) == MOI.INFEASIBLE
            end

            # Framework verify-and-retry path.
            for v in variants
                m, _ = generate_problem(:markov_decision_process, 300, feasible, 0; variant=v, optimizer=HiGHS.Optimizer)
                @test num_variables(m) > 0
                m, _ = generate_problem(:markov_decision_process, 300, infeasible, 0; variant=v, optimizer=ipm)
                @test num_variables(m) > 0
            end

            # LP optimum of an unconstrained instance equals the policy-iteration
            # optimum (the occupation-measure LP is exact), and never exceeds the
            # reference policy's cost.
            checked = 0
            for v in (:inventory_control, :queueing_control, :machine_maintenance), seed in 0:7
                model, p = gen(v, 600, unknown, seed)
                isempty(p.budget_streams) || continue
                @test solve_status(model, HiGHS.Optimizer) == MOI.OPTIMAL
                m = p.mdp
                lb = SL._mdp_lower_bound(m, m.cost, p.criterion, p.discount, p.rhs, p.normalization)[3]
                xref = SL._mdp_occupation(m, m.reference_policy, p.criterion, p.discount, p.rhs, p.normalization)
                opt = objective_value(model)
                scale = sum(abs.(m.cost .* xref)) + 1.0
                @test opt >= lb - 1e-6 * scale
                tol = p.criterion == :discounted ? 1e-6 : 1e-3
                @test opt <= lb + tol * scale
                @test opt <= sum(m.cost .* xref) + 1e-6 * scale
                checked += 1
            end
            @test checked >= 4

            # `unknown` is genuinely two-sided for the budgeted variants. Budgets
            # drawn close to the boundary can stall one algorithm, so an
            # inconclusive IPM run is retried with primal simplex.
            primal = optimizer_with_attributes(HiGHS.Optimizer, "simplex_strategy" => 4)
            for v in (:queueing_control, :constrained)
                outcomes = Set{Any}()
                for seed in 0:15
                    model, p = gen(v, 300, unknown, seed)
                    isempty(p.budget_streams) && continue
                    ts = solve_status(model, ipm)
                    ts in (MOI.OPTIMAL, MOI.INFEASIBLE) || (ts = solve_status(model, primal))
                    push!(outcomes, ts)
                end
                @test MOI.OPTIMAL in outcomes
                @test MOI.INFEASIBLE in outcomes
            end
        else
            @info "HiGHS not available; skipping markov_decision_process solver contracts"
        end
    end
end
