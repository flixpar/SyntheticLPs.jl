# Local helpers shared by the contracts below: the blend pairs in the exact
# order build_model enumerates them, the campaign task list, and stream
# production under the witness (whose swing displacements are zero).
function _pp_refinery_pairs(p)
    pairs = Tuple{Int,Int}[]
    for (q, product) in enumerate(p.product_names),
        name in SyntheticLPs._RP_BLEND_TABLE[product]
        s = findfirst(==(name), p.stream_names)
        s !== nothing && push!(pairs, (s, q))
    end
    return sort!(pairs)
end

_pp_campaign_tasks(p) =
    [t for t in 1:length(p.task_names) if p.campaign_unit[p.task_unit[t]]]

function _pp_stream_production(p, w, c, s, t)
    produced = s <= 6 ? p.cut_yield[c, s] * w.crude_feed[c, t] : 0.0
    for m in 1:length(p.mode_unit), (out, y) in p.mode_yields[m]
        out == s && (produced += y * w.mode_feed[c, m, t])
    end
    return produced
end

# Focused quality contracts for the process_planning category: registry
# shape, the exact closed-form variable and row counts, sizing accuracy,
# refinery assay and specification invariants, campaign schedule invariants,
# witness and certificate arithmetic, reproducibility, and HiGHS feasibility
# contracts on the LP relaxation and the unrelaxed campaign MIP.
@testset "Process Planning" begin
    @test :process_planning in list_categories()
    @test Set(list_variants(:process_planning)) == Set([:refinery, :campaign])
    info = problem_info(:process_planning)
    @test info[:default_variant] == :refinery
    @test occursin("refiner", lowercase(info[:description]))

    # ------------------------------------------------------------------
    # Refinery variant
    # ------------------------------------------------------------------
    @testset "Refinery" begin
        # Exact variable count, straight from the planted dimensions: crude
        # purchases, crude tank levels and CDU feed, swing-cut displacements,
        # per-mode unit feed by crude origin, assay-origin blend allocations,
        # and product sales and tank levels.
        refinery_variables(p) =
            p.n_periods * (3p.n_crudes + p.n_crudes * length(p.swing_pairs) +
                           p.n_crudes * length(p.mode_unit) +
                           p.n_crudes * length(_pp_refinery_pairs(p)) +
                           2length(p.product_names))
        refinery_rows(p) =
            p.n_periods * (p.n_crudes + p.n_crudes * length(p.stream_names) +
                           length(p.unit_names) + length(p.product_names) +
                           count(!=(0), p.spec_direction) + 2)

        for target in (60, 240, 1500, 9000), status in (feasible, infeasible, unknown)
            m, p = generate_problem(:process_planning, target, status, 3;
                                    variant = :refinery)
            @test num_variables(m) == refinery_variables(p)
            @test num_constraints(
                m; count_variable_in_set_constraints = false) == refinery_rows(p)
            @test p.feasibility_status == status
            @test p.configuration in (:topping_reform, :hydroskimming,
                                      :catalytic, :deep_conversion)
        end

        # The refinery is a pure LP: nothing to relax, no integrality.
        m, p = generate_problem(:process_planning, 600, unknown, 1;
                                variant = :refinery, relax_integer = false)
        @test num_constraints(m, VariableRef, MOI.ZeroOne) == 0

        # Sizing: the horizon is solved from the target so every request
        # lands within 10% across four decades; realised sizes are monotone.
        for target in (60, 200, 1000, 4000, 20000), seed in 0:3
            m, _ = generate_problem(:process_planning, target, unknown, seed;
                                    variant = :refinery)
            @test abs(num_variables(m) - target) <= 0.10 * target
        end
        sizes = [num_variables(generate_problem(:process_planning, target,
                                                unknown, 7;
                                                variant = :refinery)[1])
                 for target in (60, 100, 200, 400, 800, 1600, 3200, 6400,
                                12800, 20000, 60000, 150000)]
        @test issorted(sizes)

        # The documented cap rejects rather than silently under-sizing.
        @test_throws ArgumentError generate_problem(
            :process_planning, SyntheticLPs.MAX_REFINERY_PLANNING_VARIABLES + 1,
            unknown, 0; variant = :refinery)

        # Assay and market conventions: every crude assay sums to one barrel,
        # swing windows straddle zero, contracts are ordered, prices are
        # positive, and every stream carries non-negative qualities.
        for target in (120, 900, 5000), status in (feasible, infeasible, unknown)
            _, p = generate_problem(:process_planning, target, status, 5;
                                    variant = :refinery)
            @test all(isapprox(sum(p.cut_yield[c, :]), 1.0; atol = 1e-9)
                      for c in 1:p.n_crudes)
            @test all(p.cut_yield[c, s] >= 0.015
                      for c in 1:p.n_crudes, s in 1:6)
            @test all(p.swing_lo[c, k] <= 0.0 <= p.swing_hi[c, k]
                      for c in 1:p.n_crudes, k in 1:length(p.swing_pairs))
            @test all(p.purchase_floor[c, t] <= p.purchase_ceiling[c, t]
                      for c in 1:p.n_crudes, t in 1:p.n_periods)
            @test all(p.sales_floor[q, t] <= p.sales_ceiling[q, t]
                      for q in 1:length(p.product_names), t in 1:p.n_periods)
            @test all(p.product_price[q, t] > 0
                      for q in 1:length(p.product_names), t in 1:p.n_periods)
            @test all(p.crude_price[c] > 0 for c in 1:p.n_crudes)
            @test all(p.quality[c, s, a] >= 0
                      for c in 1:p.n_crudes, s in 1:length(p.stream_names),
                          a in 1:SyntheticLPs._RP_N_ATTRS)
            @test length(p.stream_names) == length(unique(p.stream_names))
            @test all(p.initial_product_inventory[q] <= p.product_tank[q]
                      for q in 1:length(p.product_names))
            @test sum(p.initial_crude_inventory) <= p.crude_tank_capacity
            if p.configuration == :deep_conversion
                @test :COKE in p.product_names
            end
            # Static specifications only ever sit in industry bands (the
            # constructor drops a band the planted recipe cannot meet; the
            # seasonal RVP window is checked through the witness instead).
            for q in 1:length(p.product_names), a in 1:SyntheticLPs._RP_N_ATTRS
                d = p.spec_direction[q, a]
                key = (p.product_names[q], a)
                if d == 1 && haskey(SyntheticLPs._RP_SPEC_GE, key)
                    @test p.spec_rhs[q, a, 1] in SyntheticLPs._RP_SPEC_GE[key]
                elseif d == -1 && haskey(SyntheticLPs._RP_SPEC_LE, key)
                    @test p.spec_rhs[q, a, 1] in SyntheticLPs._RP_SPEC_LE[key]
                elseif d != 0
                    @test a == SyntheticLPs._RP_RVP  # only RVP is dynamic
                end
            end
        end

        # The yield-path bound used by the certificate really is an upper
        # bound on the witness: product blend volume never exceeds the
        # best-mode conversion of the crude actually run.
        for seed in 0:3
            _, p = generate_problem(:process_planning, 900, feasible, seed;
                                    variant = :refinery)
            w = p.feasible_witness
            ypath = SyntheticLPs._rp_yield_path(
                p.cut_yield, p.unit_feed, p.mode_unit, p.mode_yields,
                _pp_refinery_pairs(p), length(p.product_names))
            pairs = _pp_refinery_pairs(p)
            for q in 1:length(p.product_names), t in 1:p.n_periods
                blended = sum(w.blend[c, k, t]
                              for c in 1:p.n_crudes, k in 1:length(pairs)
                              if pairs[k][2] == q; init = 0.0)
                run = sum(w.crude_feed[c, t] for c in 1:p.n_crudes)
                @test blended <= maximum(ypath[:, q]) * run + 1e-6
            end
        end

        # Feasible witness: re-verify the planted plan by direct arithmetic:
        # crude and product tank recursions, term/spot bounds, zero swings,
        # stream balances by crude origin, and every quality specification
        # through its blending index.
        for target in (240, 2000), seed in 0:2
            _, p = generate_problem(:process_planning, target, feasible, seed;
                                    variant = :refinery)
            w = p.feasible_witness
            C, T = p.n_crudes, p.n_periods
            P = length(p.product_names)
            pairs = _pp_refinery_pairs(p)
            @test w !== nothing
            @test p.infeasibility_certificate === nothing
            @test p.market_scenario === nothing
            for c in 1:C, t in 1:T
                @test isapprox(w.crude_inventory[c, t],
                               (t == 1 ? p.initial_crude_inventory[c] :
                                w.crude_inventory[c, t - 1]) +
                               w.purchase[c, t] - w.crude_feed[c, t]; atol = 1e-6)
                @test p.purchase_floor[c, t] - 1e-9 <= w.purchase[c, t] <=
                      p.purchase_ceiling[c, t] + 1e-9
            end
            @test all(w.swing[c, k, t] == 0.0
                      for c in 1:C, k in 1:length(p.swing_pairs), t in 1:T)
            for c in 1:C, s in 1:length(p.stream_names), t in 1:T
                consumed = sum(w.blend[c, k, t] for k in 1:length(pairs)
                               if pairs[k][1] == s; init = 0.0) +
                           sum(w.mode_feed[c, m, t]
                               for m in 1:length(p.mode_unit)
                               if p.unit_feed[p.mode_unit[m]] == s;
                               init = 0.0)
                @test isapprox(_pp_stream_production(p, w, c, s, t), consumed;
                               atol = 1e-6)
            end
            for q in 1:P, t in 1:T
                @test isapprox(w.product_inventory[q, t],
                               (t == 1 ? p.initial_product_inventory[q] :
                                w.product_inventory[q, t - 1]) +
                               sum(w.blend[c, k, t] for c in 1:C,
                                   k in 1:length(pairs) if pairs[k][2] == q;
                                   init = 0.0) -
                               w.sales[q, t]; atol = 1e-6)
                @test 0 <= w.product_inventory[q, t] <= p.product_tank[q] + 1e-9
                @test p.sales_floor[q, t] - 1e-9 <= w.sales[q, t] <=
                      p.sales_ceiling[q, t] + 1e-9
            end
            # Quality rows, including the Chevron RVP and Walter viscosity
            # blending indices exactly as build_model writes them.
            for q in 1:P, a in 1:SyntheticLPs._RP_N_ATTRS
                d = p.spec_direction[q, a]
                d == 0 && continue
                for t in 1:T
                    volume = sum(w.blend[c, k, t]
                                 for c in 1:C, k in 1:length(pairs)
                                 if pairs[k][2] == q; init = 0.0)
                    volume <= 0 && continue
                    idx = a == SyntheticLPs._RP_RVP ? x -> x^1.25 :
                          a == SyntheticLPs._RP_VIS ? cbrt : identity
                    quality = sum(idx(p.quality[c, pairs[k][1], a]) *
                                  w.blend[c, k, t]
                                  for c in 1:C, k in 1:length(pairs)
                                  if pairs[k][2] == q; init = 0.0) / volume
                    rhs = idx(p.spec_rhs[q, a, t])
                    d == 1 ? @test(quality >= rhs - 1e-9) :
                            @test(quality <= rhs + 1e-9)
                end
            end
            # JuMP re-checks every row, bound, and integrality set.
            model, _ = generate_problem(:process_planning, target, feasible,
                                        seed; variant = :refinery,
                                        relax_integer = false)
            point = Dict{VariableRef,Float64}()
            for c in 1:C, t in 1:T
                point[model[:purchase][c, t]] = w.purchase[c, t]
                point[model[:crude_inventory][c, t]] = w.crude_inventory[c, t]
                point[model[:crude_feed][c, t]] = w.crude_feed[c, t]
            end
            for c in 1:C, k in 1:length(p.swing_pairs), t in 1:T
                point[model[:swing][c, k, t]] = w.swing[c, k, t]
            end
            for c in 1:C, m in 1:length(p.mode_unit), t in 1:T
                point[model[:mode_feed][c, m, t]] = w.mode_feed[c, m, t]
            end
            for c in 1:C, k in 1:length(pairs), t in 1:T
                point[model[:blend][c, k, t]] = w.blend[c, k, t]
            end
            for q in 1:P, t in 1:T
                point[model[:sales][q, t]] = w.sales[q, t]
                point[model[:product_inventory][q, t]] = w.product_inventory[q, t]
            end
            @test isempty(primal_feasibility_report(model, point; atol = 1e-7))
        end

        # Infeasibility certificate: recompute the cumulative supply bound
        # from the stored fields and confirm the refutation. Both bounds use
        # only linear rows (crude balances, capacity rows, blend balances)
        # and variable bounds, so they refute the model as built.
        for target in (240, 2000), seed in 0:2
            _, p = generate_problem(:process_planning, target, infeasible,
                                    seed; variant = :refinery)
            cert = p.infeasibility_certificate
            @test cert !== nothing
            @test p.feasible_witness === nothing
            q, h = cert.product, cert.horizon
            @test isapprox(cert.demand, sum(p.sales_floor[q, 1:h]); atol = 1e-6)
            @test isapprox(cert.initial_inventory,
                           p.initial_product_inventory[q]; atol = 1e-6)
            ypath = SyntheticLPs._rp_yield_path(
                p.cut_yield, p.unit_feed, p.mode_unit, p.mode_yields,
                _pp_refinery_pairs(p), length(p.product_names))
            @test isapprox(cert.yield_bound, maximum(ypath[:, q]); atol = 1e-9)
            @test isapprox(cert.crude_bound,
                           min(cert.cdu_bound, cert.purchase_bound); atol = 1e-6)
            @test isapprox(cert.upper_bound,
                           cert.initial_inventory +
                           cert.yield_bound * cert.crude_bound; atol = 1e-6)
            @test cert.margin == cert.demand - cert.upper_bound
            @test cert.margin > 0
        end

        # Unknown-status instances carry the market scenario and nothing else.
        for seed in 0:3
            _, p = generate_problem(:process_planning, 700, unknown, seed;
                                    variant = :refinery)
            @test p.feasible_witness === nothing
            @test p.infeasibility_certificate === nothing
            @test p.market_scenario !== nothing
            @test 0.42 <= p.market_scenario.supply_factor <= 0.95
            @test 0.865 <= p.market_scenario.demand_factor <= 1.135
        end

        # Reproducibility and global-RNG isolation.
        Random.seed!(12345)
        _, p1 = generate_problem(:process_planning, 500, unknown, 77;
                                 variant = :refinery)
        Random.seed!(999)
        _, p2 = generate_problem(:process_planning, 500, unknown, 77;
                                 variant = :refinery)
        @test all(isequal(getfield(p1, f), getfield(p2, f))
                  for f in fieldnames(SyntheticLPs.RefineryPlanningProblem))
    end

    # ------------------------------------------------------------------
    # Campaign variant
    # ------------------------------------------------------------------
    @testset "Campaign" begin
        campaign_variables(p) = p.n_periods *
            (length(p.task_names) + 2 * length(_pp_campaign_tasks(p)) +
             count(==(:raw), p.material_kind) * p.n_tiers +
             length(p.material_names) + count(==(:final), p.material_kind))
        campaign_rows(p) =
            p.n_periods * (length(p.material_names) + length(p.unit_names) +
                           count(p.campaign_unit) +
                           4 * length(_pp_campaign_tasks(p))) -
            length(_pp_campaign_tasks(p)) * (p.campaign_length - 1)

        for target in (60, 240, 1500, 9000), status in (feasible, infeasible, unknown)
            m, p = generate_problem(:process_planning, target, status, 3;
                                    variant = :campaign)
            @test num_variables(m) == campaign_variables(p)
            @test num_constraints(
                m; count_variable_in_set_constraints = false) == campaign_rows(p)
            @test p.feasibility_status == status
            @test 2 <= p.campaign_length <= 3
        end

        # Unrelaxed model: the binary block is exactly the campaign selectors.
        m, p = generate_problem(:process_planning, 600, unknown, 1;
                                variant = :campaign, relax_integer = false)
        @test num_constraints(m, VariableRef, MOI.ZeroOne) ==
              length(_pp_campaign_tasks(p)) * p.n_periods

        for target in (60, 200, 1000, 4000, 15000), seed in 0:3
            m, _ = generate_problem(:process_planning, target, unknown, seed;
                                    variant = :campaign)
            @test abs(num_variables(m) - target) <= 0.10 * target
        end
        sizes = [num_variables(generate_problem(:process_planning, target,
                                                unknown, 7;
                                                variant = :campaign)[1])
                 for target in (60, 100, 200, 400, 800, 1600, 3200, 6400,
                                12800, 18000)]
        @test issorted(sizes)

        @test_throws ArgumentError generate_problem(
            :process_planning, SyntheticLPs.MAX_CAMPAIGN_PLANNING_VARIABLES + 1,
            unknown, 0; variant = :campaign)

        # Chain-library conventions: stoichiometric coefficients positive,
        # task output yields at most one (conversion losses), every unit
        # hosts at least one task, and campaign trains host several.
        for target in (120, 900, 5000), status in (feasible, infeasible, unknown)
            _, p = generate_problem(:process_planning, target, status, 5;
                                    variant = :campaign)
            for t in 1:length(p.task_names)
                @test all(c > 0 for (_, c) in p.task_inputs[t])
                @test all(c > 0 for (_, c) in p.task_outputs[t])
                # Oxidation steps (cumene to phenol and acetone) gain mass
                # from unmodeled air, so output mass may exceed one.
                @test sum(c for (_, c) in p.task_outputs[t]) <= 1.6
            end
            @test all(any(==(u), p.task_unit) for u in 1:length(p.unit_names))
            @test all((count(==(u), p.task_unit) > 1) == p.campaign_unit[u]
                      for u in 1:length(p.unit_names))
            # Marginal purchase prices increase across tiers (convex
            # piecewise-linear cost) and tier caps are non-negative.
            for m in 1:length(p.material_names)
                p.material_kind[m] == :raw || continue
                @test issorted(vec(p.tier_price[m, 1:p.n_tiers]))
                @test p.tier_price[m, end] > p.tier_price[m, 1]
                @test all(p.tier_cap[m, j] >= 0 for j in 1:p.n_tiers)
            end
            @test all(p.tank[m] >= p.initial_inventory[m]
                      for m in 1:length(p.material_names))
            @test all(p.sales_floor[m, t] <= p.sales_ceiling[m, t]
                      for m in 1:length(p.material_names),
                          t in 1:p.n_periods if p.material_kind[m] == :final)
            # Every task input and output is a real material index.
            @test all(1 <= mm <= length(p.material_names)
                      for t in 1:length(p.task_names)
                      for io in (p.task_inputs[t], p.task_outputs[t])
                      for (mm, _) in io)
        end

        # Feasible witness: campaign blocks are binary and exclusive, block
        # lengths respect the minimum, rates respect capacity and turndown,
        # material balances close by arithmetic, and purchases stay in tier.
        for target in (240, 2000), seed in 0:2
            _, p = generate_problem(:process_planning, target, feasible, seed;
                                    variant = :campaign)
            w = p.feasible_witness
            @test w !== nothing
            @test p.infeasibility_certificate === nothing
            @test p.market_scenario === nothing
            NT, T = length(p.task_names), p.n_periods
            cts = _pp_campaign_tasks(p)
            ci = Dict(t => i for (i, t) in enumerate(cts))
            @test all(w.active[i, t] in (0.0, 1.0)
                      for i in 1:length(cts), t in 1:T)
            for u in 1:length(p.unit_names), t in 1:T
                p.campaign_unit[u] || continue
                @test sum(w.active[ci[τ], t] for τ in cts
                          if p.task_unit[τ] == u; init = 0.0) <= 1
            end
            for (i, t) in enumerate(cts), τ in 1:T
                cap = p.unit_capacity[p.task_unit[t], τ]
                @test w.rate[t, τ] <= cap * w.active[i, τ] + 1e-6
                @test w.rate[t, τ] >=
                      p.min_rate_fraction[i] * cap * w.active[i, τ] - 1e-6
                expected_start = τ == 1 ? w.active[i, 1] :
                                 max(0.0, w.active[i, τ] - w.active[i, τ - 1])
                @test isapprox(w.starts[i, τ], expected_start; atol = 1e-9)
            end
            L = p.campaign_length
            for (i, _) in enumerate(cts), τ in 1:(T - L + 1)
                @test sum(w.active[i, k] for k in τ:(τ + L - 1)) >=
                      L * (w.active[i, τ] - (τ == 1 ? 0 : w.active[i, τ - 1])) -
                      1e-9
            end
            for u in 1:length(p.unit_names), τ in 1:T
                @test sum(w.rate[t, τ] for t in 1:NT if p.task_unit[t] == u;
                          init = 0.0) <=
                      p.unit_capacity[u, τ] + 1e-6
            end
            for m in 1:length(p.material_names), τ in 1:T
                produced = 0.0
                consumed = 0.0
                for t in 1:NT, (mm, c) in p.task_outputs[t]
                    mm == m && (produced += c * w.rate[t, τ])
                end
                for t in 1:NT, (mm, c) in p.task_inputs[t]
                    mm == m && (consumed += c * w.rate[t, τ])
                end
                bought = p.material_kind[m] == :raw ?
                         sum(w.purchase[m, :, τ]) : 0.0
                sold = p.material_kind[m] == :final ? w.sales[m, τ] : 0.0
                @test isapprox(w.inventory[m, τ],
                               (τ == 1 ? p.initial_inventory[m] :
                                w.inventory[m, τ - 1]) + produced + bought -
                               consumed - sold; atol = 1e-6)
                @test 0 <= w.inventory[m, τ] <= p.tank[m] + 1e-9
                if p.material_kind[m] == :final
                    @test p.sales_floor[m, τ] - 1e-9 <= w.sales[m, τ] <=
                          p.sales_ceiling[m, τ] + 1e-9
                end
            end
            @test all(0 <= w.purchase[m, j, τ] <= p.tier_cap[m, j] + 1e-9
                      for m in 1:length(p.material_names), j in 1:p.n_tiers,
                          τ in 1:T)
            # JuMP re-checks every row, bound, and integrality set of the
            # unrelaxed MIP at the witness.
            model, _ = generate_problem(:process_planning, target, feasible,
                                        seed; variant = :campaign,
                                        relax_integer = false)
            finals = [m for m in 1:length(p.material_names)
                      if p.material_kind[m] == :final]
            raws = [m for m in 1:length(p.material_names)
                    if p.material_kind[m] == :raw]
            point = Dict{VariableRef,Float64}()
            for t in 1:NT, τ in 1:T
                point[model[:rate][t, τ]] = w.rate[t, τ]
            end
            for t in cts, τ in 1:T
                point[model[:active][ci[t], τ]] = w.active[ci[t], τ]
                point[model[:starts][ci[t], τ]] = w.starts[ci[t], τ]
            end
            for m in raws, j in 1:p.n_tiers, τ in 1:T
                point[model[:purchase][m, j, τ]] = w.purchase[m, j, τ]
            end
            for m in 1:length(p.material_names), τ in 1:T
                point[model[:inventory][m, τ]] = w.inventory[m, τ]
            end
            for m in finals, τ in 1:T
                point[model[:sales][m, τ]] = w.sales[m, τ]
            end
            @test isempty(primal_feasibility_report(model, point; atol = 1e-7))
        end

        # Infeasibility certificate: the producing-task capacity bound and
        # the raw-supply bound are recomputed from stored fields. The task
        # bound uses `rate <= capacity * active <= capacity`, valid with the
        # selectors relaxed to [0, 1], so it refutes the LP relaxation too.
        for target in (240, 2000), seed in 0:2
            _, p = generate_problem(:process_planning, target, infeasible,
                                    seed; variant = :campaign)
            cert = p.infeasibility_certificate
            @test cert !== nothing
            @test p.feasible_witness === nothing
            m, h = cert.material, cert.horizon
            @test isapprox(cert.demand, sum(p.sales_floor[m, 1:h]); atol = 1e-6)
            @test isapprox(cert.initial_inventory,
                           p.initial_inventory[m]; atol = 1e-6)
            @test isapprox(cert.upper_bound,
                           cert.initial_inventory +
                           min(cert.task_bound, cert.raw_bound); atol = 1e-6)
            @test cert.margin == cert.demand - cert.upper_bound > 0
        end

        for seed in 0:3
            _, p = generate_problem(:process_planning, 700, unknown, seed;
                                    variant = :campaign)
            @test p.feasible_witness === nothing
            @test p.infeasibility_certificate === nothing
            @test p.market_scenario !== nothing
            @test 0.55 <= p.market_scenario.supply_factor <= 0.89
            @test 0.875 <= p.market_scenario.demand_factor <= 1.125
        end

        Random.seed!(12345)
        _, p1 = generate_problem(:process_planning, 500, unknown, 77;
                                 variant = :campaign)
        Random.seed!(999)
        _, p2 = generate_problem(:process_planning, 500, unknown, 77;
                                 variant = :campaign)
        @test all(isequal(getfield(p1, f), getfield(p2, f))
                  for f in fieldnames(SyntheticLPs.CampaignPlanningProblem))
    end

    # ------------------------------------------------------------------
    # Solver-backed feasibility contracts (HiGHS)
    # ------------------------------------------------------------------
    if HAS_HIGHS
        @testset "HiGHS Contracts" begin
            # Both variants honour their statuses on the default-relaxed
            # model. The refinery is a pure LP; infeasibility proofs at very
            # large sizes can exceed a short time limit, so contracts are
            # checked at moderate sizes where proofs are fast.
            for variant in (:refinery, :campaign),
                target in (100, 400, 2000), seed in 1:4

                m, _ = generate_problem(:process_planning, target, feasible,
                                        seed; variant = variant)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == MOI.OPTIMAL

                m, _ = generate_problem(:process_planning, target, infeasible,
                                        seed; variant = variant)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                set_time_limit_sec(m, 60.0)
                optimize!(m)
                @test termination_status(m) == MOI.INFEASIBLE
            end

            # The campaign witness must survive integrality: the unrelaxed
            # MIP is feasible whenever a feasible instance was requested.
            for target in (100, 400), seed in 1:3
                m, _ = generate_problem(:process_planning, target, feasible,
                                        seed; variant = :campaign,
                                        relax_integer = false)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                set_time_limit_sec(m, 90.0)
                optimize!(m)
                @test termination_status(m) == MOI.OPTIMAL
            end

            # Unknown-status instances are a genuine mix at every size: the
            # golden-ratio band straddles the feasibility flip.
            for variant in (:refinery, :campaign), target in (150, 1000)
                optimal = 0
                infeasible_count = 0
                for seed in 1:20
                    m, _ = generate_problem(:process_planning, target, unknown,
                                            seed; variant = variant)
                    set_optimizer(m, HiGHS.Optimizer)
                    set_silent(m)
                    set_time_limit_sec(m, 30.0)
                    optimize!(m)
                    if termination_status(m) == MOI.OPTIMAL
                        optimal += 1
                    elseif termination_status(m) == MOI.INFEASIBLE
                        infeasible_count += 1
                    end
                end
                @test optimal > 0
                @test infeasible_count > 0
            end
        end
    end
end
