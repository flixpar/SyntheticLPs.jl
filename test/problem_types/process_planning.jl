# Focused quality contracts for the process_planning category: registry shape,
# the exact variable-count formulas, sizing accuracy, flowsheet and process-network
# invariants, witness and certificate arithmetic, model structure (which variants
# carry binaries), and HiGHS-backed feasibility contracts.
@testset "Process Planning" begin
    PP = SyntheticLPs

    @test :process_planning in list_categories()
    @test Set(list_variants(:process_planning)) ==
        Set([:refinery, :mode_switching, :hydrogen_network, :campaign, :capacity_expansion])
    info = problem_info(:process_planning)
    @test info[:default_variant] == :refinery
    @test occursin("refinery", lowercase(info[:description]))

    # Exact variable counts, straight from the stored flowsheet: crude purchase,
    # run and tank per crude; a feed flow per (unit, admissible stream) plus the
    # unit's throughput; a blend flow per (grade, component); intermediate tanks;
    # blendstock purchases; spot sales; and product sales and tanks.
    refinery_variables(p) =
        p.data.n_periods * (
            3 * p.flowsheet.n_crudes +
            sum(length(u.feeds) + 1 for u in p.flowsheet.units; init=0) +
            sum(length(g.components) for g in p.flowsheet.products; init=0) +
            length(p.flowsheet.storable) +
            length(p.flowsheet.purchasable) +
            length(p.flowsheet.spot) +
            2 * length(p.flowsheet.products)
        )

    # The same, with feeds replicated per operating mode and each mode carrying a
    # throughput, a run indicator and a changeover.
    mode_variables(p) =
        p.data.n_periods * (
            3 * p.flowsheet.n_crudes +
            sum(length(u.modes) * (length(u.feeds) + 3) for u in p.flowsheet.units; init=0) +
            sum(length(g.components) for g in p.flowsheet.products; init=0) +
            length(p.flowsheet.storable) +
            length(p.flowsheet.purchasable) +
            length(p.flowsheet.spot) +
            2 * length(p.flowsheet.products)
        )

    expansion_variables(p) =
        p.n_periods *
        (4 * length(p.technologies) + length(p.raw_chemicals) + length(p.sellable_chemicals))

    hydrogen_variables(p) = refinery_variables(p) + 6 * p.data.n_periods

    campaign_tasks(p) = [t for t in eachindex(p.task_names) if p.campaign_unit[p.task_unit[t]]]

    campaign_variables(p) =
        p.n_periods * (
            length(p.task_names) +
            2 * length(campaign_tasks(p)) +
            count(==(:raw), p.material_kind) * p.n_tiers +
            length(p.material_names) +
            count(==(:final), p.material_kind)
        )

    # Rows: a balance per material; an exclusivity row per campaign train; per
    # campaign task a capacity gate and a turndown row every period, the start
    # definition (one row in period 1, three after it) and a minimum-run row per
    # admissible start; a pool row per regional feedstock pool with more than one
    # purchase variable and a ceiling row per multi-site market. Single-task unit
    # capacity and late-start prohibitions are variable bounds, not rows.
    function campaign_rows(p)
        T, L = p.n_periods, p.campaign_length
        pools = count(g -> length(g) * p.n_tiers >= 2, p.supply_groups)
        per_task = 2T + 1 + 3 * (T - 1) + (T - L + 1)
        return T * (
            length(p.material_names) + count(p.campaign_unit) + pools + length(p.market_groups)
        ) + length(campaign_tasks(p)) * per_task
    end

    @testset "Variable counts and sizing" begin
        for target in (60, 240, 1500, 9000), status in (feasible, infeasible, unknown)
            m, p = generate_problem(:process_planning, target, status, 3)
            @test num_variables(m) == refinery_variables(p)
            @test p.feasibility_status == status

            m, p = generate_problem(:process_planning, target, status, 3; variant=:mode_switching)
            @test num_variables(m) == mode_variables(p)
            @test p.feasibility_status == status

            m, p = generate_problem(:process_planning, target, status, 3; variant=:hydrogen_network)
            @test num_variables(m) == hydrogen_variables(p)
            @test p.feasibility_status == status

            m, p = generate_problem(:process_planning, target, status, 3; variant=:campaign)
            @test num_variables(m) == campaign_variables(p)
            @test num_constraints(m; count_variable_in_set_constraints=false) == campaign_rows(p)
            @test PP.campaign_row_count(p) == campaign_rows(p)
            @test p.feasibility_status == status

            m, p = generate_problem(
                :process_planning, target, status, 3; variant=:capacity_expansion
            )
            @test num_variables(m) == expansion_variables(p)
            @test p.feasibility_status == status
        end

        # Every request lands within 20% of the target across four decades, and
        # realised sizes are monotone in the target.
        for variant in
            (:refinery, :mode_switching, :hydrogen_network, :campaign, :capacity_expansion)
            for target in (50, 120, 500, 2000, 20000),
                status in (feasible, infeasible, unknown),
                seed in 0:1

                m, _ = generate_problem(:process_planning, target, status, seed; variant=variant)
                @test abs(num_variables(m) - target) <= 0.20 * target
            end
            sizes = [
                num_variables(
                    generate_problem(:process_planning, target, unknown, 7; variant=variant)[1]
                ) for target in (50, 100, 200, 400, 800, 1600, 3200, 6400, 12800)
            ]
            @test issorted(sizes)
        end

        # The variants scale to 100k variables within 10% (the campaign as a
        # multi-site company whose rows grow with it), and the campaign's
        # documented cap is enforced before anything is allocated.
        for variant in (:refinery, :campaign, :capacity_expansion)
            m, _ = generate_problem(:process_planning, 100_000, feasible, 1; variant=variant)
            @test abs(num_variables(m) - 100_000) <= 0.10 * 100_000
            @test num_constraints(m; count_variable_in_set_constraints=false) >=
                0.1 * num_variables(m)
        end
        m, p = generate_problem(:process_planning, 100_000, unknown, 2; variant=:campaign)
        @test p.n_sites > 1
        @test num_constraints(m; count_variable_in_set_constraints=false) == campaign_rows(p)
        @test PP.MAX_CAMPAIGN_PLANNING_VARIABLES == 1_000_000
        @test_throws ArgumentError generate_problem(
            :process_planning, PP.MAX_CAMPAIGN_PLANNING_VARIABLES + 1, unknown, 0; variant=:campaign
        )

        # Complexity tracks the scale of the request. A topping refinery has the
        # smallest per-period block, so left to size error alone it wins ties at
        # every target and a large request comes back as a bare
        # crude-cut-and-blend LP; `_pp_level_floor` states the complexity the
        # target deserves instead.
        for variant in (:refinery, :mode_switching, :hydrogen_network)
            for target in (200, 500, 2000, 20000), seed in 0:9
                _, p = generate_problem(:process_planning, target, unknown, seed; variant=variant)
                @test SyntheticLPs.n_units(p.flowsheet) >= 3
            end
            # Above the cracking floor a conversion unit that cracks heavy feed
            # is always present, not merely some hydrotreater.
            for target in (900, 5000), seed in 0:9
                _, p = generate_problem(:process_planning, target, unknown, seed; variant=variant)
                keys = [u.key for u in p.flowsheet.units]
                @test any(k -> k in (:fcc, :hydrocracker, :coker), keys)
            end
        end

        # The horizon is genuinely multi-period wherever the target has room.
        for target in (200, 1000, 8000), seed in 0:2
            _, p = generate_problem(:process_planning, target, unknown, seed)
            @test p.data.n_periods >= 2
            _, p = generate_problem(
                :process_planning, target, unknown, seed; variant=:capacity_expansion
            )
            @test p.n_periods >= 3
            _, p = generate_problem(
                :process_planning, target, unknown, seed; variant=:hydrogen_network
            )
            @test p.data.n_periods >= 2
            _, p = generate_problem(:process_planning, target, unknown, seed; variant=:campaign)
            @test p.n_periods >= 3
        end
    end

    @testset "Flowsheet invariants" begin
        for variant in (:refinery, :mode_switching, :hydrogen_network),
            target in (120, 900, 6000),
            status in (feasible, infeasible, unknown)

            _, p = generate_problem(:process_planning, target, status, 5; variant=variant)
            fs = p.flowsheet
            S = length(fs.stream_classes)

            # Assays are complete distributions over the instance's cut slate.
            @test size(fs.cut_yields) == (fs.n_crudes, length(fs.cut_classes))
            @test all(isapprox(sum(fs.cut_yields[c, :]), 1.0; atol=1e-9) for c in 1:fs.n_crudes)
            @test all(>(0.0), fs.cut_yields)
            @test all(>(0.0), fs.crude_sulfur)
            @test all(>(0.0), fs.crude_api)

            # Stream properties are physical.
            @test size(fs.qualities) == (S, PP.PP_N_QUALITIES)
            @test all(>(0.0), fs.qualities[:, PP.PP_Q_DENSITY])
            @test all(>=(0.0), fs.qualities[:, PP.PP_Q_SULFUR])

            # The network is a DAG by construction: a unit only takes feeds made
            # at a strictly lower stage, and its outputs sit at its own stage or
            # later (a stream made by several units carries the latest one).
            for unit in fs.units
                @test !isempty(unit.feeds)
                @test all(fs.stream_stage[s] < unit.stage for s in unit.feeds)
                @test all(fs.stream_stage[s] >= unit.stage for s in unit.outputs)
                for mode in unit.modes
                    @test size(mode.yields) == (length(unit.feeds), length(unit.outputs))
                    @test all(>=(0.0), mode.yields)
                    # Conversion moves volume around; nothing multiplies it away.
                    @test all(0.5 <= sum(mode.yields[f, :]) <= 1.5 for f in eachindex(unit.feeds))
                end
            end

            # Every stream is made by the crude unit, made by a unit, or bought,
            # and every stream has somewhere to go.
            made = falses(S)
            for unit in fs.units, s in unit.outputs
                made[s] = true
            end
            for c in 1:fs.n_crudes, k in eachindex(fs.cut_classes)
                made[fs.cut_stream[c, k]] = true
            end
            for s in fs.purchasable
                made[s] = true
            end
            @test all(made)
            sink = falses(S)
            for unit in fs.units, s in unit.feeds
                sink[s] = true
            end
            for grade in fs.products, s in grade.components
                sink[s] = true
            end
            for s in fs.spot
                sink[s] = true
            end
            @test all(sink)

            # Grades are non-empty and their windows are consistent.
            for grade in fs.products
                @test !isempty(grade.components)
                @test all(grade.spec_min[q] <= grade.spec_max[q] for q in 1:PP.PP_N_QUALITIES)
            end

            # Data is dimensioned to the flowsheet and the horizon.
            T = p.data.n_periods
            @test size(p.data.crude_price) == (fs.n_crudes, T)
            @test size(p.data.unit_capacity) == (length(fs.units), T)
            @test size(p.data.demand_min) == (length(fs.products), T)
            @test all(p.data.demand_min .<= p.data.demand_max)
            @test all(p.data.unit_min_throughput .<= p.data.unit_capacity)
            @test all(p.data.cdu_min_throughput .<= p.data.cdu_capacity)
            @test all(p.data.stream_initial_inventory .<= p.data.stream_tank_capacity .+ 1e-9)
            @test all(p.data.crude_initial_inventory .<= p.data.crude_tank_capacity .+ 1e-9)
            @test 0 <= p.data.renewable_min_fraction <= p.data.renewable_max_fraction <= 1

            # Upgraded blend components stay integrated with the refinery;
            # only bulk cuts, by-products, or otherwise dead-end streams receive
            # a merchant outlet.
            upgraded = Set((
                :reformate_mid,
                :reformate_high,
                :alkylate,
                :isomerate,
                :ulsd_component,
                :hc_diesel,
                :jet_component,
            ))
            @test all(!(fs.stream_classes[s] in upgraded) for s in fs.spot)
        end
    end

    @testset "Planted operation" begin
        for variant in (:refinery, :mode_switching)
            for target in (150, 800, 4000), seed in 0:2
                _, p = generate_problem(:process_planning, target, feasible, seed; variant=variant)
                plan = p.feasible_witness
                @test plan !== nothing
                @test p.infeasibility_certificate === nothing
                # Re-check the plan against every row of the model by arithmetic.
                @test PP.refinery_plan_satisfies(p.flowsheet, p.data, plan)

                fs, data = p.flowsheet, p.data
                T = data.n_periods
                @test size(plan.crude_run) == (fs.n_crudes, T)
                @test size(plan.unit_mode) == (length(fs.units), T)
                @test all(
                    1 <= plan.unit_mode[u, t] <= length(fs.units[u].modes) for
                    u in eachindex(fs.units), t in 1:T
                )
                # Something is actually made in every period.
                @test all(sum(view(plan.crude_run, :, t)) > 0 for t in 1:T)
                if variant == :mode_switching
                    @test PP._pp_mode_schedule_satisfies(
                        plan.unit_mode, p.initial_mode, p.minimum_run
                    )
                end
                # Blend qualities of the plan sit inside the stated windows.
                for (g, grade) in enumerate(fs.products), t in 1:T
                    volumes = view(plan.blend[g], :, t)
                    sum(volumes) > 0 || continue
                    quality = PP._pp_blend_quality(fs, grade, volumes)
                    for q in 1:PP.PP_N_QUALITIES
                        tol = 1e-6 * max(1.0, abs(quality[q]))
                        isfinite(grade.spec_min[q]) && @test quality[q] >= grade.spec_min[q] - tol
                        isfinite(grade.spec_max[q]) && @test quality[q] <= grade.spec_max[q] + tol
                    end
                end
            end

            # A witness only exists for a requested-feasible instance.
            for status in (infeasible, unknown)
                _, p = generate_problem(:process_planning, 400, status, 1; variant=variant)
                @test p.feasible_witness === nothing
            end
        end

        # The campaign switches modes over the horizon whenever a unit has more
        # than one, so the changeover rows are not decoration.
        _, p = generate_problem(:process_planning, 4000, feasible, 4; variant=:mode_switching)
        multimode = [
            u for u in eachindex(p.flowsheet.units) if length(p.flowsheet.units[u].modes) > 1
        ]
        if !isempty(multimode) && p.data.n_periods >= 4
            @test any(length(unique(p.feasible_witness.unit_mode[u, :])) > 1 for u in multimode)
        end

        # The environmental extension is coupled to actual hydroprocessing
        # feeds, not an independent side budget. Its full H2, sulfur, and carbon
        # witness is checked from those feeds period by period.
        for target in (50, 500, 3000), seed in 0:4
            _, p = generate_problem(
                :process_planning, target, feasible, seed; variant=:hydrogen_network
            )
            @test p.feasible_witness !== nothing
            @test PP.refinery_hydrogen_plan_satisfies(p)
            w = p.feasible_witness
            demand = sum(
                p.hydrogen.unit_h2_rate[u][f] * w.refinery.unit_feed[u][f, t] for
                u in eachindex(p.flowsheet.units) for f in eachindex(p.flowsheet.units[u].feeds) for
                t in 1:p.data.n_periods
            )
            @test demand > 0
            @test sum(w.sulfur_recovered) > 0
            @test sum(w.carbon_emissions) > 0
        end
        for status in (infeasible, unknown)
            _, p = generate_problem(:process_planning, 500, status, 2; variant=:hydrogen_network)
            @test p.feasible_witness === nothing
        end
    end

    @testset "Infeasibility certificates" begin
        for variant in (:refinery, :mode_switching, :hydrogen_network)
            kinds = Set{PP.RefineryInfeasibilityKind}()
            for target in (150, 800, 4000), seed in 0:5
                _, p = generate_problem(
                    :process_planning, target, infeasible, seed; variant=variant
                )
                certificate = p.infeasibility_certificate
                @test certificate !== nothing
                @test PP.refinery_certificate_holds(p.flowsheet, p.data, certificate)
                push!(kinds, certificate.kind)

                # Both kinds are the aggregate potential argument: the contracted
                # volume is strictly above the bound, and the bound is what the
                # potential argument recomputes.
                @test certificate.required > certificate.achievable
                @test isapprox(
                    certificate.achievable, PP._pp_production_bound(p.flowsheet, p.data); rtol=1e-9
                )
                @test isapprox(certificate.required, sum(p.data.demand_min); rtol=1e-9)
            end
            # The crude-supply curtailment is the default; contracts above the
            # conversion bound appear across seeds too. (The single-row
            # specification refutation is no longer planted: presolve decides it.)
            @test PP.refinery_crude_supply_below_contracts in kinds
            @test kinds == Set(instances(PP.RefineryInfeasibilityKind))

            # A certificate only exists for a requested-infeasible instance, and
            # a feasible instance's data does not support one.
            for status in (feasible, unknown)
                _, p = generate_problem(:process_planning, 400, status, 1; variant=variant)
                @test p.infeasibility_certificate === nothing
            end
            _, feasible_problem = generate_problem(
                :process_planning, 400, feasible, 2; variant=variant
            )
            _, broken = generate_problem(:process_planning, 400, infeasible, 2; variant=variant)
            @test !PP.refinery_certificate_holds(
                feasible_problem.flowsheet, feasible_problem.data, broken.infeasibility_certificate
            )
        end
    end

    @testset "Process campaigns" begin
        for target in (60, 600, 5000, 40000), status in (feasible, infeasible, unknown)
            _, p = generate_problem(:process_planning, target, status, 5; variant=:campaign)
            @test p.period_days == 7.0
            @test p.n_sites == maximum(p.material_site)
            @test (p.n_sites > 1) == (target > PP._CP_SINGLE_SITE_LIMIT)
            # Every raw belongs to exactly one regional pool, pools are keyed by
            # material, and markets only couple a grade made at several sites.
            raws = [m for m in eachindex(p.material_kind) if p.material_kind[m] == :raw]
            @test sort(reduce(vcat, p.supply_groups)) == raws
            @test all(length(unique(p.material_names[g])) == 1 for g in p.supply_groups)
            @test all(length(unique(p.material_site[g])) == length(g) for g in p.supply_groups)
            @test all(length(g) >= 2 for g in p.market_groups)
            @test all(p.material_kind[m] == :final for g in p.market_groups for m in g)
            @test size(p.supply_cap) == (length(p.supply_groups), p.n_periods)
            # Tasks only touch materials of their own site.
            for t in eachindex(p.task_names)
                site = p.unit_site[p.task_unit[t]]
                @test all(p.material_site[m] == site for (m, _) in p.task_inputs[t])
                @test all(p.material_site[m] == site for (m, _) in p.task_outputs[t])
            end
            @test 2 <= p.campaign_length <= 3
            @test !isempty(p.task_names)
            @test !isempty(p.material_names)
            @test all(any(==(u), p.task_unit) for u in eachindex(p.unit_names))
            @test all(
                (count(==(u), p.task_unit) > 1) == p.campaign_unit[u] for
                u in eachindex(p.unit_names)
            )
            @test all(c > 0 for io in (p.task_inputs..., p.task_outputs...) for (_, c) in io)
            @test all(p.tank .>= p.initial_inventory)
            for m in eachindex(p.material_names)
                p.material_kind[m] == :raw || continue
                @test issorted(vec(p.tier_price[m, 1:p.n_tiers]))
                @test all(p.tier_cap[m, :] .>= 0)
            end
        end

        for target in (60, 500, 3000, 30000), seed in 0:3
            _, p = generate_problem(:process_planning, target, feasible, seed; variant=:campaign)
            @test p.feasible_witness !== nothing
            @test p.infeasibility_certificate === nothing
            @test p.market_scenario === nothing
            @test PP.campaign_plan_satisfies(p)
            w = p.feasible_witness
            L = p.campaign_length
            for i in axes(w.starts, 1), t in axes(w.starts, 2)
                if w.starts[i, t] > 0.5
                    @test t + L - 1 <= p.n_periods
                    @test all(w.active[i, t:(t + L - 1)] .== 1.0)
                end
            end
        end

        certificate_types = Set{DataType}()
        for target in (60, 500, 3000, 30000), seed in 0:5
            _, p = generate_problem(:process_planning, target, infeasible, seed; variant=:campaign)
            @test p.feasible_witness === nothing
            @test p.infeasibility_certificate !== nothing
            @test p.market_scenario === nothing
            @test PP.campaign_certificate_holds(p)
            cert = p.infeasibility_certificate
            push!(certificate_types, typeof(cert))
            if cert isa PP.CampaignFeedstockCertificate
                # Recompute the pool argument by hand: the potential is one on
                # the pooled raws, zero on every other raw, and no task creates
                # pooled-raw content.
                y = cert.potential
                members = p.supply_groups[cert.group]
                @test all(y[members] .== 1.0)
                @test all(
                    y[m] == 0.0 for
                    m in eachindex(y) if p.material_kind[m] == :raw && !(m in members)
                )
                for t in eachindex(p.task_names)
                    made = sum(c * y[m] for (m, c) in p.task_outputs[t])
                    fed = sum(c * y[m] for (m, c) in p.task_inputs[t])
                    @test made <= fed + 1e-9
                end
                demand = sum(
                    y[m] * p.sales_floor[m, τ] for m in eachindex(y) for
                    τ in 1:p.n_periods if p.material_kind[m] == :final
                )
                supply = sum(
                    min(
                        p.supply_cap[cert.group, τ],
                        sum(p.tier_cap[m, j] for m in members for j in 1:p.n_tiers),
                    ) for τ in 1:p.n_periods
                )
                opening = sum(y .* p.initial_inventory)
                @test demand ≈ cert.demand rtol = 1e-9
                @test opening + supply ≈ cert.upper_bound rtol = 1e-9
                # A planted margin, not a knife edge.
                @test cert.margin >= 0.05 * cert.demand
                # The curtailed supply follows the contracted need of every
                # period (no short window), and the pool is among the broadest.
                need = [PP._cp_pool_need(p, y, τ) for τ in 1:p.n_periods]
                busy = findall(>(1e-3), need)
                ratios = [p.supply_cap[cert.group, τ] / need[τ] for τ in busy]
                @test maximum(ratios) - minimum(ratios) <= 1e-9 * maximum(ratios)
                breadths = [
                    PP._cp_pool_breadth(p, PP._cp_pool_bound_parts(p, g)[1]) for
                    g in eachindex(p.supply_groups)
                ]
                @test PP._cp_pool_breadth(p, y) >= 0.5 * maximum(breadths)
            end
        end
        # The feedstock pool is the default argument.
        @test PP.CampaignFeedstockCertificate in certificate_types

        for seed in 0:4
            _, p = generate_problem(:process_planning, 700, unknown, seed; variant=:campaign)
            @test p.feasible_witness === nothing
            @test p.infeasibility_certificate === nothing
            @test p.market_scenario !== nothing
            @test -0.35 <= p.market_scenario.supply_factor <= 0.95
        end
    end

    @testset "Capacity expansion network" begin
        for target in (120, 900, 6000), status in (feasible, infeasible, unknown)
            _, p = generate_problem(
                :process_planning, target, status, 5; variant=:capacity_expansion
            )
            J = length(p.chemicals)
            @test !isempty(p.technologies)
            @test !isempty(p.raw_chemicals)
            @test !isempty(p.sellable_chemicals)
            @test all(p.chemicals[j].purchasable for j in p.raw_chemicals)
            @test all(p.chemicals[j].sellable for j in p.sellable_chemicals)
            # Raw materials are bought, not made; nothing is both.
            @test all(!(p.chemicals[j].purchasable && p.chemicals[j].sellable) for j in 1:J)
            for technology in p.technologies
                # Inputs come from strictly lower layers, so the network is a DAG.
                @test all(p.chemicals[j].layer < technology.layer for (j, _) in technology.inputs)
                @test all(coefficient > 0 for (_, coefficient) in technology.inputs)
                @test technology.outputs[1].first == technology.main_output
                @test technology.outputs[1].second == 1.0
                @test p.chemicals[technology.main_output].layer == technology.layer
                @test 0 <= technology.min_expansion <= technology.max_expansion
                @test technology.existing_capacity >= 0
            end
            @test all(p.demand_min .<= p.demand_max)
            @test issorted(p.discount; rev=true)
            @test all(0 .< p.discount .<= 1)
        end

        for target in (150, 900, 5000), seed in 0:2
            m, p = generate_problem(
                :process_planning, target, feasible, seed; variant=:capacity_expansion
            )
            @test p.feasible_witness !== nothing
            @test p.infeasibility_certificate === nothing
            @test p.budget_scenario === nothing
            @test PP.process_expansion_plan_satisfies(p)
            # The plan really invests: capacity grows somewhere over the horizon.
            w = p.feasible_witness
            @test sum(w.expansion) > 0
            # The capital budget is a real row with room above the plan's spend.
            @test constraint_by_name(m, "capital_budget") !== nothing
            spend = PP._pp_expansion_capital_spend(p.technologies, w.expansion, w.expand)
            @test 1.10 * spend - 1e-6 <= p.capital_budget <= 1.50 * spend + 1e-6
            # A process the plan never runs is still a buildable option.
            @test all(t.max_expansion > 1e-2 for t in p.technologies)
        end

        kinds = Set{PP.ProcessExpansionInfeasibilityKind}()
        for target in (150, 900, 5000), seed in 0:5
            _, p = generate_problem(
                :process_planning, target, infeasible, seed; variant=:capacity_expansion
            )
            @test p.feasible_witness === nothing
            @test p.budget_scenario === nothing
            @test PP.process_expansion_certificate_holds(p)
            certificate = p.infeasibility_certificate
            push!(kinds, certificate.kind)
            # The capital potential is a valid dual by hand: zero on raws, never
            # negative, and no process creates more potential than one unit of
            # new capacity costs it in the relaxation.
            potential = certificate.potential
            @test all(potential[j] == 0.0 for j in p.raw_chemicals)
            @test all(>=(0.0), potential)
            unit_costs = [
                t.variable_investment + t.fixed_investment / t.max_expansion for t in p.technologies
            ]
            nets = [
                sum(c * potential[j] for (j, c) in t.outputs) -
                sum(a * potential[j] for (j, a) in t.inputs) for t in p.technologies
            ]
            @test all(nets .<= unit_costs .+ 1e-9 .* unit_costs)
            τ = certificate.period
            required =
                sum(potential[j] * p.demand_min[j, τ] for j in axes(p.demand_min, 1)) - sum(
                    max(nets[i], 0.0) * p.technologies[i].existing_capacity for
                    i in eachindex(p.technologies)
                )
            @test certificate.required ≈ required rtol = 1e-9
            @test certificate.achievable == p.capital_budget
            # A planted margin, not a knife edge.
            @test certificate.required >= 1.05 * certificate.achievable
            @test certificate.achievable > 0
        end
        @test kinds == Set([PP.expansion_capital_below_requirement])

        # Unknown instances carry their budget scenario, and only finished
        # chemicals are sold forward.
        for seed in 0:3
            _, p = generate_problem(
                :process_planning, 900, unknown, seed; variant=:capacity_expansion
            )
            @test p.budget_scenario !== nothing
            @test -0.15 <= p.budget_scenario.budget_share <= 0.95
            @test p.feasible_witness === nothing
            @test p.infeasibility_certificate === nothing
            for j in p.sellable_chemicals
                startswith(String(p.chemicals[j].name), "product_") && continue
                @test all(iszero, view(p.demand_min, j, :))
            end
        end

        # A feasible instance's data does not support the certificate of a
        # different, broken one.
        _, healthy = generate_problem(
            :process_planning, 600, feasible, 3; variant=:capacity_expansion
        )
        _, broken = generate_problem(
            :process_planning, 600, infeasible, 3; variant=:capacity_expansion
        )
        transplanted = PP.ProcessCapacityExpansionProblem(
            healthy.n_periods,
            healthy.chemicals,
            healthy.technologies,
            healthy.raw_chemicals,
            healthy.sellable_chemicals,
            healthy.purchase_cost,
            healthy.availability,
            healthy.sale_price,
            healthy.demand_min,
            healthy.demand_max,
            healthy.discount,
            healthy.capital_budget,
            nothing,
            broken.infeasibility_certificate,
            nothing,
            infeasible,
        )
        @test !PP.process_expansion_certificate_holds(transplanted)
    end

    @testset "Model structure" begin
        # The refinery variant is a pure LP: nothing to relax.
        m, _ = generate_problem(:process_planning, 900, unknown, 2; relax_integer=false)
        @test num_constraints(m, VariableRef, MOI.ZeroOne) == 0
        @test num_constraints(m, VariableRef, MOI.Integer) == 0
        @test objective_sense(m) == MOI.MAX_SENSE

        # Mode switching carries a run and an exact start indicator per unit,
        # mode, and period.
        m, p = generate_problem(
            :process_planning, 2000, unknown, 2; variant=:mode_switching, relax_integer=false
        )
        @test num_constraints(m, VariableRef, MOI.ZeroOne) ==
            2 * p.data.n_periods * sum(length(u.modes) for u in p.flowsheet.units; init=0)
        @test num_variables(m) == mode_variables(p)
        relaxed, _ = generate_problem(:process_planning, 2000, unknown, 2; variant=:mode_switching)
        @test num_constraints(relaxed, VariableRef, MOI.ZeroOne) == 0
        @test num_variables(relaxed) == num_variables(m)

        # The H2/SRU/carbon extension remains a pure LP.
        m, p = generate_problem(
            :process_planning, 2000, unknown, 2; variant=:hydrogen_network, relax_integer=false
        )
        @test num_constraints(m, VariableRef, MOI.ZeroOne) == 0
        @test num_variables(m) == hydrogen_variables(p)

        # Campaign trains carry an active and an exact start indicator per
        # campaign task and period.
        m, p = generate_problem(
            :process_planning, 2000, unknown, 2; variant=:campaign, relax_integer=false
        )
        @test num_constraints(m, VariableRef, MOI.ZeroOne) ==
            2 * p.n_periods * length(campaign_tasks(p))
        @test num_variables(m) == campaign_variables(p)

        # Capacity expansion carries one expansion indicator per process and period.
        m, p = generate_problem(
            :process_planning, 2000, unknown, 2; variant=:capacity_expansion, relax_integer=false
        )
        @test num_constraints(m, VariableRef, MOI.ZeroOne) == p.n_periods * length(p.technologies)

        # Reproducibility: identical data and model for the same seed.
        for variant in
            (:refinery, :mode_switching, :hydrogen_network, :campaign, :capacity_expansion)
            m1, p1 = generate_problem(:process_planning, 700, feasible, 11; variant=variant)
            m2, p2 = generate_problem(:process_planning, 700, feasible, 11; variant=variant)
            @test sprint(print, m1) == sprint(print, m2)
            @test num_variables(m1) == num_variables(m2)
            @test typeof(p1) == typeof(p2)
        end

        # Edge sizes stay well formed.
        for target in (1, 5, 25),
            variant in
            (:refinery, :mode_switching, :hydrogen_network, :campaign, :capacity_expansion),
            status in (feasible, infeasible, unknown)

            m, _ = generate_problem(:process_planning, target, status, 0; variant=variant)
            @test num_variables(m) > 0
            @test num_constraints(m; count_variable_in_set_constraints=false) > 0
        end
    end

    @testset "Solver feasibility contracts" begin
        if HAS_HIGHS
            function solved_status(model; limit=120.0)
                set_optimizer(model, HiGHS.Optimizer)
                set_silent(model)
                set_time_limit_sec(model, limit)
                optimize!(model)
                status = termination_status(model)
                if status == MOI.OTHER_ERROR
                    # HiGHS's dual simplex occasionally aborts ("excessive dual
                    # values") while proving a refinery-family LP infeasible
                    # (a few percent of requests); the interior-point method
                    # settles the same model.
                    set_attribute(model, "solver", "ipm")
                    optimize!(model)
                    status = termination_status(model)
                end
                return status
            end

            for variant in
                (:refinery, :mode_switching, :hydrogen_network, :campaign, :capacity_expansion)
                for target in (120, 900, 4000), seed in 0:2
                    m, _ = generate_problem(
                        :process_planning, target, feasible, seed; variant=variant
                    )
                    @test solved_status(m) == MOI.OPTIMAL
                    m, _ = generate_problem(
                        :process_planning, target, infeasible, seed; variant=variant
                    )
                    @test solved_status(m) == MOI.INFEASIBLE
                end
            end

            # A multi-site campaign company keeps both contracts.
            m, _ = generate_problem(:process_planning, 30_000, feasible, 0; variant=:campaign)
            @test solved_status(m) == MOI.OPTIMAL
            m, _ = generate_problem(:process_planning, 30_000, infeasible, 0; variant=:campaign)
            @test solved_status(m) == MOI.INFEASIBLE

            # The planted operation is a feasible point of the unrelaxed integer
            # model too, not just of the relaxation.
            for variant in (:mode_switching, :campaign, :capacity_expansion)
                m, _ = generate_problem(
                    :process_planning, 600, feasible, 1; variant=variant, relax_integer=false
                )
                @test solved_status(m; limit=180.0) == MOI.OPTIMAL
            end

            # The planted refutations need simplex work: HiGHS presolve alone
            # does not decide a requested-infeasible instance.
            for variant in (:refinery, :hydrogen_network, :campaign, :capacity_expansion),
                target in (1000, 3000),
                seed in 0:1

                m, _ = generate_problem(
                    :process_planning, target, infeasible, seed; variant=variant
                )
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                set_time_limit_sec(m, 60.0)
                set_attribute(m, "solver", "simplex")
                optimize!(m)
                # (A rare dual-simplex abort still counts: it happened after
                # presolve handed the model over.)
                @test termination_status(m) in (MOI.INFEASIBLE, MOI.OTHER_ERROR)
                @test MOI.get(m, MOI.SimplexIterations()) > 0
            end

            # Unknown-status instances are genuinely undecided: across seeds and
            # scales the corpus contains both outcomes.
            for variant in
                (:refinery, :mode_switching, :hydrogen_network, :campaign, :capacity_expansion)
                outcomes = Set{Any}()
                for target in (200, 2000), seed in 0:7
                    m, _ = generate_problem(
                        :process_planning, target, unknown, seed; variant=variant
                    )
                    push!(outcomes, solved_status(m))
                end
                @test MOI.OPTIMAL in outcomes
                @test MOI.INFEASIBLE in outcomes
            end
        else
            @info "Skipping process_planning solver tests (HiGHS unavailable)"
        end
    end
end
