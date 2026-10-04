# Focused quality contracts for the inventory category (four variants):
# registry shape, exact column/row formulas and size fidelity up to 100k,
# data invariants, planted witnesses checked against the built models,
# aggregate infeasibility certificates recomputed from the struct fields,
# the exact feasibility characterisation of `multi_item`, the LP relevance
# of the lot-sizing setup structure, reproducibility under a dirty global RNG,
# and HiGHS contracts (infeasible instances must need simplex work).
@testset "Inventory" begin
    @test :inventory in list_categories()
    @test Set(list_variants(:inventory)) ==
        Set([:standard, :lot_sizing, :multi_echelon, :multi_item])
    @test problem_info(:inventory)[:default_variant] == :standard

    nrows(m) = num_constraints(m; count_variable_in_set_constraints=false)
    refs = (:standard, :lot_sizing, :multi_echelon, :multi_item)

    # ---------------------------------------------------------------- sizing
    function std_counts(p)
        N, T = p.n_skus, p.n_periods
        cols = sum(T - L for L in p.lead_time) + N * T + count(>(0.0), p.demand)
        rows = N * T + count(>(0.0), p.fill_rate)
        for v in 1:p.n_vendors, t in 1:T
            rows += any(p.vendor[i] == v && t <= T - p.lead_time[i] for i in 1:N)
        end
        for z in 1:p.n_zones
            skus = findall(==(z), p.zone)
            isempty(skus) && continue
            rows += T + count(t -> any(t > p.lead_time[i] for i in skus), 1:T)
        end
        return cols, rows
    end
    function ls_counts(p)
        cols = sum(
            SyntheticLPs._lot_sizing_item_columns(p.demand[i, :], p.window) for i in 1:p.n_items
        )
        nw = sum(min(t, p.window) for i in 1:p.n_items for t in 1:p.n_periods if p.demand[i, t] > 0)
        return cols, count(>(0.0), p.demand) + nw + p.n_periods
    end
    function me_counts(p)
        P, R, S, T = p.n_products, p.n_dcs, p.n_stores, p.n_periods
        cols = P * (R * (T - 1) + sum(T - L for L in p.lane_transit) + R * T + S * T)
        thr = count(
            ((r, t),) -> any(p.lane_dc[l] == r && t <= T - p.lane_transit[l] for l in 1:p.n_lanes),
            [(r, t) for r in 1:R for t in 1:T],
        )
        rows = P * (R + S) * T + (T - 1) + thr + R * T + (P > 1 ? S * T : 0)
        return cols, rows
    end
    mi_counts(p) = (2 * p.n_items * p.n_periods, p.n_items * p.n_periods + p.n_periods)
    counts = Dict(
        :standard => std_counts,
        :lot_sizing => ls_counts,
        :multi_echelon => me_counts,
        :multi_item => mi_counts,
    )

    for v in refs, target in (60, 600, 4000), status in (feasible, infeasible, unknown), seed in 0:1
        m, p = generate_problem(ProblemVariant(:inventory, v), target, status, seed)
        cols, rows = counts[v](p)
        @test num_variables(m) == cols
        @test nrows(m) == rows
        @test abs(cols - target) <= max(0.12 * target, 16)   # one store/SKU of granularity
    end
    for v in refs
        m, p = generate_problem(ProblemVariant(:inventory, v), 100_000, unknown, 0)
        @test abs(num_variables(m) - 100_000) <= 5_000
        @test nrows(m) >= 30_000      # rows scale with columns
    end

    # ---------------------------------------------------------- standard
    for target in (100, 2000), seed in 0:2
        m, p = generate_problem(ProblemVariant(:inventory, :standard), target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing && p.infeasibility_certificate === nothing
        @test all(1 .<= p.lead_time .< p.n_periods)
        @test all(0.0 .<= p.fill_rate .< 1.0)
        point = Dict{VariableRef, Float64}()
        for i in 1:p.n_skus, t in 1:p.n_periods
            point[m[:I][i, t]] = w.stock[i, t]
            t <= p.n_periods - p.lead_time[i] && (point[m[:q][i, t]] = w.orders[i, t])
            p.demand[i, t] > 0 && (point[m[:u][i, t]] = 0.0)
        end
        @test isempty(primal_feasibility_report(m, point; atol=1e-6))
    end
    for target in (100, 2000), seed in 0:3
        _, p = generate_problem(ProblemVariant(:inventory, :standard), target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing && p.feasible_witness === nothing
        @test all(p.vendor[i] == cert.vendor && p.fill_rate[i] > 0 for i in cert.skus)
        req = sum(
            max(
                0.0,
                p.fill_rate[i] * sum(p.demand[i, :]) - p.initial_inventory[i] -
                sum(p.pipeline[i, :]),
            ) for i in cert.skus
        )
        T = p.n_periods
        order_periods = [
            t for t in 1:T if
            any(p.vendor[i] == cert.vendor && t <= T - p.lead_time[i] for i in 1:p.n_skus)
        ]
        @test cert.required ≈ req
        @test cert.available ≈ sum(p.vendor_capacity[cert.vendor, t] for t in order_periods)
        @test cert.required >= 1.1 * cert.available * (1 - 1e-9)
    end

    # -------------------------------------------------------- lot_sizing
    for target in (100, 2000), seed in 0:2
        m, p = generate_problem(ProblemVariant(:inventory, :lot_sizing), target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing && p.infeasibility_certificate === nothing
        @test all(p.demand .>= 0)
        @test all(p.demand .<= p.gross_demand .+ 1e-9)
        load = zeros(p.n_periods)
        for i in 1:p.n_items
            for t in 1:p.n_periods
                if p.demand[i, t] > 0
                    s = w.source[i][t]
                    @test s in w.setups[i]
                    @test t - p.window < s <= t                    # inside the window
                    load[s] += p.proc_time[i] * p.demand[i, t]
                else
                    @test w.source[i][t] == 0
                end
            end
            for s in w.setups[i]
                load[s] += p.setup_time[i]
            end
        end
        @test load ≈ w.load
        @test all(w.load .<= p.capacity)
        w_item, w_prod, w_dem, y_item, y_period, _ = SyntheticLPs._lot_sizing_layout(p)
        point = Dict{VariableRef, Float64}()
        for k in eachindex(w_item)
            point[m[:w][k]] = w.source[w_item[k]][w_dem[k]] == w_prod[k] ? 1.0 : 0.0
        end
        for k in eachindex(y_item)
            point[m[:y][k]] = y_period[k] in w.setups[y_item[k]] ? 1.0 : 0.0
        end
        @test isempty(primal_feasibility_report(m, point; atol=1e-7))
    end
    for target in (100, 2000), seed in 0:3
        _, p = generate_problem(ProblemVariant(:inventory, :lot_sizing), target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing && p.feasible_witness === nothing
        h = cert.horizon
        lbs = map(1:p.n_items) do i
            # Disjoint windows by an independent left-to-right sweep.
            c, last = 0, 0
            for t in 1:h
                if p.demand[i, t] > 0 && t - p.window + 1 > last
                    c += 1
                    last = t
                end
            end
            c
        end
        @test cert.setup_lower_bounds == lbs
        req = sum(
            p.proc_time[i] * sum(p.demand[i, 1:h]) + p.setup_time[i] * lbs[i] for i in 1:p.n_items
        )
        @test cert.required ≈ req
        @test cert.available ≈ sum(p.capacity[1:h])
        @test cert.required >= 1.1 * cert.available * (1 - 1e-9)
    end

    # ----------------------------------------------------- multi_echelon
    for target in (100, 3000), seed in 0:2
        m, p = generate_problem(ProblemVariant(:inventory, :multi_echelon), target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing && p.infeasibility_certificate === nothing
        P, R, S, T = p.n_products, p.n_dcs, p.n_stores, p.n_periods
        @test all(p.lane_store[p.primary_lane[s]] == s for s in 1:S)
        @test all(0 .<= p.lane_transit .<= 1)
        point = Dict{VariableRef, Float64}()
        for pp in 1:P, r in 1:R, t in 1:T
            t <= T - 1 && (point[m[:f][pp, r, t]] = w.plant_to_dc[pp, r, t])
            point[m[:J][pp, r, t]] = w.dc_stock[pp, r, t]
        end
        for pp in 1:P, s in 1:S, t in 1:T
            point[m[:K][pp, s, t]] = w.store_stock[pp, s, t]
        end
        for pp in 1:P, l in 1:p.n_lanes, t in 1:(T - p.lane_transit[l])
            s = p.lane_store[l]
            point[m[:g][pp, l, t]] = l == p.primary_lane[s] ? w.dc_to_store[pp, s, t] : 0.0
        end
        @test isempty(primal_feasibility_report(m, point; atol=1e-6))
    end
    for target in (100, 3000), seed in 0:3
        _, p = generate_problem(
            ProblemVariant(:inventory, :multi_echelon), target, infeasible, seed
        )
        cert = p.infeasibility_certificate
        @test cert !== nothing && p.feasible_witness === nothing
        h = cert.horizon
        req = sum(
            p.plant_hours[q] * max(
                0.0, sum(p.demand[q, :, 1:h]) - sum(p.dc_initial[q, :]) - sum(p.store_initial[q, :])
            ) for q in 1:p.n_products
        )
        @test cert.required ≈ req
        @test cert.available ≈ sum(p.plant_capacity[1:(h - 1)])
        @test cert.required >= 1.1 * cert.available * (1 - 1e-9)
    end

    # -------------------------------------------------------- multi_item
    for target in (100, 2000), seed in 0:2
        m, p = generate_problem(ProblemVariant(:inventory, :multi_item), target, feasible, seed)
        w = p.feasible_witness
        @test w !== nothing && p.infeasibility_certificate === nothing
        @test p.binding_ratio <= 0.92 + 1e-9
        point = Dict{VariableRef, Float64}()
        for i in 1:p.n_items, t in 1:p.n_periods
            point[m[:x][i, t]] = w.production[i, t]
            point[m[:I][i, t]] = w.inventory[i, t]
        end
        @test isempty(primal_feasibility_report(m, point; atol=1e-6))
        # Initial stock covers the first period: no single row decides.
        @test all(p.initial_inventory .>= p.demand[:, 1])
    end
    for target in (100, 2000), seed in 0:3
        _, p = generate_problem(ProblemVariant(:inventory, :multi_item), target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing && p.feasible_witness === nothing
        h = cert.horizon
        req = sum(
            p.usage[i] * max(0.0, sum(p.demand[i, 1:h]) - p.initial_inventory[i]) for
            i in 1:p.n_items
        )
        @test cert.required ≈ req
        @test cert.available ≈ sum(p.capacity[1:h])
        @test cert.required >= 1.1 * cert.available * (1 - 1e-9)
    end

    for v in refs, seed in 0:3
        _, p = generate_problem(ProblemVariant(:inventory, v), 500, unknown, seed)
        @test p.feasible_witness === nothing
        @test p.infeasibility_certificate === nothing
    end

    # ------------------------------------------------------ reproducibility
    for v in refs, status in (feasible, infeasible, unknown)
        Random.seed!(11)
        _, p1 = generate_problem(ProblemVariant(:inventory, v), 700, status, 9)
        Random.seed!(1234)
        _, p2 = generate_problem(ProblemVariant(:inventory, v), 700, status, 9)
        for f in fieldnames(typeof(p1))
            f in (:feasible_witness, :infeasibility_certificate) && continue
            @test isequal(getfield(p1, f), getfield(p2, f))
        end
        if p1.infeasibility_certificate !== nothing
            @test p1.infeasibility_certificate.required == p2.infeasibility_certificate.required
        end
    end

    @testset "HiGHS contracts" begin
        if HAS_HIGHS
            function solve!(m)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                return termination_status(m)
            end
            for v in refs, target in (150, 1500), status in (feasible, infeasible), seed in 0:2
                m, _ = generate_problem(ProblemVariant(:inventory, v), target, status, seed)
                @test solve!(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
                @test MOI.get(m, MOI.SimplexIterations()) > 0   # never presolve-only
            end
            for v in refs
                outcomes = Set{MOI.TerminationStatusCode}()
                for seed in 0:9
                    m, p = generate_problem(ProblemVariant(:inventory, v), 600, unknown, seed)
                    ts = solve!(m)
                    push!(outcomes, ts)
                    if v == :multi_item
                        # Exact characterisation for a single shared resource.
                        @test ts == (p.binding_ratio <= 1.0 ? MOI.OPTIMAL : MOI.INFEASIBLE)
                    end
                end
                @test outcomes == Set([MOI.OPTIMAL, MOI.INFEASIBLE])
            end

            # Lot sizing: the setup structure survives relaxation — removing
            # setup times and costs strictly lowers the relaxed optimum.
            gaps = 0
            for seed in 0:3
                m, p = generate_problem(
                    ProblemVariant(:inventory, :lot_sizing), 1500, feasible, seed
                )
                solve!(m)
                z = objective_value(m)
                q = deepcopy(p)
                q.setup_time .= 0.0
                q.setup_cost .= 0.0
                m0 = SyntheticLPs.build_model(q)
                relax_integrality(m0)
                solve!(m0)
                gaps += z > 1.01 * objective_value(m0)
            end
            @test gaps >= 3
        end
    end
end
