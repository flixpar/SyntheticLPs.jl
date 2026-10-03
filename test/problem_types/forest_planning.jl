# Focused quality contracts for the forest_planning category: registry shape,
# exact variable-count sizing and row formulas, yield-table and landscape
# invariants, Model I / Model II structural contracts, the planted
# area-control witness checked against every row without a solver, the
# Lagrangian (DP) infeasibility certificate recomputed from the column data,
# reproducibility, and HiGHS feasibility contracts.

"Recompute the per-column total harvest volume (sum of harvest-definition coefficients)."
fp_column_totals(p) = [sum(p.vol_amount[p.vol_ptr[j]:(p.vol_ptr[j + 1] - 1)]; init=0.0) for j in eachindex(p.col_source)]

"Zone of every column's source."
function fp_column_zone(p)
    S = length(p.stratum_area)
    return [src <= S ? p.stratum_zone[src] : p.node_zone[src - S] for src in Int.(p.col_source)]
end

"Area clearcut per (zone, period) window row for column areas `x` (Dict of non-empty rows)."
function fp_greenup_rows(p, x)
    T, w = p.n_periods, p.greenup_window
    zone = fp_column_zone(p)
    rows = Dict{Tuple{Int, Int}, Float64}()
    for j in eachindex(p.col_source), c in (Int(p.col_cut1[j]), Int(p.col_cut2[j]))
        c == 0 && continue
        for t in c:min(T, c + w - 1)
            rows[(zone[j], t)] = get(rows, (zone[j], t), 0.0) + x[j]
        end
    end
    return rows
end

@testset "Forest Planning" begin
    @test :forest_planning in list_categories()
    @test Set(list_variants(:forest_planning)) == Set([:model_i, :model_ii])
    info = problem_info(:forest_planning)
    @test info[:default_variant] == :model_i
    @test occursin("forest", lowercase(info[:description]))
    variants = (:model_i, :model_ii)

    # Sizing: area columns plus T*K harvest accounting variables land on the
    # target (exact up to one column); rows are strata + Model II nodes +
    # harvest definitions + 2(T-1) even-flow rows + non-empty green-up rows +
    # the ending-inventory row.
    for v in variants, target in (10, 50, 200, 1000, 5000), status in (feasible, infeasible, unknown), seed in 0:1
        m, p = generate_problem(:forest_planning, target, status, seed; variant=v)
        T, K = p.n_periods, length(p.products)
        @test num_variables(m) == length(p.col_source) + T * K
        expected = max(target, 30 + T * K)   # 30-column floor keeps tiny instances schedulable
        @test expected - 1 <= num_variables(m) <= expected
        n_green = length(fp_greenup_rows(p, ones(length(p.col_source))))
        expected_rows = length(p.stratum_area) + length(p.node_period) + T * K + 2 * (T - 1) + n_green + 1
        @test num_constraints(m; count_variable_in_set_constraints=false) == expected_rows
        @test v == :model_ii || isempty(p.node_period)
    end

    # Large targets: exact sizing and rows growing with columns (constructor
    # only); the documented cap rejects larger requests.
    for v in variants
        p = SyntheticLPs.ForestPlanningProblem{v}(100_000, unknown, 3)
        @test SyntheticLPs.forest_num_variables(p) == 100_000
        n_rows = length(p.stratum_area) + length(p.node_period) + length(p.zone_area)
        @test n_rows >= 1_500
        cap = SyntheticLPs.FOREST_PLANNING_MAX_VARIABLES
        @test cap == 1_000_000
        @test_throws ArgumentError SyntheticLPs.ForestPlanningProblem{v}(cap + 1, unknown, 0)
        @test_throws ArgumentError generate_problem(:forest_planning, cap + 1, unknown, 0; variant=v)
    end

    # Yield tables: Chapman-Richards volume is zero up to the regeneration
    # lag and strictly increasing afterwards, bounded by the asymptote; the
    # sawlog share stays within (0, saw_max).
    _, p = generate_problem(:forest_planning, 3000, unknown, 5; variant=:model_ii)
    for sm in p.stand_models
        spec = p.forest_types[sm.type_index]
        ages = collect(0.0:5.0:300.0)
        vols = [SyntheticLPs._forest_volume(sm, a) for a in ages]
        @test all(vols[ages .<= sm.lag] .== 0.0)
        post = vols[ages .> sm.lag]
        @test all(diff(post) .> 0)
        @test maximum(vols) < sm.max_volume
        @test sm.regime in (:existing, :natural, :planted)
        @test sm.regime == :natural ? sm.lag == spec.natural_lag : sm.lag == 0.0
        fr = [SyntheticLPs._forest_saw_fraction(spec, sm, a) for a in ages]
        @test all(0.0 .< fr .< spec.saw_max)
    end

    # Landscape and column-store invariants shared by both formulations.
    for v in variants, target in (400, 4000), seed in 0:2
        _, p = generate_problem(:forest_planning, target, unknown, seed; variant=v)
        S, N, T = length(p.stratum_area), length(p.node_period), p.n_periods
        n = length(p.col_source)
        @test all(p.stratum_area .> 0)
        @test p.zone_area ≈ [sum(p.stratum_area[p.stratum_zone .== z]) for z in eachindex(p.zone_area)]
        @test all(0.25 .<= p.greenup_fraction .<= 0.40)
        @test p.greenup_window == max(1, ceil(Int, 20 / p.period_length))
        # unique (type, site, age class) analysis areas within a watershed
        keys = [(p.stratum_zone[s], p.stratum_model[s], p.stratum_age[s]) for s in 1:S]
        @test allunique(keys)
        @test all(p.stand_models[m].regime == :existing for m in p.stratum_model)
        @test length(p.vol_ptr) == n + 1
        @test all(1 .<= p.vol_period .<= T)
        @test all(1 .<= p.vol_product .<= length(p.products))
        @test all(p.vol_amount .> 0)
        @test all(p.col_ending_inventory .>= 0)
        @test all(1 .<= p.col_source .<= S + N)
        counts = zeros(Int, S + N)
        for src in p.col_source
            counts[src] += 1
        end
        # Every source but the streamed tail has a real choice.
        @test all(counts .>= 2)
        for s in 1:S
            # one do-nothing (grow-to-end, unthinned) column per stratum
            @test count(j -> p.col_source[j] == s && p.col_cut1[j] == 0 && p.col_thin[j] == 0, 1:n) == 1
        end
        for j in 1:n
            c1, c2, th = Int(p.col_cut1[j]), Int(p.col_cut2[j]), Int(p.col_thin[j])
            @test c2 == 0 || (c1 > 0 && c2 > c1)
            @test th == 0 || c1 == 0 || th < c1
            # harvest entries appear exactly at the thinning / clearcut periods
            ps = Set(Int.(p.vol_period[p.vol_ptr[j]:(p.vol_ptr[j + 1] - 1)]))
            @test ps ⊆ Set(filter(>(0), [th, c1, c2]))
        end
        if v == :model_i
            @test all(p.col_dest .== 0)
        else
            @test all(p.col_cut2 .== 0)
            # Network: arcs run forward in time and enter nodes of the same
            # watershed; a node exists only if its stand can still be cut.
            zone = fp_column_zone(p)
            for j in 1:n
                d = Int(p.col_dest[j])
                d == 0 && continue
                src = Int(p.col_source[j])
                src_period = src <= S ? 0 : p.node_period[src - S]
                @test p.node_period[d] == p.col_cut1[j] > src_period
                @test p.node_zone[d] == zone[j]
                @test p.col_ending_inventory[j] == 0.0
                src_model = src <= S ? p.stratum_model[src] : p.node_model[src - S]
                @test p.node_model[d] == p.regen_targets[src_model][p.col_regen1[j]]
            end
            inflow = zeros(Int, N)
            for d in p.col_dest
                d > 0 && (inflow[d] += 1)
            end
            @test all(inflow .>= 1)
            @test allunique([(p.node_model[i], p.node_zone[i], p.node_period[i]) for i in 1:N])
        end
    end

    # Planted witness: checked arithmetically against every row family.
    for v in variants, target in (100, 1000, 6000), seed in 0:2
        m, p = generate_problem(:forest_planning, target, feasible, seed; variant=v)
        w = p.feasible_witness
        @test w !== nothing
        @test p.infeasibility_certificate === nothing
        x = w.areas
        S, N, T, K = length(p.stratum_area), length(p.node_period), p.n_periods, length(p.products)
        @test length(x) == length(p.col_source)
        @test all(x .>= 0)
        bal = zeros(S + N)
        for j in eachindex(x)
            bal[p.col_source[j]] += x[j]
            p.col_dest[j] > 0 && (bal[S + p.col_dest[j]] -= x[j])
        end
        scale = sum(p.stratum_area)
        @test all(abs.(bal[1:S] .- p.stratum_area) .<= 1e-8 * scale)
        @test all(abs.(bal[(S + 1):end]) .<= 1e-8 * scale)
        H = zeros(T, K)
        for j in eachindex(x), e in p.vol_ptr[j]:(p.vol_ptr[j + 1] - 1)
            H[p.vol_period[e], p.vol_product[e]] += p.vol_amount[e] * x[j]
        end
        @test H ≈ w.harvest rtol = 1e-9
        @test all(isapprox.(sum(H; dims=2), w.flat_volume; rtol=1e-7))   # perfectly even flow
        @test w.flat_volume > 0
        δ = p.even_flow_tolerance
        @test 0.05 <= δ <= 0.20
        for t in 2:T
            tot, prev = sum(H[t, :]), sum(H[t - 1, :])
            @test tot >= (1 - δ) * prev
            @test tot <= (1 + δ) * prev
        end
        for t in 1:T, k in 1:K
            @test p.min_supply[k] <= H[t, k] + 1e-9
            @test H[t, k] <= p.max_supply[k]
        end
        @test p.min_supply ≈ p.base_min_supply
        for ((z, t), area) in fp_greenup_rows(p, x)
            @test area <= p.greenup_fraction[z] * p.zone_area[z] * (1 + 1e-9)
        end
        ei = sum(p.col_ending_inventory .* x)
        @test ei ≈ w.ending_inventory rtol = 1e-9
        @test ei >= p.min_ending_inventory
        @test p.min_ending_inventory ≈ p.base_min_ending_inventory
        @test p.supply_scale == p.inventory_scale == 1.0
        @test p.lagrangian_scale >= 1.0
        if target <= 1000
            point = Dict{VariableRef, Float64}(m[:area][j] => x[j] for j in eachindex(x))
            for t in 1:T, k in 1:K
                point[m[:harvest][t, k]] = H[t, k]
            end
            @test isempty(primal_feasibility_report(m, point; atol=1e-6))
        end
    end

    # Lagrangian certificate: DP values are dual feasible for every column,
    # their area-weighted sum is the stored bound, and it falls short of the
    # aggregated requirements with margin.
    for v in variants, target in (100, 1000, 6000), seed in 0:2
        _, p = generate_problem(:forest_planning, target, infeasible, seed; variant=v)
        cert = p.infeasibility_certificate
        @test cert !== nothing
        @test p.feasible_witness === nothing
        S = length(p.stratum_area)
        μ = cert.inventory_weight
        @test μ >= 0
        π = cert.source_values
        @test length(π) == S + length(p.node_period)
        ch = fp_column_totals(p)
        slack = [
            π[p.col_source[j]] - (ch[j] + μ * p.col_ending_inventory[j] + (p.col_dest[j] > 0 ? π[S + p.col_dest[j]] : 0.0))
            for j in eachindex(ch)
        ]
        @test minimum(slack) >= -1e-9 * maximum(abs, π)
        # tight: every source attains its value on some column (exact DP)
        for src in eachindex(π)
            @test minimum(slack[p.col_source .== src]) <= 1e-9 * max(1.0, abs(π[src]))
        end
        @test cert.bound ≈ sum(p.stratum_area .* π[1:S]) rtol = 1e-10
        required = p.n_periods * sum(p.min_supply) + μ * p.min_ending_inventory
        @test cert.required ≈ required rtol = 1e-10
        @test cert.bound * 1.02 < cert.required
        # requirements are scaled-up planted contracts, each satisfiable alone
        @test p.min_supply ≈ p.supply_scale .* p.base_min_supply
        @test p.min_ending_inventory ≈ p.inventory_scale * p.base_min_ending_inventory
        @test p.supply_scale >= 1.0 && p.inventory_scale >= 1.0
        @test all(p.min_supply .<= p.max_supply)
        ei_max = sum(
            p.stratum_area[s] * maximum(p.col_ending_inventory[p.col_source .== s]) for s in 1:S
        )
        v == :model_i && @test p.min_ending_inventory < ei_max
    end

    # Unknown: natural contracts on a continuum from the planted level to
    # just past the certified-infeasible level; no witness or certificate.
    for v in variants, seed in 0:9
        _, p = generate_problem(:forest_planning, 500, unknown, seed; variant=v)
        @test p.feasible_witness === nothing
        @test p.infeasibility_certificate === nothing
        @test 1.0 <= p.supply_scale <= 1.08 * p.lagrangian_scale + 1e-12
        @test 1.0 <= p.inventory_scale <= p.supply_scale + 1e-12
    end

    # Reproducibility, including isolation from a dirty global RNG.
    for v in variants, status in (feasible, infeasible, unknown)
        Random.seed!(987)
        m1, p1 = generate_problem(:forest_planning, 700, status, 42; variant=v)
        Random.seed!(12345)
        m2, p2 = generate_problem(:forest_planning, 700, status, 42; variant=v)
        for f in fieldnames(typeof(p1))
            f in (:feasible_witness, :infeasibility_certificate, :forest_types, :stand_models) && continue
            @test isequal(getfield(p1, f), getfield(p2, f))
        end
        if p1.feasible_witness !== nothing
            @test p1.feasible_witness.areas == p2.feasible_witness.areas
        end
        if p1.infeasibility_certificate !== nothing
            @test p1.infeasibility_certificate.source_values == p2.infeasibility_certificate.source_values
        end
        @test sprint(print, m1) == sprint(print, m2)
    end

    @testset "Forest Planning HiGHS contracts" begin
        if HAS_HIGHS
            for v in variants, target in (200, 1500), status in (feasible, infeasible), seed in 0:3
                m, _ = generate_problem(:forest_planning, target, status, seed; variant=v)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end
            for v in variants, status in (feasible, infeasible)
                m, _ = generate_problem(:forest_planning, 400, status, 1; variant=v, optimizer=HiGHS.Optimizer)
                @test num_variables(m) > 0
            end
            # The planted schedule is feasible, so the optimum is at least its NPV.
            for v in variants
                m, p = generate_problem(:forest_planning, 800, feasible, 2; variant=v)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test objective_value(m) >= sum(p.col_npv .* p.feasible_witness.areas) - 1e-6 * abs(objective_value(m))
            end
            # `unknown` is genuinely two-sided.
            for v in variants
                outcomes = Set{MOI.TerminationStatusCode}()
                for seed in 0:11
                    m, _ = generate_problem(:forest_planning, 500, unknown, seed; variant=v)
                    set_optimizer(m, HiGHS.Optimizer)
                    set_silent(m)
                    optimize!(m)
                    push!(outcomes, termination_status(m))
                end
                @test MOI.OPTIMAL in outcomes
                @test MOI.INFEASIBLE in outcomes
            end
        end
    end
end
