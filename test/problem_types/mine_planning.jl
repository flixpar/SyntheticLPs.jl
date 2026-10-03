# Focused quality contracts for the mine_planning category: registry shape,
# exact variable/row formulas, block-model and precedence invariants
# (topological order, precedence closure, transitive reduction), planted
# witnesses checked against every row of the built MIP without a solver,
# closure-flow certificate arithmetic recomputed from the struct fields,
# reproducibility, and HiGHS-backed feasibility contracts (including the
# validity of the closure bound against the LP it bounds).
@testset "Mine Planning" begin
    MP = SyntheticLPs
    variants = (:cpit, :pcpsp, :stockpile)
    mp_ref(v) = ProblemVariant(:mine_planning, v)

    @test :mine_planning in list_categories()
    @test Set(list_variants(:mine_planning)) == Set(variants)
    info = problem_info(:mine_planning)
    @test info[:default_variant] == :cpit
    @test occursin("mine", lowercase(info[:description]))

    # --- helpers -------------------------------------------------------------
    n_rows(m) = num_constraints(m; count_variable_in_set_constraints=false)
    n_pairs(p) = p isa MP.MineCPITProblem ? 0 : length(p.pair_block)
    n_bins(p) = p isa MP.MineStockpileProblem ? p.n_bins : 0
    blocks_with_pairs(p) = p isa MP.MineCPITProblem ? 0 : length(unique(p.pair_block))
    mill_eligible(p) =
        p isa MP.MineCPITProblem ? copy(p.is_ore) :
        [b in Set(p.pair_block[j] for j in eachindex(p.pair_block) if p.pair_dest[j] == 1) for b in 1:length(p.blocks)]
    feed_capacity(p) = p isa MP.MineCPITProblem ? p.processing_capacity : p.mill_capacity
    min_feed(p) = p isa MP.MineCPITProblem ? p.min_processing : p.min_mill_feed

    function expected_rows(p)
        bm, T = p.blocks, p.n_periods
        B, A = length(bm), length(bm.arc_succ)
        base = B * (T - 1) + A * T + T          # chain + precedence + mining capacity
        if p isa MP.MineCPITProblem
            return base + T                      # milling row
        elseif p isa MP.MinePCPSPProblem
            mill = any(==(1), p.pair_dest) ? 4 : 0
            leach = any(==(2), p.pair_dest) ? 2 : 0
            return base + T * blocks_with_pairs(p) + T * (mill + leach)
        else
            return base + T * blocks_with_pairs(p) + T * (4 + p.n_bins)
        end
    end

    # The witness as a variable assignment of the built model.
    function witness_point(m, p)
        bm, T = p.blocks, p.n_periods
        wit = p.feasible_witness
        point = Dict{VariableRef, Float64}()
        x = m[:x]
        for b in 1:length(bm), t in 1:T
            point[x[b, t]] = (wit.mining_period[b] != 0 && wit.mining_period[b] <= t) ? 1.0 : 0.0
        end
        if !(p isa MP.MineCPITProblem)
            y = m[:y]
            for j in eachindex(p.pair_block), t in 1:T
                b = p.pair_block[j]
                chosen = wit.destination[b] == p.pair_dest[j] && wit.mining_period[b] == t
                point[y[j, t]] = chosen ? 1.0 : 0.0
            end
        end
        if p isa MP.MineStockpileProblem
            for s in 1:p.n_bins, t in 1:T
                point[m[:r][s, t]] = wit.reclaim[s, t]
                point[m[:inv][s, t]] = wit.inventory[s, t]
            end
        end
        return point
    end

    # --- registry-level sizing --------------------------------------------------
    for v in variants, target in (50, 200, 1000, 5000), status in (feasible, infeasible, unknown), seed in 0:1
        m, p = generate_problem(mp_ref(v), target, status, seed)
        bm, T = p.blocks, p.n_periods
        B = length(bm)
        @test 3 <= T <= 24
        @test num_variables(m) == T * (B + n_pairs(p) + 2 * n_bins(p))
        @test n_rows(m) == expected_rows(p)
        if v == :cpit
            @test B == max(4, round(Int, target / T))
            @test abs(num_variables(m) - target) <= T
        else
            # Prefix sizing: each block adds T * (1 + its destinations).
            @test abs(num_variables(m) - target) <= 3 * T
        end
        @test abs(num_variables(m) - target) <= 0.1 * target || target <= 50
        # Rows grow with the model (chain + slope-pattern precedence per block
        # variable): never a wide-thin LP.
        @test n_rows(m) >= num_variables(m) || target < 1000
    end

    # Large targets: constructor-only sizing (no model build) stays exact.
    for v in variants
        p = generate_problem(mp_ref(v), 100_000, feasible, 4)[2]
        nv = p.n_periods * (length(p.blocks) + n_pairs(p) + 2 * n_bins(p))
        @test abs(nv - 100_000) <= 3 * p.n_periods
    end
    big = MP.MineCPITProblem(1_000_000, feasible, 0)
    @test abs(length(big.blocks) * big.n_periods - 1_000_000) <= big.n_periods

    # Sizing cap: above the documented limit the request is rejected.
    cap = MP.MINE_PLANNING_MAX_VARIABLES
    @test cap == 1_000_000
    @test_throws ArgumentError MP.MineCPITProblem(cap + 1, unknown, 0)
    @test_throws ArgumentError MP.MinePCPSPProblem(cap + 1, unknown, 0)
    @test_throws ArgumentError MP.MineStockpileProblem(cap + 1, unknown, 0)
    @test_throws ArgumentError generate_problem(:mine_planning, cap + 1, unknown, 0)

    # --- block model and precedence invariants ---------------------------------
    for v in variants, target in (80, 700, 6000), seed in 0:2
        _, p = generate_problem(mp_ref(v), target, unknown, seed)
        bm = p.blocks
        B = length(bm)
        econ = p.economics
        @test bm.pattern in (:five, :nine)
        @test all(1 .<= bm.i .<= bm.nx) && all(1 .<= bm.j .<= bm.ny) && all(1 .<= bm.z .<= bm.nz)
        @test allunique(zip(bm.i, bm.j, bm.z))
        # Arcs: sorted by successor, unique, predecessor earlier in the
        # topological order and exactly one bench higher (so no arc is implied
        # by a longer path: the arc set is transitively reduced).
        @test issorted(bm.arc_succ)
        @test allunique(zip(bm.arc_succ, bm.arc_pred))
        @test all(bm.arc_pred .< bm.arc_succ)
        @test all(bm.z[bm.arc_succ] .== bm.z[bm.arc_pred] .+ 1)
        offsets = Set(MP._mine_pattern_offsets(bm.pattern))
        @test all((bm.i[a] - bm.i[b], bm.j[a] - bm.j[b]) in offsets for (b, a) in zip(bm.arc_succ, bm.arc_pred))
        # Closure: every block below the surface has an arc to every in-grid
        # pattern position one bench up, i.e. the pit contains all of them.
        n_arcs = zeros(Int, B)
        for b in bm.arc_succ
            n_arcs[b] += 1
        end
        for b in 1:B
            expected = bm.z[b] == 1 ? 0 :
                count(1 <= bm.i[b] + di <= bm.nx && 1 <= bm.j[b] + dj <= bm.ny for (di, dj) in offsets)
            @test n_arcs[b] == expected
        end
        # Grounded data: positive tonnage and grades, depth-increasing mining
        # cost, oxide rock lighter than sulfide, arsenic in ppm range.
        @test all(>(0), bm.tonnage)
        @test all(>(0), bm.grade)
        @test all(>(0), bm.contaminant)
        unit = bm.mining_cost ./ bm.tonnage
        expected_unit = [
            (econ.mining_cost_surface + econ.mining_cost_per_bench * (bm.z[b] - 1)) * (bm.oxide[b] ? 0.9 : 1.0)
            for b in 1:B
        ]
        @test unit ≈ expected_unit rtol = 1e-12
        if any(bm.oxide) && !all(bm.oxide)
            @test sum(bm.tonnage[bm.oxide]) / count(bm.oxide) < sum(bm.tonnage[.!bm.oxide]) / count(.!bm.oxide)
        end
        # Mill-profitable ore is a minority, as in real deposits.
        ore = [bm.grade[b] > MP._mine_mill_cutoff(econ, bm.oxide[b]) for b in 1:B]
        @test 1 <= count(ore) <= 0.6 * B
        if v == :cpit
            @test p.is_ore == ore
        end
    end

    # --- planted witnesses: every row of the built MIP, no solver --------------
    for v in variants, target in (60, 400, 3000), seed in 0:2
        m, p = generate_problem(mp_ref(v), target, feasible, seed; relax_integer=false)
        wit = p.feasible_witness
        @test wit !== nothing
        @test p.infeasibility_certificate === nothing
        mined = wit.mining_period .!= 0
        k = count(mined)
        @test k >= 1
        @test all(mined[1:k]) && !any(mined[(k + 1):end])          # a prefix: closed
        @test issorted(wit.mining_period[1:k])                       # shells in order
        @test all(0 .<= wit.mining_period .<= p.n_periods)
        # Integral, satisfies integrality and every linear row.
        report = primal_feasibility_report(m, witness_point(m, p); atol=1e-6)
        @test isempty(report)
        # Planted margins: the plan uses at most 97% of each capacity.
        T = p.n_periods
        w = p.blocks.tonnage
        for t in 1:T
            mined_t = sum(w[b] for b in 1:length(w) if wit.mining_period[b] == t; init=0.0)
            @test mined_t <= p.mining_capacity[t] / 1.03 + 1e-9
        end
    end

    # --- closure-flow certificates: recomputed from the struct fields ----------
    modes = Dict(v => Set{Symbol}() for v in variants)
    for v in variants, target in (60, 300, 2000), seed in 0:5
        _, p = generate_problem(mp_ref(v), target, infeasible, seed)
        cert = p.infeasibility_certificate
        @test cert !== nothing
        @test p.feasible_witness === nothing
        push!(modes[v], cert.mode)
        bm = p.blocks
        B, T, k = length(bm), p.n_periods, cert.periods
        w = bm.tonnage
        elig = mill_eligible(p)
        @test cert.mode in (:ramp_up, :exhaustion, :head_grade)
        @test cert.mode == :exhaustion ? k == T : 1 <= k <= min(3, T - 1)
        @test cert.mining_budget ≈ sum(p.mining_capacity[1:k]) rtol = 1e-12
        # Weights: mill-eligible tonnage, or tonnage-weighted grade excess
        # over the threshold in head-grade mode.
        if cert.mode == :head_grade
            @test v != :cpit
            @test cert.weights ≈ [elig[b] ? w[b] * max(bm.grade[b] - cert.grade_threshold, 0.0) : 0.0 for b in 1:B]
            @test cert.feed_multiplier ≈ p.head_grade_min - cert.grade_threshold rtol = 1e-12
            @test cert.feed_multiplier > 0
            @test p.head_grade_min <= 0.8 * maximum(bm.grade[elig]) + 1e-12
        else
            @test cert.weights == [elig[b] ? w[b] : 0.0 for b in 1:B]
            @test cert.feed_multiplier == 1.0
        end
        # Requirement: the multiplier times the contracted feed of periods 1..k.
        @test cert.requirement ≈ cert.feed_multiplier * sum(min_feed(p)[1:k]) rtol = 1e-9
        # The flow: nonnegative, within terminal capacities, and conserved at
        # every block (inflow from successors' arcs and the source equals
        # outflow to predecessors and the sink).
        c = cert.weights .- cert.lambda .* w
        scale = max(1.0, maximum(abs, c))
        tol = 1e-7 * scale
        @test cert.lambda >= 0
        @test all(cert.arc_flow .>= -tol)
        @test all(-tol .<= cert.source_flow .<= max.(c, 0.0) .+ tol)
        @test all(-tol .<= cert.sink_flow .<= max.(-c, 0.0) .+ tol)
        net = cert.source_flow .- cert.sink_flow
        for (kk, (b, a)) in enumerate(zip(bm.arc_succ, bm.arc_pred))
            net[b] -= cert.arc_flow[kk]
            net[a] += cert.arc_flow[kk]
        end
        @test maximum(abs, net) <= 1e-6 * scale
        bound = cert.lambda * cert.mining_budget + sum(c[b] - cert.source_flow[b] for b in 1:B if c[b] > 0; init=0.0)
        @test bound ≈ cert.bound rtol = 1e-9 atol = 1e-9
        # The contradiction, with the planted >= 10% margin.
        @test cert.requirement >= 1.1 * cert.bound - 1e-9
        @test cert.bound >= -1e-9
        # No single row is contradictory: every per-period minimum feed stays
        # at or below 95% of its mill capacity.
        @test all(min_feed(p) .<= 0.95 .* feed_capacity(p) .+ 1e-9)
    end
    @test :ramp_up in modes[:cpit]
    @test :head_grade in modes[:pcpsp] || :head_grade in modes[:stockpile]
    @test any(:exhaustion in modes[v] for v in variants)

    # Unknown: natural instances, no planted claim.
    for v in variants, seed in 0:2
        _, p = generate_problem(mp_ref(v), 300, unknown, seed)
        @test p.feasible_witness === nothing
        @test p.infeasibility_certificate === nothing
        @test all(min_feed(p) .<= 0.95 .* feed_capacity(p) .+ 1e-9)
    end

    # --- reproducibility, including isolation from the global RNG -----------------
    for v in variants, status in (feasible, infeasible, unknown)
        Random.seed!(987)
        m1, p1 = generate_problem(mp_ref(v), 250, status, 42)
        Random.seed!(12345)
        m2, p2 = generate_problem(mp_ref(v), 250, status, 42)
        for f in fieldnames(MP.MineBlockModel)
            @test isequal(getfield(p1.blocks, f), getfield(p2.blocks, f))
        end
        @test feed_capacity(p1) == feed_capacity(p2)
        @test min_feed(p1) == min_feed(p2)
        @test p1.mining_capacity == p2.mining_capacity
        @test sprint(print, m1) == sprint(print, m2)
    end

    if HAS_HIGHS
        # End-to-end feasibility contract across variants, sizes and seeds.
        for v in variants, target in (100, 600), status in (feasible, infeasible), seed in 0:3
            m, _ = generate_problem(mp_ref(v), target, status, seed)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            @test termination_status(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
        end

        # The same contract through the framework's verify-and-retry path.
        for v in variants, status in (feasible, infeasible)
            m, _ = generate_problem(mp_ref(v), 300, status, 7; optimizer=HiGHS.Optimizer)
            @test num_variables(m) > 0
        end

        # `unknown` is genuinely two-sided.
        for v in variants
            outcomes = Set{MOI.TerminationStatusCode}()
            for seed in 0:19
                m, _ = generate_problem(mp_ref(v), 150, unknown, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                push!(outcomes, termination_status(m))
            end
            @test MOI.OPTIMAL in outcomes
            @test MOI.INFEASIBLE in outcomes
        end

        # The certificate bound is a valid upper bound on the LP it relaxes:
        # max weights'x over the precedence closure polytope within the
        # mining budget, solved directly.
        for v in variants, seed in 0:3
            _, p = generate_problem(mp_ref(v), 400, infeasible, seed)
            cert = p.infeasibility_certificate
            bm = p.blocks
            lp = Model(HiGHS.Optimizer)
            set_silent(lp)
            @variable(lp, 0 <= z[1:length(bm)] <= 1)
            for (b, a) in zip(bm.arc_succ, bm.arc_pred)
                @constraint(lp, z[b] <= z[a])
            end
            @constraint(lp, sum(bm.tonnage[b] * z[b] for b in 1:length(bm)) <= cert.mining_budget)
            @objective(lp, Max, sum(cert.weights[b] * z[b] for b in 1:length(bm)))
            optimize!(lp)
            @test termination_status(lp) == MOI.OPTIMAL
            @test objective_value(lp) <= cert.bound * (1 + 1e-7) + 1e-6
        end
    end
end
