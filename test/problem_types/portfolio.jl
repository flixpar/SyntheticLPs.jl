# Focused quality contracts for the portfolio category: shared factor-market
# helpers, exact sizing, bounded nonzeros at scale, planted witnesses and typed
# infeasibility certificates (checked arithmetically), reproducibility, and
# HiGHS-backed feasibility contracts.
using SparseArrays

function _portfolio_test_nnz(model)
    total = 0
    for (F, S) in list_of_constraint_types(model)
        F <: AffExpr || continue
        for c in all_constraints(model, F, S)
            total += length(constraint_object(c).func.terms)
        end
    end
    return total
end

@testset "Portfolio Helpers" begin
    rng = MersenneTwister(5)
    # Water-filling respects every cap exactly and sums to one (the old
    # clip-then-renormalize reference violated caps).
    for _ in 1:200
        n = rand(rng, 3:40)
        target = rand(rng, n) .^ 3
        caps = rand(rng, n) .* (3 / n) .+ 1e-3
        sum(caps) <= 1.05 && (caps .*= 1.2 / sum(caps))
        x = SyntheticLPs._portfolio_waterfill(target, caps)
        @test sum(x) ≈ 1.0 atol = 1e-10
        @test all(x .<= caps .+ 1e-12)
        @test all(x .>= 0)
    end
    # Exact CVaR versus a brute-force scan over every breakpoint.
    for _ in 1:50
        losses = randn(rng, rand(rng, 5:60))
        beta = rand(rng, (0.8, 0.9, 0.95))
        c = 1 / ((1 - beta) * length(losses))
        brute = minimum(a + c * sum(max.(losses .- a, 0)) for a in losses)
        value, alpha = SyntheticLPs._portfolio_cvar(losses, beta)
        @test value ≈ brute
        @test alpha + c * sum(max.(losses .- alpha, 0)) ≈ value
    end
    # The Lagrangian floor bound is a valid lower bound (and tight here) on
    # the capped simplex with group floors.
    if HAS_HIGHS
        for _ in 1:20
            n = rand(rng, 4:25)
            costs = randn(rng, n)
            caps = (rand(rng, n) .+ 0.2) .* (3 / n)
            groups = [mod1(i, 3) for i in 1:n]
            floors = rand(rng, 3) .* 0.25
            bound, _, μ = SyntheticLPs._portfolio_floor_bound(costs, caps, groups, floors)
            @test all(>=(0), μ)
            lp = Model(HiGHS.Optimizer)
            set_silent(lp)
            @variable(lp, 0 <= x[i=1:n] <= caps[i])
            @constraint(lp, sum(x) == 1)
            for g in 1:3
                @constraint(lp, sum(x[i] for i in 1:n if groups[i] == g) >= floors[g])
            end
            @objective(lp, Min, costs' * x)
            optimize!(lp)
            if termination_status(lp) == MOI.OPTIMAL
                @test bound <= objective_value(lp) + 1e-9
                @test bound ≈ objective_value(lp) atol = 1e-8
            end
        end
    end
    # Fractional-knapsack extreme over the capped simplex versus enumeration
    # of vertices in a tiny case.
    values = [0.3, -0.2, 0.1, 0.5]
    caps = [0.5, 0.4, 0.6, 0.3]
    @test SyntheticLPs._portfolio_extreme_on_capped_simplex(values, caps, :min) ≈ 0.4 * -0.2 + 0.6 * 0.1
    @test SyntheticLPs._portfolio_extreme_on_capped_simplex(values, caps, :max) ≈ 0.3 * 0.5 + 0.5 * 0.3 + 0.2 * 0.1
end

@testset "Portfolio Sizing and Scale" begin
    @test Set(list_variants(:portfolio)) == Set((:cvar, :tracking_error))
    @test ProblemVariant(:portfolio) == ProblemVariant(:portfolio, :cvar)
    for target in (40, 200, 1000, 5000), status in (feasible, infeasible, unknown)
        model, prob = generate_problem(:portfolio, target, status, 2; variant=:cvar)
        mk = prob.market
        K = SyntheticLPs._portfolio_n_factors(mk)
        @test num_variables(model) == 3 * mk.n_assets + K + mk.n_scenarios + 1 == target
        @test allequal(diff(mk.idiosyncratic.colptr))
        @test sum(mk.benchmark) ≈ 1 && all(>(0), mk.benchmark)

        target >= 60 || continue
        model, prob = generate_problem(:portfolio, target, status, 2; variant=:tracking_error)
        mk = prob.market
        K = SyntheticLPs._portfolio_n_factors(mk)
        @test num_variables(model) == length(prob.investable) + K + mk.n_scenarios == target
        @test num_constraints(model; count_variable_in_set_constraints=false) == 2 * mk.n_scenarios + K + 2
    end
    for v in (:cvar, :tracking_error), target in (3, 20)
        model, _ = generate_problem(:portfolio, target, unknown, 1; variant=v)
        @test num_variables(model) > 0
    end
    for v in (:cvar, :tracking_error)
        elapsed = @elapsed model, _ = generate_problem(:portfolio, 100_000, feasible, 0; variant=v)
        @test num_variables(model) == 100_000
        @test _portfolio_test_nnz(model) <= 8_000_000
        @test elapsed < 60
    end
end

@testset "Portfolio CVaR Witness and Certificates" begin
    modes = Dict{Symbol, Int}()
    for seed in 1:30
        _, prob = generate_problem(:portfolio, 600, seed <= 6 ? feasible : infeasible, seed; variant=:cvar)
        mk = prob.market
        n = mk.n_assets
        ns = 1 + mk.n_styles
        if seed <= 6
            w = prob.feasible_witness
            @test w isa SyntheticLPs.CVaRWitness
            @test prob.infeasibility_certificate === nothing && prob.infeasibility_mode == :none
            x = w.weights
            @test sum(x) ≈ 1 atol = 1e-10
            @test all(0 .<= x .<= 0.9 .* prob.max_position .+ 1e-12)
            f = SyntheticLPs._portfolio_exposures(mk, x)
            @test f ≈ w.exposures
            @test all(prob.exposure_lower .< f[1:ns] .< prob.exposure_upper)
            @test all(f[(ns + 1):end] .<= prob.sector_upper .+ 1e-12)
            regions = SyntheticLPs._portfolio_group_sums(x, prob.region, length(prob.region_upper))
            @test all(regions .<= prob.region_upper .+ 1e-12)
            classes = SyntheticLPs._portfolio_group_sums(x, prob.asset_class, length(prob.class_lower))
            @test all(prob.class_lower .- 1e-12 .<= classes .<= prob.class_upper .+ 1e-12)
            @test sum(abs.(x .- mk.benchmark)) ≈ w.turnover
            @test w.turnover < prob.turnover_limit
            losses = -SyntheticLPs._portfolio_scenario_returns(mk, x)
            cvar, _ = SyntheticLPs._portfolio_cvar(losses, prob.cvar_level)
            @test cvar ≈ w.cvar
            c = 1 / ((1 - prob.cvar_level) * mk.n_scenarios)
            @test w.alpha + c * sum(max.(losses .- w.alpha, 0)) ≈ w.cvar
            @test w.cvar < prob.cvar_limit
        else
            @test prob.feasible_witness === nothing
            cert = prob.infeasibility_certificate
            modes[prob.infeasibility_mode] = get(modes, prob.infeasibility_mode, 0) + 1
            if prob.infeasibility_mode == :crash_tail
                @test cert isa SyntheticLPs.CVaRTailCertificate
                @test length(cert.tail) == ceil(Int, (1 - prob.cvar_level) * mk.n_scenarios)
                # Mean tail loss per asset recomputed from the market model.
                unit = zeros(n)
                for i in (1, n)
                    unit .= 0
                    unit[i] = 1
                    r = SyntheticLPs._portfolio_scenario_returns(mk, unit)
                    @test -sum(r[cert.tail]) / length(cert.tail) ≈ cert.asset_tail_loss[i]
                end
                # Lagrangian bound recomputed from the stored multipliers.
                λ = cert.budget_multiplier
                μ = cert.class_multipliers
                @test all(>=(0), μ)
                bound = λ + sum(μ .* prob.class_lower)
                for i in 1:n
                    bound += min(0.0, cert.asset_tail_loss[i] - λ - μ[prob.asset_class[i]]) * prob.max_position[i]
                end
                @test bound ≈ cert.loss_bound
                @test cert.cvar_limit == prob.cvar_limit <= 0.85 * cert.loss_bound
            elseif prob.infeasibility_mode == :class_floor
                @test cert isa SyntheticLPs.ClassFloorCertificate
                @test sum(prob.class_lower) ≈ cert.floor_sum
                @test cert.floor_sum >= 1.05 - 1e-12
            else
                @test cert isa SyntheticLPs.TurnoverSectorCertificate
                bench_sector = SyntheticLPs._portfolio_exposures(mk, mk.benchmark)[(ns + 1):end]
                @test sum(bench_sector[g] - prob.sector_upper[g] for g in cert.sectors) ≈ cert.deficit
                @test cert.turnover_limit == prob.turnover_limit <= 0.85 * 2 * cert.deficit
            end
        end
    end
    @test Set(keys(modes)) == Set((:crash_tail, :class_floor, :turnover_sector))
    @test modes[:crash_tail] >= 8
end

@testset "Portfolio Tracking Witness and Certificate" begin
    for seed in 1:5
        _, prob = generate_problem(:portfolio, 800, feasible, seed; variant=:tracking_error)
        mk = prob.market
        w = prob.feasible_witness
        @test w isa SyntheticLPs.TrackingErrorWitness
        x = zeros(mk.n_assets)
        x[prob.investable] .= w.weights
        @test sum(x) ≈ 1 atol = 1e-10
        @test all(x[prob.investable] .<= 0.9 .* prob.max_position[prob.investable] .+ 1e-12)
        f = SyntheticLPs._portfolio_exposures(mk, x)
        @test f ≈ w.exposures
        @test all(prob.exposure_lower .<= f .<= prob.exposure_upper)
        te = SyntheticLPs._tracking_error(mk, x, prob.benchmark_returns)
        @test te ≈ w.tracking_error
        @test te < prob.te_budget
        @test prob.benchmark_returns ≈ SyntheticLPs._portfolio_scenario_returns(mk, mk.benchmark)

        _, bad = generate_problem(:portfolio, 800, infeasible, seed; variant=:tracking_error)
        cert = bad.infeasibility_certificate
        @test cert isa SyntheticLPs.TrackingErrorCertificate
        energy = findall(==(bad.energy_sector), bad.market.sector)
        @test issubset(energy, bad.excluded)
        @test maximum(bad.market.style_loadings[bad.investable, cert.factor]) ≈ cert.max_investable_loading
        @test cert.band_lower == bad.exposure_lower[cert.factor]
        @test cert.band_lower > cert.max_investable_loading + 1e-3
        # The factor row alone is satisfiable under the caps (no single-row
        # presolve contradiction); only the budget row exposes it.
        row_max = sum(max(bad.market.style_loadings[i, cert.factor], 0.0) * bad.max_position[i] for i in bad.investable)
        @test row_max > cert.band_lower
        @test !isempty(prob.excluded)
    end
end

@testset "Portfolio Reproducibility" begin
    for v in (:cvar, :tracking_error), status in (feasible, infeasible, unknown)
        m1, p1 = generate_problem(:portfolio, 400, status, 9; variant=v)
        m2, p2 = generate_problem(:portfolio, 400, status, 9; variant=v)
        @test p1.market.factor_returns == p2.market.factor_returns
        @test p1.market.idiosyncratic == p2.market.idiosyncratic
        mktempdir() do dir
            a = joinpath(dir, "a.mps")
            b = joinpath(dir, "b.mps")
            write_to_file(m1, a)
            write_to_file(SyntheticLPs.build_model(p2), b)
            @test read(a, String) == read(b, String)
        end
    end
end

@testset "Portfolio Feasibility Contracts" begin
    if HAS_HIGHS
        solve_status = function (model)
            set_optimizer(model, HiGHS.Optimizer)
            set_silent(model)
            set_time_limit_sec(model, 60.0)
            optimize!(model)
            return termination_status(model)
        end
        for v in (:cvar, :tracking_error), seed in 1:4
            model, _ = generate_problem(:portfolio, 500, feasible, seed; variant=v, optimizer=HiGHS.Optimizer)
            @test solve_status(model) == MOI.OPTIMAL
            model, _ = generate_problem(:portfolio, 500, infeasible, seed; variant=v, optimizer=HiGHS.Optimizer)
            @test solve_status(model) in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
        end
        # `unknown` is the natural mandate with no repair: both outcomes occur.
        for v in (:cvar, :tracking_error)
            outcomes = Set{MOI.TerminationStatusCode}()
            for seed in 1:10
                model, _ = generate_problem(:portfolio, 500, unknown, seed; variant=v)
                push!(outcomes, solve_status(model))
            end
            @test outcomes == Set((MOI.OPTIMAL, MOI.INFEASIBLE))
        end
    end
end
