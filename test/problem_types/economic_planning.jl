# Focused quality contracts for the economic_planning category: registry shape,
# exact sizing formulas and caps, productivity of the input-output table
# (Hawkins-Simon, spectral radius, nonnegative Leontief inverse), data
# invariants, witness arithmetic against every row of the built model, Farkas
# certificates re-derived from the built model's rows and bounds (no solver),
# two-sided `unknown`, reproducibility, and HiGHS feasibility contracts.
using SparseArrays

@testset "Economic Planning" begin
    SL = SyntheticLPs
    @test :economic_planning in list_categories()
    @test Set(list_variants(:economic_planning)) == Set([:dynamic_leontief, :energy_system])
    info = problem_info(:economic_planning)
    @test info[:default_variant] == :dynamic_leontief
    DL = ProblemVariant(:economic_planning, :dynamic_leontief)
    ES = ProblemVariant(:economic_planning, :energy_system)

    # Generic Farkas evaluation on a built model. `mult` maps constraint refs to
    # multipliers (>= 0 on >= rows, <= 0 on <= rows, free on equalities); every
    # feasible x satisfies y'Ax >= y'b, while y'Ax <= Σ_j max over x_j's bounds of
    # (y'A)_j x_j. Returns (y'b, that maximum); the system is infeasible when the
    # first exceeds the second. A positive aggregated coefficient on a column
    # without an upper bound makes the maximum infinite.
    function ep_farkas(mult::Dict)
        r = Dict{VariableRef, Float64}()
        yb = 0.0
        for (con, y) in mult
            y == 0 && continue
            obj = constraint_object(con)
            set = obj.set
            b = if set isa MOI.LessThan
                @test y <= 0
                set.upper
            elseif set isa MOI.GreaterThan
                @test y >= 0
                set.lower
            else
                set.value
            end
            yb += y * b
            for (v, a) in obj.func.terms
                r[v] = get(r, v, 0.0) + y * a
            end
        end
        bmax = 0.0
        for (v, rj) in r
            abs(rj) <= 1e-9 && continue
            if rj > 0
                has_upper_bound(v) || return (yb, Inf)
                bmax += rj * upper_bound(v)
            else
                has_lower_bound(v) || return (yb, Inf)
                bmax += rj * lower_bound(v)
            end
        end
        return (yb, bmax)
    end

    # ---------------------------------------------------------------------
    # dynamic_leontief
    # ---------------------------------------------------------------------
    dl_count(p) = p.n_periods * (3 * p.n_sectors + 2 * length(p.tradable) + 2)
    function dl_witness(m, p)
        w = p.feasible_witness
        d = Dict{VariableRef, Float64}()
        for (sym, arr) in (
            (:x, w.output),
            (:N, w.new_capacity),
            (:K, w.capacity),
            (:imp, w.imports),
            (:ex, w.exports),
        )
            for idx in eachindex(m[sym])
                d[m[sym][idx]] = arr[idx]
            end
        end
        for t in eachindex(w.consumption)
            d[m[:C][t]] = w.consumption[t]
            d[m[:F][t]] = w.debt[t]
        end
        return d
    end
    function dl_mult(m, p)
        c = p.infeasibility_certificate
        d = Dict{Any, Float64}(m[:consumption_target] => 1.0)
        for t in 1:p.n_periods
            for s in 1:p.n_sectors
                d[m[:balance][s, t]] = c.balance_multipliers[s, t]
            end
            for i in eachindex(p.tradable)
                d[m[:import_limit][i, t]] = -c.import_multipliers[i, t]
            end
            for k in axes(c.labor_multipliers, 1)
                d[m[:labor][k, t]] = -c.labor_multipliers[k, t]
            end
        end
        for s in 1:p.n_sectors
            d[m[:capacity][s, 1]] = -c.capacity_multipliers[s]
        end
        return d
    end

    @testset "dynamic_leontief sizing" begin
        for target in (50, 100, 400, 1500, 6000),
            status in (feasible, infeasible, unknown),
            seed in 0:2

            m, p = generate_problem(DL, target, status, seed)
            T, n, ntr = p.n_periods, p.n_sectors, length(p.tradable)
            nk = size(p.labor_coef, 1)
            @test num_variables(m) == dl_count(p)
            @test num_constraints(m; count_variable_in_set_constraints=false) ==
                T * (3n + ntr + nk + 1) + T
            @test abs(num_variables(m) - target) <= max(2T, 0.05 * target)
            @test 3 <= T <= 25
            @test 2 <= ntr <= n - 1
        end
        cap = SL.LEONTIEF_MAX_VARIABLES
        @test cap == 1_000_000
        @test_throws ArgumentError SL.DynamicLeontiefProblem(cap + 1, unknown, 0)
        @test_throws ArgumentError generate_problem(DL, cap + 1, unknown, 0)
        big = SL.DynamicLeontiefProblem(cap, feasible, 0)
        @test abs(dl_count(big) - cap) <= 0.001 * cap
        @test big.n_sectors <= SL.LEONTIEF_MAX_SECTORS
        # Sparse at scale: suppliers per column grow only logarithmically.
        @test nnz(big.A) <= 3 * (6 + 2 * log(big.n_sectors)) * big.n_sectors
        m, p = generate_problem(DL, 100_000, feasible, 3)
        @test abs(num_variables(m) - 100_000) <= 100
    end

    @testset "dynamic_leontief productivity and data" begin
        ranges = Float64[]
        for target in (60, 300, 2000, 20000), seed in 0:3
            _, p = generate_problem(DL, target, unknown, seed)
            n = p.n_sectors
            price = p.unit_price
            @test all(>(0), price)
            @test all(>=(0), nonzeros(p.A))
            @test all(>=(0), nonzeros(p.B))
            # Hybrid units are a diagonal similarity of a value table whose column
            # sums (intermediate-input shares) are below one: productive.
            Ii, Jj, Vv = findnz(p.A)
            colsum = zeros(n)
            for (i, j, v) in zip(Ii, Jj, Vv)
                colsum[j] += price[i] * v / price[j]
            end
            @test maximum(colsum) <= 0.9 + 1e-9
            @test minimum(colsum) > 0
            push!(ranges, maximum(Vv) / minimum(Vv))
            if n <= 60
                IA = Matrix(I - Matrix(p.A))
                @test all(det(IA[1:k, 1:k]) > 0 for k in 1:n)          # Hawkins-Simon
                @test maximum(abs.(eigvals(Matrix(p.A)))) < 1          # spectral radius
                @test all(inv(IA) .>= -1e-12)                         # nonnegative Leontief inverse
            end
            @test issorted([
                findfirst(==(b), (:primary, :manufacturing, :construction, :services)) for
                b in p.sector_block
            ])
            @test p.tradable == collect(1:length(p.tradable))
            @test all(b in (:primary, :manufacturing) for b in p.sector_block[p.tradable])
            @test all(0 .< p.survival .< 1)
            @test all(0 .<= p.gestation_share .< 1)
            @test all(0 .< p.import_ceiling .< 1)
            @test all(>(0), sum(p.labor_coef; dims=1))
            @test sum(p.consumption_bundle .* price) ≈ 1 atol = 1e-9
            @test all(>=(0), p.government_demand)
            @test issorted(p.consumption_floor)
            # Every capital column is supplied by construction/equipment sectors.
            @test all(j -> !isempty(nzrange(p.B, j)), 1:n)
        end
        # Physical units make the coefficient range span many orders of magnitude.
        @test maximum(ranges) > 1e4
    end

    @testset "dynamic_leontief witness" begin
        for target in (50, 300, 1500, 5000), seed in 0:2
            m, p = generate_problem(DL, target, feasible, seed)
            w = p.feasible_witness
            @test w !== nothing
            @test p.infeasibility_certificate === nothing
            @test isempty(primal_feasibility_report(m, dl_witness(m, p); atol=1e-7))
            # Explicit balance arithmetic for one interior period.
            t = min(2, p.n_periods)
            x = w.output[:, t]
            lhs = x .- p.A * x .- p.consumption_bundle .* w.consumption[t]
            lhs[p.tradable] .+= w.imports[:, t] .- w.exports[:, t]
            inv_goods = (1 .- p.gestation_share) .* w.new_capacity[:, t]
            t < p.n_periods && (inv_goods .+= p.gestation_share .* w.new_capacity[:, t + 1])
            lhs .-= p.B * inv_goods
            @test maximum(
                abs.(lhs .- p.government_demand[:, t]) ./
                (abs.(p.government_demand[:, t]) .+ x .+ 1e-9),
            ) < 1e-8
            @test all(w.consumption .>= p.consumption_floor)
            @test dot(p.consumption_weight, w.consumption) >= p.consumption_target
            @test all(w.output[:, 1] .<= p.initial_capacity)
        end
    end

    @testset "dynamic_leontief certificate" begin
        modes = Set{Symbol}()
        for target in (50, 300, 1500, 5000), seed in 0:4
            m, p = generate_problem(DL, target, infeasible, seed)
            c = p.infeasibility_certificate
            @test c !== nothing
            @test p.feasible_witness === nothing
            push!(modes, c.mode)
            @test all(>=(0), c.balance_multipliers)
            @test c.import_multipliers == c.balance_multipliers[p.tradable, :]
            @test all(>=(0), c.labor_multipliers)
            @test all(>=(0), c.capacity_multipliers)
            @test (c.mode == :labor_capital) == any(>(0), c.capacity_multipliers)
            @test c.consumption_target == p.consumption_target
            ρ̃ = zeros(p.n_sectors)
            ρ̃[p.tradable] .= p.import_ceiling
            total = 0.0
            for t in 1:p.n_periods
                piv = c.balance_multipliers[:, t]
                # M'π_t <= Σ_k μ_kt ℓ_k + ν (ν only in period 1), M = I + diag(ρ̃) - A.
                # Tolerance: the floating-point scale of the terms being combined
                # (π, A'π >= 0), since entries with zero cover cancel exactly.
                upstream = transpose(p.A) * piv
                u = piv .+ ρ̃ .* piv .- upstream
                cover = vec(transpose(c.labor_multipliers[:, t]) * p.labor_coef)
                t == 1 && (cover .+= c.capacity_multipliers)
                @test all(u .<= cover .+ 1e-10 .* ((1 .+ ρ̃) .* piv .+ upstream .+ cover))
                # Scaled so the consumption column cancels against the target row.
                @test dot(piv, p.consumption_bundle) ≈ p.consumption_weight[t] rtol = 1e-9
                resource =
                    dot(c.labor_multipliers[:, t], p.labor_supply[:, t]) +
                    (t == 1 ? dot(c.capacity_multipliers, p.initial_capacity) : 0.0)
                @test (resource - dot(piv, p.government_demand[:, t])) / p.consumption_weight[t] ≈
                    c.consumption_bounds[t] rtol = 1e-8
                total += p.consumption_weight[t] * c.consumption_bounds[t]
            end
            @test p.consumption_target >= 1.1 * total * (1 - 1e-9)
            # Re-derived from the built model's rows and bounds.
            yb, bmax = ep_farkas(dl_mult(m, p))
            @test isfinite(bmax)
            @test yb > bmax
            @test yb - bmax ≈ p.consumption_target - total rtol = 1e-6
        end
        @test modes == Set([:labor, :labor_capital])
        for seed in 0:5
            _, p = generate_problem(DL, 400, unknown, seed)
            @test p.feasible_witness === nothing
            @test p.infeasibility_certificate === nothing
        end
    end

    @testset "dynamic_leontief reproducibility" begin
        for status in (feasible, infeasible, unknown)
            Random.seed!(1)
            _, p1 = generate_problem(DL, 900, status, 17)
            Random.seed!(2)
            _, p2 = generate_problem(DL, 900, status, 17)
            for f in fieldnames(typeof(p1))
                v1, v2 = getfield(p1, f), getfield(p2, f)
                if v1 isa SL.LeontiefWitness || v1 isa SL.LeontiefCertificate
                    @test all(getfield(v1, g) == getfield(v2, g) for g in fieldnames(typeof(v1)))
                else
                    @test isequal(v1, v2)
                end
            end
        end
    end

    # ---------------------------------------------------------------------
    # energy_system
    # ---------------------------------------------------------------------
    es_count(p) = (L=SL._es_layout(p); L.n_act + 2 * L.n_capt + L.n_flow)
    function es_witness(m, p)
        w = p.feasible_witness
        d = Dict{VariableRef, Float64}()
        for (sym, arr) in ((:act, w.act), (:cap, w.cap), (:ncap, w.ncap), (:flow, w.flow))
            for (i, v) in enumerate(m[sym])
                d[v] = arr[i]
            end
        end
        return d
    end
    function es_mult(m, p)
        c = p.infeasibility_certificate
        R, T, S = p.n_regions, p.n_periods, p.n_slices
        NC = length(p.commodities)
        d = Dict{Any, Float64}()
        add!(con, y) = (d[con] = get(d, con, 0.0) + y)
        for (i, t) in enumerate(c.periods)
            w = c.period_weight[i]
            for r in 1:R
                for ci in 1:NC
                    add!(m[:bal][((r - 1) * NC + ci - 1) * T + t], w * c.commodity_value[r, ci])
                end
                for s in 1:S
                    add!(m[:bal_elc][((r - 1) * T + t - 1) * S + s], w * c.electricity_value[r])
                end
            end
            c.mode == :emission_cap && add!(m[:emission][t], -c.emission_weight * w)
        end
        c.mode == :carbon_budget && add!(m[:budget], -c.emission_weight)
        for (col, y) in c.capact_multipliers
            add!(m[:capact][col], y)
        end
        for (col, y) in c.transfer_multipliers
            add!(m[:transfer][col], y)
        end
        for (col, y) in c.growth_multipliers
            add!(m[:growth][col], y)
        end
        return d
    end

    @testset "energy_system sizing" begin
        for target in (50, 100, 300, 1000, 3000, 10000),
            status in (feasible, infeasible, unknown),
            seed in 0:2

            m, p = generate_problem(ES, target, status, seed)
            L = SL._es_layout(p)
            R, T, S = p.n_regions, p.n_periods, p.n_slices
            NC = length(p.commodities)
            @test num_variables(m) == es_count(p)
            @test length(p.lines) == SL._es_n_lines(R)
            n_reserve = count(isfinite, p.supply_reserve)
            n_capact = sum(T * L.slots[k] for k in eachindex(p.tech_name) if L.cap_index[k] > 0)
            n_cap_tech = count(>(0), L.cap_index)
            @test num_constraints(m; count_variable_in_set_constraints=false) ==
                R * NC * T +
                  R * T * S +
                  n_capact +
                  L.n_capt +
                  n_cap_tech * (T - 1) +
                  R * T +
                  T +
                  1 +
                  n_reserve
            @test abs(num_variables(m) - target) <= max(0.005 * target, target < 300 ? 2 : 0)
        end
        cap = SL.ENERGY_SYSTEM_MAX_VARIABLES
        @test cap == 1_000_000
        @test_throws ArgumentError SL.EnergySystemProblem(cap + 1, unknown, 0)
        @test_throws ArgumentError generate_problem(ES, cap + 1, unknown, 0)
        big = SL.EnergySystemProblem(cap, unknown, 0)
        @test abs(es_count(big) - cap) <= 0.005 * cap
        m, p = generate_problem(ES, 100_000, feasible, 3)
        @test abs(num_variables(m) - 100_000) <= 500
    end

    @testset "energy_system data" begin
        for target in (80, 600, 4000, 20000), seed in 0:2
            _, p = generate_problem(ES, target, unknown, seed)
            L = SL._es_layout(p)
            @test sum(p.slice_duration) ≈ 1
            @test all(sum(p.profiles; dims=2) .≈ 1)
            @test all(0 .<= p.tech_avail .<= 0.98 + 1e-12)
            @test all(>=(0), p.tech_input_coef)
            @test all(>=(0), p.tech_emission)
            @test all(>=(0), p.demand)
            for k in eachindex(p.tech_name)
                p.tech_kind[k] == :supply && continue
                @test p.tech_life[k] >= 1
                @test p.tech_growth[k] > 1
                @test all(p.tech_residual[k, :] .<= p.tech_potential[k] + 1e-9)
                p.tech_kind[k] == :gen && @test p.tech_capfac[k] == SL.ES_PJ_PER_GW_YEAR
                # Combustion plants and devices emit by fuel carbon content.
                if p.tech_input[k] > 0 &&
                    p.commodities[p.tech_input[k]] in keys(SL.ES_EMISSION_FACTOR)
                    @test p.tech_emission[k] ≈
                        SL.ES_EMISSION_FACTOR[p.commodities[p.tech_input[k]]] *
                          p.tech_input_coef[k]
                end
            end
            # Interconnectors form a connected network over the regions.
            R = p.n_regions
            seen = falses(R)
            seen[1] = true
            stack = [1]
            while !isempty(stack)
                a = pop!(stack)
                for (u, v) in p.lines
                    for (x, y) in ((u, v), (v, u))
                        if x == a && !seen[y]
                            seen[y] = true
                            push!(stack, y)
                        end
                    end
                end
            end
            @test all(seen)
            @test all(0.9 .<= p.line_efficiency .<= 0.995)
        end
    end

    @testset "energy_system witness" begin
        for target in (50, 300, 1500, 6000), seed in 0:2
            m, p = generate_problem(ES, target, feasible, seed)
            w = p.feasible_witness
            @test w !== nothing
            @test p.infeasibility_certificate === nothing
            @test all(>=(0), w.act)
            @test all(iszero, w.flow)
            @test isempty(primal_feasibility_report(m, es_witness(m, p); atol=1e-7))
        end
    end

    @testset "energy_system certificate" begin
        modes = Set{Symbol}()
        for target in (50, 300, 1500, 6000), seed in 0:5
            m, p = generate_problem(ES, target, infeasible, seed)
            c = p.infeasibility_certificate
            @test c !== nothing
            @test p.feasible_witness === nothing
            push!(modes, c.mode)
            @test all(>=(0), c.commodity_value)
            @test all(>=(0), c.electricity_value)
            @test c.emission_weight == (c.mode == :supply_shortfall ? 0.0 : 1.0)
            # Planted margin: the certified minimum exceeds the limit by >= 15%.
            lb = c.demand_value - c.available_value
            if c.mode == :supply_shortfall
                @test lb > 0
            else
                @test lb >= 1.15 * c.emission_limit * (1 - 1e-9)
                @test c.emission_limit > 0
            end
            # Re-derived from the built model's rows and bounds.
            yb, bmax = ep_farkas(es_mult(m, p))
            @test isfinite(bmax)
            @test yb > bmax
            @test isapprox(
                yb - bmax, lb - c.emission_weight * c.emission_limit; rtol=1e-6, atol=1e-8 * abs(yb)
            )
        end
        @test modes == Set([:emission_cap, :carbon_budget, :supply_shortfall])
    end

    @testset "energy_system reproducibility" begin
        for status in (feasible, infeasible, unknown)
            Random.seed!(3)
            _, p1 = generate_problem(ES, 2500, status, 11)
            Random.seed!(4)
            _, p2 = generate_problem(ES, 2500, status, 11)
            for f in fieldnames(typeof(p1))
                v1, v2 = getfield(p1, f), getfield(p2, f)
                if v1 isa SL.EnergySystemWitness || v1 isa SL.EnergySystemCertificate
                    @test all(
                        isequal(getfield(v1, g), getfield(v2, g)) for g in fieldnames(typeof(v1))
                    )
                else
                    @test isequal(v1, v2)
                end
            end
        end
    end

    @testset "economic_planning HiGHS contracts" begin
        if HAS_HIGHS
            for ref in (DL, ES),
                target in (60, 400, 1500), status in (feasible, infeasible),
                seed in 0:2

                m, _ = generate_problem(ref, target, status, seed)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
            end
            # Through the framework's verify-and-retry path.
            for ref in (DL, ES), status in (feasible, infeasible)
                m, _ = generate_problem(ref, 500, status, 1; optimizer=HiGHS.Optimizer)
                @test num_variables(m) > 0
            end
            # `unknown` lands on both sides of the frontier.
            for (ref, target) in ((DL, 400), (ES, 3000))
                outcomes = Set{MOI.TerminationStatusCode}()
                for seed in 0:11
                    m, _ = generate_problem(ref, target, unknown, seed)
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
