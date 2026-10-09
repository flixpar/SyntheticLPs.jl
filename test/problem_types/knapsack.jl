# Focused quality contracts for the knapsack category: registry shape (the
# single-row `standard` knapsack is gone), exact sizing and row formulas that
# grow with the instance, bounded nnz, data invariants, witness and
# certificate arithmetic recomputed from the struct fields, reproducibility,
# and HiGHS contracts (including a genuinely mixed `unknown`).

kn_rows(m) = num_constraints(m; count_variable_in_set_constraints=false)

function kn_status(m)
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    optimize!(m)
    return termination_status(m)
end

@testset "Knapsack" begin
    @test :knapsack in list_categories()
    @test Set(list_variants(:knapsack)) ==
        Set([:multiple_choice, :multidimensional, :bounded, :mixed_integer_set])
    @test problem_info(:knapsack)[:default_variant] == :multiple_choice

    @testset "multiple_choice" begin
        for target in (1, 2, 7, 11, 60, 500, 4000), status in (feasible, infeasible, unknown)
            m, p = generate_problem("knapsack/multiple_choice", target, status, 1)
            @test num_variables(m) == length(p.values) == target
            @test kn_rows(m) == p.n_classes + 3 * p.n_clusters + 2
            # Classes partition the columns into consecutive blocks of 3-8
            # (tiny remainders aside).
            @test vcat(collect.(p.class_options)...) == collect(1:target)
            target >= 11 && @test all(3 <= length(r) <= 9 for r in p.class_options)
            @test p.n_clusters == clamp(round(Int, p.n_classes / 25), 1, p.n_classes)
            @test sort(unique(p.class_cluster)) == collect(1:p.n_clusters)
            @test all(>(0), p.local_usage) && all(>(0), p.shared_usage) && all(>(0), p.values)
        end
        _, big = generate_problem("knapsack/multiple_choice", 100_000, unknown, 0)
        @test length(big.values) == 100_000
        @test big.n_classes + 3 * big.n_clusters + 2 >= 0.15 * 100_000

        for target in (40, 900, 6000), seed in 0:2
            m, p = generate_problem("knapsack/multiple_choice", target, feasible, seed)
            w = p.feasible_witness
            @test w !== nothing && p.infeasibility_certificate === nothing
            @test [w.choice[g] in p.class_options[g] for g in 1:p.n_classes] == trues(p.n_classes)
            loc = SyntheticLPs._mck_totals(p.local_usage, w.choice, p.class_cluster, p.n_clusters)
            sh = vec(SyntheticLPs._mck_totals(p.shared_usage, w.choice, ones(Int, p.n_classes), 1))
            @test all(loc .* 1.02 .<= p.local_capacity)
            @test all(sh .* 1.01 .<= p.shared_capacity)
            point = Dict(v => 0.0 for v in all_variables(m))
            for j in w.choice
                point[m[:x][j]] = 1.0
            end
            @test isempty(primal_feasibility_report(m, point; atol=1e-6))
        end
        for target in (40, 900, 6000), seed in 0:2
            _, p = generate_problem("knapsack/multiple_choice", target, infeasible, seed)
            c = p.infeasibility_certificate
            @test c !== nothing && p.feasible_witness === nothing
            mins = [
                minimum(p.shared_usage[c.resource, j] for j in p.class_options[g]) for
                g in 1:p.n_classes
            ]
            @test c.min_usage ≈ mins
            @test c.min_usage_total ≈ sum(mins)
            @test c.capacity == p.shared_capacity[c.resource]
            @test c.min_usage_total >= 1.04 * c.capacity * (1 - 1e-12)
        end
    end

    @testset "multidimensional" begin
        for target in (1, 3, 30, 400, 5000), status in (feasible, infeasible, unknown)
            m, p = generate_problem("knapsack/multidimensional", target, status, 2)
            @test num_variables(m) == p.n_items == target
            @test p.n_local == clamp(round(Int, target / 12), 2, max(2, target))
            @test p.n_programs == clamp(round(Int, target / 40), 1, target)
            used = falses(p.n_local)
            for i in 1:target, k in 1:length(p.local_usage[i])
                used[SyntheticLPs._mdkp_resource(p.window_start[i], k, p.n_local)] = true
            end
            @test kn_rows(m) == count(used) + 3 + count(>(0), p.program_floor)
            @test all(2 <= length(u) <= 6 || length(u) == p.n_local for u in p.local_usage)
            @test all(0 .<= p.program_floor .<= [count(==(q), p.program) for q in 1:p.n_programs])
        end
        # Rows scale (the old variant had 3-5 rows at any size), nnz ~7/column.
        m, p = generate_problem("knapsack/multidimensional", 20_000, unknown, 0)
        @test kn_rows(m) >= 0.1 * 20_000
        @test sum(length, p.local_usage) + 3 * p.n_items + p.n_items <= 10 * p.n_items

        for target in (50, 800, 5000), seed in 0:2
            m, p = generate_problem("knapsack/multidimensional", target, feasible, seed)
            w = p.feasible_witness
            @test w !== nothing && p.infeasibility_certificate === nothing
            sel = falses(p.n_items)
            sel[w.selected] .= true
            loc = SyntheticLPs._mdkp_local_totals(sel, p.window_start, p.local_usage, p.n_local)
            @test all(loc .<= p.local_capacity)
            @test all(p.global_usage * sel .<= p.global_capacity)
            @test all(
                count(i -> sel[i] && p.program[i] == q, 1:p.n_items) >= p.program_floor[q] for
                q in 1:p.n_programs
            )
            point = Dict(m[:x][i] => (sel[i] ? 1.0 : 0.0) for i in 1:p.n_items)
            @test isempty(primal_feasibility_report(m, point; atol=1e-6))
        end
        for target in (50, 800, 5000), seed in 0:2
            _, p = generate_problem("knapsack/multidimensional", target, infeasible, seed)
            c = p.infeasibility_certificate
            @test c !== nothing && p.feasible_witness === nothing
            @test c.floors == p.program_floor
            lightest = sum(
                sum(
                    sort([
                        p.global_usage[c.resource, i] for i in 1:p.n_items if p.program[i] == q
                    ])[1:c.floors[q]];
                    init=0.0,
                ) for q in 1:p.n_programs
            )
            @test lightest ≈ c.lightest_sum rtol = 1e-9
            @test c.capacity == p.global_capacity[c.resource]
            @test c.lightest_sum >= 1.04 * c.capacity * (1 - 1e-12)
        end
    end

    @testset "bounded" begin
        for target in (1, 2, 9, 50, 700, 6000), status in (feasible, infeasible, unknown)
            m, p = generate_problem("knapsack/bounded", target, status, 3)
            @test num_variables(m) == length(p.item_of) == target
            @test p.n_knapsacks == clamp(round(Int, target / 28), 1, target)
            ncols = [count(==(i), p.item_of) for i in 1:p.n_items]
            item_rows = count(i -> p.commitment[i] > 0 || ncols[i] > 1, 1:p.n_items)
            @test kn_rows(m) == item_rows + length(unique(p.knapsack_of))
            # Lanes stay inside the item's region; each item has 1-5 lanes.
            @test all(
                p.knapsack_region[p.knapsack_of[j]] == p.item_region[p.item_of[j]] for j in 1:target
            )
            @test all(1 .<= ncols .<= 5)
            @test all(0 .<= p.commitment .<= p.stock)
            @test all(has_upper_bound(v) for v in all_variables(m))
        end
        for target in (60, 900, 6000), seed in 0:2
            m, p = generate_problem("knapsack/bounded", target, feasible, seed)
            w = p.feasible_witness
            @test w !== nothing && p.infeasibility_certificate === nothing
            q = w.quantities
            shipped = [
                sum(q[j] for j in eachindex(q) if p.item_of[j] == i; init=0) for i in 1:p.n_items
            ]
            @test all(p.commitment .<= shipped .<= p.stock)
            load = zeros(p.n_knapsacks)
            for j in eachindex(q)
                load[p.knapsack_of[j]] += p.unit_weight[j] * q[j]
            end
            @test all(load .<= p.capacity)
            point = Dict(m[:x][j] => Float64(q[j]) for j in eachindex(q))
            @test isempty(primal_feasibility_report(m, point; atol=1e-6))
        end
        for target in (60, 900, 6000), seed in 0:2
            _, p = generate_problem("knapsack/bounded", target, infeasible, seed)
            c = p.infeasibility_certificate
            @test c !== nothing && p.feasible_witness === nothing
            @test c.items == [i for i in 1:p.n_items if p.item_region[i] == c.region]
            @test c.min_unit_weight ≈ [
                minimum(p.unit_weight[j] for j in eachindex(p.item_of) if p.item_of[j] == i) for
                i in c.items
            ]
            @test c.committed_weight ≈ sum(p.commitment[c.items] .* c.min_unit_weight)
            @test c.region_capacity ≈
                sum(p.capacity[k] for k in 1:p.n_knapsacks if p.knapsack_region[k] == c.region)
            @test c.committed_weight >= 1.04 * c.region_capacity * (1 - 1e-9)
        end
    end

    @testset "mixed_integer_set" begin
        @test_nowarn generate_problem("knapsack/mixed_integer_set", 1, unknown, 1)
        for target in (1, 5, 80, 600, 3000), status in (feasible, infeasible, unknown)
            m, p = generate_problem("knapsack/mixed_integer_set", target, status, 1)
            @test num_variables(m) == p.n_integer + p.n_continuous == target
            @test kn_rows(m) == p.n_rows + 1
            @test length(p.row_indices) == p.n_rows
            @test all(length(p.row_indices[r]) == length(p.row_coefficients[r]) for r in 1:p.n_rows)
            @test all(
                allunique(p.row_indices[r]) && all(1 <= i <= target for i in p.row_indices[r]) for
                r in 1:p.n_rows
            )
            @test count(p.dense_rows) <= 16
            @test all(length(p.row_indices[r]) <= 24 for r in 1:p.n_rows if !p.dense_rows[r])
        end
        # nnz grows linearly: the old generator was O(n^2) (18.7 GB MPS at 50k).
        _, big = generate_problem("knapsack/mixed_integer_set", 100_000, feasible, 0)
        @test sum(length, big.row_indices) <= 30 * 100_000

        for target in (40, 500, 3000), seed in 0:2
            m, p = generate_problem("knapsack/mixed_integer_set", target, feasible, seed)
            point = Dict{VariableRef, Float64}()
            for i in 1:p.n_integer
                point[m[:integer_items][i]] = p.planted_integer[i]
            end
            for j in 1:p.n_continuous
                point[m[:continuous_items][j]] = p.planted_continuous[j]
            end
            @test isempty(primal_feasibility_report(m, point; atol=1e-6))
        end
        for target in (40, 500, 3000), seed in 0:2
            _, p = generate_problem("knapsack/mixed_integer_set", target, infeasible, seed)
            c = p.infeasibility_certificate
            @test c !== nothing
            @test all(>=(0), c.row_multipliers)
            upper = vcat(Float64.(p.integer_upper), p.continuous_upper)
            bound, _ = SyntheticLPs.mik_dual_bound(
                p.row_indices, p.row_coefficients, p.capacities, p.profits, upper, c.row_multipliers
            )
            @test bound ≈ c.dual_bound rtol = 1e-9
            @test p.minimum_profit >= 1.01 * bound * (1 - 1e-12)
            # Far below the box bound, so presolve's activity bounds cannot see it.
            @test bound < sum(p.profits .* upper)
        end
    end

    # Reproducibility, independent of the global RNG.
    for v in ("multiple_choice", "multidimensional", "bounded", "mixed_integer_set"),
        status in (feasible, infeasible, unknown)

        Random.seed!(1)
        _, p1 = generate_problem("knapsack/$v", 700, status, 9)
        Random.seed!(99)
        _, p2 = generate_problem("knapsack/$v", 700, status, 9)
        for f in fieldnames(typeof(p1))
            a, b = getfield(p1, f), getfield(p2, f)
            if a isa Union{Number, Symbol, AbstractArray, FeasibilityStatus, Nothing}
                @test isequal(a, b)
            end
        end
    end

    @testset "Knapsack HiGHS contracts" begin
        if HAS_HIGHS
            for v in ("multiple_choice", "multidimensional", "bounded", "mixed_integer_set")
                for target in (80, 1500), status in (feasible, infeasible), seed in 0:1
                    m, _ = generate_problem("knapsack/$v", target, status, seed)
                    @test kn_status(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
                end
                outcomes = Set{MOI.TerminationStatusCode}()
                for target in (300, 2000), seed in 0:9
                    m, _ = generate_problem("knapsack/$v", target, unknown, seed)
                    push!(outcomes, kn_status(m))
                end
                @test MOI.OPTIMAL in outcomes
                @test MOI.INFEASIBLE in outcomes
            end
        end
    end
end
