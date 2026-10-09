# Focused quality contracts for the container_loading category: exact sizing
# and row formulas, data invariants (realistic container catalogue, pattern
# fit), witness and certificate arithmetic recomputed from the struct fields,
# reproducibility, tiny-target robustness, and HiGHS contracts with a mixed
# `unknown`.

cl_rows(m) = num_constraints(m; count_variable_in_set_constraints=false)

function cl_status(m)
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    optimize!(m)
    return termination_status(m)
end

@testset "Container Loading" begin
    @test Set(list_variants(:container_loading)) == Set([:standard, :two_dimensional_bin_packing])
    @test problem_info(:container_loading)[:default_variant] == :standard

    @testset "standard" begin
        @test_nowarn generate_problem("container_loading/standard", 2, unknown, 1)
        for target in (2, 11, 100, 1000, 10_000), status in (feasible, infeasible, unknown)
            m, p = generate_problem("container_loading/standard", target, status, 1)
            n, b = SyntheticLPs.container_loading_dimensions(target)
            @test (p.n_items, p.n_containers) == (n, b)
            @test num_variables(m) == n * b + b
            target >= 100 && @test abs(num_variables(m) - target) <= 0.02 * target
            target >= 1000 && @test 4 <= n / b <= 8
            @test cl_rows(m) == n + n * b + 3 * b
            @test b >= 2 && n >= 2
            # Capacities come from the ISO catalogue.
            for c in 1:b
                t = SyntheticLPs.CONTAINER_TYPES[p.container_type[c]]
                @test p.capacities[:, c] == [t.payload, t.volume, t.pallets]
            end
            @test all(>(0), p.item_requirements) && all(>(0), p.costs)
        end
        # The old search always chose 2 containers; now the fleet scales.
        @test SyntheticLPs.container_loading_dimensions(100_000)[2] >= 100

        for target in (60, 800, 5000), seed in 0:2
            m, p = generate_problem("container_loading/standard", target, feasible, seed)
            w = p.feasible_witness
            @test w !== nothing && p.infeasibility_certificate === nothing
            load = zeros(3, p.n_containers)
            for i in 1:p.n_items
                load[:, w.assignment[i]] .+= p.item_requirements[:, i]
            end
            @test all(load .<= p.capacities .+ 1e-9)
            point = Dict(v => 0.0 for v in all_variables(m))
            for i in 1:p.n_items
                point[m[:assign][i, w.assignment[i]]] = 1.0
                point[m[:used][w.assignment[i]]] = 1.0
            end
            @test isempty(primal_feasibility_report(m, point; atol=1e-6))
        end
        for target in (60, 800, 5000), seed in 0:2
            _, p = generate_problem("container_loading/standard", target, infeasible, seed)
            c = p.infeasibility_certificate
            @test c.total_requirement ≈ sum(p.item_requirements[c.dimension, :])
            @test c.fleet_capacity ≈ sum(p.capacities[c.dimension, :])
            @test c.total_requirement >= 1.04 * c.fleet_capacity * (1 - 1e-9)
        end
    end

    @testset "two_dimensional_bin_packing" begin
        @test_nowarn generate_problem(
            "container_loading/two_dimensional_bin_packing", 2, unknown, 1
        )
        for target in (4, 30, 300, 3000, 20_000), status in (feasible, infeasible, unknown)
            m, p = generate_problem(
                "container_loading/two_dimensional_bin_packing", target, status, 2
            )
            P, Q = length(p.strip_class), length(p.sheet_type)
            @test num_variables(m) == P + Q == max(target, 4)
            S, K = length(p.sheet_widths), length(p.class_heights)
            used_cs = length(
                unique(
                    vcat(
                        [(p.strip_class[q], p.strip_sheet[q]) for q in 1:P],
                        [(c, p.sheet_type[q]) for q in 1:Q for c in p.sheet_classes[q]],
                    ),
                ),
            )
            @test cl_rows(m) == length(p.demands) + used_cs + length(unique(p.sheet_type))
            @test p.class_heights == sort(unique(p.item_heights))
            # Strip patterns fit across the sheet width with items no taller
            # than the class; sheet patterns stack strips within the height.
            for q in 1:P
                @test all(
                    p.item_heights[i] <= p.class_heights[p.strip_class[q]] for i in p.strip_items[q]
                )
                @test sum(p.item_widths[p.strip_items[q]] .* p.strip_counts[q]) <=
                    p.sheet_widths[p.strip_sheet[q]]
            end
            for q in 1:Q
                @test sum(p.class_heights[p.sheet_classes[q]] .* p.sheet_counts[q]) <=
                    p.sheet_heights[p.sheet_type[q]]
            end
            @test allunique(zip(p.strip_class, p.strip_sheet, p.strip_items, p.strip_counts))
            @test allunique(zip(p.sheet_type, p.sheet_classes, p.sheet_counts))
        end
        _, big = generate_problem(
            "container_loading/two_dimensional_bin_packing", 100_000, unknown, 0
        )
        @test length(big.strip_class) + length(big.sheet_type) == 100_000
        @test length(big.demands) >= 0.04 * 100_000

        for target in (60, 900, 5000), seed in 0:2
            m, p = generate_problem(
                "container_loading/two_dimensional_bin_packing", target, feasible, seed
            )
            w = p.feasible_witness
            point = Dict(v => 0.0 for v in all_variables(m))
            for i in eachindex(p.demands)
                q = w.strip_pattern[i]
                @test p.strip_items[q] == [i] && p.strip_sheet[q] == w.sheet_type[i]
                @test p.strip_counts[q][1] * w.strip_runs[i] >= p.demands[i]
                point[m[:strip_runs][q]] += w.strip_runs[i]
            end
            for c in axes(w.sheet_runs, 1), s in axes(w.sheet_runs, 2)
                w.sheet_runs[c, s] > 0 || continue
                point[m[:sheet_runs][w.sheet_pattern[c, s]]] += w.sheet_runs[c, s]
            end
            @test isempty(primal_feasibility_report(m, point; atol=1e-6))
        end
        for target in (60, 900, 5000), seed in 0:2
            _, p = generate_problem(
                "container_loading/two_dimensional_bin_packing", target, infeasible, seed
            )
            c = p.infeasibility_certificate
            @test c.demand_area ≈ sum(p.item_widths .* p.item_heights .* p.demands)
            @test c.supply_area ≈ sum(p.sheet_widths .* p.sheet_heights .* p.availability)
            @test c.demand_area >= 1.04 * c.supply_area
        end
    end

    for v in ("standard", "two_dimensional_bin_packing"), status in (feasible, infeasible, unknown)
        Random.seed!(5)
        _, p1 = generate_problem("container_loading/$v", 800, status, 13)
        Random.seed!(55)
        _, p2 = generate_problem("container_loading/$v", 800, status, 13)
        for f in fieldnames(typeof(p1))
            a, b = getfield(p1, f), getfield(p2, f)
            a isa Union{Number, AbstractArray, FeasibilityStatus, Nothing} && @test isequal(a, b)
        end
    end

    @testset "Container Loading HiGHS contracts" begin
        if HAS_HIGHS
            for v in ("standard", "two_dimensional_bin_packing")
                for target in (100, 1500), status in (feasible, infeasible), seed in 0:1
                    m, _ = generate_problem("container_loading/$v", target, status, seed)
                    @test cl_status(m) == (status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE)
                end
                outcomes = Set{MOI.TerminationStatusCode}()
                for target in (300, 1500), seed in 0:9
                    m, _ = generate_problem("container_loading/$v", target, unknown, seed)
                    push!(outcomes, cl_status(m))
                end
                @test MOI.OPTIMAL in outcomes
                @test MOI.INFEASIBLE in outcomes
            end
        end
    end
end
