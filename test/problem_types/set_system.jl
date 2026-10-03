# Focused quality contracts for the set_system category: registry shape, exact
# column counts down to tiny targets, data invariants of the four application
# set systems, witness and certificate arithmetic recomputed from the struct
# fields, reproducibility, and HiGHS feasibility contracts on the default LP
# relaxation.

const SET_VARIANTS = (:set_cover, :set_packing, :set_partitioning, :combinatorial_auction)
_ss_ref(v) = ProblemVariant(:set_system, v)

@testset "Set System" begin
    @test Set(list_variants(:set_system)) == Set(SET_VARIANTS)
    @test problem_info(:set_system)[:default_variant] == :set_cover

    @testset "Tiny target robustness" begin
        for v in SET_VARIANTS
            @test_nowarn generate_problem(_ss_ref(v), 2, unknown, 1)
            @test_nowarn generate_problem(_ss_ref(v), 3, infeasible, 1)
        end
        tiny = generate_dataset(
            num_problems=4,
            size_distribution=Uniform(2, 3),
            problem_types=[:set_system],
            seed=1,
        )
        @test length(tiny) == 4
    end

    @testset "Exact sizing" begin
        for v in SET_VARIANTS, target in (2, 7, 60, 500, 4000), status in (feasible, infeasible, unknown)
            m, _ = generate_problem(_ss_ref(v), target, status, 2)
            @test num_variables(m) == target
        end
        # 100k builds in seconds and stays exact (constructors only).
        @test length(SyntheticLPs.SetCoverProblem(100_000, unknown, 0).columns) == 100_000
        @test length(SyntheticLPs.SetPackingProblem(100_000, infeasible, 0).columns) == 100_000
        @test length(SyntheticLPs.CombinatorialAuctionProblem(100_000, unknown, 0).bundles) ==
            100_000
    end

    @testset "Set cover: location covering" begin
        for status in (feasible, infeasible, unknown), seed in 0:2
            _, p = generate_problem(_ss_ref(:set_cover), 2000, status, seed)
            @test all(c -> !isempty(c) && issorted(c) && allunique(c), p.columns)
            @test all(c -> all(1 .<= c .<= p.n_elements), p.columns)
            covered = falses(p.n_elements)
            foreach(c -> covered[c] .= true, p.columns)
            @test all(covered)                        # anchors guarantee coverability
            @test all(>(0), p.costs)
            # Heavy-tailed coverage: some sites cover many more points than others.
            sizes = length.(p.columns)
            @test maximum(sizes) >= 3 * minimum(sizes)
            if status == feasible
                covered .= false
                foreach(j -> covered[p.columns[j]] .= true, p.feasible_witness.sites)
                @test all(covered)
                @test length(p.feasible_witness.sites) <= p.maximum_selected
            elseif status == infeasible
                pts = p.infeasibility_certificate.points
                owner = zeros(Int, p.n_elements)
                ok = true
                for (j, c) in enumerate(p.columns)
                    @test count(in(Set(pts)), c) <= 1
                end
                @test p.maximum_selected < length(pts)
                @test length(pts) >= 20
            end
        end
    end

    @testset "Set packing: train paths" begin
        for status in (feasible, infeasible, unknown), seed in 0:2
            _, p = generate_problem(_ss_ref(:set_packing), 2000, status, seed)
            n_trains = length(p.train_rows)
            @test length(p.mandatory) == n_trains
            # Each path holds exactly one train element: its own.
            train_set = Set(p.train_rows)
            for (j, c) in enumerate(p.columns)
                @test filter(in(train_set), c) == [p.train_rows[p.train_of[j]]]
            end
            if status == feasible
                paths = p.feasible_witness.paths
                used = Int[]
                foreach(j -> append!(used, p.columns[j]), paths)
                @test allunique(used)                 # disjoint cells and trains
                run = Set(p.train_of[paths])
                @test all(r in run for r in findall(p.mandatory))
            elseif status == infeasible
                cert = p.infeasibility_certificate
                @test sum(cert.occupancy) > length(cert.cells)
                @test sum(cert.occupancy) >= 1.05 * length(cert.cells)
                @test all(p.mandatory[cert.trains])
                cells = Set(cert.cells)
                for (k, r) in enumerate(cert.trains)
                    for j in findall(==(r), p.train_of)
                        @test count(in(cells), p.columns[j]) >= cert.occupancy[k]
                    end
                end
                @test length(cert.trains) >= 2
            end
        end
    end

    @testset "Set partitioning" begin
        for seed in 0:2
            _, p = generate_problem(_ss_ref(:set_partitioning), 2000, feasible, seed)
            cols = p.feasible_witness.columns
            @test sort!(reduce(vcat, p.columns[cols])) == collect(1:p.n_elements)
            @test length(cols) <= p.maximum_selected

            _, q = generate_problem(_ss_ref(:set_partitioning), 2000, infeasible, seed)
            cert = q.infeasibility_certificate
            @test cert.max_size == maximum(length, q.columns)
            @test cert.bound ≈ q.n_elements / cert.max_size
            @test q.maximum_selected <= cert.bound - 0.5
        end
    end

    @testset "Combinatorial auction" begin
        for status in (feasible, infeasible, unknown), seed in 0:2
            _, p = generate_problem(_ss_ref(:combinatorial_auction), 2000, status, seed)
            @test all(length(b) == length(q) for (b, q) in zip(p.bundles, p.quantities))
            @test all(b -> issorted(b) && allunique(b), p.bundles)
            @test all(all(1 .<= q .<= p.supply[b]) for (b, q) in zip(p.bundles, p.quantities))
            @test any(>(1), p.supply)                 # genuinely multi-unit
            @test length(unique(p.bidder_of)) < length(p.bidder_of)   # XOR bidders exist
            if status == feasible
                acc = p.feasible_witness.accepted
                @test allunique(p.bidder_of[acc])
                used = zeros(Int, p.n_items)
                for b in acc, (t, i) in enumerate(p.bundles[b])
                    used[i] += p.quantities[b][t]
                end
                @test all(used .<= p.supply)
                @test sum(p.bid_values[acc]) >= p.reserve
            elseif status == infeasible
                cert = p.infeasibility_certificate
                @test all(>=(0), cert.prices) && all(>=(0), cert.surplus)
                for b in eachindex(p.bundles)
                    priced = sum(
                        p.quantities[b][t] * cert.prices[i] for (t, i) in enumerate(p.bundles[b])
                    )
                    @test priced + cert.surplus[p.bidder_of[b]] >= p.bid_values[b] - 1e-9
                end
                @test cert.bound ≈ sum(p.supply .* cert.prices) + sum(cert.surplus)
                @test p.reserve >= 1.02 * cert.bound
            end
        end
    end

    @testset "Reproducibility" begin
        for v in SET_VARIANTS, status in (feasible, infeasible, unknown)
            _, a = generate_problem(_ss_ref(v), 900, status, 5)
            _, b = generate_problem(_ss_ref(v), 900, status, 5)
            for f in fieldnames(typeof(a))
                x, y = getfield(a, f), getfield(b, f)
                (x === nothing || isbits(x) || x isa AbstractArray) && @test x == y
            end
        end
    end

    @testset "HiGHS feasibility contracts" begin
        if HAS_HIGHS
            for v in SET_VARIANTS, target in (300, 2500), seed in 0:1
                for (status, expected) in ((feasible, MOI.OPTIMAL), (infeasible, MOI.INFEASIBLE))
                    m, _ = generate_problem(_ss_ref(v), target, status, seed)
                    set_optimizer(m, HiGHS.Optimizer)
                    set_silent(m)
                    set_time_limit_sec(m, 60.0)
                    optimize!(m)
                    @test termination_status(m) == expected
                end
            end
        else
            @info "HiGHS not available; skipping set_system solver contracts"
        end
    end
end
