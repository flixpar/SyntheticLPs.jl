# Focused quality contracts for the tsp category: registry wiring, data
# contracts, variable-count formulas, the Hall-deficit structure behind every
# infeasible branch, tiny-target clamping, and the HiGHS feasibility contracts
# for every variant (including the natural MIPs checked without relaxation).
@testset "TSP Variants" begin
    @test list_variants(:tsp) == [
        :assignment_relaxation,
        :asymmetric,
        :flow,
        :multiple_salespersons,
        :precedence,
        :prize_collecting,
        :standard,
        :time_windows,
    ]
    @test problem_info(:tsp)[:default_variant] == :standard
    @test ProblemVariant("tsp") == ProblemVariant(:tsp, :standard)

    # Symmetric road metrics for the symmetric-data variants (the
    # time-window variant stores it as travel_time); genuinely asymmetric
    # travel times for the ATSP variant.
    for v in (
        :standard,
        :flow,
        :multiple_salespersons,
        :precedence,
        :prize_collecting,
        :time_windows,
        :assignment_relaxation,
    )
        _, p = generate_problem(ProblemVariant(:tsp, v), 100, unknown, 0)
        mat = hasproperty(p, :dist) ? p.dist : p.travel_time
        @test mat == mat'
        @test all(iszero, mat[i, i] for i in axes(mat, 1))
    end
    # Sparse ATSP: a sorted candidate-arc list without loops, positive
    # direction-dependent travel times, no stop with fewer than two incoming
    # candidates, and (feasible) every planted-tour leg among the candidates.
    for status in (feasible, unknown), s in 0:2
        _, p = generate_problem("tsp/asymmetric", 2000, status, s)
        n = p.n_stops
        @test issorted(p.arcs) && allunique(p.arcs)
        @test all(i != j for (i, j) in p.arcs)
        @test length(p.travel_time) == length(p.arcs)
        @test all(>(0), p.travel_time)
        time = Dict(zip(p.arcs, p.travel_time))
        @test any(time[(i, j)] != time[(j, i)] for (i, j) in p.arcs if haskey(time, (j, i)))
        @test any(!haskey(time, (j, i)) for (i, j) in p.arcs)       # asymmetric support
        indeg = zeros(Int, n)
        outdeg = zeros(Int, n)
        for (i, j) in p.arcs
            indeg[j] += 1
            outdeg[i] += 1
        end
        @test minimum(indeg) >= 2 && minimum(outdeg) >= 2
        # Sparse: the candidate graph keeps a small fraction of the n(n-1) arcs.
        @test length(p.arcs) <= (p.out_degree + 3) * n
        if status == feasible
            tour = p.planted_tour
            @test first(tour) == last(tour) == 1
            @test sort(tour[2:(end - 1)]) == collect(2:n)
            @test all((tour[t - 1], tour[t]) in keys(time) for t in 2:length(tour))
        else
            @test isempty(p.planted_tour)
        end
    end

    # Variable-count formulas, straight from each struct's n_stops.
    count_formulas = [
        (:standard => (p -> p.n_stops^2 - 1)),
        (:asymmetric => (p -> length(p.arcs) + p.n_stops - 1)),
        (:flow => (p -> 2 * p.n_stops * (p.n_stops - 1))),
        (:time_windows => (p -> p.n_stops^2)),
        (:assignment_relaxation => (p -> p.n_stops * (p.n_stops - 1))),
        (:multiple_salespersons => (p -> p.n_stops^2 - 1)),
        (:precedence => (p -> p.n_stops^2 - 1)),
        (:prize_collecting => (p -> 2 * p.n_stops * (p.n_stops - 1) + p.n_stops - 1)),
    ]
    for (v, f) in count_formulas
        m, p = generate_problem(ProblemVariant(:tsp, v), 100, unknown, 0)
        @test num_variables(m) == f(p)
    end

    # The infeasible branch sizes n against the *delivered* count after the
    # Hall block deletes k*(n-k) arcs, via a per-variant delivered() lambda.
    # These assertions tie those lambdas to the models actually built (for
    # the application variants, in their district mode).
    delivered_formulas = [
        (:standard => ((n, k) -> n^2 - 1 - k * (n - k))),
        (:flow => ((n, k) -> 2 * (n^2 - n) - 2 * k * (n - k))),
        (:assignment_relaxation => ((n, k) -> n^2 - n - k * (n - k))),
        (:multiple_salespersons => ((n, k) -> n^2 - 1 - k * (n - k))),
        (:prize_collecting => ((n, k) -> 2n^2 - n - 1 - 2k * (n - k))),
    ]
    for (v, f) in delivered_formulas, s in 1:6
        m, p = generate_problem(ProblemVariant(:tsp, v), 1200, infeasible, s)
        hasproperty(p, :infeasibility_mode) && p.infeasibility_mode != :district && continue
        @test num_variables(m) == f(p.n_stops, length(p.blocked_set))
        @test abs(num_variables(m) - 1200) <= 0.05 * 1200
    end

    # Hall district: every in-arc to the district S originates in the
    # gateway set T, T is disjoint from S, excludes the depot, and is one node
    # short of S (the degree-row deficit that makes these instances
    # infeasible even in the LP relaxation). The district scales with n
    # (k >= 3 once n >= 8) so no degree row is a singleton or doubleton, and
    # district stops keep in-arcs from every gateway.
    for v in (:standard, :flow, :assignment_relaxation, :multiple_salespersons, :prize_collecting)
        for s in 1:4
            _, p = generate_problem(ProblemVariant(:tsp, v), 1200, infeasible, s)
            hasproperty(p, :infeasibility_mode) && p.infeasibility_mode != :district && continue
            S, T = p.blocked_set, p.gate_set
            @test length(T) == length(S) - 1
            @test length(S) >= 3
            @test isempty(intersect(S, T))
            @test !(1 in S) && !(1 in T)
            for j in S, i in 1:p.n_stops
                i == j && continue
                @test p.arc_ok[i, j] == (i in T)
            end
        end
    end
    for s in 1:3
        _, p = generate_problem("tsp/asymmetric", 2000, infeasible, s)
        S, T = p.blocked_set, p.gate_set
        @test length(T) == length(S) - 1 >= 2
        @test isempty(intersect(S, T)) && !(1 in S) && !(1 in T)
        @test all(i in T for (i, j) in p.arcs if j in S)
        @test all(count(a -> a[2] == j, p.arcs) >= 2 for j in S)
        @test all(count(a -> a[1] == j, p.arcs) >= 2 for j in S)
    end

    # Time-window data contract: nonempty windows, and the planted tour's
    # travel time fits the route budget of a feasible instance.
    _, p = generate_problem("tsp/time_windows", 100, feasible, 0)
    @test all(p.window_start[j] <= p.window_end[j] for j in 2:p.n_stops)
    tour_time = sum(
        p.travel_time[p.planted_tour[i - 1], p.planted_tour[i]] for i in 2:length(p.planted_tour)
    )
    @test tour_time <= p.route_budget

    # Application-variant data contracts and relaxation-proof
    # infeasibility certificates.
    _, p = generate_problem("tsp/prize_collecting", 100, feasible, 4)
    @test 0 < p.prize_quota <= sum(p.prizes)
    modes = Symbol[]
    for s in 0:11
        _, p = generate_problem("tsp/prize_collecting", 300, infeasible, s)
        push!(modes, p.infeasibility_mode)
        if p.infeasibility_mode == :over_total
            @test p.prize_quota > sum(p.prizes)
        else
            # District: quota attainable row-by-row, but above the LP maximum
            # total - min_{j in S} prize_j by a margin.
            @test p.prize_quota <= sum(p.prizes)
            @test p.prize_quota - (sum(p.prizes) - minimum(p.prizes[p.blocked_set])) >= 0.25 * 10.0
        end
    end
    @test :district in modes && :over_total in modes

    _, p = generate_problem("tsp/multiple_salespersons", 100, feasible, 4)
    @test p.n_salespersons * p.min_stops <= p.n_stops - 1 <= p.n_salespersons * p.max_stops
    modes = Symbol[]
    for s in 0:11
        _, p = generate_problem("tsp/multiple_salespersons", 300, infeasible, s)
        push!(modes, p.infeasibility_mode)
        if p.infeasibility_mode == :fleet_capacity
            @test p.n_salespersons * p.max_stops < p.n_stops - 1
            @test isempty(p.blocked_set)
        else
            @test length(p.gate_set) == length(p.blocked_set) - 1
        end
    end
    @test :district in modes && :fleet_capacity in modes

    _, p = generate_problem("tsp/precedence", 100, infeasible, 4)
    @test length(p.precedence_pairs) == 3
    @test p.precedence_pairs[1][2] == p.precedence_pairs[2][1]
    @test p.precedence_pairs[2][2] == p.precedence_pairs[3][1]
    @test p.precedence_pairs[3][2] == p.precedence_pairs[1][1]
end

# Tiny targets clamp to n = 5, where the Hall-block size must also fall back to
# k = 2. Each of these used to throw during construction.
@testset "TSP Tiny Target Robustness" begin
    @test_nowarn generate_problem("tsp/standard", 3, infeasible, 1)
    @test_nowarn generate_problem("tsp/asymmetric", 3, infeasible, 1)
    @test_nowarn generate_problem("tsp/flow", 3, infeasible, 1)
    @test_nowarn generate_problem("tsp/time_windows", 3, unknown, 1)
    @test_nowarn generate_problem("tsp/multiple_salespersons", 3, infeasible, 1)
    @test_nowarn generate_problem("tsp/precedence", 3, infeasible, 1)
    @test_nowarn generate_problem("tsp/prize_collecting", 3, infeasible, 1)
    @test_nowarn generate_problem("tsp/assignment_relaxation", 3, infeasible, 1)
    @test_nowarn generate_problem("tsp/asymmetric", 3, unknown, 1)
    @test_nowarn generate_problem("tsp/asymmetric", 3, feasible, 1)
    for target in (3, 12, 40), s in 1:3
        @test_nowarn generate_problem("tsp/multiple_salespersons", target, infeasible, s)
        @test_nowarn generate_problem("tsp/prize_collecting", target, infeasible, s)
    end
end

# Large-target sizing (construction only, no solve): the sparse ATSP keeps its
# variable count within 5% of the target at 100k, where it has ~10k stops.
@testset "TSP Large Target Sizing" begin
    for status in (feasible, infeasible, unknown)
        m, p = generate_problem("tsp/asymmetric", 100_000, status, 0)
        @test abs(num_variables(m) - 100_000) <= 0.05 * 100_000
        @test p.n_stops > 5000
    end
end

@testset "TSP Feasibility Contracts" begin
    if HAS_HIGHS
        # tsp variants: feasible requests deliver a relaxed-feasible model and
        # infeasible requests a relaxed-infeasible one (Hall-deficit arc block
        # / route-budget shortfall), by construction rather than heuristic repair.
        for variant in list_variants(:tsp)
            ref = "tsp/$variant"
            for s in 1:5
                m, _ = generate_problem(ref, 120, feasible, s; optimizer=HiGHS.Optimizer)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) == MOI.OPTIMAL
                m, _ = generate_problem(ref, 120, infeasible, s; optimizer=HiGHS.Optimizer)
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                @test termination_status(m) in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
            end
        end

        # The Hall district is not refuted by presolve alone: the solver has
        # to run simplex iterations (k in {2, 3} used to be caught by presolve).
        for v in (:standard, :asymmetric, :flow, :assignment_relaxation), s in 1:2
            m, _ = generate_problem(ProblemVariant(:tsp, v), 1500, infeasible, s)
            set_optimizer(m, HiGHS.Optimizer)
            set_silent(m)
            optimize!(m)
            @test termination_status(m) == MOI.INFEASIBLE
            @test MOI.get(m, MOI.SimplexIterations()) > 0
        end

        # The newly integrated natural MIPs and lifted-MTZ variants also honor
        # the contract without relaxing integrality.
        for ref in (
            "tsp/standard",
            "tsp/asymmetric",
            "tsp/multiple_salespersons",
            "tsp/precedence",
            "tsp/prize_collecting",
        )
            for status in (feasible, infeasible), s in 1:2
                m, _ = generate_problem(
                    ref,
                    80,
                    status,
                    s;
                    relax_integer=false,
                    optimizer=HiGHS.Optimizer,
                    feasibility_timeout=30.0,
                )
                set_optimizer(m, HiGHS.Optimizer)
                set_silent(m)
                optimize!(m)
                expected = status == feasible ? MOI.OPTIMAL : MOI.INFEASIBLE
                @test termination_status(m) == expected
            end
        end

        # Reconstruct every route of one m-TSP solution and verify that the
        # modeled stop-count bounds hold route by route, not just in aggregate.
        m, p = generate_problem("tsp/multiple_salespersons", 100, feasible, 17; relax_integer=false)
        set_optimizer(m, HiGHS.Optimizer)
        set_silent(m)
        optimize!(m)
        @test termination_status(m) == MOI.OPTIMAL
        x = m[:x]
        route_lengths = Int[]
        for first_stop in 2:p.n_stops
            value(x[1, first_stop]) > 0.5 || continue
            current = first_stop
            route_length = 1
            while value(x[current, 1]) <= 0.5
                successors = [j for j in 2:p.n_stops if j != current && value(x[current, j]) > 0.5]
                @test length(successors) == 1
                current = only(successors)
                route_length += 1
                @test route_length <= p.n_stops - 1
            end
            push!(route_lengths, route_length)
        end
        @test length(route_lengths) == p.n_salespersons
        @test all(p.min_stops <= len <= p.max_stops for len in route_lengths)
    end
end
