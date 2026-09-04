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
    _, p = generate_problem("tsp/asymmetric", 100, unknown, 0)
    @test count(p.dist[i, j] != p.dist[j, i] for i in 1:p.n_stops, j in 1:p.n_stops if i != j) > 0
    @test length(p.row_weight) == length(p.col_weight) == p.grid_side
    @test all(
        p.dist[i, j] <= p.dist[i, k] + p.dist[k, j] for
        i in 1:p.n_stops, j in 1:p.n_stops, k in 1:p.n_stops
    )

    # Variable-count formulas, straight from each struct's n_stops.
    count_formulas = [
        (:standard => (p -> p.n_stops^2 - 1)),
        (:asymmetric => (p -> p.n_stops^2 - 1)),
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
    # These assertions tie those lambdas to the models actually built.
    delivered_formulas = [
        (:standard => ((n, k) -> n^2 - 1 - k * (n - k))),
        (:asymmetric => ((n, k) -> n^2 - 1 - k * (n - k))),
        (:flow => ((n, k) -> 2 * (n^2 - n) - 2 * k * (n - k))),
        (:assignment_relaxation => ((n, k) -> n^2 - n - k * (n - k))),
    ]
    for (v, f) in delivered_formulas, s in 1:3
        m, p = generate_problem(ProblemVariant(:tsp, v), 120, infeasible, s)
        @test num_variables(m) == f(p.n_stops, length(p.blocked_set))
    end

    # Hall block: every in-arc to the blocked set S originates in the gate
    # set T, T is disjoint from S and one node short of it (the degree-row
    # deficit that makes these instances infeasible even in the LP
    # relaxation), and blocked stops keep at least one allowed in-arc.
    for v in (:standard, :asymmetric, :flow, :assignment_relaxation)
        for s in 1:3
            _, p = generate_problem(ProblemVariant(:tsp, v), 120, infeasible, s)
            @test length(p.gate_set) == length(p.blocked_set) - 1
            @test isempty(intersect(p.blocked_set, p.gate_set))
            for j in p.blocked_set, i in 1:p.n_stops
                (i in p.gate_set || i == j) && continue
                @test !p.arc_ok[i, j]
            end
            for j in p.blocked_set
                @test any(p.arc_ok[i, j] for i in 1:p.n_stops)
            end
        end
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
    _, p = generate_problem("tsp/prize_collecting", 100, infeasible, 4)
    @test p.prize_quota > sum(p.prizes)

    _, p = generate_problem("tsp/multiple_salespersons", 100, feasible, 4)
    @test p.n_salespersons * p.min_stops <= p.n_stops - 1 <= p.n_salespersons * p.max_stops
    _, p = generate_problem("tsp/multiple_salespersons", 100, infeasible, 4)
    @test p.n_salespersons * p.max_stops < p.n_stops - 1

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
