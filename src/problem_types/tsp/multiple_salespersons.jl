using JuMP
using Random

"""
    TSPMultipleSalespersonsProblem <: ProblemGenerator

Balanced multiple-salesperson TSP. A fixed fleet leaves and returns to one depot,
every stop is assigned to exactly one route, and every route contains between
`min_stops` and `max_stops` customers. Anchored lifted order constraints make
the per-route limits exact in integer solutions.

Unrelaxed MIPs (`relax_integer=false`) grow hard quickly: around
`target_variables >= 300` HiGHS may not prove optimality within the central
verifier's default `feasibility_timeout` (10 s), so pass a larger timeout (the
tests use 30 s at `target = 80`). The default relaxed LPs are unaffected.

# Variable count

One binary `x` per allowed arc plus one order variable per stop. With complete
support that is `n(n-1) + (n-1) = n^2 - 1`; a district-mode infeasible instance
deletes the `k(n-k)` arcs of its Hall block and sizes `n` against the delivered
count `n^2 - 1 - k(n-k)`.

# Feasibility

  - `feasible`: the balanced partition (`remainder` routes of `quotient + 1`
    stops, the rest of `quotient`) is an explicit witness; complete support.
  - `infeasible`: one of two LP-valid certificates, chosen per instance
    (`infeasibility_mode`):
      - `:district` (≈75%, the default): the shared Hall-deficit district block
        (`_tsp_hall_block`): a district `S` of `k ≈ 6–12%` of the stops can be
        entered only from `k-1` gateway stops `T` (depot excluded from both), so
        the stop in-degree rows of `S` sum to `k` but draw on only `k-1` unit
        out-degrees — a contradiction over `2k-1` dense degree rows that
        presolve does not see; the route limits are sampled as for `unknown`.
      - `:fleet_capacity`: `fleet · max_stops < n - 1`, contradicted by the
        aggregate row `n - 1 ≤ max_stops · depot_out` together with
        `depot_out = fleet` (a two-row argument presolve detects).
  - `unknown`: loose route limits around the balanced split on complete support.
"""
struct TSPMultipleSalespersonsProblem <: ProblemGenerator
    n_stops::Int
    n_salespersons::Int
    min_stops::Int
    max_stops::Int
    locations::Vector{Tuple{Float64, Float64}}
    dist::Matrix{Float64}
    arc_ok::Matrix{Bool}
    blocked_set::Vector{Int}
    gate_set::Vector{Int}
    infeasibility_mode::Symbol
end

function TSPMultipleSalespersonsProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    n0 = max(5, round(Int, sqrt(target_variables + 1)))
    # Mode and district size are drawn unconditionally (RNG alignment across
    # statuses); only an infeasible request uses them.
    mode = rand(rng) < 0.75 ? :district : :fleet_capacity
    k = _tsp_hall_size(rng, n0)
    district = feasibility_status == infeasible && mode == :district
    n = district ? _tsp_pick_n(n0, target_variables, k, m -> m^2 - 1 - k * (m - k)) : n0

    n_customers = n - 1
    max_fleet = min(6, max(2, fld(n_customers, 2)))
    n_salespersons = rand(rng, 2:max_fleet)
    quotient, remainder = divrem(n_customers, n_salespersons)

    if feasibility_status == infeasible && !district
        min_stops = 1
        max_stops = max(1, fld(n_customers - 1, n_salespersons))
    elseif feasibility_status == feasible
        # The balanced partition with `remainder` routes of quotient+1 stops is
        # an explicit feasible witness.
        min_stops = quotient
        max_stops = quotient + (remainder > 0)
    else
        min_stops = max(1, quotient - 1)
        max_stops = min(n_customers, quotient + (remainder > 0) + 2)
    end

    locations = _tsp_stops(rng, n)
    dist = _tsp_distance(rng, locations)
    arc_ok, S, T = if district
        _tsp_hall_block(rng, n, k, locations)
    else
        _tsp_full_support(n), Int[], Int[]
    end
    resolved_mode = feasibility_status == infeasible ? mode : :none
    return TSPMultipleSalespersonsProblem(
        n, n_salespersons, min_stops, max_stops, locations, dist, arc_ok, S, T, resolved_mode
    )
end

function build_model(prob::TSPMultipleSalespersonsProblem)
    model = Model()
    n = prob.n_stops
    nodes = 1:n
    stops = 2:n
    fleet = prob.n_salespersons
    max_stops = prob.max_stops
    ok(i, j) = prob.arc_ok[i, j]

    @variable(model, x[i in nodes, j in nodes; ok(i, j)], Bin)
    @variable(model, 1 <= u[j in stops] <= max_stops)
    @objective(model, Min, sum(prob.dist[i, j] * x[i, j] for i in nodes, j in nodes if ok(i, j)))

    depot_out = sum(x[1, j] for j in stops if ok(1, j))
    @constraint(model, depot_out == fleet)
    @constraint(model, sum(x[j, 1] for j in stops if ok(j, 1)) == fleet)
    for j in stops
        @constraint(model, sum(x[i, j] for i in nodes if ok(i, j)) == 1)
        @constraint(model, sum(x[j, k] for k in nodes if ok(j, k)) == 1)
        # A route's first stop has position exactly one (a district stop has
        # no depot arc, so the row reduces to the variable bound and is skipped).
        ok(1, j) && @constraint(model, u[j] <= 1 + (max_stops - 1) * (1 - x[1, j]))
        # With exact unit increments below, a returning stop's position is the
        # number of customers on its route.
        @constraint(model, u[j] >= prob.min_stops * x[j, 1])
    end

    if max_stops > 1
        for i in stops, j in stops
            (i != j && ok(i, j)) || continue
            if ok(j, i)
                @constraint(
                    model,
                    u[i] - u[j] + max_stops * x[i, j] + (max_stops - 2) * x[j, i] <= max_stops - 1
                )
            else
                @constraint(model, u[i] - u[j] + max_stops * x[i, j] <= max_stops - 1)
            end
        end
    else
        # A one-stop route cannot contain a customer-to-customer arc.
        for i in stops, j in stops
            (i != j && ok(i, j)) || continue
            @constraint(model, x[i, j] == 0)
        end
    end

    # This redundant-for-integers aggregate row materially strengthens the LP
    # and provides a direct relaxation-proof infeasibility certificate.
    @constraint(model, n - 1 <= max_stops * depot_out)

    return model
end

register_variant(
    :tsp,
    :multiple_salespersons,
    TSPMultipleSalespersonsProblem,
    "Balanced multiple-salesperson TSP with exact per-route stop limits and lifted order constraints";
    tags=[:routing, :big_m],
)
