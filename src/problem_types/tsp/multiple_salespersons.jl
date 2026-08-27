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
"""
struct TSPMultipleSalespersonsProblem <: ProblemGenerator
    n_stops::Int
    n_salespersons::Int
    min_stops::Int
    max_stops::Int
    locations::Vector{Tuple{Float64,Float64}}
    dist::Matrix{Float64}
end

function TSPMultipleSalespersonsProblem(target_variables::Int,
                                        feasibility_status::FeasibilityStatus,
                                        seed::Int)
    Random.seed!(seed)
    n = max(5, round(Int, sqrt(target_variables + 1)))
    n_customers = n - 1
    max_fleet = min(6, max(2, fld(n_customers, 2)))
    n_salespersons = rand(2:max_fleet)
    quotient, remainder = divrem(n_customers, n_salespersons)

    if feasibility_status == infeasible
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

    locations = _tsp_stops(n)
    dist = _tsp_distance(locations)
    return TSPMultipleSalespersonsProblem(
        n, n_salespersons, min_stops, max_stops, locations, dist,
    )
end

function build_model(prob::TSPMultipleSalespersonsProblem)
    model = Model()
    n = prob.n_stops
    nodes = 1:n
    stops = 2:n
    fleet = prob.n_salespersons
    max_stops = prob.max_stops

    @variable(model, x[i in nodes, j in nodes; i != j], Bin)
    @variable(model, 1 <= u[j in stops] <= max_stops)
    @objective(model, Min,
        sum(prob.dist[i, j] * x[i, j] for i in nodes, j in nodes if i != j))

    depot_out = sum(x[1, j] for j in stops)
    @constraint(model, depot_out == fleet)
    @constraint(model, sum(x[j, 1] for j in stops) == fleet)
    for j in stops
        @constraint(model, sum(x[i, j] for i in nodes if i != j) == 1)
        @constraint(model, sum(x[j, k] for k in nodes if k != j) == 1)
        # A route's first stop has position exactly one.
        @constraint(model, u[j] <= 1 + (max_stops - 1) * (1 - x[1, j]))
        # With exact unit increments below, a returning stop's position is the
        # number of customers on its route.
        @constraint(model, u[j] >= prob.min_stops * x[j, 1])
    end

    if max_stops > 1
        for i in stops, j in stops
            i == j && continue
            @constraint(model,
                u[i] - u[j] + max_stops * x[i, j] +
                (max_stops - 2) * x[j, i] <= max_stops - 1)
        end
    else
        # A one-stop route cannot contain a customer-to-customer arc.
        for i in stops, j in stops
            i == j && continue
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
    "Balanced multiple-salesperson TSP with exact per-route stop limits and lifted order constraints",
)
