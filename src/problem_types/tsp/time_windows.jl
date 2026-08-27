using JuMP
using Random

"""
    TSPTimeWindowsProblem <: ProblemGenerator

Generator for the travelling-salesman problem with time windows (TSPTW) —
appointment-delivery routing — as a mixed-integer program with a meaningful
continuous relaxation.

# Overview
A delivery vehicle leaves its home base (node 1) at time 0, must serve every
stop exactly once within that stop's time window, and return before the end of
the shift. Data is a symmetric metric of travel times (a shared-geography road
matrix converted to minutes), a positive service time per stop, a delivery
window `[a_j, b_j]` per stop (tight for "appointment" stops), an
EV-style route budget on total travel time, and a shift horizon.

The formulation propagates time along the selected arcs (no MTZ block is
needed — with `τ > 0` any depot-free cycle would force arrival times to
strictly increase around the cycle, which is impossible):

- Binary arc variables `x[i,j] ∈ {0,1}` select which arcs are traversed.
- Continuous arrival times `t[j] ∈ [a_j, b_j]` for stops `j = 2..n` (waiting
  for a window to open is allowed, so `t_j` is the service start, not the
  arrival).
- A continuous return time `r ∈ [0, L]` for the leg back to the base.

Key structural couplings:
- **Degree constraints**: one in-arc and one out-arc per node.
- **Time propagation with per-arc big-M**: `t_j ≥ t_i + s_i + τ_ij − M_ij(1−x_ij)`
  where `M_ij = max(0, b_i + s_i + τ_ij − a_j)` is the smallest value that makes
  the row non-binding when `x_ij = 0` (at `t_i = b_i` it reduces to
  `t_j ≥ a_j`, already a variable bound). The `max(0, ·)` clamp matters: a raw
  negative `M` would make the row *more* binding on unused arcs and cut off
  genuine tours.
- **Budget row** `Σ τ_ij x_ij ≤ F`: total route (travel) time fits the vehicle's
  charge/planned duration.
- **Shift row** `r ≤ L` with return propagation `r ≥ t_j + s_j + τ_j1 − M_j(1−x_j1)`.

This is a MIP whose continuous relaxation is a genuine tour relaxation: the
relaxed model is a useful LP test instance, but a fractional `x` is not an
implementable tour.

# Fields
- `n_stops::Int`: Total node count `n` (node 1 = home base, nodes `2..n` = stops)
- `locations::Vector{Tuple{Float64,Float64}}`: Node coordinates (index 1 = home base)
- `travel_time::Matrix{Float64}`: Symmetric travel-time matrix in minutes;
  `travel_time[i,j] = travel_time[j,i] > 0` off the diagonal, `[i,i] = 0`
- `service::Vector{Float64}`: Service duration per node in minutes (index 1 = 0)
- `window_start::Vector{Float64}`: Window opening `a_j` per node (index 1 = 0)
- `window_end::Vector{Float64}`: Window closing `b_j` per node (index 1 = 0)
- `route_budget::Float64`: Maximum total travel time `F`
- `shift_length::Float64`: Maximum return time `L`
- `planted_tour::Vector{Int}`: The planted `[1, …, 1]` tour (equal to the data
  tour; windows are built around it in the feasible branch)
"""
struct TSPTimeWindowsProblem <: ProblemGenerator
    n_stops::Int
    locations::Vector{Tuple{Float64,Float64}}
    travel_time::Matrix{Float64}
    service::Vector{Float64}
    window_start::Vector{Float64}
    window_end::Vector{Float64}
    route_budget::Float64
    shift_length::Float64
    planted_tour::Vector{Int}
end

"""
    TSPTimeWindowsProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a TSPTW instance.

# Variable-count formula
The model creates one binary `x` per arc (`n*(n-1)`), one arrival time per stop
(`n-1`), and one return time:

    total = n*(n-1) + (n-1) + 1 = n^2

So `n = max(5, round(Int, sqrt(target_variables)))`. No variables are deleted
in any feasibility branch, so the delivered count is `n^2` for all three
statuses. For `target = 100` this gives `n = 10` (100 vars); for `target = 500`,
`n = 22` (484 vars).

# Arguments
- `target_variables`: Target number of decision variables (`x`, `t`, and `r`)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Feasibility
- `feasible`: a concrete tour is planted and the windows are built around its
  schedule. Window openings `a_j` sit at or below the tour's no-wait arrival
  (tight for ~25% appointment stops, earlier otherwise); window closings `b_j`
  sit at or above the *realized* arrival computed by a forward pass with
  waiting, so serving the planted tour in order — waiting where a window opens
  late — satisfies every row. `F` exceeds the planted tour's travel time by
  10-30% and `L` its return time by 5-20%. The planted integer point survives
  `relax_integrality` verbatim (relaxation only widens `x`), so the delivered
  relaxation is feasible and bounded.
- `infeasible`: the route budget is set strictly below the minimum possible
  total travel time: `F = 0.85 · Σ_i min_{j≠i} τ_ij`. Summing the out-degree
  rows gives `Σ_{ij} τ_ij x_ij = Σ_i (Σ_{j≠i} τ_ij x_ij) ≥ Σ_i min_{j≠i} τ_ij > F`
  for every feasible `x`, using only the degree rows — so the model is
  infeasible even in the LP relaxation. Windows and shift are sampled naturally.
- `unknown`: windows are sampled independently of any tour (`a_j` uniform over
  the first 60% of the horizon, widths 20-90 min) and the budget sits at
  0.9-1.2× a random reference tour's travel time; the instance may or may not
  admit a feasible tour.
"""
function TSPTimeWindowsProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # total = n^2  =>  n ≈ sqrt(target). Status-independent (no deletions).
    n = max(5, round(Int, sqrt(target_variables)))

    # --- Geography, travel times, service times ---
    locations = _tsp_stops(n)
    dist = _tsp_distance(locations)
    minutes_per_km = rand(1.2:0.1:2.0)             # urban driving pace
    tau = round.(dist .* minutes_per_km, digits = 2)
    # Time propagation is the subtour eliminator; it needs strictly positive
    # travel times on distinct stops (guaranteed by the distance floor).
    @assert minimum(tau[i, j] for i in 1:n, j in 1:n if i != j) > 0
    service = vcat(0.0, [rand(5.0:0.5:25.0) for _ in 2:n])

    # --- A concrete tour and its no-wait schedule (shared by all branches) ---
    tour = [1; shuffle(collect(2:n)); 1]
    arr = zeros(n)                                  # no-wait arrivals, depot at 0
    for idx in 2:n
        i, j = tour[idx-1], tour[idx]
        arr[j] = arr[i] + service[i] + tau[i, j]
    end
    tour_tau = sum(tau[tour[idx-1], tour[idx]] for idx in 2:n+1)
    no_wait_return = arr[tour[n]] + service[tour[n]] + tau[tour[n], 1]
    horizon = round(no_wait_return * 1.5, digits = 2)

    a = zeros(n)
    b = zeros(n)
    if feasibility_status == feasible
        # Plant windows around the tour schedule. `a` rounds down (it must not
        # exceed the realized arrival) and `b` rounds up (it must not cut it).
        tight = rand(n) .< 0.25                     # appointment stops
        for j in 2:n
            target_a = tight[j] ? arr[j] : max(0.0, arr[j] * (0.75 + 0.2 * rand()))
            a[j] = max(0.0, round(target_a - 0.01, digits = 2))
        end
        # Forward pass WITH waiting: the schedule the vehicle actually runs.
        t = zeros(n)
        for idx in 2:n
            i, j = tour[idx-1], tour[idx]
            t[j] = max(t[i] + service[i] + tau[i, j], a[j])
        end
        for j in 2:n
            b[j] = round(t[j] + 0.01, digits = 2)   # > t[j] after rounding
        end
        planted_return = t[tour[n]] + service[tour[n]] + tau[tour[n], 1]
        F = round(tour_tau * (1.10 + 0.20 * rand()), digits = 2)
        L = round(planted_return * (1.05 + 0.15 * rand()), digits = 2)

    else
        # Natural windows, independent of the tour (the infeasible and unknown
        # branches differ only in the budget F).
        for j in 2:n
            a[j] = round(rand() * 0.6 * horizon, digits = 2)
            b[j] = round(a[j] + rand(20.0:5.0:90.0), digits = 2)
        end
        L = round(horizon * 1.2, digits = 2)
        if feasibility_status == infeasible
            # Below the degree-row lower bound on total travel time
            # (see constructor docstring), so infeasible even relaxed.
            sum_min = sum(minimum(tau[i, j] for j in 1:n if j != i) for i in 1:n)
            F = round(0.85 * sum_min, digits = 2)
        else
            # unknown: a plausible budget that may or may not suffice.
            F = round(tour_tau * (0.9 + 0.3 * rand()), digits = 2)
        end
    end

    return TSPTimeWindowsProblem(n, locations, tau, service, a, b, F, L, tour)
end

"""
    build_model(prob::TSPTimeWindowsProblem)

Build a JuMP model for the TSPTW using time propagation along selected arcs.
Deterministic — uses only data from the struct fields.

Node indexing: node `1` is the home base (departure at time 0); nodes `2..n`
are stops. All arcs of the complete graph are available.

Decision variables:
- `x[i,j] ∈ {0,1}`: arc `(i,j)` is traversed
- `t[j] ∈ [a_j, b_j]`: service start time at stop `j` (waiting allowed)
- `r ∈ [0, L]`: return time at the home base

# Returns
- `model`: The JuMP model
"""
function build_model(prob::TSPTimeWindowsProblem)
    model = Model()

    n = prob.n_stops
    nodes = 1:n
    stops = 2:n
    tau = prob.travel_time
    s = prob.service
    a = prob.window_start
    b = prob.window_end

    # --- Variables: one binary x per arc, arrival per stop, return time ---
    # Count = n*(n-1) + (n-1) + 1 = n^2
    @variable(model, x[i in nodes, j in nodes; i != j], Bin)
    @variable(model, a[j] <= t[j in stops] <= b[j])
    @variable(model, 0 <= r <= prob.shift_length)

    # --- Objective: minimize total travel time ---
    @objective(model, Min,
        sum(tau[i, j] * x[i, j] for i in nodes, j in nodes if i != j))

    # --- Degree constraints: exactly one in-arc and one out-arc per node ---
    for j in nodes
        @constraint(model, sum(x[i, j] for i in nodes if i != j) == 1)   # in
        @constraint(model, sum(x[j, k] for k in nodes if k != j) == 1)   # out
    end

    # --- Route budget: total travel time within the vehicle's range ---
    @constraint(model,
        sum(tau[i, j] * x[i, j] for i in nodes, j in nodes if i != j) <=
        prob.route_budget)

    # --- Time propagation, stop -> stop (waiting allowed; M is the smallest
    # value that makes the row non-binding at x = 0; clamped at 0) ---
    for i in stops, j in stops
        i == j && continue
        M = max(0.0, b[i] + s[i] + tau[i, j] - a[j])
        @constraint(model, t[j] >= t[i] + s[i] + tau[i, j] - M * (1 - x[i, j]))
    end

    # --- Time propagation, depot -> stop (the base departs at time 0) ---
    for j in stops
        M = max(0.0, tau[1, j] - a[j])
        @constraint(model, t[j] >= tau[1, j] - M * (1 - x[1, j]))
    end

    # --- Return legs: stop -> depot, within the shift ---
    for j in stops
        M = max(0.0, b[j] + s[j] + tau[j, 1])
        @constraint(model, r >= t[j] + s[j] + tau[j, 1] - M * (1 - x[j, 1]))
    end

    return model
end

# Register the variant (standard remains the category default; do NOT pass
# default = true here).
register_variant(
    :tsp,
    :time_windows,
    TSPTimeWindowsProblem,
    "Travelling-salesman problem with delivery time windows, service times, a route-duration budget, and a shift limit; a MIP whose time-propagation relaxation is a genuine tour relaxation",
)
