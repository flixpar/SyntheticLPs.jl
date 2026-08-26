using JuMP
using Random

"""
    TimeWindowedTSPProblem <: ProblemGenerator

Generator for the Traveling Salesman Problem with Time Windows (TSP-TW),
formulated as a mixed-integer program with Miller-Tucker-Zemlin (MTZ) subtour
elimination plus arrival-time variables.

# Overview
A technician based at a depot (city 1) must visit `n-1` customer sites exactly
once and return. Travel costs are rounded-Euclidean distances over clustered
city locations; every customer `i` has an integer service duration `d[i]` and
an appointment window `[early[i], late[i]]` that the visit must start inside.
The objective minimizes total travel distance — the field-service / last-mile
routing model.

On top of the directed MTZ core (`x[i,j] ∈ {0,1}` arcs, degree 1 in/out,
position potentials `u`), continuous arrival times `t[i] ≥ 0` propagate along
the tour:

- The depot leaves at time `0` (it has no variable and no window).
- For every arc `i → j` with `j ≠ 1`:
  `t[j] ≥ t[i] + d[i] + c[i,j] − M[i,j]·(1 − x[i,j])`, so a used arc forces the
  arrival at `j` to respect the departure from `i`, while the per-arc big-M
  `M[i,j] = max(0, l[i] + d[i] + c[i,j] − e[j])` (tightened from the window
  data: `t[i] ≤ l[i]`, `t[j] ≥ e[j]`) keeps the row vacuous when the arc is
  unused.
- Appointment windows `e[i] ≤ t[i] ≤ l[i]`, declared as variable bounds so
  presolve propagates them into the big-M analysis.

This is a MIP whose continuous relaxation is a meaningful routing relaxation
(cf. the CLAUDE.md "Model classes" section): the relaxed model is a useful LP
test instance, but a fractional `x` is not a directly implementable schedule.

# Feasibility
- `feasible`: a base nearest-neighbor tour is built first and every window is
  placed around that tour's arrival times (with random slack), so the base
  tour satisfies all windows by construction.
- `infeasible`: two mutually exclusive jobs. Customers `A` and `B` (a
  farthest pair) both get a long service `D = max(c[1,A], c[1,B]) + n + 10`
  and the same early deadline `L = D + c[A,B] − n − 5`; every other customer
  is available "all day" (`[0, H]`). Any tour visits `A` and `B` in some
  order; chaining the time constraints along the tour segment between them
  (which avoids the depot) gives an arrival at the later one of at least
  `D + c[A,B] − n = L + 5` (Euclidean triangle inequality with rounding slack
  `< n`), violating the shared deadline. Each job is individually reachable
  (`L = max(c[1,A], c[1,B]) + c[A,B] + 5`), so the infeasibility is a genuine
  schedule conflict, not a trivially unreachable window. Note that this
  infeasibility lives at the integer level — the LP relaxation is feasible —
  so feasibility-contract verification must be run with `relax_integer = false`.
- `unknown`: windows are the base-tour arrivals plus uniform noise — mostly
  feasible, occasionally not, as in real overbooked schedules.

# Fields
- `n::Int`: number of cities (1 depot + `n-1` customers)
- `points::Vector{Tuple{Int,Int}}`: city coordinates
- `costs::Matrix{Int}`: rounded-Euclidean arc costs (`costs[i,i] = 0`,
  symmetric, off-diagonal entries ≥ 1)
- `services::Vector{Int}`: service duration per city (`services[1] = 0`)
- `early::Vector{Int}` / `late::Vector{Int}`: appointment windows per city
  (depot entries unused)
- `base_tour::Vector{Int}`: the construction tour (a permutation of `1:n`
  starting at the depot)
- `base_arrivals::Vector{Int}`: arrival times along `base_tour`
  (`base_arrivals[1] = 0`)
- `tw_pair::Tuple{Int,Int}`: the mutually exclusive job pair for `infeasible`
  instances, `(0, 0)` otherwise
"""
struct TimeWindowedTSPProblem <: ProblemGenerator
    n::Int
    points::Vector{Tuple{Int,Int}}
    costs::Matrix{Int}
    services::Vector{Int}
    early::Vector{Int}
    late::Vector{Int}
    base_tour::Vector{Int}
    base_arrivals::Vector{Int}
    tw_pair::Tuple{Int,Int}
end

"""
    TimeWindowedTSPProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a TSP-TW instance.

# Variable-count formula
The model has `n(n-1)` binary arc variables, `n-1` position potentials, and
`n-1` arrival times (the depot has neither potential nor time variable):

    total = n*(n-1) + (n-1) + (n-1) = n^2 + n - 2

so `n ≈ round(sqrt(target_variables))` (clamped to `n ≥ 3`). For
`target = 100` this gives `n = 10` (108 vars); for `target = 500`, `n = 22`
(504 vars).

# Arguments
- `target_variables`: Target number of decision variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function TimeWindowedTSPProblem(target_variables::Int,
                                feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    n = max(3, round(Int, sqrt(target_variables)))
    points = _tsp_clustered_points(n)
    costs = _tsp_euclidean_costs(points)
    services = [0; rand(10:60, n - 1)]

    # --- Base tour: nearest neighbor from the depot, lowest-index tie-break ---
    base_tour = Int[1]
    unvisited = Set(2:n)
    while !isempty(unvisited)
        cur = base_tour[end]
        candidates = sort!(collect(unvisited))
        dists = [costs[cur, j] for j in candidates]
        nxt = candidates[argmin(dists)]
        push!(base_tour, nxt)
        delete!(unvisited, nxt)
    end

    # --- Arrival times along the base tour (depot leaves at time 0) ---
    base_arrivals = zeros(Int, n)
    for k in 2:n
        base_arrivals[base_tour[k]] = base_arrivals[base_tour[k-1]] +
            services[base_tour[k-1]] + costs[base_tour[k-1], base_tour[k]]
    end

    early = zeros(Int, n)
    late = zeros(Int, n)
    tw_pair = (0, 0)

    if feasibility_status == feasible
        # Windows around the base-tour arrivals, with random slack on both
        # sides: the base tour satisfies every window by construction.
        for i in 2:n
            early[i] = max(0, base_arrivals[i] - rand(0:30))
            late[i] = base_arrivals[i] + rand(10:40)
        end

    elseif feasibility_status == infeasible
        # Two mutually exclusive jobs: A and B (a farthest pair) get a long
        # service and the same early deadline, so no tour can serve both.
        # margin = n absorbs the rounding slack of the Euclidean
        # triangle-inequality argument chaining travel times along the tour
        # segment between A and B (≤ n-1 arcs, slack ≤ 0.5n < n).
        margin = n
        A, B = (0, 0)
        best_cost = -1
        for i in 2:n, j in i+1:n
            if costs[i, j] > best_cost
                best_cost = costs[i, j]
                A, B = i, j
            end
        end
        tw_pair = (A, B)
        D = max(costs[1, A], costs[1, B]) + margin + 10
        services[A] = D
        services[B] = D
        L = D + costs[A, B] - margin - 5
        early[A] = 0; late[A] = L
        early[B] = 0; late[B] = L
        # Everyone else is available all day; the horizon dominates any
        # reachable arrival (≤ Σd + (n-1)·c_max).
        H = sum(services) + n * maximum(costs) + 10
        for i in 2:n
            if i != A && i != B
                early[i] = 0
                late[i] = H
            end
        end

    else
        # unknown: noisy windows around the base tour — realistic schedules
        # that are usually but not provably feasible.
        for i in 2:n
            early[i] = max(0, base_arrivals[i] + rand(-40:40))
            late[i] = early[i] + rand(20:80)
        end
    end

    return TimeWindowedTSPProblem(n, points, costs, services, early, late,
                                  base_tour, base_arrivals, tw_pair)
end

"""
    build_model(prob::TimeWindowedTSPProblem)

Build a JuMP model for the TSP-TW using the directed MTZ formulation plus
arrival-time variables. Deterministic — uses only data from the struct fields.

Node indexing: city `1` is the depot (no time variable; it leaves at time 0).
Decision variables:

- `x[i,j] ∈ {0,1}`: arc `(i,j)` is traversed
- `2 ≤ u[i] ≤ n` (continuous): position potential of city `i ≠ 1`
- `e[i] ≤ t[i] ≤ l[i]` (continuous): arrival time at city `i ≠ 1` (the
  appointment window is the variable's bound, which presolve exploits)

# Returns
- `model`: The JuMP model
"""
function build_model(prob::TimeWindowedTSPProblem)
    model = Model()

    n = prob.n
    cities = 1:n
    d = prob.services

    # --- Variables ---
    @variable(model, x[i in cities, j in cities; i != j], Bin)
    @variable(model, 2 <= u[2:n] <= n)
    @variable(model, prob.early[i] <= t[i in 2:n] <= prob.late[i])

    # --- Objective: minimize total travel distance ---
    @objective(model, Min,
        sum(prob.costs[i, j] * x[i, j] for i in cities, j in cities if i != j))

    # --- Degree constraints: one out-arc and one in-arc per city ---
    for i in cities
        @constraint(model, sum(x[i, j] for j in cities if j != i) == 1)   # out
        @constraint(model, sum(x[j, i] for j in cities if j != i) == 1)   # in
    end

    # --- MTZ subtour elimination (potentials relative to depot city 1) ---
    for i in 2:n, j in 2:n
        i == j && continue
        @constraint(model, u[i] - u[j] + n * x[i, j] <= n - 1)
    end

    # --- Time propagation along used arcs (no deadline on returning) ---
    # Per-arc big-M tightened from the window data: t_i <= l_i and t_j >= e_j
    # give M_ij = max(0, l_i + d_i + c_ij - e_j), which keeps the row exact for
    # x = 1 and vacuous (RHS <= e_j <= t_j) for x = 0.
    # Arcs out of the depot: the technician leaves at time 0 (l_1 = 0, d_1 = 0).
    for j in 2:n
        M1j = max(0, prob.costs[1, j] - prob.early[j])
        @constraint(model,
            t[j] >= d[1] + prob.costs[1, j] - M1j * (1 - x[1, j]))
    end
    # Arcs between customers.
    for i in 2:n, j in 2:n
        i == j && continue
        Mij = max(0, prob.late[i] + d[i] + prob.costs[i, j] - prob.early[j])
        @constraint(model,
            t[j] >= t[i] + d[i] + prob.costs[i, j] - Mij * (1 - x[i, j]))
    end

    return model
end

# Register the variant (lazily creates the :tsp category).
register_variant(
    :tsp,
    :time_windows,
    TimeWindowedTSPProblem,
    "TSP with service durations and appointment windows (field-service routing), MTZ subtour elimination plus arrival-time variables; a MIP whose relaxation is a routing relaxation",
)
