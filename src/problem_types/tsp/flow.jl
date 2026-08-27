using JuMP
using Random

"""
    TSPFlowProblem <: ProblemGenerator

Generator for the symmetric travelling-salesman problem formulated with
**single-commodity flow** (Gavish–Graves) subtour elimination — the same
formulation family as `vehicle_routing/cvrp`, applied to the one-vehicle tour.

# Overview
Identical story to `tsp/standard` (a tour over delivery stops with symmetric
road distances), and the *same data-generating process* via the shared
`_tsp_stops`/`_tsp_distance` helpers. The variants therefore differ in
formulation class, not in data distribution — though not in instance: the two
formulations size `n` differently (`n ≈ sqrt(target/2)` here vs.
`n ≈ sqrt(target+1)` for `standard`), so equal `target`/`seed` pairs yield
different draws.

The formulation:
- Binary arc variables `x[i,j] ∈ {0,1}` select which arcs are traversed.
- Continuous "supply" variables `f[i,j] ≥ 0` count how many *remaining stops*
  the vehicle still has to serve after arriving at `j` via arc `(i,j)`.

Key structural couplings:
- **Degree constraints** force exactly one incoming and one outgoing arc at
  every node, including the home base.
- **Flow conservation**: each stop consumes exactly one unit of supply
  (`inflow − outflow = 1`); the depot (node 1) sources exactly `n-1` units.
- **Capacity coupling** `f[i,j] ≤ (n-1)·x[i,j]` forbids supply on unused arcs.

Because supply originates only at the depot and drains one unit per stop, no
depot-free cycle can carry flow — subtours are eliminated. The LP relaxation of
this formulation is markedly stronger than MTZ's, which is exactly why it is a
distinct variant: the two deliver structurally different LPs over the same kind
of data. This is a MIP whose continuous relaxation is a genuine depot-anchored
tour relaxation; a fractional `x` is not an implementable tour.

# Fields
- `n_stops::Int`: Total node count `n` (node 1 = home base, nodes `2..n` = stops)
- `locations::Vector{Tuple{Float64,Float64}}`: Node coordinates (index 1 = home base)
- `dist::Matrix{Float64}`: Symmetric road-distance matrix; `dist[i,j] = dist[j,i]`,
  `dist[i,i] = 0`
- `arc_ok::Matrix{Bool}`: Allowed-arc mask; `arc_ok[i,j]` is true iff the model
  creates variables for arc `(i,j)` (always false on the diagonal)
- `blocked_set::Vector{Int}`: The Hall-deficit set `S` (empty unless infeasible)
- `gate_set::Vector{Int}`: The gate set `T` with `|T| = |S| - 1` (empty unless infeasible)
"""
struct TSPFlowProblem <: ProblemGenerator
    n_stops::Int
    locations::Vector{Tuple{Float64,Float64}}
    dist::Matrix{Float64}
    arc_ok::Matrix{Bool}
    blocked_set::Vector{Int}
    gate_set::Vector{Int}
end

"""
    TSPFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a symmetric TSP instance with the single-commodity-flow formulation.

# Variable-count formula
On a complete directed graph over `n` nodes with no self-loops there are
`n*(n-1)` arcs. The model creates one binary `x` and one continuous `f` per arc:

    total = 2 * n * (n - 1)

So `n = max(5, round(Int, sqrt(target_variables / 2)))`. For `target = 100`
this gives `n = 7` (84 vars); for `target = 500`, `n = 16` (480 vars). The
infeasible branch deletes `2*k*(n-k)` variables (`x` and `f` on each blocked
arc) and sizes `n` against the *delivered* count.

# Arguments
- `target_variables`: Target number of decision variables (`x` plus `f`)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Feasibility
- `feasible`: complete arc support. Any permutation of the stops is a tour, and
  the relaxation is nonempty — an explicit witness is `x[i,j] = 1/(n-1)` on all
  arcs with the star supply `f[1,j] = 1`, `f` zero elsewhere (each stop consumes
  its unit directly from the depot, and every capacity row reads
  `1 ≤ (n-1)·(1/(n-1))`) — so both the MIP and the delivered relaxation are
  feasible.
- `infeasible`: Hall-deficit arc block (a set `S` of `k` stops keeps only the
  in-arcs from `k-1` gates `T`), which contradicts the degree rows alone —
  `k = Σ_{j∈S} indeg(j) ≤ Σ_{i∈T} outdeg(i) = k-1` — so the model is infeasible
  even in the LP relaxation (the conservation and capacity rows only shrink the
  feasible set further).
- `unknown`: a natural instance, identical to the feasible branch.
"""
function TSPFlowProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # total = 2 * n * (n - 1)  ≈ 2 * n^2  =>  n ≈ sqrt(target / 2).
    n0 = max(5, round(Int, sqrt(target_variables / 2)))

    # Block size k is drawn unconditionally (RNG alignment across statuses);
    # the infeasible branch sizes n against the delivered count
    # 2*(n^2 - n) - 2*k*(n - k) (the block removes both x and f on each of its
    # k*(n-k) deleted arcs).
    n, k = _tsp_plan_dimensions(n0, target_variables, feasibility_status,
                                (m, kk) -> 2 * (m^2 - m) - 2 * kk * (m - kk))

    # --- Geography and symmetric road distances (shared with tsp/standard) ---
    locations = _tsp_stops(n)
    dist = _tsp_distance(locations)

    # --- Resolve feasibility intent ---
    arc_ok, S, T = _tsp_arc_support(n, k, feasibility_status)

    return TSPFlowProblem(n, locations, dist, arc_ok, S, T)
end

"""
    build_model(prob::TSPFlowProblem)

Build a JuMP model for the symmetric TSP using the single-commodity flow
(Gavish–Graves) formulation. Deterministic — uses only data from the struct
fields.

Node indexing: node `1` is the home base; nodes `2..n` are stops. An arc `(i,j)`
has variables only where `arc_ok[i, j]` is true (the complete graph minus any
Hall block).

Decision variables (two per allowed arc):
- `x[i,j] ∈ {0,1}`: arc `(i,j)` is traversed
- `f[i,j] ≥ 0`: remaining-stop supply carried on arc `(i,j)`

# Returns
- `model`: The JuMP model
"""
function build_model(prob::TSPFlowProblem)
    model = Model()

    n = prob.n_stops
    nodes = 1:n
    stops = 2:n
    depot = 1
    ok(i, j) = prob.arc_ok[i, j]

    # --- Variables: one binary x and one continuous f per allowed arc ---
    # Count = 2 * n * (n - 1) (before any arc deletions)
    @variable(model, x[i in nodes, j in nodes; ok(i, j)], Bin)
    @variable(model, f[i in nodes, j in nodes; ok(i, j)] >= 0)

    # --- Objective: minimize total travel distance ---
    @objective(model, Min,
        sum(prob.dist[i, j] * x[i, j] for i in nodes, j in nodes if ok(i, j)))

    # --- Degree constraints: exactly one in-arc and one out-arc per node ---
    for j in nodes
        @constraint(model, sum(x[i, j] for i in nodes if ok(i, j)) == 1)   # in
        @constraint(model, sum(x[j, k] for k in nodes if ok(j, k)) == 1)   # out
    end

    # --- Supply conservation ---
    # At each stop: inbound supply - outbound supply = 1 (the stop consumes
    # its own unit).
    for j in stops
        @constraint(model,
            sum(f[i, j] for i in nodes if ok(i, j)) -
            sum(f[j, k] for k in nodes if ok(j, k)) == 1)
    end
    # At the home base: net outflow of supply = n - 1 (one unit per stop).
    @constraint(model,
        sum(f[depot, j] for j in nodes if ok(depot, j)) -
        sum(f[i, depot] for i in nodes if ok(i, depot)) == n - 1)

    # --- Capacity coupling: supply only on used arcs ---
    for i in nodes, j in nodes
        ok(i, j) || continue
        @constraint(model, f[i, j] <= (n - 1) * x[i, j])
    end

    return model
end

# Register the variant (standard remains the category default; do NOT pass
# default = true here).
register_variant(
    :tsp,
    :flow,
    TSPFlowProblem,
    "Symmetric travelling-salesman problem with single-commodity-flow subtour elimination; a MIP whose continuous relaxation is a genuine depot-anchored tour relaxation",
)
