using JuMP
using Random

"""
    TSPAssignmentRelaxationProblem <: ProblemGenerator

Generator for a **strengthened degree LP relaxation** of the symmetric
travelling-salesman problem, delivered as a standalone LP test instance.

# Overview
This variant is an LP: the arc variables are continuous fractions
`x[i,j] ∈ [0, 1]` (integrality is never declared, in the manner of
`nurse_scheduling` and `unit_commitment`), so the model is an LP even with
`relax_integer = false`. The formulation keeps:

- **Degree constraints**: exactly one incoming and one outgoing arc at every
  node (a fractional cycle cover; the bipartite assignment relaxation of the
  tour);
- **Pairwise two-cycle cuts** `x[i,j] + x[j,i] ≤ 1` on every unordered pair,
  which rule out the integer two-cycles admitted by the plain directed
  assignment relaxation.

Exponential subtour-elimination (DFJ) cuts are deliberately *not* included —
they cannot be enumerated polynomially — so the polytope is genuinely weaker
than the TSP's: optimal solutions can be fractional, and integer optima can
decompose into several disconnected subtours. This is the intended object: the
LP lower bound used by TSP solvers, dense and cheap, as an LP-solver test
instance. Never present a solution of this model as a tour.

# Fields
- `n_stops::Int`: Total node count `n` (node 1 = home base, nodes `2..n` = stops)
- `locations::Vector{Tuple{Float64,Float64}}`: Node coordinates (index 1 = home base)
- `dist::Matrix{Float64}`: Symmetric road-distance matrix; `dist[i,j] = dist[j,i]`,
  `dist[i,i] = 0`
- `arc_ok::Matrix{Bool}`: Allowed-arc mask; `arc_ok[i,j]` is true iff the model
  creates a variable for arc `(i,j)` (always false on the diagonal)
- `blocked_set::Vector{Int}`: The Hall-deficit set `S` (empty unless infeasible)
- `gate_set::Vector{Int}`: The gate set `T` with `|T| = |S| - 1` (empty unless infeasible)
"""
struct TSPAssignmentRelaxationProblem <: ProblemGenerator
    n_stops::Int
    locations::Vector{Tuple{Float64,Float64}}
    dist::Matrix{Float64}
    arc_ok::Matrix{Bool}
    blocked_set::Vector{Int}
    gate_set::Vector{Int}
end

"""
    TSPAssignmentRelaxationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct the strengthened degree LP relaxation of a symmetric TSP.

# Variable-count formula
On a complete directed graph over `n` nodes with no self-loops there are
`n*(n-1)` arcs, and this LP creates exactly one continuous `x` per arc:

    total = n*(n-1) = n^2 - n

So `n = max(5, round(Int, (1 + sqrt(1 + 4*target_variables)) / 2))`. For
`target = 100` this gives `n = 11` (110 vars); for `target = 500`, `n = 23`
(506 vars). The infeasible branch deletes `k*(n-k)` arc variables and sizes `n`
against the *delivered* count.

# Arguments
- `target_variables`: Target number of decision variables (arc fractions `x`)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Feasibility
- `feasible`: complete arc support — `x[i,j] = 1/(n-1)` on all arcs satisfies
  every degree row and every pairwise subtour row (each row reads
  `2/(n-1) ≤ 1` for `n ≥ 3`), so the LP is feasible and bounded.
- `infeasible`: Hall-deficit arc block (a set `S` of `k` stops keeps only the
  in-arcs from `k-1` gates `T`), which contradicts the degree rows alone —
  `k = Σ_{j∈S} indeg(j) ≤ Σ_{i∈T} outdeg(i) = k-1` — so the LP is infeasible.
- `unknown`: a natural instance, identical to the feasible branch.
"""
function TSPAssignmentRelaxationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing ---
    # total = n^2 - n  =>  n = (1 + sqrt(1 + 4*target)) / 2.
    n0 = max(5, round(Int, (1 + sqrt(1 + 4 * target_variables)) / 2))

    # Draw the Hall-block size unconditionally so the RNG stream stays aligned
    # across statuses (ignored unless infeasible).
    k = n0 >= 8 ? rand(2:3) : 2

    n = n0
    if feasibility_status == infeasible
        # Delivered count: n^2 - n - k*(n - k) variables.
        n = _tsp_pick_n(n0, target_variables, k, m -> m^2 - m - k * (m - k))
    end

    # --- Geography and symmetric road distances (shared with tsp/standard) ---
    locations = _tsp_stops(n)
    dist = _tsp_distance(locations)

    # --- Resolve feasibility intent ---
    arc_ok = _tsp_full_support(n)
    S = Int[]
    T = Int[]
    if feasibility_status == infeasible
        arc_ok, S, T = _tsp_hall_block(n, k)
    end

    return TSPAssignmentRelaxationProblem(n, locations, dist, arc_ok, S, T)
end

"""
    build_model(prob::TSPAssignmentRelaxationProblem)

Build a JuMP model for the strengthened degree LP relaxation of the symmetric
TSP. Deterministic — uses only data from the struct fields.

Node indexing: node `1` is the home base; nodes `2..n` are stops. An arc `(i,j)`
has a variable only where `arc_ok[i, j]` is true (the complete graph minus any
Hall block).

Decision variables (one per allowed arc):
- `x[i,j] ∈ [0, 1]`: fraction of arc `(i,j)` selected — continuous by design,
  so the delivered model is an LP even with `relax_integer = false`

# Returns
- `model`: The JuMP model
"""
function build_model(prob::TSPAssignmentRelaxationProblem)
    model = Model()

    n = prob.n_stops
    nodes = 1:n
    ok(i, j) = prob.arc_ok[i, j]

    # --- Variables: one continuous arc fraction per allowed arc ---
    # Count = n*(n-1) (before any arc deletions)
    @variable(model, 0 <= x[i in nodes, j in nodes; ok(i, j)] <= 1)

    # --- Objective: minimize total travel distance of the fractional cover ---
    @objective(model, Min,
        sum(prob.dist[i, j] * x[i, j] for i in nodes, j in nodes if ok(i, j)))

    # --- Degree constraints: exactly one in-arc and one out-arc per node ---
    for j in nodes
        @constraint(model, sum(x[i, j] for i in nodes if ok(i, j)) == 1)   # in
        @constraint(model, sum(x[j, k] for k in nodes if ok(j, k)) == 1)   # out
    end

    # --- Pairwise subtour elimination: no two-node subtour on any pair ---
    for i in nodes, j in i+1:n
        (ok(i, j) && ok(j, i)) || continue
        @constraint(model, x[i, j] + x[j, i] <= 1)
    end

    return model
end

# Register the variant (standard remains the category default; do NOT pass
# default = true here).
register_variant(
    :tsp,
    :assignment_relaxation,
    TSPAssignmentRelaxationProblem,
    "Strengthened degree LP relaxation of the travelling-salesman problem with pairwise two-cycle cuts; a pure LP whose solutions may be fractional and may contain subtours",
)
