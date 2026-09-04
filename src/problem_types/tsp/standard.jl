using JuMP
using Random

"""
    TSPStandardProblem <: ProblemGenerator

Generator for the symmetric travelling-salesman problem (TSP) — a courier or
field-service tour over delivery stops — formulated with the Miller–Tucker–Zemlin
(MTZ) subtour-elimination constraints as a mixed-integer program with a
meaningful continuous relaxation.

# Overview

A vehicle starts at its home base (node 1), must visit every stop exactly once,
and returns. The network is a complete directed graph over `n` nodes (no
self-loops); road distances are symmetric, so opposite arcs cost the same. The
objective minimizes total travel distance.

The formulation is the lifted polynomial-size MTZ model:

  - Binary arc variables `x[i,j] ∈ {0,1}` select which arcs are traversed.
  - Continuous order variables `u[j] ∈ [1, n-1]` for stops `j = 2..n` record the
    visit position of stop `j` along the tour (the home base is position 0 and
    carries no `u`).

Key structural couplings:

  - **Degree constraints** force exactly one incoming and one outgoing arc at
    every node, including the home base.
  - **Lifted MTZ constraints**
    `u[i] - u[j] + (n-1)x[i,j] + (n-3)x[j,i] ≤ n-2` over stop-to-stop arcs
    eliminate subtours and strengthen the continuous relaxation by using both
    directions of each customer pair.

The MTZ relaxation is much weaker than exponential subtour-elimination (DFJ)
models, but it is a compact and structurally rich LP: dense degree rows plus an
`O(n²)` block of big-M order rows coupling binary and continuous variables.
This is a MIP whose continuous relaxation is a genuine tour relaxation (cf. the
CLAUDE.md "Model classes" section): the relaxed model is a useful LP test
instance, but a fractional `x` is not an implementable tour.

# Fields

  - `n_stops::Int`: Total node count `n` (node 1 = home base, nodes `2..n` = stops)
  - `locations::Vector{Tuple{Float64,Float64}}`: Node coordinates (index 1 = home base)
  - `dist::Matrix{Float64}`: Symmetric road-distance matrix over nodes `1..n`;
    `dist[i,j] = dist[j,i]`, `dist[i,i] = 0`
  - `arc_ok::Matrix{Bool}`: Allowed-arc mask; `arc_ok[i,j]` is true iff the model
    creates a variable for arc `(i,j)` (always false on the diagonal)
  - `blocked_set::Vector{Int}`: The Hall-deficit set `S` (empty unless infeasible)
  - `gate_set::Vector{Int}`: The gate set `T` with `|T| = |S| - 1` (empty unless infeasible)
"""
struct TSPStandardProblem <: ProblemGenerator
    n_stops::Int
    locations::Vector{Tuple{Float64, Float64}}
    dist::Matrix{Float64}
    arc_ok::Matrix{Bool}
    blocked_set::Vector{Int}
    gate_set::Vector{Int}
end

"""
    TSPStandardProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a symmetric TSP instance with the MTZ formulation.

# Variable-count formula

On a complete directed graph over `n` nodes with no self-loops there are
`n*(n-1)` arcs. The model creates one binary `x` per arc plus one continuous
order variable per stop (nodes `2..n`):

    total = n*(n-1) + (n-1) = n^2 - 1

So `n = max(5, round(Int, sqrt(target_variables + 1)))`. For `target = 100`
this gives `n = 10` (99 vars); for `target = 500`, `n = 22` (483 vars). The
infeasible branch deletes `k*(n-k)` arc variables (see below) and sizes `n`
against the *delivered* count.

# Arguments

  - `target_variables`: Target number of decision variables (`x` plus `u`)
  - `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
  - `seed`: Random seed for reproducibility

# Feasibility

  - `feasible`: complete arc support. Any permutation of the stops is a tour, and
    the relaxation is nonempty — an explicit witness is `x[i,j] = 1/(n-1)` on all
    arcs with every `u ≡ 1` (each degree row sums to 1, and every lifted MTZ row
    reduces to `1 + (n-3)/(n-1) = (2n-4)/(n-1) ≤ n-2`, which holds for `n ≥ 3`)
    — so both the MIP and the delivered relaxation are feasible.
  - `infeasible`: a Hall-deficit arc block stands in for road closures cutting off
    a district: a set `S` of `k` stops keeps only the in-arcs from `k-1` gate
    nodes `T`. The in-degree rows of `S` must sum to `k` yet can draw only on the
    `k-1` unit out-degrees of `T` — a contradiction obtained from the degree rows
    alone, so the model is infeasible even in the LP relaxation (the MTZ rows only
    shrink the feasible set further). No degree row degenerates to an empty sum
    because `|T| ≥ 1`.
  - `unknown`: a natural instance, identical to the feasible branch (complete-graph
    TSPs are always feasible; the bias is moot, but the branch is kept explicit).
"""
function TSPStandardProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)

    # --- Dimension sizing ---
    # total = n^2 - 1  =>  n ≈ sqrt(target + 1).
    n0 = max(5, round(Int, sqrt(target_variables + 1)))

    # Block size k is drawn unconditionally (RNG alignment across statuses);
    # the infeasible branch sizes n against the delivered count
    # n^2 - 1 - k*(n - k), because the block deletes k*(n-k) arc variables.
    n, k = _tsp_plan_dimensions(
        rng, n0, target_variables, feasibility_status, (m, kk) -> m^2 - 1 - kk * (m - kk)
    )

    # --- Geography and symmetric road distances ---
    locations = _tsp_stops(rng, n)
    dist = _tsp_distance(rng, locations)

    # --- Resolve feasibility intent ---
    # feasible / unknown: complete support — see constructor docstring.
    arc_ok, S, T = _tsp_arc_support(rng, n, k, feasibility_status)

    return TSPStandardProblem(n, locations, dist, arc_ok, S, T)
end

"""
    build_model(prob::TSPStandardProblem)

Build a JuMP model for the symmetric TSP using the Miller–Tucker–Zemlin (MTZ)
formulation. Deterministic — uses only data from the struct fields.

Node indexing: node `1` is the home base; nodes `2..n` are stops. An arc `(i,j)`
has a variable only where `arc_ok[i, j]` is true (the complete graph minus any
Hall block).

Decision variables (one per allowed arc, one per stop):

  - `x[i,j] ∈ {0,1}`: arc `(i,j)` is traversed
  - `u[j] ∈ [1, n-1]`: visit position of stop `j` along the tour

# Returns

  - `model`: The JuMP model
"""
function build_model(prob::TSPStandardProblem)
    model = Model()

    n = prob.n_stops
    nodes = 1:n
    stops = 2:n
    ok(i, j) = prob.arc_ok[i, j]

    # --- Variables: one binary x per allowed arc, one order var per stop ---
    # Count = n*(n-1) + (n-1) = n^2 - 1 (before any arc deletions)
    @variable(model, x[i in nodes, j in nodes; ok(i, j)], Bin)
    @variable(model, 1 <= u[j in stops] <= n - 1)

    # --- Objective: minimize total travel distance ---
    @objective(model, Min, sum(prob.dist[i, j] * x[i, j] for i in nodes, j in nodes if ok(i, j)))

    # --- Degree constraints: exactly one in-arc and one out-arc per node ---
    for j in nodes
        @constraint(model, sum(x[i, j] for i in nodes if ok(i, j)) == 1)   # in
        @constraint(model, sum(x[j, k] for k in nodes if ok(j, k)) == 1)   # out
    end

    # --- Lifted MTZ subtour elimination over stop-to-stop arcs ---
    for i in stops, j in stops
        (i != j && ok(i, j)) || continue
        if ok(j, i)
            @constraint(model, u[i] - u[j] + (n - 1) * x[i, j] + (n - 3) * x[j, i] <= n - 2)
        else
            @constraint(model, u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)
        end
    end

    return model
end

# Register the variant (category default).
register_variant(
    :tsp,
    :standard,
    TSPStandardProblem,
    "Symmetric travelling-salesman problem over clustered delivery stops with lifted Miller-Tucker-Zemlin subtour elimination";
    default=true,
)
