using JuMP
using Random

"""
    TSPAsymmetricProblem <: ProblemGenerator

Generator for the asymmetric travelling-salesman problem (ATSP) with
traffic-dependent travel times, formulated with the Miller–Tucker–Zemlin (MTZ)
subtour-elimination constraints as a mixed-integer program with a meaningful
continuous relaxation.

# Overview
An urban courier must visit every stop exactly once and return to the home base
(node 1). Unlike the symmetric `tsp/standard` variant, travel times here are
**direction-dependent**: one-way streets and peak-period congestion make
`time[i,j] ≠ time[j,i]` in general, so the natural data is a directed matrix.

The travel-time matrix is built the way real road networks are used:
- a symmetric base distance per pair (shared geography with the other tsp
  variants),
- a direction-dependent congestion factor `0.8..1.5` per directed arc
  (counter-flow directions stay fast),
- a Floyd–Warshall **metric closure**, i.e. every entry becomes the shortest
  path through the perturbed network — this is exactly "route via side streets
  when the direct road is congested", and it restores the triangle inequality
  while preserving asymmetry.

The formulation is the same MTZ model as `tsp/standard` (it applies verbatim to
asymmetric costs):

- Binary arc variables `x[i,j] ∈ {0,1}` select which arcs are traversed.
- Continuous order variables `u[j] ∈ [1, n-1]` for stops `j = 2..n`.
- Degree constraints (one in-arc, one out-arc per node) and MTZ constraints
  `u[i] - u[j] + n·x[i,j] ≤ n-1` over stop-to-stop arcs.

This is a MIP whose continuous relaxation is a genuine tour relaxation: the
relaxed model is a useful LP test instance, but a fractional `x` is not an
implementable tour.

# Fields
- `n_stops::Int`: Total node count `n` (node 1 = home base, nodes `2..n` = stops)
- `locations::Vector{Tuple{Float64,Float64}}`: Node coordinates (index 1 = home base)
- `dist::Matrix{Float64}`: Asymmetric travel-time matrix over nodes `1..n`
  (minutes); `dist[i,i] = 0`, triangle-inequality-respecting up to rounding,
  generically `dist[i,j] ≠ dist[j,i]`
- `arc_ok::Matrix{Bool}`: Allowed-arc mask; `arc_ok[i,j]` is true iff the model
  creates a variable for arc `(i,j)` (always false on the diagonal)
- `blocked_set::Vector{Int}`: The Hall-deficit set `S` (empty unless infeasible)
- `gate_set::Vector{Int}`: The gate set `T` with `|T| = |S| - 1` (empty unless infeasible)
"""
struct TSPAsymmetricProblem <: ProblemGenerator
    n_stops::Int
    locations::Vector{Tuple{Float64,Float64}}
    dist::Matrix{Float64}
    arc_ok::Matrix{Bool}
    blocked_set::Vector{Int}
    gate_set::Vector{Int}
end

"""
    TSPAsymmetricProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an asymmetric TSP instance with the MTZ formulation.

# Variable-count formula
Identical to `tsp/standard`: one binary `x` per arc plus one continuous order
variable per stop over a complete directed graph on `n` nodes:

    total = n*(n-1) + (n-1) = n^2 - 1

So `n = max(5, round(Int, sqrt(target_variables + 1)))` (the infeasible branch
sizes `n` against the *delivered* count after the Hall block deletes
`k*(n-k)` arc variables).

# Arguments
- `target_variables`: Target number of decision variables (`x` plus `u`)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility

# Feasibility
Identical mechanism to `tsp/standard`; asymmetry of the cost matrix plays no
role in feasibility.
- `feasible`: complete arc support — any permutation is a tour, and the
  relaxation is nonempty (witness `x[i,j] = 1/(n-1)` with every `u ≡ 1`).
- `infeasible`: Hall-deficit arc block (a set `S` of `k` stops keeps only the
  in-arcs from `k-1` gates `T`), which contradicts the degree rows alone —
  `k = Σ_{j∈S} indeg(j) ≤ Σ_{i∈T} outdeg(i) = k-1` — so the model is infeasible
  even in the LP relaxation.
- `unknown`: a natural instance, identical to the feasible branch.
"""
function TSPAsymmetricProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # --- Dimension sizing (same law as tsp/standard) ---
    n0 = max(5, round(Int, sqrt(target_variables + 1)))

    # Draw the Hall-block size unconditionally so the RNG stream stays aligned
    # across statuses (ignored unless infeasible).
    k = n0 >= 8 ? rand(2:3) : 2

    n = n0
    if feasibility_status == infeasible
        n = _tsp_pick_n(n0, target_variables, k, m -> m^2 - 1 - k * (m - k))
    end

    # --- Geography ---
    locations = _tsp_stops(n)

    # --- Asymmetric travel times: symmetric base, per-direction congestion ---
    base = _tsp_distance(locations)
    dist = zeros(n, n)
    for i in 1:n, j in 1:n
        if i != j
            congestion = 0.8 + 0.7 * rand()          # 0.80 .. 1.50 per direction
            dist[i, j] = base[i, j] * congestion
        end
    end

    # --- Metric closure: every entry becomes the shortest path through the
    # perturbed network ("route around the traffic"), which restores the
    # triangle inequality while keeping the matrix asymmetric. ---
    for m in 1:n, i in 1:n, j in 1:n
        through = dist[i, m] + dist[m, j]
        through < dist[i, j] && (dist[i, j] = through)
    end
    dist = round.(dist, digits = 2)

    # --- Resolve feasibility intent ---
    arc_ok = _tsp_full_support(n)
    S = Int[]
    T = Int[]
    if feasibility_status == infeasible
        arc_ok, S, T = _tsp_hall_block(n, k)
    end

    return TSPAsymmetricProblem(n, locations, dist, arc_ok, S, T)
end

"""
    build_model(prob::TSPAsymmetricProblem)

Build a JuMP model for the asymmetric TSP using the Miller–Tucker–Zemlin (MTZ)
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
function build_model(prob::TSPAsymmetricProblem)
    model = Model()

    n = prob.n_stops
    nodes = 1:n
    stops = 2:n
    ok(i, j) = prob.arc_ok[i, j]

    # --- Variables: one binary x per allowed arc, one order var per stop ---
    @variable(model, x[i in nodes, j in nodes; ok(i, j)], Bin)
    @variable(model, 1 <= u[j in stops] <= n - 1)

    # --- Objective: minimize total travel time ---
    @objective(model, Min,
        sum(prob.dist[i, j] * x[i, j] for i in nodes, j in nodes if ok(i, j)))

    # --- Degree constraints: exactly one in-arc and one out-arc per node ---
    for j in nodes
        @constraint(model, sum(x[i, j] for i in nodes if ok(i, j)) == 1)   # in
        @constraint(model, sum(x[j, k] for k in nodes if ok(j, k)) == 1)   # out
    end

    # --- MTZ subtour elimination over stop-to-stop arcs ---
    for i in stops, j in stops
        (i != j && ok(i, j)) || continue
        @constraint(model, u[i] - u[j] + n * x[i, j] <= n - 1)
    end

    return model
end

# Register the variant (standard remains the category default; do NOT pass
# default = true here).
register_variant(
    :tsp,
    :asymmetric,
    TSPAsymmetricProblem,
    "Asymmetric travelling-salesman problem with traffic-dependent travel times (metric closure of a congested road network) and MTZ subtour elimination; a MIP whose continuous relaxation is a compact big-M tour relaxation",
)
