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

Travel times are shortest paths on an explicit urban street grid. Horizontal
streets are one-way and alternate direction by row; vertical avenues are
two-way. Each street has a sampled positive congestion weight. The resulting
directed shortest-path matrix is strongly connected, satisfies the directed
triangle inequality exactly, and is genuinely asymmetric rather than being an
independent perturbation of every city pair.

The formulation is the same MTZ model as `tsp/standard` (it applies verbatim to
asymmetric costs):

  - Binary arc variables `x[i,j] ∈ {0,1}` select which arcs are traversed.
  - Continuous order variables `u[j] ∈ [1, n-1]` for stops `j = 2..n`.
  - Degree constraints (one in-arc, one out-arc per node) and lifted MTZ
    constraints using both directions of each stop pair.

This is a MIP whose continuous relaxation is a genuine tour relaxation: the
relaxed model is a useful LP test instance, but a fractional `x` is not an
implementable tour.

# Fields

  - `n_stops::Int`: Total node count `n` (node 1 = home base, nodes `2..n` = stops)
  - `locations::Vector{Tuple{Float64,Float64}}`: Street-grid coordinates (index 1 = home base)
  - `grid_side::Int`: Side length of the one-way street grid
  - `row_weight::Vector{Int}`: Congestion weight for each one-way horizontal street
  - `col_weight::Vector{Int}`: Congestion weight for each two-way vertical avenue
  - `dist::Matrix{Float64}`: Asymmetric travel-time matrix over nodes `1..n`
    (minutes); `dist[i,i] = 0`, satisfies the directed triangle inequality exactly,
    generically `dist[i,j] ≠ dist[j,i]`
  - `arc_ok::Matrix{Bool}`: Allowed-arc mask; `arc_ok[i,j]` is true iff the model
    creates a variable for arc `(i,j)` (always false on the diagonal)
  - `blocked_set::Vector{Int}`: The Hall-deficit set `S` (empty unless infeasible)
  - `gate_set::Vector{Int}`: The gate set `T` with `|T| = |S| - 1` (empty unless infeasible)
"""
struct TSPAsymmetricProblem <: ProblemGenerator
    n_stops::Int
    locations::Vector{Tuple{Float64, Float64}}
    grid_side::Int
    row_weight::Vector{Int}
    col_weight::Vector{Int}
    dist::Matrix{Float64}
    arc_ok::Matrix{Bool}
    blocked_set::Vector{Int}
    gate_set::Vector{Int}
end

"""
Shortest paths from one vertex of the alternating one-way street grid.

`distances` and `buckets` are caller-owned scratch buffers, refilled on every
call so one pair can be reused across the per-source calls of a single
instance; the defaults allocate a fresh pair. Returning `distances` hands back
the shared buffer, so copy out any values before the next call.
"""
function _tsp_street_shortest_paths(
    S::Int,
    row_weight::Vector{Int},
    col_weight::Vector{Int},
    source::Int,
    distances::Vector{Int}=fill(typemax(Int), S * S),
    buckets::Vector{Vector{Int}}=[Int[] for _ in 0:(3 * (S * S - 1))],
)
    infinity = typemax(Int)
    fill!(distances, infinity)
    foreach(empty!, buckets)
    distances[source] = 0

    # Positive weights are at most three, so a Dial bucket queue is simpler and
    # faster than repeatedly scanning the full grid. A shortest simple path has
    # at most n_vertices - 1 edges.
    max_distance = 3 * (S * S - 1)
    push!(buckets[1], source)

    for distance in 0:max_distance
        while !isempty(buckets[distance + 1])
            vertex = pop!(buckets[distance + 1])
            distances[vertex] == distance || continue
            row = div(vertex - 1, S) + 1
            col = rem(vertex - 1, S) + 1

            # Odd rows run west, even rows east.
            next_col = col + (iseven(row) ? 1 : -1)
            if 1 <= next_col <= S
                next_vertex = (row - 1) * S + next_col
                candidate = distance + row_weight[row]
                if candidate <= max_distance && candidate < distances[next_vertex]
                    distances[next_vertex] = candidate
                    push!(buckets[candidate + 1], next_vertex)
                end
            end

            # Avenues are traversable in both directions.
            for next_row in (row - 1, row + 1)
                1 <= next_row <= S || continue
                next_vertex = (next_row - 1) * S + col
                candidate = distance + col_weight[col]
                if candidate <= max_distance && candidate < distances[next_vertex]
                    distances[next_vertex] = candidate
                    push!(buckets[candidate + 1], next_vertex)
                end
            end
        end
    end
    return distances
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
function TSPAsymmetricProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)

    # --- Dimension sizing (same law as tsp/standard) ---
    n0 = max(5, round(Int, sqrt(target_variables + 1)))

    # Block size k is drawn unconditionally (RNG alignment across statuses);
    # the infeasible branch sizes n against the delivered count.
    n, k = _tsp_plan_dimensions(
        rng, n0, target_variables, feasibility_status, (m, kk) -> m^2 - 1 - kk * (m - kk)
    )

    # --- Explicit one-way street geography ---
    grid_side = 2 * n
    depot_vertex = (div(grid_side, 2) - 1) * grid_side + div(grid_side, 2)
    fixed_vertices = [depot_vertex, 1, grid_side]
    remaining = [v for v in 1:(grid_side * grid_side) if !(v in fixed_vertices)]
    city_vertices = vcat(fixed_vertices, shuffle(rng, remaining)[1:(n - 3)])
    city_coords = [(div(v - 1, grid_side) + 1, rem(v - 1, grid_side) + 1) for v in city_vertices]
    locations = [(Float64(row), Float64(col)) for (row, col) in city_coords]
    row_weight = rand(rng, 1:3, grid_side)
    col_weight = rand(rng, 1:3, grid_side)

    # Directed shortest-path closure of the street network. The two fixed
    # endpoints on row 1 make asymmetry deterministic; the central depot and
    # remaining random stops retain realistic spatial variety.
    dist = zeros(n, n)
    infinity = typemax(Int)
    street_distances = fill(infinity, grid_side * grid_side)
    buckets = [Int[] for _ in 0:(3 * (grid_side ^ 2 - 1))]
    for i in 1:n
        _tsp_street_shortest_paths(
            grid_side, row_weight, col_weight, city_vertices[i], street_distances, buckets
        )
        for j in 1:n
            d = street_distances[city_vertices[j]]
            d == infinity && error("tsp/asymmetric street grid unexpectedly disconnected")
            dist[i, j] = Float64(d)
        end
    end
    @assert any(dist[i, j] != dist[j, i] for i in 1:n for j in (i + 1):n)

    # --- Resolve feasibility intent ---
    arc_ok, blocked_set, gate_set = _tsp_arc_support(rng, n, k, feasibility_status)

    return TSPAsymmetricProblem(
        n, locations, grid_side, row_weight, col_weight, dist, arc_ok, blocked_set, gate_set
    )
end

"""
    build_model(prob::TSPAsymmetricProblem)

Build a JuMP model for the asymmetric TSP using the lifted
Miller–Tucker–Zemlin (MTZ) formulation. Deterministic — uses only data from the
struct fields.

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

# Register the variant (standard remains the category default; do NOT pass
# default = true here).
register_variant(
    :tsp,
    :asymmetric,
    TSPAsymmetricProblem,
    "Asymmetric travelling-salesman problem with shortest-path travel times on an alternating one-way street grid and lifted MTZ subtour elimination; a MIP whose continuous relaxation is a compact big-M tour relaxation",
)
