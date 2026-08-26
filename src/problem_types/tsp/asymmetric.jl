using JuMP
using Random

"""
    AsymmetricTSPProblem <: ProblemGenerator

Generator for the Asymmetric Traveling Salesman Problem (ATSP) on a one-way
street grid, formulated as a mixed-integer program with single-commodity-flow
(Gavish-Graves) subtour elimination.

# Overview
`n` cities sit at distinct points of an `S × S` street grid. Horizontal streets
are one-way and alternate direction by row parity (even rows eastbound, odd
rows westbound); vertical avenues are two-way. Every street direction carries
an integer congestion weight in `{1,2,3}`. The cost of an arc `(i,j)` is the
directed shortest-path travel time along the street network — a realistic
urban-logistics metric: it satisfies the directed triangle inequality and is
genuinely asymmetric (crossing town against the one-way rows forces detours).
The same data model describes sequence-dependent machine setup costs.

The formulation mirrors the package's CVRP generator specialized to a single
vehicle with unit demands:

- Binary arc variables `x[i,j] ∈ {0,1}` for every directed pair `i ≠ j`.
- **Degree constraints**: each city has exactly one incoming and one outgoing
  arc.
- **Single-commodity flow** `f[i,j] ≥ 0`: `n-1` units of flow leave the depot
  (city 1), one unit is consumed at every other city, and
  `f[i,j] ≤ (n-1)·x[i,j]` forces flow onto used arcs only. This anchors all
  flow to the depot, so the support of `x` must contain a depot-rooted
  arborescence; combined with the degree constraints the support is exactly a
  Hamiltonian tour. (The LP relaxation is correspondingly a genuine routing
  relaxation, stronger than MTZ.)

This is a MIP whose continuous relaxation is a meaningful routing relaxation
(cf. the CLAUDE.md "Model classes" section): the relaxed model is a useful LP
test instance, but a fractional `x` is not a directly implementable tour.

# Feasibility
- `feasible` / `unknown`: the street network is strongly connected (rows
  alternate direction and columns are two-way, so any city can reach any
  other), hence a Hamiltonian tour always exists.
- `infeasible`: a bridge closure forbids every city-to-city arc crossing a
  west/east cut. No tour can cross the cut, and this holds even in the LP
  relaxation: summing flow balance over the depot-free side demands a positive
  net inflow while the cut carries no flow.

# Fields
- `n::Int`: number of cities
- `grid_side::Int`: street-grid side length `S = 2n`
- `city_coords::Vector{Tuple{Int,Int}}`: city locations on the grid
- `row_weight::Vector{Int}`: congestion weight of each row's one-way direction
- `col_weight::Vector{Int}`: congestion weight of each (two-way) column
- `costs::Matrix{Int}`: directed shortest-path arc costs (`costs[i,i] = 0`,
  off-diagonal entries ≥ 1)
- `forbidden::Set{Tuple{Int,Int}}`: directed arcs that are not available
  (empty unless the status is `infeasible`)
- `partition::Tuple{Vector{Int},Vector{Int}}`: the (west, east) city indices
  of the bridge cut (empty unless the status is `infeasible`)
"""
struct AsymmetricTSPProblem <: ProblemGenerator
    n::Int
    grid_side::Int
    city_coords::Vector{Tuple{Int,Int}}
    row_weight::Vector{Int}
    col_weight::Vector{Int}
    costs::Matrix{Int}
    forbidden::Set{Tuple{Int,Int}}
    partition::Tuple{Vector{Int},Vector{Int}}
end

"""
    AsymmetricTSPProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an ATSP instance on a one-way street grid.

# Variable-count formula
The model has one binary `x` and one continuous `f` per directed pair of
cities:

    total = 2 * n * (n - 1)

so `n ≈ round(sqrt(target_variables / 2))` (clamped to `n ≥ 3`). For
`target = 100` this gives `n = 7` (84 vars); for `target = 500`, `n = 16`
(480 vars). Infeasible instances forbid the crossing arcs of the bridge cut,
so their actual variable count is correspondingly smaller.

# Arguments
- `target_variables`: Target number of decision variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function AsymmetricTSPProblem(target_variables::Int,
                              feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    n = max(3, round(Int, sqrt(target_variables / 2)))
    S = 2n                          # even grid side; rows alternate direction

    # Distinct city locations and per-street congestion weights.
    perm = randperm(S * S)
    city_coords = [(div(p - 1, S) + 1, rem(p - 1, S) + 1) for p in perm[1:n]]
    row_weight = rand(1:3, S)       # one-way horizontal streets
    col_weight = rand(1:3, S)       # two-way vertical avenues

    # Directed shortest-path costs (Dijkstra per city). The grid is strongly
    # connected by construction (see _tsp_grid_shortest_path), so all costs
    # are finite; the guard turns a future regression into a loud error rather
    # than a silently corrupt instance.
    costs = zeros(Int, n, n)
    for i in 1:n
        dist = _tsp_grid_shortest_path(S, row_weight, col_weight,
                                       (city_coords[i][1] - 1) * S + city_coords[i][2])
        for j in 1:n
            costs[i, j] = dist[(city_coords[j][1] - 1) * S + city_coords[j][2]]
        end
    end
    all(isfinite, costs) ||
        error("tsp/asymmetric: one-way street grid is not strongly connected " *
              "(grid_side = $S); costs contain Inf")

    forbidden = Set{Tuple{Int,Int}}()
    partition = (Int[], Int[])
    if feasibility_status == infeasible
        # Bridge closure: split the city by grid column (west/east) and forbid
        # every arc crossing the cut in either direction. No tour can cross the
        # cut; moreover the cut carries no flow, so flow balance on the
        # depot-free side is unsatisfiable even in the LP relaxation.
        order = sortperm(city_coords; by=p -> p[2])
        west = order[1:div(n, 2)]
        east = order[div(n, 2)+1:end]
        partition = (west, east)
        for i in west, j in east
            push!(forbidden, (i, j))
            push!(forbidden, (j, i))
        end
    end
    # feasible/unknown: leave the strongly connected network; a tour always exists.

    return AsymmetricTSPProblem(n, S, city_coords, row_weight, col_weight,
                                costs, forbidden, partition)
end

"""
    _tsp_grid_shortest_path(S, row_weight, col_weight, source) -> Vector{Int}

Single-source shortest paths on the one-way street grid: horizontal streets
are one-way and alternate direction by row parity (even rows eastbound, odd
rows westbound) with weight `row_weight[r]`; vertical avenues are two-way with
weight `col_weight[c]`. Edge weights are positive integers in `{1,2,3}`, so
Dijkstra runs with a Dial bucket queue over distances `0 .. 3*(S²-1)`.

Strong connectivity: any node can move vertically anywhere along a two-way
avenue; to move against a row's one-way direction, step one row over (opposite
direction) and back. Hence all distances are finite for any `S ≥ 2`.
"""
function _tsp_grid_shortest_path(S::Int, row_weight::Vector{Int},
                                 col_weight::Vector{Int}, source::Int)
    V = S * S
    INF = typemax(Int)
    dist = fill(INF, V)
    dist[source] = 0
    maxd = 3 * (V - 1)                       # longest possible simple path
    buckets = [Int[] for _ in 0:maxd]
    push!(buckets[1], source)
    for d in 0:maxd
        while !isempty(buckets[d + 1])
            v = pop!(buckets[d + 1])
            dist[v] == d || continue         # stale entry
            r = div(v - 1, S) + 1
            c = rem(v - 1, S) + 1
            # Horizontal: one-way street, direction fixed by row parity.
            c2 = c + (iseven(r) ? 1 : -1)
            if 1 <= c2 <= S
                u = (r - 1) * S + c2
                w = d + row_weight[r]
                # The `w <= maxd` check keeps a disconnected grid from indexing
                # past the buckets; unreachable nodes keep `dist = Inf` and are
                # caught by the constructor's finiteness guard.
                if w <= maxd && w < dist[u]
                    dist[u] = w
                    push!(buckets[w + 1], u)
                end
            end
            # Vertical: two-way avenue.
            for r2 in (r - 1, r + 1)
                1 <= r2 <= S || continue
                u = (r2 - 1) * S + c
                w = d + col_weight[c]
                if w <= maxd && w < dist[u]
                    dist[u] = w
                    push!(buckets[w + 1], u)
                end
            end
        end
    end
    return dist
end

"""
    build_model(prob::AsymmetricTSPProblem)

Build a JuMP model for the ATSP using the single-commodity flow (Gavish-Graves)
formulation. Deterministic — uses only data from the struct fields.

Node indexing: city `1` is the depot. Decision variables (over the available
directed arcs `(i,j)`, `i ≠ j`):

- `x[i,j] ∈ {0,1}`: arc `(i,j)` is traversed
- `f[i,j] ≥ 0`: single-commodity flow carried on arc `(i,j)`; `n-1` units
  leave the depot and one unit is consumed at each other city

# Returns
- `model`: The JuMP model
"""
function build_model(prob::AsymmetricTSPProblem)
    model = Model()

    n = prob.n
    cities = 1:n
    depot = 1
    available(i, j) = i != j && !((i, j) in prob.forbidden)

    # --- Variables: one binary x and one continuous f per available arc ---
    @variable(model, x[i in cities, j in cities; available(i, j)], Bin)
    @variable(model, f[i in cities, j in cities; available(i, j)] >= 0)

    # --- Objective: minimize total travel cost ---
    @objective(model, Min,
        sum(prob.costs[i, j] * x[i, j] for i in cities, j in cities if available(i, j)))

    # --- Degree constraints: one out-arc and one in-arc per city ---
    for i in cities
        @constraint(model, sum(x[i, j] for j in cities if available(i, j)) == 1)   # out
        @constraint(model, sum(x[j, i] for j in cities if available(j, i)) == 1)   # in
    end

    # --- Single-commodity flow conservation ---
    # n-1 units leave the depot; one unit is consumed at each other city.
    @constraint(model,
        sum(f[depot, j] for j in cities if available(depot, j)) == n - 1)
    for i in 2:n
        @constraint(model,
            sum(f[j, i] for j in cities if available(j, i)) -
            sum(f[i, j] for j in cities if available(i, j)) == 1)
    end

    # --- Capacity coupling: flow only on used arcs ---
    for i in cities, j in cities
        available(i, j) || continue
        @constraint(model, f[i, j] <= (n - 1) * x[i, j])
    end

    return model
end

# Register the variant (lazily creates the :tsp category).
register_variant(
    :tsp,
    :asymmetric,
    AsymmetricTSPProblem,
    "Asymmetric TSP on a one-way street grid with congestion-weighted shortest-path costs, single-commodity-flow (Gavish-Graves) subtour elimination; a MIP whose relaxation is a routing relaxation",
)
