using JuMP
using Random

"""
    EuclideanTSPProblem <: ProblemGenerator

Generator for the symmetric Euclidean Traveling Salesman Problem (TSP),
formulated as a mixed-integer program with Miller-Tucker-Zemlin (MTZ) subtour
elimination.

# Overview
`n` cities are placed as integer coordinates on a `[0, 1000]²` grid with a
realistic clustered layout (dense neighborhoods plus scattered outliers); the
cost of an arc is the rounded Euclidean distance. The objective minimizes total
travel distance over a closed tour visiting every city exactly once. This is
the canonical real-world delivery-routing instance.

The formulation is the classic directed MTZ model (symmetric costs, directed
arc variables — every undirected tour corresponds to two directed tours of the
same cost, so the formulation is exact):

- Binary arc variables `x[i,j] ∈ {0,1}` for every directed pair `i ≠ j`.
- **Degree constraints**: each city has exactly one incoming and one outgoing
  arc.
- **MTZ subtour elimination**: continuous position potentials `2 ≤ u[i] ≤ n`
  for `i ≠ 1` (city 1, the depot, is implicitly at position 1) with
  `u[i] - u[j] + n·x[i,j] ≤ n-1` for every pair `i,j ≠ 1`. Any subtour avoiding
  the depot telescopes around its cycle to `n·|S| ≤ (n-1)·|S|`, a
  contradiction; any Hamiltonian tour satisfies the constraints with `u[i]` set
  to the visit position.

This is a MIP whose continuous relaxation is a routing relaxation (cf. the
CLAUDE.md "Model classes" section): the relaxed model is a useful LP test
instance, but a fractional `x` is not a directly implementable tour.

# Feasibility
- `feasible` / `unknown`: the complete network always admits a Hamiltonian
  tour, so both are feasible by construction.
- `infeasible`: a bridge closure cuts the map into a west and an east half and
  forbids every arc crossing the cut in either direction, so no tour can ever
  leave its half of the map and the MIP is infeasible. Note that this
  infeasibility lives at the integer level — the LP relaxation is feasible
  (fractional within-half matchings) — so feasibility-contract verification
  must be run with `relax_integer = false`.

# Fields
- `n::Int`: number of cities
- `points::Vector{Tuple{Int,Int}}`: city coordinates
- `costs::Matrix{Int}`: rounded-Euclidean arc costs (`costs[i,i] = 0`,
  symmetric, off-diagonal entries ≥ 1)
- `forbidden::Set{Tuple{Int,Int}}`: directed arcs that are not available
  (empty unless the status is `infeasible`)
- `partition::Tuple{Vector{Int},Vector{Int}}`: the (west, east) city indices
  of the bridge cut (empty unless the status is `infeasible`)
"""
struct EuclideanTSPProblem <: ProblemGenerator
    n::Int
    points::Vector{Tuple{Int,Int}}
    costs::Matrix{Int}
    forbidden::Set{Tuple{Int,Int}}
    partition::Tuple{Vector{Int},Vector{Int}}
end

"""
    EuclideanTSPProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a Euclidean TSP instance.

# Variable-count formula
The model has `n(n-1)` binary arc variables and `n-1` position potentials:

    total = n*(n-1) + (n-1) = n^2 - 1

so `n ≈ round(sqrt(target_variables))` (clamped to `n ≥ 3`). For `target = 100`
this gives `n = 10` (99 vars); for `target = 500`, `n = 22` (483 vars).
Infeasible instances forbid the crossing arcs of the bridge cut, so their
actual variable count is correspondingly smaller.

# Arguments
- `target_variables`: Target number of decision variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function EuclideanTSPProblem(target_variables::Int,
                             feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    n = max(3, round(Int, sqrt(target_variables)))
    points = _tsp_clustered_points(n)
    costs = _tsp_euclidean_costs(points)

    forbidden = Set{Tuple{Int,Int}}()
    partition = (Int[], Int[])
    if feasibility_status == infeasible
        # Bridge closure: split the map into west/east halves by x-coordinate
        # and forbid every arc crossing the cut in either direction. A
        # Hamiltonian tour must cross the cut, so the MIP is infeasible.
        order = sortperm(points; by=p -> p[1])
        west = order[1:div(n, 2)]
        east = order[div(n, 2)+1:end]
        partition = (west, east)
        for i in west, j in east
            push!(forbidden, (i, j))
            push!(forbidden, (j, i))
        end
    end
    # feasible/unknown: leave the complete network; a tour always exists.

    return EuclideanTSPProblem(n, points, costs, forbidden, partition)
end

"""
    build_model(prob::EuclideanTSPProblem)

Build a JuMP model for the Euclidean TSP using the directed MTZ formulation.
Deterministic — uses only data from the struct fields.

Node indexing: city `1` is the depot. Decision variables (over the available
directed arcs `(i,j)`, `i ≠ j`):

- `x[i,j] ∈ {0,1}`: arc `(i,j)` is traversed
- `2 ≤ u[i] ≤ n` (continuous): position potential of city `i ≠ 1`

# Returns
- `model`: The JuMP model
"""
function build_model(prob::EuclideanTSPProblem)
    model = Model()

    n = prob.n
    cities = 1:n
    available(i, j) = i != j && !((i, j) in prob.forbidden)

    # --- Variables: one binary arc per available directed pair ---
    @variable(model, x[i in cities, j in cities; available(i, j)], Bin)
    @variable(model, 2 <= u[2:n] <= n)

    # --- Objective: minimize total travel distance ---
    @objective(model, Min,
        sum(prob.costs[i, j] * x[i, j] for i in cities, j in cities if available(i, j)))

    # --- Degree constraints: one out-arc and one in-arc per city ---
    for i in cities
        @constraint(model, sum(x[i, j] for j in cities if available(i, j)) == 1)   # out
        @constraint(model, sum(x[j, i] for j in cities if available(j, i)) == 1)   # in
    end

    # --- MTZ subtour elimination (potentials relative to depot city 1) ---
    for i in 2:n, j in 2:n
        available(i, j) || continue
        @constraint(model, u[i] - u[j] + n * x[i, j] <= n - 1)
    end

    return model
end

# Register the variant (lazily creates the :tsp category).
register_variant(
    :tsp,
    :euclidean,
    EuclideanTSPProblem,
    "Symmetric Euclidean TSP over clustered city locations with directed Miller-Tucker-Zemlin (MTZ) subtour elimination; a MIP whose relaxation is a routing relaxation",
    default = true,
)
