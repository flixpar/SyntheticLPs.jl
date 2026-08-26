using JuMP
using Random

"""
    TSPAssignmentRelaxationProblem <: ProblemGenerator

Generator for the assignment (2-matching) relaxation of the Traveling Salesman
Problem — a pure LP.

# Overview
Cities are placed on a clustered integer grid with rounded-Euclidean costs,
exactly as in the `tsp/euclidean` variant. The model keeps only the degree
constraints of the TSP: continuous arc weights `x[i,j] ∈ [0,1]` with exactly
one unit of outgoing and one unit of incoming arc mass per city, minimizing
total travel cost. There is no subtour elimination, so an optimal solution
generally decomposes into several fractional subtours — this is precisely the
assignment / 2-matching relaxation used as the root bound of TSP
branch-and-bound.

This variant is an LP relaxation of a MIP: the natural model uses binary arc
variables, but here the arcs are LP-relaxed to the continuous interval
`[0,1]` (cf. the CLAUDE.md "Model classes" section). The optimal value lower-
bounds the true TSP optimum, and a fractional `x` is not a directly
implementable tour.

# Feasibility
- `feasible` / `unknown`: the complete network always admits an assignment
  (`x[i,j] = 1/(n-1)` on every off-diagonal arc), so both are feasible by
  construction.
- `infeasible`: one city is completely cut off — every arc into and out of it
  is forbidden — so its degree rows sum over the empty set and the LP is
  infeasible. (A bridge cut between two nonempty regions would NOT make this
  relaxation infeasible: a fractional matching exists within each side.)

# Fields
- `n::Int`: number of cities
- `points::Vector{Tuple{Int,Int}}`: city coordinates
- `costs::Matrix{Int}`: rounded-Euclidean arc costs (`costs[i,i] = 0`,
  symmetric, off-diagonal entries ≥ 1)
- `forbidden::Set{Tuple{Int,Int}}`: directed arcs that are not available
  (empty unless the status is `infeasible`)
"""
struct TSPAssignmentRelaxationProblem <: ProblemGenerator
    n::Int
    points::Vector{Tuple{Int,Int}}
    costs::Matrix{Int}
    forbidden::Set{Tuple{Int,Int}}
end

"""
    TSPAssignmentRelaxationProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct an instance of the assignment relaxation of the TSP.

# Variable-count formula
The model has one continuous arc variable per directed pair of cities:

    total = n * (n - 1)

so `n ≈ round(sqrt(target_variables))` (clamped to `n ≥ 3`). For
`target = 100` this gives `n = 10` (90 vars); for `target = 500`, `n = 22`
(462 vars). Infeasible instances forbid all arcs touching one city, so their
actual variable count is correspondingly smaller.

# Arguments
- `target_variables`: Target number of decision variables
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function TSPAssignmentRelaxationProblem(target_variables::Int,
                                        feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    n = max(3, round(Int, sqrt(target_variables)))
    points = _tsp_clustered_points(n)
    costs = _tsp_euclidean_costs(points)

    forbidden = Set{Tuple{Int,Int}}()
    if feasibility_status == infeasible
        # One completely cut-off city: forbid every arc into and out of city
        # n, so its degree rows sum over the empty set and the LP is
        # infeasible.
        for i in 1:n
            i == n && continue
            push!(forbidden, (n, i))
            push!(forbidden, (i, n))
        end
    end
    # feasible/unknown: leave the complete network; an assignment always exists.

    return TSPAssignmentRelaxationProblem(n, points, costs, forbidden)
end

"""
    build_model(prob::TSPAssignmentRelaxationProblem)

Build a JuMP model for the assignment relaxation of the TSP. Deterministic —
uses only data from the struct fields.

Decision variables (over the available directed arcs `(i,j)`, `i ≠ j`):

- `x[i,j] ∈ [0,1]` (continuous): amount of arc mass on `(i,j)`

# Returns
- `model`: The JuMP model
"""
function build_model(prob::TSPAssignmentRelaxationProblem)
    model = Model()

    n = prob.n
    cities = 1:n
    available(i, j) = i != j && !((i, j) in prob.forbidden)

    # --- Variables: continuous arc weights on available arcs ---
    @variable(model, 0 <= x[i in cities, j in cities; available(i, j)] <= 1)

    # --- Objective: minimize total travel cost ---
    @objective(model, Min,
        sum(prob.costs[i, j] * x[i, j] for i in cities, j in cities if available(i, j)))

    # --- Degree constraints: one unit of out-mass and in-mass per city ---
    for i in cities
        @constraint(model, sum(x[i, j] for j in cities if available(i, j)) == 1)   # out
        @constraint(model, sum(x[j, i] for j in cities if available(j, i)) == 1)   # in
    end

    # No subtour elimination: optimal solutions decompose into subtours, which
    # is exactly what makes this the assignment relaxation of the TSP.

    return model
end

# Register the variant (lazily creates the :tsp category).
register_variant(
    :tsp,
    :assignment_relaxation,
    TSPAssignmentRelaxationProblem,
    "Assignment (2-matching) relaxation of the TSP: degree constraints only, continuous arc weights; a pure LP whose optimum decomposes into subtours",
)
