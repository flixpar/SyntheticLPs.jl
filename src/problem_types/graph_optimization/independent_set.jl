using JuMP
using Random

"""
    IndependentSetWitness

A planted independent set: `vertices` are pairwise non-adjacent, so the 0/1
indicator of `vertices` satisfies every clique row, and
`length(vertices) >= minimum_selected` meets the cardinality floor.
"""
struct IndependentSetWitness
    vertices::Vector{Int}
end

"""
    CliquePartitionCertificate

LP-valid upper bound on `sum(x)` for a clique-packing model. `parts` partition
the vertices; every part with `part_rows[p] > 0` is a subset of the model's
clique row `cliques[part_rows[p]]` (singletons have `part_rows[p] == 0` and use
the bound `x_v <= 1`). Summing those rows gives `sum(x) <= length(parts) ==
bound` for every `x` in `[0,1]^n`, so a floor `minimum_selected > bound` is
infeasible even in the LP relaxation. Presolve cannot see it: the argument
aggregates thousands of rows.
"""
struct CliquePartitionCertificate
    parts::Vector{Vector{Int}}
    part_rows::Vector{Int}
    bound::Int
end

"""
    IndependentSetProblem <: ProblemGenerator

Weighted maximum independent set on a unit-disk interference graph — choosing a
set of wireless transmitters (or dispersed facility sites) that may operate
simultaneously, where any two sites within the interference radius conflict.

# Data

Sites are scattered over a square with Gaussian hotspots (venues, dense office
floors) over a uniform background, so the conflict graph has heterogeneous
degrees and large natural cliques. Weights are lognormal traffic demands, larger
in hotspots.

# Formulation (clique formulation)

    max  sum_v w_v x_v
    s.t. sum_{v in K} x_v <= 1        for every clique K of a greedy edge clique cover
         sum_v x_v >= minimum_selected  (service floor; omitted when 0)
         x binary

The clique cover implies every edge row and gives the strong *clique*
formulation, whose LP relaxation is far from the trivially half-integral
edge relaxation of sparse random graphs. Variables: exactly `target_variables`
(one per site).

# Feasibility

  - `feasible`: a greedy independent set is the `feasible_witness`; the floor is
    85–100% of its size.
  - `infeasible`: the floor exceeds a clique-partition bound (see
    [`CliquePartitionCertificate`](@ref)) — every LP-feasible point selects at
    most `bound` sites. The argument aggregates one row per part, so presolve
    does not detect it.
  - `unknown`: the floor is drawn between the greedy size and the
    clique-partition bound; the LP optimum lies somewhere in between.
"""
struct IndependentSetProblem <: ProblemGenerator
    n_vertices::Int
    xs::Vector{Float64}
    ys::Vector{Float64}
    edges::Vector{Tuple{Int, Int}}
    cliques::Vector{Vector{Int}}
    weights::Vector{Float64}
    minimum_selected::Int
    feasible_witness::Union{Nothing, IndependentSetWitness}
    infeasibility_certificate::Union{Nothing, CliquePartitionCertificate}
end

function IndependentSetProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 2 || throw(ArgumentError("independent set needs at least 2 variables"))
    rng = MersenneTwister(seed)
    n = target_variables
    avg_degree = 5.0 + 5.0 * rand(rng)
    xs, ys, _ = _graph_geometric_points(rng, n, avg_degree)
    edges = _graph_pairs_within(xs, ys, 1.0)
    adj = _graph_adjacency(n, edges)
    cliques = _graph_clique_cover(adj, _graph_cell_groups(xs, ys, 1.0))

    # Demand grows with local crowding (hotspot sites carry more traffic).
    weights = round.(
        _graph_lognormal_weights(rng, n; median=40.0, sigma=0.5) .*
        [1 + 0.05 * length(adj[v]) for v in 1:n];
        digits=2,
    )

    greedy = _graph_greedy_independent_set(adj, weights)
    parts, part_rows = _graph_clique_partition(n, cliques)
    bound = length(parts)

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        minimum_selected = max(1, floor(Int, (0.85 + 0.15 * rand(rng)) * length(greedy)))
        witness = IndependentSetWitness(greedy)
    elseif feasibility_status == infeasible
        minimum_selected = bound + max(1, ceil(Int, 0.02 * bound))
        certificate = CliquePartitionCertificate(parts, part_rows, bound)
    else
        minimum_selected = _graph_floor_between(rng, length(greedy), bound)
    end

    return IndependentSetProblem(
        n, xs, ys, edges, cliques, weights, minimum_selected, witness, certificate
    )
end

function build_model(prob::IndependentSetProblem)
    model = Model()
    @variable(model, x[1:prob.n_vertices], Bin)
    @objective(model, Max, sum(prob.weights[v] * x[v] for v in 1:prob.n_vertices))
    for clique in prob.cliques
        @constraint(model, sum(x[v] for v in clique) <= 1)
    end
    if prob.minimum_selected > 0
        @constraint(model, sum(x) >= prob.minimum_selected)
    end
    return model
end

register_variant(
    :graph_optimization,
    :independent_set,
    IndependentSetProblem,
    "Weighted maximum independent set on a hotspot unit-disk interference graph, in the clique formulation";
    default=true,
)
