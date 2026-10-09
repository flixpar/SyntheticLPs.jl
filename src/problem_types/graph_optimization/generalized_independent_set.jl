using JuMP
using Random

"""
    GeneralizedIndependentSetProblem <: ProblemGenerator

Generalized independent set (GISP, Colombi–Mansini–Savelsbergh 2017) as wind
farm layout: candidate turbine sites, hard conflicts between sites closer than
the minimum spacing, and soft conflicts (wake losses) between sites that lie
roughly along the prevailing wind direction within the wake length. Building
both endpoints of a soft edge activates its penalty variable.

# Formulation

    max  sum_v b_v x_v - sum_e p_e y_e
    s.t. sum_{v in K} x_v <= 1              for every clique K of a greedy clique
                                            cover of the hard-spacing graph
         x_u + x_v - y_e <= 1               for every wake pair e = (u, v)
         sum_v x_v >= minimum_turbines      (installed-capacity floor; omitted when 0)
         x, y binary

`b_v` is the annual energy yield from a smooth wind-resource field; the wake
penalty `p_e` decays with distance and grows with the downstream site's yield.
Variables: `n + |wake pairs|`, exactly `target_variables` (the strongest wake
pairs are kept).

# Feasibility

  - `feasible`: a greedy spacing-feasible layout is planted (`feasible_witness`);
    the floor is 85–100% of its size.
  - `infeasible`: the floor exceeds a clique-partition bound on the number of
    turbines ([`CliquePartitionCertificate`](@ref)).
  - `unknown`: the floor lies between the greedy layout size and that bound.
"""
struct GeneralizedIndependentSetProblem <: ProblemGenerator
    n_vertices::Int
    xs::Vector{Float64}
    ys::Vector{Float64}
    hard_edges::Vector{Tuple{Int, Int}}
    hard_cliques::Vector{Vector{Int}}
    soft_edges::Vector{Tuple{Int, Int}}
    vertex_benefits::Vector{Float64}
    edge_penalties::Vector{Float64}
    minimum_selected::Int
    feasible_witness::Union{Nothing, IndependentSetWitness}
    infeasibility_certificate::Union{Nothing, CliquePartitionCertificate}
end

function GeneralizedIndependentSetProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 6 ||
        throw(ArgumentError("generalized independent set needs at least 6 variables"))
    rng = MersenneTwister(seed)

    # Sites: mostly uniform over the concession with some clustering on ridges.
    n = max(4, round(Int, (0.5 + 0.15 * rand(rng)) * target_variables))
    n_soft = target_variables - n
    spacing_degree = 3.0 + 3.0 * rand(rng)
    xs, ys, side = _graph_geometric_points(rng, n, spacing_degree; hotspot_share=0.3)
    hard_edges = _graph_pairs_within(xs, ys, 1.0)
    adj = _graph_adjacency(n, hard_edges)
    hard_cliques = _graph_clique_cover(adj, _graph_cell_groups(xs, ys, 1.0))

    # Wind resource: smooth field of a few broad ridges (yield in GWh/yr).
    ridges = [(side * rand(rng), side * rand(rng), 0.3 + 0.5 * rand(rng)) for _ in 1:4]
    ridge_width = max(1.0, side / 3)
    vertex_benefits = [
        round(
            10.0 *
            (
                1.0 + sum(
                    h * exp(-((xs[v] - rx)^2 + (ys[v] - ry)^2) / (2ridge_width^2)) for
                    (rx, ry, h) in ridges
                )
            ) *
            (0.95 + 0.1 * rand(rng));
            digits=2,
        ) for v in 1:n
    ]

    # Wake pairs: beyond the hard spacing, within the wake length, and within
    # ±30° of the prevailing wind axis. Grow the wake length until there are
    # enough candidates, then keep the strongest `n_soft`.
    wind = 2pi * rand(rng)
    axis = (cos(wind), sin(wind))
    soft_edges = Tuple{Int, Int}[]
    edge_penalties = Float64[]
    wake = 3.0
    while true
        candidates = Tuple{Int, Int}[]
        strengths = Float64[]
        for (u, v) in _graph_pairs_within(xs, ys, wake; min_radius=1.0)
            dx, dy = xs[v] - xs[u], ys[v] - ys[u]
            d = hypot(dx, dy)
            alignment = abs(dx * axis[1] + dy * axis[2]) / d
            # Past the concession size (tiny instances) any pair may interact.
            (alignment >= cos(pi / 6) || wake > side) || continue
            push!(candidates, (u, v))
            push!(strengths, alignment * (1.0 / d)^2)
        end
        if length(candidates) >= n_soft || wake > 2 * side
            keep = sort!(sortperm(strengths; rev=true)[1:min(n_soft, length(candidates))])
            soft_edges = candidates[keep]
            edge_penalties = [
                round(
                    0.6 *
                    min(vertex_benefits[u], vertex_benefits[v]) *
                    strengths[e] *
                    (0.9 + 0.2 * rand(rng));
                    digits=2,
                ) for (e, (u, v)) in zip(keep, soft_edges)
            ]
            break
        end
        wake *= 1.5
    end

    greedy = _graph_greedy_independent_set(adj, vertex_benefits)
    parts, part_rows = _graph_clique_partition(n, hard_cliques)
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

    return GeneralizedIndependentSetProblem(
        n,
        xs,
        ys,
        hard_edges,
        hard_cliques,
        soft_edges,
        vertex_benefits,
        edge_penalties,
        minimum_selected,
        witness,
        certificate,
    )
end

function build_model(prob::GeneralizedIndependentSetProblem)
    model = Model()
    @variable(model, x[1:prob.n_vertices], Bin)
    @variable(model, y[1:length(prob.soft_edges)], Bin)
    @objective(
        model,
        Max,
        sum(prob.vertex_benefits[v] * x[v] for v in 1:prob.n_vertices) -
            sum(prob.edge_penalties[e] * y[e] for e in eachindex(prob.soft_edges)),
    )
    for clique in prob.hard_cliques
        @constraint(model, sum(x[v] for v in clique) <= 1)
    end
    for (e, (u, v)) in enumerate(prob.soft_edges)
        @constraint(model, x[u] + x[v] - y[e] <= 1)
    end
    if prob.minimum_selected > 0
        @constraint(model, sum(x) >= prob.minimum_selected)
    end
    return model
end

register_variant(
    :graph_optimization,
    :generalized_independent_set,
    GeneralizedIndependentSetProblem,
    "Generalized independent set as wind-farm layout: spacing cliques, wake-loss soft conflicts, and a capacity floor";
    tags=[:energy, :packing],
    min_target_variables=6,
)
