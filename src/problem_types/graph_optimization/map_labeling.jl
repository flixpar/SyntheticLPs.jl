using JuMP
using Random

const MAP_LABEL_POSITIONS = (:NE, :NW, :SE, :SW, :N)
# Cartographic preference order (Imhof): upper right first.
const MAP_LABEL_PREFERENCE = (1.0, 0.9, 0.8, 0.7, 0.85)

"""
    MapLabelingProblem <: ProblemGenerator

Maximum-weight point-feature map labeling. Each feature (a town on a map) has
four candidate label boxes around its symbol (NE, NW, SE, SW; a fifth, centred
above, for a few features so the candidate total matches the target exactly).
Box sizes follow the name length and the feature's importance class; two
candidates conflict when their boxes overlap. Candidates covering another
feature's symbol are allowed but heavily discounted.

# Formulation (clique formulation)

    max  sum_j value_j place_j
    s.t. sum_{j in K} place_j <= 1   for every clique K of a greedy clique cover of
                                     the conflict graph (each feature's candidate
                                     set is a seed clique, grown to maximality)
         sum_j place_j >= minimum_placed   (coverage floor; omitted when 0)
         place binary

Pairwise-intersecting axis-parallel boxes share a common point (Helly), so the
cliques are the "stacks" of overlapping labels in dense map regions — the
classic strong formulation for label placement.

# Feasibility

  - `feasible`: greedy labeling by importance (`feasible_witness`, the placed
    candidates); the floor is 85–100% of its size.
  - `infeasible`: the floor exceeds a clique-partition bound on the number of
    placed labels ([`CliquePartitionCertificate`](@ref)).
  - `unknown`: the floor lies between the greedy count and that bound.
"""
struct MapLabelingProblem <: ProblemGenerator
    n_features::Int
    feature_candidates::Vector{Vector{Int}}
    boxes::Vector{NTuple{4, Float64}}        # (xmin, ymin, xmax, ymax) per candidate
    conflicts::Vector{Tuple{Int, Int}}
    cliques::Vector{Vector{Int}}
    label_values::Vector{Float64}
    minimum_placed::Int
    feasible_witness::Union{Nothing, IndependentSetWitness}
    infeasibility_certificate::Union{Nothing, CliquePartitionCertificate}
end

_map_overlap(a::NTuple{4, Float64}, b::NTuple{4, Float64}) =
    a[1] < b[3] && b[1] < a[3] && a[2] < b[4] && b[2] < a[4]

_map_contains(a::NTuple{4, Float64}, x::Float64, y::Float64) = a[1] < x < a[3] && a[2] < y < a[4]

function MapLabelingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 8 || throw(ArgumentError("map labeling needs at least 8 variables"))
    rng = MersenneTwister(seed)
    n_candidates = target_variables
    n_features = n_candidates ÷ 4
    n_extra = n_candidates - 4 * n_features          # features with a fifth (N) box

    # Feature importance (population, lognormal) sets the font class.
    importance = _graph_lognormal_weights(rng, n_features; median=20.0, sigma=1.0)
    major_cut = sort(importance; rev=true)[max(1, ceil(Int, 0.1 * n_features))]
    scale = [importance[f] >= major_cut ? 1.4 : 1.0 for f in 1:n_features]
    widths = [scale[f] * (0.35 + 0.07 * rand(rng, 3:14)) for f in 1:n_features]
    heights = [0.35 * scale[f] for f in 1:n_features]

    # Map extent: candidate boxes overlap ~`crowding` others on average
    # (expected overlaps ≈ 16 ρ w̄ h̄ for 4 boxes per feature at density ρ).
    crowding = 3.0 + 4.0 * rand(rng)
    mean_area = sum(widths .* heights) / n_features
    density = crowding / (16 * mean_area)
    # `_graph_geometric_points` sizes its square by a unit-disk degree; convert.
    px, py, side = _graph_geometric_points(rng, n_features, pi * density; hotspot_share=0.5)

    # Candidate boxes, feature by feature; the first `n_extra` features (random
    # importance, since features are unordered) get the centred-above box.
    gap = 0.05
    boxes = NTuple{4, Float64}[]
    owner = Int[]
    position = Int[]
    feature_candidates = [Int[] for _ in 1:n_features]
    for f in 1:n_features
        x, y, w, h = px[f], py[f], widths[f], heights[f]
        options = (
            (x + gap, y + gap, x + gap + w, y + gap + h),            # NE
            (x - gap - w, y + gap, x - gap, y + gap + h),            # NW
            (x + gap, y - gap - h, x + gap + w, y - gap),            # SE
            (x - gap - w, y - gap - h, x - gap, y - gap),            # SW
            (x - w / 2, y + 2gap, x + w / 2, y + 2gap + h),          # N
        )
        for p in 1:(f <= n_extra ? 5 : 4)
            push!(boxes, options[p])
            push!(owner, f)
            push!(position, p)
            push!(feature_candidates[f], length(boxes))
        end
    end

    # Box-overlap conflicts via grid bucketing on box centres (cell = largest
    # box extent, so overlapping boxes lie in neighbouring cells).
    cell = 2 * maximum(max(b[3] - b[1], b[4] - b[2]) for b in boxes)
    cx = [(b[1] + b[3]) / 2 for b in boxes]
    cy = [(b[2] + b[4]) / 2 for b in boxes]
    buckets = _graph_buckets(cx, cy, cell)
    conflicts = Tuple{Int, Int}[]
    for i in eachindex(boxes)
        gx, gy = floor(Int, cx[i] / cell), floor(Int, cy[i] / cell)
        for dx in -1:1, dy in -1:1
            bucket = get(buckets, (gx + dx, gy + dy), nothing)
            bucket === nothing && continue
            for j in bucket
                j > i || continue
                owner[i] == owner[j] && continue
                _map_overlap(boxes[i], boxes[j]) && push!(conflicts, (i, j))
            end
        end
    end
    sort!(conflicts)

    # Values: importance x position preference, discounted when the box hides
    # another feature's symbol.
    point_buckets = _graph_buckets(px, py, cell)
    label_values = zeros(n_candidates)
    for j in eachindex(boxes)
        hides = false
        gx, gy = floor(Int, cx[j] / cell), floor(Int, cy[j] / cell)
        for dx in -1:1, dy in -1:1
            bucket = get(point_buckets, (gx + dx, gy + dy), nothing)
            bucket === nothing && continue
            for f in bucket
                f != owner[j] && _map_contains(boxes[j], px[f], py[f]) && (hides = true)
            end
        end
        value = importance[owner[j]] * MAP_LABEL_PREFERENCE[position[j]] * (hides ? 0.25 : 1.0)
        label_values[j] = round(value * (0.95 + 0.1 * rand(rng)); digits=2)
    end

    # Clique cover of the conflict graph plus same-feature exclusivity.
    all_pairs = copy(conflicts)
    for candidates in feature_candidates
        for a in 1:(length(candidates) - 1), b in (a + 1):length(candidates)
            push!(all_pairs, (candidates[a], candidates[b]))
        end
    end
    adj = _graph_adjacency(n_candidates, all_pairs)
    cliques = _graph_clique_cover(adj, feature_candidates)

    # Greedy labeling by importance: best still-free candidate per feature.
    placed = Int[]
    blocked = falses(n_candidates)
    for f in sortperm(importance; rev=true)
        free = [j for j in feature_candidates[f] if !blocked[j]]
        isempty(free) && continue
        j = free[argmax(label_values[free])]
        push!(placed, j)
        blocked[adj[j]] .= true
        blocked[j] = true
    end
    sort!(placed)

    parts, part_rows = _graph_clique_partition(n_candidates, cliques)
    if length(parts) > n_features
        # Feature sets are themselves a clique partition (each feature's set is
        # a seed, contained in the clique grown from it).
        rows_of = [Int[] for _ in 1:n_candidates]
        for (r, K) in enumerate(cliques), j in K
            push!(rows_of[j], r)
        end
        parts = [copy(c) for c in feature_candidates]
        part_rows = [
            rows_of[c[1]][findfirst(r -> issubset(c, cliques[r]), rows_of[c[1]])] for
            c in feature_candidates
        ]
    end
    bound = length(parts)

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        minimum_placed = max(1, floor(Int, (0.85 + 0.15 * rand(rng)) * length(placed)))
        witness = IndependentSetWitness(placed)
    elseif feasibility_status == infeasible
        minimum_placed = bound + max(1, ceil(Int, 0.02 * bound))
        certificate = CliquePartitionCertificate(parts, part_rows, bound)
    else
        minimum_placed = _graph_floor_between(rng, length(placed), bound)
    end

    return MapLabelingProblem(
        n_features,
        feature_candidates,
        boxes,
        conflicts,
        cliques,
        label_values,
        minimum_placed,
        witness,
        certificate,
    )
end

function build_model(prob::MapLabelingProblem)
    model = Model()
    n_candidates = length(prob.label_values)
    @variable(model, place[1:n_candidates], Bin)
    @objective(model, Max, sum(prob.label_values[j] * place[j] for j in 1:n_candidates))
    for clique in prob.cliques
        @constraint(model, sum(place[j] for j in clique) <= 1)
    end
    if prob.minimum_placed > 0
        @constraint(model, sum(place) >= prob.minimum_placed)
    end
    return model
end

register_variant(
    :graph_optimization,
    :map_labeling,
    MapLabelingProblem,
    "Point-feature map labeling with geometric box-overlap conflicts in the clique formulation";
    tags=[:combinatorial, :packing],
    min_target_variables=8,
)
