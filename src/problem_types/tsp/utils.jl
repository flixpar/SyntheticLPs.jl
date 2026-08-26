using Random

# Shared data-generation helpers for the `tsp` category.
#
# Every variant file in this category shares the SyntheticLPs module namespace,
# so these helpers must be defined exactly once (duplicated definitions across
# variant files would silently overwrite each other). All helpers draw from the
# caller's RNG stream — they never call Random.seed! — so instance generation is
# deterministic for a fixed seed. Callers invoke them in a fixed order.

"""
    _tsp_clustered_points(n) -> Vector{Tuple{Int,Int}}

Sample `n` distinct city locations as integer coordinates in `[0, 1000]²` with
a realistic urban structure: roughly 70% of the cities are drawn around
`max(2, n ÷ 3)` randomly placed cluster centers (neighborhoods) with Gaussian
spread, and the rest are scattered uniformly (suburbs / outliers).

The draw order is fixed (cluster members first, then scattered points) so the
RNG stream is deterministic. Duplicate coordinates — possible in principle but
rare on a 1001² lattice for `n ≤ ~45` — are jittered to a nearby free point.
"""
function _tsp_clustered_points(n::Int)
    grid_max = 1000
    n_clusters = max(2, n ÷ 3)
    centers = [(rand(0:grid_max), rand(0:grid_max)) for _ in 1:n_clusters]
    spread = rand(80:150)

    points = Tuple{Int,Int}[]
    n_clustered = min(n, round(Int, 0.7 * n))
    for _ in 1:n_clustered
        cx, cy = rand(centers)
        x = clamp(cx + round(Int, randn() * spread), 0, grid_max)
        y = clamp(cy + round(Int, randn() * spread), 0, grid_max)
        _tsp_push_unique_point!(points, (x, y), grid_max)
    end
    while length(points) < n
        _tsp_push_unique_point!(points, (rand(0:grid_max), rand(0:grid_max)), grid_max)
    end
    return points
end

# Push `p` onto `points`, jittering deterministically if it is already taken.
function _tsp_push_unique_point!(points::Vector{Tuple{Int,Int}},
                                 p::Tuple{Int,Int}, grid_max::Int)
    p in points || return push!(points, p)
    # Extremely rare on a 1001² lattice; probe a fixed pattern of neighbors.
    for d in 1:grid_max
        for q in ((clamp(p[1] + d, 0, grid_max), p[2]),
                  (clamp(p[1] - d, 0, grid_max), p[2]),
                  (p[1], clamp(p[2] + d, 0, grid_max)),
                  (p[1], clamp(p[2] - d, 0, grid_max)))
            q in points || return push!(points, q)
        end
    end
    error("tsp: could not find a distinct lattice point near $p")
end

"""
    _tsp_euclidean_costs(points) -> Matrix{Int}

Symmetric rounded-Euclidean distance matrix for `points`. Distinct integer
lattice points are at least distance 1 apart, so off-diagonal entries are ≥ 1.
"""
function _tsp_euclidean_costs(points::Vector{Tuple{Int,Int}})
    n = length(points)
    costs = zeros(Int, n, n)
    for i in 1:n, j in 1:n
        i == j && continue
        dx = points[i][1] - points[j][1]
        dy = points[i][2] - points[j][2]
        costs[i, j] = round(Int, hypot(dx, dy))
    end
    return costs
end
