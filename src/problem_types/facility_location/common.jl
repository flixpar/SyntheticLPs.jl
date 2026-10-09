# Shared geography helpers for the facility_location family. Every randomised
# helper takes the caller's `rng` first; call them from constructors only.

using Random

_fl_dist(a::Tuple{Float64, Float64}, b::Tuple{Float64, Float64}) = hypot(a[1] - b[1], a[2] - b[2])

"""
    _fl_clustered_points(rng, n, centers, spread, span; rural_fraction=0.2)

Scatter `n` points over `[0, span]^2`: a `1 - rural_fraction` share is drawn
around randomly chosen `centers` with Gaussian `spread`, the rest uniformly
(rural scatter between towns).
"""
function _fl_clustered_points(
    rng::AbstractRNG,
    n::Int,
    centers::Vector{Tuple{Float64, Float64}},
    spread::Float64,
    span::Float64;
    rural_fraction::Float64=0.2,
)
    points = Vector{Tuple{Float64, Float64}}(undef, n)
    for i in 1:n
        if rand(rng) < rural_fraction
            points[i] = (span * rand(rng), span * rand(rng))
        else
            c = centers[rand(rng, 1:length(centers))]
            points[i] = (
                clamp(c[1] + spread * randn(rng), 0.0, span),
                clamp(c[2] + spread * randn(rng), 0.0, span),
            )
        end
    end
    return points
end

"""
    _fl_nearest_sites(sites, queries, k) -> Vector{Vector{Int}}

For every query point, the indices of its `min(k, length(sites))` nearest
sites, sorted by distance (ties by index). Uses a uniform bucket grid, so the
cost is near-linear in `length(sites) + length(queries) * k` instead of the
dense `length(sites) * length(queries)` scan. Deterministic (no RNG).
"""
function _fl_nearest_sites(
    sites::Vector{Tuple{Float64, Float64}}, queries::Vector{Tuple{Float64, Float64}}, k::Int
)
    n = length(sites)
    k = min(k, n)
    result = [Int[] for _ in queries]
    (n == 0 || k == 0) && return result

    xmin = min(minimum(first, sites), minimum(first, queries; init=Inf))
    xmax = max(maximum(first, sites), maximum(first, queries; init=(-Inf)))
    ymin = min(minimum(last, sites), minimum(last, queries; init=Inf))
    ymax = max(maximum(last, sites), maximum(last, queries; init=(-Inf)))
    g = max(1, ceil(Int, sqrt(n / 2)))
    width = max(max(xmax - xmin, ymax - ymin) / g, 1e-9)
    gx = floor(Int, (xmax - xmin) / width) + 1
    gy = floor(Int, (ymax - ymin) / width) + 1
    cell_of(p) = (
        clamp(floor(Int, (p[1] - xmin) / width) + 1, 1, gx),
        clamp(floor(Int, (p[2] - ymin) / width) + 1, 1, gy),
    )
    buckets = [Int[] for _ in 1:(gx * gy)]
    for (s, p) in enumerate(sites)
        cx, cy = cell_of(p)
        push!(buckets[(cy - 1) * gx + cx], s)
    end

    candidates = Tuple{Float64, Int}[]
    for (q, p) in enumerate(queries)
        empty!(candidates)
        cx, cy = cell_of(p)
        ring = 0
        while true
            for iy in (cy - ring):(cy + ring), ix in (cx - ring):(cx + ring)
                max(abs(ix - cx), abs(iy - cy)) == ring || continue
                (1 <= ix <= gx && 1 <= iy <= gy) || continue
                for s in buckets[(iy - 1) * gx + ix]
                    push!(candidates, (_fl_dist(p, sites[s]), s))
                end
            end
            # Any site in ring `ring + 1` or beyond is at least `ring * width`
            # away from the query, so the k best found so far are final.
            if length(candidates) >= k
                partialsort!(candidates, k)
                candidates[k][1] <= ring * width && break
            end
            ring > max(gx, gy) && break
            ring += 1
        end
        sort!(candidates)
        result[q] = [s for (_, s) in candidates[1:k]]
    end
    return result
end
