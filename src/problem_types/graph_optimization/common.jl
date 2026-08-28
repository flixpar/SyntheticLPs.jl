using Random

# Return `count` distinct undirected edges, avoiding `forbidden`. Edges are
# sorted both internally and as a collection so construction does not depend on
# Set iteration order.
function _graph_sample_edges(
    rng::AbstractRNG,
    n_vertices::Int,
    count::Int;
    forbidden::Set{Tuple{Int,Int}}=Set{Tuple{Int,Int}}(),
    planted_independent::Set{Int}=Set{Int}(),
)
    maximum = n_vertices * (n_vertices - 1) ÷ 2 - length(forbidden)
    count <= maximum || throw(ArgumentError("requested too many distinct graph edges"))

    edges = Set{Tuple{Int,Int}}()
    attempts = 0
    attempt_limit = max(100, 20 * count)
    while length(edges) < count && attempts < attempt_limit
        u = rand(rng, 1:n_vertices)
        v = rand(rng, 1:n_vertices)
        attempts += 1
        u == v && continue
        edge = minmax(u, v)
        edge in forbidden && continue
        (u in planted_independent && v in planted_independent) && continue
        push!(edges, edge)
    end

    # Dense requests can make rejection sampling slow. Finish deterministically
    # from a seed-shuffled list of the remaining admissible pairs.
    if length(edges) < count
        remaining = Tuple{Int,Int}[]
        for u in 1:(n_vertices - 1), v in (u + 1):n_vertices
            edge = (u, v)
            if !(edge in forbidden) && !(edge in edges) &&
               !(u in planted_independent && v in planted_independent)
                push!(remaining, edge)
            end
        end
        shuffle!(rng, remaining)
        append_count = count - length(edges)
        append_count <= length(remaining) ||
            throw(ArgumentError("not enough admissible graph edges"))
        union!(edges, @view remaining[1:append_count])
    end
    return sort!(collect(edges))
end

function _graph_weights(rng::AbstractRNG, n::Int; low::Int=1, high::Int=100)
    return Float64[rand(rng, low:high) for _ in 1:n]
end
