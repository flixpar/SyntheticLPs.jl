using Random

function _set_random_column(
    rng::AbstractRNG,
    n_elements::Int;
    max_size::Int=max(1, min(n_elements, 8)),
)
    effective_max = max(1, min(n_elements, max_size))
    # Mix short uniform columns with a heavier upper tail.
    size = if rand(rng) < 0.7
        rand(rng, 1:min(effective_max, 4))
    else
        rand(rng, 1:effective_max)
    end
    return sort!(randperm(rng, n_elements)[1:size])
end

# Start with a shuffled exact partition, then add random columns. The leading
# `n_planted` columns are a constructive exact-cover/packing witness and ensure
# that every row is nonempty.
function _set_columns_with_partition(
    rng::AbstractRNG,
    n_elements::Int,
    n_columns::Int;
    max_size::Int=max(2, min(n_elements, 6)),
)
    order = randperm(rng, n_elements)
    columns = Vector{Vector{Int}}()
    cursor = 1
    while cursor <= n_elements
        remaining = n_elements - cursor + 1
        block_size = min(remaining, rand(rng, 1:max_size))
        push!(columns, sort!(order[cursor:(cursor + block_size - 1)]))
        cursor += block_size
    end
    length(columns) <= n_columns ||
        throw(ArgumentError("too few columns for the planted partition"))
    n_planted = length(columns)
    while length(columns) < n_columns
        push!(columns, _set_random_column(rng, n_elements; max_size=max_size))
    end
    return columns, n_planted
end

function _set_positive_coefficients(
    rng::AbstractRNG,
    n::Int;
    low::Int=1,
    high::Int=100,
)
    return Float64[rand(rng, low:high) for _ in 1:n]
end

function _set_elements_to_columns(columns::Vector{Vector{Int}}, n_elements::Int)
    incidence = [Int[] for _ in 1:n_elements]
    for (j, elements) in enumerate(columns), i in elements
        push!(incidence[i], j)
    end
    return incidence
end
