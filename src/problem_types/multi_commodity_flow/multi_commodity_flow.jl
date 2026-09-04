# multi_commodity_flow category
#
# Entry point for the `multi_commodity_flow` problem category. A category groups one or
# more variant formulations; the category is registered lazily from its
# first variant's `register_variant` call (or call `register_category`
# explicitly to give the category its own description). Add a variant by
# creating a file in this folder and including it below.

# Directed Hamilton cycle plus extra arcs sampled by rejection. Large sparse
# instances never materialize all n(n-1) ordered pairs.
function _discrete_mcf_topology(rng::AbstractRNG, n_nodes::Int, n_arcs::Int)
    arcs = Tuple{Int, Int}[]
    seen = Set{Tuple{Int, Int}}()
    order = randperm(rng, n_nodes)
    for idx in 1:n_nodes
        arc = (order[idx], order[idx == n_nodes ? 1 : idx + 1])
        push!(arcs, arc)
        push!(seen, arc)
    end
    attempts = 0
    max_attempts = 50 * n_arcs
    while length(arcs) < n_arcs && attempts < max_attempts
        attempts += 1
        i = rand(rng, 1:n_nodes)
        j = rand(rng, 1:n_nodes)
        i == j && continue
        arc = (i, j)
        arc in seen && continue
        push!(seen, arc)
        push!(arcs, arc)
    end
    return arcs
end

include("standard.jl")
include("binary_capacity.jl")
include("integer_flow.jl")
