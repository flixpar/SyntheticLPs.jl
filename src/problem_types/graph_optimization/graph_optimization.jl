# graph optimization category

register_category(
    :graph_optimization,
    "Binary graph packing and covering formulations on reproducible sparse graphs",
)

include("common.jl")
include("independent_set.jl")
include("generalized_independent_set.jl")
include("vertex_cover.jl")
include("vertex_coloring.jl")
include("map_labeling.jl")
include("quasi_clique.jl")
