# Inverse-optimization category: objective inference from exact decisions,
# noisy decision panels, and observed shortest paths.

register_category(
    :inverse_optimization,
    "Inverse linear optimization from exact, noisy, or network-structured observed decisions",
)

include("common.jl")
include("classical.jl")
include("noisy_observations.jl")
include("shortest_path.jl")
