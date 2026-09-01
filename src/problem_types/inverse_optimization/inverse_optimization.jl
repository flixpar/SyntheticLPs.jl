# Inverse optimization: infer forward-model parameters from exact decisions,
# imperfect decision panels, target values, routes, and market outcomes.

register_category(
    :inverse_optimization,
    "Inverse optimization from exact, noisy, value-targeted, network-structured, and market-clearing observations",
)

# Shared machinery for box-identified generic inverse LPs, followed by the
# simplex-normalized packing families.
include("inverse_data.jl")
include("common.jl")

include("standard.jl")
include("classical.jl")
include("linf.jl")
include("noisy_observations.jl")
include("optimal_value.jl")
include("shortest_path.jl")
include("shortest_path_layered.jl")
include("market_clearing.jl")
