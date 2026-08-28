# set-system category

register_category(
    :set_system,
    "Binary covering, packing, partitioning, and winner-determination models on sparse set systems",
)

include("common.jl")
include("set_cover.jl")
include("set_packing.jl")
include("set_partitioning.jl")
include("combinatorial_auction.jl")
