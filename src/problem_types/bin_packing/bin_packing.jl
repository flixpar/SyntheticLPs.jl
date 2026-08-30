# bin_packing category
#
# Entry point for the `bin_packing` problem category.

register_category(
    :bin_packing,
    "One-dimensional item packing with identical-bin and typed-fleet formulations",
)
include("standard.jl")
include("heterogeneous.jl")
