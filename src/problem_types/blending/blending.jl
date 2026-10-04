# blending category
#
# Entry point for the `blending` problem category: secondary-aluminium alloy
# blending (scrap, primary metal and master alloys charged into composition
# windows). `common.jl` holds the alloy/material catalog and helpers; each
# variant file is a structurally different blending LP on top of it.

register_category(
    :blending,
    "Alloy blending: scrap, primary metal and master alloys charged so every melt meets " *
    "its composition window",
)

include("common.jl")
include("standard.jl")
include("multi_period.jl")
include("robust.jl")
