# transportation category
#
# Entry point for the `transportation` problem category: shipping from sources
# to customers over sparse geographic lane networks. `common.jl` holds the
# shared geography, lane-set and exact max-flow plumbing; each remaining file
# is one variant.

register_category(
    :transportation,
    "Transportation and distribution LPs on sparse geographic lane networks: capacitated source-to-customer shipping, multimodal shipping under regional emission caps, fixed-charge lane selection, and two-echelon transshipment through capacitated distribution centres",
)

include("common.jl")
include("standard.jl")
include("transshipment.jl")
include("emission_constrained.jl")
include("fixed_charge.jl")
