# Hub location and hub-and-spoke network design.
#
# Flows between origin-destination pairs are consolidated through intermediate
# hubs, with a discount factor alpha on inter-hub legs capturing economies of
# scale. The family spans the classical problem classes of the literature
# (Alumur & Kara, "Network hub location problems: the state of the art",
# Networks & Spatial Economics 2009; Campbell & O'Kelly, "Twenty-five years of
# hub location research", Transportation Science 2012):
#
#   p_hub_median       uncapacitated single-allocation p-hub median (airline)
#   compact_single_allocation compact origin-indexed p-hub median
#   r_allocation       uncapacitated r-allocation p-hub median (airline, backup hubs)
#   multiple_allocation fixed-charge multiple-allocation hub location (parcel)
#   capacitated        capacitated single-allocation hub location (postal)
#   hub_covering       service-threshold hub set covering (express / airline)
#   hub_network        single allocation over an incomplete hub network with
#                      modular backbone links (telecom backbone)
#   budgeted_backbone  exact-p hub and capacitated physical-link investment

register_category(
    :hub_location,
    "Hub location and hub-and-spoke network design: route origin-destination " *
    "traffic through consolidated hubs with discounted inter-hub economies of scale",
)

include("hub_data.jl")
include("p_hub_median.jl")
include("compact_single_allocation.jl")
include("r_allocation.jl")
include("multiple_allocation.jl")
include("capacitated.jl")
include("hub_covering.jl")
include("hub_network.jl")
include("budgeted_backbone.jl")
