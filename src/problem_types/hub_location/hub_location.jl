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
#   r_allocation       uncapacitated r-allocation p-hub median (airline, backup hubs)
#   multiple_allocation fixed-charge multiple-allocation hub location (parcel)
#   capacitated        capacitated single-allocation hub location (postal)
#   hub_network        single allocation over an incomplete hub network with
#                      modular backbone links (telecom backbone)

register_category(:hub_location,
    "Hub location and hub-and-spoke network design: route origin-destination " *
    "traffic through consolidated hubs with discounted inter-hub economies of scale")

include("hub_data.jl")
include("p_hub_median.jl")
include("r_allocation.jl")
include("multiple_allocation.jl")
include("capacitated.jl")
include("hub_network.jl")
