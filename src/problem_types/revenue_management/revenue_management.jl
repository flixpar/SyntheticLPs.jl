# revenue_management category
#
# Deterministic bid-price capacity allocation and stochastic overbooking share a
# coherent hub-and-spoke itinerary generator and typed product metadata.

register_category(
    :revenue_management,
    "Network revenue-management LPs for deterministic capacity allocation and stochastic overbooking with denied-service recourse",
)

include("standard.jl")
include("stochastic_overbooking.jl")
