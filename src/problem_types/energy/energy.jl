# energy category
#
# Entry point for the `energy` problem category. A category groups one or
# more variant formulations; the category is registered lazily from its
# first variant's `register_variant` call (or call `register_category`
# explicitly to give the category its own description). Add a variant by
# creating a file in this folder and including it below.

# Category-level description (it now groups several formulations).
register_category(:energy, "Power-systems optimization: economic dispatch, ramping, reserves, storage, transmission, and DC optimal power flow")

include("standard.jl")
include("ramping.jl")
include("reserves.jl")
include("storage.jl")
include("transmission.jl")
include("dc_opf.jl")
