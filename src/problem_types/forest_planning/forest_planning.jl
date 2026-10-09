# forest_planning category
#
# Forest harvest scheduling (timber supply) LPs in the tradition of USFS
# FORPLAN / Spectrum and industrial Woodstock models: Johnson & Scheurman's
# Model I (whole-horizon stratum prescriptions) and Model II (rotation-level
# area flows through regeneration nodes) over a shared landscape, yield and
# economics model (`common.jl`).

register_category(
    :forest_planning,
    "Forest harvest scheduling LPs (Model I prescriptions and Model II regeneration-node networks) with even-flow, watershed green-up, mill-supply and ending-inventory constraints",
)

include("common.jl")
include("model_i.jl")
include("model_ii.jl")
