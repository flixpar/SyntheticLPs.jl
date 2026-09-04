# Multi-period refinery and chemical-process planning.
#
# `common.jl` builds the shared refinery flowsheet — crude assays, distillation
# cuts, conversion units with per-mode yields, and specification-constrained
# finished grades — together with the planted operation and the two structural
# infeasibility certificates. `refinery` is the pure LP over a fixed operating
# mode; `mode_switching` leaves the mode choice and its changeovers to the
# solver; `capacity_expansion` is the long-range process-network investment
# model, whose network is chemical rather than petroleum.

register_category(
    :process_planning,
    "Multi-period refinery and chemical-process planning: crude selection and " *
    "distillation, fixed-yield conversion, intermediate tankage, blending to " *
    "product specifications, operating-mode selection, and long-range capacity " *
    "expansion",
)

include("common.jl")
include("refinery.jl")
include("mode_switching.jl")
include("capacity_expansion.jl")
