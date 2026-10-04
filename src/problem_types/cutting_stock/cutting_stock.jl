# cutting_stock category
#
# One-dimensional cutting stock in its two classic LP forms: the
# Gilmore-Gomory pattern master LP (`standard`, multi-period `due_dates`,
# multi-machine `setup_cost`) over a shared near-linear sparse pattern
# enumerator (`common.jl`), and the pseudo-polynomial arc-flow formulation
# (`arc_flow`).

register_category(
    :cutting_stock,
    "One-dimensional cutting stock: Gilmore-Gomory pattern LPs (multi-stock, multi-period, multi-machine) and the arc-flow formulation",
)

include("common.jl")
include("standard.jl")
include("due_dates.jl")
include("setup_cost.jl")
include("arc_flow.jl")
