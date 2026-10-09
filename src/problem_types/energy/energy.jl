# energy category
#
# Power-systems operations LPs in two families that share `common.jl`:
#
#   * multi-area, multi-period economic dispatch (`standard`, `reserves`,
#     `storage`, `hydrothermal`): one dispatch core — zones joined by lossy
#     tie-lines, a technology-grounded fleet with must-run floors and ramp limits,
#     curtailable renewables — with each variant adding one real coupling
#     (emissions budget, reserve co-optimization, storage state of charge, hydro
#     cascades);
#   * bus-level DC power flow on a meshed transmission grid (`dc_opf`,
#     `security_constrained_dc_opf`).

register_category(
    :energy,
    "Power-systems operations: multi-area economic dispatch with ramping, emissions, reserves, storage and hydro cascades, and DC optimal power flow with N-1 security",
)

include("common.jl")
include("standard.jl")
include("reserves.jl")
include("storage.jl")
include("hydrothermal.jl")
include("dc_opf.jl")
include("security_constrained_dc_opf.jl")
