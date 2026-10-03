# economic_planning category
#
# Dynamic multi-sector economy and energy-economy planning LPs — the family of
# Netlib's PILOT models, Dantzig's dynamic Leontief staircase models, and
# today's TIMES/MESSAGE/MARKAL energy-system models: staircase multi-period
# structure, many equality balance rows, hybrid physical/monetary units, and
# wide coefficient ranges.

register_category(
    :economic_planning,
    "Dynamic economy-wide planning LPs: multi-sector input-output (dynamic Leontief) growth models and technology-rich energy-system (TIMES/MESSAGE-style) capacity expansion models",
)

include("dynamic_leontief.jl")
