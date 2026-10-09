# mine_planning category
#
# Open-pit mine production scheduling over a 3D block model, the source of the
# MineLib benchmark LPs (Espinoza, Goycoolea, Moreno & Newman, "MineLib: a
# library of open pit mining problems", Annals of OR 2013) whose relaxations
# motivated the Bienstock-Zuckerberg algorithm:
#
#   cpit       constrained pit limit: fixed destinations, mining and milling
#              capacities, minimum mill-feed contract
#   pcpsp      precedence-constrained production scheduling with destinations
#              (mill / heap leach / dump), head-grade and arsenic blending, and
#              metal-production capacities
#   stockpile  scheduling with grade-binned stockpiles carrying inventory across
#              periods (linear stockpile models of Moreno et al., EJOR 2017)

register_category(
    :mine_planning,
    "Open-pit mine production scheduling over a precedence-closed 3D block model " *
    "(MineLib-style CPIT/PCPSP): cumulative extraction variables, slope precedence, " *
    "capacity and blending rows, discounted NPV",
)

include("block_model.jl")
include("cpit.jl")
include("pcpsp.jl")
include("stockpile.jl")
