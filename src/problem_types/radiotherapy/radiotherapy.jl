# IMRT fluence-map optimization.
#
# The shared case generator builds spatially correlated pencil-beam dose data;
# the variants apply complementary convex and mixed-integer formulations used
# in treatment planning: voxelwise and minimax dose penalties, mean-tail-dose
# (CVaR) goals, setup-robust planning, and beam-angle selection.

register_category(
    :radiotherapy,
    "IMRT fluence-map LP/MILP planning with spatial dose, DVH tails, setup scenarios, beam selection, and deliverability regularization",
)

include("common.jl")
include("weighted_deviation.jl")
include("mean_tail_dose.jl")
include("minmax_deviation.jl")
include("robust_fluence.jl")
include("beam_angle_selection.jl")
