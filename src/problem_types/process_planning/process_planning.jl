# Multi-period refinery and chemical-process planning.
#
# `common.jl` builds the shared refinery flowsheet — crude assays, distillation
# cuts, conversion units with per-mode yields, and specification-constrained
# finished grades — together with the planted operation and two structural
# infeasibility certificates. `refinery` is the pure LP over a fixed operating
# mode; `mode_switching` leaves modes, starts, and minimum runs to the solver;
# `hydrogen_network` couples hydroprocessing to H2, sulfur, and carbon;
# `campaign` is a state-task-network petrochemical scheduling MILP; and
# `capacity_expansion` is the long-range chemical-network investment MILP.

using Random
using Distributions

register_category(
    :process_planning,
    "Multi-period refinery and chemical-process planning: crude selection and " *
    "distillation, fixed-yield conversion, intermediate tankage, blending to " *
    "product specifications, operating-mode selection, hydrogen and emissions " *
    "management, process campaigns, and long-range capacity expansion",
)

"""Low-discrepancy seed position used by the campaign market scenarios."""
_pp_seed_position(seed::Int) = mod(seed * 0.6180339887498949, 1.0)

"""Positive, mean-one annual profile used by process campaign demand."""
function _pp_seasonal_deviation(rng::AbstractRNG, amp::Real, phase::Real,
                                n_periods::Int; period_days::Real=7.0)
    row = [1 + amp * cos(2pi * period_days * (t - 1) / 365.25 + phase)
           for t in 1:n_periods]
    row .*= rand(rng, Uniform(0.97, 1.03), n_periods)
    row .= max.(row, 0.12)
    row ./= sum(row) / n_periods
    return row
end

include("common.jl")
include("refinery.jl")
include("mode_switching.jl")
include("hydrogen_network.jl")
include("campaign.jl")
include("capacity_expansion.jl")
