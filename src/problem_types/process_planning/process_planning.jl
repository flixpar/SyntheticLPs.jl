# process_planning category

using Random
using Distributions

register_category(
    :process_planning,
    "Multi-period production planning for petroleum refineries and chemical process plants: crude assays, conversion units, blending, quality specifications, campaigns, contracts, and inventory",
)

"""
Position of `seed` in `[0, 1)` by the golden-ratio sequence, so any block of
consecutive seeds spreads evenly along a market-scenario band and produces a
genuine feasibility mix instead of a binomial accident concentrated at one
end. Shared by both variants' `unknown`-status market scenarios.
"""
_pp_seed_position(seed::Int) = mod(seed * 0.6180339887498949, 1.0)

"""
Seasonal demand profile for one product: cosine deviation of amplitude `amp`
peaked at `phase`, with mild per-period noise, floored so strong seasonality
(paving asphalt runs 0.1x in winter) stays non-negative, and renormalised to
mean one over the horizon so the reference plan sells what it produces.
Shared by both variants' demand generation.
"""
function _pp_seasonal_deviation(rng::AbstractRNG, amp::Real, phase::Real,
                                n_periods::Int)
    row = [1 + amp * cos(2π * (t - phase) / n_periods) for t in 1:n_periods]
    row .*= rand(rng, Uniform(0.97, 1.03), n_periods)
    row .= max.(row, 0.12)
    row ./= sum(row) / n_periods
    return row
end

include("refinery.jl")
include("campaign.jl")
