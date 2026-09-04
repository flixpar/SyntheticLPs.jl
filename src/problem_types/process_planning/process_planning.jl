# process_planning category

register_category(
    :process_planning,
    "Multi-period production planning for petroleum refineries and chemical process plants: crude assays, conversion units, blending, quality specifications, campaigns, contracts, and inventory",
)

include("refinery.jl")
include("campaign.jl")
