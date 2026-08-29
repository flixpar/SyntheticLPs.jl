# operating_room_scheduling category
#
# Entry point for the `operating_room_scheduling` problem category. Groups variants
# covering tactical and operational Operating Room (OR) planning and scheduling:
# - :standard: Advance elective case scheduling across multi-day horizons with
#   specialized OR equipment, surgeon availability, regular/overtime capacity,
#   PACU recovery beds, and patient urgency weighting.
# - :daily_sequencing: Daily multi-stage operational scheduling with pre-op, intra-op
#   OR with cleaning times, post-op PACU recovery, and surgeon no-overlap constraints.
# - :master_surgical_schedule: Master Surgical Schedule (MSS) block planning with
#   downstream bed leveling, quota constraints, and room-specialty affinities.
# - :robust_elective: Robust advance scheduling with budget of uncertainty protecting
#   against surgery duration overruns.

register_category(
    :operating_room_scheduling,
    "Operating room planning and scheduling optimizing surgical case assignment, sequencing, multi-stage recovery, and capacity management",
)

include("standard.jl")
include("daily_sequencing.jl")
include("master_surgical_schedule.jl")
include("robust_elective.jl")
