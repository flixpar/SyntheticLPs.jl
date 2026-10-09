# diet_problem category
#
# Entry point for the `diet_problem` problem category. `common.jl` holds the
# shared food-composition table, dietary reference intakes and helper bounds;
# each variant file builds a structurally different diet LP on top of it.

register_category(
    :diet_problem,
    "Least-cost diet planning on a role-correlated food-composition table with " *
    "Dietary-Reference-Intake requirements",
)

include("common.jl")
include("standard.jl")
include("food_groups.jl")
include("food_aid.jl")
