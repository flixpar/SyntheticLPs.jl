# tsp category
#
# Entry point for the `tsp` problem category. A category groups one or more
# variant formulations; the category is registered explicitly below because it
# carries a description of its own. Add a variant by creating a file in this
# folder and including it below.

register_category(:tsp, "Traveling salesman problem: Euclidean (MTZ), asymmetric one-way-street ATSP (Gavish-Graves), time-windowed TSP, and the assignment (2-matching) LP relaxation")

# Shared data-generation helpers. Every variant file in this category shares
# the SyntheticLPs module namespace, so these must be defined exactly once,
# before the variant files that use them.
include("utils.jl")

include("euclidean.jl")
include("asymmetric.jl")
include("time_windows.jl")
include("assignment_relaxation.jl")
