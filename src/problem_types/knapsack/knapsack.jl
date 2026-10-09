# knapsack category
#
# Entry point for the `knapsack` problem category. Every variant has many rows
# that grow with the instance: the single-row `standard` knapsack (solved by
# Dantzig's greedy, and by presolve alone) was removed, and `bounded` was
# rebuilt as a bounded multiple knapsack.

register_category(
    :knapsack,
    "Knapsack-family resource-allocation LPs: multiple-choice, sparse multi-dimensional, bounded multiple, and HEM-MIK-style knapsack sets",
)

include("multiple_choice.jl")
include("multidimensional.jl")
include("bounded.jl")
include("mixed_integer_set.jl")
