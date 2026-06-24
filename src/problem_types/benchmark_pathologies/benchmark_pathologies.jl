# benchmark_pathologies category
#
# Solver-stress LPs with deliberately controlled numerical or degeneracy
# structure. These are not domain applications; they are realistic benchmark
# complements for testing algorithms on edge cases.

include("scaling_stress.jl")
include("degenerate_network.jl")
