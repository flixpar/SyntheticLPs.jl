# stochastic_program category
#
# Entry point for the `stochastic_program` problem category: extensive-form
# (deterministic-equivalent) stochastic LPs. `standard` is a two-stage
# capacity/recourse model with the dual block-angular structure;
# `multistage_alm` is a multistage asset-liability model on a scenario tree.

register_category(
    :stochastic_program,
    "Extensive-form stochastic linear programs: two-stage recourse (dual block-angular) and multistage scenario-tree models",
)

include("standard.jl")
include("multistage_alm.jl")
