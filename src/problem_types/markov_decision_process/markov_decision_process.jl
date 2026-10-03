# markov_decision_process category
#
# Entry point for the `markov_decision_process` category: occupation-measure
# (dual) LPs of finite Markov decision processes built from operational models
# (inventory, queueing, maintenance) plus constrained MDPs with coupling
# budget rows. `common.jl` holds the shared sparse kernel, the LP builder,
# policy iteration, and the witness / certificate machinery.

register_category(
    :markov_decision_process,
    "Occupation-measure LPs of finite Markov decision processes (discounted or average cost) from operational models — seasonal pricing and replenishment, tandem-queue admission and service-rate control, condition-based maintenance with spares — and constrained MDPs whose budget rows couple every state-action frequency",
)

include("common.jl")
include("inventory_control.jl")
include("queueing_control.jl")
include("machine_maintenance.jl")
include("constrained.jl")
