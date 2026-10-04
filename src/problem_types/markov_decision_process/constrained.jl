using JuMP
using Random

const CONSTRAINED_MDP_BASES = (:inventory_control, :queueing_control, :machine_maintenance)

"""
    ConstrainedMDP <: AbstractMDPProblem

Constrained MDP (Altman 1999): minimize the expected operating cost of one of
the operational models subject to two or three budget rows on its secondary
cost streams — the LP form in which a constrained MDP is actually solved, whose
optimal policies randomize in up to (number of budget rows) states.

# Structure

`base_model` is sampled from `:inventory_control`, `:queueing_control`, and
`:machine_maintenance`; `base` is that variant's instance built without its own
service row (see their docstrings for states, actions, and costs), and `mdp`,
`criterion`, `discount`, `rhs`, and `normalization` are shared with it. The
budget rows always include the base's service stream (`:shortage`,
`:rejection`, `:downtime` — stream 1) and at least one stream that conflicts
with it (inventory: `:on_hand`; queueing and maintenance: a random one of the
other two); with probability 1/2 all three streams are budgeted. Every budget
row has a nonzero coefficient on most state-action pairs, so these are dense
coupling rows on top of the sparse balance rows.

# Feasibility (see `_mdp_plant_budgets`)

  - `feasible`: the witness mixes the reference policy's occupation measure
    with that of the policy optimal for a random weighting of the budgeted
    streams (a near-Pareto point); each budget sits 5-25% above the witness.
  - `infeasible`: a Dirichlet weighting `w` of the budget rows (the most
    conflicting of three draws) and its certified minimum `L_w`; budgets satisfy
    `Σ w_j B_j = (1 - m) L_w` with `m ∈ [0.08, 0.25]` while — whenever the
    streams conflict, which the stream choice ensures — every single budget
    exceeds its own stream's optimum. No single row is infeasible; only the
    combination is, and refuting it requires the whole transition structure.
  - `unknown`: `B_j = L_j + u_j (R_j - L_j)` with `u_j ∈ [0.4, 1.4]`, between
    each stream's own optimum `L_j` and beyond the reference policy's value
    `R_j` (all `u_j >= 1` is always achievable by the reference policy);
    whether the budgets are jointly achievable is decided by the LP.
"""
struct ConstrainedMDP <: AbstractMDPProblem
    base_model::Symbol
    base::AbstractMDPProblem
    mdp::MDPData
    criterion::Symbol
    discount::Float64
    rhs::Vector{Float64}
    normalization::Float64
    budget_streams::Vector{Int}
    budgets::Vector{Float64}
    feasible_witness::Union{Nothing, MDPOccupationWitness}
    infeasibility_certificate::Union{Nothing, MDPDualCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    ConstrainedMDP(target_variables, feasibility_status, seed)

Sample a constrained MDP over one of the operational models with about
`target_variables` state-action pairs (the base model's exact count). Targets
above `MDP_MAX_PAIRS` raise an `ArgumentError`.
"""
function ConstrainedMDP(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    _mdp_check_target(target_variables, "constrained")
    rng = MersenneTwister(seed)
    base_model = CONSTRAINED_MDP_BASES[rand(rng, 1:3)]
    base_seed = rand(rng, 0:(2^31 - 1))
    all_three = rand(rng) < 0.5
    partner = rand(rng, 2:3)
    ctor = Dict(
        :inventory_control => InventoryControlMDP,
        :queueing_control => QueueingControlMDP,
        :machine_maintenance => MachineMaintenanceMDP,
    )[base_model]
    base = ctor(target_variables, unknown, base_seed; service_row=false)
    m = base.mdp

    budget_streams = if all_three
        [1, 2, 3]
    elseif base_model == :inventory_control
        [1, 2]      # shortage vs on-hand stock: the classic service/capital conflict
    else
        [1, partner]
    end
    budgets, witness, certificate = _mdp_plant_budgets(
        rng,
        m,
        base.criterion,
        base.discount,
        base.rhs,
        base.normalization,
        feasibility_status,
        budget_streams,
    )
    return ConstrainedMDP(
        base_model,
        base,
        m,
        base.criterion,
        base.discount,
        base.rhs,
        base.normalization,
        budget_streams,
        budgets,
        witness,
        certificate,
        feasibility_status,
    )
end

register_variant(
    :markov_decision_process,
    :constrained,
    ConstrainedMDP,
    "Constrained MDP (Altman) occupation-measure LP over an inventory, tandem-queue, or maintenance model: minimize operating cost subject to 2-3 dense budget rows on conflicting service / capital / energy / labour streams, with joint infeasibility certified by a weighted value-function Farkas certificate while every single budget stays individually achievable";
    tags=[:markov],
    max_target_variables=1_000_000,
)
