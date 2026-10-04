using JuMP
using Random

const QUEUEING_MDP_STREAMS = [:rejection, :energy, :congestion]

"""
    QueueingControlMDP <: AbstractMDPProblem

Admission and service-rate control of a two-station tandem queue with finite
buffers (blocking after service), uniformized into a discrete-time MDP
(Lippman 1975; Stidham & Weber 1993 for the control structure) and written as
the occupation-measure LP.

# State and actions

State `(n1, n2)`, `n1 ∈ 0:buffer1`, `n2 ∈ 0:buffer2` customers at the
front-end and back-office stations; index `n1 * (buffer2 + 1) + n2 + 1`.

Action `(a, k1, k2)`: admit (`a = 1`) or reject (`a = 0`) the next arrival
(forced `a = 0` when station 1 is full), and service-rate levels
`k1 ∈ 1:K1`, `k2 ∈ 1:K2` from the stations' rate menus. A station that cannot
complete a service (empty, or station 1 blocked by a full station 2) has the
single idle level `0` — rate choices that cannot change the transition law are
not offered, so no two columns are parallel. Pairs of a state are ordered
`a`-major, then `k1`, then `k2`; `action_label = 100a + 10k1 + k2`.

# Dynamics and costs

Uniformization with `Λ = arrival_rate + max(rates1) + max(rates2)`: an
admitted arrival (probability `a λ / Λ`), a station-1 completion moving a job
downstream (`μ1[k1] / Λ`), a station-2 departure (`μ2[k2] / Λ`), and a
self-loop for the remainder. Cost per unit time: `holding1 * n1 + holding2 *
n2 + energy1[k1] + energy2[k2] + rejection_penalty * λ * (1 - a)`, with convex
power-law energy menus (`energy ∝ μ^α`, `α ∈ [1.5, 3]`).

Secondary streams: `:rejection = λ (1 - a)` (lost-customer rate), `:energy`,
and `:congestion = n1 + n2` (work in process; by Little's law a response-time
SLA). Peak load exceeds even the fastest bottleneck rate
(`λ / min(max μ1, max μ2) ∈ [1.03, 1.3]` — the regime where admission control
matters), so a positive rejection rate is unavoidable under every policy.

# Feasibility

The unconstrained LP is always feasible and bounded. With probability 1/2
(always for `infeasible`) one blocking-rate SLA row `Σ rejection·x <= B` is
added (`budget_streams == [1]`): `feasible` puts `B` above the rejection rate
of the reference threshold policy (the witness), `infeasible` 8-25% below the
certified minimum over all policies (an `MDPDualCertificate`), `unknown` on
either side of that minimum. See `_mdp_plant_budgets`.
"""
struct QueueingControlMDP <: AbstractMDPProblem
    buffer1::Int
    buffer2::Int
    arrival_rate::Float64
    rates1::Vector{Float64}
    rates2::Vector{Float64}
    energy1::Vector{Float64}
    energy2::Vector{Float64}
    holding1::Float64
    holding2::Float64
    rejection_penalty::Float64
    uniformization_rate::Float64
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

"""Exact state-action pair count of the tandem-queue MDP."""
function _queueing_mdp_pairs(N1::Int, N2::Int, K1::Int, K2::Int)
    total = 0
    for n1 in 0:N1
        adm = n1 < N1 ? 2 : 1
        c1 = n1 > 0 ? K1 : 1
        # n2 = 0: station 2 idle; 0 < n2 < N2: both menus; n2 = N2: station 1 blocked.
        total += adm * (c1 + (N2 - 1) * c1 * K2 + K2)
    end
    return total
end

"""Buffers `(N1, N2)` with `N2 ≈ ratio * N1` whose pair count is closest to `target`."""
function _queueing_mdp_size(target::Int, K1::Int, K2::Int, ratio::Float64)
    best = (typemax(Int), 1, 1)
    for N1 in 1:2000
        base = max(1, round(Int, ratio * N1))
        for N2 in max(1, base - 1):(base + 1)
            err = abs(_queueing_mdp_pairs(N1, N2, K1, K2) - target)
            if err < best[1]
                best = (err, N1, N2)
            end
        end
        _queueing_mdp_pairs(N1, max(1, base - 1), K1, K2) > 2 * target && break
    end
    return best[2], best[3]
end

"""
    QueueingControlMDP(target_variables, feasibility_status, seed)

Sample a tandem-queue control MDP with about `target_variables` state-action
pairs (exact count `_queueing_mdp_pairs`). Targets above `MDP_MAX_PAIRS` raise
an `ArgumentError`. The keyword `service_row` forces the
optional service row on (`true`) or off (`false`) instead of sampling it
(`nothing`); `ConstrainedMDP` builds its base model with `service_row=false`.
"""
function QueueingControlMDP(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int;
    service_row::Union{Nothing, Bool}=nothing,
)
    _mdp_check_target(target_variables, "queueing_control")
    rng = MersenneTwister(seed)

    K1 = rand(rng, 2:3)
    K2 = rand(rng, 2:3)
    ratio = 0.5 + rand(rng)
    N1, N2 = _queueing_mdp_size(target_variables, K1, K2, ratio)

    # Loads: the bottleneck's fastest rate is still below peak demand.
    λ = 1.0
    ρ_hi = 1.03 + 0.27 * rand(rng)
    ρ_lo = 0.8 + 0.25 * rand(rng)
    ρ1, ρ2 = rand(rng) < 0.5 ? (ρ_hi, ρ_lo) : (ρ_lo, ρ_hi)
    μ1max, μ2max = λ / ρ1, λ / ρ2
    low1, low2 = 0.4 + 0.2 * rand(rng), 0.4 + 0.2 * rand(rng)
    rates1 = collect(range(low1 * μ1max, μ1max; length=K1))
    rates2 = collect(range(low2 * μ2max, μ2max; length=K2))
    holding1 = 0.5 + 1.5 * rand(rng)
    holding2 = holding1 * (0.8 + 1.2 * rand(rng))
    α = 1.5 + 1.5 * rand(rng)
    e_scale = (holding1 + holding2) * (N1 + N2) / 4 * (0.2 + 0.8 * rand(rng))
    energy1 = [e_scale * (μ / μ1max)^α for μ in rates1]
    energy2 = [e_scale * (μ / μ2max)^α for μ in rates2]
    rejection_penalty = (0.3 + 1.2 * rand(rng)) * (holding1 + holding2) * (N1 + N2) / 2
    Λ = λ + μ1max + μ2max
    criterion, discount = _mdp_criterion(rng, (100.0, 2000.0))
    has_service_row = rand(rng) < 0.5 || feasibility_status == infeasible
    service_row === nothing || (has_service_row = service_row)
    τ_admit = clamp(round(Int, (0.6 + 0.3 * rand(rng)) * N1), 1, N1)
    θ1 = clamp(round(Int, (0.2 + 0.4 * rand(rng)) * N1), 1, N1)
    θ2 = clamp(round(Int, (0.2 + 0.4 * rand(rng)) * N2), 1, N2)

    idx(n1, n2) = n1 * (N2 + 1) + n2 + 1
    S = (N1 + 1) * (N2 + 1)
    b = MDPBuilder(3; sizehint=_queueing_mdp_pairs(N1, N2, K1, K2))
    ref = Vector{Int}(undef, S)
    for n1 in 0:N1, n2 in 0:N2
        s = idx(n1, n2)
        _mdp_begin_state!(b)
        admits = n1 < N1 ? (0, 1) : (0,)
        k1s = (n1 > 0 && n2 < N2) ? (1:K1) : (0:0)
        k2s = n2 > 0 ? (1:K2) : (0:0)
        aref = n1 < τ_admit ? 1 : 0
        k1ref = first(k1s) == 0 ? 0 : (n1 >= θ1 ? K1 : cld(K1, 2))
        k2ref = first(k2s) == 0 ? 0 : (n2 >= θ2 ? K2 : cld(K2, 2))
        for a in admits, k1 in k1s, k2 in k2s
            r_arr = a * λ
            r1 = k1 > 0 ? rates1[k1] : 0.0
            r2 = k2 > 0 ? rates2[k2] : 0.0
            r_arr > 0 && _mdp_add_succ!(b, idx(n1 + 1, n2), r_arr / Λ)
            r1 > 0 && _mdp_add_succ!(b, idx(n1 - 1, n2 + 1), r1 / Λ)
            r2 > 0 && _mdp_add_succ!(b, idx(n1, n2 - 1), r2 / Λ)
            stay = (Λ - r_arr - r1 - r2) / Λ
            stay > 1e-15 && _mdp_add_succ!(b, s, stay)
            en = (k1 > 0 ? energy1[k1] : 0.0) + (k2 > 0 ? energy2[k2] : 0.0)
            rej = λ * (1 - a)
            c = holding1 * n1 + holding2 * n2 + en + rejection_penalty * rej
            k = _mdp_end_pair!(b, c, (rej, en, Float64(n1 + n2)), 100a + 10k1 + k2)
            if a == aref && k1 == k1ref && k2 == k2ref
                ref[s] = k
            end
        end
    end
    mdp = _mdp_finish(b, copy(QUEUEING_MDP_STREAMS), ref)

    # Typical start: a lightly loaded system.
    light = max(1, (N1 + N2) ÷ 5)
    start = [Float64(n1 + n2 <= light) for n1 in 0:N1 for n2 in 0:N2]
    rhs, N = _mdp_rhs(criterion, discount, start)
    budget_streams = has_service_row ? [1] : Int[]
    budgets, witness, certificate = _mdp_plant_budgets(
        rng, mdp, criterion, discount, rhs, N, feasibility_status, budget_streams
    )

    return QueueingControlMDP(
        N1,
        N2,
        λ,
        rates1,
        rates2,
        energy1,
        energy2,
        holding1,
        holding2,
        rejection_penalty,
        Λ,
        mdp,
        criterion,
        discount,
        rhs,
        N,
        budget_streams,
        budgets,
        witness,
        certificate,
        feasibility_status,
    )
end

register_variant(
    :markov_decision_process,
    :queueing_control,
    QueueingControlMDP,
    "Occupation-measure LP of uniformized admission and service-rate control of an overloaded two-station tandem queue with finite buffers (holding, convex energy, and rejection costs), with an optional blocking-rate SLA row refuted by a value-function Farkas certificate";
    tags=[:markov],
    max_target_variables=1_000_000,
)
