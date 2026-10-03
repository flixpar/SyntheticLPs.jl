# Shared machinery for the markov_decision_process category: a sparse
# state-action kernel, the occupation-measure LP builder, policy evaluation /
# policy iteration on sparse LU factorizations, planted occupation-measure
# witnesses, and LP-duality (Farkas) certificates. Names carry an `_mdp_`
# prefix because they live in the package namespace. Randomized helpers take
# the caller's `rng` first; everything else is deterministic.

using JuMP
using Random
using Distributions
using SparseArrays
using LinearAlgebra

"""
Largest number of state-action pairs (= LP columns) accepted by every
`markov_decision_process` variant. The kernel, its LU-based policy iteration,
and the JuMP model each hold the full sparse transition data, so larger targets
raise an `ArgumentError` instead of being silently undersized (same convention
as `network_flow/standard` and `telecom_network_design/standard`).
"""
const MDP_MAX_PAIRS = 1_000_000

"""
    MDPData

A finite Markov decision process in compressed sparse form. States are
`1:n_states`; the state-action pairs of state `s` are the contiguous range
`state_ptr[s]:(state_ptr[s+1]-1)`, so pairs are numbered state by state and
`length(cost)` is the number of pairs (one LP column each). Pair `k` moves to
state `trans_next[t]` with probability `trans_prob[t]` for
`t in trans_ptr[k]:(trans_ptr[k+1]-1)`; successors of a pair are distinct and
their probabilities sum to one.

`cost` is the primary per-pair cost (the LP objective) and `streams[j]` are
secondary, nonnegative per-pair metrics named `stream_names[j]` (shortage,
rejections, downtime, energy, ...) that budget rows may constrain.
`action_label[k]` is a compact domain-specific encoding of pair `k`'s action
(documented per variant), kept for inspection and tests.
`reference_policy[s]` is the pair a hand-written, domain-standard heuristic
(base-stock ordering, threshold admission, control-limit replacement, ...)
selects in state `s`; it is unichain by construction and seeds policy
iteration and the planted witnesses.
"""
struct MDPData
    n_states::Int
    state_ptr::Vector{Int}
    trans_ptr::Vector{Int}
    trans_next::Vector{Int}
    trans_prob::Vector{Float64}
    cost::Vector{Float64}
    streams::Vector{Vector{Float64}}
    stream_names::Vector{Symbol}
    action_label::Vector{Int}
    reference_policy::Vector{Int}
end

_mdp_npairs(m::MDPData) = length(m.cost)

"""Per-pair owning state (`pair_state[k]` is the state of pair `k`)."""
function _mdp_pair_state(m::MDPData)
    ps = Vector{Int}(undef, _mdp_npairs(m))
    for s in 1:(m.n_states)
        for k in m.state_ptr[s]:(m.state_ptr[s + 1] - 1)
            ps[k] = s
        end
    end
    return ps
end

"""Index of the stream named `name` in `m.stream_names`."""
_mdp_stream_index(m::MDPData, name::Symbol) = findfirst(==(name), m.stream_names)

"""
    MDPBuilder

Incremental constructor for `MDPData`. Call `_mdp_begin_state!` once per state
(in state order), then `_mdp_add_succ!` for each successor of the pair being
built and `_mdp_end_pair!` to commit it. Successors repeated *consecutively*
are merged on the fly (callers emit monotone successor sequences, e.g. clipped
inventory levels); `_mdp_end_pair!` merges any remaining duplicates and
renormalizes so each pair's probabilities sum to one up to rounding.
"""
mutable struct MDPBuilder
    n_streams::Int
    state_ptr::Vector{Int}
    trans_ptr::Vector{Int}
    trans_next::Vector{Int}
    trans_prob::Vector{Float64}
    cost::Vector{Float64}
    streams::Vector{Vector{Float64}}
    action_label::Vector{Int}
    pair_start::Int
end

function MDPBuilder(n_streams::Int; sizehint::Int=0, nnzhint::Int=0)
    b = MDPBuilder(
        n_streams,
        Int[],
        [1],
        Int[],
        Float64[],
        Float64[],
        [Float64[] for _ in 1:n_streams],
        Int[],
        1,
    )
    if sizehint > 0
        sizehint!(b.cost, sizehint)
        sizehint!(b.action_label, sizehint)
        sizehint!(b.trans_ptr, sizehint + 1)
        foreach(v -> sizehint!(v, sizehint), b.streams)
    end
    if nnzhint > 0
        sizehint!(b.trans_next, nnzhint)
        sizehint!(b.trans_prob, nnzhint)
    end
    return b
end

_mdp_begin_state!(b::MDPBuilder) = push!(b.state_ptr, length(b.cost) + 1)

function _mdp_add_succ!(b::MDPBuilder, s::Int, p::Float64)
    p > 0.0 || return nothing
    if length(b.trans_next) >= b.pair_start && b.trans_next[end] == s
        b.trans_prob[end] += p
    else
        push!(b.trans_next, s)
        push!(b.trans_prob, p)
    end
    return nothing
end

function _mdp_end_pair!(b::MDPBuilder, cost::Float64, streams, label::Int)
    lo = b.pair_start
    hi = length(b.trans_next)
    hi >= lo || error("MDP pair has no successors")
    # Merge non-consecutive duplicates (rare: e.g. a shock jump onto a level a
    # degradation increment also reaches), then renormalize.
    if !allunique(view(b.trans_next, lo:hi))
        nx = b.trans_next[lo:hi]
        pr = b.trans_prob[lo:hi]
        order = sortperm(nx)
        resize!(b.trans_next, lo - 1)
        resize!(b.trans_prob, lo - 1)
        for o in order
            if length(b.trans_next) >= lo && b.trans_next[end] == nx[o]
                b.trans_prob[end] += pr[o]
            else
                push!(b.trans_next, nx[o])
                push!(b.trans_prob, pr[o])
            end
        end
        hi = length(b.trans_next)
    end
    total = sum(view(b.trans_prob, lo:hi))
    view(b.trans_prob, lo:hi) ./= total
    push!(b.cost, cost)
    for j in 1:(b.n_streams)
        push!(b.streams[j], streams[j])
    end
    push!(b.action_label, label)
    push!(b.trans_ptr, length(b.trans_next) + 1)
    b.pair_start = length(b.trans_next) + 1
    return length(b.cost)
end

function _mdp_finish(b::MDPBuilder, stream_names::Vector{Symbol}, reference_policy::Vector{Int})
    n_states = length(b.state_ptr)
    state_ptr = vcat(b.state_ptr, length(b.cost) + 1)
    all(state_ptr[s + 1] > state_ptr[s] for s in 1:n_states) ||
        error("every MDP state needs at least one action")
    length(reference_policy) == n_states || error("reference policy length mismatch")
    all(state_ptr[s] <= reference_policy[s] < state_ptr[s + 1] for s in 1:n_states) ||
        error("reference policy picks a pair outside its state")
    return MDPData(
        n_states,
        state_ptr,
        b.trans_ptr,
        b.trans_next,
        b.trans_prob,
        b.cost,
        b.streams,
        stream_names,
        b.action_label,
        reference_policy,
    )
end

"""
    MDPOccupationWitness

Planted feasible point of the occupation-measure LP: `occupation[k]` is the
value of column `k` (state-action pair `k`). It is the exact (sparse-LU)
occupation measure of a stationary deterministic policy — or a convex
combination of two, which the balance rows (linear, same right-hand side)
preserve — so it satisfies every balance row (and, under the average-cost
criterion, the normalization row) and every budget row with the planted margin.
`policies` lists the per-state pair choice of each mixed policy and `mix` their
convex weights.
"""
struct MDPOccupationWitness
    occupation::Vector{Float64}
    policies::Vector{Vector{Int}}
    mix::Vector{Float64}
end

"""
    MDPDualCertificate

LP-duality (Farkas) certificate that the budget rows cannot all hold. With
`w = weights` (one nonnegative multiplier per budget row, aligned with the
problem's `budget_streams`) and the combined per-pair metric
`d_w(k) = Σ_j w_j * streams[budget_streams[j]][k]`, the stored per-state
`potential` satisfies, for every state-action pair `k` of state `s`:

  - discounted criterion: `potential[s] - γ Σ_t P(t|k) potential[t] <= d_w(k)`;
  - average criterion:    `gain + potential[s] - Σ_t P(t|k) potential[t] <= d_w(k)`.

Multiplying balance row `s` by `potential[s]` (and the average-cost
normalization row by `gain`) and summing shows that EVERY nonnegative point of
the balance rows has `Σ_k d_w(k) x_k >= lower_bound`, where `lower_bound` is
`Σ_s rhs[s] * potential[s]` (discounted) or `normalization * gain` (average).
Since `weighted_budget = Σ_j w_j B_j < lower_bound`, the budget rows are
jointly unsatisfiable — a contradiction derived from LP rows alone (there are
no integer variables, so it is independent of `relax_integer`). The potential
is the optimal value function of `d_w` from policy iteration, corrected by the
MacQueen (discounted) or Odoni (average) bound, so `lower_bound` is essentially
the true minimum of `d_w` over all policies: refuting the budgets takes the
whole transition structure, not a presolve-level aggregate argument.
"""
struct MDPDualCertificate
    weights::Vector{Float64}
    potential::Vector{Float64}
    gain::Float64
    lower_bound::Float64
    weighted_budget::Float64
end

"""
Abstract supertype of the markov_decision_process variants. Every subtype has
the fields `mdp::MDPData`, `criterion::Symbol` (`:discounted` or `:average`),
`discount::Float64` (1.0 under `:average`), `rhs::Vector{Float64}` (balance-row
right-hand sides; zeros under `:average`), `normalization::Float64`,
`budget_streams::Vector{Int}`, and `budgets::Vector{Float64}` — which is all
`build_model` reads — plus `feasible_witness`, `infeasibility_certificate`, and
`feasibility_status`.
"""
abstract type AbstractMDPProblem <: ProblemGenerator end

# --- sizing / scalar helpers --------------------------------------------------

function _mdp_check_target(target::Int, name::AbstractString)
    target >= 1 || throw(ArgumentError("target_variables must be >= 1 (got $target)."))
    target <= MDP_MAX_PAIRS || throw(
        ArgumentError(
            "markov_decision_process/$name supports at most $MDP_MAX_PAIRS state-action " *
            "pairs; requested $target.",
        ),
    )
    return nothing
end

"""
    _mdp_criterion(rng, horizon_range) -> (criterion, discount)

Sample the optimality criterion: discounted (65%) with discount factor
`1 - 1/H` for an effective planning horizon `H` drawn log-uniformly from
`horizon_range` (periods / transitions), or long-run average cost (35%,
discount reported as 1.0). Both draws are always made.
"""
function _mdp_criterion(rng::AbstractRNG, horizon_range::Tuple{Float64, Float64})
    u = rand(rng)
    lo, hi = log(horizon_range[1]), log(horizon_range[2])
    H = exp(lo + rand(rng) * (hi - lo))
    return u < 0.65 ? (:discounted, 1.0 - 1.0 / H) : (:average, 1.0)
end

"""
    _mdp_rhs(criterion, discount, start_weights) -> (rhs, normalization)

Balance-row right-hand sides. The LP is scaled so the occupation measure sums
to `normalization = n_states` under both criteria (average state occupancy 1,
which keeps column values well above solver tolerances). Discounted: the
state-relevance (initial) distribution `μ` mixes 70% of a domain "typical
start" profile with 30% uniform mass, so every state has a positive right-hand
side and no balance row can be forced to zero by presolve;
`rhs = (1-γ) * n_states * μ`. Average cost: the balance rows are homogeneous
and `Σ x = normalization` is a separate row.
"""
function _mdp_rhs(criterion::Symbol, discount::Float64, start_weights::Vector{Float64})
    S = length(start_weights)
    N = Float64(S)
    criterion == :average && return zeros(S), N
    w = start_weights ./ sum(start_weights)
    mu = 0.7 .* w .+ 0.3 / S
    return (1.0 - discount) * N .* mu, N
end

# --- policy evaluation / iteration ----------------------------------------------

"""Sparse `I - γ P_π` (rows = from-states) for the per-state pair choice `policy`."""
function _mdp_policy_matrix(m::MDPData, policy::Vector{Int}, γ::Float64)
    S = m.n_states
    I_ = Int[]
    J_ = Int[]
    V_ = Float64[]
    sizehint!(I_, 4S)
    sizehint!(J_, 4S)
    sizehint!(V_, 4S)
    for s in 1:S
        push!(I_, s)
        push!(J_, s)
        push!(V_, 1.0)
        k = policy[s]
        for t in m.trans_ptr[k]:(m.trans_ptr[k + 1] - 1)
            push!(I_, s)
            push!(J_, m.trans_next[t])
            push!(V_, -γ * m.trans_prob[t])
        end
    end
    return sparse(I_, J_, V_, S, S)
end

"""One-step look-ahead `d[k] + γ Σ_t P(t|k) v[t]` for pair `k`."""
@inline function _mdp_q(m::MDPData, d::Vector{Float64}, v::Vector{Float64}, γ::Float64, k::Int)
    acc = 0.0
    @inbounds for t in m.trans_ptr[k]:(m.trans_ptr[k + 1] - 1)
        acc += m.trans_prob[t] * v[m.trans_next[t]]
    end
    return d[k] + γ * acc
end

"""
    _mdp_policy_iteration(m, d, γ, policy0; max_iter=50) -> (v, policy)

Howard policy iteration minimizing the discounted per-pair cost `d` (`γ < 1`)
from `policy0`. Each evaluation is a sparse LU solve of `(I - γ P_π) v = d_π`;
improvement switches a state only on a strict relative gain of 1e-12, so the
iteration terminates. Nothing derived from `v` relies on convergence for its
*validity* (the bounds below correct for any residual) — only for tightness.
"""
function _mdp_policy_iteration(
    m::MDPData, d::Vector{Float64}, γ::Float64, policy0::Vector{Int}; max_iter::Int=50
)
    S = m.n_states
    policy = copy(policy0)
    v = zeros(S)
    for _ in 1:max_iter
        A = _mdp_policy_matrix(m, policy, γ)
        v = lu(A) \ [d[policy[s]] for s in 1:S]
        changed = false
        for s in 1:S
            best_k = policy[s]
            best = _mdp_q(m, d, v, γ, best_k)
            thresh = best - 1e-12 * (abs(best) + 1.0)
            for k in m.state_ptr[s]:(m.state_ptr[s + 1] - 1)
                k == policy[s] && continue
                q = _mdp_q(m, d, v, γ, k)
                if q < thresh && q < best
                    best = q
                    best_k = k
                end
            end
            if best_k != policy[s]
                policy[s] = best_k
                changed = true
            end
        end
        changed || break
    end
    return v, policy
end

"""
    _mdp_occupation(m, policy, criterion, γ, rhs, normalization) -> Vector{Float64}

Per-pair occupation measure of the stationary deterministic `policy` (zero on
pairs the policy does not use), by one sparse LU solve:

  - discounted: `(I - γ P_π)ᵀ y = rhs`, the scaled discounted state-visit
    frequencies (`Σ y = normalization`);
  - average: the stationary distribution scaled to `normalization`, from
    `(I - P_π)ᵀ y = 0`: one equation is dropped and `y_r = 1` is pinned for a
    state `r` that is recurrent under `π` (the most-visited state of a Cesàro
    power iteration), then `y` is rescaled to sum to `normalization`. Pinning
    keeps the system sparse — replacing an equation by `Σ y = N` would add a
    dense row and wreck the LU fill. Requires `π` unichain, which every
    reference policy is by construction.

Round-off on transient states (`|y| <= 1e-12 * normalization`) is clipped to
zero; a materially negative entry raises (it would mean `π` is multichain).
"""
function _mdp_occupation(
    m::MDPData,
    policy::Vector{Int},
    criterion::Symbol,
    γ::Float64,
    rhs::Vector{Float64},
    normalization::Float64,
)
    S = m.n_states
    if criterion == :discounted
        At = sparse(transpose(_mdp_policy_matrix(m, policy, γ)))
        y = lu(At) \ rhs
    else
        At = sparse(transpose(_mdp_policy_matrix(m, policy, 1.0)))   # (I - P_π)ᵀ
        Pt = spdiagm(0 => ones(S)) - At                                  # P_πᵀ
        u = fill(1.0 / S, S)
        visits = zeros(S)
        for _ in 1:200
            u = Pt * u
            visits .+= u
        end
        r = argmax(visits)
        keep = [s for s in 1:S if s != r]
        y = zeros(S)
        y[r] = 1.0
        if S > 1
            y[keep] = lu(At[keep, keep]) \ (-Vector(At[keep, r]))
        end
        y .*= normalization / sum(y)
    end
    minimum(y) >= -1e-9 * normalization ||
        error("occupation measure has a negative entry $(minimum(y)); policy is not unichain")
    tol = 1e-12 * normalization
    y = [abs(yi) <= tol ? 0.0 : max(yi, 0.0) for yi in y]
    x = zeros(_mdp_npairs(m))
    for s in 1:S
        x[policy[s]] = y[s]
    end
    return x
end

"""Combined per-pair metric `Σ_j w_j * streams[budget_streams[j]]`."""
function _mdp_combined_stream(m::MDPData, budget_streams::Vector{Int}, w::Vector{Float64})
    d = zeros(_mdp_npairs(m))
    for (j, sj) in enumerate(budget_streams)
        w[j] == 0.0 && continue
        d .+= w[j] .* m.streams[sj]
    end
    return d
end

"""
    _mdp_lower_bound(m, d, criterion, γ, rhs, normalization) -> (potential, gain, lb, policy)

Provably valid lower bound `lb` on `Σ_k d[k] x_k` over every nonnegative point
of the balance rows, with the dual vector proving it (see
`MDPDualCertificate`), and the policy-iteration policy that (nearly) attains it.

  - Discounted: `v` from policy iteration, then the MacQueen correction
    `v + δ/(1-γ)` with `δ = min_k (d + γ P v - v)[k]` makes every pair's dual
    inequality hold no matter how far policy iteration got.
  - Average: policy iteration on the surrogate discount `1 - 1e-4` gives a
    relative value function `h`; the Odoni bound `g = min_k (d + P h - h)[k]`
    makes `(g, h)` dual feasible for the average-cost LP, within about
    `1e-4 * span(h)` of the optimal gain.

A safety slack of `1e-9 * (max|d| + 1)` is taken off every pair inequality so
the certificate also verifies under floating-point recomputation.
"""
function _mdp_lower_bound(
    m::MDPData,
    d::Vector{Float64},
    criterion::Symbol,
    γ::Float64,
    rhs::Vector{Float64},
    normalization::Float64,
)
    ps = _mdp_pair_state(m)
    γe = criterion == :discounted ? γ : 1.0 - 1e-4
    v, policy = _mdp_policy_iteration(m, d, γe, m.reference_policy)
    safety = 1e-9 * (maximum(abs, d) + 1.0)
    γc = criterion == :discounted ? γ : 1.0
    δ = Inf
    for k in 1:_mdp_npairs(m)
        δ = min(δ, _mdp_q(m, d, v, γc, k) - v[ps[k]])
    end
    if criterion == :discounted
        potential = v .+ (δ - safety) / (1.0 - γ)
        return potential, 0.0, sum(rhs .* potential), policy
    end
    gain = δ - safety
    return v, gain, normalization * gain, policy
end

"""Value `Σ_k s[k] x[k]` of a per-pair stream at a per-pair point."""
_mdp_value(s::Vector{Float64}, x::Vector{Float64}) = sum(s[k] * x[k] for k in eachindex(x); init=0.0)

# --- budget planting ---------------------------------------------------------------

"""
    _mdp_plant_budgets(rng, m, criterion, γ, rhs, N, status, streams)
        -> (budgets, witness, certificate)

Plant budget rows `Σ_k streams_j[k] x_k <= B_j` for the stream indices
`streams` (nonnegative per-pair metrics), consistently with `status`:

  - `feasible`: the witness is the occupation measure of the domain reference
    policy — with several rows, mixed with the policy that is optimal for a
    random weighting of the budget streams, so the budgets sit near the
    Pareto frontier and bind at the optimum; each `B_j` is the witness's value
    plus a 5-25% relative margin and an absolute floor, so the witness meets
    every row strictly. With no rows, the reference occupation measure is
    stored as the witness of the (always feasible) unconstrained LP.
  - `infeasible`: a weight vector `w` (all on a single row, or Dirichlet over
    several — the most conflicting of three draws) and the certified combined
    lower bound `L_w`; budgets are placed with `Σ_j w_j B_j = (1 - m) L_w`,
    `m ∈ [0.08, 0.25]`. With several rows the deficit is spread so that,
    whenever the streams genuinely conflict, every `B_j` individually exceeds
    its own optimum `L_j`: each row alone is satisfiable and only the
    combination is refuted.
  - `unknown`: `B_j = L_j + u_j * max(R_j - L_j, 0.3 L_j)` between each stream's
    own optimum `L_j` and the reference policy's value `R_j`, with `u` drawn on
    both sides of 0 for a single row (`[-0.5, 1]`) and in `[0.4, 1.4]` for
    several rows, where the rows' conflict decides; no witness or certificate.
"""
function _mdp_plant_budgets(
    rng::AbstractRNG,
    m::MDPData,
    criterion::Symbol,
    γ::Float64,
    rhs::Vector{Float64},
    N::Float64,
    status::FeasibilityStatus,
    streams::Vector{Int},
)
    nb = length(streams)
    # Draws made unconditionally keep the RNG stream aligned across statuses.
    margins = 0.05 .+ 0.2 .* rand(rng, max(nb, 1))
    infeasible_margin = 0.08 + 0.17 * rand(rng)
    us = nb == 1 ? [-0.5 + 1.5 * rand(rng)] : 0.4 .+ 1.0 .* rand(rng, max(nb, 1))
    mix_theta = 0.3 + 0.5 * rand(rng)
    raw_weights = [rand(rng, Gamma(1.0, 1.0), max(nb, 1)) for _ in 1:3]

    budgets = zeros(nb)
    if nb == 0
        status == infeasible &&
            error("an infeasible MDP instance needs at least one budget row")
        status == feasible || return budgets, nothing, nothing
        x = _mdp_occupation(m, m.reference_policy, criterion, γ, rhs, N)
        return budgets, MDPOccupationWitness(x, [copy(m.reference_policy)], [1.0]), nothing
    end

    xref = _mdp_occupation(m, m.reference_policy, criterion, γ, rhs, N)
    R = [_mdp_value(m.streams[j], xref) for j in streams]
    # Natural scale of each stream (reference value, floored by the stream's
    # largest per-pair value at occupancy 1e-3 * N).
    scale = [max(abs(R[i]), maximum(m.streams[j]) * N * 1e-3, 1e-9) for (i, j) in enumerate(streams)]

    if status == feasible
        x = xref
        pols, mixw = [copy(m.reference_policy)], [1.0]
        if nb > 1
            w = raw_weights[1][1:nb] ./ scale
            _, _, _, pol = _mdp_lower_bound(m, _mdp_combined_stream(m, streams, w), criterion, γ, rhs, N)
            # The frontier policy can be multichain under the average
            # criterion; the reference alone is then the witness.
            xo = try
                _mdp_occupation(m, pol, criterion, γ, rhs, N)
            catch
                nothing
            end
            if xo !== nothing
                x = mix_theta .* xref .+ (1.0 - mix_theta) .* xo
                pols, mixw = [copy(m.reference_policy), pol], [mix_theta, 1.0 - mix_theta]
            end
        end
        for (i, j) in enumerate(streams)
            val = _mdp_value(m.streams[j], x)
            budgets[i] = val + margins[i] * abs(val) + 0.01 * scale[i]
        end
        return budgets, MDPOccupationWitness(x, pols, mixw), nothing
    end

    # Each stream's own certified optimum (with its proof, reused below).
    own = [_mdp_lower_bound(m, m.streams[j], criterion, γ, rhs, N) for j in streams]
    L = [o[3] for o in own]

    if status == unknown
        for i in 1:nb
            budgets[i] = L[i] + us[i] * max(R[i] - L[i], 0.3 * abs(L[i]))
        end
        return budgets, nothing, nothing
    end

    # infeasible
    if nb == 1
        pot, gain, lb, _ = own[1]
        lb > 0 || error("budget stream has a nonpositive optimum; cannot plant infeasibility")
        budgets[1] = (1.0 - infeasible_margin) * lb
        return budgets, nothing, MDPDualCertificate([1.0], pot, gain, lb, budgets[1])
    end
    best = nothing
    for trial in 1:3
        w = raw_weights[trial][1:nb] ./ scale
        w ./= sum(w)
        pot, gain, lb, _ = _mdp_lower_bound(m, _mdp_combined_stream(m, streams, w), criterion, γ, rhs, N)
        rel = (lb - sum(w .* L)) / max(abs(lb), 1e-12)
        if best === nothing || rel > best[1]
            best = (rel, w, pot, gain, lb)
        end
    end
    _, w, pot, gain, lb = best
    lb > 0 || error("combined budget stream has a nonpositive optimum; cannot plant infeasibility")
    target = (1.0 - infeasible_margin) * lb          # Σ_j w_j B_j
    pool = target - sum(w .* L)
    if pool > 0
        # Every row individually satisfiable (B_j > L_j); only the weighted
        # combination is refuted.
        share = raw_weights[1][1:nb] ./ sum(raw_weights[1][1:nb])
        for i in 1:nb
            budgets[i] = L[i] + pool * share[i] / w[i]
        end
    else
        # The streams barely conflict: every budget goes below its own optimum.
        budgets .= (1.0 - infeasible_margin) .* L
    end
    return budgets, nothing, MDPDualCertificate(w, pot, gain, lb, sum(w .* budgets))
end

# --- model -------------------------------------------------------------------

"""
    _mdp_balance_matrix(m, γ) -> SparseMatrixCSC (n_states × n_pairs)

Column `k` (pair `k` of state `s`) has `+1` in row `s` and `-γ P(t|k)` in row
`t` for every successor `t`, merged when `t == s`. Exact zeros (a probability-1
self-loop under `γ = 1`) are dropped.
"""
function _mdp_balance_matrix(m::MDPData, γ::Float64)
    n = _mdp_npairs(m)
    total = n + length(m.trans_next)
    I_ = Vector{Int}(undef, total)
    J_ = Vector{Int}(undef, total)
    V_ = Vector{Float64}(undef, total)
    idx = 0
    for s in 1:(m.n_states)
        for k in m.state_ptr[s]:(m.state_ptr[s + 1] - 1)
            idx += 1
            I_[idx], J_[idx], V_[idx] = s, k, 1.0
            for t in m.trans_ptr[k]:(m.trans_ptr[k + 1] - 1)
                idx += 1
                I_[idx], J_[idx], V_[idx] = m.trans_next[t], k, -γ * m.trans_prob[t]
            end
        end
    end
    A = sparse(I_, J_, V_, m.n_states, n)
    dropzeros!(A)
    return A
end

"""
    build_model(prob::AbstractMDPProblem)

Occupation-measure (dual) LP of the MDP (Manne 1960; budget rows as in
Altman's constrained MDPs). Deterministic — reads only struct fields.

  - `x[k] >= 0`: scaled discounted / long-run frequency of state-action pair `k`.
  - `balance[s]`: `Σ_{k∈A(s)} x[k] - γ Σ_k P(s|k) x[k] = rhs[s]` per state
    (`γ = 1`, `rhs = 0` under the average criterion).
  - `normalization` (average criterion only): `Σ_k x[k] = normalization`.
  - `budget[j]`: `Σ_k streams[budget_streams[j]][k] x[k] <= budgets[j]`.
  - Objective: minimize `Σ_k cost[k] x[k]`.
"""
function build_model(prob::AbstractMDPProblem)
    m = prob.mdp
    n = _mdp_npairs(m)
    model = Model()
    @variable(model, x[1:n] >= 0)
    γ = prob.criterion == :discounted ? prob.discount : 1.0
    At = sparse(transpose(_mdp_balance_matrix(m, γ)))   # column s = row s of A
    rows = Vector{AffExpr}(undef, m.n_states)
    for s in 1:(m.n_states)
        span = At.colptr[s]:(At.colptr[s + 1] - 1)
        e = AffExpr(0.0)
        sizehint!(e.terms, length(span))
        for p in span
            add_to_expression!(e, At.nzval[p], x[At.rowval[p]])
        end
        rows[s] = e
    end
    @constraint(model, balance[s in 1:(m.n_states)], rows[s] == prob.rhs[s])
    if prob.criterion == :average
        @constraint(model, normalization, sum(x) == prob.normalization)
    end
    budget_rows = Vector{AffExpr}(undef, length(prob.budget_streams))
    for (j, sj) in enumerate(prob.budget_streams)
        d = m.streams[sj]
        e = AffExpr(0.0)
        for k in 1:n
            d[k] != 0.0 && add_to_expression!(e, d[k], x[k])
        end
        budget_rows[j] = e
    end
    @constraint(model, budget[j in eachindex(budget_rows)], budget_rows[j] <= prob.budgets[j])
    obj = AffExpr(0.0)
    sizehint!(obj.terms, n)
    for k in 1:n
        m.cost[k] != 0.0 && add_to_expression!(obj, m.cost[k], x[k])
    end
    @objective(model, Min, obj)
    return model
end

# --- distribution helpers ----------------------------------------------------------

"""
    _mdp_truncated_pmf(dist, tail) -> Vector{Float64}

Probability mass of a nonnegative integer distribution on `0:D`, where `D` is
the smallest value with upper tail `P(X > D) <= tail`; the residual tail is
lumped into `D`, so the vector sums to one.
"""
function _mdp_truncated_pmf(dist::DiscreteUnivariateDistribution, tail::Float64)
    D = 0
    while ccdf(dist, D) > tail
        D += 1
    end
    p = [pdf(dist, d) for d in 0:D]
    p[end] += max(1.0 - sum(p), 0.0)
    return p ./ sum(p)
end

"""Negative binomial with mean `μ` and dispersion `r` (variance `μ + μ²/r`)."""
_mdp_negbin(μ::Float64, r::Float64) = NegativeBinomial(r, r / (r + μ))
