using JuMP
using Random
using Distributions

const MAINTENANCE_MDP_CYCLES = (1, 2, 4, 13, 26, 52)
const MAINTENANCE_MDP_STREAMS = [:downtime, :labour, :spares_held]

"""
    MachineMaintenanceMDP <: AbstractMDPProblem

Condition-based maintenance of a deteriorating production asset with a
spare-parts stock and a seasonal production calendar, jointly optimizing
maintenance, operating speed, and spare replenishment (Wang 2002 survey;
Elwany & Gebraeel 2008 for joint maintenance / spares), written as the
occupation-measure LP.

# State and actions

State `(c, k, e)`: condition level `c ∈ 0:n_conditions` (`0` = as new,
`n_conditions` = failed), spare parts on hand `k ∈ 0:max_spares`, and phase
`e ∈ 1:n_phases` of the production calendar (`{1,2,4,13,26,52}`); index
`(e-1)(C+1)(K+1) + c(K+1) + k + 1`.

Maintenance action `m` (codes): `1:n_speeds` run at speed level `m`;
`n_speeds+1` imperfect repair (only for `c >= 2`); `n_speeds+2` preventive
replacement (`1 <= c < C`, needs a spare); `n_speeds+3` corrective
replacement (failed, needs a spare); `n_speeds+4` emergency replacement with
an expedited part (failed, no spare on hand); `n_speeds+5` wait (failed, no
spare). Combined with a spare order `q ∈ 0:min(max_order, max_spares -
k_after)` delivered next period (`k_after = k -` spares used). Pairs of a state
are ordered `m`-major, `q` inner; `action_label = 100m + q`.

# Dynamics and costs

Running at speed `v` degrades the condition by a Poisson increment with rate
`wear_rate * speed[v]^wear_exponent * (1 + acceleration * c / C) *
phase_load[e]` (tail below 1e-4 lumped, capped at failure), plus a sudden
shock failure with probability `shock_prob * speed[v]`. Repair lowers the
condition by a uniform `1:repair_depth`; replacements restore `c = 0`; waiting
stays failed. Spares follow `k' = k_after + q`; the phase advances cyclically.
Per-period cost: `-phase_margin[e] * output + action cost + spare_holding *
k_after + order_fixed * [q > 0] + spare_cost * q`, where output is
`speed[v] * (1 - efficiency_loss * (c/C)^2)` when running and the
non-downtime fraction of the period otherwise.

Secondary streams: `:downtime` (fraction of the period the asset is down:
planned repair/replacement windows, corrective and emergency outages, full
periods while waiting), `:labour` (maintenance crew hours), `:spares_held`
(`k_after`). Shocks and wear make some downtime unavoidable under every policy.

# Feasibility

The unconstrained LP is always feasible and bounded. With probability 1/2
(always for `infeasible`) one availability row `Σ downtime·x <= B` is added
(`budget_streams == [1]`): `feasible` puts `B` above the downtime of the
control-limit reference policy (the witness), `infeasible` 8-25% below the
certified minimum over all policies (an `MDPDualCertificate`), `unknown` on
either side of that minimum. See `_mdp_plant_budgets`.
"""
struct MachineMaintenanceMDP <: AbstractMDPProblem
    n_conditions::Int
    max_spares::Int
    max_order::Int
    n_phases::Int
    speeds::Vector{Float64}
    wear_rate::Float64
    wear_exponent::Float64
    acceleration::Float64
    shock_prob::Float64
    repair_depth::Int
    phase_load::Vector{Float64}
    phase_margin::Vector{Float64}
    efficiency_loss::Float64
    downtime::Vector{Float64}       # repair, preventive, corrective, emergency, wait
    action_cost::Vector{Float64}    # repair, preventive, corrective, emergency, wait
    labour_hours::Vector{Float64}   # repair, preventive, corrective, emergency, wait
    spare_holding::Float64
    order_fixed::Float64
    spare_cost::Float64
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

"""Maintenance actions `(code, spares_used)` available in condition `c` with `k` spares."""
function _maintenance_mdp_actions(c::Int, k::Int, C::Int, V::Int)
    acts = Tuple{Int, Int}[]
    if c < C
        for v in 1:V
            push!(acts, (v, 0))
        end
        c >= 2 && push!(acts, (V + 1, 0))
        (c >= 1 && k >= 1) && push!(acts, (V + 2, 1))
    elseif k >= 1
        push!(acts, (V + 3, 1))
    else
        push!(acts, (V + 4, 0))
        push!(acts, (V + 5, 0))
    end
    return acts
end

"""Exact state-action pair count of the maintenance MDP."""
function _maintenance_mdp_pairs(C::Int, K::Int, Q::Int, V::Int, E::Int)
    total = 0
    for c in 0:C, k in 0:K
        for (_, used) in _maintenance_mdp_actions(c, k, C, V)
            total += min(Q, K - (k - used)) + 1
        end
    end
    return E * total
end

"""
    _maintenance_mdp_size(target, V, Q_draw, C_pref, K_pref) -> (E, C, K)

Grid search over calendar length, condition resolution, and spare capacity for
the exact pair count closest to `target`, mildly preferring the sampled
condition resolution and spare capacity.
"""
function _maintenance_mdp_size(target::Int, V::Int, Q_draw::Int, C_pref::Int, K_pref::Int)
    best = (Inf, 1, 4, 1)
    for C in 3:80, K in 1:24
        base = _maintenance_mdp_pairs(C, K, min(Q_draw, K), V, 1)
        for E in MAINTENANCE_MDP_CYCLES
            score =
                abs(base * E - target) / target +
                0.02 * abs(log(C / C_pref)) +
                0.02 * abs(log(K / K_pref))
            if score < best[1]
                best = (score, E, C, K)
            end
        end
    end
    return best[2], best[3], best[4]
end

"""
    MachineMaintenanceMDP(target_variables, feasibility_status, seed)

Sample a condition-based maintenance MDP with about `target_variables`
state-action pairs (exact count `_maintenance_mdp_pairs`). Targets above
`MDP_MAX_PAIRS` raise an `ArgumentError`. The keyword `service_row` forces the
optional service row on (`true`) or off (`false`) instead of sampling it
(`nothing`); `ConstrainedMDP` builds its base model with `service_row=false`.
"""
function MachineMaintenanceMDP(
    target_variables::Int,
    feasibility_status::FeasibilityStatus,
    seed::Int;
    service_row::Union{Nothing, Bool}=nothing,
)
    _mdp_check_target(target_variables, "machine_maintenance")
    rng = MersenneTwister(seed)

    V = rand(rng, 1:3)
    Q_draw = rand(rng, 1:3)
    C_pref = clamp(round(Int, 3.0 * target_variables^0.18 * (0.8 + 0.4 * rand(rng))), 4, 60)
    K_pref = clamp(round(Int, 1.2 * target_variables^0.15 * (0.8 + 0.4 * rand(rng))), 1, 20)
    E, C, K = _maintenance_mdp_size(target_variables, V, Q_draw, C_pref, K_pref)
    Q = min(Q_draw, K)

    speeds = V == 1 ? [1.0] : collect(range(0.8, 1.2; length=V))
    # Wear: mean time from new to failure at nominal speed ≈ 8-30 periods.
    life = 8.0 + 22.0 * rand(rng)
    acceleration = 0.5 + 1.5 * rand(rng)
    wear_rate = C / life * (1 + acceleration / 2)^(-1)
    wear_exponent = 1.5 + 1.5 * rand(rng)
    shock_prob = 0.003 + 0.017 * rand(rng)
    repair_depth = max(2, round(Int, C * (0.2 + 0.3 * rand(rng))))
    amp = 0.1 + 0.3 * rand(rng)
    shift = rand(rng)
    season = [E == 1 ? 1.0 : 1.0 + amp * sin(2π * ((e - 1) / E - shift)) for e in 1:E]
    phase_load =
        0.9 .+ 0.2 .* (season .- minimum(season)) ./ max(maximum(season) - minimum(season), 1e-9)
    margin = 50.0 + 450.0 * rand(rng)
    phase_margin = margin .* season
    efficiency_loss = 0.1 + 0.3 * rand(rng)
    # repair, preventive, corrective, emergency, wait
    downtime = [
        0.1 + 0.2 * rand(rng),
        0.2 + 0.3 * rand(rng),
        0.5 + 0.4 * rand(rng),
        0.6 + 0.4 * rand(rng),
        1.0,
    ]
    part = margin * (1.0 + 3.0 * rand(rng))
    action_cost = [
        0.1 * part * (1 + rand(rng)),
        part * (1.0 + 0.3 * rand(rng)),
        part * (2.0 + 2.0 * rand(rng)),
        part * (3.0 + 3.0 * rand(rng)),
        0.0,
    ]
    labour_hours = [
        4.0 + 8.0 * rand(rng),
        8.0 + 8.0 * rand(rng),
        16.0 + 16.0 * rand(rng),
        20.0 + 20.0 * rand(rng),
        0.0,
    ]
    spare_cost = part * (0.6 + 0.3 * rand(rng))
    spare_holding = spare_cost * (0.2 + 0.2 * rand(rng)) / 52
    order_fixed = spare_cost * (0.1 + 0.4 * rand(rng))
    control_limit = clamp(round(Int, C * (0.5 + 0.35 * rand(rng))), 1, C - 1)
    spare_target = clamp(round(Int, K * (0.3 + 0.5 * rand(rng))), 1, K)
    criterion, discount = _mdp_criterion(rng, (20.0, 400.0))
    has_service_row = rand(rng) < 0.5 || feasibility_status == infeasible
    service_row === nothing || (has_service_row = service_row)

    # Degradation outcome per (speed, condition, phase): successor condition
    # distribution, shared by every spare level.
    vref = argmin(abs.(speeds .- 1.0))
    deg_lv = Array{Vector{Int}}(undef, V, C, E)
    deg_pr = Array{Vector{Float64}}(undef, V, C, E)
    for v in 1:V, c in 0:(C - 1), e in 1:E
        rate = wear_rate * speeds[v]^wear_exponent * (1 + acceleration * c / C) * phase_load[e]
        pmf = _mdp_truncated_pmf(Poisson(rate), 1e-4)
        shock = min(shock_prob * speeds[v], 0.5)
        lv = Int[]
        pr = Float64[]
        for (di, p) in enumerate(pmf)
            c2 = min(c + di - 1, C)
            if !isempty(lv) && lv[end] == c2
                pr[end] += (1 - shock) * p
            else
                push!(lv, c2)
                push!(pr, (1 - shock) * p)
            end
        end
        if lv[end] == C
            pr[end] += shock
        else
            push!(lv, C)
            push!(pr, shock)
        end
        deg_lv[v, c + 1, e] = lv
        deg_pr[v, c + 1, e] = pr
    end

    Sp = (C + 1) * (K + 1)
    idx(c, k, e) = (e - 1) * Sp + c * (K + 1) + k + 1
    b = MDPBuilder(3; sizehint=_maintenance_mdp_pairs(C, K, Q, V, E))
    ref = Vector{Int}(undef, E * Sp)
    for e in 1:E, c in 0:C, k in 0:K
        s = idx(c, k, e)
        _mdp_begin_state!(b)
        e2 = mod1(e + 1, E)
        # Reference: control-limit replacement, base-stock spares, nominal speed.
        mref = if c == C
            k >= 1 ? V + 3 : V + 4
        elseif c >= control_limit && k >= 1
            V + 2
        else
            vref
        end
        for (mcode, used) in _maintenance_mdp_actions(c, k, C, V)
            k_after = k - used
            qref = k_after < spare_target ? min(Q, spare_target - k_after) : 0
            for q in 0:min(Q, K - k_after)
                k2 = k_after + q
                if mcode <= V
                    for (c2, p) in zip(deg_lv[mcode, c + 1, e], deg_pr[mcode, c + 1, e])
                        _mdp_add_succ!(b, idx(c2, k2, e2), p)
                    end
                    output = speeds[mcode] * (1 - efficiency_loss * (c / C)^2)
                    act_cost, dt, lab = 0.0, 0.0, 0.0
                else
                    a = mcode - V
                    if a == 1          # imperfect repair: c - U{1..repair_depth}, floored at 0
                        for r in repair_depth:-1:1
                            _mdp_add_succ!(b, idx(max(c - r, 0), k2, e2), 1.0 / repair_depth)
                        end
                    elseif a == 5      # wait: stays failed
                        _mdp_add_succ!(b, idx(C, k2, e2), 1.0)
                    else               # replacements restore as-new
                        _mdp_add_succ!(b, idx(0, k2, e2), 1.0)
                    end
                    dt = downtime[a]
                    output = 1.0 - dt
                    act_cost, lab = action_cost[a], labour_hours[a]
                end
                cost =
                    -phase_margin[e] * output +
                    act_cost +
                    spare_holding * k_after +
                    order_fixed * (q > 0) +
                    spare_cost * q
                kk = _mdp_end_pair!(b, cost, (dt, lab, Float64(k_after)), 100mcode + q)
                if mcode == mref && q == qref
                    ref[s] = kk
                end
            end
        end
    end
    mdp = _mdp_finish(b, copy(MAINTENANCE_MDP_STREAMS), ref)

    # Typical start: a young asset with some spares, start of the calendar.
    start = zeros(E * Sp)
    for c in 0:max(1, C ÷ 3), k in min(1, K):K
        start[idx(c, k, 1)] = 1.0
    end
    rhs, N = _mdp_rhs(criterion, discount, start)
    budget_streams = has_service_row ? [1] : Int[]
    budgets, witness, certificate = _mdp_plant_budgets(
        rng, mdp, criterion, discount, rhs, N, feasibility_status, budget_streams
    )

    return MachineMaintenanceMDP(
        C,
        K,
        Q,
        E,
        speeds,
        wear_rate,
        wear_exponent,
        acceleration,
        shock_prob,
        repair_depth,
        phase_load,
        phase_margin,
        efficiency_loss,
        downtime,
        action_cost,
        labour_hours,
        spare_holding,
        order_fixed,
        spare_cost,
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
    :machine_maintenance,
    MachineMaintenanceMDP,
    "Occupation-measure LP of condition-based maintenance with speed control, imperfect repair, shock failures, a spare-parts stock, and a seasonal production calendar, with an optional availability (downtime) row refuted by a value-function Farkas certificate";
    tags=[:markov],
    max_target_variables=1_000_000,
)
