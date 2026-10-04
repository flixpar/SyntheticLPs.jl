using JuMP
using Random
using Distributions

"""
    MineCPITProblem <: ProblemGenerator

Constrained pit limit problem (CPIT, the MineLib benchmark class of Espinoza,
Goycoolea, Moreno & Newman 2013): schedule the extraction of an open-pit block
model over `T` periods to maximise discounted cash flow, subject to slope
precedence and per-period mining and milling capacities. Each block has a fixed
destination decided by its grade — ore (above the milling breakeven grade) is
milled, everything else is dumped.

# Formulation ("by" / cumulative variables, as in MineLib)

`x[b, t] in {0, 1}` is 1 if block `b` has been mined by the end of period `t`
(relaxed to `[0, 1]` under the default `relax_integer=true`, the LP that
Bienstock–Zuckerberg-type algorithms solve).

  - chain: `x[b, t-1] <= x[b, t]`
  - precedence: `x[b, t] <= x[a, t]` for every slope arc `b -> a` (a block
    needs the 5 or 9 blocks above it)
  - mining capacity: `sum_b w_b (x[b,t] - x[b,t-1]) <= mining_capacity[t]`
  - milling: `min_processing[t] <= sum_{b ore} w_b (x[b,t] - x[b,t-1]) <=
    processing_capacity[t]` (the lower side only where positive)
  - objective: `max sum_t delta_t sum_b v_b (x[b,t] - x[b,t-1])`, written in the
    telescoped "by" form, with `v_b` the undiscounted block value

The minimum mill feed is what makes the instance non-trivial: without it,
mining nothing is feasible. It models a mill-feed (offtake) contract over a
window of periods.

# Feasibility control

  - `feasible`: a whole-block schedule is planted by mining the topological
    (nested-shell) block order greedily into periods within capacities shrunk
    by 3-8%; minimum feeds are the natural contract clipped to 88-95% of the
    plan's ore in each period. The integral plan is stored as a
    [`MinePlanWitness`](@ref).
  - `infeasible`: the minimum feed of the first `k` periods is raised above a
    planted upper bound on the ore reachable with `k` periods of mining
    capacity — a [`MineClosureCertificate`](@ref) (Lagrangian max-closure flow
    bound, 10-35% margin) that combines precedence, chain, capacity and feed
    rows, so no single row exposes it. `k` is 1-3 (`:ramp_up`: the ore is
    buried under too much overburden for the contracted start-up fleet; the
    fleet may be reduced to no less than 30% of steady state, or mill and
    contract enlarged together by at most 1.6x) or the whole horizon
    (`:exhaustion`, 20% of the time and as fallback).
  - `unknown`: the natural contract (60-125% of the ore reserve spread over a
    window of periods, capped at 50-90% of mill capacity), with the start-up
    periods drawn at 60-120% of the closure bound on reachable ore — a genuine
    two-sided boundary (pre-strip and deposit life decide), no planted claim.

# Sizing

Variables are exactly `n_blocks * n_periods`; `n_periods = _mine_horizon(target)`
and `n_blocks = max(4, round(target / n_periods))`, so the count is within
`n_periods / 2` of the target. Rows: `n_blocks * (n_periods - 1)` chain rows,
`n_arcs * n_periods` precedence rows (5-9 arcs per block below the surface),
and 2 capacity rows per period.

# Fields

  - `blocks::MineBlockModel`, `economics::MineEconomics`, `n_periods::Int`
  - `is_ore::Vector{Bool}`: fixed destination (true = mill)
  - `block_value::Vector{Float64}`: undiscounted value, k\$ (ore: revenue minus
    milling and mining cost; waste: minus mining cost)
  - `mining_capacity`, `processing_capacity`, `min_processing`: per period, kt
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct MineCPITProblem <: ProblemGenerator
    blocks::MineBlockModel
    economics::MineEconomics
    n_periods::Int
    is_ore::Vector{Bool}
    block_value::Vector{Float64}
    mining_capacity::Vector{Float64}
    processing_capacity::Vector{Float64}
    min_processing::Vector{Float64}
    feasible_witness::Union{Nothing, MinePlanWitness}
    infeasibility_certificate::Union{Nothing, MineClosureCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    MineCPITProblem(target_variables, feasibility_status, seed)

Construct a CPIT instance with about `target_variables` schedule variables
(at most `MINE_PLANNING_MAX_VARIABLES`).
"""
function MineCPITProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= MINE_PLANNING_MAX_VARIABLES || throw(
        ArgumentError(
            "mine_planning/cpit supports at most $MINE_PLANNING_MAX_VARIABLES variables; " *
            "requested $target_variables.",
        ),
    )
    rng = MersenneTwister(seed)
    T = _mine_horizon(rng, target_variables)
    K = max(4, round(Int, target_variables / T))
    econ = _mine_economics(rng)
    bm = _mine_block_model(rng, K, econ)
    B = length(bm)
    w = bm.tonnage

    # Fixed destinations and block values.
    is_ore = [bm.grade[b] > _mine_mill_cutoff(econ, bm.oxide[b]) for b in 1:B]
    block_value = [
        (
            if is_ore[b]
                w[b] * (
                    bm.grade[b] / 100 * _mine_mill_recovery(econ, bm.oxide[b]) * econ.price -
                    econ.mill_cost
                )
            else
                0.0
            end
        ) - bm.mining_cost[b] for b in 1:B
    ]

    # Natural capacities: the fleet moves the pit in about T / rho periods and
    # the mill is sized 10-60% above the average ore rate.
    total = sum(w)
    ore_total = sum(w[b] for b in 1:B if is_ore[b]; init=0.0)
    wmax = maximum(w)
    rho = rand(rng, Uniform(0.8, 1.25))
    M = max(total / (T * rho), 3.0 * wmax)
    kappa = rand(rng, Uniform(1.1, 1.6))
    P = max(kappa * M * ore_total / total, 2.5 * wmax)
    mining_capacity = fill(M, T)
    processing_capacity = fill(P, T)
    _mine_ramp_up!(rng, mining_capacity, processing_capacity)

    # Natural mill-feed contract over a window of periods.
    min_processing = _mine_feed_contract(rng, processing_capacity, ore_total)

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        eps_cap = rand(rng, Uniform(0.03, 0.08))
        eps_feed = rand(rng, Uniform(0.05, 0.12))
        mining_period = zeros(Int, B)
        ore_mined = zeros(T)
        nxt = 1
        for t in 1:T
            mined = 0.0
            while nxt <= B
                b = nxt
                mined + w[b] <= mining_capacity[t] / (1 + eps_cap) || break
                if is_ore[b]
                    ore_mined[t] + w[b] <= processing_capacity[t] / (1 + eps_cap) || break
                    ore_mined[t] += w[b]
                end
                mined += w[b]
                mining_period[b] = t
                nxt += 1
            end
        end
        min_processing = [min(min_processing[t], (1 - eps_feed) * ore_mined[t]) for t in 1:T]
        witness = MinePlanWitness(
            mining_period, [is_ore[b] ? 1 : 0 for b in 1:B], zeros(0, T), zeros(0, T)
        )
    elseif feasibility_status == infeasible
        margin = rand(rng, Uniform(0.1, 0.35))
        weights = [is_ore[b] ? w[b] : 0.0 for b in 1:B]
        certificate = _mine_feed_infeasibility!(
            rng, bm, weights, mining_capacity, processing_capacity, min_processing, margin
        )
    else
        weights = [is_ore[b] ? w[b] : 0.0 for b in 1:B]
        _mine_unknown_startup!(
            rng, bm, weights, mining_capacity, processing_capacity, min_processing
        )
    end

    return MineCPITProblem(
        bm,
        econ,
        T,
        is_ore,
        block_value,
        mining_capacity,
        processing_capacity,
        min_processing,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::MineCPITProblem)

Build the cumulative ("by") CPIT model; deterministic. Variables `x[b, t]`
(binary). See [`MineCPITProblem`](@ref) for the rows.
"""
function build_model(prob::MineCPITProblem)
    bm = prob.blocks
    B, T = length(bm), prob.n_periods
    w = bm.tonnage
    delta = _mine_discount(prob.economics.discount_rate, T)
    model = Model()
    @variable(model, x[1:B, 1:T], Bin)

    _mine_add_chain_and_precedence!(model, x, bm, T)
    ore_blocks = [b for b in 1:B if prob.is_ore[b]]
    for t in 1:T
        @constraint(model, _mine_period_tonnage(x, w, 1:B, t) <= prob.mining_capacity[t])
        _mine_add_feed_row!(
            model,
            _mine_period_tonnage(x, w, ore_blocks, t),
            prob.min_processing[t],
            prob.processing_capacity[t],
        )
    end

    obj = AffExpr(0.0)
    for t in 1:T, b in 1:B
        add_to_expression!(obj, _mine_by_coefficient(prob.block_value[b], delta, t), x[b, t])
    end
    @objective(model, Max, obj)
    return model
end

register_variant(
    :mine_planning,
    :cpit,
    MineCPITProblem,
    "Constrained pit limit (MineLib CPIT) open-pit production scheduling: cumulative block-extraction variables over a precedence-closed 3D block model with 1-5/1-9 slope precedence, per-period mining and milling capacities and a minimum mill-feed contract, maximizing NPV";
    default=true,
    tags=[:mining, :degenerate],
    max_target_variables=1_000_000,
)
