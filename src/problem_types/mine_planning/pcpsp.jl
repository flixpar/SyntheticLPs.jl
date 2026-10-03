using JuMP
using Random
using Distributions

"""
    MinePCPSPProblem <: ProblemGenerator

Precedence-constrained production scheduling with destinations (PCPSP, the
richest MineLib class): blocks are scheduled as in CPIT, but where each mined
block goes is itself a decision. A copper porphyry with an oxide cap feeds two
plants — a flotation mill (sulfide-friendly, concentrate sold subject to an
arsenic penalty limit) and a heap leach with SX-EW (oxide only) — or the waste
dump.

# Formulation

  - `x[b, t] in {0, 1}`: block `b` mined by the end of period `t` (cumulative
    "by" variables; relaxed under the default `relax_integer=true`)
  - `y[j, t] >= 0`: fraction of block `pair_block[j]` sent in period `t` to
    plant `pair_dest[j]` (1 = mill, 2 = heap leach); a block has a pair only for
    plants it is eligible for (mill: grade at least half the milling breakeven;
    leach: oxide at least half the leaching breakeven). The waste dump is
    implicit (whatever is mined and not sent to a plant).

Rows per period `t` (plus the chain and precedence rows of every variant):

  - linking: `sum_{j of b} y[j, t] <= x[b, t] - x[b, t-1]` for every block with
    a pair
  - mining capacity: `sum_b w_b (x[b,t] - x[b,t-1]) <= mining_capacity[t]`
  - mill feed: `min_mill_feed[t] <= sum_{j mill} w y[j,t] <= mill_capacity[t]`
  - leach: `sum_{j leach} w y[j,t] <= leach_capacity[t]`
  - head grade: `sum_{j mill} w (g - head_grade_min) y[j,t] >= 0`
  - arsenic: `sum_{j mill} w (a - arsenic_max) / 1000 y[j,t] <= 0` (t As)
  - concentrate and cathode capacities: `sum_{j mill} metal_j y[j,t] <=
    mill_metal_capacity[t]`, same for the leach, metal in t Cu
  - objective: discounted plant margins minus discounted mining cost.

The blending rows are ratio constraints linearised by multiplying through by
the feed — non-unimodular coupling that survives the LP relaxation.

# Feasibility control

  - `feasible`: a greedy whole-block schedule (nested-shell order, each block
    to its most valuable plant with room, mining stops for the period when the
    mill is full) within capacities shrunk by 3-8%; minimum feed, head-grade
    and arsenic specs are the natural ones clipped to the plan with 5-12%
    margins. Stored as a [`MinePlanWitness`](@ref) (`destination` 1 = mill,
    2 = leach, 0 = dump).
  - `infeasible`: a [`MineClosureCertificate`](@ref) in one of two modes,
    drawn 50/50: `:ramp_up` (start-up mill feed above the closure bound on
    reachable mill-eligible tonnage, with a reduced start-up fleet when
    needed; `:exhaustion` as last resort), or `:head_grade` (minimum feed in
    the first one or two periods at a head grade the reachable ore cannot
    sustain: the head-grade spec is raised to `tau + (1 + margin) * bound /
    feed` for a threshold grade `tau`, but kept below 80% of the richest
    eligible grade, so no row is trivially contradictory).
  - `unknown`: natural specs (head grade 70-110% of the average ore grade,
    arsenic limit 100-180% of the average ore arsenic, feed contract at 60-90%
    of mill capacity), no claim.

# Sizing

Variables are `n_periods * (n_blocks + n_pairs)`; the pit is the prefix of
the block order whose count is closest to the target (within about
`1.5 * n_periods`).

# Fields

  - `blocks`, `economics`, `n_periods`
  - `pair_block`, `pair_dest`: eligible (block, plant) pairs
  - `pair_value`: undiscounted plant margin of the whole block, k\$
  - `pair_metal`: recovered copper of the whole block, t
  - `mining_capacity`, `mill_capacity`, `min_mill_feed`, `leach_capacity`,
    `mill_metal_capacity`, `leach_metal_capacity`: per period
  - `head_grade_min` (% Cu), `arsenic_max` (ppm)
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct MinePCPSPProblem <: ProblemGenerator
    blocks::MineBlockModel
    economics::MineEconomics
    n_periods::Int
    pair_block::Vector{Int}
    pair_dest::Vector{Int}
    pair_value::Vector{Float64}
    pair_metal::Vector{Float64}
    mining_capacity::Vector{Float64}
    mill_capacity::Vector{Float64}
    min_mill_feed::Vector{Float64}
    leach_capacity::Vector{Float64}
    mill_metal_capacity::Vector{Float64}
    leach_metal_capacity::Vector{Float64}
    head_grade_min::Float64
    arsenic_max::Float64
    feasible_witness::Union{Nothing, MinePlanWitness}
    infeasibility_certificate::Union{Nothing, MineClosureCertificate}
    feasibility_status::FeasibilityStatus
end

"""
Mill and leach eligibility of every block (marginal material at half the
breakeven grade is eligible, so the LP can blend it).
"""
function _mine_plant_eligibility(bm::MineBlockModel, econ::MineEconomics)
    B = length(bm)
    mill = [bm.grade[b] >= 0.5 * _mine_mill_cutoff(econ, bm.oxide[b]) for b in 1:B]
    leach = [bm.oxide[b] && bm.grade[b] >= 0.5 * _mine_leach_cutoff(econ) for b in 1:B]
    return mill, leach
end

"""
    _mine_head_grade_infeasibility!(rng, bm, eligible, mining_capacity,
                                    feed_capacity, min_feed, grade_min, margin)
        -> (head_grade_min, certificate) or nothing

`:head_grade` mode: over the first `k in 1:2` periods the mill must receive at
least 60-85% of its capacity at a head grade of at least `head_grade_min`.
For thresholds `tau` at the 40/60/80% tonnage quantiles of the eligible
grades (only the 60% quantile on pits above 20,000 blocks, to bound the
max-flow work), the closure bound `U(tau)` on `sum_b w_b (g_b - tau)^+ x[b, k]`
(mining budget of periods `1..k`) gives the smallest provably infeasible spec
`tau + (1 + margin) U(tau) / feed`. Returns `nothing` (leaving the vectors
untouched) when the resulting spec would reach 80% of the richest eligible
grade.
"""
function _mine_head_grade_infeasibility!(
    rng::AbstractRNG,
    bm::MineBlockModel,
    eligible::Vector{Bool},
    mining_capacity::Vector{Float64},
    feed_capacity::Vector{Float64},
    min_feed::Vector{Float64},
    grade_min::Float64,
    margin::Float64,
)
    T = length(mining_capacity)
    k = rand(rng, 1:min(2, T - 1))
    theta = rand(rng, Uniform(0.6, 0.85))
    level = [max(min_feed[t], theta * feed_capacity[t]) for t in 1:k]
    feed = sum(level)
    budget = sum(mining_capacity[1:k])
    elig = [b for b in eachindex(eligible) if eligible[b]]
    isempty(elig) && return nothing
    gmax = maximum(bm.grade[b] for b in elig)
    sorted = sort(elig; by=b -> bm.grade[b])
    cumw = cumsum(bm.tonnage[sorted])
    best = nothing
    for qtl in (length(bm) > 20_000 ? (0.6,) : (0.4, 0.6, 0.8))
        tau = bm.grade[sorted[searchsortedfirst(cumw, qtl * cumw[end])]]
        weights = [eligible[b] ? bm.tonnage[b] * max(bm.grade[b] - tau, 0.0) : 0.0 for b in eachindex(eligible)]
        closure = _mine_closure_bound(bm, weights, budget)
        spec = tau + (1 + margin) * closure[1] / feed
        if best === nothing || spec < best[1]
            best = (spec, tau, weights, closure)
        end
    end
    spec, tau, weights, closure = best
    head_grade_min = max(grade_min, spec)
    head_grade_min <= 0.8 * gmax || return nothing
    min_feed[1:k] .= level
    mu = head_grade_min - tau
    cert = _mine_certificate(:head_grade, k, budget, weights, tau, mu, closure, mu * feed)
    return head_grade_min, cert
end

"""
    MinePCPSPProblem(target_variables, feasibility_status, seed)

Construct a PCPSP instance with about `target_variables` variables (at most
`MINE_PLANNING_MAX_VARIABLES`).
"""
function MinePCPSPProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 1 || throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= MINE_PLANNING_MAX_VARIABLES || throw(
        ArgumentError(
            "mine_planning/pcpsp supports at most $MINE_PLANNING_MAX_VARIABLES variables; " *
            "requested $target_variables.",
        ),
    )
    rng = MersenneTwister(seed)
    T = _mine_horizon(rng, target_variables)
    econ = _mine_economics(rng)
    bm0 = _mine_block_model(rng, max(4, cld(target_variables, T)), econ)
    mill0, leach0 = _mine_plant_eligibility(bm0, econ)
    K = _mine_prefix_size([1 + mill0[b] + leach0[b] for b in 1:length(bm0)], T, 0, target_variables)
    bm = _mine_truncate(bm0, K)
    B = K
    mill_ok, leach_ok = mill0[1:B], leach0[1:B]
    w, g = bm.tonnage, bm.grade

    pair_block, pair_dest = Int[], Int[]
    for b in 1:B
        mill_ok[b] && (push!(pair_block, b); push!(pair_dest, 1))
        leach_ok[b] && (push!(pair_block, b); push!(pair_dest, 2))
    end
    mill_rec = [_mine_mill_recovery(econ, bm.oxide[b]) for b in 1:B]
    recovery(j) = pair_dest[j] == 1 ? mill_rec[pair_block[j]] : econ.leach_recovery
    cost(j) = pair_dest[j] == 1 ? econ.mill_cost : econ.leach_cost
    pair_value = [
        w[pair_block[j]] * (g[pair_block[j]] / 100 * recovery(j) * econ.price - cost(j)) for
        j in eachindex(pair_block)
    ]
    pair_metal = [10.0 * w[pair_block[j]] * g[pair_block[j]] * recovery(j) for j in eachindex(pair_block)]

    # Preferred plant of each block in a whole-block plan: the most valuable
    # profitable one (0 = dump).
    best_pair = zeros(Int, B)
    for j in eachindex(pair_block)
        b = pair_block[j]
        pair_value[j] > 0 || continue
        if best_pair[b] == 0 || pair_value[j] > pair_value[best_pair[b]]
            best_pair[b] = j
        end
    end
    mill_ore = [b for b in 1:B if best_pair[b] > 0 && pair_dest[best_pair[b]] == 1]
    leach_ore = [b for b in 1:B if best_pair[b] > 0 && pair_dest[best_pair[b]] == 2]

    # Natural capacities.
    total = sum(w)
    wmax = maximum(w)
    rho = rand(rng, Uniform(0.8, 1.25))
    M = max(total / (T * rho), 3.0 * wmax)
    mill_t = sum(w[mill_ore]; init=0.0)
    leach_t = sum(w[leach_ore]; init=0.0)
    C = max(rand(rng, Uniform(1.1, 1.6)) * M * mill_t / total, 2.5 * wmax)
    L = max(rand(rng, Uniform(1.0, 1.5)) * M * leach_t / total, 2.5 * wmax)
    mining_capacity = fill(M, T)
    mill_capacity = fill(C, T)
    _mine_ramp_up!(rng, mining_capacity, mill_capacity)
    leach_capacity = fill(L, T)
    avg_mill_grade = isempty(mill_ore) ? maximum(g) : sum(w[b] * g[b] for b in mill_ore) / mill_t
    avg_leach_grade = isempty(leach_ore) ? maximum(g) : sum(w[b] * g[b] for b in leach_ore) / leach_t
    mill_metal_capacity =
        rand(rng, Uniform(0.95, 1.3)) * 10.0 * econ.mill_recovery_sulfide * avg_mill_grade .* mill_capacity
    leach_metal_capacity = rand(rng, Uniform(0.95, 1.3)) * 10.0 * econ.leach_recovery * avg_leach_grade .* leach_capacity
    mill_metal_capacity .= max.(mill_metal_capacity, 3.0 * maximum(pair_metal; init=1.0))
    leach_metal_capacity .= max.(leach_metal_capacity, 3.0 * maximum(pair_metal; init=1.0))

    # Natural specs and mill-feed contract.
    head_grade_min = rand(rng, Uniform(0.7, 1.1)) * avg_mill_grade
    avg_as = isempty(mill_ore) ? maximum(bm.contaminant) : sum(w[b] * bm.contaminant[b] for b in mill_ore) / mill_t
    arsenic_max = rand(rng, Uniform(1.0, 1.8)) * avg_as
    min_mill_feed = _mine_feed_contract(rng, mill_capacity, sum(w[b] for b in 1:B if mill_ok[b]))

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        eps_cap = rand(rng, Uniform(0.03, 0.08))
        eps_spec = rand(rng, Uniform(0.05, 0.12))
        mining_period = zeros(Int, B)
        destination = zeros(Int, B)
        feed = zeros(T)
        feed_grade = zeros(T)     # sum w g
        feed_as = zeros(T)        # sum w a
        nxt = 1
        for t in 1:T
            mined = mill_metal = leach_tons = leach_metal = 0.0
            while nxt <= B
                b = nxt
                mined + w[b] <= mining_capacity[t] / (1 + eps_cap) || break
                j = best_pair[b]
                dest = 0
                if j > 0 && pair_dest[j] == 1
                    if feed[t] + w[b] <= mill_capacity[t] / (1 + eps_cap) &&
                       mill_metal + pair_metal[j] <= mill_metal_capacity[t] / (1 + eps_cap)
                        dest = 1
                    else
                        break   # mill full: stop mining for the period
                    end
                elseif j > 0 && pair_dest[j] == 2
                    if leach_tons + w[b] <= leach_capacity[t] / (1 + eps_cap) &&
                       leach_metal + pair_metal[j] <= leach_metal_capacity[t] / (1 + eps_cap)
                        dest = 2
                    end
                end
                if dest == 1
                    feed[t] += w[b]
                    feed_grade[t] += w[b] * g[b]
                    feed_as[t] += w[b] * bm.contaminant[b]
                    mill_metal += pair_metal[j]
                elseif dest == 2
                    leach_tons += w[b]
                    leach_metal += pair_metal[j]
                end
                mined += w[b]
                mining_period[b] = t
                destination[b] = dest
                nxt += 1
            end
        end
        fed = [t for t in 1:T if feed[t] > 0]
        if !isempty(fed)
            head_grade_min = min(head_grade_min, (1 - eps_spec) * minimum(feed_grade[t] / feed[t] for t in fed))
            arsenic_max = max(arsenic_max, (1 + eps_spec) * maximum(feed_as[t] / feed[t] for t in fed))
        end
        min_mill_feed = [min(min_mill_feed[t], (1 - eps_spec) * feed[t]) for t in 1:T]
        witness = MinePlanWitness(mining_period, destination, zeros(0, T), zeros(0, T))
    elseif feasibility_status == infeasible
        margin = rand(rng, Uniform(0.1, 0.35))
        result = nothing
        if rand(rng) < 0.5
            result = _mine_head_grade_infeasibility!(
                rng, bm, mill_ok, mining_capacity, mill_capacity, min_mill_feed, head_grade_min, margin
            )
        end
        if result === nothing
            weights = [mill_ok[b] ? w[b] : 0.0 for b in 1:B]
            certificate = _mine_feed_infeasibility!(
                rng, bm, weights, mining_capacity, mill_capacity, min_mill_feed, margin
            )
        else
            head_grade_min, certificate = result
        end
    else
        weights = [mill_ok[b] ? w[b] : 0.0 for b in 1:B]
        _mine_unknown_startup!(rng, bm, weights, mining_capacity, mill_capacity, min_mill_feed)
    end

    return MinePCPSPProblem(
        bm,
        econ,
        T,
        pair_block,
        pair_dest,
        pair_value,
        pair_metal,
        mining_capacity,
        mill_capacity,
        min_mill_feed,
        leach_capacity,
        mill_metal_capacity,
        leach_metal_capacity,
        head_grade_min,
        arsenic_max,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::MinePCPSPProblem)

Build the PCPSP model; deterministic. Variables `x[b, t]` (binary, cumulative
extraction) and `y[j, t] >= 0` (plant destination fractions).
"""
function build_model(prob::MinePCPSPProblem)
    bm = prob.blocks
    B, T = length(bm), prob.n_periods
    w, g = bm.tonnage, bm.grade
    P = length(prob.pair_block)
    delta = _mine_discount(prob.economics.discount_rate, T)
    model = Model()
    @variable(model, x[1:B, 1:T], Bin)
    @variable(model, y[1:P, 1:T] >= 0)

    _mine_add_chain_and_precedence!(model, x, bm, T)
    pairs_of = [Int[] for _ in 1:B]
    for j in 1:P
        push!(pairs_of[prob.pair_block[j]], j)
    end
    mill = [j for j in 1:P if prob.pair_dest[j] == 1]
    leach = [j for j in 1:P if prob.pair_dest[j] == 2]
    for t in 1:T
        for b in 1:B
            isempty(pairs_of[b]) && continue
            expr = AffExpr(0.0)
            for j in pairs_of[b]
                add_to_expression!(expr, 1.0, y[j, t])
            end
            add_to_expression!(expr, -1.0, x[b, t])
            t > 1 && add_to_expression!(expr, 1.0, x[b, t - 1])
            @constraint(model, expr <= 0)
        end
        @constraint(model, _mine_period_tonnage(x, w, 1:B, t) <= prob.mining_capacity[t])
        if !isempty(mill)
            feed, grade_row, arsenic, metal = AffExpr(0.0), AffExpr(0.0), AffExpr(0.0), AffExpr(0.0)
            for j in mill
                b = prob.pair_block[j]
                add_to_expression!(feed, w[b], y[j, t])
                add_to_expression!(grade_row, w[b] * (g[b] - prob.head_grade_min), y[j, t])
                add_to_expression!(arsenic, w[b] * (bm.contaminant[b] - prob.arsenic_max) / 1000, y[j, t])
                add_to_expression!(metal, prob.pair_metal[j], y[j, t])
            end
            _mine_add_feed_row!(model, feed, prob.min_mill_feed[t], prob.mill_capacity[t])
            @constraint(model, grade_row >= 0)
            @constraint(model, arsenic <= 0)
            @constraint(model, metal <= prob.mill_metal_capacity[t])
        end
        if !isempty(leach)
            tons, metal = AffExpr(0.0), AffExpr(0.0)
            for j in leach
                add_to_expression!(tons, w[prob.pair_block[j]], y[j, t])
                add_to_expression!(metal, prob.pair_metal[j], y[j, t])
            end
            @constraint(model, tons <= prob.leach_capacity[t])
            @constraint(model, metal <= prob.leach_metal_capacity[t])
        end
    end

    obj = AffExpr(0.0)
    for t in 1:T
        for j in 1:P
            add_to_expression!(obj, delta[t] * prob.pair_value[j], y[j, t])
        end
        for b in 1:B
            add_to_expression!(obj, _mine_by_coefficient(-bm.mining_cost[b], delta, t), x[b, t])
        end
    end
    @objective(model, Max, obj)
    return model
end

register_variant(
    :mine_planning,
    :pcpsp,
    MinePCPSPProblem,
    "Precedence-constrained production scheduling with destinations (MineLib PCPSP): cumulative block extraction plus mill / heap-leach / dump split variables, head-grade and arsenic blending rows, mill and leach tonnage and metal capacities, minimum mill feed, maximizing NPV",
)
