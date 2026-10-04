using JuMP
using Random
using Distributions

"""
    MineStockpileProblem <: ProblemGenerator

Open-pit production scheduling with grade-binned stockpiles, following the
linear stockpile models of Moreno, Rezakhah, Newman & Ferreira ("Linear models
for stockpiling in open-pit mine production scheduling problems", EJOR 2017).
Sulfide ore above an economic threshold can either go straight to the mill or
onto one of `S` stockpiles, each holding a grade range `[lo_s, hi_s)`; the
stockpiles carry inventory across periods and are reclaimed to the mill later
(typically while pre-stripping, or once the pit is exhausted). Reclaimed ore is
valued and blended at its bin's lower grade edge `lo_s`, a conservative linear
approximation that never credits metal the pile does not contain.

# Formulation

  - `x[b, t] in {0, 1}`: block `b` mined by the end of period `t` (cumulative)
  - `y[j, t] >= 0`: fraction of block `pair_block[j]` sent to its destination
    `pair_dest[j]` (1 = mill direct, 2 = stockpile bin `pair_bin[j]`)
  - `r[s, t] >= 0`: tonnage reclaimed from bin `s` to the mill in period `t`
  - `0 <= inv[s, t] <= stockpile_capacity[s]`: end-of-period inventory, kt

Rows per period `t` (plus chain and precedence):

  - linking: `sum_{j of b} y[j, t] <= x[b, t] - x[b, t-1]`
  - mining capacity
  - mill feed: `min_mill_feed[t] <= sum_{j mill} w y + sum_s r[s,t] <= mill_capacity[t]`
  - head grade: `sum_{j mill} w (g - head_grade_min) y + sum_s (lo_s - head_grade_min) r[s,t] >= 0`
  - concentrate capacity: recovered copper of direct feed plus reclaim (at
    `lo_s`) within `mill_metal_capacity[t]`
  - rehandling fleet: `sum_s r[s,t] <= reclaim_capacity[t]`
  - inventory balance: `inv[s,t] = inv[s,t-1] + sum_{j in bin s} w y[j,t] - r[s,t]`
    (`inv[s,0] = 0`)
  - objective: discounted mill margin of direct feed and reclaim (minus
    rehandling), minus stockpile placement and mining costs.

# Feasibility control

  - `feasible`: greedy whole-block schedule — ore to the mill while it has
    room, stockpile-grade ore that does not fit (and marginal stockpile-grade
    ore) to its bin while the bin has room, mining stops for the period when
    high-grade ore no longer fits; spare mill capacity is then filled by
    reclaiming the richest bins first. Capacities shrunk by 3-8%; feed and
    head-grade specs clipped to the plan with 5-12% margins. The plan, with
    per-bin reclaim and inventory, is the [`MinePlanWitness`](@ref).
  - `infeasible`: as `pcpsp` (`:ramp_up` / `:exhaustion` / `:head_grade`
    [`MineClosureCertificate`](@ref)); stockpiles only defer ore (reclaim never
    exceeds what was stockpiled from mined blocks, and the initial inventory is
    zero), so the closure bound covers direct feed plus reclaim.
  - `unknown`: natural specs and contract as in `pcpsp` (start-up feed at
    60-120% of the closure bound on reachable mill feed), no claim.

# Sizing

Variables are `n_periods * (n_blocks + n_pairs + 2 * n_bins)`; the pit is the
prefix of the block order whose count is closest to the target.

# Fields

  - `blocks`, `economics`, `n_periods`, `n_bins`
  - `bin_lower`, `bin_upper`: grade range of each bin, % Cu
  - `pair_block`, `pair_dest`, `pair_bin`: eligible (block, destination) pairs
  - `pair_value`: undiscounted value of sending the whole block, k\$ (mill
    margin, or minus the placement cost for a stockpile)
  - `pair_metal`: recovered copper of the whole block at the mill, t (0 for
    stockpile pairs)
  - `reclaim_value`, `reclaim_metal`: per kt reclaimed from each bin, k\$ and t
  - `rehandle_cost`, `placement_cost`: \$/t
  - `mining_capacity`, `mill_capacity`, `min_mill_feed`, `mill_metal_capacity`,
    `reclaim_capacity`: per period; `stockpile_capacity`: per bin
  - `head_grade_min`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct MineStockpileProblem <: ProblemGenerator
    blocks::MineBlockModel
    economics::MineEconomics
    n_periods::Int
    n_bins::Int
    bin_lower::Vector{Float64}
    bin_upper::Vector{Float64}
    pair_block::Vector{Int}
    pair_dest::Vector{Int}
    pair_bin::Vector{Int}
    pair_value::Vector{Float64}
    pair_metal::Vector{Float64}
    reclaim_value::Vector{Float64}
    reclaim_metal::Vector{Float64}
    rehandle_cost::Float64
    placement_cost::Float64
    mining_capacity::Vector{Float64}
    mill_capacity::Vector{Float64}
    min_mill_feed::Vector{Float64}
    mill_metal_capacity::Vector{Float64}
    reclaim_capacity::Vector{Float64}
    stockpile_capacity::Vector{Float64}
    head_grade_min::Float64
    feasible_witness::Union{Nothing, MinePlanWitness}
    infeasibility_certificate::Union{Nothing, MineClosureCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    MineStockpileProblem(target_variables, feasibility_status, seed)

Construct a stockpiling instance with about `target_variables` variables (at
most `MINE_PLANNING_MAX_VARIABLES`).
"""
function MineStockpileProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    target_variables <= MINE_PLANNING_MAX_VARIABLES || throw(
        ArgumentError(
            "mine_planning/stockpile supports at most $MINE_PLANNING_MAX_VARIABLES variables; " *
            "requested $target_variables.",
        ),
    )
    rng = MersenneTwister(seed)
    T = _mine_horizon(rng, target_variables)
    S = rand(rng, 2:3)
    econ = _mine_economics(rng)
    rehandle_cost = rand(rng, Uniform(0.5, 1.5))
    placement_cost = rand(rng, Uniform(0.2, 0.6))

    # Bins: geometric grade ranges from the stockpiling threshold (ore that
    # still pays for rehandling at its bin's lower edge) up to a high-grade
    # limit; richer ore always goes direct.
    sulfide_rec = econ.mill_recovery_sulfide
    g_lo = 1.05 * 100.0 * (econ.mill_cost + rehandle_cost) / (econ.price * sulfide_rec)
    g_hi = g_lo * rand(rng, Uniform(2.0, 3.0))
    edges = [g_lo * (g_hi / g_lo)^(s / S) for s in 0:S]
    bin_lower, bin_upper = edges[1:S], edges[2:(S + 1)]
    bin_of(bm, b) =
        (!bm.oxide[b] && g_lo <= bm.grade[b] < g_hi) ? searchsortedlast(edges, bm.grade[b]) : 0

    offset = 2 * S * T
    bm0 = _mine_block_model(rng, max(4, cld(max(target_variables - offset, 4), T)), econ)
    mill0, _ = _mine_plant_eligibility(bm0, econ)
    counts = [1 + mill0[b] + (bin_of(bm0, b) > 0) for b in 1:length(bm0)]
    K = _mine_prefix_size(counts, T, offset, target_variables)
    bm = _mine_truncate(bm0, K)
    B = K
    mill_ok = mill0[1:B]
    w, g = bm.tonnage, bm.grade

    pair_block, pair_dest, pair_bin = Int[], Int[], Int[]
    for b in 1:B
        mill_ok[b] && (push!(pair_block, b); push!(pair_dest, 1); push!(pair_bin, 0))
        s = bin_of(bm, b)
        s > 0 && (push!(pair_block, b); push!(pair_dest, 2); push!(pair_bin, s))
    end
    mill_rec = [_mine_mill_recovery(econ, bm.oxide[b]) for b in 1:B]
    pair_value = [
        if pair_dest[j] == 1
            w[pair_block[j]] *
            (g[pair_block[j]] / 100 * mill_rec[pair_block[j]] * econ.price - econ.mill_cost)
        else
            -w[pair_block[j]] * placement_cost
        end for j in eachindex(pair_block)
    ]
    pair_metal = [
        if pair_dest[j] == 1
            10.0 * w[pair_block[j]] * g[pair_block[j]] * mill_rec[pair_block[j]]
        else
            0.0
        end for j in eachindex(pair_block)
    ]
    reclaim_value = [
        bin_lower[s] / 100 * sulfide_rec * econ.price - econ.mill_cost - rehandle_cost for s in 1:S
    ]
    reclaim_metal = [10.0 * bin_lower[s] * sulfide_rec for s in 1:S]

    mill_pair = zeros(Int, B)
    sp_pair = zeros(Int, B)
    for j in eachindex(pair_block)
        pair_dest[j] == 1 ? (mill_pair[pair_block[j]] = j) : (sp_pair[pair_block[j]] = j)
    end
    profitable = [mill_pair[b] > 0 && pair_value[mill_pair[b]] > 0 for b in 1:B]
    mill_ore = [b for b in 1:B if profitable[b]]

    # Natural capacities.
    total = sum(w)
    wmax = maximum(w)
    rho = rand(rng, Uniform(0.8, 1.25))
    M = max(total / (T * rho), 3.0 * wmax)
    mill_t = sum(w[mill_ore]; init=0.0)
    C = max(rand(rng, Uniform(1.0, 1.5)) * M * mill_t / total, 2.5 * wmax)
    mining_capacity = fill(M, T)
    mill_capacity = fill(C, T)
    _mine_ramp_up!(rng, mining_capacity, mill_capacity)
    avg_mill_grade = isempty(mill_ore) ? maximum(g) : sum(w[b] * g[b] for b in mill_ore) / mill_t
    mill_metal_capacity =
        rand(rng, Uniform(0.95, 1.3)) * 10.0 * sulfide_rec * avg_mill_grade .* mill_capacity
    mill_metal_capacity .= max.(mill_metal_capacity, 3.0 * maximum(pair_metal; init=1.0))
    reclaim_capacity = fill(rand(rng, Uniform(0.3, 0.6)) * C, T)
    stockpile_capacity = [rand(rng, Uniform(0.6, 1.5)) * C for _ in 1:S]

    head_grade_min = rand(rng, Uniform(0.7, 1.1)) * avg_mill_grade
    min_mill_feed = _mine_feed_contract(rng, mill_capacity, sum(w[b] for b in 1:B if mill_ok[b]))

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        eps_cap = rand(rng, Uniform(0.03, 0.08))
        eps_spec = rand(rng, Uniform(0.05, 0.12))
        shrink = 1 / (1 + eps_cap)
        mining_period = zeros(Int, B)
        destination = zeros(Int, B)
        reclaim = zeros(S, T)
        inventory = zeros(S, T)
        feed = zeros(T)
        feed_grade = zeros(T)
        nxt = 1
        for t in 1:T
            mined = metal = 0.0
            stock = t > 1 ? inventory[:, t - 1] : zeros(S)
            while nxt <= B
                b = nxt
                mined + w[b] <= shrink * mining_capacity[t] || break
                s = sp_pair[b] > 0 ? pair_bin[sp_pair[b]] : 0
                room = s > 0 && stock[s] + w[b] <= shrink * stockpile_capacity[s]
                dest = 0
                if profitable[b]
                    j = mill_pair[b]
                    if feed[t] + w[b] <= shrink * mill_capacity[t] &&
                        metal + pair_metal[j] <= shrink * mill_metal_capacity[t]
                        dest = 1
                        feed[t] += w[b]
                        feed_grade[t] += w[b] * g[b]
                        metal += pair_metal[j]
                    elseif room
                        dest = 2
                    else
                        break   # mill and bin full: stop mining for the period
                    end
                elseif room
                    dest = 2
                end
                dest == 2 && (stock[s] += w[b])
                mined += w[b]
                mining_period[b] = t
                destination[b] = dest
                nxt += 1
            end
            # Fill spare mill capacity from the richest bins.
            reclaimed = 0.0
            for s in S:-1:1
                reclaim_value[s] > 0 || continue
                amount = min(
                    stock[s],
                    shrink * mill_capacity[t] - feed[t],
                    shrink * reclaim_capacity[t] - reclaimed,
                    (shrink * mill_metal_capacity[t] - metal) / reclaim_metal[s],
                )
                amount > 0 || continue
                reclaim[s, t] = amount
                stock[s] -= amount
                reclaimed += amount
                feed[t] += amount
                feed_grade[t] += amount * bin_lower[s]
                metal += amount * reclaim_metal[s]
            end
            inventory[:, t] .= stock
        end
        fed = [t for t in 1:T if feed[t] > 0]
        if !isempty(fed)
            head_grade_min = min(
                head_grade_min, (1 - eps_spec) * minimum(feed_grade[t] / feed[t] for t in fed)
            )
        end
        min_mill_feed = [min(min_mill_feed[t], (1 - eps_spec) * feed[t]) for t in 1:T]
        witness = MinePlanWitness(mining_period, destination, reclaim, inventory)
    elseif feasibility_status == infeasible
        margin = rand(rng, Uniform(0.1, 0.35))
        result = nothing
        if rand(rng) < 0.5
            result = _mine_head_grade_infeasibility!(
                rng,
                bm,
                mill_ok,
                mining_capacity,
                mill_capacity,
                min_mill_feed,
                head_grade_min,
                margin,
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

    return MineStockpileProblem(
        bm,
        econ,
        T,
        S,
        bin_lower,
        bin_upper,
        pair_block,
        pair_dest,
        pair_bin,
        pair_value,
        pair_metal,
        reclaim_value,
        reclaim_metal,
        rehandle_cost,
        placement_cost,
        mining_capacity,
        mill_capacity,
        min_mill_feed,
        mill_metal_capacity,
        reclaim_capacity,
        stockpile_capacity,
        head_grade_min,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::MineStockpileProblem)

Build the stockpiling model; deterministic. Variables `x[b, t]` (binary),
`y[j, t]`, `r[s, t]`, `inv[s, t]` (continuous).
"""
function build_model(prob::MineStockpileProblem)
    bm = prob.blocks
    B, T, S = length(bm), prob.n_periods, prob.n_bins
    w, g = bm.tonnage, bm.grade
    P = length(prob.pair_block)
    delta = _mine_discount(prob.economics.discount_rate, T)
    model = Model()
    @variable(model, x[1:B, 1:T], Bin)
    @variable(model, y[1:P, 1:T] >= 0)
    @variable(model, r[1:S, 1:T] >= 0)
    @variable(model, 0 <= inv[s = 1:S, 1:T] <= prob.stockpile_capacity[s])

    _mine_add_chain_and_precedence!(model, x, bm, T)
    pairs_of = [Int[] for _ in 1:B]
    for j in 1:P
        push!(pairs_of[prob.pair_block[j]], j)
    end
    mill = [j for j in 1:P if prob.pair_dest[j] == 1]
    in_bin = [[j for j in 1:P if prob.pair_bin[j] == s] for s in 1:S]
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

        feed = AffExpr(0.0)
        grade_row = AffExpr(0.0)
        metal = AffExpr(0.0)
        for j in mill
            b = prob.pair_block[j]
            add_to_expression!(feed, w[b], y[j, t])
            add_to_expression!(grade_row, w[b] * (g[b] - prob.head_grade_min), y[j, t])
            add_to_expression!(metal, prob.pair_metal[j], y[j, t])
        end
        for s in 1:S
            add_to_expression!(feed, 1.0, r[s, t])
            add_to_expression!(grade_row, prob.bin_lower[s] - prob.head_grade_min, r[s, t])
            add_to_expression!(metal, prob.reclaim_metal[s], r[s, t])
        end
        _mine_add_feed_row!(model, feed, prob.min_mill_feed[t], prob.mill_capacity[t])
        @constraint(model, grade_row >= 0)
        @constraint(model, metal <= prob.mill_metal_capacity[t])
        @constraint(model, sum(r[s, t] for s in 1:S) <= prob.reclaim_capacity[t])
        for s in 1:S
            inflow = AffExpr(0.0)
            for j in in_bin[s]
                add_to_expression!(inflow, w[prob.pair_block[j]], y[j, t])
            end
            if t > 1
                @constraint(model, inv[s, t] - inv[s, t - 1] - inflow + r[s, t] == 0)
            else
                @constraint(model, inv[s, t] - inflow + r[s, t] == 0)
            end
        end
    end

    obj = AffExpr(0.0)
    for t in 1:T
        for j in 1:P
            add_to_expression!(obj, delta[t] * prob.pair_value[j], y[j, t])
        end
        for s in 1:S
            add_to_expression!(obj, delta[t] * prob.reclaim_value[s], r[s, t])
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
    :stockpile,
    MineStockpileProblem,
    "Open-pit production scheduling with grade-binned stockpiles (linear stockpile model of Moreno et al. 2017): cumulative block extraction, direct-feed and stockpile split variables, per-bin inventory balances carried across periods, reclaim and rehandling capacities, head-grade blending, minimum mill feed, maximizing NPV";
    tags=[:mining, :staircase, :blending, :degenerate],
    max_target_variables=1_000_000,
)
