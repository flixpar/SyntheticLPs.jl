using JuMP
using Random
using Distributions
using StatsBase

"""
    MultiPeriodBlendingCertificate

Cumulative melt-capacity proof for `MultiPeriodBlendingProblem`: plant `plant`
must ship `Σ_t Σ_g demand[g, plant, t]` over the horizon with no opening
finished-goods stock, every tonne of output needs at least `1 / max_yield[g]`
tonnes of charge, and the furnaces melt at most `horizon · melt_capacity`.
`required` (minimum charge) exceeds `achievable` (melt capacity) by ≥ 8%.
"""
struct MultiPeriodBlendingCertificate
    plant::Int
    achievable::Float64
    required::Float64
end

"""
    MultiPeriodBlendingPlan

Planted operation of `MultiPeriodBlendingProblem`: `blend[k]` tonnes on blend
variable `k` (`prob.blend_vars`), `charge[g, k, t]`, `buy[i, k, t]`,
`stock[i, k, t]` (raw materials) and `finished[g, k, t]` (finished goods).
"""
struct MultiPeriodBlendingPlan
    blend::Vector{Float64}
    charge::Array{Float64, 3}
    buy::Array{Float64, 3}
    stock::Array{Float64, 3}
    finished::Array{Float64, 3}
end

"""
    MultiPeriodBlendingProblem <: ProblemGenerator

Multi-plant, multi-period alloy production planning with scrap purchasing and
inventories: each casthouse buys scrap classes, primary metal and master alloys
on a regional market whose prices and volumes move from period to period, holds
raw-material and finished-goods stock, and blends each period's alloy demand
inside the composition windows.

# Formulation

Indices: materials `i` (primary grades, master alloys and scrap classes at
class-average chemistry), plants `k` with an alloy portfolio, grades `g`,
periods `t = 1..T`. Variables (all ≥ 0):

  - `x[i, g, k, t]` tonnes of material charged to grade `g` (compatible pairs);
  - `charge[g, k, t]` melt mass of each grade;
  - `buy[i, k, t]`, `stock[i, k, t]` raw purchases and end-of-period stock;
  - `fg[g, k, t]` finished-goods stock.

Minimize purchases at period prices plus holding and melting cost. Rows:

  - charge balance `Σ_i x = charge` and element-mass composition rows
    (`Σ_i comp·x ≥ lo·charge`, `≤ hi·charge`, active ones only) per
    `(g, k, t)`;
  - finished goods `fg[t] = fg[t−1] + Σ_i yield·x − demand[g,k,t]`, `fg[0] = 0`;
  - raw stock `stock[t] = stock[t−1] + buy − Σ_g x`, `stock[0] = 0`;
  - melt capacity `Σ_g charge[g,k,t] ≤ melt_capacity[k]` and scrap-yard
    capacity `Σ_scrap stock[i,k,t] ≤ yard_capacity[k]` per plant and period;
  - market volume `Σ_k buy[i,k,t] ≤ market[i,t]` for scrap and primary metal
    (master alloys are bought freely at list price).

Period-to-period price swings make it worth buying ahead into the yard and
melting ahead into finished stock, so the staircase of balance rows binds
together with the capacity and composition rows.

# Sizing

Per plant-period, `Σ_{g∈G_k} (|compatible(g)| + 2) + 2|materials|` variables.
Plants are added (each with a 1–8 grade portfolio) until one period of all
plants times a nominal horizon reaches the target; the horizon is then
`round(target / per_period)` periods (1–104).

# Feasibility

  - `feasible`: a hand-written recipe per `(g, k, t)` produces exactly that
    period's demand, bought in the same period (no stock); windows are the
    registered ones widened around every recipe; capacities and market volumes
    are 1.02–1.30× the planted use. Stored as a `MultiPeriodBlendingPlan`.
  - `infeasible`: one plant's melt capacity is cut below the cumulative charge
    its horizon demand needs (`MultiPeriodBlendingCertificate`) — an
    aggregation of all its periods' capacity, balance and output rows.
  - `unknown`: nominal demands and capacities with no planted point.
"""
struct MultiPeriodBlendingProblem <: ProblemGenerator
    n_plants::Int
    n_periods::Int
    materials::BlendMaterials
    portfolio::Vector{Vector{Int}}
    blend_vars::Vector{Tuple{Int, Int, Int, Int}}
    price::Matrix{Float64}
    market::Matrix{Float64}
    holding_cost::Vector{Float64}
    finished_holding::Vector{Float64}
    melt_cost::Float64
    demand::Array{Float64, 3}
    lo::Matrix{Float64}
    hi::Matrix{Float64}
    melt_capacity::Vector{Float64}
    yard_capacity::Vector{Float64}
    feasible_witness::Union{Nothing, MultiPeriodBlendingPlan}
    infeasibility_certificate::Union{Nothing, MultiPeriodBlendingCertificate}
    feasibility_status::FeasibilityStatus
end

function _mp_materials(rng::AbstractRNG, grades::Vector{Int}, n_classes::Int)
    kind, source, comps, cost, yield = _blend_base_materials()
    keep = Int[]
    for i in eachindex(kind)
        if kind[i] == :primary
            (source[i] == 1 || rand(rng) < 0.5) && push!(keep, i)
        else
            e = _BLEND_HARDENERS[source[i]].element
            any(_BLEND_GRADES[g].lo[e] > 0 for g in grades) && push!(keep, i)
        end
    end
    # Scrap classes the portfolio can use, most common first, at class chemistry.
    usable = [c for c in eachindex(_BLEND_SCRAP_CLASSES) if
              any(_blend_family(g) in _BLEND_SCRAP_CLASSES[c].families for g in grades)]
    w = _BLEND_CLASS_WEIGHTS[usable]
    chosen = usable[sample(rng, eachindex(usable), Weights(w), min(n_classes, length(usable)); replace=false)]
    k2 = [kind[i] for i in keep]
    s2 = [source[i] for i in keep]
    c2 = [_blend_assay.(comps[i]) for i in keep]
    p2 = [cost[i] for i in keep]
    y2 = [yield[i] for i in keep]
    for c in sort(chosen)
        spec = _BLEND_SCRAP_CLASSES[c]
        push!(k2, :scrap); push!(s2, c); push!(c2, _blend_assay.(collect(spec.comp)))
        push!(p2, spec.cost); push!(y2, spec.yield)
    end
    return BlendMaterials(k2, s2, reduce(hcat, c2), p2, y2, zeros(Int, length(k2)))
end

function _mp_portfolio_size(mats::BlendMaterials, portfolio::Vector{Int})
    n = 2 * length(mats.kind)
    for g in portfolio
        n += 2 + count(i -> blend_compatible(mats, i, g), eachindex(mats.kind))
    end
    return n
end

"""
    mp_plan_satisfies(prob, plan=prob.feasible_witness; atol=1e-7)

Check a `MultiPeriodBlendingPlan` against every row and bound, without a solver.
"""
function mp_plan_satisfies(
    prob::MultiPeriodBlendingProblem,
    plan::Union{Nothing, MultiPeriodBlendingPlan}=prob.feasible_witness;
    atol::Float64=1e-7,
)
    plan === nothing && return false
    mats = prob.materials
    nI, K, T, G = length(mats.kind), prob.n_plants, prob.n_periods, length(_BLEND_GRADES)
    tol(v) = atol * max(1.0, abs(v))
    all(>=(-atol), plan.blend) && all(>=(-atol), plan.charge) && all(>=(-atol), plan.buy) || return false
    all(>=(-atol), plan.stock) && all(>=(-atol), plan.finished) || return false
    mass = zeros(G, K, T)
    out = zeros(G, K, T)
    element = zeros(length(BLEND_ELEMENTS), G, K, T)
    used = zeros(nI, K, T)
    for (v, (i, g, k, t)) in enumerate(prob.blend_vars)
        q = plan.blend[v]
        mass[g, k, t] += q
        out[g, k, t] += mats.yield[i] * q
        element[:, g, k, t] .+= q .* view(mats.comp, :, i)
        used[i, k, t] += q
    end
    for k in 1:K, t in 1:T
        for g in prob.portfolio[k]
            abs(mass[g, k, t] - plan.charge[g, k, t]) <= tol(mass[g, k, t]) || return false
            m = plan.charge[g, k, t]
            for e in eachindex(BLEND_ELEMENTS)
                element[e, g, k, t] + tol(m) >= prob.lo[e, g] * m || return false
                element[e, g, k, t] <= prob.hi[e, g] * m + tol(m) || return false
            end
            previous = t == 1 ? 0.0 : plan.finished[g, k, t - 1]
            expected = previous + out[g, k, t] - prob.demand[g, k, t]
            abs(plan.finished[g, k, t] - expected) <= tol(prob.demand[g, k, t]) || return false
        end
        sum(plan.charge[g, k, t] for g in prob.portfolio[k]) <= prob.melt_capacity[k] + tol(prob.melt_capacity[k]) ||
            return false
        for i in 1:nI
            previous = t == 1 ? 0.0 : plan.stock[i, k, t - 1]
            expected = previous + plan.buy[i, k, t] - used[i, k, t]
            abs(plan.stock[i, k, t] - expected) <= tol(max(plan.buy[i, k, t], used[i, k, t])) || return false
        end
        yard = sum(plan.stock[i, k, t] for i in 1:nI if mats.kind[i] == :scrap; init=0.0)
        yard <= prob.yard_capacity[k] + tol(yard) || return false
    end
    for i in 1:nI, t in 1:T
        isfinite(prob.market[i, t]) || continue
        bought = sum(plan.buy[i, k, t] for k in 1:K)
        bought <= prob.market[i, t] + tol(bought) || return false
    end
    return true
end

function _mp_max_yield(mats::BlendMaterials, g::Int)
    return maximum(mats.yield[i] for i in eachindex(mats.kind) if blend_compatible(mats, i, g))
end

function _mp_required_charge(prob::MultiPeriodBlendingProblem, k::Int)
    return sum(
        prob.demand[g, k, t] / _mp_max_yield(prob.materials, g) for g in prob.portfolio[k], t in 1:prob.n_periods
    )
end

"""
    mp_certificate_holds(prob::MultiPeriodBlendingProblem)

Recompute the cumulative melt-capacity certificate and check it.
"""
function mp_certificate_holds(prob::MultiPeriodBlendingProblem)
    cert = prob.infeasibility_certificate
    cert === nothing && return false
    1 <= cert.plant <= prob.n_plants || return false
    required = _mp_required_charge(prob, cert.plant)
    achievable = prob.n_periods * prob.melt_capacity[cert.plant]
    isapprox(required, cert.required; rtol=1e-9) && isapprox(achievable, cert.achievable; rtol=1e-9) || return false
    return achievable < required * (1 - 1e-9)
end

"""
    MultiPeriodBlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-plant, multi-period alloy production instance (see the type).
"""
function MultiPeriodBlendingProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    rng = MersenneTwister(seed)
    target = max(target_variables, 1)
    nominal_T = clamp(round(Int, target^0.3 * rand(rng, Uniform(0.8, 1.2))), 2, 52)
    portfolio_size = clamp(round(Int, target^0.18), 1, 8)
    n_classes = clamp(round(Int, 1 + target^0.2), 2, length(_BLEND_SCRAP_CLASSES))

    # Plants and their portfolios; materials follow the union of grades.
    portfolio = Vector{Vector{Int}}()
    grades_all = Int[]
    material_seed = rand(rng, UInt32)
    mats = nothing
    per_period = 0
    while isempty(portfolio) || per_period * nominal_T < target
        size_k = clamp(portfolio_size + rand(rng, -1:1), 1, 8)
        grades = Int[]
        while length(grades) < size_k
            g = _blend_sample_grade(rng)
            g in grades || push!(grades, g)
        end
        push!(portfolio, sort(grades))
        union!(grades_all, grades)
        # The material list depends on the portfolio union, so it is redrawn
        # from a dedicated stream as plants are added.
        mats = _mp_materials(MersenneTwister(material_seed), sort(grades_all), n_classes)
        per_period = sum(_mp_portfolio_size(mats, p) for p in portfolio)
        length(portfolio) >= 40 && break
    end
    K = length(portfolio)
    T = clamp(round(Int, target / per_period), 1, 104)
    nI = length(mats.kind)
    G = length(_BLEND_GRADES)

    blend_vars = Tuple{Int, Int, Int, Int}[]
    for k in 1:K, t in 1:T, g in portfolio[k], i in 1:nI
        blend_compatible(mats, i, g) && push!(blend_vars, (i, g, k, t))
    end

    # Prices follow a mean-reverting market walk per material.
    price = zeros(Float64, nI, T)
    for i in 1:nI
        level = 1.0
        vol = mats.kind[i] == :scrap ? 0.06 : 0.03
        for t in 1:T
            level = exp(0.7 * log(level) + rand(rng, Normal(0.0, vol)))
            price[i, t] = mats.cost[i] * level
        end
    end
    holding_cost = [0.004 * mats.cost[i] for i in 1:nI]
    finished_holding = [0.006 * _BLEND_GRADES[g].price for g in 1:G]
    melt_cost = rand(rng, Uniform(60.0, 120.0))

    base = [clamp(rand(rng, LogNormal(log(60.0), 0.6)), 5.0, 600.0) for g in 1:G, k in 1:K]
    season = rand(rng, Uniform(0.0, 2pi), K)
    demand = zeros(Float64, G, K, T)
    for k in 1:K, g in portfolio[k], t in 1:T
        demand[g, k, t] = base[g, k] * (1 + 0.15 * sin(2pi * t / 13 + season[k])) * rand(rng, LogNormal(0.0, 0.15))
    end

    lo = [Float64(_BLEND_GRADES[g].lo[e]) for e in eachindex(BLEND_ELEMENTS), g in 1:G]
    hi = [Float64(_BLEND_GRADES[g].hi[e]) for e in eachindex(BLEND_ELEMENTS), g in 1:G]
    melt_capacity = zeros(Float64, K)
    yard_capacity = zeros(Float64, K)
    market = fill(Inf, nI, T)
    witness = nothing
    compat = [[i for i in 1:nI if blend_compatible(mats, i, g)] for g in 1:G]

    if feasibility_status == unknown
        melt_need = [sum(demand[g, k, t] / 0.93 for g in portfolio[k]) for k in 1:K, t in 1:T]
        for k in 1:K
            melt_capacity[k] = sum(view(melt_need, k, :)) / T * rand(rng, Uniform(0.95, 1.5))
            yard_capacity[k] = sum(view(melt_need, k, :)) / T * rand(rng, Uniform(0.3, 1.5))
        end
        # Regional market volumes: primary metal could cover ~80% of the melt
        # and scrap ~60%, each scaled by a per-material, per-period draw, so a
        # thin market in some period may or may not be bridged by stock.
        total_need = vec(sum(melt_need; dims=1))
        n_primary = count(==(:primary), mats.kind)
        n_scrap = count(==(:scrap), mats.kind)
        for i in 1:nI, t in 1:T
            mats.kind[i] == :hardener && continue
            share = mats.kind[i] == :primary ? 0.8 / n_primary : 0.6 / n_scrap
            market[i, t] = total_need[t] * share * rand(rng, Uniform(0.5, 1.6))
        end
    else
        blend = zeros(Float64, length(blend_vars))
        charge = zeros(Float64, G, K, T)
        used = zeros(Float64, nI, K, T)
        index = Dict(v => n for (n, v) in enumerate(blend_vars))
        for k in 1:K, t in 1:T, g in portfolio[k]
            cand = compat[g]
            fractions, composition = _blend_recipe(rng, g, cand, mats.kind, mats.source, mats.comp)
            mass = demand[g, k, t] / sum(fractions[j] * mats.yield[cand[j]] for j in eachindex(cand))
            for (j, i) in enumerate(cand)
                q = mass * fractions[j]
                blend[index[(i, g, k, t)]] = q
                used[i, k, t] += q
            end
            charge[g, k, t] = sum(mass * fractions[j] for j in eachindex(cand))
            for e in eachindex(BLEND_ELEMENTS)
                lo[e, g] > 0 && (lo[e, g] = min(lo[e, g], 0.98 * composition[e]))
                hi[e, g] = max(hi[e, g], 1.02 * composition[e])
            end
        end
        # The planted output meets each period's demand exactly (demand is
        # set to the recipe's output so rounding cannot leave a shortfall).
        finished = zeros(Float64, G, K, T)
        for k in 1:K, g in portfolio[k], t in 1:T
            demand[g, k, t] = sum(mats.yield[i] * blend[index[(i, g, k, t)]] for i in compat[g])
        end
        buy = copy(used)
        stock = zeros(Float64, nI, K, T)
        for k in 1:K
            melt_capacity[k] = maximum(sum(charge[g, k, t] for g in portfolio[k]) for t in 1:T) * rand(rng, Uniform(1.03, 1.20))
            yard_capacity[k] = maximum(sum(used[i, k, t] for i in 1:nI if mats.kind[i] == :scrap; init=0.0) for t in 1:T) *
                               rand(rng, Uniform(0.5, 1.5)) + 1.0
        end
        for i in 1:nI, t in 1:T
            mats.kind[i] == :hardener && continue
            bought = sum(buy[i, k, t] for k in 1:K)
            market[i, t] = bought > 0 ? bought * rand(rng, Uniform(1.02, 1.30)) : rand(rng, Uniform(5.0, 50.0))
        end
        witness = MultiPeriodBlendingPlan(blend, charge, buy, stock, finished)
    end

    certificate = nothing
    if feasibility_status == infeasible
        k = rand(rng, 1:K)
        tmp = MultiPeriodBlendingProblem(
            K, T, mats, portfolio, blend_vars, price, market, holding_cost, finished_holding, melt_cost,
            demand, lo, hi, melt_capacity, yard_capacity, nothing, nothing, feasibility_status,
        )
        required = _mp_required_charge(tmp, k)
        melt_capacity[k] = required / (T * rand(rng, Uniform(1.08, 1.25)))
        certificate = MultiPeriodBlendingCertificate(k, T * melt_capacity[k], required)
        witness = nothing
    end

    prob = MultiPeriodBlendingProblem(
        K, T, mats, portfolio, blend_vars, price, market, holding_cost, finished_holding, melt_cost, demand,
        lo, hi, melt_capacity, yard_capacity, witness, certificate, feasibility_status,
    )
    feasibility_status == feasible && @assert mp_plan_satisfies(prob)
    feasibility_status == infeasible && @assert mp_certificate_holds(prob)
    return prob
end

"""
    build_model(prob::MultiPeriodBlendingProblem)

Build the multi-period alloy production LP (deterministic; see the type).
"""
function build_model(prob::MultiPeriodBlendingProblem)
    model = Model()
    mats = prob.materials
    K, T, nI = prob.n_plants, prob.n_periods, length(mats.kind)
    V = length(prob.blend_vars)
    keys_gkt = [(g, k, t) for k in 1:K for t in 1:T for g in prob.portfolio[k]]
    @variable(model, x[1:V] >= 0)
    @variable(model, charge[keys_gkt] >= 0)
    @variable(model, buy[1:nI, 1:K, 1:T] >= 0)
    @variable(model, stock[1:nI, 1:K, 1:T] >= 0)
    @variable(model, fg[keys_gkt] >= 0)
    @objective(
        model,
        Min,
        sum(prob.price[i, t] * buy[i, k, t] + prob.holding_cost[i] * stock[i, k, t] for i in 1:nI, k in 1:K, t in 1:T) +
        sum(prob.finished_holding[key[1]] * fg[key] + prob.melt_cost * charge[key] for key in keys_gkt)
    )
    groups = Dict{Tuple{Int, Int, Int}, Vector{Int}}()
    uses = Dict{Tuple{Int, Int, Int}, Vector{Int}}()
    for (v, (i, g, k, t)) in enumerate(prob.blend_vars)
        push!(get!(groups, (g, k, t), Int[]), v)
        push!(get!(uses, (i, k, t), Int[]), v)
    end
    for key in keys_gkt
        g, k, t = key
        vs = groups[key]
        materials = [prob.blend_vars[v][1] for v in vs]
        @constraint(model, sum(x[v] for v in vs) - charge[key] == 0)
        mins, maxs = _blend_active_rows(mats.comp, materials, view(prob.lo, :, g), view(prob.hi, :, g))
        for e in mins
            carriers = [v for v in vs if mats.comp[e, prob.blend_vars[v][1]] > 0]
            @constraint(model, sum(mats.comp[e, prob.blend_vars[v][1]] * x[v] for v in carriers) - prob.lo[e, g] * charge[key] >= 0)
        end
        for e in maxs
            carriers = [v for v in vs if mats.comp[e, prob.blend_vars[v][1]] > 0]
            @constraint(model, sum(mats.comp[e, prob.blend_vars[v][1]] * x[v] for v in carriers) - prob.hi[e, g] * charge[key] <= 0)
        end
        produced = @expression(model, sum(mats.yield[prob.blend_vars[v][1]] * x[v] for v in vs))
        if t == 1
            @constraint(model, fg[key] - produced == -prob.demand[g, k, t])
        else
            @constraint(model, fg[key] - fg[(g, k, t - 1)] - produced == -prob.demand[g, k, t])
        end
    end
    for k in 1:K, t in 1:T
        for i in 1:nI
            consumed = haskey(uses, (i, k, t)) ? sum(x[v] for v in uses[(i, k, t)]) : 0.0
            previous = t == 1 ? 0.0 : stock[i, k, t - 1]
            @constraint(model, stock[i, k, t] - previous - buy[i, k, t] + consumed == 0)
        end
        @constraint(model, sum(charge[(g, k, t)] for g in prob.portfolio[k]) <= prob.melt_capacity[k])
        scrap = [i for i in 1:nI if mats.kind[i] == :scrap]
        isempty(scrap) || @constraint(model, sum(stock[i, k, t] for i in scrap) <= prob.yard_capacity[k])
    end
    for i in 1:nI, t in 1:T
        isfinite(prob.market[i, t]) || continue
        @constraint(model, sum(buy[i, k, t] for k in 1:K) <= prob.market[i, t])
    end
    return model
end

register_variant(
    :blending,
    :multi_period,
    MultiPeriodBlendingProblem,
    "Multi-plant, multi-period alloy production planning: scrap and primary purchasing on a " *
    "moving market, raw and finished-goods inventories, melt and yard capacities, composition windows";
    tags=[:production, :blending, :staircase, :block_angular],
)
