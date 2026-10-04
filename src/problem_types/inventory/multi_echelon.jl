using JuMP
using Random
using Distributions

"""
Planted distribution plan: just-in-time replenishment along each store's
primary lane, with safety stock retained at every node. `plant_to_dc[p, r, t]`
is the production shipped to DC `r`, `dc_to_store[p, s, t]` the shipment to
store `s` on its primary lane (secondary lanes unused), and `dc_stock` /
`store_stock` the end-of-period inventories.
"""
struct MultiEchelonPlanWitness
    plant_to_dc::Array{Float64, 3}
    dc_to_store::Array{Float64, 3}
    dc_stock::Array{Float64, 3}
    store_stock::Array{Float64, 3}
end

"""
Plant prefix-capacity certificate. Summing every DC and store balance row of
product `p` over periods `1..horizon` shows the network must have produced at
least `D_p(1..horizon) - I0_p` units in periods `1..horizon - 1` (production
reaches a DC one period after it is made; network stock and in-transit
quantities are nonnegative). Weighting by plant hours per unit, the plant needs
`required = Σ_p hours[p] max(0, ...)` hours in those periods, but its capacity
rows supply only `available < required`. The refutation combines balance rows
of the whole network with the plant rows — presolve does not see it.
"""
struct MultiEchelonPrefixCertificate
    horizon::Int
    required::Float64
    available::Float64
end

"""
    MultiEchelonInventoryProblem <: ProblemGenerator

Two-echelon, multi-product distribution planning: plant → regional DCs →
stores.

# Overview

A plant makes `n_products` product groups, ships them to `n_dcs` regional
distribution centers (one-period transit), and the DCs replenish `n_stores`
stores. Every store has a primary lane from its regional DC (0–1 period
transit) and, for about a third of stores, a costlier secondary lane from a
neighbouring DC used when the primary DC is short. Demand at stores must be met
(no backlog).

Columns: `f[p, r, t]` plant → DC (`t < n_periods`), `g[p, ℓ, t]` lane
shipments (`t <= n_periods - transit[ℓ]`), `J[p, r, t]` DC stock, `K[p, s, t]`
store stock.

Rows:

  - flow balance per product, node, and period (DCs and stores);
  - plant capacity per period: `Σ_p hours[p] Σ_r f[p,r,t] <= plant_capacity[t]`;
  - DC throughput per DC and period: `Σ_p Σ_{ℓ from r} g[p,ℓ,t] <= dc_throughput[r,t]`;
  - DC storage per DC and period: `Σ_p cube[p] J[p,r,t] <= dc_storage[r]`;
  - store shelf space (multi-product only) per store and period.

Objective: production, transport (distance-based), and holding costs.

The previous version was a single-product star with a handful of locations —
100k variables came from a 4,000-period horizon — and `(L - 1) · T`
structurally dead return arcs. Here size comes from the network (hundreds of
stores at weekly granularity), secondary lanes are live alternatives, and the
plant/DC capacity rows couple everything.

# Feasibility control

  - `feasible`: the JIT primary-lane plan is planted
    ([`MultiEchelonPlanWitness`](@ref)); capacities are drawn above it.
  - `infeasible`: the plant capacity is cut so that the binding demand prefix
    needs 10–35% more plant hours than it has
    ([`MultiEchelonPrefixCertificate`](@ref)).
  - `unknown`: the same ratio drawn as `1 ± U(0.03, 0.30)`.

# Fields

  - `n_products`, `n_dcs`, `n_stores`, `n_periods`, `n_lanes::Int`
  - `lane_dc`, `lane_store`, `lane_transit::Vector{Int}`, `lane_cost::Vector{Float64}`
  - `primary_lane::Vector{Int}`: per store
  - `demand::Array{Float64,3}`: `n_products × n_stores × n_periods`
  - `dc_initial::Matrix{Float64}`, `store_initial::Matrix{Float64}`: `n_products × nodes`
  - `plant_hours`, `cube`, `production_cost`, `dc_holding`, `store_holding::Vector{Float64}`: per product
  - `plant_dc_cost::Vector{Float64}`: per DC
  - `plant_capacity::Vector{Float64}`, `dc_throughput::Matrix{Float64}`,
    `dc_storage::Vector{Float64}`, `store_shelf::Vector{Float64}`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct MultiEchelonInventoryProblem <: ProblemGenerator
    n_products::Int
    n_dcs::Int
    n_stores::Int
    n_periods::Int
    n_lanes::Int
    lane_dc::Vector{Int}
    lane_store::Vector{Int}
    lane_transit::Vector{Int}
    lane_cost::Vector{Float64}
    primary_lane::Vector{Int}
    demand::Array{Float64, 3}
    dc_initial::Matrix{Float64}
    store_initial::Matrix{Float64}
    plant_hours::Vector{Float64}
    cube::Vector{Float64}
    production_cost::Vector{Float64}
    dc_holding::Vector{Float64}
    store_holding::Vector{Float64}
    plant_dc_cost::Vector{Float64}
    plant_capacity::Vector{Float64}
    dc_throughput::Matrix{Float64}
    dc_storage::Vector{Float64}
    store_shelf::Vector{Float64}
    feasible_witness::Union{Nothing, MultiEchelonPlanWitness}
    infeasibility_certificate::Union{Nothing, MultiEchelonPrefixCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _multi_echelon_prefix_requirement(demand, dc_initial, store_initial, hours, horizon)

Plant hours needed in periods `1..horizon - 1` to serve all store demand of
periods `1..horizon` (see [`MultiEchelonPrefixCertificate`](@ref)).
"""
function _multi_echelon_prefix_requirement(demand, dc_initial, store_initial, hours, horizon::Int)
    P = size(demand, 1)
    return sum(
        hours[p] * max(
            0.0,
            sum(view(demand, p, :, 1:horizon)) - sum(view(dc_initial, p, :)) -
            sum(view(store_initial, p, :)),
        ) for p in 1:P
    )
end

"""
    MultiEchelonInventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a two-echelon distribution planning instance with about
`target_variables` columns.
"""
function MultiEchelonInventoryProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    rng = MersenneTwister(seed)
    target = max(target_variables, 30)

    # Several product groups share shelves, DC storage, and plant hours (with
    # a single product the store chains decouple and presolve folds them away).
    P = target <= 300 ? 1 : rand(rng, 2:4)
    T = target <= 500 ? rand(rng, 4:6) : (target <= 10_000 ? rand(rng, 8:13) : rand(rng, 13:26))
    second_prob = 0.35
    # Columns ≈ P·T·(stores·(1 + 1 + 0.35) + dcs·2): size stores from the target.
    stores_per_dc = rand(rng, 8:30)
    per_store = P * T * (2 + second_prob + 2 / stores_per_dc)
    R = max(1, round(Int, target / per_store / stores_per_dc))

    # Geography: DCs spread over a region; stores clustered around their DC.
    # Stores (with their lanes) are added until the column budget is used.
    dc_xy = [(100 * rand(rng), 100 * rand(rng)) for _ in 1:R]
    dist(a, b) = hypot(a[1] - b[1], a[2] - b[2])
    lane_dc, lane_store, lane_transit, lane_cost = Int[], Int[], Int[], Float64[]
    primary_lane = Int[]
    store_dc = Int[]
    store_xy = Tuple{Float64, Float64}[]
    rate = rand(rng, Uniform(0.02, 0.05))       # cost per unit-distance
    cols = P * (R * (T - 1) + R * T)
    while cols < target || length(store_dc) < 2
        s = length(store_dc) + 1
        r = mod1(s, R)
        xy = (dc_xy[r][1] + 12 * randn(rng), dc_xy[r][2] + 12 * randn(rng))
        push!(store_dc, r)
        push!(store_xy, xy)
        d = dist(dc_xy[r], xy)
        push!(lane_dc, r)
        push!(lane_store, s)
        push!(lane_transit, d > 25 ? 1 : 0)
        push!(lane_cost, rate * (5 + d))
        push!(primary_lane, length(lane_dc))
        cols += P * (T + T - lane_transit[end])
        if R > 1 && rand(rng) < second_prob
            # Nearest other DC as the backup lane, at an expedite premium.
            others = [q for q in 1:R if q != r]
            q = others[argmin([dist(dc_xy[o], xy) for o in others])]
            d2 = dist(dc_xy[q], xy)
            push!(lane_dc, q)
            push!(lane_store, s)
            push!(lane_transit, d2 > 25 ? 1 : 0)
            push!(lane_cost, rate * (5 + d2) * 1.3)
            cols += P * (T - lane_transit[end])
        end
    end
    S = length(store_dc)
    n_lanes = length(lane_dc)
    plant_xy = (100 * rand(rng), 100 * rand(rng))
    plant_dc_cost = [rate * 0.6 * (10 + dist(plant_xy, dc_xy[r])) for r in 1:R]

    # Demand: product popularity x store size x seasonality.
    phase = 2π * rand(rng)
    amp = 0.3 * rand(rng)
    product_scale = rand(rng, LogNormal(0.0, 0.5), P)
    store_size = rand(rng, LogNormal(log(30.0), 0.6), S)
    demand = zeros(P, S, T)
    for p in 1:P, s in 1:S
        demand[p, s, :] = _inventory_demand(
            rng,
            T,
            product_scale[p] * store_size[s];
            amp=amp,
            phase=phase + 0.3 * randn(rng),
            trend=0.005 * randn(rng),
            cv=0.2 + 0.2 * rand(rng),
            intermittent=rand(rng) < 0.05,
        )
    end
    plant_hours = rand(rng, LogNormal(log(0.02), 0.4), P)
    cube = rand(rng, LogNormal(log(0.03), 0.5), P)
    production_cost = rand(rng, LogNormal(log(8.0), 0.5), P)
    dc_holding = production_cost .* rand(rng, Uniform(0.003, 0.006), P)
    store_holding = dc_holding .* rand(rng, Uniform(1.5, 3.0), P)

    # --- Planted JIT plan along primary lanes ---------------------------------
    store_initial = zeros(P, S)
    dc_initial = zeros(P, R)
    dc_to_store = zeros(P, S, T)          # shipped on the primary lane in period t
    store_stock = zeros(P, S, T)
    dc_need = zeros(P, R, T)              # what each DC ships in period t
    for p in 1:P, s in 1:S
        L = lane_transit[primary_lane[s]]
        avg = sum(demand[p, s, :]) / T
        safety = avg * rand(rng, Uniform(0.2, 0.6))
        store_initial[p, s] = round(sum(demand[p, s, 1:L]; init=0.0) + safety; digits=1)
        level = store_initial[p, s]
        for t in 1:T
            arrival = 0.0
            if t > L
                arrival = max(0.0, demand[p, s, t] + safety - level)
                dc_to_store[p, s, t - L] = arrival
                dc_need[p, store_dc[s], t - L] += arrival
            end
            level += arrival - demand[p, s, t]
            store_stock[p, s, t] = level
        end
    end
    plant_to_dc = zeros(P, R, T)
    dc_stock = zeros(P, R, T)
    for p in 1:P, r in 1:R
        avg = sum(dc_need[p, r, :]) / T
        safety = avg * rand(rng, Uniform(0.3, 0.8))
        dc_initial[p, r] = round(dc_need[p, r, 1] + safety; digits=1)
        level = dc_initial[p, r]
        for t in 1:T
            arrival = 0.0
            if t > 1
                arrival = max(0.0, dc_need[p, r, t] + safety - level)
                plant_to_dc[p, r, t - 1] = arrival
            end
            level += arrival - dc_need[p, r, t]
            dc_stock[p, r, t] = level
        end
    end

    # Capacities with headroom above the plan.
    plant_load = [sum(plant_hours[p] * plant_to_dc[p, r, t] for p in 1:P, r in 1:R) for t in 1:T]
    plant_base = sum(plant_load) / max(T - 1, 1) * rand(rng, Uniform(1.05, 1.25))
    plant_capacity = [max(plant_base, plant_load[t] * 1.05, 1.0) for t in 1:T]
    dc_throughput = zeros(R, T)
    for r in 1:R
        out = [sum(dc_to_store[p, s, t] for p in 1:P, s in 1:S if store_dc[s] == r) for t in 1:T]
        base = sum(out) / T * rand(rng, Uniform(1.05, 1.3))
        for t in 1:T
            dc_throughput[r, t] = max(base, out[t] * 1.05, 1.0)
        end
    end
    dc_storage = [
        max(
            maximum(sum(cube[p] * dc_stock[p, r, t] for p in 1:P) for t in 1:T) *
            rand(rng, Uniform(1.2, 1.8)),
            1.0,
        ) for r in 1:R
    ]
    store_shelf = [
        max(
            maximum(sum(cube[p] * store_stock[p, s, t] for p in 1:P) for t in 1:T) *
            rand(rng, Uniform(1.2, 1.8)),
            0.1,
        ) for s in 1:S
    ]

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        witness = MultiEchelonPlanWitness(plant_to_dc, dc_to_store, dc_stock, store_stock)
    else
        ratio = _inventory_scale_ratio(rng, feasibility_status)
        best_h, best_r = 2, -1.0
        for h in 2:T
            req = _multi_echelon_prefix_requirement(
                demand, dc_initial, store_initial, plant_hours, h
            )
            r = req / sum(plant_capacity[1:(h - 1)])
            if r > best_r
                best_h, best_r = h, r
            end
        end
        plant_capacity .*= best_r / ratio
        if feasibility_status == infeasible
            req = _multi_echelon_prefix_requirement(
                demand, dc_initial, store_initial, plant_hours, best_h
            )
            certificate = MultiEchelonPrefixCertificate(
                best_h, req, sum(plant_capacity[1:(best_h - 1)])
            )
        end
    end

    return MultiEchelonInventoryProblem(
        P,
        R,
        S,
        T,
        n_lanes,
        lane_dc,
        lane_store,
        lane_transit,
        lane_cost,
        primary_lane,
        demand,
        dc_initial,
        store_initial,
        plant_hours,
        cube,
        production_cost,
        dc_holding,
        store_holding,
        plant_dc_cost,
        plant_capacity,
        dc_throughput,
        dc_storage,
        store_shelf,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::MultiEchelonInventoryProblem)

Build the two-echelon distribution LP. Deterministic. Variables: `f[p, r, t]`
(`t < T`), `g[p, ℓ, t]` (`t <= T - transit[ℓ]`), `J[p, r, t]`, `K[p, s, t]`.
"""
function build_model(prob::MultiEchelonInventoryProblem)
    model = Model()
    P, R, S, T = prob.n_products, prob.n_dcs, prob.n_stores, prob.n_periods
    nl = prob.n_lanes
    @variable(model, f[1:P, 1:R, 1:(T - 1)] >= 0)
    @variable(model, g[p = 1:P, l = 1:nl, t = 1:(T - prob.lane_transit[l])] >= 0)
    @variable(model, J[1:P, 1:R, 1:T] >= 0)
    @variable(model, K[1:P, 1:S, 1:T] >= 0)
    single = P == 1
    if single
        for s in 1:S, t in 1:T
            set_upper_bound(K[1, s, t], prob.store_shelf[s] / prob.cube[1])
        end
    end

    dc_lanes = [findall(==(r), prob.lane_dc) for r in 1:R]
    store_lanes = [findall(==(s), prob.lane_store) for s in 1:S]

    # DC balance: initial + arrivals from the plant - shipments = stock.
    for p in 1:P, r in 1:R, t in 1:T
        expr = AffExpr(t == 1 ? prob.dc_initial[p, r] : 0.0)
        t > 1 && add_to_expression!(expr, 1.0, J[p, r, t - 1])
        t > 1 && add_to_expression!(expr, 1.0, f[p, r, t - 1])
        for l in dc_lanes[r]
            t <= T - prob.lane_transit[l] && add_to_expression!(expr, -1.0, g[p, l, t])
        end
        add_to_expression!(expr, -1.0, J[p, r, t])
        @constraint(model, expr == 0)
    end
    # Store balance: arrivals on every lane - demand = stock.
    for p in 1:P, s in 1:S, t in 1:T
        expr = AffExpr(t == 1 ? prob.store_initial[p, s] : 0.0)
        t > 1 && add_to_expression!(expr, 1.0, K[p, s, t - 1])
        for l in store_lanes[s]
            L = prob.lane_transit[l]
            t > L && add_to_expression!(expr, 1.0, g[p, l, t - L])
        end
        add_to_expression!(expr, -1.0, K[p, s, t])
        @constraint(model, expr == prob.demand[p, s, t])
    end
    # Plant capacity.
    for t in 1:(T - 1)
        @constraint(
            model,
            sum(prob.plant_hours[p] * f[p, r, t] for p in 1:P, r in 1:R) <= prob.plant_capacity[t]
        )
    end
    # DC throughput and storage.
    for r in 1:R, t in 1:T
        terms = [(p, l) for p in 1:P for l in dc_lanes[r] if t <= T - prob.lane_transit[l]]
        isempty(terms) ||
            @constraint(model, sum(g[p, l, t] for (p, l) in terms) <= prob.dc_throughput[r, t])
        @constraint(model, sum(prob.cube[p] * J[p, r, t] for p in 1:P) <= prob.dc_storage[r])
    end
    # Store shelf space (a variable bound when there is a single product).
    if !single
        for s in 1:S, t in 1:T
            @constraint(model, sum(prob.cube[p] * K[p, s, t] for p in 1:P) <= prob.store_shelf[s])
        end
    end

    @objective(
        model,
        Min,
        sum(
                (prob.production_cost[p] + prob.plant_dc_cost[r]) * f[p, r, t] for
                p in 1:P, r in 1:R, t in 1:(T - 1)
            ) +
            sum(
                prob.lane_cost[l] * g[p, l, t] for p in 1:P, l in 1:nl for
                t in 1:(T - prob.lane_transit[l])
            ) +
            sum(prob.dc_holding[p] * J[p, r, t] for p in 1:P, r in 1:R, t in 1:T) +
            sum(prob.store_holding[p] * K[p, s, t] for p in 1:P, s in 1:S, t in 1:T)
    )
    return model
end

register_variant(
    :inventory,
    :multi_echelon,
    MultiEchelonInventoryProblem,
    "Two-echelon multi-product distribution planning (plant -> regional DCs -> stores) with primary and backup lanes, transit times, plant/DC throughput/storage/shelf capacities, a planted JIT plan, and a network-wide prefix capacity certificate";
    tags=[:logistics, :network, :multicommodity, :staircase],
)
