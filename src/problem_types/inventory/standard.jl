using JuMP
using Random
using Distributions

"""
Planted replenishment plan: orders `orders[i, t]` (placed in period `t`, zero
where no order column exists), end-of-period stock `stock[i, t]`, no lost
sales. It covers every period's demand with the SKU's safety stock retained,
and all shared capacities (vendor, storage zone, receiving) are drawn with
headroom above what it uses.
"""
struct ReplenishmentPlanWitness
    orders::Matrix{Float64}
    stock::Matrix{Float64}
end

"""
Vendor allocation certificate. Every SKU in `skus` comes from vendor `vendor`
and has a service-level row `Σ_t lost[i,t] <= (1 - fill_rate[i]) Σ_t demand[i,t]`.
Summing its balance rows over the horizon gives `Σ_t q[i,t] >= fill_rate[i]
D_i - initial_inventory[i] - pipeline[i]`, so the vendor must ship at least
`required = Σ_i max(0, ...)` units, while its capacity rows allow only
`available < required` over all order periods. Uses balance, service, and
vendor rows only — an aggregate argument presolve does not see.
"""
struct VendorAllocationCertificate
    vendor::Int
    skus::Vector{Int}
    required::Float64
    available::Float64
end

"""
    InventoryProblem <: ProblemGenerator

Multi-SKU replenishment planning at a distribution center (the `inventory`
category's `standard` variant).

# Overview

A distribution center plans weekly purchase orders for `n_skus` SKUs bought
from `n_vendors` vendors with vendor-specific lead times, over `n_periods`
periods. Columns per SKU:

  - `q[i, t] >= 0` — order placed in period `t` (arrives `lead_time[i]` later),
    for `t <= n_periods - lead_time[i]`;
  - `I[i, t] >= 0` — end-of-period stock;
  - `u[i, t] ∈ [0, demand[i, t]]` — lost sales (periods with positive demand).

Rows:

  - stock balance: `I[i,t-1] + q[i,t-L_i] + pipeline[i,t] + u[i,t] - I[i,t] = demand[i,t]`
    (`I[i,0] = initial_inventory[i]`; open orders arrive as `pipeline`);
  - service level for A/B-class SKUs: `Σ_t u[i,t] <= (1 - fill_rate[i]) Σ_t demand[i,t]`;
  - vendor capacity (allocation), per vendor and order period:
    `Σ_{i from v} q[i,t] <= vendor_capacity[v,t]`;
  - storage, per temperature/handling zone and period:
    `Σ_{i in z} volume[i] I[i,t] <= zone_capacity[z]`;
  - receiving, per zone and period: `Σ_{i in z} pallets[i] q[i,t-L_i] <= receiving_capacity[z,t]`.

Objective: purchase cost plus holding cost plus lost-sales penalties (margin
plus goodwill). The SKU chains are coupled by vendor, storage, and receiving
rows — a multi-commodity staircase, unlike the single-resource production
model of `multi_item` or the network of `multi_echelon`.

# Feasibility control

  - `feasible`: an order-up-to plan without lost sales is planted
    ([`ReplenishmentPlanWitness`](@ref)); capacities are drawn above it.
  - `infeasible`: one vendor's allocation is cut so that its service-level
    SKUs need 10–35% more units than it can ship
    ([`VendorAllocationCertificate`](@ref)).
  - `unknown`: the same ratio drawn as `1 ± U(0.03, 0.30)`.

Lost sales make every instance without service rows trivially feasible, so the
service rows are what give the infeasibility certificate teeth.

# Fields

  - `n_skus`, `n_periods`, `n_vendors`, `n_zones::Int`
  - `vendor::Vector{Int}`, `zone::Vector{Int}`, `lead_time::Vector{Int}`: per SKU
  - `demand::Matrix{Float64}`, `pipeline::Matrix{Float64}`: `n_skus × n_periods`
  - `initial_inventory`, `fill_rate`, `volume`, `pallets::Vector{Float64}`: per SKU
  - `unit_cost`, `holding_cost`, `lost_sale_cost::Vector{Float64}`: per SKU
  - `vendor_capacity::Matrix{Float64}`: `n_vendors × n_periods`
  - `zone_capacity::Vector{Float64}`, `receiving_capacity::Matrix{Float64}`
  - `feasible_witness`, `infeasibility_certificate`, `feasibility_status`
"""
struct InventoryProblem <: ProblemGenerator
    n_skus::Int
    n_periods::Int
    n_vendors::Int
    n_zones::Int
    vendor::Vector{Int}
    zone::Vector{Int}
    lead_time::Vector{Int}
    demand::Matrix{Float64}
    pipeline::Matrix{Float64}
    initial_inventory::Vector{Float64}
    fill_rate::Vector{Float64}
    volume::Vector{Float64}
    pallets::Vector{Float64}
    unit_cost::Vector{Float64}
    holding_cost::Vector{Float64}
    lost_sale_cost::Vector{Float64}
    vendor_capacity::Matrix{Float64}
    zone_capacity::Vector{Float64}
    receiving_capacity::Matrix{Float64}
    feasible_witness::Union{Nothing, ReplenishmentPlanWitness}
    infeasibility_certificate::Union{Nothing, VendorAllocationCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _replenishment_required(prob_fields..., skus) -> Float64

Units vendor-sourced SKUs `skus` must receive over the horizon to meet their
fill rates (see [`VendorAllocationCertificate`](@ref)).
"""
function _replenishment_required(demand, pipeline, initial_inventory, fill_rate, skus)
    return sum(
        (
            max(0.0, fill_rate[i] * sum(view(demand, i, :)) - initial_inventory[i] - sum(view(pipeline, i, :)))
            for i in skus
        );
        init=0.0,
    )
end

"""
    InventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a multi-SKU replenishment instance with about `target_variables`
columns (within one SKU's columns).
"""
function InventoryProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be >= 1 (got $target_variables)."))
    rng = MersenneTwister(seed)
    target = max(target_variables, 12)

    T = target <= 300 ? rand(rng, 4:8) : (target <= 10_000 ? rand(rng, 8:16) : rand(rng, 13:30))
    n_vendor_hint = max(1, round(Int, target / (3T * rand(rng, 6:15))))
    vendor_lead = [rand(rng, 1:min(3, T - 1)) for _ in 1:n_vendor_hint]
    phase = 2π * rand(rng)
    amp = 0.35 * rand(rng)

    # SKUs until the column budget is used.
    vendor, lead_time, demand_rows = Int[], Int[], Vector{Vector{Float64}}()
    cols = 0
    while cols < target
        v = rand(rng, 1:n_vendor_hint)
        L = vendor_lead[v]
        base = rand(rng, LogNormal(log(40.0), 1.0))
        d = _inventory_demand(
            rng, T, base; amp=amp, phase=phase + 0.5 * randn(rng), trend=0.01 * randn(rng),
            cv=0.15 + 0.35 * rand(rng), intermittent=rand(rng) < 0.12,
        )
        push!(vendor, v)
        push!(lead_time, L)
        push!(demand_rows, d)
        cols += (T - L) + T + count(>(0.0), d)
    end
    N = length(vendor)
    # Compact vendor ids.
    used = sort(unique(vendor))
    vmap = Dict(v => k for (k, v) in enumerate(used))
    vendor = [vmap[v] for v in vendor]
    V = length(used)
    demand = permutedims(reduce(hcat, demand_rows))
    n_zones = max(1, round(Int, N / rand(rng, 20:60)))
    zone = [rand(rng, 1:n_zones) for _ in 1:N]

    # SKU economics and service classes (ABC by demand volume).
    unit_cost = rand(rng, LogNormal(log(12.0), 0.8), N)
    holding_cost = unit_cost .* rand(rng, Uniform(0.003, 0.008), N)   # weekly carrying
    lost_sale_cost = unit_cost .* rand(rng, Uniform(0.3, 1.5), N)
    volume = rand(rng, LogNormal(log(0.05), 0.7), N)                  # m3 per unit
    pallets = volume ./ rand(rng, Uniform(0.8, 1.6), N)
    totals = [sum(demand[i, :]) for i in 1:N]
    rank = sortperm(totals; rev=true)
    fill_rate = zeros(N)
    for (k, i) in enumerate(rank)
        q = k / N
        fill_rate[i] = q <= 0.2 ? rand(rng, Uniform(0.95, 0.99)) : (q <= 0.5 ? rand(rng, Uniform(0.85, 0.95)) : 0.0)
    end

    # --- Planted order-up-to plan (no lost sales) --------------------------------
    initial_inventory = zeros(N)
    pipeline = zeros(N, T)
    orders = zeros(N, T)
    stock = zeros(N, T)
    for i in 1:N
        L = lead_time[i]
        avg = totals[i] / T
        safety = avg * rand(rng, Uniform(0.2, 0.8))
        # Open orders cover the lead-time gap; initial stock is the safety level.
        for t in 1:L
            pipeline[i, t] = demand[i, t]
        end
        initial_inventory[i] = round(safety + avg * rand(rng, Uniform(0.0, 0.5)); digits=1)
        level = initial_inventory[i]
        for t in 1:T
            arrival = pipeline[i, t]
            if t > L
                arrival = max(0.0, demand[i, t] + safety - level)
                orders[i, t - L] = arrival
            end
            level += arrival - demand[i, t]
            stock[i, t] = level
        end
    end

    # Capacities around the plan: vendor allocations flat-ish with headroom,
    # zone storage above the plan's peak, receiving above its arrivals.
    vendor_capacity = zeros(V, T)
    for v in 1:V
        skus = findall(==(v), vendor)
        load = [sum(orders[i, t] for i in skus) for t in 1:T]
        base = sum(load) / T * rand(rng, Uniform(1.0, 1.3))
        for t in 1:T
            vendor_capacity[v, t] = max(base, load[t] * rand(rng, Uniform(1.05, 1.25)), 1.0)
        end
    end
    zone_capacity = zeros(n_zones)
    receiving_capacity = zeros(n_zones, T)
    for z in 1:n_zones
        skus = findall(==(z), zone)
        if isempty(skus)
            zone_capacity[z] = 1.0
            receiving_capacity[z, :] .= 1.0
            continue
        end
        peak = maximum(sum(volume[i] * stock[i, t] for i in skus) for t in 1:T)
        zone_capacity[z] = peak * rand(rng, Uniform(1.1, 1.5))
        arrivals = [
            sum(pallets[i] * (t > lead_time[i] ? orders[i, t - lead_time[i]] : 0.0) for i in skus) for
            t in 1:T
        ]
        base = sum(arrivals) / T * rand(rng, Uniform(1.05, 1.3))
        for t in 1:T
            receiving_capacity[z, t] = max(base, arrivals[t] * 1.1, 1.0)
        end
    end

    witness = nothing
    certificate = nothing
    if feasibility_status == feasible
        witness = ReplenishmentPlanWitness(orders, stock)
    else
        # The vendor whose service-level SKUs weigh most on its allocation.
        req = zeros(V)
        avail = [sum(vendor_capacity[v, t] for t in 1:T if any(vendor[i] == v && t <= T - lead_time[i] for i in 1:N); init=0.0) for v in 1:V]
        for v in 1:V
            skus = [i for i in 1:N if vendor[i] == v && fill_rate[i] > 0]
            req[v] = _replenishment_required(demand, pipeline, initial_inventory, fill_rate, skus)
        end
        vstar = argmax(req ./ max.(avail, 1e-9))
        if req[vstar] <= 0.0
            # No service-level SKU at any vendor yet: put the largest vendor's
            # SKUs under contract.
            vstar = argmax([count(==(v), vendor) for v in 1:V])
            for i in findall(==(vstar), vendor)
                fill_rate[i] = rand(rng, Uniform(0.9, 0.98))
            end
        end
        skus = [i for i in 1:N if vendor[i] == vstar && fill_rate[i] > 0]
        required = _replenishment_required(demand, pipeline, initial_inventory, fill_rate, skus)
        ratio = _inventory_scale_ratio(rng, feasibility_status)
        vendor_capacity[vstar, :] .*= required / (ratio * avail[vstar])
        if feasibility_status == infeasible
            certificate = VendorAllocationCertificate(vstar, skus, required, required / ratio)
        end
    end

    return InventoryProblem(
        N,
        T,
        V,
        n_zones,
        vendor,
        zone,
        lead_time,
        demand,
        pipeline,
        initial_inventory,
        fill_rate,
        volume,
        pallets,
        unit_cost,
        holding_cost,
        lost_sale_cost,
        vendor_capacity,
        zone_capacity,
        receiving_capacity,
        witness,
        certificate,
        feasibility_status,
    )
end

"""
    build_model(prob::InventoryProblem)

Build the multi-SKU replenishment LP. Deterministic. Variables: `q[i, t]` for
`t <= n_periods - lead_time[i]`, `I[i, t]`, `u[i, t]`.
"""
function build_model(prob::InventoryProblem)
    model = Model()
    N, T = prob.n_skus, prob.n_periods
    @variable(model, q[i = 1:N, t = 1:(T - prob.lead_time[i])] >= 0)
    @variable(model, I[1:N, 1:T] >= 0)
    @variable(model, 0 <= u[i = 1:N, t = 1:T; prob.demand[i, t] > 0] <= prob.demand[i, t])

    for i in 1:N, t in 1:T
        L = prob.lead_time[i]
        expr = AffExpr(prob.pipeline[i, t] + (t == 1 ? prob.initial_inventory[i] : 0.0))
        t > 1 && add_to_expression!(expr, 1.0, I[i, t - 1])
        t > L && add_to_expression!(expr, 1.0, q[i, t - L])
        prob.demand[i, t] > 0 && add_to_expression!(expr, 1.0, u[i, t])
        add_to_expression!(expr, -1.0, I[i, t])
        @constraint(model, expr == prob.demand[i, t])
    end
    for i in 1:N
        prob.fill_rate[i] > 0 || continue
        @constraint(
            model,
            sum(u[i, t] for t in 1:T if prob.demand[i, t] > 0) <= (1 - prob.fill_rate[i]) * sum(prob.demand[i, :])
        )
    end
    vendor_skus = [findall(==(v), prob.vendor) for v in 1:prob.n_vendors]
    for v in 1:prob.n_vendors, t in 1:T
        skus = [i for i in vendor_skus[v] if t <= T - prob.lead_time[i]]
        isempty(skus) && continue
        @constraint(model, sum(q[i, t] for i in skus) <= prob.vendor_capacity[v, t])
    end
    zone_skus = [findall(==(z), prob.zone) for z in 1:prob.n_zones]
    for z in 1:prob.n_zones
        isempty(zone_skus[z]) && continue
        for t in 1:T
            @constraint(model, sum(prob.volume[i] * I[i, t] for i in zone_skus[z]) <= prob.zone_capacity[z])
            arriving = [i for i in zone_skus[z] if t > prob.lead_time[i]]
            isempty(arriving) && continue
            @constraint(
                model,
                sum(prob.pallets[i] * q[i, t - prob.lead_time[i]] for i in arriving) <=
                    prob.receiving_capacity[z, t]
            )
        end
    end

    @objective(
        model,
        Min,
        sum(prob.unit_cost[i] * q[i, t] for i in 1:N for t in 1:(T - prob.lead_time[i])) +
        sum(prob.holding_cost[i] * I[i, t] for i in 1:N, t in 1:T) +
        sum(prob.lost_sale_cost[i] * u[i, t] for i in 1:N, t in 1:T if prob.demand[i, t] > 0)
    )
    return model
end

register_variant(
    :inventory,
    :standard,
    InventoryProblem,
    "Multi-SKU distribution-center replenishment LP: SKU stock chains with vendor lead times, lost sales and service-level rows, coupled by vendor allocation, zone storage, and receiving capacity rows, with a planted order-up-to plan and a vendor-allocation certificate";
    default=true,
    tags=[:logistics, :staircase, :block_angular],
)
