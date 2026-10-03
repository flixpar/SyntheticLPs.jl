using JuMP
using Random
using Distributions

# Shared multi-echelon (plant -> distribution center -> customer), multi-product,
# multi-period network used by the `standard`, `carbon`, and `multi_product`
# supply-chain variants. Each variant switches on a different coupling family on
# top of the same flow-balance backbone:
#
#   standard       DC opening decisions with per-arc linking, rail/intermodal
#                  modal capacity per period
#   carbon         the standard network plus a horizon carbon budget over
#                  production, linehaul, and last-mile emissions
#   multi_product  many products, specialized plants with product-line
#                  capacities, and shared lane capacity across products
#                  (multicommodity bundle rows); no design decisions

"""Linehaul modes: per-unit-distance cost, fixed terminal cost, and CO2 per unit-km."""
const _SCN_MODES = (
    truck=(rate=0.10, terminal=0.0, emission=0.10),
    rail=(rate=0.055, terminal=6.0, emission=0.03),
    intermodal=(rate=0.07, terminal=4.0, emission=0.05),
)

"""
    SupplyChainNetwork

Data of a multi-echelon production–distribution network over `n_periods`.
A shipment's unit cost is `lane_cost[lane] + production_cost[plant, product]`.

Plants ship products they can make (`plant_products`) to distribution centers
over `lanes = (plant, dc, mode)`; DCs hold inventory and deliver to customers
over `arcs = (dc, customer)`; customers order the products in
`customer_products`. Shipment variables exist for every lane × capable product ×
period, delivery variables for every arc × ordered product × period, and
inventory variables for every DC × product × period. Capacity data that a
variant does not use is left at `Inf` / empty and generates no row.
"""
struct SupplyChainNetwork
    n_plants::Int
    n_dcs::Int
    n_customers::Int
    n_products::Int
    n_periods::Int
    plant_location::Vector{Tuple{Float64, Float64}}
    dc_location::Vector{Tuple{Float64, Float64}}
    customer_location::Vector{Tuple{Float64, Float64}}
    customer_region::Vector{Int}
    modes::Vector{Symbol}
    lanes::Vector{NTuple{3, Int}}
    production_cost::Matrix{Float64}
    lane_cost::Vector{Float64}
    lane_distance::Vector{Float64}
    plant_products::Vector{Vector{Int}}
    arcs::Vector{Tuple{Int, Int}}
    arc_cost::Vector{Float64}
    arc_distance::Vector{Float64}
    customer_products::Vector{Vector{Int}}
    demand::Array{Float64, 3}
    resource_use::Matrix{Float64}
    plant_capacity::Matrix{Float64}
    line_capacity::Matrix{Float64}
    lane_capacity::Vector{Float64}
    mode_capacity::Matrix{Float64}
    dc_throughput::Vector{Float64}
    dc_storage::Vector{Float64}
    initial_stock::Matrix{Float64}
    holding_cost::Matrix{Float64}
    dc_fixed_cost::Vector{Float64}
    design::Bool
end

"""
Planted plan of a requested-feasible supply-chain network instance: DC opening
values (`1.0`/`0.0`; all ones without design decisions), lane shipments and arc
deliveries aligned with [`_scn_ship_keys`](@ref) / [`_scn_deliver_keys`](@ref),
and DC inventories `stock[d, k, t]`. Every capacity in the network was sized
above this plan's usage, so it satisfies every row of the built model.
"""
struct SupplyChainNetworkWitness
    open::Vector{Float64}
    ship::Vector{Float64}
    stock::Array{Float64, 3}
    deliver::Vector{Float64}
end

"""
    _scn_ship_keys(net) -> Vector{NTuple{3,Int}}

`(lane, product, period)` for every shipment variable, in model order.
"""
function _scn_ship_keys(net::SupplyChainNetwork)
    keys = NTuple{3, Int}[]
    for (l, (p, _, _)) in enumerate(net.lanes), k in net.plant_products[p], t in 1:net.n_periods
        push!(keys, (l, k, t))
    end
    return keys
end

"""
    _scn_deliver_keys(net) -> Vector{NTuple{3,Int}}

`(arc, product, period)` for every delivery variable, in model order.
"""
function _scn_deliver_keys(net::SupplyChainNetwork)
    keys = NTuple{3, Int}[]
    for (a, (_, c)) in enumerate(net.arcs), k in net.customer_products[c], t in 1:net.n_periods
        push!(keys, (a, k, t))
    end
    return keys
end

"""
    _scn_num_variables(net)

Exact variable count: DC opening decisions (design only) + shipments + DC
inventories + deliveries.
"""
function _scn_num_variables(net::SupplyChainNetwork)
    T = net.n_periods
    ships = sum(length(net.plant_products[l[1]]) for l in net.lanes) * T
    delivers = sum(length(net.customer_products[a[2]]) for a in net.arcs) * T
    return (net.design ? net.n_dcs : 0) + ships + net.n_dcs * net.n_products * T + delivers
end

_scn_distance(a::Tuple{Float64, Float64}, b::Tuple{Float64, Float64}) =
    hypot(a[1] - b[1], a[2] - b[2])

"""
    _scn_shape(target, variant) -> (n_products, n_periods, product_share)

Products, periods, and the expected share of products each customer orders.
Periods grow slowly with the target (4 at 1k, 6 at 10k, 8 at 100k);
`multi_product` carries 2–12 products of which customers order about 60%, the
other variants one to three products ordered by everyone.
"""
function _scn_shape(target::Int, variant::Symbol)
    lt = log10(max(target, 10))
    T = clamp(round(Int, 1.5 * lt), 2, 10)
    if variant == :multi_product
        K = clamp(round(Int, 2 * lt - 1), 2, 12)
        share = 0.6
    else
        K = clamp(round(Int, lt - 1), 1, 3)
        share = 1.0
    end
    return K, T, share
end

"""
    _scn_instance(rng, target, variant) -> (net, witness, regions)

Sample a network sized to `target` variables and plant a feasible plan.
Capacities are sized above the plan's usage; the caller then applies the
status-specific changes. `regions[r]` lists the customers of region `r`.
"""
function _scn_instance(rng::AbstractRNG, target_variables::Int, variant::Symbol)
    target = max(target_variables, 30)
    K, T, share = _scn_shape(target, variant)
    design = variant != :multi_product
    modes = if variant == :multi_product
        [:truck]
    elseif variant == :carbon || target >= 2_000
        [:truck, :rail, :intermodal]
    else
        [:truck, :rail]
    end

    # --- Customers: enough that 2-lane-per-customer delivery covers ~60% of the
    # target and 4-5 lanes cover well above it; extra arcs fill the rest. ---
    # Fixed-point sizing: DCs and plants grow with customers; the DC stock block
    # and the linehaul shipments are subtracted before solving for customers at
    # an average of three delivery arcs each (arcs range over 2..5 per customer).
    per_arc = max(1.0, share * K) * T
    lanes_per_dc = 2.5 * (1 + 0.5 * (length(modes) - 1))
    plant_share = variant == :multi_product ? 0.45 : 1.0
    C, D, P = 2, 2, 2
    for _ in 1:6
        D = clamp(round(Int, C / 9) + (C >= 6 ? 2 : 0), 2, 400)
        P = clamp(round(Int, D / 3), 2, 150)
        dense = (design ? D : 0) + D * K * T + D * lanes_per_dc * max(1.0, plant_share * K) * T
        C = max(2, round(Int, (target - dense) / (3.0 * per_arc)))
    end
    n_regions = clamp(round(Int, sqrt(C) / 2), 2, 30)

    centers = [(rand(rng, Uniform(8, 92)), rand(rng, Uniform(8, 92))) for _ in 1:n_regions]
    region_weight = rand(rng, Dirichlet(fill(1.5, n_regions)))
    customer_region = [rand(rng, Categorical(region_weight)) for _ in 1:C]
    customer_location = [
        (
            clamp(centers[r][1] + rand(rng, Normal(0, 6)), 0, 100),
            clamp(centers[r][2] + rand(rng, Normal(0, 6)), 0, 100),
        ) for r in customer_region
    ]
    dc_location = Vector{Tuple{Float64, Float64}}(undef, D)
    for d in 1:D
        r = d <= n_regions ? d : rand(rng, Categorical(region_weight))
        dc_location[d] = (
            clamp(centers[r][1] + rand(rng, Normal(0, 9)), 0, 100),
            clamp(centers[r][2] + rand(rng, Normal(0, 9)), 0, 100),
        )
    end
    plant_location = [(rand(rng, Uniform(0, 100)), rand(rng, Uniform(0, 100))) for _ in 1:P]

    # --- Products: plants are specialized; each product has >= 1 capable plant ---
    plant_products = if variant == :multi_product
        lists = [Int[] for _ in 1:P]
        for p in 1:P, k in 1:K
            rand(rng) < 0.45 && push!(lists[p], k)
        end
        for k in 1:K
            any(k in lists[p] for p in 1:P) || push!(lists[rand(rng, 1:P)], k)
        end
        for p in 1:P
            isempty(lists[p]) && push!(lists[p], rand(rng, 1:K))
            sort!(lists[p])
        end
        lists
    else
        [collect(1:K) for _ in 1:P]
    end
    customer_products = [
        share >= 1.0 ? collect(1:K) : sort(randperm(rng, K)[1:clamp(rand(rng, Binomial(K, share)), 1, K)]) for
        _ in 1:C
    ]

    # --- Demand: customer scale x product mix x seasonality x noise ---
    customer_scale = rand(rng, LogNormal(log(40.0), 0.6), C)
    product_scale = rand(rng, Uniform(0.5, 1.5), K)
    amplitude = rand(rng, Uniform(0.05, 0.35), K)
    phase = rand(rng, Uniform(0, 2π), K)
    demand = zeros(Float64, C, K, T)
    for c in 1:C, k in customer_products[c], t in 1:T
        season = 1 + amplitude[k] * sin(2π * (t - 1) / T + phase[k])
        demand[c, k, t] = customer_scale[c] * product_scale[k] * season * rand(rng, LogNormal(0.0, 0.12))
    end

    # --- Last-mile arcs: two nearest DCs per customer, then nearest extras ---
    dc_order = [sortperm([_scn_distance(dc_location[d], customer_location[c]) for d in 1:D]) for c in 1:C]
    arcs = Tuple{Int, Int}[]
    extras = Tuple{Float64, Int, Int}[]
    for c in 1:C
        for r in 1:min(2, D)
            push!(arcs, (dc_order[c][r], c))
        end
        for r in 3:min(5, D)
            d = dc_order[c][r]
            push!(extras, (_scn_distance(dc_location[d], customer_location[c]) * rand(rng, Uniform(0.8, 1.25)), d, c))
        end
    end

    # --- Linehaul lanes: each DC from its 2-3 nearest plants plus the nearest
    # capable plant of any product not yet covered; mode availability by distance ---
    lanes = NTuple{3, Int}[]
    lane_distance = Float64[]
    for d in 1:D
        order = sortperm([_scn_distance(plant_location[p], dc_location[d]) for p in 1:P])
        chosen = order[1:min(P, rand(rng, 2:3))]
        covered = Set{Int}(k for p in chosen for k in plant_products[p])
        for k in 1:K
            k in covered && continue
            p = first(q for q in order if k in plant_products[q])
            push!(chosen, p)
            union!(covered, plant_products[p])
        end
        for p in unique(chosen)
            dist = _scn_distance(plant_location[p], dc_location[d])
            for (mi, m) in enumerate(modes)
                available = m == :truck || (m == :rail && dist > 25 && rand(rng) < 0.6) ||
                            (m == :intermodal && dist > 18 && rand(rng) < 0.5)
                available || continue
                push!(lanes, (p, d, mi))
                push!(lane_distance, dist)
            end
        end
    end

    # Fill extra arcs until the exact variable count reaches the target.
    sort!(extras; by=first)
    base = (design ? D : 0) + sum(length(plant_products[l[1]]) for l in lanes) * T + D * K * T
    count = base + sum(length(customer_products[c]) for (_, c) in arcs) * T
    for (_, d, c) in extras
        step = length(customer_products[c]) * T
        count + step / 2 > target && continue
        push!(arcs, (d, c))
        count += step
    end
    sort!(arcs; by=a -> (a[2], a[1]))
    arc_distance = [_scn_distance(dc_location[d], customer_location[c]) for (d, c) in arcs]

    # --- Costs ---
    production_cost = rand(rng, LogNormal(log(20.0), 0.2), P, K)
    lane_cost = [
        _SCN_MODES[modes[mi]].terminal +
        _SCN_MODES[modes[mi]].rate * lane_distance[i] * rand(rng, Uniform(0.9, 1.1)) for
        (i, (_, _, mi)) in enumerate(lanes)
    ]
    arc_cost = [1.5 + 0.18 * arc_distance[i] * rand(rng, Uniform(0.9, 1.1)) for i in eachindex(arcs)]
    holding_cost = rand(rng, Uniform(0.3, 1.0), D, K)
    resource_use = [rand(rng, Uniform(0.7, 1.4)) for _ in 1:P, _ in 1:K]

    # --- Planted plan ---
    is_open = trues(D)
    if design
        is_open .= [rand(rng) < 0.65 for _ in 1:D]
    end
    arcs_of = [Int[] for _ in 1:C]
    for (a, (_, c)) in enumerate(arcs)
        push!(arcs_of[c], a)
    end
    for c in 1:C
        any(is_open[arcs[a][1]] for a in arcs_of[c]) || (is_open[arcs[arcs_of[c][1]][1]] = true)
    end
    deliver_keys = NTuple{3, Int}[]
    for (a, (_, c)) in enumerate(arcs), k in customer_products[c], t in 1:T
        push!(deliver_keys, (a, k, t))
    end
    arc_weight = zeros(Float64, length(arcs))
    for c in 1:C
        total = 0.0
        for a in arcs_of[c]
            if is_open[arcs[a][1]]
                arc_weight[a] = rand(rng, Uniform(0.5, 1.5)) / (5 + arc_distance[a])
                total += arc_weight[a]
            end
        end
        for a in arcs_of[c]
            arc_weight[a] /= total
        end
    end
    deliver = [arc_weight[a] * demand[arcs[a][2], k, t] for (a, k, t) in deliver_keys]
    outflow = zeros(Float64, D, K, T)
    for (i, (a, k, t)) in enumerate(deliver_keys)
        outflow[arcs[a][1], k, t] += deliver[i]
    end
    cover = rand(rng, Uniform(0.1, 0.45), D)
    stock = zeros(Float64, D, K, T)
    initial_stock = zeros(Float64, D, K)
    inflow = zeros(Float64, D, K, T)
    for d in 1:D, k in 1:K
        initial_stock[d, k] = cover[d] * outflow[d, k, 1]
        for t in 1:T
            stock[d, k, t] = cover[d] * outflow[d, k, min(t + 1, T)]
            previous = t == 1 ? initial_stock[d, k] : stock[d, k, t - 1]
            inflow[d, k, t] = outflow[d, k, t] + stock[d, k, t] - previous
        end
    end
    ship_keys = NTuple{3, Int}[]
    for (l, (p, _, _)) in enumerate(lanes), k in plant_products[p], t in 1:T
        push!(ship_keys, (l, k, t))
    end
    lane_weight = [
        rand(rng, Uniform(0.5, 1.5)) * (variant == :carbon && modes[mi] != :truck ? 3.0 : 1.0) for
        (_, _, mi) in lanes
    ]
    weight_total = zeros(Float64, D, K)
    for (l, (p, d, _)) in enumerate(lanes), k in plant_products[p]
        weight_total[d, k] += lane_weight[l]
    end
    ship = [
        lane_weight[l] / weight_total[lanes[l][2], k] * inflow[lanes[l][2], k, t] for (l, k, t) in ship_keys
    ]

    # --- Capacities sized above the plan ---
    plant_use = zeros(Float64, P, T)
    line_use = zeros(Float64, P, K, T)
    lane_use = zeros(Float64, length(lanes), T)
    mode_use = zeros(Float64, length(modes), T)
    for (i, (l, k, t)) in enumerate(ship_keys)
        p, _, mi = lanes[l]
        plant_use[p, t] += resource_use[p, k] * ship[i]
        line_use[p, k, t] += ship[i]
        lane_use[l, t] += ship[i]
        mode_use[mi, t] += ship[i]
    end
    mean_plant = sum(plant_use) / (P * T)
    plant_capacity = [plant_use[p, t] * rand(rng, Uniform(1.08, 1.3)) + 0.02 * mean_plant for p in 1:P, t in 1:T]
    line_capacity = fill(Inf, P, K)
    lane_capacity = fill(Inf, length(lanes))
    mode_capacity = fill(Inf, length(modes), T)
    if variant == :multi_product
        for p in 1:P, k in plant_products[p]
            line_capacity[p, k] = maximum(line_use[p, k, :]) * rand(rng, Uniform(1.1, 1.35)) + 0.5
        end
        mean_lane = sum(lane_use) / max(length(lane_use), 1)
        for l in eachindex(lanes)
            lane_capacity[l] = maximum(lane_use[l, :]) * rand(rng, Uniform(1.1, 1.4)) + 0.05 * mean_lane
        end
    else
        mean_mode = sum(mode_use) / length(mode_use)
        for (mi, m) in enumerate(modes), t in 1:T
            m == :truck && continue
            mode_capacity[mi, t] = mode_use[mi, t] * rand(rng, Uniform(1.1, 1.3)) + 0.02 * mean_mode
        end
    end
    dc_out = [maximum(sum(outflow[d, :, t]) for t in 1:T) for d in 1:D]
    dc_stock = [maximum(sum(stock[d, :, t]) for t in 1:T) for d in 1:D]
    open_out = [dc_out[d] for d in 1:D if is_open[d]]
    open_stock = [dc_stock[d] for d in 1:D if is_open[d]]
    typical_out = sort(open_out)[cld(length(open_out), 2)]
    typical_stock = sort(open_stock)[cld(length(open_stock), 2)]
    dc_throughput = [
        is_open[d] ? dc_out[d] * rand(rng, Uniform(1.1, 1.35)) : typical_out * rand(rng, Uniform(0.6, 1.4)) for
        d in 1:D
    ]
    dc_storage = [
        is_open[d] ? dc_stock[d] * rand(rng, Uniform(1.15, 1.5)) + 1.0 :
        typical_stock * rand(rng, Uniform(0.6, 1.4)) + 1.0 for d in 1:D
    ]
    dc_fixed_cost = [dc_throughput[d] * T * rand(rng, Uniform(0.8, 2.0)) for d in 1:D]

    net = SupplyChainNetwork(
        P,
        D,
        C,
        K,
        T,
        plant_location,
        dc_location,
        customer_location,
        customer_region,
        modes,
        lanes,
        production_cost,
        lane_cost,
        lane_distance,
        plant_products,
        arcs,
        arc_cost,
        arc_distance,
        customer_products,
        demand,
        resource_use,
        plant_capacity,
        line_capacity,
        lane_capacity,
        mode_capacity,
        dc_throughput,
        dc_storage,
        initial_stock,
        holding_cost,
        dc_fixed_cost,
        design,
    )
    witness = SupplyChainNetworkWitness(Float64.(is_open), ship, stock, deliver)
    regions = [findall(==(r), customer_region) for r in 1:n_regions]
    return net, witness, regions
end

"""
    _scn_scale_capacities!(net, factor)

Scale every finite capacity of the network (plant, line, lane, mode, DC
throughput) by `factor` — a correlated network-wide supply condition used by
`unknown` instances.
"""
function _scn_scale_capacities!(net::SupplyChainNetwork, factor::Float64)
    net.plant_capacity .*= factor
    net.line_capacity .*= factor
    net.lane_capacity .*= factor
    net.mode_capacity .*= factor
    net.dc_throughput .*= factor
    return net
end

"""
    _scn_build_model(net) -> (model, ship, deliver)

Build the network LP/MIP backbone. Rows: plant resource capacity per period;
product-line capacity per plant/product/period and shared lane capacity per
lane/period (when finite); modal capacity per period (when finite); DC
inventory balance per DC/product/period; DC throughput and storage per period
(scaled by the opening decision under design); customer demand per
customer/product/period; and, under design, per-arc linking
`Σ_k deliver[a,k,t] ≤ (Σ_k demand[c,k,t]) open[d]` — the strong, disaggregated
form whose LP relaxation still prices DC opening.
"""
function _scn_build_model(net::SupplyChainNetwork)
    model = Model()
    P, D, C, K, T = net.n_plants, net.n_dcs, net.n_customers, net.n_products, net.n_periods
    ship_keys = _scn_ship_keys(net)
    deliver_keys = _scn_deliver_keys(net)
    nS, nD = length(ship_keys), length(deliver_keys)

    if net.design
        @variable(model, open[1:D], Bin)
    end
    @variable(model, ship[1:nS] >= 0)
    @variable(model, stock[1:D, 1:K, 1:T] >= 0)
    @variable(model, deliver[1:nD] >= 0)

    objective = AffExpr(0.0)
    for (i, (l, k, _)) in enumerate(ship_keys)
        add_to_expression!(objective, net.lane_cost[l] + net.production_cost[net.lanes[l][1], k], ship[i])
    end
    for (i, (a, _, _)) in enumerate(deliver_keys)
        add_to_expression!(objective, net.arc_cost[a], deliver[i])
    end
    for d in 1:D, k in 1:K, t in 1:T
        add_to_expression!(objective, net.holding_cost[d, k], stock[d, k, t])
    end
    if net.design
        for d in 1:D
            add_to_expression!(objective, net.dc_fixed_cost[d], open[d])
        end
    end
    @objective(model, Min, objective)

    plant_load = [AffExpr(0.0) for _ in 1:P, _ in 1:T]
    line_load = Dict{NTuple{3, Int}, AffExpr}()
    lane_load = [AffExpr(0.0) for _ in eachindex(net.lanes), _ in 1:T]
    mode_load = [AffExpr(0.0) for _ in eachindex(net.modes), _ in 1:T]
    inbound = [AffExpr(0.0) for _ in 1:D, _ in 1:K, _ in 1:T]
    for (i, (l, k, t)) in enumerate(ship_keys)
        p, d, mi = net.lanes[l]
        add_to_expression!(plant_load[p, t], net.resource_use[p, k], ship[i])
        if isfinite(net.line_capacity[p, k])
            add_to_expression!(get!(() -> AffExpr(0.0), line_load, (p, k, t)), 1.0, ship[i])
        end
        add_to_expression!(lane_load[l, t], 1.0, ship[i])
        add_to_expression!(mode_load[mi, t], 1.0, ship[i])
        add_to_expression!(inbound[d, k, t], 1.0, ship[i])
    end
    outbound = [AffExpr(0.0) for _ in 1:D, _ in 1:K, _ in 1:T]
    dc_out = [AffExpr(0.0) for _ in 1:D, _ in 1:T]
    received = [AffExpr(0.0) for _ in 1:C, _ in 1:K, _ in 1:T]
    arc_flow = [AffExpr(0.0) for _ in eachindex(net.arcs), _ in 1:T]
    for (i, (a, k, t)) in enumerate(deliver_keys)
        d, c = net.arcs[a]
        add_to_expression!(outbound[d, k, t], 1.0, deliver[i])
        add_to_expression!(dc_out[d, t], 1.0, deliver[i])
        add_to_expression!(received[c, k, t], 1.0, deliver[i])
        add_to_expression!(arc_flow[a, t], 1.0, deliver[i])
    end

    @constraint(model, plant_capacity[p = 1:P, t = 1:T], plant_load[p, t] <= net.plant_capacity[p, t])
    line_keys = sort!(collect(keys(line_load)))
    @constraint(model, line_capacity[key in line_keys], line_load[key] <= net.line_capacity[key[1], key[2]])
    finite_lanes = [l for l in eachindex(net.lanes) if isfinite(net.lane_capacity[l])]
    @constraint(model, lane_capacity[l in finite_lanes, t = 1:T], lane_load[l, t] <= net.lane_capacity[l])
    finite_modes = [mi for mi in eachindex(net.modes) if all(isfinite, net.mode_capacity[mi, :])]
    @constraint(
        model, mode_capacity[mi in finite_modes, t = 1:T], mode_load[mi, t] <= net.mode_capacity[mi, t]
    )
    @constraint(
        model,
        dc_balance[d = 1:D, k = 1:K, t = 1:T],
        (t == 1 ? net.initial_stock[d, k] : stock[d, k, t - 1]) + inbound[d, k, t] - outbound[d, k, t] ==
        stock[d, k, t]
    )
    gate(d) = net.design ? open[d] : 1.0
    @constraint(model, dc_throughput[d = 1:D, t = 1:T], dc_out[d, t] <= net.dc_throughput[d] * gate(d))
    @constraint(
        model, dc_storage[d = 1:D, t = 1:T], sum(stock[d, k, t] for k in 1:K) <= net.dc_storage[d] * gate(d)
    )
    demand_nodes = [(c, k, t) for c in 1:C for k in net.customer_products[c] for t in 1:T]
    @constraint(model, demand[n in demand_nodes], received[n...] >= net.demand[n...])
    if net.design
        @constraint(
            model,
            arc_linking[a in eachindex(net.arcs), t = 1:T],
            arc_flow[a, t] <=
            sum(net.demand[net.arcs[a][2], k, t] for k in net.customer_products[net.arcs[a][2]]) *
            open[net.arcs[a][1]]
        )
    end
    return model, ship, deliver
end
