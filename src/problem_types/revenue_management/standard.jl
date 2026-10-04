using JuMP
using Random
using Distributions

"""
Fare classes from full fare to the deepest discount: fare multiplier on the
market's base fare and a quality term (fewer restrictions are more attractive
at equal price).
"""
const _RM_FARE_CLASSES = (
    (name=:Y, fare=2.6, quality=0.6),
    (name=:B, fare=1.9, quality=0.4),
    (name=:M, fare=1.45, quality=0.2),
    (name=:Q, fare=1.1, quality=0.0),
    (name=:V, fare=0.8, quality=-0.2),
)

"""Seat counts of the aircraft types used to size flights."""
const _RM_FLEET = (50.0, 76.0, 100.0, 150.0, 180.0, 220.0, 300.0)

"""A scheduled flight: one operation of a route on a day in a departure bank."""
struct RMFlight
    origin::Int
    destination::Int
    day::Int
    bank::Int
end

"""An origin–destination market on one departure day (one MNL choice set)."""
struct RMMarket
    origin::Int
    destination::Int
    day::Int
end

"""A sellable product: an itinerary (one or two flights) in a fare class."""
struct RMChoiceProduct
    market::Int
    flights::Vector{Int}
    fare_class::Symbol
end

"""
Planted feasible point: in every market-day `m` the airline offers all
products and sells the fraction `offer_fraction[m]` (≤ 1) of the MNL
full-offer sales, `sales[j] = θ_m Λ_m v_j / (V_m + v0_m)`; `no_purchase[m]` closes
the market balance. With `θ ≤ 1` every sales-based scale row holds (the check
reduces to `θ ≤ 1`), capacities were sized above the plan's flight loads, and
contract minimum loads below them.
"""
struct RMChoiceWitness
    sales::Vector{Float64}
    no_purchase::Vector{Float64}
    offer_fraction::Vector{Float64}
end

"""
Unattainable minimum-load contract. For every market-day `m` using `flight`,
let `S` be its products on the flight with total attraction `V_S`; the market's
scale rows `x_j ≤ (v_j / v0) x0` (summed over `S`) and its balance row
`Σ x + x0 = Λ` give `Σ_{j∈S} x_j ≤ Λ V_S / (V_S + v0) = market_bounds[k]`.
Summing over the markets bounds the flight's load by `sellable_bound`, below the
contracted `min_load` by `margin`. The proof combines the flight's min-load row
with the balance and scale rows of every market feeding it, so no single row
(or bound) reveals it to presolve.
"""
struct RMChoiceCertificate
    flight::Int
    min_load::Float64
    markets::Vector{Int}
    market_bounds::Vector{Float64}
    sellable_bound::Float64
    margin::Float64
end

"""
    RevenueManagementProblem <: ProblemGenerator

Choice-based network revenue management: the sales-based linear program (SBLP)
of Gallego, Ratliff & Shebalov (2015) for a multi-hub airline schedule with
multinomial-logit (MNL) customer choice.

# Overview

Airports (a few hubs and many spokes) are linked by spoke–hub and hub–hub
routes flown in daily departure banks over several days. Each origin–destination
market on each day is one MNL choice set whose products are its nonstop and
one-stop itineraries (connecting at a hub in the same bank) in several fare
classes; product attraction falls with price, stops, and distance from the
market's preferred departure bank. Decision variables are expected sales
`sales[j] ≥ 0` per product and no-purchase volume `no_purchase[m] ≥ 0` per
market-day. Constraints:

  - market balance: `Σ_{j∈m} sales[j] + no_purchase[m] = Λ_m`;
  - sales-based scale rows (one per product): `sales[j] ≤ (v_j / v0_m) no_purchase[m]`,
    which make the LP equivalent to the choice-based deterministic LP;
  - flight capacity: `Σ_{j uses f} sales[j] ≤ capacity[f]`;
  - minimum-load contracts on a few flights (charter / public-service
    guarantees): `Σ_{j uses f} sales[j] ≥ min_load[f]`.

The objective maximizes expected revenue. Unlike the independent-demand DLP
(whose same-itinerary fare-class columns are parallel and collapse under
presolve), every column here has its own scale row, and rows grow one-for-one
with products.

# Feasibility control

  - `feasible`: contracts at `U(0.70, 0.95)` of the planted plan's flight loads;
    stores [`RMChoiceWitness`](@ref).
  - `infeasible`: one contracted flight's minimum load is set `U(5%, 12%)` above
    the most its markets can sell under the choice model (the aircraft is
    up-gauged if needed so the contract fits the cabin); stores
    [`RMChoiceCertificate`](@ref). Discovering it needs simplex work.
  - `unknown`: contracts at an instance-wide tightness `U(0.9, 2.0)` times the
    planted plan's flight loads (up-gauging the aircraft when a contract would
    exceed 97% of the cabin): below one the plan meets them; above it, whether
    all contracts can be met together under the choice model, the shared
    connecting demand, and the other flights' capacities is left to the
    instance.

# Size

Variables are exactly `n_products + n_markets`; markets (in gravity-weighted
random order) and their day copies are added until the target is reached, the
last market-day trimmed to land exactly on it (targets below 2 give 2). Rows
are `n_products + n_markets + n_flights + n_contracts`.
"""
struct RevenueManagementProblem <: ProblemGenerator
    n_airports::Int
    n_hubs::Int
    n_days::Int
    n_banks::Int
    airport_location::Vector{Tuple{Float64, Float64}}
    flights::Vector{RMFlight}
    capacity::Vector{Float64}
    markets::Vector{RMMarket}
    market_size::Vector{Float64}
    no_purchase_attraction::Vector{Float64}
    products::Vector{RMChoiceProduct}
    fare::Vector{Float64}
    attraction::Vector{Float64}
    contracted_flights::Vector{Int}
    min_load::Vector{Float64}
    feasible_witness::Union{Nothing, RMChoiceWitness}
    infeasibility_certificate::Union{Nothing, RMChoiceCertificate}
    feasibility_status::FeasibilityStatus
end

"""
    _rm_dimensions(target) -> (days, fare_classes, banks, hubs, spokes)

Schedule dimensions for a target: more days, fare classes, banks, and hubs as
the target grows, and enough spokes that the candidate markets comfortably
exceed the target.
"""
function _rm_dimensions(target::Int)
    T = max(target, 2)
    days = clamp(round(Int, 2.3 * log10(T) - 5), 1, 14)
    classes = T < 300 ? 3 : (T < 5_000 ? 4 : 5)
    banks = T < 2_000 ? 2 : (T < 50_000 ? 3 : 4)
    hubs = T < 3_000 ? 1 : (T < 100_000 ? 2 : 3)
    per_market = days * (2.0 * classes + 1)
    spokes = clamp(ceil(Int, 1.7 * sqrt(hubs * T / per_market)) + 2, 3, 600)
    return days, classes, banks, hubs, spokes
end

"""
    _rm_network(rng, hubs, spokes, banks)

Airports on a 2500 × 2500 km map (hubs central), each spoke served from its
nearest hub and, with probability 0.35, a second hub. Returns locations,
populations, the route set, and per-route operating banks.
"""
function _rm_network(rng::AbstractRNG, hubs::Int, spokes::Int, banks::Int)
    n = hubs + spokes
    location = Vector{Tuple{Float64, Float64}}(undef, n)
    population = zeros(Float64, n)
    for h in 1:hubs
        location[h] = (rand(rng, Uniform(700, 1800)), rand(rng, Uniform(700, 1800)))
        population[h] = rand(rng, LogNormal(log(4.0), 0.3))
    end
    for s in (hubs + 1):n
        location[s] = (rand(rng, Uniform(0, 2500)), rand(rng, Uniform(0, 2500)))
        population[s] = rand(rng, LogNormal(log(0.8), 0.7))
    end
    dist(a, b) = hypot(location[a][1] - location[b][1], location[a][2] - location[b][2])
    route_banks = Dict{Tuple{Int, Int}, Vector{Int}}()
    for h in 1:hubs, g in 1:hubs
        h == g && continue
        route_banks[(h, g)] = collect(1:banks)
    end
    for s in (hubs + 1):n
        order = sortperm([dist(s, h) for h in 1:hubs])
        served = [order[1]]
        hubs >= 2 && rand(rng) < 0.35 && push!(served, order[2])
        frequency = clamp(round(Int, 1 + population[s] * banks / 2), 1, banks)
        for h in served
            ops = sort(randperm(rng, banks)[1:frequency])
            route_banks[(s, h)] = ops
            route_banks[(h, s)] = ops
        end
    end
    return location, population, route_banks
end

"""
    _rm_itineraries(o, d, hubs, route_banks) -> Vector{Vector{Tuple{Int,Int,Int}}}

Nonstop and one-stop (same-bank hub connection) itineraries from `o` to `d`,
each a list of `(origin, destination, bank)` legs; at most four, nonstops first.
"""
function _rm_itineraries(o::Int, d::Int, hubs::Int, route_banks)
    itineraries = Vector{Vector{Tuple{Int, Int, Int}}}()
    for b in get(route_banks, (o, d), Int[])
        push!(itineraries, [(o, d, b)])
    end
    for h in 1:hubs
        (h == o || h == d) && continue
        first_banks = get(route_banks, (o, h), Int[])
        second_banks = get(route_banks, (h, d), Int[])
        for b in first_banks
            b in second_banks && push!(itineraries, [(o, h, b), (h, d, b)])
        end
    end
    return itineraries[1:min(end, 4)]
end

"""
    _rm_sellable_bound(prob_attraction, no_purchase, market_size, market_of, flight_products, f)

Upper bound on flight `f`'s load implied by the market balance and scale rows:
`Σ_m Λ_m V_{m,f} / (V_{m,f} + v0_m)` over the market-days feeding the flight.
Returns `(bound, markets, per_market_bounds)`.
"""
function _rm_sellable_bound(attraction, no_purchase, market_size, market_of, flight_products, f)
    by_market = Dict{Int, Float64}()
    for j in flight_products[f]
        m = market_of[j]
        by_market[m] = get(by_market, m, 0.0) + attraction[j]
    end
    markets = sort!(collect(keys(by_market)))
    bounds = [market_size[m] * by_market[m] / (by_market[m] + no_purchase[m]) for m in markets]
    return sum(bounds), markets, bounds
end

function _rm_cabin(needed::Float64)
    for seats in _RM_FLEET
        seats >= needed && return seats
    end
    return 50.0 * ceil(needed / 50.0)
end

function RevenueManagementProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    target = max(target_variables, 2)
    days, n_classes, banks, hubs, spokes = _rm_dimensions(target)
    classes = n_classes == 5 ? _RM_FARE_CLASSES : _RM_FARE_CLASSES[[1:(n_classes - 1); 5]]

    local location, population, route_banks, candidates
    while true
        location, population, route_banks = _rm_network(rng, hubs, spokes, banks)
        n = hubs + spokes
        candidates = Tuple{Float64, Int, Int, Vector{Vector{Tuple{Int, Int, Int}}}}[]
        capacity_estimate = 0
        for o in 1:n, d in 1:n
            o == d && continue
            its = _rm_itineraries(o, d, hubs, route_banks)
            isempty(its) && continue
            od_distance = hypot(location[o][1] - location[d][1], location[o][2] - location[d][2])
            weight = population[o] * population[d] / max(od_distance, 150.0)^0.7
            # Efraimidis–Spirakis key: ascending order is a weighted random
            # permutation, so big city pairs tend to come first.
            key = -log(rand(rng)) / weight
            push!(candidates, (key, o, d, its))
            capacity_estimate += days * (length(its) * n_classes + 1)
        end
        capacity_estimate >= target && break
        spokes = ceil(Int, 1.4 * spokes)
    end
    sort!(candidates; by=first)
    n_airports = hubs + spokes
    dist(a, b) = hypot(location[a][1] - location[b][1], location[a][2] - location[b][2])

    # --- Markets, products, and the flights they use ---
    flight_index = Dict{Tuple{Int, Int, Int, Int}, Int}()
    flights = RMFlight[]
    markets = RMMarket[]
    market_size = Float64[]
    no_purchase = Float64[]
    products = RMChoiceProduct[]
    fare = Float64[]
    attraction = Float64[]
    day_factor = [rand(rng, Uniform(0.75, 1.25)) for _ in 1:days]
    total = 0
    for (_, o, d, its) in candidates
        total >= target && break
        distance = dist(o, d)
        base_fare = 60.0 + 0.11 * distance
        price_sensitivity = rand(rng, Uniform(1.2, 2.4))
        preferred_bank = rand(rng, 1:banks)
        gravity = 35.0 * population[o] * population[d] / max(distance / 1000, 0.3)^0.4
        for day in 1:days
            total >= target && break
            remaining = target - total
            full = length(its) * length(classes) + 1
            n_products_here = min(full, remaining) - 1
            n_products_here < 1 && (n_products_here = 1)
            push!(markets, RMMarket(o, d, day))
            m = length(markets)
            push!(market_size, gravity * day_factor[day] * rand(rng, LogNormal(0.0, 0.25)))
            V = 0.0
            count = 0
            for c in classes, it in its
                count >= n_products_here && break
                legs = Int[]
                for (a, b, bank) in it
                    key = (a, b, day, bank)
                    idx = get!(flight_index, key) do
                        push!(flights, RMFlight(a, b, day, bank))
                        length(flights)
                    end
                    push!(legs, idx)
                end
                stops = length(it) - 1
                price = base_fare * c.fare * (stops == 0 ? 1.0 : rand(rng, Uniform(0.85, 0.95)))
                bank_gap = abs(it[1][3] - preferred_bank) / banks
                utility =
                    c.quality - price_sensitivity * price / base_fare - 0.7 * stops -
                    0.8 * bank_gap + 0.15 * randn(rng)
                push!(products, RMChoiceProduct(m, legs, c.name))
                push!(fare, round(price; digits=2))
                push!(attraction, exp(utility))
                V += attraction[end]
                count += 1
            end
            purchase_probability = rand(rng, Uniform(0.35, 0.75))
            push!(no_purchase, V * (1 - purchase_probability) / purchase_probability)
            total += count + 1
        end
    end

    n_products = length(products)
    n_markets = length(markets)
    n_flights = length(flights)
    market_of = [p.market for p in products]
    flight_products = [Int[] for _ in 1:n_flights]
    for (j, p) in enumerate(products), f in p.flights
        push!(flight_products[f], j)
    end
    market_attraction = zeros(Float64, n_markets)
    for j in 1:n_products
        market_attraction[market_of[j]] += attraction[j]
    end

    # --- Planted plan and capacities ---
    offer_fraction = rand(rng, Uniform(0.45, 0.85), n_markets)
    sales = [
        offer_fraction[market_of[j]] * market_size[market_of[j]] * attraction[j] /
        (market_attraction[market_of[j]] + no_purchase[market_of[j]]) for j in 1:n_products
    ]
    no_purchase_sales = copy(market_size)
    for j in 1:n_products
        no_purchase_sales[market_of[j]] -= sales[j]
    end
    load = [sum((sales[j] for j in flight_products[f]); init=0.0) for f in 1:n_flights]
    capacity = [_rm_cabin(load[f] / rand(rng, Uniform(0.85, 0.97))) for f in 1:n_flights]

    # --- Minimum-load contracts ---
    n_contracts = max(1, round(Int, 0.06 * n_flights))
    contracted = sort(randperm(rng, n_flights)[1:n_contracts])
    min_load = zeros(Float64, n_flights)
    witness = nothing
    certificate = nothing
    if feasibility_status == unknown
        # One instance-level contract tightness relative to the planted loads:
        # below 1 the planted plan meets every contract; above it, the
        # contracts compete for shared connecting demand and capacity.
        tightness = rand(rng, Uniform(0.9, 2.0))
        for f in contracted
            min_load[f] = load[f] * tightness * rand(rng, Uniform(0.97, 1.03))
            if min_load[f] > 0.97 * capacity[f]
                capacity[f] = _rm_cabin(min_load[f] / rand(rng, Uniform(0.80, 0.95)))
            end
        end
    else
        for f in contracted
            min_load[f] = load[f] * rand(rng, Uniform(0.70, 0.95))
        end
    end
    if feasibility_status == feasible
        witness = RMChoiceWitness(sales, no_purchase_sales, offer_fraction)
    elseif feasibility_status == infeasible
        f = contracted[rand(rng, 1:n_contracts)]
        bound, feeding, bounds = _rm_sellable_bound(
            attraction, no_purchase, market_size, market_of, flight_products, f
        )
        min_load[f] = bound * rand(rng, Uniform(1.05, 1.12))
        if min_load[f] > 0.97 * capacity[f]
            capacity[f] = _rm_cabin(min_load[f] / rand(rng, Uniform(0.80, 0.95)))
        end
        certificate = RMChoiceCertificate(f, min_load[f], feeding, bounds, bound, min_load[f] - bound)
    end

    return RevenueManagementProblem(
        n_airports,
        hubs,
        days,
        banks,
        location,
        flights,
        capacity,
        markets,
        market_size,
        no_purchase,
        products,
        fare,
        attraction,
        contracted,
        min_load,
        witness,
        certificate,
        feasibility_status,
    )
end

function build_model(prob::RevenueManagementProblem)
    model = Model()
    n_products = length(prob.products)
    n_markets = length(prob.markets)
    n_flights = length(prob.flights)
    market_products = [Int[] for _ in 1:n_markets]
    flight_products = [Int[] for _ in 1:n_flights]
    for (j, p) in enumerate(prob.products)
        push!(market_products[p.market], j)
        for f in p.flights
            push!(flight_products[f], j)
        end
    end

    @variable(model, sales[1:n_products] >= 0)
    @variable(model, no_purchase[1:n_markets] >= 0)
    @objective(model, Max, sum(prob.fare[j] * sales[j] for j in 1:n_products))
    @constraint(
        model,
        market_balance[m = 1:n_markets],
        sum(sales[j] for j in market_products[m]) + no_purchase[m] == prob.market_size[m]
    )
    @constraint(
        model,
        scale[j = 1:n_products],
        sales[j] -
        prob.attraction[j] / prob.no_purchase_attraction[prob.products[j].market] *
        no_purchase[prob.products[j].market] <= 0
    )
    @constraint(
        model,
        flight_capacity[f = 1:n_flights],
        sum(sales[j] for j in flight_products[f]) <= prob.capacity[f]
    )
    @constraint(
        model,
        contract[f in prob.contracted_flights],
        sum(sales[j] for j in flight_products[f]) >= prob.min_load[f]
    )
    return model
end

register_variant(
    :revenue_management,
    :standard,
    RevenueManagementProblem,
    "Choice-based network revenue management (sales-based LP with MNL choice) on a multi-hub banked airline schedule over several days, with flight capacities and minimum-load contracts";
    default=true,
    tags=[:finance, :block_angular],
)
