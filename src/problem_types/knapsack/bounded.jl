using JuMP
using Random

"""
    BoundedAllocationWitness

Planted integer allocation for a `feasible` instance: `quantities[j]` is the
number of units on column `j` (aligned with `item_of` / `knapsack_of`). Each
item ships between its commitment and its stock, and each vehicle's capacity
is its planted load times a factor `>= 1.04`.
"""
struct BoundedAllocationWitness
    quantities::Vector{Int}
end

"""
    RegionalCommitmentCertificate

Relaxation-valid infeasibility proof. Every column of an item homed in region
`region` loads a vehicle of that region, and one unit of item `i` weighs at
least `min_unit_weight[i]` on any eligible vehicle, so meeting the region's
commitments loads at least

    committed_weight = sum_{i in region} commitment[i] * min_unit_weight[i]

while the region's vehicles hold `region_capacity <= committed_weight / 1.04`.
Farkas multipliers: 1 on each regional capacity row and `min_unit_weight[i]` on
the lower side of each regional item row.
"""
struct RegionalCommitmentCertificate
    region::Int
    items::Vector{Int}
    min_unit_weight::Vector{Float64}
    committed_weight::Float64
    region_capacity::Float64
end

"""
    BoundedKnapsackProblem <: ProblemGenerator

Bounded multiple knapsack with contracted minimum deliveries: allocate stock
lots (bounded unit counts) to capacity-limited vehicles, maximizing margin.

# Overview

Items (stock lots) carry `u_i in 1:12` units of a product with a lognormal unit
weight and unit price. Vehicles are grouped into `clamp(round(K / 40), 1, 6)`
regions; each item is homed in a region and may load `2-5` of its region's
vehicles (lanes). Column `(i, k)` ships `x_ik` units in `[0, u_i]` (general
integer; relaxed by default) with vehicle-specific unit weight
`w_i * kappa_k * noise` (packaging and handling differ by vehicle type) and
unit margin `price_i - lane_cost_k * w_i` (floored at 10% of the price).
About a third of the items have a contracted minimum delivery `l_i`.

```text
max  sum_{(i,k)} p_ik x_ik
s.t. l_i <= sum_k x_ik <= u_i             for every item (a ranged row when l_i > 0)
     sum_{(i,k)} w_ik x_ik <= C_k          for every vehicle
     0 <= x_ik <= u_i, integer
```

Each column sits in exactly two rows with non-unit weights, so the relaxation
is a generalized-assignment / generalized-flow LP (not totally unimodular).
The column count equals `target_variables` exactly; vehicles
`K = clamp(round(n / 28), 1, n)`, rows about `0.32 n`.

# Feasibility

  - `feasible`: each item ships a planted `round(u_i * U(0.3, 0.9))` units
    split over its lanes; vehicle capacity is the planted load times
    `U(1.04, 1.25)` plus a small allowance (capped below the full-stock load);
    commitments are at most the planted shipment. Witness:
    `BoundedAllocationWitness`.
  - `infeasible`: same instance, then the largest region's commitments are
    raised (round-robin, up to the stock) until their lightest-lane weight is
    `U(1.04, 1.12)` times the region's vehicle capacity (capacities are scaled
    down if every unit is already committed). Certificate:
    `RegionalCommitmentCertificate`.
  - `unknown`: a per-instance share `U(0.2, 0.9)` of items is committed at
    `U(0.5, 1.0)` of stock, and capacities are `U(0.3, 0.65)` of the full-stock
    load expected on each vehicle; the LP decides.
"""
struct BoundedKnapsackProblem <: ProblemGenerator
    n_items::Int
    n_knapsacks::Int
    n_regions::Int
    item_region::Vector{Int}
    knapsack_region::Vector{Int}
    stock::Vector{Int}
    commitment::Vector{Int}
    item_of::Vector{Int}
    knapsack_of::Vector{Int}
    unit_weight::Vector{Float64}
    unit_profit::Vector{Float64}
    capacity::Vector{Float64}
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, BoundedAllocationWitness}
    infeasibility_certificate::Union{Nothing, RegionalCommitmentCertificate}
end

function BoundedKnapsackProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    target_variables >= 1 ||
        throw(ArgumentError("target_variables must be positive (got $target_variables)"))
    rng = MersenneTwister(seed)
    n = target_variables

    K = clamp(round(Int, n / 28), 1, n)
    R = clamp(round(Int, K / 40), 1, 6)
    knapsack_region = [mod(k - 1, R) + 1 for k in 1:K]
    shuffle!(rng, knapsack_region)
    region_knapsacks = [findall(==(r), knapsack_region) for r in 1:R]
    kappa = [0.85 + 0.4 * rand(rng) for _ in 1:K]          # vehicle packaging factor
    lane_cost = [0.05 + 0.25 * rand(rng) for _ in 1:K]     # cost per unit weight

    # Items and their lanes until exactly n columns exist.
    item_region = Int[]
    stock = Int[]
    item_of = Int[]
    knapsack_of = Int[]
    base_weight = Float64[]
    price = Float64[]
    remaining = n
    while remaining > 0
        r = rand(rng, 1:R)
        pool = region_knapsacks[r]
        e = min(rand(rng, 2:5), length(pool), remaining)
        # Do not leave a single dangling column if the pool allows two.
        remaining - e == 1 && e > 1 && length(pool) >= 2 && (e -= 1)
        push!(item_region, r)
        i = length(item_region)
        push!(stock, rand(rng, 1:12))
        w = 8.0 * exp(0.6 * randn(rng))
        push!(base_weight, w)
        push!(price, w * (1.0 + 1.5 * rand(rng)) * exp(0.2 * randn(rng)))
        for k in Random.shuffle(rng, pool)[1:e]
            push!(item_of, i)
            push!(knapsack_of, k)
        end
        remaining -= e
    end
    N = length(stock)
    unit_weight = [base_weight[item_of[j]] * kappa[knapsack_of[j]] * exp(0.08 * randn(rng)) for j in 1:n]
    unit_profit = [
        max(0.1 * price[item_of[j]], price[item_of[j]] - lane_cost[knapsack_of[j]] * base_weight[item_of[j]] * 3.0)
        for j in 1:n
    ]
    cols_of = [Int[] for _ in 1:N]
    for j in 1:n
        push!(cols_of[item_of[j]], j)
    end
    # Expected full-stock load per vehicle (stock split evenly over lanes).
    full_load = zeros(K)
    for j in 1:n
        i = item_of[j]
        full_load[knapsack_of[j]] += unit_weight[j] * stock[i] / length(cols_of[i])
    end
    max_load = zeros(K)
    for j in 1:n
        max_load[knapsack_of[j]] += unit_weight[j] * stock[item_of[j]]
    end

    commitment = zeros(Int, N)
    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible || feasibility_status == infeasible
        q = zeros(Int, n)
        shipped = zeros(Int, N)
        for i in 1:N
            total = round(Int, stock[i] * (0.3 + 0.6 * rand(rng)))
            for _ in 1:total
                q[cols_of[i][rand(rng, 1:length(cols_of[i]))]] += 1
            end
            shipped[i] = total
        end
        load = zeros(K)
        for j in 1:n
            load[knapsack_of[j]] += unit_weight[j] * q[j]
        end
        allowance = 2.0 * 8.0
        capacity = [
            max(load[k], min(load[k] * (1.04 + 0.21 * rand(rng)) + allowance, 0.85 * max_load[k])) for k in 1:K
        ]
        for i in 1:N
            rand(rng) < 0.35 && (commitment[i] = floor(Int, shipped[i] * (0.5 + 0.5 * rand(rng))))
        end
        if feasibility_status == feasible
            feasible_witness = BoundedAllocationWitness(q)
        else
            rstar = argmax(r -> length(region_knapsacks[r]), 1:R)
            items = [i for i in 1:N if item_region[i] == rstar]
            minw = [minimum(unit_weight[j] for j in cols_of[i]) for i in items]
            region_cap = sum(capacity[k] for k in region_knapsacks[rstar])
            target = (1.04 + 0.08 * rand(rng)) * region_cap
            current = sum(commitment[items[t]] * minw[t] for t in eachindex(items); init=0.0)
            order = randperm(rng, length(items))
            progressed = true
            while current < target && progressed
                progressed = false
                for t in order
                    current >= target && break
                    i = items[t]
                    commitment[i] < stock[i] || continue
                    step = min(stock[i] - commitment[i], max(1, stock[i] ÷ 4))
                    commitment[i] += step
                    current += step * minw[t]
                    progressed = true
                end
            end
            if current < target
                scale = current / target
                for k in region_knapsacks[rstar]
                    capacity[k] *= scale
                end
                region_cap = sum(capacity[k] for k in region_knapsacks[rstar])
            end
            infeasibility_certificate = RegionalCommitmentCertificate(
                rstar, items, minw, current, region_cap
            )
        end
    else
        share = 0.2 + 0.7 * rand(rng)
        for i in 1:N
            rand(rng) < share && (commitment[i] = round(Int, stock[i] * (0.5 + 0.5 * rand(rng))))
        end
        rho = 0.30 + 0.35 * rand(rng)
        capacity = [full_load[k] * rho * (0.9 + 0.2 * rand(rng)) for k in 1:K]
    end

    return BoundedKnapsackProblem(
        N,
        K,
        R,
        item_region,
        knapsack_region,
        stock,
        commitment,
        item_of,
        knapsack_of,
        unit_weight,
        unit_profit,
        capacity,
        feasibility_status,
        feasible_witness,
        infeasibility_certificate,
    )
end

function build_model(prob::BoundedKnapsackProblem)
    model = Model()
    n = length(prob.item_of)
    @variable(model, 0 <= x[j = 1:n] <= prob.stock[prob.item_of[j]], Int)
    @objective(model, Max, sum(prob.unit_profit[j] * x[j] for j in 1:n))

    item_expr = [AffExpr() for _ in 1:prob.n_items]
    load = [AffExpr() for _ in 1:prob.n_knapsacks]
    for j in 1:n
        add_to_expression!(item_expr[prob.item_of[j]], 1.0, x[j])
        add_to_expression!(load[prob.knapsack_of[j]], prob.unit_weight[j], x[j])
    end
    for i in 1:prob.n_items
        if prob.commitment[i] > 0
            @constraint(model, prob.commitment[i] <= item_expr[i] <= prob.stock[i])
        elseif length(item_expr[i].terms) > 1
            # A single-lane item without a commitment is already bounded by
            # its column's upper bound; no redundant singleton row.
            @constraint(model, item_expr[i] <= prob.stock[i])
        end
    end
    for k in 1:prob.n_knapsacks
        isempty(load[k].terms) && continue
        @constraint(model, load[k] <= prob.capacity[k])
    end
    return model
end

register_variant(
    :knapsack,
    :bounded,
    BoundedKnapsackProblem,
    "Bounded multiple knapsack: allocate bounded stock lots to capacity-limited vehicles with lane-specific weights and contracted minimum deliveries",
)
