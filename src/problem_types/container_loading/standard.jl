using JuMP
using Random

"""
    ContainerLoadingWitness

Planted loading for a `feasible` instance: `assignment[i]` is the container of
consignment `i`. Every used container respects its payload, volume and floor
(pallet-position) limits in plain arithmetic.
"""
struct ContainerLoadingWitness
    assignment::Vector{Int}
end

"""
    ContainerOverloadCertificate

Relaxation-valid infeasibility proof: each consignment is fully assigned
(equality rows) and container `b` holds at most `capacity[d, b] * used[b] <=
capacity[d, b]` of dimension `dimension`, so the fleet can carry at most
`fleet_capacity` while the consignments need `total_requirement >= 1.04 *
fleet_capacity`. Multipliers: the requirement on each assignment row, 1 on
each capacity row of that dimension, the capacity on each `used <= 1` bound.
"""
struct ContainerOverloadCertificate
    dimension::Int
    total_requirement::Float64
    fleet_capacity::Float64
end

"""
Container catalogue: (name, payload kg, volume m^3, pallet positions, relative
cost). Real ISO dry/reefer box figures (rounded).
"""
const CONTAINER_TYPES = (
    (name="20ft", payload=28_200.0, volume=33.2, pallets=11.0, cost=1.00),
    (name="40ft", payload=26_700.0, volume=67.7, pallets=24.0, cost=1.45),
    (name="40ft_HC", payload=26_500.0, volume=76.4, pallets=24.0, cost=1.55),
    (name="45ft_HC", payload=27_700.0, volume=86.0, pallets=27.0, cost=1.75),
    (name="40ft_reefer", payload=27_400.0, volume=59.3, pallets=23.0, cost=2.10),
)

"""
    ContainerLoadingProblem <: ProblemGenerator

Multi-container loading: assign palletised consignments to a heterogeneous
fleet of ISO containers under payload, volume and floor-position limits,
minimising the cost of the containers used.

# Overview

Consignments have 1-6 pallets, a volume of 1.1-1.9 m^3 per pallet (stacked
height) and a density drawn from a light/bulky vs dense goods mixture
(lognormal around 140 or 520 kg/m^3), so some are weight-critical and others
volume- or floor-critical. Containers are drawn from `CONTAINER_TYPES` (a
20 ft box carries as much weight as a 45 ft high-cube but a third of the
volume), with a type-dependent cost.

```text
min  sum_b cost_b used_b
s.t. sum_b assign_ib = 1                         for every consignment
     assign_ib <= used_b                         for every pair (strong linking)
     sum_i req_id assign_ib <= cap_bd used_b     for every container, d in (kg, m^3, pallets)
     binary
```

Sizing: `n_items * n_containers + n_containers` columns with 4-8 consignments
per fleet slot, chosen to land as close to the target as possible (the old
search always picked 2 containers, so each assignment row was a doubleton that
presolve eliminated — half the columns — and 50k instances were degenerate
25,000 x 2 LPs that hit the time limit).

# Feasibility

  - `feasible`: first-fit-decreasing loading over the fleet (consignments
    shrunk 8% at a time until it fits) — `ContainerLoadingWitness`.
  - `infeasible`: densities raised so total weight is `U(1.04, 1.12)` times the
    fleet payload (`ContainerOverloadCertificate`).
  - `unknown`: weight and volume utilisation of the whole fleet drawn
    independently in `U(0.86, 1.01)`; whether consignments can be split so that
    every container respects all three limits at once is decided by the LP.
"""
struct ContainerLoadingProblem <: ProblemGenerator
    n_items::Int
    n_containers::Int
    container_type::Vector{Int}
    item_requirements::Matrix{Float64}   # 3 x n_items (kg, m^3, pallets)
    capacities::Matrix{Float64}          # 3 x n_containers
    costs::Vector{Float64}
    feasibility_status::FeasibilityStatus
    feasible_witness::Union{Nothing, ContainerLoadingWitness}
    infeasibility_certificate::Union{Nothing, ContainerOverloadCertificate}
end

"""
    container_loading_dimensions(target) -> (n_items, n_containers)

Closest `n * b + b` to `target` with `4 <= n / b <= 8` (small targets relax
the ratio).
"""
function container_loading_dimensions(target::Int)
    target = max(target, 6)
    best = (typemax(Int), Inf, 2, 2)
    for b in 2:max(2, isqrt(target))
        n = max(2, round(Int, (target - b) / b))
        ratio = n / b
        penalty = ratio < 4 ? 4 - ratio : (ratio > 8 ? ratio - 8 : 0.0)
        key = (abs(n * b + b - target), penalty)
        if key < (best[1], best[2])
            best = (key[1], key[2], n, b)
        end
    end
    # Prefer realistic ratios when they cost at most 1% of the target.
    for b in 2:max(2, isqrt(target))
        n = max(2, round(Int, (target - b) / b))
        if 4 <= n / b <= 8 && abs(n * b + b - target) <= max(best[1], 0.01 * target)
            return n, b
        end
    end
    return best[3], best[4]
end

function _container_ffd(req::Matrix{Float64}, cap::Matrix{Float64})
    n = size(req, 2)
    nb = size(cap, 2)
    score = [maximum(req[d, i] / maximum(cap[d, :]) for d in 1:3) for i in 1:n]
    order = sortperm(score; rev=true)
    load = zeros(3, nb)
    assignment = zeros(Int, n)
    for i in order
        placed = false
        for b in 1:nb
            if all(load[d, b] + req[d, i] <= cap[d, b] for d in 1:3)
                load[:, b] .+= req[:, i]
                assignment[i] = b
                placed = true
                break
            end
        end
        placed || return nothing
    end
    return assignment
end

function ContainerLoadingProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    n_items, n_containers = container_loading_dimensions(target_variables)

    # Fleet: mostly 40 ft boxes, some 20 ft, high-cubes and reefers.
    type_weights = [0.25, 0.30, 0.25, 0.10, 0.10]
    cdf = cumsum(type_weights)
    container_type = [min(searchsortedfirst(cdf, rand(rng)), 5) for _ in 1:n_containers]
    sort!(container_type)
    capacities = zeros(3, n_containers)
    costs = zeros(n_containers)
    base_cost = 1500.0 + 1000.0 * rand(rng)
    for b in 1:n_containers
        t = CONTAINER_TYPES[container_type[b]]
        capacities[:, b] .= (t.payload, t.volume, t.pallets)
        costs[b] = base_cost * t.cost * (0.95 + 0.1 * rand(rng))
    end

    # Consignments.
    req = zeros(3, n_items)
    for i in 1:n_items
        pallets = Float64(rand(rng, 1:6))
        vol = pallets * (1.1 + 0.8 * rand(rng))
        density = rand(rng) < 0.55 ? 140.0 * exp(0.35 * randn(rng)) : 520.0 * exp(0.3 * randn(rng))
        req[:, i] .= (vol * density, vol, pallets)
    end
    # Scale volume and weight so the fleet is used at a realistic level
    # (consignments are generated per fleet slot, so normalise once).
    fleet = vec(sum(capacities; dims=2))
    for d in 1:2
        util = sum(req[d, :]) / fleet[d]
        req[d, :] .*= (0.70 + 0.12 * rand(rng)) / util
    end
    req[3, :] .*= min(1.0, 0.75 * fleet[3] / sum(req[3, :]))

    feasible_witness = nothing
    infeasibility_certificate = nothing
    if feasibility_status == feasible
        assignment = _container_ffd(req, capacities)
        while assignment === nothing
            req .*= 0.92
            assignment = _container_ffd(req, capacities)
        end
        feasible_witness = ContainerLoadingWitness(assignment)
    elseif feasibility_status == infeasible
        factor = (1.04 + 0.08 * rand(rng)) * fleet[1] / sum(req[1, :])
        req[1, :] .*= factor
        total = sum(req[1, :])
        infeasibility_certificate = ContainerOverloadCertificate(1, total, fleet[1])
        total >= 1.04 * fleet[1] * (1 - 1e-12) || error("container_loading: certificate margin lost")
    else
        for d in 1:2
            req[d, :] .*= (0.86 + 0.15 * rand(rng)) * fleet[d] / sum(req[d, :])
        end
    end

    return ContainerLoadingProblem(
        n_items,
        n_containers,
        container_type,
        req,
        capacities,
        costs,
        feasibility_status,
        feasible_witness,
        infeasibility_certificate,
    )
end

function build_model(prob::ContainerLoadingProblem)
    model = Model()
    I = 1:prob.n_items
    B = 1:prob.n_containers
    @variable(model, assign[I, B], Bin)
    @variable(model, used[B], Bin)
    @objective(model, Min, sum(prob.costs[b] * used[b] for b in B))
    @constraint(model, assignment[i in I], sum(assign[i, b] for b in B) == 1)
    @constraint(model, linking[i in I, b in B], assign[i, b] <= used[b])
    @constraint(
        model,
        capacity[d in 1:3, b in B],
        sum(prob.item_requirements[d, i] * assign[i, b] for i in I) <= prob.capacities[d, b] * used[b]
    )
    return model
end

register_variant(
    :container_loading,
    :standard,
    ContainerLoadingProblem,
    "Loading palletised consignments into a heterogeneous ISO container fleet under payload, volume and floor limits";
    default=true,
    tags=[:logistics, :partitioning, :big_m],
)
