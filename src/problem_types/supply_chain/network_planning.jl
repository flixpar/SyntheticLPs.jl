using JuMP
using Random
using Distributions

const _NETWORK_PLANNING_PROFILES = (:regional_stable, :seasonal_prebuild, :disruption)

"""
Maximum accepted variable target for `supply_chain/network_planning`.

Each shipment coordinate is stored in an arc vector, two coefficient
dictionaries, JuMP's variable store, and several row expressions. Targets above
one million therefore require multi-gigabyte working sets in realistic Julia
sessions. Reject them before allocating rather than silently returning a much
smaller model.
"""
const MAX_NETWORK_PLANNING_VARIABLES = 1_000_000

struct NetworkPlanningWitness
    production::Array{Float64, 3}
    inventory::Array{Float64, 3}
    shipment::Dict{NTuple{4, Int}, Float64}
end

struct NetworkPlanningInfeasibilityCertificate
    product::Int
    period::Int
    demand::Float64
    supply_bound::Float64
    lane_bound::Float64
    upper_bound::Float64
    margin::Float64
end

struct NetworkPlanningDisruption
    period::Int
    plant::Int
    production_factor::Float64
    shipment_surcharge::Float64
end

struct NetworkPlanningNominalScenario
    supply_factor::Float64
    lane_factor::Float64
    minimum_local_service::Float64
end

"""
    SupplyChainNetworkPlanningProblem <: ProblemGenerator

Multi-period, multi-product supply-chain network-planning LP. Plants produce
specialized products under shared resource capacity, carry inventory between
periods, and ship on a sparse, period-specific plant/customer network.

Demand is an equality: there are no unmet-demand or backlog variables. This
prevents a nominally feasible request from being satisfied by service
shortfalls and prevents profitable over-shipment or disposal. `feasible_witness`
is populated only for a requested-feasible instance;
`infeasibility_certificate` only for a requested-infeasible instance; and
`nominal_scenario` only for an unknown-status sample. `build_model` is
deterministic.

The structural `profile` is one of:

  - `:regional_stable`: regionally clustered lanes and comparatively stable demand;
  - `:seasonal_prebuild`: a late demand peak, cheaper early production, and
    inventory prebuild in the construction plan (stored only for feasible requests);
  - `:disruption`: a plant outage in one period, sparser alternate lanes, and
    reduced/surcharged capacity around the disruption.

Only open `(plant, customer, product, period)` arcs have shipment variables.
Thus the exact variable count is
`2 * n_plants * n_products * n_periods + length(shipment_arcs)`.
Targets through `MAX_NETWORK_PLANNING_VARIABLES` are supported; larger targets
raise `ArgumentError` before arc allocation. Unknown-status instances apply a
correlated network-wide supply scenario and preserve local inbound lane
capacity, leaving aggregate feasibility naturally unspecified without creating
systematic singleton demand cuts.
"""
struct SupplyChainNetworkPlanningProblem <: ProblemGenerator
    profile::Symbol
    n_plants::Int
    n_customers::Int
    n_products::Int
    n_periods::Int
    plant_locations::Vector{Tuple{Float64, Float64}}
    customer_locations::Vector{Tuple{Float64, Float64}}
    plant_regions::Vector{Int}
    customer_regions::Vector{Int}
    specialization::Matrix{Float64}
    resource_use::Matrix{Float64}
    production_cost::Array{Float64, 3}
    holding_cost::Array{Float64, 3}
    demand::Array{Float64, 3}
    initial_inventory::Matrix{Float64}
    production_capacity::Array{Float64, 3}
    plant_capacity::Matrix{Float64}
    inventory_capacity::Matrix{Float64}
    shipment_arcs::Vector{NTuple{4, Int}}
    shipment_cost::Dict{NTuple{4, Int}, Float64}
    lane_capacity::Dict{NTuple{4, Int}, Float64}
    feasible_witness::Union{Nothing, NetworkPlanningWitness}
    infeasibility_certificate::Union{Nothing, NetworkPlanningInfeasibilityCertificate}
    disruption::Union{Nothing, NetworkPlanningDisruption}
    nominal_scenario::Union{Nothing, NetworkPlanningNominalScenario}
end

_network_profile(seed::Int) =
    _NETWORK_PLANNING_PROFILES[mod(seed, length(_NETWORK_PLANNING_PROFILES)) + 1]

function _network_period_range(profile::Symbol)
    profile == :regional_stable && return 3:6
    profile == :seasonal_prebuild && return 5:8
    return 4:7
end

function _network_density(profile::Symbol)
    profile == :regional_stable && return (0.30, 0.46)
    profile == :seasonal_prebuild && return (0.24, 0.38)
    return (0.16, 0.30)
end

"""
Choose dimensions and an exact sparse-arc budget. The search counts the two
dense plant/product/time blocks and every shipment variable that will actually
be created. Customer candidates are computed analytically, avoiding a scan
whose size grows with the requested target.
"""
function _choose_network_planning_dimensions(target_variables::Int, profile::Symbol)
    target_variables <= MAX_NETWORK_PLANNING_VARIABLES || throw(
        ArgumentError(
            "supply_chain/network_planning supports target_variables <= " *
            "$(MAX_NETWORK_PLANNING_VARIABLES); requested $target_variables. " *
            "Larger sparse-arc models require a multi-gigabyte working set.",
        ),
    )
    target = max(target_variables, 1)
    density_lo, density_hi = _network_density(profile)
    density_mid = (density_lo + density_hi) / 2
    best_score = (typemax(Int), Inf, typemax(Int), typemax(Int))
    best = (2, 2, 2, first(_network_period_range(profile)), 1)

    for n_periods in _network_period_range(profile), n_products in 2:5, n_plants in 2:20
        dense_vars = 2 * n_plants * n_products * n_periods
        max_degree = min(n_plants, max(2, ceil(Int, density_hi * n_plants)))
        expected_degree = clamp(density_mid * n_plants, 1.0, max_degree)
        denominators = (1.0, expected_degree, Float64(max_degree))
        customer_candidates = Set([2])
        for degree in denominators
            estimate = (target - dense_vars) / (n_products * n_periods * degree)
            for delta in -2:2
                push!(customer_candidates, max(2, round(Int, estimate) + delta))
            end
        end

        for n_customers in customer_candidates
            demand_nodes = n_customers * n_products * n_periods
            max_arcs = max_degree * demand_nodes
            if profile == :disruption
                # One plant is unavailable in one period.
                max_arcs -=
                    n_customers * n_products * max(0, max_degree - min(max_degree, n_plants - 1))
            end
            arc_budget = clamp(target - dense_vars, demand_nodes, max_arcs)
            delivered = dense_vars + arc_budget
            size_error = abs(delivered - target)
            density_error = abs(arc_budget / (n_plants * demand_nodes) - density_mid)
            shape_penalty = abs(n_customers - 3 * n_plants)
            score = (size_error, density_error, 0, shape_penalty)
            if score < best_score
                best_score = score
                best = (n_plants, n_customers, n_products, n_periods, arc_budget)
            end
        end
    end
    return best
end

_network_distance(a::Tuple{Float64, Float64}, b::Tuple{Float64, Float64}) =
    hypot(a[1] - b[1], a[2] - b[2])

function _network_locations(rng::AbstractRNG, n_plants::Int, n_customers::Int)
    n_regions = min(n_plants, n_customers, max(2, round(Int, sqrt(n_customers))))
    centers = [
        (rand(rng, Uniform(10.0, 90.0)), rand(rng, Uniform(10.0, 90.0))) for _ in 1:n_regions
    ]

    plant_regions = [mod1(i, n_regions) for i in 1:n_plants]
    shuffle!(rng, plant_regions)
    customer_regions = rand(rng, 1:n_regions, n_customers)

    plant_locations = Tuple{Float64, Float64}[]
    for region in plant_regions
        center = centers[region]
        push!(
            plant_locations,
            (
                clamp(center[1] + rand(rng, Normal(0, 11)), 0, 100),
                clamp(center[2] + rand(rng, Normal(0, 11)), 0, 100),
            ),
        )
    end
    customer_locations = Tuple{Float64, Float64}[]
    for region in customer_regions
        center = centers[region]
        push!(
            customer_locations,
            (
                clamp(center[1] + rand(rng, Normal(0, 7)), 0, 100),
                clamp(center[2] + rand(rng, Normal(0, 7)), 0, 100),
            ),
        )
    end
    return plant_locations, customer_locations, plant_regions, customer_regions
end

function _network_specialization(rng::AbstractRNG, n_plants::Int, n_products::Int)
    specialization = rand(rng, Uniform(0.48, 0.88), n_plants, n_products)
    for k in 1:n_products
        n_specialists = clamp(ceil(Int, n_plants / 3), 1, n_plants)
        for p in randperm(rng, n_plants)[1:n_specialists]
            specialization[p, k] = rand(rng, Uniform(1.22, 1.65))
        end
    end
    # Product bulk differs, while specialization makes a unit cheaper in the
    # shared plant resource at plants designed for that product.
    product_bulk = rand(rng, Uniform(0.75, 1.55), n_products)
    resource_use = [
        product_bulk[k] / sqrt(specialization[p, k]) for p in 1:n_plants, k in 1:n_products
    ]
    return specialization, resource_use
end

function _network_demand(
    rng::AbstractRNG,
    profile::Symbol,
    n_customers::Int,
    n_products::Int,
    n_periods::Int,
    disruption_period::Int,
)
    customer_scale = rand(rng, LogNormal(log(38.0), 0.35), n_customers)
    product_scale = rand(rng, Uniform(0.65, 1.45), n_products)
    seasonal = ones(n_periods)
    if profile == :regional_stable
        phase = rand(rng, Uniform(0, 2π))
        seasonal .= [1 + 0.07 * sin(2π * (t - 1) / n_periods + phase) for t in 1:n_periods]
    elseif profile == :seasonal_prebuild
        peak = max(3, ceil(Int, 0.72 * n_periods))
        seasonal .= [0.63 + 1.12 * exp(-0.5 * ((t - peak) / 1.05)^2) for t in 1:n_periods]
    else
        seasonal .= [1 + 0.04 * (t - 1) for t in 1:n_periods]
        seasonal[disruption_period] *= 1.22
    end

    demand = zeros(Float64, n_customers, n_products, n_periods)
    for c in 1:n_customers, k in 1:n_products, t in 1:n_periods
        demand[c, k, t] =
            customer_scale[c] * product_scale[k] * seasonal[t] * rand(rng, Uniform(0.92, 1.08))
    end
    return demand
end

function _network_arcs(
    rng::AbstractRNG,
    profile::Symbol,
    arc_budget::Int,
    n_plants::Int,
    n_customers::Int,
    n_products::Int,
    n_periods::Int,
    plant_locations,
    customer_locations,
    plant_regions,
    customer_regions,
    specialization,
    disruption_period::Int,
    disrupted_plant::Int,
)
    _, density_hi = _network_density(profile)
    max_degree = min(n_plants, max(2, ceil(Int, density_hi * n_plants)))
    arcs = NTuple{4, Int}[]
    extras = Tuple{Float64, NTuple{4, Int}}[]
    degree = Dict{Tuple{Int, Int, Int}, Int}()

    for c in 1:n_customers, k in 1:n_products, t in 1:n_periods
        node = (c, k, t)
        eligible = [
            p for p in 1:n_plants if
            !(profile == :disruption && t == disruption_period && p == disrupted_plant)
        ]
        scored = Tuple{Float64, Int}[]
        for p in eligible
            distance = _network_distance(plant_locations[p], customer_locations[c])
            region_penalty = if plant_regions[p] == customer_regions[c]
                0.0
            else
                (profile == :regional_stable ? 28.0 : 12.0)
            end
            specialist_penalty = 24.0 / specialization[p, k]
            time_jitter = rand(rng, Uniform(0, 4))
            push!(scored, (distance + region_penalty + specialist_penalty + time_jitter, p))
        end
        sort!(scored)
        primary = (scored[1][2], c, k, t)
        push!(arcs, primary)
        degree[node] = 1
        for (score, p) in scored[2:end]
            push!(extras, (score + rand(rng, Uniform(0, 7)), (p, c, k, t)))
        end
    end

    sort!(extras; by=first)
    for (_, arc) in extras
        length(arcs) >= arc_budget && break
        _, c, k, t = arc
        node = (c, k, t)
        period_max = if profile == :disruption && t == disruption_period
            min(max_degree, n_plants - 1)
        else
            max_degree
        end
        degree[node] >= period_max && continue
        push!(arcs, arc)
        degree[node] += 1
    end
    @assert length(arcs) == arc_budget
    sort!(arcs)
    return arcs
end

function _network_reference_plan(
    rng::AbstractRNG,
    profile::Symbol,
    demand,
    shipment_arcs,
    plant_locations,
    customer_locations,
    specialization,
    n_plants::Int,
    n_customers::Int,
    n_products::Int,
    n_periods::Int,
    disruption_period::Int,
)
    inbound = Dict{Tuple{Int, Int, Int}, Vector{NTuple{4, Int}}}()
    for arc in shipment_arcs
        p, c, k, t = arc
        push!(get!(inbound, (c, k, t), NTuple{4, Int}[]), arc)
    end

    witness_shipment = Dict(arc => 0.0 for arc in shipment_arcs)
    for c in 1:n_customers, k in 1:n_products, t in 1:n_periods
        arcs = inbound[(c, k, t)]
        weights = Float64[]
        for arc in arcs
            p = arc[1]
            distance = _network_distance(plant_locations[p], customer_locations[c])
            disruption_bias = profile == :disruption && t == disruption_period ? 0.8 : 1.0
            push!(
                weights,
                disruption_bias * specialization[p, k] / (10 + distance) *
                rand(rng, Uniform(0.9, 1.1)),
            )
        end
        weights ./= sum(weights)
        for (arc, weight) in zip(arcs, weights)
            witness_shipment[arc] = demand[c, k, t] * weight
        end
    end

    outflow = zeros(Float64, n_plants, n_products, n_periods)
    for (arc, value) in witness_shipment
        p, _, k, t = arc
        outflow[p, k, t] += value
    end

    initial_inventory = zeros(Float64, n_plants, n_products)
    witness_production = zeros(Float64, n_plants, n_products, n_periods)
    witness_inventory = zeros(Float64, n_plants, n_products, n_periods)
    for p in 1:n_plants, k in 1:n_products
        total_out = sum(outflow[p, k, :])
        initial_inventory[p, k] = total_out * rand(rng, Uniform(0.01, 0.035))
        previous = initial_inventory[p, k]
        for t in 1:n_periods
            desired = if profile == :seasonal_prebuild
                peak = max(3, ceil(Int, 0.72 * n_periods))
                if t < peak
                    0.20 * total_out * t / max(peak - 1, 1)
                else
                    0.20 * total_out * max(n_periods - t, 0) / max(n_periods - peak + 1, 1)
                end
            elseif profile == :disruption
                if t < disruption_period
                    0.12 * total_out * t / max(disruption_period - 1, 1)
                else
                    0.04 * total_out * max(n_periods - t, 0) / max(n_periods - disruption_period + 1, 1)
                end
            else
                0.025 * total_out * (1 - t / (n_periods + 1))
            end
            production = max(0.0, outflow[p, k, t] + desired - previous)
            inventory = previous + production - outflow[p, k, t]
            witness_production[p, k, t] = production
            witness_inventory[p, k, t] = max(0.0, inventory)
            previous = witness_inventory[p, k, t]
        end
    end
    return initial_inventory, witness_production, witness_inventory, witness_shipment
end

function SupplyChainNetworkPlanningProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)
    profile = _network_profile(seed)
    n_plants, n_customers, n_products, n_periods, arc_budget = _choose_network_planning_dimensions(
        target_variables, profile
    )

    disruption_period =
        profile == :disruption ? clamp(ceil(Int, 0.55 * n_periods), 2, n_periods) : 0
    disrupted_plant = profile == :disruption ? rand(rng, 1:n_plants) : 0
    disruption = if profile == :disruption
        NetworkPlanningDisruption(disruption_period, disrupted_plant, 0.35, 1.55)
    else
        nothing
    end
    plant_locations, customer_locations, plant_regions, customer_regions = _network_locations(
        rng, n_plants, n_customers
    )
    specialization, resource_use = _network_specialization(rng, n_plants, n_products)
    demand = _network_demand(rng, profile, n_customers, n_products, n_periods, disruption_period)
    shipment_arcs = _network_arcs(
        rng,
        profile,
        arc_budget,
        n_plants,
        n_customers,
        n_products,
        n_periods,
        plant_locations,
        customer_locations,
        plant_regions,
        customer_regions,
        specialization,
        disruption_period,
        disrupted_plant,
    )
    initial_inventory, witness_production, witness_inventory, witness_shipment = _network_reference_plan(
        rng,
        profile,
        demand,
        shipment_arcs,
        plant_locations,
        customer_locations,
        specialization,
        n_plants,
        n_customers,
        n_products,
        n_periods,
        disruption_period,
    )

    production_cost = zeros(Float64, n_plants, n_products, n_periods)
    holding_cost = zeros(Float64, n_plants, n_products, n_periods)
    for p in 1:n_plants, k in 1:n_products, t in 1:n_periods
        time_factor = profile == :seasonal_prebuild ? 0.82 + 0.09 * (t - 1) : 0.96 + 0.025 * (t - 1)
        if profile == :disruption && t == disruption_period && p == disrupted_plant
            time_factor *= 1.8
        end
        production_cost[p, k, t] =
            rand(rng, LogNormal(log(24.0), 0.18)) * time_factor / specialization[p, k]
        holding_cost[p, k, t] = rand(rng, Uniform(0.35, 1.3)) * (1 + 0.05 * (t - 1))
    end

    shipment_cost = Dict{NTuple{4, Int}, Float64}()
    lane_capacity = Dict{NTuple{4, Int}, Float64}()
    for arc in shipment_arcs
        p, c, k, t = arc
        distance = _network_distance(plant_locations[p], customer_locations[c])
        regional_factor = plant_regions[p] == customer_regions[c] ? 0.88 : 1.16
        disruption_factor = if profile == :disruption && t == disruption_period
            1.55
        elseif profile == :disruption && abs(t - disruption_period) == 1
            1.12
        else
            1.0
        end
        shipment_cost[arc] =
            (2.5 + distance * rand(rng, Uniform(0.32, 0.52))) *
            regional_factor *
            disruption_factor *
            rand(rng, Uniform(0.92, 1.08))
        node_demand = demand[c, k, t]
        lane_capacity[arc] =
            witness_shipment[arc] * rand(rng, Uniform(1.12, 1.35)) +
            node_demand * rand(rng, Uniform(0.025, 0.08))
        if profile == :disruption && t == disruption_period
            lane_capacity[arc] *= rand(rng, Uniform(0.82, 0.96))
            lane_capacity[arc] = max(lane_capacity[arc], 1.06 * witness_shipment[arc])
        end
    end

    production_capacity = zeros(Float64, n_plants, n_products, n_periods)
    plant_capacity = zeros(Float64, n_plants, n_periods)
    inventory_capacity = zeros(Float64, n_plants, n_products)
    for p in 1:n_plants, k in 1:n_products, t in 1:n_periods
        baseline = sum(demand[:, k, t]) / n_plants
        production_capacity[p, k, t] =
            witness_production[p, k, t] * rand(rng, Uniform(1.12, 1.32)) +
            0.04 * baseline * specialization[p, k]
    end
    for p in 1:n_plants, t in 1:n_periods
        used = sum(resource_use[p, k] * witness_production[p, k, t] for k in 1:n_products)
        plant_capacity[p, t] = used * rand(rng, Uniform(1.10, 1.25)) + 0.02 * sum(demand[:, :, t])
        if profile == :disruption && t == disruption_period && p == disrupted_plant
            # The disrupted plant has no lanes in this period, but retains a
            # small production crew that may rebuild inventory.
            plant_capacity[p, t] *= disruption.production_factor
            plant_capacity[p, t] = max(plant_capacity[p, t], 1.04 * used)
        end
    end
    for p in 1:n_plants, k in 1:n_products
        peak_inventory = maximum(witness_inventory[p, k, :])
        average_out =
            sum((witness_shipment[a] for a in shipment_arcs if a[1] == p && a[3] == k); init=0.0) /
            n_periods
        inventory_capacity[p, k] = max(
            1.18 * peak_inventory + 0.08 * average_out + 1.0, 1.10 * initial_inventory[p, k]
        )
    end

    nominal_scenario = nothing
    if feasibility_status == unknown
        # Draw a coherent network-wide supply condition, then add small
        # plant/product/time effects. This creates natural aggregate tightness
        # without independently damaging every sparse coordinate.
        supply_factor = rand(rng, Uniform(0.65, 1.20))
        lane_factor = rand(rng, Uniform(0.92, 1.14))
        plant_effect = rand(rng, Uniform(0.95, 1.05), n_plants)
        product_effect = rand(rng, Uniform(0.95, 1.05), n_products)
        period_effect = rand(rng, Uniform(0.96, 1.04), n_periods)
        for p in 1:n_plants, k in 1:n_products, t in 1:n_periods
            production_capacity[p, k, t] *=
                supply_factor * plant_effect[p] * product_effect[k] * period_effect[t]
        end
        for p in 1:n_plants, t in 1:n_periods
            plant_capacity[p, t] *= supply_factor * plant_effect[p] * period_effect[t]
        end

        lane_plant_effect = rand(rng, Uniform(0.96, 1.04), n_plants)
        lane_period_effect = rand(rng, Uniform(0.97, 1.03), n_periods)
        for arc in shipment_arcs
            p, _, _, t = arc
            lane_capacity[arc] *= lane_factor * lane_plant_effect[p] * lane_period_effect[t]
        end

        # Sparse lanes must not turn `unknown` into a hidden singleton-cut
        # generator. Preserve a modest, randomized local service margin at every
        # customer/product/period node; aggregate production/resource conditions
        # still determine whether the complete instance is feasible.
        inbound = Dict{Tuple{Int, Int, Int}, Vector{NTuple{4, Int}}}()
        for arc in shipment_arcs
            _, c, k, t = arc
            push!(get!(inbound, (c, k, t), NTuple{4, Int}[]), arc)
        end
        minimum_local_service = Inf
        for c in 1:n_customers, k in 1:n_products, t in 1:n_periods
            node_arcs = inbound[(c, k, t)]
            service_ratio = rand(rng, Uniform(1.03, 1.16))
            required = service_ratio * demand[c, k, t]
            available = sum(lane_capacity[a] for a in node_arcs)
            if available < required
                multiplier = required / available
                for arc in node_arcs
                    lane_capacity[arc] *= multiplier
                end
                available = required
            end
            minimum_local_service = min(minimum_local_service, available / demand[c, k, t])
        end
        nominal_scenario = NetworkPlanningNominalScenario(
            supply_factor, lane_factor, minimum_local_service
        )
    end

    infeasibility_certificate = nothing

    if feasibility_status == infeasible
        certificate_product = rand(rng, 1:n_products)
        certificate_period =
            profile == :disruption ? disruption_period : rand(rng, max(1, n_periods - 2):n_periods)
        k = certificate_product
        tau = certificate_period

        # Limit the chosen product's cumulative available supply. The resulting
        # cut remains valid in the LP: through tau, deliveries cannot exceed
        # initial stock plus the sum of every product/shared production upper
        # bound, nor the sum of all inbound lane limits.
        initial_inventory[:, k] .*= rand(rng, Uniform(0.05, 0.18))
        certificate_demand = sum(demand[:, k, 1:tau])
        desired_supply = rand(rng, Uniform(0.58, 0.78)) * certificate_demand
        initial_total = sum(initial_inventory[:, k])
        production_upper = [
            min(production_capacity[p, k, t], plant_capacity[p, t] / resource_use[p, k]) for
            p in 1:n_plants, t in 1:tau
        ]
        raw_production = sum(production_upper)
        scale = clamp((desired_supply - initial_total) / max(raw_production, eps()), 0.0, 0.95)
        for p in 1:n_plants, t in 1:tau
            # Set the product bound from the effective pre-cut bound. Merely
            # scaling the raw product bound is insufficient when shared plant
            # capacity was the tighter member of the minimum.
            production_capacity[p, k, t] = scale * production_upper[p, t]
        end

        certificate_supply_bound =
            sum(initial_inventory[:, k]) + sum(
                min(production_capacity[p, k, t], plant_capacity[p, t] / resource_use[p, k]) for
                p in 1:n_plants, t in 1:tau
            )
        certificate_lane_bound = sum(
            lane_capacity[a] for a in shipment_arcs if a[3] == k && a[4] <= tau
        )
        certificate_upper_bound = min(certificate_supply_bound, certificate_lane_bound)
        certificate_margin = certificate_demand - certificate_upper_bound
        @assert certificate_margin > 1e-8
        infeasibility_certificate = NetworkPlanningInfeasibilityCertificate(
            certificate_product,
            certificate_period,
            certificate_demand,
            certificate_supply_bound,
            certificate_lane_bound,
            certificate_upper_bound,
            certificate_margin,
        )
    end

    feasible_witness = if feasibility_status == feasible
        NetworkPlanningWitness(witness_production, witness_inventory, witness_shipment)
    else
        nothing
    end

    return SupplyChainNetworkPlanningProblem(
        profile,
        n_plants,
        n_customers,
        n_products,
        n_periods,
        plant_locations,
        customer_locations,
        plant_regions,
        customer_regions,
        specialization,
        resource_use,
        production_cost,
        holding_cost,
        demand,
        initial_inventory,
        production_capacity,
        plant_capacity,
        inventory_capacity,
        shipment_arcs,
        shipment_cost,
        lane_capacity,
        feasible_witness,
        infeasibility_certificate,
        disruption,
        nominal_scenario,
    )
end

function build_model(prob::SupplyChainNetworkPlanningProblem)
    model = Model()
    P, C, K, T = prob.n_plants, prob.n_customers, prob.n_products, prob.n_periods
    arcs = prob.shipment_arcs

    @variable(model, 0 <= produce[p = 1:P, k = 1:K, t = 1:T] <= prob.production_capacity[p, k, t])
    @variable(model, 0 <= inventory[p = 1:P, k = 1:K, t = 1:T] <= prob.inventory_capacity[p, k])
    @variable(model, 0 <= ship[a in arcs] <= prob.lane_capacity[a])

    @objective(
        model,
        Min,
        sum(prob.production_cost[p, k, t] * produce[p, k, t] for p in 1:P, k in 1:K, t in 1:T) +
            sum(prob.holding_cost[p, k, t] * inventory[p, k, t] for p in 1:P, k in 1:K, t in 1:T) +
            sum(prob.shipment_cost[a] * ship[a] for a in arcs)
    )

    outgoing = Dict{Tuple{Int, Int, Int}, Vector{NTuple{4, Int}}}()
    incoming = Dict{Tuple{Int, Int, Int}, Vector{NTuple{4, Int}}}()
    for arc in arcs
        p, c, k, t = arc
        push!(get!(outgoing, (p, k, t), NTuple{4, Int}[]), arc)
        push!(get!(incoming, (c, k, t), NTuple{4, Int}[]), arc)
    end

    @constraint(
        model,
        inventory_balance[p = 1:P, k = 1:K, t = 1:T],
        (t == 1 ? prob.initial_inventory[p, k] : inventory[p, k, t - 1]) + produce[p, k, t] -
        sum((ship[a] for a in get(outgoing, (p, k, t), NTuple{4, Int}[])); init=0.0) ==
            inventory[p, k, t]
    )

    # Equality prevents both service shortfall and cost-free dumping at customers.
    @constraint(
        model,
        demand_balance[c = 1:C, k = 1:K, t = 1:T],
        sum(ship[a] for a in incoming[(c, k, t)]) == prob.demand[c, k, t]
    )

    @constraint(
        model,
        resource_capacity[p = 1:P, t = 1:T],
        sum(prob.resource_use[p, k] * produce[p, k, t] for k in 1:K) <= prob.plant_capacity[p, t]
    )
    return model
end

register_variant(
    :supply_chain,
    :network_planning,
    SupplyChainNetworkPlanningProblem,
    "Multi-period, multi-product supply-chain network-planning LP with sparse period-specific lanes, specialized production, shared resource capacity, inventory carryover, exact service, structural profiles, and constructive feasibility certificates",
)
