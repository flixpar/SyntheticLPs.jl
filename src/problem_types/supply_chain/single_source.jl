using JuMP
using Random
using StatsBase
using Distributions

"""
    SingleSourceSupplyChainProblem <: ProblemGenerator

Generator for single-source capacitated supply chain (facility location) problems.

Each customer must be served in full by exactly one open facility (a single-source
assignment), in contrast to the standard supply chain model where a customer's
demand can be split across several facilities. This adds a binary assignment
matrix `z[f, c]` whose `sum_f z[f, c] == 1` enforces single sourcing, together
with flow-gating constraints `x[(f,c,m)] <= demand[c] * z[f, c]` that allow
shipping on a lane only from the facility a customer is assigned to.

This problem models realistic supply chain networks with:

  - Geographic clustering of customers and facilities
  - Multiple transportation modes with infrastructure availability
  - K-nearest connectivity guarantees for feasible instances
  - Facility opening costs and capacities
  - Mode-specific capacity constraints
  - Single-source assignment of every customer to one facility

# Overview

Models single-source strategic supply-chain network design. The decisions open
facilities (`y`), assign each customer to exactly one facility (`z`), and ship
that customer's demand from its assigned facility over available transportation
modes (`x`). The objective minimizes fixed facility cost plus mode-specific
transportation cost. Constraints assign each customer to one facility, gate lane
flow by the assignment, satisfy customer demand, gate shipments by open-facility
capacity, and limit aggregate shipment volume by transportation mode.

# Fields

All data generated in the constructor based on `target_variables` and `feasibility_status`:

  - `n_facilities::Int`: Number of potential facility locations
  - `n_customers::Int`: Number of customer locations
  - `transport_modes::Vector{String}`: Selected transport modes
  - `facility_locs::Vector{Tuple{Float64,Float64}}`: Geographic facility locations
  - `customer_locs::Vector{Tuple{Float64,Float64}}`: Geographic customer locations
  - `cluster_centers::Vector{Tuple{Float64,Float64}}`: Cluster centers for customer distribution
  - `cluster_weights::Vector{Float64}`: Weights for cluster importance
  - `fixed_costs::Dict{Int, Float64}`: Fixed cost to open each facility
  - `demands::Dict{Int, Float64}`: Demand at each customer location
  - `capacities::Dict{Int, Float64}`: Capacity of each facility
  - `transport_costs::Dict{Tuple{Int,Int,String}, Float64}`: Transport cost per (facility, customer, mode)
  - `mode_capacities::Dict{String, Float64}`: Total capacity available for each transport mode
  - `total_demand::Float64`: Total demand across all customers
"""
struct SingleSourceSupplyChainProblem <: ProblemGenerator
    n_facilities::Int
    n_customers::Int
    transport_modes::Vector{String}
    facility_locs::Vector{Tuple{Float64, Float64}}
    customer_locs::Vector{Tuple{Float64, Float64}}
    cluster_centers::Vector{Tuple{Float64, Float64}}
    cluster_weights::Vector{Float64}
    fixed_costs::Dict{Int, Float64}
    demands::Dict{Int, Float64}
    capacities::Dict{Int, Float64}
    transport_costs::Dict{Tuple{Int, Int, String}, Float64}
    mode_capacities::Dict{String, Float64}
    total_demand::Float64
end

"""
    SingleSourceSupplyChainProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a single-source supply chain problem instance with geographic clustering
and connectivity logic.

# Variable-count formula

The built model has three variable blocks:

  - `y[1:n_facilities]`                            -> `n_facilities`
  - `x[valid_combinations]`                        -> `n_lanes` (available (facility, customer, mode) lanes)
  - `z[1:n_facilities, 1:n_customers]` (Bin)       -> `n_facilities * n_customers`

Total `n_facilities * (1 + n_customers) + n_lanes`. Customers are generated one
at a time — location, demand, and lane availability per (facility, mode) — and
generation stops at whichever customer count lands the exact total closest to
`target_variables` (at least 4 customers), so the realised size tracks the
target within about one customer's worth of columns (`n_facilities + ` its
lanes; ≲ 2% from 1k up).

# Sophisticated Feasibility Logic

  - **Geographic clustering**: Customers clustered using Dirichlet-weighted clusters with log-normal spread
  - **Facility placement**: Beta-distributed strategic placement with market-access consideration
  - **K-nearest connectivity**: For feasible instances, ensures each customer connects to K nearest facilities
  - **Single-source feasibility**: For feasible instances, builds an explicit capacity-respecting
    assignment of each customer to one facility, guaranteeing a single-source solution exists
  - **Capacity deficit**: For infeasible instances, drives total facility capacity below total
    demand with a margin, so no assignment can be served

# Arguments

  - `target_variables`: Target number of variables (see variable-count formula above)
  - `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
  - `seed`: Random seed for reproducibility
"""
function SingleSourceSupplyChainProblem(
    target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int
)
    rng = MersenneTwister(seed)

    # --- Regime by scale (facility count, modes, geography, costs) ---
    if target_variables <= 250
        n_facilities = rand(rng, DiscreteUniform(3, 6))
        n_transport_modes = rand(rng, DiscreteUniform(1, 2))
        grid_width = rand(rng, Uniform(200.0, 800.0))
        grid_height = rand(rng, Uniform(200.0, 800.0))
        infrastructure_density = rand(rng, Beta(5, 2)) * 0.3 + 0.7  # 0.7-1.0
        clustering_factor = rand(rng, Beta(3, 2)) * 0.6 + 0.25  # 0.25-0.85
        min_fixed_cost = max(100000.0, rand(rng, LogNormal(log(300000), 0.5)))
        max_fixed_cost = min_fixed_cost * rand(rng, Uniform(1.8, 3.5))
        base_demand = rand(rng, Uniform(80.0, 150.0))
        min_demand = base_demand
        max_demand = base_demand * rand(rng, Uniform(3.0, 8.0))
    elseif target_variables <= 1000
        n_facilities = rand(rng, DiscreteUniform(5, 12))
        n_transport_modes = rand(rng, DiscreteUniform(2, 3))
        grid_width = rand(rng, Uniform(800.0, 2000.0))
        grid_height = rand(rng, Uniform(800.0, 2000.0))
        infrastructure_density = rand(rng, Beta(3, 2)) * 0.4 + 0.5  # 0.5-0.9
        clustering_factor = rand(rng, Beta(2, 3)) * 0.5 + 0.2  # 0.2-0.7
        min_fixed_cost = max(300000.0, rand(rng, LogNormal(log(800000), 0.6)))
        max_fixed_cost = min_fixed_cost * rand(rng, Uniform(2.0, 4.0))
        base_demand = rand(rng, Uniform(150.0, 300.0))
        min_demand = base_demand
        max_demand = base_demand * rand(rng, Uniform(4.0, 12.0))
    else
        n_facilities = rand(rng, DiscreteUniform(8, 20))
        n_transport_modes = rand(rng, DiscreteUniform(3, 4))
        grid_width = rand(rng, Uniform(2000.0, 5000.0))
        grid_height = rand(rng, Uniform(2000.0, 5000.0))
        infrastructure_density = rand(rng, Beta(2, 3)) * 0.4 + 0.4  # 0.4-0.8
        clustering_factor = rand(rng, Beta(1, 3)) * 0.4 + 0.15  # 0.15-0.55
        min_fixed_cost = max(500000.0, rand(rng, LogNormal(log(1500000), 0.7)))
        max_fixed_cost = min_fixed_cost * rand(rng, Uniform(2.5, 5.0))
        base_demand = rand(rng, Uniform(300.0, 600.0))
        min_demand = base_demand
        max_demand = base_demand * rand(rng, Uniform(6.0, 20.0))
    end

    # Additional parameters
    capacity_factor = rand(rng, Uniform(1.2, 2.2))
    mode_capacity_factor = rand(rng, Uniform(0.25, 0.65))

    # Transport modes and costs
    all_transport_modes = ["truck", "rail", "ship", "air"]
    transport_base_costs = Dict(
        "truck" => rand(rng, Gamma(4, 0.25)),
        "rail" => rand(rng, Gamma(3, 0.2)),
        "ship" => rand(rng, Gamma(2, 0.15)),
        "air" => rand(rng, Gamma(6, 0.5)),
    )

    transport_modes = sample(
        rng, all_transport_modes, min(n_transport_modes, length(all_transport_modes)); replace=false
    )
    # Feasible requests guarantee every customer a lane on one fallback mode to
    # its K nearest facilities, so a single-source assignment is buildable.
    fallback_mode = ("truck" in transport_modes) ? "truck" : transport_modes[1]
    K = min(max(3, ceil(Int, n_facilities ÷ 3)), n_facilities)

    # Geographic clusters, sized from a first estimate of the customer count
    # (the exact count is settled while customers are generated below).
    eff = 0.6 * n_transport_modes * infrastructure_density
    n_customers_hint = max(
        4, round(Int, (target_variables - n_facilities) / (n_facilities * (1.0 + eff)))
    )
    n_clusters = max(2, round(Int, sqrt(n_customers_hint) * clustering_factor))
    cluster_centers = [(grid_width * rand(rng), grid_height * rand(rng)) for _ in 1:n_clusters]
    cluster_weights = rand(rng, Dirichlet(ones(n_clusters)))

    # Facility locations (more dispersed)
    facility_locs = Vector{Tuple{Float64, Float64}}()
    for _ in 1:n_facilities
        if rand(rng) < 0.4
            center = rand(rng, cluster_centers)
            spread_x = grid_width * 0.12
            spread_y = grid_height * 0.12
            x = clamp(center[1] + rand(rng, Normal(0, spread_x)), 0, grid_width)
            y = clamp(center[2] + rand(rng, Normal(0, spread_y)), 0, grid_height)
        else
            x = grid_width * rand(rng, Beta(1.5, 1.5))
            y = grid_height * rand(rng, Beta(1.5, 1.5))
        end
        push!(facility_locs, (x, y))
    end

    # --- Customers, generated one at a time until the variable count lands on
    # the target. Each customer brings one assignment column per facility plus
    # its available lanes, so the realised size is known exactly as we go:
    #   vars = n_facilities + n_facilities * n_customers + n_lanes.
    customer_locs = Vector{Tuple{Float64, Float64}}()
    demand_multipliers = Float64[]
    # Lane data: (f, c, mode) => (distance, terrain factor, efficiency factor)
    lane_draws = Dict{Tuple{Int, Int, String}, NTuple{3, Float64}}()
    diag_len = sqrt(grid_width^2 + grid_height^2)
    base_spread = grid_width * (1 - clustering_factor) * 0.08
    total_vars = n_facilities
    while true
        c = length(customer_locs) + 1
        cluster_idx = sample(rng, 1:n_clusters, Weights(cluster_weights))
        center = cluster_centers[cluster_idx]
        spread = rand(rng, LogNormal(log(base_spread), 0.3))
        loc = (
            clamp(center[1] + rand(rng, Normal(0, spread)), 0, grid_width),
            clamp(center[2] + rand(rng, Normal(0, spread)), 0, grid_height),
        )
        multiplier = rand(rng, LogNormal(log(1.0), 0.4))
        dvec = [
            hypot(facility_locs[f][1] - loc[1], facility_locs[f][2] - loc[2]) for
            f in 1:n_facilities
        ]
        lanes = Tuple{Int, Int, String}[]
        draws = NTuple{3, Float64}[]
        for f in 1:n_facilities, mode in transport_modes
            prob_available = if mode == "truck"
                0.98
            elseif mode == "rail"
                min(0.8, 0.3 + 0.5 * (dvec[f] / diag_len))
            elseif mode == "ship"
                any(l -> abs(l[2]) < grid_height * 0.1, (facility_locs[f], loc)) ? 0.8 : 0.0
            else  # air
                dvec[f] > diag_len * 0.3 ? 0.7 : 0.2
            end
            if rand(rng) < prob_available * infrastructure_density
                push!(lanes, (f, c, mode))
                push!(
                    draws,
                    (
                        dvec[f],
                        rand(rng, LogNormal(log(1.0), 0.15)),
                        rand(rng, Beta(3, 2)) * 0.4 + 0.8,
                    ),
                )
            end
        end
        if feasibility_status == feasible
            for f in sortperm(dvec)[1:K]
                (f, c, fallback_mode) in lanes && continue
                push!(lanes, (f, c, fallback_mode))
                push!(
                    draws,
                    (
                        dvec[f],
                        rand(rng, LogNormal(log(1.0), 0.15)),
                        rand(rng, Beta(3, 2)) * 0.4 + 0.8,
                    ),
                )
            end
        end
        next_total = total_vars + n_facilities + length(lanes)
        # Stop at whichever of "without" / "with" this customer is closer to
        # the target (at least 4 customers).
        if c > 4 && abs(next_total - target_variables) >= abs(total_vars - target_variables)
            break
        end
        push!(customer_locs, loc)
        push!(demand_multipliers, multiplier)
        for (lane, d) in zip(lanes, draws)
            lane_draws[lane] = d
        end
        total_vars = next_total
    end
    n_customers = length(customer_locs)

    # Facility fixed costs (correlated with market access and location)
    fixed_costs = Dict{Int, Float64}()
    for f in 1:n_facilities
        distances_to_customers = [
            sqrt((facility_locs[f][1] - c[1])^2 + (facility_locs[f][2] - c[2])^2) for
            c in customer_locs
        ]
        market_potential = sum(exp.(-distances_to_customers ./ (grid_width * 0.2)))
        location_factor = (facility_locs[f][1] / grid_width + facility_locs[f][2] / grid_height) / 2
        base_cost =
            min_fixed_cost +
            (max_fixed_cost - min_fixed_cost) *
            (0.2 + 0.5 * market_potential / n_customers + 0.3 * location_factor)
        cost_multiplier = rand(rng, LogNormal(log(1.0), 0.25))
        fixed_costs[f] = base_cost * cost_multiplier
    end

    # Customer demands (correlated with the weight of the nearest cluster)
    demands = Dict{Int, Float64}()
    for c in 1:n_customers
        distances_to_clusters = [
            sqrt((customer_locs[c][1] - center[1])^2 + (customer_locs[c][2] - center[2])^2) for
            center in cluster_centers
        ]
        _, cluster_idx = findmin(distances_to_clusters)
        cluster_influence = cluster_weights[cluster_idx]
        base_demand_val = min_demand + (max_demand - min_demand) * (0.2 + 0.8 * cluster_influence)
        demands[c] = base_demand_val * demand_multipliers[c]
    end

    # Facility capacities
    total_demand = sum(values(demands))
    avg_capacity = (total_demand / n_facilities) * capacity_factor
    capacities = Dict{Int, Float64}()
    for f in 1:n_facilities
        relative_cost =
            (fixed_costs[f] - minimum(values(fixed_costs))) /
            max(eps(), (maximum(values(fixed_costs)) - minimum(values(fixed_costs))))
        base_capacity = avg_capacity * (0.6 + 0.8 * relative_cost)
        capacity_multiplier = rand(rng, Gamma(3, 1/3))
        capacities[f] = base_capacity * capacity_multiplier
    end

    # Transport costs on the available lanes
    max_demand_realised = maximum(values(demands))
    transport_costs = Dict{Tuple{Int, Int, String}, Float64}()
    for ((f, c, mode), (distance, terrain_factor, efficiency_factor)) in lane_draws
        base_cost = get(transport_base_costs, mode, 1.0)
        volume_factor = 1.0 - 0.25 * (demands[c] / max_demand_realised)
        transport_costs[(f, c, mode)] =
            base_cost * distance * terrain_factor * volume_factor * efficiency_factor
    end

    # Mode capacities
    mode_capacities = Dict{String, Float64}()
    for mode in transport_modes
        base_capacity = total_demand * mode_capacity_factor
        capacity_multiplier = if mode == "truck"
            rand(rng, Gamma(4, 0.25))
        elseif mode == "rail"
            rand(rng, Gamma(6, 0.33))
        elseif mode == "ship"
            rand(rng, Gamma(9, 0.33))
        else  # air
            rand(rng, Gamma(2, 0.25))
        end
        mode_capacities[mode] = base_capacity * capacity_multiplier
    end

    # --- Feasibility enforcement ---
    if feasibility_status == feasible
        # Every customer already has a fallback-mode lane to its K nearest
        # facilities (added while customers were generated).
        # Build an explicit single-source assignment: greedily assign each customer
        # (largest demand first) to the nearest facility that still has residual
        # capacity and a valid route. Bump capacities/mode capacity so this fits.
        residual = Dict(f => capacities[f] for f in 1:n_facilities)
        assigned_to = Dict{Int, Int}()
        customer_order = sort(1:n_customers; by=c -> -demands[c])
        for c in customer_order
            dvec = [
                sqrt(
                    (facility_locs[f][1] - customer_locs[c][1])^2 +
                    (facility_locs[f][2] - customer_locs[c][2])^2,
                ) for f in 1:n_facilities
            ]
            order = sortperm(dvec)
            # facilities with a route to c on the fallback mode
            routed = [f for f in order if haskey(transport_costs, (f, c, fallback_mode))]
            chosen = nothing
            for f in routed
                if residual[f] >= demands[c]
                    chosen = f
                    break
                end
            end
            if chosen === nothing
                # No facility has room: assign to the nearest routed facility
                # (K >= 3 fallback lanes exist) and grow its capacity.
                chosen = routed[1]
                capacities[chosen] += 1.05 * demands[c]
                residual[chosen] =
                    capacities[chosen] -
                    sum(demands[cc] for (cc, ff) in assigned_to if ff == chosen; init=0.0)
            end
            assigned_to[c] = chosen
            residual[chosen] -= demands[c]
        end

        # Guarantee each facility's capacity covers its assigned single-source load.
        assigned_load = Dict(f => 0.0 for f in 1:n_facilities)
        for (c, f) in assigned_to
            assigned_load[f] += demands[c]
        end
        for f in 1:n_facilities
            if capacities[f] < 1.05 * assigned_load[f]
                capacities[f] = 1.05 * assigned_load[f]
            end
        end

        # Ensure the fallback mode alone can move all demand.
        if mode_capacities[fallback_mode] < 1.05 * total_demand
            mode_capacities[fallback_mode] = 1.05 * total_demand
        end

    elseif feasibility_status == infeasible
        # CAPACITY DEFICIT: drive total facility capacity strictly below total demand
        # with a margin, so no assignment (single-source or otherwise) can satisfy demand.
        margin = rand(rng, Uniform(0.7, 0.9))  # target total capacity = margin * total_demand
        total_capacity = sum(values(capacities))
        target_total = margin * total_demand
        if total_capacity > target_total
            scale = target_total / total_capacity
            for f in 1:n_facilities
                capacities[f] *= scale
            end
        end
    end
    # For unknown, leave the natural instance unchanged.

    return SingleSourceSupplyChainProblem(
        n_facilities,
        n_customers,
        transport_modes,
        facility_locs,
        customer_locs,
        cluster_centers,
        cluster_weights,
        fixed_costs,
        demands,
        capacities,
        transport_costs,
        mode_capacities,
        total_demand,
    )
end

"""
    build_model(prob::SingleSourceSupplyChainProblem)

Build a JuMP model for the single-source supply chain problem (deterministic).

# Arguments

  - `prob`: SingleSourceSupplyChainProblem instance

# Returns

  - `model`: The JuMP model
"""
function build_model(prob::SingleSourceSupplyChainProblem)
    model = Model()

    # Open-facility decisions
    @variable(model, y[1:prob.n_facilities], Bin)

    # Valid (facility, customer, mode) lanes with an available route
    valid_combinations = [
        (f, c, m) for
        f in 1:prob.n_facilities, c in 1:prob.n_customers, m in prob.transport_modes if
        haskey(prob.transport_costs, (f, c, m))
    ]

    # Flow decisions on valid lanes
    @variable(model, x[valid_combinations] >= 0)

    # Single-source assignment decisions
    @variable(model, z[1:prob.n_facilities, 1:prob.n_customers], Bin)

    # Objective: minimize fixed facility cost + transportation cost
    @objective(
        model,
        Min,
        sum(prob.fixed_costs[f] * y[f] for f in 1:prob.n_facilities) +
            sum(prob.transport_costs[combo] * x[combo] for combo in valid_combinations)
    )

    # Each customer assigned to exactly one facility
    for c in 1:prob.n_customers
        @constraint(model, sum(z[f, c] for f in 1:prob.n_facilities) == 1)
    end

    # Flow gating: a lane (f,c,m) can carry flow only if customer c is assigned to f
    for (f, c, m) in valid_combinations
        @constraint(model, x[(f, c, m)] <= prob.demands[c] * z[f, c])
    end

    by_customer = [Tuple{Int, Int, String}[] for _ in 1:prob.n_customers]
    by_facility = [Tuple{Int, Int, String}[] for _ in 1:prob.n_facilities]
    by_mode = Dict(m => Tuple{Int, Int, String}[] for m in prob.transport_modes)
    for combo in valid_combinations
        push!(by_facility[combo[1]], combo)
        push!(by_customer[combo[2]], combo)
        push!(by_mode[combo[3]], combo)
    end

    # Customer demand satisfaction
    for c in 1:prob.n_customers
        @constraint(model, sum(x[combo] for combo in by_customer[c]; init=0.0) >= prob.demands[c])
    end

    # Facility capacity (also links opening decision y)
    for f in 1:prob.n_facilities
        @constraint(
            model, sum(x[combo] for combo in by_facility[f]; init=0.0) <= prob.capacities[f] * y[f]
        )
    end

    # Transport mode capacity (restored for consistency with the standard SC model)
    for m in prob.transport_modes
        @constraint(
            model, sum(x[combo] for combo in by_mode[m]; init=0.0) <= prob.mode_capacities[m]
        )
    end

    return model
end

# Register the variant
register_variant(
    :supply_chain,
    :single_source,
    SingleSourceSupplyChainProblem,
    "Single-source capacitated supply chain where each customer is served in full by exactly one facility";
    tags=[:location, :bipartite, :partitioning, :big_m],
)
