using JuMP
using Random
using StatsBase
using Distributions

"""
    SupplyChainProblem <: ProblemGenerator

Generator for supply chain optimization problems that minimize facility and transportation costs while
meeting customer demands and respecting capacity constraints.

This generator randomly samples between multiple realistic supply chain variants:
- Multi-echelon supply chains (suppliers → warehouses → customers)
- Global supply chains with international tariffs and duties
- Direct-to-consumer e-commerce fulfillment with service tiers
- Multi-period planning with inventory holding costs
- Make-or-buy decisions with production and procurement options

Each variant captures different real-world supply chain structures and constraints.

# Fields
All data generated in constructor based on target_variables and feasibility_status.
Fields vary by variant but generally include:
- Network structure (nodes, arcs, echelons)
- Cost parameters (fixed costs, variable costs, tariffs, holding costs)
- Capacity constraints (facility capacities, transport capacities)
- Demand parameters (customer demands, time periods)
- Variant-specific data (service levels, exchange rates, lead times, etc.)
"""
struct SupplyChainProblem <: ProblemGenerator
    variant::Symbol
    data::Dict{Symbol, Any}
end

#=============================================================================
  HELPER FUNCTIONS FOR COMMON OPERATIONS
=============================================================================#

"""
Generate geographic locations with clustering for realistic spatial distribution.
"""
function generate_clustered_locations(n_locations::Int, n_clusters::Int,
                                      grid_width::Float64, grid_height::Float64,
                                      clustering_factor::Float64)
    # Generate cluster centers
    cluster_centers = [(grid_width * rand(), grid_height * rand()) for _ in 1:n_clusters]
    cluster_weights = rand(Dirichlet(ones(n_clusters)))

    # Generate locations clustered around centers
    locations = Vector{Tuple{Float64,Float64}}()
    for _ in 1:n_locations
        cluster_idx = sample(1:n_clusters, Weights(cluster_weights))
        center = cluster_centers[cluster_idx]

        base_spread = grid_width * (1 - clustering_factor) * 0.08
        spread = rand(LogNormal(log(base_spread), 0.3))

        x = clamp(center[1] + rand(Normal(0, spread)), 0, grid_width)
        y = clamp(center[2] + rand(Normal(0, spread)), 0, grid_height)
        push!(locations, (x, y))
    end

    return locations, cluster_centers, cluster_weights
end

"""
Calculate Euclidean distance between two locations.
"""
euclidean_distance(loc1::Tuple{Float64,Float64}, loc2::Tuple{Float64,Float64}) =
    sqrt((loc1[1] - loc2[1])^2 + (loc1[2] - loc2[2])^2)

"""
Generate transport costs based on distance and mode characteristics.
"""
function generate_transport_cost(distance::Float64, mode::String,
                                 base_costs::Dict{String,Float64},
                                 demand::Float64, max_demand::Float64)
    base_cost = get(base_costs, mode, 1.0)
    terrain_factor = rand(LogNormal(log(1.0), 0.15))
    volume_discount = 1.0 - 0.25 * (demand / max_demand)
    efficiency_factor = rand(Beta(3, 2)) * 0.4 + 0.8

    return base_cost * distance * terrain_factor * volume_discount * efficiency_factor
end

#=============================================================================
  VARIANT 1: MULTI-ECHELON SUPPLY CHAIN (30% probability)

  Models a two-echelon distribution network: Suppliers → Warehouses → Customers
  Common in retail, manufacturing, and logistics industries.

  Features:
  - Supplier facilities with production/sourcing capacity
  - Distribution warehouses with storage capacity
  - Flow from suppliers to warehouses to customers
  - Warehouse inventory holding costs
  - Facility opening decisions at both echelons
=============================================================================#

function generate_multi_echelon_variant(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem dimensions
    # Variables = n_suppliers (binary) + n_warehouses (binary) +
    #             n_suppliers * n_warehouses (flow) + n_warehouses * n_customers (flow)
    if target_variables <= 250
        n_suppliers = rand(DiscreteUniform(2, 5))
        n_warehouses = rand(DiscreteUniform(3, 7))
        n_customers = rand(DiscreteUniform(15, 30))
        grid_size = rand(Uniform(500.0, 1000.0))
        clustering_factor = rand(Beta(3, 2)) * 0.5 + 0.3
    elseif target_variables <= 1000
        n_suppliers = rand(DiscreteUniform(4, 10))
        n_warehouses = rand(DiscreteUniform(6, 15))
        n_customers = rand(DiscreteUniform(25, 60))
        grid_size = rand(Uniform(1000.0, 2500.0))
        clustering_factor = rand(Beta(2, 3)) * 0.4 + 0.25
    else
        n_suppliers = rand(DiscreteUniform(8, 20))
        n_warehouses = rand(DiscreteUniform(12, 30))
        n_customers = rand(DiscreteUniform(50, 150))
        grid_size = rand(Uniform(2500.0, 5000.0))
        clustering_factor = rand(Beta(1, 3)) * 0.3 + 0.2
    end

    # Generate geographic locations
    n_clusters = max(3, round(Int, sqrt(n_customers) * clustering_factor))
    customer_locs, cluster_centers, cluster_weights = generate_clustered_locations(
        n_customers, n_clusters, grid_size, grid_size, clustering_factor
    )

    # Warehouses placed strategically near customer clusters
    warehouse_locs = Vector{Tuple{Float64,Float64}}()
    for _ in 1:n_warehouses
        if rand() < 0.6
            center = rand(cluster_centers)
            spread = grid_size * 0.15
            x = clamp(center[1] + rand(Normal(0, spread)), 0, grid_size)
            y = clamp(center[2] + rand(Normal(0, spread)), 0, grid_size)
        else
            x = grid_size * rand(Beta(1.5, 1.5))
            y = grid_size * rand(Beta(1.5, 1.5))
        end
        push!(warehouse_locs, (x, y))
    end

    # Suppliers placed more dispersed (representing different sourcing regions)
    supplier_locs = [(grid_size * rand(Beta(1.2, 1.2)), grid_size * rand(Beta(1.2, 1.2)))
                     for _ in 1:n_suppliers]

    # Generate customer demands
    base_demand = target_variables <= 250 ? rand(Uniform(100.0, 200.0)) :
                  target_variables <= 1000 ? rand(Uniform(200.0, 400.0)) :
                  rand(Uniform(400.0, 800.0))

    demands = Dict{Int, Float64}()
    for c in 1:n_customers
        distances_to_clusters = [euclidean_distance(customer_locs[c], center) for center in cluster_centers]
        _, cluster_idx = findmin(distances_to_clusters)
        cluster_influence = cluster_weights[cluster_idx]

        demand_val = base_demand * (0.3 + 1.4 * cluster_influence) * rand(LogNormal(log(1.0), 0.35))
        demands[c] = demand_val
    end
    total_demand = sum(values(demands))
    max_demand = maximum(values(demands))

    # Supplier capacities and fixed costs
    supplier_capacity_factor = rand(Uniform(1.3, 2.0))
    avg_supplier_capacity = (total_demand / n_suppliers) * supplier_capacity_factor

    supplier_capacities = Dict{Int, Float64}()
    supplier_fixed_costs = Dict{Int, Float64}()
    base_supplier_cost = rand(LogNormal(log(500000), 0.6))

    for s in 1:n_suppliers
        capacity = avg_supplier_capacity * rand(Gamma(3, 1/3))
        supplier_capacities[s] = capacity

        # Larger capacity suppliers tend to have higher fixed costs
        capacity_factor = capacity / avg_supplier_capacity
        supplier_fixed_costs[s] = base_supplier_cost * (0.5 + 0.5 * capacity_factor) * rand(LogNormal(log(1.0), 0.25))
    end

    # Warehouse capacities and fixed costs
    warehouse_capacity_factor = rand(Uniform(1.2, 1.8))
    avg_warehouse_capacity = (total_demand / n_warehouses) * warehouse_capacity_factor

    warehouse_capacities = Dict{Int, Float64}()
    warehouse_fixed_costs = Dict{Int, Float64}()
    warehouse_holding_costs = Dict{Int, Float64}()
    base_warehouse_cost = rand(LogNormal(log(200000), 0.5))

    for w in 1:n_warehouses
        capacity = avg_warehouse_capacity * rand(Gamma(3, 1/3))
        warehouse_capacities[w] = capacity

        # Location-dependent costs (proximity to customers)
        distances_to_customers = [euclidean_distance(warehouse_locs[w], c) for c in customer_locs]
        market_potential = sum(exp.(-distances_to_customers ./ (grid_size * 0.25)))
        location_premium = market_potential / n_customers

        warehouse_fixed_costs[w] = base_warehouse_cost * (0.6 + 0.8 * location_premium) * rand(LogNormal(log(1.0), 0.2))

        # Holding costs proportional to warehouse sophistication/location
        warehouse_holding_costs[w] = rand(Uniform(0.15, 0.35)) * (warehouse_fixed_costs[w] / base_warehouse_cost)
    end

    # Transport costs: Supplier → Warehouse
    supplier_warehouse_costs = Dict{Tuple{Int,Int}, Float64}()
    base_cost_sw = rand(Gamma(3, 0.25))

    for s in 1:n_suppliers
        for w in 1:n_warehouses
            distance = euclidean_distance(supplier_locs[s], warehouse_locs[w])
            # Bulk shipments from suppliers to warehouses are cheaper per unit
            cost = base_cost_sw * distance * rand(LogNormal(log(1.0), 0.2)) * rand(Beta(2, 1))
            supplier_warehouse_costs[(s, w)] = cost
        end
    end

    # Transport costs: Warehouse → Customer
    warehouse_customer_costs = Dict{Tuple{Int,Int}, Float64}()
    base_cost_wc = rand(Gamma(4, 0.3))

    for w in 1:n_warehouses
        for c in 1:n_customers
            distance = euclidean_distance(warehouse_locs[w], customer_locs[c])
            # Last-mile delivery costs (higher per unit)
            cost = base_cost_wc * distance * rand(LogNormal(log(1.0), 0.18)) *
                   (1.0 - 0.2 * (demands[c] / max_demand))  # Volume discount
            warehouse_customer_costs[(w, c)] = cost
        end
    end

    # Feasibility adjustments
    if feasibility_status == feasible
        # Ensure sufficient capacity at both echelons
        if sum(values(supplier_capacities)) < total_demand
            scale = 1.1 * total_demand / sum(values(supplier_capacities))
            for s in 1:n_suppliers
                supplier_capacities[s] *= scale
            end
        end

        if sum(values(warehouse_capacities)) < total_demand
            scale = 1.1 * total_demand / sum(values(warehouse_capacities))
            for w in 1:n_warehouses
                warehouse_capacities[w] *= scale
            end
        end

        # Ensure each customer can be served by at least 2 nearby warehouses
        for c in 1:n_customers
            distances = [(w, euclidean_distance(warehouse_locs[w], customer_locs[c]))
                        for w in 1:n_warehouses]
            sort!(distances, by=x->x[2])

            for (w, dist) in distances[1:min(2, n_warehouses)]
                # Ensure route exists with reasonable cost
                if !haskey(warehouse_customer_costs, (w, c))
                    warehouse_customer_costs[(w, c)] = base_cost_wc * dist * 1.1
                end
            end
        end

    elseif feasibility_status == infeasible
        # Make infeasible by constraining warehouse capacity
        scale = rand(Uniform(0.65, 0.85))
        for w in 1:n_warehouses
            warehouse_capacities[w] *= scale
        end
    end

    return Dict(
        :n_suppliers => n_suppliers,
        :n_warehouses => n_warehouses,
        :n_customers => n_customers,
        :supplier_capacities => supplier_capacities,
        :warehouse_capacities => warehouse_capacities,
        :demands => demands,
        :supplier_fixed_costs => supplier_fixed_costs,
        :warehouse_fixed_costs => warehouse_fixed_costs,
        :warehouse_holding_costs => warehouse_holding_costs,
        :supplier_warehouse_costs => supplier_warehouse_costs,
        :warehouse_customer_costs => warehouse_customer_costs,
        :total_demand => total_demand
    )
end

#=============================================================================
  VARIANT 2: GLOBAL SUPPLY CHAIN WITH TARIFFS (25% probability)

  Models international supply chains with cross-border considerations.
  Common in global manufacturing and sourcing operations.

  Features:
  - Multiple geographic regions with different cost structures
  - Tariffs and duties on cross-border shipments
  - Exchange rate considerations
  - Regional production facilities and markets
  - Import quotas and trade restrictions
=============================================================================#

function generate_global_tariff_variant(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem dimensions
    # Variables = n_facilities (binary) + n_facilities * n_customers (continuous flows)
    if target_variables <= 250
        n_regions = rand(DiscreteUniform(2, 3))
        facilities_per_region = rand(DiscreteUniform(2, 4))
        customers_per_region = rand(DiscreteUniform(5, 12))
        grid_size = rand(Uniform(1000.0, 2000.0))
    elseif target_variables <= 1000
        n_regions = rand(DiscreteUniform(3, 5))
        facilities_per_region = rand(DiscreteUniform(3, 6))
        customers_per_region = rand(DiscreteUniform(8, 18))
        grid_size = rand(Uniform(2000.0, 4000.0))
    else
        n_regions = rand(DiscreteUniform(4, 7))
        facilities_per_region = rand(DiscreteUniform(4, 9))
        customers_per_region = rand(DiscreteUniform(12, 30))
        grid_size = rand(Uniform(3000.0, 6000.0))
    end

    n_facilities = n_regions * facilities_per_region
    n_customers = n_regions * customers_per_region

    # Define regions with characteristics
    region_names = ["North America", "Europe", "Asia Pacific", "Latin America",
                   "Middle East", "Africa", "South Asia"]
    selected_regions = sample(region_names, n_regions, replace=false)

    # Region characteristics: labor cost factor, logistics cost factor, market size factor
    region_chars = Dict{String, Dict{Symbol, Float64}}()
    for region in selected_regions
        region_chars[region] = Dict(
            :labor_cost_factor => rand(LogNormal(log(1.0), 0.5)),
            :logistics_cost_factor => rand(LogNormal(log(1.0), 0.4)),
            :market_size_factor => rand(Gamma(2, 0.5)),
            :exchange_rate => rand(LogNormal(log(1.0), 0.25))  # Relative to base currency
        )
    end

    # Assign facilities and customers to regions
    facility_regions = Dict{Int, String}()
    customer_regions = Dict{Int, String}()

    for (idx, region) in enumerate(selected_regions)
        for i in 1:facilities_per_region
            facility_regions[(idx-1) * facilities_per_region + i] = region
        end
        for i in 1:customers_per_region
            customer_regions[(idx-1) * customers_per_region + i] = region
        end
    end

    # Generate locations within regional zones
    region_zone_size = grid_size / sqrt(n_regions)
    facility_locs = Vector{Tuple{Float64,Float64}}()
    customer_locs = Vector{Tuple{Float64,Float64}}()

    for (idx, region) in enumerate(selected_regions)
        # Define region zone (simplified grid layout)
        zone_row = (idx - 1) ÷ ceil(Int, sqrt(n_regions))
        zone_col = (idx - 1) % ceil(Int, sqrt(n_regions))
        zone_x_start = zone_col * region_zone_size
        zone_y_start = zone_row * region_zone_size

        # Facilities in this region
        for _ in 1:facilities_per_region
            x = zone_x_start + region_zone_size * rand(Beta(2, 2))
            y = zone_y_start + region_zone_size * rand(Beta(2, 2))
            push!(facility_locs, (x, y))
        end

        # Customers in this region (more clustered)
        n_local_clusters = rand(DiscreteUniform(1, 3))
        local_cluster_centers = [
            (zone_x_start + region_zone_size * rand(),
             zone_y_start + region_zone_size * rand())
            for _ in 1:n_local_clusters
        ]

        for _ in 1:customers_per_region
            center = rand(local_cluster_centers)
            spread = region_zone_size * 0.15
            x = clamp(center[1] + rand(Normal(0, spread)), zone_x_start, zone_x_start + region_zone_size)
            y = clamp(center[2] + rand(Normal(0, spread)), zone_y_start, zone_y_start + region_zone_size)
            push!(customer_locs, (x, y))
        end
    end

    # Generate customer demands
    base_demand = target_variables <= 250 ? rand(Uniform(120.0, 250.0)) :
                  target_variables <= 1000 ? rand(Uniform(250.0, 500.0)) :
                  rand(Uniform(500.0, 1000.0))

    demands = Dict{Int, Float64}()
    for c in 1:n_customers
        region = customer_regions[c]
        market_factor = region_chars[region][:market_size_factor]
        demand_val = base_demand * market_factor * rand(LogNormal(log(1.0), 0.4))
        demands[c] = demand_val
    end
    total_demand = sum(values(demands))

    # Facility capacities and costs
    capacity_factor = rand(Uniform(1.4, 2.2))
    avg_capacity = (total_demand / n_facilities) * capacity_factor

    capacities = Dict{Int, Float64}()
    fixed_costs = Dict{Int, Float64}()
    production_costs = Dict{Int, Float64}()
    base_fixed_cost = rand(LogNormal(log(600000), 0.6))

    for f in 1:n_facilities
        region = facility_regions[f]
        labor_factor = region_chars[region][:labor_cost_factor]

        capacity = avg_capacity * rand(Gamma(3, 1/3))
        capacities[f] = capacity

        # Fixed costs influenced by region
        fixed_costs[f] = base_fixed_cost * labor_factor * rand(LogNormal(log(1.0), 0.3))

        # Variable production costs (per unit)
        production_costs[f] = rand(Uniform(5.0, 25.0)) * labor_factor
    end

    # Generate tariff rates between regions
    tariff_rates = Dict{Tuple{String, String}, Float64}()
    for region1 in selected_regions
        for region2 in selected_regions
            if region1 == region2
                tariff_rates[(region1, region2)] = 0.0  # No tariff within region
            else
                # Tariff rates typically 0% to 25%, with most being modest
                tariff_rates[(region1, region2)] = rand(Beta(2, 8)) * 0.25
            end
        end
    end

    # Transport costs with tariffs
    transport_costs = Dict{Tuple{Int,Int}, Float64}()
    tariff_costs = Dict{Tuple{Int,Int}, Float64}()
    base_transport_cost = rand(Gamma(3, 0.3))

    for f in 1:n_facilities
        f_region = facility_regions[f]
        logistics_factor = region_chars[f_region][:logistics_cost_factor]

        for c in 1:n_customers
            c_region = customer_regions[c]
            distance = euclidean_distance(facility_locs[f], customer_locs[c])

            # Base transport cost
            base_cost = base_transport_cost * distance * logistics_factor *
                       rand(LogNormal(log(1.0), 0.2))

            # Cross-border premium (additional logistics complexity)
            cross_border_premium = f_region != c_region ? rand(Uniform(1.15, 1.4)) : 1.0

            transport_costs[(f, c)] = base_cost * cross_border_premium

            # Tariff cost (percentage of product value)
            # Approximate product value from production cost
            tariff_rate = tariff_rates[(f_region, c_region)]
            tariff_costs[(f, c)] = production_costs[f] * tariff_rate
        end
    end

    # Regional quotas (some regions limit imports)
    regional_import_quotas = Dict{String, Float64}()
    for region in selected_regions
        if rand() < 0.3  # 30% of regions have import quotas
            region_demand = sum(demands[c] for c in 1:n_customers if customer_regions[c] == region)
            # Quota allows 60-90% to be imported
            quota_factor = rand(Uniform(0.6, 0.9))
            regional_import_quotas[region] = region_demand * quota_factor
        end
    end

    # Feasibility adjustments
    if feasibility_status == feasible
        # Ensure each region has sufficient production capacity
        for region in selected_regions
            region_demand = sum(demands[c] for c in 1:n_customers if customer_regions[c] == region)
            region_facilities = [f for f in 1:n_facilities if facility_regions[f] == region]
            region_capacity = sum(capacities[f] for f in region_facilities)

            # If local capacity is insufficient, ensure imports can cover gap
            if region_capacity < region_demand * 0.3
                # Boost local capacity
                for f in region_facilities
                    capacities[f] *= 1.5
                end
            end
        end

        # Ensure global capacity is sufficient
        if sum(values(capacities)) < total_demand
            scale = 1.15 * total_demand / sum(values(capacities))
            for f in 1:n_facilities
                capacities[f] *= scale
            end
        end

        # Relax quotas if they're too tight
        for (region, quota) in regional_import_quotas
            region_demand = sum(demands[c] for c in 1:n_customers if customer_regions[c] == region)
            if quota < region_demand * 0.5
                regional_import_quotas[region] = region_demand * 0.7
            end
        end

    elseif feasibility_status == infeasible
        # Make infeasible through tight import quotas
        for region in selected_regions
            region_demand = sum(demands[c] for c in 1:n_customers if customer_regions[c] == region)
            region_facilities = [f for f in 1:n_facilities if facility_regions[f] == region]
            region_capacity = sum(capacities[f] for f in region_facilities)

            # If local production is insufficient, add a tight quota
            if region_capacity < region_demand * 0.9
                quota_factor = rand(Uniform(0.3, 0.5))
                regional_import_quotas[region] = (region_demand - region_capacity) * quota_factor
            end
        end
    end

    return Dict(
        :n_facilities => n_facilities,
        :n_customers => n_customers,
        :n_regions => n_regions,
        :facility_regions => facility_regions,
        :customer_regions => customer_regions,
        :region_characteristics => region_chars,
        :capacities => capacities,
        :demands => demands,
        :fixed_costs => fixed_costs,
        :production_costs => production_costs,
        :transport_costs => transport_costs,
        :tariff_costs => tariff_costs,
        :tariff_rates => tariff_rates,
        :regional_import_quotas => regional_import_quotas,
        :total_demand => total_demand
    )
end

#=============================================================================
  VARIANT 3: DIRECT-TO-CONSUMER E-COMMERCE FULFILLMENT (20% probability)

  Models modern e-commerce fulfillment networks with service level requirements.
  Common in online retail and last-mile delivery operations.

  Features:
  - Fulfillment centers with picking/packing capacity
  - Multiple shipping speed tiers (standard, expedited, express)
  - Service level requirements (% of customers within delivery radius)
  - Distance-dependent delivery costs
  - Returns handling capacity
=============================================================================#

function generate_ecommerce_fulfillment_variant(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem dimensions
    # Variables = n_fulfillment_centers (binary) + n_fulfillment_centers * n_customers * n_speed_tiers
    n_speed_tiers = 3  # Standard, Expedited, Express

    if target_variables <= 250
        n_fulfillment_centers = rand(DiscreteUniform(3, 6))
        n_customers = rand(DiscreteUniform(20, 40))
        grid_size = rand(Uniform(600.0, 1200.0))
        clustering_factor = rand(Beta(2, 2)) * 0.4 + 0.4
    elseif target_variables <= 1000
        n_fulfillment_centers = rand(DiscreteUniform(6, 15))
        n_customers = rand(DiscreteUniform(35, 70))
        grid_size = rand(Uniform(1200.0, 3000.0))
        clustering_factor = rand(Beta(2, 3)) * 0.35 + 0.3
    else
        n_fulfillment_centers = rand(DiscreteUniform(12, 30))
        n_customers = rand(DiscreteUniform(60, 180))
        grid_size = rand(Uniform(2500.0, 5000.0))
        clustering_factor = rand(Beta(1, 3)) * 0.3 + 0.25
    end

    # Generate customer clusters (representing metropolitan areas)
    n_clusters = max(4, round(Int, sqrt(n_customers) * clustering_factor))
    customer_locs, cluster_centers, cluster_weights = generate_clustered_locations(
        n_customers, n_clusters, grid_size, grid_size, clustering_factor
    )

    # Fulfillment centers strategically placed near major markets
    fc_locs = Vector{Tuple{Float64,Float64}}()
    for _ in 1:n_fulfillment_centers
        if rand() < 0.7  # 70% near major markets
            center = sample(cluster_centers, Weights(cluster_weights))
            spread = grid_size * 0.12
            x = clamp(center[1] + rand(Normal(0, spread)), 0, grid_size)
            y = clamp(center[2] + rand(Normal(0, spread)), 0, grid_size)
        else  # 30% in strategic secondary locations
            x = grid_size * rand(Beta(1.5, 1.5))
            y = grid_size * rand(Beta(1.5, 1.5))
        end
        push!(fc_locs, (x, y))
    end

    # Customer order volumes (demand)
    base_order_volume = target_variables <= 250 ? rand(Uniform(50.0, 120.0)) :
                        target_variables <= 1000 ? rand(Uniform(120.0, 280.0)) :
                        rand(Uniform(250.0, 550.0))

    demands = Dict{Int, Float64}()
    for c in 1:n_customers
        distances_to_clusters = [euclidean_distance(customer_locs[c], center) for center in cluster_centers]
        _, cluster_idx = findmin(distances_to_clusters)

        # Urban areas have higher order volumes
        urban_factor = cluster_weights[cluster_idx]
        order_volume = base_order_volume * (0.4 + 1.2 * urban_factor) * rand(LogNormal(log(1.0), 0.45))
        demands[c] = order_volume
    end
    total_demand = sum(values(demands))

    # Fulfillment center capacities and costs
    capacity_factor = rand(Uniform(1.25, 1.9))
    avg_capacity = (total_demand / n_fulfillment_centers) * capacity_factor

    fc_capacities = Dict{Int, Float64}()
    fc_fixed_costs = Dict{Int, Float64}()
    fc_variable_costs = Dict{Int, Float64}()
    base_fc_cost = rand(LogNormal(log(400000), 0.55))

    for fc in 1:n_fulfillment_centers
        # Capacity with variation
        capacity = avg_capacity * rand(Gamma(3, 1/3))
        fc_capacities[fc] = capacity

        # Proximity to customers affects real estate costs
        distances_to_customers = [euclidean_distance(fc_locs[fc], c) for c in customer_locs]
        proximity_premium = sum(exp.(-distances_to_customers ./ (grid_size * 0.2))) / n_customers

        # Fixed costs (warehouse, automation, etc.)
        fc_fixed_costs[fc] = base_fc_cost * (0.7 + 0.6 * proximity_premium) *
                            (capacity / avg_capacity)^0.7 * rand(LogNormal(log(1.0), 0.25))

        # Variable costs per unit (picking, packing)
        fc_variable_costs[fc] = rand(Uniform(2.5, 6.0))
    end

    # Shipping costs by speed tier
    speed_tiers = ["standard", "expedited", "express"]
    speed_cost_multipliers = Dict(
        "standard" => 1.0,
        "expedited" => rand(Uniform(1.8, 2.4)),
        "express" => rand(Uniform(3.2, 4.5))
    )

    shipping_costs = Dict{Tuple{Int,Int,String}, Float64}()
    base_shipping_cost = rand(Gamma(5, 0.4))

    for fc in 1:n_fulfillment_centers
        for c in 1:n_customers
            distance = euclidean_distance(fc_locs[fc], customer_locs[c])
            base_cost = base_shipping_cost * distance * rand(LogNormal(log(1.0), 0.15))

            for tier in speed_tiers
                multiplier = speed_cost_multipliers[tier]
                shipping_costs[(fc, c, tier)] = base_cost * multiplier
            end
        end
    end

    # Service level requirements
    # At least X% of customers must have a fulfillment center within Y distance
    service_level_pct = rand(Uniform(0.85, 0.95))
    service_level_radius = grid_size * rand(Uniform(0.25, 0.40))

    # Speed tier capacity allocation (what % of demand uses each tier)
    tier_demand_pct = Dict(
        "standard" => rand(Uniform(0.65, 0.80)),
        "expedited" => rand(Uniform(0.12, 0.22)),
        "express" => rand(Uniform(0.05, 0.12))
    )
    # Normalize to sum to 1
    total_pct = sum(values(tier_demand_pct))
    for tier in speed_tiers
        tier_demand_pct[tier] /= total_pct
    end

    # Feasibility adjustments
    if feasibility_status == feasible
        # Ensure service level can be met
        customers_needing_coverage = ceil(Int, n_customers * service_level_pct)

        # Check which customers are covered
        covered_customers = Set{Int}()
        for fc in 1:n_fulfillment_centers
            for c in 1:n_customers
                if euclidean_distance(fc_locs[fc], customer_locs[c]) <= service_level_radius
                    push!(covered_customers, c)
                end
            end
        end

        # If insufficient, add FCs or expand radius
        if length(covered_customers) < customers_needing_coverage
            service_level_radius *= 1.3
        end

        # Ensure sufficient capacity
        if sum(values(fc_capacities)) < total_demand
            scale = 1.1 * total_demand / sum(values(fc_capacities))
            for fc in 1:n_fulfillment_centers
                fc_capacities[fc] *= scale
            end
        end

    elseif feasibility_status == infeasible
        # Make infeasible through capacity constraints
        scale = rand(Uniform(0.7, 0.88))
        for fc in 1:n_fulfillment_centers
            fc_capacities[fc] *= scale
        end
    end

    return Dict(
        :n_fulfillment_centers => n_fulfillment_centers,
        :n_customers => n_customers,
        :speed_tiers => speed_tiers,
        :fc_capacities => fc_capacities,
        :demands => demands,
        :fc_fixed_costs => fc_fixed_costs,
        :fc_variable_costs => fc_variable_costs,
        :shipping_costs => shipping_costs,
        :tier_demand_pct => tier_demand_pct,
        :service_level_pct => service_level_pct,
        :service_level_radius => service_level_radius,
        :fc_locs => fc_locs,
        :customer_locs => customer_locs,
        :total_demand => total_demand
    )
end

#=============================================================================
  VARIANT 4: MULTI-PERIOD INVENTORY PLANNING (15% probability)

  Models supply chain planning over multiple time periods with inventory.
  Common in seasonal businesses and production planning.

  Features:
  - Multiple time periods with varying demand
  - Inventory holding costs
  - Facility opening/closing decisions
  - Time-varying production and transportation costs
  - Inventory balance constraints
=============================================================================#

function generate_multi_period_inventory_variant(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem dimensions
    # Variables = n_facilities * n_periods (binary open/close) +
    #             n_facilities * n_customers * n_periods (flow) +
    #             n_facilities * n_periods (inventory)

    if target_variables <= 250
        n_facilities = rand(DiscreteUniform(3, 6))
        n_customers = rand(DiscreteUniform(8, 15))
        n_periods = rand(DiscreteUniform(3, 5))
        grid_size = rand(Uniform(500.0, 1000.0))
    elseif target_variables <= 1000
        n_facilities = rand(DiscreteUniform(5, 12))
        n_customers = rand(DiscreteUniform(12, 30))
        n_periods = rand(DiscreteUniform(4, 8))
        grid_size = rand(Uniform(1000.0, 2500.0))
    else
        n_facilities = rand(DiscreteUniform(10, 25))
        n_customers = rand(DiscreteUniform(25, 80))
        n_periods = rand(DiscreteUniform(6, 12))
        grid_size = rand(Uniform(2000.0, 4000.0))
    end

    # Generate locations
    n_clusters = max(2, round(Int, sqrt(n_customers) * 0.5))
    customer_locs, _, cluster_weights = generate_clustered_locations(
        n_customers, n_clusters, grid_size, grid_size, 0.5
    )

    facility_locs = [(grid_size * rand(Beta(1.5, 1.5)), grid_size * rand(Beta(1.5, 1.5)))
                     for _ in 1:n_facilities]

    # Generate time-varying demands (seasonal pattern)
    base_demand = target_variables <= 250 ? rand(Uniform(80.0, 150.0)) :
                  target_variables <= 1000 ? rand(Uniform(150.0, 300.0)) :
                  rand(Uniform(300.0, 600.0))

    # Seasonal pattern (e.g., retail with holiday season)
    seasonal_factors = [rand(Uniform(0.6, 0.9)) for _ in 1:(n_periods÷2)]
    append!(seasonal_factors, [rand(Uniform(1.1, 1.6)) for _ in 1:(n_periods - length(seasonal_factors))])
    shuffle!(seasonal_factors)

    demands = Dict{Tuple{Int,Int}, Float64}()  # (customer, period)
    for c in 1:n_customers
        base_cust_demand = base_demand * rand(LogNormal(log(1.0), 0.4))
        for t in 1:n_periods
            demand_val = base_cust_demand * seasonal_factors[t] * rand(LogNormal(log(1.0), 0.2))
            demands[(c, t)] = demand_val
        end
    end

    # Facility capacities (per period production capacity)
    avg_period_demand = sum(values(demands)) / n_periods
    capacity_factor = rand(Uniform(1.3, 2.0))
    avg_capacity = (avg_period_demand / n_facilities) * capacity_factor

    capacities = Dict{Int, Float64}()
    for f in 1:n_facilities
        capacities[f] = avg_capacity * rand(Gamma(3, 1/3))
    end

    # Fixed costs (per period operating cost)
    fixed_costs = Dict{Int, Float64}()
    base_fixed_cost = rand(LogNormal(log(50000), 0.5))
    for f in 1:n_facilities
        fixed_costs[f] = base_fixed_cost * (capacities[f] / avg_capacity)^0.6 *
                        rand(LogNormal(log(1.0), 0.25))
    end

    # Inventory holding costs (per unit per period)
    holding_costs = Dict{Int, Float64}()
    base_holding_cost = rand(Uniform(0.5, 2.5))
    for f in 1:n_facilities
        holding_costs[f] = base_holding_cost * rand(LogNormal(log(1.0), 0.3))
    end

    # Transport costs (vary by period due to fuel costs, etc.)
    transport_costs = Dict{Tuple{Int,Int,Int}, Float64}()  # (facility, customer, period)
    base_transport_cost = rand(Gamma(3, 0.25))

    # Period cost multipliers (e.g., higher in peak season)
    period_cost_multipliers = [rand(Uniform(0.9, 1.15)) for _ in 1:n_periods]

    for f in 1:n_facilities
        for c in 1:n_customers
            distance = euclidean_distance(facility_locs[f], customer_locs[c])
            base_cost = base_transport_cost * distance * rand(LogNormal(log(1.0), 0.2))

            for t in 1:n_periods
                transport_costs[(f, c, t)] = base_cost * period_cost_multipliers[t]
            end
        end
    end

    # Maximum inventory capacity at facilities
    max_inventory = Dict{Int, Float64}()
    for f in 1:n_facilities
        # Can typically store 2-4 periods worth of production
        max_inventory[f] = capacities[f] * rand(Uniform(2.0, 4.0))
    end

    # Feasibility adjustments
    if feasibility_status == feasible
        # Ensure sufficient capacity in each period
        for t in 1:n_periods
            period_demand = sum(demands[(c, t)] for c in 1:n_customers)
            total_capacity = sum(values(capacities))

            if total_capacity < period_demand
                scale = 1.1 * period_demand / total_capacity
                for f in 1:n_facilities
                    capacities[f] *= scale
                end
            end
        end

    elseif feasibility_status == infeasible
        # Create capacity shortage in peak period
        peak_period = argmax([sum(demands[(c, t)] for c in 1:n_customers) for t in 1:n_periods])
        scale = rand(Uniform(0.65, 0.85))
        for f in 1:n_facilities
            capacities[f] *= scale
        end
    end

    total_demand = sum(values(demands))

    return Dict(
        :n_facilities => n_facilities,
        :n_customers => n_customers,
        :n_periods => n_periods,
        :capacities => capacities,
        :demands => demands,
        :fixed_costs => fixed_costs,
        :holding_costs => holding_costs,
        :transport_costs => transport_costs,
        :max_inventory => max_inventory,
        :seasonal_factors => seasonal_factors,
        :total_demand => total_demand
    )
end

#=============================================================================
  VARIANT 5: MAKE-OR-BUY SUPPLY CHAIN (10% probability)

  Models supply chains with make-or-buy decisions for components/products.
  Common in manufacturing with outsourcing options.

  Features:
  - Internal production facilities with production costs
  - External suppliers with procurement costs and lead times
  - Quality tiers (in-house vs. outsourced)
  - Capacity constraints on both production and procurement
  - Strategic sourcing decisions
=============================================================================#

function generate_make_or_buy_variant(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Determine problem dimensions
    # Variables = n_facilities (binary) + n_products (binary make/buy decisions) +
    #             n_facilities * n_products (production) +
    #             n_suppliers * n_products (procurement) +
    #             n_facilities * n_customers (distribution)

    if target_variables <= 250
        n_facilities = rand(DiscreteUniform(2, 4))
        n_suppliers = rand(DiscreteUniform(3, 6))
        n_products = rand(DiscreteUniform(3, 6))
        n_customers = rand(DiscreteUniform(10, 20))
        grid_size = rand(Uniform(500.0, 1000.0))
    elseif target_variables <= 1000
        n_facilities = rand(DiscreteUniform(3, 8))
        n_suppliers = rand(DiscreteUniform(5, 12))
        n_products = rand(DiscreteUniform(5, 10))
        n_customers = rand(DiscreteUniform(18, 40))
        grid_size = rand(Uniform(1000.0, 2500.0))
    else
        n_facilities = rand(DiscreteUniform(6, 15))
        n_suppliers = rand(DiscreteUniform(10, 25))
        n_products = rand(DiscreteUniform(8, 18))
        n_customers = rand(DiscreteUniform(35, 100))
        grid_size = rand(Uniform(2000.0, 4000.0))
    end

    # Generate locations
    n_clusters = max(2, round(Int, sqrt(n_customers) * 0.4))
    customer_locs, _, _ = generate_clustered_locations(
        n_customers, n_clusters, grid_size, grid_size, 0.45
    )

    facility_locs = [(grid_size * rand(Beta(1.5, 1.5)), grid_size * rand(Beta(1.5, 1.5)))
                     for _ in 1:n_facilities]
    supplier_locs = [(grid_size * rand(Beta(1.2, 1.2)), grid_size * rand(Beta(1.2, 1.2)))
                     for _ in 1:n_suppliers]

    # Customer demands by product
    base_demand = target_variables <= 250 ? rand(Uniform(60.0, 120.0)) :
                  target_variables <= 1000 ? rand(Uniform(120.0, 250.0)) :
                  rand(Uniform(250.0, 500.0))

    demands = Dict{Tuple{Int,Int}, Float64}()  # (customer, product)
    product_popularity = [rand(Beta(2, 5)) for _ in 1:n_products]  # Some products more popular

    for c in 1:n_customers
        cust_demand_factor = rand(LogNormal(log(1.0), 0.4))
        for p in 1:n_products
            # Not all customers demand all products
            if rand() < 0.7  # 70% chance customer needs this product
                demand_val = base_demand * product_popularity[p] * cust_demand_factor *
                            rand(LogNormal(log(1.0), 0.35))
                demands[(c, p)] = demand_val
            end
        end
    end

    # Facility production capacities and costs
    facility_capacities = Dict{Tuple{Int,Int}, Float64}()  # (facility, product)
    facility_fixed_costs = Dict{Int, Float64}()
    production_costs = Dict{Tuple{Int,Int}, Float64}()  # (facility, product)

    base_facility_cost = rand(LogNormal(log(500000), 0.6))

    for f in 1:n_facilities
        facility_fixed_costs[f] = base_facility_cost * rand(LogNormal(log(1.0), 0.3))

        for p in 1:n_products
            # Facilities have different capabilities for different products
            product_demand = sum(get(demands, (c, p), 0.0) for c in 1:n_customers)
            capacity_share = rand(Uniform(0.15, 0.45))

            facility_capacities[(f, p)] = product_demand * capacity_share * rand(Gamma(2, 0.5))

            # Production costs with economies of scale
            base_prod_cost = rand(Uniform(8.0, 35.0))
            scale_factor = (facility_capacities[(f, p)] / product_demand)^(-0.15)  # Economies of scale
            production_costs[(f, p)] = base_prod_cost * scale_factor * rand(LogNormal(log(1.0), 0.2))
        end
    end

    # Supplier procurement capacities and costs
    supplier_capacities = Dict{Tuple{Int,Int}, Float64}()  # (supplier, product)
    procurement_costs = Dict{Tuple{Int,Int}, Float64}()  # (supplier, product)
    supplier_quality_tier = Dict{Int, Float64}()  # Quality factor (higher = better)

    for s in 1:n_suppliers
        supplier_quality_tier[s] = rand(Beta(3, 2))  # Most suppliers are decent quality

        for p in 1:n_products
            # Suppliers specialize in certain products
            if rand() < 0.4  # 40% chance supplier can provide this product
                product_demand = sum(get(demands, (c, p), 0.0) for c in 1:n_customers)
                capacity_share = rand(Uniform(0.1, 0.35))

                supplier_capacities[(s, p)] = product_demand * capacity_share * rand(Gamma(2, 0.6))

                # Procurement costs (typically competitive with or cheaper than in-house)
                avg_production_cost = mean(production_costs[(f, p)] for f in 1:n_facilities)
                quality_premium = 0.8 + 0.4 * supplier_quality_tier[s]
                procurement_costs[(s, p)] = avg_production_cost * rand(Uniform(0.75, 1.15)) *
                                           quality_premium * rand(LogNormal(log(1.0), 0.25))
            end
        end
    end

    # Distribution costs: facility to customer
    distribution_costs = Dict{Tuple{Int,Int}, Float64}()  # (facility, customer)
    base_dist_cost = rand(Gamma(3, 0.3))

    for f in 1:n_facilities
        for c in 1:n_customers
            distance = euclidean_distance(facility_locs[f], customer_locs[c])
            distribution_costs[(f, c)] = base_dist_cost * distance * rand(LogNormal(log(1.0), 0.18))
        end
    end

    # Inbound costs: supplier to facility
    inbound_costs = Dict{Tuple{Int,Int}, Float64}()  # (supplier, facility)
    base_inbound_cost = rand(Gamma(2, 0.2))

    for s in 1:n_suppliers
        for f in 1:n_facilities
            distance = euclidean_distance(supplier_locs[s], facility_locs[f])
            inbound_costs[(s, f)] = base_inbound_cost * distance * rand(LogNormal(log(1.0), 0.2))
        end
    end

    # Feasibility adjustments
    if feasibility_status == feasible
        # Ensure each product can be sourced sufficiently
        for p in 1:n_products
            product_demand = sum(get(demands, (c, p), 0.0) for c in 1:n_customers)

            # Total make capacity
            make_capacity = sum(get(facility_capacities, (f, p), 0.0) for f in 1:n_facilities)
            # Total buy capacity
            buy_capacity = sum(get(supplier_capacities, (s, p), 0.0) for s in 1:n_suppliers)

            total_capacity = make_capacity + buy_capacity

            if total_capacity < product_demand
                scale = 1.15 * product_demand / max(total_capacity, eps())
                # Scale up both make and buy options
                for f in 1:n_facilities
                    if haskey(facility_capacities, (f, p))
                        facility_capacities[(f, p)] *= scale
                    end
                end
                for s in 1:n_suppliers
                    if haskey(supplier_capacities, (s, p))
                        supplier_capacities[(s, p)] *= scale
                    end
                end
            end
        end

    elseif feasibility_status == infeasible
        # Make infeasible by reducing supplier capacity dramatically
        for s in 1:n_suppliers
            for p in 1:n_products
                if haskey(supplier_capacities, (s, p))
                    supplier_capacities[(s, p)] *= rand(Uniform(0.3, 0.6))
                end
            end
        end
        # And also constrain production
        for f in 1:n_facilities
            for p in 1:n_products
                facility_capacities[(f, p)] *= rand(Uniform(0.6, 0.8))
            end
        end
    end

    total_demand = sum(values(demands))

    return Dict(
        :n_facilities => n_facilities,
        :n_suppliers => n_suppliers,
        :n_products => n_products,
        :n_customers => n_customers,
        :demands => demands,
        :facility_capacities => facility_capacities,
        :supplier_capacities => supplier_capacities,
        :facility_fixed_costs => facility_fixed_costs,
        :production_costs => production_costs,
        :procurement_costs => procurement_costs,
        :distribution_costs => distribution_costs,
        :inbound_costs => inbound_costs,
        :supplier_quality_tier => supplier_quality_tier,
        :total_demand => total_demand
    )
end

#=============================================================================
  MAIN CONSTRUCTOR: Samples between variants
=============================================================================#

"""
    SupplyChainProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)

Construct a supply chain problem instance by randomly selecting from multiple realistic variants.

Variants are sampled with the following probabilities based on real-world frequency:
- Multi-echelon supply chain (30%): Suppliers → Warehouses → Customers
- Global supply chain with tariffs (25%): International sourcing with customs/duties
- Direct-to-consumer fulfillment (20%): E-commerce with service level requirements
- Multi-period inventory planning (15%): Time-varying demand with inventory
- Make-or-buy supply chain (10%): Production vs. procurement decisions

# Arguments
- `target_variables`: Target number of variables (problem dimensions scale accordingly)
- `feasibility_status`: Desired feasibility status (feasible, infeasible, or unknown)
- `seed`: Random seed for reproducibility
"""
function SupplyChainProblem(target_variables::Int, feasibility_status::FeasibilityStatus, seed::Int)
    Random.seed!(seed)

    # Sample variant with realistic weights
    variant_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    variant_types = [:multi_echelon, :global_tariff, :ecommerce_fulfillment,
                     :multi_period_inventory, :make_or_buy]

    variant = sample(variant_types, Weights(variant_weights))

    # Generate data based on selected variant
    data = if variant == :multi_echelon
        generate_multi_echelon_variant(target_variables, feasibility_status, seed)
    elseif variant == :global_tariff
        generate_global_tariff_variant(target_variables, feasibility_status, seed)
    elseif variant == :ecommerce_fulfillment
        generate_ecommerce_fulfillment_variant(target_variables, feasibility_status, seed)
    elseif variant == :multi_period_inventory
        generate_multi_period_inventory_variant(target_variables, feasibility_status, seed)
    else  # :make_or_buy
        generate_make_or_buy_variant(target_variables, feasibility_status, seed)
    end

    return SupplyChainProblem(variant, data)
end

#=============================================================================
  BUILD MODEL: Variant-specific model construction
=============================================================================#

"""
    build_model(prob::SupplyChainProblem)

Build a JuMP model for the supply chain problem (deterministic).
The specific formulation depends on which variant was selected.

# Arguments
- `prob`: SupplyChainProblem instance

# Returns
- `model`: The JuMP model
"""
function build_model(prob::SupplyChainProblem)
    if prob.variant == :multi_echelon
        return build_multi_echelon_model(prob.data)
    elseif prob.variant == :global_tariff
        return build_global_tariff_model(prob.data)
    elseif prob.variant == :ecommerce_fulfillment
        return build_ecommerce_fulfillment_model(prob.data)
    elseif prob.variant == :multi_period_inventory
        return build_multi_period_inventory_model(prob.data)
    else  # :make_or_buy
        return build_make_or_buy_model(prob.data)
    end
end

# Multi-echelon model builder
function build_multi_echelon_model(data::Dict{Symbol, Any})
    model = Model()

    n_suppliers = data[:n_suppliers]
    n_warehouses = data[:n_warehouses]
    n_customers = data[:n_customers]

    # Binary variables: open supplier/warehouse
    @variable(model, y_supplier[1:n_suppliers], Bin)
    @variable(model, y_warehouse[1:n_warehouses], Bin)

    # Flow variables
    @variable(model, x_sw[1:n_suppliers, 1:n_warehouses] >= 0)  # Supplier to warehouse
    @variable(model, x_wc[1:n_warehouses, 1:n_customers] >= 0)  # Warehouse to customer

    # Objective: minimize total costs
    @objective(model, Min,
        sum(data[:supplier_fixed_costs][s] * y_supplier[s] for s in 1:n_suppliers) +
        sum(data[:warehouse_fixed_costs][w] * y_warehouse[w] for w in 1:n_warehouses) +
        sum(data[:supplier_warehouse_costs][(s,w)] * x_sw[s,w]
            for s in 1:n_suppliers, w in 1:n_warehouses) +
        sum(data[:warehouse_customer_costs][(w,c)] * x_wc[w,c]
            for w in 1:n_warehouses, c in 1:n_customers) +
        sum(data[:warehouse_holding_costs][w] * sum(x_sw[s,w] for s in 1:n_suppliers)
            for w in 1:n_warehouses)
    )

    # Customer demand satisfaction
    for c in 1:n_customers
        @constraint(model, sum(x_wc[w,c] for w in 1:n_warehouses) >= data[:demands][c])
    end

    # Warehouse flow conservation
    for w in 1:n_warehouses
        @constraint(model,
            sum(x_sw[s,w] for s in 1:n_suppliers) >= sum(x_wc[w,c] for c in 1:n_customers)
        )
    end

    # Supplier capacity constraints
    for s in 1:n_suppliers
        @constraint(model,
            sum(x_sw[s,w] for w in 1:n_warehouses) <= data[:supplier_capacities][s] * y_supplier[s]
        )
    end

    # Warehouse capacity constraints
    for w in 1:n_warehouses
        @constraint(model,
            sum(x_wc[w,c] for c in 1:n_customers) <= data[:warehouse_capacities][w] * y_warehouse[w]
        )
    end

    return model
end

# Global tariff model builder
function build_global_tariff_model(data::Dict{Symbol, Any})
    model = Model()

    n_facilities = data[:n_facilities]
    n_customers = data[:n_customers]
    n_regions = data[:n_regions]

    # Binary variables: open facility
    @variable(model, y[1:n_facilities], Bin)

    # Flow variables
    @variable(model, x[1:n_facilities, 1:n_customers] >= 0)

    # Objective: minimize total costs (fixed + production + transport + tariffs)
    @objective(model, Min,
        sum(data[:fixed_costs][f] * y[f] for f in 1:n_facilities) +
        sum(data[:production_costs][f] * x[f,c] for f in 1:n_facilities, c in 1:n_customers) +
        sum(data[:transport_costs][(f,c)] * x[f,c] for f in 1:n_facilities, c in 1:n_customers) +
        sum(data[:tariff_costs][(f,c)] * x[f,c] for f in 1:n_facilities, c in 1:n_customers)
    )

    # Customer demand satisfaction
    for c in 1:n_customers
        @constraint(model, sum(x[f,c] for f in 1:n_facilities) >= data[:demands][c])
    end

    # Facility capacity constraints
    for f in 1:n_facilities
        @constraint(model, sum(x[f,c] for c in 1:n_customers) <= data[:capacities][f] * y[f])
    end

    # Regional import quotas
    for (region, quota) in data[:regional_import_quotas]
        region_customers = [c for c in 1:n_customers if data[:customer_regions][c] == region]
        region_facilities = [f for f in 1:n_facilities if data[:facility_regions][f] == region]
        other_facilities = [f for f in 1:n_facilities if data[:facility_regions][f] != region]

        # Total imports into this region must not exceed quota
        @constraint(model,
            sum(x[f,c] for f in other_facilities, c in region_customers) <= quota
        )
    end

    return model
end

# E-commerce fulfillment model builder
function build_ecommerce_fulfillment_model(data::Dict{Symbol, Any})
    model = Model()

    n_fulfillment_centers = data[:n_fulfillment_centers]
    n_customers = data[:n_customers]
    speed_tiers = data[:speed_tiers]

    # Binary variables: open fulfillment center
    @variable(model, y[1:n_fulfillment_centers], Bin)

    # Flow variables by speed tier
    @variable(model, x[1:n_fulfillment_centers, 1:n_customers, speed_tiers] >= 0)

    # Objective: minimize total costs
    @objective(model, Min,
        sum(data[:fc_fixed_costs][fc] * y[fc] for fc in 1:n_fulfillment_centers) +
        sum(data[:fc_variable_costs][fc] * sum(x[fc,c,tier] for c in 1:n_customers, tier in speed_tiers)
            for fc in 1:n_fulfillment_centers) +
        sum(data[:shipping_costs][(fc,c,tier)] * x[fc,c,tier]
            for fc in 1:n_fulfillment_centers, c in 1:n_customers, tier in speed_tiers)
    )

    # Customer demand satisfaction (across all tiers)
    for c in 1:n_customers
        @constraint(model,
            sum(x[fc,c,tier] for fc in 1:n_fulfillment_centers, tier in speed_tiers) >= data[:demands][c]
        )
    end

    # Fulfillment center capacity constraints
    for fc in 1:n_fulfillment_centers
        @constraint(model,
            sum(x[fc,c,tier] for c in 1:n_customers, tier in speed_tiers) <=
            data[:fc_capacities][fc] * y[fc]
        )
    end

    # Service level requirement: % of customers must have nearby FC
    # Count customers within service radius
    customers_within_radius = Dict{Int, Vector{Int}}()
    for fc in 1:n_fulfillment_centers
        customers_within_radius[fc] = Int[]
        for c in 1:n_customers
            distance = euclidean_distance(data[:fc_locs][fc], data[:customer_locs][c])
            if distance <= data[:service_level_radius]
                push!(customers_within_radius[fc], c)
            end
        end
    end

    # At least service_level_pct of customers must be served by an open FC within radius
    min_covered_customers = ceil(Int, n_customers * data[:service_level_pct])
    for c in 1:n_customers
        eligible_fcs = [fc for fc in 1:n_fulfillment_centers if c in customers_within_radius[fc]]
        if !isempty(eligible_fcs)
            # If customer is within radius of any FC, at least one nearby FC should be open if serving them
            @constraint(model,
                sum(x[fc,c,tier] for fc in eligible_fcs, tier in speed_tiers) >=
                data[:demands][c] * sum(y[fc] for fc in eligible_fcs) / max(length(eligible_fcs), 1)
            )
        end
    end

    return model
end

# Multi-period inventory model builder
function build_multi_period_inventory_model(data::Dict{Symbol, Any})
    model = Model()

    n_facilities = data[:n_facilities]
    n_customers = data[:n_customers]
    n_periods = data[:n_periods]

    # Binary variables: facility open in each period
    @variable(model, y[1:n_facilities, 1:n_periods], Bin)

    # Flow variables
    @variable(model, x[1:n_facilities, 1:n_customers, 1:n_periods] >= 0)

    # Inventory variables
    @variable(model, inv[1:n_facilities, 0:n_periods] >= 0)

    # Set initial inventory to zero
    for f in 1:n_facilities
        fix(inv[f, 0], 0.0; force=true)
    end

    # Objective: minimize total costs
    @objective(model, Min,
        sum(data[:fixed_costs][f] * y[f,t] for f in 1:n_facilities, t in 1:n_periods) +
        sum(data[:transport_costs][(f,c,t)] * x[f,c,t]
            for f in 1:n_facilities, c in 1:n_customers, t in 1:n_periods) +
        sum(data[:holding_costs][f] * inv[f,t] for f in 1:n_facilities, t in 1:n_periods)
    )

    # Customer demand satisfaction
    for c in 1:n_customers, t in 1:n_periods
        @constraint(model,
            sum(x[f,c,t] for f in 1:n_facilities) >= data[:demands][(c,t)]
        )
    end

    # Inventory balance constraints
    for f in 1:n_facilities, t in 1:n_periods
        @constraint(model,
            inv[f, t-1] + data[:capacities][f] * y[f,t] ==
            sum(x[f,c,t] for c in 1:n_customers) + inv[f,t]
        )
    end

    # Maximum inventory constraints
    for f in 1:n_facilities, t in 1:n_periods
        @constraint(model, inv[f,t] <= data[:max_inventory][f])
    end

    return model
end

# Make-or-buy model builder
function build_make_or_buy_model(data::Dict{Symbol, Any})
    model = Model()

    n_facilities = data[:n_facilities]
    n_suppliers = data[:n_suppliers]
    n_products = data[:n_products]
    n_customers = data[:n_customers]

    # Binary variables: open facility
    @variable(model, y_facility[1:n_facilities], Bin)

    # Production variables (facility, product)
    production_combinations = [(f, p) for f in 1:n_facilities, p in 1:n_products
                               if haskey(data[:facility_capacities], (f, p))]
    @variable(model, produce[production_combinations] >= 0)

    # Procurement variables (supplier, product)
    procurement_combinations = [(s, p) for s in 1:n_suppliers, p in 1:n_products
                                if haskey(data[:supplier_capacities], (s, p))]
    @variable(model, procure[procurement_combinations] >= 0)

    # Distribution variables (facility, customer, product)
    # Only for customers that demand the product
    distribution_combinations = [(f, c, p) for f in 1:n_facilities, c in 1:n_customers, p in 1:n_products
                                 if haskey(data[:demands], (c, p))]
    @variable(model, distribute[distribution_combinations] >= 0)

    # Objective: minimize total costs
    @objective(model, Min,
        # Facility fixed costs
        sum(data[:facility_fixed_costs][f] * y_facility[f] for f in 1:n_facilities) +
        # Production costs
        sum(data[:production_costs][(f,p)] * produce[(f,p)] for (f,p) in production_combinations) +
        # Procurement costs
        sum(data[:procurement_costs][(s,p)] * procure[(s,p)] for (s,p) in procurement_combinations) +
        # Inbound logistics (supplier to facility)
        sum(data[:inbound_costs][(s,f)] * sum(procure[(s,p)] for (ss,p) in procurement_combinations if ss == s)
            for s in 1:n_suppliers, f in 1:n_facilities if haskey(data[:inbound_costs], (s,f))) +
        # Distribution costs
        sum(data[:distribution_costs][(f,c)] * sum(distribute[(ff,c,p)] for (ff,cc,p) in distribution_combinations if ff == f && cc == c)
            for f in 1:n_facilities, c in 1:n_customers)
    )

    # Customer demand satisfaction
    for c in 1:n_customers, p in 1:n_products
        if haskey(data[:demands], (c, p))
            valid_distributions = [(f, c, p) for (f, cc, pp) in distribution_combinations
                                  if cc == c && pp == p]
            @constraint(model,
                sum(distribute[combo] for combo in valid_distributions) >= data[:demands][(c,p)]
            )
        end
    end

    # Facility flow balance (production + procurement = distribution)
    for f in 1:n_facilities, p in 1:n_products
        produced = sum(produce[(ff,pp)] for (ff,pp) in production_combinations if ff == f && pp == p)

        # Procurement arriving at this facility for this product
        # Simplified: assume procured goods are distributed across facilities proportionally
        total_procured = sum(procure[(s,pp)] for (s,pp) in procurement_combinations if pp == p)
        received = total_procured / n_facilities  # Simplified distribution

        distributed = sum(distribute[(ff,c,pp)] for (ff,c,pp) in distribution_combinations
                         if ff == f && pp == p)

        @constraint(model, produced + received >= distributed)
    end

    # Production capacity constraints
    for (f, p) in production_combinations
        @constraint(model, produce[(f,p)] <= data[:facility_capacities][(f,p)] * y_facility[f])
    end

    # Procurement capacity constraints
    for (s, p) in procurement_combinations
        @constraint(model, procure[(s,p)] <= data[:supplier_capacities][(s,p)])
    end

    return model
end

# Register the problem type
register_problem(
    :supply_chain,
    SupplyChainProblem,
    "Supply chain optimization with multiple realistic variants: multi-echelon distribution, global sourcing with tariffs, e-commerce fulfillment, multi-period inventory planning, and make-or-buy decisions"
)
